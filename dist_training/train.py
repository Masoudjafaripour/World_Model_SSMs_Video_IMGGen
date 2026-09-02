"""Run one arm: one (phase, strategy, layout, seq_len, model) configuration.

Protocol note that governs the whole study: TP and PP are mathematically exact,
DP changes the global batch and therefore the model you get. So the global batch
in tokens is held constant across every arm and the difference is absorbed into
gradient accumulation. Loss then becomes a *sanity check*, not a result, and
every arm differs only in wall-clock, memory and utilisation.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

try:
    from tqdm import tqdm
except ImportError:                        # optional dependency; run still works
    def tqdm(it, *a, **k):
        return it

try:
    import wandb
except ImportError:                        # optional dependency; --wandb no-ops without it
    wandb = None

import costmodel as cm
import metrics as M
import parallel as P
from data import build_dataset
from model import ModelCfg, VideoDiT, apply_lora, flow_loss, trainable_fraction


@dataclass
class RunCfg:
    # identity
    phase: str = "pretrain"          # pretrain | finetune
    strategy: str = "ddp"
    model_size: str = "S"            # S | L
    seq_len: int = 8192
    micro_bs: int = 1
    dp: int = 1
    tp: int = 1
    pp: int = 1
    ckpt: bool = False
    seed: int = 0
    # protocol
    global_batch_tokens: int = 524_288      # held constant across all arms
    steps: int = 300
    warmup: int = 20
    lr: float = 1e-4
    lora_r: int = 16
    # data
    data_source: str = "synthetic"
    data_root: str = "data/latents"
    split: str = "train"
    n_clips: int = 4096
    in_dim: int = 16
    action_dim: int = 8
    # io
    out_dir: str = "results"
    init_from: str = ""               # pretrained checkpoint for the finetune arm
    save_every: int = 0               # >0 enables DCP mid-run checkpointing
    gpu: str = "a6000"
    # W&B (all opt-in; --wandb=false by default, so nothing changes unless set)
    wandb: bool = False
    wandb_project: str = "video-wm-dist-training"
    wandb_group: str = ""             # defaults to out_dir's basename (dryrun/main)
    wandb_mode: str = "online"        # online | offline | disabled

    def key(self) -> str:
        return (f"{self.phase}_{self.model_size}_{self.strategy}_s{self.seq_len}"
                f"_b{self.micro_bs}_dp{self.dp}tp{self.tp}pp{self.pp}"
                f"_{'ckpt' if self.ckpt else 'nockpt'}_seed{self.seed}")


SIZES = {
    "T": ModelCfg(dim=128, depth=2, head_dim=32, max_seq=1024),  # CPU smoke only
    "S": ModelCfg(dim=1024, depth=24),                            # 464 M
    "L": ModelCfg(dim=2048, depth=32),                            # 2.44 B
}


# --------------------------------------------------------------------------- #
def make_model(cfg: RunCfg, mcfg: ModelCfg, device):
    torch.manual_seed(cfg.seed)
    model = VideoDiT(mcfg)
    if cfg.init_from and Path(cfg.init_from).exists():
        sd = torch.load(cfg.init_from, map_location="cpu", weights_only=True)
        model.load_state_dict(sd, strict=False)
    if cfg.phase == "finetune":
        model = apply_lora(model, r=cfg.lora_r)
    if cfg.ckpt:
        model = P.enable_checkpointing(model)
    return model


def save_dcp(model, opt, step, path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict
    ms, os_ = get_state_dict(model, opt)
    dcp.save({"model": ms, "optim": os_, "step": step}, checkpoint_id=str(path))


def load_dcp(model, opt, path):
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (get_state_dict,
                                                         set_state_dict)
    ms, os_ = get_state_dict(model, opt)
    state = {"model": ms, "optim": os_, "step": 0}
    dcp.load(state, checkpoint_id=str(path))
    set_state_dict(model, opt, model_state_dict=state["model"],
                   optim_state_dict=state["optim"])
    return state.get("step", 0)


# --------------------------------------------------------------------------- #
def run(cfg: RunCfg) -> dict:
    rank, world, local = P.init_dist()
    device = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    out = Path(cfg.out_dir)
    (out / "runs" / cfg.key()).mkdir(parents=True, exist_ok=True)
    run_dir = out / "runs" / cfg.key()

    mcfg = SIZES[cfg.model_size]
    mcfg.in_dim, mcfg.action_dim = cfg.in_dim, cfg.action_dim
    mcfg.max_seq = max(mcfg.max_seq, cfg.seq_len)
    assert mcfg.dim % cfg.tp == 0 and mcfg.n_heads % cfg.tp == 0, "dim/heads must divide tp"

    # --- protocol: fix global batch tokens, absorb the rest into accumulation ---
    accum = max(1, cfg.global_batch_tokens // (cfg.micro_bs * cfg.seq_len * max(cfg.dp, 1)))

    # DDP needs no mesh; TP/FSDP do. Keeping this narrow avoids constructing a
    # device mesh on setups where it would fail for unrelated reasons.
    needs_mesh = world > 1 and cfg.strategy in ("fsdp", "zero2", "tp", "tp_dp")
    mesh = P.build_mesh(world, cfg.dp, cfg.tp) if needs_mesh else None
    model = make_model(cfg, mcfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    rho = trainable_fraction(model)
    n_trainable = int(n_params * rho)

    # --- data -------------------------------------------------------------- #
    ds = build_dataset(argparse.Namespace(**{**asdict(cfg), "seq_len": cfg.seq_len}))
    sampler = (DistributedSampler(ds, num_replicas=cfg.dp,
                                  rank=rank // max(cfg.tp, 1), seed=cfg.seed)
               if cfg.dp > 1 else None)
    nw = 4 if torch.cuda.is_available() else 0
    dl = DataLoader(ds, batch_size=cfg.micro_bs, sampler=sampler,
                    shuffle=(sampler is None), num_workers=nw,
                    pin_memory=torch.cuda.is_available(), drop_last=True,
                    persistent_workers=nw > 0,
                    **({"prefetch_factor": 4} if nw else {}))

    # --- pipeline arm takes a separate path -------------------------------- #
    use_pp = cfg.strategy == "pp"
    if use_pp:
        stage_mod, stage, SchedCls = P.build_pp(model, mcfg, cfg.pp, rank, device,
                                                cfg.micro_bs, cfg.seq_len)
        sched = SchedCls(stage, n_microbatches=accum,
                         loss_fn=lambda o, y: torch.nn.functional.mse_loss(o[0], y))
        params = stage_mod.parameters()
        nosync = lambda: __import__("contextlib").nullcontext()
        net = stage_mod
    else:
        net, nosync = P.wrap(model, cfg.strategy, mesh=mesh, dp=cfg.dp,
                             tp=cfg.tp, device=device)
        params = net.parameters()

    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=cfg.lr,
                            betas=(0.9, 0.95), fused=torch.cuda.is_available())

    start_step = 0
    ckpt_dir = run_dir / "dcp"
    if cfg.save_every and ckpt_dir.exists():
        start_step = load_dcp(net, opt, ckpt_dir)

    # --- measure ------------------------------------------------------------ #
    flops_tok = cm.train_flops_per_token(n_params, mcfg.depth, cfg.seq_len, mcfg.dim)
    timer = M.StepTimer(warmup=cfg.warmup)
    loss_log, it = [], iter(dl)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    wb = None
    if cfg.wandb and P.is_main():
        if wandb is None:
            print("[warn] --wandb requested but the wandb package is not "
                  "installed; skipping. pip install wandb")
        else:
            wb = wandb.init(project=cfg.wandb_project,
                            group=cfg.wandb_group or Path(cfg.out_dir).name,
                            name=cfg.key(), config=asdict(cfg),
                            mode=cfg.wandb_mode, reinit=True)

    monitor = M.GpuMonitor(run_dir / "gpu_telemetry.csv") if P.is_main() else None
    live_plot = (M.LiveGpuPlot(run_dir / "gpu_telemetry.csv",
                               run_dir / "gpu_usage_live.png", wandb_run=wb)
                if P.is_main() else None)
    comm = M.CommCounter() if world > 1 else None
    t_start, oom, err = time.time(), False, ""

    step_iter = tqdm(range(start_step, cfg.steps), desc=cfg.key(),
                     disable=not P.is_main(), leave=False, dynamic_ncols=True)
    try:
        with (monitor or __import__("contextlib").nullcontext()), \
             (live_plot or __import__("contextlib").nullcontext()), \
             (comm or __import__("contextlib").nullcontext()):
            for step in step_iter:
                with timer:
                    opt.zero_grad(set_to_none=True)
                    tot = 0.0
                    if use_pp:
                        # ScheduleGPipe.step() takes the *whole* accumulated
                        # batch in one call and does its own internal chunking
                        # into n_microbatches -- it must not be called per
                        # microbatch (that undersizes the batch dim it expects
                        # to split, e.g. "Expecting 128 arg_mbs but got 1").
                        xs, as_ = [], []
                        for _ in range(accum):
                            try:
                                x, a = next(it)
                            except StopIteration:
                                it = iter(dl)
                                x, a = next(it)
                            xs.append(x); as_.append(a)
                        x = torch.cat(xs, dim=0).to(device, non_blocking=True)
                        a = torch.cat(as_, dim=0).to(device, non_blocking=True)
                        with torch.autocast("cuda", torch.bfloat16,
                                            enabled=torch.cuda.is_available()):
                            sched.step(x, target=x) if rank == cfg.pp - 1 else sched.step(x)
                        # GPipe computes loss internally per chunk; not surfaced
                        # here, so the loss curve stays a sanity check only for pp.
                    else:
                        for micro in range(accum):
                            try:
                                x, a = next(it)
                            except StopIteration:
                                it = iter(dl)
                                x, a = next(it)
                            x, a = x.to(device, non_blocking=True), a.to(device, non_blocking=True)
                            last = micro == accum - 1
                            with torch.autocast("cuda", torch.bfloat16,
                                                enabled=torch.cuda.is_available()):
                                # no_sync on all but the last microbatch: this is
                                # what makes DP comm amortise over accumulation.
                                with (nosync() if not last else
                                      __import__("contextlib").nullcontext()):
                                    l = flow_loss(net, x, a) / accum
                                    l.backward()
                            tot += float(l.detach()) * accum
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in (params if not use_pp else stage_mod.parameters())
                         if p.requires_grad], 1.0)
                    opt.step()
                loss_log.append({"step": step, "loss": tot / max(accum, 1),
                                 "t": round(time.time() - t_start, 2)})
                if P.is_main() and hasattr(step_iter, "set_postfix"):
                    step_iter.set_postfix(loss=f"{tot / max(accum, 1):.4f}")
                if wb is not None:
                    wb.log({"loss": tot / max(accum, 1),
                           "step_time": timer.times[-1] if timer.times else None},
                          step=step)
                if P.is_main() and step % 20 == 0:
                    (run_dir / "loss.jsonl").write_text(
                        "\n".join(json.dumps(r) for r in loss_log) + "\n")
                if cfg.save_every and step and step % cfg.save_every == 0:
                    save_dcp(net, opt, step, ckpt_dir)
                    timer = M.StepTimer(warmup=cfg.warmup)  # re-warm after resume
    except torch.cuda.OutOfMemoryError:
        oom, err = True, "OOM"
    except Exception as e:                                   # record, do not crash the sweep
        err = f"{type(e).__name__}: {e}"[:300]

    # --- record ------------------------------------------------------------- #
    st = timer.stats()
    tokens_per_step = cfg.micro_bs * cfg.seq_len * accum * max(cfg.dp, 1)
    tps = tokens_per_step / st["step_time"] if st["n_timed"] else float("nan")
    pred = cm.predict_step_time(
        cfg.strategy, cm.Machine(n_gpus=world, peak_flops=cm.PEAK_FLOPS[cfg.gpu]),
        n_params=n_params, n_trainable=n_trainable, depth=mcfg.depth, dim=mcfg.dim,
        seq_len=cfg.seq_len, micro_bs=cfg.micro_bs, accum=accum, dp=cfg.dp,
        tp=cfg.tp, pp=cfg.pp, checkpointing=cfg.ckpt)

    rec = {
        **asdict(cfg), "key": cfg.key(), "world": world, "accum": accum,
        "n_params": n_params, "n_trainable": n_trainable, "rho": rho,
        "flops_per_token": flops_tok, "tokens_per_step": tokens_per_step,
        **st, "tokens_per_s": tps,
        "mfu": M.mfu(tps, flops_tok, cm.PEAK_FLOPS[cfg.gpu], world) if st["n_timed"] else None,
        **M.memory_stats(),
        "comm_bytes_measured": (comm.bytes / max(st["n_timed"], 1)) if comm else 0,
        "comm_bytes_predicted": pred["comm_bytes"],
        "pred_step_time": pred["step_time"], "pred_tokens_per_s": pred["tokens_per_s"],
        "tp_dp_ratio": cm.tp_dp_ratio(cfg.seq_len, mcfg.dim, cfg.micro_bs, accum, rho),
        "comm_compute": cm.comm_compute_ratio(
            cfg.strategy, n_params=n_params, n_trainable=n_trainable,
            depth=mcfg.depth, dim=mcfg.dim, seq_len=cfg.seq_len,
            micro_bs=cfg.micro_bs, accum=accum, dp=cfg.dp, tp=cfg.tp, pp=cfg.pp,
            peak=cm.PEAK_FLOPS[cfg.gpu], n_gpus=world),
        "final_loss": loss_log[-1]["loss"] if loss_log else None,
        "wall_s": round(time.time() - t_start, 1),
        "oom": oom, "error": err, "ok": (not err) and st["n_timed"] > 0,
        "throttled": M.throttled(run_dir / "gpu_telemetry.csv"),
        "torch": torch.__version__, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if P.is_main():
        (run_dir / "loss.jsonl").write_text(
            "\n".join(json.dumps(r) for r in loss_log) + "\n")
        (run_dir / "record.json").write_text(json.dumps(rec, indent=2, default=str))
        M.append_jsonl(out / "results.jsonl", rec)
        if cfg.phase == "pretrain" and cfg.strategy == "single" and not err:
            torch.save(model.state_dict(), out / f"pretrained_{cfg.model_size}.pt")
    if wb is not None:
        wb.log({f"final/{k}": v for k, v in rec.items()
               if isinstance(v, (int, float)) and v is not None})
        wb.finish()
    P.cleanup()
    return rec


def main():
    p = argparse.ArgumentParser()
    # `from __future__ import annotations` makes __annotations__ strings, so the
    # type is taken from the default value instead.
    def as_bool(v):
        return str(v).lower() in ("1", "true", "yes")
    for fld in dataclasses.fields(RunCfg):
        d = fld.default
        p.add_argument(f"--{fld.name}", default=d,
                       type=as_bool if isinstance(d, bool) else type(d))
    cfg = RunCfg(**vars(p.parse_args()))
    rec = run(cfg)
    if P.is_main():
        print(json.dumps({k: rec[k] for k in
                          ("key", "ok", "tokens_per_s", "mfu", "peak_alloc_gb",
                           "step_time", "error")}, indent=2))


if __name__ == "__main__":
    main()
