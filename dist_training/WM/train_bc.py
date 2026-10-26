"""BC (+ world loss) training on cached LIBERO features. 1 GPU or N GPUs (DDP).

  python train_bc.py --name t_overfit --size T --overfit_episodes 8 --steps 2000
  torchrun --standalone --nproc_per_node=4 train_bc.py --name s_ddp4 --size S

Checkpoint policy (README "Checkpoints"): only rank 0 writes, and a run dir
holds at most last.pt (resume; atomic overwrite), best.pt (weights, bf16, best
val BC loss) and final.pt (weights, bf16). last.pt is deleted when a run
finishes. Rerunning the same --name resumes from last.pt.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # dist_training/metrics.py
import metrics as M  # noqa: E402
from libero_data import LiberoFeatures  # noqa: E402
from wla_model import SIZES, WLAPolicy, wla_loss  # noqa: E402


@dataclass
class Cfg:
    name: str = "debug"
    size: str = "S"
    lam: float = 0.0                 # world-loss weight; 0 = plain BC
    future_k: int = 4
    steps: int = 20000
    batch: int = 64                  # per GPU
    lr: float = 3e-4
    wd: float = 0.05
    warmup: int = 500
    eval_every: int = 500
    eval_batches: int = 50
    save_every_min: float = 20.0     # how often last.pt is refreshed
    max_minutes: float = 0.0         # stop cleanly (saving last.pt) before the job limit
    overfit_episodes: int = 0
    workers: int = 4
    seed: int = 0
    data: str = os.environ.get("WM_DATA", "") + "/libero_siglip"
    runs: str = os.environ.get("WM_RUNS", "runs")
    wandb: bool = False


def is_main():
    return not dist.is_initialized() or dist.get_rank() == 0


def lr_at(step, cfg):
    if step < cfg.warmup:
        return cfg.lr * (step + 1) / cfg.warmup
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return cfg.lr * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def atomic_save(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def weights_bf16(model):
    return {k: v.detach().to(torch.bfloat16).cpu() for k, v in model.state_dict().items()}


# --------------------------------------------------------------------------- #
# Live plot: losses, throughput and per-GPU utilisation in one PNG
# --------------------------------------------------------------------------- #
class LivePlot:
    def __init__(self, run_dir: Path, period: float = 60.0):
        self.dir, self.period, self._stop = run_dir, period, threading.Event()

    def render(self):
        import csv
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        mf = self.dir / "metrics.jsonl"
        if not mf.exists():
            return
        rows = [json.loads(l) for l in mf.read_text().splitlines() if l.strip()]
        tr = [r for r in rows if r["kind"] == "train"]
        va = [r for r in rows if r["kind"] == "val"]
        fig, ax = plt.subplots(2, 2, figsize=(11, 7))
        a = ax[0, 0]
        a.plot([r["step"] for r in tr], [r["bc"] for r in tr], lw=1, label="train BC")
        a.plot([r["step"] for r in va], [r["bc"] for r in va], "o-", ms=3, label="val BC")
        a.set_yscale("log"); a.set_title("BC loss (L1, normalised)"); a.legend()
        a = ax[0, 1]
        a.plot([r["step"] for r in tr], [r["world"] for r in tr], lw=1, label="train world")
        a.plot([r["step"] for r in va], [r["world"] for r in va], "o-", ms=3, label="val world")
        a.set_title("world loss (cos + MSE)"); a.legend()
        a = ax[1, 0]
        a.plot([r["step"] for r in tr], [r["samples_s"] for r in tr], lw=1, label="samples/s (all GPUs)")
        a2 = a.twinx()
        a2.plot([r["step"] for r in tr], [r["data_frac"] for r in tr], "r", lw=0.8, label="data-wait fraction")
        a2.set_ylim(0, 1)
        a.set_title("throughput"); a.legend(loc="upper left"); a2.legend(loc="upper right")
        a = ax[1, 1]
        gf = self.dir / "gpu_telemetry.csv"
        if gf.exists():
            by = {}
            with open(gf) as f:
                for r in csv.DictReader(f):
                    try:
                        by.setdefault(r["gpu"], []).append((float(r["t"]) / 60, float(r["util_pct"])))
                    except (ValueError, KeyError):
                        continue
            for g, v in sorted(by.items()):
                a.plot([p[0] for p in v], [p[1] for p in v], lw=0.8, label=f"gpu{g}")
            a.legend(fontsize=7)
        a.set_ylim(0, 105); a.set_xlabel("min"); a.set_title("GPU util % (rank-0 node)")
        fig.suptitle(self.dir.name); fig.tight_layout()
        tmp = self.dir / "live.tmp.png"
        fig.savefig(tmp, dpi=90); plt.close(fig)
        os.replace(tmp, self.dir / "live.png")

    def _loop(self):
        while not self._stop.wait(self.period):
            try:
                self.render()
            except Exception as e:                      # never kill training over a plot
                print(f"[liveplot] {e}", flush=True)

    def __enter__(self):
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        try:
            self.render()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, dl, cfg, device):
    model.eval()
    tot = torch.zeros(3, device=device)
    for i, b in enumerate(dl):
        if i >= cfg.eval_batches:
            break
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        with torch.autocast("cuda", torch.bfloat16):
            _, bc, wl = wla_loss(model, b, cfg.lam)
        tot += torch.stack([bc.float(), wl.float(), torch.ones((), device=device)])
    if dist.is_initialized():
        dist.all_reduce(tot)
    model.train()
    return (tot[0] / tot[2]).item(), (tot[1] / tot[2]).item()


def main(cfg: Cfg):
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank, local = int(os.environ.get("RANK", 0)), int(os.environ.get("LOCAL_RANK", 0))
    if world > 1:
        dist.init_process_group("nccl")
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    torch.manual_seed(cfg.seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = Path(cfg.runs) / cfg.name
    run_dir.mkdir(parents=True, exist_ok=True)

    mcfg = SIZES[cfg.size]
    train_ds = LiberoFeatures(cfg.data, "train", mcfg.horizon, cfg.future_k, cfg.overfit_episodes)
    val_ds = LiberoFeatures(cfg.data, "val", mcfg.horizon, cfg.future_k, cfg.overfit_episodes)
    tsamp = DistributedSampler(train_ds, seed=cfg.seed) if world > 1 else None
    vsamp = DistributedSampler(val_ds, shuffle=True, seed=1) if world > 1 else None
    dl_kw = dict(batch_size=cfg.batch, num_workers=cfg.workers, pin_memory=True,
                 drop_last=True, persistent_workers=cfg.workers > 0)
    train_dl = DataLoader(train_ds, sampler=tsamp, shuffle=tsamp is None, **dl_kw)
    val_dl = DataLoader(val_ds, sampler=vsamp, shuffle=vsamp is None,
                        generator=torch.Generator().manual_seed(1), **dl_kw)

    model = WLAPolicy(mcfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    net = DDP(model, device_ids=[local]) if world > 1 else model
    decay = [p for n, p in model.named_parameters() if p.ndim >= 2 and "emb" not in n and "pos" not in n]
    no_decay = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and "emb" not in n and "pos" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=cfg.lr, betas=(0.9, 0.95), fused=True)

    # --- resume ---------------------------------------------------------- #
    step, best, epoch = 0, float("inf"), 0
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        step, best, epoch = ck["step"], ck["best"], ck["epoch"]
        if is_main():
            print(f"[resume] {last} at step {step} (best val BC {best:.4f})", flush=True)
    if is_main():
        (run_dir / "config.json").write_text(json.dumps(
            {**asdict(cfg), "model": asdict(mcfg), "n_params": n_params, "world": world,
             "global_batch": cfg.batch * world, "train_samples": len(train_ds),
             "val_samples": len(val_ds)}, indent=1))
        print(f"[{cfg.name}] {n_params/1e6:.1f}M params, {world} GPU(s), global batch "
              f"{cfg.batch*world}, {len(train_ds)} train / {len(val_ds)} val samples", flush=True)

    wb = None
    if cfg.wandb and is_main():
        import wandb
        wb = wandb.init(project="wla-libero", name=cfg.name, config=asdict(cfg),
                        dir=str(run_dir), mode="offline", resume="allow", id=cfg.name)

    def save_last():
        if is_main():
            atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                         "best": best, "epoch": epoch, "cfg": asdict(cfg)}, last)

    from contextlib import ExitStack
    t_job, t_saved = time.time(), time.time()
    stopped_early = False
    with ExitStack() as stack:
        if is_main():
            stack.enter_context(M.GpuMonitor(run_dir / "gpu_telemetry.csv", period=5.0))
            stack.enter_context(LivePlot(run_dir))
        mf = open(run_dir / "metrics.jsonl", "a") if is_main() else None
        acc = torch.zeros(3, device=device)
        t_data = t_win = 0.0
        t0 = time.time()
        while step < cfg.steps:
            if tsamp is not None:
                tsamp.set_epoch(epoch)
            td = time.time()
            for b in train_dl:
                b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
                t_data += time.time() - td
                for g in opt.param_groups:
                    g["lr"] = lr_at(step, cfg)
                with torch.autocast("cuda", torch.bfloat16):
                    loss, bc, wl = wla_loss(net, b, cfg.lam)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                acc += torch.stack([bc.detach().float(), wl.detach().float(), torch.ones((), device=device)])
                step += 1

                if step % 50 == 0:
                    torch.cuda.synchronize()
                    dt = time.time() - t0
                    r = (acc / acc[2]).tolist()
                    rec = {"kind": "train", "step": step, "bc": r[0], "world": r[1],
                           "lr": lr_at(step, cfg), "samples_s": 50 * cfg.batch * world / dt,
                           "data_frac": t_data / dt, "min": (time.time() - t_job) / 60}
                    if is_main():
                        mf.write(json.dumps(rec) + "\n"); mf.flush()
                        if wb: wb.log(rec, step=step)
                        if step % 500 == 0:
                            print(f"step {step:6d}  bc {r[0]:.4f}  world {r[1]:.4f}  "
                                  f"{rec['samples_s']:.0f} samples/s  data {rec['data_frac']:.0%}",
                                  flush=True)
                    acc.zero_(); t_data = 0.0; t0 = time.time()

                if step % cfg.eval_every == 0 or step == cfg.steps:
                    vbc, vwl = evaluate(net, val_dl, cfg, device)
                    if is_main():
                        mf.write(json.dumps({"kind": "val", "step": step, "bc": vbc, "world": vwl}) + "\n")
                        mf.flush()
                        if wb: wb.log({"val_bc": vbc, "val_world": vwl}, step=step)
                        print(f"  val @ {step}: bc {vbc:.4f}  world {vwl:.4f}", flush=True)
                        if vbc < best:
                            best = vbc
                            atomic_save({"model": weights_bf16(model), "step": step,
                                         "val_bc": vbc, "cfg": asdict(cfg)}, run_dir / "best.pt")
                    t0 = time.time()

                if (time.time() - t_saved) / 60 > cfg.save_every_min:
                    save_last(); t_saved = time.time()
                if cfg.max_minutes and step % 50 == 0:
                    # rank 0 decides, so every rank stops at the same step
                    stop = torch.tensor(float((time.time() - t_job) / 60 > cfg.max_minutes),
                                        device=device)
                    if dist.is_initialized():
                        dist.broadcast(stop, 0)
                    if stop.item():
                        stopped_early = True
                        break
                if step >= cfg.steps:
                    break
                td = time.time()
            else:
                epoch += 1
                continue
            break

        if stopped_early:
            save_last()
            if is_main():
                print(f"[stop] time budget reached at step {step}; resubmit to resume", flush=True)
        elif is_main():
            atomic_save({"model": weights_bf16(model), "step": step, "cfg": asdict(cfg)},
                        run_dir / "final.pt")
            last.unlink(missing_ok=True)
            print(f"[done] step {step}, best val BC {best:.4f} -> {run_dir}", flush=True)
        if mf:
            mf.close()
    if wb:
        wb.finish()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for f in dataclasses.fields(Cfg):
        d = f.default
        p.add_argument(f"--{f.name}", default=d,
                       type=(lambda v: str(v).lower() in ("1", "true", "yes")) if isinstance(d, bool) else type(d))
    main(Cfg(**vars(p.parse_args())))
