"""Sweep driver: a resumable job queue over arms.

Resume is at *arm* granularity, which is the granularity that matters. Each arm
is ~300 steps, so a kill loses at most one arm. Mid-run resume (DCP) is enabled
only for the long convergence run, where it is actually needed -- see
--save_every in train.py.

  python src/sweep.py --config configs/sweep.yaml            # run / resume
  python src/sweep.py --config configs/sweep.yaml --dry_run  # 20 steps per arm
  python src/sweep.py --config configs/sweep.yaml --list     # show queue state
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import yaml

try:
    from tqdm import tqdm
except ImportError:                        # optional dependency; sweep still runs
    def tqdm(it, *a, **k):
        return it

SRC = Path(__file__).parent


def load_done(results: Path) -> set:
    done = set()
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                r = json.loads(line)
                if r.get("ok") or r.get("oom"):     # OOM is a valid, final result
                    done.add(r["key"])
            except json.JSONDecodeError:
                continue
    return done


def layouts(strategy: str, n: int):
    """Valid (dp, tp, pp) for a strategy on n GPUs."""
    if strategy == "single":
        return [(1, 1, 1)]
    if strategy in ("ddp", "zero2", "fsdp"):
        return [(n, 1, 1)]
    if strategy == "tp":
        return [(1, n, 1)]
    if strategy == "pp":
        return [(1, 1, n)]
    if strategy == "tp_dp":
        return [(n // t, t, 1) for t in (2,) if n % t == 0 and n // t > 1]
    return []


def build_queue(c: dict) -> list:
    arms = []
    for phase in c["phases"]:
        for size in c["model_sizes"]:
            for strat in c["strategies"]:
                for dp, tp, pp in layouts(strat, c["n_gpus"]):
                    for s in c["seq_lens"]:
                        for mb in c["micro_bs"]:
                            for ck in c["checkpointing"]:
                                for seed in range(c["seeds"]):
                                    arms.append(dict(
                                        phase=phase, model_size=size, strategy=strat,
                                        seq_len=s, micro_bs=mb, dp=dp, tp=tp, pp=pp,
                                        ckpt=ck, seed=seed, **c.get("common", {})))
    return arms


def key_of(a: dict) -> str:
    return (f"{a['phase']}_{a['model_size']}_{a['strategy']}_s{a['seq_len']}"
            f"_b{a['micro_bs']}_dp{a['dp']}tp{a['tp']}pp{a['pp']}"
            f"_{'ckpt' if a['ckpt'] else 'nockpt'}_seed{a['seed']}")


def estimate_hours(arms: list, c: dict) -> float:
    """Predicted wall-clock for a queue, so the booking is sized before it is
    made rather than after. Uses the same cost model the experiment validates."""
    sys.path.insert(0, str(SRC))
    import costmodel as cm
    from train import SIZES as sizes          # single source of truth
    steps, tot = c.get("steps", 60), 0.0
    for a in arms:
        m = sizes[a["model_size"]]
        N = m.n_params()
        nt = N if a["phase"] == "pretrain" else int(N * 0.015)
        gb = a.get("global_batch_tokens", c.get("common", {}).get(
            "global_batch_tokens", 524288))
        acc = max(1, gb // (a["micro_bs"] * a["seq_len"] * max(a["dp"], 1)))
        r = cm.predict_step_time(
            a["strategy"], cm.Machine(n_gpus=a["dp"] * a["tp"] * a["pp"]),
            n_params=N, n_trainable=nt, depth=m.depth, dim=m.dim,
            seq_len=a["seq_len"], micro_bs=a["micro_bs"], accum=acc,
            dp=a["dp"], tp=a["tp"], pp=a["pp"], checkpointing=a["ckpt"])
        tot += r["step_time"] * steps + 60      # + process startup
    return tot / 3600


def launch(arm: dict, n_gpus: int, steps: int, out: str, wandb: bool = False,
          wandb_project: str = "video-wm-dist-training", wandb_group: str = "",
          wandb_mode: str = "online") -> int:
    world = arm["dp"] * arm["tp"] * arm["pp"]
    args = [f"--{k}={v}" for k, v in arm.items()]
    args += [f"--steps={steps}", f"--out_dir={out}",
             f"--wandb={wandb}", f"--wandb_project={wandb_project}",
             f"--wandb_group={wandb_group or Path(out).name}",
             f"--wandb_mode={wandb_mode}"]
    if world > 1:
        cmd = [sys.executable, "-m", "torch.distributed.run",
               f"--nproc_per_node={world}", "--standalone",
               str(SRC / "train.py"), *args]
    else:
        cmd = [sys.executable, str(SRC / "train.py"), *args]
    env = {**os.environ, "PYTHONPATH": str(SRC), "OMP_NUM_THREADS": "8",
           "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1"}
    return subprocess.run(cmd, env=env).returncode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/sweep.yaml")
    p.add_argument("--dry_run", action="store_true",
                   help="20 steps per arm: catches every OOM/hang/typo in ~2h")
    p.add_argument("--list", action="store_true")
    p.add_argument("--cooldown", type=float, default=20.0,
                   help="seconds between arms so thermal drift does not "
                        "correlate with arm ordering")
    p.add_argument("--wandb", action="store_true",
                   help="log each arm to Weights & Biases (opt-in, off by default)")
    p.add_argument("--wandb_project", default="video-wm-dist-training")
    p.add_argument("--wandb_group", default="",
                   help="defaults to the out_dir basename (dryrun/main) so all "
                        "arms of one sweep group together in the W&B UI")
    p.add_argument("--wandb_mode", default="online",
                   choices=["online", "offline", "disabled"],
                   help="use offline if this node has no internet access")
    a = p.parse_args()

    c = yaml.safe_load(Path(a.config).read_text())
    out = Path(c.get("out_dir", "results")) / ("dryrun" if a.dry_run else "main")
    out.mkdir(parents=True, exist_ok=True)
    results = out / "results.jsonl"

    queue = build_queue(c)
    done = load_done(results)
    todo = [x for x in queue if key_of(x) not in done]

    # Randomised order: thermal drift must not align with condition.
    random.Random(c.get("order_seed", 0)).shuffle(todo)

    if a.list:
        print(f"{len(queue)} arms, {len(done)} done, {len(todo)} remaining")
        print(f"  est. {estimate_hours(todo, c):.1f} GPU-h  "
              f"(+ ~40% for OOM bisection, reruns, cooldown)")
        for x in todo[:40]:
            print("  ", key_of(x))
        return

    steps = 20 if a.dry_run else c.get("steps", 300)
    print(f"[sweep] {len(todo)}/{len(queue)} arms remaining -> {out}")
    t0 = time.time()
    pbar = tqdm(todo, desc="sweep", unit="arm", dynamic_ncols=True)
    for i, arm in enumerate(pbar, 1):
        k = key_of(arm)
        if hasattr(pbar, "set_postfix_str"):
            pbar.set_postfix_str(k)
        print(f"\n[{i}/{len(todo)}] {k}", flush=True)
        rc = launch(arm, c["n_gpus"], steps, str(out), wandb=a.wandb,
                   wandb_project=a.wandb_project, wandb_group=a.wandb_group,
                   wandb_mode=a.wandb_mode)
        if rc != 0:
            # Record the failure so resume does not retry it forever.
            with open(results, "a") as f:
                f.write(json.dumps({"key": k, **arm, "ok": False,
                                    "error": f"launcher rc={rc}"}) + "\n")
        subprocess.run([sys.executable, str(SRC / "plots.py"),
                        "--results", str(results), "--out", str(out / "plots")])
        time.sleep(a.cooldown)
    print(f"\n[sweep] complete in {(time.time()-t0)/3600:.2f} h")


if __name__ == "__main__":
    main()
