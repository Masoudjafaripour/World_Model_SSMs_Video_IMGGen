"""Standalone smoke test for the W&B integration -- no GPUs, no sweep queue,
just confirms wandb.init/log/finish reaches the dashboard end to end. Use this
to isolate "is wandb itself working" from "is the training pipeline working".

  python test_wandb.py                     # online, live at wandb.ai
  python test_wandb.py --mode offline      # no network; `wandb sync` later
"""
from __future__ import annotations

import argparse
import math
import time

import wandb


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--project", default="video-wm-dist-training")
    p.add_argument("--mode", default="online", choices=["online", "offline", "disabled"])
    p.add_argument("--steps", type=int, default=50)
    a = p.parse_args()

    run = wandb.init(project=a.project, group="smoketest", job_type="test",
                     name=f"wandb-smoke-{int(time.time())}", mode=a.mode,
                     config={"steps": a.steps})
    print(f"[test_wandb] project={a.project} mode={a.mode}")
    print(f"[test_wandb] run url: {run.url or '(offline -- no url until synced)'}")

    for step in range(a.steps):
        loss = 2.0 * math.exp(-step / 15) + 0.05 * math.sin(step)
        wandb.log({"loss": loss, "fake_metric": math.cos(step / 5)}, step=step)
        time.sleep(0.05)

    wandb.finish()
    print("[test_wandb] done -- check the run url above")


if __name__ == "__main__":
    main()
