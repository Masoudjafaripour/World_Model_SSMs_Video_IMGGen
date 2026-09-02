"""Measurement instrumentation.

Three things are logged that are easy to omit and expensive to omit:
  1. per-rank MFU, not aggregate throughput -- the only way to compare arms that
     place different amounts of work on each device;
  2. clock and throttle state, sampled continuously -- a throttling GPU looks
     exactly like communication overhead in a throughput plot;
  3. idle fraction per rank -- this is what exposes the pipeline bubble and
     data-parallel load imbalance directly rather than by inference.
"""
from __future__ import annotations

import csv
import json
import subprocess
import threading
import time
from pathlib import Path

import torch


# --------------------------------------------------------------------------- #
# GPU telemetry, sampled in a background thread for the whole run
# --------------------------------------------------------------------------- #
_FIELDS = ("index,temperature.gpu,power.draw,clocks.sm,utilization.gpu,"
           "memory.used,clocks_throttle_reasons.active")


class GpuMonitor:
    def __init__(self, out_csv: Path, period: float = 2.0):
        self.out, self.period, self._stop = Path(out_csv), period, threading.Event()
        self.t = None

    def _poll(self):
        with open(self.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t", "gpu", "temp_c", "power_w", "sm_clock_mhz",
                        "util_pct", "mem_used_mb", "throttle_hex"])
            t0 = time.time()
            while not self._stop.is_set():
                try:
                    r = subprocess.run(
                        ["nvidia-smi", f"--query-gpu={_FIELDS}",
                         "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, timeout=5)
                    for line in r.stdout.strip().splitlines():
                        w.writerow([round(time.time() - t0, 1)]
                                   + [c.strip() for c in line.split(",")])
                    f.flush()
                except Exception:
                    pass
                self._stop.wait(self.period)

    def __enter__(self):
        self.t = threading.Thread(target=self._poll, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.t:
            self.t.join(timeout=5)


class LiveGpuPlot:
    """Background thread that regenerates a per-GPU utilization/memory PNG from
    the telemetry CSV as GpuMonitor writes it, so a running arm is watchable
    live (e.g. during --dry_run) instead of only after it finishes."""

    def __init__(self, csv_path: Path, out_png: Path, period: float = 5.0,
                wandb_run=None):
        self.csv_path, self.out_png = Path(csv_path), Path(out_png)
        self.period, self._stop = period, threading.Event()
        self.wandb_run = wandb_run          # optional: also stream the PNG to W&B
        self.t = None

    def _read(self) -> dict:
        by: dict = {}
        if not self.csv_path.exists():
            return by
        try:
            with open(self.csv_path) as f:
                for row in csv.DictReader(f):
                    try:
                        g = row["gpu"]
                        by.setdefault(g, []).append(
                            (float(row["t"]), float(row["util_pct"]),
                             float(row["mem_used_mb"])))
                    except (ValueError, KeyError):
                        continue
        except Exception:
            pass
        return by

    def _render(self):
        by = self._read()
        if not by:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5), sharex=True)
        for gpu, v in sorted(by.items(), key=lambda kv: int(kv[0])):
            t = [p[0] for p in v]
            ax1.plot(t, [p[1] for p in v], label=f"gpu{gpu}", lw=1)
            ax2.plot(t, [p[2] for p in v], label=f"gpu{gpu}", lw=1)
        ax1.set_ylabel("utilization (%)")
        ax1.set_ylim(0, 100)
        ax1.legend(fontsize=7, ncol=min(len(by), 4))
        ax2.set_ylabel("memory used (MiB)")
        ax2.set_xlabel("s within run")
        fig.suptitle("Live per-GPU usage (updates while training)")
        self.out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.out_png)
        plt.close(fig)
        if self.wandb_run is not None:
            try:
                import wandb
                self.wandb_run.log({"gpu_usage_live": wandb.Image(str(self.out_png))})
            except Exception:
                pass

    def _loop(self):
        while not self._stop.is_set():
            try:
                self._render()
            except Exception:
                pass
            self._stop.wait(self.period)

    def __enter__(self):
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self.t:
            self.t.join(timeout=5)
        try:
            self._render()          # final frame reflecting the full run
        except Exception:
            pass


def throttled(csv_path: Path) -> bool:
    """True if any sample reported a non-zero throttle reason other than idle."""
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                v = row.get("throttle_hex", "0x0")
                if v and int(v, 16) & ~0x1:   # bit 0 = GpuIdle, benign
                    return True
    except Exception:
        pass
    return False


# --------------------------------------------------------------------------- #
# NCCL byte counters
# --------------------------------------------------------------------------- #
class CommCounter:
    """Wraps c10d collectives to accumulate measured bytes, so the analytic
    model in costmodel.py can be checked rather than trusted."""

    def __init__(self):
        self.bytes = 0
        self._orig = {}

    def __enter__(self):
        import torch.distributed as dist
        for name in ("all_reduce", "all_gather_into_tensor",
                     "reduce_scatter_tensor", "broadcast"):
            fn = getattr(dist, name, None)
            if fn is None:
                continue
            self._orig[name] = fn

            def wrap(_fn=fn, _self=self):
                def inner(tensor, *a, **k):
                    try:
                        _self.bytes += tensor.numel() * tensor.element_size()
                    except Exception:
                        pass
                    return _fn(tensor, *a, **k)
                return inner
            setattr(dist, name, wrap())
        return self

    def __exit__(self, *exc):
        import torch.distributed as dist
        for name, fn in self._orig.items():
            setattr(dist, name, fn)


# --------------------------------------------------------------------------- #
# Timing / MFU
# --------------------------------------------------------------------------- #
class StepTimer:
    """CUDA-synchronised steady-state timing. Warmup steps are discarded:
    allocator growth, cuDNN autotune and NCCL warmup make early steps
    unrepresentative, and that is true after a resume as well as a cold start."""

    def __init__(self, warmup: int = 20):
        self.warmup, self.n, self.times = warmup, 0, []
        self._t = None

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.n += 1
        if self.n > self.warmup:
            self.times.append(time.perf_counter() - self._t)

    def stats(self) -> dict:
        if not self.times:
            return {"step_time": float("nan"), "step_time_p50": float("nan"),
                    "step_time_std": float("nan"), "n_timed": 0}
        t = sorted(self.times)
        mean = sum(t) / len(t)
        var = sum((x - mean) ** 2 for x in t) / max(len(t) - 1, 1)
        return {"step_time": mean, "step_time_p50": t[len(t) // 2],
                "step_time_std": var ** 0.5, "n_timed": len(t)}


def mfu(tokens_per_s: float, flops_per_token: float, peak: float, n_gpus: int) -> float:
    return tokens_per_s * flops_per_token / (peak * n_gpus)


def memory_stats() -> dict:
    if not torch.cuda.is_available():
        return {"peak_alloc_gb": 0.0, "peak_reserved_gb": 0.0}
    return {"peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9,
            "peak_reserved_gb": torch.cuda.max_memory_reserved() / 1e9}


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
