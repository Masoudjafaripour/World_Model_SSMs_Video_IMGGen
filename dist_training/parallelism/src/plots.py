"""Regenerate every figure from results.jsonl. Called after each arm, so the
sweep is inspectable while it runs and a kill never loses analysis.

Figures map onto the predictions in the README:
  fig1  throughput vs sequence length, per strategy          -> prediction 1, 2
  fig2  TP/DP communication ratio, analytic vs measured      -> the headline
  fig3  DP overhead vs gradient accumulation                 -> prediction 3
  fig4  pipeline bubble, measured vs (P-1)/(M+P-1)           -> prediction 4
  fig5  memory split, and which arms fit at all              -> prediction 5
  fig6  pretrain vs finetune regime flip                     -> the contribution
  fig7  cost-model calibration: predicted vs measured        -> contribution 1
  fig8  thermal / clock timeline                             -> validity check
  fig9  loss curves                                          -> sanity check only
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.autolayout": True})
C = {"single": "#444", "ddp": "#1f77b4", "zero2": "#17becf", "fsdp": "#2ca02c",
     "tp": "#d62728", "pp": "#ff7f0e", "tp_dp": "#9467bd"}


def load(p: Path) -> list:
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return [r for r in rows if r.get("ok")]


def _save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)


def fig_throughput_vs_seqlen(rows, out):
    for phase in ("pretrain", "finetune"):
        sub = [r for r in rows if r["phase"] == phase]
        if not sub:
            continue
        fig, ax = plt.subplots(figsize=(5, 3.4))
        g = defaultdict(list)
        for r in sub:
            g[r["strategy"]].append((r["seq_len"], r["tokens_per_s"]))
        for s, pts in sorted(g.items()):
            d = defaultdict(list)
            for x, y in pts:
                d[x].append(y)
            xs = sorted(d)
            ax.plot(xs, [sum(d[x]) / len(d[x]) for x in xs], "o-",
                    color=C.get(s, None), label=s)
        ax.set(xscale="log", yscale="log", xlabel="sequence length (tokens)",
               ylabel="tokens/s", title=f"Throughput vs sequence length ({phase})")
        ax.legend(fontsize=7)
        _save(fig, out, f"fig1_throughput_{phase}.png")


def fig_comm_ratio(rows, out):
    """Two panels, because the naive expectation and the true one differ.

    Left  eq(1)/(2): TP/DP communication ratio. FLAT in s under the fixed
          global-batch protocol -- s cancels against M. Separated by phase, it
          shows the 1/rho amplification, which is the real regime-flip signal.
    Right eq(3): fraction of step time spent on communication. This DECREASES
          with s, because attention FLOPs grow with s while TP's bytes per token
          do not. TP's relative overhead improves at long sequence length.
    """
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(8.2, 3.4))
    for phase, mk in (("pretrain", "o"), ("finetune", "s")):
        sub = sorted([r for r in rows if r["phase"] == phase and r.get("tp_dp_ratio")],
                     key=lambda r: r["seq_len"])
        if sub:
            axl.plot([r["seq_len"] for r in sub], [r["tp_dp_ratio"] for r in sub],
                     mk + "-", label=f"{phase} analytic")
    meas = defaultdict(dict)
    for r in rows:
        if r.get("comm_bytes_measured"):
            meas[(r["phase"], r["seq_len"])][r["strategy"]] = r["comm_bytes_measured"]
    xs = [(s, d["tp"] / d["ddp"]) for (ph, s), d in sorted(meas.items())
          if "tp" in d and d.get("ddp", 0) > 0]
    if xs:
        axl.scatter([x[0] for x in xs], [x[1] for x in xs], marker="x", c="k",
                    zorder=5, label="measured")
    axl.axhline(1, ls=":", c="gray")
    axl.set(xscale="log", yscale="log", xlabel="sequence length (tokens)",
            ylabel=r"$C_{TP}/C_{DP}$",
            title="(a) comm volume ratio — flat in $s$")
    axl.legend(fontsize=7)

    g = defaultdict(list)
    for r in rows:
        if r.get("comm_compute") is not None:
            g[r["strategy"]].append((r["seq_len"], r["comm_compute"]))
    for st, pts in sorted(g.items()):
        d = defaultdict(list)
        for x, y in pts:
            d[x].append(y)
        ks = sorted(d)
        axr.plot(ks, [sum(d[k]) / len(d[k]) for k in ks], "o-", color=C.get(st),
                 label=st)
    axr.set(xscale="log", yscale="log", xlabel="sequence length (tokens)",
            ylabel=r"$T_{comm}/T_{compute}$",
            title="(b) comm fraction — falls with $s$")
    axr.legend(fontsize=7)
    _save(fig, out, "fig2_comm_ratio.png")


def fig_dp_amortisation(rows, out):
    sub = [r for r in rows if r["strategy"] in ("ddp", "fsdp", "zero2")
           and r.get("accum")]
    if not sub:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    base = {r["seq_len"]: r["tokens_per_s"] for r in rows if r["strategy"] == "single"}
    g = defaultdict(list)
    for r in sub:
        b = base.get(r["seq_len"])
        if b:
            g[r["strategy"]].append((r["accum"], 1 - r["tokens_per_s"] / (b * r["world"])))
    for s, pts in sorted(g.items()):
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] * 100 for p in pts], "o-", color=C.get(s))
        ax.plot([], [], "o-", color=C.get(s), label=s)
    ax.axhline(5, ls=":", c="r", label="5% (prediction 3)")
    ax.set(xscale="log", xlabel=r"gradient accumulation $M$",
           ylabel="overhead vs ideal scaling (%)", title="DP comm amortisation")
    ax.legend(fontsize=7)
    _save(fig, out, "fig3_dp_amortisation.png")


def fig_pipeline_bubble(rows, out):
    sub = [r for r in rows if r["strategy"] == "pp" and r.get("accum")]
    if not sub:
        return
    fig, ax = plt.subplots(figsize=(5, 3.4))
    base = {r["seq_len"]: r["tokens_per_s"] for r in rows if r["strategy"] == "single"}
    pts = []
    for r in sub:
        b = base.get(r["seq_len"])
        if b:
            pts.append((r["accum"], 1 - r["tokens_per_s"] / (b * r["world"]), r["pp"]))
    pts.sort()
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o", c=C["pp"], label="measured")
    P_ = pts[0][2] if pts else 4
    ms = sorted({p[0] for p in pts})
    ax.plot(ms, [(P_ - 1) / (m + P_ - 1) for m in ms], "--", c="k",
            label=r"$(P-1)/(M+P-1)$")
    ax.set(xscale="log", xlabel=r"microbatches $M$", ylabel="idle fraction",
           title=f"Pipeline bubble (P={P_})")
    ax.legend(fontsize=7)
    _save(fig, out, "fig4_pipeline_bubble.png")


def fig_memory(rows, out):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    sub = sorted([r for r in rows if r.get("peak_alloc_gb")],
                 key=lambda r: (r["phase"], r["strategy"], r["seq_len"]))
    if not sub:
        return
    lbl = [f"{r['strategy']}·{r['phase'][:4]}·s{r['seq_len']//1024}k" for r in sub]
    ax.bar(range(len(sub)), [r["peak_alloc_gb"] for r in sub],
           color=[C.get(r["strategy"], "#999") for r in sub])
    ax.axhline(44, ls="--", c="r", lw=1, label="A6000 usable (~44 GB)")
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(lbl, rotation=90, fontsize=5)
    ax.set(ylabel="peak allocated (GB)", title="Peak memory per rank")
    ax.legend(fontsize=7)
    _save(fig, out, "fig5_memory.png")


def fig_regime_flip(rows, out):
    """The contribution: strategy ranking reverses between phases."""
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.2), sharey=True)
    for ax, phase in zip(axes, ("pretrain", "finetune")):
        sub = [r for r in rows if r["phase"] == phase]
        if not sub:
            continue
        g = defaultdict(list)
        for r in sub:
            g[r["strategy"]].append(r["tokens_per_s"])
        ks = sorted(g, key=lambda k: -sum(g[k]) / len(g[k]))
        base = max((sum(v) / len(v) for v in g.values()), default=1)
        ax.bar(range(len(ks)), [(sum(g[k]) / len(g[k])) / base for k in ks],
               color=[C.get(k, "#999") for k in ks])
        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels(ks, rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{phase}  (rho={sub[0].get('rho', 1):.3f})", fontsize=9)
    axes[0].set_ylabel("relative throughput")
    fig.suptitle("Regime flip: binding constraint moves weights -> activations",
                 fontsize=9)
    _save(fig, out, "fig6_regime_flip.png")


def fig_costmodel(rows, out):
    sub = [r for r in rows if r.get("pred_tokens_per_s") and r.get("tokens_per_s")]
    if not sub:
        return
    fig, ax = plt.subplots(figsize=(4.2, 4))
    for s in {r["strategy"] for r in sub}:
        d = [r for r in sub if r["strategy"] == s]
        ax.scatter([r["pred_tokens_per_s"] for r in d],
                   [r["tokens_per_s"] for r in d], s=18, color=C.get(s), label=s)
    lo = min(min(r["pred_tokens_per_s"], r["tokens_per_s"]) for r in sub)
    hi = max(max(r["pred_tokens_per_s"], r["tokens_per_s"]) for r in sub)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    err = sum(abs(r["pred_tokens_per_s"] - r["tokens_per_s"]) / r["tokens_per_s"]
              for r in sub) / len(sub)
    ax.set(xscale="log", yscale="log", xlabel="predicted tokens/s",
           ylabel="measured tokens/s",
           title=f"Cost-model calibration (MAPE {err*100:.1f}%)")
    ax.legend(fontsize=7)
    _save(fig, out, "fig7_costmodel.png")


def fig_thermal(run_dirs, out):
    import csv
    fig, ax = plt.subplots(figsize=(6, 3))
    ax2 = ax.twinx()
    n = 0
    for d in sorted(run_dirs)[:12]:
        f = d / "gpu_telemetry.csv"
        if not f.exists():
            continue
        by = defaultdict(list)
        with open(f) as fh:
            for row in csv.DictReader(fh):
                try:
                    by[row["gpu"]].append((float(row["t"]), float(row["temp_c"]),
                                           float(row["sm_clock_mhz"])))
                except (ValueError, KeyError):
                    pass
        for gpu, v in by.items():
            ax.plot([p[0] for p in v], [p[1] for p in v], lw=0.7, alpha=0.7)
            ax2.plot([p[0] for p in v], [p[2] for p in v], lw=0.5, alpha=0.3, c="gray")
            n += 1
    if not n:
        plt.close(fig)
        return
    ax.axhline(83, ls="--", c="r", lw=1)
    ax.set(xlabel="s within run", ylabel="temp (C)",
           title="Thermal / clock stability (gray = SM clock)")
    ax2.set_ylabel("SM clock (MHz)")
    _save(fig, out, "fig8_thermal.png")


def fig_loss(run_dirs, out):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    n = 0
    for d in sorted(run_dirs):
        f = d / "loss.jsonl"
        if not f.exists():
            continue
        pts = [json.loads(x) for x in f.read_text().splitlines() if x.strip()]
        if len(pts) < 5:
            continue
        ax.plot([p["step"] for p in pts], [p["loss"] for p in pts], lw=0.9,
                alpha=0.8, label=d.name[:34])
        n += 1
    if not n:
        plt.close(fig)
        return
    ax.set(xlabel="step", ylabel="flow-matching loss",
           title="Loss — sanity check only (global batch held constant)")
    if n <= 8:
        ax.legend(fontsize=5)
    _save(fig, out, "fig9_loss.png")


def summary_table(rows, out: Path):
    if not rows:
        return
    cols = ["key", "phase", "strategy", "seq_len", "accum", "tokens_per_s", "mfu",
            "peak_alloc_gb", "comm_bytes_measured", "comm_bytes_predicted",
            "step_time", "step_time_std", "throttled", "final_loss"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "---|" * len(cols)]
    for r in sorted(rows, key=lambda r: (r["phase"], r["strategy"], r["seq_len"])):
        lines.append("| " + " | ".join(
            f"{r.get(c):.4g}" if isinstance(r.get(c), float) else str(r.get(c))
            for c in cols) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/main/results.jsonl")
    ap.add_argument("--out", default="results/main/plots")
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    rows = load(res)
    out.mkdir(parents=True, exist_ok=True)
    run_dirs = [d for d in (res.parent / "runs").glob("*") if d.is_dir()]

    fig_throughput_vs_seqlen(rows, out)
    fig_comm_ratio(rows, out)
    fig_dp_amortisation(rows, out)
    fig_pipeline_bubble(rows, out)
    fig_memory(rows, out)
    fig_regime_flip(rows, out)
    fig_costmodel(rows, out)
    fig_thermal(run_dirs, out)
    fig_loss(run_dirs, out)
    summary_table(rows, out)
    print(f"[plots] {len(rows)} ok rows -> {out}")


if __name__ == "__main__":
    main()
