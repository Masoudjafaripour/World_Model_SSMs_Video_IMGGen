"""Analytic cost model — the part that makes this a contribution rather than a
benchmark table.

Per optimizer step, data parallelism moves a volume set by the *trainable
parameter count*, once; tensor parallelism moves a volume set by the *tokens
processed*, at every layer. With |theta| ~ 12 L d^2 and rho the trainable
fraction:

    C_TP / C_DP  ~  (b * s * M_accum) / (3 * d * rho)                      (1)

IMPORTANT -- read this before interpreting any figure. Equation (1) is stated
per step at fixed (b, s, M). This study holds the *global batch in tokens*
constant, B = b * s * M * dp, which is what makes the arms comparable. Under
that protocol M scales as 1/s, the s cancels, and (1) collapses to

    C_TP / C_DP  ~  B / (3 * d * dp * rho)                                 (2)

i.e. INDEPENDENT of sequence length. Tensor-parallel communication is set by
tokens processed, and a fixed token budget is a fixed token budget however it is
shaped. The naive expectation that "longer sequences punish TP" is wrong under a
fixed-batch protocol; it only holds if the token budget grows with s.

What genuinely varies with s is the comm-to-COMPUTE ratio, because attention
FLOPs grow with s while TP's bytes per token do not:

    T_comm / T_compute  ~  8 L d / (6N + 12 L s d)                         (3)

which DECREASES with s. So TP's relative overhead improves at long sequence
length, while its motivation -- activation memory, linear in s -- strengthens.
Both effects favour TP as s grows; only the interconnect works against it.
Equation (3) and the rho-amplification in (2) are the two testable claims.
"""
from __future__ import annotations

from dataclasses import dataclass

BF16 = 2  # bytes

# Dense tensor-core peaks, no sparsity.
PEAK_FLOPS = {"a6000": 154.8e12, "v100": 125.0e12, "a100": 312e12, "h100": 989e12}


def ring_factor(n: int) -> float:
    """Bus volume per rank for ring all-reduce, in units of the payload."""
    return 0.0 if n <= 1 else 2 * (n - 1) / n


# --------------------------------------------------------------------------- #
# FLOPs
# --------------------------------------------------------------------------- #
def train_flops_per_token(n_params: int, depth: int, seq_len: int, dim: int) -> float:
    """6N for the weights + 12*L*s*d for attention score/value matmuls."""
    return 6 * n_params + 12 * depth * seq_len * dim


# --------------------------------------------------------------------------- #
# Communication volume, bytes moved per rank per optimizer step
# --------------------------------------------------------------------------- #
def comm_bytes(strategy: str, *, n_params: int, n_trainable: int, depth: int,
               dim: int, seq_len: int, micro_bs: int, accum: int,
               dp: int = 1, tp: int = 1, pp: int = 1) -> float:
    tok = micro_bs * seq_len * dim * BF16          # one activation tensor
    s = strategy.lower()
    if s in ("single", "1gpu"):
        return 0.0
    if s == "ddp":                                  # one grad all-reduce per step
        return ring_factor(dp) * n_trainable * BF16
    if s in ("fsdp", "zero3"):                      # 2 all-gathers + 1 reduce-scatter
        return (2 * (dp - 1) / dp * n_params * BF16
                + (dp - 1) / dp * n_trainable * BF16)
    if s == "zero2":                                # grad reduce-scatter + param all-gather
        return 2 * (dp - 1) / dp * n_trainable * BF16
    if s == "tp":                                   # 2 all-reduces fwd + 2 bwd, per block
        return ring_factor(tp) * 4 * depth * tok * accum
    if s == "pp":                                   # p2p activations at each boundary
        return 2 * (pp - 1) * tok * accum
    if s == "tp_dp":
        return (ring_factor(tp) * 4 * depth * tok * accum
                + ring_factor(dp) * n_trainable * BF16)
    raise ValueError(f"unknown strategy {strategy}")


def tp_dp_ratio(seq_len: int, dim: int, micro_bs: int, accum: int, rho: float) -> float:
    """Equation (1): per-step TP/DP communication ratio at fixed (b, s, M).
    Under the fixed-global-batch protocol this is s-independent -- see (2)."""
    return (micro_bs * seq_len * accum) / (3 * dim * max(rho, 1e-9))


def comm_compute_ratio(strategy: str, *, n_params, n_trainable, depth, dim,
                       seq_len, micro_bs, accum, dp=1, tp=1, pp=1,
                       busbw=20e9, peak=PEAK_FLOPS["a6000"], mfu=0.35,
                       n_gpus=4) -> float:
    """Equation (3) generalised: fraction of step time spent moving bytes.
    This is the quantity that actually varies with sequence length."""
    cb = comm_bytes(strategy, n_params=n_params, n_trainable=n_trainable,
                    depth=depth, dim=dim, seq_len=seq_len, micro_bs=micro_bs,
                    accum=accum, dp=dp, tp=tp, pp=pp)
    tokens = micro_bs * seq_len * accum * dp
    fl = train_flops_per_token(n_params, depth, seq_len, dim) * tokens
    return (cb / busbw) / (fl / (peak * mfu * n_gpus))


# --------------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------------- #
def memory_bytes(strategy: str, *, n_params: int, n_trainable: int, depth: int,
                 dim: int, seq_len: int, micro_bs: int, dp: int = 1, tp: int = 1,
                 pp: int = 1, checkpointing: bool = False,
                 act_bytes_per_token_layer: int = 16) -> dict:
    """Rough but calibrated split into weights / optimizer / activations."""
    s = strategy.lower()
    w = n_params * BF16
    # Adam: fp32 master + m + v, on trainable params only.
    opt = n_trainable * 12 + n_trainable * BF16

    if s in ("fsdp", "zero3"):
        w, opt = w / dp, opt / dp
    elif s == "zero2":
        opt = opt / dp
    elif s in ("tp", "tp_dp"):
        w, opt = w / tp, opt / tp
    elif s == "pp":
        w, opt = w / pp, opt / pp

    layers_local = depth / (pp if s == "pp" else 1)
    act = act_bytes_per_token_layer * micro_bs * seq_len * dim * layers_local
    if s in ("tp", "tp_dp"):
        act /= tp                       # with sequence parallelism on norm regions
    if checkpointing:
        act = act / max(layers_local, 1) * (layers_local ** 0.5) + act * 0.05
    return {"weights": w, "optimizer": opt, "activations": act,
            "total": w + opt + act}


# --------------------------------------------------------------------------- #
# Predicted step time, and the configuration selector
# --------------------------------------------------------------------------- #
@dataclass
class Machine:
    n_gpus: int = 4
    peak_flops: float = PEAK_FLOPS["a6000"]
    mfu: float = 0.35             # calibrate from a single-GPU run
    busbw: float = 20e9           # B/s, from nccl-tests all_reduce_perf
    vram: float = 48e9
    reserve: float = 4e9          # allocator + fragmentation headroom


def predict_step_time(strategy: str, m: Machine, *, n_params, n_trainable, depth,
                      dim, seq_len, micro_bs, accum, dp=1, tp=1, pp=1,
                      checkpointing=False) -> dict:
    tokens = micro_bs * seq_len * accum * dp
    fl = train_flops_per_token(n_params, depth, seq_len, dim) * tokens
    t_compute = fl / (m.peak_flops * m.mfu * m.n_gpus)
    if checkpointing:
        t_compute *= 1.33                      # one extra forward
    cb = comm_bytes(strategy, n_params=n_params, n_trainable=n_trainable,
                    depth=depth, dim=dim, seq_len=seq_len, micro_bs=micro_bs,
                    accum=accum, dp=dp, tp=tp, pp=pp)
    t_comm = cb / m.busbw
    # DP/FSDP collectives overlap with backward; TP sits on the critical path.
    overlap = {"ddp": 0.85, "fsdp": 0.7, "zero2": 0.8, "zero3": 0.7,
               "pp": 0.9, "tp": 0.0, "tp_dp": 0.3, "single": 1.0}.get(strategy, 0.0)
    t = t_compute + t_comm * (1 - overlap)
    if strategy == "pp":                        # bubble
        t /= max(1e-9, 1 - (pp - 1) / (accum + pp - 1))
    mem = memory_bytes(strategy, n_params=n_params, n_trainable=n_trainable,
                       depth=depth, dim=dim, seq_len=seq_len, micro_bs=micro_bs,
                       dp=dp, tp=tp, pp=pp, checkpointing=checkpointing)
    return {"step_time": t, "t_compute": t_compute, "t_comm": t_comm,
            "comm_bytes": cb, "tokens": tokens, "tokens_per_s": tokens / t,
            "mem_total": mem["total"], "fits": mem["total"] < m.vram - m.reserve}


def select_config(m: Machine, *, n_params, n_trainable, depth, dim, seq_len,
                  target_batch_tokens: int, candidates=None) -> list:
    """Enumerate feasible (strategy, dp, tp, pp, micro_bs, accum, ckpt) and rank
    by predicted throughput. Contribution 1: this should match exhaustive
    measurement on held-out configurations."""
    out = []
    cands = candidates or ["ddp", "zero2", "fsdp", "tp", "pp", "tp_dp"]
    for strat in cands:
        for tp in (1, 2, 4):
            for pp in (1, 2, 4):
                dp = m.n_gpus // (tp * pp)
                if dp < 1 or tp * pp * dp != m.n_gpus:
                    continue
                if strat == "tp" and (tp == 1 or pp != 1 or dp != 1):
                    continue
                if strat == "pp" and (pp == 1 or tp != 1 or dp != 1):
                    continue
                if strat == "tp_dp" and (tp == 1 or dp == 1 or pp != 1):
                    continue
                if strat in ("ddp", "zero2", "fsdp") and (tp != 1 or pp != 1):
                    continue
                for ckpt in (False, True):
                    for mb in (1, 2, 4, 8):
                        acc = max(1, target_batch_tokens // (mb * seq_len * max(dp, 1)))
                        r = predict_step_time(strat, m, n_params=n_params,
                                              n_trainable=n_trainable, depth=depth,
                                              dim=dim, seq_len=seq_len, micro_bs=mb,
                                              accum=acc, dp=dp, tp=tp, pp=pp,
                                              checkpointing=ckpt)
                        if not r["fits"]:
                            continue
                        out.append({"strategy": strat, "dp": dp, "tp": tp, "pp": pp,
                                    "micro_bs": mb, "accum": acc, "ckpt": ckpt, **r})
    return sorted(out, key=lambda r: -r["tokens_per_s"])
