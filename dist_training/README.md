# Parallelism for video world models: pretraining vs fine-tuning

Video world models sit at an unusual point in the parallelism design space:
sequence lengths of ~10<sup>4</sup> latent tokens make **activations**, not
parameters, the dominant memory cost, while parameter counts stay modest. This
inverts guidance derived from LLM training — and inverts again between
pretraining and parameter-efficient fine-tuning.

Target hardware: **4× RTX A6000, single node, PCIe**. That is the regime most
academic labs actually have, and the one the parallelism literature does not
address.

---

## Correction to a natural first guess

It is tempting to reason: *TP communication is proportional to tokens, DP
communication is proportional to parameters, so longer sequences must punish
TP.* Under this study's protocol that is **wrong**, and the code surfaced it
before any GPU time was spent. Worth stating up front, because the figures only
make sense with it.

Per step at fixed (b, s, M), with |θ| ≈ 12Ld² and ρ the trainable fraction:

```
C_TP / C_DP  ≈  (b · s · M) / (3 · d · ρ)                     (1)
```

But the arms are only comparable if the **global batch in tokens** is held
constant, B = b·s·M·dp. Under that constraint M ∝ 1/s, the s cancels, and (1)
becomes

```
C_TP / C_DP  ≈  B / (3 · d · dp · ρ)                          (2)
```

**independent of sequence length.** TP communication is set by tokens processed,
and a fixed token budget is a fixed token budget however it is shaped.

What *does* vary with s is the comm-to-**compute** ratio, because attention
FLOPs grow with s while TP's bytes per token do not:

```
T_comm / T_compute  ≈  8·L·d / (6N + 12·L·s·d)                (3)
```

which **decreases** with s. So TP's relative overhead improves at long sequence
length, while its motivation (activation memory, linear in s) strengthens. Both
effects favour TP as s grows. Only the interconnect works against it.

The two testable claims are therefore **(3)**, and the **1/ρ amplification** in
(2) — which is the regime flip.

| | pretraining (ρ=1) | LoRA fine-tune (ρ≈0.015) |
|---|---|---|
| C_TP/C_DP, model S, B=512k | **171×** | **11 400×** |
| binding constraint | optimizer state + gradient comm | activation memory |
| right answer | ZeRO-2 / FSDP | DDP is nearly free; use checkpointing |

---

## Install

```bash
pip install -r requirements.txt      # torch>=2.6, numpy, pyyaml, matplotlib
```

## Run

```bash
bash preflight.sh                                   # 1. validate hardware
python sweep.py --config sweep.yaml --list          # 2. see queue + GPU-h
python sweep.py --config sweep.yaml --dry_run       # 3. 20 steps/arm, ~2h
python sweep.py --config sweep.yaml                 # 4. the real sweep
python plots.py --results results/main/results.jsonl --out results/main/plots
```

Step 3 is not optional. It catches every OOM, hang and config typo before the
exclusive window opens. Going into a 5-day booking without it is how the booking
becomes 10 days.

**Resume is automatic.** Kill the sweep at any point and rerun the same command;
completed arms are skipped. Granularity is one arm (~10 min), which is the
granularity that matters. Mid-run resume uses `torch.distributed.checkpoint`
(resharding-aware, so a TP=4 checkpoint reloads under TP=2) and is enabled only
via `--save_every` for the single long convergence run, where it is actually
needed.

---

## Layout

```
model.py           Latent video DiT + LoRA. Heads inferred from runtime width,
                   so TP sharding works without the module knowing its degree.
data.py            Memory-mapped latents. Synthetic (default) or real.
parallel.py        single | ddp | zero2 | fsdp | tp | pp | tp_dp
costmodel.py       Eqs (1)-(3), memory model, and the config selector
metrics.py         MFU, NCCL byte counters, GPU telemetry, step timing
train.py           One arm
sweep.py           Resumable queue + GPU-hour estimator
plots.py           9 figures + summary.md, regenerated after every arm
encode_latents.py  VAE latent encoding (run on the V100s)
preflight.sh       Topology, ECC, P2P, bus bandwidth
```

### Results tree

```
results/main/
  results.jsonl              append-only, one row per arm — the resume ledger
  runs/<arm_key>/
    record.json              full config + all metrics
    loss.jsonl               loss trace (sanity check only)
    gpu_telemetry.csv        temp / power / clock / throttle, 2 s sampling
    dcp/                     mid-run checkpoint, if --save_every
  plots/
    fig1_throughput_{pretrain,finetune}.png
    fig2_comm_ratio.png      (a) volume ratio, flat in s  (b) comm fraction, falls
    fig3_dp_amortisation.png
    fig4_pipeline_bubble.png
    fig5_memory.png
    fig6_regime_flip.png     ← the contribution
    fig7_costmodel.png       predicted vs measured, with MAPE
    fig8_thermal.png
    fig9_loss.png
    summary.md
```

---

## Protocol

TP and PP are mathematically exact. **DP changes the global batch and therefore
the model you get.** So `global_batch_tokens` is held constant across every arm
and the difference is absorbed into gradient accumulation. Quality is then
identical by construction, loss is a *sanity check* rather than a result, and
arms differ only in wall-clock, memory and utilisation.

If you instead hold micro-batch constant, you conflate systems throughput with
optimization and no comparison is interpretable.

### Models

| | dim | depth | params | single-GPU memory @ s=8192 |
|---|---|---|---|---|
| S | 1024 | 24 | 464 M | 10.6 GB — fits, so this measures *overhead* |
| L | 2048 | 32 | 2.44 B | 47.6 GB — **does not fit**, so sharding is forced |

Model L exceeding the ~44 GB usable ceiling is deliberate: it is what makes FSDP
necessary rather than optional.

### Budget

```
sweep       112 arms    76 GPU-h    main grid
ablations    40 arms    18 GPU-h    checkpointing × micro-batch at s=8192
variance     12 arms     5 GPU-h    3 seeds — establishes the noise floor first
                        ─────────
                        99 GPU-h  →  ~5.8 days serial with 40% overhead
```

`--list` prints this from the cost model for whatever config you give it, so
the booking is sized before it is made.

**Run `variance.yaml` first.** If seed-to-seed spread exceeds an effect you
later want to claim, that effect is not real.

---

## Predictions (write these down before running)

1. **TP=4 across root complexes** is much worse than TP2×DP2 — possibly worse
   than the single-GPU baseline. Check `nvidia-smi topo -m` first: if the A6000s
   are on two root complexes, flat TP=4 crosses the CPU link.
2. **TP's comm/compute fraction falls with s** (eq. 3), from ~1.0 at s=1024 to
   ~0.6 at s=8192 — *not* the naive linear-in-s penalty. DP stays flat and ~200×
   lower.
3. **DP overhead drops below 5%** once M ≳ 8. TP has no equivalent knob: its
   comm is per-layer, per-microbatch, on the critical path, and never amortises.
4. **PP bubble tracks (P−1)/(M+P−1)** within ~20%.
5. **On model L, FSDP is the only arm that trains without heroics.**
6. **Gradient checkpointing buys more memory per unit slowdown** (~33% time for
   ~5× activation memory) than TP does on PCIe.

If 2, 5 and the ρ-amplification hold, the study is complete without a larger
cluster.

---

## Validity traps

- **Lock clocks.** `sudo nvidia-smi -i 0,1,2,3 -lgc <freq> && -pm 1`. Four
  A6000s at 300 W in one chassis will throttle, and throttling looks exactly
  like communication overhead in a throughput plot. Arm order is randomised
  (`order_seed`) so thermal drift cannot correlate with condition, and
  `fig8_thermal.png` plus the `throttled` column exist to catch it anyway.
- **Normalise ECC.** GPUs reporting 46068 vs 49140 MiB have different ECC
  settings; a 3 GB ceiling difference corrupts every OOM-threshold comparison.
- **Confirm P2P.** A6000 is professional silicon so P2P works over PCIe — unlike
  consumer RTX cards, where the driver disables it and NCCL silently stages
  through host memory. `NCCL_DEBUG=INFO` tells you which path was taken; if you
  don't know, you can't interpret the number.
- **Discard warmup after a resume**, not just at cold start. Allocator state,
  cuDNN autotune and NCCL warmup all reset. `StepTimer` is re-armed after every
  DCP save for this reason.
- **Never `DataParallel`.** Single-process, replicates every step, gives
  plausible-looking garbage.
- **Don't mix in the V100s.** No bf16, no FlashAttention-2, ~half the throughput.
  DP would pin to the slowest rank and MFU would not be comparable across
  devices. Use them for latent encoding, or for a separate *homogeneous* 4×V100
  replicate — that second replicate is genuinely valuable, since same-code
  different-interconnect is what isolates causation.

---

## Data

Default is `synthetic`: procedurally generated latent trajectories with exact
action labels. No download, unlimited, exact control over s, fully reproducible.
Since dataset identity affects none of the six predictions, this removes three
failure modes at zero scientific cost.

For one real-video replicate:

```bash
# pretrain source — general video, no actions
CUDA_VISIBLE_DEVICES=4,5,6,7 python encode_latents.py \
    --videos /data/ssv2/videos --out data/latents/pretrain --frames 32
# finetune source — action-conditioned
python encode_latents.py --videos /data/bridge \
    --actions /data/bridge/actions.npy --out data/latents/finetune
```

Then set `data_source: memmap`, `data_root: data/latents/pretrain`.

Note: SSv2 is now behind registration on Qualcomm's developer portal (the old
`20bn.com` links are dead) — verify access before planning around it. Avoid
YouTube-link-based sets (Kinetics, Panda-70M): link rot means the dataset isn't
reproducible across your own runs, let alone anyone else's.

After encoding, a clip is ~64 KB, so 100k clips fits in page cache and the
dataloader cannot stall. That matters because a dataloader stall is
indistinguishable from a comm stall in a profile.

---

## Config selector

The part that makes this a contribution rather than a benchmark table. Given
(s, d, |θ|, ρ, VRAM, measured bus bandwidth), predict the best configuration
without running it:

```python
import costmodel as cm
from model import ModelCfg
m = ModelCfg(dim=2048, depth=32); N = m.n_params()
best = cm.select_config(cm.Machine(n_gpus=4, busbw=20e9),   # busbw from preflight
                        n_params=N, n_trainable=N, depth=m.depth, dim=m.dim,
                        seq_len=8192, target_batch_tokens=524288)
print(best[0])   # -> zero2, dp4, mb1, M=16, no ckpt, 22.0 GB
```

`fig7_costmodel.png` reports MAPE against measurement. Showing this matches
exhaustive search on **held-out** configurations is what converts "we measured
things" into "we predict things".

---

## Test coverage

Verified on CPU (no GPU in the authoring environment):

| | status |
|---|---|
| model fwd/bwd, LoRA (ρ=0.015 measured) | ✅ |
| `n_params()` analytic estimate | ✅ 0.03% error vs actual |
| cost model, eqs (1)–(3), config selector | ✅ |
| `single` arm, pretrain + finetune, end-to-end | ✅ |
| `ddp` arm, 2 ranks over gloo, comm counters | ✅ |
| sweep queue, partial resume, incremental plots | ✅ |
| all 9 figures + summary.md | ✅ (fabricated data) |
| **`fsdp` / `zero2` / `tp` / `tp_dp` / `pp`** | ⚠️ **unverified — needs GPUs** |

Run `--dry_run` on the real node before the exclusive window: the sharded arms
are the ones most likely to need a fix, and the dry run is what finds it cheaply.
A `T` model size (dim=128, depth=2) exists for fast smoke tests.
