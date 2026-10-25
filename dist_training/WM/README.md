# World-Language-Action model: multi-GPU behaviour cloning

Language-conditioned behaviour cloning (BC) with an auxiliary **world-model**
loss, pretrained on open robot datasets and trained data-parallel on a GPU cluster
(4× L40S per node).

**Status:** setup. Goal right now is a *working pipeline*, kept simple:
single node first, multi-node later.

---

## Idea

```
 instruction ──► frozen text encoder (cached once) ─────┐
 frames t-1,t ─► frozen vision encoder ─► patch tokens ──┤
 proprio ──────► MLP ────────────────────────────────────┤
                                                         ▼
                        trainable transformer (~300M)
                         │                          │
              action head (chunk of H     world head (vision-encoder
              future actions)             features of frame t+k)
```

`loss = L_BC(actions) + λ · L_world(future features)`

Predicting *features* of a future frame, not pixels, keeps the world loss cheap
and stable. λ = 0 is plain BC and is the baseline every result is compared to.

## Architecture

One token sequence per sample, processed by a single trunk:

```
 [ lang ×16 | obs t-1 ×49 | obs t ×49 | proprio ×1 | world queries ×49 | action queries ×H ]
 └──────────────── context (102) ───────────────────┘ └──── learned queries ──────────────┘
                                                          ≈ 160 tokens total (H = 8)
```

### Blocks

| # | Block | In → Out | Trainable | Notes |
|---|---|---|---|---|
| 1 | **Language encoder** | instruction → 16 × 768 | ❄️ frozen | SigLIP text tower. Cached once per unique instruction, never run during training. |
| 2 | **Vision encoder** | 224² RGB → 196 × 768 → **49 × 768** | ❄️ frozen | SigLIP-B/16, 2×2 avg-pool to a 7×7 grid. Run once per frame: the frames at t−1 and t feed the context, and the frame at t+k becomes the world target. |
| 3 | **Projectors** | 768 → d, state → d | 🔥 | One linear layer each for language and vision. 2-layer MLP for proprio (normalised joint/EEF state). |
| 4 | **Embeddings** | + d | 🔥 | Learned position + type embedding (lang / obs t−1 / obs t / proprio / world / action). |
| 5 | **Trunk** | L × d | 🔥 | Pre-norm transformer, d = 1024, 24 layers, 16 heads, MLP ×4, bf16, SDPA attention. **~300M parameters**, the only big block. |
| 6 | **World head** | 49 world queries → 49 × 768 | 🔥 | Predicts the frozen-encoder features of frame **t+k** (k = 4 to start). Target is stop-grad and LayerNorm'd. Loss: cosine + MSE. |
| 7 | **Action head** | H action queries → H × 7 | 🔥 | v0: linear + **L1 regression** on a normalised action chunk (H = 8). v1: swap in a flow-matching head. |

### Attention mask

| query ↓ attends to → | context | world queries | action queries |
|---|---|---|---|
| context | ✅ | ❌ | ❌ |
| world queries | ✅ | ✅ | ❌ |
| action queries | ✅ | ✅ | ✅ |

Action queries can read the predicted future. That is how the world loss is
meant to help the policy. Nothing reads the actions, so λ = 0 reduces exactly to
plain BC with extra unused tokens.

### Loss

```
L = L1(â_{t:t+H}, a_{t:t+H})  +  λ · [ (1 − cos(ẑ_{t+k}, z_{t+k})) + MSE(ẑ_{t+k}, z_{t+k}) ]
```

Actions are normalised per dataset (1st/99th percentile → [−1, 1]), and the
gripper dimension is kept binary.

### Inference

The encoders run once per control step, then one trunk forward pass predicts an
H-step action chunk. The world head is optional at test time: it is not needed
to act, and it is useful only for visualising "what the model expects".

### Deliberately left out of v0

| Left out | Added in |
|---|---|
| pretrained VLM backbone | option B |
| wrist camera | v1 |
| flow-matching actions | v1 |
| pixel-space prediction | later, via `../model.py`'s `VideoDiT` |
| multi-dataset embodiment tokens | step 6 |

---

## Efficiency study: what each pipeline choice buys

The question: how much compute and wall-clock does each choice save, and what
does it cost in validation quality? Same policy model, same data, 1 node × 4 L40S.

### Arms (each adds one change to the previous)

| # | Arm | Encoder | Input per step | Tokens/frame |
|---|---|---|---|---|
| A0 | **reference** | SigLIP **fine-tuned** (fwd + bwd) | 224 px frames | 49 |
| A1 | + frozen encoder | SigLIP frozen (fwd only, every step) | 224 px frames | 49 |
| A2 | + pre-encoded features | none at train time | cached features | 196 (unpooled) |
| A3 | + 2×2 token pooling | none | cached features | **49** (our v0) |
| A4 | + large batch | none | cached features | 49, batch ×4 / GPU |
| A5 | + batch sweep | none | cached features | find the useful-batch limit |

* A0's batch is the largest that fits before OOM. A1–A3 use the same batch where
  it fits. A4/A5 then scale the batch.
* A2 → A3 is our analogue of temporal compression: fewer tokens per frame.
* Frame decoding is excluded from all arms (frames come from `img.npy`). The
  study isolates the encoder and token-count effects.

### Measured per arm

| Metric | How |
|---|---|
| frames/s/GPU, tokens/s/GPU | timed steps after warmup (`train_bc.py` metrics) |
| time per epoch | 1 pass over the 273k-frame train split, 4 GPUs |
| peak VRAM / GPU | `torch.cuda.max_memory_allocated` |
| MFU | FLOPs counted with `torch.utils.flop_counter` on one step, against the L40S BF16 dense peak (362 TFLOPS, datasheet; verify) |
| **val BC loss vs total compute** | total FLOPs (encoder fwd(+bwd) + policy fwd+bwd) up to each eval, log-scale x-axis |
| steps / wall-clock / compute to best val | from `metrics.jsonl` |
| later: LIBERO success rate | closed-loop, for the best arms only |

### Output

`report.py` turns the runs into a one-page figure (six panels + summary
table, as in the reference report) under `$WM_RUNS/efficiency/`, plus a short
write-up with the TL;DR numbers.

### Budget

Throughput, VRAM and MFU need only ~10 min per arm. Convergence (val vs
compute) needs a full run per arm: A0/A1 are the expensive ones. They run to a
fixed compute budget, not to convergence, unless they are still improving.

---

## DAgger (after BC works)

BC trains only on expert states, so at test time small errors push the policy
into states it never saw, and the errors compound. DAgger fixes this by
labelling the states *the learner* visits with expert actions, then retraining
on the aggregate.

**Gate:** start only once BC (step 3/4) reaches a non-trivial LIBERO success
rate, roughly ≥ 30% on one suite. Below that, rollouts are mostly failure states
far from the task, and labelling them teaches little.

**Where:** LIBERO (simulation) only. Bridge is offline real-robot data with no
expert to query, so it stays pure BC.

### The expert

LIBERO demos are human teleoperation, so there is no built-in expert to query.
Options:

| Expert | Pros | Cons |
|---|---|---|
| **A strong public LIBERO policy** (e.g. a π0/π0.5 or OpenVLA-OFT LIBERO checkpoint, ~95%+ success) | Ready now, same 7-DoF delta-EEF action space | Large (3–7B). Inference costs GPU time. Its mistakes become labels. |
| Privileged-state policy (trained by us on sim state, not pixels) | Small, fast, cheap to query | Must be trained first, and must itself be good |

**Default:** a public LIBERO checkpoint as teacher. Verify the exact checkpoint,
licence and action convention before use.

### Loop

```
D ← LIBERO demos;  π_0 ← BC policy (step 4)
for i = 1..N (≈ 5–10 iterations):
    roll out π_mix = β_i·expert + (1−β_i)·π_{i−1} in LIBERO   (β_i: 1 → 0, e.g. 0.5^i)
    at every visited state: query expert → action chunk label
    D ← D ∪ {(obs, lang, proprio, expert chunk, future frame)}
    π_i ← fine-tune π_{i−1} on D  (4-GPU DDP; demos + all DAgger data)
    eval π_i on held-out init states → success rate
```

* The executed action comes from `π_mix`, but the **label is always the
  expert's**. The future-frame world target is whatever the simulator actually
  rendered at t+k.
* Mixing (β) keeps early rollouts near the task. It decays to pure learner
  rollouts.
* Rollouts and expert queries run as a separate job (parallel MuJoCo envs with
  EGL rendering on a GPU node). Training is the usual DDP job. The two
  communicate only through shards on `$SCRATCH`, so either side can be re-run on
  its own.

### Storage

| What | Where | Kept |
|---|---|---|
| DAgger data | `$WM_DATA/dagger/iter_<i>/*.tar` | every iteration (this *is* the dataset) |
| policy per iteration | `$WM_RUNS/<run>/dagger/iter_<i>/best.pt` | best only, weights only |
| rollout videos | a few per iteration | for inspection only |

### Measured

* Success rate vs DAgger iteration, against BC on the **same number of extra
  expert-only rollouts** (β = 1 throughout). This control shows whether the gain
  comes from on-policy states or just from more expert data.
* Effect of the world loss on DAgger (λ = 0 vs λ > 0).

---

## Why DDP

The model fits on one 48 GB GPU, and sequences are a few hundred tokens. In the
regime studied in [`../README.md`](../README.md), that makes plain DDP the right
call. TP/PP would only add communication over PCIe. FSDP/ZeRO-2 come back only if
the model grows past ~1B.

## Data

| Role | Dataset | Notes |
|---|---|---|
| fine-tune + closed-loop eval | **LIBERO** | simulated, small, standard benchmark. Starts here. |
| pretraining | **BridgeData V2** | ~60k language-labelled episodes, WidowX |
| optional 2nd embodiment | Fractal (RT-1, OXE subset) | only if Bridge alone is too narrow |

Sources are the LeRobot / Hugging Face versions, which avoid a TensorFlow/RLDS
dependency. They are converted once into **tar shards** (224 px frames,
normalised actions) and unpacked into `$SLURM_TMPDIR` at job start, because
Lustre handles millions of small files badly. Check sizes before every download.

## Plan

1. **Env:** venv on `$SCRATCH` from the cluster wheelhouse; CPU smoke test.
2. **Data:** LIBERO → shards + action normalisation stats.
3. **1-GPU BC baseline** (λ = 0): overfit a few episodes, then train.
4. **World loss:** λ > 0 vs λ = 0 on LIBERO success rate.
5. **1 node, 4-GPU DDP:** throughput, checkpoint/resume across 12–24 h jobs.
6. **DAgger on top of BC** (LIBERO only), once BC works. See [DAgger](#dagger-after-bc-works).
7. **Pretrain on Bridge (+ LIBERO-90)** → fine-tune / eval on LIBERO suites.
8. **Multi-node DDP**, then ablations (λ, horizon k, size, data mix, BC vs BC+DAgger).

---

## Storage: where things go

`/project/aip-zaiane` is **full** (12/12 TiB), and `$HOME` is only 50 GB. So:

| What | Where |
|---|---|
| code, configs, this README | here, in git |
| venv | `$SCRATCH/World_Model_SSMs_Video_IMGGen/wm_venv` |
| raw downloads + tar shards | `$SCRATCH/World_Model_SSMs_Video_IMGGen/dist_training/WM/data/` |
| runs: logs, metrics, eval videos, checkpoints | `$SCRATCH/World_Model_SSMs_Video_IMGGen/dist_training/WM/runs/<run>/` |
| HF / pip / W&B caches | `$SCRATCH/hf_cache`, `$SCRATCH/cache` |
| per-job hot data | `$SLURM_TMPDIR` (node-local, wiped after the job) |

`$SCRATCH` is **not backed up** and idle files are purged. Raw data can be
re-downloaded, but final checkpoints cannot. Copy a run's `best.pt` to
`/project/aip-zaiane/...` once space has been freed there.

## Checkpoints: only the important ones

Each run keeps at most three files, and never a checkpoint every N steps:

| File | Contents | When written |
|---|---|---|
| `last.pt` | weights + optimizer + step + RNG | overwritten periodically (atomic write: tmp → rename). Exists only for resume. |
| `best.pt` | weights only, bf16 | when the validation metric improves |
| `final.pt` | weights only, bf16 | end of run, then `last.pt` is deleted |

At ~300M parameters a weights-only bf16 file is ~0.6 GB, and `last.pt` is ~4 GB.
Only rank 0 writes.
