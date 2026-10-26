# Language-Conditioned Behaviour Cloning with an Auxiliary World-Model Loss, Trained Data-Parallel

**Status (2026-10-26):** pipeline working end to end on LIBERO, on 1 GPU and on
1 node × 4 GPUs. The main comparison, plain BC (λ = 0) vs BC + world loss
(λ > 0), is running at the 300M scale. Closed-loop simulator evaluation is being
set up.

---

## Abstract

We train a language-conditioned visuomotor policy by **behaviour cloning (BC)**
and add an auxiliary **world-model** objective: the same network that predicts
an action chunk also predicts the visual features of a future frame. Frozen
SigLIP encoders are run once over the dataset and cached, so training touches
only a small transformer over cached features. The data loader then takes
1–2% of step time, and data-parallel training reaches **3.4× on 4 GPUs (85%
scaling efficiency)**. On LIBERO (40 tasks, 1693 demonstrations), a 27M-parameter
policy reaches a held-out validation BC loss of **0.131** in 17 min on one GPU, or
5 min on four. The open question this setup is built to answer is whether the
world loss improves closed-loop task success over plain BC.

---

## 1. Introduction

Behaviour cloning learns a policy by regressing expert actions from observations.
It is simple and scales with data and compute. But it learns nothing about
*how the world changes*, and it degrades once the policy leaves the expert's
state distribution.

This project tests two additions on top of a plain, scalable BC pipeline:

1. **An auxiliary world-model loss.** The policy also predicts the encoder
   features of the frame k steps ahead, and the action tokens may attend to that
   prediction.
2. **DAgger** (planned). Retraining on states the policy itself visits, labelled
   by an expert, to reduce compounding error.

Everything is built to run data-parallel, scaling gradually from 1 GPU to 1 node
(4 GPUs) and then to several nodes.

---

## 2. Method

### 2.1 Problem formulation

A demonstration is a sequence of frames $o_t$, robot states $s_t$ and actions
$a_t \in \mathbb{R}^7$, with an instruction $\ell$. Frozen encoders give
language tokens $E_\text{txt}(\ell) \in \mathbb{R}^{32 \times 768}$ and image
tokens $E(o_t) \in \mathbb{R}^{49 \times 768}$ (SigLIP-B/16 patch tokens,
2×2-pooled to 7×7).

A trainable transformer $f_\theta$ predicts an action chunk and future features:

$$(\hat a_{t:t+H},\ \hat z_{t+k}) = f_\theta\big(E_\text{txt}(\ell),\ E(o_{t-1}),\ E(o_t),\ s_t\big), \qquad H = 8,\ k = 4$$

### 2.2 Objective

$$\mathcal L(\theta) = \underbrace{\big\lVert \hat a_{t:t+H} - a_{t:t+H} \big\rVert_1}_{\text{behaviour cloning}} \;+\; \lambda\,\underbrace{\Big[\big(1 - \cos(\hat z_{t+k}, z_{t+k})\big) + \big\lVert \hat z_{t+k} - z_{t+k} \big\rVert_2^2\Big]}_{\text{world model}}, \qquad z_{t+k} = \mathrm{LN}\big(E(o_{t+k})\big)$$

* Actions are normalised per dimension to $[-1, 1]$ using the 1st/99th
  percentiles of the training split. The gripper is already $\pm 1$.
* Action steps past the end of an episode are masked out of the L1 term.
* The world target is a frozen-encoder output (stop-gradient by construction)
  and is LayerNorm'd. Predicting *features* rather than pixels keeps this term
  cheap and stable.
* **λ = 0 is exactly plain BC** (see §2.3), the baseline for every comparison.

### 2.3 Architecture

One token sequence per sample (188 tokens) through a single transformer trunk:

```
 [ lang ×32 | obs t-1 ×49 | obs t ×49 | proprio ×1 | world queries ×49 | action queries ×8 ]
 └──────────────────── context (131) ──────────────┘ └────────── learned queries ──────────┘
```

| # | Block | In → Out | Trainable | Notes |
|---|---|---|---|---|
| 1 | Language encoder | instruction → 32 × 768 | frozen | SigLIP text tower, run once per task string |
| 2 | Vision encoder | 224² RGB → 196 × 768 → 49 × 768 | frozen | SigLIP-B/16, 2×2 average pool, run once per frame |
| 3 | Projectors | 768 → d; state → d | yes | linear for language/vision, 2-layer MLP for the 8-d state |
| 4 | Embeddings | + d | yes | learned position + token-type embedding |
| 5 | Trunk | 188 × d | yes | pre-norm transformer, SDPA attention, bf16 autocast |
| 6 | World head | 49 world queries → 49 × 768 | yes | predicts $z_{t+k}$ |
| 7 | Action head | 8 action queries → 8 × 7 | yes | linear, L1 regression (v0) |

**Attention mask.** Context attends to context. World queries attend to context
and world queries. Action queries attend to everything. Nothing attends to the
action queries, so the world prediction cannot see the actions. The actions *can*
read the predicted future, and that is how the world loss is meant to help.

**Model sizes**

| Size | d | Layers | Heads | Parameters | Use |
|---|---|---|---|---|---|
| T | 256 | 4 | 4 | 3.9M | pipeline / overfit checks |
| S | 512 | 8 | 8 | 26.8M | first real runs |
| B | 1024 | 24 | 16 | ≈ 303M | main runs |

**Inference.** Encoders run once per control step, then one trunk pass predicts
an 8-step action chunk. The world head is not needed to act.

### 2.4 Data-parallel training

The largest model fits on one 48 GB GPU, and sequences are short (188 tokens).
That is the regime where plain **DDP** is the right choice (cf. the parallelism
study in [`../parallelism/README.md`](../parallelism/README.md)). Tensor/pipeline parallelism would only
add PCIe traffic. Batch size is set per GPU, so the global batch grows with the
GPU count, and the learning rate is scaled by √(global-batch ratio).

---

## 3. Experimental setup

### 3.1 Data

**LIBERO** (spatial / object / goal / long suites), from the
`physical-intelligence/libero` release (CC-BY-4.0):

| Episodes | Frames | Tasks | Control rate | Train / val |
|---|---|---|---|---|
| 1693 | 273,465 | 40 | 10 Hz | 260,231 / 13,234 frames (5% of episodes per task held out) |

Frames are resized to 224², encoded once (11 min on one L40S) and stored as
fp16 features. Only the agent-view camera is used in v0. The release's images
are rotated 180°, and closed-loop evaluation applies the same rotation.

Planned later: **BridgeData V2** for pretraining (~60k episodes), then optionally
Fractal (RT-1) as a second embodiment.

### 3.2 Hardware

Nodes with 4× NVIDIA L40S (48 GB), PCIe, no NVLink. Measured NCCL all-reduce bus
bandwidth (256 MB, bf16):

| Scope | Transport | Bus bandwidth |
|---|---|---|
| 1 node, 4 GPUs | PCIe | **18.8 GB/s** |
| 2 nodes, 8 GPUs | RoCE | **12.2 GB/s** (65% of intra-node) |

### 3.3 Training protocol

AdamW (β = 0.9/0.95, weight decay 0.05 on matrices), linear warmup then cosine
decay, gradient clipping 1.0, bf16 autocast. Validation BC loss is computed on the
held-out episodes every 500 steps.

---

## 4. Results

### 4.1 Pipeline validation

**r0** (T model, 1 GPU, 8 episodes, train = val): BC loss **0.104 → 0.014** in
3000 steps (2 min). The model can memorise, so data, normalisation, masking and
optimisation are correct end to end.

### 4.2 Single-GPU baseline vs 1-node DDP (S model, λ = 0)

Both runs see the same 1.28M samples (≈ 4.9 epochs).

| Run | GPUs | Global batch | Steps | Samples/s | Wall-clock | Best val BC | Final train BC |
|---|---|---|---|---|---|---|---|
| r1 | 1 | 64 | 20,000 | 1,320 | 16.7 min | **0.131** | 0.094 |
| r2 | 4 | 256 | 5,000 | **4,500** | **5.1 min** | 0.135 | 0.099 |

* **Scaling:** 3.4× throughput on 4 GPUs, i.e. **85% efficiency**. The data
  loader accounts for only 1–2% of step time.
* **Batch size:** 4× the global batch costs ~2.5% in validation BC at an equal
  sample budget.
* **Generalisation gap:** validation BC plateaus at ≈ 0.133 from ~12k steps
  while training BC keeps falling (0.094). With ~40 demos per task, the model
  starts memorising. Closed-loop success, not validation loss, is the deciding
  metric.

### 4.3 In progress

| Run | Setup | Question |
|---|---|---|
| r3a / r3b | B (≈ 303M), 4 GPUs, λ = 0 vs λ = 0.5 | does the world loss help at scale? |
| sim | LIBERO simulator (MuJoCo 3.3, headless EGL) | closed-loop success rate per suite |

---

## 5. Planned work

### 5.1 Efficiency study: what each pipeline choice buys

Same policy, same data, 1 node × 4 GPUs. Each arm adds one change to the
previous:

| Arm | Encoder at train time | Tokens / frame |
|---|---|---|
| A0 reference | SigLIP fine-tuned (fwd + bwd) | 49 |
| A1 | SigLIP frozen, run every step | 49 |
| A2 | none: pre-encoded features | 196 (unpooled) |
| A3 | none: pre-encoded + 2×2 pooling (v0) | 49 |
| A4 / A5 | as A3, with batch scaled up / swept | 49 |

Measured per arm: frames/s/GPU, tokens/s/GPU, time per epoch, peak VRAM, MFU
(FLOPs from `torch.utils.flop_counter`), and **validation BC vs total compute**
(encoder + policy FLOPs). Frame decoding is excluded from all arms. `report.py`
will produce a one-page figure with summary table.

### 5.2 Closed-loop evaluation

Roll out each policy in the LIBERO simulator from the benchmark's fixed initial
states, and report success rate per suite. This is the metric that decides
λ = 0 vs λ > 0.

### 5.3 DAgger (once BC reaches ≳ 30% success on a suite)

LIBERO demonstrations are human teleoperation, so there is no scripted expert.
The default teacher is a strong public LIBERO policy (checkpoint, licence and
action convention to be verified). A privileged-state policy is the fallback.

```
D ← demos;  π_0 ← BC policy
for i = 1..N (≈ 5–10):
    roll out π_mix = β_i·expert + (1−β_i)·π_{i−1}      (β_i: 1 → 0)
    label every visited state with the expert's action chunk
    D ← D ∪ new samples;  π_i ← fine-tune π_{i−1} on D  (DDP)
    evaluate π_i (success rate)
```

The executed action comes from π_mix, but the label is always the expert's.
Rollout and training are separate jobs that communicate through data shards.
**Control:** BC on the same amount of expert-only rollouts (β = 1), which
separates "on-policy states" from "more data". Simulation only: Bridge has no
expert to query.

### 5.4 Scale-out

Pretraining on BridgeData V2 (+ LIBERO-90), then fine-tuning and evaluating on the
LIBERO suites. Multi-node DDP (2 → 4 nodes) only while measured scaling
efficiency holds. Ablations: λ, k, model size, data mix, BC vs BC + DAgger.

### 5.5 Left out of v0

Pretrained VLM backbone · wrist camera · flow-matching action head ·
pixel-space prediction (via `VideoDiT` in `../parallelism/src/model.py`) · per-embodiment tokens.

---

## 6. Reproducing

All jobs are submitted from this folder. Arguments after the script name go to
`train_bc.py`, and resubmitting the same `--name` resumes from `last.pt`.

```bash
sbatch slurm/setup_env.sh                       # venv + CPU smoke test
sbatch slurm/check_gpu.sh                       # 4-GPU NCCL check
sbatch slurm/multinode.sh smoke_test.py         # 2-node NCCL check
J=$(sbatch --parsable slurm/prepare_libero.sh)  # download
sbatch --dependency=afterok:$J --gres=gpu:l40s:1 --cpus-per-task=16 --mem=48G \
       --time=02:00:00 slurm/prepare_libero.sh encode

# r1: 1 GPU
sbatch --time=06:00:00 slurm/train.sh --name r1_S_1gpu --size S --steps 20000 --batch 64 --lr 3e-4
# r2: 1 node, 4 GPUs
sbatch --gres=gpu:l40s:4 --cpus-per-task=32 --mem=160G --time=03:00:00 \
       slurm/train.sh --name r2_S_4gpu --size S --steps 5000 --batch 64 --lr 6e-4 --warmup 200
# r3: world loss on/off at 300M
sbatch --gres=gpu:l40s:4 --cpus-per-task=32 --mem=160G --time=03:00:00 \
       slurm/train.sh --name r3_B_4gpu_lam0.5 --size B --lam 0.5 --steps 5000 --batch 64
# multi-node
sbatch --nodes=2 --time=12:00:00 slurm/multinode.sh train_bc.py --name b_2n --size B
```

### Code

| File | Role |
|---|---|
| `prepare_libero.py` | download, frozen-encoder feature cache, split, normalisation stats |
| `libero_data.py` | samples (obs t−1, obs t, state, action chunk, future features) from the cache |
| `wla_model.py` | policy, attention mask, loss |
| `train_bc.py` | 1-GPU / DDP training, live plot, resume, checkpoint policy |
| `smoke_test.py`, `sim_test.py` | environment, NCCL and simulator checks |
| `slurm/` | job scripts; `env.sh` / `sim_env.sh` hold the module stack and paths |

---

## Appendix A. Storage

Project storage is full, and home is small, so only code lives in git. Everything
heavy is on `$SCRATCH`, which is **not backed up** and is purged when idle.

| What | Where |
|---|---|
| code, configs, this README | git |
| venv | `$SCRATCH/World_Model_SSMs_Video_IMGGen/wm_venv` |
| raw data + feature cache (≈ 185 GB) | `$SCRATCH/World_Model_SSMs_Video_IMGGen/dist_training/WM/data/` |
| runs (metrics, live plot, checkpoints) | `$SCRATCH/.../WM/runs/<name>/`, linked here as `runs/` (git-ignored) |
| job logs | `$SCRATCH/.../WM/logs/` |
| HF / pip caches | `$SCRATCH/hf_cache`, `$SCRATCH/cache` |

Raw data can be re-downloaded. Final checkpoints worth keeping must be copied to
project storage once space is available.

## Appendix B. Checkpoint policy

Only rank 0 writes, and a run directory holds at most three checkpoints:

| File | Contents | Written |
|---|---|---|
| `last.pt` | weights + optimizer + step | every 20 min (atomic tmp → rename), and on a time-budget stop. Deleted at the end of the run. |
| `best.pt` | weights only, bf16 | when validation BC improves |
| `final.pt` | weights only, bf16 | end of the run |

Each run also writes `metrics.jsonl`, `gpu_telemetry.csv`, `config.json` and
`live.png`. The live plot refreshes every 60 s and shows train/val losses,
throughput, data-wait fraction and per-GPU utilisation.
