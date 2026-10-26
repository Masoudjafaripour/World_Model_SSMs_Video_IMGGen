# Distributed training

Two projects about training world models across many GPUs:

| Folder | What |
|---|---|
| [`parallelism/`](parallelism/README.md) | Which parallelism strategy (DDP, ZeRO-2, FSDP, TP, PP, TP×DP) is best for a video diffusion world model, for pretraining vs LoRA fine-tuning. Includes a cost model that predicts the best configuration. |
| [`WM/`](WM/README.md) | Language-conditioned behaviour cloning with an auxiliary world-model loss on robot data (LIBERO), trained data-parallel from 1 GPU up to several nodes. |

`WM/train_bc.py` reuses the GPU telemetry from `parallelism/src/metrics.py`.
