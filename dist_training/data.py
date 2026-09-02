"""Latent clip datasets.

Everything is pre-encoded and memory-mapped. A dataloader stall is
indistinguishable from a communication stall in a profile, so the dataloader is
deliberately made incapable of stalling.

Two sources:
  synthetic  -- procedurally generated latent trajectories with exact action
                labels. No download, unlimited, exact control over sequence
                length. This is the primary source for measurement runs.
  memmap     -- real video latents produced by tools/encode_latents.py from
                SSv2 (pretrain) or RT-1/Bridge (fine-tune).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SyntheticLatents(Dataset):
    """Deterministic advected-blob dynamics in latent space.

    Tokens follow a smooth field whose drift is set by the action, so the
    action label is genuinely predictive and the fine-tuning arm has a real
    conditioning shift to learn. Cost is negligible next to the model.
    """

    def __init__(self, n: int, seq_len: int, in_dim: int, action_dim: int = 8, seed: int = 0):
        self.n, self.s, self.d, self.a = n, seq_len, in_dim, action_dim
        g = torch.Generator().manual_seed(seed)
        # A small shared basis keeps clips correlated (learnable) but distinct.
        self.basis = torch.randn(16, in_dim, generator=g)
        self.seed = seed

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        g = torch.Generator().manual_seed(self.seed * 1_000_003 + i)
        act = torch.randn(self.a, generator=g)
        coef = torch.randn(16, generator=g)
        phase = torch.linspace(0, 6.283, self.s).unsqueeze(1)
        drift = act[: min(self.a, 4)].sum() * 0.5
        w = torch.sin(phase * (1 + coef[:8].abs().mean()) + drift)  # [S,1]
        x = (w * (coef @ self.basis)) + 0.1 * torch.randn(self.s, self.d, generator=g)
        return x.float(), act.float()


class MemmapLatents(Dataset):
    """Real latents: one .npy of [N, S, D] plus optional [N, A] actions."""

    def __init__(self, root: str, seq_len: int, split: str = "train"):
        root = Path(root)
        self.x = np.load(root / f"{split}_latents.npy", mmap_mode="r")
        ap = root / f"{split}_actions.npy"
        self.a = np.load(ap, mmap_mode="r") if ap.exists() else None
        self.s = seq_len
        assert self.x.shape[1] >= seq_len, f"clip has {self.x.shape[1]} tokens < {seq_len}"

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        x = torch.from_numpy(np.asarray(self.x[i, : self.s])).float()
        a = (torch.from_numpy(np.asarray(self.a[i])).float() if self.a is not None
             else torch.zeros(8))
        return x, a


def build_dataset(cfg) -> Dataset:
    if cfg.data_source == "synthetic":
        return SyntheticLatents(cfg.n_clips, cfg.seq_len, cfg.in_dim,
                                cfg.action_dim, seed=cfg.seed)
    return MemmapLatents(cfg.data_root, cfg.seq_len, cfg.split)
