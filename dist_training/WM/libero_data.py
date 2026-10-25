"""Dataset over the cached LIBERO features (prepare_libero.py).

A sample is anchored at frame t of an episode:
  obs t-1 (clamped to the episode start), obs t, state t,
  actions t..t+H-1 (masked past the episode end), obs t+k (clamped to the end).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class LiberoFeatures(Dataset):
    def __init__(self, root: str, split: str, horizon: int = 8, future_k: int = 4,
                 max_episodes: int = 0):
        root = Path(root)
        self.vis = np.load(root / "vis.npy", mmap_mode="r")
        low = np.load(root / "lowdim.npz")
        self.state, self.action = low["state"], low["action"]
        ep = np.load(root / "episodes.npz")
        self.lang = torch.from_numpy(np.load(root / "lang.npy"))
        self.lang_mask = torch.from_numpy(np.load(root / "lang_mask.npy"))
        stats = json.loads((root / "stats.json").read_text())
        self.H, self.k = horizon, future_k

        # normalise actions to [-1, 1] by 1st/99th pct; the gripper (last dim) is
        # already -1/+1 in LIBERO and is left as is
        lo, hi = (np.array(stats["action"][q], np.float32) for q in ("q01", "q99"))
        lo[-1], hi[-1] = -1.0, 1.0
        self.a_lo, self.a_hi = lo, hi
        self.s_mean = np.array(stats["state"]["mean"], np.float32)
        self.s_std = np.array(stats["state"]["std"], np.float32) + 1e-6

        # overfit mode uses the first train episodes for both splits
        keep = np.flatnonzero(ep["split"] == (0 if split == "train" or max_episodes else 1))
        if max_episodes:
            keep = keep[:max_episodes]
        self.ep_start, self.ep_len, self.ep_task = ep["start"][keep], ep["length"][keep], ep["task"][keep]
        self.index = np.concatenate([np.stack([np.full(n, e), np.arange(n)], 1)
                                     for e, n in enumerate(self.ep_len)])

    def norm_action(self, a):
        return np.clip(2 * (a - self.a_lo) / (self.a_hi - self.a_lo) - 1, -1, 1)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        s, n = int(self.ep_start[e]), int(self.ep_len[e])
        steps = np.arange(t, t + self.H)
        valid = steps < n
        acts = self.action[s + np.minimum(steps, n - 1)]
        task = int(self.ep_task[e])
        return {
            "lang": self.lang[task],
            "lang_mask": self.lang_mask[task],
            "obs_prev": torch.from_numpy(np.array(self.vis[s + max(t - 1, 0)])),
            "obs_cur": torch.from_numpy(np.array(self.vis[s + t])),
            "obs_future": torch.from_numpy(np.array(self.vis[s + min(t + self.k, n - 1)])),
            "state": torch.from_numpy((self.state[s + t] - self.s_mean) / self.s_std),
            "action": torch.from_numpy(self.norm_action(acts).astype(np.float32)),
            "action_valid": torch.from_numpy(valid.astype(np.float32)),
        }
