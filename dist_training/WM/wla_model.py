"""World-Language-Action policy, v0 (see README "Architecture").

Consumes cached frozen-encoder features (prepare_libero.py), so the encoders
are not part of this module. The trunk is the only large block.

Token layout, one sequence per sample:
  [ lang ×32 | obs t-1 ×49 | obs t ×49 | proprio ×1 | world queries ×49 | action queries ×H ]
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

FEAT_DIM = 768          # SigLIP-B width (vision patches and text tokens)
N_PATCH = 49            # 7x7 pooled patch grid per frame
N_LANG = 32
TYPE_IDS = dict(lang=0, obs_prev=1, obs_cur=2, proprio=3, world=4, action=5)


@dataclass
class WLACfg:
    dim: int = 1024
    depth: int = 24
    heads: int = 16
    mlp_ratio: int = 4
    horizon: int = 8            # action chunk length H
    action_dim: int = 7
    state_dim: int = 8

    @property
    def seq_len(self) -> int:
        return N_LANG + 2 * N_PATCH + 1 + N_PATCH + self.horizon


SIZES = {
    "T": WLACfg(dim=256, depth=4, heads=4),       # ~3M:   pipeline / overfit checks
    "S": WLACfg(dim=512, depth=8, heads=8),       # ~25M:  first real runs
    "B": WLACfg(dim=1024, depth=24, heads=16),    # ~300M: README target
}


class Block(nn.Module):
    def __init__(self, cfg: WLACfg):
        super().__init__()
        d = cfg.dim
        self.heads = cfg.heads
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, cfg.mlp_ratio * d), nn.GELU(),
                                 nn.Linear(cfg.mlp_ratio * d, d))

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.heads, D // self.heads) \
                                       .permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.proj(o.transpose(1, 2).reshape(B, L, D))
        return x + self.mlp(self.n2(x))


class WLAPolicy(nn.Module):
    def __init__(self, cfg: WLACfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.lang_proj = nn.Linear(FEAT_DIM, d)
        self.vis_proj = nn.Linear(FEAT_DIM, d)
        self.state_proj = nn.Sequential(nn.Linear(cfg.state_dim, d), nn.GELU(), nn.Linear(d, d))
        self.world_q = nn.Parameter(torch.randn(1, N_PATCH, d) * 0.02)
        self.action_q = nn.Parameter(torch.randn(1, cfg.horizon, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, cfg.seq_len, d) * 0.02)
        self.type_emb = nn.Embedding(len(TYPE_IDS), d)
        self.register_buffer("type_ids", self._type_ids(), persistent=False)
        self.register_buffer("base_mask", self._base_mask(), persistent=False)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(d)
        self.world_head = nn.Linear(d, FEAT_DIM)
        self.action_head = nn.Linear(d, cfg.action_dim)

    def _type_ids(self):
        h = self.cfg.horizon
        parts = [("lang", N_LANG), ("obs_prev", N_PATCH), ("obs_cur", N_PATCH),
                 ("proprio", 1), ("world", N_PATCH), ("action", h)]
        return torch.cat([torch.full((n,), TYPE_IDS[k]) for k, n in parts])

    def _base_mask(self):
        """True = may attend. Context sees context; world sees context + world;
        actions see everything. Nothing reads the action queries."""
        L, ctx = self.cfg.seq_len, N_LANG + 2 * N_PATCH + 1
        wq = ctx + N_PATCH
        m = torch.zeros(L, L, dtype=torch.bool)
        m[:ctx, :ctx] = True
        m[ctx:wq, :wq] = True
        m[wq:, :] = True
        return m

    def forward(self, lang, lang_mask, obs_prev, obs_cur, state):
        B = lang.shape[0]
        x = torch.cat([self.lang_proj(lang), self.vis_proj(obs_prev), self.vis_proj(obs_cur),
                       self.state_proj(state).unsqueeze(1),
                       self.world_q.expand(B, -1, -1), self.action_q.expand(B, -1, -1)], 1)
        x = x + self.pos + self.type_emb(self.type_ids)

        # padded language tokens are never attended to
        key_ok = torch.ones(B, self.cfg.seq_len, dtype=torch.bool, device=x.device)
        key_ok[:, :N_LANG] = lang_mask
        mask = (self.base_mask[None] & key_ok[:, None, :]).unsqueeze(1)    # [B,1,L,L]

        for b in self.blocks:
            x = b(x, mask)
        x = self.norm(x)
        world = self.world_head(x[:, -(N_PATCH + self.cfg.horizon):-self.cfg.horizon])
        action = self.action_head(x[:, -self.cfg.horizon:])
        return action, world


def wla_loss(model, batch, lam: float):
    """L1 on the normalised action chunk + lam * (cosine + MSE) on the future
    frame's LayerNorm'd features. Returns (total, bc, world)."""
    act, world = model(batch["lang"], batch["lang_mask"], batch["obs_prev"],
                       batch["obs_cur"], batch["state"])
    valid = batch["action_valid"].unsqueeze(-1)                         # [B,H,1]
    bc = (F.l1_loss(act.float(), batch["action"], reduction="none") * valid).sum() \
        / (valid.sum() * act.shape[-1]).clamp(min=1)
    tgt = F.layer_norm(batch["obs_future"].float(), (FEAT_DIM,))
    w = world.float()
    wl = (1 - F.cosine_similarity(w, tgt, dim=-1)).mean() + F.mse_loss(w, tgt)
    return bc + lam * wl, bc, wl
