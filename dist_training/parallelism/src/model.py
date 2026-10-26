"""Latent video world model (DiT). This is the *workload*, not the contribution.

Design constraints that matter for the experiment:
  - heads are inferred from runtime tensor width, so tensor-parallel sharding of
    qkv/out projections works without the module knowing its TP degree;
  - conditioning (diffusion timestep + action) enters via adaLN-zero, so the
    conditioning vector is small and cheap to pass across pipeline stages;
  - no fused/custom kernels, so FLOP accounting stays analytic and honest.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelCfg:
    dim: int = 1024
    depth: int = 24
    head_dim: int = 64
    in_dim: int = 16          # VAE latent channels x patch (flattened per token)
    action_dim: int = 8       # continuous action vector; 0 disables action conditioning
    mlp_ratio: int = 4
    max_seq: int = 8192

    @property
    def n_heads(self) -> int:
        return self.dim // self.head_dim

    def n_params(self) -> int:
        d, r = self.dim, self.mlp_ratio
        per_block = 4 * d * d + 2 * r * d * d + 6 * d * d  # attn + mlp + adaLN modulation
        return (self.depth * per_block + 2 * self.in_dim * d
                + self.max_seq * d + 2 * d * d)  # + pos-emb + timestep mlp


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([a.cos(), a.sin()], dim=-1).to(t.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.dim, 3 * cfg.dim, bias=False)
        self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        qkv = self.qkv(x)
        # Local head count, not the config value: under ColwiseParallel the output
        # width is dim/tp. Inferring it here is what makes the module TP-agnostic.
        n_local = qkv.shape[-1] // (3 * self.head_dim)
        q, k, v = qkv.view(B, S, 3, n_local, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)  # flash/mem-efficient backend
        return self.proj(o.transpose(1, 2).reshape(B, S, n_local * self.head_dim))


class Block(nn.Module):
    """Pre-norm transformer block with adaLN-zero conditioning."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        d, h = cfg.dim, cfg.mlp_ratio * cfg.dim
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(cfg)
        self.fc1, self.fc2 = nn.Linear(d, h, bias=False), nn.Linear(h, d, bias=False)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d, bias=True))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        sa, ga, ba, sm, gm, bm = self.mod(c).unsqueeze(1).chunk(6, dim=-1)
        x = x + ga * self.attn(self.n1(x) * (1 + sa) + ba)
        x = x + gm * self.fc2(F.gelu(self.fc1(self.n2(x) * (1 + sm) + bm)))
        return x


class Embed(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Linear(cfg.in_dim, cfg.dim, bias=False)
        self.pos = nn.Parameter(torch.zeros(1, cfg.max_seq, cfg.dim))
        self.t_mlp = nn.Sequential(nn.Linear(cfg.dim, cfg.dim), nn.SiLU(),
                                   nn.Linear(cfg.dim, cfg.dim))
        self.a_mlp = (nn.Linear(cfg.action_dim, cfg.dim) if cfg.action_dim else None)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x, t, a=None):
        h = self.tok(x) + self.pos[:, : x.shape[1]]
        c = self.t_mlp(timestep_embedding(t, self.cfg.dim))
        if self.a_mlp is not None and a is not None:
            c = c + self.a_mlp(a)
        return h, c


class Head(nn.Module):
    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.norm = nn.LayerNorm(cfg.dim, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(cfg.dim, cfg.in_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x, c):
        return self.out(self.norm(x))


class VideoDiT(nn.Module):
    """Rectified-flow velocity prediction on VAE latents, action-conditioned."""

    def __init__(self, cfg: ModelCfg):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.head = Head(cfg)

    def forward(self, x, t, a=None):
        h, c = self.embed(x, t, a)
        for b in self.blocks:
            h = b(h, c)
        return self.head(h, c)


# --------------------------------------------------------------------------- #
# Rectified-flow objective. Kept trivial on purpose: the systems measurement
# must not be perturbed by a fancy loss.
# --------------------------------------------------------------------------- #
def flow_loss(model, x1: torch.Tensor, a: torch.Tensor | None = None):
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], device=x1.device, dtype=x1.dtype)
    xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
    return F.mse_loss(model(xt, t * 1000, a), x1 - x0)


# --------------------------------------------------------------------------- #
# LoRA — the fine-tuning arm. Wraps nn.Linear in place.
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        self.a = nn.Parameter(torch.zeros(r, base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.normal_(self.a, std=0.02)
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(x, self.a), self.b) * self.scale


def apply_lora(model: nn.Module, r: int = 16, targets=("qkv", "proj")) -> nn.Module:
    """Freeze the trunk, add LoRA to attention projections. Returns the model."""
    for p in model.parameters():
        p.requires_grad_(False)
    for mod in model.modules():
        for name, child in list(mod.named_children()):
            if name in targets and isinstance(child, nn.Linear):
                setattr(mod, name, LoRALinear(child, r))
    # adaLN stays trainable: action conditioning is the point of the fine-tune.
    for mod in model.modules():
        if isinstance(mod, Embed) and mod.a_mlp is not None:
            for p in mod.a_mlp.parameters():
                p.requires_grad_(True)
    return model


def trainable_fraction(model: nn.Module) -> float:
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return tr / max(tot, 1)
