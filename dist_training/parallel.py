"""Parallelism strategies, all built on torch.distributed primitives.

Deliberately thin. The contribution is the measurement and the cost model, not
the kernels -- hand-rolling column/row-parallel linears costs a week and buys
nothing this experiment measures.

Strategies
  single  1 GPU, accumulation raised to hold the global batch fixed
  ddp     replicate weights, all-reduce gradients
  zero2   shard optimizer state + gradients
  fsdp    shard parameters, gradients and optimizer state (ZeRO-3)
  tp      Megatron-style intra-layer sharding
  pp      GPipe-style inter-layer sharding
  tp_dp   TP within a PCIe pair x DP across pairs  <- expected best on this node
"""
from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn

from model import Block

STRATEGIES = ("single", "ddp", "zero2", "fsdp", "tp", "pp", "tp_dp")


# --------------------------------------------------------------------------- #
# Process group
# --------------------------------------------------------------------------- #
def init_dist():
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(backend)
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
    return rank, world, local


def cleanup():
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def build_mesh(world: int, dp: int, tp: int):
    from torch.distributed.device_mesh import init_device_mesh
    assert dp * tp == world, f"dp*tp={dp*tp} != world={world}"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if tp > 1 and dp > 1:
        return init_device_mesh(dev, (dp, tp), mesh_dim_names=("dp", "tp"))
    if tp > 1:
        return init_device_mesh(dev, (tp,), mesh_dim_names=("tp",))
    return init_device_mesh(dev, (dp,), mesh_dim_names=("dp",))


# --------------------------------------------------------------------------- #
# Gradient checkpointing
# --------------------------------------------------------------------------- #
def enable_checkpointing(model: nn.Module):
    from torch.utils.checkpoint import checkpoint

    class Ckpt(nn.Module):
        def __init__(self, mod):
            super().__init__()
            self.mod = mod

        def forward(self, *a):
            return checkpoint(self.mod, *a, use_reentrant=False)

    if hasattr(model, "blocks"):
        model.blocks = nn.ModuleList(Ckpt(b) for b in model.blocks)
    return model


# --------------------------------------------------------------------------- #
# Tensor parallel
# --------------------------------------------------------------------------- #
def apply_tp(model: nn.Module, mesh):
    """Megatron plan: qkv/fc1 column-sharded, proj/fc2 row-sharded.

    The residual stream stays replicated, so adaLN modulation is left alone --
    it is small and sharding it would add collectives for no memory win.
    """
    from torch.distributed.tensor.parallel import (ColwiseParallel,
                                                   RowwiseParallel,
                                                   parallelize_module)
    tp_mesh = mesh["tp"] if "tp" in getattr(mesh, "mesh_dim_names", ()) else mesh
    for block in model.blocks:
        b = block.mod if isinstance(block, nn.Module) and hasattr(block, "mod") else block
        parallelize_module(b, tp_mesh, {
            "attn.qkv": ColwiseParallel(),
            "attn.proj": RowwiseParallel(),
            "fc1": ColwiseParallel(),
            "fc2": RowwiseParallel(),
        })
    return model


# --------------------------------------------------------------------------- #
# Pipeline parallel
# --------------------------------------------------------------------------- #
class PipeStage(nn.Module):
    """One contiguous slice of blocks. Conditioning is threaded through every
    stage; it is [B, dim] so the extra point-to-point traffic is negligible."""

    def __init__(self, model, lo, hi, first, last):
        super().__init__()
        self.embed = model.embed if first else None
        self.blocks = nn.ModuleList(model.blocks[lo:hi])
        self.head = model.head if last else None

    def forward(self, x, c, t=None, a=None):
        if self.embed is not None:
            x, c = self.embed(x, t, a)
        for b in self.blocks:
            x = b(x, c)
        if self.head is not None:
            x = self.head(x, c)
        return x, c


def build_pp(model, cfg, pp: int, rank: int, device, micro_bs: int, seq_len: int):
    """Returns (stage_module, schedule_factory). Layers are split evenly; use
    uneven bounds here if the node is heterogeneous."""
    from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

    per = cfg.depth // pp
    lo, hi = rank * per, (rank + 1) * per if rank < pp - 1 else cfg.depth
    stage_mod = PipeStage(model, lo, hi, rank == 0, rank == pp - 1).to(device)

    d = cfg.dim
    if rank == 0:
        ex = (torch.zeros(micro_bs, seq_len, cfg.in_dim, device=device),
              torch.zeros(micro_bs, d, device=device),
              torch.zeros(micro_bs, device=device),
              torch.zeros(micro_bs, cfg.action_dim, device=device))
    else:
        ex = (torch.zeros(micro_bs, seq_len, d, device=device),
              torch.zeros(micro_bs, d, device=device))
    stage = PipelineStage(stage_mod, rank, pp, device, input_args=ex)
    return stage_mod, stage, ScheduleGPipe


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
@contextmanager
def _fsdp2_nosync(model):
    """FSDP2's set_requires_gradient_sync is a plain setter, not a context
    manager (unlike DDP's no_sync / FSDP1's no_sync) -- wrap it so callers can
    use `with nosync():` uniformly across strategies."""
    model.set_requires_gradient_sync(False)
    try:
        yield
    finally:
        model.set_requires_gradient_sync(True)


def wrap(model, strategy: str, *, mesh=None, dp=1, tp=1, device="cuda"):
    """Apply a strategy. Returns (model, no_sync_ctx_factory)."""
    s = strategy.lower()

    if s == "single":
        return model.to(device), lambda: nullcontext()

    if s == "ddp":
        from torch.nn.parallel import DistributedDataParallel as DDP
        ids = [torch.cuda.current_device()] if torch.cuda.is_available() else None
        m = DDP(model.to(device), device_ids=ids, gradient_as_bucket_view=True)
        return m, m.no_sync

    if s in ("fsdp", "zero2", "tp_dp", "tp"):
        if s in ("tp", "tp_dp"):
            model = apply_tp(model.to(device), mesh)
            if s == "tp":
                return model, lambda: nullcontext()
        dp_mesh = mesh["dp"] if (mesh is not None and
                                 "dp" in getattr(mesh, "mesh_dim_names", ())) else mesh
        try:  # FSDP2
            from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
            pol = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                       reduce_dtype=torch.float32)
            reshard = (s != "zero2")   # ZeRO-2: keep params gathered after fwd
            for b in model.blocks:
                fully_shard(b, mesh=dp_mesh, mp_policy=pol,
                            reshard_after_forward=reshard)
            fully_shard(model, mesh=dp_mesh, mp_policy=pol,
                        reshard_after_forward=reshard)
            return model.to(device), lambda: _fsdp2_nosync(model)
        except ImportError:  # FSDP1 fallback
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import ShardingStrategy
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            import functools
            strat = (ShardingStrategy.SHARD_GRAD_OP if s == "zero2"
                     else ShardingStrategy.FULL_SHARD)
            m = FSDP(model, sharding_strategy=strat, device_id=device,
                     auto_wrap_policy=functools.partial(
                         transformer_auto_wrap_policy,
                         transformer_layer_cls={Block}))
            return m, m.no_sync

    raise ValueError(f"unknown strategy {strategy}")
