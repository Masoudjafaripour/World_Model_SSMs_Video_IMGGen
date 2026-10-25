"""Environment smoke test. Plain `python smoke_test.py` on CPU, or under
`torchrun --nproc_per_node=N` on GPUs to also check NCCL all-reduce and bus bandwidth."""
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F


def main():
    import datasets, einops, numpy, pyarrow, transformers, webdataset  # noqa: F401

    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    cuda = torch.cuda.is_available()
    if rank == 0:
        print(f"torch {torch.__version__}  cuda {torch.version.cuda}  "
              f"transformers {transformers.__version__}  datasets {datasets.__version__}  "
              f"pyarrow {pyarrow.__version__}")
        print(f"cuda available: {cuda}  devices: {torch.cuda.device_count()}")

    device = torch.device("cpu")
    if cuda:
        local = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        if rank == 0:
            print(f"gpu: {torch.cuda.get_device_name(local)}  "
                  f"bf16: {torch.cuda.is_bf16_supported()}")

    # one transformer-sized attention + matmul in bf16, forward and backward
    dtype = torch.bfloat16 if cuda else torch.float32
    q = torch.randn(2, 16, 160, 64, device=device, dtype=dtype, requires_grad=True)
    o = F.scaled_dot_product_attention(q, q, q)
    w = torch.randn(1024, 4096, device=device, dtype=dtype, requires_grad=True)
    (o.sum() + (torch.randn(160, 1024, device=device, dtype=dtype) @ w).sum()).backward()
    if rank == 0:
        print("sdpa + matmul fwd/bwd: ok")

    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
        x = torch.ones(1, device=device) * (rank + 1)
        dist.all_reduce(x)
        assert x.item() == world * (world + 1) / 2, x.item()

        # ring all-reduce busbw, as nccl-tests reports it
        x = torch.empty(256 * 1024 * 1024 // 2, dtype=torch.bfloat16, device=device)
        for _ in range(5):
            dist.all_reduce(x)
        if cuda:
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(20):
            dist.all_reduce(x)
        if cuda:
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t) / 20
        if rank == 0:
            nb = x.numel() * 2
            print(f"all_reduce ok on {world} ranks; 256 MB busbw "
                  f"{nb / dt * 2 * (world - 1) / world / 1e9:.1f} GB/s")
        dist.destroy_process_group()

    if rank == 0:
        print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
