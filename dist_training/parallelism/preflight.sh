#!/usr/bin/env bash
# Run this BEFORE designing the sweep. The numbers it prints determine which
# strategies are worth running at all, and one of them (bus bandwidth) predicts
# most of the results on its own.
set -u
OUT=${1:-results/preflight}
mkdir -p "$OUT"
echo "=== preflight -> $OUT ==="

echo "--- 1. topology: are the A6000s on one root complex or two? ---"
# PIX/PXB within a pair and SYS/NODE across it means TP=4 crosses the CPU link,
# and TP2xDP2 will beat flat TP=4. NV# means NVLink bridges are installed --
# check for this first, it changes the whole experiment.
nvidia-smi topo -m | tee "$OUT/topo.txt"

echo "--- 2. link width/gen: Gen4 x16, or silently x8? ---"
nvidia-smi -q | grep -A4 "GPU Link Info" | tee "$OUT/linkinfo.txt"

echo "--- 3. ECC state must match across ranks ---"
# GPUs reporting different total memory (46068 vs 49140 MiB) have different ECC
# settings; a ~3 GB ceiling difference corrupts any OOM-threshold comparison.
#   sudo nvidia-smi -i 0,1,2,3 -e 0   &&   reboot
nvidia-smi --query-gpu=index,name,memory.total,ecc.mode.current \
           --format=csv | tee "$OUT/ecc.txt"

echo "--- 4. lock clocks: thermal drift must not correlate with arm order ---"
nvidia-smi --query-gpu=index,clocks.max.sm --format=csv | tee "$OUT/clocks.txt"
echo "  then: sudo nvidia-smi -i 0,1,2,3 -lgc <freq_below_throttle>"
echo "  and:  sudo nvidia-smi -i 0,1,2,3 -pm 1"

echo "--- 5. P2P actually enabled? (A6000 should say Yes; RTX 3090 would not) ---"
python - <<'PY' | tee "$OUT/p2p.txt"
import torch
n = torch.cuda.device_count()
print("visible GPUs:", n)
for i in range(n):
    for j in range(n):
        if i != j:
            print(f"  {i}->{j} p2p={torch.cuda.can_device_access_peer(i, j)}")
PY

echo "--- 6. bus bandwidth: THE number. Put it in Machine(busbw=...) ---"
# Restricted to the 4 A6000s (target hardware) -- this node also has 4 V100s at
# indices 4-7 for latent encoding, and mixing them in would pin to the slowest rank.
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Written to a real file (not piped via stdin) because torch.multiprocessing.spawn
# re-execs __main__ from disk for each worker; `python - <<PY` has no file to re-exec.
BUSBW_PY="$(mktemp --suffix=.py)"
cat > "$BUSBW_PY" <<'PY'
# Ring all-reduce busbw, measured the same way nccl-tests reports it.
import os, time, torch, torch.distributed as dist
if not torch.cuda.is_available():
    print("no CUDA; skipping"); raise SystemExit
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29555")
import torch.multiprocessing as mp
def w(rank, n):
    dist.init_process_group("nccl", rank=rank, world_size=n)
    torch.cuda.set_device(rank)
    for mb in (8, 64, 256, 512):
        x = torch.empty(mb*1024*1024//2, dtype=torch.bfloat16, device="cuda")
        for _ in range(5): dist.all_reduce(x)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20): dist.all_reduce(x)
        torch.cuda.synchronize(); dt = (time.perf_counter()-t)/20
        if rank == 0:
            nb = x.numel()*2
            print(f"  {mb:4d} MB  algbw {nb/dt/1e9:7.2f} GB/s   "
                  f"busbw {nb/dt*2*(n-1)/n/1e9:7.2f} GB/s")
    dist.destroy_process_group()
if __name__ == "__main__":
    n = torch.cuda.device_count()
    mp.spawn(w, args=(n,), nprocs=n)
PY
python "$BUSBW_PY" | tee "$OUT/busbw.txt"
rm -f "$BUSBW_PY"

echo "--- 7. host resources for the latent cache ---"
free -g | head -2 | tee "$OUT/mem.txt"
df -h . | tee -a "$OUT/mem.txt"

echo
echo "=== next: python src/sweep.py --config configs/sweep.yaml --dry_run ==="
