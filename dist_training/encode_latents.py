"""Pre-encode video clips to VAE latents, once, offline.

Run this on the V100s -- it is embarrassingly parallel, needs no communication,
and leaves the A6000s free. After encoding, a clip is ~64 KB, so 100k clips fits
in page cache and the dataloader can no longer stall. That matters because a
dataloader stall is indistinguishable from a communication stall in a profile.

  # pretrain source (general video, no actions)
  CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/encode_latents.py \
      --videos /data/ssv2/videos --out data/latents/pretrain --frames 32

  # finetune source (action-conditioned)
  python tools/encode_latents.py --videos /data/bridge --actions /data/bridge/act.npy \
      --out data/latents/finetune --frames 32
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True)
    p.add_argument("--actions", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--vae", default="stabilityai/sd-vae-ft-mse")
    p.add_argument("--frames", type=int, default=32)
    p.add_argument("--res", type=int, default=256)
    p.add_argument("--patch", type=int, default=2)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--split", default="train")
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()

    from diffusers import AutoencoderKL          # pip install diffusers decord
    import decord

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(a.vae).to(dev).eval().requires_grad_(False)

    files = sorted(Path(a.videos).rglob("*.webm")) + sorted(Path(a.videos).rglob("*.mp4"))
    if a.limit:
        files = files[: a.limit]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    lat, kept = [], []
    for i, f in enumerate(files):
        try:
            vr = decord.VideoReader(str(f), width=a.res, height=a.res)
            idx = np.linspace(0, len(vr) - 1, a.frames).astype(int)
            v = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
            v = (v.float() / 127.5 - 1).to(dev)
            with torch.no_grad(), torch.autocast(dev, torch.bfloat16, enabled=dev == "cuda"):
                z = vae.encode(v).latent_dist.sample() * 0.18215   # [T,4,H/8,W/8]
            # Patchify to tokens: [T, 4, h, w] -> [T*(h/p)*(w/p), 4*p*p]
            T, C, H, W = z.shape
            pz = z.reshape(T, C, H // a.patch, a.patch, W // a.patch, a.patch)
            tok = pz.permute(0, 2, 4, 1, 3, 5).reshape(-1, C * a.patch * a.patch)
            lat.append(tok.float().cpu().numpy().astype(np.float16))
            kept.append(i)
        except Exception as e:
            print(f"skip {f.name}: {e}")
        if (i + 1) % 200 == 0:
            print(f"{i+1}/{len(files)}  tokens/clip={lat[-1].shape[0]}", flush=True)

    x = np.stack(lat)
    np.save(out / f"{a.split}_latents.npy", x)
    print(f"wrote {x.shape} -> {out}  ({x.nbytes/1e9:.2f} GB, "
          f"{x.nbytes/len(x)/1e3:.0f} KB/clip)")

    if a.actions:
        acts = np.load(a.actions)[kept].astype(np.float16)
        np.save(out / f"{a.split}_actions.npy", acts)
        print(f"wrote actions {acts.shape}")


if __name__ == "__main__":
    main()
