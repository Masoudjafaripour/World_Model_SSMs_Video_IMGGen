"""Step 2: LIBERO -> frozen-encoder feature cache.

v0 keeps the vision and text encoders frozen, so they run once here instead of
every training step. Training then reads features only, which means no image
decoding and a dataloader that cannot stall.

  python prepare_libero.py --stage download   # CPU job: HF -> $WM_DATA/raw/pi_libero
  python prepare_libero.py --stage encode     # 1-GPU job: -> $WM_DATA/libero_siglip

Output ($WM_DATA/libero_siglip/):
  vis.npy        fp16 [N, 49, 768]  agent-view SigLIP patch tokens, 2x2-pooled to 7x7
  lowdim.npz     per frame: episode, frame, task, state [8], action [7]
  episodes.npz   per episode: start offset, length, task, split (0 train / 1 val)
  lang.npy       fp16 [n_tasks, 32, 768] SigLIP text tokens, + lang_mask.npy
  img.npy        uint8 [N, 224, 224, 3] frames (online-encoder arms of the efficiency study)
  vis196.npy     fp16 [N, 196, 768] unpooled tokens (pooling arm of the efficiency study)
  tasks.json     task strings
  stats.json     action/state normalisation (1st/99th pct, train split only)

Source: physical-intelligence/libero (LeRobot v2, CC-BY-4.0): LIBERO
spatial/object/goal/10, 40 tasks, 1693 episodes. Its images are already rotated
180 degrees, so closed-loop eval must apply the same rotation.
"""
from __future__ import annotations

import argparse
import json
import os
from io import BytesIO
from pathlib import Path

import numpy as np

REPO = "physical-intelligence/libero"
ENCODER = "google/siglip-base-patch16-224"
LANG_TOKENS = 32
VAL_FRAC = 0.05


def data_root() -> Path:
    return Path(os.environ["WM_DATA"])


def download(raw: Path):
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, repo_type="dataset", local_dir=raw, max_workers=8)
    snapshot_download(ENCODER)                      # into $HF_HOME for the encode job


class Episodes:
    """One item = one episode: (images uint8 [T,224,224,3], lowdim dict)."""

    def __init__(self, raw: Path):
        self.files = sorted(raw.glob("data/chunk-*/episode_*.parquet"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        import pyarrow.parquet as pq
        from PIL import Image
        t = pq.read_table(self.files[i]).to_pydict()
        imgs = np.stack([np.asarray(Image.open(BytesIO(im["bytes"])).convert("RGB")
                                    .resize((224, 224), Image.BILINEAR))
                         for im in t["image"]])
        low = {"episode": np.asarray(t["episode_index"], np.int32),
               "frame": np.asarray(t["frame_index"], np.int32),
               "task": np.asarray(t["task_index"], np.int32),
               "state": np.asarray(t["state"], np.float32),
               "action": np.asarray(t["actions"], np.float32)}
        return imgs, low


def encode(raw: Path, out: Path, batch: int = 256):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, SiglipTextModel, SiglipVisionModel

    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    meta = [json.loads(l) for l in (raw / "meta/episodes.jsonl").read_text().splitlines()]
    tasks = [json.loads(l)["task"] for l in (raw / "meta/tasks.jsonl").read_text().splitlines()]
    n_frames = sum(m["length"] for m in meta)

    # --- language: once per task ------------------------------------------ #
    tok = AutoTokenizer.from_pretrained(ENCODER)
    txt = SiglipTextModel.from_pretrained(ENCODER, torch_dtype=torch.float16).to(dev).eval()
    ids = tok(tasks, padding="max_length", max_length=64, truncation=True,
              return_tensors="pt").input_ids.to(dev)
    with torch.no_grad():
        h = txt(input_ids=ids).last_hidden_state[:, :LANG_TOKENS]
    np.save(out / "lang.npy", h.cpu().numpy().astype(np.float16))
    np.save(out / "lang_mask.npy", (ids[:, :LANG_TOKENS] != tok.pad_token_id).cpu().numpy())
    (out / "tasks.json").write_text(json.dumps(tasks, indent=1))
    del txt

    # --- vision: every agent-view frame ----------------------------------- #
    vis_model = SiglipVisionModel.from_pretrained(ENCODER, torch_dtype=torch.float16).to(dev).eval()
    mean = torch.tensor([0.5, 0.5, 0.5], device=dev).view(1, 3, 1, 1)
    vis = np.lib.format.open_memmap(out / "vis.npy", mode="w+", dtype=np.float16,
                                    shape=(n_frames, 49, 768))
    # Also kept for the efficiency study (README "Efficiency study"): raw 224 px
    # frames for the online-encoder arms, and unpooled tokens for the pooling arm.
    img = np.lib.format.open_memmap(out / "img.npy", mode="w+", dtype=np.uint8,
                                    shape=(n_frames, 224, 224, 3))
    vis196 = np.lib.format.open_memmap(out / "vis196.npy", mode="w+", dtype=np.float16,
                                       shape=(n_frames, 196, 768))
    ds = Episodes(raw)
    assert len(ds) == len(meta), (len(ds), len(meta))
    dl = DataLoader(ds, batch_size=None, num_workers=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)) - 1,
                    prefetch_factor=4)
    lows, starts, lengths, off = [], [], [], 0
    for e, (imgs, low) in enumerate(dl):
        T = imgs.shape[0]
        for s in range(0, T, batch):
            x = imgs[s:s + batch].to(dev).permute(0, 3, 1, 2).half().div(255)
            x = (x - mean) / 0.5                       # SigLIP normalisation
            with torch.no_grad():
                p = vis_model(pixel_values=x).last_hidden_state      # [b,196,768]
            vis196[off + s: off + s + p.shape[0]] = p.cpu().numpy()
            p = p.view(-1, 14, 14, 768).permute(0, 3, 1, 2)
            p = F.avg_pool2d(p.float(), 2).flatten(2).transpose(1, 2)  # [b,49,768]
            vis[off + s: off + s + p.shape[0]] = p.half().cpu().numpy()
        img[off: off + T] = imgs.numpy()
        lows.append({k: v.numpy() for k, v in low.items()})
        starts.append(off); lengths.append(T); off += T
        if e % 100 == 0:
            print(f"  episode {e}/{len(ds)}  frames {off}/{n_frames}", flush=True)
    assert off == n_frames, (off, n_frames)
    vis.flush(); img.flush(); vis196.flush()

    low = {k: np.concatenate([l[k] for l in lows]) for k in lows[0]}
    np.savez(out / "lowdim.npz", **low)

    # --- split: hold out VAL_FRAC of episodes per task --------------------- #
    ep_task = np.array([low["task"][s] for s in starts])
    split = np.zeros(len(starts), np.int8)
    rng = np.random.default_rng(0)
    for t in np.unique(ep_task):
        idx = np.flatnonzero(ep_task == t)
        split[rng.choice(idx, max(1, round(VAL_FRAC * len(idx))), replace=False)] = 1
    np.savez(out / "episodes.npz", start=np.array(starts), length=np.array(lengths),
             task=ep_task, split=split)

    # --- normalisation stats, train split only ----------------------------- #
    train = np.repeat(split == 0, lengths)
    stats = {k: {"q01": np.quantile(low[k][train], 0.01, axis=0).tolist(),
                 "q99": np.quantile(low[k][train], 0.99, axis=0).tolist(),
                 "mean": low[k][train].mean(0).tolist(),
                 "std": low[k][train].std(0).tolist()} for k in ("action", "state")}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(f"done: {n_frames} frames, {len(starts)} episodes "
          f"({int(split.sum())} val), {len(tasks)} tasks -> {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["download", "encode"], required=True)
    a = p.parse_args()
    raw = data_root() / "raw" / "pi_libero"
    if a.stage == "download":
        download(raw)
    else:
        encode(raw, data_root() / "libero_siglip")


if __name__ == "__main__":
    main()
