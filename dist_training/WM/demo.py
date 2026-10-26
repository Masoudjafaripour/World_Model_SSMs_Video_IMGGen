"""Offline demos of a trained policy (no simulator needed).

  python demo.py --run r1_S_1gpu

Writes to $WM_RUNS/<run>/demo/:
  input_reliance.json/.png  val BC loss when the instruction, the image or the robot
                            state is swapped for another sample's: what the policy uses
  actions_ep<e>.png         predicted vs demonstrated actions over a held-out episode
  lang_swap.png             one LIBERO-goal scene, every goal instruction: does the
                            predicted motion follow the language?
  attention_ep<e>.png       where the action tokens look in the current frame
  episode_ep<e>.gif         animated: frame + attention, actions with a moving cursor
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from libero_data import LiberoFeatures
from wla_model import N_LANG, N_PATCH, SIZES, WLAPolicy

ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]
# LIBERO-goal: one shared scene, ten instructions -- the cleanest language test
GOAL_TASKS = [
    "open the middle drawer of the cabinet", "put the bowl on the stove",
    "put the wine bottle on top of the cabinet", "open the top drawer and put the bowl inside",
    "put the bowl on top of the cabinet", "push the plate to the front of the stove",
    "put the cream cheese in the bowl", "turn on the stove", "put the bowl on the plate",
    "put the wine bottle on the rack",
]


def load(run_dir: Path, device):
    ck = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    model = WLAPolicy(SIZES[cfg["size"]])
    model.load_state_dict({k: v.float() for k, v in ck["model"].items()})
    return model.to(device).eval(), cfg, ck.get("step")


def to(b, device):
    # cached features are fp16; training ran under autocast, the demo runs in fp32
    return {k: v.to(device).float() if v.is_floating_point() else v.to(device)
            for k, v in b.items()}


def bc_l1(model, b):
    act, _ = model(b["lang"], b["lang_mask"], b["obs_prev"], b["obs_cur"], b["state"])
    valid = b["action_valid"].unsqueeze(-1)
    return ((act - b["action"]).abs() * valid).sum() / (valid.sum() * act.shape[-1])


@torch.no_grad()
def input_reliance(model, ds, device, out: Path, n_batches=60):
    """Val BC loss with each input replaced by another sample's (a batch roll)."""
    dl = DataLoader(ds, batch_size=256, shuffle=True, num_workers=6,
                    generator=torch.Generator().manual_seed(0))
    swaps = {"none (true inputs)": [], "instruction": ["lang", "lang_mask"],
             "image (t-1, t)": ["obs_prev", "obs_cur"], "robot state": ["state"]}
    tot = {k: 0.0 for k in swaps}
    n = 0
    for i, b in enumerate(dl):
        if i >= n_batches:
            break
        b = to(b, device)
        for name, keys in swaps.items():
            bb = dict(b)
            for k in keys:
                bb[k] = torch.roll(b[k], 1, 0)
            tot[name] += bc_l1(model, bb).item()
        n += 1
    res = {k: v / n for k, v in tot.items()}
    (out / "input_reliance.json").write_text(json.dumps(res, indent=1))
    fig, ax = plt.subplots(figsize=(6, 3))
    names = list(res)
    ax.barh(names[::-1], [res[k] for k in names[::-1]], color=["#888"] * 3 + ["#2a6"])
    for y, k in enumerate(names[::-1]):
        ax.text(res[k], y, f" {res[k]:.3f}", va="center")
    ax.set_xlabel("val BC loss (L1, normalised)  — higher = input matters more")
    ax.set_title("Which inputs does the policy use?\n(each input swapped with another sample's)")
    fig.tight_layout(); fig.savefig(out / "input_reliance.png", dpi=110); plt.close(fig)
    return res


@torch.no_grad()
def episode_actions(model, ds, img, device, e, out: Path, tasks):
    s, n = int(ds.ep_start[e]), int(ds.ep_len[e])
    idx = [i for i, (ee, _) in enumerate(ds.index) if ee == e]
    items = [ds[i] for i in idx]
    b = to({k: torch.stack([it[k] for it in items]) for k in items[0]}, device)
    act, _ = model(b["lang"], b["lang_mask"], b["obs_prev"], b["obs_cur"], b["state"])
    pred, gt = act[:, 0].cpu().numpy(), b["action"][:, 0].cpu().numpy()

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(3, 6, height_ratios=[1.1, 1, 1])
    for j, t in enumerate(np.linspace(0, n - 1, 6).astype(int)):
        a = fig.add_subplot(gs[0, j]); a.imshow(img[s + t]); a.set_title(f"t = {t}"); a.axis("off")
    T = np.arange(n)
    for d in range(7):
        a = fig.add_subplot(gs[1 + d // 4, d % 4]) if d < 4 else fig.add_subplot(gs[2, (d - 4) * 2:(d - 4) * 2 + 2])
        a.plot(T, gt[:, d], "k", lw=1.2, label="demonstration")
        a.plot(T, pred[:, d], "C1", lw=1.2, label="policy")
        a.set_title(ACTION_NAMES[d]); a.set_ylim(-1.1, 1.1)
        if d == 0:
            a.legend(fontsize=8)
    l1 = np.abs(pred - gt).mean()
    fig.suptitle(f"Held-out episode {e}: \"{tasks[int(ds.ep_task[e])]}\"   (mean |error| {l1:.3f}, "
                 f"actions normalised to [-1, 1])")
    fig.tight_layout(); fig.savefig(out / f"actions_ep{e}.png", dpi=90); plt.close(fig)
    return float(l1)


@torch.no_grad()
def lang_swap(model, ds, img, device, out: Path, tasks):
    goal_ids = [tasks.index(g) for g in GOAL_TASKS if g in tasks]
    eps = [e for e in range(len(ds.ep_task)) if int(ds.ep_task[e]) in goal_ids]
    if not goal_ids or not eps:
        print("lang_swap: no LIBERO-goal episode in this split; skipped")
        return None
    e = eps[0]
    i0 = [i for i, (ee, t) in enumerate(ds.index) if ee == e and t == 0][0]
    it = ds[i0]
    K = len(goal_ids)
    b = to({k: it[k].unsqueeze(0).repeat(K, *[1] * it[k].dim()) for k in it}, device)
    b["lang"] = ds.lang[goal_ids].to(device).float()
    b["lang_mask"] = ds.lang_mask[goal_ids].to(device)
    act, _ = model(b["lang"], b["lang_mask"], b["obs_prev"], b["obs_cur"], b["state"])
    act = act.cpu().numpy()                                   # [K, H, 7]
    path = np.cumsum(act[:, :, :3], axis=1)                   # integrated normalised deltas

    fig = plt.figure(figsize=(14, 5))
    a = fig.add_subplot(1, 3, 1); a.imshow(img[int(ds.ep_start[e])]); a.axis("off")
    a.set_title("same scene and state for every instruction")
    for j, (u, v, lbl) in enumerate([(0, 1, "x–y (top view)"), (0, 2, "x–z (side view)")]):
        a = fig.add_subplot(1, 3, 2 + j)
        for k, g in enumerate(goal_ids):
            a.plot(np.r_[0, path[k, :, u]], np.r_[0, path[k, :, v]], "-o", ms=2,
                   label=tasks[g] if j == 0 else None, color=plt.cm.tab10(k))
        a.set_title(f"predicted 8-step motion, {lbl}"); a.axis("equal")
        a.set_xlabel(ACTION_NAMES[u]); a.set_ylabel(ACTION_NAMES[v])
    fig.legend(loc="lower center", ncol=3, fontsize=7)
    fig.tight_layout(rect=(0, 0.16, 1, 1)); fig.savefig(out / "lang_swap.png", dpi=100); plt.close(fig)
    # spread between instructions vs the model's own error, both in normalised units
    spread = np.abs(act - act.mean(0, keepdims=True)).mean()
    return {"instructions": K, "mean_abs_deviation_between_instructions": float(spread)}


@torch.no_grad()
def attention(model, ds, img, device, e, out: Path, tasks):
    """Last-block attention from the action queries to the current frame's 7x7 patches."""
    blk = model.blocks[-1]
    cap = {}

    def hook(mod, args):
        x, mask = args
        B, L, D = x.shape
        h = mod.heads
        q, k, _ = mod.qkv(mod.n1(x)).view(B, L, 3, h, D // h).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(D // h)
        att = att.masked_fill(~mask, float("-inf")).softmax(-1).mean(1)       # [B,L,L]
        lo = N_LANG + N_PATCH                                                 # obs t block
        cap["a"] = att[:, -model.cfg.horizon:, lo:lo + N_PATCH].mean(1)       # [B,49]

    hd = blk.register_forward_pre_hook(hook)
    s, n = int(ds.ep_start[e]), int(ds.ep_len[e])
    ts = np.linspace(0, n - 1, 5).astype(int)
    idx = [i for i, (ee, t) in enumerate(ds.index) if ee == e and t in set(ts.tolist())]
    items = [ds[i] for i in idx]
    b = to({k: torch.stack([it[k] for it in items]) for k in items[0]}, device)
    model(b["lang"], b["lang_mask"], b["obs_prev"], b["obs_cur"], b["state"])
    hd.remove()
    att = cap["a"].view(-1, 1, 7, 7)
    att = F.interpolate(att / att.amax((2, 3), keepdim=True), size=224, mode="bilinear")[:, 0].cpu().numpy()

    fig, ax = plt.subplots(1, len(ts), figsize=(3 * len(ts), 3.4))
    for j, t in enumerate(ts):
        ax[j].imshow(img[s + t]); ax[j].imshow(att[j], cmap="jet", alpha=0.45)
        ax[j].set_title(f"t = {t}"); ax[j].axis("off")
    fig.suptitle(f"Action-token attention on the current frame (last layer): \"{tasks[int(ds.ep_task[e])]}\"")
    fig.tight_layout(); fig.savefig(out / f"attention_ep{e}.png", dpi=90); plt.close(fig)


@torch.no_grad()
def episode_gif(model, ds, img, device, e, out: Path, tasks, stride=2, fps=10):
    """Animated episode: frame + action-token attention (left), predicted vs
    demonstrated actions with a moving cursor (right)."""
    import imageio.v2 as imageio
    blk, cap = model.blocks[-1], {}

    def hook(mod, args):
        x, mask = args
        B, L, D = x.shape
        q, k, _ = mod.qkv(mod.n1(x)).view(B, L, 3, mod.heads, D // mod.heads).permute(2, 0, 3, 1, 4)
        att = ((q @ k.transpose(-1, -2)) / math.sqrt(D // mod.heads)) \
            .masked_fill(~mask, float("-inf")).softmax(-1).mean(1)
        lo = N_LANG + N_PATCH
        cap["a"] = att[:, -model.cfg.horizon:, lo:lo + N_PATCH].mean(1)

    s, n = int(ds.ep_start[e]), int(ds.ep_len[e])
    idx = [i for i, (ee, _) in enumerate(ds.index) if ee == e]
    items = [ds[i] for i in idx]
    b = to({k: torch.stack([it[k] for it in items]) for k in items[0]}, device)
    hd = blk.register_forward_pre_hook(hook)
    act, _ = model(b["lang"], b["lang_mask"], b["obs_prev"], b["obs_cur"], b["state"])
    hd.remove()
    pred, gt = act[:, 0].cpu().numpy(), b["action"][:, 0].cpu().numpy()
    att = cap["a"].view(-1, 1, 7, 7)
    att = F.interpolate(att / att.amax((2, 3), keepdim=True), size=224,
                        mode="bilinear")[:, 0].cpu().numpy()

    dims = [0, 1, 2, 6]                                   # dx, dy, dz, gripper
    fig = plt.figure(figsize=(9, 4.2), dpi=72)
    gs = fig.add_gridspec(len(dims), 2, width_ratios=[1, 1.25])
    ax_img = fig.add_subplot(gs[:, 0]); ax_img.axis("off")
    im = ax_img.imshow(img[s]); ov = ax_img.imshow(att[0], cmap="jet", alpha=0.4, vmin=0, vmax=1)
    cursors = []
    for r, d in enumerate(dims):
        a = fig.add_subplot(gs[r, 1])
        a.plot(gt[:, d], "k", lw=1, label="demo"); a.plot(pred[:, d], "C1", lw=1, label="policy")
        a.set_ylim(-1.1, 1.1); a.set_ylabel(ACTION_NAMES[d], fontsize=8); a.tick_params(labelsize=7)
        if r == 0:
            a.legend(fontsize=7, loc="upper right", ncol=2)
        cursors.append(a.axvline(0, color="r", lw=1))
    fig.suptitle(f"\"{tasks[int(ds.ep_task[e])]}\" (held-out demo; colour = where the policy looks)",
                 fontsize=9)
    fig.tight_layout()
    frames = []
    for t in range(0, n, stride):
        im.set_data(img[s + t]); ov.set_data(att[t])
        for c in cursors:
            c.set_xdata([t, t])
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    plt.close(fig)
    path = out / f"episode_ep{e}.gif"
    imageio.mimsave(path, frames, duration=1000 * stride / fps / 2, loop=0)   # 2x real time
    return path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="r1_S_1gpu")
    p.add_argument("--episodes", type=int, default=3)
    a = p.parse_args()
    data = Path(os.environ["WM_DATA"]) / "libero_siglip"
    run_dir = Path(os.environ["WM_RUNS"]) / a.run
    out = run_dir / "demo"; out.mkdir(exist_ok=True)
    device = torch.device("cuda")

    model, cfg, step = load(run_dir, device)
    ds = LiberoFeatures(str(data), "val", model.cfg.horizon, cfg["future_k"])
    img = np.load(data / "img.npy", mmap_mode="r")
    tasks = json.loads((data / "tasks.json").read_text())

    summary = {"run": a.run, "best_step": step, "input_reliance": input_reliance(model, ds, device, out)}
    # episodes from different tasks, spread over the val split
    picks, seen = [], set()
    for e in range(len(ds.ep_task)):
        if int(ds.ep_task[e]) not in seen:
            picks.append(e); seen.add(int(ds.ep_task[e]))
    picks = picks[:: max(1, len(picks) // a.episodes)][: a.episodes]
    summary["episode_l1"] = {int(e): episode_actions(model, ds, img, device, e, out, tasks) for e in picks}
    for e in picks:
        attention(model, ds, img, device, e, out, tasks)
        episode_gif(model, ds, img, device, e, out, tasks)
    summary["lang_swap"] = lang_swap(model, ds, img, device, out, tasks)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(f"DEMO DONE -> {out}")


if __name__ == "__main__":
    main()
