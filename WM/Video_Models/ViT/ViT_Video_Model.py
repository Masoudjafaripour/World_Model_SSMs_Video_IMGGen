# ViT_Video_Model.py
# Concise ViT + rectified-flow diffusion world model for the maze environment.
# Same task/data as Video_GridWM.py (current frame + action -> next frame),
# but the CNN regressor is replaced with a small ViT that denoises the next
# frame in pixel space, conditioned on the current frame (concatenated in
# patch space) and the action + diffusion time (via adaLN-zero). Small
# enough (~1M params) to fine-tune in well under a minute on CPU/GPU.

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")         # headless-safe: results are saved to disk, not shown
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def _find_upward(start: Path, name: str) -> Path:
    """Walk up from `start` to the nearest ancestor containing `name`."""
    for parent in (start, *start.parents):
        if (parent / name).exists():
            return parent
    raise FileNotFoundError(f"could not locate '{name}' above {start}")


_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_find_upward(_HERE, "Video_GridWM.py")))

from Video_GridWM import (
    GRID, CELL, IMG, DEVICE, START, GOAL,
    render_state, step, sample_batch, decode_agent,
    bfs_oracle_path, to_rgb, show_maze_state, ACTION_NAMES,
)

PATCH = CELL                 # one ViT patch == one maze cell
N_PATCHES = GRID * GRID
N_ACTIONS = 4
N_CHANNELS = 4                # walls / start / goal / agent

RESULTS_DIR = _HERE / "results"       # ViT/results, self-contained with this model
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Patchify / unpatchify (non-overlapping, grid-aligned with the maze cells)
# -----------------------------
def patchify(img):
    B, C, _, _ = img.shape
    p = img.unfold(2, PATCH, PATCH).unfold(3, PATCH, PATCH)          # B,C,G,G,P,P
    return p.permute(0, 2, 3, 1, 4, 5).reshape(B, N_PATCHES, C * PATCH * PATCH)


def unpatchify(tokens, channels=N_CHANNELS):
    B = tokens.shape[0]
    t = tokens.view(B, GRID, GRID, channels, PATCH, PATCH)
    return t.permute(0, 3, 1, 4, 2, 5).reshape(B, channels, IMG, IMG)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * freqs[None]
    return torch.cat([a.cos(), a.sin()], dim=-1)


# -----------------------------
# Tiny ViT block (pre-norm attention + MLP, adaLN-zero conditioning)
# -----------------------------
class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        B, S, _ = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        return self.proj(o.transpose(1, 2).reshape(B, S, -1))


class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio=4):
        super().__init__()
        hidden = mlp_ratio * dim
        self.n1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(dim, n_heads)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.mod[1].weight)
        nn.init.zeros_(self.mod[1].bias)

    def forward(self, x, c):
        sa, ga, ba, sm, gm, bm = self.mod(c).unsqueeze(1).chunk(6, dim=-1)
        x = x + ga * self.attn(self.n1(x) * (1 + sa) + ba)
        x = x + gm * self.fc2(F.gelu(self.fc1(self.n2(x) * (1 + sm) + bm)))
        return x


class ViTVideoWorldModel(nn.Module):
    """
    Rectified-flow diffusion world model: predicts the next maze frame given
    the current frame + action, by denoising in pixel space with a small ViT.
    Current-frame patches are concatenated (channel-wise) with the noisy
    next-frame patches, so the ViT attends jointly over "what is" and "what
    might be next"; action + diffusion time enter via adaLN-zero.

    The network directly predicts the clean target frame x1 (not the
    velocity x1-x0): velocity targets blow up as t->1 (dividing by (1-t)),
    which this tiny model + plain AdamW training can't fit well, while x1
    is bounded in [0,1] regardless of t. Velocity is only reconstructed at
    sampling time, where the (1-t) factor stays well away from 0 by
    construction (see sample_next).
    """

    def __init__(self, dim=96, depth=4, n_heads=4, mlp_ratio=4):
        super().__init__()
        self.dim = dim
        patch_dim = N_CHANNELS * PATCH * PATCH

        self.patch_embed = nn.Linear(2 * patch_dim, dim)   # obs || noisy-next
        self.pos = nn.Parameter(torch.zeros(1, N_PATCHES, dim))
        nn.init.normal_(self.pos, std=0.02)

        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_emb = nn.Embedding(N_ACTIONS, dim)

        self.blocks = nn.ModuleList(Block(dim, n_heads, mlp_ratio) for _ in range(depth))

        self.out_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.out = nn.Linear(dim, patch_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, obs, xt, t, action):
        h = self.patch_embed(torch.cat([patchify(obs), patchify(xt)], dim=-1)) + self.pos
        c = self.t_mlp(timestep_embedding(t * 1000, self.dim)) + self.action_emb(action)

        for block in self.blocks:
            h = block(h, c)

        return unpatchify(self.out(self.out_norm(h)))


# -----------------------------
# Rectified-flow objective + few-step sampling
# -----------------------------
def flow_loss(model, obs, action, next_obs, pos_weight=12.0):
    x0 = torch.randn_like(next_obs)
    t = torch.rand(obs.shape[0], device=obs.device)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * next_obs

    x1_pred = model(obs, xt, t, action)   # predicts the clean frame directly

    # start/goal/agent pixels are a tiny fraction of the image; unweighted MSE
    # lets the model shrug them off and still get a low loss (same issue
    # Video_GridWM.py's CNN loss corrects for with the same 12x weight)
    weight = torch.ones_like(next_obs)
    weight[next_obs > 0.5] = pos_weight
    return (weight * (x1_pred - next_obs) ** 2).mean()


@torch.no_grad()
def sample_next(model, obs, action, steps=4):
    """DDIM-style update from an x1-predicting model: at step k (t_k=k/steps),
    move a 1/(steps-k) fraction of the way from x toward the model's current
    x1 prediction. The last step (k=steps-1) always lands exactly on x1_pred,
    so (1-t) is never divided anywhere near 0."""
    x = torch.randn_like(obs)

    for k in range(steps):
        t = torch.full((obs.shape[0],), k / steps, device=obs.device)
        x1_pred = model(obs, x, t, action)
        x = x + (x1_pred - x) / (steps - k)

    return x


def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)


def load_checkpoint(path):
    model = ViTVideoWorldModel().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    return model


def plot_loss_curve(losses, path=RESULTS_DIR / "ViT_Video_WM_loss.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(losses) + 1), losses, marker="o")
    plt.xlabel("epoch")
    plt.ylabel("flow loss")
    plt.title("ViT diffusion world model - training loss")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_model(epochs=15, steps_per_epoch=150, batch_size=32, lr=3e-4, model=None,
                 loss_plot_path=RESULTS_DIR / "ViT_Video_WM_loss.png"):
    model = model or ViTVideoWorldModel().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        total = 0.0

        for _ in range(steps_per_epoch):
            obs, action, next_obs = sample_batch(batch_size)
            loss = flow_loss(model, obs, action, next_obs)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total += loss.item()

        losses.append(total / steps_per_epoch)
        print(f"epoch {epoch + 1:02d} | flow loss {losses[-1]:.5f}")

        # rewritten every epoch so you can watch training progress mid-run
        if loss_plot_path is not None:
            plot_loss_curve(losses, loss_plot_path)

    return model, losses


# -----------------------------
# Prediction and greedy planning (mirrors Video_GridWM.py)
# -----------------------------
@torch.no_grad()
def predict_next(model, agent, action, steps=4):
    obs = render_state(agent).unsqueeze(0).to(DEVICE)
    act = torch.tensor([action], device=DEVICE)

    pred = sample_next(model, obs, act, steps=steps)[0].cpu().clamp(0, 1)
    pred_agent = decode_agent(pred)

    return pred_agent, pred


@torch.no_grad()
def plan_with_learned_model(model, start=START, goal=GOAL, max_steps=30, steps=4):
    agent = start
    path = [agent]
    frames = [render_state(agent)]
    actions = []           # actions[i] is the move taken from path[i] to path[i+1]
    visited = set()

    for _ in range(max_steps):
        if agent == goal:
            break

        best = None

        for action in range(4):
            pred_agent, pred_img = predict_next(model, agent, action, steps=steps)

            dist = abs(pred_agent[0] - goal[0]) + abs(pred_agent[1] - goal[1])
            revisit_penalty = 3.0 if pred_agent in visited else 0.0
            stuck_penalty = 1.0 if pred_agent == agent else 0.0
            score = dist + revisit_penalty + stuck_penalty

            if best is None or score < best[0]:
                best = (score, action, pred_agent, pred_img)

        _, action, agent, pred_img = best

        visited.add(agent)
        path.append(agent)
        frames.append(pred_img)
        actions.append(action)

    return path, frames, actions


# -----------------------------
# Rollout visualization: static grid (PNG) + animated GIF/MP4, all with
# readable per-frame captions (position, action taken, distance-to-goal).
# -----------------------------
LEGEND_ITEMS = [
    ("walls", (0.0, 0.0, 0.0)),
    ("start", (0.85, 0.1, 0.1)),
    ("goal", (0.0, 0.8, 0.2)),
    ("agent", (0.1, 0.2, 0.9)),
]


def _frame_caption(t, pos, goal, action=None):
    dist = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
    tag = "  ✓ goal" if pos == goal else f"  dist={dist}"
    if action is None:
        return f"t={t}  start {pos}{tag}"
    return f"t={t}  {ACTION_NAMES[action]} → {pos}{tag}"


def _draw_frame(ax, img, caption):
    ax.imshow(to_rgb(img))
    ax.set_title(caption, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_rollout(path, frames, actions, goal=GOAL, max_show=12,
                  save_path=RESULTS_DIR / "ViT_Video_WM_rollout.png"):
    n = min(len(frames), max_show)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 3.0))
    axes = [axes] if n == 1 else list(axes)

    for t in range(n):
        action = actions[t - 1] if t > 0 else None
        _draw_frame(axes[t], frames[t], _frame_caption(t, path[t], goal, action))

    fig.legend(handles=[Patch(facecolor=c, label=l) for l, c in LEGEND_ITEMS],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.02), frameon=False)
    steps_shown = "" if n == len(frames) else f" (first {n} of {len(frames)} steps)"
    fig.suptitle(f"Greedy rollout: {path[0]} → {goal}{steps_shown}")
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def _frame_to_array(img, caption, figsize=(3.2, 3.52), dpi=100):
    # dims chosen to land on multiples of 16 (320x352) so ffmpeg's mp4 encoder
    # doesn't need to pad/resize each frame
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    _draw_frame(ax, img, caption)
    fig.legend(handles=[Patch(facecolor=c, label=l) for l, c in LEGEND_ITEMS],
               loc="lower center", ncol=4, fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


def save_rollout_video(path, frames, actions, goal=GOAL, fps=2, hold_last_secs=1.5,
                        gif_path=RESULTS_DIR / "ViT_Video_WM_rollout.gif",
                        mp4_path=RESULTS_DIR / "ViT_Video_WM_rollout.mp4"):
    """Render the full rollout as an annotated GIF + MP4 (needs `imageio[ffmpeg]`)."""
    try:
        import imageio.v2 as imageio
    except ImportError as e:
        raise ImportError(
            "GIF/MP4 export needs imageio: pip install imageio imageio-ffmpeg"
        ) from e

    arrays = [
        _frame_to_array(frames[t], _frame_caption(t, path[t], goal, actions[t - 1] if t > 0 else None))
        for t in range(len(frames))
    ]
    arrays += [arrays[-1]] * max(0, round(fps * hold_last_secs) - 1)   # hold on final frame

    imageio.mimsave(gif_path, arrays, duration=1.0 / fps, loop=0)
    imageio.mimsave(mp4_path, arrays, fps=fps)

    return gif_path, mp4_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train / fine-tune the ViT diffusion maze world model.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to fine-tune from (or run inference on)")
    parser.add_argument("--save", type=str, default=str(RESULTS_DIR / "ViT_Video_WM.pt"))
    parser.add_argument("--denoising-steps", type=int, default=4,
                         help="Euler integration steps for rectified-flow sampling at inference")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--infer-only", action="store_true",
                         help="skip training; requires --resume, just re-runs planning + saves rollout outputs")
    args = parser.parse_args()

    print("Device:", DEVICE)
    print("Results dir:", RESULTS_DIR)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if args.infer_only:
        if not args.resume:
            parser.error("--infer-only requires --resume <checkpoint>")
        model = load_checkpoint(args.resume)
        print(f"Loaded checkpoint from {args.resume} (no training)")
    else:
        init_model = load_checkpoint(args.resume) if args.resume else None

        model, losses = train_model(
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            lr=args.lr,
            model=init_model,
        )

        save_checkpoint(model, args.save)
        print(f"Saved checkpoint to {args.save}")
        print(f"Saved loss curve to {RESULTS_DIR / 'ViT_Video_WM_loss.png'}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params / 1e6:.2f}M")

    learned_path, learned_frames, learned_actions = plan_with_learned_model(model, steps=args.denoising_steps)
    print("\nLearned ViT-diffusion path:")
    print(learned_path)

    oracle = bfs_oracle_path()
    print("\nTrue BFS shortest path:")
    print(oracle)

    if not args.no_plot:
        plot_rollout(learned_path, learned_frames, learned_actions)
        print(f"Saved rollout plot to {RESULTS_DIR / 'ViT_Video_WM_rollout.png'}")

        gif_path, mp4_path = save_rollout_video(learned_path, learned_frames, learned_actions)
        print(f"Saved rollout gif to {gif_path}")
        print(f"Saved rollout mp4 to {mp4_path}")
