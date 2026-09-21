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

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")         # headless-safe: results are saved to disk, not shown
import matplotlib.pyplot as plt


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
    bfs_oracle_path, to_rgb, show_maze_state,
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
def flow_loss(model, obs, action, next_obs):
    x0 = torch.randn_like(next_obs)
    t = torch.rand(obs.shape[0], device=obs.device)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * next_obs

    v_pred = model(obs, xt, t, action)
    v_target = next_obs - x0
    return F.mse_loss(v_pred, v_target)


@torch.no_grad()
def sample_next(model, obs, action, steps=4):
    x = torch.randn_like(obs)
    dt = 1.0 / steps

    for i in range(steps):
        t = torch.full((obs.shape[0],), i * dt, device=obs.device)
        x = x + model(obs, x, t, action) * dt

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

    return path, frames


def plot_rollout(path, frames, max_show=12, save_path=RESULTS_DIR / "ViT_Video_WM_rollout.png"):
    n = min(len(frames), max_show)
    plt.figure(figsize=(2.2 * n, 2.5))

    for t in range(n):
        plt.subplot(1, n, t + 1)
        show_maze_state(frames[t], f"t={t}\n{path[t]}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train / fine-tune the ViT diffusion maze world model.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--steps-per-epoch", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to fine-tune from")
    parser.add_argument("--save", type=str, default=str(RESULTS_DIR / "ViT_Video_WM.pt"))
    parser.add_argument("--sample-steps", type=int, default=4, help="Euler steps at inference")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("Device:", DEVICE)
    print("Results dir:", RESULTS_DIR)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

    learned_path, learned_frames = plan_with_learned_model(model, steps=args.sample_steps)
    print("\nLearned ViT-diffusion path:")
    print(learned_path)

    oracle = bfs_oracle_path()
    print("\nTrue BFS shortest path:")
    print(oracle)

    if not args.no_plot:
        plot_rollout(learned_path, learned_frames)
        print(f"Saved rollout plot to {RESULTS_DIR / 'ViT_Video_WM_rollout.png'}")
