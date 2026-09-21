# ViT + Rectified-Flow Diffusion Video World Model

[`ViT_Video_Model.py`](ViT_Video_Model.py) predicts the next maze frame given the current frame + action, using a small **Vision Transformer trained as a rectified-flow denoiser**: instead of regressing the next frame in one forward pass (as [`../Video_GridWM.py`](../Video_GridWM.py)'s CNN does), it predicts a *velocity field* and integrates it over a few Euler steps at inference time.

## Method

* **Tokens = maze cells.** Each 6×6 patch is exactly one grid cell (7×7 = 49 tokens), so tokens align 1:1 with cells — no patch/cell decoding ambiguity.
* **Conditioning.** Current-frame patches are concatenated (channel-wise) with the noisy next-frame patches before the patch embedding, so the ViT attends jointly over "what is" and "what might be next." The action and diffusion timestep are injected per-block via **adaLN-zero** modulation (same mechanism as `dist_training/model.py`'s `VideoDiT`, here at toy scale: dim 96, depth 4, ~0.73M params — small enough to fine-tune in minutes on CPU/GPU).

## Equations

Rectified-flow interpolation between noise and the true next frame $x_1$:

$$x_0 \sim \mathcal{N}(0, I), \qquad t \sim \mathcal{U}(0,1), \qquad x_t = (1-t)\,x_0 + t\,x_1$$

Training target is the constant velocity along that straight-line path, regressed with MSE:

$$\mathcal{L}(\theta) = \mathbb{E}_{x_1, x_0, t}\left[\, \lVert v_\theta(x_t, t, o, a) - (x_1 - x_0) \rVert^2 \,\right]$$

where $o$ is the current frame and $a$ the action, entering via adaLN-zero:

$$c = \mathrm{MLP}_t(\mathrm{sinusoid}(t)) + \mathrm{Embed}_a(a), \qquad [\gamma_1,\beta_1,\alpha_1,\gamma_2,\beta_2,\alpha_2] = \mathrm{MLP}(c)$$
$$x \mathrel{+}= \alpha_1 \odot \mathrm{Attn}\big(\mathrm{LN}(x)(1+\gamma_1)+\beta_1\big), \qquad x \mathrel{+}= \alpha_2 \odot \mathrm{MLP}\big(\mathrm{LN}(x)(1+\gamma_2)+\beta_2\big)$$

Inference does $K$-step Euler integration from pure noise:

$$x_{k+1} = x_k + v_\theta(x_k, t_k, o, a)\,\Delta t, \qquad \Delta t = 1/K,\ \ x_0 \sim \mathcal{N}(0,I)$$

## Code

```python
class ViTVideoWorldModel(nn.Module):
    def __init__(self, dim=96, depth=4, n_heads=4, mlp_ratio=4):
        ...
        self.patch_embed = nn.Linear(2 * patch_dim, dim)   # obs || noisy-next
        self.t_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.action_emb = nn.Embedding(N_ACTIONS, dim)
        self.blocks = nn.ModuleList(Block(dim, n_heads, mlp_ratio) for _ in range(depth))

    def forward(self, obs, xt, t, action):
        h = self.patch_embed(torch.cat([patchify(obs), patchify(xt)], dim=-1)) + self.pos
        c = self.t_mlp(timestep_embedding(t * 1000, self.dim)) + self.action_emb(action)
        for block in self.blocks:
            h = block(h, c)
        return unpatchify(self.out(self.out_norm(h)))


def flow_loss(model, obs, action, next_obs):
    x0 = torch.randn_like(next_obs)
    t = torch.rand(obs.shape[0], device=obs.device)
    xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * next_obs
    return F.mse_loss(model(obs, xt, t, action), next_obs - x0)


@torch.no_grad()
def sample_next(model, obs, action, steps=4):
    x = torch.randn_like(obs)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((obs.shape[0],), i * dt, device=obs.device)
        x = x + model(obs, x, t, action) * dt
    return x
```

## Commands

The script uses the `Agg` matplotlib backend (headless-safe) and writes everything to `ViT/results/` (self-contained next to this model) — no `--save` path needed for the default run.

```bash
# Pre-train from scratch (~2 min on a single GPU: 15 epochs x 150 steps)
python ViT_Video_Model.py --epochs 15 --steps-per-epoch 150 --batch-size 32 --lr 3e-4

# Fine-tune from a checkpoint (quick: few epochs, lower LR)
python ViT_Video_Model.py \
    --resume results/ViT_Video_WM.pt --epochs 3 --steps-per-epoch 100 --lr 1e-4 \
    --save results/ViT_Video_WM_ft.pt

# More Euler steps at inference (slower, usually sharper next-frame samples), no plotting
python ViT_Video_Model.py --resume results/ViT_Video_WM.pt --epochs 0 --sample-steps 8 --no-plot
```

Each run writes to `results/` (created next to `ViT_Video_Model.py`):

* `ViT_Video_WM.pt` — model checkpoint (`--save` to change the path; `--resume` to continue from one)
* `ViT_Video_WM_loss.png` — training loss curve, rewritten after every epoch so you can watch it update mid-run
* `ViT_Video_WM_rollout.png` — the learned-model greedy-planning rollout (skipped with `--no-plot`)

## Result

Flow loss drops from ~0.84 to a ~0.35 plateau over 15 epochs (2250 steps, ~2 min on a V100). The denoiser learns plausible next-frame structure, but the same one-step-greedy planner used by `Video_GridWM.py` still struggles on the maze — it reaches fewer, noisier hops than the CNN baseline and is more prone to a slightly wrong agent-position decode, since diffusion samples are noisier than the CNN's direct regression. This isolates the greedy-planning limitation (already noted in the top-level README for the CNN maze model) from the quality of the dynamics model itself.
