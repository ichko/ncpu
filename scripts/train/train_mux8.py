#!/usr/bin/env python3
"""
E_mux8 — Train an NCA to implement an 8-to-1 multiplexer.

    Y = D[S]

where S ∈ {0..7} (3-bit selector) picks which of the 8 data bits D[0..7] to output.

Grid: 64×96, r=4
  x=10  D[7:0]  — data inputs   (8 circles, MSB top)
  x=32  S[2:0]  — selector      (3 circles, MSB top, encodes 0–7)
  x=54  Y       — output        (1 circle)

Usage:
    uv run python scripts/train/train_mux8.py
    uv run python scripts/train/train_mux8.py --seed 1
    uv run python scripts/train/train_mux8.py --steps 50000
"""

import json
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.utils import freeze_frame, git_info, make_grid, save_grid_image

# ── Args ───────────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--seed",   type=int, default=0)
parser.add_argument("--steps",  type=int, default=30_000)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(args.seed)
DEVICE = args.device if torch.cuda.is_available() else "cpu"

# ── Config ─────────────────────────────────────────────────────────────────────

TOTAL_STEPS = args.steps
PLOT_EVERY  = 500
SAVE_EVERY  = 5000

H, W     = 96, 64
R        = 4
AMONG_SP = 2
STEP_PX  = 2 * R + AMONG_SP   # 10 px
X_DATA   = 10
X_SEL    = 32
X_OUT    = 54

NCA_CFG = dict(
    channels=16,
    hidden_channels=[128],
    fire_rate=0.5,
    alive_threshold=0,
    zero_initialization=False,
    kernel_size=7,
    num_perception_kernels=3,
    read_only_dims=[1],
    padding_type="zeros",
)
OPTIM = dict(lr=1e-4, batch_size=16, steps_min=48, steps_max=96)
N_LAST = 8

# ── Screen helpers ─────────────────────────────────────────────────────────────

def _draw_col(screen, bits, cx):
    if bits is None:
        return
    n  = len(bits)
    v  = n * 2*R + AMONG_SP * (n - 1)
    tm = (H - v) // 2
    for i, bit in enumerate(bits):
        cy = tm + R + i * STEP_PX
        cv2.circle(screen, (cx, cy), R, 255 if bit else 0, -1)

def make_screen(d_bits=None, s_bits=None, out_bit=None):
    screen = np.full((H, W), 128, dtype=np.uint8)
    _draw_col(screen, d_bits, X_DATA)
    _draw_col(screen, s_bits, X_SEL)
    if out_bit is not None:
        _draw_col(screen, [out_bit], X_OUT)
    return screen

def int_to_bits(n, width):
    return [(n >> (width - 1 - i)) & 1 for i in range(width)]

def get_sample():
    d = [random.randint(0, 1) for _ in range(8)]
    s = random.randint(0, 7)
    y = d[s]
    inp = make_screen(d_bits=d, s_bits=int_to_bits(s, 3))
    out = make_screen(out_bit=y)
    return torch.from_numpy(inp).float(), torch.from_numpy(out).float()

# Precompute masks
out_mask = torch.from_numpy(make_screen(out_bit=1) > 200).float().to(DEVICE)  # (H,W)

# ── Run directory ──────────────────────────────────────────────────────────────

ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path("runs") / f"E_mux8_s{args.seed}_{ts}"
run_dir.mkdir(parents=True)
(run_dir / "checkpoints").mkdir()
(run_dir / "rollouts").mkdir()
(run_dir / "snapshots").mkdir()

json.dump({
    "run":  {"name": run_dir.name, "experiment": "E_mux8", "seed": args.seed,
             "total_steps": TOTAL_STEPS, "device": DEVICE},
    "ds":   {"H": H, "W": W, "r": R, "among_sp": AMONG_SP,
             "x_data": X_DATA, "x_sel": X_SEL, "x_out": X_OUT},
    "nca":  NCA_CFG,
    "optim": OPTIM,
    "env":  {"python": sys.version, "torch": torch.__version__, "git": git_info()},
}, open(run_dir / "config.json", "w"), indent=2)

log_path = run_dir / "log.jsonl"
ckpt_dir = run_dir / "checkpoints"
print(f"E_mux8  seed={args.seed}  run: {run_dir.name}\n")

# ── Model ──────────────────────────────────────────────────────────────────────

nca   = NeuralCA(**NCA_CFG).to(DEVICE)
optim = torch.optim.Adam(nca.parameters(), lr=OPTIM["lr"])

# ── Loss helpers ───────────────────────────────────────────────────────────────

def rollout_loss(rollout, out_norm):
    pred   = rollout[:, -N_LAST:, 0]                       # (B, N, H, W)
    target = out_norm.unsqueeze(1).expand_as(pred)
    mask   = out_mask.unsqueeze(0).unsqueeze(0)
    return ((pred - target) ** 2 * mask).sum() / (mask.sum() * pred.shape[0] * N_LAST)

def count_valid(ch0_last, out_norm):
    pred   = (ch0_last * out_mask).sum(dim=(-2,-1)) / out_mask.sum()   # (B,)
    target = (out_norm  * out_mask).sum(dim=(-2,-1)) / out_mask.sum()  # (B,)
    return ((pred > 0) == (target > 0)).float().mean().item()

# ── Training loop ──────────────────────────────────────────────────────────────

metrics = []

pbar = tqdm(range(TOTAL_STEPS), ncols=100, desc=f"mux8 s{args.seed}")

for step in pbar:
    # Sample batch
    batch     = [get_sample() for _ in range(OPTIM["batch_size"])]
    inp_batch = torch.stack([s[0] for s in batch]).to(DEVICE)   # (B, H, W)
    out_batch = torch.stack([s[1] for s in batch]).to(DEVICE)   # (B, H, W)

    inp_norm = inp_batch / 128.0 - 1.0
    out_norm = out_batch / 128.0 - 1.0

    # Initial state
    state = torch.zeros(OPTIM["batch_size"], NCA_CFG["channels"], H, W, device=DEVICE)
    state[:, 0] = inp_norm
    state[:, 1] = inp_norm

    # Forward
    T       = random.randint(OPTIM["steps_min"], OPTIM["steps_max"])
    rollout = nca(state, steps=T)   # (B, T+1, C, H, W)

    loss = rollout_loss(rollout, out_norm)
    optim.zero_grad()
    loss.backward()
    optim.step()

    with torch.no_grad():
        valid = count_valid(rollout[:, -1, 0], out_norm)

    loss_val = loss.item()
    metrics.append({"step": step, "loss": loss_val, "valid": valid})

    pbar.set_description(f"mux8 s{args.seed}  loss={loss_val:.4f}  acc={valid*100:.1f}%")

    with open(log_path, "a") as f:
        f.write(json.dumps({"step": step, "loss": loss_val, "num_valid_bits": valid}) + "\n")

    if step % PLOT_EVERY != 0:
        continue

    # Checkpoint
    torch.save(nca.state_dict(), ckpt_dir / f"nca_{step:07d}.pt")
    torch.save(nca.state_dict(), ckpt_dir / "nca_latest.pt")

    # Curves
    losses = [m["loss"]  for m in metrics]
    accs   = [m["valid"] for m in metrics]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3))
    axes[0].scatter(range(len(losses)), losses, s=0.4, alpha=0.3, color="steelblue")
    axes[0].set_yscale("log"); axes[0].set_title("Loss"); axes[0].set_xlabel("step")
    axes[1].scatter(range(len(accs)), [a * 100 for a in accs],
                    s=0.4, alpha=0.3, color="darkorange")
    axes[1].set_ylim(0, 105); axes[1].set_title("Accuracy (%)")
    axes[1].set_xlabel("step")
    fig.suptitle(f"E_mux8 — seed={args.seed}  step={step}", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "curves.png", dpi=110)
    plt.close(fig)

    # Snapshot: 4 random examples
    snap_b = min(OPTIM["batch_size"], 4)
    save_grid_image(
        run_dir / "snapshot_latest.png",
        [inp_norm[:snap_b].cpu(), out_norm[:snap_b].cpu(),
         rollout[:snap_b, -1, 0].detach().cpu(),
         (rollout[:snap_b, -1, 0].detach() - out_norm[:snap_b]).abs().cpu()],
        row_vmin=[None, None, None, 0],
        row_vmax=[None, None, None, 2],
    )

    # Rollout GIF
    gif_b  = min(OPTIM["batch_size"], 4)
    gif_t  = rollout[:gif_b, ::2]
    C_dim  = gif_t.shape[2]
    frames = gif_t.permute(2, 0, 1, 3, 4).reshape(
        C_dim * gif_b, gif_t.shape[1], H, W
    ).detach().cpu().numpy()
    frames_rgb = torch.from_numpy(
        media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis")
    )
    grid = make_grid(frames_rgb, nrow=gif_b, padding=1)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=8)
    media.write_video(str(run_dir / "rollout_latest.gif"),
                      grid.numpy(), fps=10, codec="gif")

    if step % SAVE_EVERY == 0 and step > 0:
        import shutil
        shutil.copy(run_dir / "snapshot_latest.png",
                    run_dir / "snapshots" / f"snapshot_{step:07d}.png")
        shutil.copy(run_dir / "rollout_latest.gif",
                    run_dir / "rollouts"  / f"rollout_{step:07d}.gif")

print(f"\nDone. Run: {run_dir}")
