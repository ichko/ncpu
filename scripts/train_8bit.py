#!/usr/bin/env python3
"""
8-bit adder NCA training script.

Usage:
    uv run python scripts/train_8bit.py

Outputs (under runs/<timestamp>/):
    config.json          — run configuration
    log.jsonl            — one JSON line per step: {"step": N, "loss": X}
    loss_curve.png       — overwritten every PLOT_EVERY steps
    bits_curve.png       — overwritten every PLOT_EVERY steps
    rollout_NNNNNNN.gif  — channel-0 rollout grid saved every PLOT_EVERY steps
    snapshot_NNNNNNN.png — 4-row IO snapshot saved every PLOT_EVERY steps
    checkpoints/         — NCA weights + trainer pickle
"""

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
from tqdm import tqdm
from argparse import Namespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.dataset import NCPUDataset, sample_8bit_adder
from ncpu.nca import NeuralCA
from ncpu.trainer import NCPUTrainer
from ncpu.utils import freeze_frame, git_info, make_grid, save_grid_image


# ── Config ────────────────────────────────────────────────────────────────────
# 16 input bits (8+8) → 2 columns of 8 rows each
# 9 output bits (max sum = 510 < 512)
#
# Grid 128×128, same side_spacing=17 as 4-bit for consistent visual separation.
#   input col 0 at x=17, col 1 at x=27, output at x=111
#   ~76px edge-to-edge gap between input and output

DEVICE = "cuda"
TOTAL_STEPS = 200_000
PLOT_EVERY = 500

ds_config = Namespace(
    W=80,
    H=112,
    r=4,
    spacing=(2, 21),
    sampler=sample_8bit_adder,
    balanced=False,
)

nca_config = Namespace(
    channels=16,
    hidden_channels=[128],
    fire_rate=0.5,
    alive_threshold=0,
    zero_initialization=False,
    kernel_size=5,
    num_perception_kernels=3,
    read_only_dims=[1],
)

optim_config = Namespace(
    lr=0.0001,
    batch_size=8,
    gaussian_noise=-1,
    grad_clip=None,
    steps_min=64,
    steps_max=128,
)

# ── Run directory ─────────────────────────────────────────────────────────────
run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = Path("runs") / run_name
run_dir.mkdir(parents=True, exist_ok=True)
rollouts_dir = run_dir / "rollouts"
rollouts_dir.mkdir()
snapshots_dir = run_dir / "snapshots"
snapshots_dir.mkdir()

print(f"Device : {DEVICE}")
print(f"Run dir: {run_dir}\n")

with open(run_dir / "config.json", "w") as f:
    json.dump(
        {
            "run": {
                "name": run_name,
                "total_steps": TOTAL_STEPS,
                "plot_every": PLOT_EVERY,
                "device": DEVICE,
            },
            "ds": {**vars(ds_config), "sampler": ds_config.sampler.__name__},
            "nca": vars(nca_config),
            "optim": vars(optim_config),
            "env": {
                "python": sys.version,
                "torch": torch.__version__,
                "git": git_info(),
            },
        },
        f,
        indent=2,
    )

# ── Model & trainer ───────────────────────────────────────────────────────────
dataset = NCPUDataset(ds_config)
nca = NeuralCA(**vars(nca_config)).to(DEVICE)
trainer = NCPUTrainer(
    nca,
    dataset.get_dataloader(batch_size=optim_config.batch_size),
    lr=optim_config.lr,
    gaussian_noise=optim_config.gaussian_noise,
    grad_clip=optim_config.grad_clip,
    checkpoint_pattern=str(run_dir / "checkpoints" / "nca_{step:06d}.pt"),
)

trainer.sanity_check()

log_path = run_dir / "log.jsonl"
print(f"\nLogging to: {log_path}")
print(f"{'─'*60}\n")

# ── Training loop ─────────────────────────────────────────────────────────────
pbar = tqdm(range(TOTAL_STEPS), ncols=100)

for step in pbar:
    info = trainer.optim_step(steps=(optim_config.steps_min, optim_config.steps_max))
    loss = info["loss"]
    num_valid_bits = info["num_valid_bits"]

    grad_norm = trainer.metrics[-1].get("grad_norm") if trainer.metrics else None
    pbar.set_description(
        f"loss={loss:.4f}  bits={num_valid_bits:.2f}"
        + (f"  gnorm={grad_norm:.3f}" if grad_norm else "")
    )

    with open(log_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "step": step,
                    "loss": loss,
                    "num_valid_bits": num_valid_bits,
                    "grad_norm": grad_norm,
                }
            )
            + "\n"
        )

    if step % PLOT_EVERY != 0:
        continue

    # ── Periodic reporting ────────────────────────────────────────────────
    rollout = info["rollout"]  # (B, T, C, H, W) on CPU
    nca_out = info["nca_out"]  # (B, H, W)
    out = info["out"]  # (B, H, W)
    inp = info["inp"]  # (B, H, W)
    B, T, C, H, W = rollout.shape

    print(f"\n{'─'*60}")
    print(f"  step     : {step}")
    print(
        f"  loss     : {loss:.8f}   bits: {num_valid_bits:.2f} / {len(trainer.bit_masks)}"
    )
    print(
        f"  nca_out  : min={nca_out.min():.3f}  max={nca_out.max():.3f}"
        f"  mean={nca_out.mean():.4f}"
    )

    # ── Checkpoint ────────────────────────────────────────────────────────
    trainer.save_checkpoint()

    # ── IO snapshot (4 rows: input, target, nca_out, |nca_out - target|) ─
    snap_b = min(B, 4)
    diff = (nca_out[:snap_b] - out[:snap_b]).abs()
    snap_path = snapshots_dir / f"snapshot_{step:07d}.png"
    save_grid_image(
        snap_path,
        [
            inp[:snap_b].cpu(),
            out[:snap_b].cpu(),
            nca_out[:snap_b].detach().cpu(),
            diff.detach().cpu(),
        ],
        row_vmin=[None, None, None, 0],
        row_vmax=[None, None, None, 2],
    )
    shutil.copy(snap_path, run_dir / "snapshot_latest.png")

    # ── Loss curve ────────────────────────────────────────────────────────
    losses = [m["loss"] for m in trainer.metrics]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(losses)), losses, s=0.5, alpha=0.4, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("masked MSE loss")
    ax.set_title(f"8-bit adder NCA — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close(fig)

    # ── Valid bits curve ──────────────────────────────────────────────────
    bits_vals = [m["num_valid_bits"] for m in trainer.metrics if "num_valid_bits" in m]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(bits_vals)), bits_vals, s=0.5, alpha=0.4, color="darkorange")
    ax.set_ylim(0, len(trainer.bit_masks))
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("mean valid bits")
    ax.set_title(f"8-bit adder NCA — valid bits — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "bits_curve.png", dpi=120)
    plt.close(fig)

    # ── Rollout GIF — cap batch and subsample frames to keep memory sane ──
    gif_b = min(B, 4)
    gif_t = rollout[:gif_b, ::2]  # every other frame
    gif_t_len = gif_t.shape[1]
    frames = gif_t.permute(2, 0, 1, 3, 4).reshape(C * gif_b, gif_t_len, H, W).numpy()
    frames_rgb = torch.from_numpy(
        media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis")
    )  # (C*gif_b, gif_t_len, H, W, 3)
    grid = make_grid(frames_rgb, nrow=gif_b, padding=1)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=8)

    gif_path = rollouts_dir / f"rollout_{step:07d}.gif"
    media.write_video(str(gif_path), grid.numpy(), fps=10, codec="gif")
    shutil.copy(gif_path, run_dir / "rollout_latest.gif")

    print(f"  → saved: loss_curve.png, rollout_{step:07d}.gif, snapshot_{step:07d}.png")
