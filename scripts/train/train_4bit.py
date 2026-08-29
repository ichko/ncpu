#!/usr/bin/env python3
"""
4-bit adder NCA training script.

Usage:
    uv run python scripts/train/train_4bit.py

Outputs (under runs/<timestamp>/):
    config.json          — run configuration
    log.jsonl            — one JSON line per step: {"step": N, "loss": X}
    loss_curve.png       — overwritten every PLOT_EVERY steps
    rollout_NNNNNNN.gif  — channel-0 rollout grid saved every PLOT_EVERY steps
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.dataset import NCPUDataset, sample_4bit_adder
from ncpu.nca import NeuralCA
from ncpu.trainer import NCPUTrainer
from ncpu.utils import freeze_frame, git_info, make_grid, save_grid_image


# ── Config ───────────────────────────────────────────────────────────────────
# spacing=(among_spacing, side_spacing):
#   input col 0 at x = side_spacing
#   input col 1 at x = side_spacing + (2*r + among_spacing)
#   output      at x = W - side_spacing
#
# Sanity check: input col1 and output at the same x — no propagation needed.
# side_spacing = (W - 2*r - among_spacing) / 2 = (64 - 8 - 2) / 2 = 27
#   col0=27, col1=37, output=37  →  output overlaps input col1

DEVICE = "cuda"
TOTAL_STEPS = 100_000
PLOT_EVERY = 250

ds_config = Namespace(
    W=64,
    H=64,
    r=4,
    spacing=(2, 17),
    sampler=sample_4bit_adder,
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
    batch_size=12,
    gaussian_noise=-1,
    grad_clip=None,  # disabled for sanity check
    steps_min=30,
    steps_max=64,
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

# ── Model & trainer ──────────────────────────────────────────────────────────
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
        f"loss={loss:.4f}  bits={num_valid_bits:.2f}" + (f"  gnorm={grad_norm:.3f}" if grad_norm else "")
    )

    with open(log_path, "a") as f:
        f.write(json.dumps({"step": step, "loss": loss, "num_valid_bits": num_valid_bits, "grad_norm": grad_norm}) + "\n")

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
    print(f"  loss     : {loss:.8f}")
    print(f"  rollout  : shape={tuple(rollout.shape)}")
    print(
        f"             min={rollout.min():.3f}  max={rollout.max():.3f}"
        f"  mean={rollout.mean():.4f}  std={rollout.std():.4f}"
    )
    print(
        f"  nca_out  : min={nca_out.min():.3f}  max={nca_out.max():.3f}"
        f"  mean={nca_out.mean():.4f}  std={nca_out.std():.4f}"
    )
    print(
        f"  target   : min={out.min():.3f}  max={out.max():.3f}"
        f"  mean={out.mean():.4f}  std={out.std():.4f}"
    )

    # ── Checkpoint ────────────────────────────────────────────────────────
    trainer.save_checkpoint()

    # ── IO snapshot (4 rows: input, target, nca_out, |nca_out - target|) ─
    diff = (nca_out - out).abs()
    snap_path = snapshots_dir / f"snapshot_{step:07d}.png"
    save_grid_image(
        snap_path,
        [inp.cpu(), out.cpu(), nca_out.detach().cpu(), diff.detach().cpu()],
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
    ax.set_title(f"4-bit adder NCA — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close(fig)

    # ── Valid bits curve ──────────────────────────────────────────────────
    bits_vals = [m["num_valid_bits"] for m in trainer.metrics if "num_valid_bits" in m]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(bits_vals)), bits_vals, s=0.5, alpha=0.4, color="darkorange")
    ax.set_xlabel("optimisation step")
    ax.set_ylabel("mean valid bits")
    ax.set_title(f"4-bit adder NCA — valid bits — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "bits_curve.png", dpi=120)
    plt.close(fig)

    # ── Rollout GIF — rows=channels, columns=batch items ─────────────────────
    # reshape (B, T, C, H, W) → (C*B, T, H, W) so make_grid produces C rows × B cols
    frames = rollout.permute(2, 0, 1, 3, 4).reshape(C * B, T, H, W).numpy()
    frames_rgb = torch.from_numpy(
        media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis")
    )  # (C*B, T, H, W, 3)
    grid = make_grid(frames_rgb, nrow=B, padding=1)  # (T, H_grid, W_grid, 3)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=15)  # 1.5s pause at 10fps

    gif_path = rollouts_dir / f"rollout_{step:07d}.gif"
    media.write_video(str(gif_path), grid.numpy(), fps=10, codec="gif")
    shutil.copy(gif_path, run_dir / "rollout_latest.gif")

    print(f"  → saved: loss_curve.png, rollout_{step:07d}.gif, snapshot_{step:07d}.png")
