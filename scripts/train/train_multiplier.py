#!/usr/bin/env python3
"""
4-bit multiplier NCA training script.

Usage:
    uv run python scripts/train/train_multiplier.py --seed 0
    uv run python scripts/train/train_multiplier.py --seed 1 --steps 50000

Outputs under runs/mul4_s<seed>_<timestamp>/

Layout: 8 input bits (A[0..3], B[0..3]) in two left columns,
        8 output bits (product[0..7]) in single right column.
Grid: 80×112, same as 4-bit adder cols2.
NCA:  kernel_size=7, no alive masking — identical to adder setup.
"""

import json
import shutil
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from argparse import Namespace

import matplotlib.pyplot as plt
import mediapy as media
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.dataset import NCPUDataset, sample_4bit_multiplier
from ncpu.loss import output_masked_rollout_loss
from ncpu.nca import NeuralCA
from ncpu.trainer import NCPUTrainer
from ncpu.utils import (
    freeze_frame,
    git_info,
    make_grid,
    make_io_screen,
    save_grid_image,
)

# ── Args ───────────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=50_000)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

torch.manual_seed(args.seed)

# ── Config ─────────────────────────────────────────────────────────────────────

TOTAL_STEPS = args.steps
PLOT_EVERY = 500
MEDIA_EVERY = 5000
DEVICE = args.device
NOISE_MAX_STD = 0.3  # ramps from 0 → this over training
NOISE_MAX_PROB = 0.3  # ramps from 0 → this over training

ds_config = Namespace(
    W=112,
    H=112,
    r=4,
    spacing=(2, 21),
    sampler=sample_4bit_multiplier,
    balanced=False,
    screen_fn=make_io_screen,
)

nca_config = Namespace(
    channels=16,
    hidden_channels=[128],
    fire_rate=0.5,
    alive_threshold=0,
    zero_initialization=True,
    kernel_size=7,
    num_perception_kernels=3,
    read_only_dims=[1],
    padding_type="zeros",
)

optim_config = Namespace(
    lr=1e-4,
    batch_size=8,
    gaussian_noise=-1,
    grad_clip=None,
    steps_min=64,
    steps_max=128,
)

# ── Run directory ──────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"mul4_s{args.seed}_{timestamp}"
run_dir = Path("runs") / run_name
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "rollouts").mkdir()
(run_dir / "snapshots").mkdir()
checkpoints_dir = run_dir / "checkpoints"
checkpoints_dir.mkdir()

with open(run_dir / "config.json", "w") as f:
    json.dump(
        {
            "run": {
                "name": run_name,
                "experiment": "mul4",
                "seed": args.seed,
                "total_steps": TOTAL_STEPS,
                "plot_every": PLOT_EVERY,
                "device": DEVICE,
            },
            "ds": {
                **{
                    k: v
                    for k, v in vars(ds_config).items()
                    if k not in ("sampler", "screen_fn")
                },
                "sampler": ds_config.sampler.__name__,
                "screen_fn": ds_config.screen_fn.__name__,
            },
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

print(f"4-bit multiplier  seed={args.seed}")
print(f"Run dir: {run_dir}\n")

# ── Model & trainer ────────────────────────────────────────────────────────────

dataset = NCPUDataset(ds_config)
nca = NeuralCA(**vars(nca_config)).to(DEVICE)
trainer = NCPUTrainer(
    nca,
    dataset.get_dataloader(batch_size=optim_config.batch_size),
    lr=optim_config.lr,
    gaussian_noise=optim_config.gaussian_noise,
    grad_clip=optim_config.grad_clip,
    checkpoint_pattern=str(checkpoints_dir / "nca_{step:06d}.pt"),
    loss_fn=output_masked_rollout_loss,
    input_dims=(0, 1),
)
trainer.ds = dataset
trainer.sanity_check()

log_path = run_dir / "log.jsonl"

# ── Training loop ──────────────────────────────────────────────────────────────

pbar = tqdm(range(TOTAL_STEPS), ncols=100, desc=f"mul4 s{args.seed}")

for step in pbar:
    progress = step / max(TOTAL_STEPS - 1, 1)
    trainer.nca.gaussian_noise.std = NOISE_MAX_STD * progress
    trainer.nca.gaussian_noise.prob = NOISE_MAX_PROB * progress

    info = trainer.optim_step(
        steps=(optim_config.steps_min, optim_config.steps_max),
        return_rollout=(step % PLOT_EVERY == 0),
    )
    loss = info["loss"]
    num_valid_bits = info["num_valid_bits"]
    grad_norm = trainer.metrics[-1].get("grad_norm") if trainer.metrics else None

    pbar.set_description(
        f"mul4 s{args.seed}  loss={loss:.4f}  bits={num_valid_bits:.2f}/8"
    )

    with open(log_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "step": step,
                    "loss": loss,
                    "num_valid_bits": num_valid_bits,
                    "grad_norm": grad_norm,
                    "ts": datetime.now().isoformat(),
                }
            )
            + "\n"
        )

    if step % PLOT_EVERY != 0:
        continue

    rollout = info["rollout"]
    nca_out = info["nca_out"]
    out = info["out"]
    inp = info["inp"]
    B, T, C, H, W = rollout.shape

    trainer.save_checkpoint()

    # loss curve
    losses = [m["loss"] for m in trainer.metrics]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(losses)), losses, s=0.5, alpha=0.4, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"4-bit multiplier seed={args.seed} — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close(fig)

    # bits curve
    bits_vals = [m["num_valid_bits"] for m in trainer.metrics if "num_valid_bits" in m]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(bits_vals)), bits_vals, s=0.5, alpha=0.4, color="darkorange")
    ax.set_ylim(0, len(trainer.bit_masks) + 0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("valid bits")
    ax.set_title(f"4-bit multiplier seed={args.seed} — valid bits — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "bits_curve.png", dpi=120)
    plt.close(fig)

    # snapshot
    snap_b = min(B, 4)
    diff = (nca_out - out).abs()
    save_grid_image(
        run_dir / "snapshot_latest.png",
        [
            inp[:snap_b].cpu(),
            out[:snap_b].cpu(),
            nca_out[:snap_b].detach().cpu(),
            diff[:snap_b].detach().cpu(),
        ],
        row_vmin=[None, None, None, 0],
        row_vmax=[None, None, None, 2],
    )

    # rollout gif — channel 0, first 4 batch items
    gif_b = min(B, 4)
    gif_t = rollout[:gif_b, ::2]
    frames = (
        gif_t.permute(2, 0, 1, 3, 4).reshape(C * gif_b, gif_t.shape[1], H, W).numpy()
    )
    frames_rgb = torch.from_numpy(media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis"))
    grid = make_grid(frames_rgb, nrow=gif_b, padding=1)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=8)
    media.write_video(
        str(run_dir / "rollout_latest.gif"), grid.numpy(), fps=10, codec="gif"
    )

    if step % MEDIA_EVERY == 0 and step > 0:
        shutil.copy(
            run_dir / "rollout_latest.gif",
            run_dir / "rollouts" / f"rollout_{step:07d}.gif",
        )
        shutil.copy(
            run_dir / "snapshot_latest.png",
            run_dir / "snapshots" / f"snapshot_{step:07d}.png",
        )

print(f"\nDone. Run: {run_dir}")
