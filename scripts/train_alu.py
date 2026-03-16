#!/usr/bin/env python3
"""
8-bit ALU NCA training script.

Operations (3-bit opcode):
    0: ADD    A + B + carry_in
    1: SUB    A - B - carry_in
    2: AND    A & B
    3: OR     A | B
    4: XOR    A ^ B
    5: NOT    ~A
    6: SHL    A << 1, carry_in -> LSB
    7: SHR    A >> 1, carry_in -> MSB

Outputs (under runs/<timestamp>/):
    config.json, log.jsonl, loss_curve.png, bits_curve.png,
    rollout_latest.gif, snapshot_latest.png, checkpoints/

Usage:
    uv run python scripts/train_alu.py
    uv run python scripts/train_alu.py --resume last
    uv run python scripts/train_alu.py --resume 20260313_120000
"""

import json
import shutil
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.dataset import ALUDataset
from ncpu.loss import output_masked_rollout_loss
from ncpu.nca import NeuralCA

from ncpu.trainer import NCPUTrainer
from ncpu.utils import freeze_frame, git_info, make_grid, save_grid_image


# ── Media helpers ─────────────────────────────────────────────────────────────


def make_rollout_gif(rollout):
    B, T, C, H, W = rollout.shape
    gif_b = min(B, 4)
    gif_t = rollout[:gif_b, ::2]
    frames = (
        gif_t.permute(2, 0, 1, 3, 4).reshape(C * gif_b, gif_t.shape[1], H, W).numpy()
    )
    frames_rgb = torch.from_numpy(media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis"))
    grid = make_grid(frames_rgb, nrow=gif_b, padding=1)
    return freeze_frame(grid, timesteps=[0, -1], repeat=8)


def save_snapshot_latest(run_dir, inp, out, nca_out):
    snap_b = min(inp.shape[0], 4)
    diff = (nca_out[:snap_b] - out[:snap_b]).abs()
    save_grid_image(
        run_dir / "snapshot_latest.png",
        [
            inp[:snap_b].cpu(),
            out[:snap_b].cpu(),
            nca_out[:snap_b].detach().cpu(),
            diff.detach().cpu(),
        ],
        row_vmin=[None, None, None, 0],
        row_vmax=[None, None, None, 2],
    )


def save_rollout_latest(run_dir, rollout):
    grid = make_rollout_gif(rollout)
    media.write_video(
        str(run_dir / "rollout_latest.gif"), grid.numpy(), fps=10, codec="gif"
    )


def save_media(step, run_dir, snapshots_dir, rollouts_dir, inp, out, nca_out, rollout):
    snap_b = min(inp.shape[0], 4)
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
    grid = make_rollout_gif(rollout)
    gif_path = rollouts_dir / f"rollout_{step:07d}.gif"
    media.write_video(str(gif_path), grid.numpy(), fps=10, codec="gif")
    shutil.copy(gif_path, run_dir / "rollout_latest.gif")
    print(f"  → saved: snapshot_{step:07d}.png, rollout_{step:07d}.gif")


# ── Config ────────────────────────────────────────────────────────────────────

DEVICE = "cuda"
TOTAL_STEPS = 300_000
PLOT_EVERY = 500
MEDIA_EVERY = 5000

_parser = ArgumentParser()
_parser.add_argument(
    "--resume", type=str, default=None, help="Run name to resume, or 'last'"
)
_args = _parser.parse_args()

if _args.resume == "last":
    _runs = sorted(
        [
            d.name
            for d in Path("runs").iterdir()
            if (d / "checkpoints" / "trainer.pkl").exists()
        ]
    )
    if not _runs:
        raise RuntimeError("No resumable runs found in runs/")
    RESUME_RUN = _runs[-1]
else:
    RESUME_RUN = _args.resume

ds_config = Namespace(
    W=96,
    H=96,
    r=4,
)

nca_config = Namespace(
    channels=16,
    hidden_channels=[128],
    fire_rate=0.5,
    alive_threshold=0,
    zero_initialization=True,
    kernel_size=5,
    num_perception_kernels=3,
    read_only_dims=[1],
    padding_type="zeros",
)

optim_config = Namespace(
    lr=0.0001,
    batch_size=8,
    gaussian_noise=-1,
    grad_clip=None,
    steps_min=64,
    steps_max=96,
)

# ── Run directory ─────────────────────────────────────────────────────────────
if RESUME_RUN is not None:
    run_name = RESUME_RUN
    run_dir = Path("runs") / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
else:
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "run": {
                    "name": run_name,
                    "total_steps": TOTAL_STEPS,
                    "plot_every": PLOT_EVERY,
                    "device": DEVICE,
                    "task": "alu_8bit",
                },
                "ds": vars(ds_config),
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

rollouts_dir = run_dir / "rollouts"
rollouts_dir.mkdir(exist_ok=True)
snapshots_dir = run_dir / "snapshots"
snapshots_dir.mkdir(exist_ok=True)

print(f"Device : {DEVICE}")
print(f"Run dir: {run_dir}\n")

# ── Model & trainer ───────────────────────────────────────────────────────────
dataset = ALUDataset(ds_config)
dataloader = dataset.get_dataloader(batch_size=optim_config.batch_size)

if RESUME_RUN is not None:
    trainer = NCPUTrainer.load_trainer(run_dir / "checkpoints")
    trainer.nca.to(DEVICE)
    trainer.to(DEVICE)
    trainer.dataloader = dataloader
    trainer.ds = dataset
    trainer.dataset_iter = iter(dataloader)
    trainer.optim = torch.optim.Adam(trainer.nca.parameters(), lr=optim_config.lr)
    trainer.loss_fn = output_masked_rollout_loss
    # recompute masks — ensures correct device and layout after any changes
    left_mask, right_mask = dataset.get_io_mask()
    trainer.inp_mask = torch.tensor(left_mask).to(DEVICE)
    trainer.out_mask = torch.tensor(right_mask).to(DEVICE)
    trainer.out_mask_binary = (trainer.out_mask > 128).float()
    trainer.inp_mask_binary = (trainer.inp_mask > 128).float()
    trainer.bit_masks = dataset.get_output_bit_masks().to(DEVICE)
    print(f"Resumed from step {trainer.learning_step}")
else:
    nca = NeuralCA(**vars(nca_config)).to(DEVICE)
    trainer = NCPUTrainer(
        nca,
        dataloader,
        lr=optim_config.lr,
        gaussian_noise=optim_config.gaussian_noise,
        grad_clip=optim_config.grad_clip,
        checkpoint_pattern=str(run_dir / "checkpoints" / "nca_{step:06d}.pt"),
        loss_fn=output_masked_rollout_loss,
    )
    trainer.sanity_check()


log_path = run_dir / "log.jsonl"
print(f"\nLogging to: {log_path}")
print(f"{'─'*60}\n")

# ── Training loop ─────────────────────────────────────────────────────────────
start_step = trainer.learning_step
pbar = tqdm(range(start_step, start_step + TOTAL_STEPS), ncols=100)

for step in pbar:
    info = trainer.optim_step(
        steps=(optim_config.steps_min, optim_config.steps_max),
        return_rollout=(step % PLOT_EVERY == 0),
    )
    loss = info["loss"]
    num_valid_bits = info["num_valid_bits"]

    grad_norm = trainer.metrics[-1].get("grad_norm") if trainer.metrics else None
    pbar.set_description(
        f"loss={loss:.4f}  bits={num_valid_bits:.2f}/9"
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
                    "ts": datetime.now().isoformat(),
                }
            )
            + "\n"
        )

    if step % PLOT_EVERY != 0:
        continue

    # ── Periodic reporting ────────────────────────────────────────────────
    rollout = info["rollout"]
    nca_out = info["nca_out"]
    out = info["out"]
    inp = info["inp"]

    print(f"\n{'─'*60}")
    print(f"  step : {step}   loss: {loss:.8f}   bits: {num_valid_bits:.2f} / 12")
    print(
        f"  nca_out: min={nca_out.min():.3f}  max={nca_out.max():.3f}  mean={nca_out.mean():.4f}"
    )

    # ── Checkpoint ────────────────────────────────────────────────────────
    trainer.save_checkpoint()

    # ── Loss curve ────────────────────────────────────────────────────────
    losses = [m["loss"] for m in trainer.metrics]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(losses)), losses, s=0.5, alpha=0.4, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("masked MSE loss")
    ax.set_title(f"8-bit ALU NCA — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close(fig)

    # ── Valid bits curve ──────────────────────────────────────────────────
    bits_vals = [m["num_valid_bits"] for m in trainer.metrics if "num_valid_bits" in m]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(bits_vals)), bits_vals, s=0.5, alpha=0.4, color="darkorange")
    ax.set_ylim(0, 12)
    ax.set_xlabel("step")
    ax.set_ylabel("mean valid bits")
    ax.set_title(f"8-bit ALU NCA — valid bits / 12 — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "bits_curve.png", dpi=120)
    plt.close(fig)

    save_snapshot_latest(run_dir, inp, out, nca_out)
    save_rollout_latest(run_dir, rollout)

    if step % MEDIA_EVERY == 0:
        save_media(
            step, run_dir, snapshots_dir, rollouts_dir, inp, out, nca_out, rollout
        )

    print(f"  → saved: loss_curve.png, bits_curve.png")
