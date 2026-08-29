#!/usr/bin/env python3
"""
8-bit ALU NCA training script (E4).

Operations (3-bit opcode):
    0: ADD   A + B + carry_in
    1: SUB   A - B - carry_in  (carry_out = NOT borrow)
    2: AND   A & B
    3: OR    A | B
    4: XOR   A ^ B
    5: NOT   ~A
    6: SHL   A << 1, carry_in → LSB
    7: SHR   A >> 1, carry_in → MSB

Layout (128×112 grid, r=4):
    Left   : A[0..7] (sub-col 0) + B[0..7] (sub-col 1)
    Middle : opcode[0..2] + carry_in, single column at x=64
    Right  : result[0..7] + carry_out, single column at x=108

Usage:
    uv run python scripts/train/train_alu.py
    uv run python scripts/train/train_alu.py --seed 1
    uv run python scripts/train/train_alu.py --steps 200000
"""

import json
import shutil
import sys
from argparse import ArgumentParser, Namespace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.dataset import ALUDataset
from ncpu.loss import output_masked_rollout_loss
from ncpu.nca import NeuralCA
from ncpu.trainer import NCPUTrainer
from ncpu.utils import freeze_frame, git_info, make_grid, save_grid_image

# ── Args ───────────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--seed",   type=int, default=0)
parser.add_argument("--steps",  type=int, default=200_000)
parser.add_argument("--device", default="cuda")
parser.add_argument("--resume", type=str, default=None,
                    help="Run name to resume, or 'last'")
args = parser.parse_args()

torch.manual_seed(args.seed)

# ── Config ─────────────────────────────────────────────────────────────────────

TOTAL_STEPS = args.steps
PLOT_EVERY = 500
MEDIA_EVERY = 5000
DEVICE = args.device

# Grid: wide enough for 3 regions (left inputs / middle opcode / right output).
# A/B at x=20,30  |  opcode+carry at x=64  |  result+carry at x=108
# Horizontal gap A→opcode ≈ 26px, opcode→output ≈ 36px.
# With k=7 (3px/step reach) + fire_rate=0.5: steps_max=256 covers full carry chain.
ds_config = Namespace(
    W=128,
    H=112,
    r=4,
    spacing=(2, 20),
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

if args.resume == "last":
    e4_runs = sorted(d.name for d in Path("runs").iterdir()
                     if d.name.startswith("E4_") and (d / "checkpoints" / "trainer.pkl").exists())
    if not e4_runs:
        raise RuntimeError("No resumable E4 runs found")
    args.resume = e4_runs[-1]

if args.resume:
    run_dir = Path("runs") / args.resume
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    checkpoints_dir = run_dir / "checkpoints"
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"E4_alu8_s{args.seed}_{timestamp}"
    run_dir  = Path("runs") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "rollouts").mkdir()
    (run_dir / "snapshots").mkdir()
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir()
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "run":   {"name": run_name, "experiment": "E4", "seed": args.seed,
                      "total_steps": TOTAL_STEPS, "plot_every": PLOT_EVERY, "device": DEVICE},
            "ds":    vars(ds_config),
            "nca":   vars(nca_config),
            "optim": vars(optim_config),
            "env":   {"python": sys.version, "torch": torch.__version__, "git": git_info()},
        }, f, indent=2)

print(f"ALU 8-bit  seed={args.seed}  {'(resuming) ' if args.resume else ''}run: {run_dir.name}\n")

# ── Model & trainer ────────────────────────────────────────────────────────────

dataset = ALUDataset(ds_config)

if args.resume:
    trainer = NCPUTrainer.load_trainer(checkpoints_dir)
    trainer.nca.to(DEVICE)
    trainer.to(DEVICE)
    trainer.dataloader    = dataset.get_dataloader(batch_size=optim_config.batch_size)
    trainer.ds            = dataset
    trainer.dataset_iter  = iter(trainer.dataloader)
    trainer.optim         = torch.optim.Adam(trainer.nca.parameters(), lr=optim_config.lr)
    trainer.loss_fn       = output_masked_rollout_loss
    left_mask, right_mask = dataset.get_io_mask()
    trainer.inp_mask        = torch.tensor(left_mask).to(DEVICE)
    trainer.out_mask        = torch.tensor(right_mask).to(DEVICE)
    trainer.out_mask_binary = (trainer.out_mask > 128).float()
    trainer.inp_mask_binary = (trainer.inp_mask > 128).float()
    trainer.bit_masks       = dataset.get_output_bit_masks().to(DEVICE)
    print(f"Resumed from step {trainer.learning_step}")
else:
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

start_step = trainer.learning_step
pbar = tqdm(range(start_step, start_step + TOTAL_STEPS), ncols=100, desc=f"alu s{args.seed}")

for step in pbar:
    info = trainer.optim_step(
        steps=(optim_config.steps_min, optim_config.steps_max),
        return_rollout=(step % PLOT_EVERY == 0),
    )
    loss = info["loss"]
    num_valid_bits = info["num_valid_bits"]
    grad_norm = trainer.metrics[-1].get("grad_norm") if trainer.metrics else None

    pbar.set_description(
        f"alu s{args.seed}  loss={loss:.4f}  bits={num_valid_bits:.2f}/8"
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
    ax.set_title(f"8-bit ALU seed={args.seed} — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close(fig)

    # bits curve
    bits_vals = [m["num_valid_bits"] for m in trainer.metrics if "num_valid_bits" in m]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(bits_vals)), bits_vals, s=0.5, alpha=0.4, color="darkorange")
    ax.set_ylim(0, 8.5)
    ax.set_xlabel("step")
    ax.set_ylabel("valid bits / 8")
    ax.set_title(f"8-bit ALU seed={args.seed} — valid bits — step {step}")
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

    # rollout gif
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
