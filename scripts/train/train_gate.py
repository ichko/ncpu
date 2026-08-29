#!/usr/bin/env python3
"""
Gate benchmark training script (E1).

Usage:
    uv run python scripts/train/train_gate.py --gate XOR --seed 0
    uv run python scripts/train/train_gate.py --gate half_adder --seed 1

Supported gates: AND, OR, XOR, NAND, NOR, XNOR, half_adder, majority3

Outputs (under runs/E1_<gate>_s<seed>_<timestamp>/):
    config.json, log.jsonl, loss_curve.png, bits_curve.png,
    snapshot_latest.png, rollout_latest.gif, checkpoints/
"""

import json
import shutil
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapy as media
import torch
from tqdm import tqdm
from argparse import Namespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.dataset import (
    NCPUDataset,
    sample_AND_gate,
    sample_OR_gate,
    sample_XOR_gate,
    sample_NAND_gate,
    sample_NOR_gate,
    sample_XNOR_gate,
    sample_half_adder,
    sample_majority3,
)
from ncpu.loss import output_masked_rollout_loss
from ncpu.nca import NeuralCA
from ncpu.trainer import NCPUTrainer
from ncpu.utils import (
    freeze_frame,
    git_info,
    make_grid,
    save_grid_image,
    make_io_screen_cols1,
)

# ── Gate registry ──────────────────────────────────────────────────────────────

GATES = {
    "AND": sample_AND_gate,
    "OR": sample_OR_gate,
    "XOR": sample_XOR_gate,
    "NAND": sample_NAND_gate,
    "NOR": sample_NOR_gate,
    "XNOR": sample_XNOR_gate,
    "half_adder": sample_half_adder,
    "majority3": sample_majority3,
}

# ── Args ───────────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--gate", required=True, choices=GATES.keys())
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=50_000)
parser.add_argument("--kernel_size", type=int, default=5, choices=[3, 5, 7])
parser.add_argument("--alive_threshold", type=float, default=0.0)
parser.add_argument("--zero_init", action="store_true", default=False)
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

torch.manual_seed(args.seed)

# ── Config ─────────────────────────────────────────────────────────────────────

TOTAL_STEPS = args.steps
PLOT_EVERY = 500
DEVICE = args.device

ds_config = Namespace(
    W=48,
    H=48,
    r=4,
    spacing=(2, 10),
    sampler=GATES[args.gate],
    balanced=False,
    screen_fn=make_io_screen_cols1,
)

nca_config = Namespace(
    channels=16,
    hidden_channels=[128],
    fire_rate=0.5,
    alive_threshold=args.alive_threshold,
    zero_initialization=args.zero_init,
    kernel_size=args.kernel_size,
    num_perception_kernels=3,
    read_only_dims=[1],
    padding_type="zeros",
)

optim_config = Namespace(
    lr=1e-4,
    batch_size=16,
    gaussian_noise=-1,
    grad_clip=None,
    steps_min=32,
    steps_max=64,
)

# ── Run directory ──────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
is_ablation = args.kernel_size != 5 or args.alive_threshold != 0.0 or args.zero_init
experiment = (
    "E1"
    if not is_ablation
    else (
        "E_KS"
        if args.kernel_size != 5
        else ("E_AM" if args.alive_threshold != 0.0 else "E_ZI")
    )
)
kernel_tag = "" if args.kernel_size == 5 else f"_k{args.kernel_size}"
alive_tag = "" if args.alive_threshold == 0.0 else f"_am{args.alive_threshold:.2f}"
zinit_tag = "_zi" if args.zero_init else ""
run_name = f"{experiment}_{args.gate}{kernel_tag}{alive_tag}{zinit_tag}_s{args.seed}_{timestamp}"
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
                "experiment": experiment,
                "gate": args.gate,
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

print(f"Gate   : {args.gate}  seed={args.seed}")
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

pbar = tqdm(range(TOTAL_STEPS), ncols=100, desc=f"{args.gate} s{args.seed}")

for step in pbar:
    info = trainer.optim_step(
        steps=(optim_config.steps_min, optim_config.steps_max),
        return_rollout=(step % PLOT_EVERY == 0),
    )
    loss = info["loss"]
    num_valid_bits = info["num_valid_bits"]
    grad_norm = trainer.metrics[-1].get("grad_norm") if trainer.metrics else None

    pbar.set_description(
        f"{args.gate} s{args.seed}  loss={loss:.4f}  bits={num_valid_bits:.2f}"
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

    rollout = info["rollout"]  # (B, T, C, H, W)
    nca_out = info["nca_out"]  # (B, H, W)
    out = info["out"]
    inp = info["inp"]
    B, T, C, H, W = rollout.shape

    trainer.save_checkpoint()

    # loss curve
    losses = [m["loss"] for m in trainer.metrics]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(range(len(losses)), losses, s=0.5, alpha=0.4, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"{args.gate} seed={args.seed} — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close(fig)

    # bits curve
    bits_vals = [m["num_valid_bits"] for m in trainer.metrics if "num_valid_bits" in m]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(range(len(bits_vals)), bits_vals, s=0.5, alpha=0.4, color="darkorange")
    ax.set_ylim(0, len(trainer.bit_masks) + 0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("valid bits")
    ax.set_title(f"{args.gate} seed={args.seed} — valid bits — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "bits_curve.png", dpi=120)
    plt.close(fig)

    # snapshot
    diff = (nca_out - out).abs()
    snap_b = min(B, 4)
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

    # rollout gif (channel 0 only, first 4 batch items)
    gif_b = min(B, 4)
    ch0 = rollout[:gif_b, ::2, 0:1]  # (B, T//2, 1, H, W)
    frames = ch0.permute(2, 0, 1, 3, 4).reshape(gif_b, ch0.shape[1], H, W).numpy()
    frames_rgb = torch.from_numpy(media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis"))
    grid = make_grid(frames_rgb, nrow=gif_b, padding=1)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=8)
    media.write_video(
        str(run_dir / "rollout_latest.gif"), grid.numpy(), fps=10, codec="gif"
    )

print(f"\nDone. Run: {run_dir}")
