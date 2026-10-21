#!/usr/bin/env python3
"""8-bit multiplier NCA training.

Two 8-bit operands (16 input bits, two left columns) -> 16-bit product
(two right columns). Same grid, NCA, and optimizer configuration as the
8-bit adder (E3_adder8_cols2); only the output layout (two columns via
make_io_screen_cols2io) and the sampler (sample_8bit_multiplier) differ.

Trains until improvement stops: a moving average of correct output bits is
tracked, and training halts once it has not improved for PATIENCE steps.

    uv run python scripts/train_multiplier8.py --seed 0

Outputs under runs/MUL8_cols2_s<seed>_<timestamp>/
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
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.dataset import NCPUDataset, sample_8bit_multiplier
from ncpu.loss import output_masked_rollout_loss
from ncpu.nca import NeuralCA
from ncpu.trainer import NCPUTrainer
from ncpu.utils import freeze_frame, git_info, make_grid, save_grid_image, make_io_screen_cols2io

# ── Args ─────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=400_000, help="hard cap on steps")
parser.add_argument("--device", default="cuda")
# early stopping (improvement = a rise in the moving-average of correct bits)
parser.add_argument("--patience", type=int, default=40_000,
                    help="stop if the best moving-average valid-bits has not improved for this many steps")
parser.add_argument("--min-delta", type=float, default=0.05, help="min bits improvement that counts")
args = parser.parse_args()

torch.manual_seed(args.seed)

PLOT_EVERY, MEDIA_EVERY, DEVICE = 500, 5000, args.device
W, H = 80, 112                                  # same as the 8-bit adder (cols2)

ds_config = Namespace(W=W, H=H, r=4, spacing=(2, 21),
                      sampler=sample_8bit_multiplier, balanced=False,
                      screen_fn=make_io_screen_cols2io)
nca_config = Namespace(channels=16, hidden_channels=[128], fire_rate=0.5,
                       alive_threshold=0, zero_initialization=False, kernel_size=7,
                       num_perception_kernels=3, read_only_dims=[1], padding_type="zeros")
optim_config = Namespace(lr=1e-4, batch_size=8, gaussian_noise=-1, grad_clip=None,
                         steps_min=64, steps_max=128)

# ── Run directory ─────────────────────────────────────────────────────────────
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name = f"MUL8_cols2_s{args.seed}_{timestamp}"
run_dir = Path("runs") / run_name
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "rollouts").mkdir()
(run_dir / "snapshots").mkdir()
checkpoints_dir = run_dir / "checkpoints"
checkpoints_dir.mkdir()

with open(run_dir / "config.json", "w") as f:
    json.dump({
        "run": {"name": run_name, "task": "mul8", "layout": "cols2", "seed": args.seed,
                "max_steps": args.steps, "patience": args.patience, "min_delta": args.min_delta,
                "plot_every": PLOT_EVERY, "device": DEVICE},
        "ds": {**{k: v for k, v in vars(ds_config).items() if k not in ("sampler", "screen_fn")},
               "sampler": ds_config.sampler.__name__, "screen_fn": ds_config.screen_fn.__name__},
        "nca": vars(nca_config), "optim": vars(optim_config),
        "env": {"python": sys.version, "torch": torch.__version__, "git": git_info()},
    }, f, indent=2)

print(f"8-bit multiplier  seed={args.seed}\nRun dir: {run_dir}\n")

# ── Model & trainer ─────────────────────────────────────────────────────────
dataset = NCPUDataset(ds_config)
nca = NeuralCA(**vars(nca_config)).to(DEVICE)
trainer = NCPUTrainer(
    nca, dataset.get_dataloader(batch_size=optim_config.batch_size),
    lr=optim_config.lr, gaussian_noise=optim_config.gaussian_noise, grad_clip=optim_config.grad_clip,
    checkpoint_pattern=str(checkpoints_dir / "nca_{step:06d}.pt"),
    loss_fn=output_masked_rollout_loss, input_dims=(0, 1),
)
trainer.ds = dataset
trainer.sanity_check()
n_out_bits = len(trainer.bit_masks)
print(f"output bits: {n_out_bits}")

log_path = run_dir / "log.jsonl"

# ── Training loop (until improvement stops) ────────────────────────────────────
bits_ema = None
best_bits = -1.0
best_step = 0
pbar = tqdm(range(args.steps), ncols=100, desc=f"mul8 s{args.seed}")

for step in pbar:
    info = trainer.optim_step(
        steps=(optim_config.steps_min, optim_config.steps_max),
        return_rollout=(step % PLOT_EVERY == 0),
    )
    loss = info["loss"]
    vb = info["num_valid_bits"]
    bits_ema = vb if bits_ema is None else 0.99 * bits_ema + 0.01 * vb
    grad_norm = trainer.metrics[-1].get("grad_norm") if trainer.metrics else None

    if bits_ema > best_bits + args.min_delta:
        best_bits, best_step = bits_ema, step

    pbar.set_description(
        f"mul8 s{args.seed}  loss={loss:.4f}  bits={vb:.2f}/{n_out_bits}  "
        f"ema={bits_ema:.2f}  best={best_bits:.2f}@{best_step}"
    )

    with open(log_path, "a") as f:
        f.write(json.dumps({"step": step, "loss": loss, "num_valid_bits": vb,
                            "bits_ema": bits_ema, "grad_norm": grad_norm,
                            "ts": datetime.now().isoformat()}) + "\n")

    # early stop once the moving average has plateaued for `patience` steps
    if step - best_step > args.patience and step > args.patience:
        print(f"\nEarly stop at step {step}: no improvement since {best_step} "
              f"(best ema {best_bits:.3f}/{n_out_bits}).")
        trainer.save_checkpoint()
        break

    if step % PLOT_EVERY != 0:
        continue

    rollout, nca_out, out, inp = info["rollout"], info["nca_out"], info["out"], info["inp"]
    B, T, C, Hh, Ww = rollout.shape
    trainer.save_checkpoint()

    losses = [m["loss"] for m in trainer.metrics]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(losses)), losses, s=0.5, alpha=0.4, color="steelblue")
    ax.set_yscale("log"); ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title(f"8-bit multiplier s{args.seed} — step {step}")
    fig.tight_layout(); fig.savefig(run_dir / "loss_curve.png", dpi=120); plt.close(fig)

    bits_vals = [m["num_valid_bits"] for m in trainer.metrics if "num_valid_bits" in m]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(bits_vals)), bits_vals, s=0.5, alpha=0.4, color="darkorange")
    ax.set_ylim(0, n_out_bits + 0.5); ax.set_xlabel("step"); ax.set_ylabel("valid bits")
    ax.set_title(f"8-bit multiplier s{args.seed} — valid bits — step {step}")
    fig.tight_layout(); fig.savefig(run_dir / "bits_curve.png", dpi=120); plt.close(fig)

    snap_b = min(B, 4)
    diff = (nca_out - out).abs()
    save_grid_image(
        run_dir / "snapshot_latest.png",
        [inp[:snap_b].cpu(), out[:snap_b].cpu(),
         nca_out[:snap_b].detach().cpu(), diff[:snap_b].detach().cpu()],
        row_vmin=[None, None, None, 0], row_vmax=[None, None, None, 2],
    )

    # rollout gif — optional (needs ffmpeg); never let a missing encoder kill training
    gif_ok = False
    try:
        gif_b = min(B, 4)
        gif_t = rollout[:gif_b, ::2]
        frames = gif_t.permute(2, 0, 1, 3, 4).reshape(C * gif_b, gif_t.shape[1], Hh, Ww).numpy()
        frames_rgb = torch.from_numpy(media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis"))
        grid = make_grid(frames_rgb, nrow=gif_b, padding=1)
        grid = freeze_frame(grid, timesteps=[0, -1], repeat=8)
        media.write_video(str(run_dir / "rollout_latest.gif"), grid.numpy(), fps=10, codec="gif")
        gif_ok = True
    except Exception as e:
        if step == 0:
            print(f"(rollout gif disabled: {e})")

    if step % MEDIA_EVERY == 0 and step > 0:
        if gif_ok:
            shutil.copy(run_dir / "rollout_latest.gif", run_dir / "rollouts" / f"rollout_{step:07d}.gif")
        shutil.copy(run_dir / "snapshot_latest.png", run_dir / "snapshots" / f"snapshot_{step:07d}.png")

print(f"\nDone. Run: {run_dir}  best ema bits {best_bits:.3f}/{n_out_bits} @ step {best_step}")
