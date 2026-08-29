#!/usr/bin/env python3
"""
Extrapolation experiment: train adder NCA on 1–max_train_bits, test on larger.

Usage:
    uv run python scripts/train/train_extrapolation.py --seed 0
    uv run python scripts/train/train_extrapolation.py --seed 0 --max_train_bits 5 --steps 100000

Grid is fixed to accommodate 8-bit inputs (9 output bits), bottom-aligned so
LSB is always at the same absolute y position across all bit widths.

Outputs under runs/extrap_s<seed>_<timestamp>/
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
from argparse import Namespace
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.dataset import ExtrapolationDataset
from ncpu.loss import output_masked_rollout_loss
from ncpu.nca import NeuralCA
from ncpu.trainer import NCPUTrainer
from ncpu.utils import freeze_frame, git_info, make_grid, save_grid_image

# ── Args ───────────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--seed",           type=int, default=0)
parser.add_argument("--steps",          type=int, default=100_000)
parser.add_argument("--max_train_bits", type=int, default=5)
parser.add_argument("--device",         default="cuda")
args = parser.parse_args()

torch.manual_seed(args.seed)

# ── Config ─────────────────────────────────────────────────────────────────────

TOTAL_STEPS    = args.steps
MAX_BITS       = 8           # grid always sized for 8-bit inputs
PLOT_EVERY     = 500
MEDIA_EVERY    = 5000
VAL_EVERY      = 1000        # steps between validation runs
VAL_BATCHES    = 4           # batches per val k
DEVICE         = args.device
NOISE_MAX_STD  = 0.3
NOISE_MAX_PROB = 0.3
BATCH_SIZE     = 8

# spacing=(among, side): side doubles as bottom margin for bottom_y = H - side
ds_config = Namespace(
    W=80, H=112, r=4,
    spacing=(2, 21),
    min_bits=1,
    max_bits=args.max_train_bits,
    batch_size=BATCH_SIZE,
)

nca_config = Namespace(
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

optim_config = Namespace(
    lr=1e-4,
    batch_size=BATCH_SIZE,
    gaussian_noise=-1,
    grad_clip=None,
    steps_min=64,
    steps_max=128,
)

# ── Run directory ──────────────────────────────────────────────────────────────

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
run_name  = f"extrap_train{args.max_train_bits}_s{args.seed}_{timestamp}"
run_dir   = Path("runs") / run_name
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "rollouts").mkdir()
(run_dir / "snapshots").mkdir()
checkpoints_dir = run_dir / "checkpoints"
checkpoints_dir.mkdir()

with open(run_dir / "config.json", "w") as f:
    json.dump({
        "run":   {"name": run_name, "seed": args.seed, "total_steps": TOTAL_STEPS,
                  "max_train_bits": args.max_train_bits, "max_bits": MAX_BITS,
                  "plot_every": PLOT_EVERY, "device": DEVICE},
        "ds":    vars(ds_config),
        "nca":   vars(nca_config),
        "optim": vars(optim_config),
        "env":   {"python": sys.version, "torch": torch.__version__, "git": git_info()},
    }, f, indent=2)

print(f"Extrapolation  train_bits=1–{args.max_train_bits}  seed={args.seed}")
print(f"Run dir: {run_dir}\n")

# ── Model & trainer ────────────────────────────────────────────────────────────

dataset = ExtrapolationDataset(**vars(ds_config))
nca     = NeuralCA(**vars(nca_config)).to(DEVICE)
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

# ── Validation datasets (one per held-out bit width) ───────────────────────────

val_ks = list(range(args.max_train_bits + 1, MAX_BITS + 1))
val_datasets = {
    k: ExtrapolationDataset(
        W=ds_config.W, H=ds_config.H, r=ds_config.r,
        spacing=ds_config.spacing, min_bits=k, max_bits=k,
        batch_size=BATCH_SIZE,
    )
    for k in val_ks
}


def run_validation(trainer, val_datasets, n_batches, steps_range):
    """Run a few no-grad batches per held-out k; return {k: {val_loss, val_bits}}."""
    results = {}
    original_iter     = trainer.dataset_iter
    original_masks    = trainer.bit_masks
    original_out_mask = trainer.out_mask_binary
    original_inp_mask = trainer.inp_mask_binary
    try:
        for k, val_ds in val_datasets.items():
            trainer.dataset_iter    = iter(val_ds.get_dataloader(batch_size=BATCH_SIZE))
            trainer.bit_masks       = val_ds.get_output_bit_masks().to(trainer.device)
            inp_m, out_m            = val_ds.get_io_mask()
            trainer.out_mask_binary = (torch.tensor(out_m) > 128).float().to(trainer.device)
            trainer.inp_mask_binary = (torch.tensor(inp_m) > 128).float().to(trainer.device)
            losses, bits = [], []
            with torch.no_grad():
                for _ in range(n_batches):
                    info = trainer.optim_step(steps=steps_range[1], return_rollout=False)
                    losses.append(info["loss"])
                    bits.append(info["num_valid_bits"])
            results[k] = {
                "val_loss": sum(losses) / len(losses),
                "val_bits": sum(bits) / len(bits),
            }
    finally:
        trainer.dataset_iter    = original_iter
        trainer.bit_masks       = original_masks
        trainer.out_mask_binary = original_out_mask
        trainer.inp_mask_binary = original_inp_mask
    return results


log_path = run_dir / "log.jsonl"

# ── Training loop ──────────────────────────────────────────────────────────────

pbar = tqdm(range(TOTAL_STEPS), ncols=110, desc=f"extrap s{args.seed}")

for step in pbar:
    progress = step / max(TOTAL_STEPS - 1, 1)
    trainer.nca.gaussian_noise.std  = NOISE_MAX_STD  * progress
    trainer.nca.gaussian_noise.prob = NOISE_MAX_PROB * progress

    info = trainer.optim_step(
        steps=(optim_config.steps_min, optim_config.steps_max),
        return_rollout=(step % PLOT_EVERY == 0),
    )
    loss           = info["loss"]
    num_valid_bits = info["num_valid_bits"]
    current_k      = dataset.current_k
    grad_norm      = trainer.metrics[-1].get("grad_norm") if trainer.metrics else None

    pbar.set_description(
        f"extrap s{args.seed}  k={current_k}  loss={loss:.4f}  bits={num_valid_bits:.2f}/{current_k+1}"
    )

    with open(log_path, "a") as f:
        f.write(json.dumps({
            "step": step, "loss": loss,
            "num_valid_bits": num_valid_bits,
            "current_k": current_k,
            "grad_norm": grad_norm,
            "ts": datetime.now().isoformat(),
        }) + "\n")

    if step % VAL_EVERY == 0 and step > 0:
        val_results = run_validation(
            trainer, val_datasets, VAL_BATCHES,
            (optim_config.steps_min, optim_config.steps_max),
        )
        with open(log_path, "a") as f:
            for k, r in val_results.items():
                f.write(json.dumps({
                    "step": step, "phase": "val", "k": k,
                    "val_loss": r["val_loss"], "val_bits": r["val_bits"],
                    "val_n_bits": k + 1,
                    "ts": datetime.now().isoformat(),
                }) + "\n")

    if step % PLOT_EVERY != 0:
        continue

    rollout = info["rollout"]
    nca_out = info["nca_out"]
    out     = info["out"]
    inp     = info["inp"]
    B, T, C, H, W = rollout.shape

    trainer.save_checkpoint()

    # parse val records from log for plotting
    val_log = {k: {"steps": [], "losses": [], "bits": []} for k in val_ks}
    try:
        with open(log_path) as f:
            for line in f:
                d = json.loads(line)
                if d.get("phase") == "val" and d["k"] in val_log:
                    val_log[d["k"]]["steps"].append(d["step"])
                    val_log[d["k"]]["losses"].append(d["val_loss"])
                    val_log[d["k"]]["bits"].append(d["val_bits"])
    except Exception:
        pass

    val_colors = plt.cm.autumn([(i + 1) / (len(val_ks) + 1) for i in range(len(val_ks))])

    # loss curve — training scatter + val lines per k
    train_losses = [m["loss"] for m in trainer.metrics]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(train_losses)), train_losses, s=0.5, alpha=0.3, color="steelblue", label="train")
    for (k, vd), col in zip(val_log.items(), val_colors):
        if vd["steps"]:
            ax.plot(vd["steps"], vd["losses"], color=col, linewidth=1.2, label=f"val k={k}")
    ax.set_yscale("log")
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title(f"Extrapolation seed={args.seed} train_bits=1–{args.max_train_bits} — step {step}")
    ax.legend(fontsize=7, markerscale=4)
    fig.tight_layout(); fig.savefig(run_dir / "loss_curve.png", dpi=120); plt.close(fig)

    # bits curve — training scatter coloured by k + val lines per k
    bits_vals = [m["num_valid_bits"] for m in trainer.metrics if "num_valid_bits" in m]
    k_vals    = [m.get("current_k", 1) for m in trainer.metrics if "num_valid_bits" in m]
    fig, ax = plt.subplots(figsize=(10, 4))
    sc = ax.scatter(range(len(bits_vals)), bits_vals, c=k_vals, s=0.5,
                    cmap="plasma", vmin=1, vmax=MAX_BITS, alpha=0.4)
    fig.colorbar(sc, ax=ax, label="train k")
    for (k, vd), col in zip(val_log.items(), val_colors):
        if vd["steps"]:
            ax.plot(vd["steps"], vd["bits"], color=col, linewidth=1.2, label=f"val k={k}")
    ax.set_ylim(0, MAX_BITS + 1.5)
    ax.set_xlabel("step"); ax.set_ylabel("valid bits")
    ax.set_title(f"Extrapolation seed={args.seed} — valid bits by k — step {step}")
    ax.legend(fontsize=7, markerscale=4)
    fig.tight_layout(); fig.savefig(run_dir / "bits_curve.png", dpi=120); plt.close(fig)

    # snapshot
    snap_b = min(B, 4)
    diff = (nca_out - out).abs()
    save_grid_image(
        run_dir / "snapshot_latest.png",
        [inp[:snap_b].cpu(), out[:snap_b].cpu(),
         nca_out[:snap_b].detach().cpu(), diff[:snap_b].detach().cpu()],
        row_vmin=[None, None, None, 0],
        row_vmax=[None, None, None, 2],
    )

    # rollout gif
    gif_b = min(B, 4)
    gif_t = rollout[:gif_b, ::2]
    frames = gif_t.permute(2, 0, 1, 3, 4).reshape(C * gif_b, gif_t.shape[1], H, W).numpy()
    frames_rgb = torch.from_numpy(media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis"))
    grid = make_grid(frames_rgb, nrow=gif_b, padding=1)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=8)
    media.write_video(str(run_dir / "rollout_latest.gif"), grid.numpy(), fps=10, codec="gif")

    if step % MEDIA_EVERY == 0 and step > 0:
        shutil.copy(run_dir / "rollout_latest.gif",
                    run_dir / "rollouts" / f"rollout_{step:07d}.gif")
        shutil.copy(run_dir / "snapshot_latest.png",
                    run_dir / "snapshots" / f"snapshot_{step:07d}.png")

print(f"\nDone. Run: {run_dir}")
