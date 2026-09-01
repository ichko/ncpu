#!/usr/bin/env python3
"""
Unified gate training script (E1): train a SINGLE gate as one NCA rule.

This is the one-script-for-gates entry point. It replaces the old
train_gate.py (E1 benchmark) and train_gates_noise_robust.py (multi-gate +
noise) with a single script that trains one gate at a time and optionally
injects stochastic Gaussian noise at each rollout step (mimicking damage /
radiation). When both noise args are left at 0, per-step noise is sampled
randomly over a range during training and the model is periodically
evaluated across the whole noise range (robust mode).

Supported gates: AND, OR, XOR, NAND, NOR, XNOR, half_adder, majority3
  - AND/OR/XOR/NAND/NOR/XNOR take `--n-inputs` inputs (default 2).
  - half_adder is fixed at 2 inputs -> [sum, carry].
  - majority3 is fixed at 3 inputs -> 1 output.

Usage:
    uv run python scripts/train/train_gate.py --gate XOR --seed 0
    uv run python scripts/train/train_gate.py --gate AND --n-inputs 4
    uv run python scripts/train/train_gate.py --gate OR --gaussian-noise 0.4 \
        --gaussian-noise-fire-rate 0.5 --seed 1
    uv run python scripts/train/train_gate.py --gate NAND --kernel_size 7 --steps 30000

Outputs (under runs/E1_<gate>_s<seed>_<timestamp>/):
    config.json, log.jsonl, loss_curve.png, bits_curve.png, eval_*_curve.png,
    snapshot_latest.png, rollout_latest.gif, checkpoints/
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.config import make_gate_config
from ncpu.dataset import (
    NCPUDataset,
    sample_half_adder,
    sample_majority3,
    sample_n_input_gate,
)
from ncpu.loss import output_masked_rollout_loss
from ncpu.nca import NeuralCA
from ncpu.trainer import NCPUTrainer
from ncpu.utils import freeze_frame, git_info, make_grid, save_grid_image, make_io_screen_cols1

# ── Gate registry ──────────────────────────────────────────────────────────────
# n_inputs: None means "use --n-inputs" (general boolean gates); a fixed int
# means the gate has a fixed arity and overrides --n-inputs.
GATES = {
    "AND": (lambda n: sample_n_input_gate(n, "AND"), None),
    "OR": (lambda n: sample_n_input_gate(n, "OR"), None),
    "XOR": (lambda n: sample_n_input_gate(n, "XOR"), None),
    "NAND": (lambda n: sample_n_input_gate(n, "NAND"), None),
    "NOR": (lambda n: sample_n_input_gate(n, "NOR"), None),
    "XNOR": (lambda n: sample_n_input_gate(n, "XNOR"), None),
    "half_adder": (lambda n: sample_half_adder, 2),
    "majority3": (lambda n: sample_majority3, 3),
}

# ── Args ───────────────────────────────────────────────────────────────────────

parser = ArgumentParser(description="Train a single NCA rule for one logic gate (E1).")
parser.add_argument("--gate", required=True, choices=GATES.keys())
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=50_000)
parser.add_argument("--n-inputs", type=int, default=2, dest="n_inputs",
                    help="number of inputs for general boolean gates (default 2; e.g. 3 or 4). Ignored for half_adder/majority3.")
parser.add_argument("--gaussian-noise", type=float, default=0.0,
                    help="gaussian noise std injected each rollout step. If 0 and "
                         "--gaussian-noise-fire-rate is also 0, per-step noise is sampled "
                         "randomly over [0,1] (robust mode).")
parser.add_argument("--gaussian-noise-fire-rate", type=float, default=0.0,
                    help="probability of injecting noise each rollout step")
parser.add_argument("--kernel_size", type=int, default=5, choices=[3, 5, 7])
parser.add_argument("--alive_threshold", type=float, default=0.0)
parser.add_argument("--zero_init", action="store_true", default=False)
parser.add_argument("--eval-every", type=int, default=1_000, dest="eval_every",
                    help="evaluate the persisted model every N steps (default 1000)")
parser.add_argument("--device", default="cuda")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Fixed-arity gates override --n-inputs.
if GATES[args.gate][1] is not None:
    args.n_inputs = GATES[args.gate][1]

sampler = GATES[args.gate][0](args.n_inputs)

# ── Config ─────────────────────────────────────────────────────────────────────

TOTAL_STEPS = args.steps
PLOT_EVERY = 500
DEVICE = args.device
FIXED_NOISE = args.gaussian_noise > 0 or args.gaussian_noise_fire_rate > 0

# Grid: spread `n_inputs` input columns across the board automatically.
cfg = make_gate_config(n_inputs=args.n_inputs, H=48, side_sp=10, among_sp=2)
ds_config = Namespace(
    W=cfg.W,
    H=cfg.H,
    r=cfg.r,
    spacing=cfg.spacing,
    sampler=sampler,
    balanced=False,
    screen_fn=lambda left_input=[], right_input=[], **kw: make_io_screen_cols1(
        cfg.H, cfg.W, cfg.r, cfg.spacing,
        left_input=left_input, right_input=right_input,
        n_input_cols=args.n_inputs,
    ),
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
    gaussian_noise=args.gaussian_noise,
    gaussian_noise_fire_rate=args.gaussian_noise_fire_rate,
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
noise_tag = f"n{int(args.gaussian_noise*100)}fr{int(args.gaussian_noise_fire_rate*100)}" if FIXED_NOISE else "robust"
run_name = (
    f"{experiment}_{args.gate}{kernel_tag}{alive_tag}{zinit_tag}"
    f"_ni{args.n_inputs}_{noise_tag}_s{args.seed}_{timestamp}"
)
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
                "n_inputs": args.n_inputs,
                "seed": args.seed,
                "total_steps": TOTAL_STEPS,
                "plot_every": PLOT_EVERY,
                "device": DEVICE,
            },
            "ds": {
                **{k: v for k, v in vars(ds_config).items() if k not in ("sampler", "screen_fn")},
                "sampler": ds_config.sampler.__name__ if hasattr(ds_config.sampler, "__name__") else str(ds_config.sampler),
                "screen_fn": "make_io_screen_cols1",
            },
            "nca": {**vars(nca_config), "gaussian_noise": args.gaussian_noise,
                    "gaussian_noise_fire_rate": args.gaussian_noise_fire_rate},
            "optim": vars(optim_config),
            "eval": {"eval_every": args.eval_every, "fixed_noise": FIXED_NOISE},
            "env": {"python": sys.version, "torch": torch.__version__, "git": git_info()},
        },
        f,
        indent=2,
    )

print(f"Gate   : {args.gate}  n_inputs={args.n_inputs}  seed={args.seed}  noise={noise_tag}")
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

# Fixed eval batch reused every periodic evaluation so curves are comparable.
eval_batch = next(iter(dataset.get_dataloader(batch_size=optim_config.batch_size)))
if FIXED_NOISE:
    eval_noise_configs = [(args.gaussian_noise, args.gaussian_noise_fire_rate)]
else:
    EVAL_NOISE_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]
    eval_noise_configs = [
        (std, fire_rate) for std in EVAL_NOISE_LEVELS for fire_rate in EVAL_NOISE_LEVELS
    ]
N_EVAL_ROLLOUTS = 5
EVAL_STEPS = 64
eval_metrics = []


def run_eval(trainer, eval_batch, noise_configs, n_rollouts=N_EVAL_ROLLOUTS, steps=EVAL_STEPS):
    """Evaluate the persisted model (save -> load round-trip), not the in-memory one."""
    trainer.save_checkpoint()
    trainer.load_checkpoint(trainer.learning_step)
    ckpt = Path(trainer.checkpoint_pattern.format(step=trainer.learning_step))
    result = trainer.evaluate(eval_batch, noise_configs, n_rollouts=n_rollouts, steps=steps)
    return {**result, "checkpoint": str(ckpt)}


# ── Training loop ──────────────────────────────────────────────────────────────

pbar = tqdm(range(TOTAL_STEPS), ncols=100, desc=f"{args.gate} s{args.seed}")

eval_bits = eval_loss = None
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
        + (f"  eval={eval_bits:.3f}" if step > 0 and eval_bits is not None else "")
    )

    with open(log_path, "a") as f:
        f.write(json.dumps({
            "step": step, "loss": loss,
            "num_valid_bits": num_valid_bits,
            "grad_norm": grad_norm,
            "gate": args.gate, "n_inputs": args.n_inputs, "seed": args.seed,
            "ts": datetime.now().isoformat(),
        }) + "\n")

    eval_bits = eval_loss = None
    if step % args.eval_every == 0:
        eval_result = run_eval(trainer, eval_batch, eval_noise_configs)
        eval_bits, eval_loss = eval_result["eval_bits"], eval_result["eval_loss"]
        eval_metrics.append({"step": step, "eval_bits": eval_bits, "eval_loss": eval_loss})
        with open(log_path, "a") as f:
            f.write(json.dumps({
                "step": step, "eval_bits": eval_bits, "eval_loss": eval_loss,
                "gate": args.gate, "n_inputs": args.n_inputs, "seed": args.seed,
                "per_cfg": eval_result["per_cfg"],
            }) + "\n")
        print(f"\n  EVAL step {step}: bits={eval_bits:.3f}  loss={eval_loss:.4f}  ({len(eval_noise_configs)*N_EVAL_ROLLOUTS} rollouts)")

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
    ax.set_xlabel("step"); ax.set_ylabel("loss")
    ax.set_title(f"{args.gate} n={args.n_inputs} seed={args.seed} — step {step}")
    fig.tight_layout(); fig.savefig(run_dir / "loss_curve.png", dpi=120); plt.close(fig)

    # bits curve
    bits_vals = [m["num_valid_bits"] for m in trainer.metrics if "num_valid_bits" in m]
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(range(len(bits_vals)), bits_vals, s=0.5, alpha=0.4, color="darkorange")
    if eval_metrics:
        ax.plot([m["step"] for m in eval_metrics],
                [m["eval_bits"] for m in eval_metrics],
                color="seagreen", marker="o", lw=1, label="eval bits")
        ax.legend()
    ax.set_ylim(0, len(trainer.bit_masks) + 0.5)
    ax.set_xlabel("step"); ax.set_ylabel("valid bits")
    ax.set_title(f"{args.gate} n={args.n_inputs} seed={args.seed} — valid bits — step {step}")
    fig.tight_layout(); fig.savefig(run_dir / "bits_curve.png", dpi=120); plt.close(fig)

    # eval curves
    if eval_metrics:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot([m["step"] for m in eval_metrics],
                [m["eval_bits"] for m in eval_metrics],
                color="seagreen", marker="o", lw=1)
        ax.set_ylim(0, len(trainer.bit_masks) + 0.5)
        ax.set_xlabel("step"); ax.set_ylabel("mean eval bit accuracy")
        ax.set_title(f"{args.gate} — eval bits — step {step}")
        fig.tight_layout(); fig.savefig(run_dir / "eval_bits_curve.png", dpi=120); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot([m["step"] for m in eval_metrics],
                [m["eval_loss"] for m in eval_metrics],
                color="seagreen", marker="o", lw=1)
        ax.set_yscale("log")
        ax.set_xlabel("step"); ax.set_ylabel("eval loss")
        ax.set_title(f"{args.gate} — eval loss — step {step}")
        fig.tight_layout(); fig.savefig(run_dir / "eval_loss_curve.png", dpi=120); plt.close(fig)

    # snapshot
    diff = (nca_out - out).abs()
    snap_b = min(B, 4)
    save_grid_image(
        run_dir / "snapshot_latest.png",
        [inp[:snap_b].cpu(), out[:snap_b].cpu(),
         nca_out[:snap_b].detach().cpu(), diff[:snap_b].detach().cpu()],
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
    media.write_video(str(run_dir / "rollout_latest.gif"), grid.numpy(), fps=10, codec="gif")

print(f"\nDone. Run: {run_dir}")
