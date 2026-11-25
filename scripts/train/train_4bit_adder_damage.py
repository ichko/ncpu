#!/usr/bin/env python3
"""
4-bit adder NCA trained against a mixture of programmatic damages (D4).

Extends the white-Gaussian noise sweeps (train_4bit_adder_noise.py) with the
full damage library in ncpu.noise: perlin, salt&pepper, occlusion, blur,
brightness/contrast, quantization, jitter, speckle on the input screen, plus
state-side damages (state_gaussian, state_perlin, channel_dropout,
block_dropout) during the rollout.

Each damage spec is "kind:amplitude[:prob[:k=v,...]]", e.g.:
    --damage "perlin:40:0.5:octaves=4" --damage "occlusion:2:0.5:block_size=8"
With --damage-mode sample (default) one input damage and one state damage
are picked at random per step; "all" stacks them.

Evaluation builds a robustness matrix: a fixed eval batch is rolled out under
clean conditions, each training damage alone, and all damages combined.

Usage:
    uv run python scripts/train/train_4bit_adder_damage.py
    uv run python scripts/train/train_4bit_adder_damage.py --seed 1 --steps 30000
    uv run python scripts/train/train_4bit_adder_damage.py --damage "perlin:40:0.5" \
        --state-damage "state_gaussian:0.5:0.5" --eval-every 1000 --save-rollout-npz
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

from ncpu.dataset import NCPUDataset, sample_4bit_adder
from ncpu.loss import combined_loss
from ncpu.nca import NeuralCA
from ncpu.noise import (
    DEFAULT_INPUT_DAMAGE_SPECS,
    DEFAULT_STATE_DAMAGE_SPECS,
    MIX_INPUT_DAMAGE_SPECS,
    MIX_STATE_DAMAGE_SPECS,
    Damage,
    damages_slug,
    parse_damage_specs,
)
from ncpu.trainer import NCPUTrainer
from ncpu.utils import (
    freeze_frame,
    git_info,
    make_grid,
    save_gif,
    save_grid_image,
    setup_device,
)

# ── Args ───────────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=30_000)
parser.add_argument("--device", default="cuda")
parser.add_argument(
    "--resume", type=str, default=None, help="Run name to resume, or 'last'"
)
parser.add_argument(
    "--eval-every",
    type=int,
    default=1000,
    dest="eval_every",
    help="evaluate the robustness matrix every N steps (default 1000)",
)
parser.add_argument(
    "--save-rollout-npz",
    action="store_true",
    help="also save rollout arrays as compressed NPZ (a rollout GIF is always written)",
)
parser.add_argument(
    "--damage",
    action="append",
    default=None,
    help="input damage spec 'kind:amplitude[:prob[:k=v,...]]'; repeatable. "
    "Default: mild 'gaussian:0.5:0.5'. Pass 'none' to disable input damages, "
    "'mix' for the full input damage mix",
)
parser.add_argument(
    "--state-damage",
    action="append",
    default=None,
    dest="state_damage",
    help="state damage spec (state_gaussian/state_perlin/channel_dropout/"
    "block_dropout); repeatable. Default: mild 'state_gaussian:0.5:0.5'. "
    "Pass 'none' to disable state damages, 'mix' for the full state damage mix",
)
parser.add_argument(
    "--damage-mode",
    choices=["sample", "all"],
    default="sample",
    help="sample: one input+state damage per step; all: stack them all",
)
parser.add_argument(
    "--eval-damage",
    action="append",
    default=None,
    dest="eval_damage",
    help="extra input damage spec to add to the eval matrix; repeatable",
)
args = parser.parse_args()

torch.manual_seed(args.seed)

# ── Config ─────────────────────────────────────────────────────────────────────

TOTAL_STEPS = args.steps
PLOT_EVERY = 500
MEDIA_EVERY = 5000
EVAL_EVERY = args.eval_every
DEVICE = setup_device(args.device)
SAVE_ROLLOUT_NPZ = args.save_rollout_npz

# amplitude is in pixel units on the raw [0,255] screen for input damages,
# and on the [-10,10] state scale for state damages. No flags -> mild
# gaussian:0.5:0.5 per side; the literal spec "mix" selects the full mix.
input_damages = parse_damage_specs(
    args.damage or DEFAULT_INPUT_DAMAGE_SPECS, mix_specs=MIX_INPUT_DAMAGE_SPECS
)
state_damages = parse_damage_specs(
    args.state_damage or DEFAULT_STATE_DAMAGE_SPECS, mix_specs=MIX_STATE_DAMAGE_SPECS
)

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
    fire_rate=0.99,
    alive_threshold=0.1,
    zero_initialization=False,
    kernel_size=7,
    read_only_dims=[1],
    padding_type="constant",
    gaussian_noise=0.0,  # legacy per-step gaussian; use state_gaussian damage
    gaussian_noise_fire_rate=0.0,
    state_damages=state_damages,
)

optim_config = Namespace(
    lr=1e-3,
    batch_size=8,
    gaussian_noise=-1,  # legacy input noise disabled; input_damages replace it
    grad_clip=1.0,
    steps_min=30,
    steps_max=80,
)

# ── Run directory ──────────────────────────────────────────────────────────────

if args.resume == "last":
    d4_runs = sorted(
        d.name
        for d in Path("runs").iterdir()
        if d.name.startswith("D4_") and (d / "checkpoints" / "trainer.pkl").exists()
    )
    if not d4_runs:
        raise RuntimeError("No resumable D4 runs found")
    args.resume = d4_runs[-1]

if args.resume:
    run_dir = Path("runs") / args.resume
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    checkpoints_dir = run_dir / "checkpoints"
    print(f"Resuming run {run_dir.name}: damage config comes from the pickled trainer")
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dmg_tag = damages_slug([*input_damages, *state_damages])
    run_name = f"D4_adder4_{dmg_tag}_s{args.seed}_{timestamp}"
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
                    "experiment": "D4",
                    "seed": args.seed,
                    "total_steps": TOTAL_STEPS,
                    "plot_every": PLOT_EVERY,
                    "eval_every": EVAL_EVERY,
                    "save_rollout_npz": SAVE_ROLLOUT_NPZ,
                    "device": DEVICE,
                    "damage_mode": args.damage_mode,
                    "damage_tag": dmg_tag,
                    "cuda_available": bool(torch.cuda.is_available()),
                    "cuda_device_name": (
                        torch.cuda.get_device_name(0)
                        if torch.cuda.is_available()
                        else None
                    ),
                },
                "ds": {
                    k: (v.__name__ if callable(v) else v)
                    for k, v in vars(ds_config).items()
                },
                "nca": {
                    **vars(nca_config),
                    "state_damages": [d.to_dict() for d in state_damages],
                },
                "optim": vars(optim_config),
                "input_damages": [d.to_dict() for d in input_damages],
                "env": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "git": git_info(),
                },
            },
            f,
            indent=2,
            default=str,
        )

print(
    f"4-bit adder damage training  seed={args.seed}  "
    f"{'(resuming) ' if args.resume else ''}run: {run_dir.name}"
)
print(f"  input damages: {[d.label() for d in input_damages]}")
print(
    f"  state damages: {[d.label() for d in state_damages]}  mode={args.damage_mode}\n"
)

# ── Model & trainer ────────────────────────────────────────────────────────────

dataset = NCPUDataset(ds_config)

if args.resume:
    trainer = NCPUTrainer.load_trainer(checkpoints_dir)
    trainer.nca.to(DEVICE)
    trainer.to(DEVICE)
    trainer.dataloader = dataset.get_dataloader(batch_size=optim_config.batch_size)
    trainer.ds = dataset
    trainer.dataset_iter = iter(trainer.dataloader)
    trainer.optim = torch.optim.Adam(trainer.nca.parameters(), lr=optim_config.lr)
    trainer.loss_fn = combined_loss
    left_mask, right_mask = dataset.get_io_mask()
    trainer.inp_mask = torch.tensor(left_mask).to(DEVICE)
    trainer.out_mask = torch.tensor(right_mask).to(DEVICE)
    trainer.out_mask_binary = (trainer.out_mask > 128).float()
    trainer.inp_mask_binary = (trainer.inp_mask > 128).float()
    trainer.bit_masks = dataset.get_output_bit_masks().to(DEVICE)
    input_damages = trainer.input_damages
    state_damages = trainer.nca.state_damages
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
        loss_fn=combined_loss,
        input_implant_type="all",
        input_damages=input_damages,
        damage_mode=args.damage_mode,
    )
    trainer.ds = dataset
    trainer.sanity_check()

_dev = next(trainer.nca.parameters()).device
print(f"model device: {_dev}")
if DEVICE.startswith("cuda") and _dev.type != "cuda":
    raise RuntimeError(f"model is on {_dev} but --device {DEVICE!r} was requested")

# ── Eval robustness matrix ─────────────────────────────────────────────────────

eval_batch = next(iter(dataset.get_dataloader(batch_size=optim_config.batch_size)))

eval_damage_configs = [{"label": "clean"}]
for d in input_damages:
    eval_damage_configs.append({"label": f"in_{d.label()}", "input_damages": [d]})
for d in state_damages:
    eval_damage_configs.append({"label": f"st_{d.label()}", "state_damages": [d]})
for d in parse_damage_specs(args.eval_damage):
    eval_damage_configs.append({"label": f"eval_{d.label()}", "input_damages": [d]})
eval_damage_configs.append({"label": "all_in", "input_damages": list(input_damages)})
eval_damage_configs.append(
    {
        "label": "all",
        "input_damages": list(input_damages),
        "state_damages": list(state_damages),
    }
)

eval_metrics = []
log_path = run_dir / "log.jsonl"

# ── Training loop ──────────────────────────────────────────────────────────────

start_step = trainer.learning_step
pbar = tqdm(
    range(start_step, start_step + TOTAL_STEPS), ncols=100, desc=f"dmg s{args.seed}"
)

for step in pbar:
    info = trainer.optim_step(
        steps=(optim_config.steps_min, optim_config.steps_max),
        return_rollout=(step % PLOT_EVERY == 0),
    )
    if step == start_step:
        print(
            f"first batch device: inp={info['inp'].device} nca_out={info['nca_out'].device}"
        )
    loss = info["loss"]
    num_valid_bits = info["num_valid_bits"]
    grad_norm = trainer.metrics[-1].get("grad_norm") if trainer.metrics else None

    pbar.set_description(
        f"dmg s{args.seed}  loss={loss:.4f}  bits={num_valid_bits:.2f}/5"
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

    if step % EVAL_EVERY == 0 and step > 0:
        trainer.save_checkpoint()
        trainer.load_checkpoint(trainer.learning_step)
        result = trainer.evaluate(
            eval_batch, [], n_rollouts=3, steps=80, damage_configs=eval_damage_configs
        )
        eval_metrics.append(
            {
                "step": step,
                **{
                    cfg["label"]: {"bits": cfg["bits"], "loss": cfg["loss"]}
                    for cfg in result["per_cfg"]
                },
            }
        )
        with open(log_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "step": step,
                        "eval_bits": result["eval_bits"],
                        "eval_loss": result["eval_loss"],
                        "per_cfg": result["per_cfg"],
                        "ts": datetime.now().isoformat(),
                    }
                )
                + "\n"
            )
        summary = "  ".join(f"{c['label']}={c['bits']:.2f}" for c in result["per_cfg"])
        print(f"\n  EVAL step {step}: {summary}")

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
    ax.set_title(f"4-bit adder + damages seed={args.seed} — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close(fig)

    # bits curve
    bits_vals = [m["num_valid_bits"] for m in trainer.metrics if "num_valid_bits" in m]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.scatter(range(len(bits_vals)), bits_vals, s=0.5, alpha=0.4, color="darkorange")
    ax.set_ylim(0, 5.5)
    ax.set_xlabel("step")
    ax.set_ylabel("valid bits / 5")
    ax.set_title(f"4-bit adder + damages seed={args.seed} — valid bits — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "bits_curve.png", dpi=120)
    plt.close(fig)

    # eval curves: per damage label over training steps
    if eval_metrics:
        labels = [c["label"] for c in eval_damage_configs]
        fig, ax = plt.subplots(figsize=(11, 5))
        for label in labels:
            xs = [m["step"] for m in eval_metrics if label in m]
            ys = [m[label]["bits"] for m in eval_metrics if label in m]
            if xs:
                ax.plot(xs, ys, marker="o", lw=1, ms=3, label=label)
        ax.set_ylim(0, 5.5)
        ax.set_xlabel("step")
        ax.set_ylabel("mean eval bit accuracy")
        ax.set_title(
            f"Robustness matrix seed={args.seed} — eval bits per damage — step {step}"
        )
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(run_dir / "eval_bits_curve.png", dpi=120)
        plt.close(fig)

        # bar chart of the latest eval per damage label
        latest = eval_metrics[-1]
        fig, ax = plt.subplots(figsize=(11, 4))
        vals = [latest[label]["bits"] for label in labels if label in latest]
        ax.bar(range(len(vals)), vals, color="seagreen")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(
            [l for l in labels if l in latest], rotation=45, ha="right", fontsize=7
        )
        ax.set_ylim(0, 5.5)
        ax.set_ylabel("eval bit accuracy")
        ax.set_title(f"Robustness matrix seed={args.seed} — step {step}")
        fig.tight_layout()
        fig.savefig(run_dir / "eval_bits_bar.png", dpi=120)
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

    # rollout output: always a GIF per run; NPZ on top when requested
    gif_b = min(B, 4)
    gif_t = rollout[:gif_b, ::2]
    frames = (
        gif_t.permute(2, 0, 1, 3, 4).reshape(C * gif_b, gif_t.shape[1], H, W).numpy()
    )
    frames_rgb = torch.from_numpy(media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis"))
    grid = make_grid(frames_rgb, nrow=gif_b, padding=1)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=8)
    try:
        # Pillow-based GIF writer: no ffmpeg binary needed (works on cluster nodes)
        save_gif(run_dir / "rollout_latest.gif", grid.numpy(), fps=10)
    except Exception as e:
        print(f"WARNING: rollout gif failed: {e}")
    if SAVE_ROLLOUT_NPZ:
        np.savez_compressed(run_dir / "rollout_latest.npz", rollout=gif_t.numpy())

    if step % MEDIA_EVERY == 0 and step > 0:
        shutil.copy(
            run_dir / "rollout_latest.gif",
            run_dir / "rollouts" / f"rollout_{step:07d}.gif",
        )
        if SAVE_ROLLOUT_NPZ:
            shutil.copy(
                run_dir / "rollout_latest.npz",
                run_dir / "rollouts" / f"rollout_{step:07d}.npz",
            )
        shutil.copy(
            run_dir / "snapshot_latest.png",
            run_dir / "snapshots" / f"snapshot_{step:07d}.png",
        )

print(f"\nDone. Run: {run_dir}")
