#!/usr/bin/env python3
"""
Gate training under programmatic damages (D5): one gate x one damage set x one seed.

Same recipe as train_gate.py (E1), but instead of legacy Gaussian noise the
model is trained against a set of programmatic damages from ncpu.noise:
input-side damages (perlin, salt&pepper, occlusion, blur, ...) on the raw
input screen and state-side damages during the rollout. With --damage-mode
sample (default) one input + one state damage are drawn at random per step.

Sweep shape: gates x damages x seeds, e.g.
    uv run python scripts/train/train_gate_damage.py --gate XOR --seed 1 \
        --damage "perlin:60:0.8"
    uv run python scripts/train/train_gate_damage.py --gate AND --seed 2 \
        --damage "salt_pepper:0.1:1.0" --damage "occlusion:3:0.8:block_size=12"
    uv run python scripts/train/train_gate_damage.py --gate OR --seed 3   # default mix

Each damage spec is "kind:amplitude[:prob[:k=v,...]]"; input amplitudes are in
[0,255] pixel units, state amplitudes on the [-10,10] state scale.

Runs are named D5_<gate>_ni<n>_<damage-slug>_s<seed>_<ts>/ so that
grouping by (gate, damage set) is trivial — see
scripts/analyze/analyze_damage_sweep.py, which plots per-seed loss curves in
different colors per damage group.

Usage:
    uv run python scripts/train/train_gate_damage.py --gate XOR --seed 1 --steps 30000
    uv run python scripts/train/train_gate_damage.py --gate AND --damage "blur:2.0:0.8" \
        --eval-every 1000 --save-rollout-npz
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
from ncpu.noise import (
    DEFAULT_INPUT_DAMAGE_SPECS,
    DEFAULT_STATE_DAMAGE_SPECS,
    damages_slug,
    parse_damage_specs,
)
from ncpu.trainer import NCPUTrainer
from ncpu.utils import (
    setup_device,
    freeze_frame,
    git_info,
    make_grid,
    make_io_screen_cols1,
    save_grid_image,
)

# ── Gate registry (same as train_gate.py) ──────────────────────────────────────
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

parser = ArgumentParser(
    description="Train one logic gate under programmatic damages (D5)."
)
parser.add_argument("--gate", required=True, choices=GATES.keys())
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=50_000)
parser.add_argument(
    "--n-inputs",
    type=int,
    default=2,
    dest="n_inputs",
    help="inputs for general boolean gates; ignored for half_adder/majority3",
)
parser.add_argument(
    "--damage",
    action="append",
    default=None,
    help="input damage spec 'kind:amplitude[:prob[:k=v,...]]'; repeatable. "
    "Defaults to the full input-side mix; pass 'none' to disable "
    "input damages (e.g. for state-only ablations)",
)
parser.add_argument(
    "--state-damage",
    action="append",
    default=None,
    dest="state_damage",
    help="state damage spec; repeatable. Defaults to the full state-side mix; "
    "pass 'none' to disable state damages",
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
parser.add_argument("--kernel_size", type=int, default=5, choices=[3, 5, 7])
parser.add_argument("--alive_threshold", type=float, default=0.0)
parser.add_argument("--zero_init", action="store_true", default=False)
parser.add_argument(
    "--eval-every",
    type=int,
    default=1_000,
    dest="eval_every",
    help="evaluate the robustness matrix every N steps (default 1000)",
)
parser.add_argument("--device", default="cuda")
parser.add_argument(
    "--resume", type=str, default=None, help="Run name to resume, or 'last'"
)
parser.add_argument(
    "--save-rollout-npz",
    action="store_true",
    help="also save rollout arrays as compressed NPZ (a rollout GIF is always written)",
)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

if GATES[args.gate][1] is not None:
    args.n_inputs = GATES[args.gate][1]
sampler = GATES[args.gate][0](args.n_inputs)

input_damages = parse_damage_specs(args.damage or DEFAULT_INPUT_DAMAGE_SPECS)
state_damages = parse_damage_specs(args.state_damage or DEFAULT_STATE_DAMAGE_SPECS)
using_defaults = args.damage is None and args.state_damage is None
dmg_tag = "mix" if using_defaults else damages_slug([*input_damages, *state_damages])

# ── Config ─────────────────────────────────────────────────────────────────────

TOTAL_STEPS = args.steps
PLOT_EVERY = 500
DEVICE = setup_device(args.device)
SAVE_ROLLOUT_NPZ = args.save_rollout_npz
EVAL_EVERY = args.eval_every

cfg = make_gate_config(n_inputs=args.n_inputs, H=48, side_sp=10, among_sp=2)
ds_config = Namespace(
    W=cfg.W,
    H=cfg.H,
    r=cfg.r,
    spacing=cfg.spacing,
    sampler=sampler,
    balanced=False,
    screen_fn=lambda left_input=[], right_input=[], **kw: make_io_screen_cols1(
        cfg.H,
        cfg.W,
        cfg.r,
        cfg.spacing,
        left_input=left_input,
        right_input=right_input,
        n_input_cols=1,
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
    gaussian_noise=0.0,
    gaussian_noise_fire_rate=0.0,
    state_damages=state_damages,
)

optim_config = Namespace(
    lr=1e-4,
    batch_size=16,
    gaussian_noise=-1,  # legacy input noise disabled; input_damages replace it
    grad_clip=None,
    steps_min=32,
    steps_max=64,
)

# ── Run directory ──────────────────────────────────────────────────────────────

if args.resume == "last":
    d5_runs = sorted(
        d.name
        for d in Path("runs").iterdir()
        if d.name.startswith("D5_") and (d / "checkpoints" / "trainer.pkl").exists()
    )
    if not d5_runs:
        raise RuntimeError("No resumable D5 runs found")
    args.resume = d5_runs[-1]

if args.resume:
    run_dir = Path("runs") / args.resume
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    checkpoints_dir = run_dir / "checkpoints"
    print(f"Resuming run {run_dir.name}: damage config comes from the pickled trainer")
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"D5_{args.gate}_ni{args.n_inputs}_{dmg_tag}_s{args.seed}_{timestamp}"
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
                    "experiment": "D5",
                    "gate": args.gate,
                    "n_inputs": args.n_inputs,
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
                    **{
                        k: v
                        for k, v in vars(ds_config).items()
                        if k not in ("sampler", "screen_fn")
                    },
                    "sampler": (
                        sampler.__name__
                        if hasattr(sampler, "__name__")
                        else str(sampler)
                    ),
                    "screen_fn": "make_io_screen_cols1",
                },
                "nca": {
                    **{
                        k: v
                        for k, v in vars(nca_config).items()
                        if k != "state_damages"
                    },
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
    f"Gate: {args.gate} n_inputs={args.n_inputs} seed={args.seed} damage={dmg_tag} mode={args.damage_mode}"
)
print(f"  input damages: {[d.label() for d in input_damages]}")
print(f"  state damages: {[d.label() for d in state_damages]}")
print(f"Run dir: {run_dir}\n")

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
    trainer.loss_fn = output_masked_rollout_loss
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
        loss_fn=output_masked_rollout_loss,
        input_dims=(0, 1),
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

N_EVAL_ROLLOUTS = 5
EVAL_STEPS = 64
eval_metrics = []
log_path = run_dir / "log.jsonl"


def run_eval(
    trainer, eval_batch, damage_configs, n_rollouts=N_EVAL_ROLLOUTS, steps=EVAL_STEPS
):
    """Evaluate the persisted model (save -> load round-trip), not the in-memory one."""
    trainer.save_checkpoint()
    trainer.load_checkpoint(trainer.learning_step)
    ckpt = Path(trainer.checkpoint_pattern.format(step=trainer.learning_step))
    result = trainer.evaluate(
        eval_batch,
        [],
        n_rollouts=n_rollouts,
        steps=steps,
        damage_configs=damage_configs,
    )
    return {**result, "checkpoint": str(ckpt)}


# ── Training loop ──────────────────────────────────────────────────────────────

start_step = trainer.learning_step
pbar = tqdm(
    range(start_step, start_step + TOTAL_STEPS),
    ncols=100,
    desc=f"{args.gate} dmg s{args.seed}",
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
        f"{args.gate} dmg s{args.seed}  loss={loss:.4f}  bits={num_valid_bits:.2f}"
    )

    with open(log_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "step": step,
                    "loss": loss,
                    "num_valid_bits": num_valid_bits,
                    "grad_norm": grad_norm,
                    "gate": args.gate,
                    "n_inputs": args.n_inputs,
                    "seed": args.seed,
                    "ts": datetime.now().isoformat(),
                }
            )
            + "\n"
        )

    if step % EVAL_EVERY == 0 and step > 0:
        eval_result = run_eval(trainer, eval_batch, eval_damage_configs)
        eval_metrics.append(
            {
                "step": step,
                **{
                    cfg["label"]: {"bits": cfg["bits"], "loss": cfg["loss"]}
                    for cfg in eval_result["per_cfg"]
                },
            }
        )
        with open(log_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "step": step,
                        "eval_bits": eval_result["eval_bits"],
                        "eval_loss": eval_result["eval_loss"],
                        "gate": args.gate,
                        "n_inputs": args.n_inputs,
                        "seed": args.seed,
                        "per_cfg": eval_result["per_cfg"],
                    }
                )
                + "\n"
            )
        summary = "  ".join(
            f"{c['label']}={c['bits']:.2f}" for c in eval_result["per_cfg"]
        )
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
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(range(len(losses)), losses, s=0.5, alpha=0.4, color="steelblue")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(
        f"{args.gate} ni={args.n_inputs} dmg={dmg_tag} seed={args.seed} — step {step}"
    )
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
    ax.set_title(f"{args.gate} — valid bits — step {step}")
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
        ax.set_ylim(0, len(trainer.bit_masks) + 0.5)
        ax.set_xlabel("step")
        ax.set_ylabel("mean eval bit accuracy")
        ax.set_title(f"{args.gate} robustness matrix seed={args.seed} — step {step}")
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(run_dir / "eval_bits_curve.png", dpi=120)
        plt.close(fig)

        latest = eval_metrics[-1]
        fig, ax = plt.subplots(figsize=(11, 4))
        vals = [latest[label]["bits"] for label in labels if label in latest]
        ax.bar(range(len(vals)), vals, color="seagreen")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(
            [l for l in labels if l in latest], rotation=45, ha="right", fontsize=7
        )
        ax.set_ylim(0, len(trainer.bit_masks) + 0.5)
        ax.set_ylabel("eval bit accuracy")
        ax.set_title(f"{args.gate} robustness matrix seed={args.seed} — step {step}")
        fig.tight_layout()
        fig.savefig(run_dir / "eval_bits_bar.png", dpi=120)
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

    # rollout output: channel 0 only, first 4 batch items; always a GIF per
    # run, plus NPZ on top when requested
    gif_b = min(B, 4)
    ch0 = rollout[:gif_b, ::2, 0:1]
    frames = ch0.permute(2, 0, 1, 3, 4).reshape(gif_b, ch0.shape[1], H, W).numpy()
    frames_rgb = torch.from_numpy(media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis"))
    grid = make_grid(frames_rgb, nrow=gif_b, padding=1)
    grid = freeze_frame(grid, timesteps=[0, -1], repeat=8)
    media.write_video(
        str(run_dir / "rollout_latest.gif"), grid.numpy(), fps=10, codec="gif"
    )
    if SAVE_ROLLOUT_NPZ:
        np.savez_compressed(run_dir / "rollout_latest.npz", rollout=ch0.numpy())

    if step % 5000 == 0 and step > 0:
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
