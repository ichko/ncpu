#!/usr/bin/env python3
"""
Window-sweep gate training: one size-agnostic rule computes AND / OR / XOR.

A board is tiled with 32×32 windows, each holding one gate (selected by a
read-only channel). A ``WindowSweep`` slides a 32×32 window across the board and
applies the local NCA rule ``inner_steps`` times per hop. Because the rule only
ever sees a 32×32 crop, it is structurally incapable of learning board size or
absolute position — the same checkpoint runs on a 1024×1024 canvas in the web
demo.

Usage:
    uv run python scripts/train/train_window.py --seed 0
    uv run python scripts/train/train_window.py --seed 0 --n-cols 4 --inner-steps 48
    uv run python scripts/train/train_window.py --resume last

Outputs under runs/window_gates_<cols>x<rows>_s<seed>_<timestamp>/
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
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.sweep import SweepPlan, WindowSweep
from ncpu.utils import git_info
from ncpu.window import (
    NCA_CHANNELS,
    READ_ONLY_DIMS,
    WINDOW_SIZE,
    WORK_CHANNEL,
    WindowGateDataset,
)

# ── Args ───────────────────────────────────────────────────────────────────────

parser = ArgumentParser(
    description="Train a size-agnostic gate rule via a rolling window."
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=20_000)
parser.add_argument("--device", default="cuda")
parser.add_argument("--resume", type=str, default=None, help="run name, or 'last'")
parser.add_argument("--eval-every", type=int, default=500, dest="eval_every")
parser.add_argument("--save-rollout-npz", action="store_true")
parser.add_argument(
    "--n-rows",
    type=int,
    default=1,
    dest="n_rows",
    help="window rows on the training board",
)
parser.add_argument(
    "--n-cols",
    type=int,
    default=4,
    dest="n_cols",
    help="window columns on the training board",
)
parser.add_argument("--wstride", type=int, default=WINDOW_SIZE)
parser.add_argument("--hstride", type=int, default=WINDOW_SIZE)
parser.add_argument(
    "--inner-steps",
    type=int,
    default=48,
    dest="inner_steps",
    help="NCA updates per window hop",
)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=8, dest="batch_size")
parser.add_argument(
    "--n-eval-rollouts",
    type=int,
    default=8,
    dest="n_eval_rollouts",
    help="stochastic rollouts averaged per periodic eval",
)
parser.add_argument(
    "--gate-prob",
    type=float,
    default=0.5,
    dest="gate_prob",
    help="probability a training tile holds a gate (rest are empty)",
)
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

DEVICE = args.device
PLOT_EVERY = 500
MEDIA_EVERY = 5000

BOARD_H = args.n_rows * WINDOW_SIZE
BOARD_W = args.n_cols * WINDOW_SIZE

# ── Sweep plan ─────────────────────────────────────────────────────────────────

plan = SweepPlan(
    H=BOARD_H,
    W=BOARD_W,
    wh=WINDOW_SIZE,
    ww=WINDOW_SIZE,
    hstride=args.hstride,
    wstride=args.wstride,
    n_passes=1,
)

nca_config = Namespace(
    channels=NCA_CHANNELS,
    hidden_channels=[128],
    fire_rate=0.5,
    alive_threshold=0,
    zero_initialization=False,
    kernel_size=7,
    num_perception_kernels=3,
    read_only_dims=READ_ONLY_DIMS,
    padding_type="zeros",
)

# ── Run directory ──────────────────────────────────────────────────────────────

if args.resume == "last":
    runs = sorted(
        d.name
        for d in Path("runs").iterdir()
        if "window_gates" in d.name and (d / "checkpoints" / "nca_last.pt").exists()
    )
    if not runs:
        raise RuntimeError("No resumable window_gates runs found")
    args.resume = runs[-1]

if args.resume:
    run_dir = Path("runs") / args.resume
    ckpt_dir = run_dir / "checkpoints"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
else:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"window_gates_{args.n_cols}x{args.n_rows}_s{args.seed}_{stamp}"
    run_dir = Path("runs") / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "rollouts").mkdir()
    (run_dir / "snapshots").mkdir()
    with open(run_dir / "config.json", "w") as f:
        json.dump(
            {
                "run": {
                    "name": run_dir.name,
                    "seed": args.seed,
                    "total_steps": args.steps,
                    "device": DEVICE,
                },
                "board": {
                    "H": BOARD_H,
                    "W": BOARD_W,
                    "n_rows": args.n_rows,
                    "n_cols": args.n_cols,
                    "window": WINDOW_SIZE,
                    "wstride": args.wstride,
                    "hstride": args.hstride,
                    "inner_steps": args.inner_steps,
                    "gate_prob": args.gate_prob,
                },
                "plan": {
                    "describe": plan.describe(),
                    "caveats": plan.caveats(),
                    "n_hops": plan.n_hops,
                },
                "nca": vars(nca_config),
                "optim": {"lr": args.lr, "batch_size": args.batch_size},
                "env": {
                    "python": sys.version,
                    "torch": torch.__version__,
                    "git": git_info(),
                },
            },
            f,
            indent=2,
        )

print(f"window gates  cols={args.n_cols} rows={args.n_rows}  seed={args.seed}")
print(f"  board {BOARD_H}x{BOARD_W}  {plan.describe()}")
print(f"Run dir: {run_dir}\n")

# ── Model / sweep / optimiser ─────────────────────────────────────────────────

dataset = WindowGateDataset(
    n_rows=args.n_rows, n_cols=args.n_cols, seed=args.seed, gate_prob=args.gate_prob
)
dataloader = dataset.get_dataloader(batch_size=args.batch_size)
data_iter = iter(dataloader)

nca = NeuralCA(**vars(nca_config)).to(DEVICE)
sweep = WindowSweep(nca)
optim = torch.optim.Adam(nca.parameters(), lr=args.lr)

start_step = 0
train_losses, train_accs = [], []

if args.resume:
    ckpt = torch.load(ckpt_dir / "nca_last.pt", map_location=DEVICE, weights_only=False)
    nca.load_state_dict(ckpt["nca"])
    optim.load_state_dict(ckpt["optim"])
    start_step = ckpt["step"]
    train_losses = ckpt.get("train_losses", [])
    train_accs = ckpt.get("train_accs", [])
    print(f"resumed from step {start_step}")


# ── Loss / metrics ─────────────────────────────────────────────────────────────


def loss_fn(board, target, out_mask):
    pred = board[:, WORK_CHANNEL]
    return ((pred - target) ** 2 * out_mask).sum() / (out_mask.sum() * board.shape[0])


def gate_accuracy(board, target, out_mask):
    pred = board[:, WORK_CHANNEL]
    correct = ((pred > 0) == (target > 0)).float()
    return (correct * out_mask).sum().item() / out_mask.sum().item()


def save_checkpoint(step):
    torch.save(
        {
            "nca": nca.state_dict(),
            "optim": optim.state_dict(),
            "step": step,
            "train_losses": train_losses,
            "train_accs": train_accs,
        },
        ckpt_dir / "nca_last.pt",
    )
    torch.save(nca.state_dict(), ckpt_dir / f"nca_{step:06d}.pt")


# Fixed eval board so periodic evals are comparable.
eval_batch = next(data_iter)
eval_state, eval_target, eval_mask = [t.to(DEVICE) for t in eval_batch]


def run_eval():
    accs = []
    with torch.no_grad():
        for _ in range(args.n_eval_rollouts):
            board = sweep(eval_state, plan, inner_steps=args.inner_steps)
            accs.append(gate_accuracy(board, eval_target, eval_mask))
    return sum(accs) / len(accs)


# ── Training loop ──────────────────────────────────────────────────────────────

pbar = tqdm(range(start_step, args.steps), ncols=100, desc=f"win s{args.seed}")
log_path = run_dir / "log.jsonl"

for step in pbar:
    state, target, out_mask = next(data_iter)
    state = state.to(DEVICE)
    target = target.to(DEVICE)
    out_mask = out_mask.to(DEVICE)

    optim.zero_grad()
    board = sweep(state, plan, inner_steps=args.inner_steps)
    loss = loss_fn(board, target, out_mask)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(nca.parameters(), 1.0)
    optim.step()

    with torch.no_grad():
        acc = gate_accuracy(board, target, out_mask)
    train_losses.append(loss.item())
    train_accs.append(acc)

    pbar.set_description(f"win s{args.seed}  loss={loss.item():.4f}  acc={acc:.3f}")

    with open(log_path, "a") as f:
        f.write(
            json.dumps(
                {
                    "step": step,
                    "loss": loss.item(),
                    "gate_acc": acc,
                    "grad_norm": grad_norm.item(),
                    "ts": datetime.now().isoformat(),
                }
            )
            + "\n"
        )

    if step % args.eval_every == 0:
        eval_acc = run_eval()
        with open(log_path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "step": step,
                        "phase": "eval",
                        "eval_acc": eval_acc,
                        "ts": datetime.now().isoformat(),
                    }
                )
                + "\n"
            )
        print(f"\n  EVAL step {step}: gate_acc={eval_acc:.3f}")

    if step % PLOT_EVERY != 0:
        continue

    save_checkpoint(step)

    # loss curve
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(
        range(len(train_losses)), train_losses, s=0.5, alpha=0.4, color="steelblue"
    )
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(f"window gates s{args.seed} — loss — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close(fig)

    # accuracy curve
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.scatter(range(len(train_accs)), train_accs, s=0.5, alpha=0.4, color="darkorange")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("step")
    ax.set_ylabel("gate accuracy")
    ax.set_title(f"window gates s{args.seed} — accuracy — step {step}")
    fig.tight_layout()
    fig.savefig(run_dir / "acc_curve.png", dpi=120)
    plt.close(fig)

    if args.save_rollout_npz:
        with torch.no_grad():
            np.savez_compressed(
                run_dir / "rollout_latest.npz",
                state=state[0].cpu().numpy(),
                board=board[0].cpu().numpy(),
                target=target[0].cpu().numpy(),
            )

    if step % MEDIA_EVERY == 0 and step > 0:
        shutil.copy(
            run_dir / "loss_curve.png", run_dir / "snapshots" / f"loss_{step:07d}.png"
        )

save_checkpoint(args.steps)
print(f"\nDone. Run: {run_dir}")
