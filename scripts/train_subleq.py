#!/usr/bin/env python3
"""Train the subleq ALU NCA.

The NCA is the ALU of a one-instruction (subleq) machine: given two operand
words A, B drawn as circles it must, at the LAST rollout step, reproduce the
input screen with the output column correctly filled:

    result = (B - A) mod 2**word_bits
    branch = 1 if signed(result) <= 0 else 0

Setup (per the design discussion):
    * 3x3 fixed Sobel perception, 2-layer ReLU rule (one hidden layer)
    * 100x100 grid, neutral-grey background, 0-bit=-1 / 1-bit=+1 circles
    * output initialised to 0 (== background); faint 0.1 scaffold grid is part
      of the TARGET only, forcing the signal to propagate everywhere
    * supervision on the LAST frame only: fullscreen MSE + extra weight on the
      output circles

Usage:
    uv run python scripts/train_subleq.py
    uv run python scripts/train_subleq.py --steps 40000 --seed 1
    uv run python scripts/train_subleq.py --resume last
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
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.subleq import SubleqLayout, DartboardLayout, AddresserLayout, MemTileLayout, Gather2Layout, GatherDiscLayout, GatherColLayout, GatherDartLayout
from ncpu.utils import git_info

# ── Args ─────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=30_000)
parser.add_argument("--device", default="cuda")
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--word-bits", type=int, default=8)
parser.add_argument("--layout", choices=["columns", "dartboard", "addresser", "memtile", "gather2", "gatherdisc", "gathercol", "gatherdart"], default="columns")
parser.add_argument("--cells", type=int, default=6, help="addresser: number of memory cells")
parser.add_argument("--width", type=int, default=60)
parser.add_argument("--height", type=int, default=100)
parser.add_argument("--rollout-min", type=int, default=140)
parser.add_argument("--rollout-max", type=int, default=170)
parser.add_argument("--channels", type=int, default=16)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--kernel", type=int, default=3)
parser.add_argument("--loss", choices=["fullscreen", "masked"], default="fullscreen",
                    help="fullscreen = whole-screen MSE (+out_weight); masked = output-cells only (adder recipe)")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr-final", type=float, default=None, help="exp-decay lr to this by the last step")
parser.add_argument("--hard-frac", type=float, default=0.0, help="fraction of samples drawn near R=0 (branch boundary)")
parser.add_argument("--init-from", type=str, default=None, help="checkpoint .pt to warm-start nca weights from")
parser.add_argument("--out-weight", type=float, default=5.0,
                    help="extra weight on the output-circle pixels in the loss")
parser.add_argument("--supervise", choices=["last", "all"], default="all",
                    help="supervise the target on every rollout step, or only the last")
parser.add_argument("--plot-every", type=int, default=500)
parser.add_argument("--media-every", type=int, default=5000,
                    help="archive a checkpoint + snapshot + rollout gif every N steps")
parser.add_argument("--resume", type=str, default=None, help="run name, or 'last'")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = args.device

# ── Geometry, model config ───────────────────────────────────────────────────
if args.layout == "dartboard":
    layout = DartboardLayout(W=args.width, H=args.height, word_bits=args.word_bits)
elif args.layout == "addresser":
    layout = AddresserLayout(n_cells=args.cells, word_bits=args.word_bits)
elif args.layout == "memtile":
    layout = MemTileLayout(n_cells=args.cells, word_bits=args.word_bits)
elif args.layout == "gather2":
    layout = Gather2Layout(n_cells=args.cells, word_bits=args.word_bits)
elif args.layout == "gatherdisc":
    layout = GatherDiscLayout(n_cells=args.cells, word_bits=args.word_bits)
elif args.layout == "gathercol":
    layout = GatherColLayout(n_cells=args.cells, word_bits=args.word_bits)
elif args.layout == "gatherdart":
    layout = GatherDartLayout(n_cells=args.cells, word_bits=args.word_bits)
else:
    layout = SubleqLayout(W=args.width, H=args.height, word_bits=args.word_bits)

nca_config = dict(
    channels=args.channels,
    hidden_channels=[args.hidden],   # single hidden layer -> 2-layer ReLU rule
    fire_rate=0.5,
    alive_threshold=0,               # alive masking OFF (every cell updates every step)
    zero_initialization=True,
    kernel_size=args.kernel,
    perception="static_sobel",
    padding_type="zeros",
    read_only_dims=[1],              # channel 1 holds a frozen copy of the input
)
INPUT_DIMS = (0, 1)                   # implant input into ch0 (output) and ch1 (frozen)

# ── Run directory ────────────────────────────────────────────────────────────
if args.resume == "last":
    runs = sorted(d.name for d in Path("runs").iterdir()
                  if "SUBLEQ" in d.name and (d / "checkpoints" / "nca_last.pt").exists())
    if not runs:
        raise RuntimeError("No resumable SUBLEQ runs found")
    args.resume = runs[-1]

if args.resume:
    run_dir = Path("runs") / args.resume
    ckpt_dir = run_dir / "checkpoints"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
else:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{stamp}_SUBLEQ_{args.layout}_w{args.word_bits}_s{args.seed}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "rollouts").mkdir()
    (run_dir / "snapshots").mkdir()
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "run": {"name": run_dir.name, "seed": args.seed, "total_steps": args.steps,
                    "device": DEVICE},
            "layout": {"kind": args.layout, "W": layout.W, "H": layout.H,
                       "word_bits": layout.word_bits, "out_bits": layout.out_bits,
                       "r": getattr(layout, "r", None), "gap": getattr(layout, "gap", None),
                       "pad": getattr(layout, "pad", None)},
            "nca": nca_config,
            "optim": {"lr": args.lr, "batch_size": args.batch_size,
                      "rollout": [args.rollout_min, args.rollout_max],
                      "out_weight": args.out_weight, "supervise": args.supervise,
                      "loss": args.loss, "lr_final": args.lr_final,
                      "hard_frac": args.hard_frac, "init_from": args.init_from},
            "env": {"python": sys.version, "torch": torch.__version__, "git": git_info()},
        }, f, indent=2)

print(f"subleq  seed={args.seed}  {'(resume) ' if args.resume else ''}run: {run_dir.name}")
print(f"layout: {args.layout} {layout.W}x{layout.H} out_bits={layout.out_bits}  rollout={args.rollout_min}..{args.rollout_max}")

# ── Model / optimiser ────────────────────────────────────────────────────────
nca = NeuralCA(**nca_config).to(DEVICE)
optim = torch.optim.Adam(nca.parameters(), lr=args.lr)
start_step = 0
metrics = []

if args.init_from:                       # warm-start weights (fresh optimiser)
    init_ck = torch.load(args.init_from, map_location=DEVICE, weights_only=False)
    nca.load_state_dict(init_ck["nca"])
    print(f"warm-started nca from {args.init_from}")

if args.resume:
    ckpt = torch.load(ckpt_dir / "nca_last.pt", map_location=DEVICE)
    nca.load_state_dict(ckpt["nca"])
    optim.load_state_dict(ckpt["optim"])
    start_step = ckpt["step"]
    metrics = ckpt.get("metrics", [])
    print(f"resumed from step {start_step}")

bit_masks = layout.output_bit_masks().to(DEVICE)          # (out_bits, H, W)
bm_sum = bit_masks.sum(dim=(-1, -2))                      # (out_bits,)
out_mask = bit_masks.sum(dim=0).clamp(0, 1)              # (H, W) union


def implant(inp):
    bs = inp.shape[0]
    state = torch.zeros(bs, args.channels, layout.H, layout.W, device=DEVICE)
    for d in INPUT_DIMS:
        state[:, d] = inp
    return state


def run_batch(bs, steps):
    inp, tgt = layout.sample_batch(bs, device=DEVICE, hard_frac=args.hard_frac)
    rollout = nca.forward(implant(inp), steps=steps)
    last = rollout[:, -1, 0]
    seq = rollout[:, 1:, 0]                          # every step (skip the seed frame)
    T = seq.shape[1]
    diff = (seq - tgt.unsqueeze(1)) ** 2             # (B, T, H, W)
    if args.loss == "masked":                        # output-cells only, over all steps (adder recipe)
        loss = (diff * out_mask).sum() / (out_mask.sum() * bs * T)
    elif args.supervise == "all":                    # fullscreen every step + output weight
        loss = diff.mean() + args.out_weight * (diff * out_mask).sum() / (out_mask.sum() * bs * T)
    else:                                            # fullscreen last frame only
        d = (last - tgt) ** 2
        loss = d.mean() + args.out_weight * (d * out_mask).sum() / (out_mask.sum() * bs)
    return inp, tgt, rollout, last, loss


def bit_accuracy(last, tgt):
    nca_bit = (last.unsqueeze(1) * bit_masks).sum(dim=(-1, -2)) / bm_sum  # (B, out_bits)
    tgt_bit = (tgt.unsqueeze(1) * bit_masks).sum(dim=(-1, -2)) / bm_sum
    correct = ((nca_bit > 0) == (tgt_bit > 0)).float()                   # (B, out_bits)
    br = correct[:, layout.word_bits].mean().item() if layout.out_bits > layout.word_bits else 0.0
    return (correct.mean().item(),
            correct[:, :layout.word_bits].mean().item(),
            br)


def save_snapshot(path, inp, tgt, last, n=4):
    n = min(n, inp.shape[0])
    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8))
    rows = [("input", inp), ("target", tgt), ("nca last", last),
            ("|diff|", (last - tgt).abs())]
    for r, (name, data) in enumerate(rows):
        for c in range(n):
            ax = axes[r, c]
            vmin, vmax = (0, 2) if name == "|diff|" else (-1, 1)
            ax.imshow(data[c].detach().cpu(), cmap="viridis", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_rollout_gif(path, rollout, n=4, stride=3, fps=12):
    """Channel-0 evolution for `n` samples, tiled side by side."""
    n = min(n, rollout.shape[0])
    seq = rollout[:n, ::stride, 0].detach().cpu().numpy()          # (n, T', H, W)
    frames = np.concatenate([seq[i] for i in range(n)], axis=2)    # (T', H, n*W)
    rgb = media.to_rgb(frames, vmin=-1, vmax=1, cmap="viridis")
    media.write_video(str(path), rgb, fps=fps, codec="gif")


# ── Training loop ────────────────────────────────────────────────────────────
log_path = run_dir / "log.jsonl"
pbar = tqdm(range(start_step, start_step + args.steps), ncols=110, desc=f"subleq s{args.seed}")

lr_final = args.lr_final if args.lr_final is not None else args.lr
for step in pbar:
    if lr_final != args.lr and args.steps > 1:          # exponential lr decay over the run
        frac = (step - start_step) / (args.steps - 1)
        lr_now = args.lr * (lr_final / args.lr) ** frac
        for pg in optim.param_groups:
            pg["lr"] = lr_now
    steps = int(np.random.randint(args.rollout_min, args.rollout_max + 1))
    inp, tgt, rollout, last, loss = run_batch(args.batch_size, steps)

    optim.zero_grad()
    loss.backward()
    optim.step()

    with torch.no_grad():
        acc9, acc_res, acc_br = bit_accuracy(last, tgt)
    metrics.append({"step": step, "loss": loss.item(), "acc": acc9,
                    "acc_result": acc_res, "acc_branch": acc_br})
    pbar.set_description(
        f"subleq s{args.seed} loss={loss.item():.4f} bits={acc9*layout.out_bits:.1f}/{layout.out_bits} "
        f"(res={acc_res*layout.word_bits:.1f}/{layout.word_bits} br={acc_br:.2f})"
    )
    with open(log_path, "a") as f:
        f.write(json.dumps(metrics[-1]) + "\n")

    if step % args.plot_every != 0:
        continue

    torch.save({"nca": nca.state_dict(), "optim": optim.state_dict(), "step": step,
                "metrics": metrics, "nca_config": nca_config,
                "layout": {"W": layout.W, "H": layout.H, "word_bits": layout.word_bits}},
               ckpt_dir / "nca_last.pt")

    xs = [m["step"] for m in metrics]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    a1.scatter(xs, [m["loss"] for m in metrics], s=1, alpha=0.4)
    a1.set_yscale("log"); a1.set_title("loss"); a1.set_xlabel("step")
    a2.plot(xs, [m["acc"] for m in metrics], label="all bits", lw=0.8)
    a2.plot(xs, [m["acc_result"] for m in metrics], label="result", lw=0.8)
    a2.plot(xs, [m["acc_branch"] for m in metrics], label="branch", lw=0.8)
    a2.set_ylim(0, 1.02); a2.set_title("bit accuracy"); a2.set_xlabel("step"); a2.legend()
    fig.tight_layout(); fig.savefig(run_dir / "curves.png", dpi=110); plt.close(fig)

    save_snapshot(run_dir / "snapshot_latest.png", inp, tgt, last)
    save_rollout_gif(run_dir / "rollout_latest.gif", rollout)

    # archive a checkpoint + snapshot + gif every media_every steps
    if step % args.media_every == 0 and step > 0:
        torch.save({"nca": nca.state_dict(), "optim": optim.state_dict(), "step": step,
                    "metrics": metrics, "nca_config": nca_config,
                    "layout": {"W": layout.W, "H": layout.H, "word_bits": layout.word_bits}},
                   ckpt_dir / f"nca_{step:06d}.pt")
        shutil.copy(run_dir / "snapshot_latest.png", run_dir / "snapshots" / f"snapshot_{step:07d}.png")
        shutil.copy(run_dir / "rollout_latest.gif", run_dir / "rollouts" / f"rollout_{step:07d}.gif")

print(f"\nDone. Run: {run_dir}")
