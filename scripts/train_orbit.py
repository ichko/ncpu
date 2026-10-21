#!/usr/bin/env python3
"""Train the orbit-counter NCA — a loop you can see.

The NCA must run a bright comet around a dim closed track forever and tick a
tally register every time the comet passes a fixed tap point.  The comet's
position is the program counter; there is no clock channel and no halt channel.

Setup:
    * 3x3 fixed Sobel perception, 2-layer ReLU rule (mirrors train_subleq.py)
    * fire_rate = 1.0 and alive masking OFF — a stochastic firing rate jitters
      the orbital period, which fights a phase-locked target
    * channel 0 = mutable state, channel 1 = frozen map of the track geometry
    * the comet is supervised on every frame; the counter is unsupervised for
      `settle` frames after each crossing so the carry ripple has time to run

The metric that matters is `pos_err` on a rollout much longer than the training
horizon: flat = a genuine loop, rising = a drifting clock, exploding = the comet
died or the NCA unrolled the training window.

Usage:
    uv run python scripts/train_orbit.py
    uv run python scripts/train_orbit.py --counter binary --steps 40000
    uv run python scripts/train_orbit.py --resume last
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
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.gif import write_gif
from ncpu.nca import NeuralCA
from ncpu.orbit import OrbitLayout

try:                            # ncpu.utils pulls in cv2, which needs libGL —
    from ncpu.utils import git_info   # absent on the unite compute nodes
except ImportError:
    def git_info():
        return None

# ── Args ─────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=30_000)
parser.add_argument("--device", default="cuda")
parser.add_argument("--batch-size", type=int, default=8)
# geometry
parser.add_argument("--width", type=int, default=44)
parser.add_argument("--height", type=int, default=44)
parser.add_argument("--r", type=int, default=2)
parser.add_argument("--marks", type=int, default=5)
parser.add_argument("--mark-step", type=int, default=7)
parser.add_argument("--counter", choices=["unary", "binary"], default="unary")
parser.add_argument("--tail", type=int, default=6)
parser.add_argument("--speed", type=float, default=0.5, help="comet px per NCA step (<= 1)")
parser.add_argument("--tap", default="bottom")
parser.add_argument("--settle", type=int, default=None, help="counter grace frames (None -> derived)")
# rollout
parser.add_argument("--rollout-min", type=int, default=None, help="default: 1.1 laps")
parser.add_argument("--rollout-max", type=int, default=None, help="default: 1.9 laps")
# model
parser.add_argument("--channels", type=int, default=16)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--kernel", type=int, default=3)
parser.add_argument("--fire-rate", type=float, default=1.0)
# loss
parser.add_argument("--track-weight", type=float, default=5.0)
parser.add_argument("--counter-weight", type=float, default=5.0)
parser.add_argument("--comet-weight", type=float, default=20.0,
                    help="weight on the comet's CURRENT pixels — stops the erase collapse")
parser.add_argument("--comet-dilate", type=int, default=1,
                    help="dilate the per-frame comet mask by this many px")
parser.add_argument("--bright-weight", type=float, default=1.0,
                    help="hinge penalty when peak brightness on the track falls short")
parser.add_argument("--grad-clip", type=float, default=1.0, help="0 disables")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr-final", type=float, default=None, help="exp-decay lr to this by the last step")
parser.add_argument("--init-from", type=str, default=None, help="checkpoint .pt to warm-start from")
# reporting
parser.add_argument("--plot-every", type=int, default=500)
parser.add_argument("--media-every", type=int, default=5000)
parser.add_argument("--eval-steps", type=int, default=1500,
                    help="long-rollout extrapolation probe — several laps past the horizon")
parser.add_argument("--eval-batch", type=int, default=2)
parser.add_argument("--resume", type=str, default=None, help="run name, or 'last'")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
DEVICE = args.device

# ── Geometry, model config ───────────────────────────────────────────────────
layout = OrbitLayout(
    W=args.width, H=args.height, r=args.r, marks=args.marks,
    mark_step=args.mark_step, counter=args.counter, tail=args.tail,
    speed=args.speed, tap=args.tap, settle=args.settle,
)

ROLL_MIN = args.rollout_min if args.rollout_min else int(1.1 * layout.lap_steps)
ROLL_MAX = args.rollout_max if args.rollout_max else int(1.9 * layout.lap_steps)

nca_config = dict(
    channels=args.channels,
    hidden_channels=[args.hidden],   # single hidden layer -> 2-layer ReLU rule
    fire_rate=args.fire_rate,        # 1.0: deterministic, so the period is controllable
    alive_threshold=0,               # alive masking OFF (every cell updates every step)
    zero_initialization=True,
    kernel_size=args.kernel,
    perception="static_sobel",
    padding_type="zeros",
    read_only_dims=[1],              # channel 1 holds the frozen track map
)
STATE_DIM, GEOM_DIM = 0, 1

# ── Run directory ────────────────────────────────────────────────────────────
if args.resume == "last":
    runs = sorted(d.name for d in Path("runs").iterdir()
                  if "ORBIT" in d.name and (d / "checkpoints" / "nca_last.pt").exists())
    if not runs:
        raise RuntimeError("No resumable ORBIT runs found")
    args.resume = runs[-1]

if args.resume:
    run_dir = Path("runs") / args.resume
    ckpt_dir = run_dir / "checkpoints"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
else:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / f"{stamp}_ORBIT_{args.counter}_m{args.marks}_s{args.seed}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "rollouts").mkdir()
    (run_dir / "snapshots").mkdir()
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "run": {"name": run_dir.name, "seed": args.seed, "total_steps": args.steps,
                    "device": DEVICE},
            "layout": {"W": layout.W, "H": layout.H, "r": layout.r,
                       "track": [layout.w, layout.h], "L": layout.L,
                       "lap_steps": layout.lap_steps, "speed": layout.speed,
                       "counter": layout.counter, "marks": layout.marks,
                       "mark_step": layout.mark_step, "tap": layout.tap_name,
                       "tap_s": layout.tap_s, "settle": layout.settle,
                       "tail": layout.tail},
            "nca": nca_config,
            "optim": {"lr": args.lr, "lr_final": args.lr_final,
                      "batch_size": args.batch_size, "rollout": [ROLL_MIN, ROLL_MAX],
                      "track_weight": args.track_weight,
                      "counter_weight": args.counter_weight,
                      "init_from": args.init_from},
            "env": {"python": sys.version, "torch": torch.__version__, "git": git_info()},
        }, f, indent=2)

print(f"orbit  seed={args.seed}  {'(resume) ' if args.resume else ''}run: {run_dir.name}")
print(layout.describe())
print(f"rollout={ROLL_MIN}..{ROLL_MAX} steps ({ROLL_MIN/layout.lap_steps:.1f}..{ROLL_MAX/layout.lap_steps:.1f} laps)"
      f"  long eval={args.eval_steps} ({args.eval_steps/layout.lap_steps:.1f} laps)")

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
    ckpt = torch.load(ckpt_dir / "nca_last.pt", map_location=DEVICE, weights_only=False)
    nca.load_state_dict(ckpt["nca"])
    optim.load_state_dict(ckpt["optim"])
    start_step = ckpt["step"]
    metrics = ckpt.get("metrics", [])
    print(f"resumed from step {start_step}")

track_mask = layout.track_mask().to(DEVICE)          # (H, W)
mark_masks = layout.mark_masks().to(DEVICE)          # (marks, H, W)
mark_area = mark_masks.sum(dim=(-1, -2))             # (marks,)
reg_mask = layout.register_mask().to(DEVICE)         # (H, W)
comet_table = layout.comet_masks(args.comet_dilate).to(DEVICE)   # (L, H, W)


def implant(inp, geom):
    """ch0 = mutable seed state, ch1 = frozen track map, rest zero."""
    bs = inp.shape[0]
    state = torch.zeros(bs, args.channels, layout.H, layout.W, device=DEVICE)
    state[:, STATE_DIM] = inp
    state[:, GEOM_DIM] = geom
    return state


def run_batch(bs, steps):
    inp, geom, tgt, cmask, meta = layout.sample_batch(bs, steps, device=DEVICE)
    rollout = nca.forward(implant(inp, geom), steps=steps)
    seq = rollout[:, 1:, STATE_DIM]                       # (B, T, H, W), skip the seed frame
    tgt_s, cmask_s = tgt[:, 1:], cmask[:, 1:]
    diff = (seq - tgt_s) ** 2
    T = seq.shape[1]

    bg_loss = diff.mean()
    trk_loss = (diff * track_mask).sum() / (track_mask.sum() * bs * T)
    cw = cmask_s[..., None, None]
    cnt_loss = (diff * reg_mask * cw).sum() / (reg_mask.sum() * cmask_s.sum() + 1e-8)

    # Weight on the comet's CURRENT position, not the whole corridor. The
    # corridor is ~99% empty, so an unweighted track loss is minimised by
    # deleting the comet (run 1 did exactly that within ~300 steps).
    comet_w = comet_table[meta["arc"][:, 1:]]                      # (B, T, H, W)
    comet_loss = (diff * comet_w).sum() / (comet_w.sum() + 1e-8)

    # Hinge on peak brightness over the track: directly punishes a dead comet.
    peak_nca = (seq * track_mask).amax(dim=(-1, -2))                # (B, T)
    peak_tgt = (tgt_s * track_mask).amax(dim=(-1, -2))
    bright_loss = F.relu(peak_tgt - peak_nca).pow(2).mean()

    loss = (bg_loss
            + args.track_weight * trk_loss
            + args.comet_weight * comet_loss
            + args.counter_weight * cnt_loss
            + args.bright_weight * bright_loss)
    parts = {"bg": bg_loss.item(), "track": trk_loss.item(),
             "comet": comet_loss.item(), "counter": cnt_loss.item(),
             "bright": bright_loss.item()}
    return rollout, tgt, cmask, meta, loss, parts


def read_counter(frames):
    """(B, T, H, W) -> (B, T, marks) mean activation per register mark."""
    return (frames.unsqueeze(2) * mark_masks).sum(dim=(-1, -2)) / mark_area


@torch.no_grad()
def score(rollout, tgt, cmask, meta):
    """pos_err (px), counter bit accuracy, comet contrast."""
    seq = rollout[:, :, STATE_DIM]                       # (B, T, H, W) incl. seed
    pred_arc = layout.decode_arc(seq)                    # (B, T)
    pos_err = layout.arc_error(pred_arc, meta["true_arc"])

    nca_marks = read_counter(seq)
    tgt_marks = read_counter(tgt)
    correct = ((nca_marks > 0) == (tgt_marks > 0)).float().mean(-1)   # (B, T)
    cnt_acc = (correct * cmask).sum() / cmask.sum().clamp(min=1)

    contrast = (seq * track_mask).amax(dim=(-1, -2))      # (B, T) peak comet brightness
    return (pos_err.mean().item(), pos_err[:, -1].mean().item(),
            cnt_acc.item(), contrast.min().item(), pos_err)


@torch.no_grad()
def long_eval(steps, bs):
    """Extrapolation probe: run far past the training horizon."""
    nca.eval()
    inp, geom, tgt, cmask, meta = layout.sample_batch(bs, steps, device=DEVICE)
    rollout = nca.forward(implant(inp, geom), steps=steps)
    _, _, cnt_acc, contrast, pos_err = score(rollout, tgt, cmask, meta)
    nca.train()
    per_t = pos_err.mean(0).cpu().numpy()                 # (T,)
    return per_t, cnt_acc, contrast


def save_snapshot(path, tgt, rollout, n_times=6):
    """One sample, several time points: target / nca / |diff|."""
    T = rollout.shape[1]
    ts = np.linspace(0, T - 1, n_times).astype(int)
    nca_seq = rollout[0, :, STATE_DIM].detach().cpu()
    tgt_seq = tgt[0].detach().cpu()
    fig, axes = plt.subplots(3, n_times, figsize=(2.0 * n_times, 6.2))
    rows = [("target", tgt_seq), ("nca ch0", nca_seq),
            ("|diff|", (nca_seq - tgt_seq).abs())]
    for r, (name, data) in enumerate(rows):
        for c, t in enumerate(ts):
            ax = axes[r, c]
            vmin, vmax = (0, 2) if name == "|diff|" else (-1, 1)
            ax.imshow(data[t], cmap="viridis", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={t}", fontsize=8)
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def save_rollout_gif(path, tgt, rollout, n=2, stride=2, fps=20, zoom=3):
    """target on top, nca below, n samples side by side."""
    n = min(n, rollout.shape[0])
    nca_seq = rollout[:n, ::stride, STATE_DIM].detach().cpu().numpy()
    tgt_seq = tgt[:n, ::stride].detach().cpu().numpy()
    top = np.concatenate([tgt_seq[i] for i in range(n)], axis=2)
    bot = np.concatenate([nca_seq[i] for i in range(n)], axis=2)
    frames = np.concatenate([top, bot], axis=1)
    write_gif(path, frames, fps=fps, vmin=-1, vmax=1, zoom=zoom)


def save_eval_plot(path, per_t, lap_steps):
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(np.arange(len(per_t)), per_t, lw=0.9)
    ax.axvspan(ROLL_MIN, ROLL_MAX, color="tab:green", alpha=0.12,
               label="training horizon")
    for k in range(1, int(len(per_t) / lap_steps) + 1):
        ax.axvline(k * lap_steps, color="grey", lw=0.4, alpha=0.5)
    ax.set_xlabel("NCA step"); ax.set_ylabel("comet position error (px)")
    ax.set_title("extrapolation: flat = a real loop, rising = a drifting clock")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ── Training loop ────────────────────────────────────────────────────────────
log_path = run_dir / "log.jsonl"
pbar = tqdm(range(start_step, start_step + args.steps), ncols=120, desc=f"orbit s{args.seed}")

lr_final = args.lr_final if args.lr_final is not None else args.lr
for step in pbar:
    if lr_final != args.lr and args.steps > 1:          # exponential lr decay over the run
        frac = (step - start_step) / (args.steps - 1)
        for pg in optim.param_groups:
            pg["lr"] = args.lr * (lr_final / args.lr) ** frac

    steps = int(np.random.randint(ROLL_MIN, ROLL_MAX + 1))
    rollout, tgt, cmask, meta, loss, parts = run_batch(args.batch_size, steps)

    optim.zero_grad()
    loss.backward()
    grad_norm = None
    if args.grad_clip > 0:
        grad_norm = torch.nn.utils.clip_grad_norm_(nca.parameters(), args.grad_clip).item()
    optim.step()

    pos_err, pos_err_last, cnt_acc, contrast, _ = score(rollout, tgt, cmask, meta)
    metrics.append({"step": step, "loss": loss.item(), "rollout": steps,
                    "pos_err": pos_err, "pos_err_last": pos_err_last,
                    "count_acc": cnt_acc, "contrast": contrast,
                    "grad_norm": grad_norm, **{f"L_{k}": v for k, v in parts.items()}})
    pbar.set_description(
        f"orbit s{args.seed} loss={loss.item():.4f} pos={pos_err:.1f}px "
        f"(end {pos_err_last:.1f}) cnt={cnt_acc:.2f} peak={contrast:.2f}"
    )
    with open(log_path, "a") as f:
        f.write(json.dumps(metrics[-1]) + "\n")

    if step % args.plot_every != 0:
        continue

    ckpt_blob = {"nca": nca.state_dict(), "optim": optim.state_dict(), "step": step,
                 "metrics": metrics, "nca_config": nca_config,
                 "layout": {"W": layout.W, "H": layout.H, "r": layout.r,
                            "marks": layout.marks, "mark_step": layout.mark_step,
                            "counter": layout.counter, "tap": layout.tap_name,
                            "tail": layout.tail, "speed": layout.speed,
                            "settle": layout.settle}}
    torch.save(ckpt_blob, ckpt_dir / "nca_last.pt")

    xs = [m["step"] for m in metrics]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 3.8))
    a1.scatter(xs, [m["loss"] for m in metrics], s=1, alpha=0.4)
    a1.set_yscale("log"); a1.set_title("loss"); a1.set_xlabel("step")
    a2.plot(xs, [m["pos_err"] for m in metrics], lw=0.7, label="mean")
    a2.plot(xs, [m["pos_err_last"] for m in metrics], lw=0.7, label="last frame")
    a2.set_yscale("log"); a2.set_title("comet position error (px)")
    a2.set_xlabel("step"); a2.legend(fontsize=8)
    a3.plot(xs, [m["count_acc"] for m in metrics], lw=0.8, label="counter acc")
    a3.plot(xs, [m["contrast"] for m in metrics], lw=0.8, label="min peak brightness")
    a3.set_ylim(-1.05, 1.6); a3.set_title("counter / aliveness")
    a3.set_xlabel("step"); a3.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run_dir / "curves.png", dpi=110); plt.close(fig)

    save_snapshot(run_dir / "snapshot_latest.png", tgt, rollout)
    save_rollout_gif(run_dir / "rollout_latest.gif", tgt, rollout)

    if step % args.media_every == 0 and step > 0:
        per_t, ev_cnt, ev_contrast = long_eval(args.eval_steps, args.eval_batch)
        save_eval_plot(run_dir / "eval_latest.png", per_t, layout.lap_steps)
        in_win = float(per_t[ROLL_MIN:ROLL_MAX + 1].mean())
        out_win = float(per_t[ROLL_MAX + 1:].mean()) if len(per_t) > ROLL_MAX + 1 else float("nan")
        print(f"\n[step {step}] long eval {args.eval_steps} steps: "
              f"pos_err in-window={in_win:.2f}px  beyond={out_win:.2f}px  "
              f"final={per_t[-1]:.2f}px  counter={ev_cnt:.2f}  peak>={ev_contrast:.2f}")
        with open(run_dir / "eval.jsonl", "a") as f:
            f.write(json.dumps({"step": step, "eval_steps": args.eval_steps,
                                "pos_err_in_window": in_win,
                                "pos_err_beyond": out_win,
                                "pos_err_final": float(per_t[-1]),
                                "count_acc": ev_cnt, "contrast": ev_contrast}) + "\n")
        torch.save(ckpt_blob, ckpt_dir / f"nca_{step:06d}.pt")
        shutil.copy(run_dir / "snapshot_latest.png", run_dir / "snapshots" / f"snapshot_{step:07d}.png")
        shutil.copy(run_dir / "rollout_latest.gif", run_dir / "rollouts" / f"rollout_{step:07d}.gif")

print(f"\nDone. Run: {run_dir}")
