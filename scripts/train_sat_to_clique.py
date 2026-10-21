#!/usr/bin/env python3
"""Train an NCA to perform the 3SAT -> CLIQUE Karp reduction as morphogenesis.

The input screen carries only the literal codes — one signed one-hot strip per
literal occurrence, written down the left edge and across the top.  The NCA must
grow, in place, the ``3m x 3m`` adjacency matrix of the reduced graph: body cell
(u, v) = +1 if occurrences u and v are joined by an edge, -1 if not.  Edge iff
the two occurrences sit in different clauses and their literals are not
complementary; see ``ncpu.karp`` for the reference implementation and for why
the diagonal blocks are handed over as geometry.

Setup (mirrors scripts/train_subleq.py so results stay comparable):
    * 3x3 fixed Sobel perception, 2-layer ReLU rule
    * channel 0 = mutable state, channel 1 = frozen copy of the input
    * alive masking OFF, fire_rate 0.5
    * whole-screen MSE with extra weight on the body cells, and a further
      upweight on NON-edges, which are the minority class

Metrics, in order of how much they mean:
    exact       fraction of samples whose ENTIRE scored adjacency matrix is
                right -- this is what "performed the reduction" means
    nonedge     recall on non-edges (the minority class; the whole task)
    cell        per-cell accuracy, printed next to the constant-rule baseline
                ("every off-block pair is an edge"), which is already ~1-1/(2n)
    extrap      exact-adjacency accuracy on a LARGER instance the rule has
                never seen -- the rule is convolutional, so this is free to try
                and is the headline claim of the track

Rollout depth is derived from the layout, never hard-coded: a 3x3 kernel moves
information one pixel per step, so the requirement is an L1 distance in pixels
(``CliqueLayout.min_rollout``).  A run below it is physically impossible, and
the script refuses to start.

Reproducibility: ``--seed`` seeds torch, numpy AND the ``random`` module, which
is what ``ncpu.sat``'s samplers draw from by default, so the formula stream is
a function of the seed alone.  Training and the extrapolation eval draw from
*separate* ``random.Random`` streams, so changing --eval-batch or --eval-every
cannot shift the training instances.

Memory: the rollout is stepped and only the channel-0 frames are kept, because
``NeuralCA.forward(steps=T)`` stacks the whole (B, T, C, H, W) sequence -- ~7 GB
at n=8, m=8 before the activations are counted, which would bite exactly when
demonstrating size extrapolation.  Same compute.  ``--grad-checkpoint K``
recomputes the rollout in K-step segments in the backward pass instead of
storing it; the startup banner prints the estimate.

Usage:
    uv run python scripts/train_sat_to_clique.py
    uv run python scripts/train_sat_to_clique.py --vars 6 --clauses 6 --steps 40000
    uv run python scripts/train_sat_to_clique.py --layout compact --block-hint none
    uv run python scripts/train_sat_to_clique.py --resume last
"""

import json
import random
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
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.gif import write_gif
from ncpu.karp import CliqueLayout, CompactCliqueLayout
from ncpu.nca import NeuralCA

try:                                  # ncpu.utils pulls in cv2, which needs libGL —
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
# instance shape
parser.add_argument("--vars", type=int, default=5, help="n: variables per instance")
parser.add_argument("--clauses", type=int, default=4, help="m: clauses (nodes = 3m)")
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--r", type=int, default=1,
                    help="lattice radius (step = 3r). The step multiplies straight "
                         "into the rollout depth, so keep it tight: a body cell "
                         "carries one bit, not a word.")
parser.add_argument("--layout", choices=["onehot", "compact"], default="onehot",
                    help="literal code: signed one-hot (n lanes) or binary index "
                         "(log2 n + 1 lanes)")
parser.add_argument("--block-hint", choices=["diag", "wall", "none"], default="diag",
                    help="how the instance-independent i != j geometry is handled")
parser.add_argument("--want", choices=["any", "sat", "unsat", "unique"], default="any",
                    help="training distribution; the reduction is defined for all "
                         "formulas, so 'any' needs no balancing")
# rollout / model
parser.add_argument("--rollout-min", type=int, default=None,
                    help="default: layout.suggested_rollout()")
parser.add_argument("--rollout-max", type=int, default=None,
                    help="default: 1.2 x rollout-min")
parser.add_argument("--force-short", action="store_true",
                    help="allow a rollout below the layout's light cone (it cannot work)")
parser.add_argument("--channels", type=int, default=16)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--kernel", type=int, default=3)
parser.add_argument("--fire-rate", type=float, default=0.5)
parser.add_argument("--grad-checkpoint", type=int, default=0, metavar="K",
                    help="recompute the rollout in K-step segments during backward "
                         "instead of storing them; ~sqrt memory for ~1.3x compute. "
                         "0 = off")
# loss
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr-final", type=float, default=None,
                    help="exp-decay lr to this by the last step")
parser.add_argument("--body-weight", type=float, default=5.0,
                    help="extra weight on the body pixels the NCA must produce")
parser.add_argument("--nonedge-weight", type=float, default=4.0,
                    help="further multiplier on NON-edge body pixels (minority class)")
parser.add_argument("--supervise", choices=["last", "all"], default="all")
# extrapolation eval
parser.add_argument("--extrap-vars", type=int, default=None, help="default: vars + 1")
parser.add_argument("--extrap-clauses", type=int, default=None, help="default: clauses + 2")
parser.add_argument("--no-extrap", action="store_true")
parser.add_argument("--eval-every", type=int, default=1000)
parser.add_argument("--eval-batch", type=int, default=8)
# bookkeeping
parser.add_argument("--plot-every", type=int, default=500)
parser.add_argument("--media-every", type=int, default=5000)
parser.add_argument("--resume", type=str, default=None, help="run name, or 'last'")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
# ncpu.sat's samplers default to the GLOBAL random module, so without this the
# formula stream is not a function of --seed at all.
random.seed(args.seed)
# ...and the eval/extrapolation path gets its own stream, so evaluating never
# perturbs the training instances (Track A's pattern).
TRAIN_RNG = random.Random(args.seed)
EVAL_RNG = random.Random(args.seed + 7919)
DEVICE = args.device

# ── Geometry ─────────────────────────────────────────────────────────────────
Layout = CliqueLayout if args.layout == "onehot" else CompactCliqueLayout
layout = Layout(n_vars=args.vars, m_clauses=args.clauses, k=args.k, r=args.r,
                block_hint=args.block_hint)

ROLL_MIN = args.rollout_min if args.rollout_min is not None else layout.suggested_rollout()
ROLL_MAX = args.rollout_max if args.rollout_max is not None else int(1.2 * ROLL_MIN)
if ROLL_MIN < layout.min_rollout() and not args.force_short:
    raise SystemExit(
        f"rollout-min {ROLL_MIN} is below the light cone: a 3x3 kernel moves one "
        f"pixel per step and the farthest header is {layout.min_rollout()} px away, "
        f"so the task is physically impossible. Use --rollout-min "
        f">= {layout.min_rollout()} (suggested {layout.suggested_rollout()}), a "
        f"smaller --r, fewer --clauses, or --force-short to insist."
    )

extrap = None
if not args.no_extrap:
    extrap = Layout(n_vars=args.extrap_vars or (args.vars + 1),
                    m_clauses=args.extrap_clauses or (args.clauses + 2),
                    k=args.k, r=args.r, block_hint=args.block_hint)
    EXTRAP_ROLL = extrap.suggested_rollout()

nca_config = dict(
    channels=args.channels,
    hidden_channels=[args.hidden],   # single hidden layer -> 2-layer ReLU rule
    fire_rate=args.fire_rate,
    alive_threshold=0,               # alive masking OFF (every cell updates)
    zero_initialization=True,
    kernel_size=args.kernel,
    perception="static_sobel",
    padding_type="zeros",
    read_only_dims=[1],              # channel 1 holds a frozen copy of the input
)
INPUT_DIMS = (0, 1)                  # implant into ch0 (mutable) and ch1 (frozen)

# ── Run directory ────────────────────────────────────────────────────────────
if args.resume == "last":
    runs = sorted(d.name for d in Path("runs").iterdir()
                  if "KARP" in d.name and (d / "checkpoints" / "nca_last.pt").exists())
    if not runs:
        raise RuntimeError("No resumable KARP runs found")
    args.resume = runs[-1]

if args.resume:
    run_dir = Path("runs") / args.resume
    ckpt_dir = run_dir / "checkpoints"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
else:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / (f"{stamp}_KARP_clique_{args.layout}_"
                              f"n{args.vars}m{args.clauses}_s{args.seed}")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "rollouts").mkdir()
    (run_dir / "snapshots").mkdir()
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "run": {"name": run_dir.name, "seed": args.seed,
                    "total_steps": args.steps, "device": DEVICE},
            "layout": {"kind": args.layout, "W": layout.W, "H": layout.H,
                       "n": layout.n, "m": layout.m, "k": layout.k, "N": layout.N,
                       "header_lanes": layout.D, "block_hint": layout.block_hint,
                       "r": layout.r, "step": layout.step,
                       "min_rollout": layout.min_rollout(),
                       "edge_baseline": layout.edge_baseline(),
                       "extrap": None if extrap is None else
                       {"n": extrap.n, "m": extrap.m, "W": extrap.W,
                        "rollout": EXTRAP_ROLL}},
            "nca": nca_config,
            "optim": {"lr": args.lr, "lr_final": args.lr_final,
                      "batch_size": args.batch_size, "rollout": [ROLL_MIN, ROLL_MAX],
                      "body_weight": args.body_weight,
                      "nonedge_weight": args.nonedge_weight,
                      "grad_checkpoint": args.grad_checkpoint,
                      "supervise": args.supervise, "want": args.want},
            "env": {"python": sys.version, "torch": torch.__version__, "git": git_info()},
        }, f, indent=2)

print(f"karp 3SAT->CLIQUE  seed={args.seed}  "
      f"{'(resume) ' if args.resume else ''}run: {run_dir.name}")
print(layout.describe())
print(f"rollout {ROLL_MIN}..{ROLL_MAX}  (light cone {layout.min_rollout()} px)")
if extrap is not None:
    print("extrapolation eval: " + extrap.describe() + f"  rollout {EXTRAP_ROLL}")


def activation_gb(lay, batch, steps, segment=0):
    """Rough BPTT activation cost: per step the rule keeps the 3C perception,
    the hidden layer and two C-wide state tensors.  With checkpointing only one
    segment's worth is stored, plus the channel-0 frames, which are always kept
    because the loss reads them."""
    per_cell = 3 * args.channels + args.hidden + 2 * args.channels
    per_step = lay.W * lay.H * batch * per_cell * 4
    kept = steps if segment <= 0 else segment
    frames = lay.W * lay.H * batch * steps * 4
    return (per_step * kept + frames) / 1024 ** 3


_cost = activation_gb(layout, args.batch_size, ROLL_MAX, args.grad_checkpoint)
_note = f" with --grad-checkpoint {args.grad_checkpoint}" if args.grad_checkpoint > 0 else ""
print(f"BPTT activations ~{_cost:.1f} GB{_note} at batch {args.batch_size}, "
      f"rollout <={ROLL_MAX}")
if args.grad_checkpoint <= 0 and _cost > 8:
    print(f"  (--grad-checkpoint 64 would cut that to "
          f"~{activation_gb(layout, args.batch_size, ROLL_MAX, 64):.1f} GB "
          f"for ~1.3x compute)")

# ── Model / optimiser ────────────────────────────────────────────────────────
nca = NeuralCA(**nca_config).to(DEVICE)
optim = torch.optim.Adam(nca.parameters(), lr=args.lr)
start_step = 0
metrics = []

if args.resume:
    ckpt = torch.load(ckpt_dir / "nca_last.pt", map_location=DEVICE)
    nca.load_state_dict(ckpt["nca"])
    optim.load_state_dict(ckpt["optim"])
    start_step = ckpt["step"] + 1        # the saved step is finished; resume after it
    metrics = ckpt.get("metrics", [])
    print(f"resumed from step {start_step}")

_MASKS = {}


def masks_for(lay):
    """(body, header, scored) masks on DEVICE, cached per layout."""
    key = id(lay)
    if key not in _MASKS:
        _MASKS[key] = (lay.body_mask().to(DEVICE),        # (H, W) scored body pixels
                       lay.header_mask().to(DEVICE),      # (H, W) must be preserved
                       lay.scored_mask().to(DEVICE))      # (N, N) bool
    return _MASKS[key]


def implant(inp, ch=args.channels):
    state = torch.zeros(inp.shape[0], ch, inp.shape[-2], inp.shape[-1], device=inp.device)
    for d in INPUT_DIMS:
        state[:, d] = inp
    return state


def weight_map(tgt, lay=layout):
    """Body pixels weighted, non-edge body pixels weighted more.

    Non-edges are the minority class by construction (two occurrences clash only
    when they share a variable with opposite signs, i.e. with probability about
    1/(2n)), so an unweighted loss is happy to paint the whole body +1.
    """
    return masks_for(lay)[0] * torch.where(tgt < 0, args.nonedge_weight, 1.0)


def _segment(state, k):
    """k NCA steps, returning the new state and the channel-0 frames."""
    frames = []
    for _ in range(int(k)):
        state = nca.forward(state, steps=1)[:, -1]
        frames.append(state[:, 0])
    return state, torch.stack(frames, dim=1)


def rollout_frames(state, steps):
    """Step the NCA, keeping only the channel-0 frames (B, T, H, W).

    ``NeuralCA.forward(steps=T)`` stacks the whole (B, T, C, H, W) sequence,
    which at these depths is gigabytes of tensor the loss never reads (~7 GB at
    n=8, m=8, batch 8).  Stepping here costs the same compute and keeps only
    channel 0.  The seed frame is not included, so the frames are steps 1..T --
    the same thing the old ``rollout[:, 1:, 0]`` slice produced.

    With --grad-checkpoint K the segments are recomputed in the backward pass
    instead of stored (checkpoint preserves RNG state, so the fire-rate masks
    are reproduced exactly).
    """
    K = args.grad_checkpoint if args.grad_checkpoint > 0 else steps
    out, done = [], 0
    while done < steps:
        k = min(K, steps - done)
        if args.grad_checkpoint > 0 and torch.is_grad_enabled():
            state, seg = checkpoint(_segment, state, k, use_reentrant=False)
        else:
            state, seg = _segment(state, k)
        out.append(seg)
        done += k
    return torch.cat(out, dim=1)


def run_batch(bs, steps, lay=layout, grad=True, rng=None):
    inp, tgt, adj, _ = lay.sample_batch(bs, device=DEVICE, rng=rng or TRAIN_RNG,
                                        want=args.want)
    seq = rollout_frames(implant(inp), steps)        # (B, T, H, W), seed excluded
    last = seq[:, -1]
    if not grad:
        return inp, tgt, adj, seq, last, None
    T = seq.shape[1]
    diff = (seq - tgt.unsqueeze(1)) ** 2             # (B, T, H, W)
    w = weight_map(tgt, lay).unsqueeze(1)            # (B, 1, H, W)
    if args.supervise == "all":
        loss = diff.mean() + args.body_weight * (diff * w).sum() / (w.sum() * T)
    else:
        d = (last - tgt) ** 2
        wl = weight_map(tgt, lay)
        loss = d.mean() + args.body_weight * (d * wl).sum() / wl.sum()
    return inp, tgt, adj, seq, last, loss


def score(last, tgt, adj, lay=layout):
    """The metrics that matter, all over the scored (off-block) cells only."""
    _, header_mask, sc = masks_for(lay)
    got = lay.decode_adjacency(last)                          # (B, N, N) bool
    ok = (got == adj)[:, sc]                                  # (B, K)
    edge = adj[:, sc]
    n_edge = edge.sum().clamp(min=1)
    n_non = (~edge).sum().clamp(min=1)
    return {
        "exact": ok.all(dim=-1).float().mean().item(),        # the real metric
        "cell": ok.float().mean().item(),
        "acc_edge": (ok & edge).sum().item() / n_edge.item(),
        "acc_nonedge": (ok & ~edge).sum().item() / n_non.item(),
        "baseline": edge.float().mean().item(),               # "always edge" accuracy
        "header_mse": (((last - tgt) ** 2) * header_mask).sum().item()
                      / (header_mask.sum().item() * last.shape[0]),
    }


@torch.no_grad()
def eval_layout(lay, steps, bs):
    """Held-out score on a LARGER layout, from its own RNG stream.

    EVAL_RNG keeps the extrapolation eval out of the training instance stream,
    so changing --eval-batch or --eval-every cannot shift what the model trains
    on.  The state is stepped rather than stacked: only the final frame is
    scored, so there is no reason to hold (B, T, C, H, W) at the extrapolation
    shape, which is the biggest tensor this script would ever build.
    """
    inp, tgt, adj, _ = lay.sample_batch(bs, device=DEVICE, rng=EVAL_RNG,
                                        want=args.want)
    # the fire-rate masks come from torch's RNG, so an eval of a different size
    # would otherwise shift every subsequent training step's dropout pattern.
    cpu_state = torch.get_rng_state()
    cuda_state = (torch.cuda.get_rng_state(DEVICE)
                  if torch.cuda.is_available() and str(DEVICE).startswith("cuda")
                  else None)
    try:
        x = implant(inp)
        for _ in range(steps):
            x = nca.forward(x, steps=1)[:, -1]
        return score(x[:, 0], tgt, adj, lay=lay)
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state(cuda_state, DEVICE)


def save_snapshot(path, inp, tgt, last, n=4):
    n = min(n, inp.shape[0])
    fig, axes = plt.subplots(4, n, figsize=(2.2 * n, 8.8))
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


def save_rollout_gif(path, tgt, frames, n=2, stride=2, fps=18, zoom=4):
    """Target on top, the NCA growing the reduction below.

    ``frames`` is the (B, T, H, W) channel-0 sequence from ``rollout_frames``.
    """
    n = min(n, frames.shape[0])
    nca_seq = frames[:n, ::stride].detach().cpu().numpy()
    tgt_seq = np.repeat(tgt[:n].detach().cpu().numpy()[:, None], nca_seq.shape[1], axis=1)
    top = np.concatenate([tgt_seq[i] for i in range(n)], axis=2)
    bot = np.concatenate([nca_seq[i] for i in range(n)], axis=2)
    write_gif(path, np.concatenate([top, bot], axis=1), fps=fps, vmin=-1, vmax=1, zoom=zoom)


def save_curves(path):
    xs = [m["step"] for m in metrics]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
    a1.scatter(xs, [m["loss"] for m in metrics], s=1, alpha=0.4)
    a1.set_yscale("log"); a1.set_title("loss"); a1.set_xlabel("step")
    a2.plot(xs, [m["exact"] for m in metrics], lw=1.0, label="exact adjacency")
    a2.plot(xs, [m["cell"] for m in metrics], lw=0.8, label="per-cell")
    a2.plot(xs, [m["acc_nonedge"] for m in metrics], lw=0.8, label="non-edge recall")
    a2.plot(xs, [m["baseline"] for m in metrics], lw=0.8, ls=":", color="grey",
            label="all-edges baseline")
    ex = [(m["step"], m["extrap_exact"]) for m in metrics if m.get("extrap_exact") is not None]
    if ex:
        a2.plot([e[0] for e in ex], [e[1] for e in ex], lw=1.0, ls="--",
                label=f"exact @ n={extrap.n} m={extrap.m}")
    a2.set_ylim(0, 1.02); a2.set_xlabel("step"); a2.legend(fontsize=8)
    a2.set_title("did it perform the reduction?")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def save_checkpoint(step):
    torch.save({"nca": nca.state_dict(), "optim": optim.state_dict(), "step": step,
                "metrics": metrics, "nca_config": nca_config,
                "layout": {"kind": args.layout, "n": layout.n, "m": layout.m,
                           "k": layout.k, "r": layout.r,
                           "block_hint": layout.block_hint,
                           "W": layout.W, "H": layout.H}},
               ckpt_dir / "nca_last.pt")


# ── Training loop ────────────────────────────────────────────────────────────
log_path = run_dir / "log.jsonl"
pbar = tqdm(range(start_step, start_step + args.steps), ncols=125,
            desc=f"karp s{args.seed}")

lr_final = args.lr_final if args.lr_final is not None else args.lr
inp = tgt = frames = last = None            # so --steps 0 cannot NameError below
for step in pbar:
    if lr_final != args.lr and args.steps > 1:          # exponential lr decay
        frac = (step - start_step) / (args.steps - 1)
        for pg in optim.param_groups:
            pg["lr"] = args.lr * (lr_final / args.lr) ** frac

    steps = int(np.random.randint(ROLL_MIN, ROLL_MAX + 1))
    inp, tgt, adj, frames, last, loss = run_batch(args.batch_size, steps)

    optim.zero_grad()
    loss.backward()
    optim.step()

    with torch.no_grad():
        row = {"step": step, "loss": loss.item(), "rollout": steps}
        row.update(score(last, tgt, adj))
        row["extrap_exact"] = None
        if extrap is not None and step % args.eval_every == 0:
            ev = eval_layout(extrap, EXTRAP_ROLL, args.eval_batch)
            row["extrap_exact"] = ev["exact"]
            row["extrap_cell"] = ev["cell"]
            row["extrap_nonedge"] = ev["acc_nonedge"]

    metrics.append(row)
    pbar.set_description(
        f"karp s{args.seed} loss={row['loss']:.4f} exact={row['exact']:.2f} "
        f"cell={row['cell']:.3f}(base {row['baseline']:.3f}) "
        f"nonedge={row['acc_nonedge']:.2f}"
        + (f" xtrp={row['extrap_exact']:.2f}" if row["extrap_exact"] is not None else "")
    )
    with open(log_path, "a") as f:
        f.write(json.dumps(row) + "\n")

    if step % args.plot_every != 0:
        continue

    save_checkpoint(step)
    save_curves(run_dir / "curves.png")
    save_snapshot(run_dir / "snapshot_latest.png", inp, tgt, last)
    save_rollout_gif(run_dir / "rollout_latest.gif", tgt, frames)

    if step % args.media_every == 0 and step > 0:
        shutil.copy(ckpt_dir / "nca_last.pt", ckpt_dir / f"nca_{step:06d}.pt")
        shutil.copy(run_dir / "snapshot_latest.png",
                    run_dir / "snapshots" / f"snapshot_{step:07d}.png")
        shutil.copy(run_dir / "rollout_latest.gif",
                    run_dir / "rollouts" / f"rollout_{step:07d}.gif")

# The loop above only saves on a --plot-every boundary, so a 30k-step run would
# otherwise last save at 29 500 and throw the final 500 steps away.
if metrics and frames is not None:
    final = metrics[-1]["step"]
    save_checkpoint(final)
    save_curves(run_dir / "curves.png")
    save_snapshot(run_dir / "snapshot_latest.png", inp, tgt, last)
    save_rollout_gif(run_dir / "rollout_latest.gif", tgt, frames)
    print(f"final checkpoint at step {final}")

print(f"\nDone. Run: {run_dir}")
