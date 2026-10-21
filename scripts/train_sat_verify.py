#!/usr/bin/env python3
"""Train the 3SAT verifier NCA (Karp exploration, track A).

The NCA is the polynomial-time NP verifier.  A formula is drawn as a clause
matrix (``ncpu.sat.SATLayout``): n variable columns by m clause rows, with the
candidate assignment as a row of +-1 discs along the top.  The flag column and
the verdict disc are left at background on the input, and the NCA must fill
them in by the LAST rollout step:

    flag_i  = +1 iff clause i is satisfied by the assignment   (an OR)
    verdict = +1 iff every flag is +1                          (an AND)

Why it should be learnable: the computation is local and directional —
broadcast each assignment value down its variable column, multiply against the
literal sign in place (a literal is satisfied iff sign x assignment > 0), OR
along each clause row into the flag column, AND down the flag column into the
verdict.

Depth is the binding constraint
-------------------------------
A 3x3 kernel moves information ONE PIXEL per step, and the lattice pitch is
``step = 2r + gap`` pixels, so the requirement is a distance in PIXELS, not in
lattice cells.  The three legs add rather than max (the OR cannot start before
the assignment value arrives, the AND cannot start before the flags do), which
is exactly ``SATLayout.min_rollout() = (n + m + 1) * step``: with the verdict at
the bottom of the flag column the two critical paths cost the same, either
``m*step`` down to the last clause row + ``n*step`` across + ``1*step`` down to
the verdict, or ``1*step`` down to clause 0 + ``n*step`` across + ``m*step`` for
the flag to fall the length of the column.  The rollout here is always derived
from that method — never from a constant — and it is printed at startup, so an
under-depth run is obvious immediately instead of being logged as a learning
failure.

This is not an artifact of the encoding: a GLOBAL AND costs depth >= the grid
diameter, because some signal has to cross the whole height.  Any single-rollout
verifier pays it, and it is what bounds the instance size one rollout can check.

The cost lever is the lattice pitch, not the rule.  At the honest operating
point n=6, m=26 the pitch is 6 px, min_rollout is 198 and a 1.5-2.5x rollout is
300-500 steps of backprop-through-time.  ``--r 1`` gives pitch 3: half the depth
and a quarter of the cells, i.e. roughly 8x cheaper, at the price of 5-pixel
discs instead of 13-pixel ones (less pixel redundancy per encoded value).
``--grad-checkpoint K`` trades recomputation for activation memory when the
larger pitch is worth keeping.  The startup banner prints the estimate.

Setup follows ``scripts/train_subleq.py``: 3x3 fixed Sobel perception, 2-layer
ReLU rule, input implanted into channel 0 (the canvas) and a frozen channel 1,
alive masking off, fire_rate 0.5, zero padding.

The output region needs THREE weights, not one
---------------------------------------------
The output discs are wildly imbalanced in two directions at once, and a plain
per-pixel MSE over them has a well-defined failure mode for each.

Inside the flag column, ~95% of clauses are satisfied at the phase ratio, so
per-pixel MSE is dominated by the trivially-positive discs and "stamp the whole
column satisfied" is a deep local minimum — the first GPU probe sat in it from
step ~1500 with ``flag_acc`` equal to ``flag_base`` to three decimals.
``--flag-neg-weight`` (default: the inverse class frequency, measured on a probe
batch at startup, ~18x) multiplies flag discs whose target is NEGATIVE, applied
per sample because which clauses fail varies by instance.  ``--flag-neg-weight
1`` reproduces the unweighted behaviour exactly.  Watch ``flag_pos_rate_pred``:
the fix is working when it falls from 1.0 toward the truth rate.

The verdict needs its own loss weight
-------------------------------------
The output region is 27 discs, and only ONE of them is the verdict.  By pixel
count that is ~4% of the output loss term — for the single hard bit, the one
whose constant baseline is 0.50 while the flags' is 0.95.  That is the textbook
"per-bit high, exact-instance low" credit-assignment failure, so the verdict
disc gets its own per-pixel multiplier, ``--verdict-weight``, defaulting to the
whole flag column's weighted mass, which equalises the two regions' share of the
output term (that is exactly m when the flags are unweighted, so the control
arm is unaffected).
The realised shares are printed at startup rather than left implied.
``--verdict-warmup`` ramps that weight in over the first N steps, on the theory
that the AND is only learnable once the flags are roughly right; it is off by
default because a low early verdict weight is also an invitation to settle into
"always SAT".

Read the metrics against their constant-predictor baselines
-----------------------------------------------------------
Three accuracies are reported and they are NOT interchangeable:
    flag     per-clause bit accuracy over the m flag discs
    verdict  the single verdict disc — the hard one, because it is the AND
    exact    all m + 1 bits correct on the same instance
Each is printed next to the majority-class baseline measured on the same batch,
because the raw numbers mislead.  Tier 0 (results/karp/tier0_ratio433.json) at
n=6, m=26: verdict baseline 0.501, exact 0.501, but FLAGS 0.950 — 95% of clauses
are satisfied, so "always satisfied" scores 0.95 on flags.  A flag accuracy
below ~0.95 is worse than a constant.  ``pred_sat`` next to ``sat_rate`` catches
the other degenerate rule.

Default shape is n=6, m=26 (ratio 4.33, the phase transition): ratio 2 is 100%
satisfiable with a dozen solutions and does not exercise the AND.  Batch balance
lives in ``SATLayout.sample_batch`` and must not be defeated; ``--sat-frac`` /
``--near-miss-frac`` surface it.

Usage:
    uv run python scripts/train_sat_verify.py
    uv run python scripts/train_sat_verify.py --r 1 --batch-size 8
    uv run python scripts/train_sat_verify.py --resume last
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
from ncpu.nca import NeuralCA
from ncpu.sat import SATLayout
from ncpu.utils import git_info

# ── Args ─────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=30_000)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch-size", type=int, default=4)
# instance shape — default sits at the phase ratio (m/n = 4.33)
parser.add_argument("--n-vars", type=int, default=6)
parser.add_argument("--m-clauses", type=int, default=26)
parser.add_argument("--k", type=int, default=3, help="literals per clause")
parser.add_argument("--r", type=int, default=2,
                    help="disc radius; lattice pitch is 2r+gap px and the required "
                         "rollout scales with it — r=1 roughly halves the depth")
parser.add_argument("--gap", type=int, default=None, help="px between discs (default r)")
parser.add_argument("--pad", type=int, default=None, help="border px (default 2r)")
parser.add_argument("--sat-frac", type=float, default=0.5,
                    help="fraction of assignments that DO satisfy the formula")
parser.add_argument("--near-miss-frac", type=float, default=0.5,
                    help="of the negatives, fraction built by flipping one variable "
                         "of a real solution (the informative ones)")
# rollout: always derived from layout.min_rollout() unless given explicitly
parser.add_argument("--rollout-min", type=int, default=None)
parser.add_argument("--rollout-max", type=int, default=None)
parser.add_argument("--rollout-scale-min", type=float, default=1.5,
                    help="rollout-min = scale x layout.min_rollout() when not given")
parser.add_argument("--rollout-scale-max", type=float, default=2.5)
# model
parser.add_argument("--channels", type=int, default=16)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--kernel", type=int, default=3)
# optimisation
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr-final", type=float, default=None, help="exp-decay lr to this by the last step")
parser.add_argument("--grad-clip", type=float, default=1.0,
                    help="clip the global gradient norm to this before each optimiser "
                         "step. 0 disables clipping (and reproduces runs made before it "
                         "existed bit for bit — the norm is still measured and logged)")
parser.add_argument("--out-weight", type=float, default=5.0,
                    help="extra weight on the output region (flags + verdict) in the loss")
parser.add_argument("--verdict-weight", type=float, default=None,
                    help="per-pixel weight of the verdict disc relative to a flag pixel. "
                         "Default m, which equalises the two regions' share of the output "
                         "term; 1.0 restores plain per-pixel weighting (~4%% verdict)")
parser.add_argument("--flag-neg-weight", type=float, default=None,
                    help="per-pixel weight of flag discs whose target is NEGATIVE "
                         "(unsatisfied clauses) relative to positive ones. Default is the "
                         "inverse class frequency measured on a probe batch (~18x at the "
                         "phase ratio), which equalises the two classes' loss mass; "
                         "1 restores plain per-pixel weighting (~95%% of it positive)")
parser.add_argument("--flag-probe", type=int, default=64,
                    help="instances used to measure the flag class balance at startup")
parser.add_argument("--verdict-warmup", type=int, default=0, metavar="N",
                    help="ramp the verdict weight 1 -> --verdict-weight over the first N "
                         "steps (flags first, then the AND). 0 = full weight from step 0")
parser.add_argument("--supervise", choices=["last", "all"], default="last",
                    help="'last' supervises only the final --last-frames frames (the "
                         "default: the answer physically cannot exist before the light "
                         "cone closes, so supervising every step penalises propagation); "
                         "'all' supervises every rollout step, as train_subleq.py does")
parser.add_argument("--last-frames", type=int, default=8,
                    help="supervise='last': how many final frames to score, so the "
                         "answer has to be a settled state rather than a transient")
parser.add_argument("--grad-checkpoint", type=int, default=0, metavar="K",
                    help="recompute the rollout in K-step segments during backward "
                         "instead of storing them; ~sqrt memory for ~1.3x compute. "
                         "0 = off")
parser.add_argument("--init-from", type=str, default=None, help="checkpoint .pt to warm-start from")
# bookkeeping
parser.add_argument("--plot-every", type=int, default=500)
parser.add_argument("--media-every", type=int, default=5000,
                    help="archive a checkpoint + snapshot + rollout gif every N steps")
parser.add_argument("--eval-batch", type=int, default=64,
                    help="held-out batch size for the periodic accuracy estimate")
parser.add_argument("--resume", type=str, default=None, help="run name, or 'last'")
parser.add_argument("--runs-dir", default="runs",
                    help="where run directories live. Point it somewhere scratch for "
                         "throwaway verification runs, so they never share a directory "
                         "(or a glob) with real training runs")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
# SATLayout.sample_batch draws from a `random` instance. Training and eval get
# SEPARATE streams, so running an eval never perturbs the training instance
# sequence and a run stays reproducible instance for instance.
TRAIN_RNG = random.Random(args.seed)
EVAL_RNG = random.Random(args.seed + 10_000)
PROBE_RNG = random.Random(args.seed + 30_000)   # startup class-balance measurement
DEVICE = args.device

if DEVICE.startswith("cuda"):
    if not torch.cuda.is_available():
        raise SystemExit(f"--device {DEVICE} but torch.cuda.is_available() is False")
    free, total = torch.cuda.mem_get_info(torch.device(DEVICE))
    print(f"GPU {torch.cuda.get_device_name(DEVICE)}: "
          f"{free / 1024 ** 3:.1f} of {total / 1024 ** 3:.1f} GB free")

# ── Geometry, depth, cost ────────────────────────────────────────────────────
layout = SATLayout(n_vars=args.n_vars, m_clauses=args.m_clauses, k=args.k,
                   r=args.r, gap=args.gap, pad=args.pad)
DEPTH = layout.min_rollout()             # (n + m + 1) * step, in pixels == steps
if args.rollout_min is None:
    args.rollout_min = int(round(args.rollout_scale_min * DEPTH))
if args.rollout_max is None:
    args.rollout_max = max(args.rollout_min, int(round(args.rollout_scale_max * DEPTH)))
if args.rollout_min < DEPTH:
    print(f"WARNING: rollout-min {args.rollout_min} < min_rollout {DEPTH}: the answer "
          f"cannot physically reach the verdict disc in that many steps.")

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
INPUT_DIMS = (0, 1)                   # implant input into ch0 (canvas) and ch1 (frozen)

bit_masks = layout.output_bit_masks("verify").to(DEVICE)   # (m + 1, H, W): flags, then verdict
flag_mask = bit_masks[:-1].sum(dim=0).clamp(0, 1)          # (H, W) union of the m flag discs
verdict_mask = bit_masks[-1]                               # (H, W) the single verdict disc
DISC_PX = float(verdict_mask.sum())                        # pixels in one disc
FLAG_PX = float(flag_mask.sum())
VERDICT_PX = DISC_PX

# Class balance INSIDE the flag column. At the phase ratio ~95% of clauses are
# satisfied, so an unweighted per-pixel MSE over the flags is dominated by the
# trivially-satisfied ones and "stamp the whole column satisfied" is a deep
# local minimum — which is exactly where the first probe run parked. Measure the
# rate rather than hard-coding it: it moves with m, --sat-frac and
# --near-miss-frac. The probe draws from its own RNG stream, so it perturbs
# neither the training nor the eval instance sequence.
_, _, _probe_bits, _ = layout.sample_batch(args.flag_probe, mode="verify", rng=PROBE_RNG,
                                           sat_frac=args.sat_frac,
                                           near_miss_frac=args.near_miss_frac)
FLAG_POS_RATE = float(_probe_bits[:, :-1].float().mean())
FLAG_NEG_RATE = 1.0 - FLAG_POS_RATE
if args.flag_neg_weight is None:                           # inverse class frequency
    args.flag_neg_weight = FLAG_POS_RATE / max(FLAG_NEG_RATE, 1e-6)

# Mean weighted flag mass per clause disc, in units of one positive flag pixel.
# Exactly 1.0 in the unweighted case, so that path stays numerically untouched.
FLAG_MASS = (1.0 if args.flag_neg_weight == 1.0
             else FLAG_POS_RATE + args.flag_neg_weight * FLAG_NEG_RATE)
if args.verdict_weight is None:
    # equalise the verdict's loss mass with the whole (weighted) flag column;
    # reduces to m when the flags are unweighted, so the control arm is unchanged
    args.verdict_weight = float(layout.m) * FLAG_MASS

_flag_mass_px = layout.m * DISC_PX * FLAG_MASS
_verdict_mass_px = args.verdict_weight * DISC_PX
_total_mass = _flag_mass_px + _verdict_mass_px
VERDICT_SHARE = _verdict_mass_px / _total_mass
FLAG_POS_SHARE = layout.m * DISC_PX * FLAG_POS_RATE / _total_mass
FLAG_NEG_SHARE = (layout.m * DISC_PX * args.flag_neg_weight * FLAG_NEG_RATE / _total_mass
                  if args.flag_neg_weight != 1.0 else
                  layout.m * DISC_PX * FLAG_NEG_RATE / _total_mass)

layout_config = dict(n=layout.n, m=layout.m, k=layout.k, r=layout.r, gap=layout.gap,
                     pad=layout.pad, W=layout.W, H=layout.H, min_rollout=DEPTH,
                     out_bits=layout.m + 1, flag_px=FLAG_PX, verdict_px=VERDICT_PX)


def disc_px(r):
    """Pixels in a radius-r disc — the redundancy each encoded value carries."""
    return int(sum(1 for dy in range(-r, r + 1) for dx in range(-r, r + 1)
                   if dx * dx + dy * dy <= r * r))


def activation_gb(lay, batch, steps, segment=0):
    """Rough BPTT activation cost: per step the rule keeps the 3C perception,
    the hidden layer and two C-wide state tensors.  Checkpointing stores only
    one segment's worth plus the channel-0 frames."""
    per_cell = 3 * args.channels + args.hidden + 2 * args.channels
    per_step = lay.W * lay.H * batch * per_cell * 4
    kept = steps if segment <= 0 else segment
    frames = lay.W * lay.H * batch * steps * 4          # channel-0 frames, always kept
    return (per_step * kept + frames) / 1024 ** 3


# ── Run directory ────────────────────────────────────────────────────────────
RUNS_DIR = Path(args.runs_dir)
RUNS_DIR.mkdir(parents=True, exist_ok=True)

if args.resume == "last":
    runs = sorted(d.name for d in RUNS_DIR.iterdir()
                  if "SATVERIFY" in d.name and (d / "checkpoints" / "nca_last.pt").exists())
    if not runs:
        raise RuntimeError("No resumable SATVERIFY runs found")
    args.resume = runs[-1]

if args.resume:
    run_dir = RUNS_DIR / args.resume
    ckpt_dir = run_dir / "checkpoints"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
    # A resumed run keeps writing to the same log, so silently changing the loss
    # weighting mid-run would corrupt the trace. Say so, loudly.
    prev = json.loads((run_dir / "config.json").read_text()).get("optim", {})
    for key, now in [("flag_neg_weight", args.flag_neg_weight),
                     ("verdict_weight", args.verdict_weight),
                     ("out_weight", args.out_weight)]:
        was = prev.get(key)
        if was is not None and abs(float(was) - float(now)) > 1e-9:
            print(f"WARNING: resuming with {key}={now:.4g} but this run was started with "
                  f"{float(was):.4g} — pass --{key.replace('_', '-')} {float(was):.6g} "
                  f"to continue it unchanged.")
else:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RUNS_DIR / f"{stamp}_SATVERIFY_n{layout.n}_m{layout.m}_r{layout.r}_s{args.seed}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "rollouts").mkdir()
    (run_dir / "snapshots").mkdir()
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "run": {"name": run_dir.name, "seed": args.seed, "total_steps": args.steps,
                    "device": DEVICE, "task": "sat_verify"},
            "layout": layout_config,
            "nca": nca_config,
            "optim": {"lr": args.lr, "lr_final": args.lr_final, "batch_size": args.batch_size,
                      "rollout": [args.rollout_min, args.rollout_max],
                      "out_weight": args.out_weight, "verdict_weight": args.verdict_weight,
                      "verdict_warmup": args.verdict_warmup,
                      "flag_neg_weight": args.flag_neg_weight, "grad_clip": args.grad_clip,
                      "flag_pos_rate_probe": FLAG_POS_RATE, "flag_probe": args.flag_probe,
                      "loss_share": {"flag_pos": FLAG_POS_SHARE, "flag_neg": FLAG_NEG_SHARE,
                                     "verdict": VERDICT_SHARE},
                      "supervise": args.supervise,
                      "last_frames": args.last_frames, "grad_checkpoint": args.grad_checkpoint,
                      "sat_frac": args.sat_frac, "near_miss_frac": args.near_miss_frac,
                      "init_from": args.init_from},
            "env": {"python": sys.version, "torch": torch.__version__, "git": git_info()},
        }, f, indent=2)

print("NOTE ON PLATEAUS: at n=3 m=3 this task sat at EXACTLY its baselines for ~5,500 "
      "steps — flat\n  loss, f+pred pinned, verdict at chance — and then broke through to "
      "99.5% verdict by ~7k.\n  A flat early plateau is not evidence of a dead run. Read "
      "steps_since_verdict_improve in\n  eval.jsonl to tell 'never learned' from 'learned "
      "and then lost it'; nca_best.pt holds the\n  best held-out verdict weights either way.")
print(f"sat-verify  seed={args.seed}  {'(resume) ' if args.resume else ''}run: {run_dir.name}")
print(layout.describe())
print(f"min_rollout {DEPTH} px steps (= (n+m+1)*step: {layout.m}*step down to the last "
      f"clause row + {layout.n}*step across + 1*step down to the verdict; the other "
      f"critical path, 1 + {layout.n} + {layout.m}, costs the same) "
      f"-> rollout {args.rollout_min}..{args.rollout_max}  "
      f"[{layout.m} flag bits + verdict]")
print(f"flag class balance ({args.flag_probe}-instance probe): "
      f"{FLAG_POS_RATE:.1%} of the {layout.m} flag discs positive, {FLAG_NEG_RATE:.1%} negative "
      f"-> --flag-neg-weight {args.flag_neg_weight:.4g}")
print(f"output loss shares (1 disc = {DISC_PX:.0f} px): "
      f"flags+ {FLAG_POS_SHARE:5.1%}   flags- {FLAG_NEG_SHARE:5.1%}   "
      f"verdict {VERDICT_SHARE:5.1%} at --verdict-weight {args.verdict_weight:.4g}"
      + (f" (ramped in over {args.verdict_warmup} steps)" if args.verdict_warmup > 0 else ""))
if args.flag_neg_weight == 1.0:
    print(f"  (unweighted flags: the {FLAG_NEG_RATE:.1%} of discs that carry the ANSWER get "
          f"{FLAG_NEG_SHARE:.1%} of the loss — this is the control arm's configuration)")
cost = activation_gb(layout, args.batch_size, args.rollout_max, args.grad_checkpoint)
note = f" with --grad-checkpoint {args.grad_checkpoint}" if args.grad_checkpoint > 0 else ""
print(f"BPTT activations ~{cost:.1f} GB{note} at batch {args.batch_size}"
      f"   grad-clip {args.grad_clip:g}" + (" (OFF)" if args.grad_clip <= 0 else ""))
if args.r > 1:                      # the pitch is the cost lever, so price it up front
    fine = SATLayout(layout.n, layout.m, k=layout.k, r=1)
    fine_roll = int(round(args.rollout_scale_max * fine.min_rollout()))
    print(f"  (--r 1: pitch {fine.step}px, {fine.W}x{fine.H}, min_rollout {fine.min_rollout()}, "
          f"rollout <={fine_roll}, ~{activation_gb(fine, args.batch_size, fine_roll, args.grad_checkpoint):.1f} GB "
          f"— cheaper, but {disc_px(1)}-pixel discs instead of {disc_px(layout.r)}-pixel ones)")
if args.grad_checkpoint <= 0 and cost > 8:
    print(f"  (--grad-checkpoint 64 would cut that to "
          f"~{activation_gb(layout, args.batch_size, args.rollout_max, 64):.1f} GB for ~1.3x compute)")

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

M = layout.m


def output_weight_mask(bits, step):
    """Per-pixel loss weight over the output region.

    Positive flag pixels weigh 1; flag discs whose TARGET is negative — the
    unsatisfied clauses, i.e. the ones that carry the answer — weigh
    ``--flag-neg-weight``, which must be applied per sample because which
    clauses fail varies by instance.  The verdict disc weighs
    ``--verdict-weight``, ramped in linearly by ``--verdict-warmup``.

    Returns (H, W) when the flags are unweighted (the mask is then the same for
    every instance, which keeps the control arm's arithmetic untouched) and
    (B, H, W) otherwise.
    """
    vw = args.verdict_weight
    if args.verdict_warmup > 0:
        frac = min(1.0, max(0, step - start_step) / args.verdict_warmup)
        vw = 1.0 + frac * (vw - 1.0)
    if args.flag_neg_weight == 1.0:
        return flag_mask + vw * verdict_mask                          # (H, W)
    per_flag = torch.where(bits[:, :M].bool(), 1.0, args.flag_neg_weight)   # (B, m)
    return (torch.einsum("bm,mhw->bhw", per_flag, bit_masks[:-1])
            + vw * verdict_mask)                                      # (B, H, W)


def save_ckpt(path, step):
    torch.save({"nca": nca.state_dict(), "optim": optim.state_dict(), "step": step,
                "metrics": metrics, "nca_config": nca_config, "layout": layout_config},
               path)


def implant(inp):
    bs = inp.shape[0]
    state = torch.zeros(bs, args.channels, layout.H, layout.W, device=DEVICE)
    for d in INPUT_DIMS:
        state[:, d] = inp
    return state


def sample(bs, rng=None):
    return layout.sample_batch(bs, mode="verify", device=DEVICE, rng=rng or TRAIN_RNG,
                               sat_frac=args.sat_frac, near_miss_frac=args.near_miss_frac)


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
    which at the depths min_rollout demands is gigabytes by itself.  Stepping
    here costs the same compute and keeps only what the loss reads.  With
    --grad-checkpoint K the segments are recomputed in the backward pass
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


def screen_loss(pred, tgt, weight_mask):
    """Fullscreen MSE plus an extra weight on the output discs.

    ``pred`` is (B, T, H, W), ``tgt`` is (B, H, W) broadcast over T, and
    ``weight_mask`` is (H, W) or (B, H, W) from ``output_weight_mask``.
    """
    diff = (pred - tgt.unsqueeze(1)) ** 2
    bs, T = diff.shape[0], diff.shape[1]
    if weight_mask.dim() == 2:
        out = (diff * weight_mask).sum() / (weight_mask.sum() * bs * T)
    else:                                   # per-sample: the sum already spans the batch
        out = (diff * weight_mask.unsqueeze(1)).sum() / (weight_mask.sum() * T)
    loss = diff.mean() + args.out_weight * out

    # Same quantity per sample, detached, purely diagnostic: a batch loss 25x its
    # neighbours is either one pathological instance (loss_max_sample >> loss) or
    # a genuine optimisation blow-up (they move together). Logging both is what
    # tells those apart, so neither has to be assumed.
    with torch.no_grad():
        wm = weight_mask if weight_mask.dim() == 3 else weight_mask.unsqueeze(0)
        per_out = (diff * wm.unsqueeze(1)).sum((1, 2, 3)) / (wm.sum((-1, -2)) * T)
        per_sample = diff.mean((1, 2, 3)) + args.out_weight * per_out
    return loss, per_sample


def run_batch(bs, steps, step):
    inp, tgt, bits, _ = sample(bs)
    frames = rollout_frames(implant(inp), steps)
    last = frames[:, -1]
    seq = frames if args.supervise == "all" else frames[:, -args.last_frames:]
    loss, per_sample = screen_loss(seq, tgt, output_weight_mask(bits, step))
    return inp, tgt, bits, last, loss, per_sample


def accuracies(last, bits):
    """Model accuracy and the majority-class baseline, side by side.

    Reported separately on purpose.  The verdict is an AND over m flags, so flag
    accuracy can sit at 0.99 while the verdict is chance — and at the phase
    ratio ~95% of clauses are satisfied, so a flag score under its baseline is
    worse than the constant rule "everything is satisfied".

    ``flag_pos_rate_pred`` is that constant rule made directly observable: the
    fraction of flag discs the model decodes as satisfied.  1.0 against a truth
    rate of ~0.95 means the whole column is being stamped positive, which is
    otherwise only inferable from flag_acc coinciding with flag_base.
    """
    got = SATLayout.decode_bits(last, bit_masks)              # (B, m + 1)
    correct = got == bits
    flag_ones = bits[:, :M].float().mean().item()
    sat_rate = bits[:, M].float().mean().item()
    all_zero = (bits.sum(dim=1) == 0).float().mean().item()
    return {
        "flag_acc": correct[:, :M].float().mean().item(),
        "verdict_acc": correct[:, M].float().mean().item(),
        "exact_acc": correct.all(dim=1).float().mean().item(),
        "flag_base": max(flag_ones, 1 - flag_ones),
        "verdict_base": max(sat_rate, 1 - sat_rate),
        "exact_base": max(sat_rate, all_zero),
        "sat_rate": sat_rate,
        "pred_sat": got[:, M].float().mean().item(),
        "flag_pos_rate": flag_ones,                          # truth
        "flag_pos_rate_pred": got[:, :M].float().mean().item(),   # model
    }


class frozen_rng:
    """Run a block without advancing the training RNG streams.

    Evals and gif rollouts draw torch RNG for the fire-rate masks, so without
    this the training trajectory would silently depend on how often you
    evaluate.  Inside the block the seed is fixed too, so the stochastic part of
    an eval is the same at every step and the eval curve tracks learning rather
    than fire-mask luck.
    """

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.seed)

    def __exit__(self, *exc):
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)


EVAL_BATCH = None


@torch.no_grad()
def evaluate(bs, steps):
    """Held-out accuracy, on a FIXED set of instances drawn once from EVAL_RNG.

    Fixed rather than fresh so the curve moves when the model does, not when the
    instances do; ``eval_sat_extrap.py`` is where the large-sample unbiased
    numbers come from at the end of a run.
    """
    global EVAL_BATCH
    if EVAL_BATCH is None:
        inp, _, bits, _ = sample(bs, rng=EVAL_RNG)   # its own stream: never disturbs training
        EVAL_BATCH = (inp, bits)
    inp, bits = EVAL_BATCH
    with frozen_rng(args.seed + 10_000):
        x = implant(inp)
        for _ in range(steps):
            x = nca.forward(x, steps=1)[:, -1]                # step-wise: never stack the rollout
    return accuracies(x[:, 0], bits)


def save_snapshot(path, inp, tgt, last, n=4):
    n = min(n, inp.shape[0])
    fig, axes = plt.subplots(4, n, figsize=(2 * n, 8), squeeze=False)
    rows = [("input", inp), ("target", tgt), ("nca last", last),
            ("|diff|", (last - tgt).abs())]
    for r, (name, data) in enumerate(rows):
        for c in range(n):
            ax = axes[r][c]
            vmin, vmax = (0, 2) if name == "|diff|" else (-1, 1)
            ax.imshow(data[c].detach().cpu(), cmap="viridis", vmin=vmin, vmax=vmax,
                      interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


@torch.no_grad()
def save_rollout_gif(path, inp, steps, n=3, stride=None, fps=20, zoom=2):
    """Fresh no-grad rollout of `n` samples, tiled side by side."""
    n = min(n, inp.shape[0])
    stride = stride or max(1, steps // 120)
    x = implant(inp[:n])
    frames = []
    with frozen_rng(args.seed + 20_000):
        for t in range(steps):
            x = nca.forward(x, steps=1)[:, -1]
            if t % stride == 0:
                frames.append(x[:, 0].cpu().numpy())
    seq = np.stack(frames, axis=1)                                  # (n, T', H, W)
    write_gif(path, np.concatenate(list(seq), axis=2), fps=fps, vmin=-1, vmax=1, zoom=zoom)


# ── Training loop ────────────────────────────────────────────────────────────
# Best-so-far on the fixed held-out batch. Two jobs: keep a copy of the solved
# weights (an n=3 m=3 run reached 99.5% verdict at ~10k and then degraded, which
# loses the result unless it was saved), and measure staleness — "flat and never
# peaked" is a young run, "flat but peaked 6k steps ago" is a collapsing one, and
# they look identical in the loss curve.
best_verdict, best_verdict_step = -1.0, None
for _m in metrics:                                  # restore after --resume
    if _m.get("eval_verdict_acc", -1.0) > best_verdict:
        best_verdict, best_verdict_step = _m["eval_verdict_acc"], _m["step"]

log_path = run_dir / "log.jsonl"
pbar = tqdm(range(start_step, start_step + args.steps), ncols=125, desc=f"sat s{args.seed}")

lr_final = args.lr_final if args.lr_final is not None else args.lr
for step in pbar:
    if lr_final != args.lr and args.steps > 1:          # exponential lr decay over the run
        frac = (step - start_step) / (args.steps - 1)
        lr_now = args.lr * (lr_final / args.lr) ** frac
        for pg in optim.param_groups:
            pg["lr"] = lr_now
    steps = int(np.random.randint(args.rollout_min, args.rollout_max + 1))
    inp, tgt, bits, last, loss, per_sample = run_batch(args.batch_size, steps, step)

    optim.zero_grad()
    loss.backward()
    # max_norm=inf measures the norm without touching the gradients (the scale
    # factor clamps to exactly 1.0), so the pre-clip norm is logged either way.
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        nca.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")))
    clipped = bool(args.grad_clip > 0 and grad_norm > args.grad_clip)
    optim.step()

    with torch.no_grad():
        acc = accuracies(last, bits)
    metrics.append({"step": step, "loss": loss.item(), "rollout": steps,
                    "loss_max_sample": float(per_sample.max()),
                    "grad_norm": grad_norm, "grad_clipped": int(clipped), **acc})
    clip_rate = np.mean([m["grad_clipped"] for m in metrics[-200:]])
    stale = "-" if best_verdict_step is None else f"{step - best_verdict_step}"
    pbar.set_description(
        f"sat s{args.seed} l={loss.item():.4f} "
        f"flag={acc['flag_acc']:.3f}/{acc['flag_base']:.3f} "
        f"f+={acc['flag_pos_rate_pred']:.2f}/{acc['flag_pos_rate']:.2f} "
        f"vd={acc['verdict_acc']:.3f}/{acc['verdict_base']:.3f} "
        f"ex={acc['exact_acc']:.3f} gn={grad_norm:.2f}{'*' if clipped else ''} "
        f"clip={clip_rate:.0%} stale={stale}"
    )
    with open(log_path, "a") as f:
        f.write(json.dumps(metrics[-1]) + "\n")

    if step % args.plot_every != 0:
        continue

    eval_steps = (args.rollout_min + args.rollout_max) // 2
    ev = evaluate(args.eval_batch, eval_steps)
    metrics[-1].update({f"eval_{k}": v for k, v in ev.items()})
    metrics[-1]["eval_rollout"] = eval_steps

    improved = ev["verdict_acc"] > best_verdict + 1e-9
    if improved:
        best_verdict, best_verdict_step = ev["verdict_acc"], step
        save_ckpt(ckpt_dir / "nca_best.pt", step)   # keep the solved weights
    metrics[-1].update({"eval_verdict_best": best_verdict,
                        "eval_verdict_best_step": best_verdict_step,
                        "steps_since_verdict_improve": step - best_verdict_step})
    with open(run_dir / "eval.jsonl", "a") as f:   # log.jsonl is per-step, this is per-eval
        f.write(json.dumps({k: v for k, v in metrics[-1].items()
                            if k == "step" or k.startswith(("eval_", "steps_since"))}) + "\n")

    save_ckpt(ckpt_dir / "nca_last.pt", step)

    xs = [m["step"] for m in metrics]
    ev_xs = [m["step"] for m in metrics if "eval_verdict_acc" in m]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 4))
    a1.scatter(xs, [m["loss"] for m in metrics], s=1, alpha=0.4, label="batch mean")
    a1.scatter(xs, [m.get("loss_max_sample", np.nan) for m in metrics], s=1, alpha=0.25,
               color="tab:red", label="worst sample")
    a1.set_yscale("log"); a1.set_title("loss"); a1.set_xlabel("step")
    a1.legend(fontsize=7)
    for key, base, colour, label in [("flag_acc", "flag_base", "tab:blue", "flag bits"),
                                     ("exact_acc", "exact_base", "tab:green", "exact instance"),
                                     ("verdict_acc", "verdict_base", "tab:red", "verdict")]:
        a2.plot(xs, [m[key] for m in metrics], lw=0.8, color=colour, label=label)
        a2.plot(xs, [m[base] for m in metrics], lw=0.8, ls=":", color=colour, alpha=0.55,
                label=f"{label} baseline")
    a2.plot(xs, [m["flag_pos_rate_pred"] for m in metrics], lw=0.8, color="tab:orange",
            alpha=0.8, label="flag +rate (pred)")
    a2.plot(xs, [m["flag_pos_rate"] for m in metrics], lw=0.8, ls=":", color="tab:orange",
            alpha=0.55, label="flag +rate (truth)")
    if ev_xs:
        a2.plot(ev_xs, [m["eval_verdict_acc"] for m in metrics if "eval_verdict_acc" in m],
                lw=1.4, ls="--", color="k", label=f"verdict (held-out x{args.eval_batch})")
    a2.set_ylim(0, 1.02); a2.set_title("accuracy vs constant-predictor baseline")
    a2.set_xlabel("step"); a2.legend(fontsize=7, ncol=2)
    if best_verdict_step is not None:      # where the held-out verdict last improved
        a2.axvline(best_verdict_step, color="k", lw=0.8, ls="-.", alpha=0.5)

    # gradient norms: are the loss spikes optimisation-side or instance-side?
    gn = [m.get("grad_norm", np.nan) for m in metrics]
    a3.scatter(xs, gn, s=1, alpha=0.4, color="tab:purple")
    if args.grad_clip > 0:
        a3.axhline(args.grad_clip, color="k", lw=0.8, ls="--",
                   label=f"clip {args.grad_clip:g} (fires {clip_rate:.0%} of last 200)")
        a3.legend(fontsize=7)
    a3.set_yscale("log"); a3.set_title("pre-clip gradient norm"); a3.set_xlabel("step")
    fig.tight_layout(); fig.savefig(run_dir / "curves.png", dpi=110); plt.close(fig)

    save_snapshot(run_dir / "snapshot_latest.png", inp, tgt, last)

    # archive a checkpoint + snapshot + gif every media_every steps
    if step % args.media_every == 0 and step > 0:
        save_ckpt(ckpt_dir / f"nca_{step:06d}.pt", step)
        save_rollout_gif(run_dir / "rollout_latest.gif", inp, eval_steps)
        shutil.copy(run_dir / "snapshot_latest.png", run_dir / "snapshots" / f"snapshot_{step:07d}.png")
        shutil.copy(run_dir / "rollout_latest.gif", run_dir / "rollouts" / f"rollout_{step:07d}.gif")

if args.steps > 0:                       # the periodic saves land on plot_every
    final_step = start_step + args.steps - 1   # boundaries, so save the last step too
    save_ckpt(ckpt_dir / "nca_last.pt", final_step)
    save_ckpt(ckpt_dir / f"nca_{final_step:06d}.pt", final_step)
    save_snapshot(run_dir / "snapshot_latest.png", inp, tgt, last)
    save_rollout_gif(run_dir / "rollout_latest.gif", inp,
                     (args.rollout_min + args.rollout_max) // 2)
    print(f"final checkpoint: {ckpt_dir / f'nca_{final_step:06d}.pt'}")

print(f"\nDone. Run: {run_dir}")
