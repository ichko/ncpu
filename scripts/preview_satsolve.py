#!/usr/bin/env python3
"""Look at the solve task, and check the loss, with no model involved.

Two jobs:

1. **Render the task.**  The solver's input is the clause matrix with the
   assignment row blank; the answer is the assignment row filled in.  The GIF
   animates a real WalkSAT search *in the encoding* — each flip redraws one
   disc — which is the behaviour the NCA is being asked to learn.  If this does
   not look like something a local rule could do, nothing downstream will work.

2. **Validate the energy before spending a GPU on it.**  The satisfaction
   energy in ``ncpu.satsolve`` is a surrogate for "is this formula satisfied",
   and a surrogate can be checked without training:

   * does it fall monotonically as WalkSAT approaches a solution?
   * does it rank assignments the same way the exact unsatisfied-clause count
     does (rank correlation)?
   * how hard does it punish ONE broken clause, relative to the spread of the
     energy?  This is the number that separates the aggregation choices: a
     plain mean over clauses barely notices a single abandoned clause, which is
     precisely the failure mode the log-domain aggregate exists to prevent.

Outputs (project root):
    satsolve_preview.png    task screens, masks, energy curves, aggregate check
    satsolve_preview.gif    a WalkSAT search animated on the clause matrix

Usage:
    uv run python scripts/preview_satsolve.py
    uv run python scripts/preview_satsolve.py --n 8 --m 34 --noise-scale 0.5
"""

import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu import sat
from ncpu.gif import write_gif
from ncpu.sat import SATLayout
from ncpu.satsolve import (
    AGGREGATES,
    RandomWalkSolver,
    aggregate_energy,
    assignment_union_mask,
    clause_satisfaction,
    clause_tensors,
    count_unsat,
    read_discs,
    seed_noise,
    walksat_once,
)

parser = ArgumentParser()
parser.add_argument("--n", type=int, default=6)
parser.add_argument("--m", type=int, default=26)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--r", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--noise-scale", type=float, default=0.5)
parser.add_argument("--beta", type=float, default=4.0)
parser.add_argument("--tau", type=float, default=8.0)
parser.add_argument("--rollout-mult", type=float, default=3.0)
parser.add_argument("--gap-instances", type=int, default=32,
                    help="instances used for the aggregate check")
parser.add_argument("--baseline-instances", type=int, default=60,
                    help="instances for the any-frame degenerate-baseline curve")
parser.add_argument("--min-flips", type=int, default=12,
                    help="keep resampling until the animated search takes at least this many flips")
parser.add_argument("--soft-scale", type=float, default=0.25,
                    help="disc magnitude for the 'undecided' regime check")
parser.add_argument("--u-ref", type=int, default=5,
                    help="reference number of unsatisfied clauses for normalising energies")
parser.add_argument("--fps", type=int, default=6)
parser.add_argument("--zoom", type=int, default=3)
parser.add_argument("--out-dir", default=".")
args = parser.parse_args()

rng = random.Random(args.seed)
torch.manual_seed(args.seed)

layout = SATLayout(n_vars=args.n, m_clauses=args.m, k=args.k, r=args.r)
print(layout.describe())
print(f"min_rollout (light cone, pixels) = {layout.min_rollout()} steps  ->  "
      f"suggested solver rollout {int(args.rollout_mult * layout.min_rollout())} "
      f"({args.rollout_mult:g} sweeps)")

asn_masks = layout.assignment_masks()
asn_union = assignment_union_mask(layout)


def energies(frames, var_idx, sign):
    """(T, H, W) screens -> {aggregate: (T,)} plus the exact unsat count."""
    a = read_discs(frames.unsqueeze(0), asn_masks)             # (1, T, n)
    clause_s = clause_satisfaction(a, var_idx, sign, beta=args.beta)
    out = {name: aggregate_energy(clause_s, name, args.tau)[0].numpy() for name in AGGREGATES}
    return out, a


# ── a real WalkSAT search, rendered in the encoding ──────────────────────────
# Keep looking until the search takes a few flips: a two-flip solve makes a
# pretty but uninformative animation, and the point of the GIF is to show what
# the NCA has to reproduce — a trajectory, not a lucky guess.
for _attempt in range(200):
    formula, sols = sat.sample_instance(args.n, args.m, args.k, rng, want="sat")
    trace = []
    solved, final, flips, _u = walksat_once(formula, args.n, max_flips=400, rng=rng,
                                            trace=trace)
    if solved and flips >= args.min_flips:
        break
print(f"\ninstance: {len(sols)} solution(s) of {2 ** args.n};  walksat solved={solved} "
      f"in {flips} flips ({len(trace)} states)")

frames = np.stack([layout.input_screen(formula, a) for a in trace])
blank = layout.input_screen(formula)
seeded = seed_noise(torch.from_numpy(blank).clone(), asn_union, args.noise_scale)
target = layout.target_screen(formula, trace[-1])

var_idx, sign = clause_tensors([formula])
curves, _a = energies(torch.from_numpy(frames), var_idx, sign)
true_unsat = np.array([count_unsat(formula, a) for a in trace])

# rank agreement between each surrogate and the exact count
def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


# ── the aggregation check: what does ONE broken clause cost? ─────────────────
# For each instance, evaluate EVERY assignment (the cube is small at these n),
# bucket them by the exact number of unsatisfied clauses, and average the energy
# per bucket.  Normalise each curve so that u=0 is 0 and u=U_REF is 1.  Then the
# value at u=1 is directly comparable across aggregates: it is the share of the
# whole energy range that a SINGLE abandoned clause already costs.  A mean over
# clauses is linear in u by construction, so it charges roughly 1/U_REF for the
# first broken clause — that is exactly how a model gets away with satisfying 25
# clauses out of 26 forever.
U_REF = args.u_ref


def assignment_table(formula, max_enum=4096, scale=1.0):
    """Every assignment (or a sample), its exact unsat count, and its energies.

    ``scale`` shrinks the assignment discs towards the background without
    changing their signs, which is what an UNDECIDED state looks like — the
    discs early in a rollout, straight after the noise seed.  The thresholded
    assignment (and therefore the exact unsat count) is unchanged, so any
    difference between aggregates at ``scale < 1`` is purely about how they
    behave in the regime where training actually starts.
    """
    n = args.n
    if 2 ** n <= max_enum:
        assignments = [tuple((i >> (n - 1 - v)) & 1 for v in range(n)) for i in range(2 ** n)]
    else:
        assignments = [tuple(probe_rng.randint(0, 1) for _ in range(n)) for _ in range(max_enum)]
    screens = np.stack([layout.input_screen(formula, a) for a in assignments])
    if scale != 1.0:
        screens = screens * (1.0 - (1.0 - scale) * asn_union.numpy())
    e, _ = energies(torch.from_numpy(screens), *clause_tensors([formula]))
    u = np.array([count_unsat(formula, a) for a in assignments])
    return e, u


probe_rng = random.Random(args.seed + 1)
SCALES = (1.0, args.soft_scale)
buckets = {sc: {name: {u: [] for u in range(U_REF + 1)} for name in AGGREGATES}
           for sc in SCALES}
rhos = {sc: {name: [] for name in AGGREGATES} for sc in SCALES}
for _ in range(args.gap_instances):
    f, _s = sat.sample_instance(args.n, args.m, args.k, probe_rng, want="sat")
    for sc in SCALES:
        e, u = assignment_table(f, scale=sc)
        if not all((u == want).any() for want in (0, U_REF)):
            continue
        for name in AGGREGATES:
            rhos[sc][name].append(spearman(e[name], u))
            base = e[name][u == 0].mean()
            top = e[name][u == U_REF].mean()
            if top - base < 1e-9:
                continue
            for want in range(U_REF + 1):
                sel = u == want
                if sel.any():
                    buckets[sc][name][want].append((e[name][sel].mean() - base) / (top - base))

print(f"\nrank agreement with the exact unsat count over ALL {2 ** args.n} assignments "
      f"({len(rhos[1.0]['mean'])} instances):")
print(f"  {'aggregate':<10s} committed discs   undecided discs (x{args.soft_scale:g})")
for name in AGGREGATES:
    print(f"  {name:<10s} {np.mean(rhos[1.0][name]):+.3f}            "
          f"{np.mean(rhos[args.soft_scale][name]):+.3f}")

curve = {sc: {} for sc in SCALES}
for sc in SCALES:
    for name in AGGREGATES:
        curve[sc][name] = [float(np.mean(buckets[sc][name][u])) if buckets[sc][name][u]
                           else float("nan") for u in range(U_REF + 1)]

for sc in SCALES:
    label = "committed discs (+-1)" if sc == 1.0 else f"undecided discs (x{sc:g})"
    print(f"\nnormalised energy by number of unsatisfied clauses — {label} "
          f"(0 -> 0, {U_REF} -> 1):")
    print("  aggregate " + "".join(f"  u={u:<6d}" for u in range(U_REF + 1)))
    for name in AGGREGATES:
        print(f"  {name:9s} " + "".join(f"  {v:<7.3f}" for v in curve[sc][name]))

print(f"""
reading of the above:
  * with COMMITTED discs, logprod is linear in the unsat count and coincides with
    mean ({curve[1.0]['logprod'][1]:.2f} vs {curve[1.0]['mean'][1]:.2f} for one broken clause).  That is not a
    defect: -log of a saturated clause is a constant, so logprod IS the WalkSAT
    objective (number of unsatisfied clauses) up to a factor of m.
  * the aggregates separate where training actually lives — undecided discs.
    There -log s diverges as a clause collapses, so logprod's gradient on a
    failing clause is 1/(m s_i) instead of the mean's constant 1/m: the worse a
    clause, the harder it pulls, and no amount of easy clauses buys it off.
  * softmin puts {curve[1.0]['softmin'][1]:.2f} of the range on the FIRST broken clause — the sharpest
    worst-clause pressure available, at the cost of a nearly flat gradient on
    every other clause.  Use it as the ablation, not the default.
  * min is rank-blind beyond one clause (rho {np.mean(rhos[1.0]['min']):+.2f}): it cannot tell one
    unsatisfied clause from five, so it carries almost no training signal.
""")

# ── what an any-frame metric is worth ────────────────────────────────────────
# Scoring a rollout by "did ANY decoded frame satisfy the formula" has no fixed
# floor.  A random walk that visits enough distinct assignments passes through a
# solution eventually, so the metric climbs to 1.0 while the walker learns
# nothing.  The floor for that metric is a visit-matched random walk — measured
# here — and the metric with a fixed floor is the assignment the rollout
# actually RETURNS, whose baseline is a single random draw.
base_rng = random.Random(args.seed + 2)
base_formulas = [sat.sample_instance(args.n, args.m, args.k, base_rng, want="sat")[0]
                 for _ in range(args.baseline_instances)]
visit_grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
any_rates, final_rates, seen_counts = [], [], []
for v in visit_grid:
    walker = RandomWalkSolver(visits=v, restarts=1, mode="walk", seed=args.seed)
    outs = [walker(f, args.n) for f in base_formulas]
    any_rates.append(float(np.mean([any(o.meta["any"]) for o in outs])))
    final_rates.append(float(np.mean([o.solved for o in outs])))
    seen_counts.append(float(np.mean([o.meta["distinct"][0] for o in outs])))

print("\nthe any-frame trap — a random walk that learns nothing, scored two ways:")
print("  visits  distinct   any-frame   returned-answer")
for v, a, f, d in zip(visit_grid, any_rates, final_rates, seen_counts):
    print(f"  {v:6d}  {d:8.1f}   {a:9.3f}   {f:15.3f}")
print("  any-frame runs away to 1.000; the returned answer stays at the "
      "single-draw floor.\n  Report the returned answer, and score any-frame "
      "against a visit-matched walk.")

# ── figure ───────────────────────────────────────────────────────────────────
out_dir = Path(args.out_dir)
write_gif(out_dir / "satsolve_preview.gif", frames, fps=args.fps, vmin=-1, vmax=1,
          zoom=args.zoom)

fig = plt.figure(figsize=(14, 9.5))
grid = fig.add_gridspec(3, 5, height_ratios=[1.25, 1.0, 1.0])

panels = [("input: assignment row blank", blank, -1, 1),
          (f"seeded (noise {args.noise_scale})", seeded.numpy(), -1, 1),
          ("walksat solution + flags", target, -1, 1),
          ("assignment discs (the answer)", asn_union.numpy(), 0, 1),
          ("output discs (writable)", layout.output_bit_masks("solve").sum(0).clamp(0, 1).numpy(),
           0, 1)]
for j, (name, data, lo, hi) in enumerate(panels):
    ax = fig.add_subplot(grid[0, j])
    ax.imshow(data, cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(name, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

ax = fig.add_subplot(grid[1, :3])
for name in AGGREGATES:
    v = curves[name]
    span = v.max() - v.min()
    ax.plot((v - v.min()) / (span if span > 1e-9 else 1.0), lw=1.1,
            label=f"{name} (rho={spearman(v, true_unsat):+.2f})")
ax2 = ax.twinx()
ax2.plot(true_unsat, color="0.3", lw=1.4, ls="--")
ax2.set_ylabel("exact unsat clauses", fontsize=8, color="0.3")
ax.set_xlabel("walksat flip"); ax.set_ylabel("energy (min-max normalised)", fontsize=8)
ax.set_title("energy surrogates along a real search (dashed grey = ground truth)",
             fontsize=9)
ax.legend(fontsize=7, loc="upper right")

ax = fig.add_subplot(grid[1, 3:])
us = np.arange(U_REF + 1)
for name in AGGREGATES:
    ax.plot(us, curve[1.0][name], "-o", ms=4, lw=1.2,
            label=f"{name}  (u=1: {curve[1.0][name][1]:.2f})")
ax.plot(us, us / U_REF, ":", color="0.5", lw=1, label="linear reference")
ax.set_xlabel("unsatisfied clauses"); ax.set_ylabel("normalised energy", fontsize=8)
ax.set_title("what one abandoned clause costs, by aggregate", fontsize=9)
ax.legend(fontsize=7)
ax.tick_params(labelsize=8)

ax = fig.add_subplot(grid[2, :])
ax.plot(visit_grid, any_rates, "-o", ms=4, color="#bc4749",
        label="any decoded frame satisfied (no fixed floor)")
ax.plot(visit_grid, final_rates, "-o", ms=4, color="#2a6f97",
        label="assignment actually returned (floor = one random draw)")
ax.set_xscale("log", base=2)
ax.set_xlabel("assignments visited by a random walk that learns nothing")
ax.set_ylabel("solve rate", fontsize=8)
ax.set_ylim(-0.03, 1.03)
ax.set_title("why the solve-rate metric is the FINAL assignment: an any-frame "
             "metric rewards jitter", fontsize=9)
ax.legend(fontsize=8)

fig.suptitle(layout.describe(), fontsize=9)
fig.tight_layout()
fig.savefig(out_dir / "satsolve_preview.png", dpi=130)
plt.close(fig)

print(f"\nwrote {out_dir / 'satsolve_preview.png'} and {out_dir / 'satsolve_preview.gif'}")
