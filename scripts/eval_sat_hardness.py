#!/usr/bin/env python3
"""Where does the exponential show up?  The 3SAT hardness curve, by search cost.

This is the baseline artifact of Track C, and it needs no NCA at all: run a
plain WalkSAT over random 3SAT instances and measure *steps to solution* as the
clause ratio m/n is swept across the satisfiability threshold at 4.267.  The
classic result is easy-hard-easy — cheap when the formula is under-constrained
and has a huge solution set, cheap again when it is so over-constrained that it
is trivially UNSAT, and expensive in between.

The point for this project: the hardness is visible in *steps*, which is the
one currency an NCA rollout also trades in.  So the same curve can be redrawn
with a trained NCA as the solver (``--solver nca``) and put side by side with
this one.  That comparison is the thesis of the track.

Four panels, all written to the project root:

  (a) median search cost vs ratio, one curve per n.  Cost is total work —
      flips (or NCA steps) summed over restarts, including the failed ones.  Filter: an instance counts
      as satisfiable iff the solver actually solved it inside the budget.  The
      RISING branch is real.  The falling branch to the right of the threshold
      is contaminated by selection bias — above 4.267 most instances are UNSAT,
      and the only ones that get solved are the easy ones — so it is drawn
      dashed and should not be read as "hard instances got easy".
  (b) solve-rate-within-budget vs ratio: the phase transition itself, getting
      sharper with n.
  (c) the honest curve: at a small n where brute force can decide
      satisfiability exactly (``sat.sample_instance(want="sat")``), the median
      cost over *provably* satisfiable instances, with no censoring and with
      bootstrap 95% CIs.  It is **easy-hard (monotone)**, and there is no peak
      to find: the falling branch of the textbook easy-hard-easy picture is
      *refutation* getting cheap above the threshold, and a stochastic local
      searcher is incomplete — it can exhibit a solution but never prove there
      is none.  Restricted to satisfiable instances its cost can only rise as
      solutions grow scarce.  A peak in this panel means too few instances per
      point, not a phase effect.  The twin axis shows the median solution count
      collapsing from hundreds to one — which is also the reason a per-pixel
      MSE target is incoherent for a solver (see ``ncpu.satsolve``).
  (d) cost vs n at a fixed ratio just below the threshold, log-y: this is the
      exponential.  Two curves, and the pair is the point: the per-instance
      budget is either MATCHED to the measured search work at each size, or
      FROZEN at the smallest size's value.  A frozen budget buys fewer and
      fewer search moves as n grows, so performance decays for purely budgetary
      reasons and reads as a failure to extrapolate; plotting both keeps the
      two effects separable.  Costs are censoring-robust quantiles over *all*
      instances (unsolved counted as infinite).

Usage:
    uv run python scripts/eval_sat_hardness.py                  # ~90 s on CPU
    uv run python scripts/eval_sat_hardness.py --instances 40 --max-flips 200000
    uv run python scripts/eval_sat_hardness.py --solver nca \
        --checkpoint runs/<run>/checkpoints/nca_last.pt --n-list 6 --scale-n-list 6 8 10
"""

import json
import statistics as st
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu import sat
from ncpu.satsolve import (
    RandomWalkSolver,
    WalkSATSolver,
    load_nca_solver,
    search_work,
)

parser = ArgumentParser()
parser.add_argument("--solver", choices=["walksat", "nca", "random"], default="walksat")
parser.add_argument("--checkpoint", default=None, help="required for --solver nca")
parser.add_argument("--device", default="cpu")

parser.add_argument("--ratios", type=float, nargs="+",
                    default=[2.0, 2.5, 3.0, 3.5, 3.8, 4.0, 4.267, 4.5, 5.0, 5.5, 6.0])
parser.add_argument("--n-list", type=int, nargs="+", default=None,
                    help="variable counts for the ratio sweep (default 50 100, or 6 for nca)")
parser.add_argument("--instances", type=int, default=20, help="instances per ratio point")

parser.add_argument("--scale-n-list", type=int, nargs="+", default=None,
                    help="variable counts for the size sweep at the threshold ratio")
parser.add_argument("--scale-instances", type=int, default=16)
parser.add_argument("--scale-ratio", type=float, default=4.1,
                    help="ratio for the size sweep.  Just BELOW the threshold on "
                         "purpose: instances there are still almost all satisfiable, "
                         "so nothing is censored and the growth in cost is a clean "
                         "measurement rather than a survivorship artifact")

parser.add_argument("--exact-n", type=int, default=14,
                    help="n for the brute-force-filtered panel; 0 disables it")
parser.add_argument("--exact-instances", type=int, default=120,
                    help="instances per point; the monotone claim needs error bars, and "
                         "at 25 the curve is noise")
parser.add_argument("--exact-max-ratio", type=float, default=5.5,
                    help="rejection sampling a satisfiable instance gets slow above this")

# solver budgets
parser.add_argument("--max-flips", type=int, default=50_000, help="walksat: flips per attempt")
parser.add_argument("--rollout", type=int, default=None,
                    help="nca: steps per attempt (default: --rollout-mult x the layout light cone)")
parser.add_argument("--rollout-mult", default="auto",
                    help="nca: rollout in SEARCH SWEEPS (multiples of min_rollout()); "
                         "'auto' sizes it from the measured walksat flip count at each "
                         "instance size so the budget stays matched as n grows")
parser.add_argument("--max-rollout", type=int, default=20_000,
                    help="nca: cap on the derived rollout")
parser.add_argument("--allow-under-depth", action="store_true",
                    help="nca: permit a rollout below the light cone (impossible, not hard)")
parser.add_argument("--visits", type=int, default=100,
                    help="random: assignments visited per attempt")
parser.add_argument("--scale-budget", choices=["matched", "fixed", "both"], default="both",
                    help="size sweep: 'matched' scales the per-instance budget with the "
                         "measured search work, 'fixed' freezes it at the smallest size "
                         "(the confounded control), 'both' plots the two together")
parser.add_argument("--restarts", type=int, default=1)
parser.add_argument("--noise", type=float, default=0.5,
                    help="walksat: random-walk probability; nca: assignment seed noise scale")
parser.add_argument("--check-every", type=int, default=4, help="nca: decode every k steps")

parser.add_argument("--k-lit", type=int, default=3, help="literals per clause")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="sat_hardness_curve.png")
parser.add_argument("--json", default="sat_hardness_curve.json")
args = parser.parse_args()

if args.n_list is None:
    args.n_list = [6] if args.solver == "nca" else [50, 100]
if args.scale_n_list is None:
    args.scale_n_list = ([6, 8, 10, 12] if args.solver == "nca"
                         else [20, 40, 60, 80, 100, 120, 140, 160])

import random  # noqa: E402  (after argparse so the seed is known)

rng = random.Random(args.seed)


# ── the pluggable solver ─────────────────────────────────────────────────────

def parse_mult(value):
    return value if value == "auto" else float(value)


def base_solver(**over):
    if args.solver == "walksat":
        kw = dict(max_flips=args.max_flips, restarts=args.restarts, noise=args.noise,
                  seed=args.seed)
        kw.update(over)
        return WalkSATSolver(**kw)
    if args.solver == "random":
        kw = dict(visits=args.visits, restarts=args.restarts, mode="walk", seed=args.seed)
        kw.update(over)
        return RandomWalkSolver(**kw)
    if not args.checkpoint:
        raise SystemExit("--solver nca needs --checkpoint")
    kw = dict(rollout=args.rollout, rollout_mult=parse_mult(args.rollout_mult),
              restarts=args.restarts, noise_scale=args.noise,
              check_every=args.check_every, seed=args.seed,
              allow_under_depth=args.allow_under_depth)
    kw.update(over)
    return load_nca_solver(args.checkpoint, device=args.device, **kw)


solver = base_solver()
if args.solver == "walksat":
    budget = args.max_flips * args.restarts
elif args.solver == "random":
    budget = args.visits * args.restarts
else:
    budget = args.rollout * args.restarts if args.rollout else None
print(f"solver: {solver.describe()}   budget/instance = {budget} {solver.unit}")


def sized_solver(n, m, mode):
    """A solver configured for this instance size under a budget policy.

    ``matched`` scales the per-instance budget with the measured search work at
    that size — a fixed budget is the classic confound: search cost grows with
    n, so a frozen budget buys fewer and fewer moves and the resulting decline
    reads as a failure to extrapolate when it is really a shrinking allowance.
    ``fixed`` deliberately reproduces that control so the two effects can be
    told apart in the same figure.
    """
    n0, m0 = SCALE_REF
    ratio = search_work(n, m, args.k_lit) / search_work(n0, m0, args.k_lit)
    if args.solver == "walksat":
        cap = args.max_flips if mode == "fixed" else int(round(args.max_flips * ratio))
        return base_solver(max_flips=max(1, cap))
    if args.solver == "random":
        v = args.visits if mode == "fixed" else int(round(args.visits * ratio))
        return base_solver(visits=max(1, v))
    if mode == "fixed":
        return base_solver(rollout=FIXED_ROLLOUT, rollout_mult=parse_mult(args.rollout_mult))
    return base_solver(rollout=args.rollout, rollout_mult=parse_mult(args.rollout_mult))


def censored_quantile(costs, solved, q):
    """The q-quantile of cost over ALL instances, unsolved ones counted as +inf.

    This is the statistic to use once instances start hitting the budget.  The
    "median over solved instances" is biased *downwards* exactly where the
    problem is hardest — at the threshold only the lucky instances finish, so
    the average of the survivors looks easy.  A quantile below the solve rate is
    immune: its position in the sorted list does not depend on how big the
    censored values would have been.  Returns nan when q lands in the censored
    tail, which is the honest answer ("more than the budget").
    """
    vals = sorted(c if s else float("inf") for c, s in zip(costs, solved))
    if not vals:
        return float("nan")
    idx = min(max(int(np.ceil(q * len(vals))) - 1, 0), len(vals) - 1)
    v = vals[idx]
    return float(v) if np.isfinite(v) else float("nan")


def summarise(costs, solved, extra=None):
    """Per-point statistics.  ``median`` is over the SOLVED instances only —
    convenient but downward-biased once instances are censored, so ``q25`` (a
    censoring-robust quantile over all instances) is what the scaling panel
    uses; ``solve_rate`` carries the rest of the information."""
    hit = [c for c, s in zip(costs, solved) if s]
    row = {
        "instances": len(costs),
        "solved": int(sum(solved)),
        "solve_rate": float(np.mean(solved)) if costs else float("nan"),
        "median": float(st.median(hit)) if hit else float("nan"),
        "mean": float(st.mean(hit)) if hit else float("nan"),
        "p90": float(np.percentile(hit, 90)) if hit else float("nan"),
        "max": float(max(hit)) if hit else float("nan"),
        "q25": censored_quantile(costs, solved, 0.25),
        "q50": censored_quantile(costs, solved, 0.50),
        "budget": budget,
    }
    row.update(extra or {})
    return row


def bootstrap_median_ci(costs, solved, reps=400, seed=0):
    """95% CI of the median over solved instances — the error bars the monotone
    claim needs.  Returns (nan, nan) with too few solves."""
    hit = np.array([c for c, s in zip(costs, solved) if s], dtype=float)
    if len(hit) < 5:
        return float("nan"), float("nan")
    gen = np.random.default_rng(seed)
    meds = [float(np.median(gen.choice(hit, size=len(hit), replace=True))) for _ in range(reps)]
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def run_point(n, m, count, want=None, use_solver=None):
    """Solve ``count`` instances at (n, m).  ``want`` uses the exact brute-force
    filter from ``sat.sample_instance``; ``None`` takes uniform random ones."""
    sol = use_solver or solver
    costs, solved, restarts, nsol = [], [], [], []
    any_frame, distinct, sweeps = [], [], []
    for _ in range(count):
        if want is None:
            formula = sat.random_ksat(n, m, args.k_lit, rng=rng)
        else:
            formula, sols = sat.sample_instance(n, m, args.k_lit, rng=rng, want=want)
            nsol.append(len(sols))
        out = sol(formula, n)
        costs.append(out.work)
        solved.append(bool(out.solved))          # the RETURNED answer, not any frame
        restarts.append(out.restarts)
        if out.meta.get("any") is not None:
            any_frame.append(bool(any(out.meta["any"])))
        if out.meta.get("distinct"):
            distinct.append(float(np.median(out.meta["distinct"])))
        if out.meta.get("sweeps"):
            sweeps.append(float(out.meta["sweeps"]))
    extra = {"median_restarts": float(st.median(restarts))}
    if nsol:
        extra["median_solutions"] = float(st.median(nsol))
    if any_frame:                                 # diagnostic, never the headline
        extra["any_frame_rate"] = float(np.mean(any_frame))
    if distinct:
        extra["median_distinct"] = float(np.median(distinct))
    if sweeps:
        extra["median_sweeps"] = float(np.median(sweeps))
    lo, hi = bootstrap_median_ci(costs, solved, seed=args.seed)
    extra["ci_lo"], extra["ci_hi"] = lo, hi
    return summarise(costs, solved, extra)


results = {
    "config": vars(args) | {"solver_desc": solver.describe(), "unit": solver.unit},
    "ratio_sweep": {}, "exact_sweep": {}, "scale_sweep": {},
}
t_start = time.time()

# ── (a, b) ratio sweep, uniform instances, solver-as-filter ──────────────────
for n in args.n_list:
    rows = []
    bar = tqdm(args.ratios, ncols=100, desc=f"ratio sweep n={n}")
    for ratio in bar:
        m = int(round(n * ratio))
        row = run_point(n, m, args.instances) | {"ratio": ratio, "m": m, "n": n}
        rows.append(row)
        bar.set_postfix(solved=f"{row['solved']}/{row['instances']}",
                        med=f"{row['median']:.0f}")
    results["ratio_sweep"][str(n)] = rows

# ── (c) exact brute-force-filtered sweep at small n ──────────────────────────
if args.exact_n:
    n = args.exact_n
    rows = []
    ratios = [r for r in args.ratios if r <= args.exact_max_ratio]
    bar = tqdm(ratios, ncols=100, desc=f"exact sweep n={n}")
    for ratio in bar:
        m = int(round(n * ratio))
        row = run_point(n, m, args.exact_instances, want="sat") | {
            "ratio": ratio, "m": m, "n": n}
        rows.append(row)
        bar.set_postfix(med=f"{row['median']:.0f}", sols=f"{row['median_solutions']:.0f}")
    results["exact_sweep"][str(n)] = rows

# ── (d) size sweep: matched vs fixed budget ──────────────────────────────────
# SCALE_REF is the smallest size in the sweep; the "fixed" control freezes every
# solver's allowance at that size's value, which is exactly the confound to
# expose.  "matched" lets the allowance grow with the measured search work.
SCALE_REF = (args.scale_n_list[0], int(round(args.scale_n_list[0] * args.scale_ratio)))
FIXED_ROLLOUT = None
if args.solver == "nca":
    ref_layout = solver.layout_for(*SCALE_REF)
    FIXED_ROLLOUT = base_solver().steps_for(ref_layout)
    print(f"fixed-budget control: {FIXED_ROLLOUT} steps "
          f"({FIXED_ROLLOUT / ref_layout.min_rollout():.1f} sweeps at n={SCALE_REF[0]})")

modes = ["matched", "fixed"] if args.scale_budget == "both" else [args.scale_budget]
results["scale_sweep"] = {}
for mode in modes:
    rows = []
    bar = tqdm(args.scale_n_list, ncols=100, desc=f"size sweep ({mode}) @ {args.scale_ratio:.3f}")
    for n in bar:
        m = int(round(n * args.scale_ratio))
        row = run_point(n, m, args.scale_instances, use_solver=sized_solver(n, m, mode))
        row |= {"ratio": args.scale_ratio, "m": m, "n": n, "mode": mode}
        rows.append(row)
        bar.set_postfix(solved=f"{row['solved']}/{row['instances']}",
                        med=f"{row['median']:.0f}")
    results["scale_sweep"][mode] = rows

results["seconds"] = time.time() - t_start

# ── plot ─────────────────────────────────────────────────────────────────────
def log_if_possible(ax, *series):
    """Log-scale only when there is something positive to show.

    A solver that solves nothing (an untrained NCA, or a random baseline that
    finds the answer on its first draw and costs 0) leaves an axis with no
    positive values, and matplotlib refuses to log-scale that."""
    values = [v for s in series for v in s if np.isfinite(v) and v > 0]
    if values:
        ax.set_yscale("log")


unit = solver.unit
fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
(ax_a, ax_b), (ax_c, ax_d) = axes
colors = plt.cm.viridis(np.linspace(0.15, 0.8, max(2, len(args.n_list))))

for c, (key, rows) in zip(colors, results["ratio_sweep"].items()):
    x = [r["ratio"] for r in rows]
    y = [r["median"] for r in rows]
    rate = [r["solve_rate"] for r in rows]
    left = [i for i, r in enumerate(x) if r <= sat.PHASE_RATIO]
    right = [i for i, r in enumerate(x) if r >= sat.PHASE_RATIO]
    ax_a.plot([x[i] for i in left], [y[i] for i in left], "-o", ms=4, color=c,
              label=f"n={key}")
    ax_a.plot([x[i] for i in right], [y[i] for i in right], "--o", ms=4, color=c,
              alpha=0.55, mfc="none")
    ax_b.plot(x, rate, "-o", ms=4, color=c, label=f"n={key}")

for ax, title in ((ax_a, f"(a) median search cost in {unit} (solved instances only)"),
                  (ax_b, "(b) solve rate within budget")):
    ax.axvline(sat.PHASE_RATIO, color="crimson", lw=1, ls=":", zorder=0)
    ax.set_xlabel("clause ratio  m/n")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
log_if_possible(ax_a, *[[r["median"] for r in rows]
                        for rows in results["ratio_sweep"].values()])
ax_a.set_ylabel(f"{unit}, all restarts (median)")
ax_a.text(sat.PHASE_RATIO + 0.05, ax_a.get_ylim()[0] * 1.4, "  dashed: censored,\n  "
          "selection-biased", fontsize=7, color="0.35")
ax_b.set_ylabel("fraction solved")
ax_b.set_ylim(-0.03, 1.03)

if results["exact_sweep"]:
    for key, rows in results["exact_sweep"].items():
        x = [r["ratio"] for r in rows]
        med = np.array([r["median"] for r in rows])
        lo = np.array([r["ci_lo"] for r in rows])
        hi = np.array([r["ci_hi"] for r in rows])
        err = np.vstack([np.maximum(med - lo, 0), np.maximum(hi - med, 0)])
        ax_c.errorbar(x, med, yerr=err, fmt="-o", ms=4, lw=1.2, capsize=2.5,
                      color="#2a6f97", label=f"median {unit}, n={key} (95% CI)")
        ax2 = ax_c.twinx()
        ax2.plot(x, [r["median_solutions"] for r in rows], "-s", ms=3.5, color="#bc4749",
                 alpha=0.8)
        ax2.set_yscale("log")
        ax2.set_ylabel("median # satisfying assignments", color="#bc4749", fontsize=9)
        ax2.tick_params(axis="y", colors="#bc4749")
    log_if_possible(ax_c, *[[r["median"] for r in rows]
                            for rows in results["exact_sweep"].values()])
    ax_c.legend(fontsize=8, loc="upper left")
ax_c.axvline(sat.PHASE_RATIO, color="crimson", lw=1, ls=":", zorder=0)
ax_c.set_xlabel("clause ratio  m/n")
ax_c.set_ylabel(f"{unit} (median)")
ax_c.set_title("(c) easy-hard (MONOTONE), exact SAT filter — no peak, and none is\n"
               "expected: an incomplete searcher cannot refute", fontsize=9)

styles = {"matched": ("-o", "#4a4e69", 1.0), "fixed": ("--s", "#9a8c98", 0.8)}
results["scale_fit"] = {}
ax_d2 = ax_d.twinx()
for mode, rows in results["scale_sweep"].items():
    fmt, color, alpha = styles.get(mode, ("-o", "#4a4e69", 1.0))
    ns = [r["n"] for r in rows]
    med = [r["q50"] for r in rows]          # censoring-robust; see censored_quantile
    lab = ("budget matched to search work" if mode == "matched"
           else "budget frozen at the smallest size (control)")
    ax_d.plot(ns, med, fmt, ms=5, color=color, alpha=alpha, label=lab)
    ax_d2.plot(ns, [r["solve_rate"] for r in rows], ":" if mode == "fixed" else "-",
               marker="s", ms=3.5, color="#bc4749", alpha=0.55 if mode == "fixed" else 0.85)
    okp = [(n, y) for n, y in zip(ns, med) if np.isfinite(y) and y > 0]
    if len(okp) >= 2:
        xs = np.array([p[0] for p in okp], dtype=float)
        ys = np.log(np.array([p[1] for p in okp], dtype=float))
        slope, intercept = np.polyfit(xs, ys, 1)
        if mode == "matched":
            ax_d.plot(xs, np.exp(intercept + slope * xs), "--", lw=1, color="crimson",
                      label=f"exp fit: x{np.exp(slope):.3f} per variable")
        results["scale_fit"][mode] = {"log_slope": float(slope),
                                      "per_variable": float(np.exp(slope))}
ax_d2.set_ylim(-0.03, 1.03)
ax_d2.set_ylabel("solve rate within budget", color="#bc4749", fontsize=9)
ax_d2.tick_params(axis="y", colors="#bc4749")
log_if_possible(ax_d, *[[r["q50"] for r in rows]
                        for rows in results["scale_sweep"].values()])
ax_d.set_xlabel("variables n")
ax_d.set_ylabel(f"{unit}")
ax_d.set_title(f"(d) cost vs n at m/n={args.scale_ratio:.3f} (just below the threshold)",
               fontsize=10)
ax_d.legend(fontsize=7)

for ax in axes.ravel():
    ax.grid(alpha=0.25, lw=0.5)
fig.suptitle(f"3SAT hardness by search cost — solver: {solver.describe()}", fontsize=11)
fig.tight_layout()
fig.savefig(args.out, dpi=130)
plt.close(fig)

with open(args.json, "w") as f:
    json.dump(results, f, indent=2)

# ── console summary ──────────────────────────────────────────────────────────
print(f"\nratio sweep ({unit}, median over solved):")
for key, rows in results["ratio_sweep"].items():
    print(f"  n={key}")
    for r in rows:
        print(f"    m/n={r['ratio']:5.3f}  m={r['m']:4d}  solved={r['solved']:3d}/"
              f"{r['instances']:<3d}  median={r['median']:9.1f}  p90={r['p90']:9.1f}  "
              f"q25={r['q25']:9.1f}")
for key, rows in results["exact_sweep"].items():
    print(f"exact sweep n={key} (provably satisfiable instances, no censoring):")
    for r in rows:
        print(f"    m/n={r['ratio']:5.3f}  m={r['m']:4d}  median={r['median']:8.1f}  "
              f"[{r['ci_lo']:6.1f}, {r['ci_hi']:6.1f}]  p90={r['p90']:8.1f}  "
              f"median #solutions={r['median_solutions']:8.1f}")
    print("    an incomplete local searcher cannot refute, so on satisfiable instances "
          "this\n    curve can only rise: easy-hard, not easy-hard-easy.")
for mode, rows in results["scale_sweep"].items():
    print(f"size sweep at m/n={args.scale_ratio:.3f}  [{mode} budget]:")
    for r in rows:
        extra = ""
        if "median_sweeps" in r:
            extra = f"  sweeps={r['median_sweeps']:.1f}"
        if "any_frame_rate" in r:
            extra += (f"  any-frame={r['any_frame_rate']:.2f}"
                      f"  distinct={r.get('median_distinct', float('nan')):.0f}")
        print(f"    n={r['n']:4d}  m={r['m']:4d}  solved={r['solved']:3d}/"
              f"{r['instances']:<3d}  q25={r['q25']:9.1f}  q50={r['q50']:9.1f}{extra}")
    if mode in results.get("scale_fit", {}):
        print(f"    exponential fit: x{results['scale_fit'][mode]['per_variable']:.3f} "
              f"{unit} per variable")
print(f"\n{results['seconds']:.1f}s — wrote {args.out} and {args.json}")
