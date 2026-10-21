#!/usr/bin/env python3
"""Run a trained 3SAT verifier on LARGER instances than it was trained on.

This is the headline measurement of track A.  An NCA rule is a 3x3 perception
plus a 1x1 MLP applied at every cell with the same weights everywhere, so it is
size-agnostic: the weights learned on a 6 x 26 clause matrix can be *run*
unchanged on a 12 x 52 one.  Nothing is retrained here — only the lattice grows.

Three things the script is careful about, because each one manufactures a fake
result if you get it wrong:

1.  **The rollout must grow with the instance.**  Information moves one pixel
    per step, and ``SATLayout.min_rollout() = (n + m + 1) * step`` is the depth
    at which the verdict disc can first be right — the legs (assignment down,
    OR across, flag down to the verdict) are sequentially dependent, so they
    add, and they scale with the lattice pitch rather than with n + m.
    Every evaluated size therefore gets a rollout scaled to *its own*
    ``min_rollout()``.  A fixed rollout would show accuracy collapsing with
    size for a purely geometric reason and would be misread as a failure to
    extrapolate — so that control is plotted too (faded, dashed), unless
    --skip-fixed-rollout.  Where the two curves diverge is the whole point.

2.  **Accuracies are reported against their constant-predictor baselines.**
    At the phase ratio ~95% of clauses are satisfied, so "always satisfied"
    scores ~0.95 on flags: a flag number without its baseline is misleading.
    Tier 0 at n=6, m=26 measured verdict 0.501, exact 0.501, flags 0.950.  The
    baselines are recomputed per sweep point (they move with m) and drawn as
    dotted lines in the matching colour.

3.  **A constant predictor is made visible.**  Verdict accuracy is also split
    into SAT-recall and UNSAT-recall, and the predicted SAT rate is printed
    next to the true one.

Sweeps (n and m move independently):
    --m-list         m varies, n held at the training n
    --n-list         n varies, m held at the training m
    --joint-n-list   both grow, holding the training clause/variable ratio
    --rollout-mults  accuracy vs rollout length at one size

Cost note: the rollout grows linearly with m, and so does the grid, so a point
at n=12, m=52 is ~8x the work of the training size.  Checkpoints trained with
``--r 1`` (pitch 3 px) evaluate roughly 8x cheaper at the same n, m.

Outputs (project root):
    sat_extrap.png     summary plot
    sat_extrap.json    every number behind it

Usage:
    uv run python scripts/eval_sat_extrap.py --run last
    uv run python scripts/eval_sat_extrap.py --checkpoint runs/<run>/checkpoints/nca_last.pt
    uv run python scripts/eval_sat_extrap.py --run last --samples 128 --joint-n-list 6 9 12 15
"""

import json
import random
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ncpu.nca import NeuralCA
from ncpu.sat import SATLayout, sample_instance

parser = ArgumentParser()
parser.add_argument("--checkpoint", default=None, help="path to a nca_*.pt")
parser.add_argument("--run", default=None,
                    help="run directory under runs/ (or 'last' for the newest SATVERIFY run); "
                         "uses its checkpoints/nca_last.pt")
parser.add_argument("--samples", type=int, default=64, help="instances per sweep point")
parser.add_argument("--batch", type=int, default=32, help="instances per forward pass")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--sat-frac", type=float, default=0.5)
parser.add_argument("--near-miss-frac", type=float, default=0.5)
parser.add_argument("--rollout-mult", type=float, default=2.0,
                    help="rollout = mult x min_rollout() of the *evaluated* layout "
                         "(2.0 is the middle of the default training window)")
parser.add_argument("--skip-fixed-rollout", action="store_true",
                    help="drop the control curve that keeps the training rollout length")
parser.add_argument("--m-list", type=int, nargs="*", default=None,
                    help="m sweep at the training n (default: around the training m)")
parser.add_argument("--n-list", type=int, nargs="*", default=None,
                    help="n sweep at the training m")
parser.add_argument("--joint-n-list", type=int, nargs="*", default=None,
                    help="n sweep with m scaled to hold the training ratio")
parser.add_argument("--rollout-mults", type=float, nargs="*",
                    default=[0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0])
parser.add_argument("--rollout-sweep-size", type=int, nargs=2, default=None,
                    metavar=("N", "M"),
                    help="(n, m) for the rollout-length sweep (default: the biggest joint size)")
parser.add_argument("--max-n", type=int, default=16,
                    help="refuse larger n: the ground truth is brute-forced over 2**n")
parser.add_argument("--out-dir", default=str(ROOT))
args = parser.parse_args()

# SATLayout.sample_batch draws from a `random` instance; give this script its
# own stream so it neither disturbs nor depends on any other.
RNG = random.Random(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
DEVICE = args.device

# ── checkpoint ───────────────────────────────────────────────────────────────
if args.run == "last":
    runs = sorted(d for d in (ROOT / "runs").iterdir()
                  if "SATVERIFY" in d.name and (d / "checkpoints" / "nca_last.pt").exists())
    if not runs:
        raise RuntimeError("no SATVERIFY runs with a checkpoint under runs/")
    args.run = runs[-1].name
if args.run:
    args.checkpoint = str(ROOT / "runs" / args.run / "checkpoints" / "nca_last.pt")
if not args.checkpoint:
    parser.error("give --checkpoint or --run")

ck = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
nca_config = ck["nca_config"]
train_cfg = ck["layout"]
nca = NeuralCA(**nca_config).to(DEVICE)
nca.load_state_dict(ck["nca"])
nca.eval()

TRAIN_N, TRAIN_M = train_cfg["n"], train_cfg["m"]
K = train_cfg.get("k", 3)
R = train_cfg.get("r", 2)
GAP, PAD = train_cfg.get("gap"), train_cfg.get("pad")
C = nca_config["channels"]

# The pitch must match training: the rule was learned at one lattice spacing.
train_layout = SATLayout(TRAIN_N, TRAIN_M, k=K, r=R, gap=GAP, pad=PAD)
TRAIN_DEPTH = train_layout.min_rollout()
TRAIN_ROLLOUT = int(round(args.rollout_mult * TRAIN_DEPTH))
RATIO = TRAIN_M / TRAIN_N

print(f"checkpoint {args.checkpoint}  (step {ck.get('step')})")
print(f"trained at {train_layout.describe()}")
print(f"min_rollout {TRAIN_DEPTH} -> training-scale rollout {TRAIN_ROLLOUT} "
      f"({args.rollout_mult}x); ratio {RATIO:.2f}")
print(f"device {DEVICE}   {args.samples} instances per sweep point\n")

if args.m_list is None:
    args.m_list = sorted({max(2, int(round(TRAIN_M * f))) for f in (0.5, 0.75, 1.0, 1.25, 1.5)})
if args.n_list is None:
    args.n_list = sorted({max(K, int(round(TRAIN_N * f))) for f in (0.75, 1.0, 1.5, 2.0, 2.5)})
if args.joint_n_list is None:
    args.joint_n_list = sorted({max(K, int(round(TRAIN_N * f))) for f in (1.0, 1.5, 2.0)})


# ── evaluation ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(layout, masks, rollout, n_samples):
    """Accuracy of the loaded rule on freshly sampled instances of `layout`."""
    t = dict(flag_ok=0, flag_tot=0, flag_ones=0, verdict_ok=0, exact_ok=0, total=0,
             sat_ok=0, sat_tot=0, unsat_ok=0, unsat_tot=0, pred_sat=0, all_zero=0)
    done = 0
    while done < n_samples:
        b = min(args.batch, n_samples - done)
        inp, _, bits, _ = layout.sample_batch(b, mode="verify", device=DEVICE, rng=RNG,
                                              sat_frac=args.sat_frac,
                                              near_miss_frac=args.near_miss_frac)
        x = torch.zeros(b, C, layout.H, layout.W, device=DEVICE)
        x[:, 0] = inp
        x[:, 1] = inp
        for _ in range(rollout):
            x = nca.forward(x, steps=1)[:, -1]        # step-wise: never stack the rollout
        got = SATLayout.decode_bits(x[:, 0], masks)   # (b, m + 1)
        correct = got == bits
        truth_sat = bits[:, -1].bool()

        t["flag_ok"] += int(correct[:, :-1].sum())
        t["flag_tot"] += int(correct[:, :-1].numel())
        t["flag_ones"] += int(bits[:, :-1].sum())
        t["verdict_ok"] += int(correct[:, -1].sum())
        t["exact_ok"] += int(correct.all(dim=1).sum())
        t["total"] += b
        t["sat_ok"] += int(correct[truth_sat, -1].sum())
        t["sat_tot"] += int(truth_sat.sum())
        t["unsat_ok"] += int(correct[~truth_sat, -1].sum())
        t["unsat_tot"] += int((~truth_sat).sum())
        t["pred_sat"] += int(got[:, -1].sum())
        t["all_zero"] += int((bits.sum(dim=1) == 0).sum())
        done += b

    safe = lambda a, b: (a / b) if b else float("nan")
    ones = safe(t["flag_ones"], t["flag_tot"])
    sat = safe(t["sat_tot"], t["total"])
    return {
        "flag": safe(t["flag_ok"], t["flag_tot"]),
        "verdict": safe(t["verdict_ok"], t["total"]),
        "exact": safe(t["exact_ok"], t["total"]),
        # majority-class baselines, measured on the same instances
        "flag_base": max(ones, 1 - ones),
        "verdict_base": max(sat, 1 - sat),
        "exact_base": max(sat, safe(t["all_zero"], t["total"])),
        "sat_recall": safe(t["sat_ok"], t["sat_tot"]),
        "unsat_recall": safe(t["unsat_ok"], t["unsat_tot"]),
        "true_sat_rate": sat,
        "pred_sat_rate": safe(t["pred_sat"], t["total"]),
        "rollout": rollout, "n_samples": t["total"],
    }


def feasible(n, m):
    """Cheap pre-check: does a satisfiable instance of this shape turn up fast?

    ``sample_batch`` needs ``want='sat'``, which rejection-samples; well above
    the phase ratio that can spin for 20k tries before raising.  Better to skip
    the point in a second than to stall the sweep.
    """
    try:
        sample_instance(n, m, K, RNG, want="sat", max_tries=400)
        return True
    except RuntimeError:
        return False


def point(n, m, rollout=None, mult=None, label=""):
    """One sweep point: build a fresh layout, run the same weights on it."""
    if n > args.max_n:
        print(f"  skip n={n}: above --max-n {args.max_n} (ground truth is brute-forced)")
        return None
    layout = SATLayout(n, m, k=K, r=R, gap=GAP, pad=PAD)
    depth = layout.min_rollout()
    if rollout is None:
        rollout = int(round((mult if mult is not None else args.rollout_mult) * depth))
    if not feasible(n, m):
        print(f"  skip n={n} m={m} (ratio {m / n:.2f}): satisfiable instances too rare here")
        return None
    masks = layout.output_bit_masks("verify").to(DEVICE)
    t0 = time.time()
    res = evaluate(layout, masks, rollout, args.samples)
    res.update(n=n, m=m, ratio=m / n, W=layout.W, H=layout.H, min_rollout=depth,
               mult=rollout / depth, seconds=time.time() - t0)
    under = " UNDER-DEPTH" if rollout < depth else ""
    print(f"  {label}n={n:3d} m={m:3d} ratio={m / n:4.2f} {layout.W:4d}x{layout.H:<4d} "
          f"min_rollout={depth:5d} rollout={rollout:5d} ({rollout / depth:.2f}x){under} | "
          f"flag {res['flag']:.3f}/{res['flag_base']:.3f}  "
          f"verdict {res['verdict']:.3f}/{res['verdict_base']:.3f}  "
          f"exact {res['exact']:.3f}/{res['exact_base']:.3f} | "
          f"sat-rec {res['sat_recall']:.2f} unsat-rec {res['unsat_recall']:.2f} "
          f"pred-sat {res['pred_sat_rate']:.2f}  [{res['seconds']:.0f}s]")
    return res


def sweep(pairs, title):
    """Every (n, m) at a scaled rollout, and (optionally) at the training one."""
    print(title + "   (accuracy/baseline pairs)")
    scaled, fixed = [], []
    for n, m in pairs:
        r = point(n, m, mult=args.rollout_mult)
        if r is not None:
            scaled.append(r)
        if not args.skip_fixed_rollout:
            r = point(n, m, rollout=TRAIN_ROLLOUT, label="[fixed rollout] ")
            if r is not None:
                fixed.append(r)
    print()
    return {"scaled": scaled, "fixed": fixed}


results = {}
results["m_sweep"] = sweep([(TRAIN_N, m) for m in args.m_list],
                           f"m sweep (n held at the training n={TRAIN_N}):")
results["n_sweep"] = sweep([(n, TRAIN_M) for n in args.n_list],
                           f"n sweep (m held at the training m={TRAIN_M}):")
results["joint_sweep"] = sweep([(n, max(1, int(round(RATIO * n)))) for n in args.joint_n_list],
                               f"joint sweep (both grow at the training ratio {RATIO:.2f}):")

if args.rollout_sweep_size:
    rn, rm = args.rollout_sweep_size
else:
    rn = min(max(args.joint_n_list), args.max_n)
    rm = max(1, int(round(RATIO * rn)))
sweep_layout = SATLayout(rn, rm, k=K, r=R, gap=GAP, pad=PAD)
print(f"rollout-length sweep at n={rn} m={rm} (min_rollout {sweep_layout.min_rollout()}):")
results["rollout_sweep"] = {
    "n": rn, "m": rm, "min_rollout": sweep_layout.min_rollout(),
    "points": [p for p in (point(rn, rm, mult=mu) for mu in args.rollout_mults) if p is not None],
}
print()

meta = {"checkpoint": str(args.checkpoint), "step": ck.get("step"),
        "train": {"n": TRAIN_N, "m": TRAIN_M, "ratio": RATIO, "min_rollout": TRAIN_DEPTH,
                  "rollout": TRAIN_ROLLOUT, "k": K, "r": R, "step_px": train_layout.step},
        "eval": {"samples": args.samples, "rollout_mult": args.rollout_mult,
                 "sat_frac": args.sat_frac, "near_miss_frac": args.near_miss_frac,
                 "seed": args.seed, "device": DEVICE}}

out_dir = Path(args.out_dir)
json_path = out_dir / "sat_extrap.json"
json_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
print(f"wrote {json_path}")

# ── plot ─────────────────────────────────────────────────────────────────────
# verdict is drawn last so it stays visible where it coincides with exact
METRICS = [("flag", "flag_base", "tab:blue", "flag bits"),
           ("exact", "exact_base", "tab:green", "exact instance"),
           ("verdict", "verdict_base", "tab:red", "verdict")]


def draw(ax, block, xkey, title, xlabel, train_x=None):
    for key, style, alpha, suffix in [("scaled", "-o", 1.0, ""),
                                      ("fixed", "--x", 0.4, " (fixed rollout)")]:
        pts = block.get(key, [])
        if not pts:
            continue
        xs = [p[xkey] for p in pts]
        for metric, base, colour, label in METRICS:
            ax.plot(xs, [p[metric] for p in pts], style, color=colour, alpha=alpha,
                    ms=4, lw=1.4, label=label + suffix)
            if key == "scaled":       # baselines depend on the instances, not the rollout
                ax.plot(xs, [p[base] for p in pts], ":", color=colour, alpha=0.5, lw=1.0,
                        label=label + " baseline")
    ticks = sorted({p[xkey] for key in ("scaled", "fixed") for p in block.get(key, [])})
    if ticks:
        ax.set_xticks(ticks)
    if train_x is not None:
        ax.axvline(train_x, color="k", lw=0.8, ls="-.", alpha=0.6)
        ax.annotate("trained here", (train_x, 0.03), fontsize=7, rotation=90,
                    ha="right", va="bottom", color="k", alpha=0.7)
    ax.set_ylim(0, 1.03)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel("accuracy", fontsize=9)
    ax.grid(alpha=0.15)


fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
draw(axes[0][0], results["m_sweep"], "m", f"clauses m (n = {TRAIN_N})", "m clauses", TRAIN_M)
draw(axes[0][1], results["n_sweep"], "n", f"variables n (m = {TRAIN_M})", "n variables", TRAIN_N)
draw(axes[1][0], results["joint_sweep"], "n",
     f"both, ratio {RATIO:.2f} held", "n variables (m = ratio x n)", TRAIN_N)

ax = axes[1][1]
pts = results["rollout_sweep"]["points"]
if pts:
    xs = [p["mult"] for p in pts]
    for metric, base, colour, label in METRICS:
        ax.plot(xs, [p[metric] for p in pts], "-o", color=colour, ms=4, lw=1.4, label=label)
        ax.plot(xs, [p[base] for p in pts], ":", color=colour, alpha=0.5, lw=1.0)
    ax.axvline(1.0, color="k", lw=0.8, ls="-.", alpha=0.6)
    ax.annotate("min_rollout: answer cannot arrive before here", (1.0, 0.03), fontsize=7,
                rotation=90, ha="right", va="bottom", alpha=0.7)
ax.set_ylim(0, 1.03)
ax.set_title(f"rollout length at n={rn} m={rm} "
             f"(min_rollout {results['rollout_sweep']['min_rollout']})", fontsize=10)
ax.set_xlabel("rollout / min_rollout", fontsize=9)
ax.set_ylabel("accuracy", fontsize=9)
ax.grid(alpha=0.15)

handles, labels = axes[0][0].get_legend_handles_labels()
seen, uniq = set(), []
for h, l in zip(handles, labels):
    if l not in seen:
        seen.add(l); uniq.append((h, l))
if uniq:
    fig.legend([h for h, _ in uniq], [l for _, l in uniq], loc="lower center",
               ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.005))
fig.suptitle(f"3SAT verifier: size extrapolation of one rule  "
             f"(trained n={TRAIN_N} m={TRAIN_M} r={R}, step {ck.get('step')}, "
             f"{args.samples} instances/point; dotted = constant-predictor baseline)",
             fontsize=11)
fig.tight_layout(rect=(0, 0.08, 1, 1))
png_path = out_dir / "sat_extrap.png"
fig.savefig(png_path, dpi=130)
plt.close(fig)
print(f"wrote {png_path}")
