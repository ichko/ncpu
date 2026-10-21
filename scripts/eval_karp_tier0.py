#!/usr/bin/env python3
"""Tier 0 well-posedness gate for the Karp tracks — no model, no GPU.

Per docs/karp_eval_protocol.md, a track does not get GPU time until its task
survives this. Everything here is cheap, and it catches the failures that would
otherwise cost hours to discover:

  1. degenerate baseline  — what does the best input-INDEPENDENT predictor score?
                            every later model number is meaningless without it.
  2. shortcut probes      — can a LINEAR model on the raw input pixels, or a
                            trivial classifier on summary statistics, predict the
                            answer? the verdict is an AND of ORs of products, so
                            a linear probe should sit at baseline. if it does
                            not, the answer is leaking and the NCA result would
                            not measure the intended computation.
  3. light cone           — can the answer physically reach the output disc in T
                            steps? if not the task is impossible, not hard, and
                            must never be logged as a learning failure.
  4. target well-posedness— is the target a deterministic function of the input?
                            for solve mode this is the multiple-solutions trap.

Usage:
    uv run python scripts/eval_karp_tier0.py
    uv run python scripts/eval_karp_tier0.py --n 6 --m 12 --samples 4000
"""

import json
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.linear_model import LogisticRegression

from ncpu.sat import (
    PHASE_RATIO,
    SATLayout,
    clause_flags,
    evaluate,
    n_solutions,
    sample_instance,
)

parser = ArgumentParser()
parser.add_argument("--n", type=int, default=6)
parser.add_argument("--m", type=int, default=26)   # ratio 4.33; m=12 fails this gate's own item 4
parser.add_argument("--samples", type=int, default=3000)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", default="results/karp/tier0.json")
args = parser.parse_args()

rng = random.Random(args.seed)
layout = SATLayout(n_vars=args.n, m_clauses=args.m)
print(layout.describe())
print()

report = {"n": args.n, "m": args.m, "samples": args.samples,
          "ratio": args.m / args.n, "phase_ratio": PHASE_RATIO}

# ── gather a verify-mode sample (uses the layout's own balancing) ────────────
inp, tgt, bits, meta = layout.sample_batch(args.samples, mode="verify", rng=rng)
X = inp.reshape(args.samples, -1).numpy()
bits = bits.numpy()
flags = bits[:, :-1]
verdict = bits[:, -1]

# ── 1. degenerate baselines ──────────────────────────────────────────────────
# best constant predictor = always answer the majority class.
maj_verdict = max(verdict.mean(), 1 - verdict.mean())
maj_flag = max(flags.mean(), 1 - flags.mean())
# an input-independent predictor gets a whole instance right only if it happens
# to match every bit; the best such is the per-bit majority pattern.
# For EXACT-instance accuracy the best constant predictor is the MODAL bit
# pattern, not the per-bit majority pattern: those optimise different targets.
# Assembling it bit-by-bit is also unstable here -- the verdict is balanced at
# ~0.50 by design and all-flags-1 <=> verdict-1 perfectly, so a hair of sampling
# noise flips the verdict bit and collapses the baseline 0.52 -> 0.000
# discontinuously. The mode is what "best input-independent predictor" means.
maj_pattern = (bits.mean(0) > 0.5).astype(int)
patterns, counts = np.unique(bits, axis=0, return_counts=True)
mode_pattern = patterns[counts.argmax()]
const_exact = float(counts.max() / len(bits))
maj_exact = float((bits == maj_pattern).all(1).mean())

print("1. degenerate baselines (best input-INDEPENDENT predictor)")
print(f"   verdict accuracy        {maj_verdict:.3f}   (verdict is {verdict.mean():.3f} positive)")
print(f"   per-clause-flag accuracy {maj_flag:.3f}   (flags are {flags.mean():.3f} positive)")
print(f"   exact-instance accuracy  {const_exact:.3f}   (modal pattern; "
      f"per-bit-majority pattern would give {maj_exact:.3f})")
report["baseline"] = {"verdict": float(maj_verdict), "flag": float(maj_flag),
                      "exact": const_exact, "exact_majority_pattern": maj_exact, "verdict_pos_rate": float(verdict.mean()),
                      "flag_pos_rate": float(flags.mean())}

# ── 2. shortcut probes ───────────────────────────────────────────────────────
split = int(0.7 * args.samples)


def probe(features, y, label):
    """Accuracy AND the probe's own majority baseline.

    Each probe must be judged against the baseline of ITS OWN target: scoring a
    clause-flag probe against the verdict's baseline manufactures a leak that
    is not there (flags are ~95% positive, the verdict ~50%).
    """
    if len(np.unique(y[:split])) < 2:
        print(f"   {label:36s} degenerate (single class)")
        return None
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(features[:split], y[:split])
    acc = float(clf.score(features[split:], y[split:]))
    yt = y[split:]
    base = float(max(yt.mean(), 1 - yt.mean()))
    print(f"   {label:36s} {acc:.3f}   (own baseline {base:.3f}, "
          f"lift {acc - base:+.3f})")
    return {"acc": acc, "baseline": base, "lift": acc - base}


# summary features a NON-computing model could read off: how many literals of
# each sign per clause, how many occurrences per variable, assignment balance.
summ = []
for f, a in meta:
    pos = [sum(1 for l in c if l > 0) for c in f]
    occ = [sum(1 for c in f for l in c if abs(l) - 1 == v) for v in range(args.n)]
    summ.append(pos + occ + list(a) + [sum(a)])
summ = np.asarray(summ, dtype=np.float32)

print("\n2. shortcut probes (should sit AT the baseline above)")
lin_verdict = probe(X, verdict, "linear on raw pixels -> verdict")
sum_verdict = probe(summ, verdict, "summary stats -> verdict")
lin_flag0 = probe(X, flags[:, 0], "linear on raw pixels -> clause 0")
report["shortcut"] = {"linear_pixels_verdict": lin_verdict,
                      "summary_verdict": sum_verdict,
                      "linear_pixels_flag0": lin_flag0}

leak = [k for k, v in report["shortcut"].items()
        if v is not None and v["lift"] > 0.10]
report["leak_suspected"] = leak

# ── 3. light cone ────────────────────────────────────────────────────────────
# worst case: an assignment value at the far corner must reach its clause row,
# then travel along the row to the flag column, then down to the verdict.
vx, vy = layout.verdict_center
far = layout.min_rollout()
# information also has to make the return trip for the AND to be conditioned on
# every clause, so the honest requirement is the one-way distance; the AND then
# resolves down the flag column.
required = far
print("\n3. light cone")
print(f"   farthest input disc -> verdict, L1 distance: {required} cells")
print(f"   one cell per step => rollout must exceed {required}")
print(f"   recommended rollout (3x headroom): {3 * required}")
report["light_cone"] = {"required_steps": int(required), "recommended_rollout": int(3 * required)}

# ── 4. target well-posedness ─────────────────────────────────────────────────
print("\n4. target well-posedness")
det_ok = all(bits[i, -1] == evaluate(*meta[i]) for i in range(args.samples))
flag_ok = all(list(bits[i, :-1]) == clause_flags(*meta[i]) for i in range(args.samples))
print(f"   rendered target matches reference semantics: verdict={det_ok} flags={flag_ok}")

counts = []
for _ in range(200):
    f, _s = sample_instance(args.n, args.m, 3, rng, want="any")
    counts.append(n_solutions(f, args.n))
counts = np.asarray(counts)
frac_unique = float((counts == 1).mean())
frac_sat = float((counts >= 1).mean())
print(f"   at ratio {args.m / args.n:.2f}: satisfiable {frac_sat:.2f}, "
      f"uniquely satisfiable {frac_unique:.2f}, "
      f"median solutions among SAT {np.median(counts[counts >= 1]) if frac_sat else 0:.0f}")
print(f"   -> solve mode needs want='unique' (or the energy loss); "
      f"{1 - frac_unique:.0%} of instances here have an ambiguous target")
report["well_posed"] = {"render_matches_reference": bool(det_ok and flag_ok),
                        "frac_sat": frac_sat, "frac_unique": frac_unique}

# ── verdict on the gate ──────────────────────────────────────────────────────
print("\n" + "=" * 62)
passed = det_ok and flag_ok and not leak
print("TIER 0:", "PASS" if passed else "ATTENTION")
if leak:
    print("  shortcut probes beat the baseline by >0.10:", ", ".join(leak))
    print("  -> the answer is partly inferable without the intended computation")
if not (det_ok and flag_ok):
    print("  rendered target disagrees with reference semantics — encoding bug")
report["passed"] = bool(passed)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2))
print(f"\nwrote {out}")
