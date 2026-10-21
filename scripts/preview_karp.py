#!/usr/bin/env python3
"""Render the 3SAT -> CLIQUE reduction as the NCA will see it — no model involved.

Use this to eyeball the task before training: is the literal code readable, is
the body really blank on the input, does the grown matrix match the reference
graph, and is the certificate where it should be?

Panels:
    1. INPUT   — literal codes written down the left and across the top, body blank
    2. TARGET  — the same headers plus the grown 3m x 3m adjacency matrix, with
                 one worked cell traced: row code x column code -> edge?
    3. REFERENCE — the reduced graph as a matrix, with an m-clique circled
                 (the clique is the certificate that the formula is satisfiable)

Outputs (project root):
    karp_clique_preview.png

Usage:
    uv run python scripts/preview_karp.py
    uv run python scripts/preview_karp.py --vars 6 --clauses 5 --layout compact
    uv run python scripts/preview_karp.py --check          # answer preservation
    uv run python scripts/preview_karp.py --check --probe-out results/karp/tier0_clique.json

``--check`` runs the track's Tier 0 gate (docs/karp_eval_protocol.md):

    1. answer preservation   sat(brute force)  <=>  max_clique == m, on random
                             instances at the trained ratio AND above the phase
                             transition so the UNSAT side is actually exercised.
                             Only the biconditional is asserted -- the UNSAT
                             clique sizes are reported, never assumed (they are
                             usually m-1, but not always).
    2. shortcut probes       a linear model on the raw input pixels, and one on
                             summary statistics, must NOT beat the baseline of
                             the cell they are predicting.  Includes a positive
                             control that must.
    3. light cone            printed by layout.describe(); the trainer refuses
                             to start below it.
    4. render round-trip     decode(target) == adjacency(), and the input is
                             blank on every scored cell.
"""

import json
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.karp import (
    CliqueLayout,
    baseline_scores,
    CompactCliqueLayout,
    assignment_to_clique,
    clique_to_assignment,
    max_clique,
    sat_to_clique,
    shortcut_probe,
    verify_reduction,
    verify_render,
)
from ncpu.sat import sample_instance

parser = ArgumentParser()
parser.add_argument("--vars", type=int, default=5)
parser.add_argument("--clauses", type=int, default=4)
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--r", type=int, default=1,
                    help="lattice radius; step = 3r, and step multiplies straight "
                         "into the required rollout depth")
parser.add_argument("--layout", choices=["onehot", "compact"], default="onehot")
parser.add_argument("--block-hint", choices=["diag", "wall", "none"], default="diag")
parser.add_argument("--want", choices=["any", "sat", "unsat", "unique"], default="sat")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--check", action="store_true",
                    help="also run the answer-preservation + render round-trip checks")
parser.add_argument("--trials", type=int, default=200, help="--check: random instances")
parser.add_argument("--probe-samples", type=int, default=1500,
                    help="--check: instances for the Tier 0 shortcut probe (0 = skip)")
parser.add_argument("--probe-cells", type=int, default=3,
                    help="--check: how many scored cells to probe (most balanced, "
                         "median, most skewed)")
parser.add_argument("--probe-out", default=None,
                    help="--check: write the Tier 0 report to this JSON path")
parser.add_argument("--out-dir", default=".")
args = parser.parse_args()

rng = random.Random(args.seed)
Layout = CliqueLayout if args.layout == "onehot" else CompactCliqueLayout
layout = Layout(n_vars=args.vars, m_clauses=args.clauses, k=args.k, r=args.r,
                block_hint=args.block_hint)
print(layout.describe())


def lit_str(lit):
    return f"{'' if lit > 0 else '~'}x{abs(lit)}"


def formula_str(formula):
    return " & ".join("(" + " | ".join(lit_str(l) for l in c) + ")" for c in formula)


# ── the checks ───────────────────────────────────────────────────────────────
def unsat_note(stats):
    """What the UNSAT clique sizes actually did — measured, never asserted.

    The load-bearing property is the biconditional ``sat <=> clique == m``, and
    that is what verify_reduction asserts.  "UNSAT lands at exactly m-1" is NOT
    a property: it is the common case only (measured 94-100% depending on the
    shape), so it is printed as a fraction rather than claimed.
    """
    if not stats["unsat"]:
        return "  (no UNSAT instances at this ratio -- the UNSAT side is untested here)"
    frac = stats["unsat_at_m_minus_1_frac"]
    return (f"  UNSAT clique sizes {stats['unsat_clique_hist']}: "
            f"{stats['unsat_at_m_minus_1']}/{stats['unsat']} = {frac:.1%} at m-1, "
            f"the rest lower. Only 'never m' is guaranteed; 'exactly m-1' is not.")


report = {}
if args.check:
    print(f"\n1. answer preservation over {args.trials} random instances "
          f"(n={args.vars}, m={args.clauses}) ...")
    stats = verify_reduction(trials=args.trials, n_vars=args.vars,
                             m_clauses=args.clauses, k=args.k, rng=rng)
    print(f"  OK: {stats['sat']} satisfiable / {stats['unsat']} unsatisfiable; "
          f"max-clique sizes {stats['max_clique_hist']}  (sat <=> clique == m)")
    print(unsat_note(stats))
    # a ratio above the phase transition, so the UNSAT side is actually exercised
    hard_m = max(args.clauses, int(round(4.8 * args.vars)))
    hard = verify_reduction(trials=max(20, args.trials // 5), n_vars=args.vars,
                            m_clauses=hard_m, k=args.k, rng=rng)
    print(f"  OK at ratio {hard_m / args.vars:.1f} (m={hard_m}): {hard['sat']} sat / "
          f"{hard['unsat']} unsat; sizes {hard['max_clique_hist']}")
    print(unsat_note(hard))
    print(f"\n4. render round-trip: {verify_render(layout, trials=20, rng=rng)} instances OK")
    base = baseline_scores(layout, trials=500, rng=rng)
    print(f"\n1b. degenerate baseline (\"every off-block pair is an edge\", "
          f"{base['trials']} instances): per-cell {base['cell']:.3f} "
          f"(analytic {base['analytic_cell']:.3f}), "
          f"exact-adjacency {base['exact']:.4f} over {base['scored_cells']} scored cells "
          f"-- read every model number against these")
    report = {"layout": layout.describe(), "n": args.vars, "m": args.clauses,
              "k": args.k, "block_hint": args.block_hint,
              "answer_preservation": {"trained_shape": stats, f"m{hard_m}": hard},
              "baseline": base,
              "light_cone": {"min_rollout_px": layout.min_rollout(),
                             "suggested": layout.suggested_rollout()}}

    if args.probe_samples > 0:
        print("\n2. shortcut probes -- each judged against ITS OWN cell's baseline")
        probe = shortcut_probe(layout, samples=args.probe_samples, seed=args.seed,
                               n_cells=args.probe_cells)
        report["shortcut"] = probe
        if probe["leak"]:
            print("  ATTENTION: beat their own baseline by >0.10:",
                  ", ".join(probe["leak"]))
        else:
            print("  OK: no probe beats its own cell's baseline by >0.10 -- the edge "
                  "bits are not inferable without the row x column product")
        if not probe["control_ok"]:
            print("  WARNING: the positive control did NOT separate; the null result "
                  "above is not trustworthy")
        report["passed"] = not probe["leak"] and probe["control_ok"]
        print("\nTIER 0:", "PASS" if report["passed"] else "ATTENTION")

    if args.probe_out:
        out_json = Path(args.probe_out)
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(report, indent=2, default=str))
        print(f"wrote {out_json}")

# ── one instance to look at ──────────────────────────────────────────────────
formula, _ = sample_instance(args.vars, args.clauses, args.k, rng, want=args.want)
occ, adj = sat_to_clique(formula)
inp, tgt, _ = layout.screens_for(formula)

clique = max_clique(adj)
satisfiable = len(clique) == layout.m
assignment = clique_to_assignment(occ, clique, layout.n) if satisfiable else None
print(f"\nformula: {formula_str(formula)}")
print(f"max clique = {len(clique)}/{layout.m} -> "
      f"{'SAT, assignment ' + str(assignment) if satisfiable else 'UNSAT'}")
if satisfiable:
    print(f"clique nodes {clique} = " +
          ", ".join(f"c{occ[u][0]}:{lit_str(occ[u][2])}" for u in clique))
    assert assignment_to_clique(formula, assignment, layout.k) is not None

node_label = [f"c{i}:{lit_str(lit)}" for i, _, lit in occ]

# ── figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.6))


def frame_axes(ax, screen, title):
    ax.imshow(screen, cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=11)
    ax.set_xticks(layout.head_x + layout.body_x)
    ax.set_xticklabels([f"x{d + 1}" if args.layout == "onehot" else f"L{d}"
                        for d in range(layout.D)] + node_label,
                       rotation=90, fontsize=6.5)
    ax.set_yticks(layout.head_y + layout.body_y)
    ax.set_yticklabels([f"x{d + 1}" if args.layout == "onehot" else f"L{d}"
                        for d in range(layout.D)] + node_label, fontsize=6.5)
    ax.tick_params(length=1.5, pad=1)
    split = (layout.head_x[-1] + layout.body_x[0]) / 2
    ax.axvline(split, color="w", lw=1.0, alpha=0.8)
    ax.axhline(split, color="w", lw=1.0, alpha=0.8)
    for b in range(1, layout.m):                       # clause-block boundaries
        p = (layout.body_x[b * layout.k - 1] + layout.body_x[b * layout.k]) / 2
        ax.axvline(p, color="w", lw=0.4, alpha=0.35)
        ax.axhline(p, color="w", lw=0.4, alpha=0.35)


# pick one worked cell to trace on the target: a complementary (non-edge) pair
# if there is one, else any off-block pair.
cands = [(u, v) for u in range(layout.N) for v in range(layout.N)
         if layout.scored[u, v] and not adj[u, v]]
if not cands:
    cands = [(u, v) for u in range(layout.N) for v in range(layout.N) if layout.scored[u, v]]
u, v = cands[len(cands) // 2]
verdict = "edge" if adj[u, v] else "NO edge (complementary)"

frame_axes(axes[0], inp,
           f"INPUT  ({layout.W}x{layout.H})\nliteral codes only — the body is empty")
frame_axes(axes[1], tgt,
           "TARGET — the reduced instance, grown in place\n"
           f"circled: row {node_label[u]}  x  col {node_label[v]}  ->  {verdict}")

ax = axes[1]
cx, cy = layout.body_x[v], layout.body_y[u]
ax.plot([layout.head_x[0] - layout.r, cx], [cy, cy], color="w", lw=1.0, alpha=0.9)
ax.plot([cx, cx], [layout.head_y[0] - layout.r, cy], color="w", lw=1.0, alpha=0.9)
ax.add_patch(Circle((cx, cy), layout.r + 2.0, fill=False, color="w", lw=1.6))

# reference panel
ax = axes[2]
ax.imshow(adj.astype(float), cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
ax.set_title(f"REFERENCE graph ({layout.N} nodes) — red = the m-clique certificate\n"
             f"max clique {len(clique)}/{layout.m} -> "
             f"{'SATISFIABLE' if satisfiable else 'UNSATISFIABLE'}", fontsize=11)
ax.set_xticks(range(layout.N)); ax.set_xticklabels(node_label, rotation=90, fontsize=7)
ax.set_yticks(range(layout.N)); ax.set_yticklabels(node_label, fontsize=7)
for b in range(1, layout.m):
    ax.axvline(b * layout.k - 0.5, color="w", lw=0.5, alpha=0.4)
    ax.axhline(b * layout.k - 0.5, color="w", lw=0.5, alpha=0.4)
for i in clique:                                        # the certificate
    for j in clique:
        if i != j:
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                   color="crimson", lw=1.2))
    ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False,
                           color="crimson", lw=1.2, ls=":"))

fig.suptitle(
    f"3SAT -> CLIQUE as morphogenesis    {formula_str(formula)}    "
    f"[{type(layout).__name__}, block_hint={layout.block_hint}, "
    f"header {layout.D} lanes, min rollout {layout.min_rollout()} px, "
    f"all-edges baseline {layout.edge_baseline():.3f}]",
    fontsize=11.5,
)
fig.text(0.5, 0.025,
         "values: teal 0.0 = background   |   yellow +1 = edge / positive literal   |   "
         "dark -1 = no edge / negated literal\n"
         "the k x k diagonal blocks are instance-independent geometry: given on the "
         "input, and excluded from every score",
         ha="center", va="bottom", fontsize=9, color="0.25")
fig.tight_layout(rect=(0, 0.105, 1, 0.95))
out = Path(args.out_dir) / "karp_clique_preview.png"
fig.savefig(out, dpi=140)
plt.close(fig)
print(f"\nwrote {out}")
