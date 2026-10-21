#!/usr/bin/env python3
"""Annotated diagram of the 3SAT-on-a-grid encoding and how the NCA computes it.

Explains in one figure: what a formula looks like as pixels, what the NCA has to
fill in, the local dataflow that turns an assignment into a verdict, why the
rollout depth is what it is, and why every metric is printed beside a baseline.

Usage:
    uv run python scripts/explain_sat_encoding.py
Writes sat_encoding_explained.png to the project root.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.sat import SATLayout, clause_flags, evaluate

N, M = 5, 6
formula = (
    (1, -3, 5), (-1, 2, 4), (2, -4, -5), (-2, 3, 5), (1, -2, -3), (-3, 4, -5),
)

# Pick an assignment that fails EXACTLY ONE clause: the instructive case, where
# the verdict genuinely depends on the AND rather than on a glance.
assignment = next(
    a for a in ((b >> 4 & 1, b >> 3 & 1, b >> 2 & 1, b >> 1 & 1, b & 1)
                for b in range(32))
    if sum(clause_flags(formula, a)) == M - 1
)

lay = SATLayout(n_vars=N, m_clauses=M, r=3)
inp = lay.input_screen(formula, assignment)
tgt = lay.target_screen(formula, assignment)
flags = clause_flags(formula, assignment)
verdict = evaluate(formula, assignment)
bad = flags.index(0)

CMAP, VMIN, VMAX = "RdBu", -1.0, 1.0
INK = "#111111"


def show(ax, img, title, caption):
    ax.imshow(img, cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=20)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#aaa"); s.set_linewidth(0.8)
    ax.set_xlabel(caption, fontsize=8.8, labelpad=10, linespacing=1.55,
                  color=INK, loc="left")
    for j, x in enumerate(lay.var_x):
        ax.text(x, -2.5, f"x{j}", ha="center", va="bottom", fontsize=9, color=INK)
    ax.text(lay.flag_x, -2.5, "flag", ha="center", va="bottom", fontsize=9,
            fontweight="bold", color=INK)
    ax.text(-6, lay.asn_y, "assign →", ha="right", va="center", fontsize=8.5,
            color="#2166ac", fontweight="bold")
    ax.text(-6, lay.clause_y[len(lay.clause_y) // 2], "clauses",
            ha="right", va="center", fontsize=8.5, color=INK, rotation=90)


fig = plt.figure(figsize=(18.2, 9.6))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 0.60], width_ratios=[1, 1, 1, 1.35],
                      hspace=0.46, wspace=0.42,
                      left=0.055, right=0.98, top=0.865, bottom=0.065)
fig.suptitle("3SAT on a grid: what the NCA sees, and what it has to compute",
             fontsize=17, fontweight="bold", y=0.965)

# ── 1. input ─────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
show(ax, inp, "1.  INPUT — what it is given",
     "A formula drawn as its clause matrix.\n"
     "blue +1 = literal positive · red −1 = negated\n"
     "white = that variable is absent from the clause\n"
     "Top row is the candidate assignment.\n"
     "The flag column is BLANK — that is the answer.")
ax.add_patch(Rectangle((lay.flag_x - 5, lay.clause_y[0] - 5), 10,
                       lay.verdict_center[1] - lay.clause_y[0] + 10,
                       fill=False, ec="#b2182b", lw=1.6, ls=(0, (3, 2))))

# ── 2. target ────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
show(ax, tgt, "2.  TARGET — what it must produce",
     f"Same formula, plus the answer column.\n"
     f"orange = per-clause flag, +1 iff satisfied\n"
     f"green  = verdict, the AND of all {M} flags\n"
     f"Clause c{bad} fails here, so the verdict is UNSAT\n"
     f"— one red disc in {M} flips the whole answer.")
for i in range(M):
    ec = "#e08214" if flags[i] else "#b2182b"
    ax.add_patch(Rectangle((lay.flag_x - 5, lay.clause_y[i] - 5), 10, 10,
                           fill=False, ec=ec, lw=2.0))
ax.add_patch(Rectangle((lay.flag_x - 5, lay.verdict_center[1] - 5), 10, 10,
                       fill=False, ec="#1b7837", lw=2.6))

# ── 3. dataflow ──────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
show(ax, inp * 0.30, "3.  THE LOCAL DATAFLOW",
     "① broadcast each assignment DOWN its column\n"
     "② multiply in place — a literal is satisfied iff\n"
     "     sign × assignment > 0 — then OR ALONG the row\n"
     "③ AND DOWN the flag column into the verdict\n"
     "Every step is local: no cell sees the whole grid.")
r = 2
ax.add_patch(FancyArrowPatch((lay.var_x[1], lay.asn_y + 5),
                             (lay.var_x[1], lay.clause_y[r] - 5),
                             arrowstyle="-|>", mutation_scale=14,
                             color="#2166ac", lw=2.4))
ax.add_patch(FancyArrowPatch((lay.var_x[0], lay.clause_y[r]),
                             (lay.flag_x - 5, lay.clause_y[r]),
                             arrowstyle="-|>", mutation_scale=14,
                             color="#e08214", lw=2.4))
ax.add_patch(FancyArrowPatch((lay.flag_x, lay.clause_y[0] + 5),
                             (lay.flag_x, lay.verdict_center[1] - 5),
                             arrowstyle="-|>", mutation_scale=14,
                             color="#1b7837", lw=2.4))
ax.text(lay.var_x[1] + 4, lay.asn_y + 13, "①", fontsize=15,
        color="#2166ac", fontweight="bold")
ax.text(lay.var_x[2], lay.clause_y[r] - 9, "②", fontsize=15,
        color="#8a4b00", fontweight="bold", ha="center")
ax.text(lay.flag_x - 13, lay.clause_y[3], "③", fontsize=15,
        color="#1b7837", fontweight="bold", ha="center")

# ── 4. worked ────────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 3]); ax.axis("off")
ax.set_title("4.  THE SAME INSTANCE, WORKED", fontsize=12.5,
             fontweight="bold", pad=20, loc="left")
rows = ["assignment:  " + "  ".join(
    f"x{j}={'T' if v else 'F'}" for j, v in enumerate(assignment)),
    "─" * 46, ""]
for i, cl in enumerate(formula):
    parts = []
    for lit in cl:
        v = abs(lit) - 1
        ok = (lit > 0) == bool(assignment[v])
        parts.append(("" if lit > 0 else "!") + f"x{v}" + ("+" if ok else "-"))
    tail = "flag +1" if flags[i] else "flag -1   <-- FAILS"
    rows.append(f"c{i}:  " + " v ".join(f"{p:5s}" for p in parts) + f"  ->  {tail}")
rows += ["", "─" * 46,
         f"verdict = AND(flags) = {'+1 (SAT)' if verdict else '-1 (UNSAT)'}"]
ax.text(0.0, 0.94, "\n".join(rows), fontsize=9.3, family="monospace",
        va="top", ha="left", linespacing=1.9, transform=ax.transAxes)

# ── bottom text ──────────────────────────────────────────────────────────────
ax = fig.add_subplot(gs[1, :2]); ax.axis("off")
ax.text(0, 1.0, "WHY THE ROLLOUT IS AS DEEP AS IT IS", fontsize=11.5,
        fontweight="bold", va="top", transform=ax.transAxes)
ax.text(0, 0.845,
        "A 3×3 kernel moves information ONE PIXEL per step, so required depth is a distance in\n"
        "pixels, not a count of lattice cells — it scales with the lattice pitch `step`.\n\n"
        "The legs ADD, because they are sequentially dependent: the clause OR cannot start before\n"
        "the assignment arrives, and the verdict AND cannot start before the flags do.\n\n"
        "       m·step    assignment down to the last clause row\n"
        "    +  n·step    across to the flag column\n"
        "    +  1·step    down to the verdict\n"
        "    =  (n + m + 1)·step        →  99 steps at n=6, m=26, r=1\n\n"
        "Below this the task is IMPOSSIBLE, not hard — a failure that would look like a learning\n"
        "failure. The verdict sits at the BOTTOM of the flag column for the same reason: at the top\n"
        "corner every flag must travel the full height back up, adding a whole m·step leg\n"
        "(348 → 198 steps at r=2).",
        fontsize=9.2, va="top", ha="left", linespacing=1.5, transform=ax.transAxes)

ax = fig.add_subplot(gs[1, 2:]); ax.axis("off")
ax.text(0, 1.0, "WHY EVERY NUMBER IS PRINTED NEXT TO A BASELINE", fontsize=11.5,
        fontweight="bold", va="top", transform=ax.transAxes)
ax.text(0, 0.845,
        "At the phase ratio m/n ≈ 4.267 about 95% of clause flags are satisfied. A model that answers\n"
        "“satisfied” for every clause therefore scores 95% while computing nothing at all.\n\n"
        "     per-clause flag accuracy      baseline 0.95     ← nearly uninformative alone\n"
        "     verdict accuracy              baseline 0.50     ← the bit that matters\n"
        "     exact-instance accuracy       baseline 0.50     ← every output bit correct\n\n"
        "That 95% is a trap for TRAINING too, not just for reporting: the 5% of discs carrying the\n"
        "answer received only 2.5% of the loss, and the first probe duly collapsed to stamping every\n"
        "clause satisfied. `--flag-neg-weight` re-weights the rare class to correct it.\n\n"
        "`f+pred` = fraction of flags the model predicts positive. 1.00 = stamping, 0.95 = discriminating.",
        fontsize=9.2, va="top", ha="left", linespacing=1.5, transform=ax.transAxes)

out = Path(__file__).resolve().parent.parent / "sat_encoding_explained.png"
fig.savefig(out, dpi=150, facecolor="white")
print(f"wrote {out}  (clause c{bad} fails, verdict={'SAT' if verdict else 'UNSAT'})")
