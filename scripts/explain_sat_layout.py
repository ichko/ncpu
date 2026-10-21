#!/usr/bin/env python3
"""Annotated spec sheet for the SATLayout pixel geometry.

Companion to scripts/explain_sat_encoding.py, which covers the dataflow. This
one covers the LAYOUT: how the lattice is built, what every parameter does in
pixels, which region is which, and how W/H/min_rollout fall out of (n, m, r).

Usage:
    uv run python scripts/explain_sat_layout.py
Writes sat_layout_explained.png to the project root.
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

# n=4 so clauses have an ABSENT variable (k=3 of 4) -- the symbol an n=3
# training set structurally cannot contain, which cost us a whole result.
N, M, R = 4, 4, 2
formula = ((1, -2, 3), (-1, 3, -4), (2, -3, 4), (1, 2, -4))
assignment = (1, 0, 1, 0)

lay = SATLayout(n_vars=N, m_clauses=M, r=R)
inp = lay.input_screen(formula, assignment)
tgt = lay.target_screen(formula, assignment)
CMAP, VMIN, VMAX = "RdBu", -1.0, 1.0
INK = "#111"

fig = plt.figure(figsize=(17.6, 10.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.72], width_ratios=[1.15, 1, 1],
                      hspace=0.34, wspace=0.28,
                      left=0.05, right=0.975, top=0.885, bottom=0.05)
fig.suptitle("SATLayout — pixel geometry and encoding spec",
             fontsize=17, fontweight="bold", y=0.963)

# ── panel 1: the annotated full screen ───────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.imshow(inp, cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")
ax.set_title(f"1.  REGION MAP   ({lay.W}x{lay.H} px, n={N} m={M} r={R})",
             fontsize=12.5, fontweight="bold", pad=18)
ax.set_xticks(lay.var_x + [lay.flag_x])
ax.set_xticklabels([f"x{j}\n{x}" for j, x in enumerate(lay.var_x)] + [f"flag\n{lay.flag_x}"],
                   fontsize=7.5)
ax.set_yticks([lay.asn_y] + lay.clause_y + [lay.verdict_y])
ax.set_yticklabels([f"assign  {lay.asn_y}"] + [f"c{i}  {y}" for i, y in enumerate(lay.clause_y)]
                   + [f"verdict  {lay.verdict_y}"], fontsize=7.5)
ax.tick_params(length=2)

def region(x0, y0, x1, y1, color, label, lx, ly, ha="left"):
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                           ec=color, lw=1.8, ls=(0, (4, 2))))
    ax.text(lx, ly, label, color=color, fontsize=8.5, fontweight="bold",
            ha=ha, va="center")

h = lay.r + 1
rx = lay.W + 2
region(lay.var_x[0] - h, lay.asn_y - h, lay.var_x[-1] + h, lay.asn_y + h,
       "#2166ac", "assignment row", rx, lay.asn_y)
region(lay.var_x[0] - h, lay.clause_y[0] - h, lay.var_x[-1] + h, lay.clause_y[-1] + h,
       "#111111", "clause matrix\n(n x m literals)", rx,
       (lay.clause_y[0] + lay.clause_y[-1]) / 2 - 3)
region(lay.flag_x - h, lay.clause_y[0] - h, lay.flag_x + h, lay.clause_y[-1] + h,
       "#e08214", "flag column\n(the answer)", rx,
       (lay.clause_y[0] + lay.clause_y[-1]) / 2 + 6)
region(lay.flag_x - h, lay.verdict_y - h, lay.flag_x + h, lay.verdict_y + h,
       "#1b7837", "verdict", rx, lay.verdict_y)
ax.set_xlim(-2, lay.W + 26)
ax.set_xlabel("column x / row y are PIXEL coordinates", fontsize=8.5, labelpad=8)

# ── panel 2: the lattice, dimensioned ────────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
z = inp[: lay.clause_y[1] + 2 * lay.r + 2, : lay.var_x[1] + 2 * lay.r + 2]
ax.imshow(z, cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")
ax.set_title("2.  THE LATTICE, DIMENSIONED", fontsize=12.5, fontweight="bold", pad=18)
ax.set_xticks(np.arange(-0.5, z.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, z.shape[0], 1), minor=True)
ax.grid(which="minor", color="#bbb", lw=0.4)
ax.set_xticks([]); ax.set_yticks([])


def dim(p0, p1, label, color, off=(0, 0), ha="center", va="center"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="<|-|>", mutation_scale=9,
                                 color=color, lw=1.6, shrinkA=0, shrinkB=0))
    ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], label,
            color=color, fontsize=9, fontweight="bold", ha=ha, va=va)


ax.add_patch(FancyArrowPatch((-0.5, lay.asn_y), (lay.var_x[0] - lay.r - 0.5, lay.asn_y),
                             arrowstyle="<|-|>", mutation_scale=9, color="#6a3d9a", lw=1.6))
ax.text((lay.var_x[0] - lay.r) / 2, lay.asn_y - 1.4, f"pad={lay.pad}", color="#6a3d9a",
        fontsize=9, fontweight="bold", ha="center")
ax.add_patch(FancyArrowPatch((lay.var_x[0] - lay.r, lay.clause_y[0] + lay.r + 2.0),
                             (lay.var_x[0] + lay.r, lay.clause_y[0] + lay.r + 2.0),
                             arrowstyle="<|-|>", mutation_scale=9, color="#b2182b",
                             lw=1.6, shrinkA=0, shrinkB=0))
ax.text(lay.var_x[0] - lay.r - 1.2, lay.clause_y[0] + lay.r + 2.0, f"2r={2*lay.r}",
        color="#b2182b", fontsize=9, fontweight="bold", ha="right", va="center")
dim((lay.var_x[0] + lay.r, lay.asn_y - lay.r - 1.6),
    (lay.var_x[1] - lay.r, lay.asn_y - lay.r - 1.6),
    f"gap={lay.gap}", "#e08214", off=(0, -1.5))
dim((lay.var_x[0], lay.clause_y[0] + lay.r + 2.6),
    (lay.var_x[1], lay.clause_y[0] + lay.r + 2.6),
    f"step={lay.step}", "#1b7837", off=(0, 1.6))
ax.set_xlabel(f"step = 2r + gap = {lay.step} px between disc centres.\n"
              f"A disc of radius r={lay.r} is {int((np.abs(np.arange(-R,R+1))[:,None]**2 + np.abs(np.arange(-R,R+1))[None,:]**2 <= R*R).sum())} pixels.\n"
              "Depth scales with step, so r is the main cost lever.",
              fontsize=8.5, labelpad=8, loc="left", linespacing=1.5)

# ── panel 3: input vs target ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
pair = np.concatenate([inp, np.full((4, lay.W), np.nan), tgt], axis=0)
ax.imshow(pair, cmap=CMAP, vmin=VMIN, vmax=VMAX, interpolation="nearest")
ax.set_title("3.  INPUT  vs  TARGET", fontsize=12.5, fontweight="bold", pad=18)
ax.set_xticks([]); ax.set_yticks([])
off = lay.H + 4
ax.text(1, 2, "INPUT", fontsize=9, fontweight="bold", color=INK, va="top")
ax.text(1, off + 2, "TARGET", fontsize=9, fontweight="bold", color=INK, va="top")
for i, f in enumerate(clause_flags(formula, assignment)):
    ax.add_patch(Rectangle((lay.flag_x - h, off + lay.clause_y[i] - h), 2 * h, 2 * h,
                           fill=False, ec="#e08214", lw=1.5))
ax.add_patch(Rectangle((lay.flag_x - h, off + lay.verdict_y - h), 2 * h, 2 * h,
                       fill=False, ec="#1b7837", lw=2.0))
ax.add_patch(Rectangle((lay.flag_x - h, lay.clause_y[0] - h), 2 * h,
                       lay.verdict_y - lay.clause_y[0] + 2 * h,
                       fill=False, ec="#b2182b", lw=1.5, ls=(0, (3, 2))))
ax.set_xlabel("Identical except the answer column.\n"
              "On the INPUT it is background (0.0) — the model\n"
              "must not be able to read the answer off its own input.\n"
              "Verified: input is identically zero on all output pixels.",
              fontsize=8.5, labelpad=8, loc="left", linespacing=1.5)

# ── bottom text ──────────────────────────────────────────────────────────────
def block(col, title, body):
    a = fig.add_subplot(gs[1, col]); a.axis("off")
    a.text(0, 1.0, title, fontsize=11.5, fontweight="bold", va="top",
           transform=a.transAxes)
    a.text(0, 0.845, body, fontsize=9.0, va="top", ha="left", family="monospace",
           linespacing=1.62, transform=a.transAxes)


block(0, "PIXEL VALUES", 
      "  +1.0   blue    literal positive / TRUE / satisfied\n"
      "   0.0   white   BACKGROUND — two distinct meanings:\n"
      "                   in the clause matrix, 'this variable\n"
      "                   is absent from this clause'\n"
      "                   in the answer column, 'not yet written'\n"
      "  -1.0   red     literal negated / FALSE / unsatisfied\n\n"
      "Decoding: mask-average each disc, threshold at 0.\n\n"
      "The absent symbol is why n must be > k in training:\n"
      "at n=k=3 every clause holds every variable, so 0.0\n"
      "never appears in the matrix and the model never\n"
      "learns it. Such a model collapses at n>=4.")

block(1, "GEOMETRY", 
      f"  step = 2r + gap          = {lay.step} px\n"
      f"  pad  = 2r                = {lay.pad} px\n\n"
      f"  var_x[j] = pad + r + j*step\n"
      f"  asn_y    = pad + r\n"
      f"  clause_y[i] = asn_y + (i+1)*step\n"
      f"  flag_x   = pad + r + n*step\n"
      f"  verdict_y = clause_y[-1] + step\n\n"
      f"  W = flag_x + r + pad     = {lay.W}\n"
      f"  H = verdict_y + r + pad  = {lay.H}\n\n"
      "  min_rollout = (n + m + 1) * step\n"
      "      m*step  assignment -> last clause row\n"
      "    + n*step  across to the flag column\n"
      "    + 1*step  down to the verdict")

sizes = [(3, 3, 1), (6, 8, 1), (6, 16, 1), (6, 26, 1), (6, 26, 2), (10, 13, 1)]
rows = ["   n    m   r      grid    min_roll   status", "  " + "-" * 44]
status = {(3, 3, 1): "solved", (6, 8, 1): "solved 0.98", (6, 16, 1): "extrap 0.94",
          (6, 26, 1): "fails", (6, 26, 2): "fails", (10, 13, 1): "extrap 0.94"}
for n, m, r in sizes:
    L = SATLayout(n_vars=n, m_clauses=m, r=r)
    rows.append(f"  {n:2d}  {m:3d}  {r:2d}   {L.W:3d}x{L.H:<4d}    {L.min_rollout():5d}   {status[(n,m,r)]}")
block(2, "SIZES, DEPTH, AND WHAT WORKED",
      "\n".join(rows) +
      "\n\n  Inference needs ~2x min_rollout: at 1.0x the\n"
      "  same weights score exact 0.05, at 2.0x  0.91.\n"
      "  Always rescale rollout to the layout being\n"
      "  evaluated, or geometry looks like failure.")

out = Path(__file__).resolve().parent.parent / "sat_layout_explained.png"
fig.savefig(out, dpi=150, facecolor="white")
print(f"wrote {out}")
