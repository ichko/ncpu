#!/usr/bin/env python3
"""One tile of the painted program: INPUT | OP | OUTPUT.  The unit that tiles."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.program import OPS, Instr, ProgramLayout, run_program

L = ProgramLayout(n_instr=1, n_regs=4, word_bits=4, unknown=0.0, op_amp=0.5)
ins = Instr("ADD", 0, 1, 2)
regs = (5, 9, 0, 0)
out = run_program(regs, [ins], 4)[1]

REG = ["#e15759", "#f28e2b", "#59a14f", "#4e79a7"]
FLD = {"op": "#b07aa1", "sel_a": "#ff9da7", "sel_b": "#9c755f", "sel_dst": "#bab0ac"}

fig, ax = plt.subplots(figsize=(13, 11))
ax.set_xlim(-30, L.W + 46); ax.set_ylim(L.H + 14, -30); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Rectangle((0, 0), L.W, L.H, fc="#f4f4f4", ec="k", lw=1.2))

def blob(x, y, col, filled, label=None, side=1):
    ax.add_patch(Circle((x, y), L.r, fc=col if filled else "white", ec=col, lw=1.8))
    if label:
        ax.text(x + side * (L.r + 1.5), y + 0.7, label, fontsize=8, va="center",
                ha="left" if side > 0 else "right", color="#333")

xi, xo, xp = L.col_x[L.data_col[0]], L.col_x[L.data_col[1]], L.col_x[L.op_col[0]]

# INPUT column
for r in range(4):
    bits = [(regs[r] >> (3 - b)) & 1 for b in range(4)]
    for b in range(4):
        y = L.data_y[L.row_of(r, b)]
        blob(xi, y, REG[r], bool(bits[b]), label=f"r{r} bit{3-b}" if True else None, side=-1)
ax.text(xi, -6, "INPUT\n4 regs × 4 bits\n= 16 blobs", ha="center", va="bottom", fontsize=11, weight="bold")
ax.text(xi, L.H + 3, f"r0..r3 = {list(regs)}", ha="center", va="top", fontsize=9)

# OP column
for k, name in enumerate(OPS):
    blob(xp, L.op_y[L.op_row("op", k)], FLD["op"], name == ins.op, label=name)
for fld, sel, tag in (("sel_a", ins.a, "src A"), ("sel_b", ins.b, "src B"), ("sel_dst", ins.dst, "dst")):
    for k in range(4):
        blob(xp, L.op_y[L.op_row(fld, k)], FLD[fld], k == sel, label=f"{tag} = r{k}")
ax.text(xp, -6, f"OP\n{ins.op} r{ins.a} r{ins.b} → r{ins.dst}\n20 blobs, ±0.5", ha="center", va="bottom",
        fontsize=11, weight="bold", color=FLD["op"])

# OUTPUT column
for r in range(4):
    bits = [(out[r] >> (3 - b)) & 1 for b in range(4)]
    for b in range(4):
        y = L.data_y[L.row_of(r, b)]
        ax.add_patch(Circle((xo, y), L.r, fc="none", ec=REG[r], lw=1.8, ls=(0, (2, 2))))
        if r == ins.dst:
            ax.add_patch(Circle((xo, y), L.r * 0.45, fc=REG[r] if bits[b] else "white", ec=REG[r], lw=1))
ax.text(xo, -6, "OUTPUT\nblank at t=0 (all 0)\nNCA must fill it", ha="center", va="bottom", fontsize=11, weight="bold")
ax.text(xo, L.H + 3, f"target {list(out)}\n(r2 = r0 + r1, rest copied)", ha="center", va="top", fontsize=9)

# arrows: what flows where
for r in (0, 1):
    y = L.data_y[L.row_of(r, 2)]
    ax.add_patch(FancyArrowPatch((xi + 4, y), (xp - 4, y), arrowstyle="->", color=REG[r], lw=1.5, mutation_scale=12))
y2 = L.data_y[L.row_of(2, 2)]
ax.add_patch(FancyArrowPatch((xp + 4, y2), (xo - 4, y2), arrowstyle="->", color=REG[2], lw=2.2, mutation_scale=14))
ax.text((xp + xo) / 2, y2 - 3, "write r2", ha="center", fontsize=8, color=REG[2])
for r in (0, 1, 3):
    y = L.data_y[L.row_of(r, 1)]
    ax.add_patch(FancyArrowPatch((xi + 4, y), (xo - 4, y), arrowstyle="->", color="#999", lw=0.9, ls="--",
                                 mutation_scale=10, connectionstyle="arc3,rad=0.0"))
ax.text((xi + xo) / 2, L.data_y[L.row_of(3, 1)] + 4.5, "copy-through (unchanged)", ha="center", fontsize=8, color="#777")

# next tile hint
ax.add_patch(FancyArrowPatch((xo + 6, L.H / 2), (L.W + 20, L.H / 2), arrowstyle="->", color="k", lw=2, mutation_scale=16))
ax.text(L.W + 22, L.H / 2 + 1, "OUTPUT is the\nINPUT of the\nnext tile", fontsize=10, va="center", weight="bold")

ax.text(L.W / 2, -24, f"one tile: {xo - xi} px wide · {L.H} px tall · blob r={L.r} · row pitch {L.pitch} · "
        f"K = {2 * L.min_rollout(2)} CA steps", ha="center", fontsize=9.5, color="#333")
ax.text(L.W / 2, -20, "filled = +1 · hollow = −1 · data blobs ±1 · op blobs ±0.5 · unresolved = 0", ha="center", fontsize=9.5, color="#333")
fig.suptitle("The tile: INPUT | OP | OUTPUT", fontsize=15, weight="bold")
fig.tight_layout()
p = Path(__file__).resolve().parent.parent / "tile_diagram.png"
fig.savefig(p, dpi=150, facecolor="white"); print("wrote", p)
