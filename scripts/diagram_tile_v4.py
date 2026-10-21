#!/usr/bin/env python3
"""Design sketch, tile v4: memory on top, ALU underneath.

     MEM in  ── copy ──▶  MEM out
      ▼ load                ▲ store
    selA/selB             dst
     [A][B] ─▶ [op] ─▶ [R] ┘

Data flows DOWN out of memory into the operand slots, RIGHT through the op
into the result, and UP into the destination column of the next memory grid.
Memory itself copies straight across on the top row.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
OPS = ("ADD", "COPY")                        # start small: two operations, one op column

OP, A, B, D = "ADD", 0, 1, 2
regs, out = (5, 9, 0, 0), (5, 9, 14, 0)
NR, WB = 4, 4
R_, P, PAD, VG = 2, 6, 5, 3
AG = 12                                      # SAME air on both sides of the op column
OPC_N = 1                                    # op columns (one column holds up to word_bits ops)
G = 2 * AG + (OPC_N - 1) * P                 # B --AG-- op --AG-- R

def bits(v): return [(v >> (WB - 1 - b)) & 1 for b in range(WB)]

# ── geometry ────────────────────────────────────────────────────────────────
grid_w = (NR - 1) * P
x_min = [PAD + c * P for c in range(NR)]
x_mout = [x_min[-1] + G + c * P for c in range(NR)]
tile_w = x_mout[0] - x_min[0]
# input selectors (selA, selB) ABOVE the grid they read; output selector (dst) BELOW the grid it writes
y_selA = PAD
y_selB = y_selA + P
y_mem = [y_selB + P + VG + b * P for b in range(WB)]
y_dst = y_mem[-1] + P + VG
y_alu = [y_dst + P + VG + b * P for b in range(WB)]
# ALU columns: A, B under the END of MEM in, op block in the gap, R under the START of MEM out
x_a, x_b = x_min[2], x_min[3]
x_op = [x_min[-1] + AG + c * P for c in range(OPC_N)]
x_r = x_mout[0]
H, W = y_alu[-1] + PAD + 1, x_mout[-1] + PAD + 1

REG = ["#e15759", "#f28e2b", "#59a14f", "#4e79a7"]
SEL_A, SEL_B, SEL_D, OPC, RES = "#ff9da7", "#9c755f", "#bab0ac", "#b07aa1", "#2ca02c"

def draw(ax, stage, title):
    ax.set_xlim(-12, W + 8); ax.set_ylim(H + 6, -10); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc="#f4f4f4", ec="k", lw=1))
    def blob(x, y, col, filled=None):
        if filled is None:
            ax.add_patch(Circle((x, y), R_, fc="none", ec=col, lw=1.0, ls=(0, (1.5, 1.5))))
            ax.add_patch(Circle((x, y), 0.5, fc=col, ec="none"))
        else:
            ax.add_patch(Circle((x, y), R_, fc=col if filled else "white", ec=col, lw=1.5))
    for c in range(NR):
        for b in range(WB): blob(x_min[c], y_mem[b], REG[c], bits(regs[c])[b])
        blob(x_min[c], y_selA, SEL_A, c == A); blob(x_min[c], y_selB, SEL_B, c == B)
        blob(x_mout[c], y_selA, SEL_A, False); blob(x_mout[c], y_selB, SEL_B, False)   # next instr's
        for b in range(WB):
            v = (bits(out[c])[b] if stage >= 4 else None) if c == D else (bits(regs[c])[b] if stage >= 3 else None)
            blob(x_mout[c], y_mem[b], REG[c], v)
        blob(x_mout[c], y_dst, SEL_D, c == D)
    for b in range(WB):
        blob(x_a, y_alu[b], REG[A], bits(regs[A])[b] if stage >= 1 else None)
        blob(x_b, y_alu[b], REG[B], bits(regs[B])[b] if stage >= 1 else None)
        blob(x_r, y_alu[b], RES, bits(out[D])[b] if stage >= 2 else None)
    for k, name in enumerate(OPS):
        c, r = divmod(k, WB); blob(x_op[c], y_alu[r], OPC, name == OP)
        ax.text(x_op[c] + R_ + 1, y_alu[r] + 0.6, name, fontsize=5, va="center", color=OPC)
    # labels
    ax.text(np.mean(x_min), -1.5, "MEM in", ha="center", va="bottom", fontsize=8, weight="bold")
    ax.text(np.mean(x_mout), -1.5, "MEM out", ha="center", va="bottom", fontsize=8, weight="bold")
    for c in range(NR):
        ax.text(x_min[c] + R_ + 0.3, y_mem[-1] + 0.6, "", fontsize=5)
    ax.text(x_min[0] - R_ - 1, y_selA + 0.6, "selA", ha="right", va="center", fontsize=6, color=SEL_A)
    ax.text(x_min[0] - R_ - 1, y_selB + 0.6, "selB", ha="right", va="center", fontsize=6, color=SEL_B)
    ax.text(x_mout[-1] + R_ + 1, y_dst + 0.6, "dst", ha="left", va="center", fontsize=6, color="#777")
    ax.text(x_mout[-1] + R_ + 1, y_selA + 0.6, "next A", ha="left", va="center", fontsize=5.5, color=SEL_A)
    ax.text(x_mout[-1] + R_ + 1, y_selB + 0.6, "next B", ha="left", va="center", fontsize=5.5, color=SEL_B)
    ax.text(x_a, y_alu[-1] + P, "A", ha="center", va="top", fontsize=7, color=REG[A], weight="bold")
    ax.text(x_b, y_alu[-1] + P, "B", ha="center", va="top", fontsize=7, color=REG[B], weight="bold")
    ax.text(np.mean(x_op), y_alu[-1] + P, "op", ha="center", va="top", fontsize=7, color=OPC, weight="bold")
    ax.text(x_r, y_alu[-1] + P, "R", ha="center", va="top", fontsize=7, color=RES, weight="bold")
    ax.set_title(title, fontsize=9, loc="left")
    arr = lambda p, q, col, lw=1.4, rad=0.0: ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->", color=col, lw=lw, mutation_scale=9, connectionstyle=f"arc3,rad={rad}"))
    if stage == 1:
        arr((x_min[A], y_mem[-1] + R_), (x_a, y_alu[0] - R_), REG[A])
        arr((x_min[B], y_mem[-1] + R_), (x_b, y_alu[0] - R_), REG[B])
    if stage == 2:
        arr((x_b + R_, np.mean(y_alu)), (x_r - R_, np.mean(y_alu)), RES, lw=1.6)
    if stage == 3:
        for c in range(NR):
            if c != D: arr((x_min[c] + R_, y_mem[c]), (x_mout[c] - R_, y_mem[c]), REG[c], lw=0.9)
    if stage == 4:
        arr((x_r, y_alu[0] - R_), (x_mout[D], y_mem[-1] + R_), RES, lw=1.6)

fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.ravel()
draw(axes[0], 0, "t = 0   painted: MEM in, selA/selB, op, dst.   A, B, R, MEM out unresolved")
draw(axes[1], 1, "stage 1  LOAD  ↓   A ← mem[selA],  B ← mem[selB]")
draw(axes[2], 2, "stage 2  EXECUTE  →   R ← op(A, B)")
draw(axes[3], 3, "stage 3  COPY  →   MEM out[c] ← MEM in[c],  c ≠ dst")
draw(axes[4], 4, "stage 4  STORE  ↑   MEM out[dst] ← R        (MEM out = next tile's MEM in)")
axes[5].axis("off")
axes[5].text(0, 0.95,
    f"tile v4 — ADD r0 r1 → r2 on [5, 9, 0, 0]\n\n"
    f"one tile: {tile_w} px wide × {H} px tall     (v3: 78×61, v2: 58×61)\n\n"
    "top         selA, selB rows above the grid they read\n"
    "mem row     MEM in ──── copy ────▶ MEM out\n"
    "below       dst row under the grid it writes\n"
    "ALU row     [A][B] under the END of MEM in,\n"
    "            [op] in the gap, [R] under the START of MEM out\n"
    f"            ops = {OPS}: one op column, {len(OPS)} blobs\n"
    f"            B →{AG}px→ op →{AG}px→ R  (symmetric)\n\n"
    "flow:  load ↓   execute →   copy →   store ↑\n\n"
    "in a chain: selA/selB above a grid = who reads it,\n"
    "dst below a grid = who wrote it.\n\n"
    "hops:\n"
    f"  load    selected column → A/B slot    ~{y_alu[0]-y_mem[-1]} px down, ≤{3*P} across\n"
    f"  exec    A,B → R                       ~{x_r-x_b} px across\n"
    f"  copy    MEM in col c → MEM out col c  {tile_w} px straight across\n"
    f"  store   R → dst column                ~{y_alu[0]-y_mem[-1]} px up, ≤{3*P} across",
    fontsize=9.5, va="top", family="monospace")
fig.suptitle("Tile v4 — input selectors above, output selector below, ALU underneath: load ↓ execute → copy → store ↑", fontsize=14, weight="bold")
fig.tight_layout()
p = Path(__file__).resolve().parent.parent / "tile_v4_design.png"
fig.savefig(p, dpi=140, facecolor="white"); print("wrote", p, f"tile {tile_w}x{H}")
