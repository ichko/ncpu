#!/usr/bin/env python3
"""Design sketch, tile v3: explicit LOAD -> EXECUTE -> COPY -> STORE stages.

    selA  o . . .
    selB  . o . .
         [ MEM in ]   [A][B]  [op]  [R]   [ MEM out ]
                                             dst  . . o .

Stage 1  LOAD     A <- mem[selA], B <- mem[selB]
Stage 2  EXECUTE  R <- op(A, B)
Stage 3  COPY     mem_out[c] <- mem_in[c] for every c != dst
Stage 4  STORE    mem_out[dst] <- R
Every stage is a separate supervision target, in that time order.
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.program import OPS

OP, A, B, D = "ADD", 0, 1, 2
regs, out = (5, 9, 0, 0), (5, 9, 14, 0)
NR, WB = 4, 4
R_, P, PAD, G = 2, 6, 8, 12

def bits(v): return [(v >> (WB - 1 - b)) & 1 for b in range(WB)]

# ── geometry ────────────────────────────────────────────────────────────────
x_min = [PAD + c * P for c in range(NR)]
x_ab = [x_min[-1] + G + c * P for c in range(2)]            # A, B slots
x_op = [x_ab[-1] + G + c * P for c in range(2)]             # 2x4 op block
x_r = x_op[-1] + G                                           # result slot
x_mout = [x_r + G + c * P for c in range(NR)]
y_selA, y_selB = PAD, PAD + P
y_bits = [y_selB + P + 4 + b * P for b in range(WB)]
y_dst = y_bits[-1] + P + 4
y_lab = y_bits[0] - P - 1
H, W = y_dst + PAD + 1, x_mout[-1] + PAD + 1

REG = ["#e15759", "#f28e2b", "#59a14f", "#4e79a7"]
SEL_A, SEL_B, SEL_D, OPC, SLOT, RES = "#ff9da7", "#9c755f", "#bab0ac", "#b07aa1", "#6c8ebf", "#2ca02c"

def draw(ax, stage, title):
    """stage 0 = t0, 1 = loaded, 2 = executed, 3 = copied, 4 = stored"""
    ax.set_xlim(-14, W + 6); ax.set_ylim(H + 8, -18); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc="#f4f4f4", ec="k", lw=1))
    def blob(x, y, col, filled=None, small=False):
        if filled is None:                       # unresolved
            ax.add_patch(Circle((x, y), R_, fc="none", ec=col, lw=1.1, ls=(0, (1.5, 1.5))))
            ax.add_patch(Circle((x, y), 0.5, fc=col, ec="none"))
        else:
            ax.add_patch(Circle((x, y), R_, fc=col if filled else "white", ec=col, lw=1.6))
    for c in range(NR):                          # MEM in
        for b in range(WB): blob(x_min[c], y_bits[b], REG[c], bits(regs[c])[b])
        blob(x_min[c], y_selA, SEL_A, c == A); blob(x_min[c], y_selB, SEL_B, c == B)
    for c in range(NR):                          # MEM out
        for b in range(WB):
            if c == D:   v = bits(out[c])[b] if stage >= 4 else None
            else:        v = bits(regs[c])[b] if stage >= 3 else None
            blob(x_mout[c], y_bits[b], REG[c], v)
        blob(x_mout[c], y_dst, SEL_D, c == D)
    for b in range(WB):                          # A, B slots
        blob(x_ab[0], y_bits[b], REG[A], bits(regs[A])[b] if stage >= 1 else None)
        blob(x_ab[1], y_bits[b], REG[B], bits(regs[B])[b] if stage >= 1 else None)
    for k, name in enumerate(OPS):               # op block
        c, r = divmod(k, WB); blob(x_op[c], y_bits[r], OPC, name == OP)
    for b in range(WB):                          # result
        blob(x_r, y_bits[b], RES, bits(out[D])[b] if stage >= 2 else None)
    # labels
    for c in range(NR):
        ax.text(x_min[c], y_bits[-1] + P + 0.5, f"r{c}", ha="center", va="top", fontsize=6.5, color=REG[c])
        ax.text(x_mout[c], y_bits[-1] + P + 0.5, f"r{c}", ha="center", va="top", fontsize=6.5, color=REG[c])
    ax.text(x_ab[0], y_lab, "A", ha="center", va="bottom", fontsize=8, color=SLOT, weight="bold")
    ax.text(x_ab[1], y_lab, "B", ha="center", va="bottom", fontsize=8, color=SLOT, weight="bold")
    ax.text(np.mean(x_op), y_lab, "op", ha="center", va="bottom", fontsize=8, color=OPC, weight="bold")
    ax.text(x_r, y_lab, "R", ha="center", va="bottom", fontsize=8, color=RES, weight="bold")
    ax.text(np.mean(x_min), y_lab - 12, "MEM in", ha="center", va="bottom", fontsize=8, weight="bold")
    ax.text(np.mean(x_mout), y_lab - 12, "MEM out", ha="center", va="bottom", fontsize=8, weight="bold")
    ax.text(x_min[0] - R_ - 1, y_selA + 0.6, "A", ha="right", va="center", fontsize=6.5, color=SEL_A)
    ax.text(x_min[0] - R_ - 1, y_selB + 0.6, "B", ha="right", va="center", fontsize=6.5, color=SEL_B)
    ax.text(x_mout[-1] + R_ + 1, y_dst + 0.6, "dst", ha="left", va="center", fontsize=6.5, color="#777")
    ax.set_title(title, fontsize=9, loc="left")
    ym = np.mean(y_bits)
    if stage == 1:
        for c, xs in ((A, x_ab[0]), (B, x_ab[1])):
            ax.add_patch(FancyArrowPatch((x_min[c] + R_, y_bits[1] if c == A else y_bits[2]), (xs - R_, y_bits[1] if c == A else y_bits[2]),
                                         arrowstyle="->", color=REG[c], lw=1.4, mutation_scale=9, connectionstyle="arc3,rad=-0.25"))
    if stage == 2:
        ax.add_patch(FancyArrowPatch((x_ab[1] + R_, ym), (x_r - R_, ym), arrowstyle="->", color=RES, lw=1.6, mutation_scale=10))
    if stage == 3:
        for c in range(NR):
            if c != D:
                ax.add_patch(FancyArrowPatch((x_min[c] + R_, y_bits[0] - 3), (x_mout[c] - R_, y_bits[0] - 3), arrowstyle="->",
                                             color=REG[c], lw=0.9, mutation_scale=7, connectionstyle="arc3,rad=-0.12"))
    if stage == 4:
        ax.add_patch(FancyArrowPatch((x_r + R_, y_bits[2]), (x_mout[D] - R_, y_bits[2]), arrowstyle="->", color=RES, lw=1.6, mutation_scale=10))

fig, axes = plt.subplots(2, 3, figsize=(21, 9))
axes = axes.ravel()
draw(axes[0], 0, "t = 0   painted: MEM in, selectors, op, dst.  A, B, R, MEM out unresolved")
draw(axes[1], 1, "stage 1  LOAD:   A ← mem[selA],  B ← mem[selB]")
draw(axes[2], 2, "stage 2  EXECUTE:   R ← op(A, B)")
draw(axes[3], 3, "stage 3  COPY:   MEM out[c] ← MEM in[c]  for c ≠ dst")
draw(axes[4], 4, "stage 4  STORE:   MEM out[dst] ← R      → MEM out is the next tile's MEM in")
axes[5].axis("off")
axes[5].text(0, 0.95, f"tile v3 — ADD r0 r1 → r2 on [5, 9, 0, 0]\n\n"
             f"one tile: {x_mout[0] - x_min[0]} px wide × {H} px tall\n"
             f"(v2 was 58 × 61,  v1 was 48 × 127)\n\n"
             "sections, left → right, all on the same bit rows:\n"
             "  MEM in   4 cols × 4 bits, selA / selB rows above\n"
             "  A, B     operand slots, 1 col each\n"
             "  op       2×4 one-hot block\n"
             "  R        result slot, 1 col\n"
             "  MEM out  4 cols × 4 bits, dst row below\n\n"
             "four supervision targets in time order:\n"
             "  K1  A, B filled\n"
             "  K2  + R filled\n"
             "  K3  + MEM out non-dst columns filled\n"
             "  K4  + MEM out dst column = R\n\n"
             "type channels (fixed): data · sel_in · op · sel_out\n"
             "  + slot_a · slot_b · result  (or one 'alu' channel)\n\n"
             "every hop is now short and has a marker at both ends:\n"
             "  load   selA-marked column → A slot      (~30 px)\n"
             "  exec   A,B → R across the op block       (~30 px)\n"
             "  copy   MEM in col c → MEM out col c      (~76 px, straight)\n"
             "  store  R → dst-marked column             (~20 px)",
             fontsize=9.5, va="top", family="monospace")
fig.suptitle("Tile v3 — explicit LOAD / EXECUTE / COPY / STORE, each its own supervision stage", fontsize=14, weight="bold")
fig.tight_layout()
p = Path(__file__).resolve().parent.parent / "tile_v3_design.png"
fig.savefig(p, dpi=140, facecolor="white"); print("wrote", p, f"tile {x_mout[0]-x_min[0]}x{H}")
