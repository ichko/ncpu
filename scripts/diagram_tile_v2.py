#!/usr/bin/env python3
"""Design sketch for tile layout v2 (not implemented yet).

Registers side by side (one column per register, bits top->bottom MSB first).
src A / src B selectors are rows ABOVE the input grid, one above the other so
their order is fixed; dst selector is a row BELOW the output grid.  Every
selector blob sits directly above/below the register column it names.  The
opcode is a compact 2x4 one-hot block between the grids.
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

# ── example (same as before) ────────────────────────────────────────────────
OP, A, B, D = "ADD", 0, 1, 2
regs = (5, 9, 0, 0)
out = (5, 9, 14, 0)
NR, WB = 4, 4

# ── geometry (px) ───────────────────────────────────────────────────────────
R, P, PAD, GAP = 2, 6, 8, 14          # blob radius, pitch, margin, grid<->op gap
xin = [PAD + c * P for c in range(NR)]                   # input register columns
xop = [xin[-1] + GAP + c * P for c in range(2)]          # 2 op columns
xout = [xop[-1] + GAP + c * P for c in range(NR)]        # output register columns
y_selA = PAD
y_selB = PAD + P
y_bits = [y_selB + P + 4 + b * P for b in range(WB)]     # MSB first
y_dst = y_bits[-1] + P + 4
H = y_dst + PAD + 1
W = xout[-1] + PAD + 1

REG = ["#e15759", "#f28e2b", "#59a14f", "#4e79a7"]
SEL_A, SEL_B, SEL_D, OPC = "#ff9da7", "#9c755f", "#bab0ac", "#b07aa1"

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(20, 9), gridspec_kw={"width_ratios": [1.3, 1]})
ax.set_xlim(-24, W + 40); ax.set_ylim(H + 10, -22); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Rectangle((0, 0), W, H, fc="#f4f4f4", ec="k", lw=1.2))

def blob(x, y, col, filled, r=R, lw=1.8, ls="-"):
    ax.add_patch(Circle((x, y), r, fc=col if filled else "white", ec=col, lw=lw, ls=ls))

# input register grid
for c in range(NR):
    bits = [(regs[c] >> (WB - 1 - b)) & 1 for b in range(WB)]
    for b in range(WB):
        blob(xin[c], y_bits[b], REG[c], bits[b])
    ax.text(xin[c], y_bits[-1] + P + 1, f"r{c}", ha="center", va="top", fontsize=9, color=REG[c], weight="bold")
for b in range(WB):
    ax.text(xin[0] - R - 1.5, y_bits[b] + 0.7, f"bit {WB-1-b}", ha="right", va="center", fontsize=7.5, color="#555")

# selectors above input: A on top row, B just below -> order is geometric
for c in range(NR):
    blob(xin[c], y_selA, SEL_A, c == A)
    blob(xin[c], y_selB, SEL_B, c == B)
ax.text(xin[0] - R - 1.5, y_selA + 0.7, "src A", ha="right", va="center", fontsize=8, color=SEL_A, weight="bold")
ax.text(xin[0] - R - 1.5, y_selB + 0.7, "src B", ha="right", va="center", fontsize=8, color=SEL_B, weight="bold")
ax.add_patch(FancyArrowPatch((xin[A], y_selA + R + 0.5), (xin[A], y_bits[0] - R - 0.8), arrowstyle="-|>", color=SEL_A, lw=1.2, mutation_scale=9))
ax.add_patch(FancyArrowPatch((xin[B], y_selB + R + 0.5), (xin[B], y_bits[0] - R - 0.8), arrowstyle="-|>", color=SEL_B, lw=1.2, mutation_scale=9))

# opcode: 2x4 one-hot block, rows aligned with the bit rows
for k, name in enumerate(OPS):
    c, r = divmod(k, WB)
    blob(xop[c], y_bits[r], OPC, name == OP)
    ax.text(xop[c] + (R + 1.3 if c == 1 else -(R + 1.3)), y_bits[r] + 0.7, name,
            ha="left" if c == 1 else "right", va="center", fontsize=6.5, color=OPC)
ax.text((xop[0] + xop[1]) / 2, y_bits[0] - P - 1, "op", ha="center", va="bottom", fontsize=9, color=OPC, weight="bold")

# output grid: small dots = still to be filled; the dst column shown as the target it must become
for c in range(NR):
    bits = [(out[c] >> (WB - 1 - b)) & 1 for b in range(WB)]
    for b in range(WB):
        blob(xout[c], y_bits[b], REG[c], False, r=R, lw=1.2, ls=(0, (1.5, 1.5)))
        ax.add_patch(Circle((xout[c], y_bits[b]), 0.55, fc=REG[c], ec="none"))
        if c == D:   # ghost of what it must become
            blob(xout[c], y_bits[b], REG[c], bits[b], r=R * 0.5, lw=1)
    ax.text(xout[c], y_bits[-1] + P + 1, f"r{c}", ha="center", va="top", fontsize=9, color=REG[c], weight="bold")

# dst selector below output
for c in range(NR):
    blob(xout[c], y_dst, SEL_D, c == D)
ax.text(xout[-1] + R + 1.5, y_dst + 0.7, "dst", ha="left", va="center", fontsize=8, color="#777", weight="bold")
ax.add_patch(FancyArrowPatch((xout[D], y_dst - R - 0.5), (xout[D], y_bits[-1] + R + 0.8), arrowstyle="-|>", color=SEL_D, lw=1.2, mutation_scale=9))

# flow arrows
ymid = (y_bits[0] + y_bits[-1]) / 2
ax.add_patch(FancyArrowPatch((xin[-1] + R + 1, ymid), (xop[0] - R - 1, ymid), arrowstyle="->", color="#444", lw=1.4, mutation_scale=11))
ax.add_patch(FancyArrowPatch((xop[-1] + R + 1, ymid), (xout[0] - R - 1, ymid), arrowstyle="->", color="#444", lw=1.4, mutation_scale=11))
ax.add_patch(FancyArrowPatch((xout[-1] + R + 3, ymid), (W + 14, ymid), arrowstyle="->", color="k", lw=2, mutation_scale=15))
ax.text(W + 16, ymid + 0.7, "→ INPUT of\nnext tile", fontsize=9, va="center", weight="bold")

# headers
ax.text(np.mean(xin), -3, "INPUT\nr0 r1 r2 r3 side by side\nbits top→bottom (MSB first)", ha="center", va="bottom", fontsize=9.5, weight="bold")
ax.text(np.mean(xop), -3, "OP", ha="center", va="bottom", fontsize=9.5, weight="bold", color=OPC)
ax.text(np.mean(xout), -3, "OUTPUT\ndots = unresolved (0)\nmust become " + str(list(out)), ha="center", va="bottom", fontsize=9.5, weight="bold")
ax.text(np.mean(xin), H + 4, f"= {list(regs)}", ha="center", va="top", fontsize=9)
ax.text(W / 2, -17, f"ADD r0 r1 → r2      tile {xout[0]-xin[0]} px wide × {H} px tall  (was 48 × 127)      "
        "blob r=2, pitch 6      selectors sit directly above / below the register column they name",
        ha="center", fontsize=8.5, color="#333")

# ── right: grayscale of what the NCA would see at t=0 ───────────────────────
img = np.zeros((H, W), np.float32)
yy, xx = np.ogrid[:H, :W]
def stamp(x, y, v, r=R):
    img[(xx - x) ** 2 + (yy - y) ** 2 <= r * r] = v
for c in range(NR):
    for b in range(WB):
        stamp(xin[c], y_bits[b], 1.0 if (regs[c] >> (WB - 1 - b)) & 1 else -1.0)
    stamp(xin[c], y_selA, 0.5 if c == A else -0.5)
    stamp(xin[c], y_selB, 0.5 if c == B else -0.5)
    stamp(xout[c], y_dst, 0.5 if c == D else -0.5)
for k, name in enumerate(OPS):
    c, r = divmod(k, WB)
    stamp(xop[c], y_bits[r], 0.5 if name == OP else -0.5)
ax2.imshow(img, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
ax2.set_title(f"what the NCA sees at t=0  ({H}×{W} px, channel 0)", fontsize=10)
ax2.set_xticks([]); ax2.set_yticks([])
for t, x in (("src A", xin[0]), ("src B", xin[0])):
    pass
ax2.text(xin[0] - 3, y_selA, "A", ha="right", va="center", fontsize=8, color="w")
ax2.text(xin[0] - 3, y_selB, "B", ha="right", va="center", fontsize=8, color="w")
ax2.text(xout[-1] + 3, y_dst, "dst", ha="left", va="center", fontsize=8, color="w")
ax2.text(np.mean(xin), H - 1, "input", ha="center", va="top", fontsize=8)
ax2.text(np.mean(xop), H - 1, "op", ha="center", va="top", fontsize=8)
ax2.text(np.mean(xout), H - 1, "output", ha="center", va="top", fontsize=8)

fig.suptitle("Tile layout v2 — registers side by side, selectors aligned to their register column", fontsize=14, weight="bold")
fig.tight_layout()
p = Path(__file__).resolve().parent.parent / "tile_v2_design.png"
fig.savefig(p, dpi=150, facecolor="white"); print("wrote", p, f"tile {H}x{W}")
