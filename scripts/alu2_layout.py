#!/usr/bin/env python3
"""
Fast standalone ALU v2 layout diagram — no model loading, runs in ~1s.

Usage:
    uv run python scripts/alu2_layout.py
    uv run python scripts/alu2_layout.py && uv run python -c \
      "import cairosvg; cairosvg.svg2pdf(url='results/E_alu2/layout_diagram.svg', \
       write_to='docs/tex/images/alu/layout_diagram.pdf')"
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT_DIR = Path("results/E_alu2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Grid geometry (must match config) ─────────────────────────────────────────

H_D, W_D = 112, 96
r_d       = 4
sp        = 2          # among_sp
x_a       = 12
x_b       = 24
x_ctrl    = 48
x_out     = 82
step_px   = 2 * r_d + sp

COL_A   = "#4E79A7"
COL_B   = "#F28E2B"
COL_OP  = "#59A14F"
COL_OUT = "#E15759"
COL_BG  = "#f7f7f7"

# ── Helpers ────────────────────────────────────────────────────────────────────

def _col_centers(n, cx):
    v  = n * 2 * r_d + sp * (n - 1)
    tm = (H_D - v) // 2
    return [(cx, tm + r_d + i * step_px) for i in range(n)]

def _draw_circle(ax, cx, cy, r, color, label, fontsize=6.5):
    ax.add_patch(mpatches.Circle((cx, cy), r, color=color, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold", zorder=4)

# ── Figure ─────────────────────────────────────────────────────────────────────

scale        = 0.028
p_w          = W_D * scale
p_h          = H_D * scale
lm, rm       = 0.10, 0.10
tm_m, bm_m   = 0.05, 0.35
fig_w        = p_w + lm + rm
fig_h        = p_h + tm_m + bm_m

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.subplots_adjust(
    left   = lm / fig_w,
    right  = 1 - rm / fig_w,
    bottom = bm_m / fig_h,
    top    = 1 - tm_m / fig_h,
)

ax.set_xlim(0, W_D); ax.set_ylim(H_D, 0); ax.set_aspect("equal")
ax.set_facecolor(COL_BG)
ax.add_patch(mpatches.FancyBboxPatch((0, 0), W_D, H_D,
    boxstyle="square,pad=0", linewidth=1, edgecolor="#aaaaaa", facecolor=COL_BG))
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

for (cx, cy), lbl in zip(_col_centers(8, x_a), [f"A{7-i}" for i in range(8)]):
    _draw_circle(ax, cx, cy, r_d, COL_A, lbl)
for (cx, cy), lbl in zip(_col_centers(8, x_b), [f"B{7-i}" for i in range(8)]):
    _draw_circle(ax, cx, cy, r_d, COL_B, lbl)
ctrl_labels = ["op2", "op1", "op0", "ci", "c2", "c1", "c0"]
for (cx, cy), lbl in zip(_col_centers(7, x_ctrl), ctrl_labels):
    _draw_circle(ax, cx, cy, r_d, COL_OP, lbl)
out_labels = [f"r{7-i}" for i in range(8)] + ["co", "bt"]
for (cx, cy), lbl in zip(_col_centers(10, x_out), out_labels):
    _draw_circle(ax, cx, cy, r_d, COL_OUT, lbl)

handles = [
    mpatches.Patch(color=COL_A,   label="A"),
    mpatches.Patch(color=COL_B,   label="B"),
    mpatches.Patch(color=COL_OP,  label="Control"),
    mpatches.Patch(color=COL_OUT, label="Output"),
]
fig.legend(handles=handles, loc="lower center", fontsize=9, ncol=4,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.01),
           handlelength=1.0, handletextpad=0.3, columnspacing=0.6)

out = OUT_DIR / "layout_diagram.svg"
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved: {out}")
