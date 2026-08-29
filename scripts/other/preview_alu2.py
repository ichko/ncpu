#!/usr/bin/env python3
"""
Preview the ALU v2 grid layout.

Saves alu2_layout_preview.png to the project root.

Usage:
    uv run python scripts/other/preview_alu2.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.dataset import ALU2Dataset, ALU2_COND_NAMES, ALU2_OP_NAMES, _compute_alu2, _int_to_bits_msb

# ── Geometry ───────────────────────────────────────────────────────────────────

W, H, R, AMONG_SP = 96, 112, 4, 2
X_A, X_B, X_CTRL, X_OUT = 12, 24, 48, 82
STEP = 2 * R + AMONG_SP   # 10 px

ds = ALU2Dataset()

# Column label info
COL_INFO = {
    "A":    dict(x=X_A,    n=8,  label="A\n(8b)",   color="#4E79A7"),
    "B":    dict(x=X_B,    n=8,  label="B\n(8b)",   color="#F28E2B"),
    "CTRL": dict(x=X_CTRL, n=7,  label="CTRL\n(7b)", color="#59A14F"),
    "OUT":  dict(x=X_OUT,  n=10, label="OUT\n(10b)", color="#E15759"),
}

# Bit label lists (top to bottom)
A_BITS_LABELS    = [f"A[{7-i}]" for i in range(8)]
B_BITS_LABELS    = [f"B[{7-i}]" for i in range(8)]
CTRL_BITS_LABELS = ["op[2]", "op[1]", "op[0]", "cin", "cc[2]", "cc[1]", "cc[0]"]
OUT_BITS_LABELS  = [f"R[{7-i}]" for i in range(8)] + ["cout", "BT"]


def bit_centers(n, x, H):
    """Return list of (cx, cy) for n circles centred at column x."""
    v  = n * 2 * R + AMONG_SP * (n - 1)
    tm = (H - v) // 2
    return [(x, tm + R + i * STEP) for i in range(n)]


# ── Figure 1: annotated blank layout ──────────────────────────────────────────

def blank_screen():
    from ncpu.utils import make_alu2_screen
    return make_alu2_screen(H, W, R, AMONG_SP, X_A, X_B, X_CTRL, X_OUT)


fig1, ax = plt.subplots(figsize=(10, 5))
ax.imshow(blank_screen(), cmap="gray", vmin=0, vmax=255, origin="upper",
          extent=[0, W, H, 0], aspect="equal")
ax.set_xlim(-30, W + 80)
ax.set_ylim(H + 8, -20)

# Draw column bracket + label + bit labels
for name, info in COL_INFO.items():
    cx, n, lbl, col = info["x"], info["n"], info["label"], info["color"]
    centers = bit_centers(n, cx, H)
    labels  = {"A": A_BITS_LABELS, "B": B_BITS_LABELS,
                "CTRL": CTRL_BITS_LABELS, "OUT": OUT_BITS_LABELS}[name]

    # Bracket: vertical line + ticks
    ys = [c[1] for c in centers]
    ax.plot([cx + R + 3] * 2, [ys[0], ys[-1]], color=col, lw=1.5)
    ax.plot([cx + R + 3, cx + R + 7], [ys[0],  ys[0]],  color=col, lw=1.5)
    ax.plot([cx + R + 3, cx + R + 7], [ys[-1], ys[-1]], color=col, lw=1.5)

    # Column header above first circle
    ax.text(cx, ys[0] - R - 6, lbl, ha="center", va="bottom",
            fontsize=7, color=col, fontweight="bold")

    # Bit labels to the right
    for (bx, by), bl in zip(centers, labels):
        ax.text(bx + R + 9, by, bl, ha="left", va="center",
                fontsize=5.5, color="#333333")

# Horizontal distance arrows
arrow_kw = dict(arrowstyle="<->", color="#888888", lw=1)
for x0, x1, y, lbl in [
    (X_A,    X_CTRL,  H + 4, f"A→CTRL  {X_CTRL - X_A}px"),
    (X_CTRL, X_OUT,   H + 4, f"CTRL→OUT  {X_OUT - X_CTRL}px"),
]:
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", color="#666666", lw=1.2))
    ax.text((x0 + x1) / 2, y + 3, lbl, ha="center", va="top",
            fontsize=6, color="#555555")

ax.set_title("ALU v2 grid layout — 96×112, r=4, step=10px", fontsize=10)
ax.axis("off")
fig1.tight_layout()

# ── Figure 2: example inputs + outputs for all 8 ops ─────────────────────────

OPS_TO_SHOW = list(range(8))
N_OPS       = len(OPS_TO_SHOW)

fig2, axes = plt.subplots(N_OPS, 2, figsize=(6, N_OPS * 1.35),
                           gridspec_kw={"wspace": 0.04, "hspace": 0.08})

# Fixed operands for reproducible display
A_FIXED  = 0b10110101   # 181
B_FIXED  = 0b01101110   # 110
CIN      = 1
COND     = 2            # CS (branch if carry set)

from ncpu.utils import make_alu2_screen

for row, op in enumerate(OPS_TO_SHOW):
    result, cout, branch = _compute_alu2(A_FIXED, B_FIXED, CIN, op, COND)

    ctrl_bits = _int_to_bits_msb(op, 3) + [CIN] + _int_to_bits_msb(COND, 3)
    out_bits  = _int_to_bits_msb(result, 8) + [cout, branch]
    uses_b    = op not in (5, 6, 7)

    inp_screen = make_alu2_screen(
        H, W, R, AMONG_SP, X_A, X_B, X_CTRL, X_OUT,
        a_bits    = _int_to_bits_msb(A_FIXED, 8),
        b_bits    = _int_to_bits_msb(B_FIXED, 8) if uses_b else None,
        ctrl_bits = ctrl_bits,
    )
    out_screen = make_alu2_screen(
        H, W, R, AMONG_SP, X_A, X_B, X_CTRL, X_OUT,
        out_bits = out_bits,
    )

    for col, (screen, title_sfx) in enumerate([(inp_screen, "input"),
                                                (out_screen, "output")]):
        ax = axes[row, col]
        ax.imshow(screen, cmap="gray", vmin=0, vmax=255,
                  origin="upper", aspect="equal", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            # Op label on the left
            flag_str = f"R={result:#04x} cout={cout} BT={branch}"
            ax.set_ylabel(f"op{op}: {ALU2_OP_NAMES[op]}\n{flag_str}",
                          fontsize=5.5, rotation=0, labelpad=90, va="center")
        if row == 0:
            ax.set_title(title_sfx, fontsize=8, pad=3)

fig2.suptitle(
    f"ALU v2 — A={A_FIXED:#04x}  B={B_FIXED:#04x}  cin={CIN}  cond={COND}({ALU2_COND_NAMES[COND]})",
    fontsize=9, y=1.005,
)

# ── Save both into one combined image ─────────────────────────────────────────

out_path = Path("alu2_layout_preview.png")
fig1.savefig(out_path.with_stem("alu2_layout_annotated"), dpi=140, bbox_inches="tight")
fig2.savefig(out_path.with_stem("alu2_ops_preview"),      dpi=140, bbox_inches="tight")
plt.close("all")

# Combine with PIL
from PIL import Image
img1 = Image.open(out_path.with_stem("alu2_layout_annotated"))
img2 = Image.open(out_path.with_stem("alu2_ops_preview"))
# Stack horizontally (resize img2 to match img1 height)
h1 = img1.height
h2 = img2.height
scale = h1 / h2
w2_new = int(img2.width * scale)
img2_r = img2.resize((w2_new, h1), Image.LANCZOS)
combined = Image.new("RGB", (img1.width + w2_new + 20, h1), (255, 255, 255))
combined.paste(img1, (0, 0))
combined.paste(img2_r, (img1.width + 20, 0))
combined.save(out_path)
print(f"Saved: {out_path}  ({combined.width}×{combined.height})")
print(f"Also saved:  alu2_layout_annotated.png  &  alu2_ops_preview.png")
