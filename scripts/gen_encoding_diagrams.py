#!/usr/bin/env python3
"""
Generate clean spatial encoding diagrams for the NCPU paper.

Three separate SVGs:
  fig_encoding_gate.svg    — 2-input gate (encoding concept)
  fig_encoding_adder.svg   — 4-bit adder, two-column A+B layout
  fig_encoding_alu.svg     — 8-bit ALU, A + B + opcode layout

Usage:
    uv run python scripts/gen_encoding_diagrams.py

Saves to docs/paper-assets/diagrams/  (deletes old versions first)
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

OUT = Path("docs/paper-assets/diagrams")
OUT.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "white",
    "savefig.facecolor": "white",
    "svg.fonttype":      "none",
    "font.family":       "sans-serif",
    "font.size":         9,
})

BG    = "#e5e5e5"   # grid background
CINP  = "#2c7bb6"   # blue  — input operand
COP   = "#1a9641"   # green — opcode
COUT  = "#d7191c"   # red   — output
CBIT1 = "#ffffff"   # bit = 1 fill
CBIT0 = "#1c1c1c"   # bit = 0 fill
CGRID = "#c0c0c0"   # grid border


# ── Sizing helpers ─────────────────────────────────────────────────────────────

def col_height(n, r, sp):
    """Total height of n circles with radius r and spacing sp between them."""
    return n * 2 * r + (n - 1) * sp


def safe_H(n_max, r, sp, v_pad=None):
    """Grid height that fits n_max circles with comfortable vertical padding."""
    v_pad = v_pad if v_pad is not None else max(10, r * 1.8)
    return col_height(n_max, r, sp) + 2 * int(v_pad)


# ── Drawing primitives ─────────────────────────────────────────────────────────

def _circle(ax, cx, cy, r, bit, role_color, lw=1.1, zorder=3):
    ax.add_patch(mpatches.Circle(
        (cx, cy), r,
        facecolor=CBIT1 if bit else CBIT0,
        edgecolor=role_color,
        linewidth=lw, zorder=zorder,
    ))


def _bit_label(ax, cx, cy, bit, r, zorder=4):
    fs    = max(6.5, r * 1.55)
    color = "#666666" if bit else "#999999"
    ax.text(cx, cy, str(bit), ha="center", va="center",
            fontsize=fs, color=color, fontweight="bold", zorder=zorder)


def _col(ax, bits, cx, H, r, sp, role_color, lw=1.1):
    """Draw a vertical column of bit circles; return list of (cx, cy) tuples."""
    v  = col_height(len(bits), r, sp)
    tm = (H - v) / 2.0
    cys = []
    for i, b in enumerate(bits):
        cy = tm + r + i * (2 * r + sp)
        _circle(ax, cx, cy, r, b, role_color, lw=lw)
        _bit_label(ax, cx, cy, b, r)
        cys.append(cy)
    return cys


def _col_label(ax, cx, cys, text, color, r, fs=9.5):
    """Column label just above the topmost circle; clip_on=False so it's never hidden."""
    y = cys[0] - r - 4
    ax.text(cx, y, text,
            ha="center", va="bottom",
            fontsize=fs, color=color, fontweight="bold",
            clip_on=False)


def _grid_axes(ax, W, H):
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)       # y-down: bit 0 (MSB) at top
    ax.set_aspect("equal")
    ax.set_facecolor(BG)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), W, H,
        boxstyle="square,pad=0",
        linewidth=1.0, edgecolor=CGRID, facecolor=BG, zorder=0,
    ))
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _bit_legend(fig):
    """Bit encoding legend as figure-level legend below the axes."""
    handles = [
        mpatches.Patch(facecolor=CBIT1, edgecolor="#777777",
                       linewidth=0.8, label="circle  =  bit 1"),
        mpatches.Patch(facecolor=CBIT0, edgecolor="#777777",
                       linewidth=0.8, label="circle  =  bit 0"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=7.5, framealpha=0.0, edgecolor="none",
               bbox_to_anchor=(0.5, 0.0),
               handlelength=1.2, handleheight=0.9)


def _save(fig, name):
    for ext in ("svg", "pdf"):
        old = OUT / f"{name}.{ext}"
        old.unlink(missing_ok=True)
    out_path = OUT / f"{name}.pdf"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Panel A: XOR gate (2-in, 1-out) ───────────────────────────────────────────

def make_gate_diagram():
    r, sp  = 8, 6
    H = safe_H(2, r, sp, v_pad=14)
    W = 80

    # push columns well away from the walls (≥ 2r gap on each side)
    x_in  = 24
    x_out = W - 24

    a_bits  = [0, 1]
    out_bit = [1]

    target_ax_w = 2.6
    scale  = target_ax_w / W
    lm = 0.45; rm = 0.22        # only need small rm — no right-side text
    tm = 0.52; bm = 0.38

    fig_w = W * scale + lm + rm
    fig_h = H * scale + tm + bm

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=lm / fig_w, right=1 - rm / fig_w,
        bottom=bm / fig_h, top=1 - tm / fig_h,
    )
    _grid_axes(ax, W, H)

    in_cys  = _col(ax, a_bits,  x_in,  H, r, sp, CINP, lw=1.3)
    out_cys = _col(ax, out_bit, x_out, H, r, sp, COUT, lw=1.3)

    # "a" / "b" labels sit left of input circles, well inside the left margin
    for cy, lbl in zip(in_cys, ["a", "b"]):
        ax.text(x_in - r - 5, cy, lbl,
                ha="right", va="center", fontsize=9.5, color=CINP,
                fontweight="bold", clip_on=False)

    _col_label(ax, x_in,  in_cys,  "inputs", CINP, r)
    _col_label(ax, x_out, out_cys, "output", COUT, r)

    _bit_legend(fig)
    ax.set_title("Logic gate  (XOR: 2 inputs, 1 output)",
                 fontsize=10, pad=8, loc="center")

    _save(fig, "fig_encoding_gate")


# ── Panel B: 4-bit adder (cols2: A + B → 5-bit sum) ──────────────────────────

def make_adder_diagram():
    r, sp  = 6, 3
    H = safe_H(5, r, sp, v_pad=13)   # 5-bit sum needs the most space: 5*12+4*3+26=94
    W = 90

    step  = 2 * r + sp    # 15 — pitch between A and B columns
    x_a   = 18
    x_b   = x_a + step    # 33
    x_sum = W - 18

    a_bits   = [1, 0, 1, 1]    # 11
    b_bits   = [0, 1, 1, 0]    #  6  →  sum = 17 = 10001
    sum_bits = [1, 0, 0, 0, 1]

    target_ax_w = 3.0
    scale = target_ax_w / W
    lm = 0.22; rm = 0.22    # no right-side text
    tm = 0.52; bm = 0.38

    fig_w = W * scale + lm + rm
    fig_h = H * scale + tm + bm

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=lm / fig_w, right=1 - rm / fig_w,
        bottom=bm / fig_h, top=1 - tm / fig_h,
    )
    _grid_axes(ax, W, H)

    a_cys   = _col(ax, a_bits,   x_a,   H, r, sp, CINP, lw=1.3)
    b_cys   = _col(ax, b_bits,   x_b,   H, r, sp, CINP, lw=1.3)
    sum_cys = _col(ax, sum_bits, x_sum, H, r, sp, COUT, lw=1.3)

    _col_label(ax, x_a,   a_cys,   "A", CINP, r)
    _col_label(ax, x_b,   b_cys,   "B", CINP, r)
    _col_label(ax, x_sum, sum_cys, "S", COUT, r)

    _bit_legend(fig)
    ax.set_title("4-bit adder  (A + B → 5-bit sum)",
                 fontsize=10, pad=8, loc="center")

    _save(fig, "fig_encoding_adder")


# ── Panel C: 8-bit ALU (A + B + opcode → result) ──────────────────────────────

def make_alu_diagram():
    r, sp  = 4, 3
    H = safe_H(8, r, sp, v_pad=11)   # 8*8+7*3+22 = 106
    W = 108

    step_ab = 2 * r + sp    # 11 — pitch between A and B
    x_a     = 18
    x_b     = x_a + step_ab  # 29
    x_op    = W // 2          # 54 — opcode column centred in grid
    x_res   = W - 18          # 90

    a_bits   = [1, 0, 1, 1, 0, 1, 0, 1]   # 0xB5
    b_bits   = [0, 1, 1, 0, 1, 1, 1, 0]   # 0x6E
    op_bits  = [0, 0, 0]                   # ADD = 0b000
    res_bits = [0, 0, 1, 0, 0, 0, 1, 1]   # 0x23

    target_ax_w = 3.6
    scale = target_ax_w / W
    lm = 0.20; rm = 0.20    # no right-side text
    tm = 0.52; bm = 0.38

    fig_w = W * scale + lm + rm
    fig_h = H * scale + tm + bm

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.subplots_adjust(
        left=lm / fig_w, right=1 - rm / fig_w,
        bottom=bm / fig_h, top=1 - tm / fig_h,
    )
    _grid_axes(ax, W, H)

    a_cys   = _col(ax, a_bits,   x_a,   H, r, sp, CINP, lw=1.2)
    b_cys   = _col(ax, b_bits,   x_b,   H, r, sp, CINP, lw=1.2)
    op_cys  = _col(ax, op_bits,  x_op,  H, r, sp, COP,  lw=1.5)
    res_cys = _col(ax, res_bits, x_res, H, r, sp, COUT, lw=1.2)

    _col_label(ax, x_a,   a_cys,   "A",  CINP, r)
    _col_label(ax, x_b,   b_cys,   "B",  CINP, r)
    _col_label(ax, x_op,  op_cys,  "op", COP,  r)
    _col_label(ax, x_res, res_cys, "R",  COUT, r)

    _bit_legend(fig)
    ax.set_title("8-bit ALU  (A + B + opcode → result)",
                 fontsize=10, pad=8, loc="center")

    _save(fig, "fig_encoding_alu")


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    make_gate_diagram()
    make_adder_diagram()
    make_alu_diagram()
    print(f"\nAll saved to {OUT}/")
