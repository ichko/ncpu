#!/usr/bin/env python3
"""Annotated diagram of the painted-program layout for a 3-instruction program.

Left: every blob coloured by what it means (register / opcode field), with the
instruction listing above each opcode column.  Right: the actual painted input
screen the NCA receives (only data column 0 filled).

    uv run python scripts/diagram_layout.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.program import OPS, Instr, ProgramLayout, run_program  # noqa: E402

L = ProgramLayout(n_instr=3, n_regs=4, word_bits=4, unknown=0.0, op_amp=0.5)
program = [Instr("ADD", 0, 1, 2), Instr("COPY", 2, 2, 3), Instr("XOR", 3, 0, 1)]
regs = (5, 9, 0, 0)
trace = run_program(regs, program, L.word_bits)

REG_COL = ["#e15759", "#f28e2b", "#59a14f", "#4e79a7"]          # r0..r3
FIELD_COL = {"op": "#b07aa1", "sel_a": "#ff9da7", "sel_b": "#9c755f", "sel_dst": "#bab0ac"}

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(22, 10), gridspec_kw={"width_ratios": [1.15, 1]})

# ── left: semantic map ────────────────────────────────────────────────────────
ax.set_xlim(-2, L.W + 34); ax.set_ylim(L.H + 2, -26); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Rectangle((0, 0), L.W, L.H, fc="#f4f4f4", ec="k", lw=1))

def blob(x, y, color, filled, label=None):
    ax.add_patch(Circle((x, y), L.r, fc=color if filled else "white", ec=color, lw=1.6))
    if label:
        ax.text(x + L.r + 1.2, y + 0.8, label, fontsize=6.5, va="center", color="#333")

for i in range(L.n_instr + 1):
    c = L.data_col[i]; x = L.col_x[c]
    vals = trace[i]
    for reg in range(L.n_regs):
        bits = [(vals[reg] >> (L.word_bits - 1 - b)) & 1 for b in range(L.word_bits)]
        for b in range(L.word_bits):
            y = L.data_y[L.row_of(reg, b)]
            known = i == 0
            blob(x, y, REG_COL[reg], filled=bool(bits[b]) if known else False)
            if i == 0 and b == 0:
                ax.text(x - L.r - 1.5, y + 0.8, f"r{reg} MSB", fontsize=7, ha="right", va="center", color=REG_COL[reg], weight="bold")
            if i == 0 and b == L.word_bits - 1:
                ax.text(x - L.r - 1.5, y + 0.8, "LSB", fontsize=6, ha="right", va="center", color=REG_COL[reg])
        if i > 0:
            ax.text(x, L.data_y[L.row_of(reg, 0)] - 4.5, "?" , fontsize=7, ha="center", color=REG_COL[reg])
    title = "data 0\n(input)" if i == 0 else f"data {i}\n= out of {i-1}"
    ax.text(x, -4, title, ha="center", va="bottom", fontsize=9, weight="bold")
    ax.text(x, L.H + 1.5, f"{list(vals)}" if i == 0 else "unknown (0)", ha="center", va="top", fontsize=7.5,
            color="k" if i == 0 else "#888")

for i, ins in enumerate(program):
    c = L.op_col[i]; x = L.col_x[c]
    for k, name in enumerate(OPS):
        y = L.op_y[L.op_row("op", k)]
        blob(x, y, FIELD_COL["op"], filled=(name == ins.op), label=name if i == 0 else None)
    for field, sel, tag in (("sel_a", ins.a, "A"), ("sel_b", ins.b, "B"), ("sel_dst", ins.dst, "D")):
        for k in range(L.n_regs):
            y = L.op_y[L.op_row(field, k)]
            blob(x, y, FIELD_COL[field], filled=(k == sel), label=f"{tag}=r{k}" if i == 0 else None)
    ax.text(x, -4, f"opcode {i}\n{ins.op} r{ins.a} r{ins.b} → r{ins.dst}", ha="center", va="bottom", fontsize=8.5,
            weight="bold", color=FIELD_COL["op"])
    # bracket the four fields
    for field, lab in (("op", "op (one-hot, 8)"), ("sel_a", "src A (one-hot, 4)"),
                       ("sel_b", "src B (one-hot, 4)"), ("sel_dst", "dst (one-hot, 4)")):
        if i == L.n_instr - 1:
            y0 = L.op_y[L.op_row(field, 0)]; n = len(OPS) if field == "op" else L.n_regs
            y1 = L.op_y[L.op_row(field, n - 1)]
            ax.plot([x + L.r + 2.5] * 2, [y0 - 2, y1 + 2], color=FIELD_COL[field], lw=2)
            ax.text(x + L.r + 4, (y0 + y1) / 2 + 0.8, lab, fontsize=7.5, va="center", color=FIELD_COL[field])

ax.text(L.W / 2, -19, "filled = +1 (bit 1 / selected)     hollow = −1 (bit 0 / not selected)     data blobs ±1, opcode blobs ±0.5     "
        "unresolved data = 0 (blank)", ha="center", fontsize=8.5, color="#333")
ax.text(L.W / 2, -23.5, f"grid {L.H}×{L.W} px · blob r={L.r} · row pitch {L.pitch} · column pitch {L.col_pitch} · "
        f"K={2*L.min_rollout(2)} steps per instruction", ha="center", fontsize=8.5, color="#333")

# ── right: what the NCA actually sees at t=0 ─────────────────────────────────
inp = L.input_screen(program, regs)
ax2.imshow(inp, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
ax2.set_title("painted input screen at t=0 (channel 0)", fontsize=10)
ax2.set_xticks([]); ax2.set_yticks([])
for i in range(L.n_instr + 1):
    ax2.text(L.col_x[L.data_col[i]], -3, f"d{i}", ha="center", fontsize=8)
for i in range(L.n_instr):
    ax2.text(L.col_x[L.op_col[i]], -3, f"op{i}", ha="center", fontsize=8, color=FIELD_COL["op"])
ax2.text(L.W / 2, L.H + 6, "reference trace:  " + "  →  ".join(str(list(t)) for t in trace), ha="center", fontsize=8.5)

fig.suptitle("Painted program layout — 3 instructions, 4 registers × 4 bits", fontsize=13, weight="bold")
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / "layout_diagram.png"
fig.savefig(out, dpi=150, facecolor="white")
print("wrote", out)
