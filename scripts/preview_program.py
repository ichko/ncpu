#!/usr/bin/env python3
"""Preview + round-trip test for the painted-program layout.

    uv run python scripts/preview_program.py
    uv run python scripts/preview_program.py --instr 4 --regs 4 --word-bits 4

Prints the compiled program, the reference register trace, a compiler/decoder
round-trip check, and writes ``program_preview.png`` to the project root showing
the input screen and every supervision target.
"""

import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.program import (  # noqa: E402
    OPS,
    Instr,
    ProgramLayout,
    run_program,
    sample_program,
    sample_regs,
)

parser = ArgumentParser()
parser.add_argument("--instr", type=int, default=3)
parser.add_argument("--regs", type=int, default=4)
parser.add_argument("--word-bits", type=int, default=8)
parser.add_argument("--radius", type=float, default=2.5)
parser.add_argument("--pitch", type=int, default=6, help="vertical blob spacing")
parser.add_argument("--col-pitch", type=int, default=22, help="horizontal column spacing")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--settle", type=int, default=40, help="K: CA steps allowed per gap")
parser.add_argument("--trials", type=int, default=200, help="round-trip trials")
parser.add_argument("--out", default="program_preview.png")
args = parser.parse_args()

rng = random.Random(args.seed)
layout = ProgramLayout(
    n_instr=args.instr,
    n_regs=args.regs,
    word_bits=args.word_bits,
    r=args.radius,
    pitch=args.pitch,
    col_pitch=args.col_pitch,
)

print(f"grid {layout.H}x{layout.W}  ({layout.data_rows} data rows, {layout.op_rows} opcode rows, {layout.n_cols} columns)")
print(f"data columns {layout.data_col}, opcode columns {layout.op_col}\n")

# ── an example program ──────────────────────────────────────────────────────
program = sample_program(args.instr, args.regs, rng=rng)
regs = sample_regs(args.regs, args.word_bits, rng=rng)
trace = run_program(regs, program, args.word_bits)

print("program:")
for i, ins in enumerate(program):
    print(f"  opcode col {layout.op_col[i]}: {ins}")
print("\nreference trace (one row per data column):")
for i, t in enumerate(trace):
    print(f"  col {i}: {list(t)}")

inp = layout.input_screen(program, regs)
targets = layout.stage_targets(program, regs)

# ── round trip: paint then decode, over random programs ─────────────────────
bad = 0
for _ in range(args.trials):
    p = sample_program(args.instr, args.regs, rng=rng)
    r0 = sample_regs(args.regs, args.word_bits, rng=rng)
    tr = run_program(r0, p, args.word_bits)
    final = layout.paint(p, r0, resolved=args.instr)
    if layout.decode(final) != tr:
        bad += 1
    # the input screen must expose column 0 and nothing downstream
    dec_in = layout.decode(layout.paint(p, r0, resolved=0))
    if dec_in[0] != tr[0] or any(c != (None,) * args.regs for c in dec_in[1:]):
        bad += 1
print(f"\nround trip: {args.trials} programs, {bad} mismatches "
      f"({'PASS' if bad == 0 else 'FAIL'})")

# ── how the supervision lines up ────────────────────────────────────────────
print("\nsupervision schedule (K = %d steps per gap):" % args.settle)
print(f"  t=0            implant input screen (col 0 filled, cols 1..{args.instr} unknown)")
for i in range(args.instr):
    cols = ", ".join(str(c) for c in range(i + 2))
    print(f"  t={(i + 1) * args.settle:<4}         supervise target {i}: data columns {cols} correct")
print(f"  total rollout: {args.instr * args.settle} steps")

# ── propagation budget ──────────────────────────────────────────────────────
# A 3x3 perception moves information 1 px per step, so a gap cannot finish
# faster than the longest path it must carry a bit along.
vspan = (layout.n_regs - 1) * layout.word_bits * layout.pitch   # reg 0 <-> reg N-1
hspan = layout.col_x[layout.data_col[1]] - layout.col_x[layout.data_col[0]]
print("\nper-gap propagation budget (3x3 perception = 1 px/step):")
print(f"  worst vertical hop  (r0 <-> r{layout.n_regs - 1}): {vspan} px")
print(f"  horizontal hop      (data col -> data col): {hspan} px")
print(f"  lower bound on K:   {vspan + hspan} steps   (you asked for K={args.settle})")

# ── field map ───────────────────────────────────────────────────────────────
KINDS = {"data": 1.0, "op": -0.5}
fieldmap = np.zeros((layout.H, layout.W), dtype=np.float32)
for c, kind in layout.col_kind.items():
    for j in range(len(layout.col_y[c])):
        yy, xx = layout._disc_idx[(c, j)]
        fieldmap[yy, xx] = KINDS[kind]

# ── figure ──────────────────────────────────────────────────────────────────
panels = [("field map", fieldmap, -1), ("blob sites (diagnostic)", layout.slot_mask(), -1),
          ("input  (t=0)", inp, 0)] + [
    (f"target {i}  (t={(i + 1) * args.settle})", t, i + 1) for i, t in enumerate(targets)
]
fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 4.6))
if len(panels) == 1:
    axes = [axes]
for ax, (title, img, resolved) in zip(axes, panels):
    cmap = "viridis" if title == "field map" else "gray"
    ax.imshow(img, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    # ring the data columns that this panel supervises
    for i in range(layout.n_instr + 1):
        x = layout.col_x[layout.data_col[i]]
        if resolved < 0:
            continue
        style = dict(color="#39d353", lw=1.6) if i <= resolved else dict(color="#666", lw=0.8, ls=":")
        ax.add_patch(plt.Rectangle(
            (x - layout.r - 2, layout.data_y[0] - layout.r - 2),
            2 * layout.r + 4, layout.data_y[-1] - layout.data_y[0] + 2 * layout.r + 4,
            fill=False, **style))
fig.suptitle(
    "painted program:  data | opcode | data | opcode | data     green = supervised, dotted = still unknown",
    fontsize=9)
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / args.out
fig.savefig(out, dpi=150)
print(f"\nwrote {out}")
