#!/usr/bin/env python3
"""One input/output pair for tile v5, annotated: what is painted at t=0 and
what the final grid must be, with the register values written out."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.program import TileLayoutV5, Instr, run_program

L = TileLayoutV5(n_instr=2)
program = [Instr("SUB", 1, 0, 2), Instr("SUBLEQ", 0, 2, 3)]
regs = (5, 9, 0, 0)
trace = run_program(regs, program, 4)
inp, out = L.paint(program, regs, 0), L.paint(program, regs, L.n_stages)

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
for ax, img, title in zip(axes, (inp, out), ("INPUT  (t = 0, what we paint)", "OUTPUT  (final target)")):
    ax.imshow(img, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=11, weight="bold"); ax.set_xticks([]); ax.set_yticks([])
    for i in range(L.n_instr + 1):
        ax.text(np.mean(L.grid_x[i]), L.y_mem[0] - 9.5, f"mem {i}", ha="center", fontsize=8, color="w")
        for c in range(L.n_regs):
            ax.text(L.grid_x[i][c], L.y_mem[-1] + 3.5, f"r{c}", ha="center", va="top", fontsize=6, color="#ddd")
    ax.text(L.grid_x[0][0] - 4, L.y_selA, "A", ha="right", va="center", fontsize=8, color="w")
    ax.text(L.grid_x[0][0] - 4, L.y_selB, "B", ha="right", va="center", fontsize=8, color="w")
    ax.text(L.grid_x[-1][-1] + 4, L.y_dst, "dst", ha="left", va="center", fontsize=8, color="w")
lines = [f"registers at t=0:  r0..r3 = {list(regs)}", ""]
for i, ins in enumerate(program):
    a, b = trace[i][ins.a], trace[i][ins.b]
    r = (a - b) & 15; signed = r - 16 if r & 8 else r
    wrote = ins.op == "SUB" or signed <= 0
    lines.append(f"tile {i}:  {ins.op:<6} r{ins.a} r{ins.b} → r{ins.dst}     dst blob = {'+1' if ins.op == 'SUB' else '−1'}")
    lines.append(f"          A−B = {a}−{b} = {signed:+d} (mod 16 → {r})   "
                 + (f"write r{ins.dst} = {r}" if wrote else f"positive → r{ins.dst} keeps {trace[i][ins.dst]}"))
    lines.append(f"          mem {i+1} = {list(trace[i+1])}")
fig.text(0.5, 0.02, "\n".join(lines), ha="center", va="bottom", fontsize=9, family="monospace")
fig.suptitle("tile v5 — one input / output pair   (white disc = 1 · black disc = 0 · small black + = not set yet · grey = background;  selector rows: +1 selected, 0 not;  dst sign = op)", fontsize=9.5)
fig.tight_layout(rect=(0, 0.26, 1, 0.97))
out_p = Path(__file__).resolve().parent.parent / "v5_pair.png"
fig.savefig(out_p, dpi=150, facecolor="white"); print("wrote", out_p)
