#!/usr/bin/env python3
"""The actual supervision targets of tile v4: the grayscale grids the NCA is
scored against, one per stage, with the pixels scored at that stage outlined."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.program import TileLayoutV7 as TileLayoutV4, Instr, run_program

n_instr = int(sys.argv[1]) if len(sys.argv) > 1 else 1
L = TileLayoutV4(n_instr=n_instr)
program = [Instr("SUB", 1, 0, 2), Instr("SUBLEQ", 0, 2, 3)][:n_instr]
regs = (5, 9, 0, 0)
trace = run_program(regs, program, L.word_bits)

panels = [("t = 0   input (what is painted)", L.paint(program, regs, 0), None)]
for k in range(L.n_stages):
    panels.append((f"stage {k+1}: {L.stage_name(k+1)}", L.paint(program, regs, k + 1), L.stage_mask(program, k + 1)))

cols = min(5, len(panels)); rows = -(-len(panels) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols * L.W / 71, 4.2 * rows))
axes = np.atleast_1d(axes).ravel()
for ax, (title, img, mask) in zip(axes, panels):
    ax.imshow(img, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
    if mask is not None:
        ax.contour(mask, levels=[0.5], colors="#39d353", linewidths=1.2)
    ax.set_title(title, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
for ax in axes[len(panels):]:
    ax.axis("off")
fig.suptitle("Tile v7 supervision — " + " ; ".join(str(i) for i in program) + f"   regs {list(regs)} → " +
             " → ".join(str(list(t)) for t in trace[1:]) + "\n"
             "green outline = the blobs whose loss is turned on at that stage (everything already resolved stays supervised too)",
             fontsize=9)
fig.tight_layout()
out = Path(__file__).resolve().parent.parent / f"v7_stages_{n_instr}.png"
fig.savefig(out, dpi=150, facecolor="white"); print("wrote", out)
