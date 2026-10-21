#!/usr/bin/env python3
"""Example of the v8 encoding: one grid, instruction as three rectangle channels."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.program import TileLayoutV8, Instr, run_program

L = TileLayoutV8()
cases = [(Instr("SUB", 1, 0, 2), (5, 9, 0, 0)), (Instr("SUBLEQ", 0, 2, 3), (5, 4, 4, 0)), (Instr("NOP", 0, 0, 0), (5, 9, 0, 0))]
fig, axes = plt.subplots(len(cases), 5, figsize=(16, 3.6 * len(cases)))
for row, (ins, regs) in zip(axes, cases):
    tr = run_program(regs, [ins], 4); m = L.mask_channels(ins)
    inp, tgt = L.data_screen(tr[0]), L.data_screen(tr[1])
    panels = [("ch0 data (t=0)", inp, "gray"), ("ch A: selA rect", m[0], "Blues"), ("ch B: selB rect", m[1], "GnBu"),
              ("ch R: dst rect (+1 SUB / −1 SUBLEQ)", m[2], "PiYG"), ("blended view: input → target", None, None)]
    for ax, (title, img, cmap) in zip(row, panels):
        if img is None:
            both = np.concatenate([L.to_rgb(inp, m), np.full((L.H, 3, 3), 60, np.uint8), L.to_rgb(tgt, m)], 1)
            ax.imshow(both, interpolation="nearest")
        else:
            ax.imshow(img, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    a, b = tr[0][ins.a], tr[0][ins.b]
    row[0].set_ylabel(f"{ins.op} r{ins.a} r{ins.b} → r{ins.dst}\n{list(tr[0])} → {list(tr[1])}", fontsize=9)
fig.suptitle("v8: one data grid + three instruction channels — A blue, B cyan, dst orange = SUB / magenta = SUBLEQ", fontsize=11)
fig.tight_layout(); out = Path(__file__).resolve().parent.parent / "v8_example.png"; fig.savefig(out, dpi=140, facecolor="white"); print("wrote", out)
