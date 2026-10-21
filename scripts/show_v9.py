#!/usr/bin/env python3
"""v9: the program moves, the data stays.  Every stage target, blended RGB."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.program import TileLayoutV9, Instr, run_program

program = [Instr("SUB", 1, 0, 2), Instr("SUBLEQ", 0, 2, 3), Instr("SUB", 3, 3, 1)]
regs = (5, 9, 0, 0)
L = TileLayoutV9(n_instr=len(program), preloaded=True)
trace = run_program(regs, program, 4)
panels = [("t = 0: instr 0 already loaded, 2 in the stack", 0)] + [(f"stage {k+1}: {L.stage_name(k+1)}", k + 1) for k in range(L.n_stages)]
fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 4.6))
for ax, (title, st) in zip(axes, panels):
    ax.imshow(L.to_rgb(L.paint(program, regs, st), L.mask_channels_at(program, st)), interpolation="nearest")
    ax.set_title(title, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(f"regs {list(trace[(st) // 2])}", fontsize=8)
fig.suptitle("v9 — " + " ; ".join(str(i) for i in program) + f"    (grid {L.H}x{L.W}; A blue, B cyan, dst orange = SUB / magenta = SUBLEQ)", fontsize=10)
fig.tight_layout(); out = Path(__file__).resolve().parent.parent / "v9_stages.png"; fig.savefig(out, dpi=150, facecolor="white"); print("wrote", out, f"grid {L.H}x{L.W}")
