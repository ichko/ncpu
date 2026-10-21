#!/usr/bin/env python3
"""v10 conveyor: the program scrolls past the ALU, no stages, no clock."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.program import TileLayoutV10, Instr, run_program

program = [Instr("SUB", 1, 0, 2), Instr("SUBLEQ", 0, 2, 3), Instr("SUB", 3, 3, 1)]
regs = (5, 9, 0, 0)
L = TileLayoutV10(n_instr=3, top_pad=6)
trace = run_program(regs, program, 4)
offsets = list(range(0, L.total_travel() + 3, 1))
show = offsets[:: max(1, len(offsets) // 11)]
fig, axes = plt.subplots(1, len(show), figsize=(1.9 * len(show), 5.2))
for ax, o in zip(axes, show):
    d, m = L.picture_at(program, regs, o)
    ax.imshow(L.to_rgb(d, m), interpolation="nearest")
    ax.axhline(L.y_line + 0.5, color="#ff3b30", lw=0.8)
    ax.set_title(f"belt {o} px\nt = {L.steps_for(o)}", fontsize=7.5)
    ax.set_xlabel(f"{list(trace[L.consumed(o)])}", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle("v10 — the conveyor: instructions scroll down past the red line and are consumed; the registers update as each one passes.\n"
             "No stages, no clock, no expansion: an instruction is current when it is there.  Blank space above the stack is randomised so the top border means nothing.",
             fontsize=9.5)
fig.tight_layout(); out = Path("v10_conveyor.png"); fig.savefig(out, dpi=150, facecolor="white"); print("wrote", out, f"grid {L.H}x{L.W}")
