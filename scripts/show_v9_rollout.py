#!/usr/bin/env python3
"""v9: the data stays on the grid, the program moves down.

Shows the layout and one full rollout, marking where supervision is applied.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from ncpu.program import Instr, TileLayoutV9, run_program

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="v9_rollout.png")
a = ap.parse_args()

prog = [Instr("SUB", 1, 0, 2), Instr("SUBLEQ", 2, 3, 1), Instr("SUB", 0, 2, 3)]
regs = [5, 9, 0, 3]
L = TileLayoutV9(n_instr=len(prog), row_h=5, stack_gap=4)
trace = run_program(regs, prog, L.word_bits)

K_LOAD, K_EXEC = 40, 40
n = L.n_stages
fig = plt.figure(figsize=(1.85 * (n + 1), 4.5))
gs = fig.add_gridspec(1, n + 1, wspace=0.12, left=0.01, right=0.99, top=0.70, bottom=0.26)

for k in range(n + 1):
    ax = fig.add_subplot(gs[0, k])
    img = L.to_rgb(L.paint(prog, regs, k), L.mask_channels_at(prog, k))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if k == 0:
        ax.set_title("t = 0\nthe whole program\nis in the picture", fontsize=8.5, weight="bold")
        ax.set_xlabel("input", fontsize=8, color="#444")
    else:
        name = L.stage_name(k)
        kind = "EXEC" if L.is_exec(k - 1) else "LOAD"
        steps = K_EXEC if kind == "EXEC" else K_LOAD
        ax.set_title(f"after stage {k}\n{name}", fontsize=8.5, weight="bold",
                     color="#b45309" if kind == "EXEC" else "#1d4ed8")
        sm = L.stage_mask(prog, k)
        changed = int(sm.sum() > 0)
        ax.set_xlabel(f"{steps} CA steps\n" + ("SUPERVISED: full grid\n+ dst column weighted"
                                               if changed else "SUPERVISED: full grid\n(nothing may change)"),
                      fontsize=7.5, color="#444")
        # outline the dst column that this stage is responsible for
        if changed:
            ys, xs = np.nonzero(sm)
            ax.add_patch(Rectangle((xs.min() - 2, ys.min() - 2), np.ptp(xs) + 5, np.ptp(ys) + 5,
                                   fill=False, ec="#b45309", lw=1.8))
    for sp in ax.spines.values():
        sp.set_edgecolor("#bbb")

fig.suptitle(
    "v9 — the register grid never moves; the instruction stack moves down into it.\n"
    "LOAD: the bottom band drops onto the grid and expands into full-height selector rectangles.   "
    "EXEC: the rule computes dst ← A−B in place.\n"
    "Supervision is the whole data grid at the END of every stage (last ~10 of its 40 frames), "
    "with the dst column that changed weighted up.",
    fontsize=10, y=0.985)
fig.text(0.5, 0.045,
         f"program:  " + "    ".join(f"{i.op}(r{i.a},r{i.b})→r{i.dst}" for i in prog)
         + f"      registers:  " + " → ".join(str(tuple(t)) for t in trace),
         ha="center", fontsize=9, color="#222")
fig.savefig(a.out, dpi=150, facecolor="white")
print(f"wrote {a.out}   grid {L.H}x{L.W}, {n} stages")
