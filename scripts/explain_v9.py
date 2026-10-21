#!/usr/bin/env python3
"""One figure explaining the painted-program layout and how the simulation runs."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.program import TileLayoutV8, TileLayoutV9, Instr, run_program

fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(3, 6, height_ratios=[1.15, 1.0, 0.85], hspace=0.38, wspace=0.28)

# ── row 1: what one instruction means ───────────────────────────────────────
L8 = TileLayoutV8()
ins, regs = Instr("SUB", 1, 0, 2), (5, 9, 0, 0)
out = run_program(regs, [ins], 4)[1]
m = L8.mask_channels(ins)
panels = [("channel 0\nthe registers", L8.data_screen(regs), "gray"),
          ("channel A\nfirst operand", m[0], "Blues"),
          ("channel B\nsecond operand", m[1], "GnBu"),
          ("channel R\ndestination (+1 SUB, −1 SUBLEQ)", m[2], "PiYG")]
for i, (t, img, cm) in enumerate(panels):
    ax = fig.add_subplot(gs[0, i]); ax.imshow(img, cmap=cm, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(t, fontsize=8.5); ax.set_xticks([]); ax.set_yticks([])
ax = fig.add_subplot(gs[0, 4:])
both = np.concatenate([L8.to_rgb(L8.data_screen(regs), m), np.full((L8.H, 4, 3), 90, np.uint8), L8.to_rgb(L8.data_screen(out), m)], 1)
ax.imshow(both, interpolation="nearest"); ax.set_xticks([]); ax.set_yticks([])
ax.set_title("blended, and what the rule must produce:   r2 ← r1 − r0", fontsize=9.5, weight="bold")
ax.set_xlabel(f"{list(regs)}  →  {list(out)}      4 registers × 4 bits; white square = 1, black = 0", fontsize=8.5)
fig.text(0.012, 0.955, "1.  ONE INSTRUCTION  —  the data is a picture, the instruction is three extra channels painted over it",
         fontsize=12, weight="bold", va="top")
fig.text(0.012, 0.925, "SUB: dst ← A − B always.   SUBLEQ: dst ← A − B only if the result is ≤ 0, otherwise dst is left alone "
         "(this is the one data-dependent decision in the whole machine).", fontsize=9, va="top", color="#333")

# ── row 2: a program is a stack, and it walks down ──────────────────────────
program = [Instr("SUB", 1, 0, 2), Instr("SUBLEQ", 0, 2, 3), Instr("SUB", 3, 3, 1)]
L9 = TileLayoutV9(n_instr=3, preloaded=True)
trace = run_program(regs, program, 4)
K = {True: 50, False: 40}; t = 0
stages = [("t = 0\ngiven", 0, "")]
for k in range(L9.n_stages):
    t += K[L9.is_exec(k)]
    stages.append((f"{L9.stage_name(k+1)}\nby t = {t}", k + 1, f"{list(trace[L9.execs_done(k+1)])}"))
for i, (title, st, sub) in enumerate(stages):
    ax = fig.add_subplot(gs[1, i])
    ax.imshow(L9.to_rgb(L9.paint(program, regs, st), L9.mask_channels_at(program, st)), interpolation="nearest")
    ax.set_title(title, fontsize=8.5); ax.set_xlabel(sub, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    if "LOAD" in title:
        ax.add_patch(Rectangle((-0.5, -0.5), L9.W, L9.row_y[0] - L9.r - 1, fill=False, ec="#d62728", lw=1.5))
fig.text(0.012, 0.635, "2.  A PROGRAM  —  instructions wait as 5-pixel bands stacked above the registers; the machine runs them one at a time",
         fontsize=12, weight="bold", va="top")
fig.text(0.012, 0.607, "EXEC: the expanded instruction computes, in place.    LOAD (red box): the whole stack slides down one band, "
         "the next band expands into full-height rectangles, the old ones vanish.", fontsize=9, va="top", color="#333")

# ── row 3: the mechanics ────────────────────────────────────────────────────
ax = fig.add_subplot(gs[2, :3]); ax.axis("off")
ax.text(0, 1, "How the simulation runs\n\n"
        "•  One update rule, 16,640 parameters, applied to every pixel at once.\n"
        "•  Each cell sees a 7×7 neighbourhood of 16 channels; 4 are the picture\n"
        "   (registers, A, B, R), the other 12 are the rule's own scratch space.\n"
        "•  Every cell updates with probability ½ per step — there is no clock line\n"
        "   and no cell knows where it is on the grid.\n"
        "•  The program is painted once, at t = 0. Nothing is written in afterwards:\n"
        "   the picture is the program, the memory and the output at the same time.\n"
        "•  EXEC is 50 steps, LOAD is 40. The rule is scored on the last 10 frames\n"
        "   of each — it must reach the right picture and then hold it still.",
        fontsize=10, va="top", family="sans-serif")
ax = fig.add_subplot(gs[2, 3:]); ax.axis("off")
ax.text(0, 1, "Where it stands\n\n"
        "•  One instruction, executed in place:  99.5% of programs exactly right\n"
        "   (SUB, SUBLEQ and its ≤0 test, and no-ops) — that part works.\n"
        "•  Moving the program stack — the LOAD above — is the open problem.\n"
        "   The rule holds the registers perfectly and slides a single band, but\n"
        "   growing a 5-pixel band into a full-height rectangle while erasing the\n"
        "   previous one does not finish inside its window.\n"
        "•  Once LOAD lands, a program of any length is just more bands: the same\n"
        "   rule, run longer, with nothing added.",
        fontsize=10, va="top", family="sans-serif")
fig.suptitle("A neural cellular automaton that runs a compiled program", fontsize=15, weight="bold", y=0.995)
out_p = Path(__file__).resolve().parent.parent / "explain_v9.png"
fig.savefig(out_p, dpi=140, facecolor="white", bbox_inches="tight"); print("wrote", out_p)
