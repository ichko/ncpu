#!/usr/bin/env python3
"""Posed layout design for v11.

An instruction is ONE band row.  selA, selB and dst live on three separate
channels of that same row, so all three selectors -- including the return --
are aligned with each other and with the register columns they point at.
At t=0 the tape is parked clear of the grid: nothing is selected.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import to_rgb

R, P, PAD = 2, 8, 7
WB, NR, NB = 4, 4, 3

SLOTS = ["r0", "r1", "r2", "r3", None, "A", "B", None, "op", "R"]
X = [PAD + i * P for i in range(len(SLOTS))]
IX = {s: X[i] for i, s in enumerate(SLOTS) if s}
MEMX = [IX[f"r{c}"] for c in range(NR)]
W = X[-1] + R + PAD + 1

PARK = 3 * P                                     # clearance of the tape at t=0
y_line = PAD + 2 * P + (NB * P) + PARK + 4
y_bits = [y_line + 6 + b * P for b in range(WB)]
H = y_bits[-1] + R + PAD + 10

C = {"mem": ["#e15759", "#f28e2b", "#59a14f", "#4e79a7"], "A": "#4472d0", "B": "#17becf",
     "R": "#2ca02c", "selA": "#4472d0", "selB": "#17becf", "dst": "#ff9d3a", "op": "#b07aa1"}
PROG = [("SUB", 1, 0, 2), ("SUBLEQ", 0, 2, 3), ("NOP", 0, 0, 0)]
REGS = [5, 9, 0, 0]


def sq(ax, x, y, col, fill, lw=1.5, ls="-", r=R):
    ax.add_patch(Rectangle((x - r, y - r), 2 * r + 1, 2 * r + 1,
                           fc=col if fill else "white", ec=col, lw=lw, ls=ls))


def blend(keys):
    v = [sum(to_rgb(C[k])[i] for k in keys) / len(keys) for i in range(3)]
    return tuple(v)


def band_row(ax, y, ins, chans=("selA", "selB", "dst"), show_op=True):
    """One instruction = one row; three selector channels overlay in it."""
    op, a, b, d = ins
    if op == "NOP":
        for c in range(NR):
            sq(ax, MEMX[c], y, "#aaa", False, lw=0.8, ls=(0, (1, 2)))
        if show_op: sq(ax, IX["op"], y, "#aaa", False, lw=0.8, ls=(0, (1, 2)))
        return
    for c in range(NR):
        on = [k for k, s in (("selA", a), ("selB", b), ("dst", d)) if s == c and k in chans]
        if on: sq(ax, MEMX[c], y, blend(on), True, lw=1.2)
        else: sq(ax, MEMX[c], y, "#bbb", False, lw=0.8, ls=(0, (1, 2)))
    if show_op: sq(ax, IX["op"], y, C["op"], op == "SUB", lw=1.2)


def draw(ax, phase, title, note, parked=False):
    ax.set_xlim(-3, W + 3); ax.set_ylim(H + 3, -16); ax.set_aspect("equal"); ax.axis("off")
    ax.add_patch(Rectangle((0, 0), W, H, fc="#f6f6f6", ec="k", lw=1.2))
    op0, sa, sb, dst = PROG[0]
    regs, slots = list(REGS), {"A": None, "B": None, "R": None}
    if phase in (1, 2, 3): slots["A"], slots["B"] = regs[sa], regs[sb]
    if phase in (2, 3): slots["R"] = (regs[sa] - regs[sb]) & 15
    if phase == 3: regs[dst] = slots["R"]

    for c in range(NR):
        for b in range(WB):
            sq(ax, MEMX[c], y_bits[b], C["mem"][c], (regs[c] >> (WB - 1 - b)) & 1)
        ax.text(MEMX[c], y_bits[-1] + 7, f"r{c}", ha="center", va="top", fontsize=6.5, color=C["mem"][c])
    for key in ("A", "B", "R"):
        for b in range(WB):
            v = slots[key]
            if v is None: sq(ax, IX[key], y_bits[b], C[key], False, lw=0.9, ls=(0, (1.4, 1.4)))
            else: sq(ax, IX[key], y_bits[b], C[key], (v >> (WB - 1 - b)) & 1)
        ax.text(IX[key], y_bits[-1] + 7, key, ha="center", va="top", fontsize=8, color=C[key], weight="bold")
    live = phase > 0
    sq(ax, IX["op"], y_bits[1], C["op"], live and op0 == "SUB",
       lw=1.5 if live else 0.9, ls="-" if live else (0, (1.4, 1.4)))
    ax.text(IX["op"], y_bits[-1] + 7, "op", ha="center", va="top", fontsize=8, color=C["op"], weight="bold")

    shift = 1 if phase == 4 else 0
    for k in range(NB - shift):
        y = y_line - 4 - k * P - (PARK if parked else 0)
        band_row(ax, y, PROG[k + shift])
        if k == 0:
            ax.text(W + 1, y, "current", ha="left", va="center",
                    fontsize=5.5, color="#999", style="italic", rotation=90)
    ax.plot([2, W - 2], [y_line, y_line], color="#d62728", lw=0.9)
    if parked:
        ax.annotate("", xy=(MEMX[0] - 5, y_line - 5), xytext=(MEMX[0] - 5, y_line - PARK),
                    arrowprops=dict(arrowstyle="<->", color="#bbb", lw=0.7))
        ax.text(MEMX[0] - 8, y_line - PARK / 2, "clear", ha="right", va="center",
                fontsize=5.5, color="#aaa", style="italic")
    ax.set_title(title, fontsize=10, weight="bold", pad=6)
    ax.text(W / 2, -9, note, ha="center", va="center", fontsize=7.5, color="#333")


def chan(ax, title, chans, show_op):
    ax.set_xlim(-3, IX["op"] + 8); ax.set_ylim(14, -10); ax.set_aspect("equal"); ax.axis("off")
    band_row(ax, 2, PROG[0], chans=chans, show_op=show_op)
    for c in range(NR):
        ax.text(MEMX[c], 9, f"r{c}", ha="center", va="top", fontsize=5.5, color="#999")
    if show_op: ax.text(IX["op"], 9, "op", ha="center", va="top", fontsize=5.5, color="#999")
    ax.set_title(title, fontsize=7.5, color="#444", pad=2)


fig = plt.figure(figsize=(19.5, 7.6))
gs = fig.add_gridspec(2, 20, height_ratios=[1, 3.4], hspace=0.05)
for i, (t, ch, so) in enumerate([("channel selA", ("selA",), False), ("channel selB", ("selB",), False),
                                 ("channel dst (return)", ("dst",), False),
                                 ("op cell", (), True), ("= one band row", ("selA", "selB", "dst"), True)]):
    chan(fig.add_subplot(gs[0, i * 3:i * 3 + 3]), t, ch, so)
fig.text(0.79, 0.88, "One instruction is ONE row.\nThe two input selectors and the\nreturn selector are three channels\nof that same row — so they are\naligned with each other and with\nthe register columns.",
         fontsize=9, va="top", color="#333")

specs = [(0, "t = 0", "tape parked clear of the grid\nnothing selected, slots and op unset", True),
         (1, "phase 1   FETCH", "band at the line:  A ← mem[selA],  B ← mem[selB]\nsupervised: A, B correct; memory unchanged", False),
         (2, "phase 2   EXEC", "R ← A − B, reading the op cell\nsupervised: R correct; memory, A, B unchanged", False),
         (3, "phase 3   STORE", "mem[dst] ← R, gated by op\nSUBLEQ writes only if the signed R ≤ 0", False),
         (4, "phase 4   ADVANCE", "tape drops one row, slots + op clear\nsupervised: next band at the line", False)]
for i, (p, t, n, pk) in enumerate(specs):
    draw(fig.add_subplot(gs[1, i * 4:i * 4 + 4]), p, t, n, parked=pk)
fig.suptitle("v11 — four phases, one band row per instruction", fontsize=13, y=0.985)
fig.savefig("v11_layout.png", dpi=150, facecolor="white", bbox_inches="tight")
print(f"grid {H}x{W}")
