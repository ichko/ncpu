#!/usr/bin/env python3
"""Render the canonical v11 cycle: the actual pixels the trainer supervises."""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ncpu.program import Instr, TileLayoutV11

ap = argparse.ArgumentParser()
ap.add_argument("--n-instr", type=int, default=3)
ap.add_argument("--out", default="v11_states.png")
a = ap.parse_args()

L = TileLayoutV11(n_instr=a.n_instr)
prog = [Instr("SUB", 1, 0, 2), Instr("SUBLEQ", 0, 2, 3), Instr("NOP", 0, 0, 0)][: a.n_instr]
regs = [5, 9, 0, 0]
traj = L.cycle(prog, regs)

n = len(traj) + 1
fig, axes = plt.subplots(1, n, figsize=(1.9 * n, 4.6))
axes[0].imshow(L.to_rgb(*L.render(prog, traj[0][0])), interpolation="nearest")
axes[0].set_title("t = 0\nparked", fontsize=8, weight="bold")
for i, (st, ph, nx) in enumerate(traj):
    ax = axes[i + 1]
    ax.imshow(L.to_rgb(*L.render(prog, nx)), interpolation="nearest")
    ins = prog[st.consumed] if st.consumed < len(prog) else None
    ax.set_title(f"{ph}\n{'i=%d %s' % (st.consumed, ins.op) if ins else ''}", fontsize=8, weight="bold")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"v11 canonical cycle — grid {L.H}x{L.W}, each arrow is one supervised phase", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(a.out, dpi=150, facecolor="white")
print("wrote", a.out)
