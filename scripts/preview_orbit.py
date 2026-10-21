#!/usr/bin/env python3
"""Render the ground-truth orbit-counter target sequence — no model involved.

Use this to eyeball the task before training: is the comet readable, does the
tally tick on the right frame, is the settle window sane?

Outputs (project root):
    orbit_preview.gif    one lap and a bit, animated
    orbit_preview.png    geometry + masks + a strip of frames around a crossing

Usage:
    uv run python scripts/preview_orbit.py
    uv run python scripts/preview_orbit.py --counter binary --marks 5
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.gif import write_gif
from ncpu.orbit import OrbitLayout

parser = ArgumentParser()
parser.add_argument("--width", type=int, default=44)
parser.add_argument("--height", type=int, default=44)
parser.add_argument("--r", type=int, default=2)
parser.add_argument("--marks", type=int, default=5)
parser.add_argument("--mark-step", type=int, default=7)
parser.add_argument("--counter", choices=["unary", "binary"], default="unary")
parser.add_argument("--tail", type=int, default=6)
parser.add_argument("--speed", type=float, default=0.5)
parser.add_argument("--tap", default="bottom")
parser.add_argument("--settle", type=int, default=None)
parser.add_argument("--steps", type=int, default=None, help="default: ~1.3 laps")
parser.add_argument("--p0", type=int, default=None, help="fixed start arc position")
parser.add_argument("--c0", type=int, default=0, help="fixed start counter value")
parser.add_argument("--fps", type=int, default=25)
parser.add_argument("--zoom", type=int, default=4, help="gif upscale factor")
parser.add_argument("--out-dir", default=".")
args = parser.parse_args()

layout = OrbitLayout(
    W=args.width, H=args.height, r=args.r, marks=args.marks,
    mark_step=args.mark_step, counter=args.counter, tail=args.tail,
    speed=args.speed, tap=args.tap, settle=args.settle,
)
print(layout.describe())

steps = args.steps if args.steps is not None else int(1.3 * layout.lap_steps)
p0 = layout.L // 4 if args.p0 is None else args.p0

arc, val, cmask, _ = layout.schedule([p0], [args.c0], steps)
frames = layout.bg[None] + layout.token_delta[arc[0]] + layout.reg_delta[val[0]]

out_dir = Path(args.out_dir)
gif_path = out_dir / "orbit_preview.gif"
png_path = out_dir / "orbit_preview.png"

write_gif(gif_path, frames, fps=args.fps, vmin=-1, vmax=1, zoom=args.zoom)

# ── still figure: geometry, masks, and the frames either side of a crossing ──
cross = np.flatnonzero(np.diff(val[0]) != 0)
c = int(cross[0]) if len(cross) else steps // 2
picks = [0, max(0, c - 4), c, min(steps, c + layout.settle), steps]
labels = ["t=0", f"t={picks[1]}", f"crossing t={c}", f"+settle t={picks[3]}", f"t={steps}"]

fig, axes = plt.subplots(2, max(3, len(picks)), figsize=(2.2 * len(picks), 5.0))
for ax in axes.ravel():
    ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")

for j, (i, lab) in enumerate(zip(picks, labels)):
    ax = axes[0, j]
    ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(frames[i], cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
    ax.set_title(f"{lab}\ncount={val[0, i]} sup={int(cmask[0, i])}", fontsize=8)

panels = [("geometry (frozen ch)", layout.bg, -1, 1),
          ("track mask", layout.track_mask().numpy(), 0, 1),
          ("register mask", layout.register_mask().numpy(), 0, 1)]
for j, (name, data, lo, hi) in enumerate(panels):
    ax = axes[1, j]
    ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    ax.imshow(data, cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_title(name, fontsize=8)

if len(picks) > 3:
    ax = axes[1, 3]
    ax.axis("on")
    ax.plot(val[0], lw=1.2, label="counter")
    ax.plot(cmask[0] * layout.marks, lw=0.8, alpha=0.6, label="supervised")
    ax.set_xlabel("step", fontsize=8); ax.legend(fontsize=7); ax.tick_params(labelsize=7)

fig.suptitle(layout.describe(), fontsize=8)
fig.tight_layout()
fig.savefig(png_path, dpi=120)
plt.close(fig)

print(f"steps={steps}  crossings at {cross.tolist()}")
print(f"wrote {gif_path}  and  {png_path}")
