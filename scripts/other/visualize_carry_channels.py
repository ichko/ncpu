#!/usr/bin/env python3
"""
Carry-wave hidden channels figure for the paper.

Renders channel 0 plus the top-K hidden channels most strongly expressing
the carry wave for a full-chain 8-bit addition (127 + 1 = 128).

    uv run python scripts/other/visualize_carry_channels.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.normalizers import normalize_neg1_to_1
from ncpu.utils import make_io_screen

RUN_DIR = Path("runs/E3_adder8_cols2_s0_20260316_033201")
OUT_DIR = Path("results/E4_carry_wave")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEPS      = 64
A, B       = 127, 1                    # full-length carry chain
N_RUNS     = 32                        # avg stochastic rollouts
TOP_K_HID  = 3                         # hidden channels to show (+ ch 0)
SKIP_HID   = 3                         # skip this many top-ranked channels first
DISPLAY_TS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 48, 64]
W, H, r, spacing = 80, 112, 4, (2, 21)
OUT_X      = W - spacing[1]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load ───────────────────────────────────────────────────────────────────────

ckpts = sorted((RUN_DIR / "checkpoints").glob("nca_*.pt"))
assert ckpts, f"no checkpoints in {RUN_DIR}"
ckpt = ckpts[-1]
print(f"Loading {ckpt}")

nca = NeuralCA(
    channels=16, hidden_channels=[128], fire_rate=0.5,
    alive_threshold=0, zero_initialization=False,
    kernel_size=7, num_perception_kernels=3,
    read_only_dims=[1], padding_type="zeros",
).to(DEVICE)
nca.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True), strict=False)
nca.eval()
nca.gaussian_noise.prob = 0.0


def bits8(n): return list(map(int, f"{n:08b}"))


def make_state(a, b):
    inp = make_io_screen(H, W, r, spacing, bits8(a) + bits8(b), [])
    inp_n = normalize_neg1_to_1(torch.tensor(inp, dtype=torch.float32)).to(DEVICE)
    s = torch.zeros(1, 16, H, W, device=DEVICE)
    s[0, 0] = inp_n
    s[0, 1] = inp_n
    return s


# ── Run ───────────────────────────────────────────────────────────────────────

state = make_state(A, B)
with torch.no_grad():
    rollout = nca(state.expand(N_RUNS, -1, -1, -1).contiguous(), steps=STEPS)
rollout = rollout.mean(dim=0).cpu()   # (T+1, C, H, W)
_, C, _, _ = rollout.shape
print(f"rollout: {rollout.shape}")

# ── Rank hidden channels by carry-wave activity ───────────────────────────────

col_strip = slice(OUT_X - 5, OUT_X + 6)
window    = slice(4, 33)
scores = []
for ch in range(C):
    strip = rollout[window, ch, :, col_strip]
    scores.append((ch, strip.std(dim=0).mean().item()))

exclude = {0, 1, C - 2, C - 1}
hidden_ranked = sorted([(ch, s) for ch, s in scores if ch not in exclude],
                       key=lambda x: -x[1])
top_hidden = [ch for ch, _ in hidden_ranked[SKIP_HID:SKIP_HID + TOP_K_HID] if ch != 12]

print("hidden channel score (desc):")
for ch, s in hidden_ranked:
    mark = " *" if ch in top_hidden else "  "
    print(f"  ch{ch:2d}: {s:.4f}{mark}")

row_channels = [0] + top_hidden
row_labels   = ["ch 0"] + [f"ch {c}" for c in top_hidden]
print(f"rows: {row_channels}")

# ── Figure ────────────────────────────────────────────────────────────────────

display_ts = [t for t in DISPLAY_TS if t <= STEPS]
n_rows = len(row_channels)
n_cols = len(display_ts)

cell_w = 1.1
cell_h = cell_w * H / W
fig_w  = n_cols * cell_w
fig_h  = n_rows * cell_h

fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
fig.subplots_adjust(hspace=0.02 * cell_w / cell_h, wspace=0.02)

for row, (ch, lbl) in enumerate(zip(row_channels, row_labels)):
    vmax = 1.0 if ch == 0 else max(1.0, rollout[:, ch].abs().max().item())
    for col, t in enumerate(display_ts):
        ax = axes[row, col]
        ax.imshow(rollout[t, ch].numpy(), cmap="viridis",
                  vmin=-vmax, vmax=vmax,
                  interpolation="nearest", origin="upper", rasterized=True)
        ax.axis("off")
        if row == 0:
            ax.set_title(f"t={t}", fontsize=11, pad=2, fontweight="bold")
        if col == 0:
            ax.text(-0.05, 0.5, lbl, transform=ax.transAxes,
                    fontsize=11, va="center", ha="right")

out_svg = OUT_DIR / "carry_channels.svg"
fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved {out_svg}")
