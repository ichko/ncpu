#!/usr/bin/env python3
"""
E4: Carry wave visualization.

Loads a trained 8-bit adder checkpoint, runs inference on carefully chosen
addition examples, and plots channel 0 activations across rollout timesteps
to show information propagating from LSB to MSB (the carry wave).

Usage:
    uv run python scripts/other/visualize_carry_wave.py
    uv run python scripts/other/visualize_carry_wave.py --run_dir runs/E3_adder8_cols2_s0_20260316_033201
    uv run python scripts/other/visualize_carry_wave.py --steps 128 --out carry_wave.svg
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.normalizers import normalize_neg1_to_1
from ncpu.utils import make_io_screen

# ── Args ───────────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--run_dir", default="runs/E3_adder8_cols2_s0_20260316_033201")
parser.add_argument("--steps",   type=int, default=128)
parser.add_argument("--out",     default="results/E4_carry_wave/carry_wave.svg")
parser.add_argument("--device",  default="cuda")
args = parser.parse_args()

run_dir = Path(args.run_dir)
DEVICE  = args.device if torch.cuda.is_available() else "cpu"

# ── Load checkpoint ────────────────────────────────────────────────────────────

ckpt_dir = run_dir / "checkpoints"
ckpts = sorted(ckpt_dir.glob("nca_*.pt"))
assert ckpts, f"No checkpoints in {ckpt_dir}"
ckpt_path = ckpts[-1]
print(f"Loading checkpoint: {ckpt_path}")

nca = NeuralCA(
    channels=16,
    hidden_channels=[128],
    fire_rate=0.5,
    alive_threshold=0,
    zero_initialization=False,
    kernel_size=7,
    num_perception_kernels=3,
    read_only_dims=[1],
    padding_type="zeros",
).to(DEVICE)

state_dict = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
nca.load_state_dict(state_dict, strict=False)
nca.eval()
nca.gaussian_noise.prob = 0.0  # disable additive noise; keep stochastic update
print("NCA loaded.")

# ── Grid parameters (must match training config) ────────────────────────────

W, H, r, spacing = 80, 112, 4, (2, 21)

# ── Choose representative addition examples ─────────────────────────────────
#
#  We want cases that make the carry wave visible:
#   (1) 127 + 1  = 128  → carry ripples through all 7 lower bits
#   (2) 255 + 1  = 256  → carry ripples through all 8 bits (overflow)
#   (3) 85  + 85 = 170  → interleaved carries, visually interesting
#   (4) 170 + 85 = 255  → partial carry

examples = [
    ("127 + 1",  127, 1),
    ("255 + 1",  255, 1),
    ("85 + 85",   85, 85),
    ("170 + 85", 170, 85),
]


def int_to_bits8(n):
    return list(map(int, f"{n:08b}"))


def make_state(a_int, b_int):
    """Build NCA initial state matching the training convention:
    - inp screen = LEFT input circles only (no output circles)
    - Channels 0 and 1 both get the normalized inp screen (input_dims=(0,1))
    - Output circles start at gray (0.0 normalized) — NCA must compute them
    """
    a_bits = int_to_bits8(a_int)
    b_bits = int_to_bits8(b_int)
    s_int  = a_int + b_int
    o_bits = list(map(int, f"{s_int:09b}"))

    # Input-only screen (no right circles) — exactly as in NCPUDataset.get_sample()
    inp_screen = make_io_screen(H, W, r, spacing, a_bits + b_bits, [])
    # Target (output circles only) — used to evaluate correctness
    out_screen = make_io_screen(H, W, r, spacing, [], o_bits)

    inp_norm = normalize_neg1_to_1(
        torch.tensor(inp_screen, dtype=torch.float32)
    )
    out_norm = normalize_neg1_to_1(
        torch.tensor(out_screen, dtype=torch.float32)
    )

    state = torch.zeros(1, 16, H, W, device=DEVICE)
    state[0, 0] = inp_norm.to(DEVICE)
    state[0, 1] = inp_norm.to(DEVICE)  # read-only channel 1

    return state, inp_screen, out_screen, out_norm, o_bits


# ── Run inference ──────────────────────────────────────────────────────────────
#  Average N_RUNS stochastic rollouts per example for smooth curves.

STEPS   = args.steps
N_RUNS  = 32   # average this many stochastic rollouts per example

all_rollouts = []     # each entry: mean over N_RUNS, shape (T+1, C, H, W)
all_inp_screens = []
all_labels = []

all_out_screens = []
all_out_norms = []

with torch.no_grad():
    for label, a_int, b_int in examples:
        state, inp_screen, out_screen, out_norm, _ = make_state(a_int, b_int)
        # replicate initial state N_RUNS times
        state_batch = state.expand(N_RUNS, -1, -1, -1).contiguous()
        rollout = nca(state_batch, steps=STEPS)  # (N_RUNS, T+1, C, H, W)
        mean_rollout = rollout.mean(dim=0).cpu()  # (T+1, C, H, W)
        all_rollouts.append(mean_rollout)
        all_inp_screens.append(inp_screen)
        all_out_screens.append(out_screen)
        all_out_norms.append(out_norm)
        all_labels.append(label)

# ── Select timesteps to display ────────────────────────────────────────────────

T = STEPS + 1
display_ts = [0, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128]
display_ts = [t for t in display_ts if t < T]

# ── Figure: channel 0 heatmap grid ────────────────────────────────────────────
#  Rows = examples, cols = selected timesteps

n_rows = len(examples)
n_cols = len(display_ts)

fig, axes = plt.subplots(
    n_rows, n_cols,
    figsize=(1.6 * n_cols, 2.0 * n_rows),
    gridspec_kw={"hspace": 0.05, "wspace": 0.03},
)

vmin, vmax = -1.5, 1.5

for row, (label, rollout) in enumerate(zip(all_labels, all_rollouts)):
    for col, t in enumerate(display_ts):
        ax = axes[row, col]
        ch0 = rollout[t, 0].numpy()  # (H, W)
        ax.imshow(ch0, cmap="RdBu_r", vmin=vmin, vmax=vmax,
                  interpolation="nearest", origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0:
            ax.set_title(f"t={t}", fontsize=7, pad=2)
        if col == 0:
            ax.set_ylabel(label, fontsize=7, labelpad=2)

fig.suptitle("Carry wave — channel 0 activations over time  (8-bit adder)", fontsize=9)
fig.savefig(args.out, bbox_inches="tight")
plt.close(fig)
print(f"Saved {args.out}")

# ── Figure 2: activation diff from t=0 (shows where information is moving) ────

fig2, axes2 = plt.subplots(
    n_rows, n_cols,
    figsize=(1.6 * n_cols, 2.0 * n_rows),
    gridspec_kw={"hspace": 0.05, "wspace": 0.03},
)

for row, (label, rollout) in enumerate(zip(all_labels, all_rollouts)):
    ch0_t0 = rollout[0, 0].numpy()
    for col, t in enumerate(display_ts):
        ax = axes2[row, col]
        diff = rollout[t, 0].numpy() - ch0_t0
        ax.imshow(diff, cmap="PiYG", vmin=-1.5, vmax=1.5,
                  interpolation="nearest", origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])

        if row == 0:
            ax.set_title(f"t={t}", fontsize=7, pad=2)
        if col == 0:
            ax.set_ylabel(label, fontsize=7, labelpad=2)

out_diff = args.out.replace(".svg", "_diff.svg")
fig2.suptitle("Carry wave — Δ(channel 0)  (change from t=0)", fontsize=9)
fig2.savefig(out_diff, bbox_inches="tight")
plt.close(fig2)
print(f"Saved {out_diff}")

# ── Figure 3: all-channel activation energy over time ─────────────────────────
#  For each example, plot mean |activation| in left-half vs right-half of grid
#  over timesteps — should show wave from left (input) to right (output)

fig3, axes3 = plt.subplots(2, 2, figsize=(8, 5), sharex=True, sharey=True)

for idx, (label, rollout) in enumerate(zip(all_labels, all_rollouts)):
    ax = axes3[idx // 2, idx % 2]
    # rollout: (T+1, C, H, W)
    energy = rollout[:, :, :, :].abs().mean(dim=(1, 2))  # (T+1, W)
    # average over W slices: shows energy flow left → right
    ax.imshow(energy.T.numpy(), aspect="auto", cmap="hot",
              origin="lower", extent=[0, T-1, 0, W-1])
    ax.set_xlabel("timestep")
    ax.set_ylabel("x (columns)")
    ax.set_title(label, fontsize=8)
    ax.axvline(0, color="cyan", linewidth=0.8, linestyle="--")

fig3.suptitle("Mean |activation| by column over time  (left=input, right=output)", fontsize=9)
fig3.tight_layout()
out_wave = args.out.replace(".svg", "_wave.svg")
fig3.savefig(out_wave, bbox_inches="tight")
plt.close(fig3)
print(f"Saved {out_wave}")

# ── Figure 4: per-output-bit accuracy over time ───────────────────────────────
#  For each rollout timestep, check whether channel 0 at each output circle
#  matches the target sign (+1 = bit on, -1 = bit off).
#  This shows the carry wave as convergence order: LSB bits converge first,
#  then carry propagates upward so MSB bits converge later.
#
#  Output circles (9 bits, bit 0=MSB at top, bit 8=LSB sum at bottom):
#   x = W - side_sp = 59
#   v_size = 9*8 + 2*8 = 88,  top_margin = (112-88)//2 = 12
#   y_i = 12 + 4 + i*10  →  bit 0 (MSB) at y=16, bit 8 (LSB carry-out) at y=96

among_sp, side_sp = 2, 21
step_y = 2 * r + among_sp          # = 10
out_x  = W - side_sp               # = 59
n_out  = 9
out_top_margin = (H - (n_out * 2 * r + among_sp * (n_out - 1))) // 2  # = 12

# build circular masks for each output bit
OUT_RADIUS = r  # pixels

def circle_mask(cy, cx, rad, H, W):
    ys, xs = np.ogrid[:H, :W]
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= rad ** 2

out_masks = []
for i in range(n_out):
    cy = out_top_margin + r + i * step_y
    mask = circle_mask(cy, out_x, OUT_RADIUS, H, W)
    out_masks.append(mask)

fig4, axes4 = plt.subplots(1, 2, figsize=(10, 4))

# ── Batch carry-wave: run over many random examples, average per-bit accuracy ─
#  Two scenarios: (a) worst-case carry chain (all A=127, B=1), (b) random pairs

CARRY_BATCH = 64  # examples per scenario

scenarios = [
    ("127 + 1 (max carry chain)", [(127, 1)] * CARRY_BATCH),
    ("random 8-bit additions",    [(np.random.randint(256), np.random.randint(256))
                                   for _ in range(CARRY_BATCH)]),
]

for plot_idx, (title, pairs) in enumerate(scenarios):
    ax = axes4[plot_idx]

    # build batch initial states
    states = []
    targets = []
    for a_int, b_int in pairs:
        s_int  = a_int + b_int
        target_bits = list(map(int, f"{s_int:09b}"))  # 9 bits MSB-first
        target_norm = [1.0 if b else -1.0 for b in target_bits]
        st, _, _, _, _ = make_state(a_int, b_int)
        states.append(st)
        targets.append(target_norm)

    state_batch = torch.cat(states, dim=0)   # (CARRY_BATCH, C, H, W)
    target_arr  = np.array(targets)          # (CARRY_BATCH, 9)

    with torch.no_grad():
        rollout = nca(state_batch, steps=STEPS)  # (B, T+1, C, H, W)

    ch0 = rollout[:, :, 0].cpu().numpy()  # (B, T+1, H, W)

    # per-bit accuracy at each timestep: fraction of batch where bit is correct
    # bit_acc[bit_i, t] = fraction of samples with correct sign at that circle
    bit_acc = np.zeros((n_out, STEPS + 1))
    for bit_i, mask in enumerate(out_masks):
        circle_vals = ch0[:, :, mask].mean(axis=-1)  # (B, T+1)
        tgt_col = target_arr[:, bit_i]               # (B,)
        correct = ((circle_vals > 0) == (tgt_col[:, None] > 0))  # (B, T+1)
        bit_acc[bit_i] = correct.mean(axis=0)

    colors = plt.cm.plasma(np.linspace(0.05, 0.95, n_out))
    for bit_i, col in enumerate(colors):
        bit_name = f"b{bit_i}" + (" MSB" if bit_i == 0 else " LSB" if bit_i == n_out - 1 else "")
        ax.plot(bit_acc[bit_i], color=col, linewidth=1.5, label=bit_name)

    ax.set_xlabel("NCA timestep")
    ax.set_ylabel("fraction of outputs correct")
    ax.set_title(title, fontsize=9)
    ax.axhline(1.0, color="gray", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=6, loc="lower right", ncol=3)
    ax.set_ylim(-0.05, 1.1)

fig4.suptitle("Carry wave — per-output-bit convergence over NCA timesteps", fontsize=10)
fig4.tight_layout()
out_carry = args.out.replace(".svg", "_carry.svg")
fig4.savefig(out_carry, bbox_inches="tight")
plt.close(fig4)
print(f"Saved {out_carry}")

# ── Figure 5: spatial channel-0 snapshots at key timesteps (127+1 case) ───────
#  Publication-quality: single row, selected timesteps, only the 127+1 example

eg_idx = 0  # 127+1  (max carry chain, clearest case)
rollout = all_rollouts[eg_idx]
pub_ts = [0, 8, 16, 24, 32, 48, 64, 128]
pub_ts = [t for t in pub_ts if t <= STEPS]

fig5, axes5 = plt.subplots(1, len(pub_ts), figsize=(2.2 * len(pub_ts), 3.0),
                            gridspec_kw={"wspace": 0.04})

for col, t in enumerate(pub_ts):
    ax = axes5[col]
    ch0 = rollout[t, 0].numpy()
    im = ax.imshow(ch0, cmap="RdBu_r", vmin=-2, vmax=2,
                   interpolation="nearest", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"t = {t}", fontsize=8)

fig5.suptitle("127 + 1 = 128  —  channel 0  over time  (carry ripple)", fontsize=10)
fig5.tight_layout()
out_pub = args.out.replace(".svg", "_pub.svg")
fig5.savefig(out_pub, bbox_inches="tight")
plt.close(fig5)
print(f"Saved {out_pub}")
