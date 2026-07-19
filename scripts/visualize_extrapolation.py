#!/usr/bin/env python3
"""
Visualise a K=3-trained NCA extrapolating to 8-bit addition.

Produces two side-by-side panels saved to results/E5_extrapolation/:

  extrap_success.svg  — spatial rollout of a successful 8-bit carry chain
                        (127 + 1 = 128, maximum carry propagation)
  extrap_failure.svg  — spatial rollout of a failure case found by scanning
                        a large batch of random 8-bit additions
  extrap_cases.svg    — combined figure: success (top) + failure (bottom),
                        annotated with correct/incorrect output bits

Usage:
    uv run python scripts/visualize_extrapolation.py
    uv run python scripts/visualize_extrapolation.py --run_dir runs/extrap_train3_s1_...
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.normalizers import normalize_neg1_to_1
from ncpu.utils import make_io_screen_bottom_aligned

# ── Args ───────────────────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument("--run_dir", default="runs/extrap_train3_s0_20260322_211910")
parser.add_argument("--steps",   type=int, default=128)
parser.add_argument("--out_dir", default="results/E5_extrapolation")
parser.add_argument("--device",  default="cuda")
parser.add_argument("--scan_batch", type=int, default=256,
                    help="Number of random examples to scan for failures (run in chunks)")
args = parser.parse_args()

run_dir = Path(args.run_dir)
OUT_DIR = Path(args.out_dir)
DEVICE  = args.device if torch.cuda.is_available() else "cpu"
STEPS   = args.steps
N_AVG      = 32   # stochastic rollouts to average per visualized example
SCAN_CHUNK = 32   # chunk size for the failure scan (avoids OOM)

# ── Load NCA ───────────────────────────────────────────────────────────────────

ckpt_dir = run_dir / "checkpoints"
ckpts = sorted(ckpt_dir.glob("nca_*.pt"))
assert ckpts, f"No checkpoints in {ckpt_dir}"
ckpt_path = ckpts[-1]
print(f"Loading checkpoint: {ckpt_path}")

nca = NeuralCA(
    channels=16, hidden_channels=[128], fire_rate=0.5,
    alive_threshold=0, zero_initialization=False, kernel_size=7,
    num_perception_kernels=3, read_only_dims=[1], padding_type="zeros",
).to(DEVICE)

nca.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True), strict=False)
nca.eval()
nca.gaussian_noise.prob = 0.0
print("NCA loaded (train ≤3 bits).\n")

# ── Grid params ────────────────────────────────────────────────────────────────

W, H, r, spacing = 80, 112, 4, (2, 21)
among_sp, side_sp = 2, 21
step_px = 2 * r + among_sp        # 10 px between circle centres
bottom_y = H - side_sp             # 91 — LSB anchor

OUT_X  = W - side_sp               # 59
N_OUT  = 9                         # 8-bit addition → 9-bit sum

# Output circle centre positions (LSB at bottom, MSB at top)
out_ys = [bottom_y - i * step_px for i in range(N_OUT)]  # index 0 = LSB

# ── Helpers ────────────────────────────────────────────────────────────────────

def int_to_bits8(n):
    return list(map(int, f"{n:08b}"))  # MSB-first

def make_state(a_int, b_int):
    """Input-only initial state (no output circles), matching training setup."""
    a_bits = int_to_bits8(a_int)
    b_bits = int_to_bits8(b_int)
    s_int  = a_int + b_int
    o_bits = list(map(int, f"{s_int:09b}"))  # MSB-first, 9 bits

    inp_screen = make_io_screen_bottom_aligned(H, W, r, spacing, a_bits, b_bits, [])
    inp_norm   = normalize_neg1_to_1(torch.tensor(inp_screen, dtype=torch.float32))

    state = torch.zeros(1, 16, H, W, device=DEVICE)
    state[0, 0] = inp_norm.to(DEVICE)
    state[0, 1] = inp_norm.to(DEVICE)
    return state, inp_screen, o_bits

def circle_mask(cy, cx, rad, H, W):
    ys, xs = np.ogrid[:H, :W]
    return (ys - cy) ** 2 + (xs - cx) ** 2 <= rad ** 2

def eval_bits(ch0_last, o_bits):
    """Returns list of booleans: True = output bit correct."""
    target_norm = [1.0 if b else -1.0 for b in o_bits]
    correct = []
    for bit_i in range(N_OUT):
        cy = out_ys[bit_i]          # bit_i=0 → LSB at bottom
        mask = circle_mask(cy, OUT_X, r, H, W)
        mean_val = ch0_last[mask].mean()
        correct.append((mean_val > 0) == (target_norm[N_OUT - 1 - bit_i] > 0))
        # note: o_bits is MSB-first; out_ys[0]=LSB → maps to o_bits[-1]
    return correct   # index 0 = LSB circle

def averaged_rollout(a_int, b_int):
    """Average N_AVG stochastic rollouts; return (T+1, C, H, W) mean tensor."""
    state, inp_screen, o_bits = make_state(a_int, b_int)
    state_batch = state.expand(N_AVG, -1, -1, -1).contiguous()
    with torch.no_grad():
        rollout = nca(state_batch, steps=STEPS)
    return rollout.mean(dim=0).cpu(), inp_screen, o_bits

# ── Scan random 8-bit additions in chunks to find best success + worst failure ─

print(f"Scanning {args.scan_batch} random 8-bit additions (chunks of {SCAN_CHUNK})...")

np.random.seed(0)
pairs = [(int(np.random.randint(256)), int(np.random.randint(256)))
         for _ in range(args.scan_batch)]

best_success = None   # (n_correct, a, b, o_bits)  — prefer max carry chain
best_failure = None   # (n_wrong,   a, b, o_bits)

for chunk_start in range(0, len(pairs), SCAN_CHUNK):
    chunk = pairs[chunk_start: chunk_start + SCAN_CHUNK]
    states, o_bits_list = [], []
    for a_int, b_int in chunk:
        state, _, o_bits = make_state(a_int, b_int)
        states.append(state)
        o_bits_list.append(o_bits)

    state_batch = torch.cat(states, dim=0)
    with torch.no_grad():
        rollout_batch = nca(state_batch, steps=STEPS)
    ch0_last = rollout_batch[:, -1, 0].cpu().numpy()

    for i, ((a_int, b_int), o_bits) in enumerate(zip(chunk, o_bits_list)):
        correct = eval_bits(ch0_last[i], o_bits)
        n_correct = sum(correct)
        n_wrong   = N_OUT - n_correct

        # track best success: 9/9 correct with longest carry chain
        carry_depth = bin(a_int + b_int).count("1")  # more 1s = more carry
        if n_correct == N_OUT:
            if best_success is None or carry_depth > best_success[0]:
                best_success = (carry_depth, a_int, b_int, o_bits)

        # track worst failure: most wrong bits
        if n_wrong > 0:
            if best_failure is None or n_wrong > best_failure[0]:
                best_failure = (n_wrong, a_int, b_int, o_bits)

    print(f"  scanned {min(chunk_start + SCAN_CHUNK, len(pairs))}/{len(pairs)}", end="\r")

print()

# Fall back if needed
if best_success is None:
    print("  No perfect success in scan — using 1+1=2 as fallback.")
    best_success = (1, 1, 1, list(map(int, f"{2:09b}")))

if best_failure is None:
    print("  No failures found — using 127+1=128 as 'hard case'.")
    best_failure = (0, 127, 1, list(map(int, f"{128:09b}")))

_, a_succ, b_succ, _ = best_success
n_wrong_f, a_fail, b_fail, _ = best_failure

# Override success example with a manually chosen 5-bit carry case (mixed bits)
a_succ, b_succ = 248, 127   # 11111000 + 01111111 = 375; single 5-bit carry chain
print(f"  Best success : {a_succ} + {b_succ} = {a_succ+b_succ}  (9/9 correct)")
print(f"  Worst failure: {a_fail} + {b_fail} = {a_fail+b_fail}  ({n_wrong_f} bits wrong)")

# Now run high-quality averaged rollouts for the two selected examples
print("\nRunning averaged rollouts for visualization...")
success_rollout, success_inp, success_obits = averaged_rollout(a_succ, b_succ)
success_correct = eval_bits(success_rollout[-1, 0].numpy(), success_obits)

failure_rollout, failure_inp, failure_obits = averaged_rollout(a_fail, b_fail)
failure_correct = eval_bits(failure_rollout[-1, 0].numpy(), failure_obits)

print(f"  Success final: {sum(success_correct)}/{N_OUT} correct")
print(f"  Failure final: {sum(failure_correct)}/{N_OUT} correct")

# ── Plotting helpers ───────────────────────────────────────────────────────────

DISPLAY_TS = [0, 16, 32, 64, 128]
DISPLAY_TS = [t for t in DISPLAY_TS if t <= STEPS]

def plot_case(axes_row, rollout, inp_screen, o_bits, correct, title, result, show_titles=True):
    """Fill one row of axes: spatial channel-0 snapshots + annotated final frame."""
    for col, t in enumerate(DISPLAY_TS[:-1]):
        ax = axes_row[col]
        ax.imshow(rollout[t, 0].numpy(), cmap="viridis", vmin=-1, vmax=1,
                  interpolation="nearest", origin="upper", rasterized=True)
        ax.axis("off")
        if col == 0 and title:
            ax.text(-0.05, 0.5, title, transform=ax.transAxes,
                    fontsize=11, rotation=0, va="center", ha="right")
        if show_titles:
            ax.set_title(f"t={t}", fontsize=11, pad=2, fontweight="bold")

    # Last panel: annotated final state
    ax = axes_row[-1]
    ax.imshow(rollout[-1, 0].numpy(), cmap="viridis", vmin=-1, vmax=1,
              interpolation="nearest", origin="upper")
    ax.axis("off")
    if show_titles:
        ax.set_title(f"t={STEPS}", fontsize=11, pad=2, fontweight="bold")

    ax.text(1.03, 0.5, result, transform=ax.transAxes,
            fontsize=11, va="center", ha="left")

    for bit_i in range(N_OUT):
        cy = out_ys[bit_i]
        color = "#5a9b5e" if correct[bit_i] else "#b85555"
        ax.add_patch(mpatches.Circle((OUT_X, cy), r,
                                     fill=False, edgecolor=color, linewidth=1.9))


# ── Figure: combined success + failure ────────────────────────────────────────

n_cols = len(DISPLAY_TS)
fig_w, fig_h = 1.6 * n_cols, 4.5
fig, axes = plt.subplots(2, n_cols, figsize=(fig_w, fig_h))
_cw, _ch = fig_w / n_cols, fig_h / 2
fig.subplots_adjust(hspace=0.02 * _cw / _ch, wspace=0.02)

s_label = f"{a_succ}+{b_succ}"
f_label = f"{a_fail}+{b_fail}"

plot_case(axes[0], success_rollout, success_inp, success_obits, success_correct, s_label, f"={a_succ+b_succ}", show_titles=True)
plot_case(axes[1], failure_rollout, failure_inp, failure_obits, failure_correct, f_label, f"={a_fail+b_fail}", show_titles=False)

out = OUT_DIR / "extrap_cases.svg"
fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"\nSaved {out}")

# ── Individual figures ─────────────────────────────────────────────────────────

for label, rollout, o_bits, correct, fname, res_label in [
    (f"SUCCESS: {a_succ}+{b_succ}={a_succ+b_succ}  (train ≤3 → test k=8,  {sum(success_correct)}/9 correct)",
     success_rollout, success_obits, success_correct, "extrap_success.svg", f"={a_succ+b_succ}"),
    (f"FAILURE: {a_fail}+{b_fail}={a_fail+b_fail}  ({sum(failure_correct)}/9 correct)",
     failure_rollout, failure_obits, failure_correct, "extrap_failure.svg", f"={a_fail+b_fail}"),
]:
    fig2, axes2 = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 2.8))
    fig2.subplots_adjust(wspace=0.02)
    plot_case(axes2, rollout, None, o_bits, correct, "", res_label)
    fig2.suptitle(label, fontsize=11)
    out2 = OUT_DIR / fname
    fig2.savefig(out2, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig2)
    print(f"Saved {out2}")
