#!/usr/bin/env python3
"""
E4 analysis — 8-bit ALU v1 (ADD/SUB/AND/OR/XOR/NOT/SHL/SHR).

Layout (128×112, r=4):
  Left   : A[7:0] (col 0 at x=20) + B[7:0] (col 1 at x=30)
  Middle : opcode[2:0] at x=64
  Right  : result[7:0] at x=108

Output: 8 bits (result only — no carry, no branch flag).
Note: this run used zero_initialization=True, later identified as suboptimal
by the E_ablations experiment.

Usage:
    uv run python scripts/analyze/analyze_e4.py

Outputs under results/E4/:
    summary.csv           — per-run metrics
    loss_curve.svg        — training loss over time
    bits_curve.svg        — valid bits over time
    snapshot.svg          — best snapshot
    layout_diagram.svg    — schematic of ALU v1 grid layout
    rollout_ch0.svg       — channel 0 rollout for example inputs per opcode
    rollout_allch.svg     — all channels for highlight combo
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.utils import make_alu_screen

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results/E4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.edgecolor":    "#cccccc",
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        "#eeeeee",
    "grid.linewidth":    0.7,
    "xtick.color":       "#555555",
    "ytick.color":       "#555555",
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "text.color":        "#222222",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
    "legend.framealpha": 0.95,
    "legend.edgecolor":  "#cccccc",
    "legend.fontsize":   8,
})

COLOR = "#4E79A7"

OP_NAMES = ["ADD", "SUB", "AND", "OR", "XOR", "NOT", "SHL", "SHR"]

# ── Find runs ──────────────────────────────────────────────────────────────────

def find_runs():
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get("run", {}).get("experiment") != "E4":
            continue
        log = open(d / "log.jsonl").readlines()
        if len(log) < 10:   # skip aborted runs
            continue
        runs.append(d)
    return runs

runs = find_runs()
print(f"Found {len(runs)} E4 run(s):")
for d in runs:
    last = json.loads(open(d / "log.jsonl").readlines()[-1])
    print(f"  {d.name}  steps={last['step']}  bits={last['num_valid_bits']:.1f}")

# ── Load log ───────────────────────────────────────────────────────────────────

def load_log(run_dir):
    return [json.loads(l) for l in open(run_dir / "log.jsonl")]

assert len(runs) == 1, "Expected exactly one E4 run"
run_dir = runs[0]
log     = load_log(run_dir)

# ── Summary CSV ────────────────────────────────────────────────────────────────

N_BITS = 8

def steps_to_full_bits(log, n_bits=N_BITS):
    for entry in log:
        if entry.get("num_valid_bits", 0) >= n_bits:
            return entry["step"]
    return None

conv = steps_to_full_bits(log)
last = log[-1]
with open(OUT_DIR / "summary.csv", "w") as f:
    f.write("run,seed,final_step,final_loss,valid_bits,conv_step\n")
    f.write(f"{run_dir.name},0,{last['step']},{last['loss']:.6f},"
            f"{last['num_valid_bits']:.3f},{conv if conv is not None else 'none'}\n")

print(f"\nSteps: {last['step']}  Loss: {last['loss']:.4f}  "
      f"Bits: {last['num_valid_bits']:.1f}/8  "
      f"Conv: {conv if conv is not None else 'none'}")

# ── Loss curve ─────────────────────────────────────────────────────────────────

steps  = np.array([e["step"] for e in log])
losses = np.array([e["loss"] for e in log])
bits   = np.array([e["num_valid_bits"] for e in log])

# Bin into ~1000 equal-width buckets → mean ± std per bucket (keeps SVG tiny)
N_BINS = 1000

def bin_series(arr, n_bins=N_BINS):
    idx   = np.linspace(0, len(arr), n_bins + 1, dtype=int)
    x, mean, std = [], [], []
    for i in range(n_bins):
        chunk = arr[idx[i]:idx[i+1]]
        if len(chunk) == 0:
            continue
        x.append(steps[idx[i]])
        mean.append(chunk.mean())
        std.append(chunk.std())
    return np.array(x), np.array(mean), np.array(std)

fig, ax = plt.subplots(figsize=(10, 4))
x, mean, std = bin_series(losses)
ax.fill_between(x, np.maximum(mean - std, 1e-4), mean + std, color=COLOR, alpha=0.2)
ax.plot(x, mean, color=COLOR, linewidth=1.5, label=f"mean ± std ({N_BINS} bins)")
ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss (masked MSE)")
ax.set_title("E4 — 8-bit ALU v1: training loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "loss_curve.svg")
plt.close(fig)
print("Saved: loss_curve.svg")

# ── Valid bits curve ───────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
x, mean, std = bin_series(bits)
ax.fill_between(x, mean - std, mean + std, color="#F28E2B", alpha=0.2)
ax.plot(x, mean, color="#F28E2B", linewidth=1.5, label=f"mean ± std ({N_BINS} bins)")
ax.axhline(N_BITS, color="#aaaaaa", linestyle="--", linewidth=1,
           label=f"{N_BITS} bits (full convergence)")
if conv is not None:
    ax.axvline(conv, color="#E15759", linestyle=":", linewidth=1,
               label=f"first 8/8 at step {conv:,}")
ax.set_ylim(0, N_BITS + 0.5)
ax.set_xlabel("Training step")
ax.set_ylabel("Valid bits")
ax.set_title("E4 — 8-bit ALU v1: valid bits over time")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "bits_curve.svg")
plt.close(fig)
print("Saved: bits_curve.svg")

# ── Snapshot ───────────────────────────────────────────────────────────────────

snap = run_dir / "snapshot_latest.png"
if snap.exists():
    from PIL import Image
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.imshow(Image.open(snap), rasterized=True)
    ax.axis("off")
    ax.set_title("E4 — final snapshot (input | target | NCA output | diff)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "snapshot.svg", dpi=150)
    plt.close(fig)
    print("Saved: snapshot.svg")

# ── Layout diagram ─────────────────────────────────────────────────────────────

H_D, W_D = 112, 128
r_d      = 4
among_sp = 2
side_sp  = 20

COL_A   = "#4E79A7"
COL_B   = "#F28E2B"
COL_OP  = "#59A14F"
COL_OUT = "#E15759"
COL_BG  = "#f7f7f7"

def _col_centers(n, cx, H, r, sp):
    v  = n * 2*r + sp * (n-1)
    tm = (H - v) // 2
    return [(cx, tm + r + i*(2*r+sp)) for i in range(n)]

def _draw_circle(ax, cx, cy, r, color, label, fontsize=4.5):
    ax.add_patch(mpatches.Circle((cx, cy), r, color=color, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold", zorder=4)

step_px = 2 * r_d + among_sp   # 10 px

scale = 0.022
p_w   = W_D * scale
p_h   = H_D * scale
lm, rm = 0.10, 0.10
tm_m, bm_m = 0.55, 0.45

fig, ax = plt.subplots(figsize=(p_w + lm + rm, p_h + tm_m + bm_m))
fig.subplots_adjust(
    left   = lm / (p_w + lm + rm),
    right  = 1 - rm / (p_w + lm + rm),
    bottom = bm_m / (p_h + tm_m + bm_m),
    top    = 1 - tm_m / (p_h + tm_m + bm_m),
)

ax.set_xlim(0, W_D); ax.set_ylim(H_D, 0); ax.set_aspect("equal")
ax.set_facecolor(COL_BG)
ax.add_patch(mpatches.FancyBboxPatch((0,0), W_D, H_D,
    boxstyle="square,pad=0", linewidth=1, edgecolor="#aaaaaa", facecolor=COL_BG))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("ALU v1 layout  (128×112, r=4)", fontsize=8, pad=4)

# A col
for (cx, cy), lbl in zip(_col_centers(8, side_sp, H_D, r_d, among_sp),
                          [f"A{7-i}" for i in range(8)]):
    _draw_circle(ax, cx, cy, r_d, COL_A, lbl)
# B col
for (cx, cy), lbl in zip(_col_centers(8, side_sp + step_px, H_D, r_d, among_sp),
                          [f"B{7-i}" for i in range(8)]):
    _draw_circle(ax, cx, cy, r_d, COL_B, lbl)
# Opcode col (middle)
for (cx, cy), lbl in zip(_col_centers(3, W_D // 2, H_D, r_d, among_sp),
                          ["op2", "op1", "op0"]):
    _draw_circle(ax, cx, cy, r_d, COL_OP, lbl)
# Result col
for (cx, cy), lbl in zip(_col_centers(8, W_D - side_sp, H_D, r_d, among_sp),
                          [f"R{7-i}" for i in range(8)]):
    _draw_circle(ax, cx, cy, r_d, COL_OUT, lbl)

# Distance annotations
for x0, x1, y, lbl in [
    (side_sp, W_D//2,       H_D + 5, f"A→op  {W_D//2 - side_sp}px"),
    (W_D//2,  W_D-side_sp,  H_D + 5, f"op→R  {W_D - side_sp - W_D//2}px"),
]:
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="<->", color="#777777", lw=1.0))
    ax.text((x0+x1)/2, y+4, lbl, ha="center", va="top", fontsize=5.5, color="#555555")

handles = [
    mpatches.Patch(color=COL_A,   label="A[7:0]"),
    mpatches.Patch(color=COL_B,   label="B[7:0]"),
    mpatches.Patch(color=COL_OP,  label="opcode[2:0]"),
    mpatches.Patch(color=COL_OUT, label="result[7:0]"),
]
fig.legend(handles=handles, loc="lower center", fontsize=6.5, ncol=4,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.02))
fig.suptitle("E4 — 8-bit ALU v1 layout", fontsize=9, y=0.99)
fig.savefig(OUT_DIR / "layout_diagram.svg", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("\nSaved: layout_diagram.svg")

# ── Rollout helpers ────────────────────────────────────────────────────────────

TSTEPS = [0, 8, 16, 32, 64, 96]

def load_nca(run_dir):
    nc  = json.load(open(run_dir / "config.json"))["nca"]
    nca = NeuralCA(
        channels=nc["channels"],
        hidden_channels=nc["hidden_channels"],
        fire_rate=nc["fire_rate"],
        alive_threshold=nc["alive_threshold"],
        zero_initialization=nc["zero_initialization"],
        kernel_size=nc["kernel_size"],
        num_perception_kernels=nc["num_perception_kernels"],
        read_only_dims=nc["read_only_dims"],
        padding_type=nc.get("padding_type", "zeros"),
    )
    last_ckpt = sorted((run_dir / "checkpoints").glob("nca_*.pt"))[-1]
    nca.load_state_dict(
        torch.load(last_ckpt, map_location="cpu", weights_only=True), strict=False
    )
    nca.eval()
    return nca

def int_to_bits(n, width):
    return [int(b) for b in f"{n:0{width}b}"]

def _compute_alu(a, b, op):
    if   op == 0: return (a + b) & 0xFF
    elif op == 1: return (a - b) & 0xFF
    elif op == 2: return a & b
    elif op == 3: return a | b
    elif op == 4: return a ^ b
    elif op == 5: return (~a) & 0xFF
    elif op == 6: return (a << 1) & 0xFF
    else:         return a >> 1

ds_cfg = json.load(open(run_dir / "config.json"))["ds"]
spacing = ds_cfg["spacing"]

def make_state(a_int, b_int, op):
    uses_b = op < 5
    screen = make_alu_screen(
        H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"], spacing=spacing,
        a_bits=int_to_bits(a_int, 8),
        b_bits=int_to_bits(b_int, 8) if uses_b else None,
        opcode_bits=int_to_bits(op, 3),
    )
    img = torch.from_numpy(screen).float() / 128.0 - 1.0
    state = torch.zeros(1, 16, ds_cfg["H"], ds_cfg["W"])
    state[0, 0] = img
    state[0, 1] = img
    return state

def make_target(a_int, b_int, op):
    result = _compute_alu(a_int, b_int, op)
    screen = make_alu_screen(
        H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"], spacing=spacing,
        result_bits=int_to_bits(result, 8),
    )
    return torch.from_numpy(screen).float() / 128.0 - 1.0

def get_rollout(nca, state, max_t=96):
    with torch.no_grad():
        return nca.forward(state, steps=max_t)[0]

# One fixed A/B pair, all 8 opcodes
A_FIXED = 0b10110101   # 181
B_FIXED = 0b01101110   # 110

COMBOS = [
    (A_FIXED, B_FIXED, op, f"op{op}:{OP_NAMES[op]}  {A_FIXED}⊕{B_FIXED}={_compute_alu(A_FIXED,B_FIXED,op)}")
    for op in range(8)
]
HIGHLIGHT = 0   # ADD with carry propagation

print("\n── Rollout figures ──")
nca = load_nca(run_dir)
print(f"  Loaded: {run_dir.name}")

n_rows = len(COMBOS)
n_cols = len(TSTEPS) + 1

# ── ch0 rollout ────────────────────────────────────────────────────────────────

cell = 0.50
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * cell, n_rows * cell),
                         gridspec_kw={"hspace": 0.02, "wspace": 0.02},
                         constrained_layout=True)

for ri, (a, b, op, label) in enumerate(COMBOS):
    state   = make_state(a, b, op)
    rollout = get_rollout(nca, state)
    target  = make_target(a, b, op)

    for ci, t in enumerate(TSTEPS):
        t_idx = min(t, rollout.shape[0] - 1)
        ax = axes[ri][ci]
        ax.imshow(rollout[t_idx, 0].numpy(), cmap="viridis",
                  vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(f"t={t}", fontsize=7, pad=2)
        if ci == 0:
            ax.set_ylabel(label, fontsize=5.5)

    ax = axes[ri][n_cols - 1]
    ax.imshow(target.numpy(), cmap="viridis", vmin=-1, vmax=1,
              interpolation="nearest", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    if ri == 0:
        ax.set_title("target", fontsize=7, pad=2)

fig.suptitle(f"E4 — ALU v1, channel 0 over time  (A=0x{A_FIXED:02X}, B=0x{B_FIXED:02X})",
             fontsize=10)
fig.savefig(OUT_DIR / "rollout_ch0.svg", dpi=150, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("    Saved: rollout_ch0.svg")

# ── all-channels rollout ───────────────────────────────────────────────────────

a, b, op, hi_label = COMBOS[HIGHLIGHT]
state   = make_state(a, b, op)
rollout = get_rollout(nca, state)
C       = rollout.shape[1]
n_ts    = len(TSTEPS)

cell = 0.50
fig, axes = plt.subplots(C, n_ts,
                         figsize=(n_ts * cell, C * cell),
                         gridspec_kw={"hspace": 0.02, "wspace": 0.02},
                         constrained_layout=True)
for ch in range(C):
    for ci, t in enumerate(TSTEPS):
        t_idx = min(t, rollout.shape[0] - 1)
        ax = axes[ch][ci]
        ax.imshow(rollout[t_idx, ch].numpy(), cmap="viridis",
                  vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if ch == 0:
            ax.set_title(f"t={t}", fontsize=7, pad=2)
        if ci == 0:
            tag = "out" if ch == 0 else ("in (r/o)" if ch == 1 else f"h{ch-1}")
            ax.set_ylabel(tag, fontsize=6)

fig.suptitle(f"E4 — all channels — {hi_label}", fontsize=9)
fig.savefig(OUT_DIR / "rollout_allch.svg", dpi=150, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("    Saved: rollout_allch.svg")

print(f"\nAll outputs in {OUT_DIR}/")
