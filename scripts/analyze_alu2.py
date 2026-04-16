#!/usr/bin/env python3
"""
E_alu2 analysis — 8-bit ALU v2 (ADD/SUB/AND/OR/XOR/NOT/RCL/RCR + carry + branch).

Grid: 96×112, r=4
  x=12  A[7:0]
  x=24  B[7:0]  (blank for NOT/RCL/RCR)
  x=48  CTRL: op[2:0] + carry_in + cond[2:0]  — 7 bits
  x=82  OUT: result[7:0] + carry_out + branch_taken  — 10 bits

Improvements over ALU v1:  zi=False, grad_clip, noise injection, RCL/RCR ops,
hidden=[128], separate carry + branch outputs.

Usage:
    uv run python scripts/analyze_alu2.py

Outputs under results/E_alu2/:
    summary.csv           — per-run metrics
    loss_curve.svg        — training loss (binned mean ± std)
    bits_curve.svg        — valid bits over time
    snapshot.svg          — latest snapshot
    layout_diagram.svg    — schematic of ALU v2 grid layout
    per_op_accuracy.svg   — per-op accuracy breakdown (result / carry / branch)
    rollout_ch0.svg       — channel 0 rollout for all 8 ops (fixed inputs)
    rollout_allch.svg     — all channels for ADD operation
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.dataset import ALU2Dataset, ALU2_OP_NAMES, ALU2_COND_NAMES, _compute_alu2, _int_to_bits_msb
from ncpu.nca import NeuralCA
from ncpu.utils import make_alu2_screen

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results/E_alu2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

random.seed(42)
torch.manual_seed(42)

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

COL_A   = "#4E79A7"
COL_B   = "#F28E2B"
COL_OP  = "#59A14F"
COL_OUT = "#E15759"
COL_BG  = "#f7f7f7"

N_BITS  = 10   # output bits: result[7:0] + carry_out + branch_taken

# ── Find runs ──────────────────────────────────────────────────────────────────

def find_runs():
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get("run", {}).get("experiment") != "E_alu2":
            continue
        log = open(d / "log.jsonl").readlines()
        if len(log) < 100:
            continue
        runs.append(d)
    return sorted(runs)

runs = find_runs()
print(f"Found {len(runs)} E_alu2 run(s):")
for d in runs:
    last = json.loads(open(d / "log.jsonl").readlines()[-1])
    print(f"  {d.name}  steps={last['step']}  bits={last.get('num_valid_bits',0):.2f}/10")

# Use the run with the most steps (or best bits)
run_dir = max(runs, key=lambda d: json.loads(open(d / "log.jsonl").readlines()[-1])["step"])
print(f"\nUsing: {run_dir.name}")

# ── Load log ───────────────────────────────────────────────────────────────────

log    = [json.loads(l) for l in open(run_dir / "log.jsonl")]
steps  = np.array([e["step"] for e in log])
losses = np.array([e["loss"] for e in log])
bits   = np.array([e["num_valid_bits"] for e in log])

# ── Summary CSV ────────────────────────────────────────────────────────────────

def steps_to_thresh(log, thresh):
    for e in log:
        if e.get("num_valid_bits", 0) >= thresh:
            return e["step"]
    return None

conv8  = steps_to_thresh(log, 8)
conv9  = steps_to_thresh(log, 9)
conv10 = steps_to_thresh(log, 10)
last   = log[-1]

last10k = bits[-10000:]
pct_ge8  = (last10k >= 8).mean()
pct_ge9  = (last10k >= 9).mean()
pct_eq10 = (last10k >= 10).mean()

with open(OUT_DIR / "summary.csv", "w") as f:
    f.write("run,final_step,final_loss,last_valid_bits,first_8bit_step,first_9bit_step,first_10bit_step,"
            "last10k_pct_ge8,last10k_pct_ge9,last10k_pct_eq10\n")
    f.write(f"{run_dir.name},{last['step']},{last['loss']:.6f},"
            f"{last['num_valid_bits']:.3f},"
            f"{conv8 if conv8 else 'none'},"
            f"{conv9 if conv9 else 'none'},"
            f"{conv10 if conv10 else 'none'},"
            f"{pct_ge8:.3f},{pct_ge9:.3f},{pct_eq10:.3f}\n")

print(f"Steps: {last['step']}  Loss: {last['loss']:.4f}  "
      f"Bits: {last['num_valid_bits']:.2f}/10")
print(f"First 8/10: step {conv8}   First 9/10: step {conv9}   First 10/10: step {conv10}")
print(f"Last 10k — ≥8: {pct_ge8*100:.1f}%  ≥9: {pct_ge9*100:.1f}%  =10: {pct_eq10*100:.1f}%")

# ── Binning helper ─────────────────────────────────────────────────────────────

N_BINS = 1000

def bin_series(arr, n_bins=N_BINS):
    idx = np.linspace(0, len(arr), n_bins + 1, dtype=int)
    x, mean, std = [], [], []
    for i in range(n_bins):
        chunk = arr[idx[i]:idx[i+1]]
        if len(chunk) == 0:
            continue
        x.append(steps[idx[i]])
        mean.append(chunk.mean())
        std.append(chunk.std())
    return np.array(x), np.array(mean), np.array(std)

# ── Loss curve ─────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
x, mean, std = bin_series(losses)
ax.fill_between(x, np.maximum(mean - std, 1e-4), mean + std,
                color=COL_A, alpha=0.2)
ax.plot(x, mean, color=COL_A, linewidth=1.5, label=f"mean ± std ({N_BINS} bins)")
ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss (masked MSE)")
ax.set_title("E_alu2 — ALU v2: training loss")
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
           label=f"{N_BITS}/10 full convergence")
for thresh, step, col, lbl in [
    (8,  conv8,  "#4E79A7", "first 8/10"),
    (9,  conv9,  "#59A14F", "first 9/10"),
    (10, conv10, "#E15759", "first 10/10"),
]:
    if step is not None:
        ax.axvline(step, color=col, linestyle=":", linewidth=1.2,
                   label=f"{lbl} at step {step:,}")
ax.set_ylim(0, N_BITS + 0.5)
ax.set_xlabel("Training step")
ax.set_ylabel("Valid bits / 10")
ax.set_title("E_alu2 — ALU v2: valid bits over time")
ax.legend(fontsize=8)
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
    ax.set_title("E_alu2 — latest snapshot (input | target | NCA output | diff)", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "snapshot.svg", dpi=150)
    plt.close(fig)
    print("Saved: snapshot.svg")

# ── Layout diagram ─────────────────────────────────────────────────────────────

ds_cfg = json.load(open(run_dir / "config.json"))["ds"]
H_D    = ds_cfg["H"]       # 112
W_D    = ds_cfg["W"]       # 96
r_d    = ds_cfg["r"]       # 4
sp     = ds_cfg["among_sp"] # 2
x_a    = ds_cfg["x_a"]     # 12
x_b    = ds_cfg["x_b"]     # 24
x_ctrl = ds_cfg["x_ctrl"]  # 48
x_out  = ds_cfg["x_out"]   # 82
step_px = 2 * r_d + sp     # 10

def _col_centers(n, cx):
    v  = n * 2*r_d + sp * (n-1)
    tm = (H_D - v) // 2
    return [(cx, tm + r_d + i*step_px) for i in range(n)]

def _draw_circle(ax, cx, cy, r, color, label, fontsize=4.5):
    ax.add_patch(mpatches.Circle((cx, cy), r, color=color, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold", zorder=4)

scale = 0.028
p_w   = W_D * scale
p_h   = H_D * scale

fig, ax = plt.subplots(figsize=(p_w + 0.4, p_h + 1.1))
fig.subplots_adjust(left=0.10, right=0.92, bottom=0.22, top=0.88)

ax.set_xlim(0, W_D); ax.set_ylim(H_D, 0); ax.set_aspect("equal")
ax.set_facecolor(COL_BG)
ax.add_patch(mpatches.FancyBboxPatch((0,0), W_D, H_D,
    boxstyle="square,pad=0", linewidth=1, edgecolor="#aaaaaa", facecolor=COL_BG))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("ALU v2 layout  (96×112, r=4)", fontsize=8, pad=4)

# A col (8 bits)
for (cx, cy), lbl in zip(_col_centers(8, x_a), [f"A{7-i}" for i in range(8)]):
    _draw_circle(ax, cx, cy, r_d, COL_A, lbl)
# B col (8 bits)
for (cx, cy), lbl in zip(_col_centers(8, x_b), [f"B{7-i}" for i in range(8)]):
    _draw_circle(ax, cx, cy, r_d, COL_B, lbl)
# CTRL col (7 bits: op2 op1 op0 ci c2 c1 c0)
ctrl_labels = ["op2","op1","op0","ci","c2","c1","c0"]
for (cx, cy), lbl in zip(_col_centers(7, x_ctrl), ctrl_labels):
    _draw_circle(ax, cx, cy, r_d, COL_OP, lbl)
# OUT col (10 bits: res7..res0 cout bt)
out_labels = [f"r{7-i}" for i in range(8)] + ["co","bt"]
for (cx, cy), lbl in zip(_col_centers(10, x_out), out_labels):
    _draw_circle(ax, cx, cy, r_d, COL_OUT, lbl)

# Distance annotations
for x0, x1, y_off, lbl in [
    (x_a,   x_b,   H_D+4, f"A–B {x_b-x_a}px"),
    (x_b,   x_ctrl, H_D+4, f"B→ctrl {x_ctrl-x_b}px"),
    (x_ctrl, x_out, H_D+4, f"ctrl→out {x_out-x_ctrl}px"),
]:
    ax.annotate("", xy=(x1, y_off), xytext=(x0, y_off),
                arrowprops=dict(arrowstyle="<->", color="#777777", lw=0.9))
    ax.text((x0+x1)/2, y_off+4, lbl, ha="center", va="top",
            fontsize=5, color="#555555")

handles = [
    mpatches.Patch(color=COL_A,   label="A[7:0]"),
    mpatches.Patch(color=COL_B,   label="B[7:0]  (blank: NOT/RCL/RCR)"),
    mpatches.Patch(color=COL_OP,  label="CTRL: op[2:0]+carry_in+cond[2:0]"),
    mpatches.Patch(color=COL_OUT, label="OUT: result[7:0]+carry_out+branch_taken"),
]
fig.legend(handles=handles, loc="lower center", fontsize=6, ncol=2,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.01))
fig.suptitle("E_alu2 — 8-bit ALU v2 layout", fontsize=9, y=0.97)
fig.savefig(OUT_DIR / "layout_diagram.svg", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("Saved: layout_diagram.svg")

# ── Load NCA ───────────────────────────────────────────────────────────────────

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

print("\n── Loading NCA ──")
nca = load_nca(run_dir)
print(f"  Loaded: {run_dir.name}")

# ── Helpers ────────────────────────────────────────────────────────────────────

def make_state(a_int, b_int, carry_in, op, cond):
    uses_b = op not in (5, 6, 7)
    ctrl   = _int_to_bits_msb(op, 3) + [carry_in] + _int_to_bits_msb(cond, 3)
    screen = make_alu2_screen(
        H_D, W_D, r_d, sp, x_a, x_b, x_ctrl, x_out,
        a_bits    = _int_to_bits_msb(a_int, 8),
        b_bits    = _int_to_bits_msb(b_int, 8) if uses_b else None,
        ctrl_bits = ctrl,
    )
    img   = torch.from_numpy(screen).float() / 128.0 - 1.0
    state = torch.zeros(1, nca.channels, H_D, W_D)
    state[0, 0] = img
    state[0, 1] = img
    return state

def make_target_screen(result, carry_out, branch_taken):
    out_bits = _int_to_bits_msb(result, 8) + [carry_out, branch_taken]
    screen   = make_alu2_screen(H_D, W_D, r_d, sp, x_a, x_b, x_ctrl, x_out,
                                out_bits=out_bits)
    return torch.from_numpy(screen).float() / 128.0 - 1.0

def get_rollout(state, max_t=96):
    with torch.no_grad():
        return nca(state, steps=max_t)[0]   # (T+1, C, H, W)

# Pre-compute output bit masks
ds_ns = __import__("argparse").Namespace(**ds_cfg)
ds    = ALU2Dataset(ds_ns)
bit_masks = ds.get_output_bit_masks()   # (10, H, W)

# ── Per-op accuracy breakdown ──────────────────────────────────────────────────

print("\n── Per-op accuracy ──")
N_ACC   = 64
BATCH   = 64
T_EVAL  = 64

# op_acc[op] = dict with result_bits (0-7), carry_out (8), branch_taken (9) accuracies
op_acc = {}

for op in range(8):
    bit_correct = np.zeros(N_BITS)
    n_samples   = 0

    for batch_start in range(0, N_ACC, BATCH):
        bs = min(BATCH, N_ACC - batch_start)
        states     = []
        expected   = []   # (bs, 10) binary

        for _ in range(bs):
            a    = random.randint(0, 255)
            b    = random.randint(0, 255)
            ci   = random.randint(0, 1)
            cond = random.randint(0, 7)
            result, cout, branch = _compute_alu2(a, b, ci, op, cond)
            out_bits = _int_to_bits_msb(result, 8) + [cout, branch]
            states.append(make_state(a, b, ci, op, cond))
            expected.append(out_bits)

        state_batch = torch.cat(states, dim=0)   # (bs, C, H, W)
        with torch.no_grad():
            rollout_batch = nca(state_batch, steps=T_EVAL)   # (bs, T+1, C, H, W)
        last_ch0 = rollout_batch[:, -1, 0]   # (bs, H, W)

        for i in range(N_BITS):
            mask    = bit_masks[i]   # (H, W)
            avg_act = (last_ch0 * mask.unsqueeze(0)).sum(dim=(-2,-1)) / mask.sum()
            pred    = (avg_act > 0).int().tolist()
            gt      = [expected[j][i] for j in range(bs)]
            bit_correct[i] += sum(p == g for p, g in zip(pred, gt))

        n_samples += bs

    acc = bit_correct / n_samples
    op_acc[op] = {
        "result":       acc[:8].mean(),
        "carry_out":    acc[8],
        "branch_taken": acc[9],
        "per_bit":      acc,
    }
    print(f"  {ALU2_OP_NAMES[op]:4s}  result={acc[:8].mean()*100:.1f}%  "
          f"carry={acc[8]*100:.1f}%  branch={acc[9]*100:.1f}%")

# Plot per-op accuracy — grouped bar chart
fig, ax = plt.subplots(figsize=(4.5, 3.8))

x       = np.arange(8)
bw      = 0.24   # bar width
off     = 0.26   # centre offset (gap between bars = off - bw = 0.02)
op_lbls = ALU2_OP_NAMES

r_vals = [op_acc[op]["result"]       * 100 for op in range(8)]
c_vals = [op_acc[op]["carry_out"]    * 100 for op in range(8)]
b_vals = [op_acc[op]["branch_taken"] * 100 for op in range(8)]

ax.bar(x - off, r_vals, bw, color=COL_A,   alpha=0.85, label="result byte")
ax.bar(x,       c_vals, bw, color=COL_B,   alpha=0.85, label="carry out")
ax.bar(x + off, b_vals, bw, color=COL_OUT, alpha=0.85, label="branch taken")

ax.set_xlim(-0.5, 7.5)
ax.set_ylim(0, 104)
ax.set_xticks(x)
ax.set_xticklabels(op_lbls, fontsize=9)
ax.set_ylabel("Accuracy (%)", fontsize=9)
ax.tick_params(axis="y", labelsize=8)
ax.grid(False)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.12),
          ncol=3, frameon=True, framealpha=0.9, edgecolor="#cccccc",
          handlelength=1.2, handletextpad=0.4, columnspacing=1.0)

fig.tight_layout(rect=[0, 0.10, 1, 1])
fig.savefig(OUT_DIR / "per_op_accuracy.svg")
plt.close(fig)
print("Saved: per_op_accuracy.svg")

# ── Rollout figures ────────────────────────────────────────────────────────────

# Fixed showcase inputs
A_FIXED    = 0b10110101   # 181
B_FIXED    = 0b01101110   # 110
CI_FIXED   = 1
COND_FIXED = 6   # AL = always taken

TSTEPS = [0, 8, 16, 32, 64, 96]

def _op_label(op):
    a, b, ci, cond = A_FIXED, B_FIXED, CI_FIXED, COND_FIXED
    result, cout, branch = _compute_alu2(a, b, ci, op, cond)
    return f"op{op}:{ALU2_OP_NAMES[op]}  →{result:02X} co={cout} bt={branch}"

COMBOS = [(A_FIXED, B_FIXED, CI_FIXED, COND_FIXED, op) for op in range(8)]

print("\n── Rollout figures ──")

n_rows = len(COMBOS)
n_cols = len(TSTEPS) + 1

# ── ch0 rollout ────────────────────────────────────────────────────────────────

cell = 0.48
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * cell, n_rows * cell),
                         gridspec_kw={"hspace": 0.02, "wspace": 0.02},
                         constrained_layout=True)

for ri, (a, b, ci, cond, op) in enumerate(COMBOS):
    state   = make_state(a, b, ci, op, cond)
    rollout = get_rollout(state)
    result, cout, branch = _compute_alu2(a, b, ci, op, cond)
    target  = make_target_screen(result, cout, branch)

    for ci_idx, t in enumerate(TSTEPS):
        t_idx = min(t, rollout.shape[0] - 1)
        ax = axes[ri][ci_idx]
        ax.imshow(rollout[t_idx, 0].numpy(), cmap="viridis",
                  vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(f"t={t}", fontsize=7, pad=2)
        if ci_idx == 0:
            ax.set_ylabel(_op_label(op), fontsize=5.5)

    ax = axes[ri][n_cols - 1]
    ax.imshow(target.numpy(), cmap="viridis", vmin=-1, vmax=1,
              interpolation="nearest", rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    if ri == 0:
        ax.set_title("target", fontsize=7, pad=2)

fig.suptitle(
    f"E_alu2 — ch0 over time  "
    f"(A=0x{A_FIXED:02X}, B=0x{B_FIXED:02X}, carry_in={CI_FIXED}, cond=AL)",
    fontsize=10
)
fig.savefig(OUT_DIR / "rollout_ch0.svg", dpi=150, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("    Saved: rollout_ch0.svg")

# ── all-channels rollout (ADD, hardest carry chain) ────────────────────────────

a, b, ci, cond, op = A_FIXED, B_FIXED, CI_FIXED, COND_FIXED, 0   # ADD
state   = make_state(a, b, ci, op, cond)
rollout = get_rollout(state)
C       = rollout.shape[1]
n_ts    = len(TSTEPS)

cell = 0.48
fig, axes = plt.subplots(C, n_ts,
                         figsize=(n_ts * cell, C * cell),
                         gridspec_kw={"hspace": 0.02, "wspace": 0.02},
                         constrained_layout=True)

for ch in range(C):
    for ci_idx, t in enumerate(TSTEPS):
        t_idx = min(t, rollout.shape[0] - 1)
        ax = axes[ch][ci_idx]
        ax.imshow(rollout[t_idx, ch].numpy(), cmap="viridis",
                  vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if ch == 0:
            ax.set_title(f"t={t}", fontsize=7, pad=2)
        if ci_idx == 0:
            tag = "out" if ch == 0 else ("in(r/o)" if ch == 1 else f"h{ch-1}")
            ax.set_ylabel(tag, fontsize=6)

result, cout, branch = _compute_alu2(a, b, ci, op, cond)
fig.suptitle(
    f"E_alu2 — all channels — ADD  "
    f"0x{a:02X}+0x{b:02X}+{ci}=0x{result:02X} co={cout} bt={branch}",
    fontsize=9
)
fig.savefig(OUT_DIR / "rollout_allch.svg", dpi=150, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("    Saved: rollout_allch.svg")

# ── Training assessment ────────────────────────────────────────────────────────

print("\n── Training assessment ──")

# Overall per-bit accuracy across all ops
all_result_acc  = np.mean([op_acc[op]["result"]       for op in range(8)])
all_carry_acc   = np.mean([op_acc[op]["carry_out"]    for op in range(8)])
all_branch_acc  = np.mean([op_acc[op]["branch_taken"] for op in range(8)])

# Loss slope in last 50k steps
last50k_loss = losses[-50000:]
slope_pct    = (last50k_loss[-1] - last50k_loss[0]) / last50k_loss[0] * 100

print(f"  Mean result accuracy:  {all_result_acc*100:.1f}%")
print(f"  Mean carry_out acc:    {all_carry_acc*100:.1f}%")
print(f"  Mean branch_taken acc: {all_branch_acc*100:.1f}%  (random baseline: 50%)")
print(f"  Loss change over last 50k steps: {slope_pct:.1f}%")

if all_branch_acc < 0.65:
    verdict = (
        "branch_taken is barely above random — the model has not learned to compute "
        "zero/neg flags internally. More training is unlikely to close this gap without "
        "increasing rollout depth (steps_max) or model capacity (hidden channels)."
    )
elif slope_pct < -10:
    verdict = (
        "Loss is still falling meaningfully. More training will likely improve "
        "branch_taken accuracy further."
    )
else:
    verdict = (
        "Loss has largely plateaued. Marginal returns on continued training. "
        "Consider increasing model capacity or rollout depth."
    )
print(f"\n  Verdict: {verdict}")

with open(OUT_DIR / "assessment.txt", "w") as f:
    f.write(f"Run: {run_dir.name}\n")
    f.write(f"Final step: {last['step']}  Loss: {last['loss']:.4f}\n\n")
    f.write(f"Mean result accuracy:   {all_result_acc*100:.1f}%\n")
    f.write(f"Mean carry_out acc:     {all_carry_acc*100:.1f}%\n")
    f.write(f"Mean branch_taken acc:  {all_branch_acc*100:.1f}%  (random=50%)\n")
    f.write(f"Loss change last 50k:   {slope_pct:.1f}%\n\n")
    f.write(f"Verdict: {verdict}\n")

print(f"\nAll outputs in {OUT_DIR}/")
