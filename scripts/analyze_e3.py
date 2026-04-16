#!/usr/bin/env python3
"""
E3 analysis — 8-bit adder benchmark (cols2 layout).

All runs use cols2 layout: A[7:0] in column 0, B[7:0] in column 1, 9-bit
output (carry + sum[7:0]) on the right. Grid: 80×112, r=4.

Usage:
    uv run python scripts/analyze_e3.py

Outputs under results/E3/:
    summary.csv              — per-run metrics
    loss_curves.svg          — loss over time, all runs
    bits_curves.svg          — valid bits over time, all runs
    snapshots_grid.svg       — best snapshot
    layout_diagram.svg       — schematic of 8-bit cols2 grid layout
    rollout_ch0.svg          — channel 0 rollout for example inputs
    rollout_allch.svg        — all channels for highlight combo
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.utils import make_io_screen

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results/E3")
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

PALETTE = ["#4E79A7", "#F28E2B", "#E15759"]

# ── Find runs ──────────────────────────────────────────────────────────────────

def find_runs():
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get("run", {}).get("experiment") != "E3":
            continue
        runs.append(d)
    return runs

runs = find_runs()
print(f"Found {len(runs)} E3 run(s):")
for d in runs:
    last = json.loads(open(d / "log.jsonl").readlines()[-1])
    print(f"  {d.name}  steps={last['step']}  bits={last['num_valid_bits']}")

# ── Load logs ──────────────────────────────────────────────────────────────────

def load_log(run_dir):
    return [json.loads(l) for l in open(run_dir / "log.jsonl")]

logs = [load_log(d) for d in runs]

# ── Summary CSV ────────────────────────────────────────────────────────────────

N_BITS = 9   # 8-bit adder: carry + sum[7:0]

def steps_to_full_bits(log, n_bits=N_BITS):
    for entry in log:
        if entry.get("num_valid_bits", 0) >= n_bits:
            return entry["step"]
    return None

with open(OUT_DIR / "summary.csv", "w") as f:
    f.write("run,seed,final_step,final_loss,valid_bits,conv_step\n")
    for d, log in zip(runs, logs):
        cfg  = json.load(open(d / "config.json"))
        seed = cfg["run"]["seed"]
        last = log[-1]
        conv = steps_to_full_bits(log)
        f.write(f"{d.name},{seed},{last['step']},{last['loss']:.6f},"
                f"{last['num_valid_bits']:.3f},"
                f"{conv if conv is not None else 'none'}\n")

# ── Align series to common step axis ──────────────────────────────────────────

max_steps = max(log[-1]["step"] + 1 for log in logs)

def align_series(log, key, n_steps):
    common = np.arange(n_steps)
    steps  = np.array([e["step"] for e in log])
    vals   = np.array([e[key]   for e in log])
    return np.interp(common, steps, vals)

# ── Loss curves ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
for i, (d, log) in enumerate(zip(runs, logs)):
    series = align_series(log, "loss", max_steps)
    smooth = np.convolve(series, np.ones(200)/200, mode="valid")
    ax.plot(smooth, color=PALETTE[i % len(PALETTE)], linewidth=1.5,
            label=d.name.split("_", 3)[-1])

ax.set_yscale("log")
ax.set_xlim(0, 30000)
ax.set_xlabel("Training step")
ax.set_ylabel("Loss (masked MSE)")
ax.set_title("E3 — 8-bit adder: training loss")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT_DIR / "loss_curves.svg")
plt.close(fig)
print("\nSaved: loss_curves.svg")

# ── Valid bits curves ──────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
for i, (d, log) in enumerate(zip(runs, logs)):
    series = align_series(log, "num_valid_bits", max_steps)
    smooth = np.convolve(series, np.ones(200)/200, mode="valid")
    ax.plot(smooth, color=PALETTE[i % len(PALETTE)], linewidth=1.5,
            label=d.name.split("_", 3)[-1])

ax.axhline(N_BITS, color="#aaaaaa", linestyle="--", linewidth=1,
           label=f"{N_BITS} bits (full convergence)")
ax.set_xlim(0, 30000)
ax.set_xlabel("Training step")
ax.set_ylabel("Valid bits")
ax.set_title("E3 — 8-bit adder: valid bits over time")
ax.set_ylim(0, N_BITS + 0.3)
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(OUT_DIR / "bits_curves.svg")
plt.close(fig)
print("Saved: bits_curves.svg")

# ── Summary table ──────────────────────────────────────────────────────────────

print(f"\n{'run':<40} {'steps':>7} {'loss':>8} {'bits':>6} {'conv':>8}")
for d, log in zip(runs, logs):
    last = log[-1]
    conv = steps_to_full_bits(log)
    conv_str = str(conv) if conv is not None else "none"
    print(f"{d.name:<40} {last['step']:>7} {last['loss']:>8.4f}"
          f" {last['num_valid_bits']:>6.1f} {conv_str:>8}")

# ── Snapshot grid ──────────────────────────────────────────────────────────────

snap_dirs = [(d, d / "snapshot_latest.png") for d in runs
             if (d / "snapshot_latest.png").exists()]

if snap_dirs:
    from PIL import Image
    n = len(snap_dirs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (d, snap) in zip(axes, snap_dirs):
        ax.imshow(Image.open(snap), rasterized=True)
        ax.set_title(d.name.split("_", 3)[-1], fontsize=7)
        ax.axis("off")
    fig.suptitle("E3 — Final snapshots", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "snapshots_grid.svg", dpi=150)
    plt.close(fig)
    print("Saved: snapshots_grid.svg")

# ── Layout diagram ─────────────────────────────────────────────────────────────

H_D, W_D = 112, 80
r_d      = 4
among_sp = 2
side_sp  = 21
step_px  = 2 * r_d + among_sp

COL_A   = "#4E79A7"
COL_B   = "#F28E2B"
COL_OUT = "#59A14F"
COL_BG  = "#f7f7f7"

def _col_centers(n, cx, H, r, sp):
    v  = n * 2*r + sp * (n-1)
    tm = (H - v) // 2
    return [(cx, tm + r + i*(2*r+sp)) for i in range(n)]

def _draw_circle(ax, cx, cy, r, color, label, fontsize=4.5):
    ax.add_patch(mpatches.Circle((cx, cy), r, color=color, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold", zorder=4)

scale = 0.028
p_w   = W_D * scale
p_h   = H_D * scale
lm, rm = 0.10, 0.10
tm_m, bm_m = 0.55, 0.40
fig_w = p_w + lm + rm
fig_h = p_h + tm_m + bm_m

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.subplots_adjust(
    left   = lm / fig_w,
    right  = 1 - rm / fig_w,
    bottom = bm_m / fig_h,
    top    = 1 - tm_m / fig_h,
)

ax.set_xlim(0, W_D); ax.set_ylim(H_D, 0); ax.set_aspect("equal")
ax.set_facecolor(COL_BG)
ax.add_patch(mpatches.FancyBboxPatch((0,0), W_D, H_D,
    boxstyle="square,pad=0", linewidth=1, edgecolor="#aaaaaa", facecolor=COL_BG))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("cols2 layout\n(A col + B col, 80×112)", fontsize=8, pad=4)

for (cx, cy), lbl in zip(_col_centers(8, side_sp, H_D, r_d, among_sp),
                          [f"A{7-i}" for i in range(8)]):
    _draw_circle(ax, cx, cy, r_d, COL_A, lbl)
for (cx, cy), lbl in zip(_col_centers(8, side_sp + step_px, H_D, r_d, among_sp),
                          [f"B{7-i}" for i in range(8)]):
    _draw_circle(ax, cx, cy, r_d, COL_B, lbl)
for (cx, cy), lbl in zip(_col_centers(9, W_D - side_sp, H_D, r_d, among_sp),
                          ["C"] + [f"S{7-i}" for i in range(8)]):
    _draw_circle(ax, cx, cy, r_d, COL_OUT, lbl)

handles = [
    mpatches.Patch(color=COL_A,   label="A[7:0]"),
    mpatches.Patch(color=COL_B,   label="B[7:0]"),
    mpatches.Patch(color=COL_OUT, label="Carry + Sum[7:0]"),
]
fig.legend(handles=handles, loc="lower center", fontsize=7, ncol=3,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.02))
fig.suptitle("E3 — 8-bit adder layout", fontsize=9, y=0.99)
fig.savefig(OUT_DIR / "layout_diagram.svg", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("\nSaved: layout_diagram.svg")

# ── Rollout helpers ────────────────────────────────────────────────────────────

TSTEPS = [0, 8, 16, 32, 64]

def best_run_dir():
    return min(runs,
               key=lambda d: json.loads(open(d/"log.jsonl").readlines()[-1])["loss"])

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

def make_state(ds_cfg, a_int, b_int):
    inp_bits = int_to_bits(a_int, 8) + int_to_bits(b_int, 8)
    screen = make_io_screen(
        H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
        spacing=ds_cfg["spacing"],
        left_input=inp_bits,
        right_input=[],
    )
    img = torch.from_numpy(screen).float() / 128.0 - 1.0
    state = torch.zeros(1, 16, ds_cfg["H"], ds_cfg["W"])
    state[0, 0] = img
    state[0, 1] = img
    return state

def make_target(ds_cfg, a_int, b_int):
    s_int    = a_int + b_int
    out_bits = int_to_bits(s_int, 9)
    screen = make_io_screen(
        H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
        spacing=ds_cfg["spacing"],
        left_input=[], right_input=out_bits,
    )
    return torch.from_numpy(screen).float() / 128.0 - 1.0

def get_rollout(nca, state, max_t=96):
    with torch.no_grad():
        return nca.forward(state, steps=max_t)[0]   # (T+1, C, H, W)

# Showcase: zero, small, mid, full carry, max
COMBOS = [
    (15,  17,  "15+17=32"),
    (100, 56,  "100+56=156"),
    (255, 1,   "255+1=256 (full carry)"),
    (255, 255, "255+255=510"),
]
HIGHLIGHT = 3   # full carry propagation

print("\n── Rollout figures ──")

run_dir = best_run_dir()
ds_cfg  = json.load(open(run_dir / "config.json"))["ds"]
nca     = load_nca(run_dir)
print(f"  Using: {run_dir.name}")

n_rows = len(COMBOS)
n_cols = len(TSTEPS)

# ── ch0 rollout ────────────────────────────────────────────────────────────────
W_G, H_G = ds_cfg["W"], ds_cfg["H"]   # 80, 112
cell_w = 0.80
cell_h = cell_w * H_G / W_G           # 1.12"

# Output circle geometry (same formula as E1)
_asp    = int(ds_cfg["spacing"][0])
_ssp    = int(ds_cfg["spacing"][1])
_rr     = int(ds_cfg["r"])
_n_out  = 9   # carry + sum[7:0]
_v      = _n_out * _rr * 2 + _asp * (_n_out - 1)
_tm     = (ds_cfg["H"] - _v) // 2
_cx_out = ds_cfg["W"] - _ssp

fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(n_cols * cell_w, n_rows * cell_h))
fig.subplots_adjust(hspace=0.01, wspace=0.01)

for ri, (a, b, label) in enumerate(COMBOS):
    state   = make_state(ds_cfg, a, b)
    rollout = get_rollout(nca, state)

    for ci, t in enumerate(TSTEPS):
        t_idx = min(t, rollout.shape[0] - 1)
        ax = axes[ri][ci]
        ax.imshow(rollout[t_idx, 0].numpy(), cmap="viridis",
                  vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(f"t={t}", fontsize=9, pad=2)
        if ci == 0:
            ax.set_ylabel(f"{a}+{b}", fontsize=8)
        if ci == n_cols - 1:
            ax.text(1.03, 0.5, f"={a+b}", transform=ax.transAxes,
                    fontsize=8, va="center", ha="left")
        # gray outline circles on last timestep only
        if ci == n_cols - 1:
            for i in range(_n_out):
                cy = _tm + _rr + i * (2 * _rr + _asp)
                ax.add_patch(mpatches.Circle(
                    (_cx_out, cy), _rr, fill=False, edgecolor="#888888", linewidth=0.8,
                ))

fig.savefig(OUT_DIR / "rollout_ch0.svg", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("    Saved: rollout_ch0.svg")

# ── all-channels rollout for highlight combo ───────────────────────────────────

a, b, hi_label = COMBOS[HIGHLIGHT]
state   = make_state(ds_cfg, a, b)
rollout = get_rollout(nca, state)
C       = rollout.shape[1]
n_ts    = len(TSTEPS)

fig, axes = plt.subplots(C, n_ts,
                         figsize=(n_ts * cell_w, C * cell_h),
                         gridspec_kw={"hspace": 0, "wspace": 0},
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

fig.suptitle(f"E3 — all channels — {hi_label}", fontsize=9)
fig.savefig(OUT_DIR / "rollout_allch.svg", dpi=300, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("    Saved: rollout_allch.svg")

print(f"\nAll outputs in {OUT_DIR}/")
