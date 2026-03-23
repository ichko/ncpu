#!/usr/bin/env python3
"""
E3 analysis — extracts plots and figures from 8-bit adder runs (cols2 layout).

Usage:
    uv run python scripts/analyze_e3.py

Outputs under results/E3/:
    summary.csv              — per-run metrics
    loss_curves.pdf          — all seeds, mean ± std
    bits_curves.pdf          — valid bits, mean ± std
    convergence_bar.pdf      — steps to full convergence
    layout_diagram.pdf       — spatial encoding schematic
    snapshots_grid.pdf       — best snapshot per seed
    rollout_ch0.pdf          — channel 0 over time, representative inputs
    rollout_allch.pdf        — all channels for highlight input
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.utils import make_io_screen

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results/E3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Global style ───────────────────────────────────────────────────────────────

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

COLORS = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2"]

N_BITS    = 9    # 8 sum bits + 1 carry
max_steps = 50000

# ── Find runs ──────────────────────────────────────────────────────────────────

def find_runs():
    """Returns [(seed, run_dir), ...] sorted by seed, deduped (keep latest)."""
    seen = {}   # seed → (timestamp, run_dir)
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get("run", {}).get("experiment") != "E3":
            continue
        seed = cfg["run"]["seed"]
        ts   = cfg["run"]["name"].split("_")[-1]   # timestamp suffix
        if seed not in seen or ts > seen[seed][0]:
            seen[seed] = (ts, d)
    return sorted((seed, d) for seed, (_, d) in seen.items())

run_list = find_runs()
print(f"Found {len(run_list)} E3 run(s):")
for seed, d in run_list:
    print(f"  seed={seed}  {d.name}")

# ── Load logs ──────────────────────────────────────────────────────────────────

def load_log(run_dir):
    return [json.loads(l) for l in open(run_dir / "log.jsonl").readlines()]

logs = [load_log(d) for _, d in run_list]

# ── Summary CSV ────────────────────────────────────────────────────────────────

def steps_to_full_bits(log, n_bits):
    for entry in log:
        if entry.get("num_valid_bits", 0) >= n_bits:
            return entry["step"]
    return None

with open(OUT_DIR / "summary.csv", "w") as f:
    f.write("seed,final_loss,valid_bits,conv_step\n")
    for (seed, _), log in zip(run_list, logs):
        last = log[-1]
        conv = steps_to_full_bits(log, N_BITS)
        f.write(f"{seed},{last['loss']:.6f},{last['num_valid_bits']:.3f},"
                f"{conv if conv is not None else 'none'}\n")

# ── Helper ─────────────────────────────────────────────────────────────────────

def align_series(all_logs, key, n_steps):
    common = np.arange(n_steps)
    out = []
    for log in all_logs:
        steps = np.array([e["step"] for e in log])
        vals  = np.array([e[key]  for e in log])
        out.append(np.interp(common, steps, vals))
    return np.array(out)

def plot_mean_std(ax, label, series, color, smooth=200):
    mean = series.mean(axis=0)
    std  = series.std(axis=0)
    x    = np.arange(series.shape[1])
    if smooth > 1 and series.shape[1] > smooth:
        mean = np.convolve(mean, np.ones(smooth)/smooth, mode="valid")
        std  = np.convolve(std,  np.ones(smooth)/smooth, mode="valid")
        x    = x[smooth-1:]
    ax.plot(x, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

# ── Loss curves ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
for i, ((seed, _), log) in enumerate(zip(run_list, logs)):
    steps = [e["step"] for e in log]
    vals  = [e["loss"]  for e in log]
    ax.plot(steps, vals, color=COLORS[i % len(COLORS)], linewidth=1.2,
            alpha=0.8, label=f"seed {seed}")

ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss (masked MSE)")
ax.set_title("E3 — 8-bit adder (cols2): training loss")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "loss_curves.pdf")
plt.close(fig)
print("\nSaved: loss_curves.pdf")

# ── Valid bits curves ──────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
for i, ((seed, _), log) in enumerate(zip(run_list, logs)):
    steps = [e["step"] for e in log]
    vals  = [e["num_valid_bits"] for e in log]
    ax.plot(steps, vals, color=COLORS[i % len(COLORS)], linewidth=1.2,
            alpha=0.8, label=f"seed {seed}")

ax.set_xlabel("Training step")
ax.set_ylabel("Valid bits")
ax.set_ylim(0, N_BITS + 0.5)
ax.set_title("E3 — 8-bit adder (cols2): valid bits")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "bits_curves.pdf")
plt.close(fig)
print("Saved: bits_curves.pdf")

# ── Convergence bar chart ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(4, 4))
seed_labels = [f"seed {s}" for s, _ in run_list]
convs = []
for log in logs:
    c = steps_to_full_bits(log, N_BITS)
    convs.append(max_steps if c is None else c)

ax.bar(seed_labels, convs,
       color=[COLORS[i % len(COLORS)] for i in range(len(run_list))],
       width=0.5)
ax.set_ylabel("Steps to full convergence")
ax.set_title("E3 — Steps to reach 9/9 valid bits")
fig.tight_layout()
fig.savefig(OUT_DIR / "convergence_bar.pdf")
plt.close(fig)
print("Saved: convergence_bar.pdf")

# ── Layout diagram ─────────────────────────────────────────────────────────────
# Single panel: cols2 with 8-bit inputs (8 rows × 2 sub-cols) and 9 outputs.

H_D, W_D = 112, 80
r_d      = 4
among_sp = 2
side_sp  = 21

COL_A   = "#4E79A7"
COL_B   = "#F28E2B"
COL_OUT = "#59A14F"
COL_BG  = "#f7f7f7"

A_LABELS   = [f"a{i}" for i in range(8)]
B_LABELS   = [f"b{i}" for i in range(8)]
OUT_LABELS = [f"s{i}" for i in range(8)] + ["c"]

def _draw_circle(ax, cx, cy, r, color, label, fontsize=5):
    ax.add_patch(mpatches.Circle((cx, cy), r, color=color, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold", zorder=4)

scale  = 0.030
p_w    = W_D * scale
p_h    = H_D * scale
lm, rm = 0.10, 0.10
tm, bm = 0.38, 0.38
fig_w  = p_w + lm + rm
fig_h  = p_h + tm + bm

fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))
fig.subplots_adjust(
    left   = lm / fig_w,
    right  = 1 - rm / fig_w,
    bottom = bm / fig_h,
    top    = 1 - tm / fig_h,
)
ax.set_xlim(0, W_D)
ax.set_ylim(H_D, 0)
ax.set_aspect("equal")
ax.set_facecolor(COL_BG)
ax.add_patch(mpatches.FancyBboxPatch(
    (0, 0), W_D, H_D, boxstyle="square,pad=0",
    linewidth=1, edgecolor="#aaaaaa", facecolor=COL_BG, zorder=0,
))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("Two-column input (cols2)", fontsize=9, pad=4)

# input circles: 8 rows × 2 sub-columns (a0..a7 | b0..b7)
n_rows = 8
v = n_rows * 2*r_d + among_sp * (n_rows - 1)
tm_v = (H_D - v) // 2
for i, (lbl, col) in enumerate(zip(A_LABELS + B_LABELS, [COL_A]*8 + [COL_B]*8)):
    sub_col = i // n_rows
    row     = i % n_rows
    cx = side_sp + sub_col * (2*r_d + among_sp)
    cy = tm_v + r_d + row * (2*r_d + among_sp)
    _draw_circle(ax, cx, cy, r_d, col, lbl)

# output circles: 9 bits stacked on right
n_out = 9
v_out = n_out * 2*r_d + among_sp * (n_out - 1)
tm_out = (H_D - v_out) // 2
cx_out = W_D - side_sp
for i, lbl in enumerate(OUT_LABELS):
    cy = tm_out + r_d + i * (2*r_d + among_sp)
    _draw_circle(ax, cx_out, cy, r_d, COL_OUT, lbl)

handles = [mpatches.Patch(color=c, label=l)
           for c, l in [(COL_A, "A bits"), (COL_B, "B bits"), (COL_OUT, "Output")]]
fig.legend(handles=handles, loc="lower center", fontsize=7, ncol=3,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.02))

fig.suptitle("8-bit adder — spatial encoding (cols2, 80×112 grid)", fontsize=10, y=0.99)
fig.savefig(OUT_DIR / "layout_diagram.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("Saved: layout_diagram.pdf")

# ── Snapshots grid ─────────────────────────────────────────────────────────────

snap_items = []
for seed, d in run_list:
    snap = d / "snapshot_latest.png"
    if snap.exists():
        snap_items.append((f"seed {seed}", snap))

n = len(snap_items)
if n > 0:
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (label, snap) in zip(axes, snap_items):
        ax.imshow(Image.open(snap), rasterized=True)
        ax.set_title(label, fontsize=9)
        ax.axis("off")
    fig.suptitle("E3 — Final snapshots", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "snapshots_grid.pdf", dpi=150)
    plt.close(fig)
    print("Saved: snapshots_grid.pdf")

# ── Print summary table ────────────────────────────────────────────────────────

print(f"\n{'seed':>6} {'final_loss':>12} {'valid_bits':>12} {'conv_step':>12}")
for (seed, _), log in zip(run_list, logs):
    last = log[-1]
    conv = steps_to_full_bits(log, N_BITS)
    print(f"{seed:>6} {last['loss']:>12.4f} {last['num_valid_bits']:>12.3f} "
          f"{conv if conv is not None else 'none':>12}")

print(f"\nAll outputs in {OUT_DIR}/")

# ── Rollout visualisations ─────────────────────────────────────────────────────
# Rows = representative input combos, cols = timesteps (channel 0) + target.
# Bits in MSB-first order matching sample_8bit_adder.

TSTEPS = [0, 16, 32, 64, 96, 128]

SHOWCASE = [
    # (a_bits, b_bits, out_bits, label)
    ([0]*8, [0]*8, [0]*9, "0+0=0"),
    ([0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,1,0], "1+1=2"),
    ([0,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,1], [0,1,0,0,0,0,0,0,0], "127+1=128  (carry ×7)"),
    ([1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,1], [1,0,0,0,0,0,0,0,0], "255+1=256  (carry ×8)"),
]

def best_run_dir():
    if not run_list:
        return None
    return min((d for _, d in run_list),
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
    nca.load_state_dict(torch.load(last_ckpt, map_location="cpu", weights_only=True))
    nca.eval()
    return nca

def make_state(ds_cfg, inp_bits):
    screen = make_io_screen(
        H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
        spacing=ds_cfg["spacing"],
        left_input=inp_bits,
        right_input=[],
    )
    img   = torch.from_numpy(screen).float() / 128.0 - 1.0
    state = torch.zeros(1, 16, ds_cfg["H"], ds_cfg["W"])
    state[0, 0] = img
    state[0, 1] = img
    return state

def get_rollout(nca, state, max_t=128):
    with torch.no_grad():
        return nca.forward(state, steps=max_t)[0]   # (T+1, C, H, W)

# cell sizes scaled to 80×112 aspect ratio
cell_w = 0.62
cell_h = 0.88

print("\n── Rollout figures ──")

run_dir = best_run_dir()
if run_dir is None:
    print("  no runs found, skipping rollout figures")
else:
    ds_cfg = json.load(open(run_dir / "config.json"))["ds"]
    nca    = load_nca(run_dir)
    print(f"  loaded from {run_dir.name}")

    n_rows = len(SHOWCASE)
    n_cols = len(TSTEPS) + 1  # +1 for target

    # ── Figure 1: channel 0, all input combos × timesteps ────────────────────
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_w, n_rows * cell_h),
                             gridspec_kw={"hspace": 0.02, "wspace": 0.02},
                             constrained_layout=True)

    for r, (a_bits, b_bits, tgt_bits, label) in enumerate(SHOWCASE):
        inp_bits = a_bits + b_bits
        rollout  = get_rollout(nca, make_state(ds_cfg, inp_bits))

        for c, t in enumerate(TSTEPS):
            t_idx = min(t, rollout.shape[0] - 1)
            ax = axes[r][c]
            ax.imshow(rollout[t_idx, 0].numpy(), cmap="viridis",
                      vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={t}", fontsize=8, pad=2)
            if c == 0:
                ax.set_ylabel(label, fontsize=7)

        # red outline circles on last timestep showing output locations
        rr         = int(ds_cfg["r"])
        sp         = int(ds_cfg["spacing"][0])
        sp_side    = int(ds_cfg["spacing"][1])
        n_out      = len(tgt_bits)
        v_size     = n_out * rr * 2 + sp * (n_out - 1)
        top_margin = (ds_cfg["H"] - v_size) // 2
        cx_out     = ds_cfg["W"] - sp_side
        ax_last    = axes[r][len(TSTEPS) - 1]
        for i in range(n_out):
            cy = top_margin + rr + i * (2 * rr + sp)
            ax_last.add_patch(mpatches.Circle(
                (cx_out, cy), rr, fill=False, edgecolor="red", linewidth=1,
            ))

        # target frame
        target_screen = make_io_screen(
            H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
            spacing=ds_cfg["spacing"],
            left_input=[], right_input=tgt_bits,
        )
        target_img = torch.from_numpy(target_screen).float() / 128.0 - 1.0
        ax = axes[r][n_cols - 1]
        ax.imshow(target_img.numpy(), cmap="viridis", vmin=-1, vmax=1,
                  interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title("target", fontsize=8, pad=2)

    fig.suptitle("E3 — 8-bit adder  —  channel 0 over time", fontsize=10)
    fig.savefig(OUT_DIR / "rollout_ch0.pdf", dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("    Saved: rollout_ch0.pdf")

    # ── Figure 2: all channels for the highlight combo (255+1=256) ───────────
    a_bits, b_bits, tgt_bits, hi_label = SHOWCASE[-1]
    rollout = get_rollout(nca, make_state(ds_cfg, a_bits + b_bits))
    C_ch    = rollout.shape[1]
    n_ts    = len(TSTEPS)

    fig, axes = plt.subplots(C_ch, n_ts,
                             figsize=(n_ts * cell_w, C_ch * cell_h),
                             gridspec_kw={"hspace": 0.02, "wspace": 0.02},
                             constrained_layout=True)
    for ch in range(C_ch):
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

    fig.suptitle(f"E3 — 8-bit adder  —  all channels  —  {hi_label}", fontsize=9)
    fig.savefig(OUT_DIR / "rollout_allch.pdf", dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("    Saved: rollout_allch.pdf")

print(f"\nAll outputs in {OUT_DIR}/")
