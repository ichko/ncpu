#!/usr/bin/env python3
"""
E2 analysis — extracts plots and figures from 4-bit adder runs.

Usage:
    uv run python scripts/analyze_e2.py

Outputs under results/E2/:
    summary.csv              — per-run metrics
    loss_curves.pdf          — cols1 vs cols2, mean ± std
    bits_curves.pdf          — valid bits, mean ± std
    convergence_bar.pdf      — steps to full convergence (mean ± std)
    snapshots_grid.pdf       — best snapshot per layout
    rollout_<layout>_ch0.pdf     — channel 0 over time, representative inputs
    rollout_<layout>_allch.pdf   — all channels for highlight input
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
from ncpu.utils import make_io_screen, make_io_screen_cols1

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results/E2")
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

LAYOUT_ORDER  = ["cols1", "cols2"]
LAYOUT_COLORS = {"cols1": "#4E79A7", "cols2": "#F28E2B"}

SCREEN_FNS = {
    "make_io_screen":      make_io_screen,
    "make_io_screen_cols1": make_io_screen_cols1,
}

N_BITS   = 5   # 4-bit adder: 4 sum bits + 1 carry
max_steps = 50000

# ── Find runs ──────────────────────────────────────────────────────────────────

def find_runs():
    """Returns {layout: [run_dir, ...]} sorted by seed."""
    runs = defaultdict(list)
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get("run", {}).get("experiment") != "E2":
            continue
        layout = cfg["run"].get("layout", "cols1")
        seed   = cfg["run"]["seed"]
        runs[layout].append((seed, d))
    return {layout: [d for _, d in sorted(dirs)] for layout, dirs in runs.items()}

runs = find_runs()
print("Found runs:")
for layout in LAYOUT_ORDER:
    n = len(runs.get(layout, []))
    print(f"  {layout:<8} {n} seed(s)")

# ── Load logs ──────────────────────────────────────────────────────────────────

def load_log(run_dir):
    return [json.loads(l) for l in open(run_dir / "log.jsonl").readlines()]

logs = {layout: [load_log(d) for d in dirs] for layout, dirs in runs.items()}

# ── Summary CSV ────────────────────────────────────────────────────────────────

def steps_to_full_bits(log, n_bits):
    for entry in log:
        if entry.get("num_valid_bits", 0) >= n_bits:
            return entry["step"]
    return None

with open(OUT_DIR / "summary.csv", "w") as f:
    f.write("layout,seed,final_loss,valid_bits,conv_step\n")
    for layout in LAYOUT_ORDER:
        if layout not in logs:
            continue
        for seed_idx, log in enumerate(logs[layout]):
            last = log[-1]
            conv = steps_to_full_bits(log, N_BITS)
            f.write(f"{layout},{seed_idx},{last['loss']:.6f},"
                    f"{last['num_valid_bits']:.3f},"
                    f"{conv if conv is not None else 'none'}\n")

# ── Helper ─────────────────────────────────────────────────────────────────────

def align_series(layout_logs, key, n_steps):
    common = np.arange(n_steps)
    out = []
    for log in layout_logs:
        steps = np.array([e["step"] for e in log])
        vals  = np.array([e[key]  for e in log])
        out.append(np.interp(common, steps, vals))
    return np.array(out)

def plot_mean_std(ax, label, series, color, smooth=200):
    mean = series.mean(axis=0)
    std  = series.std(axis=0)
    x    = np.arange(series.shape[1])
    if smooth > 1:
        mean = np.convolve(mean, np.ones(smooth)/smooth, mode="valid")
        std  = np.convolve(std,  np.ones(smooth)/smooth, mode="valid")
        x    = x[smooth-1:]
    ax.plot(x, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

# ── Loss curves ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
for layout in LAYOUT_ORDER:
    if layout not in logs:
        continue
    series = align_series(logs[layout], "loss", max_steps)
    plot_mean_std(ax, layout, series, LAYOUT_COLORS[layout])

ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss (masked MSE)")
ax.set_title("E2 — 4-bit adder: training loss (mean ± std)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "loss_curves.pdf")
plt.close(fig)
print("\nSaved: loss_curves.pdf")

# ── Valid bits curves ──────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
for layout in LAYOUT_ORDER:
    if layout not in logs:
        continue
    series = align_series(logs[layout], "num_valid_bits", max_steps)
    plot_mean_std(ax, layout, series, LAYOUT_COLORS[layout])

ax.set_xlabel("Training step")
ax.set_ylabel("Valid bits (mean)")
ax.set_ylim(0, N_BITS + 0.5)
ax.set_title("E2 — 4-bit adder: valid bits (mean ± std)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "bits_curves.pdf")
plt.close(fig)
print("Saved: bits_curves.pdf")

# ── Convergence bar chart ──────────────────────────────────────────────────────

layouts_present = [l for l in LAYOUT_ORDER if l in logs]
means, stds = [], []
for layout in layouts_present:
    convs = []
    for log in logs[layout]:
        c = steps_to_full_bits(log, N_BITS)
        convs.append(max_steps if c is None else c)
    means.append(np.mean(convs))
    stds.append(np.std(convs))

fig, ax = plt.subplots(figsize=(6, 4))
colors = [LAYOUT_COLORS[l] for l in layouts_present]
ax.bar(layouts_present, means, yerr=stds, color=colors, capsize=4, width=0.5)
ax.set_ylabel("Steps to full convergence")
ax.set_title("E2 — Steps to reach 5/5 valid bits (mean ± std)")
fig.tight_layout()
fig.savefig(OUT_DIR / "convergence_bar.pdf")
plt.close(fig)
print("Saved: convergence_bar.pdf")

# ── Layout diagram ─────────────────────────────────────────────────────────────
# Schematic showing circle positions and labelling for cols1 vs cols2.

H_D, W_D   = 112, 80   # grid dims used in E2
r_d        = 4
among_sp   = 2
side_sp    = 21

COL_A   = "#4E79A7"   # blue   — A input bits
COL_B   = "#F28E2B"   # orange — B input bits
COL_OUT = "#59A14F"   # green  — output bits
COL_BG  = "#f7f7f7"   # light grey grid background

A_LABELS   = [f"a{i}" for i in range(4)]
B_LABELS   = [f"b{i}" for i in range(4)]
OUT_LABELS = ["s₀", "s₁", "s₂", "s₃", "c"]

def _draw_circle(ax, cx, cy, r, color, label, fontsize=6):
    ax.add_patch(mpatches.Circle((cx, cy), r, color=color, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold", zorder=4)

def _output_centers(H, n_out, r, among_sp, side_sp, W):
    v = n_out * 2*r + among_sp * (n_out - 1)
    tm = (H - v) // 2
    cx = W - side_sp
    return [(cx, tm + r + i*(2*r + among_sp)) for i in range(n_out)]

# Scale panels to true pixel dimensions so circles/spacing look faithful.
scale  = 0.030                  # inches per pixel (80px → 2.40", 112px → 3.36")
p_w    = W_D * scale            # 2.40"
p_h    = H_D * scale            # 3.36"
gap    = 0.10                   # gap between panels (inches)
lm, rm = 0.10, 0.10             # left / right margin
tm, bm = 0.38, 0.38             # top (title) / bottom (legend) margin
fig_w  = 2*p_w + gap + lm + rm
fig_h  = p_h + tm + bm

fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
fig.subplots_adjust(
    left   = lm / fig_w,
    right  = 1 - rm / fig_w,
    bottom = bm / fig_h,
    top    = 1 - tm / fig_h,
    wspace = gap / p_w,
)

for ax, (layout, title) in zip(axes, [
    ("cols1", "Single-column input (cols1)"),
    ("cols2", "Two-column input (cols2)"),
]):
    ax.set_xlim(0, W_D)
    ax.set_ylim(H_D, 0)   # image coords: y increases downward
    ax.set_aspect("equal")
    ax.set_facecolor(COL_BG)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), W_D, H_D, boxstyle="square,pad=0",
        linewidth=1, edgecolor="#aaaaaa", facecolor=COL_BG, zorder=0,
    ))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9, pad=4)

    # ── output circles (same for both layouts) ────────────────────────────────
    for (cx, cy), lbl in zip(_output_centers(H_D, 5, r_d, among_sp, side_sp, W_D), OUT_LABELS):
        _draw_circle(ax, cx, cy, r_d, COL_OUT, lbl)

    # ── input circles ─────────────────────────────────────────────────────────
    if layout == "cols1":
        # 8 bits stacked in a single column: a0..a3 then b0..b3
        n = 8
        v = n * 2*r_d + among_sp * (n - 1)
        tm_v = (H_D - v) // 2
        labels = A_LABELS + B_LABELS
        colors = [COL_A]*4 + [COL_B]*4
        for i, (lbl, col) in enumerate(zip(labels, colors)):
            cy = tm_v + r_d + i*(2*r_d + among_sp)
            _draw_circle(ax, side_sp, cy, r_d, col, lbl)
    else:
        # cols2: ceil(8/2)=4 rows, 2 sub-columns; col = i//4, row = i%4
        # first column → a0..a3, second column → b0..b3
        n_rows = 4
        v = n_rows * 2*r_d + among_sp * (n_rows - 1)
        tm_v = (H_D - v) // 2
        for i, (lbl, col) in enumerate(zip(A_LABELS + B_LABELS, [COL_A]*4 + [COL_B]*4)):
            c = i // n_rows
            row = i % n_rows
            cx = side_sp + c * (2*r_d + among_sp)
            cy = tm_v + r_d + row*(2*r_d + among_sp)
            _draw_circle(ax, cx, cy, r_d, col, lbl)

handles = [mpatches.Patch(color=c, label=l)
           for c, l in [(COL_A, "A bits"), (COL_B, "B bits"), (COL_OUT, "Output")]]
fig.legend(handles=handles, loc="lower center", fontsize=7, ncol=3,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.02))

fig.suptitle("4-bit adder — spatial encoding layouts", fontsize=10, y=0.99)
fig.savefig(OUT_DIR / "layout_diagram.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("Saved: layout_diagram.pdf")

# ── Snapshots grid ─────────────────────────────────────────────────────────────

layouts_with_snaps, snap_paths = [], []
for layout in LAYOUT_ORDER:
    if layout not in runs:
        continue
    best_dir = min(
        runs[layout],
        key=lambda d: json.loads(open(d / "log.jsonl").readlines()[-1])["loss"]
    )
    snap = best_dir / "snapshot_latest.png"
    if snap.exists():
        layouts_with_snaps.append(layout)
        snap_paths.append(snap)

n = len(layouts_with_snaps)
if n > 0:
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, layout, snap in zip(axes, layouts_with_snaps, snap_paths):
        ax.imshow(Image.open(snap), rasterized=True)
        ax.set_title(layout, fontsize=9)
        ax.axis("off")
    fig.suptitle("E2 — Final snapshots (best seed per layout)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "snapshots_grid.pdf", dpi=150)
    plt.close(fig)
    print("Saved: snapshots_grid.pdf")

# ── Print summary table ────────────────────────────────────────────────────────

print(f"\n{'layout':<10} {'seeds':>6} {'loss mean':>10} {'loss std':>9} {'bits mean':>10}")
for layout in LAYOUT_ORDER:
    if layout not in logs:
        continue
    losses = [json.loads(open(d / "log.jsonl").readlines()[-1])["loss"] for d in runs[layout]]
    bits   = [json.loads(open(d / "log.jsonl").readlines()[-1])["num_valid_bits"] for d in runs[layout]]
    print(f"{layout:<10} {len(losses):>6} {np.mean(losses):>10.4f} {np.std(losses):>9.4f} {np.mean(bits):>10.3f}")

print(f"\nAll outputs in {OUT_DIR}/")

# ── Rollout visualisations ─────────────────────────────────────────────────────
# Rows = representative input combos, cols = timesteps (channel 0) + target.
# All bits in MSB-first order matching sample_4bit_adder.

TSTEPS = [0, 16, 32, 64, 96, 128]

SHOWCASE = [
    # (a_bits, b_bits, out_bits, label)
    ([0,0,0,0], [0,0,0,0], [0,0,0,0,0], "0+0=0"),
    ([0,0,0,1], [0,0,0,1], [0,0,0,1,0], "1+1=2"),
    ([0,1,1,1], [0,0,0,1], [0,1,0,0,0], "7+1=8  (carry ×3)"),
    ([1,1,1,1], [0,0,0,1], [1,0,0,0,0], "15+1=16 (carry ×4)"),
]

def best_run_dir(layout):
    if layout not in runs:
        return None
    return min(runs[layout],
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

def make_state(ds_cfg, inp_bits, screen_fn):
    screen = screen_fn(
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

def get_rollout(nca, state, max_t=128):
    with torch.no_grad():
        return nca.forward(state, steps=max_t)[0]   # (T+1, C, H, W)

# cell sizes scaled to 80×112 image aspect ratio
cell_w = 0.62
cell_h = 0.88

print("\n── Rollout figures ──")

for layout in LAYOUT_ORDER:
    run_dir = best_run_dir(layout)
    if run_dir is None:
        print(f"  {layout}: no run found, skipping")
        continue

    cfg        = json.load(open(run_dir / "config.json"))
    ds_cfg     = cfg["ds"]
    screen_fn  = SCREEN_FNS[ds_cfg["screen_fn"]]
    nca        = load_nca(run_dir)
    print(f"  {layout}: loaded from {run_dir.name}")

    n_rows = len(SHOWCASE)
    n_cols = len(TSTEPS) + 1  # +1 for target

    # ── Figure 1: channel 0, all input combos × timesteps ────────────────────
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell_w, n_rows * cell_h),
                             gridspec_kw={"hspace": 0.02, "wspace": 0.02},
                             constrained_layout=True)

    for r, (a_bits, b_bits, tgt_bits, label) in enumerate(SHOWCASE):
        inp_bits = a_bits + b_bits
        rollout  = get_rollout(nca, make_state(ds_cfg, inp_bits, screen_fn))

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
        among_sp   = int(ds_cfg["spacing"][0])
        side_sp    = int(ds_cfg["spacing"][1])
        rr         = int(ds_cfg["r"])
        n_out      = len(tgt_bits)
        v_size     = n_out * rr * 2 + among_sp * (n_out - 1)
        top_margin = (ds_cfg["H"] - v_size) // 2
        cx         = ds_cfg["W"] - side_sp
        ax_last    = axes[r][len(TSTEPS) - 1]
        for i in range(n_out):
            cy = top_margin + rr + i * (2 * rr + among_sp)
            ax_last.add_patch(mpatches.Circle(
                (cx, cy), rr, fill=False, edgecolor="red", linewidth=1,
            ))

        # target frame
        target_screen = make_io_screen_cols1(
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

    fig.suptitle(f"E2 ({layout})  —  channel 0 over time", fontsize=10)
    out = OUT_DIR / f"rollout_{layout}_ch0.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"    Saved: rollout_{layout}_ch0.pdf")

    # ── Figure 2: all channels for the highlight combo (15+1=16) ─────────────
    a_bits, b_bits, tgt_bits, hi_label = SHOWCASE[-1]
    inp_bits = a_bits + b_bits
    rollout  = get_rollout(nca, make_state(ds_cfg, inp_bits, screen_fn))
    C        = rollout.shape[1]
    n_ts     = len(TSTEPS)

    fig, axes = plt.subplots(C, n_ts,
                             figsize=(n_ts * cell_w, C * cell_h),
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

    fig.suptitle(f"E2 ({layout})  —  all channels  —  {hi_label}", fontsize=9)
    out = OUT_DIR / f"rollout_{layout}_allch.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"    Saved: rollout_{layout}_allch.pdf")

print(f"\nAll outputs in {OUT_DIR}/")
