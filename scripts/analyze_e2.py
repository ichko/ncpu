#!/usr/bin/env python3
"""
E2 analysis — 4-bit adder benchmark.

Two layout variants are compared:
  cols1  — all 8 input bits in a single left column, 5 output bits in a single right column
  cols2  — A operand in left column, B operand in a second column; outputs on right

Usage:
    uv run python scripts/analyze_e2.py

Outputs under results/E2/:
    summary.csv                  — per-run metrics
    loss_curves.svg              — loss over time, cols1 vs cols2
    bits_curves.svg              — valid bits over time, cols1 vs cols2
    convergence_bar.svg          — steps to full convergence (5/5 bits)
    snapshots_grid.svg           — best snapshot per layout variant
    layout_diagram.svg           — schematic of cols1 vs cols2 grid layout
    rollout_cols1_ch0.svg        — channel 0 rollout for cols1, example inputs
    rollout_cols2_ch0.svg        — channel 0 rollout for cols2, example inputs
    rollout_cols1_allch.svg      — all channels for cols1 highlight combo
    rollout_cols2_allch.svg      — all channels for cols2 highlight combo
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.utils import make_io_screen, make_io_screen_cols1

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results/E2")
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

COLORS = {"cols1": "#4E79A7", "cols2": "#F28E2B"}

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
for layout in ["cols1", "cols2"]:
    n = len(runs.get(layout, []))
    print(f"  {layout:<8} {n} seed(s)")

# ── Load logs ──────────────────────────────────────────────────────────────────

def load_log(run_dir):
    return [json.loads(l) for l in open(run_dir / "log.jsonl")]

logs = {layout: [load_log(d) for d in dirs] for layout, dirs in runs.items()}

# ── Summary CSV ────────────────────────────────────────────────────────────────

N_BITS = 5   # 4-bit adder: sum[3:0] + carry_out

def steps_to_full_bits(log, n_bits=N_BITS):
    for entry in log:
        if entry.get("num_valid_bits", 0) >= n_bits:
            return entry["step"]
    return None

with open(OUT_DIR / "summary.csv", "w") as f:
    f.write("layout,seed,final_loss,valid_bits,conv_step\n")
    for layout in ["cols1", "cols2"]:
        if layout not in logs:
            continue
        for seed_idx, log in enumerate(logs[layout]):
            last = log[-1]
            conv = steps_to_full_bits(log)
            f.write(f"{layout},{seed_idx},{last['loss']:.6f},"
                    f"{last['num_valid_bits']:.3f},"
                    f"{conv if conv is not None else 'none'}\n")

# ── Align series ───────────────────────────────────────────────────────────────

def align_series(layout_logs, key, n_steps):
    common = np.arange(n_steps)
    out = []
    for log in layout_logs:
        steps = np.array([e["step"] for e in log])
        vals  = np.array([e[key]  for e in log])
        out.append(np.interp(common, steps, vals))
    return np.array(out)

max_steps = 50000

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
for layout in ["cols1", "cols2"]:
    if layout not in logs:
        continue
    series = align_series(logs[layout], "loss", max_steps)
    plot_mean_std(ax, layout, series, COLORS[layout])

ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss (masked MSE)")
ax.set_title("E2 — 4-bit adder: training loss (mean ± std)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "loss_curves.svg")
plt.close(fig)
print("\nSaved: loss_curves.svg")

# ── Valid bits curves ──────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
for layout in ["cols1", "cols2"]:
    if layout not in logs:
        continue
    series = align_series(logs[layout], "num_valid_bits", max_steps)
    plot_mean_std(ax, layout, series, COLORS[layout])

ax.axhline(N_BITS, color="#aaaaaa", linestyle="--", linewidth=1, label=f"{N_BITS} bits (full convergence)")
ax.set_xlabel("Training step")
ax.set_ylabel("Valid bits (mean)")
ax.set_title("E2 — 4-bit adder: valid bits (mean ± std)")
ax.set_ylim(0, N_BITS + 0.3)
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "bits_curves.svg")
plt.close(fig)
print("Saved: bits_curves.svg")

# ── Convergence bar chart ──────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(5, 4))
layouts_present = [l for l in ["cols1", "cols2"] if l in logs]
means, stds = [], []

for layout in layouts_present:
    convs = []
    for log in logs[layout]:
        c = steps_to_full_bits(log)
        convs.append(max_steps if c is None else c)
    means.append(np.mean(convs))
    stds.append(np.std(convs))

colors = [COLORS[l] for l in layouts_present]
ax.bar(layouts_present, means, yerr=stds, color=colors, capsize=6, width=0.5)
ax.set_ylabel("Steps to 5/5 valid bits")
ax.set_title("E2 — Steps to full convergence (mean ± std)")
fig.tight_layout()
fig.savefig(OUT_DIR / "convergence_bar.svg")
plt.close(fig)
print("Saved: convergence_bar.svg")

# ── Summary table ──────────────────────────────────────────────────────────────

print(f"\n{'layout':<8} {'seeds':>6} {'loss mean':>10} {'loss std':>9} {'bits mean':>10} {'conv step':>11}")
for layout in ["cols1", "cols2"]:
    if layout not in logs:
        continue
    losses = [json.loads(open(d / "log.jsonl").readlines()[-1])["loss"] for d in runs[layout]]
    bits   = [json.loads(open(d / "log.jsonl").readlines()[-1])["num_valid_bits"] for d in runs[layout]]
    convs  = []
    for log in logs[layout]:
        c = steps_to_full_bits(log)
        convs.append(max_steps if c is None else c)
    print(f"{layout:<8} {len(losses):>6} {np.mean(losses):>10.4f} {np.std(losses):>9.4f}"
          f" {np.mean(bits):>10.3f} {np.mean(convs):>11.0f}")

# ── Snapshot grid ──────────────────────────────────────────────────────────────

snap_pairs = []
for layout in ["cols1", "cols2"]:
    if layout not in runs:
        continue
    best_dir = min(
        runs[layout],
        key=lambda d: json.loads(open(d / "log.jsonl").readlines()[-1])["loss"]
    )
    snap = best_dir / "snapshot_latest.png"
    if snap.exists():
        snap_pairs.append((layout, snap))

if snap_pairs:
    from PIL import Image
    n = len(snap_pairs)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, (layout, snap) in zip(axes, snap_pairs):
        ax.imshow(Image.open(snap), rasterized=True)
        ax.set_title(layout, fontsize=9)
        ax.axis("off")
    fig.suptitle("E2 — Final snapshots (best seed per layout)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "snapshots_grid.svg", dpi=150)
    plt.close(fig)
    print("Saved: snapshots_grid.svg")

# ── Layout diagram ─────────────────────────────────────────────────────────────

H_D, W_D = 112, 80
r_d      = 4
among_sp = 2
side_sp  = 21

COL_A   = "#4E79A7"
COL_B   = "#F28E2B"
COL_OUT = "#59A14F"
COL_BG  = "#f7f7f7"

def _col_centers(n, cx, H, r, sp):
    v  = n * 2*r + sp * (n-1)
    tm = (H - v) // 2
    return [(cx, tm + r + i*(2*r+sp)) for i in range(n)]

def _draw_circle(ax, cx, cy, r, color, label, fontsize=5):
    ax.add_patch(mpatches.Circle((cx, cy), r, color=color, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold", zorder=4)

scale = 0.028
p_w   = W_D * scale
p_h   = H_D * scale
gap   = 0.15
lm, rm = 0.10, 0.10
tm_m, bm_m = 0.55, 0.40
fig_w = 2*p_w + gap + lm + rm
fig_h = p_h + tm_m + bm_m

fig, axes = plt.subplots(1, 2, figsize=(fig_w, fig_h))
fig.subplots_adjust(
    left   = lm / fig_w,
    right  = 1 - rm / fig_w,
    bottom = bm_m / fig_h,
    top    = 1 - tm_m / fig_h,
    wspace = gap / p_w,
)

# cols1: 8 input bits in one column, 5 output bits on right
ax = axes[0]
ax.set_xlim(0, W_D); ax.set_ylim(H_D, 0); ax.set_aspect("equal")
ax.set_facecolor(COL_BG)
ax.add_patch(mpatches.FancyBboxPatch((0,0), W_D, H_D,
    boxstyle="square,pad=0", linewidth=1, edgecolor="#aaaaaa", facecolor=COL_BG))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("cols1\n(8 inputs, 1 col)", fontsize=8, pad=4)

for (cx, cy), lbl in zip(_col_centers(8, side_sp, H_D, r_d, among_sp),
                          ["A3","A2","A1","A0","B3","B2","B1","B0"]):
    _draw_circle(ax, cx, cy, r_d, COL_A, lbl)
for (cx, cy), lbl in zip(_col_centers(5, W_D-side_sp, H_D, r_d, among_sp),
                          ["S4","S3","S2","S1","S0"]):
    _draw_circle(ax, cx, cy, r_d, COL_OUT, lbl)

# cols2: A in col 0, B in col 1, outputs on right
step = 2 * r_d + among_sp
ax = axes[1]
ax.set_xlim(0, W_D); ax.set_ylim(H_D, 0); ax.set_aspect("equal")
ax.set_facecolor(COL_BG)
ax.add_patch(mpatches.FancyBboxPatch((0,0), W_D, H_D,
    boxstyle="square,pad=0", linewidth=1, edgecolor="#aaaaaa", facecolor=COL_BG))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("cols2\n(A col + B col)", fontsize=8, pad=4)

for (cx, cy), lbl in zip(_col_centers(4, side_sp, H_D, r_d, among_sp),
                          ["A3","A2","A1","A0"]):
    _draw_circle(ax, cx, cy, r_d, COL_A, lbl)
for (cx, cy), lbl in zip(_col_centers(4, side_sp + step, H_D, r_d, among_sp),
                          ["B3","B2","B1","B0"]):
    _draw_circle(ax, cx, cy, r_d, COL_B, lbl)
for (cx, cy), lbl in zip(_col_centers(5, W_D-side_sp, H_D, r_d, among_sp),
                          ["S4","S3","S2","S1","S0"]):
    _draw_circle(ax, cx, cy, r_d, COL_OUT, lbl)

handles = [
    mpatches.Patch(color=COL_A,   label="A operand"),
    mpatches.Patch(color=COL_B,   label="B operand"),
    mpatches.Patch(color=COL_OUT, label="Sum output"),
]
fig.legend(handles=handles, loc="lower center", fontsize=7, ncol=3,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.02))
fig.suptitle("E2 — 4-bit adder layouts (80×112 grid, r=4)", fontsize=9, y=0.99)
fig.savefig(OUT_DIR / "layout_diagram.svg", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("\nSaved: layout_diagram.svg")

# ── Rollout helpers ────────────────────────────────────────────────────────────

TSTEPS = [0, 8, 16, 32, 64, 96]

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
    nca.load_state_dict(
        torch.load(last_ckpt, map_location="cpu", weights_only=True), strict=False
    )
    nca.eval()
    return nca

def int_to_bits(n, width):
    return [int(b) for b in f"{n:0{width}b}"]

def make_state_cols1(ds_cfg, a_int, b_int):
    inp_bits = int_to_bits(a_int, 4) + int_to_bits(b_int, 4)
    screen = make_io_screen_cols1(
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

def make_state_cols2(ds_cfg, a_int, b_int):
    inp_bits = int_to_bits(a_int, 4) + int_to_bits(b_int, 4)
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

def make_target(ds_cfg, a_int, b_int, screen_fn):
    s_int    = a_int + b_int
    out_bits = int_to_bits(s_int, 5)
    if screen_fn == "make_io_screen_cols1":
        screen = make_io_screen_cols1(
            H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
            spacing=ds_cfg["spacing"],
            left_input=[], right_input=out_bits,
        )
    else:
        screen = make_io_screen(
            H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
            spacing=ds_cfg["spacing"],
            left_input=[], right_input=out_bits,
        )
    return torch.from_numpy(screen).float() / 128.0 - 1.0

def get_rollout(nca, state, max_t=96):
    with torch.no_grad():
        return nca.forward(state, steps=max_t)[0]   # (T+1, C, H, W)

# ── Showcase combos ────────────────────────────────────────────────────────────

COMBOS = [
    (0,  0,  "0+0=0"),
    (3,  5,  "3+5=8"),
    (7,  8,  "7+8=15"),
    (15, 1,  "15+1=16 (carry)"),
    (15, 15, "15+15=30"),
]
HIGHLIGHT = 3   # 15+1=16: carry propagation is the most interesting

print("\n── Rollout figures ──")

for layout in ["cols1", "cols2"]:
    run_dir = best_run_dir(layout)
    if run_dir is None:
        print(f"  {layout}: no run found, skipping")
        continue

    ds_cfg    = json.load(open(run_dir / "config.json"))["ds"]
    screen_fn = ds_cfg["screen_fn"]
    nca       = load_nca(run_dir)
    print(f"  {layout}: loaded from {run_dir.name}")

    make_state = make_state_cols1 if layout == "cols1" else make_state_cols2
    n_rows = len(COMBOS)
    n_cols = len(TSTEPS) + 1

    # ── ch0 rollout ───────────────────────────────────────────────────────────
    cell = 0.55
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell, n_rows * cell),
                             gridspec_kw={"hspace": 0.02, "wspace": 0.02},
                             constrained_layout=True)

    for ri, (a, b, label) in enumerate(COMBOS):
        state   = make_state(ds_cfg, a, b)
        rollout = get_rollout(nca, state)
        target  = make_target(ds_cfg, a, b, screen_fn)

        for ci, t in enumerate(TSTEPS):
            t_idx = min(t, rollout.shape[0] - 1)
            ax = axes[ri][ci]
            ax.imshow(rollout[t_idx, 0].numpy(), cmap="viridis",
                      vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"t={t}", fontsize=7, pad=2)
            if ci == 0:
                ax.set_ylabel(label, fontsize=6)

        ax = axes[ri][n_cols - 1]
        ax.imshow(target.numpy(), cmap="viridis", vmin=-1, vmax=1,
                  interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title("target", fontsize=7, pad=2)

    fig.suptitle(f"E2 {layout} — channel 0 over time", fontsize=10)
    out = OUT_DIR / f"rollout_{layout}_ch0.svg"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"    Saved: rollout_{layout}_ch0.svg")

    # ── all-channels rollout for highlight combo ──────────────────────────────
    a, b, hi_label = COMBOS[HIGHLIGHT]
    state   = make_state(ds_cfg, a, b)
    rollout = get_rollout(nca, state)
    C       = rollout.shape[1]
    n_ts    = len(TSTEPS)

    cell = 0.55
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

    fig.suptitle(f"E2 {layout} — all channels — {hi_label}", fontsize=9)
    out = OUT_DIR / f"rollout_{layout}_allch.svg"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"    Saved: rollout_{layout}_allch.svg")

print(f"\nAll outputs in {OUT_DIR}/")
