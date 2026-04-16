#!/usr/bin/env python3
"""
E1 analysis — extracts plots and figures from gate benchmark runs.

Usage:
    uv run python scripts/analyze_e1.py

Outputs under results/E1/:
    summary.csv                   — per-run metrics
    loss_curves.svg               — all gates, mean ± std shading
    bits_curves.svg               — valid bits, mean ± std shading
    convergence_bar.svg           — steps to full convergence (mean ± std)
    snapshots_grid.svg            — best snapshot per gate
    layout_diagram.svg            — schematic of grid layout
    rollout_{gate}_ch0.svg        — channel 0 over time, all input combos
    rollout_{gate}_allch.svg      — all channels for highlight combo
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
from ncpu.utils import make_io_screen_cols1

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results/E1")
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

# Tableau 10 — best-in-class categorical palette
GATE_ORDER = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "half_adder", "majority3"]
COLORS = [
    "#4E79A7",  # steel blue
    "#F28E2B",  # orange
    "#E15759",  # red
    "#76B7B2",  # teal
    "#59A14F",  # green
    "#EDC948",  # golden yellow
    "#B07AA1",  # lavender
    "#FF9DA7",  # salmon
]

# ── Find runs ──────────────────────────────────────────────────────────────────

def find_runs():
    """Returns {gate: [run_dir, ...]} sorted by seed."""
    runs = defaultdict(list)
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get("run", {}).get("experiment") != "E1":
            continue
        gate = cfg["run"]["gate"]
        seed = cfg["run"]["seed"]
        runs[gate].append((seed, d))
    # sort by seed, return just dirs
    return {gate: [d for _, d in sorted(dirs)] for gate, dirs in runs.items()}

runs = find_runs()
print(f"Found runs:")
for gate in GATE_ORDER:
    n = len(runs.get(gate, []))
    print(f"  {gate:<14} {n} seed(s)")

# ── Load logs ──────────────────────────────────────────────────────────────────

def load_log(run_dir):
    return [json.loads(l) for l in open(run_dir / "log.jsonl").readlines()]

logs = {gate: [load_log(d) for d in dirs] for gate, dirs in runs.items()}

# ── Summary CSV ───────────────────────────────────────────────────────────────

N_BITS = {"half_adder": 2}

def steps_to_full_bits(log, n_bits):
    for entry in log:
        if entry.get("num_valid_bits", 0) >= n_bits:
            return entry["step"]
    return None

with open(OUT_DIR / "summary.csv", "w") as f:
    f.write("gate,seed,final_loss,valid_bits,conv_step\n")
    for gate in GATE_ORDER:
        if gate not in logs:
            continue
        n_bits = N_BITS.get(gate, 1)
        for seed_idx, log in enumerate(logs[gate]):
            last = log[-1]
            conv = steps_to_full_bits(log, n_bits)
            f.write(f"{gate},{seed_idx},{last['loss']:.6f},"
                    f"{last['num_valid_bits']:.3f},"
                    f"{conv if conv is not None else 'none'}\n")

# ── Helper: interpolate all seeds to a common step axis ───────────────────────

def align_series(gate_logs, key, n_steps):
    """Returns (n_seeds, n_steps) array, interpolated to common axis."""
    common = np.arange(n_steps)
    out = []
    for log in gate_logs:
        steps = np.array([e["step"] for e in log])
        vals  = np.array([e[key] for e in log])
        out.append(np.interp(common, steps, vals))
    return np.array(out)

max_steps = 15000

def plot_mean_std(ax, gate, series, color, label=None, smooth=50):
    mean = series.mean(axis=0)
    std  = series.std(axis=0)
    x    = np.arange(series.shape[1])
    if smooth > 1:
        mean = np.convolve(mean, np.ones(smooth)/smooth, mode="valid")
        std  = np.convolve(std,  np.ones(smooth)/smooth, mode="valid")
        x    = x[smooth-1:]
    ax.plot(x, mean, color=color, linewidth=1.5, label=label or gate)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

# ── Loss curves ───────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
for i, gate in enumerate(GATE_ORDER):
    if gate not in logs:
        continue
    series = align_series(logs[gate], "loss", max_steps)
    plot_mean_std(ax, gate, series, COLORS[i])

ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss (masked MSE)")
ax.set_title("E1 — Gate benchmark: training loss (mean ± std)")
ax.legend(loc="upper right", ncol=2)
fig.tight_layout()
fig.savefig(OUT_DIR / "loss_curves.svg")
plt.close(fig)
print(f"\nSaved: loss_curves.svg")

# ── Valid bits curves ─────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
for i, gate in enumerate(GATE_ORDER):
    if gate not in logs:
        continue
    series = align_series(logs[gate], "num_valid_bits", max_steps)
    plot_mean_std(ax, gate, series, COLORS[i])

ax.set_xlabel("Training step")
ax.set_ylabel("Valid bits (mean)")
ax.set_title("E1 — Gate benchmark: valid bits (mean ± std)")
ax.legend(loc="lower right", ncol=2)
fig.tight_layout()
fig.savefig(OUT_DIR / "bits_curves.svg")
plt.close(fig)
print(f"Saved: bits_curves.svg")

# ── Convergence bar chart ─────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
gates_present = [g for g in GATE_ORDER if g in logs]
means, stds = [], []

for gate in gates_present:
    n_bits = N_BITS.get(gate, 1)
    convs  = []
    for log in logs[gate]:
        c = steps_to_full_bits(log, n_bits)
        convs.append(max_steps if c is None else c)
    means.append(np.mean(convs))
    stds.append(np.std(convs))

colors = [COLORS[GATE_ORDER.index(g)] for g in gates_present]
ax.bar(gates_present, means, yerr=stds, color=colors, capsize=4)
ax.set_ylabel("Steps to full convergence")
ax.set_title("E1 — Steps to reach max valid bits (mean ± std)")
ax.tick_params(axis="x", rotation=30)
fig.tight_layout()
fig.savefig(OUT_DIR / "convergence_bar.svg")
plt.close(fig)
print(f"Saved: convergence_bar.svg")

# ── Snapshot grid (best seed = lowest final loss) ─────────────────────────────

gates_with_snaps = []
snap_paths = []
for gate in GATE_ORDER:
    if gate not in runs:
        continue
    # pick seed with lowest final loss
    best_dir = min(
        runs[gate],
        key=lambda d: json.loads(open(d / "log.jsonl").readlines()[-1])["loss"]
    )
    snap = best_dir / "snapshot_latest.png"
    if snap.exists():
        gates_with_snaps.append(gate)
        snap_paths.append(snap)

n = len(gates_with_snaps)
if n > 0:
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3))
    if n == 1:
        axes = [axes]
    for ax, gate, snap in zip(axes, gates_with_snaps, snap_paths):
        ax.imshow(Image.open(snap), rasterized=True)
        ax.set_title(gate, fontsize=9)
        ax.axis("off")
    fig.suptitle("E1 — Final snapshots (best seed per gate)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "snapshots_grid.svg", dpi=150)
    plt.close(fig)
    print(f"Saved: snapshots_grid.svg")

# ── Print summary table ───────────────────────────────────────────────────────

print(f"\n{'gate':<14} {'seeds':>6} {'loss mean':>10} {'loss std':>9} {'bits mean':>10}")
for gate in GATE_ORDER:
    if gate not in logs:
        continue
    losses = [json.loads(open(d / "log.jsonl").readlines()[-1])["loss"] for d in runs[gate]]
    bits   = [json.loads(open(d / "log.jsonl").readlines()[-1])["num_valid_bits"] for d in runs[gate]]
    print(f"{gate:<14} {len(losses):>6} {np.mean(losses):>10.4f} {np.std(losses):>9.4f} {np.mean(bits):>10.3f}")

print(f"\nAll outputs in {OUT_DIR}/")

# ── Layout diagram ────────────────────────────────────────────────────────────
# Schematic showing circle positions for each gate class.

H_D, W_D = 48, 48
r_d      = 4
among_sp = 2
side_sp  = 10

COL_IN  = "#4E79A7"   # blue  — input bits
COL_OUT = "#59A14F"   # green — output bits
COL_BG  = "#f7f7f7"

def _draw_circle(ax, cx, cy, r, color, label, fontsize=6):
    ax.add_patch(mpatches.Circle((cx, cy), r, color=color, zorder=3))
    ax.text(cx, cy, label, ha="center", va="center",
            fontsize=fontsize, color="white", fontweight="bold", zorder=4)

def _input_centers(n_in, H, r, among_sp, side_sp):
    v = n_in * 2*r + among_sp * (n_in - 1)
    tm = (H - v) // 2
    cx = side_sp
    return [(cx, tm + r + i*(2*r + among_sp)) for i in range(n_in)]

def _output_centers_e1(n_out, H, r, among_sp, W, side_sp):
    v = n_out * 2*r + among_sp * (n_out - 1)
    tm = (H - v) // 2
    cx = W - side_sp
    return [(cx, tm + r + i*(2*r + among_sp)) for i in range(n_out)]

GATE_PANELS = [
    ("AND / XOR\n(2-in, 1-out)", 2, ["in₀", "in₁"], 1, ["out"]),
    ("half_adder\n(2-in, 2-out)", 2, ["in₀", "in₁"], 2, ["sum", "c"]),
    ("majority3\n(3-in, 1-out)", 3, ["in₀", "in₁", "in₂"], 1, ["out"]),
]

# Scale panels to true pixel dimensions so circles/spacing look faithful.
scale  = 0.042                  # inches per pixel (48px → 2.016")
p_w    = W_D * scale            # 2.016" (square panels)
p_h    = H_D * scale            # 2.016"
gap    = 0.10                   # gap between panels (inches)
lm, rm = 0.10, 0.10
tm, bm = 0.58, 0.35             # top (suptitle + 2-line panel titles) / bottom (legend) margin
fig_w  = 3*p_w + 2*gap + lm + rm
fig_h  = p_h + tm + bm

fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
fig.subplots_adjust(
    left   = lm / fig_w,
    right  = 1 - rm / fig_w,
    bottom = bm / fig_h,
    top    = 1 - tm / fig_h,
    wspace = gap / p_w,
)

for ax, (title, n_in, in_labels, n_out, out_labels) in zip(axes, GATE_PANELS):
    ax.set_xlim(0, W_D)
    ax.set_ylim(H_D, 0)   # image coords: y increases downward
    ax.set_aspect("equal")
    ax.set_facecolor(COL_BG)
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, 0), W_D, H_D, boxstyle="square,pad=0",
        linewidth=1, edgecolor="#aaaaaa", facecolor=COL_BG, zorder=0,
    ))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=8, pad=4)

    for (cx, cy), lbl in zip(_input_centers(n_in, H_D, r_d, among_sp, side_sp), in_labels):
        _draw_circle(ax, cx, cy, r_d, COL_IN, lbl, fontsize=5)
    for (cx, cy), lbl in zip(_output_centers_e1(n_out, H_D, r_d, among_sp, W_D, side_sp), out_labels):
        _draw_circle(ax, cx, cy, r_d, COL_OUT, lbl, fontsize=5)

handles = [
    mpatches.Patch(color=COL_IN,  label="Input bits"),
    mpatches.Patch(color=COL_OUT, label="Output bits"),
]
fig.legend(handles=handles, loc="lower center", fontsize=7, ncol=2,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.02))

fig.suptitle("E1 — Gate layout (cols1, 48×48 grid)", fontsize=9, y=0.99)
fig.savefig(OUT_DIR / "layout_diagram.svg", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"\nSaved: layout_diagram.svg")

# ── Rollout visualisations ────────────────────────────────────────────────────
# For each notable gate: rows = input combos, cols = timesteps (channel 0)
# Plus an all-channels view for the most interesting combo.

TSTEPS = [0, 8, 16, 32, 64]

SHOWCASE = {
    "AND": {
        "combos": [[0,0],[0,1],[1,0],[1,1]],
        "out":    [[0],  [0],  [0],  [1]],
        "labels": ["0,0 → 0","0,1 → 0","1,0 → 0","1,1 → 1"],
        "highlight": 3,   # index of most interesting combo for all-channels view
    },
    "XOR": {
        "combos": [[0,0],[0,1],[1,0],[1,1]],
        "out":    [[0],  [1],  [1],  [0]],
        "labels": ["0,0 → 0","0,1 → 1","1,0 → 1","1,1 → 0"],
        "highlight": 3,
    },
    "half_adder": {
        "combos": [[0,0],[0,1],[1,0],[1,1]],
        "out":    [[0,0],[1,0],[1,0],[0,1]],
        "labels": ["0,0 → s=0 c=0","0,1 → s=1 c=0","1,0 → s=1 c=0","1,1 → s=0 c=1"],
        "highlight": 3,
    },
    "majority3": {
        "combos": [[0,0,0],[1,0,0],[0,1,1],[1,1,1]],
        "out":    [[0],    [0],    [1],    [1]],
        "labels": ["0,0,0 → 0","1,0,0 → 0","0,1,1 → 1","1,1,1 → 1"],
        "highlight": 2,
    },
}

def best_run_dir(gate):
    if gate not in runs:
        return None
    return min(runs[gate],
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
    nca.load_state_dict(torch.load(last_ckpt, map_location="cpu", weights_only=True), strict=False)
    nca.eval()
    return nca

def make_state(ds_cfg, inp_bits):
    screen = make_io_screen_cols1(
        H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
        spacing=ds_cfg["spacing"],
        left_input=inp_bits,
        right_input=[],   # no output circles in initial state
    )
    img = torch.from_numpy(screen).float() / 128.0 - 1.0
    state = torch.zeros(1, 16, ds_cfg["H"], ds_cfg["W"])
    state[0, 0] = img
    state[0, 1] = img
    return state

def get_rollout(nca, state, max_t=64):
    with torch.no_grad():
        return nca.forward(state, steps=max_t)[0]   # (T+1, C, H, W)

print("\n── Rollout figures ──")

for gate, sc in SHOWCASE.items():
    run_dir = best_run_dir(gate)
    if run_dir is None:
        print(f"  {gate}: no run found, skipping")
        continue

    ds_cfg = json.load(open(run_dir / "config.json"))["ds"]
    nca    = load_nca(run_dir)
    print(f"  {gate}: loaded from {run_dir.name}")

    combos   = sc["combos"]
    out_bits = sc["out"]
    labels   = sc["labels"]
    n_rows   = len(combos)
    n_cols   = len(TSTEPS)

    # ── Figure 1: channel 0, all input combos × timesteps ────────────────────
    cell = 0.84
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell, n_rows * cell))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    if n_rows == 1:
        axes = [axes]

    for r, (inp_bits, tgt_bits, label) in enumerate(zip(combos, out_bits, labels)):
        parts = label.split(" → ", 1)
        in_label  = parts[0]
        out_label = ("→ " + parts[1]) if len(parts) > 1 else ""
        rollout = get_rollout(nca, make_state(ds_cfg, inp_bits))
        for c, t in enumerate(TSTEPS):
            t_idx = min(t, rollout.shape[0] - 1)
            ax = axes[r][c]
            ax.imshow(rollout[t_idx, 0].numpy(), cmap="viridis",
                      vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(f"t={TSTEPS[c]}", fontsize=8, pad=2)
            if c == 0:
                ax.set_ylabel(in_label, fontsize=9, labelpad=6)
            if c == n_cols - 1:
                ax.text(1.03, 0.5, out_label, transform=ax.transAxes,
                        fontsize=9, va="center", ha="left")

        # red outline circles on last timestep frame showing output locations
        among_sp   = int(ds_cfg["spacing"][0])
        side_sp    = int(ds_cfg["spacing"][1])
        rr         = int(ds_cfg["r"])
        n_out      = len(tgt_bits)
        v_size     = n_out * rr * 2 + among_sp * (n_out - 1)
        top_margin = (ds_cfg["H"] - v_size) // 2
        cx         = ds_cfg["W"] - side_sp
        ax_last    = axes[r][n_cols - 1]
        for i in range(n_out):
            cy = top_margin + rr + i * (2 * rr + among_sp)
            ax_last.add_patch(mpatches.Circle(
                (cx, cy), rr, fill=False, edgecolor="#888888", linewidth=1.8,
            ))

    out = OUT_DIR / f"rollout_{gate}_ch0.svg"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"    Saved: rollout_{gate}_ch0.svg")

    # ── Figure 2: all channels for the highlight combo ────────────────────────
    hi_bits   = combos[sc["highlight"]]
    hi_label  = labels[sc["highlight"]]
    rollout   = get_rollout(nca, make_state(ds_cfg, hi_bits))
    C         = rollout.shape[1]

    n_ts = len(TSTEPS)
    cell = 0.72  # inches per cell — keeps each subplot square
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
                ax.set_title(f"t={TSTEPS[ci]}", fontsize=7, pad=2)
            if ci == 0:
                tag = "out" if ch == 0 else ("in (r/o)" if ch == 1 else f"h{ch-1}")
                ax.set_ylabel(tag, fontsize=6)

    fig.suptitle(f"{gate}  —  all channels  —  {hi_label}", fontsize=9)
    out = OUT_DIR / f"rollout_{gate}_allch.svg"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"    Saved: rollout_{gate}_allch.svg")

# ── Paper figure: unified spatial encoding overview ────────────────────────────
# Shows the encoding concept across three task complexities:
#   Panel A — 2-input gate (XOR): 2 in, 1 out
#   Panel B — 4-bit adder (cols2): A col + B col, 5 out
#   Panel C — 8-bit ALU: A col + B col + opcode col, 8 out

PAPER_OUT = Path("results")
PAPER_OUT.mkdir(exist_ok=True)

# ── shared drawing helpers ─────────────────────────────────────────────────────

def _circ(ax, cx, cy, r, bit, zorder=3):
    color = "#ffffff" if bit else "#111111"
    ec    = "#555555"
    ax.add_patch(mpatches.Circle((cx, cy), r,
                                  facecolor=color, edgecolor=ec,
                                  linewidth=0.6, zorder=zorder))

def _col(ax, bits, cx, H, r, sp):
    n  = len(bits)
    v  = n * 2*r + sp * (n-1)
    tm = (H - v) // 2
    for i, b in enumerate(bits):
        _circ(ax, cx, tm + r + i*(2*r+sp), r, b)

def _label(ax, cx, cy, txt, color, fs=6):
    ax.text(cx, cy, txt, ha="center", va="center",
            fontsize=fs, color=color, fontweight="bold", zorder=4)

def _bracket(ax, cx, top_y, bot_y, lbl, color, side="right"):
    off = 7
    sx  = cx + off if side == "right" else cx - off
    ax.plot([sx, sx], [top_y, bot_y], color=color, lw=1.2)
    ax.plot([sx, sx + (3 if side=="right" else -3)], [top_y, top_y], color=color, lw=1.2)
    ax.plot([sx, sx + (3 if side=="right" else -3)], [bot_y, bot_y], color=color, lw=1.2)
    ax.text(sx + (6 if side=="right" else -6), (top_y+bot_y)/2,
            lbl, ha="left" if side=="right" else "right",
            va="center", fontsize=5.5, color=color)

BG   = "#e8e8e8"
CINP = "#2c7bb6"
COUT = "#d7191c"
COP  = "#1a9641"

# ── Panel A: XOR gate (2-in, 1-out, 48×48) ────────────────────────────────────
PA_H, PA_W, PA_r, PA_sp, PA_side = 48, 48, 4, 2, 10
A_bits_xor  = [0, 1]
B_bit_xor   = [1]

# ── Panel B: 4-bit adder cols2 (80×112 scaled down) ──────────────────────────
PB_H, PB_W, PB_r, PB_sp, PB_side = 80, 60, 3, 2, 10
A_bits_add  = [1, 0, 1, 1]   # 11
B_bits_add  = [0, 1, 1, 0]   #  6  → sum = 17 = 0b10001
sum_bits    = [1, 0, 0, 0, 1]

# ── Panel C: 8-bit ALU (128×112 scaled down) ──────────────────────────────────
PC_H, PC_W, PC_r, PC_sp, PC_side = 80, 80, 3, 1, 8
A_bits_alu  = [1,0,1,1,0,1,0,1]   # 0xB5
B_bits_alu  = [0,1,1,0,1,1,1,0]   # 0x6E
op_bits     = [0,0,0]              # ADD
res_bits    = [0,0,1,0,0,0,1,1]   # (0xB5+0x6E)&0xFF = 0x23

scale = 0.055
lm, rm = 0.08, 0.08
tm_m, bm_m = 0.38, 0.28
gap = 0.18

pw_a = PA_W * scale
pw_b = PB_W * scale
pw_c = PC_W * scale
ph   = max(PA_H, PB_H, PC_H) * scale
fig_w = pw_a + pw_b + pw_c + 2*gap + lm + rm
fig_h = ph + tm_m + bm_m

fig, axes = plt.subplots(1, 3, figsize=(fig_w, fig_h))
fig.subplots_adjust(
    left   = lm / fig_w,
    right  = 1 - rm / fig_w,
    bottom = bm_m / fig_h,
    top    = 1 - tm_m / fig_h,
    wspace = gap / ((pw_a + pw_b + pw_c) / 3),
)

# ── Panel A ───────────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, PA_W); ax.set_ylim(PA_H, 0); ax.set_aspect("equal")
ax.set_facecolor(BG)
ax.add_patch(mpatches.FancyBboxPatch((0,0), PA_W, PA_H,
    boxstyle="square,pad=0", lw=1, edgecolor="#aaaaaa", facecolor=BG))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(a) Logic gate\n(XOR: 2 in, 1 out)", fontsize=7.5, pad=4)

# inputs
_col(ax, A_bits_xor, PA_side, PA_H, PA_r, PA_sp)
# labels on circles
n = 2; v = n*2*PA_r+PA_sp*(n-1); tm = (PA_H-v)//2
for i, (b, lbl) in enumerate(zip(A_bits_xor, ["a","b"])):
    cy = tm + PA_r + i*(2*PA_r+PA_sp)
    _label(ax, PA_side, cy, f"{'1' if b else '0'}", "#ffffff" if b else "#dddddd", fs=5)
    ax.text(PA_side - PA_r - 3, cy, lbl, ha="right", va="center", fontsize=6, color=CINP)

# output
_col(ax, B_bit_xor, PA_W-PA_side, PA_H, PA_r, PA_sp)
cy_out = PA_H//2
_label(ax, PA_W-PA_side, cy_out, "1", "#ffffff", fs=5)
ax.text(PA_W-PA_side + PA_r + 3, cy_out, "out", ha="left", va="center",
        fontsize=6, color=COUT)

# arrow
ax.annotate("", xy=(PA_W-PA_side-PA_r-1, cy_out),
            xytext=(PA_side+PA_r+1, cy_out),
            arrowprops=dict(arrowstyle="->", color="#777777", lw=0.8))

# legend patches inside panel
ax.add_patch(mpatches.Circle((5, PA_H-5), 2.5, facecolor="#ffffff",
             edgecolor="#555555", lw=0.6))
ax.text(9, PA_H-5, "= 1", va="center", fontsize=5, color="#333333")
ax.add_patch(mpatches.Circle((5, PA_H-11), 2.5, facecolor="#111111",
             edgecolor="#555555", lw=0.6))
ax.text(9, PA_H-11, "= 0", va="center", fontsize=5, color="#333333")

# ── Panel B ───────────────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, PB_W); ax.set_ylim(PB_H, 0); ax.set_aspect("equal")
ax.set_facecolor(BG)
ax.add_patch(mpatches.FancyBboxPatch((0,0), PB_W, PB_H,
    boxstyle="square,pad=0", lw=1, edgecolor="#aaaaaa", facecolor=BG))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(b) 4-bit adder\n(A col + B col → 5-bit sum)", fontsize=7.5, pad=4)

step_b = 2*PB_r+PB_sp
_col(ax, A_bits_add, PB_side,          PB_H, PB_r, PB_sp)
_col(ax, B_bits_add, PB_side+step_b,   PB_H, PB_r, PB_sp)
_col(ax, sum_bits,   PB_W-PB_side,     PB_H, PB_r, PB_sp)

n=4; v=n*2*PB_r+PB_sp*(n-1); tm=(PB_H-v)//2
ax.text(PB_side, tm-4, "A", ha="center", va="bottom", fontsize=6, color=CINP, fontweight="bold")
ax.text(PB_side+step_b, tm-4, "B", ha="center", va="bottom", fontsize=6, color=CINP, fontweight="bold")

n2=5; v2=n2*2*PB_r+PB_sp*(n2-1); tm2=(PB_H-v2)//2
ax.text(PB_W-PB_side, tm2-4, "S", ha="center", va="bottom", fontsize=6, color=COUT, fontweight="bold")

# arrow
mid_x = (PB_side + step_b + PB_r + PB_W - PB_side - PB_r) / 2
ax.annotate("", xy=(PB_W-PB_side-PB_r-1, PB_H//2),
            xytext=(PB_side+step_b+PB_r+1, PB_H//2),
            arrowprops=dict(arrowstyle="->", color="#777777", lw=0.8))

# ── Panel C ───────────────────────────────────────────────────────────────────
ax = axes[2]
ax.set_xlim(0, PC_W); ax.set_ylim(PC_H, 0); ax.set_aspect("equal")
ax.set_facecolor(BG)
ax.add_patch(mpatches.FancyBboxPatch((0,0), PC_W, PC_H,
    boxstyle="square,pad=0", lw=1, edgecolor="#aaaaaa", facecolor=BG))
ax.set_xticks([]); ax.set_yticks([])
ax.set_title("(c) 8-bit ALU\n(A + B + opcode → result)", fontsize=7.5, pad=4)

step_c = 2*PC_r+PC_sp
_col(ax, A_bits_alu, PC_side,          PC_H, PC_r, PC_sp)
_col(ax, B_bits_alu, PC_side+step_c,   PC_H, PC_r, PC_sp)

# opcode col (middle, green)
n_op=3; v_op=n_op*2*PC_r+PC_sp*(n_op-1); tm_op=(PC_H-v_op)//2
for i, b in enumerate(op_bits):
    cy = tm_op + PC_r + i*(2*PC_r+PC_sp)
    ax.add_patch(mpatches.Circle((PC_W//2, cy), PC_r,
                                  facecolor="#ffffff" if b else "#111111",
                                  edgecolor=COP, linewidth=1.0, zorder=3))

_col(ax, res_bits, PC_W-PC_side, PC_H, PC_r, PC_sp)

n=8; v=n*2*PC_r+PC_sp*(n-1); tm=(PC_H-v)//2
ax.text(PC_side,        tm-4, "A",  ha="center", va="bottom", fontsize=6, color=CINP, fontweight="bold")
ax.text(PC_side+step_c, tm-4, "B",  ha="center", va="bottom", fontsize=6, color=CINP, fontweight="bold")
ax.text(PC_W//2,    tm_op-4,  "op", ha="center", va="bottom", fontsize=6, color=COP,  fontweight="bold")
n2=8; v2=n2*2*PC_r+PC_sp*(n2-1); tm2=(PC_H-v2)//2
ax.text(PC_W-PC_side, tm2-4, "R", ha="center", va="bottom", fontsize=6, color=COUT, fontweight="bold")

ax.annotate("", xy=(PC_W-PC_side-PC_r-1, PC_H//2),
            xytext=(PC_W//2+PC_r+1, PC_H//2),
            arrowprops=dict(arrowstyle="->", color="#777777", lw=0.8))

# ── shared legend below all panels ────────────────────────────────────────────
handles = [
    mpatches.Patch(color=CINP, label="Input operand"),
    mpatches.Patch(color=COP,  label="Opcode"),
    mpatches.Patch(color=COUT, label="Output"),
    mpatches.Circle((0,0), 1, facecolor="#ffffff", edgecolor="#555555", lw=0.8, label="bit = 1"),
    mpatches.Circle((0,0), 1, facecolor="#111111", edgecolor="#555555", lw=0.8, label="bit = 0"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.01))

fig.suptitle("Spatial encoding: inputs and outputs as pixel circles on a 2-D grid",
             fontsize=9, y=0.98)
fig.savefig(PAPER_OUT / "fig_encoding_overview.svg", bbox_inches="tight", pad_inches=0.04)
plt.close(fig)
print(f"\nSaved: results/fig_encoding_overview.svg")

print(f"\nAll outputs in {OUT_DIR}/")
