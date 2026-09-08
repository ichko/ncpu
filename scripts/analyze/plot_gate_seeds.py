#!/usr/bin/env python3
"""
Generate per-gate multi-seed overlay plots from external/runs/.

One plot per gate showing all seed loss curves and valid bits curves overlaid.
Designed for paper figures.

Usage:
    uv run python scripts/analyze/plot_gate_seeds.py

Outputs under results/gate_seeds/:
    {gate}_loss_curves.png    — loss vs step, one curve per seed
    {gate}_bits_curves.png   — valid bits vs step, one curve per seed
    comparison_loss.png      — all gates loss on one axes (normalized)
    comparison_bits.png      — all gates bits on one axes (normalized)
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

RUNS_DIR = Path("external/runs")
OUT_DIR = Path("results/gate_seeds")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GATE_ORDER = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "half_adder", "majority3"]

# Distinct color palette for up to 10 seeds
SEED_COLORS = [
    "#2171B5",  # dark blue
    "#4292C6",  # blue
    "#74C476",  # green
    "#A1D99B",  # light green
    "#FDB185",  # orange
    "#FF7800",  # dark orange
    "#E31A1C",  # red
    "#B15928",  # brown
    "#6A3D9A",  # purple
    "#FF6FA6",  # pink
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#eeeeee",
    "grid.linewidth": 0.7,
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "text.color": "#222222",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.framealpha": 0.95,
    "legend.edgecolor": "#cccccc",
    "legend.fontsize": 8,
    "figure.dpi": 150,
})


def find_runs():
    runs = defaultdict(list)
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg = json.load(open(cfg_path))
        gate = cfg["run"]["gate"]
        seed = cfg["run"]["seed"]
        runs[gate].append((seed, d))
    return {gate: sorted(dirs, key=lambda x: x[0]) for gate, dirs in runs.items()}


def load_log(run_dir, key="loss"):
    entries = []
    with open(run_dir / "log.jsonl") as f:
        for line in f:
            try:
                obj = json.loads(line)
                if key in obj and "step" in obj:
                    entries.append(obj)
            except json.JSONDecodeError:
                continue
    return entries


def extract_series(log, key):
    steps = np.array([e["step"] for e in log])
    vals = np.array([e[key] for e in log])
    return steps, vals


def align_to_common(log, key, max_step):
    steps, vals = extract_series(log, key)
    common = np.arange(max_step + 1)
    # clip to avoid extrapolation beyond data
    clipped = np.minimum(common, steps[-1])
    return np.interp(common, steps, vals), common


runs = find_runs()
print(f"Found {sum(len(v) for v in runs.values())} runs across {len(runs)} gates")

# Determine max steps across all runs
max_steps = 0
for gate, dirs in runs.items():
    for seed, d in dirs:
        log = load_log(d)
        if log:
            max_steps = max(max_steps, log[-1]["step"])
print(f"Max training steps: {max_steps}")

# ── Per-gate individual plots ──────────────────────────────────────────────

for gate in GATE_ORDER:
    if gate not in runs:
        continue
    dirs = runs[gate]
    n_seeds = len(dirs)
    print(f"\n{gate}: {n_seeds} seeds")

    # ── Loss curves ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (seed, d) in enumerate(dirs):
        log = load_log(d)
        series, common = align_to_common(log, "loss", max_steps)
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax.plot(common, series, color=color, linewidth=1.2, alpha=0.85,
                label=f"seed={seed}")
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (masked MSE)")
    ax.set_title(f"{gate} — Loss curves (all seeds)")
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{gate.lower()}_loss_curves.png")
    fig.savefig(OUT_DIR / f"{gate.lower()}_loss_curves.svg")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / f'{gate.lower()}_loss_curves.png'}")

    # ── Valid bits curves ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (seed, d) in enumerate(dirs):
        log = load_log(d)
        series, common = align_to_common(log, "num_valid_bits", max_steps)
        color = SEED_COLORS[i % len(SEED_COLORS)]
        ax.plot(common, series, color=color, linewidth=1.2, alpha=0.85,
                label=f"seed={seed}")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Valid bits")
    ax.set_title(f"{gate} — Valid bits (all seeds)")
    ax.legend(loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"{gate.lower()}_bits_curves.png")
    fig.savefig(OUT_DIR / f"{gate.lower()}_bits_curves.svg")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / f'{gate.lower()}_bits_curves.png'}")

# ── Combined comparison plots (all gates on one figure) ─────────────────────

# Loss comparison (normalized to first step)
fig, ax = plt.subplots(figsize=(10, 5))
for gate_idx, gate in enumerate(GATE_ORDER):
    if gate not in runs:
        continue
    dirs = runs[gate]
    seed_losses = []
    for seed, d in dirs:
        log = load_log(d)
        series, _ = align_to_common(log, "loss", max_steps)
        seed_losses.append(series / series[0] if series[0] > 0 else series)
    seed_losses = np.array(seed_losses)
    mean_loss = seed_losses.mean(axis=0)
    std_loss = seed_losses.std(axis=0)
    color = SEED_COLORS[gate_idx % len(SEED_COLORS)]
    x = np.arange(len(mean_loss))
    ax.plot(x, mean_loss, color=color, linewidth=1.8, label=gate, alpha=0.9)
    ax.fill_between(x, mean_loss - std_loss, mean_loss + std_loss,
                    color=color, alpha=0.1)
ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss (normalized to t=0)")
ax.set_title("All gates — Loss convergence comparison")
ax.legend(loc="upper right", ncol=4)
fig.tight_layout()
fig.savefig(OUT_DIR / "comparison_loss.png")
fig.savefig(OUT_DIR / "comparison_loss.svg")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'comparison_loss.png'}")

# Bits comparison
fig, ax = plt.subplots(figsize=(10, 5))
for gate_idx, gate in enumerate(GATE_ORDER):
    if gate not in runs:
        continue
    dirs = runs[gate]
    seed_bits = []
    for seed, d in dirs:
        log = load_log(d)
        series, _ = align_to_common(log, "num_valid_bits", max_steps)
        seed_bits.append(series)
    seed_bits = np.array(seed_bits)
    mean_bits = seed_bits.mean(axis=0)
    std_bits = seed_bits.std(axis=0)
    color = SEED_COLORS[gate_idx % len(SEED_COLORS)]
    x = np.arange(len(mean_bits))
    ax.plot(x, mean_bits, color=color, linewidth=1.8, label=gate, alpha=0.9)
    ax.fill_between(x, mean_bits - std_bits, mean_bits + std_bits,
                    color=color, alpha=0.1)
    # For binary gates, show target
    n_bits = {"half_adder": 2}.get(gate, 1)
    ax.axhline(y=n_bits, color=color, linestyle="--", alpha=0.3, linewidth=0.8)
ax.set_xlabel("Training step")
ax.set_ylabel("Valid bits (mean ± std)")
ax.set_title("All gates — Valid bits convergence comparison")
ax.legend(loc="lower right", ncol=4)
fig.tight_layout()
fig.savefig(OUT_DIR / "comparison_bits.png")
fig.savefig(OUT_DIR / "comparison_bits.svg")
plt.close(fig)
print(f"  Saved: {OUT_DIR / 'comparison_bits.png'}")

print(f"\nAll outputs in {OUT_DIR}/")