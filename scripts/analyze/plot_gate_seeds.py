#!/usr/bin/env python3
"""
Generate per-gate multi-seed overlay plots from external/runs/.

Usage:
    python3 scripts/analyze/plot_gate_seeds.py [--paper]

Without --paper, outputs under results/gate_seeds/:
    {gate}_loss_curves.png|svg      — all seeds, raw
    {gate}_loss_curves_ma.png|svg   — all seeds, 50-step MA
    {gate}_bits_curves.png|svg      — all seeds, raw
    {gate}_bits_curves_ma.png|svg   — all seeds, 50-step MA
    comparison_loss.png|svg         — all gates loss (normalized)
    comparison_bits.png|svg         — all gates bits

With --paper, also outputs to paper repo without titles:
    ncpu-paper/ssci-2027/images/gate_seeds_loss.png|svg
    ncpu-paper/ssci-2027/images/gate_seeds_bits.png|svg
"""

import argparse
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
PAPER_IMAGES = Path("/home/wallace/Projects/ncpu-paper/ssci-2027/images")

GATE_ORDER = ["AND", "OR", "XOR", "NAND", "NOR", "XNOR", "half_adder", "majority3"]
SMOOTH = 50

SEED_COLORS = [
    "#2171B5", "#4292C6", "#74C476", "#A1D99B", "#FDB185",
    "#FF7800", "#E31A1C", "#B15928", "#6A3D9A", "#FF6FA6",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--paper", action="store_true", help="also generate paper-ready versions")
    return p.parse_args()


def find_runs():
    runs = defaultdict(list)
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg = json.load(open(cfg_path))
        runs[cfg["run"]["gate"]].append((cfg["run"]["seed"], d))
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


def moving_average(data, window):
    if window <= 1:
        return data
    kernel = np.ones(window) / window
    padded = np.pad(data, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def align_to_common(log, key, max_step):
    steps = np.array([e["step"] for e in log])
    vals = np.array([e[key] for e in log])
    common = np.arange(max_step + 1)
    return np.interp(common, steps, vals), common


def make_figure(fig, ax, out_png, out_svg, paper=False):
    fig.tight_layout()
    fig.savefig(out_png, dpi=300 if paper else 150)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"  Saved {out_png}")


def main():
    args = parse_args()

    runs = find_runs()
    print(f"Found {sum(len(v) for v in runs.values())} runs across {len(runs)} gates")

    max_steps = 0
    for gate, dirs in runs.items():
        for seed, d in dirs:
            log = load_log(d)
            if log:
                max_steps = max(max_steps, log[-1]["step"])
    print(f"Max training steps: {max_steps}")

    # ── Per-gate individual plots ────────────────────────────

    for gate in GATE_ORDER:
        if gate not in runs:
            continue
        print(f"\n{gate}: {len(runs[gate])} seeds")

        # Loss — raw
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (seed, d) in enumerate(runs[gate]):
            log = load_log(d)
            series, common = align_to_common(log, "loss", max_steps)
            color = SEED_COLORS[i % len(SEED_COLORS)]
            ax.plot(common, series, color=color, linewidth=0.6, alpha=0.3,
                    label=f"seed={seed}")
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss (masked MSE)")
        ax.set_title(f"{gate} — Loss")
        ax.legend(loc="upper right", ncol=2)
        make_figure(fig, ax, OUT_DIR / f"{gate.lower()}_loss_curves.png", OUT_DIR / f"{gate.lower()}_loss_curves.svg")

        # Loss — smoothed
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (seed, d) in enumerate(runs[gate]):
            log = load_log(d)
            series, common = align_to_common(log, "loss", max_steps)
            series = moving_average(series, SMOOTH)
            color = SEED_COLORS[i % len(SEED_COLORS)]
            ax.plot(common, series, color=color, linewidth=1.5, alpha=0.85,
                    label=f"seed={seed}")
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss (masked MSE)")
        ax.set_title(f"{gate} — Loss ({SMOOTH}-step MA)")
        ax.legend(loc="upper right", ncol=2)
        make_figure(fig, ax, OUT_DIR / f"{gate.lower()}_loss_curves_ma.png", OUT_DIR / f"{gate.lower()}_loss_curves_ma.svg")

        # Bits — raw
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (seed, d) in enumerate(runs[gate]):
            log = load_log(d)
            series, common = align_to_common(log, "num_valid_bits", max_steps)
            color = SEED_COLORS[i % len(SEED_COLORS)]
            ax.plot(common, series, color=color, linewidth=0.6, alpha=0.3,
                    label=f"seed={seed}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Valid bits")
        ax.set_title(f"{gate} — Valid bits")
        ax.legend(loc="lower right", ncol=2)
        make_figure(fig, ax, OUT_DIR / f"{gate.lower()}_bits_curves.png", OUT_DIR / f"{gate.lower()}_bits_curves.svg")

        # Bits — smoothed
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (seed, d) in enumerate(runs[gate]):
            log = load_log(d)
            series, common = align_to_common(log, "num_valid_bits", max_steps)
            series = moving_average(series, SMOOTH)
            color = SEED_COLORS[i % len(SEED_COLORS)]
            ax.plot(common, series, color=color, linewidth=1.5, alpha=0.85,
                    label=f"seed={seed}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Valid bits")
        ax.set_title(f"{gate} — Valid bits ({SMOOTH}-step MA)")
        ax.legend(loc="lower right", ncol=2)
        make_figure(fig, ax, OUT_DIR / f"{gate.lower()}_bits_curves_ma.png", OUT_DIR / f"{gate.lower()}_bits_curves_ma.svg")

    # ── Combined comparison plots ──────────────────────────

    fig, ax = plt.subplots(figsize=(10, 5))
    for gate_idx, gate in enumerate(GATE_ORDER):
        if gate not in runs:
            continue
        seed_losses = []
        for seed, d in runs[gate]:
            log = load_log(d)
            series, _ = align_to_common(log, "loss", max_steps)
            series = moving_average(series, SMOOTH)
            seed_losses.append(series / series[0] if series[0] > 0 else series)
        seed_losses = np.array(seed_losses)
        mean_loss = seed_losses.mean(axis=0)
        std_loss = seed_losses.std(axis=0)
        color = SEED_COLORS[gate_idx % len(SEED_COLORS)]
        x = np.arange(len(mean_loss))
        ax.plot(x, mean_loss, color=color, linewidth=1.8, label=gate, alpha=0.9)
        ax.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.1)
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (normalized to t=0)")
    ax.legend(loc="upper right", ncol=4)
    make_figure(fig, ax, OUT_DIR / "comparison_loss.png", OUT_DIR / "comparison_loss.svg")

    fig, ax = plt.subplots(figsize=(10, 5))
    for gate_idx, gate in enumerate(GATE_ORDER):
        if gate not in runs:
            continue
        seed_bits = []
        for seed, d in runs[gate]:
            log = load_log(d)
            series, _ = align_to_common(log, "num_valid_bits", max_steps)
            series = moving_average(series, SMOOTH)
            seed_bits.append(series)
        seed_bits = np.array(seed_bits)
        mean_bits = seed_bits.mean(axis=0)
        std_bits = seed_bits.std(axis=0)
        color = SEED_COLORS[gate_idx % len(SEED_COLORS)]
        x = np.arange(len(mean_bits))
        ax.plot(x, mean_bits, color=color, linewidth=1.8, label=gate, alpha=0.9)
        ax.fill_between(x, mean_bits - std_bits, mean_bits + std_bits, color=color, alpha=0.1)
        n_bits = {"half_adder": 2}.get(gate, 1)
        ax.axhline(y=n_bits, color=color, linestyle="--", alpha=0.3, linewidth=0.8)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Valid bits")
    ax.legend(loc="lower right", ncol=4)
    make_figure(fig, ax, OUT_DIR / "comparison_bits.png", OUT_DIR / "comparison_bits.svg")

    # ── Paper versions (no titles) ─────────────────────────

    if args.paper:
        print("\n── Paper versions (no titles) ──")

        fig, ax = plt.subplots(figsize=(10, 5))
        for gate_idx, gate in enumerate(GATE_ORDER):
            if gate not in runs:
                continue
            seed_losses = []
            for seed, d in runs[gate]:
                log = load_log(d)
                series, _ = align_to_common(log, "loss", max_steps)
                series = moving_average(series, SMOOTH)
                seed_losses.append(series / series[0] if series[0] > 0 else series)
            seed_losses = np.array(seed_losses)
            mean_loss = seed_losses.mean(axis=0)
            std_loss = seed_losses.std(axis=0)
            color = SEED_COLORS[gate_idx % len(SEED_COLORS)]
            x = np.arange(len(mean_loss))
            ax.plot(x, mean_loss, color=color, linewidth=1.8, label=gate, alpha=0.9)
            ax.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.1)
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss (normalized to t=0)")
        ax.legend(loc="upper right", ncol=4)
        make_figure(fig, ax, PAPER_IMAGES / "gate_seeds_loss.png", PAPER_IMAGES / "gate_seeds_loss.svg", paper=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        for gate_idx, gate in enumerate(GATE_ORDER):
            if gate not in runs:
                continue
            seed_bits = []
            for seed, d in runs[gate]:
                log = load_log(d)
                series, _ = align_to_common(log, "num_valid_bits", max_steps)
                series = moving_average(series, SMOOTH)
                seed_bits.append(series)
            seed_bits = np.array(seed_bits)
            mean_bits = seed_bits.mean(axis=0)
            std_bits = seed_bits.std(axis=0)
            color = SEED_COLORS[gate_idx % len(SEED_COLORS)]
            x = np.arange(len(mean_bits))
            ax.plot(x, mean_bits, color=color, linewidth=1.8, label=gate, alpha=0.9)
            ax.fill_between(x, mean_bits - std_bits, mean_bits + std_bits, color=color, alpha=0.1)
            n_bits = {"half_adder": 2}.get(gate, 1)
            ax.axhline(y=n_bits, color=color, linestyle="--", alpha=0.3, linewidth=0.8)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Valid bits")
        ax.legend(loc="lower right", ncol=4)
        make_figure(fig, ax, PAPER_IMAGES / "gate_seeds_bits.png", PAPER_IMAGES / "gate_seeds_bits.svg", paper=True)

    print(f"\nAll outputs in {OUT_DIR}/")
    if args.paper:
        print(f"Paper versions in {PAPER_IMAGES}/")


if __name__ == "__main__":
    main()