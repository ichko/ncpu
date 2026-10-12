#!/usr/bin/env python3
"""
Generate per-adder multi-seed overlay plots from external/runs/.

Handles E2_adder4 and E3_adder8_cols2 experiments.
Outputs under results/adder_seeds/.

Usage:
    python3 scripts/analyze/plot_adder_seeds.py
    python3 scripts/analyze/plot_adder_seeds.py --paper
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

DEFAULT_RUNS_DIRS = [Path("external/runs")]
OUT_DIR = Path("results/adder_seeds")
OUT_DIR.mkdir(parents=True, exist_ok=True)
PAPER_IMAGES = Path("/home/wallace/Projects/ncpu-paper/ssci-2027/images")

SMOOTH = 50

SEED_COLORS = [
    "#2171B5", "#4292C6", "#74C476", "#A1D99B", "#FDB185",
    "#FF7800", "#E31A1C", "#B15928", "#6A3D9A", "#FF6FA6",
]

# Human-readable labels for comparison plots
KEY_LABELS = {
    "E2_adder4_cols1": "4-bit adder",
    "E3_adder8_cols2": "8-bit adder",
}

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
    "legend.fontsize": 9,
})


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", nargs="+", type=Path, default=DEFAULT_RUNS_DIRS,
                   help="run directories containing */config.json + log.jsonl "
                        f"(default: {[str(d) for d in DEFAULT_RUNS_DIRS]})")
    p.add_argument("--only", nargs="*", default=None,
                   help="restrict per-group plots to these keys, e.g. E3_adder8_cols2 "
                        "(comparison plots still use every group found)")
    p.add_argument("--paper", action="store_true", help="also generate paper-ready versions")
    return p.parse_args()


def find_runs(runs_dirs):
    """Group adder runs by experiment+bits+layout."""
    runs = defaultdict(list)
    for runs_dir in runs_dirs:
        if not runs_dir.is_dir():
            print(f"WARNING: runs dir not found: {runs_dir}")
            continue
        for d in sorted(runs_dir.iterdir()):
            _collect_run(runs, d)
    return {k: sorted(v, key=lambda x: x[0]) for k, v in runs.items()}


def _collect_run(runs, d):
    if not d.is_dir():
        return
    cfg_path = d / "config.json"
    if not cfg_path.exists() or not (d / "log.jsonl").exists():
        return
    cfg = json.load(open(cfg_path))
    exp = cfg["run"].get("experiment")
    if exp not in ("E2", "E3"):
        return
    bits = cfg["run"].get("bits")
    layout = cfg["run"].get("layout", "cols1")
    key = f"{exp}_adder{bits}_{layout}"
    runs[key].append((cfg["run"]["seed"], d))


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


def make_figure(fig, ax, out_png, out_svg):
    fig.tight_layout()
    fig.savefig(out_png, dpi=300)
    fig.savefig(out_svg)
    plt.close(fig)
    print(f"  Saved {out_png}")


def main():
    args = parse_args()

    runs = find_runs(args.runs_dir)
    print(f"Found {sum(len(v) for v in runs.values())} adder runs across {len(runs)} groups:")
    for key, dirs in sorted(runs.items()):
        print(f"  {key}: {len(dirs)} seeds")

    plot_keys = [k for k in sorted(runs) if args.only is None or k in args.only]

    max_steps = 0
    for key, dirs in runs.items():
        for seed, d in dirs:
            log = load_log(d)
            if log:
                max_steps = max(max_steps, log[-1]["step"])
    print(f"Max training steps: {max_steps}")

    # ── Per-group individual plots ────────────────────

    for key in plot_keys:
        dirs = runs[key]
        print(f"\n{key}: {len(dirs)} seeds")

        # Loss — raw
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (seed, d) in enumerate(dirs):
            log = load_log(d)
            series, common = align_to_common(log, "loss", max_steps)
            color = SEED_COLORS[i % len(SEED_COLORS)]
            ax.plot(common, series, color=color, linewidth=0.6, alpha=0.3,
                    label=f"seed={seed}")
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss (masked MSE)")
        ax.set_title(f"{key} — Loss")
        ax.legend(loc="upper right", ncol=2)
        make_figure(fig, ax, OUT_DIR / f"{key.lower()}_loss_curves.png", OUT_DIR / f"{key.lower()}_loss_curves.svg")

        # Loss — smoothed
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (seed, d) in enumerate(dirs):
            log = load_log(d)
            series, common = align_to_common(log, "loss", max_steps)
            series = moving_average(series, SMOOTH)
            color = SEED_COLORS[i % len(SEED_COLORS)]
            ax.plot(common, series, color=color, linewidth=1.5, alpha=0.85,
                    label=f"seed={seed}")
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss (masked MSE)")
        ax.set_title(f"{key} — Loss ({SMOOTH}-step MA)")
        ax.legend(loc="upper right", ncol=2)
        make_figure(fig, ax, OUT_DIR / f"{key.lower()}_loss_curves_ma.png", OUT_DIR / f"{key.lower()}_loss_curves_ma.svg")

        # Bits — raw
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (seed, d) in enumerate(dirs):
            log = load_log(d)
            series, common = align_to_common(log, "num_valid_bits", max_steps)
            color = SEED_COLORS[i % len(SEED_COLORS)]
            ax.plot(common, series, color=color, linewidth=0.6, alpha=0.3,
                    label=f"seed={seed}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Valid bits")
        ax.set_title(f"{key} — Valid bits")
        ax.legend(loc="lower right", ncol=2)
        make_figure(fig, ax, OUT_DIR / f"{key.lower()}_bits_curves.png", OUT_DIR / f"{key.lower()}_bits_curves.svg")

        # Bits — smoothed
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (seed, d) in enumerate(dirs):
            log = load_log(d)
            series, common = align_to_common(log, "num_valid_bits", max_steps)
            series = moving_average(series, SMOOTH)
            color = SEED_COLORS[i % len(SEED_COLORS)]
            ax.plot(common, series, color=color, linewidth=1.5, alpha=0.85,
                    label=f"seed={seed}")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Valid bits")
        ax.set_title(f"{key} — Valid bits ({SMOOTH}-step MA)")
        ax.legend(loc="lower right", ncol=2)
        make_figure(fig, ax, OUT_DIR / f"{key.lower()}_bits_curves_ma.png", OUT_DIR / f"{key.lower()}_bits_curves_ma.svg")

    # ── Combined comparison plots ──────────────────────

    fig, ax = plt.subplots(figsize=(10, 5))
    for key in sorted(runs):
        seed_losses = []
        for seed, d in runs[key]:
            log = load_log(d)
            series, _ = align_to_common(log, "loss", max_steps)
            series = moving_average(series, SMOOTH)
            seed_losses.append(series / series[0] if series[0] > 0 else series)
        seed_losses = np.array(seed_losses)
        mean_loss = seed_losses.mean(axis=0)
        std_loss = seed_losses.std(axis=0)
        color = SEED_COLORS[list(runs.keys()).index(key) % len(SEED_COLORS)]
        x = np.arange(len(mean_loss))
        ax.plot(x, mean_loss, color=color, linewidth=1.8, label=KEY_LABELS.get(key, key), alpha=0.9)
        ax.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.1)
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss (normalized to t=0)")
    ax.legend(loc="upper right", ncol=3)
    make_figure(fig, ax, OUT_DIR / "comparison_loss.png", OUT_DIR / "comparison_loss.svg")

    fig, ax = plt.subplots(figsize=(10, 5))
    for key in sorted(runs):
        seed_bits = []
        for seed, d in runs[key]:
            log = load_log(d)
            series, _ = align_to_common(log, "num_valid_bits", max_steps)
            series = moving_average(series, SMOOTH)
            seed_bits.append(series)
        seed_bits = np.array(seed_bits)
        mean_bits = seed_bits.mean(axis=0)
        std_bits = seed_bits.std(axis=0)
        color = SEED_COLORS[list(runs.keys()).index(key) % len(SEED_COLORS)]
        x = np.arange(len(mean_bits))
        ax.plot(x, mean_bits, color=color, linewidth=1.8, label=KEY_LABELS.get(key, key), alpha=0.9)
        ax.fill_between(x, mean_bits - std_bits, mean_bits + std_bits, color=color, alpha=0.1)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Valid bits (mean ± std)")
    ax.legend(loc="lower right", ncol=3)
    make_figure(fig, ax, OUT_DIR / "comparison_bits.png", OUT_DIR / "comparison_bits.svg")

    # ── Paper versions (no titles) ─────────────────────

    if args.paper:
        print("\n── Paper versions (no titles) ──")

        keys = sorted(runs)
        fig, ax = plt.subplots(figsize=(10, 5))
        for key in keys:
            seed_losses = []
            for seed, d in runs[key]:
                log = load_log(d)
                series, _ = align_to_common(log, "loss", max_steps)
                series = moving_average(series, SMOOTH)
                seed_losses.append(series / series[0] if series[0] > 0 else series)
            seed_losses = np.array(seed_losses)
            mean_loss = seed_losses.mean(axis=0)
            std_loss = seed_losses.std(axis=0)
            color = SEED_COLORS[keys.index(key) % len(SEED_COLORS)]
            x = np.arange(len(mean_loss))
            ax.plot(x, mean_loss, color=color, linewidth=1.8, label=KEY_LABELS.get(key, key), alpha=0.9)
            ax.fill_between(x, mean_loss - std_loss, mean_loss + std_loss, color=color, alpha=0.1)
        ax.set_yscale("log")
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss (normalized to t=0)")
        ax.legend(loc="upper right", ncol=3)
        make_figure(fig, ax, PAPER_IMAGES / "adder_seeds_loss.png", PAPER_IMAGES / "adder_seeds_loss.svg")

        fig, ax = plt.subplots(figsize=(10, 5))
        for key in keys:
            seed_bits = []
            for seed, d in runs[key]:
                log = load_log(d)
                series, _ = align_to_common(log, "num_valid_bits", max_steps)
                series = moving_average(series, SMOOTH)
                seed_bits.append(series)
            seed_bits = np.array(seed_bits)
            mean_bits = seed_bits.mean(axis=0)
            std_bits = seed_bits.std(axis=0)
            color = SEED_COLORS[keys.index(key) % len(SEED_COLORS)]
            x = np.arange(len(mean_bits))
            ax.plot(x, mean_bits, color=color, linewidth=1.8, label=KEY_LABELS.get(key, key), alpha=0.9)
            ax.fill_between(x, mean_bits - std_bits, mean_bits + std_bits, color=color, alpha=0.1)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Valid bits")
        ax.legend(loc="lower right", ncol=3)
        make_figure(fig, ax, PAPER_IMAGES / "adder_seeds_bits.png", PAPER_IMAGES / "adder_seeds_bits.svg")

    print(f"\nAll outputs in {OUT_DIR}/")
    if args.paper:
        print(f"Paper versions in {PAPER_IMAGES}/")


if __name__ == "__main__":
    main()