#!/usr/bin/env python3
"""
Analyse extrapolation sweep results.

For each run (extrap_train<K>_s<seed>_*), reads the final validation entries
from log.jsonl and produces (all saved to results/E5_extrapolation/):

  1. extrap_heatmap.svg   — accuracy % heatmap: rows=train ceiling K,
                            cols=test bit width, averaged over seeds
  2. extrap_curves.svg    — val accuracy curves over training steps,
                            one line per test k, one subplot per train K
  3. extrap_gap.svg       — accuracy vs extrapolation gap (test_k - train_k),
                            one line per train K; shows degradation with gap
  4. extrap_bars.svg      — bar chart with error bars, grouped by train K
  5. extrap_convergence.svg — step at which each (K, test_k) first hits 95%
  6. extrap_summary.csv   — raw numbers

Usage:
    uv run python scripts/analyze_extrapolation.py
    uv run python scripts/analyze_extrapolation.py --runs_dir path/to/runs
"""

import json
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

parser = ArgumentParser()
parser.add_argument("--runs_dir", default="runs")
parser.add_argument("--out_dir",  default="results/E5_extrapolation")
parser.add_argument("--last_n",   type=int, default=5,
                    help="Average over last N val checkpoints per run")
args = parser.parse_args()

RUNS_DIR = Path(args.runs_dir)
OUT_DIR  = Path(args.out_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_BITS = 8

# ── Collect runs ───────────────────────────────────────────────────────────────

runs = []
for run_dir in sorted(RUNS_DIR.iterdir()):
    cfg_path = run_dir / "config.json"
    log_path = run_dir / "log.jsonl"
    if not cfg_path.exists() or not log_path.exists():
        continue
    cfg = json.loads(cfg_path.read_text())
    if "min_bits" not in cfg.get("ds", {}):
        continue  # not an extrapolation run

    max_train = cfg["run"]["max_train_bits"]
    seed      = cfg["run"]["seed"]
    runs.append((max_train, seed, run_dir, log_path))

if not runs:
    print(f"No extrapolation runs found in {RUNS_DIR}")
    sys.exit(1)

print(f"Found {len(runs)} extrapolation runs")

# ── Parse val records ──────────────────────────────────────────────────────────

# data[max_train][seed][k] = list of (step, val_bits_pct)
data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

for max_train, seed, run_dir, log_path in runs:
    for line in log_path.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("phase") != "val":
            continue
        k      = d["k"]
        n_bits = d.get("val_n_bits", k + 1)
        pct    = d["val_bits"] / n_bits * 100
        data[max_train][seed][k].append((d["step"], pct))

# ── Summary: final accuracy averaged over seeds ────────────────────────────────

k_values_trained = sorted(data.keys())
all_test_ks      = sorted({k for mt in data.values() for s in mt.values() for k in s})

# final_acc[max_train][test_k] = mean accuracy over seeds (last N val points)
final_acc = defaultdict(dict)
final_std = defaultdict(dict)

rows = []
for mt in k_values_trained:
    for tk in all_test_ks:
        if tk <= mt:
            continue
        seed_finals = []
        for seed, kd in data[mt].items():
            if tk not in kd:
                continue
            last = [pct for _, pct in kd[tk][-args.last_n:]]
            if last:
                seed_finals.append(np.mean(last))
        if seed_finals:
            final_acc[mt][tk] = np.mean(seed_finals)
            final_std[mt][tk] = np.std(seed_finals)
            rows.append({"max_train_bits": mt, "test_k": tk,
                         "acc_pct": final_acc[mt][tk],
                         "std_pct": final_std[mt][tk],
                         "n_seeds": len(seed_finals)})

df = pd.DataFrame(rows)
csv_path = OUT_DIR / "extrap_summary.csv"
df.to_csv(csv_path, index=False)
print(f"Saved {csv_path}")
print(df.to_string(index=False))

# ── Plot 1: heatmap ────────────────────────────────────────────────────────────

heatmap = np.full((len(k_values_trained), len(all_test_ks)), np.nan)
for i, mt in enumerate(k_values_trained):
    for j, tk in enumerate(all_test_ks):
        if tk in final_acc[mt]:
            heatmap[i, j] = final_acc[mt][tk]

fig, ax = plt.subplots(figsize=(3.5, 3.5))
im = ax.imshow(heatmap, vmin=0, vmax=100, cmap="RdYlGn", aspect="auto")

ax.set_xticks(range(len(all_test_ks)))
ax.set_xticklabels([f"k={k}" for k in all_test_ks], fontsize=12)
ax.set_yticks(range(len(k_values_trained)))
ax.set_yticklabels([f"≤{mt}" for mt in k_values_trained], fontsize=12)
ax.set_xlabel("test bit width", fontsize=12)
ax.set_ylabel("train ceiling", fontsize=12)

for i in range(len(k_values_trained)):
    for j in range(len(all_test_ks)):
        v = heatmap[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=12, color="black" if 20 < v < 80 else "white")

fig.tight_layout(pad=0.5)
p = OUT_DIR / "extrap_heatmap.svg"
fig.savefig(p, bbox_inches="tight")
plt.close(fig)
print(f"Saved {p}")

# ── Plot 2: accuracy curves over training steps ────────────────────────────────

colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(all_test_ks)))

fig, axes = plt.subplots(
    len(k_values_trained), 1,
    figsize=(10, 2.5 * len(k_values_trained)),
    sharex=True,
)
if len(k_values_trained) == 1:
    axes = [axes]

for ax, mt in zip(axes, k_values_trained):
    for tk, col in zip(all_test_ks, colors):
        if tk <= mt:
            continue
        all_seed_curves = []
        for seed, kd in data[mt].items():
            if tk not in kd:
                continue
            steps = [s for s, _ in kd[tk]]
            pcts  = [p for _, p in kd[tk]]
            ax.plot(steps, pcts, color=col, alpha=0.4, linewidth=0.8)
            all_seed_curves.append((steps, pcts))

        if len(all_seed_curves) > 1:
            common = sorted({s for steps, _ in all_seed_curves for s in steps})
            interp = [np.interp(common, steps, pcts) for steps, pcts in all_seed_curves]
            ax.plot(common, np.mean(interp, axis=0), color=col, linewidth=1.8, label=f"k={tk}")
        elif all_seed_curves:
            steps, pcts = all_seed_curves[0]
            ax.plot(steps, pcts, color=col, linewidth=1.8, label=f"k={tk}")

    ax.set_ylim(0, 105)
    ax.set_ylabel("acc (%)")
    ax.set_title(f"train ≤ {mt} bits — val by test k")
    ax.legend(fontsize=7, loc="lower right", ncol=4)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")

axes[-1].set_xlabel("training step")
fig.tight_layout()
p = OUT_DIR / "extrap_curves.svg"
fig.savefig(p)
plt.close(fig)
print(f"Saved {p}")

# ── Plot 3: accuracy vs extrapolation gap ─────────────────────────────────────
#  x = test_k - train_k (gap),  y = accuracy
#  One line per train K value; shows how performance degrades as the gap grows.

fig, ax = plt.subplots(figsize=(7, 4))
train_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(k_values_trained)))

for mt, col in zip(k_values_trained, train_colors):
    gaps, accs, stds = [], [], []
    for tk in sorted(final_acc[mt].keys()):
        gaps.append(tk - mt)
        accs.append(final_acc[mt][tk])
        stds.append(final_std[mt][tk])
    gaps, accs, stds = np.array(gaps), np.array(accs), np.array(stds)
    ax.plot(gaps, accs, marker="o", color=col, linewidth=1.6, label=f"train ≤{mt}")
    ax.fill_between(gaps, accs - stds, np.minimum(accs + stds, 100),
                    color=col, alpha=0.15)

ax.set_xlabel("extrapolation gap  (test k − train ceiling)")
ax.set_ylabel("accuracy (%)")
ax.set_title("Accuracy vs extrapolation gap")
ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")
ax.set_ylim(80, 101)
ax.legend(fontsize=8)
fig.tight_layout()
p = OUT_DIR / "extrap_gap.svg"
fig.savefig(p)
plt.close(fig)
print(f"Saved {p}")

# ── Plot 4: bar chart with error bars ──────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4))
bar_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(k_values_trained)))

n_groups = len(all_test_ks)
bar_width = 0.8 / len(k_values_trained)

for i, (mt, col) in enumerate(zip(k_values_trained, bar_colors)):
    for j, tk in enumerate(all_test_ks):
        if tk not in final_acc[mt]:
            continue
        x = j + (i - len(k_values_trained) / 2 + 0.5) * bar_width
        ax.bar(x, final_acc[mt][tk], bar_width * 0.9,
               color=col, alpha=0.85,
               yerr=final_std[mt][tk], capsize=2, error_kw={"linewidth": 0.8})

# legend patches
import matplotlib.patches as mpatches
handles = [mpatches.Patch(color=c, label=f"train ≤{mt}")
           for mt, c in zip(k_values_trained, bar_colors)]
ax.legend(handles=handles, fontsize=8)
ax.set_xticks(range(n_groups))
ax.set_xticklabels([f"k={tk}" for tk in all_test_ks])
ax.set_xlabel("test bit width")
ax.set_ylabel("accuracy (%)")
ax.set_title("Extrapolation accuracy by test width and training ceiling")
ax.set_ylim(80, 101)
ax.axhline(100, color="gray", linewidth=0.5, linestyle="--")
fig.tight_layout()
p = OUT_DIR / "extrap_bars.svg"
fig.savefig(p)
plt.close(fig)
print(f"Saved {p}")

# ── Plot 5: convergence speed ──────────────────────────────────────────────────
#  For each (K, test_k) pair, find the first validation step where mean accuracy
#  across seeds exceeds THRESH — this shows how quickly extrapolation emerges.

THRESH = 95.0

fig, ax = plt.subplots(figsize=(7, 4))

for mt, col in zip(k_values_trained, train_colors):
    conv_steps, conv_tks = [], []
    for tk in sorted(final_acc[mt].keys()):
        # build mean curve across seeds
        all_seed_curves = []
        for seed, kd in data[mt].items():
            if tk not in kd:
                continue
            steps = [s for s, _ in kd[tk]]
            pcts  = [p for _, p in kd[tk]]
            all_seed_curves.append((steps, pcts))

        if not all_seed_curves:
            continue

        common = sorted({s for steps, _ in all_seed_curves for s in steps})
        interp = [np.interp(common, steps, pcts) for steps, pcts in all_seed_curves]
        mean_curve = np.mean(interp, axis=0)

        hit = np.argmax(mean_curve >= THRESH)
        if mean_curve[hit] >= THRESH:
            conv_steps.append(common[hit])
            conv_tks.append(tk)

    if conv_steps:
        ax.plot(conv_tks, conv_steps, marker="o", color=col, linewidth=1.6,
                label=f"train ≤{mt}")

ax.set_xlabel("test bit width k")
ax.set_ylabel(f"steps to reach {THRESH:.0f}% accuracy")
ax.set_title(f"Convergence speed: steps to {THRESH:.0f}% on each test k")
ax.legend(fontsize=8)
ax.set_ylim(bottom=0)
fig.tight_layout()
p = OUT_DIR / "extrap_convergence.svg"
fig.savefig(p)
plt.close(fig)
print(f"Saved {p}")
