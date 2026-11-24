#!/usr/bin/env python3
"""
Damage-sweep analysis (D4/D5): group runs by (task, damage set) and compare seeds.

Scans runs/ for damage-training runs (config.json with "input_damages",
experiments D4 = 4-bit adder, D5 = gates). Runs are grouped into one "virtual
run" per (task, damage set) — e.g. (XOR, perlin:a60:p0.8) — across all seeds,
so each noise damage trained separately shows up as a single group with one
loss curve per seed (different colors) plus a bold mean curve.

Outputs under results/damage_sweep/ (override with --out):
    per group (task + damage set):
        loss_seeds_<group>.png    — training loss, one color per seed + mean
        bits_seeds_<group>.png    — valid bits, one color per seed + mean
        eval_clean_seeds_<group>.png — eval "clean" bit accuracy over steps, per seed
        eval_matrix_<group>.png   — latest eval robustness matrix, bars per seed
        rollout_<group>_s<seed>.gif — each run's rollout GIF, copied per group
    overview:
        summary.csv               — one row per (group, seed) run
        matrix_final_bits.csv     — rows: (task, damage), cols: seeds + mean of final bits
        overview_final_bits.png   — heatmap: tasks x damage sets, mean final bits over seeds
        overview_loss_<group>.png per group is above; overview curves in overview_curves.png

Usage:
    uv run python scripts/analyze/analyze_damage_sweep.py
    uv run python scripts/analyze/analyze_damage_sweep.py --runs-dir runs --out results/damage_sweep
"""

import argparse
import csv
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

RUNS_DIR = Path("runs")
OUT_DIR = Path("results/damage_sweep")
EXPERIMENTS = {"D4", "D5"}


def slug(s: str, max_len: int = 80) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", s).strip("-")[:max_len]


def load_run(run_dir: Path):
    cfg_path = run_dir / "config.json"
    log_path = run_dir / "log.jsonl"
    if not cfg_path.exists() or not log_path.exists():
        return None
    with open(cfg_path) as f:
        cfg = json.load(f)
    run = cfg.get("run", {})
    if run.get("experiment") not in EXPERIMENTS:
        return None
    task = run.get("gate") or "adder4"
    input_damages = [
        f"in:{d['kind']}:a{d['amplitude']:g}:p{d.get('prob', 1.0):g}"
        for d in cfg.get("input_damages", [])
    ]
    state_damages = [
        f"st:{d['kind']}:a{d['amplitude']:g}:p{d.get('prob', 1.0):g}"
        for d in cfg.get("nca", {}).get("state_damages", [])
    ]
    damage_key = "+".join(input_damages + state_damages) or "clean"

    steps, losses, bits = [], [], []
    eval_steps, eval_clean, eval_per_cfg = [], [], []
    with open(log_path) as f:
        for line in f:
            e = json.loads(line)
            if "loss" in e:
                steps.append(e["step"])
                losses.append(e["loss"])
                bits.append(e.get("num_valid_bits"))
            if "per_cfg" in e:
                eval_steps.append(e["step"])
                per = {c["label"]: c for c in e["per_cfg"]}
                eval_per_cfg.append(per)
                eval_clean.append(per.get("clean", {}).get("bits"))
    return {
        "run_dir": run_dir,
        "name": run_dir.name,
        "experiment": run.get("experiment"),
        "task": task,
        "damage_key": damage_key,
        "damage_tag": run.get("damage_tag", slug(damage_key, 40)),
        "seed": run.get("seed", -1),
        "steps": np.array(steps),
        "losses": np.array(losses, dtype=float),
        "bits": np.array([b if b is not None else np.nan for b in bits], dtype=float),
        "eval_steps": np.array(eval_steps),
        "eval_clean": np.array(
            [b if b is not None else np.nan for b in eval_clean], dtype=float
        ),
        "eval_per_cfg": eval_per_cfg,
    }


def group_runs(runs):
    groups = defaultdict(list)
    for r in runs:
        groups[(r["task"], r["damage_key"])].append(r)
    for key in groups:
        groups[key].sort(key=lambda r: r["seed"])
    return groups


def mean_curve(steps_list, values_list, grid):
    acc = np.zeros_like(grid, dtype=float)
    n = np.zeros_like(grid, dtype=float)
    for s, v in zip(steps_list, values_list):
        if len(s) == 0:
            continue
        acc += np.interp(grid, s, v, left=np.nan, right=np.nan)
        n += ~np.isnan(np.interp(grid, s, v, left=np.nan, right=np.nan))
    with np.errstate(invalid="ignore"):
        return np.where(n > 0, acc / np.maximum(n, 1), np.nan)


def plot_seed_curves(group_runs, key, out_dir, metric, ylabel, logy, fname_suffix):
    task, damage_key = key
    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = plt.get_cmap("tab10")
    grid = None
    all_steps = [r["steps"] for r in group_runs if len(r["steps"])]
    if all_steps:
        grid = np.linspace(
            min(s.min() for s in all_steps), max(s.max() for s in all_steps), 500
        )
    vals = []
    for i, r in enumerate(group_runs):
        color = cmap(i % 10)
        v = r[metric]
        ax.plot(r["steps"], v, color=color, alpha=0.75, lw=1, label=f"seed {r['seed']}")
        vals.append(v)
    if grid is not None and vals:
        m = mean_curve([r["steps"] for r in group_runs], vals, grid)
        ax.plot(grid, m, color="black", lw=2.5, label="mean")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{task} — {damage_key} — {len(group_runs)} seeds")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        out_dir / f"{fname_suffix}_{slug(task + '__' + damage_key)}.png", dpi=120
    )
    plt.close(fig)


def plot_eval(group_runs, key, out_dir):
    task, damage_key = key
    gname = slug(task + "__" + damage_key)
    cmap = plt.get_cmap("tab10")

    # clean eval bits over steps, per seed
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    for i, r in enumerate(group_runs):
        if len(r["eval_steps"]):
            ax.plot(
                r["eval_steps"],
                r["eval_clean"],
                color=cmap(i % 10),
                marker="o",
                ms=3,
                lw=1,
                label=f"seed {r['seed']}",
            )
            plotted = True
    ax.set_xlabel("step")
    ax.set_ylabel("eval clean bit accuracy")
    ax.set_title(f"{task} — {damage_key} — eval clean bits per seed")
    if plotted:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / f"eval_clean_seeds_{gname}.png", dpi=120)
    plt.close(fig)

    # latest robustness matrix: grouped bars per eval label, one color per seed
    runs_with_eval = [r for r in group_runs if r["eval_per_cfg"]]
    if runs_with_eval:
        labels = list(runs_with_eval[0]["eval_per_cfg"][-1].keys())
        x = np.arange(len(labels))
        width = 0.8 / len(runs_with_eval)
        fig, ax = plt.subplots(figsize=(12, 4))
        for i, r in enumerate(runs_with_eval):
            vals = [
                r["eval_per_cfg"][-1].get(l, {}).get("bits", np.nan) for l in labels
            ]
            ax.bar(
                x + i * width,
                vals,
                width=width,
                color=cmap(i % 10),
                label=f"seed {r['seed']}",
            )
        ax.set_xticks(x + width * (len(runs_with_eval) - 1) / 2)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("eval bit accuracy (latest)")
        ax.set_title(f"{task} — {damage_key} — final robustness matrix")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"eval_matrix_{gname}.png", dpi=120)
        plt.close(fig)


def final_bits(r):
    if len(r["eval_clean"]) and not np.isnan(r["eval_clean"][-1]):
        return float(r["eval_clean"][-1])
    if len(r["bits"]) and not np.isnan(r["bits"][-1]):
        return float(r["bits"][-1])
    return np.nan


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument(
        "--min-steps", type=int, default=0, help="ignore runs with fewer logged steps"
    )
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    runs = []
    for d in sorted(args.runs_dir.iterdir()):
        if not d.is_dir():
            continue
        r = load_run(d)
        if r is None:
            continue
        if len(r["steps"]) < args.min_steps:
            continue
        runs.append(r)
    print(f"Found {len(runs)} damage-sweep runs under {args.runs_dir}")

    groups = group_runs(runs)
    print(f"{len(groups)} (task, damage) groups")

    summary_rows = []
    for key, gruns in sorted(groups.items()):
        task, damage_key = key
        for r in gruns:
            summary_rows.append(
                {
                    "experiment": r["experiment"],
                    "task": task,
                    "damage_key": damage_key,
                    "seed": r["seed"],
                    "run": r["name"],
                    "steps_logged": len(r["steps"]),
                    "final_loss": (
                        float(r["losses"][-1]) if len(r["losses"]) else np.nan
                    ),
                    "final_bits": final_bits(r),
                }
            )
        plot_seed_curves(gruns, key, args.out, "losses", "loss", True, "loss_seeds")
        plot_seed_curves(
            gruns, key, args.out, "bits", "valid bits", False, "bits_seeds"
        )
        plot_eval(gruns, key, args.out)

        # collect each run's rollout GIF into the output dir, one per seed
        gname = slug(task + "__" + damage_key)
        for r in gruns:
            gif = r["run_dir"] / "rollout_latest.gif"
            if gif.exists():
                shutil.copy(gif, args.out / f"rollout_{gname}_s{r['seed']}.gif")

    # summary.csv — one row per run
    with open(args.out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=list(summary_rows[0].keys()) if summary_rows else []
        )
        w.writeheader()
        w.writerows(summary_rows)

    # matrix_final_bits.csv — rows (task, damage), cols seeds + mean
    matrix = {}
    for key, gruns in groups.items():
        task, damage_key = key
        matrix.setdefault(task, {})[damage_key] = {
            r["seed"]: final_bits(r) for r in gruns
        }
    seeds = sorted({r["seed"] for r in runs})
    with open(args.out / "matrix_final_bits.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "damage"] + [f"s{s}" for s in seeds] + ["mean"])
        for task in sorted(matrix):
            for damage_key, per_seed in sorted(matrix[task].items()):
                vals = [per_seed.get(s, np.nan) for s in seeds]
                mean = np.nanmean(vals) if vals else np.nan
                w.writerow(
                    [task, damage_key]
                    + [f"{v:.3f}" if not np.isnan(v) else "" for v in vals]
                    + [f"{mean:.3f}" if not np.isnan(mean) else ""]
                )

    # overview heatmap: tasks x damage sets, mean final bits over seeds
    tasks = sorted(matrix)
    damages = sorted({d for t in matrix for d in matrix[t]})
    if tasks and damages:
        M = np.full((len(tasks), len(damages)), np.nan)
        for i, t in enumerate(tasks):
            for j, d in enumerate(damages):
                vals = [v for v in matrix[t].get(d, {}).values() if not np.isnan(v)]
                if vals:
                    M[i, j] = np.mean(vals)
        fig, ax = plt.subplots(
            figsize=(max(8, len(damages) * 0.9), max(4, len(tasks) * 0.5))
        )
        im = ax.imshow(
            M,
            aspect="auto",
            cmap="viridis",
            vmin=np.nanmin(M) if np.isfinite(M).any() else 0,
            vmax=np.nanmax(M) if np.isfinite(M).any() else 1,
        )
        ax.set_xticks(range(len(damages)))
        ax.set_xticklabels(damages, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(tasks)))
        ax.set_yticklabels(tasks, fontsize=8)
        for i in range(len(tasks)):
            for j in range(len(damages)):
                if not np.isnan(M[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{M[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if M[i, j] < np.nanmean(M) else "black",
                    )
        ax.set_title("Final bit accuracy — mean over seeds (tasks x damage sets)")
        fig.colorbar(im, ax=ax, label="final bits")
        fig.tight_layout()
        fig.savefig(args.out / "overview_final_bits.png", dpi=140)
        plt.close(fig)

    print(f"Wrote analysis to {args.out}")
    for key in sorted(groups):
        task, damage_key = key
        seeds = [r["seed"] for r in groups[key]]
        print(f"  {task:10s} | {damage_key[:70]:70s} | seeds={seeds}")


if __name__ == "__main__":
    main()
