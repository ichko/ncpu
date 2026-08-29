#!/usr/bin/env python3
"""
Gate ablation analysis — kernel size (E_KS), alive masking (E_AM),
and zero initialization (E_ZI).

Baselines (k=5, am=0.0, zi=False) are pulled from matching E1 runs.

Usage:
    uv run python scripts/analyze/analyze_ablations.py

Outputs under results/E_ablations/:
    summary_ks.csv          — per-run metrics for kernel size ablation
    summary_am.csv          — per-run metrics for alive masking ablation
    summary_zi.csv          — per-run metrics for zero-init ablation
    ks_loss.svg             — loss curves per gate, k=3/5/7
    ks_convergence.svg      — convergence bar chart, kernel size
    am_loss.svg             — loss curves per gate, am=0.0/0.1
    am_convergence.svg      — convergence bar chart, alive masking
    zi_loss.svg             — loss curves per gate, zi=False/True
    zi_convergence.svg      — convergence bar chart, zero init
    ks_snapshots.svg        — final snapshot grid, rows=gates × cols=kernel sizes
    am_snapshots.svg        — final snapshot grid, rows=gates × cols=am values
    zi_snapshots.svg        — final snapshot grid, rows=gates × cols=zi values
    ks_rollout_{gate}.svg   — channel 0 over time, rows=kernel sizes
    am_rollout_{gate}.svg   — channel 0 over time, rows=am values
    zi_rollout_{gate}.svg   — channel 0 over time, rows=zi values
    ablations_combined.svg  — paper figure: all three convergence bars in one row
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.utils import make_io_screen_cols1

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results/E_ablations")
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

GATES     = ["XOR", "AND", "half_adder"]
N_BITS    = {"half_adder": 2}
max_steps = 15000

KS_VALUES = [3, 5, 7]
KS_COLORS = {3: "#E15759", 5: "#4E79A7", 7: "#59A14F"}

AM_VALUES = [0.0, 0.1]
AM_COLORS = {0.0: "#4E79A7", 0.1: "#F28E2B"}

ZI_VALUES = [False, True]
ZI_COLORS = {False: "#4E79A7", True: "#E15759"}
ZI_LABELS = {False: "zi=False (default)", True: "zi=True"}

# ── Find runs ──────────────────────────────────────────────────────────────────

def _latest_per_seed(dirs_with_seed):
    best = {}
    for seed, ts, d in dirs_with_seed:
        if seed not in best or ts > best[seed][0]:
            best[seed] = (ts, d)
    return [d for _, d in sorted(best.values(), key=lambda x: x[0])]

def find_ks_runs():
    raw = defaultdict(lambda: defaultdict(list))
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg  = json.load(open(cfg_path))
        run  = cfg.get("run", {})
        exp  = run.get("experiment", "")
        gate = run.get("gate", "")
        if gate not in GATES:
            continue
        seed = run.get("seed", 0)
        ts   = d.name.split("_")[-1]
        ks   = cfg["nca"]["kernel_size"]
        am   = cfg["nca"]["alive_threshold"]
        if exp == "E_KS":
            raw[gate][ks].append((seed, ts, d))
        elif exp == "E1" and ks == 5 and am == 0.0:
            raw[gate][5].append((seed, ts, d))
    return {gate: {ks: _latest_per_seed(items) for ks, items in gd.items()}
            for gate, gd in raw.items()}

def find_am_runs():
    raw = defaultdict(lambda: defaultdict(list))
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg  = json.load(open(cfg_path))
        run  = cfg.get("run", {})
        exp  = run.get("experiment", "")
        gate = run.get("gate", "")
        if gate not in GATES:
            continue
        seed = run.get("seed", 0)
        ts   = d.name.split("_")[-1]
        ks   = cfg["nca"]["kernel_size"]
        am   = cfg["nca"]["alive_threshold"]
        if exp == "E_AM":
            raw[gate][am].append((seed, ts, d))
        elif exp == "E1" and ks == 5 and am == 0.0:
            raw[gate][0.0].append((seed, ts, d))
    return {gate: {am: _latest_per_seed(items) for am, items in gd.items()}
            for gate, gd in raw.items()}

def find_zi_runs():
    raw = defaultdict(lambda: defaultdict(list))
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg  = json.load(open(cfg_path))
        run  = cfg.get("run", {})
        exp  = run.get("experiment", "")
        gate = run.get("gate", "")
        if gate not in GATES:
            continue
        seed = run.get("seed", 0)
        ts   = d.name.split("_")[-1]
        ks   = cfg["nca"]["kernel_size"]
        am   = cfg["nca"]["alive_threshold"]
        zi   = cfg["nca"]["zero_initialization"]
        if exp == "E_ZI":
            raw[gate][True].append((seed, ts, d))
        elif exp == "E1" and ks == 5 and am == 0.0:
            raw[gate][False].append((seed, ts, d))
    return {gate: {zi: _latest_per_seed(items) for zi, items in gd.items()}
            for gate, gd in raw.items()}

ks_runs = find_ks_runs()
am_runs = find_am_runs()
zi_runs = find_zi_runs()

print("Kernel size runs:")
for gate in GATES:
    for ks in KS_VALUES:
        n = len(ks_runs.get(gate, {}).get(ks, []))
        print(f"  {gate:<14} k={ks}  {n} seed(s)")

print("\nAlive masking runs:")
for gate in GATES:
    for am in AM_VALUES:
        n = len(am_runs.get(gate, {}).get(am, []))
        print(f"  {gate:<14} am={am}  {n} seed(s)")

print("\nZero init runs:")
for gate in GATES:
    for zi in ZI_VALUES:
        n = len(zi_runs.get(gate, {}).get(zi, []))
        print(f"  {gate:<14} zi={zi}  {n} seed(s)")

# ── Helpers ────────────────────────────────────────────────────────────────────

def load_log(run_dir):
    return [json.loads(l) for l in open(run_dir / "log.jsonl")]

def steps_to_full_bits(log, n_bits):
    for e in log:
        if e.get("num_valid_bits", 0) >= n_bits:
            return e["step"]
    return None

def align_series(logs, key, n_steps, smooth=50):
    common = np.arange(n_steps)
    arrays = []
    for log in logs:
        steps = np.array([e["step"] for e in log])
        vals  = np.array([e[key]  for e in log])
        arrays.append(np.interp(common, steps, vals))
    arr  = np.array(arrays)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    if smooth > 1:
        k    = np.ones(smooth) / smooth
        mean = np.convolve(mean, k, mode="valid")
        std  = np.convolve(std,  k, mode="valid")
        x    = common[smooth-1:]
    else:
        x = common
    return x, mean, std

def plot_band(ax, x, mean, std, color, label):
    ax.plot(x, mean, color=color, linewidth=1.5, label=label)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)

def save(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {name}")

# ── Summary CSVs ───────────────────────────────────────────────────────────────

with open(OUT_DIR / "summary_ks.csv", "w") as f:
    f.write("gate,kernel_size,seed,final_loss,valid_bits,conv_step\n")
    for gate in GATES:
        n_bits = N_BITS.get(gate, 1)
        for ks in KS_VALUES:
            for seed_i, d in enumerate(ks_runs.get(gate, {}).get(ks, [])):
                log  = load_log(d)
                last = log[-1]
                conv = steps_to_full_bits(log, n_bits)
                f.write(f"{gate},{ks},{seed_i},{last['loss']:.6f},"
                        f"{last['num_valid_bits']:.3f},"
                        f"{conv if conv is not None else 'none'}\n")

with open(OUT_DIR / "summary_am.csv", "w") as f:
    f.write("gate,alive_threshold,seed,final_loss,valid_bits,conv_step\n")
    for gate in GATES:
        n_bits = N_BITS.get(gate, 1)
        for am in AM_VALUES:
            for seed_i, d in enumerate(am_runs.get(gate, {}).get(am, [])):
                log  = load_log(d)
                last = log[-1]
                conv = steps_to_full_bits(log, n_bits)
                f.write(f"{gate},{am},{seed_i},{last['loss']:.6f},"
                        f"{last['num_valid_bits']:.3f},"
                        f"{conv if conv is not None else 'none'}\n")

with open(OUT_DIR / "summary_zi.csv", "w") as f:
    f.write("gate,zero_init,seed,final_loss,valid_bits,conv_step\n")
    for gate in GATES:
        n_bits = N_BITS.get(gate, 1)
        for zi in ZI_VALUES:
            for seed_i, d in enumerate(zi_runs.get(gate, {}).get(zi, [])):
                log  = load_log(d)
                last = log[-1]
                conv = steps_to_full_bits(log, n_bits)
                f.write(f"{gate},{zi},{seed_i},{last['loss']:.6f},"
                        f"{last['num_valid_bits']:.3f},"
                        f"{conv if conv is not None else 'none'}\n")

print("\nSaved: summary_ks.csv, summary_am.csv, summary_zi.csv")

# ── Generic plot helpers ────────────────────────────────────────────────────────

def loss_curves(runs_dict, values, colors, labels, title, fname):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, gate in zip(axes, GATES):
        for val in values:
            dirs = runs_dict.get(gate, {}).get(val, [])
            if not dirs:
                continue
            logs = [load_log(d) for d in dirs]
            x, mean, std = align_series(logs, "loss", max_steps)
            plot_band(ax, x, mean, std, colors[val], labels[val])
        ax.set_yscale("log")
        ax.set_title(gate)
        ax.set_xlabel("Training step")
        if ax is axes[0]:
            ax.set_ylabel("Loss (masked MSE)")
        ax.legend()
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save(fig, fname)

def conv_bars(runs_dict, values, colors, labels, title, fname, rot=0):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4), sharey=True)
    x_pos = np.arange(len(values))
    for ax, gate in zip(axes, GATES):
        n_bits = N_BITS.get(gate, 1)
        means, stds = [], []
        for val in values:
            dirs = runs_dict.get(gate, {}).get(val, [])
            convs = []
            for d in dirs:
                c = steps_to_full_bits(load_log(d), n_bits)
                convs.append(max_steps if c is None else c)
            means.append(np.mean(convs) if convs else max_steps)
            stds.append(np.std(convs)   if convs else 0)
        ax.bar(x_pos, means, yerr=stds, capsize=4, width=0.5,
               color=[colors[v] for v in values])
        ax.set_xticks(x_pos)
        ax.set_xticklabels([labels[v] for v in values], rotation=rot, ha="right" if rot else "center")
        ax.set_title(gate)
        if ax is axes[0]:
            ax.set_ylabel("Steps to full convergence")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    save(fig, fname)

# ── KS figures ─────────────────────────────────────────────────────────────────

loss_curves(ks_runs, KS_VALUES, KS_COLORS,
            {k: f"k={k}" for k in KS_VALUES},
            "Kernel size ablation — loss curves (mean ± std)", "ks_loss.svg")

conv_bars(ks_runs, KS_VALUES, KS_COLORS,
          {k: f"k={k}" for k in KS_VALUES},
          "Kernel size ablation — steps to convergence", "ks_convergence.svg")

# ── AM figures ─────────────────────────────────────────────────────────────────

am_display = {0.0: "am=0.0 (off)", 0.1: "am=0.1"}

loss_curves(am_runs, AM_VALUES, AM_COLORS, am_display,
            "Alive masking ablation — loss curves (mean ± std)", "am_loss.svg")

conv_bars(am_runs, AM_VALUES, AM_COLORS, am_display,
          "Alive masking ablation — steps to convergence", "am_convergence.svg", rot=10)

# ── ZI figures ─────────────────────────────────────────────────────────────────

loss_curves(zi_runs, ZI_VALUES, ZI_COLORS, ZI_LABELS,
            "Zero initialization ablation — loss curves (mean ± std)", "zi_loss.svg")

conv_bars(zi_runs, ZI_VALUES, ZI_COLORS, ZI_LABELS,
          "Zero initialization ablation — steps to convergence", "zi_convergence.svg")

# ── Print summary tables ────────────────────────────────────────────────────────

for label, runs_dict, values in [
    ("Kernel size", ks_runs, KS_VALUES),
    ("Alive masking", am_runs, AM_VALUES),
    ("Zero init", zi_runs, ZI_VALUES),
]:
    print(f"\n── {label} ──")
    print(f"{'gate':<14} {'val':>8} {'seeds':>6} {'loss mean':>10} {'conv mean':>10}")
    for gate in GATES:
        n_bits = N_BITS.get(gate, 1)
        for val in values:
            dirs = runs_dict.get(gate, {}).get(val, [])
            if not dirs:
                continue
            losses = [json.loads(open(d/"log.jsonl").readlines()[-1])["loss"] for d in dirs]
            convs  = [steps_to_full_bits(load_log(d), n_bits) or max_steps for d in dirs]
            print(f"{gate:<14} {str(val):>8} {len(dirs):>6} {np.mean(losses):>10.4f} {np.mean(convs):>10.0f}")

# ── NCA rollout helpers ────────────────────────────────────────────────────────

TSTEPS = [0, 8, 16, 32, 48, 64]

SHOWCASE = {
    "AND": {
        "combos":    [[0,0],[0,1],[1,0],[1,1]],
        "out":       [[0],  [0],  [0],  [1]],
        "labels":    ["0,0→0","0,1→0","1,0→0","1,1→1"],
        "highlight": 3,
    },
    "XOR": {
        "combos":    [[0,0],[0,1],[1,0],[1,1]],
        "out":       [[0],  [1],  [1],  [0]],
        "labels":    ["0,0→0","0,1→1","1,0→1","1,1→0"],
        "highlight": 3,
    },
    "half_adder": {
        "combos":    [[0,0],[0,1],[1,0],[1,1]],
        "out":       [[0,0],[1,0],[1,0],[0,1]],
        "labels":    ["0,0→s=0,c=0","0,1→s=1,c=0","1,0→s=1,c=0","1,1→s=0,c=1"],
        "highlight": 3,
    },
}


def load_nca(run_dir):
    nc  = json.load(open(run_dir / "config.json"))["nca"]
    nca = NeuralCA(
        channels=nc["channels"], hidden_channels=nc["hidden_channels"],
        fire_rate=nc["fire_rate"], alive_threshold=nc["alive_threshold"],
        zero_initialization=nc["zero_initialization"], kernel_size=nc["kernel_size"],
        num_perception_kernels=nc["num_perception_kernels"],
        read_only_dims=nc["read_only_dims"],
        padding_type=nc.get("padding_type", "zeros"),
    )
    ckpts = sorted((run_dir / "checkpoints").glob("nca_*.pt"))
    if not ckpts:
        return None
    nca.load_state_dict(torch.load(ckpts[-1], map_location="cpu", weights_only=True), strict=False)
    nca.eval()
    return nca


def make_state(ds_cfg, inp_bits):
    screen = make_io_screen_cols1(
        H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
        spacing=ds_cfg["spacing"], left_input=inp_bits, right_input=[],
    )
    img   = torch.from_numpy(screen).float() / 128.0 - 1.0
    state = torch.zeros(1, 16, ds_cfg["H"], ds_cfg["W"])
    state[0, 0] = img
    state[0, 1] = img
    return state


def get_rollout(nca, state, max_t=64):
    with torch.no_grad():
        return nca.forward(state, steps=max_t)[0]


def best_dir(dirs):
    return min(dirs, key=lambda d: json.loads(open(d / "log.jsonl").readlines()[-1])["loss"])


def rollout_strip(fig_title, param_labels, param_rollouts, ds_cfg, tgt_bits, out_path):
    valid = [(lbl, ro) for lbl, ro in zip(param_labels, param_rollouts) if ro is not None]
    if not valid:
        return
    n_rows = len(valid)
    n_cols = len(TSTEPS) + 1
    cell   = 0.72
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell, n_rows * cell),
                             gridspec_kw={"hspace": 0.02, "wspace": 0.02},
                             constrained_layout=True)
    if n_rows == 1:
        axes = [axes]
    for ri, (lbl, rollout) in enumerate(valid):
        for ci, t in enumerate(TSTEPS):
            t_idx = min(t, rollout.shape[0] - 1)
            ax    = axes[ri][ci]
            ax.imshow(rollout[t_idx, 0].numpy(), cmap="viridis",
                      vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"t={t}", fontsize=8, pad=2)
            if ci == 0:
                ax.set_ylabel(lbl, fontsize=7)
        target_screen = make_io_screen_cols1(
            H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
            spacing=ds_cfg["spacing"], left_input=[], right_input=tgt_bits,
        )
        target_img = torch.from_numpy(target_screen).float() / 128.0 - 1.0
        ax = axes[ri][n_cols - 1]
        ax.imshow(target_img.numpy(), cmap="viridis", vmin=-1, vmax=1,
                  interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title("target", fontsize=8, pad=2)
    fig.suptitle(fig_title, fontsize=9)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def snapshot_grid(runs_dict, values, val_labels, title, fname):
    n_g  = len(GATES)
    n_v  = len(values)
    fig, axes = plt.subplots(n_g, n_v, figsize=(2.5 * n_v, 2.5 * n_g),
                             gridspec_kw={"hspace": 0.05, "wspace": 0.02})
    any_snap = False
    for gi, gate in enumerate(GATES):
        for vi, val in enumerate(values):
            ax   = axes[gi][vi] if n_g > 1 else axes[vi]
            dirs = runs_dict.get(gate, {}).get(val, [])
            if dirs:
                d    = best_dir(dirs)
                snap = d / "snapshot_latest.png"
                if snap.exists():
                    ax.imshow(Image.open(snap), rasterized=True)
                    any_snap = True
            ax.axis("off")
            if gi == 0:
                ax.set_title(val_labels[val], fontsize=9, pad=3)
            if vi == 0:
                ax.text(-0.08, 0.5, gate, fontsize=8, rotation=0,
                        ha="right", va="center", transform=ax.transAxes)
    fig.suptitle(title, fontsize=10, y=1.01)
    if any_snap:
        p = OUT_DIR / fname
        fig.savefig(p, bbox_inches="tight", pad_inches=0.05)
        print(f"Saved: {fname}")
    plt.close(fig)


def rollout_figures(runs_dict, values, val_labels, prefix):
    print(f"\n── {prefix.upper()} rollout figures ──")
    for gate in GATES:
        sc = SHOWCASE.get(gate)
        if sc is None:
            continue
        hi_inp     = sc["combos"][sc["highlight"]]
        hi_lbl     = sc["labels"][sc["highlight"]]
        ds_cfg_ref = None
        param_labels, param_rollouts = [], []
        for val in values:
            dirs = runs_dict.get(gate, {}).get(val, [])
            if not dirs:
                param_labels.append(val_labels[val])
                param_rollouts.append(None)
                continue
            d      = best_dir(dirs)
            nca    = load_nca(d)
            ds_cfg = json.load(open(d / "config.json"))["ds"]
            ds_cfg_ref = ds_cfg
            param_labels.append(val_labels[val])
            param_rollouts.append(get_rollout(nca, make_state(ds_cfg, hi_inp)) if nca else None)
        if ds_cfg_ref is None:
            print(f"  {gate}: no data, skipping")
            continue
        out_path = OUT_DIR / f"{prefix}_rollout_{gate}.svg"
        rollout_strip(
            f"{gate}  —  {prefix} ablation  —  {hi_lbl}",
            param_labels, param_rollouts, ds_cfg_ref,
            sc["out"][sc["highlight"]], out_path,
        )
        print(f"  {gate}: saved {prefix}_rollout_{gate}.svg")

# ── KS rollout + snapshots ─────────────────────────────────────────────────────

rollout_figures(ks_runs, KS_VALUES, {k: f"k={k}" for k in KS_VALUES}, "ks")
snapshot_grid(ks_runs, KS_VALUES, {k: f"k={k}" for k in KS_VALUES},
              "KS ablation — final snapshots", "ks_snapshots.svg")

# ── AM rollout + snapshots ─────────────────────────────────────────────────────

rollout_figures(am_runs, AM_VALUES, {0.0: "am=0.0 (off)", 0.1: "am=0.1"}, "am")
snapshot_grid(am_runs, AM_VALUES, {0.0: "am=0.0 (off)", 0.1: "am=0.1"},
              "AM ablation — final snapshots", "am_snapshots.svg")

# ── ZI rollout + snapshots ─────────────────────────────────────────────────────

rollout_figures(zi_runs, ZI_VALUES, ZI_LABELS, "zi")
snapshot_grid(zi_runs, ZI_VALUES, ZI_LABELS,
              "ZI ablation — final snapshots", "zi_snapshots.svg")

# ── Paper figure: combined convergence bars (all three ablations, one row) ──────

PANEL_SPECS = [
    ("Kernel size  (k)", ks_runs, KS_VALUES, KS_COLORS,
     {k: f"k={k}" for k in KS_VALUES}, 0),
    ("Alive masking  (am)", am_runs, AM_VALUES, AM_COLORS,
     {0.0: "am=0 (off)", 0.1: "am=0.1"}, 10),
    ("Zero init  (zi)", zi_runs, ZI_VALUES, ZI_COLORS,
     ZI_LABELS, 0),
]

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)

for col, (ablation_label, runs_dict, values, colors, labels, rot) in enumerate(PANEL_SPECS):
    ax = axes[col]
    x_pos = np.arange(len(values))
    # one group per gate, each gate is a cluster of bars
    bar_w = 0.22
    offsets = np.linspace(-(len(GATES)-1)/2, (len(GATES)-1)/2, len(GATES)) * bar_w
    gate_colors = ["#4E79A7", "#59A14F", "#B07AA1"]   # AND, XOR, half-adder

    for gi, gate in enumerate(GATES):
        n_bits = N_BITS.get(gate, 1)
        means, stds = [], []
        for val in values:
            dirs = runs_dict.get(gate, {}).get(val, [])
            convs = [steps_to_full_bits(load_log(d), n_bits) or max_steps for d in dirs]
            means.append(np.mean(convs) if convs else max_steps)
            stds.append(np.std(convs)   if convs else 0)
        ax.bar(x_pos + offsets[gi], means, width=bar_w, yerr=stds, capsize=3,
               color=gate_colors[gi], alpha=0.85,
               label=gate if col == 0 else None)

    ax.axhline(max_steps, color="#cccccc", linestyle="--", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([labels[v] for v in values], rotation=rot,
                       ha="right" if rot else "center", fontsize=9)
    ax.set_title(ablation_label, fontsize=10)
    if col == 0:
        ax.set_ylabel("Steps to full convergence")
        ax.legend(fontsize=8, loc="upper left")
    ax.text(x_pos[-1], max_steps * 1.01, "did not\nconverge", ha="center",
            va="bottom", fontsize=6.5, color="#888888")

fig.suptitle("Ablation study — steps to convergence (15 000 cap = did not converge)",
             fontsize=10)
fig.tight_layout()
save(fig, "ablations_combined.svg")
print("Saved: ablations_combined.svg")

print(f"\nAll outputs in {OUT_DIR}/")
