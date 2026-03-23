#!/usr/bin/env python3
"""
8-bit ALU NCA analysis (E4).

Usage:
    uv run python scripts/analyze_alu.py

Outputs under results/E4/:
    summary.csv                 — per-run metrics
    loss_curve.pdf              — training loss (mean ± std across seeds)
    bits_curve.pdf              — valid bits over training
    opcode_accuracy.pdf         — per-opcode bit accuracy at final checkpoint
    rollout_all_opcodes.pdf     — channel 0 over time, one row per opcode
    snapshots_grid.pdf          — final snapshot per run
    layout_diagram.pdf          — ALU grid layout schematic
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.dataset import _compute_alu, _int_to_bits, ALUDataset
from ncpu.utils import make_alu_screen

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results/E4")
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

OPCODE_NAMES  = ["ADD", "SUB", "AND", "OR", "XOR", "NOT", "SHL", "SHR"]
OPCODE_COLORS = [
    "#4E79A7",  # ADD  — blue
    "#F28E2B",  # SUB  — orange
    "#E15759",  # AND  — red
    "#76B7B2",  # OR   — teal
    "#59A14F",  # XOR  — green
    "#EDC948",  # NOT  — yellow
    "#B07AA1",  # SHL  — purple
    "#FF9DA7",  # SHR  — salmon
]

# ── Find runs ─────────────────────────────────────────────────────────────────

def find_runs():
    runs = []
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get("run", {}).get("experiment") != "E4":
            continue
        runs.append(d)
    return runs

runs = find_runs()
print(f"Found {len(runs)} E4 run(s):")
for d in runs:
    cfg  = json.load(open(d / "config.json"))
    log  = [json.loads(l) for l in open(d / "log.jsonl")]
    last = log[-1]
    print(f"  {d.name}  seed={cfg['run']['seed']}  "
          f"step={last['step']}  loss={last['loss']:.4f}  bits={last['num_valid_bits']:.2f}/8")

if not runs:
    print("No E4 runs found.")
    sys.exit(0)

# ── Load logs ─────────────────────────────────────────────────────────────────

def load_log(run_dir):
    return [json.loads(l) for l in open(run_dir / "log.jsonl")]

logs = [load_log(d) for d in runs]

# ── Summary CSV ───────────────────────────────────────────────────────────────

with open(OUT_DIR / "summary.csv", "w") as f:
    f.write("run,seed,final_step,final_loss,valid_bits\n")
    for d, log in zip(runs, logs):
        cfg  = json.load(open(d / "config.json"))
        last = log[-1]
        f.write(f"{d.name},{cfg['run']['seed']},{last['step']},"
                f"{last['loss']:.6f},{last['num_valid_bits']:.3f}\n")
print("Saved: summary.csv")

# ── Helpers ───────────────────────────────────────────────────────────────────

def align_series(logs_list, key, n_steps, smooth=50):
    common = np.arange(n_steps)
    arrays = []
    for log in logs_list:
        steps = np.array([e["step"] for e in log])
        vals  = np.array([e[key]   for e in log])
        arrays.append(np.interp(common, steps, vals))
    arr  = np.array(arrays)
    mean = arr.mean(axis=0)
    std  = arr.std(axis=0)
    if smooth > 1:
        k    = np.ones(smooth) / smooth
        mean = np.convolve(mean, k, mode="valid")
        std  = np.convolve(std,  k, mode="valid")
        x    = common[smooth - 1:]
    else:
        x = common
    return x, mean, std

max_steps = max(log[-1]["step"] + 1 for log in logs)

# ── Loss curve ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
x, mean, std = align_series(logs, "loss", max_steps)
ax.plot(x, mean, color="#4E79A7", linewidth=1.5)
ax.fill_between(x, mean - std, mean + std, color="#4E79A7", alpha=0.15)
ax.set_yscale("log")
ax.set_xlabel("Training step")
ax.set_ylabel("Loss (masked MSE)")
ax.set_title(f"E4 — 8-bit ALU: training loss (mean ± std, {len(runs)} run(s))")
fig.tight_layout()
fig.savefig(OUT_DIR / "loss_curve.pdf")
plt.close(fig)
print("Saved: loss_curve.pdf")

# ── Bits curve ────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 4))
x, mean, std = align_series(logs, "num_valid_bits", max_steps)
ax.plot(x, mean, color="#F28E2B", linewidth=1.5)
ax.fill_between(x, mean - std, mean + std, color="#F28E2B", alpha=0.15)
ax.axhline(8, color="#aaaaaa", linewidth=0.8, linestyle="--", label="max (8 bits)")
ax.set_ylim(0, 8.8)
ax.set_xlabel("Training step")
ax.set_ylabel("Valid bits / 8")
ax.set_title(f"E4 — 8-bit ALU: valid bits over training")
ax.legend()
fig.tight_layout()
fig.savefig(OUT_DIR / "bits_curve.pdf")
plt.close(fig)
print("Saved: bits_curve.pdf")

# ── NCA loading helpers ───────────────────────────────────────────────────────

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
    ckpts = sorted((run_dir / "checkpoints").glob("nca_*.pt"))
    nca.load_state_dict(torch.load(ckpts[-1], map_location="cpu", weights_only=True))
    nca.eval()
    return nca


def make_state(cfg, a_bits, b_bits, opcode_bits):
    ds  = cfg["ds"]
    img = torch.from_numpy(
        make_alu_screen(ds["H"], ds["W"], ds["r"], ds["spacing"],
                        a_bits=a_bits, b_bits=b_bits, opcode_bits=opcode_bits)
    ).float() / 128.0 - 1.0
    state = torch.zeros(1, cfg["nca"]["channels"], ds["H"], ds["W"])
    state[0, 0] = img
    state[0, 1] = img   # channel 1 = read-only copy
    return state


def get_bit_masks(cfg):
    from argparse import Namespace
    return ALUDataset(Namespace(**cfg["ds"])).get_output_bit_masks()  # (8, H, W)

# ── Per-opcode accuracy evaluation ───────────────────────────────────────────

N_EVAL     = 32    # random samples per opcode
EVAL_STEPS = 128   # NCA steps
EVAL_BATCH = 4     # samples per batched NCA call

def eval_opcode_accuracy(nca, cfg, bit_masks):
    """Returns (8,) array of mean bit accuracy per opcode, printing progress."""
    from tqdm import tqdm
    acc = np.zeros(8)
    with torch.no_grad():
        pbar = tqdm(range(8), desc="eval opcodes", ncols=70)
        for op in pbar:
            name   = OPCODE_NAMES[op]
            uses_b = op < 5
            pbar.set_description(f"eval {name:<6}")
            hit = 0
            # Pre-generate all samples for this opcode
            a_ints  = torch.randint(0, 256, (N_EVAL,)).tolist()
            b_ints  = torch.randint(0, 256, (N_EVAL,)).tolist()
            results = [_compute_alu(a, b, op) for a, b in zip(a_ints, b_ints)]
            # Run in batches
            for start in range(0, N_EVAL, EVAL_BATCH):
                end    = min(start + EVAL_BATCH, N_EVAL)
                batch  = torch.cat([
                    make_state(cfg,
                               _int_to_bits(a_ints[i], 8),
                               _int_to_bits(b_ints[i], 8) if uses_b else None,
                               _int_to_bits(op, 3))
                    for i in range(start, end)
                ], dim=0).to(DEVICE)   # (B, C, H, W)
                rollout = nca.forward(batch, steps=EVAL_STEPS)   # (B, T+1, C, H, W)
                final   = rollout[:, -1, 0]                        # (B, H, W)
                for i, idx in enumerate(range(start, end)):
                    result_bits = _int_to_bits(results[idx], 8)
                    for bit_i in range(8):
                        mask = bit_masks[bit_i]   # (H, W)
                        pred = (final[i] * mask).sum() / mask.sum()
                        tgt  = 1.0 if result_bits[bit_i] else -1.0
                        hit += int((pred.item() > 0) == (tgt > 0))
            acc[op] = hit / (N_EVAL * 8)
            pbar.write(f"  {name:<6}  {acc[op]*100:5.1f}%")
    return acc

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Eval device: {DEVICE}")

best_run = min(runs, key=lambda d: json.loads(open(d / "log.jsonl").readlines()[-1])["loss"])
cfg      = json.load(open(best_run / "config.json"))
print(f"\nEvaluating per-opcode accuracy on {best_run.name} ...")
nca       = load_nca(best_run).to(DEVICE)
bit_masks = get_bit_masks(cfg).to(DEVICE)
acc       = eval_opcode_accuracy(nca, cfg, bit_masks)

print("\nPer-opcode bit accuracy:")
for op, name in enumerate(OPCODE_NAMES):
    print(f"  {name:<6}  {acc[op]*100:5.1f}%")

# ── Opcode accuracy bar chart ─────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(OPCODE_NAMES, acc * 100, color=OPCODE_COLORS, width=0.6, zorder=3)
ax.axhline(100, color="#aaaaaa", linewidth=0.8, linestyle="--")
for bar, v in zip(bars, acc * 100):
    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8, f"{v:.0f}%",
            ha="center", va="bottom", fontsize=8, color="#444444")
ax.set_ylim(0, 115)
ax.set_ylabel("Mean bit accuracy (%)")
ax.set_title(f"E4 — 8-bit ALU: per-opcode bit accuracy at final checkpoint\n"
             f"({N_EVAL} random samples per opcode, {EVAL_STEPS} NCA steps)")
fig.tight_layout()
fig.savefig(OUT_DIR / "opcode_accuracy.pdf")
plt.close(fig)
print("Saved: opcode_accuracy.pdf")

# ── Rollout: all opcodes × timesteps ─────────────────────────────────────────

TSTEPS = [0, 16, 32, 64, 96, 128]

# One representative example per opcode (clean bit patterns)
SHOWCASE = [
    (0, "ADD",  0b01100100, 0b00101110),  # 100 + 46  = 146
    (1, "SUB",  0b11001000, 0b00111000),  # 200 - 56  = 144
    (2, "AND",  0b11001100, 0b10101010),  # 0xCC & 0xAA = 0x88
    (3, "OR",   0b11001100, 0b10101010),  # 0xCC | 0xAA = 0xEE
    (4, "XOR",  0b11110000, 0b10101010),  # 0xF0 ^ 0xAA = 0x5A
    (5, "NOT",  0b11001100, 0),           # ~0xCC = 0x33
    (6, "SHL",  0b01010101, 0),           # 0x55 << 1 = 0xAA
    (7, "SHR",  0b10101010, 0),           # 0xAA >> 1 = 0x55
]

ds     = cfg["ds"]
n_rows = len(SHOWCASE)
n_cols = len(TSTEPS) + 1   # +1 for target
cell   = 0.72
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(n_cols * cell, n_rows * cell),
                          gridspec_kw={"hspace": 0.02, "wspace": 0.02},
                          constrained_layout=True)

with torch.no_grad():
    for ri, (op, name, a_int, b_int) in enumerate(SHOWCASE):
        result = _compute_alu(a_int, b_int, op)
        uses_b = op < 5
        state  = make_state(cfg,
                            _int_to_bits(a_int, 8),
                            _int_to_bits(b_int, 8) if uses_b else None,
                            _int_to_bits(op, 3)).to(DEVICE)
        rollout = nca.forward(state, steps=max(TSTEPS))[0].cpu()  # (T+1, C, H, W)

        for ci, t in enumerate(TSTEPS):
            t_idx = min(t, rollout.shape[0] - 1)
            ax    = axes[ri][ci]
            ax.imshow(rollout[t_idx, 0].numpy(), cmap="viridis",
                      vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
            ax.set_xticks([]); ax.set_yticks([])
            if ri == 0:
                ax.set_title(f"t={t}", fontsize=8, pad=2)
            if ci == 0:
                lbl = (f"{name}  "
                       f"{a_int:#04x}{'op'+format(b_int,'#04x') if uses_b else ''}"
                       f"={result:#04x}")
                ax.set_ylabel(lbl, fontsize=5.5)

        # target frame
        target_img = torch.from_numpy(
            make_alu_screen(ds["H"], ds["W"], ds["r"], ds["spacing"],
                            result_bits=_int_to_bits(result, 8))
        ).float() / 128.0 - 1.0
        ax = axes[ri][n_cols - 1]
        ax.imshow(target_img.numpy(), cmap="viridis", vmin=-1, vmax=1,
                  interpolation="nearest", rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title("target", fontsize=8, pad=2)

fig.suptitle("E4 — 8-bit ALU: channel 0 over time (one example per opcode)", fontsize=9)
fig.savefig(OUT_DIR / "rollout_all_opcodes.pdf", dpi=150, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("Saved: rollout_all_opcodes.pdf")

# ── Snapshots grid ────────────────────────────────────────────────────────────

snaps = [(d, d / "snapshot_latest.png") for d in runs if (d / "snapshot_latest.png").exists()]
if snaps:
    n = len(snaps)
    fig, axes = plt.subplots(1, n, figsize=(3 * n, 3),
                              gridspec_kw={"wspace": 0.04})
    if n == 1:
        axes = [axes]
    for ax, (d, snap) in zip(axes, snaps):
        ax.imshow(Image.open(snap), rasterized=True)
        seed = json.load(open(d / "config.json"))["run"]["seed"]
        last = load_log(d)[-1]
        ax.set_title(f"seed={seed}  step={last['step']}", fontsize=8)
        ax.axis("off")
    fig.suptitle("E4 — 8-bit ALU: final snapshots", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "snapshots_grid.pdf", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: snapshots_grid.pdf")

# ── Layout diagram ────────────────────────────────────────────────────────────
# Schematic showing the three-column ALU grid layout at true pixel scale.

H_D, W_D  = 112, 128
r_d       = 4
among_sp  = 2
side_sp   = 20

COL_A   = "#4E79A7"   # blue   — A input
COL_B   = "#76B7B2"   # teal   — B input
COL_OP  = "#F28E2B"   # orange — opcode
COL_RES = "#59A14F"   # green  — result
COL_BG  = "#f7f7f7"

def _draw_col_diag(ax, n, cx, color, labels, fontsize=5):
    v  = n * 2*r_d + among_sp * (n - 1)
    tm = (H_D - v) // 2
    for i, lbl in enumerate(labels):
        cy = tm + r_d + i * (2*r_d + among_sp)
        ax.add_patch(mpatches.Circle((cx, cy), r_d, color=color, zorder=3))
        ax.text(cx, cy, lbl, ha="center", va="center",
                fontsize=fontsize, color="white", fontweight="bold", zorder=4)

scale  = 0.028
p_w    = W_D * scale    # 3.584"
p_h    = H_D * scale    # 3.136"
lm, rm = 0.15, 0.15
tm, bm = 0.45, 0.45
fig_w  = p_w + lm + rm
fig_h  = p_h + tm + bm

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
fig.subplots_adjust(left=lm/fig_w, right=1-rm/fig_w,
                    bottom=bm/fig_h, top=1-tm/fig_h)

ax.set_xlim(0, W_D)
ax.set_ylim(H_D, 0)
ax.set_aspect("equal")
ax.set_facecolor(COL_BG)
ax.add_patch(mpatches.FancyBboxPatch((0, 0), W_D, H_D, boxstyle="square,pad=0",
             linewidth=1, edgecolor="#aaaaaa", facecolor=COL_BG, zorder=0))
ax.set_xticks([]); ax.set_yticks([])

# A column (cx = side_sp)
cx_a = side_sp
_draw_col_diag(ax, 8, cx_a, COL_A, [f"a{i}" for i in range(8)])

# B column (cx = side_sp + 2r + among_sp)
cx_b = side_sp + 2*r_d + among_sp
_draw_col_diag(ax, 8, cx_b, COL_B, [f"b{i}" for i in range(8)])

# Opcode column (cx = W//2)
cx_op = W_D // 2
_draw_col_diag(ax, 3, cx_op, COL_OP, [f"op{i}" for i in range(3)])

# Result column (cx = W - side_sp)
cx_res = W_D - side_sp
_draw_col_diag(ax, 8, cx_res, COL_RES, [f"r{i}" for i in range(8)])

# Column labels
label_y = H_D + 4
for cx, lbl, col in [
    (cx_a,  "A",      COL_A),
    (cx_b,  "B",      COL_B),
    (cx_op, "opcode", COL_OP),
    (cx_res,"result", COL_RES),
]:
    ax.text(cx, label_y, lbl, ha="center", va="top",
            fontsize=6, color=col, fontweight="bold")

handles = [
    mpatches.Patch(color=COL_A,   label="A bits (r/o ch1)"),
    mpatches.Patch(color=COL_B,   label="B bits (r/o ch1)"),
    mpatches.Patch(color=COL_OP,  label="Opcode"),
    mpatches.Patch(color=COL_RES, label="Result"),
]
fig.legend(handles=handles, loc="lower center", fontsize=7, ncol=4,
           framealpha=0.9, edgecolor="#cccccc", bbox_to_anchor=(0.5, 0.02))

fig.suptitle(f"E4 — 8-bit ALU layout ({W_D}×{H_D} grid, r={r_d})", fontsize=9, y=0.99)
fig.savefig(OUT_DIR / "layout_diagram.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("Saved: layout_diagram.pdf")

print(f"\nAll outputs in {OUT_DIR}/")
