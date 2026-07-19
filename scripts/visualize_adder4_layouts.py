#!/usr/bin/env python3
"""
4-bit adder rollout comparison: cols1 vs cols2 layouts on the same inputs.

Produces a single figure with 4 rows of channel-0 rollouts:
    rows 0-1 : cols1 (single column input layout)
    rows 2-3 : cols2 (two column input layout)

Same two input combinations are used for both layouts so they can be compared.

Usage:
    uv run python scripts/visualize_adder4_layouts.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.utils import make_io_screen, make_io_screen_cols1

RUNS_DIR = Path("runs")
OUT_DIR  = Path("results/E2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DISPLAY_TS = [0, 8, 16, 24, 32, 64]
COMBOS = [
    (3,  5,  "3+5"),
    (15, 1,  "15+1"),
]
N_RUNS = 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def find_runs(layout):
    out = []
    for d in sorted(RUNS_DIR.iterdir()):
        cfg_path = d / "config.json"
        if not cfg_path.exists() or not (d / "log.jsonl").exists():
            continue
        cfg = json.load(open(cfg_path))
        if cfg.get("run", {}).get("experiment") != "E2":
            continue
        screen_fn = cfg["ds"].get("screen_fn", "")
        is_cols2 = screen_fn == "make_io_screen"
        is_cols1 = screen_fn == "make_io_screen_cols1"
        if (layout == "cols1" and is_cols1) or (layout == "cols2" and is_cols2):
            out.append(d)
    return out


def best_run(layout):
    runs = find_runs(layout)
    return min(runs,
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
    ).to(DEVICE)
    last = sorted((run_dir / "checkpoints").glob("nca_*.pt"))[-1]
    nca.load_state_dict(torch.load(last, map_location=DEVICE, weights_only=True), strict=False)
    nca.eval()
    nca.gaussian_noise.prob = 0.0
    return nca


def int_to_bits(n, w): return [int(b) for b in f"{n:0{w}b}"]


def make_state(layout, ds_cfg, a, b):
    bits = int_to_bits(a, 4) + int_to_bits(b, 4)
    fn = make_io_screen_cols1 if layout == "cols1" else make_io_screen
    screen = fn(H=ds_cfg["H"], W=ds_cfg["W"], r=ds_cfg["r"],
                spacing=ds_cfg["spacing"],
                left_input=bits, right_input=[])
    img = torch.from_numpy(screen).float() / 128.0 - 1.0
    state = torch.zeros(1, 16, ds_cfg["H"], ds_cfg["W"], device=DEVICE)
    state[0, 0] = img.to(DEVICE)
    state[0, 1] = img.to(DEVICE)
    return state


def rollout(nca, state, max_t):
    state_b = state.expand(N_RUNS, -1, -1, -1).contiguous()
    with torch.no_grad():
        r = nca.forward(state_b, steps=max_t)
    return r.mean(dim=0).cpu()  # (T+1, C, H, W)


# ── Load both layouts ─────────────────────────────────────────────────────────

models = {}
for layout in ["cols1", "cols2"]:
    rd = best_run(layout)
    print(f"  {layout}: {rd.name}")
    models[layout] = (load_nca(rd), json.load(open(rd / "config.json"))["ds"])

max_t = max(DISPLAY_TS)

# ── Figure ────────────────────────────────────────────────────────────────────

H = next(iter(models.values()))[1]["H"]
W = next(iter(models.values()))[1]["W"]

n_rows = len(COMBOS) * 2
n_cols = len(DISPLAY_TS)
cell_w = 0.80
cell_h = cell_w * H / W
fig_w  = n_cols * cell_w
fig_h  = n_rows * cell_h

fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
fig.subplots_adjust(hspace=0.02 * cell_w / cell_h, wspace=0.02)

# Output circle geometry per layout (5 outputs for 4-bit adder)
def out_geom(ds_cfg):
    asp, ssp = int(ds_cfg["spacing"][0]), int(ds_cfg["spacing"][1])
    rr = int(ds_cfg["r"])
    n_out = 5
    v = n_out * rr * 2 + asp * (n_out - 1)
    tm = (ds_cfg["H"] - v) // 2
    return [(ds_cfg["W"] - ssp, tm + rr + i * (2 * rr + asp)) for i in range(n_out)], rr

row = 0
for layout in ["cols1", "cols2"]:
    nca, ds_cfg = models[layout]
    out_circles, rr = out_geom(ds_cfg)
    for (a, b, lbl) in COMBOS:
        state = make_state(layout, ds_cfg, a, b)
        roll  = rollout(nca, state, max_t)
        for col, t in enumerate(DISPLAY_TS):
            ax = axes[row, col]
            ax.imshow(roll[t, 0].numpy(), cmap="viridis",
                      vmin=-1, vmax=1, interpolation="nearest", rasterized=True)
            ax.axis("off")
            if row == 0:
                ax.set_title(f"t={t}", fontsize=11, pad=2, fontweight="bold")
            if col == 0:
                ax.text(-0.05, 0.5, f"{layout}\n{lbl}", transform=ax.transAxes,
                        fontsize=10, va="center", ha="right",
                        linespacing=1.15)
            if col == n_cols - 1:
                for (cx, cy) in out_circles:
                    ax.add_patch(mpatches.Circle((cx, cy), rr,
                        fill=False, edgecolor="#888888", linewidth=1.6))
                ax.text(1.03, 0.5, f"={a+b}", transform=ax.transAxes,
                        fontsize=10, va="center", ha="left")
        row += 1

out_svg = OUT_DIR / "rollout_combined_4bit.svg"
fig.savefig(out_svg, bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print(f"Saved {out_svg}")
