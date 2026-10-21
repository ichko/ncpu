#!/usr/bin/env python3
"""Find where an orbit-counter NCA hides its loop phase.

Run 1 counted laps correctly while rendering no comet at all in channel 0, which
means the phase must live in the unsupervised hidden channels.  This script
checks that directly: it rolls a checkpoint out and, for every channel, reduces
the track region to one scalar per frame and asks whether that signal oscillates
at the lap period.

Outputs (project root by default):
    orbit_channels.png   per-channel montage over time + per-channel signals
    orbit_channels.json  per-channel std / autocorrelation-at-lap-period

Usage:
    uv run python scripts/inspect_orbit.py runs/<run>/checkpoints/nca_last.pt
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA
from ncpu.orbit import OrbitLayout

parser = ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--device", default="cpu")
parser.add_argument("--steps", type=int, default=420, help="rollout length (>= 2 laps)")
parser.add_argument("--p0", type=int, default=0, help="comet start arc position")
parser.add_argument("--c0", type=int, default=0)
parser.add_argument("--out-dir", default=".")
args = parser.parse_args()

ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
lay = ck["layout"]
layout = OrbitLayout(
    W=lay["W"], H=lay["H"], r=lay["r"], marks=lay["marks"],
    mark_step=lay["mark_step"], counter=lay["counter"], tail=lay["tail"],
    speed=lay["speed"], tap=lay["tap"], settle=lay["settle"],
)
nca = NeuralCA(**ck["nca_config"]).to(args.device)
nca.load_state_dict(ck["nca"])
nca.eval()

print(f"checkpoint step {ck['step']}  {layout.describe()}")

C = ck["nca_config"]["channels"]
track = layout.track_mask().to(args.device)
n_track = track.sum()

# Deterministic single sample so the phase is known exactly.
arc, val, cmask, true_arc = layout.schedule([args.p0], [args.c0], args.steps)
tgt = (torch.from_numpy(layout.bg)[None] + torch.from_numpy(layout.token_delta)[arc[0]]
       + torch.from_numpy(layout.reg_delta)[val[0]]).to(args.device)

state = torch.zeros(1, C, layout.H, layout.W, device=args.device)
state[:, 0] = tgt[0]
state[:, 1] = torch.from_numpy(layout.bg).to(args.device)

with torch.no_grad():
    rollout = nca.forward(state, steps=args.steps)          # (1, T, C, H, W)
seq = rollout[0]                                            # (T, C, H, W)
T = seq.shape[0]

# ── per-channel scalar signal over the track region ─────────────────────────
lap = layout.lap_steps
sig = (seq * track).sum(dim=(-1, -2)) / n_track              # (T, C)
sig = sig.cpu().numpy()

# Also track the spatial spread of activity: a real travelling pulse shows up as
# a moving centre of mass, not just a global oscillation.
ys, xs = np.mgrid[: layout.H, : layout.W]
tr = track.cpu().numpy()


def autocorr_at(x, lag):
    x = x - x.mean()
    if x.std() < 1e-9 or lag >= len(x):
        return 0.0
    a, b = x[:-lag], x[lag:]
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


stats = []
for c in range(C):
    s = sig[:, c]
    stats.append({
        "channel": c,
        "std": float(s.std()),
        "range": float(s.max() - s.min()),
        "autocorr_at_lap": autocorr_at(s, int(round(lap))),
        "autocorr_at_half_lap": autocorr_at(s, int(round(lap / 2))),
    })

order = sorted(stats, key=lambda d: -abs(d["autocorr_at_lap"]) * d["std"])
print(f"\nlap = {lap:.0f} steps; channels ranked by oscillation at the lap period:")
for d in order[:6]:
    print(f"  ch{d['channel']:<3} std={d['std']:.4f}  range={d['range']:.4f}  "
          f"autocorr@lap={d['autocorr_at_lap']:+.3f}  @half={d['autocorr_at_half_lap']:+.3f}")

# ── position-sensitive readout ───────────────────────────────────────────────
# The spatial mean above is BLIND to a travelling pulse: averaging a small bump
# over the whole corridor gives nearly the same number wherever the bump sits.
# So decode a per-channel arc position instead and compare it to ground truth.
head = torch.from_numpy(layout.head_bool).to(args.device).reshape(layout.L, -1)
head_n = head.sum(1)
true_t = torch.from_numpy(true_arc[0]).to(args.device)
chance = layout.L / 4.0

print(f"\nper-channel arc tracking (chance = {chance:.1f} px):")
for c in range(C):
    frames = seq[:, c]                                        # (T, H, W)
    corridor_mean = (frames * track).sum(dim=(-1, -2)) / n_track
    dev = frames - corridor_mean[:, None, None]               # remove the DC term
    scores = (dev.reshape(T, -1) @ head.t()) / head_n         # (T, L)
    err_hi = layout.arc_error(scores.argmax(1), true_t).mean().item()
    err_lo = layout.arc_error(scores.argmin(1), true_t).mean().item()
    polarity = "bright" if err_hi <= err_lo else "dark"
    err = min(err_hi, err_lo)
    stats[c]["arc_err_px"] = err
    stats[c]["polarity"] = polarity
    flag = "  <-- TRACKS THE COMET" if err < 0.4 * chance else ""
    print(f"  ch{c:<3} arc_err={err:6.2f}px  ({polarity}){flag}")

best = min(stats, key=lambda d: d["arc_err_px"])
print(f"\nbest tracking channel: ch{best['channel']} at {best['arc_err_px']:.2f}px "
      f"(chance {chance:.1f}px)")

out_dir = Path(args.out_dir)
with open(out_dir / "orbit_channels.json", "w") as f:
    json.dump({"step": ck["step"], "lap_steps": lap, "channels": stats}, f, indent=2)

# ── figure: montage of every channel at a few times, + signal traces ────────
ts = np.linspace(0, T - 1, 6).astype(int)
fig = plt.figure(figsize=(2.0 * len(ts) + 6, 1.9 * C))
gs = fig.add_gridspec(C, len(ts) + 3)

for c in range(C):
    row = seq[:, c].cpu().numpy()
    lo, hi = float(row.min()), float(row.max())
    for j, t in enumerate(ts):
        ax = fig.add_subplot(gs[c, j])
        ax.imshow(row[t], cmap="viridis", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_title(f"t={t}", fontsize=8)
        if j == 0:
            ax.set_ylabel(f"ch{c}", fontsize=8)
    ax = fig.add_subplot(gs[c, len(ts):])
    ax.plot(sig[:, c], lw=0.8)
    for k in range(1, int(T / lap) + 1):
        ax.axvline(k * lap, color="tab:red", lw=0.6, alpha=0.6)
    ax.set_xlim(0, T - 1); ax.tick_params(labelsize=6)
    d = stats[c]
    ax.set_title(f"ch{c}  std={d['std']:.3f}  ac@lap={d['autocorr_at_lap']:+.2f}",
                 fontsize=7)

fig.suptitle(f"orbit channels — step {ck['step']}, lap={lap:.0f} steps "
             f"(red lines = lap boundaries)", fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(out_dir / "orbit_channels.png", dpi=90)
plt.close(fig)
print(f"\nwrote {out_dir / 'orbit_channels.png'} and {out_dir / 'orbit_channels.json'}")
