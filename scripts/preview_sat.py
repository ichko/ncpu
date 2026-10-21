#!/usr/bin/env python3
"""Eyeball the 3SAT verify task before (and after) training — no model needed.

Renders what ``ncpu.sat.SATLayout`` actually hands the NCA: the input screen
(formula + candidate assignment, flag column and verdict disc left blank), the
target screen (same, with the answer filled in), and the output masks the loss
and the decoder use.  It also prints the clause matrix as text with the
ground-truth flags, so the pixels can be checked against the semantics by eye.

Two numbers it exists to put in front of you before you burn a GPU-day:

* ``min_rollout()`` — the pixel depth of the dataflow.  Information moves ONE
  PIXEL per NCA step and the lattice pitch is ``2r + gap`` px, so this scales
  with ``--r``, not with n + m.  Below it the task is impossible, not hard.
  A global AND cannot cost less than the grid diameter, since some signal must
  cross the whole height; ``--r 1`` halves the pitch and so halves the depth.
* the constant-predictor baselines of the sampled batch.  At the phase ratio
  ~95% of clauses are satisfied, so "always satisfied" already scores ~0.95 on
  flags; a flag accuracy read without its baseline is actively misleading.

Pass ``--checkpoint`` to add the model's last frame, the error map and a rollout
GIF — the same figure then doubles as a post-training sanity check.

Outputs (project root by default):
    sat_verify_preview.png    geometry, masks, input/target (+ model, if given)
    sat_verify_rollout.gif    channel-0 rollout          (only with --checkpoint)

Usage:
    uv run python scripts/preview_sat.py
    uv run python scripts/preview_sat.py --r 1
    uv run python scripts/preview_sat.py --checkpoint runs/<run>/checkpoints/nca_last.pt
"""

import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ncpu.gif import write_gif
from ncpu.nca import NeuralCA
from ncpu.sat import PHASE_RATIO, SATLayout, clause_flags, evaluate

parser = ArgumentParser()
parser.add_argument("--n-vars", type=int, default=6)
parser.add_argument("--m-clauses", type=int, default=26, help="default is the phase ratio")
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--r", type=int, default=2, help="disc radius; pitch and depth scale with it")
parser.add_argument("--gap", type=int, default=None)
parser.add_argument("--pad", type=int, default=None)
parser.add_argument("--samples", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sat-frac", type=float, default=0.5)
parser.add_argument("--near-miss-frac", type=float, default=0.5)
parser.add_argument("--checkpoint", default=None, help="a nca_*.pt to run on these instances")
parser.add_argument("--rollout", type=int, default=None,
                    help="default: 1.5 x layout.min_rollout()")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--fps", type=int, default=20)
parser.add_argument("--zoom", type=int, default=2, help="gif upscale factor")
parser.add_argument("--out-dir", default=str(ROOT))
args = parser.parse_args()


def text_matrix(layout, formula, assignment):
    """The clause matrix as characters, next to the ground-truth flags."""
    lines = ["      " + " ".join(f"v{j}"[:2].rjust(2) for j in range(layout.n)) + "  | flag"]
    lines.append("  a:  " + " ".join(" +" if b else " -" for b in assignment)
                 + f"  | {'SAT' if evaluate(formula, assignment) else 'UNSAT'}  <- verdict")
    flags = clause_flags(formula, assignment)
    for i, clause in enumerate(formula):
        row = ["."] * layout.n
        for lit in clause:
            row[abs(lit) - 1] = "+" if lit > 0 else "-"
        lines.append(f" c{i}:".ljust(6) + " ".join(c.rjust(2) for c in row) + f"  | {flags[i]}")
    return "\n".join(lines)


RNG = random.Random(args.seed)   # its own stream, as the trainer and eval each have
np.random.seed(args.seed)
torch.manual_seed(args.seed)

layout = SATLayout(n_vars=args.n_vars, m_clauses=args.m_clauses, k=args.k,
                   r=args.r, gap=args.gap, pad=args.pad)
depth = layout.min_rollout()
steps = args.rollout if args.rollout is not None else int(round(1.5 * depth))

print(layout.describe())
print(f"min_rollout {depth} px steps = (n+m+1)*step at pitch {layout.step} px/cell — two "
      f"concurrent critical paths of equal cost: last clause  down {layout.m}*step + across "
      f"{layout.n}*step + 1*step to the verdict;  first clause  down 1*step + across "
      f"{layout.n}*step + flag falls {layout.m}*step down the column")
print(f"suggested training rollout {int(round(1.5 * depth))}..{int(round(2.5 * depth))}; "
      f"this preview uses {steps}")
if args.r > 1:
    fine = SATLayout(layout.n, layout.m, k=layout.k, r=1)
    print(f"  (--r 1 would give {fine.W}x{fine.H} and min_rollout {fine.min_rollout()} — "
          f"the pitch, not the rule, is what sets the cost)")
print(f"output bits: {layout.m} flags + 1 verdict = {layout.m + 1}   "
      f"ratio {layout.m / layout.n:.2f} vs phase {PHASE_RATIO}")

inp, tgt, bits, meta = layout.sample_batch(args.samples, mode="verify", rng=RNG,
                                           sat_frac=args.sat_frac,
                                           near_miss_frac=args.near_miss_frac)
masks = layout.output_bit_masks("verify")            # (m + 1, H, W)

flag_ones = bits[:, :-1].float().mean().item()
sat_rate = bits[:, -1].float().mean().item()
print(f"\nsampled {args.samples} instances: true verdicts {[int(b) for b in bits[:, -1]]}")
print(f"constant-predictor baselines on this batch: "
      f"flag {max(flag_ones, 1 - flag_ones):.3f}  verdict {max(sat_rate, 1 - sat_rate):.3f}  "
      f"(satisfied clauses {flag_ones:.3f}, SAT rate {sat_rate:.3f})")
print("Tier 0 reference at n=6 m=26 (results/karp/tier0_ratio433.json): "
      "verdict 0.501, exact 0.501, flags 0.950 — beat the FLAG baseline, not 0.5.")
print("\nsample 0 — clause matrix, assignment and ground truth:")
print(text_matrix(layout, *meta[0]))

# decoding the target with the shared decoder must reproduce the reference
for i, (formula, assignment) in enumerate(meta):
    want = clause_flags(formula, assignment) + [evaluate(formula, assignment)]
    got = [int(b) for b in bits[i]]
    assert got == want, f"decoder disagrees with reference on sample {i}: {got} != {want}"
print("\ndecoder round-trip on the target screens: OK")

# ── optional: run a checkpoint on the same instances ─────────────────────────
model_last = None
rollout = None
if args.checkpoint:
    ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    nca = NeuralCA(**ck["nca_config"]).to(args.device)
    nca.load_state_dict(ck["nca"])
    nca.eval()
    ck_layout = ck.get("layout", {})
    if (ck_layout.get("n"), ck_layout.get("m")) != (layout.n, layout.m):
        print(f"note: checkpoint was trained at n={ck_layout.get('n')} m={ck_layout.get('m')}, "
              f"running it at n={layout.n} m={layout.m} (the rule is convolutional — "
              f"see eval_sat_extrap.py)")
    if ck_layout.get("r") not in (None, layout.r):
        print(f"WARNING: checkpoint was trained at r={ck_layout.get('r')} (pitch "
              f"{2 * ck_layout['r'] + ck_layout.get('gap', ck_layout['r'])} px) but this "
              f"preview uses r={layout.r} — the rule has never seen this pitch.")
    if steps < depth:
        print(f"WARNING: rollout {steps} < min_rollout {depth}: the answer cannot arrive.")
    C = ck["nca_config"]["channels"]
    x = torch.zeros(args.samples, C, layout.H, layout.W, device=args.device)
    x[:, 0] = inp.to(args.device)
    x[:, 1] = inp.to(args.device)
    stride = max(1, steps // 120)
    frames = []
    with torch.no_grad():
        for t in range(steps):
            x = nca.forward(x, steps=1)[:, -1]
            if t % stride == 0:
                frames.append(x[:, 0].cpu())
    rollout = torch.stack(frames, dim=1)                        # (B, T', H, W)
    model_last = x[:, 0].cpu()
    got = SATLayout.decode_bits(model_last, masks)
    correct = got == bits
    print(f"\ncheckpoint step {ck.get('step')} at rollout {steps}: "
          f"flag {correct[:, :-1].float().mean():.3f} (baseline {max(flag_ones, 1 - flag_ones):.3f})  "
          f"verdict {correct[:, -1].float().mean():.3f} (baseline {max(sat_rate, 1 - sat_rate):.3f})  "
          f"exact {correct.all(1).float().mean():.3f}")

# ── figure ───────────────────────────────────────────────────────────────────
rows = [("input", inp), ("target", tgt), ("output masks", masks.sum(0)[None].expand_as(inp))]
if model_last is not None:
    rows += [("nca last", model_last), ("|diff|", (model_last - tgt).abs())]

n = args.samples
fig, axes = plt.subplots(len(rows), n, figsize=(1.9 * n + 1.2, 2.1 * len(rows)), squeeze=False)
for r, (name, data) in enumerate(rows):
    for c in range(n):
        ax = axes[r][c]
        vmin, vmax = (0, 2) if name == "|diff|" else (-1, 1)
        if name == "output masks":
            vmin, vmax = 0, 1
        ax.imshow(data[c], cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        if c == 0:
            ax.set_ylabel(name, fontsize=9)
        if r == 0:
            ax.set_title(f"#{c}  {'SAT' if bits[c, -1] else 'UNSAT'}", fontsize=9)
fig.suptitle(f"3SAT verify  n={layout.n} m={layout.m} k={layout.k} r={layout.r}  "
             f"{layout.W}x{layout.H}  min_rollout {depth}", fontsize=11)
fig.tight_layout()
png_path = Path(args.out_dir) / "sat_verify_preview.png"
fig.savefig(png_path, dpi=120)
plt.close(fig)
print(f"\nwrote {png_path}")

if rollout is not None:
    tiled = np.concatenate([rollout[i].numpy() for i in range(min(3, n))], axis=2)
    gif_path = Path(args.out_dir) / "sat_verify_rollout.gif"
    write_gif(gif_path, tiled, fps=args.fps, vmin=-1, vmax=1, zoom=args.zoom)
    print(f"wrote {gif_path}")
