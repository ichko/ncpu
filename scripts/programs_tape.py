#!/usr/bin/env python3
"""Run real programs on a v15 tape-machine checkpoint.

The machine: 4 registers x 4 bits, SUB d = a - b and SUBLEQ (d = a - b only if that is
<= 0, signed), the program a ring executed forever with no jumps. Each program runs
from several start states, closed loop, against the software reference; prints the
first wrong cycle per start state and writes <out>.png (decoded registers over time,
model next to the reference).

    uv run python scripts/programs_tape.py --run runs/TAPE_... --ckpt nca_best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="nca_best.pt")
ap.add_argument("--cycles", type=int, default=2000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default="v15_programs")
opt = ap.parse_args()

# (op, a, b, d): d <- a - b   (SUBLEQ: only if a - b <= 0 as a signed 4-bit value)
NEG = lambda v: (-v) & 15
PROGRAMS = {
    # r0 = x, r1 = 1, r2 = -1, r3 = 0:  x <- x - (-1)
    "counter": ([("SUB", 3, 3, 3), ("SUB", 3, 1, 2), ("SUB", 0, 2, 0)],
                [(x, 1, 0, 0) for x in (0, 5, 9, 14)]),
    # r0, r1 = F(k), F(k+1); r2 temp; r3 = 0:  t = -F1; t = F0 - t; F0 = F1; F1 = t
    "fib": ([("SUB", 3, 1, 2), ("SUB", 0, 2, 2), ("SUB", 1, 3, 0), ("SUB", 2, 3, 1)],
            [(0, 1, 0, 0), (2, 7, 0, 0), (5, 3, 0, 0), (13, 11, 0, 0)]),
    # r0 = -a, r1 = -b (values kept negated so SUBLEQ can write them), r3 = 0:
    #   a > b: a -= b;  b >= a: b -= a.  Ends at a = 0, b = gcd.
    "gcd": ([("SUBLEQ", 0, 1, 0), ("SUBLEQ", 1, 0, 1), ("SUB", 3, 3, 3)],
            [(NEG(a), NEG(b), 0, 0) for a, b in ((6, 4), (7, 3), (5, 5), (6, 3))]),
    # r0 = x, r1 = 3, r2 / r3 temps:  x <- 5x + 3 mod 16 (full period 16)
    "lcg": ([("SUB", 2, 2, 2), ("SUB", 2, 0, 2), ("SUB", 0, 2, 3), ("SUB", 3, 2, 3),
             ("SUB", 3, 2, 3), ("SUB", 3, 2, 3), ("SUB", 2, 2, 2), ("SUB", 2, 1, 2),
             ("SUB", 3, 2, 0)],
            [(x, 3, 0, 0) for x in (0, 1, 7, 12)]),
}

run = Path(opt.run)
ck = torch.load(run / opt.ckpt, map_location=opt.device, weights_only=False)
cfg = ck["config"]
src = (ROOT / "scripts/train_tape.py").read_text()
src = src[:src.index("# ── model")]
sys.argv = ["train_tape.py"] + [a for k, v in cfg.items() if v is not None and v != ""
                                for a in (f"--{k.replace('_', '-')}", str(v))]
sys.argv += ["--device", opt.device]
T = {"__file__": str(ROOT / "scripts/train_tape.py"), "__name__": "programs"}
exec(compile(src, "train_tape.py", "exec"), T)
args, K = T["args"], T["K"]
nca = T["NeuralCA"](channels=args.channels, hidden_channels=args.hidden, fire_rate=args.fire_rate,
                    alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
                    perception="static_sobel", padding_type="circular", read_only_dims=[],
                    state_bound=10.0).to(opt.device)
nca.load_state_dict(ck["nca"])
T["apply_fire"](nca, cfg)
nca.eval()


def start(tape, regs):
    """The last instruction is loaded (and done): instruction 0 loads next."""
    n = len(tape)
    g = T["geom"](n, n - 1 - min(args.below, n - 1))
    j = n - 1
    ring = {(j + m) % n: (g.head - (m - 1)) % g.N for m in range(1, n)}
    return dict(g=g, nr=4, tape=list(tape), ring=ring, inmem=j, regs=tuple(regs))


def bits_to_vals(b):                                  # (B, regs, bits) -> (B, regs)
    w = 2 ** torch.arange(args.word_bits - 1, -1, -1, device=b.device)
    return (b * w).sum(-1)


res = {}
print(f"{run.name}/{opt.ckpt} step {ck['step']:,}: {opt.cycles} cycles\n")
for name, (tape, starts) in PROGRAMS.items():
    runs = [start(tape, r) for r in starts]
    x = T["start_state"](runs)
    B = len(runs)
    fail = np.full(B, opt.cycles + 1)
    model_v, ref_v = [], []
    with torch.no_grad():
        for c in range(1, opt.cycles + 1):
            runs, _, _, _ = T["cycle_targets"](runs)
            for _ in range(K):
                x = nca(x, 1)[:, -1]
            got, want = T["decode"](runs, x[:, 0]), T["want_bits"](runs)
            bad = (got != want).flatten(1).any(1).cpu().numpy()
            fail[bad & (fail > opt.cycles)] = c
            model_v.append(bits_to_vals(got).cpu().numpy())
            ref_v.append(bits_to_vals(want).cpu().numpy())
    res[name] = (fail, np.stack(model_v, 1), np.stack(ref_v, 1))
    fs = " ".join(f"{f if f <= opt.cycles else 'never':>6}" for f in fail)
    print(f"{name:>8} ({len(tape)} instrs)  first wrong cycle per start state: {fs}", flush=True)

# figure: per program, register 0..3 values over the first cycles, model vs reference
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

show = min(opt.cycles, 120)
fig, ax = plt.subplots(len(res), 2, figsize=(24, 3.2 * len(res)), constrained_layout=True)
for i, (name, (fail, mv, rv)) in enumerate(res.items()):
    for j, (v, lab) in enumerate(((mv, "model"), (rv, "reference"))):
        a = ax[i, j]
        img = v[:, :show].transpose(0, 2, 1).reshape(-1, show)       # (starts*regs, cycles)
        a.imshow(img, aspect="auto", cmap="viridis", vmin=0, vmax=15, interpolation="nearest")
        for k in range(1, v.shape[0]):
            a.axhline(k * 4 - 0.5, color="white", lw=2)
        a.set_yticks([])
        a.set_title(f"{name} — {lab} (rows: 4 registers × {v.shape[0]} start states)", loc="left")
        if j == 0:
            for k, f in enumerate(fail):
                if f <= show:
                    a.plot([f - 1, f - 1], [k * 4 - 0.5, k * 4 + 3.5], color="#e34948", lw=3)
ax[-1, 0].set_xlabel("cycle"), ax[-1, 1].set_xlabel("cycle")
fig.savefig(ROOT / f"{opt.out}.png", dpi=70)
print(f"\nwrote {opt.out}.png")
