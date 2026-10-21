#!/usr/bin/env python3
"""Evaluate a trained program-NCA on held-out compiled programs.

The rule is trained only on random instruction sequences.  This script compiles
programs it has never seen -- Fibonacci at several lengths, plus a random-program
control at matched lengths -- and reports per-column bit accuracy and exact-match.

    uv run python scripts/eval_program.py runs/PROG_.../nca_best.pt
    uv run python scripts/eval_program.py runs/PROG_.../nca_best.pt --fib 1,2,3,4,5,6
"""

import json
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA  # noqa: E402
from ncpu.program import (  # noqa: E402
    fib_program,
    fib_reference,
    layout_for,
    run_program,
    sample_program,
    sample_regs,
)

parser = ArgumentParser()
parser.add_argument("checkpoint")
parser.add_argument("--device", default="cuda")
parser.add_argument("--fib", default="1,2,3,4,5,6,7,8", help="program lengths to test")
parser.add_argument("--control", type=int, default=32,
                    help="random programs per length, as a matched control")
parser.add_argument("--settle", type=int, default=0, help="override K")
args = parser.parse_args()

ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
cfg = ckpt["config"]
GEOM = dict(n_regs=cfg["regs"], word_bits=cfg["word_bits"],
            r=cfg["radius"], pitch=cfg["pitch"], col_pitch=cfg["col_pitch"])
PX_PER_STEP = cfg["kernel"] // 2
K = args.settle or cfg["settle"] or 2 * layout_for(1, **GEOM).min_rollout(PX_PER_STEP)
WB = cfg["word_bits"]

nca = NeuralCA(
    channels=cfg["channels"], hidden_channels=cfg["hidden"], fire_rate=cfg["fire_rate"],
    alive_threshold=0, zero_initialization=True, kernel_size=cfg["kernel"],
    perception="static_sobel", padding_type="zeros", read_only_dims=[],
    state_bound=cfg["state_bound"],
).to(args.device)
nca.load_state_dict(ckpt["nca"])
nca.eval()

print(f"checkpoint {args.checkpoint}  (step {ckpt.get('step')}, score {ckpt.get('score')})")
print(f"trained on: ops={cfg['ops']}, program length {cfg['instr_min']}-{cfg['instr_max']}, K={K}\n")


@torch.no_grad()
def execute(layout, program, regs):
    """Run the painted program; return the decoded bits of every data column."""
    inp = torch.from_numpy(layout.input_screen(program, regs))[None].to(args.device)
    x = torch.zeros(1, cfg["channels"], layout.H, layout.W, device=args.device)
    x[:, 0] = inp
    out = []
    for _ in range(layout.n_instr):
        x = nca(x, K)[:, -1]
        out.append(x[0, 0].cpu().numpy())
    return out


def score(program, regs):
    layout = layout_for(len(program), **GEOM)
    frames = execute(layout, program, regs)
    trace = run_program(regs, program, WB)
    bits, exact = [], []
    for i, frame in enumerate(frames):
        got = layout.decode_bits(frame, i + 1)
        want = np.array([[int(c) for c in format(v, f"0{WB}b")] for v in trace[i + 1]])
        bits.append(float((got == want).mean()))
        exact.append(float((got == want).all()))
    decoded = [layout.decode_column(f, i + 1)[0] for i, f in enumerate(frames)]
    return bits, exact, decoded, trace


results = {}
print("FIBONACCI (held out -- never trained on)")
print(f"{'len':>4} {'bit acc':>9} {'exact/col':>10} {'final ok':>9}  value (got vs want)")
for n in [int(x) for x in args.fib.split(",") if x.strip()]:
    prog, regs = fib_program(n, cfg["regs"])
    bits, exact, decoded, trace = score(prog, regs)
    got = decoded[-1][(n - 1) % 2]
    want = trace[-1][(n - 1) % 2]
    true_fib = fib_reference(n)[-1]
    note = "" if want == true_fib else f"  (wraps: true fib = {true_fib})"
    print(f"{n:>4} {np.mean(bits):>9.4f} {np.mean(exact):>10.4f} "
          f"{str(got == want):>9}  {got} vs {want}{note}")
    results[f"fib_{n}"] = {"bits": bits, "exact": exact, "final_ok": bool(got == want)}

print("\nRANDOM PROGRAM CONTROL (same ISA, same lengths)")
print(f"{'len':>4} {'bit acc':>9} {'exact/col':>10} {'all cols':>9}")
rng = random.Random(0)
opset = tuple(o.strip().upper() for o in cfg["ops"].split(","))
for n in [int(x) for x in args.fib.split(",") if x.strip()]:
    b, e, allc = [], [], []
    for _ in range(args.control):
        prog = sample_program(n, cfg["regs"], ops=opset, copy_frac=cfg["copy_frac"], rng=rng)
        regs = sample_regs(cfg["regs"], WB, rng=rng)
        bits, exact, _, _ = score(prog, regs)
        b.append(np.mean(bits)); e.append(np.mean(exact)); allc.append(float(np.all(exact)))
    print(f"{n:>4} {np.mean(b):>9.4f} {np.mean(e):>10.4f} {np.mean(allc):>9.4f}")
    results[f"rand_{n}"] = {"bits": float(np.mean(b)), "exact": float(np.mean(e)),
                            "all_cols": float(np.mean(allc))}

out = Path(args.checkpoint).parent / "eval_heldout.json"
out.write_text(json.dumps(results, indent=2))
print(f"\nwrote {out}")
