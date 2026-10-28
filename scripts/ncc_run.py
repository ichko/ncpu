#!/usr/bin/env python3
"""Compile a C-like program (src/ncpu/ncc.py) and run it on a v15 tape-machine checkpoint.

The ring is painted into the picture, the NCA runs closed loop, and every cycle the
memory read back from the picture is compared with the software machine. When the
program halts, the variables are decoded from the PICTURE and printed.

    uv run python scripts/ncc_run.py programs/gcd.c --run runs/TAPE_..._local_s0
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ncpu.ncc import compile_src, reference, run_machine, signed  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="nca_best.pt")
ap.add_argument("--max-cycles", type=int, default=0, help="0 = until the software machine halts")
ap.add_argument("--keep-going", type=int, default=0, help="1: do not stop at the first wrong cycle")
ap.add_argument("--ring-prefix", type=int, default=0,
                help="diagnostic: run only the first N instructions as the ring (vs the software machine)")
ap.add_argument("--device", default="cuda")
opt = ap.parse_args()

text = Path(opt.src).read_text()
prog = compile_src(text)
want, passes, total = run_machine(prog)
print(f"{opt.src}: {len(prog.tape)} instructions, {prog.n_regs} registers, {len(prog.blocks)} blocks; "
      f"halts after {passes} passes = {total:,} cycles")
print(f"  reference: {reference(text)}")

run = Path(opt.run)
ck = torch.load(run / opt.ckpt, map_location=opt.device, weights_only=False)
cfg = ck["config"]
src = (ROOT / "scripts/train_tape.py").read_text()
src = src[:src.index("# ── model")]
sys.argv = ["train_tape.py"] + [a for k, v in cfg.items() if v is not None and v != ""
                                for a in (f"--{k.replace('_', '-')}", str(v))]
sys.argv += ["--device", opt.device]
T = {"__file__": str(ROOT / "scripts/train_tape.py"), "__name__": "ncc"}
exec(compile(src, "train_tape.py", "exec"), T)
args, K = T["args"], T["K"]
assert args.word_bits == 6, "ncc emits 6-bit code"
nca = T["NeuralCA"](channels=args.channels, hidden_channels=args.hidden, fire_rate=args.fire_rate,
                    alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
                    perception="static_sobel", padding_type="circular", read_only_dims=[],
                    state_bound=10.0).to(opt.device)
nca.load_state_dict(ck["nca"])
T["apply_fire"](nca, cfg)
nca.eval()

if opt.ring_prefix:
    prog.tape = prog.tape[:opt.ring_prefix]
    total = opt.max_cycles or 10 * opt.ring_prefix
n = len(prog.tape)
g = T["geom"](n, n - 1 - min(args.below, n - 1))
ring = {(n - 1 + m) % n: (g.head - (m - 1)) % g.N for m in range(1, n)}
runs = [dict(g=g, nr=prog.n_regs, tape=prog.tape, ring=ring, inmem=n - 1, regs=tuple(prog.regs0))]
x = T["start_state"](runs)
cycles = opt.max_cycles or total
print(f"  {run.name}/{opt.ckpt} step {ck['step']:,}; picture {tuple(x.shape[-2:])}\n", flush=True)


def bits_to_regs(b):
    return [int("".join(str(int(v)) for v in row), 2) for row in b]


first_bad, n_bad = None, 0
with torch.no_grad():
    for c in range(1, cycles + 1):
        runs, _, _, _ = T["cycle_targets"](runs)
        for _ in range(K):
            x = nca(x, 1)[:, -1].clone()
        got = T["decode"](runs, x[:, 0])
        bad = bool((got != T["want_bits"](runs)).any())
        if bad:
            n_bad += 1
            if first_bad is None:
                first_bad = c
                wr = np.flatnonzero((got != T["want_bits"](runs)).any(-1)[0].cpu().numpy())
                print(f"  first wrong cycle {c} (pass {(c - 1) // n + 1}, instruction {(c - 1) % n}: "
                      f"{prog.tape[(c - 1) % n]}); wrong registers {wr.tolist()}", flush=True)
            if not opt.keep_going:
                break
        if c % n == 0:
            print(f"  pass {c // n} done: {'exact' if first_bad is None else f'{n_bad} wrong cycles'}",
                  flush=True)

regs = bits_to_regs(got[0].cpu().numpy())
out = {k: signed(regs[r]) for k, r in prog.var_reg.items()}
print(f"\n  NCA picture after {c:,} cycles: {out}")
print(f"  software machine:            {want}")
print("  RESULT:", "CORRECT" if first_bad is None and c == total else
      f"wrong from cycle {first_bad}" if first_bad else f"stopped at cycle {c}")
