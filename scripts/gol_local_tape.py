#!/usr/bin/env python3
"""Game of Life with the LOCAL compiler (src/ncpu/gol_local.py) on a v15 tape checkpoint.

The ring is two generations (every instruction within 3 consecutive registers), run
closed loop against the software machine. Prints the first wrong cycle per start
board and, per generation, which boards are still exact.

    uv run python scripts/gol_local_tape.py --run runs/TAPE_..._mix2_s0 --n 4 --cycles 400
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ncpu.gol_local import life  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="nca_best.pt")
ap.add_argument("--n", type=int, default=4, help="board side")
ap.add_argument("--cycles", type=int, default=0, help="0 = two generations (one ring pass)")
ap.add_argument("--boards", default="glider,blinker")
ap.add_argument("--device", default="cuda")
opt = ap.parse_args()

BOARDS = {"glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
          "blinker": [(1, 0), (1, 1), (1, 2)],
          "block": [(1, 1), (1, 2), (2, 1), (2, 2)]}

run = Path(opt.run)
ck = torch.load(run / opt.ckpt, map_location=opt.device, weights_only=False)
cfg = ck["config"]
src = (ROOT / "scripts/train_tape.py").read_text()
src = src[:src.index("# ── model")]
sys.argv = ["train_tape.py"] + [a for k, v in cfg.items() if v is not None and v != ""
                                for a in (f"--{k.replace('_', '-')}", str(v))]
sys.argv += ["--device", opt.device]
T = {"__file__": str(ROOT / "scripts/train_tape.py"), "__name__": "gol"}
exec(compile(src, "train_tape.py", "exec"), T)
args, K = T["args"], T["K"]
nca = T["NeuralCA"](channels=args.channels, hidden_channels=args.hidden, fire_rate=args.fire_rate,
                    alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
                    perception="static_sobel", padding_type="circular", read_only_dims=[],
                    state_bound=10.0).to(opt.device)
nca.load_state_dict(ck["nca"])
T["apply_fire"](nca, cfg)
nca.eval()

names = opt.boards.split(",")
progs = [life(BOARDS[k], opt.n, args.word_bits) for k in names]
p0 = progs[0]
n, NR = len(p0.tape), p0.n_regs


def start(p):
    g = T["geom"](n, n - 1 - min(args.below, n - 1))
    j = n - 1                                          # the last instruction is done: 0 loads next
    ring = {(j + m) % n: (g.head - (m - 1)) % g.N for m in range(1, n)}
    return dict(g=g, nr=NR, tape=p.tape, ring=ring, inmem=j, regs=tuple(p.regs0))


runs = [start(p) for p in progs]
x = T["start_state"](runs)
cycles = opt.cycles or n
print(f"{run.name}/{opt.ckpt} step {ck['step']:,}: {opt.n}x{opt.n} board, {n} instructions "
      f"(generations end at {p0.gen_end}), {NR} registers x {args.word_bits} bits, "
      f"picture {tuple(x.shape[-2:])}, {cycles} cycles\n", flush=True)
fail = np.full(len(runs), cycles + 1)
with torch.no_grad():
    for c in range(1, cycles + 1):
        runs, _, _, _ = T["cycle_targets"](runs)
        for _ in range(K):
            x = nca(x, 1)[:, -1].clone()
        bad = (T["decode"](runs, x[:, 0]) != T["want_bits"](runs)).flatten(1).any(1).cpu().numpy()
        newly = bad & (fail > cycles)
        fail[newly] = c
        for i in np.flatnonzero(newly):
            print(f"  {names[i]} first wrong at cycle {c} (instruction {(c - 1) % n}: "
                  f"{runs[i]['tape'][(c - 1) % n]})", flush=True)
        if c % n in p0.gen_end or c % n == 0:
            print(f"generation {2 * (c // n) + (1 if c % n == p0.gen_end[0] else 0)} done at cycle {c}: "
                  + ", ".join(f"{k} {'exact' if f > c else 'wrong'}" for k, f in zip(names, fail)),
                  flush=True)
        if (fail <= cycles).all():
            break
print("\nfirst wrong cycle: " + ", ".join(f"{k} {f if f <= cycles else 'never'}"
                                         for k, f in zip(names, fail)))
