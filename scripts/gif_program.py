#!/usr/bin/env python3
"""Roll out a trained program-NCA on one program and write the value channel
as an animated GIF, side by side with the target, plus a per-column readout.

    uv run python scripts/gif_program.py runs/PROG_.../nca_last.pt --instr 1 --out works.gif
    uv run python scripts/gif_program.py runs/PROG_.../nca_last.pt --instr 2 --out fails.gif
    uv run python scripts/gif_program.py ... --addressing free   # routing it never trained on
"""

import json
import random
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.gif import write_gif  # noqa: E402
from ncpu.nca import NeuralCA  # noqa: E402
from ncpu.program import (  # noqa: E402
    ProgramLayout, TileLayoutV2, TileLayoutV5, run_program, sample_program, sample_regs)

ap = ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("--instr", type=int, default=1)
ap.add_argument("--addressing", default=None, help="override the trained addressing mode")
ap.add_argument("--ops", default=None, help="override the op set, e.g. ADD")
ap.add_argument("--seed", type=int, default=3)
ap.add_argument("--stride", type=int, default=2, help="keep every Nth frame")
ap.add_argument("--zoom", type=int, default=4)
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default="program_rollout.gif")
args = ap.parse_args()

d = Path(args.checkpoint).parent
cfg = json.load(open(d / "config.json"))
ck = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
WB, NR, K = cfg["word_bits"], cfg["regs"], cfg["K"]
ops = tuple((args.ops or cfg["ops"]).upper().split(","))
addressing = args.addressing or cfg["addressing"]

V2 = cfg.get("layout") == "v2"
if cfg.get("layout") == "v5":
    L = TileLayoutV5(n_instr=args.instr, n_regs=NR, word_bits=WB, r=cfg["radius"],
                     pitch=cfg["pitch"], stages=cfg.get("stages_per_tile", 2))
    TYPE_CH = list(range(1, 1 + len(TileLayoutV5.TYPES)))
elif V2:
    L = TileLayoutV2(n_instr=args.instr, n_regs=NR, word_bits=WB, r=cfg["radius"],
                     pitch=cfg["pitch"], unknown=0.0 if cfg["unknown"] == "zero" else -1.0,
                     scaffold=cfg.get("scaffold", False), scaffold_cell=cfg["scaffold_cell"] or 6)
    TYPE_CH = list(range(1, 1 + len(TileLayoutV2.TYPES)))
else:
    L = ProgramLayout(n_instr=args.instr, n_regs=NR, word_bits=WB, r=cfg["radius"],
                      pitch=cfg["pitch"], col_pitch=cfg["col_pitch"],
                      unknown=0.0 if cfg["unknown"] == "zero" else -1.0,
                      scaffold_cell=cfg["scaffold_cell"], op_amp=cfg["op_amp"])
    TYPE_CH = []
nca = NeuralCA(channels=cfg["channels"], hidden_channels=cfg["hidden"], fire_rate=cfg["fire_rate"],
               alive_threshold=0, zero_initialization=False, kernel_size=cfg["kernel"],
               perception="static_sobel", padding_type="zeros", read_only_dims=TYPE_CH,
               state_bound=cfg["state_bound"]).to(args.device)
nca.load_state_dict(ck["nca"]); nca.eval()

rng = random.Random(args.seed)
prog = sample_program(args.instr, NR, ops=ops, copy_frac=1.0 if ops == ("COPY",) else 0.35,
                      rng=rng, addressing=addressing, nop_frac=cfg.get("nop_frac", 0.0))
regs = sample_regs(NR, WB, rng=rng)
trace = run_program(regs, prog, WB)
N_ST = getattr(L, "n_stages", args.instr)
target = (L.paint(prog, regs, N_ST) if hasattr(L, "n_stages")
          else L.paint(prog, regs, resolved=args.instr, scaffold=True))

frames = [L.input_screen(prog, regs)]
with torch.no_grad():
    x = torch.zeros(1, cfg["channels"], L.H, L.W, device=args.device)
    x[:, 0] = torch.from_numpy(frames[0]).to(args.device)
    if TYPE_CH:
        x[:, TYPE_CH] = torch.from_numpy(L.type_channels()).to(args.device)
    for _ in range(N_ST * K):
        x = nca(x, 1)[:, -1]
        frames.append(x[0, 0].cpu().numpy())
frames = np.stack(frames)

# readout at each stage boundary
print("program:")
for i, ins in enumerate(prog):
    print(f"  {i}: {ins}")
print(f"\ninput regs {list(regs)}")
print(f"{'stage':>6} {'t':>5}  " + "  ".join(f"col{c:<12}" for c in range(args.instr + 1)))
per = N_ST // args.instr
for s in range(args.instr):
    t = (s + 1) * per * K
    row = []
    for c in range(args.instr + 1):
        got = L.decode_column(frames[t], c)[0]
        ok = "ok" if got == trace[c] else ("--" if c > s + 1 else "WRONG")
        row.append(f"{str(list(got)):<12}{ok:>3}")
    print(f"{s:>6} {t:>5}  " + "  ".join(row))
print(f"\nreference: " + "  ".join(str(list(t)) for t in trace))

# side by side: rollout | target, with a 3px divider
sep = np.full((frames.shape[0], L.H, 3), -1.0, dtype=np.float32)
first = np.broadcast_to(frames[0], frames.shape).copy()      # left: t=0, static
tgt = np.broadcast_to(target, frames.shape).copy()           # right: final target, static
panel = np.concatenate([first, sep, frames, sep, tgt], axis=2)[:: args.stride]
out = Path(__file__).resolve().parent.parent / args.out
write_gif(out, panel, fps=15, cmap="gray", zoom=args.zoom)
print(f"\nwrote {out}  ({panel.shape[0]} frames: input | rollout | target)")
