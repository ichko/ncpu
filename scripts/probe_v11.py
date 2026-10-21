#!/usr/bin/env python3
"""Per-op accuracy for a v11 checkpoint -- the number the headline eval hides,
since NOP needs no computation and is a third of the sampled instructions."""
import argparse, glob, json, random
from pathlib import Path
import numpy as np, torch
from ncpu.nca import NeuralCA
from ncpu.program import Instr, TileLayoutV11

ap = argparse.ArgumentParser()
ap.add_argument("--run", default=""); ap.add_argument("--n", type=int, default=96)
ap.add_argument("--cur", type=int, default=1); ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
torch.set_grad_enabled(False); random.seed(a.seed)

run = Path(a.run) if a.run else Path(sorted(glob.glob("runs/V11_2026*"))[-1])
cfg = json.loads((run / "config.json").read_text())
L = TileLayoutV11(n_instr=cfg["n_instr"], n_regs=cfg["n_regs"], word_bits=cfg["word_bits"],
                  r=cfg.get("r", 2), pitch=cfg.get("pitch", 8), pad=cfg.get("pad", 6))
nca = NeuralCA(channels=cfg["channels"], hidden_channels=cfg["hidden"], fire_rate=cfg["fire_rate"],
               alive_threshold=0, zero_initialization=False, kernel_size=3, padding_type="zeros",
               state_bound=cfg.get("state_bound", 10.0),
               soft_bound=not cfg.get("hard_bound", True)).to("cuda")
sd = torch.load(run / "nca_latest.pt", map_location="cuda"); nca.load_state_dict(sd["model"])

OPS = ("SUB", "SUBLEQ", "NOP")
by, mags, bits, tot, nok = {}, [], 0, 0, 0
wrote = {"SUBLEQ": [0, 0]}
for _ in range(a.n):
    op = random.choice(OPS)
    prog = [Instr(op, random.randrange(4), random.randrange(4), random.randrange(4))
            for _ in range(a.cur)]
    regs = [random.randrange(1 << cfg["word_bits"]) for _ in range(4)]
    traj = L.cycle(prog, regs)
    d, m = L.render(prog, traj[0][0])
    x = torch.zeros(1, cfg["channels"], L.H, L.W, device="cuda")
    x[0, 0] = torch.from_numpy(d).to("cuda"); x[0, 1:5] = torch.from_numpy(m).to("cuda")
    for st, ph, nx in traj:
        done = 0
        while done < cfg["K"][ph]:
            n = min(32, cfg["K"][ph] - done); x = nca(x, n)[:, -1]; done += n
    s = x[0, 0].cpu().numpy(); got, _ = L.decode(s); want = traj[-1][2]
    rings = [[L._ring(s, L.col_x[c], L.row_y[b]) for b in range(cfg["word_bits"])] for c in range(4)]
    mags += [abs(v) for r in rings for v in r]
    bits += sum((v > 0) == ((want.regs[c] >> (cfg["word_bits"] - 1 - b_)) & 1)
                for c, r in enumerate(rings) for b_, v in enumerate(r))
    tot += 4 * cfg["word_bits"]
    ok = got == tuple(want.regs); nok += ok
    e = by.setdefault(op, [0, 0]); e[0] += ok; e[1] += 1
    if op == "SUBLEQ" and a.cur == 1:                     # did it actually have to write?
        changed = tuple(want.regs) != tuple(regs)
        if changed: wrote["SUBLEQ"][1] += 1; wrote["SUBLEQ"][0] += ok
print(f"{run.name} step {sd['step']}  len={a.cur}  n={a.n}")
print(f"  exact {nok}/{a.n} = {nok/a.n:.3f}   per-bit {bits/tot:.4f}   ring |v| mean={np.mean(mags):.3f}")
for op in OPS:
    if op in by:
        h, c = by[op]; print(f"    {op:7s} {h:3d}/{c:3d} = {h/c:.2f}")
h, c = wrote["SUBLEQ"]
if c: print(f"    SUBLEQ that actually wrote: {h}/{c} = {h/c:.2f}")
