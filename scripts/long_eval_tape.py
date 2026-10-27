#!/usr/bin/env python3
"""Long-horizon test of a v15 tape-machine checkpoint.

Random programs of several lengths run closed loop for up to --cycles cycles; for
each run we record the first cycle whose memory is wrong. Prints survival (fraction
of runs still computing right) at a few horizons and writes v15_long_eval.json.
"zeros" = share of the reference registers at 0 after cycle 100: random SUB/SUBLEQ
programs collapse towards all-zero memory (long ones fully), which is easy to hold.

    uv run python scripts/long_eval_tape.py --run runs/TAPE_... --ckpt nca_best.pt
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="nca_best.pt")
ap.add_argument("--lengths", default="8,12,32,128,512")
ap.add_argument("--cycles", type=int, default=1000)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="v15_long_eval.json")
opt = ap.parse_args()

run = Path(opt.run)
ck = torch.load(run / opt.ckpt, map_location=opt.device, weights_only=False)
cfg = ck["config"]

# the trainer's own geometry, painting and targets: its source up to the model
src = (ROOT / "scripts/train_tape.py").read_text()
src = src[:src.index("# ── model")]
sys.argv = ["train_tape.py"] + [a for k, v in cfg.items() if v is not None and v != ""
                                for a in (f"--{k.replace('_', '-')}", str(v))]
sys.argv += ["--device", opt.device]
T = {"__file__": str(ROOT / "scripts/train_tape.py"), "__name__": "long_eval"}
exec(compile(src, "train_tape.py", "exec"), T)
args, K = T["args"], T["K"]

nca = T["NeuralCA"](channels=args.channels, hidden_channels=args.hidden, fire_rate=args.fire_rate,
                    alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
                    perception="static_sobel", padding_type="circular", read_only_dims=[],
                    state_bound=10.0).to(opt.device)
nca.load_state_dict(ck["nca"])
T["apply_fire"](nca, cfg)
nca.eval()


@torch.no_grad()
def survive(n):
    rng = random.Random(opt.seed + n)
    runs = [T["fresh_run"](n, args.eval_regs, rng) for _ in range(opt.batch)]
    x = T["start_state"](runs)
    fail = np.full(opt.batch, opt.cycles + 1)       # first wrong cycle; cycles+1 = never
    zeros = []                                      # reference registers at 0, per cycle
    for c in range(1, opt.cycles + 1):
        runs, _, _, _ = T["cycle_targets"](runs)
        zeros.append(np.mean([v == 0 for r in runs for v in r["regs"]]))
        for _ in range(K):                          # one step at a time: only the last frame is kept
            x = nca(x, 1)[:, -1]
        ok = (T["decode"](runs, x[:, 0]) == T["want_bits"](runs)).flatten(1).all(1).cpu().numpy()
        fail[(~ok) & (fail > opt.cycles)] = c
        if (fail <= opt.cycles).all():
            break
    return fail, float(np.mean(zeros[99:])) if len(zeros) > 99 else float("nan")


HORIZONS = [h for h in (8, 32, 100, 300, 1000) if h <= opt.cycles]
res = {"run": run.name, "ckpt": opt.ckpt, "step": ck["step"], "cycles": opt.cycles,
       "batch": opt.batch, "lengths": {}}
print(f"{run.name}/{opt.ckpt} step {ck['step']:,}: {opt.batch} programs per length, "
      f"up to {opt.cycles} cycles\n")
print(f"{'instrs':>7}" + "".join(f"{f'@{h}':>8}" for h in HORIZONS) + f"{'median fail':>13}{'zeros':>7}{'time':>8}")
for n in (int(v) for v in opt.lengths.split(",")):
    t0 = time.time()
    fail, z = survive(n)
    med = float(np.median(fail))
    res["lengths"][n] = dict(fail=fail.tolist(), zeros_after100=z,
                            **{f"at{h}": float((fail > h).mean()) for h in HORIZONS})
    print(f"{n:>7}" + "".join(f"{(fail > h).mean():>8.3f}" for h in HORIZONS)
          + f"{('>' if med > opt.cycles else '') + str(int(min(med, opt.cycles))):>13}"
          + f"{z:>7.2f}{time.time() - t0:>7.0f}s", flush=True)
    (ROOT / opt.out).write_text(json.dumps(res, indent=1))
