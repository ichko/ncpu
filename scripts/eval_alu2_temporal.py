#!/usr/bin/env python3
"""Per-op accuracy eval for a TemporalSobelNCA ALU2 run (mirrors analyze_alu2.py eval)."""
import json, random, sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.dataset import ALU2Dataset, ALU2_OP_NAMES, _compute_alu2, _int_to_bits_msb
from ncpu.temporal_nca import TemporalSobelNCA
from ncpu.utils import make_alu2_screen

run_dir = Path(sys.argv[1])
T_EVAL  = int(sys.argv[2]) if len(sys.argv) > 2 else 64
N_ACC   = 64

random.seed(42); torch.manual_seed(42)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

cfg    = json.load(open(run_dir / "config.json"))
ds_cfg = cfg["ds"]
nc     = cfg["nca"]
nca    = TemporalSobelNCA(**nc).to(DEVICE)
last_ckpt = sorted((run_dir / "checkpoints").glob("nca_0*.pt"))[-1]
nca.load_state_dict(torch.load(last_ckpt, map_location=DEVICE, weights_only=True))
nca.eval()
print(f"loaded {last_ckpt.name}  T_eval={T_EVAL}")

H, W = ds_cfg["H"], ds_cfg["W"]
r, sp = ds_cfg["r"], ds_cfg["among_sp"]
x_a, x_b, x_ctrl, x_out = ds_cfg["x_a"], ds_cfg["x_b"], ds_cfg["x_ctrl"], ds_cfg["x_out"]

ds = ALU2Dataset(Namespace(**ds_cfg))
bit_masks = ds.get_output_bit_masks().to(DEVICE)

def make_state(a, b, ci, op, cond):
    # Current ALU2Dataset convention: single-operand ops (NOT/RCL/RCR) operate
    # on B with A forced to zero; both columns are always drawn.
    ctrl = _int_to_bits_msb(op, 3) + [ci] + _int_to_bits_msb(cond, 3)
    screen = make_alu2_screen(H, W, r, sp, x_a, x_b, x_ctrl, x_out,
                              a_bits=_int_to_bits_msb(a, 8),
                              b_bits=_int_to_bits_msb(b, 8),
                              ctrl_bits=ctrl)
    img = torch.from_numpy(screen).float() / 128.0 - 1.0
    state = torch.zeros(1, nca.channels, H, W)
    state[0, 0] = img
    state[0, 1] = img
    return state

rows = []
for op in range(8):
    bit_correct = np.zeros(10)
    states, expected = [], []
    for _ in range(N_ACC):
        a, b = random.randint(0, 255), random.randint(0, 255)
        if op in (5, 6, 7):
            a = 0
        ci, cond = random.randint(0, 1), random.randint(0, 7)
        result, cout, branch = _compute_alu2(a, b, ci, op, cond)
        expected.append(_int_to_bits_msb(result, 8) + [cout, branch])
        states.append(make_state(a, b, ci, op, cond))
    state_batch = torch.cat(states, dim=0).to(DEVICE)
    nca.reset_history()
    with torch.no_grad():
        rollout = nca(state_batch, steps=T_EVAL)
    last_ch0 = rollout[:, -1, 0]
    for i in range(10):
        mask = bit_masks[i]
        avg = (last_ch0 * mask.unsqueeze(0)).sum(dim=(-2, -1)) / mask.sum()
        pred = (avg > 0).int().tolist()
        gt = [expected[j][i] for j in range(N_ACC)]
        bit_correct[i] = sum(p == g for p, g in zip(pred, gt))
    acc = bit_correct / N_ACC
    rows.append((ALU2_OP_NAMES[op], acc[:8].mean(), acc[8], acc[9]))
    print(f"  {ALU2_OP_NAMES[op]:4s}  result={acc[:8].mean()*100:5.1f}%  carry={acc[8]*100:5.1f}%  branch={acc[9]*100:5.1f}%")

res = np.array([[r[1], r[2], r[3]] for r in rows])
print(f"\nMEAN  result={res[:,0].mean()*100:.1f}%  carry={res[:,1].mean()*100:.1f}%  branch={res[:,2].mean()*100:.1f}%")
