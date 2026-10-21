#!/usr/bin/env python3
"""Check the exported blob against PyTorch, indexing it exactly as the WGSL does.

The shader reads one flat float32 array. If its offsets or its matrix strides
disagree with what the exporter wrote, the demo silently computes nonsense, so
this reimplements the shader's arithmetic in numpy straight off the bytes and
compares a rollout against the real model. ``fire_rate`` is forced to 1 so both
sides are deterministic.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA  # noqa: E402
from ncpu.program import Instr, TileLayoutV8  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
meta = json.loads((ROOT / "demo/data/stamp_v12.json").read_text())
w = np.frombuffer((ROOT / "demo/data/stamp_v12.bin").read_bytes(), dtype="<f4")

C, HID, KS = meta["channels"], meta["hidden"], meta["kernel"]
H, W_, PERC, PAD = meta["H"], meta["W"], 3 * C, KS // 2
KK = KS * KS
O_KID, O_KSX, O_KSY = 0, KK, 2 * KK
O_W1 = 3 * KK
O_B1 = O_W1 + HID * PERC
O_W2 = O_B1 + HID
assert len(w) == O_W2 + C * HID == meta["weights"]["floats"], "blob length"


def shader_step(x):
    """One NCA step, written the way the WGSL writes it (fire mask = 1)."""
    perc = np.zeros((PERC, H, W_), np.float32)
    for c in range(C):
        for dy in range(KS):
            for dx in range(KS):
                gy, gx = np.arange(H) + dy - PAD, np.arange(W_) + dx - PAD
                v = np.zeros((H, W_), np.float32)
                ys = (gy >= 0) & (gy < H)
                xs = (gx >= 0) & (gx < W_)
                v[np.ix_(ys, xs)] = x[c][np.ix_(gy[ys], gx[xs])]
                k = dy * KS + dx
                perc[c * 3 + 0] += w[O_KID + k] * v
                perc[c * 3 + 1] += w[O_KSX + k] * v
                perc[c * 3 + 2] += w[O_KSY + k] * v

    w1 = w[O_W1:O_W1 + HID * PERC].reshape(HID, PERC)      # shader: W1[j*PERC + i]
    b1 = w[O_B1:O_B1 + HID]
    w2 = w[O_W2:O_W2 + C * HID].reshape(C, HID)            # shader: W2[o*HID + j]
    h = np.maximum(np.tensordot(w1, perc, (1, 0)) + b1[:, None, None], 0.0)
    delta = np.tensordot(w2, h, (1, 0))
    return np.clip(x + delta, -meta["state_bound"], meta["state_bound"]).astype(np.float32)


ck = torch.load(ROOT / "runs" / meta["run"] / "nca_latest.pt", map_location="cpu",
                weights_only=False)
nca = NeuralCA(channels=C, hidden_channels=HID, fire_rate=1.0, alive_threshold=0,
               zero_initialization=False, kernel_size=KS, perception="static_sobel",
               padding_type="zeros", read_only_dims=[], state_bound=10.0)
nca.load_state_dict(ck["nca"])
nca.eval()

L = TileLayoutV8(n_regs=meta["n_regs"], word_bits=meta["word_bits"],
                 r=meta["radius"], pitch=meta["pitch"])
x = np.zeros((C, H, W_), np.float32)
x[0] = L.data_screen(meta["program"]["regs0"])
ins = meta["program"]["instrs"][0]
x[1:4] = L.mask_channels(Instr(ins["op"], ins["a"], ins["b"], ins["dst"]))

xt = torch.from_numpy(x)[None]
worst = 0.0
for s in range(1, 13):
    x = shader_step(x)
    with torch.no_grad():
        xt = nca(xt, 1)[:, -1]
    d = float(np.abs(x - xt[0].numpy()).max())
    worst = max(worst, d)
    print(f"  step {s:>2}  max |shader - torch| = {d:.3e}")

print(f"\nworst deviation over 12 steps: {worst:.3e}")
assert worst < 1e-4, "shader weight layout disagrees with the model"
print("blob layout and shader arithmetic match PyTorch")
