#!/usr/bin/env python3
"""Render one rollout per ALU operation for the talk deck.

The slide used to show eight settled grids lifted out of a matplotlib figure.
These are the runs themselves: the same trained ALU, one animation per opcode,
written as sprite sheets in the format demo/talk.html walks with a `steps()`
animation.

The encoding comes straight from ncpu.dataset.ALU2Dataset, so the inputs are
built exactly the way the model was trained to read them: operand A, operand B, a
control column holding a 3-bit opcode, a carry-in and a 3-bit condition code, and
ten outputs (result byte, carry-out, branch).

    uv run python scripts/build_talk_alu.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib import colormaps
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ncpu.dataset import ALU2Dataset, _compute_alu2, _int_to_bits_msb   # noqa: E402
from ncpu.nca import NeuralCA                                            # noqa: E402

RUN = ROOT / "runs/E_alu2_s0_20260323_203754"
OUT = ROOT / "demo/assets/talk"
# 64 is the horizon the ALU was evaluated at in results/E_alu2; the rollout is
# stochastic and does not stay pinned to the answer indefinitely past it.
HORIZON = 64
SEED = 0
COND = 6                     # AL — "always", so the branch bit is deterministic

# (name, op index, a, b, carry_in) — the same examples the poster figure uses.
#
# Careful with the single-operand ops (NOT/RCL/RCR). `_compute_alu2` takes their
# operand as `b`, but this checkpoint was trained with that operand drawn in the
# **A** column and B left blank — the convention scripts/analyze_alu2.py uses.
# The current ALU2Dataset.get_sample() does the opposite (operand in B, A zeroed),
# so it must have changed after this run was trained. Follow the checkpoint:
# reading it the dataset's way makes the model look like it never learned NOT,
# when in fact it reproduces it exactly.
CASES = [
    ("add", 0, 170,  85, 0),
    ("sub", 1, 240,  15, 0),   # SUB is the weakest op at 87.5%; 200-75 is one it misses
    ("and", 2, 204, 170, 0),
    ("or",  3,  12,  48, 0),
    ("xor", 4, 255, 170, 0),
    ("not", 5,   0,  85, 0),   # operand 85 -> drawn in A
    ("rcl", 6,   0,  65, 0),   # operand 65 -> drawn in A
    ("rcr", 7,   0, 130, 0),   # operand 130 -> drawn in A
]


def main():
    cfg = json.loads((RUN / "config.json").read_text())
    ds_cfg, nca_cfg = cfg["ds"], cfg["nca"]
    H, W = ds_cfg["H"], ds_cfg["W"]

    ds = ALU2Dataset()
    for k, v in ds_cfg.items():                       # keep the trained geometry
        setattr(ds, {"r": "R", "among_sp": "AMONG_SP", "x_a": "X_A", "x_b": "X_B",
                     "x_ctrl": "X_CTRL", "x_out": "X_OUT"}.get(k, k.upper()), int(v))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nca = NeuralCA(**nca_cfg).to(dev)
    ckpt = RUN / "checkpoints/nca_latest.pt"
    nca.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False), strict=False)
    nca.eval()
    C = nca_cfg["channels"]
    print(f"model: {RUN.name}  ({ckpt.name})   device: {dev}")

    viridis = colormaps["viridis"]
    OUT.mkdir(parents=True, exist_ok=True)
    masks = ds.get_output_bit_masks().to(dev)          # (10, H, W)

    for name, op, a, b, ci in CASES:
        want, want_c, want_br = _compute_alu2(a, b, ci, op, COND)
        ctrl = _int_to_bits_msb(op, 3) + [ci] + _int_to_bits_msb(COND, 3)
        single = op in (5, 6, 7)
        screen = ds._screen(
            a_bits=_int_to_bits_msb(b if single else a, 8),
            b_bits=None if single else _int_to_bits_msb(b, 8),
            ctrl_bits=ctrl)
        seed = torch.from_numpy(screen.astype(np.float32) / 128.0 - 1.0).to(dev)

        torch.manual_seed(SEED)
        st = torch.zeros(1, C, H, W, device=dev)
        # train_alu2.py never passes input_dims, so the trainer default (1,) applies:
        # the question goes into channel 1 alone and channel 0 starts empty. (The
        # adder demo is different — that one was exported with input_dims [0, 1].)
        st[:, 1] = seed
        with torch.no_grad():
            roll = nca.forward(st, steps=HORIZON)[0].cpu().numpy()      # (T, C, H, W)

        final = torch.from_numpy(roll[-1, 0]).to(dev)
        means = (final.unsqueeze(0) * masks).sum((1, 2)) / masks.sum((1, 2))
        got_bits = (means > 0).long().cpu().numpy()    # result[7:0] MSB-first, cout, branch
        got = int("".join(map(str, got_bits[:8])), 2)
        ok = "ok " if got == want else "MISS"

        ch0 = np.clip((roll[:, 0] + 1) / 2, 0, 1)
        rgb = (viridis(ch0)[..., :3] * 255).astype(np.uint8)
        strip = Image.fromarray(rgb.transpose(1, 0, 2, 3).reshape(H, -1, 3))
        strip.convert("P", palette=Image.ADAPTIVE, colors=256).save(
            OUT / f"film_alu_{name}.png", optimize=True)
        print(f"  film_alu_{name}.png  {name.upper():<4} a={a:<4} b={b:<4} -> got {got:<4} "
              f"want {want:<4} [{ok}]  {len(rgb)} frames  "
              f"{(OUT / f'film_alu_{name}.png').stat().st_size / 1e3:5.0f} KB")


if __name__ == "__main__":
    main()
