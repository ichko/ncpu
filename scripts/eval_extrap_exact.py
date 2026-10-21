#!/usr/bin/env python3
"""Measure extrapolation two ways: per-bit and per-sum.

The original experiment (results/E5_extrapolation) only ever recorded per-bit
accuracy, which flatters the model — a sum can be 8/9 bits right and still be the
wrong number. This re-evaluates the same checkpoints and reports both, so the
talk can put them side by side.

Rolls the state forward one step at a time rather than calling forward(steps=T),
which stacks the whole sequence and would need ~9 GB for a batch this size.

    uv run python scripts/eval_extrap_exact.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from ncpu.nca import NeuralCA                                       # noqa: E402
from build_talk_extrap import bits_msb, disk, bottom_aligned_screen, output_centres  # noqa: E402

HORIZON = 128
N = 128           # samples per (run, width)
BATCH = 64
TEST_WIDTHS = [4, 6, 8]

RUNS = {
    2: ["extrap_train2_s0_20260322_181117", "extrap_train2_s1_20260322_194510"],
    3: ["extrap_train3_s0_20260322_211910", "extrap_train3_s1_20260322_225312"],
    4: ["extrap_train4_s0_20260323_002726", "extrap_train4_s1_20260323_020157"],
    6: ["extrap_train6_s0_20260323_064425", "extrap_train6_s1_20260323_081820"],
}


def evaluate(run_name, k, rng, dev):
    run = ROOT / "runs" / run_name
    cfg = json.loads((run / "config.json").read_text())
    ds, nca_cfg = cfg["ds"], cfg["nca"]
    H, W, r, spacing = ds["H"], ds["W"], ds["r"], ds["spacing"]
    C = nca_cfg["channels"]

    nca = NeuralCA(**nca_cfg).to(dev)
    ckpt = sorted((run / "checkpoints").glob("nca_*.pt"))[-1]
    nca.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False), strict=False)
    nca.eval()

    masks = torch.stack([
        torch.from_numpy(disk(cx, cy, r, H, W)) for cx, cy in output_centres(H, W, r, spacing, k + 1)
    ]).to(dev)                                                  # (n_out, H, W), LSB first

    bits_ok = bits_tot = exact_ok = total = 0
    for start in range(0, N, BATCH):
        b = min(BATCH, N - start)
        a_i = rng.integers(0, 2 ** k, b)
        b_i = rng.integers(0, 2 ** k, b)
        screens = np.stack([
            bottom_aligned_screen(H, W, r, spacing, bits_msb(int(x), k), bits_msb(int(y), k))
            for x, y in zip(a_i, b_i)])
        seed = torch.from_numpy(screens.astype(np.float32) / 128.0 - 1.0).to(dev)

        torch.manual_seed(0)
        x = torch.zeros(b, C, H, W, device=dev)
        x[:, 0] = seed
        x[:, 1] = seed
        with torch.no_grad():
            for _ in range(HORIZON):
                x = nca.forward(x, steps=1)[:, -1]               # keep only the new state

        ch0 = x[:, 0]                                            # (b, H, W)
        means = (ch0.unsqueeze(1) * masks).sum((2, 3)) / masks.sum((1, 2))
        got_bits = (means > 0).long().cpu().numpy()              # (b, n_out), LSB first
        want = a_i + b_i
        want_bits = np.stack([(want >> i) & 1 for i in range(k + 1)], 1)

        bits_ok += int((got_bits == want_bits).sum())
        bits_tot += got_bits.size
        exact_ok += int((got_bits == want_bits).all(1).sum())
        total += b

    return bits_ok / bits_tot * 100, exact_ok / total * 100


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device {dev}   {N} random sums per cell, mean of 2 seeds\n")
    bit_tbl, exact_tbl = {}, {}
    for train_k, runs in RUNS.items():
        for test_k in TEST_WIDTHS:
            bs, es = [], []
            for rn in runs:
                b, e = evaluate(rn, test_k, np.random.default_rng(0), dev)
                bs.append(b); es.append(e)
            bit_tbl[(train_k, test_k)] = float(np.mean(bs))
            exact_tbl[(train_k, test_k)] = float(np.mean(es))
            print(f"  trained <={train_k}  tested {test_k}-bit   "
                  f"bits {np.mean(bs):6.1f}%   exact sums {np.mean(es):6.1f}%")

    out = ROOT / "results/E5_extrapolation/extrap_exact.json"
    out.write_text(json.dumps({
        "n_per_cell": N, "seeds": 2, "horizon": HORIZON,
        "bits_pct": {f"{a}_{b}": v for (a, b), v in bit_tbl.items()},
        "exact_pct": {f"{a}_{b}": v for (a, b), v in exact_tbl.items()},
    }, indent=2))
    print(f"\nwrote {out}")
