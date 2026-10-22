"""Transplant a trained NCA into a wider one without changing what it computes.

Adds state channels and hidden units to a checkpoint. The old weights keep their
slots; the new channels never feed the old ones (their input columns are zero) and
the new hidden units never write to the old channels (their output weights there
are zero), so the old channels follow exactly the same trajectory. New channels get
small random write weights so they carry a nonzero signal and receive gradient.

    uv run python scripts/widen_nca.py SRC.pt DST.pt --channels 24 --hidden 384
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.nca import NeuralCA  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--channels", type=int, default=24)
ap.add_argument("--hidden", type=int, default=384)
ap.add_argument("--kernel", type=int, default=7)
ap.add_argument("--new-write-scale", type=float, default=0.1,
                help="scale of the random weights that write into the new channels")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed)
ck = torch.load(args.src, map_location="cpu", weights_only=False)
old = ck["nca"]
W1, b1, W2 = old["rule.rule.0.weight"], old["rule.rule.0.bias"], old["rule.rule.2.weight"]
h0, f0 = W1.shape[:2]
c0 = W2.shape[0]
k = f0 // c0  # perception features per channel; feature 3c+j is filter j on channel c
assert args.channels >= c0 and args.hidden >= h0

new = NeuralCA(channels=args.channels, hidden_channels=args.hidden, fire_rate=0.5,
               alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
               perception="static_sobel", padding_type="circular", read_only_dims=[],
               state_bound=10.0).state_dict()
nW1, nb1, nW2 = (new[n].clone() for n in ("rule.rule.0.weight", "rule.rule.0.bias",
                                          "rule.rule.2.weight"))
nW1[:h0] = 0
nW1[:h0, :f0] = W1          # old units read only old channels
nW1[h0:, f0:] = 0           # new units read old channels only, for now
nb1[:h0] = b1
nW2[:c0] = 0
nW2[:c0, :h0] = W2          # old channels hear only old units
nW2[c0:] *= args.new_write_scale
new["rule.rule.0.weight"], new["rule.rule.0.bias"], new["rule.rule.2.weight"] = nW1, nb1, nW2

ck["nca"] = new
ck.setdefault("config", {})
ck["config"] = dict(ck["config"], channels=args.channels, hidden=args.hidden)
ck["widened_from"] = str(args.src)
torch.save(ck, args.dst)
print(f"{c0} ch x {h0} hidden -> {args.channels} ch x {args.hidden} hidden: {args.dst}")
