#!/usr/bin/env python3
"""Export the v12 stamped machine and its program library for the browser demo.

Writes into ``demo/data``:

    stamp_v12.json    hyper-parameters, geometry, weights (base64) and every
                      program in ``stamp_programs.library()`` with its reference
                      register trace
    stamp_v12.bin     the same float32 weights as a sidecar, for local use

The demo streams one instruction every ``--k`` NCA steps. v12 was trained with 60;
40 is the measured floor (0 errors in 3,072 instructions; 30 gives 1%).

Before anything is written, every program is executed on the real PyTorch model,
``--runs`` times with independent fire masks, and every boundary is decoded and
compared with the reference trace. A program the model does not reproduce is
reported, and the export refuses to proceed unless ``--allow-fail`` is given.
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ncpu.nca import NeuralCA  # noqa: E402
from ncpu.program import Instr, TileLayoutV8  # noqa: E402
from stamp_programs import library  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/STAMP_20261018_101454_v12_s1/nca_latest.pt")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
ap.add_argument("--out", default="demo/data")
ap.add_argument("--k", type=int, default=40, help="NCA steps per streamed instruction")
ap.add_argument("--runs", type=int, default=16, help="independent executions per program")
ap.add_argument("--allow-fail", action="store_true")
ap.add_argument("--long-runs", type=int, default=4,
                help="runs for programs over 2000 instructions (Life takes minutes per run)")
ap.add_argument("--only", default="", help="comma-separated ids: verify just these, export nothing")
args = ap.parse_args()

ck = torch.load(ROOT / args.ckpt, map_location="cpu", weights_only=False)
cfg, sd = ck["config"], ck["nca"]
WB, R, PITCH = cfg["word_bits"], cfg["radius"], cfg["pitch"]

nca = NeuralCA(channels=cfg["channels"], hidden_channels=cfg["hidden"],
               fire_rate=cfg["fire_rate"], alive_threshold=0,
               zero_initialization=False, kernel_size=cfg["kernel"],
               perception="static_sobel", padding_type="zeros",
               read_only_dims=[], state_bound=10.0).to(args.device)
nca.load_state_dict(sd)
nca.eval()


def decoder(L):
    """Every register at once on the GPU: gather each bit's square minus its
    centre pixel, mean, sign. Same read as TileLayoutV8.decode_column."""
    idx = []
    for c in L.col_x:
        for y in L.row_y:
            idx.append([(y + dy) * L.W + c + dx for dy in range(-L.r, L.r + 1)
                        for dx in range(-L.r, L.r + 1) if dx or dy])
    idx = torch.tensor(idx, device=args.device)                           # (NR*WB, ring)
    wts = torch.tensor([1 << (L.word_bits - 1 - b) for b in range(L.word_bits)], device=args.device)

    def dec(ch0):                                                         # (B, H, W) -> (B, NR), -1 = unknown
        m = ch0.flatten(1)[:, idx].mean(-1).view(ch0.shape[0], len(L.col_x), L.word_bits)
        v = ((m > 0).long() * wts).sum(-1)
        return torch.where((m.abs() >= 0.25).all(-1), v, torch.full_like(v, -1))
    return dec


@torch.no_grad()
def verify(p):
    """Execute ``p`` ``args.runs`` times in one batch; per-run first cycle at which
    ANY register disagrees with the reference trace."""
    L = TileLayoutV8(n_regs=p.n_regs, word_bits=p.word_bits, r=R, pitch=PITCH)
    dec = decoder(L)
    trace = torch.tensor(p.trace(), device=args.device)
    x = torch.zeros(args.runs, cfg["channels"], L.H, L.W, device=args.device)
    x[:, 0] = torch.from_numpy(L.data_screen(p.regs0)).to(args.device)
    fail = torch.zeros(args.runs, dtype=torch.long, device=args.device)
    for c, ins in enumerate(p.instrs):
        st = L.mask_channels(Instr(ins["op"], ins["a"], ins["b"], ins["dst"]))
        x[:, 1:4] = torch.from_numpy(st).to(args.device)
        for _ in range(args.k):                                           # no K-frame history
            x = nca(x, 1)[:, -1]
        bad = (dec(x[:, 0]) != trace[c + 1]).any(-1)
        fail = torch.where((fail == 0) & bad, torch.full_like(fail, c + 1), fail)
        if len(p.instrs) > 2000 and (c + 1) % 1000 == 0:
            print(f"    {p.id}: {c + 1}/{len(p.instrs)}, {int((fail == 0).sum())}/{args.runs} exact so far", flush=True)
    return L, [None if f == 0 else int(f) for f in fail.tolist()]


# Verification results are cached by program content + checkpoint + K, so a
# re-export does not re-run Life's 330k NCA steps for nothing.
import hashlib
CACHE = ROOT / args.out / "verify_cache.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}


def key_of(p):
    blob = json.dumps([p.regs0, p.instrs, p.word_bits, args.ckpt, args.k], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()


programs, bad = [], []
only = {s for s in args.only.split(",") if s}
lib = [p for p in library() if not only or p.id in only]
print(f"verifying {len(lib)} programs, K = {args.k}")
for p in lib:
    k = key_of(p)
    if k in cache and not only:
        ok, runs = cache[k]
        L = TileLayoutV8(n_regs=p.n_regs, word_bits=p.word_bits, r=R, pitch=PITCH)
        print(f"  {p.id:<8} {ok:>2}/{runs} runs exact (cached)")
    else:
        args_runs = args.runs if len(p.instrs) <= 2000 else args.long_runs
        saved, args.runs = args.runs, args_runs
        L, fail = verify(p)
        args.runs = saved
        ok, runs = sum(f is None for f in fail), args_runs
        cache[k] = [ok, runs]
        CACHE.write_text(json.dumps(cache, indent=1))
        print(f"  {p.id:<8} {p.n_regs:>3} regs x {p.word_bits} bits {len(p.instrs):>5} instr   "
              f"{ok:>2}/{runs} runs exact" +
              ("" if ok == runs else f"   first failures at cycles {sorted(f for f in fail if f)}"), flush=True)
    if ok < runs:
        bad.append(p.id)
    programs.append({
        "id": p.id, "name": p.name, "desc": p.desc, "labels": p.labels,
        "regs0": p.regs0, "source": p.source, "inputs": p.inputs,
        "outputs": p.outputs, "signed": p.signed, "instrs": p.instrs,
        "trace": p.trace(), "n_regs": p.n_regs, "word_bits": p.word_bits, "screen": p.screen,
        "col_x": [int(v) for v in L.col_x], "row_y": [int(v) for v in L.row_y],
        "W": L.W, "H": L.H, "verified": f"{ok}/{runs}",
    })

if only:
    sys.exit(0)
if bad and not args.allow_fail:
    sys.exit(f"model did not reproduce: {bad} (re-run with --allow-fail to export anyway)")

# ── weights: identity, sobel_x, sobel_y (7x7), W1 (256x48), b1 (256), W2 (16x256) ──
kk = cfg["kernel"]
filt = sd["perception.all_filters_batch"].numpy().reshape(-1, kk, kk)
kern = [filt[0], filt[1], filt[2]]
for c in range(cfg["channels"]):                      # the grouped conv just tiles them
    for j in range(3):
        assert np.allclose(filt[c * 3 + j], kern[j]), "perception filters are not tiled"
w1 = sd["rule.rule.0.weight"].numpy().reshape(cfg["hidden"], -1)
b1 = sd["rule.rule.0.bias"].numpy()
w2 = sd["rule.rule.2.weight"].numpy().reshape(cfg["channels"], cfg["hidden"])
blob = b"".join(a.astype("<f4").tobytes() for a in [kern[0], kern[1], kern[2], w1, b1, w2])

out = ROOT / args.out
out.mkdir(parents=True, exist_ok=True)
(out / "stamp_v12.bin").write_bytes(blob)
meta = {
    "run": Path(args.ckpt).parent.name, "step": int(ck["step"]),
    "channels": cfg["channels"], "hidden": cfg["hidden"], "kernel": kk,
    "fire_rate": cfg["fire_rate"], "state_bound": 10.0,
    "k_train": cfg["k_cycle"], "k_cycle": args.k,
    "word_bits": WB, "radius": R, "pitch": PITCH,
    "colors": {k: list(v) for k, v in TileLayoutV8.COLORS.items()},
    # the same floats twice: a sidecar .bin, and base64 inline so the page can be
    # published as a single self-contained file (hosts may not serve a .bin)
    "weights": {"file": "stamp_v12.bin", "floats": len(blob) // 4,
                "order": ["k_id", "k_sx", "k_sy", "W1", "b1", "W2"],
                "b64": base64.b64encode(blob).decode("ascii")},
    "programs": programs,
}
(out / "stamp_v12.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
print(f"wrote {out / 'stamp_v12.json'} ({len(programs)} programs) and {out / 'stamp_v12.bin'}")
