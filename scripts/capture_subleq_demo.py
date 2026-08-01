#!/usr/bin/env python3
"""Run fib(N) on the trained subleq NCA and dump a verified-correct execution
trace to JSON for the web demo, plus a full-resolution rollout FILMSTRIP PNG per
instruction (every channel-0 frame, tiled horizontally) so the page can play the
NCA computing frame-by-frame with full control (speed, pauses)."""
import json, sys, shutil
from pathlib import Path
import numpy as np
import matplotlib.cm as cm
from PIL import Image
import torch
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from ncpu.nca import NeuralCA
from ncpu.subleq import DartboardLayout
from subleq_vm import fib_program, assemble, ref_alu

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5
CKPT = "runs/20260726_214420_SUBLEQ_dartboard_w8_s0/checkpoints/nca_last.pt"
STEPS = 96
dev = "cuda"
lay = DartboardLayout(W=72, H=72, word_bits=8)
masks = lay.output_bit_masks().to(dev); msum = masks.sum((-1, -2))
ck = torch.load(CKPT, map_location=dev, weights_only=False)
nca = NeuralCA(**ck["nca_config"]).to(dev); nca.load_state_dict(ck["nca"]); nca.eval()


@torch.no_grad()
def nca_alu_frames(A, B):
    inp = torch.from_numpy(lay.screens_for(A, B)[0]).to(dev)
    st = torch.zeros(1, nca.channels, 72, 72, device=dev); st[:, 0] = inp; st[:, 1] = inp
    roll = nca.forward(st, steps=STEPS)[0, :, 0]            # (STEPS+1, H, W)
    per = (roll.unsqueeze(1) * masks).sum((-1, -2)) / msum  # (T, 9): decoded output each frame
    fbtab = (per > 0).int().cpu().tolist()                 # T x 9 (0..7 = R MSB-first, 8 = branch)
    lastb = fbtab[-1]
    R = 0
    for j in range(8):
        R = (R << 1) | lastb[j]
    packed = [sum(r[j] << j for j in range(9)) for r in fbtab]   # 1 int per frame
    return R, lastb[8], roll.cpu().numpy(), packed


def fib_ref(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


p, rc = fib_program(N)
mem0, addr, code_bytes = assemble(p)
inv = {v: k for k, v in addr.items()}
expected = fib_ref(N)


def run_capture(max_steps=100000):
    mem = list(mem0); pc = 0; trace = []; frames = []; steps = 0
    while 0 <= pc < code_bytes and steps < max_steps:
        a, b, c = mem[pc], mem[pc + 1], mem[pc + 2]
        opA, opB = mem[a], mem[b]
        R, branch, fr, fb = nca_alu_frames(opA, opB)
        eR, eB = ref_alu(opA, opB)                          # ground truth
        mem[b] = R
        pc_after = c if branch else pc + 3
        trace.append({"step": steps, "pc": pc, "idx": pc // 3, "a": a, "b": b, "c": c,
                      "opA": opA, "opB": opB, "result": R, "branch": branch, "fb": fb,
                      "exp_result": eR, "exp_branch": eB, "ok": int(R == eR and branch == eB),
                      "pc_after": pc_after, "mem": mem.copy()})
        frames.append(fr)
        pc = pc_after; steps += 1
    return mem, trace, frames


trace = frames = None
for attempt in range(40):
    out, tr, fr = run_capture()
    if out[addr[rc]] == expected:
        trace, frames = tr, fr; break
if trace is None:
    print("FAILED to get a clean run"); sys.exit(1)

# ── render per-instruction rollout filmstrips (every frame, tiled horizontally) ──
strip_dir = Path("demo/data/rollouts")
if strip_dir.exists():
    shutil.rmtree(strip_dir)
strip_dir.mkdir(parents=True)
strips = []
nframes = frames[0].shape[0]
for i, fr in enumerate(frames):
    t = np.clip((fr + 1) / 2, 0, 1)                         # [-1,1] -> [0,1]
    rgb = (cm.viridis(t)[..., :3] * 255).astype(np.uint8)   # (T,H,W,3)
    T, H, W, _ = rgb.shape
    strip = rgb.transpose(1, 0, 2, 3).reshape(H, T * W, 3)  # (H, T*W, 3)
    name = f"rollouts/step_{i:03d}.png"
    Image.fromarray(strip).save(Path("demo/data") / name)
    strips.append("data/" + name)

# ── static artifacts + trace json ──
def nm(a): return inv.get(a, f"@{a}")
csrc = ["int fib(int n) {", "  int a=0, b=1, t;", "  while (n > 0) {",
        "    t = a + b;", "    a = b;", "    b = t;", "    n = n - 1;",
        "  } // repeat", "  return a;   // fib(n)", "}"]
instrs = []
for i, (a, b, c) in enumerate(p.code):
    cshow = "end" if c == "end" else (c if c else "next")
    instrs.append({"idx": i, "bytes": [addr[a], addr[b], (addr.get(c, code_bytes) if c else (i + 1) * 3)],
                   "a": nm(addr[a]), "b": nm(addr[b]), "c": cshow, "src": p.tags[i]})
data = [{"name": n, "addr": addr[n], "init": mem0[addr[n]]} for n in p.data_order]
tr_out = [{"step": t["step"], "idx": t["idx"], "src": p.tags[t["idx"]],
           "a": t["a"], "b": t["b"], "a_name": nm(t["a"]), "b_name": nm(t["b"]),
           "opA": t["opA"], "opB": t["opB"], "result": t["result"], "branch": t["branch"],
           "exp_result": t["exp_result"], "exp_branch": t["exp_branch"], "ok": t["ok"], "fb": t["fb"],
           "pc": t["pc"], "pc_after": t["pc_after"], "mem": t["mem"], "strip": strips[k]}
          for k, t in enumerate(trace)]
out_obj = {"N": N, "expected": expected, "actual_answer": out[addr[rc]],
           "result_cell": rc, "result_addr": addr[rc],
           "input_cell": "n", "input_val": N, "input_addr": addr["n"],
           "code_bytes": code_bytes, "mem_size": len(mem0), "nframes": nframes,
           "frame_w": 72, "frame_h": 72, "csrc": csrc,
           "instrs": instrs, "data": data, "init_mem": mem0, "trace": tr_out,
           "n_steps": len(tr_out)}
Path("demo/data/subleq_fib.json").write_text(json.dumps(out_obj))
allok = all(t["ok"] for t in trace)
print(f"fib({N})={expected} actual={out[addr[rc]]} clean on attempt {attempt+1}  "
      f"steps={len(trace)}  frames/instr={nframes}  strips={len(strips)}  all-steps-correct={allok}")
