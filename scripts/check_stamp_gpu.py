#!/usr/bin/env python3
"""Run the demo's own WGSL on a real WebGPU runtime and execute the program.

``scripts/check_stamp_web.py`` proves the exported blob is laid out the way the
shader indexes it; this proves the shader's arithmetic executes the whole
program correctly. The WGSL is not retyped here -- node dumps the exact string
``demo/js/stamp.js`` hands the browser.

NOT a browser check. wgpu-py compiles with naga, which is laxer than Chrome's
Tint (it accepted ``a * b ^ c``, which the WGSL spec and Tint reject), and this
script submits one step at a time instead of using the page's dynamic uniform
offsets. Both of those shipped broken while this passed. The real test is
loading ``demo/stamp_standalone.html`` in Chromium with --enable-unsafe-webgpu.

    node scratchpad/dump_wgsl.mjs stamp.wgsl
    uv run --with wgpu python scripts/check_stamp_gpu.py --wgsl stamp.wgsl
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import wgpu

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.program import Instr, TileLayoutV8  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

ap = argparse.ArgumentParser()
ap.add_argument("--wgsl", required=True, help="shader dumped from demo/js/stamp.js")
ap.add_argument("--seed", type=int, default=12345)
args = ap.parse_args()

meta = json.loads((ROOT / "demo/data/stamp_v12.json").read_text())
weights = np.frombuffer((ROOT / "demo/data/stamp_v12.bin").read_bytes(), dtype="<f4")
P = meta["program"]
C, H, W = meta["channels"], meta["H"], meta["W"]
N, K = H * W, meta["k_cycle"]
L = TileLayoutV8(n_regs=meta["n_regs"], word_bits=meta["word_bits"],
                 r=meta["radius"], pitch=meta["pitch"])

adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
device = adapter.request_device_sync()
print(f"adapter: {adapter.info['device']} ({adapter.info['backend_type']})")

module = device.create_shader_module(code=Path(args.wgsl).read_text())
pipe = device.create_compute_pipeline(
    layout=wgpu.AutoLayoutMode.auto, compute={"module": module, "entry_point": "main"})
print("shader compiled")

U = wgpu.BufferUsage
state_bytes = C * N * 4
A = device.create_buffer(size=state_bytes, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
B = device.create_buffer(size=state_bytes, usage=U.STORAGE | U.COPY_SRC | U.COPY_DST)
wbuf = device.create_buffer(size=weights.nbytes, usage=U.STORAGE | U.COPY_DST)
pbuf = device.create_buffer(size=16, usage=U.UNIFORM | U.COPY_DST)
device.queue.write_buffer(wbuf, 0, weights.tobytes())


def bind(src, dst):
    return device.create_bind_group(layout=pipe.get_bind_group_layout(0), entries=[
        {"binding": 0, "resource": {"buffer": src, "offset": 0, "size": state_bytes}},
        {"binding": 1, "resource": {"buffer": dst, "offset": 0, "size": state_bytes}},
        {"binding": 2, "resource": {"buffer": wbuf, "offset": 0, "size": weights.nbytes}},
        {"binding": 3, "resource": {"buffer": pbuf, "offset": 0, "size": 16}},
    ])


bg = {True: bind(A, B), False: bind(B, A)}
gx, gy = -(-W // 8), -(-H // 8)

x0 = np.zeros((C, N), np.float32)
x0[0] = L.data_screen(P["regs0"]).reshape(-1)
device.queue.write_buffer(A, 0, x0.tobytes())

ok = 0
step = 0
for c, ins in enumerate(P["instrs"]):
    s = L.mask_channels(Instr(ins["op"], ins["a"], ins["b"], ins["dst"]))
    device.queue.write_buffer(A, N * 4, s.astype(np.float32).reshape(-1).tobytes())
    src_is_a = True
    for _ in range(K):
        device.queue.write_buffer(pbuf, 0, struct.pack(
            "<IfII", (args.seed + step * 2654435761) & 0xFFFFFFFF, meta["fire_rate"], 0, 0))
        enc = device.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(pipe)
        cp.set_bind_group(0, bg[src_is_a])
        cp.dispatch_workgroups(gx, gy)
        cp.end()
        device.queue.submit([enc.finish()])
        src_is_a = not src_is_a
        step += 1
    assert src_is_a, "K must be even for the A/B schedule"

    raw = np.frombuffer(device.queue.read_buffer(A, 0, state_bytes), dtype="<f4")
    ch0 = raw[:N].reshape(H, W)
    got = L.decode_column(ch0)[0]
    want = tuple(P["trace"][c + 1])
    good = got == want
    ok += good
    resid = float(np.abs(raw[N:4 * N]).max())
    hid = float(np.abs(raw[4 * N:]).max())
    print(f"  cycle {c + 1}  {ins['text']:<22} -> {got}  want {want}  "
          f"{'ok' if good else 'MISMATCH'}   resid {resid:.4f} hid {hid:.4f}")

out = L.decode_column(np.frombuffer(device.queue.read_buffer(A, 0, N * 4),
                                    dtype="<f4").reshape(H, W))[0][P["output"]]
print(f"\n{ok}/{len(P['instrs'])} boundaries exact, r{P['output']} = {out}")
assert ok == len(P["instrs"]) and out == P["trace"][-1][P["output"]]
print("the shipped shader executes fib(5) correctly on real WebGPU")
