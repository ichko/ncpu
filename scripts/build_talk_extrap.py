#!/usr/bin/env python3
"""Render rollouts from the adder that was trained on sums of at most 3 bits.

The talk claims the rule keeps working on wider sums than it ever saw. This makes
that visible: the same checkpoint runs on two 3-bit sums (inside training) and
one 8-bit sum (well outside it), and all three go out as sprite sheets in the
same format as scripts/build_talk.py writes.

The encoding is bottom-aligned — the least significant bit sits at a fixed y no
matter how many bits there are — which is what makes the widths interchangeable.
The disks are drawn here in numpy rather than through ncpu.utils, which pulls in
cv2 and therefore libGL.

    uv run python scripts/build_talk_extrap.py
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
from ncpu.nca import NeuralCA                                      # noqa: E402

RUN = ROOT / "runs/extrap_train3_s0_20260322_211910"
OUT = ROOT / "demo/assets/talk"
HORIZON = 128
SEED = 0

# (a, b, bit width) — two inside the training range, one far outside it
CASES = [
    ("extrap_3bit_a", 7, 1, 3),      # 7 + 1 = 8, the longest carry 3 bits allow
    ("extrap_3bit_b", 5, 6, 3),      # 5 + 6 = 11
    # A representative 8-bit sum, not a cherry-picked one: measured over 60 random
    # pairs this checkpoint gets 85% of 8-bit sums exactly right and 96.5% of the
    # bits, against 100% at 3 bits. 255 + 1 — the longest carry there is — is one
    # of the ones it misses, so it would misrepresent the typical case.
    ("extrap_8bit",  145, 103, 8),   # 145 + 103 = 248
]


def bits_msb(v, n):
    return [(v >> (n - 1 - i)) & 1 for i in range(n)]


def disk(cx, cy, r, H, W):
    y, x = np.ogrid[:H, :W]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def bottom_aligned_screen(H, W, r, spacing, a_bits, b_bits):
    """Mirror of ncpu.utils.make_io_screen_bottom_aligned, inputs only.

    A filled cv2 circle is exactly a Euclidean disk, so this matches what the
    model was trained on.
    """
    among, side = int(spacing[0]), int(spacing[1])
    step = 2 * r + among
    bottom = H - side
    screen = np.full((H, W), 128, np.uint8)
    for bits, cx in ((a_bits, side), (b_bits, side + step)):
        for i, bit in enumerate(reversed(bits)):        # i = 0 is the LSB
            cy = bottom - i * step
            if r <= cy < H - r:
                screen[disk(cx, cy, r, H, W)] = 255 if bit else 0
    return screen


def output_centres(H, W, r, spacing, n):
    among, side = int(spacing[0]), int(spacing[1])
    step = 2 * r + among
    return [(W - side, H - side - i * step) for i in range(n)]     # LSB first


def main():
    cfg = json.loads((RUN / "config.json").read_text())
    ds, nca_cfg = cfg["ds"], cfg["nca"]
    H, W, r, spacing = ds["H"], ds["W"], ds["r"], ds["spacing"]
    ckpt = sorted((RUN / "checkpoints").glob("nca_*.pt"))[-1]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nca = NeuralCA(**nca_cfg).to(dev)
    nca.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False), strict=False)
    nca.eval()
    C = nca_cfg["channels"]
    print(f"model: {RUN.name}  (trained on <= {cfg['run']['max_train_bits']} bits)")
    print(f"checkpoint: {ckpt.name}   device: {dev}")

    viridis = colormaps["viridis"]
    OUT.mkdir(parents=True, exist_ok=True)

    for name, a, b, k in CASES:
        img = bottom_aligned_screen(H, W, r, spacing, bits_msb(a, k), bits_msb(b, k))
        seed = torch.from_numpy(img.astype(np.float32) / 128.0 - 1.0).to(dev)

        torch.manual_seed(SEED)
        st = torch.zeros(1, C, H, W, device=dev)
        st[:, 0] = seed
        st[:, 1] = seed                                  # channel 1 is read-only
        with torch.no_grad():
            roll = nca.forward(st, steps=HORIZON)[0].cpu().numpy()   # (T, C, H, W)

        # read the answer off the settled grid, the way the demo does
        n_out = k + 1
        got = 0
        for i, (cx, cy) in enumerate(output_centres(H, W, r, spacing, n_out)):
            m = roll[-1, 0][disk(cx, cy, r, H, W)].mean()
            got |= (1 if m > 0 else 0) << i
        ok = "correct" if got == a + b else f"WRONG (got {got})"

        ch0 = np.clip((roll[:, 0] + 1) / 2, 0, 1)
        rgb = (viridis(ch0)[..., :3] * 255).astype(np.uint8)          # (T, H, W, 3)
        strip = Image.fromarray(rgb.transpose(1, 0, 2, 3).reshape(H, -1, 3))
        strip.convert("P", palette=Image.ADAPTIVE, colors=256).save(
            OUT / f"film_{name}.png", optimize=True)
        print(f"  film_{name}.png   {k}-bit  {a} + {b} = {a + b}  -> {ok}   "
              f"{len(rgb)} frames  {(OUT / f'film_{name}.png').stat().st_size/1e3:.0f} KB")


if __name__ == "__main__":
    main()
