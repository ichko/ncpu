#!/usr/bin/env python3
"""Game of Life on a v15 tape-machine checkpoint.

The 4x4 GoL program (src/ncpu/gol_program.py: 768 SUB/SUBLEQ instructions, 43
registers x 6 bits, two generations per pass of the ring) runs closed loop. Board A
and board B are read out of memory as screens (register = pixel) and compared with
the software machine. Prints the first wrong cycle / generation per start board and
writes <out>.gif: the NCA's two screens, the reference's, and the memory rows of the
NCA picture, every --every cycles.

    uv run python scripts/gol_tape.py --run runs/TAPE_..._mem6_s0 --ckpt nca_best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from ncpu.gol_program import initial_regs, program  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="nca_best.pt")
ap.add_argument("--gens", type=int, default=4)
ap.add_argument("--every", type=int, default=24, help="cycles per GIF frame (24 = one cell)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--out", default="v15_gol")
opt = ap.parse_args()

H = W = 4
BOARDS = {
    "blinker": [[0, 0, 0, 0], [1, 1, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    "glider": [[0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0]],
    "block": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 1, 0], [0, 0, 0, 0]],
}

run = Path(opt.run)
ck = torch.load(run / opt.ckpt, map_location=opt.device, weights_only=False)
cfg = ck["config"]
src = (ROOT / "scripts/train_tape.py").read_text()
src = src[:src.index("# ── model")]
sys.argv = ["train_tape.py"] + [a for k, v in cfg.items() if v is not None and v != ""
                                for a in (f"--{k.replace('_', '-')}", str(v))]
sys.argv += ["--device", opt.device]
T = {"__file__": str(ROOT / "scripts/train_tape.py"), "__name__": "gol"}
exec(compile(src, "train_tape.py", "exec"), T)
args, K = T["args"], T["K"]
nca = T["NeuralCA"](channels=args.channels, hidden_channels=args.hidden, fire_rate=args.fire_rate,
                    alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
                    perception="static_sobel", padding_type="circular", read_only_dims=[],
                    state_bound=10.0).to(opt.device)
nca.load_state_dict(ck["nca"])
T["apply_fire"](nca, cfg)
nca.eval()

prog, R = program(H, W)
NR, n = R["n_regs"], len(prog)
assert args.word_bits >= 6, f"GoL needs 6-bit words; this checkpoint has {args.word_bits}"


def start(board):
    g = T["geom"](n, n - 1 - min(args.below, n - 1))
    j = n - 1                                          # the last instruction is done: 0 loads next
    ring = {(j + m) % n: (g.head - (m - 1)) % g.N for m in range(1, n)}
    return dict(g=g, nr=NR, tape=prog, ring=ring, inmem=j,
                regs=tuple(initial_regs(board, args.word_bits, R)))


def screens(bits):                                     # (B, regs, bits) -> (B, 2, H, W) alive
    alive = bits.any(-1).cpu().numpy()                 # alive = -1 (all ones), dead = 0
    return np.stack([alive[:, R["A"]:R["A"] + H * W], alive[:, R["B"]:R["B"] + H * W]], 1) \
        .reshape(-1, 2, H, W)


names = list(BOARDS)
runs = [start(BOARDS[k]) for k in names]
x = T["start_state"](runs)
g0 = runs[0]["g"]
print(f"{run.name}/{opt.ckpt} step {ck['step']:,}: {n} instructions, {NR} registers x "
      f"{args.word_bits} bits, picture {tuple(x.shape[-2:])}\n", flush=True)
cycles = opt.gens * n // 2
fail = np.full(len(runs), cycles + 1)
frames = []
with torch.no_grad():
    for c in range(1, cycles + 1):
        runs, _, _, _ = T["cycle_targets"](runs)
        x = nca(x, K)[:, -1]
        got, want = T["decode"](runs, x[:, 0]), T["want_bits"](runs)
        bad = (got != want).flatten(1).any(1).cpu().numpy()
        fail[bad & (fail > cycles)] = c
        if c % opt.every == 0 or c == cycles:
            frames.append((c, screens(got), screens(want), x[:, 0, g0.M0:g0.M1].cpu().numpy(),
                           fail.copy()))
        if c % (n // 2) == 0:
            print(f"generation {c // (n // 2)}: " + ", ".join(
                f"{k} {'ok' if f > c else f'wrong since cycle {f}'}" for k, f in zip(names, fail)),
                flush=True)

# ── GIF ──────────────────────────────────────────────────────────────────────
S, PAD = 28, 12                                         # cell px, gap


def grid(img, d, x0, y0, b, wrong):
    for yy in range(H):
        for xx in range(W):
            col = (240, 240, 235) if b[yy, xx] else (30, 30, 34)
            d.rectangle([x0 + xx * S, y0 + yy * S, x0 + (xx + 1) * S - 2, y0 + (yy + 1) * S - 2], fill=col)
    if wrong:
        d.rectangle([x0 - 4, y0 - 4, x0 + W * S + 2, y0 + H * S + 2], outline=(227, 73, 72), width=3)


mem_h = g0.M1 - g0.M0
mw = frames[0][3].shape[-1]
scale = 2
row_h = H * S + 30 + mem_h * scale + PAD
img_w = max(4 * (W * S + PAD) + 160, mw * scale + 20)
pics = []
for c, gm, gr, mem, fl in frames:
    gen = c / (n // 2)
    im = Image.new("RGB", (img_w, 40 + len(names) * row_h), (250, 250, 248))
    d = ImageDraw.Draw(im)
    d.text((10, 10), f"cycle {c}   generation {gen:.2f}   (board A -> B on even generations, B -> A on odd)",
           fill=(20, 20, 20))
    for i, k in enumerate(names):
        y0 = 40 + i * row_h
        wrong = fl[i] <= c
        d.text((10, y0 + 2), k + ("  WRONG" if wrong else ""), fill=(227, 73, 72) if wrong else (20, 20, 20))
        for j, (lab, g) in enumerate((("NCA A", gm[i, 0]), ("NCA B", gm[i, 1]),
                                      ("true A", gr[i, 0]), ("true B", gr[i, 1]))):
            x0 = 160 + j * (W * S + PAD)
            d.text((x0, y0 + 2), lab, fill=(80, 80, 80))
            grid(im, d, x0, y0 + 16, g, wrong and j < 2 and (g != gr[i, j % 2]).any())
        m = np.clip((mem[i] + 1) / 2, 0, 1)
        m = (np.stack([m] * 3, -1) * 255).astype(np.uint8)
        mi = Image.fromarray(m).resize((mw * scale, mem_h * scale), Image.NEAREST)
        im.paste(mi, (10, y0 + 22 + H * S))
    pics.append(im)
pics[0].save(ROOT / f"{opt.out}.gif", save_all=True, append_images=pics[1:] + [pics[-1]] * 10,
             duration=150, loop=0)
print(f"wrote {opt.out}.gif ({len(pics)} frames)")
