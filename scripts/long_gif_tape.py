#!/usr/bin/env python3
"""GIF of v15 programs running for many cycles, to watch them degrade.

Runs --batch random programs closed loop for --cycles cycles, keeping the picture at
the end of every cycle. Picks --show runs with spread-out failure cycles and draws,
for each, the model (left) next to the true machine state (right). A red frame marks
a run whose memory has gone wrong; the bar at the bottom shows the cycle and a red
tick where each shown run failed.

    uv run python scripts/long_gif_tape.py --run runs/TAPE_... --ckpt nca_best.pt
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--ckpt", default="nca_best.pt")
ap.add_argument("--instrs", type=int, default=8)
ap.add_argument("--cycles", type=int, default=200)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--show", type=int, default=4)
ap.add_argument("--scale", type=int, default=6)
ap.add_argument("--ms", type=int, default=100, help="ms per frame (one frame per cycle)")
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default="v15_long_run.gif")
opt = ap.parse_args()

run = Path(opt.run)
ck = torch.load(run / opt.ckpt, map_location=opt.device, weights_only=False)
cfg = ck["config"]
src = (ROOT / "scripts/train_tape.py").read_text()
src = src[:src.index("# ── model")]
sys.argv = ["train_tape.py"] + [a for k, v in cfg.items() if v is not None and v != ""
                                for a in (f"--{k.replace('_', '-')}", str(v))]
sys.argv += ["--device", opt.device]
T = {"__file__": str(ROOT / "scripts/train_tape.py"), "__name__": "long_gif"}
exec(compile(src, "train_tape.py", "exec"), T)
args, K = T["args"], T["K"]
nca = T["NeuralCA"](channels=args.channels, hidden_channels=args.hidden, fire_rate=args.fire_rate,
                    alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
                    perception="static_sobel", padding_type="circular", read_only_dims=[],
                    state_bound=10.0).to(opt.device)
nca.load_state_dict(ck["nca"])
T["apply_fire"](nca, cfg)
nca.eval()

COL = {k: np.array(v) for k, v in T["TileLayoutV8"].COLORS.items()}


def rgb(ch0, ins):
    img = np.full(ch0.shape + (3,), 0.5)
    mem = np.abs(ch0) > 0.05
    img[mem] = np.clip((ch0[mem] + 1) / 2, 0, 1)[:, None]
    mag = np.clip(np.stack([ins[0], ins[1], ins[2], -ins[2]]), 0, 1)       # A, B, SUB, SUBLEQ
    cols = np.stack([COL["A"], COL["B"], COL["R+"], COL["R-"]])
    tint = np.einsum("khw,kc->hwc", mag, cols) / np.maximum(mag.sum(0), 1e-6)[..., None]
    alpha = (mag.max(0) * np.where(mem, 0.5, 1.0))[..., None]
    return img * (1 - alpha) + tint * alpha


with torch.no_grad():
    rng = random.Random(opt.seed)
    runs = [T["fresh_run"](opt.instrs, args.eval_regs, rng) for _ in range(opt.batch)]
    x = T["start_state"](runs)
    model, truth = [x[:, :4].cpu().numpy()], [x[:, :4].cpu().numpy()]
    fail = np.full(opt.batch, opt.cycles + 1)
    for c in range(1, opt.cycles + 1):
        runs, _, t_mem, t_tape = T["cycle_targets"](runs)
        x = nca(x, K)[:, -1]
        ok = (T["decode"](runs, x[:, 0]) == T["want_bits"](runs)).flatten(1).all(1).cpu().numpy()
        fail[(~ok) & (fail > opt.cycles)] = c
        model.append(x[:, :4].cpu().numpy())
        truth.append(torch.cat([t_mem[:, None], t_tape], 1).cpu().numpy())

# runs with spread-out failure cycles (a survivor included when there is one)
bad_runs = [int(i) for i in np.argsort(fail) if fail[i] <= opt.cycles]
good_runs = [int(i) for i in range(opt.batch) if fail[i] > opt.cycles]
n_bad = min(len(bad_runs), opt.show - (1 if good_runs else 0))
pick = [bad_runs[round(q * (len(bad_runs) - 1))] for q in np.linspace(0, 1, n_bad)] if n_bad else []
pick = sorted(set(pick), key=lambda i: fail[i]) + good_runs[:opt.show - len(set(pick))]
print("failure cycle of each run:", sorted(fail.tolist()))
print("shown:", [int(fail[i]) for i in pick])

S, gap, pad, top, barh = opt.scale, 28, 6, 40, 46
FONT = ImageFont.truetype("DejaVuSans.ttf", 20)
SMALL = ImageFont.truetype("DejaVuSans.ttf", 16)
h, w = model[0].shape[2] * S, model[0].shape[3] * S
pair = 2 * w + pad
Wc = len(pick) * (pair + gap) - gap + 2 * gap
Hc = top + h + 30 + barh
frames = []
for c in range(opt.cycles + 1):
    im = Image.new("RGB", (Wc, Hc), (255, 255, 255))
    d = ImageDraw.Draw(im)
    for j, i in enumerate(pick):
        x0 = gap + j * (pair + gap)
        for k, src_ in enumerate((model, truth)):
            tile = np.kron(rgb(src_[c][i, 0], src_[c][i, 1:4]), np.ones((S, S, 1)))
            im.paste(Image.fromarray((tile * 255).astype(np.uint8)), (x0 + k * (w + pad), top))
        bad = c >= fail[i]
        if bad:
            d.rectangle([x0 - 4, top - 4, x0 + w + 3, top + h + 3], outline=(220, 30, 30), width=4)
        d.text((x0, 8), "model | truth", fill=(0, 0, 0), font=SMALL)
        status = f"wrong since {fail[i]}" if bad else ("ok" if fail[i] > opt.cycles else "ok")
        d.text((x0, top + h + 6), status, fill=(220, 30, 30) if bad else (30, 140, 60), font=SMALL)
    # progress bar
    by, bx0, bx1 = top + h + 34, gap, Wc - gap
    d.rectangle([bx0, by, bx1, by + 16], outline=(0, 0, 0), width=1)
    d.rectangle([bx0, by, bx0 + (bx1 - bx0) * c / opt.cycles, by + 16], fill=(90, 120, 220))
    for i in pick:
        if fail[i] <= opt.cycles:
            tx = bx0 + (bx1 - bx0) * fail[i] / opt.cycles
            d.line([tx, by - 4, tx, by + 20], fill=(220, 30, 30), width=3)
    d.text((bx0, by + 20), f"cycle {c} / {opt.cycles}", fill=(0, 0, 0), font=SMALL)
    frames.append(im)
frames[0].save(ROOT / opt.out, save_all=True, append_images=frames[1:], duration=opt.ms, loop=0)
print(f"wrote {opt.out}: {len(frames)} frames, {Wc}x{Hc}")
