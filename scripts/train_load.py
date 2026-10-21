#!/usr/bin/env python3
"""Isolate the LOAD stage: can the rule move the program stack at all?

No EXEC, no clock, no chaining.  Every sample starts from a canonical
stage-start state -- instruction i expanded over the grid, instructions i+1..
waiting as 5 px bands -- and the rule gets K steps to reach the next
stage-start state: bands down one pitch, the next band expanded, the old
rectangles gone.  The register grid must hold unchanged throughout.

    uv run python scripts/train_load.py --device cuda
"""

import json
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ncpu.nca import NeuralCA  # noqa: E402
from ncpu.program import TileLayoutV9, sample_program, sample_regs  # noqa: E402

ap = ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=300_000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--stack-min", type=int, default=1, help="bands still waiting AFTER the load")
ap.add_argument("--stack-max", type=int, default=3)
ap.add_argument("--channels", type=int, default=16)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--kernel", type=int, default=7)
ap.add_argument("--fire-rate", type=float, default=0.5)
ap.add_argument("--k", type=int, default=40, help="CA steps for the load")
ap.add_argument("--tail", type=int, default=10, help="supervise the last N frames")
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--init-from", default=None)
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--eval-batch", type=int, default=32)
ap.add_argument("--ckpt-every", type=int, default=5000)
ap.add_argument("--snapshot-every", type=int, default=2500)
ap.add_argument("--tag", default="load")
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed); rng = random.Random(args.seed)
DEV = args.device
run = f"LOAD_{datetime.now():%Y%m%d_%H%M%S}_{args.tag}_s{args.seed}"
run_dir = Path(__file__).resolve().parent.parent / "runs" / run
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

_layouts = {}


def layout(n):
    """A v9 layout whose stack still holds ``n`` instructions after the load."""
    if n not in _layouts:
        _layouts[n] = TileLayoutV9(n_instr=n + 1, preloaded=True)
    return _layouts[n]


nca = NeuralCA(channels=args.channels, hidden_channels=args.hidden, fire_rate=args.fire_rate,
               alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
               perception="static_sobel", padding_type="zeros", read_only_dims=[], state_bound=10.0).to(DEV)
if args.init_from:
    nca.load_state_dict(torch.load(args.init_from, map_location=DEV, weights_only=False)["nca"])
    print(f"warm-started from {args.init_from}")
opt = torch.optim.Adam(nca.parameters(), lr=args.lr)
print(f"run {run}: stacks of {args.stack_min}..{args.stack_max} waiting bands, K={args.k}, "
      f"params {sum(p.numel() for p in nca.parameters() if p.requires_grad):,}")


def make_batch(bs, n, rng):
    """Start = stage 0 (instr 0 expanded, n bands waiting); target = stage 2 (after one LOAD)."""
    L = layout(n)
    x0, tgt, progs = [], [], []
    for _ in range(bs):
        prog = sample_program(n + 1, 4, ops=("SUB", "SUBLEQ"), copy_frac=0.0, rng=rng)
        regs = sample_regs(4, 4, rng=rng)
        d = L.data_screen(regs)
        x = np.zeros((args.channels, L.H, L.W), np.float32)
        x[0] = d; x[1:4] = L.mask_channels_at(prog, 0)
        x0.append(x)
        t = np.zeros((4, L.H, L.W), np.float32)
        t[0] = d                                        # the grid must not change
        t[1:4] = L.mask_channels_at(prog, 2)            # stack down one, next band expanded
        tgt.append(t); progs.append(prog)
    f = lambda a: torch.from_numpy(np.stack(a)).to(DEV)
    return L, f(x0), f(tgt), progs


def losses(seq, tgt):
    tail = seq[:, -args.tail:, :4]
    return ((tail - tgt[:, None]) ** 2).mean()


@torch.no_grad()
def evaluate(n, bs):
    """Fraction of loads that land: rectangles right AND bands right AND grid held."""
    L, x, tgt, progs = make_batch(bs, n, random.Random(999 + n))
    for _ in range(args.k):
        x = nca(x, 1)[:, -1]
    m = x[:, 1:4].cpu().numpy(); d = x[:, 0].cpu().numpy(); t = tgt.cpu().numpy()
    split = L.row_y[0] - L.r - 1
    ok_r = ok_b = ok_d = 0
    for b in range(bs):
        rect = np.abs(t[b, 1:]) > 0.5; rect[:, :split] = False
        band = np.abs(t[b, 1:]) > 0.5; band[:, split:] = False
        ok_r += float(np.abs(m[b][rect] - t[b, 1:][rect]).mean() < 0.3) if rect.any() else 1.0
        ok_b += float(np.abs(m[b][band] - t[b, 1:][band]).mean() < 0.3) if band.any() else 1.0
        ok_d += float((L.decode_bits(d[b]) == L.decode_bits(t[b, 0])).all())
    return ok_r / bs, ok_b / bs, ok_d / bs


@torch.no_grad()
def snapshot(path, step):
    rows = []
    for n, seed in ((2, 7), (1, step + 1), (3, step + 2)):
        L, x, tgt, _ = make_batch(1, n, random.Random(seed))
        frames = [x[0, :4].cpu().numpy()]
        for _ in range(args.k):
            x = nca(x, 1)[:, -1]; frames.append(x[0, :4].cpu().numpy())
        rows.append((frames, tgt[0].cpu().numpy(), L))
    T = max(len(r[0]) for r in rows); imgs = []
    for t in range(T):
        pr = []
        for frames, tg, L in rows:
            f = frames[min(t, len(frames) - 1)]; sep = np.full((L.H, 3, 3), 40, np.uint8)
            pr.append(np.concatenate([TileLayoutV9.to_rgb(frames[0][0], frames[0][1:]), sep,
                                      TileLayoutV9.to_rgb(f[0], f[1:]), sep,
                                      TileLayoutV9.to_rgb(tg[0], tg[1:])], 1))
        W = max(r.shape[1] for r in pr); canvas = np.full((sum(r.shape[0] for r in pr) + 3 * (len(pr) - 1), W, 3), 40, np.uint8)
        y = 0
        for r in pr:
            canvas[y: y + r.shape[0], : r.shape[1]] = r; y += r.shape[0] + 3
        im = Image.fromarray(canvas); imgs.append(im.resize((im.width * 4, im.height * 4), Image.NEAREST))
    imgs[0].save(str(path), save_all=True, append_images=imgs[1:], duration=70, loop=0)


tlog = (run_dir / "train.jsonl").open("a"); elog = (run_dir / "metrics.jsonl").open("a")
best = -1.0
bar = tqdm(range(args.steps), dynamic_ncols=True)
for step in bar:
    n = rng.randint(args.stack_min, args.stack_max)
    L, x, tgt, _ = make_batch(args.batch_size, n, rng)
    seq = nca(x, args.k)
    loss = losses(seq, tgt)
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(nca.parameters(), float("inf")).item()
    opt.step()
    if step % args.log_every == 0:
        tlog.write(json.dumps({"step": step, "loss": float(loss), "grad_norm": gn, "stack": n}) + "\n")
        if step % (args.log_every * 20) == 0:
            tlog.flush()
    if step % args.eval_every == 0:
        rec = {"step": step, "loss": float(loss), "grad_norm": gn}
        for m in range(args.stack_min, args.stack_max + 1):
            r, b, d = evaluate(m, args.eval_batch)
            rec[f"rect_S{m}"] = round(r, 4); rec[f"band_S{m}"] = round(b, 4); rec[f"grid_S{m}"] = round(d, 4)
        score = float(np.mean([rec[f"rect_S{m}"] for m in range(args.stack_min, args.stack_max + 1)]))
        rec["score"] = score; elog.write(json.dumps(rec) + "\n"); elog.flush()
        if score > best:
            best = score; torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)}, run_dir / "nca_best.pt")
        bar.set_description(f"loss {float(loss):.4f} rect {score:.2f} "
                            + " ".join(f"S{m}:b{rec[f'band_S{m}']:.2f}/g{rec[f'grid_S{m}']:.2f}" for m in range(args.stack_min, args.stack_max + 1)))
    if args.ckpt_every and step % args.ckpt_every == 0:
        torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)}, run_dir / f"nca_{step:07d}.pt")
    if args.snapshot_every and step % args.snapshot_every == 0:
        torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)}, run_dir / "nca_latest.pt")
        snapshot(run_dir / f"snapshot_{step:07d}.gif", step)
torch.save({"nca": nca.state_dict(), "step": args.steps, "config": vars(args)}, run_dir / "nca_last.pt")
print("done:", run_dir)
