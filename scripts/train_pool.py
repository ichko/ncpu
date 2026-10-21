#!/usr/bin/env python3
"""Pool training: sample stage-start states from a replay buffer.

A program of ``--n-instr`` instructions has 2n-1 stages.  Backpropagating
through all of them is impossible, so instead we keep a POOL of states.  Each
step: draw a batch of pool entries (each is "some program, paused just before
stage s"), run ONE stage, score it against that stage's target, and write the
produced state back into the pool with s+1.  Entries whose program is finished
are reseeded at t=0.

Two things fall out of this:

* the rule trains on states deep inside long programs -- instruction 8, 20,
  100 -- while never unrolling more than one stage;
* the states it trains on are its OWN slightly-wrong outputs, so the target
  asks it to recover.  That is the error-contraction property chaining needs:
  without it, per-instruction error compounds and nothing beyond 2-3
  instructions can work.

    uv run python scripts/train_pool.py --device cuda --init-from runs/.../nca_latest.pt
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
from ncpu.program import TileLayoutV9, run_program, sample_program, sample_regs  # noqa: E402

ap = ArgumentParser()
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=1_000_000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--n-instr", type=int, default=8, help="program length held in the pool")
ap.add_argument("--pool-size", type=int, default=512)
ap.add_argument("--reseed-frac", type=float, default=0.06,
                help="fraction of each batch replaced with a fresh t=0 state (keeps the pool honest)")
ap.add_argument("--reset-worst", type=int, default=1,
                help="reset this many of the worst-scoring entries per step (stops unrecoverable states)")
ap.add_argument("--k", type=int, default=60, help="CA steps per stage (same for LOAD and EXEC)")
ap.add_argument("--tail", type=int, default=20, help="supervise the last N frames of the stage")
ap.add_argument("--channels", type=int, default=16)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--kernel", type=int, default=7)
ap.add_argument("--fire-rate", type=float, default=0.5)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--init-from", default=None)
ap.add_argument("--widen", action="store_true", help="net2net-widen the checkpoint into --hidden")
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--eval-batch", type=int, default=16)
ap.add_argument("--eval-lens", default="1,2,3,4,6,8")
ap.add_argument("--ckpt-every", type=int, default=5000)
ap.add_argument("--snapshot-every", type=int, default=2500)
ap.add_argument("--tag", default="pool")
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed); rng = random.Random(args.seed)
DEV = args.device
OPS = ("SUB", "SUBLEQ", "NOP")
EVAL_LENS = [int(x) for x in args.eval_lens.split(",") if x.strip()]
run = f"POOL_{datetime.now():%Y%m%d_%H%M%S}_{args.tag}_s{args.seed}"
run_dir = Path(__file__).resolve().parent.parent / "runs" / run
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

L = TileLayoutV9(n_instr=args.n_instr, preloaded=True)
N_STAGES = L.n_stages
print(f"run {run}: {args.n_instr} instructions = {N_STAGES} stages, grid {L.H}x{L.W}, "
      f"pool {args.pool_size}, K={args.k}")

_eval_layouts = {}


def elayout(n):
    if n not in _eval_layouts:
        _eval_layouts[n] = TileLayoutV9(n_instr=n, preloaded=True)
    return _eval_layouts[n]


# ── model ───────────────────────────────────────────────────────────────────

def widen_state(old, new, seed=0):
    g = torch.Generator().manual_seed(seed); out = {}
    for k, v in new.items():
        o = old[k]
        if o.shape == v.shape:
            out[k] = o; continue
        t = v.clone()
        if k == "rule.rule.0.weight":
            t[: o.shape[0]] = o; t[o.shape[0]:] = torch.randn(t.shape[0] - o.shape[0], *o.shape[1:], generator=g) * 0.01
        elif k == "rule.rule.0.bias":
            t[: o.shape[0]] = o; t[o.shape[0]:] = 0.0
        elif k == "rule.rule.2.weight":
            t[:, : o.shape[1]] = o; t[:, o.shape[1]:] = 0.0
        else:
            raise ValueError(k)
        out[k] = t; print(f"  widened {k}: {tuple(o.shape)} -> {tuple(v.shape)}")
    return out


nca = NeuralCA(channels=args.channels, hidden_channels=args.hidden, fire_rate=args.fire_rate,
               alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
               perception="static_sobel", padding_type="zeros", read_only_dims=[], state_bound=10.0).to(DEV)
if args.init_from:
    old = torch.load(args.init_from, map_location=DEV, weights_only=False)["nca"]
    nca.load_state_dict(widen_state(old, nca.state_dict()) if args.widen else old)
    print(f"  warm-started from {args.init_from}")
opt = torch.optim.Adam(nca.parameters(), lr=args.lr)
print(f"  params {sum(p.numel() for p in nca.parameters() if p.requires_grad):,}")


# ── the pool ────────────────────────────────────────────────────────────────

class Pool:
    """Entries are (state, program, regs, stage).  Stage s means: about to run
    stage s, so the state should currently look like ``mask/data at s``."""

    def __init__(self, size):
        self.x = torch.zeros(size, args.channels, L.H, L.W, device=DEV)
        self.prog = [None] * size; self.regs = [None] * size; self.stage = [0] * size
        for i in range(size):
            self.reseed(i)

    def reseed(self, i):
        p = sample_program(args.n_instr, 4, ops=OPS, copy_frac=0.0, rng=rng)
        g = sample_regs(4, 4, rng=rng)
        x = torch.zeros(args.channels, L.H, L.W, device=DEV)
        x[0] = torch.from_numpy(L.data_screen(g)).to(DEV)
        x[1:4] = torch.from_numpy(L.mask_channels_at(p, 0)).to(DEV)
        self.x[i] = x; self.prog[i] = p; self.regs[i] = g; self.stage[i] = 0

    def targets(self, idx):
        """(B, 4, H, W): the picture each entry must reach after its current stage."""
        t = np.zeros((len(idx), 4, L.H, L.W), np.float32)
        for j, i in enumerate(idx):
            s = self.stage[i] + 1
            t[j, 0] = L.paint(self.prog[i], self.regs[i], s)
            t[j, 1:] = L.mask_channels_at(self.prog[i], s)
        return torch.from_numpy(t).to(DEV)

    def advance(self, idx, x_new, per_loss):
        for j, i in enumerate(idx):
            self.x[i] = x_new[j].detach()
            self.stage[i] += 1
            if self.stage[i] >= N_STAGES:
                self.reseed(i)
        if args.reset_worst:                       # drop the least recoverable states
            for j in torch.topk(per_loss, min(args.reset_worst, len(idx))).indices.tolist():
                self.reseed(idx[j])


pool = Pool(args.pool_size)


def batch_indices():
    idx = rng.sample(range(args.pool_size), args.batch_size)
    for j in range(len(idx)):                      # a few fresh t=0 states every step
        if rng.random() < args.reseed_frac:
            pool.reseed(idx[j])
    return idx


# ── evaluation: free-running from t=0, the real task ────────────────────────

@torch.no_grad()
def evaluate(n, bs):
    Le = elayout(n); r = random.Random(1234 + n); ok = 0
    for _ in range(bs):
        p = sample_program(n, 4, ops=OPS, copy_frac=0.0, rng=r); g = sample_regs(4, 4, rng=r)
        x = torch.zeros(1, args.channels, Le.H, Le.W, device=DEV)
        x[:, 0] = torch.from_numpy(Le.data_screen(g)).to(DEV)
        x[:, 1:4] = torch.from_numpy(Le.mask_channels_at(p, 0)).to(DEV)
        for _ in range(Le.n_stages):
            x = nca(x, args.k)[:, -1]
        want = np.array([[int(c) for c in format(v, "04b")] for v in run_program(g, p, 4)[-1]])
        ok += float((Le.decode_bits(x[0, 0].cpu().numpy()) == want).all())
    return ok / bs


@torch.no_grad()
def snapshot(path, step):
    Le = elayout(min(4, args.n_instr)); r = random.Random(step + 1)
    p = sample_program(Le.n_instr, 4, ops=OPS, copy_frac=0.0, rng=r); g = sample_regs(4, 4, rng=r)
    x = torch.zeros(1, args.channels, Le.H, Le.W, device=DEV)
    x[:, 0] = torch.from_numpy(Le.data_screen(g)).to(DEV); x[:, 1:4] = torch.from_numpy(Le.mask_channels_at(p, 0)).to(DEV)
    frames = [x[0, :4].cpu().numpy()]
    for _ in range(Le.n_stages * args.k):
        x = nca(x, 1)[:, -1]; frames.append(x[0, :4].cpu().numpy())
    tgt_d = Le.paint(p, g, Le.n_stages); tgt_m = Le.mask_channels_at(p, Le.n_stages)
    imgs = []
    for t in range(0, len(frames), 3):
        f = frames[t]; sep = np.full((Le.H, 3, 3), 40, np.uint8)
        row = np.concatenate([TileLayoutV9.to_rgb(frames[0][0], frames[0][1:]), sep,
                              TileLayoutV9.to_rgb(f[0], f[1:]), sep,
                              TileLayoutV9.to_rgb(tgt_d, tgt_m)], 1)
        im = Image.fromarray(row); imgs.append(im.resize((im.width * 4, im.height * 4), Image.NEAREST))
    imgs[0].save(str(path), save_all=True, append_images=imgs[1:], duration=60, loop=0)


# ── loop ────────────────────────────────────────────────────────────────────

tlog = (run_dir / "train.jsonl").open("a"); elog = (run_dir / "metrics.jsonl").open("a")
best = -1.0
bar = tqdm(range(args.steps), dynamic_ncols=True)
for step in bar:
    idx = batch_indices()
    x = pool.x[idx].clone()
    tgt = pool.targets(idx)
    seq = nca(x, args.k)
    tail = seq[:, -args.tail:, :4]
    per = ((tail - tgt[:, None]) ** 2).mean(dim=(1, 2, 3, 4))       # one loss per entry
    loss = per.mean()
    opt.zero_grad(set_to_none=True); loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(nca.parameters(), float("inf")).item()
    opt.step()
    pool.advance(idx, seq[:, -1], per.detach())

    if step % args.log_every == 0:
        tlog.write(json.dumps({"step": step, "loss": float(loss), "grad_norm": gn,
                               "mean_stage": float(np.mean(pool.stage))}) + "\n")
        if step % (args.log_every * 20) == 0:
            tlog.flush()
    if step % args.eval_every == 0:
        rec = {"step": step, "loss": float(loss), "grad_norm": gn, "mean_stage": float(np.mean(pool.stage))}
        for m in EVAL_LENS:
            rec[f"exact_L{m}"] = round(evaluate(m, args.eval_batch), 4)
        score = float(np.mean([rec[f"exact_L{m}"] for m in EVAL_LENS]))
        rec["score"] = score; elog.write(json.dumps(rec) + "\n"); elog.flush()
        if score > best:
            best = score; torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)}, run_dir / "nca_best.pt")
        bar.set_description(f"loss {float(loss):.4f} stage~{rec['mean_stage']:.1f} "
                            + " ".join(f"L{m}:{rec[f'exact_L{m}']:.2f}" for m in EVAL_LENS))
    if args.ckpt_every and step % args.ckpt_every == 0:
        torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)}, run_dir / f"nca_{step:07d}.pt")
    if args.snapshot_every and step % args.snapshot_every == 0:
        torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)}, run_dir / "nca_latest.pt")
        snapshot(run_dir / f"snapshot_{step:07d}.gif", step)
torch.save({"nca": nca.state_dict(), "step": args.steps, "config": vars(args)}, run_dir / "nca_last.pt")
print("done:", run_dir)
