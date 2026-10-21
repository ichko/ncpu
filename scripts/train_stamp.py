#!/usr/bin/env python3
"""v12 -- the STAMPED machine: memory that persists, instructions we paint.

Everything the v9 stack had to learn about transporting a program is gone. There
is no instruction stack, no LOAD phase, no window, no program: each cycle the
HARNESS stamps the three v8 rectangles (A over the first operand's column, B over
the second's, R over the destination, R's sign = SUB / SUBLEQ) straight onto the
memory grid, the rule computes in place, and we stamp the next one.

    cycle:   x[1:4] <- rectangles(instr)        (assign, by us)
             K steps                            (the rule computes)
             supervise the last --tail frames:
                 ch 0    = memory after instr
                 ch 1-3  = 0      the rule ERASES its own stamp
                 ch 4+   = 0      hidden drained

The boundary form is therefore "memory, nothing else", identical after every
cycle and independent of which instruction just ran. Cycle N is the same problem
as cycle 1 by construction, which is the property the v9 boundary never had: its
end-of-cycle state still carried the instruction that had just executed.

Only ONE cycle is ever rolled out per optimiser step. Depth comes from the replay
pool instead: a boundary state the rule produced goes back in, is drawn later,
gets a fresh instruction stamped on it, and is trained again. The pool is GATED on
correctness -- a state enters only if that cycle actually landed right. Ungated
pooling is what destroyed the v9 run: one slip let broken states in, their targets
were unreachable, and the rule chased them down.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA  # noqa: E402
from ncpu.program import Instr, TileLayoutV8, apply_op  # noqa: E402

OPS = ("SUB", "SUBLEQ")

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=2_000_000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="stamp")
# screen
ap.add_argument("--regs", type=int, default=4)
ap.add_argument("--word-bits", type=int, default=4)
ap.add_argument("--radius", type=int, default=2)
ap.add_argument("--pitch", type=int, default=6)
# model (defaults = the v8 ALU that reached 99.5% exact on one instruction)
ap.add_argument("--channels", type=int, default=16)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--kernel", type=int, default=7)
ap.add_argument("--fire-rate", type=float, default=0.5)
ap.add_argument("--init-from", default=None, help="warm start, e.g. the v8 ALU checkpoint")
# cycle
ap.add_argument("-K", "--k-cycle", type=int, default=60, help="NCA steps per cycle")
ap.add_argument("--tail", type=int, default=20,
                help="supervised frames at the END of the cycle; the boundary must be a "
                     "fixed point, so every frame in the tail is scored against the SAME "
                     "static target")
ap.add_argument("--mask-weight", type=float, default=1.0, help="weight on erasing the stamp")
ap.add_argument("--hidden-weight", type=float, default=1.0, help="weight on draining hidden")
ap.add_argument("--boundary-noise", type=float, default=0.02,
                help="hidden noise injected at the boundary: the rest form has to be an "
                     "ATTRACTOR, not just a point the rule happens to hit")
ap.add_argument("--nop-frac", type=float, default=0.0,
                help="fraction of cycles with no stamp at all (memory must simply hold)")
ap.add_argument("--balance-subleq", type=int, default=1,
                help="resample operands so ~half the SUBLEQs refuse to write")
# training
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm (0 = measure only)")
# pool
ap.add_argument("--pool-size", type=int, default=4096)
ap.add_argument("--pool-frac", type=float, default=0.5,
                help="fraction of batches drawn from the pool; the rest start fresh at t=0 "
                     "so the rule cannot forget the true start state")
ap.add_argument("--pool-gate", type=int, default=1,
                help="only pool states whose cycle came out exactly right (see module docstring)")
ap.add_argument("--pool-max-depth", type=int, default=0,
                help="drop pooled states deeper than this many cycles (0 = unlimited)")
# bookkeeping
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--eval-every", type=int, default=2500)
ap.add_argument("--eval-batch", type=int, default=64)
ap.add_argument("--eval-cycles", default="1,2,4,8,16,32")
ap.add_argument("--ckpt-every", type=int, default=10000)
args = ap.parse_args()

DEV = args.device
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
EVAL_CYCLES = [int(c) for c in args.eval_cycles.split(",") if c.strip()]

L = TileLayoutV8(n_regs=args.regs, word_bits=args.word_bits, r=args.radius, pitch=args.pitch)
C, H, W = args.channels, L.H, L.W

root = Path(__file__).resolve().parent.parent
run_dir = root / "runs" / f"STAMP_{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}_s{args.seed}"
run_dir.mkdir(parents=True)
(run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

nca = NeuralCA(channels=C, hidden_channels=args.hidden, fire_rate=args.fire_rate,
               alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
               perception="static_sobel", padding_type="zeros", read_only_dims=[],
               state_bound=10.0).to(DEV)
if args.init_from:
    sd = torch.load(args.init_from, map_location=DEV, weights_only=False)["nca"]
    own = nca.state_dict()
    bad = [k for k in own if k in sd and own[k].shape != sd[k].shape]
    if bad:
        print("shape mismatch, loading what fits:", bad)
        sd = {k: v for k, v in sd.items() if k not in bad}
    print("warm start:", args.init_from, nca.load_state_dict(sd, strict=False))
opt = torch.optim.Adam(nca.parameters(), lr=args.lr)
print(f"run {run_dir.name}: screen {H}x{W}, K {args.k_cycle}, tail {args.tail}, "
      f"params {sum(p.numel() for p in nca.parameters()):,}")


# ── one instruction ─────────────────────────────────────────────────────────

def sample_instr(regs, rng):
    """A random instruction over the CURRENT memory. ``balance_subleq`` picks
    operands so SUBLEQ writes about half the time -- unbalanced, the rule learns
    the majority verdict and ignores the comparison."""
    if args.nop_frac and rng.random() < args.nop_frac:
        return Instr("NOP", 0, 0, 0)
    for _ in range(16):
        op = rng.choice(list(OPS))
        a, b, dst = (rng.randrange(args.regs) for _ in range(3))
        if op != "SUBLEQ" or a == b or not args.balance_subleq:
            return Instr(op, a, b, dst)
        r = (regs[a] - regs[b]) & ((1 << args.word_bits) - 1)
        if (0 < r < (1 << (args.word_bits - 1))) == (rng.random() < 0.5):
            return Instr(op, a, b, dst)
    return Instr(op, a, b, dst)


def after(regs, ins):
    nxt = list(regs)
    nxt[ins.dst] = apply_op(ins.op, regs[ins.a], regs[ins.b], regs[ins.dst], args.word_bits)
    return tuple(nxt)


_stamp_cache = {}


def stamp_of(ins):
    key = (ins.op, ins.a, ins.b, ins.dst)
    if key not in _stamp_cache:
        _stamp_cache[key] = L.mask_channels(ins)
    return _stamp_cache[key]


def stamp(x, instrs):
    """Paint the rectangles. ASSIGN, not add: at a pooled boundary the channels
    hold whatever residue the rule left, and adding on top of it asks the rule to
    hit a target it was never shown."""
    s = torch.from_numpy(np.stack([stamp_of(i) for i in instrs])).to(x.device)
    x = x.clone()
    x[:, 1:4] = s
    return x


def fresh_batch(bs, rng):
    """t=0: memory painted, stamp painted, everything else zero."""
    regs = [tuple(rng.randint(0, (1 << args.word_bits) - 1) for _ in range(args.regs))
            for _ in range(bs)]
    x = np.zeros((bs, C, H, W), np.float32)
    for i, r in enumerate(regs):
        x[i, 0] = L.data_screen(r)
    return torch.from_numpy(x).to(DEV), regs, [0] * bs


def targets(regs_next):
    """Full-frame target for the data channel. Channels 1+ target zero, which needs
    no painting. There is no "background": a pixel that should be empty is part of
    the picture exactly like a lit one, and is scored the same."""
    return torch.from_numpy(np.stack([L.data_screen(r) for r in regs_next])).to(DEV)


# ── replay pool ─────────────────────────────────────────────────────────────
# Boundary states the rule produced, with the memory they are SUPPOSED to hold.
# There is no program to remember: the next instruction is sampled fresh against
# whatever the memory now is, so a pooled state is a complete problem on its own
# and the pool is the only thing that makes cycle N reachable.
_pool = []


def pool_push(x_end, regs_next, depth, ok):
    if not args.pool_size:
        return 0
    xs = x_end.detach().cpu().numpy()
    n = 0
    for i in range(len(regs_next)):
        if args.pool_gate and not ok[i]:
            continue                                  # GATE: broken states poison the pool
        if args.pool_max_depth and depth[i] + 1 > args.pool_max_depth:
            continue
        _pool.append((xs[i], regs_next[i], depth[i] + 1))
        n += 1
    if len(_pool) > args.pool_size:
        del _pool[: len(_pool) - args.pool_size]      # FIFO: keep the freshest states
    return n


def pool_draw(bs, rng):
    if len(_pool) < bs:
        return None
    ent = [_pool[rng.randrange(len(_pool))] for _ in range(bs)]
    x = torch.from_numpy(np.stack([e[0] for e in ent])).to(DEV)
    return x, [e[1] for e in ent], [e[2] for e in ent]


# ── one cycle ───────────────────────────────────────────────────────────────

def correct(xf, regs_next):
    """Did this cycle land? Memory exactly right AND the boundary in rest form --
    a state that decodes correctly but still carries a stamp or a hot hidden layer
    is not a state the next cycle is ever trained to start from."""
    d = xf[:, 0].cpu().numpy()
    resid = xf[:, 1:4].abs().amax(dim=(1, 2, 3)).cpu().numpy()
    hid = xf[:, 4:].abs().amax(dim=(1, 2, 3)).cpu().numpy()
    out = []
    for i, r in enumerate(regs_next):
        want = np.array([[int(c) for c in format(v, f"0{args.word_bits}b")] for v in r])
        out.append(bool((L.decode_bits(d[i]) == want).all()) and resid[i] < 0.5 and hid[i] < 0.5)
    return out


def train_cycle(x, instrs, tg_d):
    x = stamp(x, instrs)
    if args.boundary_noise:
        x = x.clone()
        x[:, 4:] = x[:, 4:] + torch.randn_like(x[:, 4:]) * args.boundary_noise
    seq = nca(x, args.k_cycle)                        # (B, K+1, C, H, W)
    fr = seq[:, -args.tail:]
    dl = ((fr[:, :, 0] - tg_d[:, None]) ** 2).mean()
    ml = (fr[:, :, 1:4] ** 2).mean() * args.mask_weight
    hl = (fr[:, :, 4:] ** 2).mean() * args.hidden_weight
    loss = dl + ml + hl
    loss.backward()
    return (float(loss.detach()), float(dl.detach()), float(ml.detach()), float(hl.detach()),
            seq[:, -1].detach())


# ── evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(cycles, bs):
    """Run ``cycles`` cycles from a fresh start, a new random instruction each time,
    and ask for exact memory at every boundary. Nothing in training ever rolls out
    more than one cycle, so this is the only metric that says whether the boundary
    really is canonical."""
    erng = random.Random(991 + cycles)
    x, regs, _ = fresh_batch(bs, erng)
    exact = np.ones(bs, bool)
    per = []
    for c in range(cycles):
        instrs = [sample_instr(regs[i], erng) for i in range(bs)]
        regs = [after(regs[i], instrs[i]) for i in range(bs)]
        x = stamp(x, instrs)
        x = nca(x, args.k_cycle)[:, -1]
        d = x[:, 0].cpu().numpy()
        bits = []
        for i, r in enumerate(regs):
            want = np.array([[int(ch) for ch in format(v, f"0{args.word_bits}b")] for v in r])
            got = L.decode_bits(d[i])
            exact[i] &= bool((got == want).all())
            bits.append(float((got == want).mean()))
        per.append((float(np.mean(exact)), float(np.mean(bits))))
    return dict(exact=per[-1][0], bits=per[-1][1],
                resid=float(x[:, 1:4].abs().mean()), hid=float(x[:, 4:].abs().mean()),
                per_cycle=[round(p[0], 3) for p in per])


# ── loop ────────────────────────────────────────────────────────────────────

tlog = (run_dir / "train.jsonl").open("a")
elog = (run_dir / "metrics.jsonl").open("a")
best = -1.0
bar = tqdm(range(args.steps), dynamic_ncols=True)
for step in bar:
    drawn = pool_draw(args.batch_size, rng) if rng.random() < args.pool_frac else None
    x, regs, depth = drawn if drawn else fresh_batch(args.batch_size, rng)
    instrs = [sample_instr(r, rng) for r in regs]
    regs_next = [after(regs[i], instrs[i]) for i in range(len(regs))]
    tg_d = targets(regs_next)

    opt.zero_grad(set_to_none=True)
    loss, dl, ml, hl, x_end = train_cycle(x, instrs, tg_d)
    gn = torch.nn.utils.clip_grad_norm_(nca.parameters(),
                                        args.grad_clip or float("inf")).item()
    opt.step()
    ok = correct(x_end, regs_next)
    pushed = pool_push(x_end, regs_next, depth, ok)

    if step % args.log_every == 0:
        tlog.write(json.dumps({
            "step": step, "loss": round(loss, 6), "grad_norm": round(gn, 4),
            "data": round(dl, 6), "mask": round(ml, 6), "hid": round(hl, 6),
            "from_pool": bool(drawn), "mean_depth": round(float(np.mean(depth)), 2),
            "batch_ok": round(float(np.mean(ok)), 3),
            "pool": len(_pool), "pushed": pushed}) + "\n")
        if step % (args.log_every * 20) == 0:
            tlog.flush()

    if step % args.eval_every == 0:
        rec = {"step": step, "loss": round(loss, 6), "pool": len(_pool),
               "pool_depth": round(float(np.mean([e[2] for e in _pool])), 2) if _pool else 0.0}
        for c in EVAL_CYCLES:
            r = evaluate(c, args.eval_batch)
            rec[f"cyc{c}"] = round(r["exact"], 4)
            rec[f"bits{c}"] = round(r["bits"], 4)
            if c == EVAL_CYCLES[0]:
                rec["resid"] = round(r["resid"], 5); rec["hid"] = round(r["hid"], 5)
            if c == max(EVAL_CYCLES):
                rec["per_cycle"] = r["per_cycle"]
        score = float(np.mean([rec[f"cyc{c}"] for c in EVAL_CYCLES]))
        rec["score"] = round(score, 4)
        elog.write(json.dumps(rec) + "\n"); elog.flush()
        torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)},
                   run_dir / "nca_latest.pt")
        if score > best:
            best = score
            torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)},
                       run_dir / "nca_best.pt")
        bar.set_description(f"loss {loss:.4f} pool {len(_pool)}/{rec['pool_depth']:.1f} "
                            + " ".join(f"c{c}:{rec[f'cyc{c}']:.2f}" for c in EVAL_CYCLES))

    if args.ckpt_every and step % args.ckpt_every == 0:
        torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)},
                   run_dir / f"nca_{step:07d}.pt")
torch.save({"nca": nca.state_dict(), "step": args.steps, "config": vars(args)},
           run_dir / "nca_last.pt")
print("done:", run_dir)
