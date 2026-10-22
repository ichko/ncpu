#!/usr/bin/env python3
"""v15 -- the TAPE machine: the program is in the picture and loops forever.

The picture is a vertical torus. Memory is the v12 register grid; above and below
it runs a ring of 3 px slots (2 px band + 1 px gap). Each band is one instruction:
over every register column it carries v12's stamp values in channels 1-3
(A = 1, B = 1, R = +1 SUB / -1 SUBLEQ), squashed to 2 px tall.

    ring order, moving down:  top slot ... HEAD (just above memory) | memory |
                              first slot below ... bottom slot -> wraps to top

One cycle is LOAD (--load-steps) then EXECUTE (--exec-steps):

    load     the band on the HEAD grows into its register columns (the stamp,
             5 px wide = exactly the bit squares), the previous instruction's
             stamp shrinks out into the first slot below, and every other band
             moves down one slot. The tape moves ONLY here.
    execute  v12's job: memory computes under the stamp. The tape stands still.

One instruction is always loaded, so an n-instruction program has exactly n - 1
slots and the ring is always full: the height is a function of the program,
H = (n - 1) * (band + gap) + gap + memory bits, and after n cycles the picture is
back where it started. No empty slots, no NOPs, and the pitch is the same 3 px all
the way round, across the wrap too. Memory sits near the bottom with --below slots
under it (fewer if the program is short). Memory is 4 registers x 4 bits by
default (--regs can draw a range). A run may start from any valid state:
any instruction loaded.

Supervision, per cycle:

    end of load   ch 1-3 everywhere = bands in their new slots + the new stamp
                  (memory is NOT scored: execution may already have started)
    end of exec   ch 0 everywhere = memory after the instruction;
                  ch 1-3 everywhere = bands unchanged, stamp still loaded
                  (it only leaves during the next load)

Hidden channels are never scored. Each optimiser step rolls out --cycles cycles
(default 3 = 120 steps) in one graph; further depth comes from a correctness-GATED
replay pool mixed with fresh starts.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.nca import NeuralCA  # noqa: E402
from ncpu.program import BIT0, BIT1, TileLayoutV8, apply_op  # noqa: E402

OPS = ("SUB", "SUBLEQ")

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=2_000_000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", default="tape")
# screen
ap.add_argument("--regs", default="4,4", help="registers: min,max (drawn per step)")
ap.add_argument("--word-bits", type=int, default=4)
ap.add_argument("--radius", type=int, default=2)
ap.add_argument("--pitch", type=int, default=6)
ap.add_argument("--instrs", default="3,12", help="program length: min,max (drawn per step)")
ap.add_argument("--band", type=int, default=2, help="band height, px")
ap.add_argument("--gap", type=int, default=1, help="gap between bands (and above memory), px")
ap.add_argument("--below", type=int, default=2, help="slots under memory (they hold bands too)")
# model
ap.add_argument("--channels", type=int, default=16)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--kernel", type=int, default=7)
ap.add_argument("--fire-rate", type=float, default=0.5)
ap.add_argument("--init-from", default="runs/STAMP_20261018_101454_v12_s1/nca_latest.pt",
                help="warm start (v12 already computes under a stamp); '' = from scratch")
# cycle
ap.add_argument("--load-steps", type=int, default=20)
ap.add_argument("--exec-steps", type=int, default=20)
ap.add_argument("--tail", type=int, default=1,
                help="frames scored at each checkpoint, ending at it (1 = the checkpoint only)")
ap.add_argument("--load-weight", type=float, default=1.0)
ap.add_argument("--tape-weight", type=float, default=1.0)
ap.add_argument("--cycles", type=int, default=3,
                help="cycles per rollout (3 = 120 steps): gradients cross cycle boundaries, so "
                     "hidden state has to stay usable from one instruction to the next")
ap.add_argument("--boundary-noise", type=float, default=0.02)
# training
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--grad-clip", type=float, default=1.0)
# pool
ap.add_argument("--pool-size", type=int, default=4096, help="per program length")
ap.add_argument("--pool-frac", type=float, default=0.5)
ap.add_argument("--pool-gate", type=int, default=1)
# bookkeeping
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--eval-every", type=int, default=2500)
ap.add_argument("--eval-batch", type=int, default=32)
ap.add_argument("--eval-cycles", default="1,2,4,8,16,32")
ap.add_argument("--eval-instrs", type=int, default=8, help="program length for evaluation")
ap.add_argument("--eval-regs", type=int, default=4, help="registers for evaluation")
ap.add_argument("--ckpt-every", type=int, default=10000)
args = ap.parse_args()

DEV = args.device
torch.manual_seed(args.seed)
rng = random.Random(args.seed)
EVAL_CYCLES = [int(c) for c in args.eval_cycles.split(",") if c.strip()]
N_RANGE = tuple(int(v) for v in args.instrs.split(","))
R_RANGE = tuple(int(v) for v in args.regs.split(","))
K = args.load_steps + args.exec_steps
P = args.band + args.gap                              # slot pitch
assert min(N_RANGE) >= 2, "a 1-instruction program never moves"

_lay = {}


def lay(nr):
    """The v12 register layout for ``nr`` registers (its width is the picture's)."""
    if nr not in _lay:
        _lay[nr] = TileLayoutV8(n_regs=nr, word_bits=args.word_bits, r=args.radius, pitch=args.pitch)
    return _lay[nr]


C = args.channels
L0 = lay(R_RANGE[0])                                  # rows do not depend on the register count
BIT_TOP = L0.row_y[0] - L0.r                          # first bit row inside the v12 block
BIT_ROWS = L0.row_y[-1] + L0.r + 1 - BIT_TOP          # bit squares and the gaps between them


class Geom:
    """Pixel geometry of an n-instruction program with A slots above memory.
    Training uses B = min(--below, n - 1) slots under memory and the rest above.

    One instruction is always loaded, so the ring has exactly n - 1 slots and is
    always full: the program comes back to the same picture after n cycles, with no
    empty slot and no NOP. The height is therefore fixed by the program:

        H = (n - 1) * (band + gap) + gap + memory bits

    rows 0 .. A*P-1       slots 0 .. A-1 (each: gap rows, then band rows)
    rows A*P .. M0-1      gap
    rows M0 .. M1-1       memory bits
    rows M1 .. H-1        slots A .. n-2
    and row H wraps to row 0 with the same pitch. Where memory sits only rolls the
    picture, which a translation-invariant rule on a torus cannot see, so it is
    kept at one place: near the bottom.
    """

    def __init__(self, n, A):
        self.n, self.A, self.N = n, A, n - 1
        self.M0 = A * P + args.gap
        self.M1 = self.M0 + BIT_ROWS
        self.H = self.M1 + (self.N - A) * P
        self.off = self.M0 - BIT_TOP                  # where row 0 of the v12 block would sit
        self.head = (A - 1) % self.N                  # A = 0: the head is the last slot, via the wrap

    def slot_row(self, i):
        return (i * P if i < self.A else self.M1 + (i - self.A) * P) + args.gap


_geoms = {}


def geom(n, A):
    if (n, A) not in _geoms:
        _geoms[(n, A)] = Geom(n, A)
    return _geoms[(n, A)]


# ── machine state, on the CPU ───────────────────────────────────────────────
# A run is (tape, ring, inmem, regs): the instructions, the slot each band sits in
# (instruction index -> slot), which instruction is loaded in memory (or None) and
# the register values. Everything the targets need follows from it.

def sample_instr(rng, nr):
    return (rng.choice(OPS), *(rng.randrange(nr) for _ in range(3)))


def fresh_run(n, nr, rng):
    """A random VALID boundary state: any instruction j loaded (and executed), the
    rest in program order behind it with j + 1 on the head, any register values.
    Memory is at the bottom; which band is where is what varies."""
    g = geom(n, n - 1 - min(args.below, n - 1))
    tape = [sample_instr(rng, nr) for _ in range(n)]
    j = rng.randrange(n)
    ring = {(j + m) % n: (g.head - (m - 1)) % g.N for m in range(1, n)}
    regs = tuple(rng.randint(0, (1 << args.word_bits) - 1) for _ in range(nr))
    return dict(g=g, nr=nr, tape=tape, ring=ring, inmem=j, regs=regs)


def load_event(run):
    """The tape moves: the head band loads, the loaded one offloads into the first
    slot below memory, the rest move down one slot (wrapping)."""
    g, ring = run["g"], {}
    loaded = None
    for k, i in run["ring"].items():
        if i == g.head:
            loaded = k
        else:
            ring[k] = (i + 1) % g.N
    if run["inmem"] is not None:
        ring[run["inmem"]] = g.A % g.N            # A = n-1: the slot below memory is slot 0, via the wrap
    return dict(run, ring=ring, inmem=loaded)


def execute(run):
    if run["inmem"] is None:
        return run
    op, a, b, d = run["tape"][run["inmem"]]
    regs = list(run["regs"])
    regs[d] = apply_op(op, regs[a], regs[b], regs[d], args.word_bits)
    return dict(run, regs=tuple(regs))


# ── painting ────────────────────────────────────────────────────────────────

def cols(ins):
    op, a, b, d = ins
    return [(0, a, 1.0), (1, b, 1.0), (2, d, BIT0 if op == "SUBLEQ" else BIT1)]


def memory(run):
    g = run["g"]
    L = lay(run["nr"])
    out = np.zeros((g.H, L.W), np.float32)
    out[g.M0:g.M1] = L.data_screen(run["regs"])[BIT_TOP:BIT_TOP + BIT_ROWS]
    return out


def instr_channels(run):
    """Bands in their slots, plus the loaded instruction grown over its columns
    (5 px wide = exactly the bit squares)."""
    g = run["g"]
    L = lay(run["nr"])
    out = np.zeros((3, g.H, L.W), np.float32)
    for k, i in run["ring"].items():
        y = g.slot_row(i)
        for ch, reg, v in cols(run["tape"][k]):
            out[ch, y:y + args.band, L.col_x[reg] - L.r:L.col_x[reg] + L.r + 1] = v
    if run["inmem"] is not None:
        for ch, reg, v in cols(run["tape"][run["inmem"]]):
            out[ch, g.M0:g.M1, L.col_x[reg] - L.r:L.col_x[reg] + L.r + 1] = v
    return out


def start_state(runs):
    """t = 0: memory, bands and the loaded stamp; hidden zero."""
    x = np.zeros((len(runs), C, runs[0]["g"].H, lay(runs[0]["nr"]).W), np.float32)
    for i, run in enumerate(runs):
        x[i, 0] = memory(run)
        x[i, 1:4] = instr_channels(run)
    return torch.from_numpy(x).to(DEV)


def cycle_targets(runs):
    """Advance every run one cycle; return the new runs and the targets."""
    loaded = [load_event(r) for r in runs]
    done = [execute(r) for r in loaded]
    t_load = torch.from_numpy(np.stack([instr_channels(r) for r in loaded])).to(DEV)
    t_mem = torch.from_numpy(np.stack([memory(r) for r in done])).to(DEV)
    t_tape = torch.from_numpy(np.stack([instr_channels(r) for r in done])).to(DEV)
    return done, t_load, t_mem, t_tape


# ── reading the picture back ────────────────────────────────────────────────

_dec = {}


def decode(runs, ch0):
    """(B, H, W) -> (B, regs, bits) in {0, 1}: sign of each bit square's mean."""
    idx = []
    for run in runs:
        g, L = run["g"], lay(run["nr"])
        key = (g.n, g.A, run["nr"])
        if key not in _dec:
            _dec[key] = [[(g.off + L.row_y[b] + dy) * L.W + L.col_x[c] + dx
                          for dy in range(-L.r, L.r + 1) for dx in range(-L.r, L.r + 1)]
                         for c in range(run["nr"]) for b in range(args.word_bits)]
        idx.append(_dec[key])
    idx = torch.tensor(idx, device=DEV)                           # (B, regs*bits, 25)
    flat = ch0.flatten(1)
    m = torch.gather(flat, 1, idx.flatten(1)).view(idx.shape).mean(-1)
    return (m > 0).long().view(len(runs), -1, args.word_bits)


def want_bits(runs):
    return torch.tensor([[[(v >> (args.word_bits - 1 - b)) & 1 for b in range(args.word_bits)]
                          for v in r["regs"]] for r in runs], device=DEV)


def landed(x_load, x_end, t_load, t_tape, runs):
    """Memory exactly right at the end of execute, and every instruction pixel (bands
    and stamp) within 0.5 of its target at both checkpoints."""
    mem_ok = (decode(runs, x_end[:, 0]) == want_bits(runs)).flatten(1).all(1)
    load_ok = (x_load[:, 1:4] - t_load).abs().amax(dim=(1, 2, 3)) < 0.5
    tape_ok = (x_end[:, 1:4] - t_tape).abs().amax(dim=(1, 2, 3)) < 0.5
    return (mem_ok & load_ok & tape_ok).tolist()


# ── replay pool, one per (program length, registers) = picture shape ───────

_pool = {}


def pool_push(key, x_end, runs, depth, ok):
    p = _pool.setdefault(key, [])
    xs = x_end.detach().cpu().numpy()
    k = 0
    for i, run in enumerate(runs):
        if args.pool_gate and not ok[i]:
            continue
        p.append((xs[i], run, depth[i] + args.cycles))
        k += 1
    if len(p) > args.pool_size:
        del p[: len(p) - args.pool_size]
    return k


def pool_draw(key, bs, rng):
    p = _pool.get(key, [])
    if len(p) < bs:
        return None
    ent = [p[rng.randrange(len(p))] for _ in range(bs)]
    return (torch.from_numpy(np.stack([e[0] for e in ent])).to(DEV),
            [e[1] for e in ent], [e[2] for e in ent])


# ── model ───────────────────────────────────────────────────────────────────

nca = NeuralCA(channels=C, hidden_channels=args.hidden, fire_rate=args.fire_rate,
               alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
               perception="static_sobel", padding_type="circular", read_only_dims=[],
               state_bound=10.0).to(DEV)
root = Path(__file__).resolve().parent.parent
if args.init_from:
    sd = torch.load(root / args.init_from, map_location=DEV, weights_only=False)["nca"]
    print("warm start:", args.init_from, nca.load_state_dict(sd, strict=False))
opt = torch.optim.Adam(nca.parameters(), lr=args.lr)

run_dir = root / "runs" / f"TAPE_{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}_s{args.seed}"
run_dir.mkdir(parents=True)
(run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
print(f"run {run_dir.name}: {R_RANGE} registers, programs of {N_RANGE} instructions, "
      f"load {args.load_steps} + exec {args.exec_steps}")


def scored(seq, t, tail):
    return seq[:, t - tail + 1:t + 1]


def per_sample(a, b):
    return ((a - b) ** 2).flatten(1).mean(1)


def train_cycle(x, runs):
    """``--cycles`` cycles in one graph, every checkpoint scored; landed only if every
    cycle landed. Also returns per-op stats of the batch for the log."""
    if args.boundary_noise:
        x = x.clone()
        x[:, 4:] = x[:, 4:] + torch.randn_like(x[:, 4:]) * args.boundary_noise
    ll = ml = tl = 0.0
    ok = [True] * len(runs)
    st = {op: {"instr": [], "mem": [], "bit": [], "reg": []} for op in OPS}
    for _ in range(args.cycles):
        runs, t_load, t_mem, t_tape = cycle_targets(runs)
        seq = nca(x, K)                               # (B, K+1, C, H, W)
        fl = scored(seq, args.load_steps, args.tail)
        fe = scored(seq, K, args.tail)
        li = per_sample(fl[:, :, 1:4], t_load[:, None]) * args.load_weight
        lm = per_sample(fe[:, :, 0], t_mem[:, None])
        lt = per_sample(fe[:, :, 1:4], t_tape[:, None]) * args.tape_weight
        ll, ml, tl = ll + li.mean(), ml + lm.mean(), tl + lt.mean()
        xe = seq[:, -1].detach()
        ok = [a and b for a, b in zip(ok, landed(seq[:, args.load_steps].detach(), xe, t_load, t_tape, runs))]
        hit = decode(runs, xe[:, 0]) == want_bits(runs)
        for i, r in enumerate(runs):
            op, _, _, d = r["tape"][r["inmem"]]
            st[op]["instr"].append(float(li[i] + lt[i])); st[op]["mem"].append(float(lm[i]))
            st[op]["bit"].append(float(hit[i, d].float().mean())); st[op]["reg"].append(float(hit[i, d].all()))
        x = seq[:, -1]
    ll, ml, tl = ll / args.cycles, ml / args.cycles, tl / args.cycles
    loss = ll + ml + tl
    loss.backward()
    ops = {f"{k}_{op.lower()}": round(float(np.mean(v)), 6) for op in OPS for k, v in st[op].items() if v}
    return float(loss), float(ll), float(ml), float(tl), x.detach(), runs, ok, ops


@torch.no_grad()
def evaluate(cycles, bs):
    """Fresh tapes run ``cycles`` cycles; exact = memory right at the end of every
    execute AND every instruction pixel right at both checkpoints. prog = memory
    alone right at the end of every execute (the program computes correctly)."""
    erng = random.Random(991 + cycles)
    runs = [fresh_run(args.eval_instrs, args.eval_regs, erng) for _ in range(bs)]
    x = start_state(runs)
    exact = torch.ones(bs, dtype=torch.bool, device=DEV)
    prog = exact.clone()
    per, per_prog = [], []
    for _ in range(cycles):
        runs, t_load, t_mem, t_tape = cycle_targets(runs)
        seq = nca(x, K)
        exact &= torch.tensor(landed(seq[:, args.load_steps], seq[:, -1], t_load, t_tape, runs),
                              device=DEV)
        prog &= (decode(runs, seq[:, -1, 0]) == want_bits(runs)).flatten(1).all(1)
        x = seq[:, -1]
        per.append(float(exact.float().mean()))
        per_prog.append(float(prog.float().mean()))
    return per, per_prog


tlog = (run_dir / "train.jsonl").open("a")
elog = (run_dir / "metrics.jsonl").open("a")
best = -1.0
bar = tqdm(range(args.steps), dynamic_ncols=True)
for step in bar:
    n = rng.randint(*N_RANGE)
    nr = rng.randint(*R_RANGE)
    drawn = pool_draw((n, nr), args.batch_size, rng) if rng.random() < args.pool_frac else None
    if drawn:
        x, runs, depth = drawn
    else:
        runs = [fresh_run(n, nr, rng) for _ in range(args.batch_size)]
        x, depth = start_state(runs), [0] * args.batch_size

    opt.zero_grad(set_to_none=True)
    loss, ll, ml, tl, x_end, done, ok, ops = train_cycle(x, runs)
    gn = torch.nn.utils.clip_grad_norm_(nca.parameters(), args.grad_clip or float("inf")).item()
    opt.step()
    pushed = pool_push((n, nr), x_end, done, depth, ok)

    if step % args.log_every == 0:
        tlog.write(json.dumps({
            "step": step, "loss": round(loss, 6), "load": round(ll, 6), "mem": round(ml, 6),
            "tape": round(tl, 6), "grad_norm": round(gn, 4), "n": n, "from_pool": bool(drawn),
            "mean_depth": round(float(np.mean(depth)), 2), "batch_ok": round(float(np.mean(ok)), 3),
            "pool": sum(len(p) for p in _pool.values()), "pushed": pushed, **ops}) + "\n")
        if step % (args.log_every * 20) == 0:
            tlog.flush()

    if step % args.eval_every == 0:
        per, per_prog = evaluate(max(EVAL_CYCLES), args.eval_batch)
        rec = {"step": step, "loss": round(loss, 6),
               "pool": sum(len(p) for p in _pool.values())}
        for c in EVAL_CYCLES:
            rec[f"cyc{c}"] = round(per[c - 1], 4)
        for c in EVAL_CYCLES:
            rec[f"prog{c}"] = round(per_prog[c - 1], 4)
        score = float(np.mean([rec[f"cyc{c}"] for c in EVAL_CYCLES]))
        rec["score"] = round(score, 4)
        elog.write(json.dumps(rec) + "\n"); elog.flush()
        ck = {"nca": nca.state_dict(), "step": step, "config": vars(args)}
        torch.save(ck, run_dir / "nca_latest.pt")
        if score > best:
            best = score
            torch.save(ck, run_dir / "nca_best.pt")
        bar.set_description(f"loss {loss:.4f} pool {rec['pool']} "
                            + " ".join(f"c{c}:{rec[f'cyc{c}']:.2f}" for c in EVAL_CYCLES)
                            + " prog " + " ".join(f"{rec[f'prog{c}']:.2f}" for c in EVAL_CYCLES))

    if args.ckpt_every and step % args.ckpt_every == 0:
        torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)},
                   run_dir / f"nca_{step:07d}.pt")
torch.save({"nca": nca.state_dict(), "step": args.steps, "config": vars(args)},
           run_dir / "nca_last.pt")
print("done:", run_dir)
