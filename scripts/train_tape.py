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
ap.add_argument("--seamless", type=int, default=0,
                help="1: memory is nr * pitch wide, so the horizontal wrap gap equals the register gap")
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
ap.add_argument("--cycles-range", default="",
                help="lo,hi: cycles per rollout drawn per step (batch scaled by --cycles / drawn)")
ap.add_argument("--boundary-noise", type=float, default=0.02)
ap.add_argument("--visible-noise", type=float, default=0.0,
                help="noise on memory and instruction channels (0-3) at the start of each "
                     "rollout, so the model learns to pull a drifted picture back")
# training
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--batch-size", type=int, default=16)
ap.add_argument("--grad-clip", type=float, default=1.0)
# pool
ap.add_argument("--pool-size", type=int, default=4096, help="per program length")
ap.add_argument("--pool-gb", type=float, default=0,
                help="RAM budget for the whole pool: each (length, registers) pool gets an equal share")
ap.add_argument("--pool-frac", type=float, default=0.5)
ap.add_argument("--pool-fp16", type=int, default=0, help="store pool states as float16 (RAM)")
ap.add_argument("--deep-every", type=int, default=0,
                help="every this many steps, take a batch from the pool and run it on WITHOUT "
                     "gradients for --deep-advance cycles; memory-correct survivors go back in. "
                     "Training then replays states hundreds of cycles deep")
ap.add_argument("--deep-advance", default="10,60", help="cycles per deep advance, drawn uniformly")
ap.add_argument("--tape-noise", type=float, default=-1,
                help="noise std on the instruction channels 1-3 alone (-1 = same as --visible-noise)")
ap.add_argument("--pool-evict", default="oldest", choices=["oldest", "shallow", "balanced"],
                help="shallow: a full pool drops its shallowest states first, so deep ones stay; "
                     "balanced: it drops from the fullest depth band (<10, 10-99, 100+ cycles)")
ap.add_argument("--pool-draw", default="uniform", choices=["uniform", "balanced"],
                help="balanced: each batch comes from one depth band, bands equally likely")
ap.add_argument("--step-noise", type=float, default=0.0,
                help="gaussian noise added to every channel after EVERY step of a training "
                     "rollout (denoising: the model must keep pulling the picture back); off in eval")
ap.add_argument("--tape-hold", type=int, default=0,
                help="1: score the instruction channels on every execute frame, not only at the "
                     "checkpoint (nothing on the tape moves then, so the target is the same)")
ap.add_argument("--tape-loss", default="mse", choices=["mse", "l1"],
                help="l1 keeps pulling small tape errors to exactly zero")
ap.add_argument("--clock", type=float, default=0.0,
                help="weight of a SUPERVISED clock: the last two channels are painted once with the phase "
                     "(cos, sin = 1, 0) and every step is scored against cos/sin(2 pi t / K); the NCA keeps "
                     "it itself, nothing overwrites the state (its own time drifted: bands moved later "
                     "every cycle and regions lost sync)")
ap.add_argument("--time-free", type=int, default=0,
                help="1: at every checkpoint the picture may show this cycle's state OR the previous one "
                     "(it lags: its software machine waits a cycle); never twice in a row. The NCA may run "
                     "slow, nothing asks it to keep the trainer's time. Eval follows the same rule")
ap.add_argument("--desync", type=float, default=0.0,
                help="fraction of training samples where a random horizontal strip (1/4-1/2 of the "
                     "height) starts k steps AHEAD: that strip, all channels, is taken from the state "
                     "run k more steps; targets unchanged. Regions drifted apart in closed loop and "
                     "training never showed a desynced state")
ap.add_argument("--desync-k", type=int, default=4)
ap.add_argument("--topk", type=int, default=0,
                help="also score the mean of each sample's K worst squared pixel errors (memory and "
                     "instruction channels): one flipped band pixel is ~3e-5 of a whole-picture mean")
ap.add_argument("--topk-weight", type=float, default=1.0)
ap.add_argument("--repair", type=float, default=0.0,
                help="fraction of each batch whose bands are damaged before the rollout (faded, "
                     "ghost ink in other register columns, speckle, hidden noise); the targets stay "
                     "clean, so the model has to restore them. The right register stays strongest")
ap.add_argument("--repair-bands", default="1,3", help="bands damaged per sample: min,max")
ap.add_argument("--mean-fire", type=int, default=0,
                help="1: no stochastic fire; every cell applies fire_rate * delta every step "
                     "(the same mean update, deterministic)")
ap.add_argument("--ema", type=float, default=0.0,
                help="EMA decay of the weights (0 = off). Evals and checkpoints ('nca') use the "
                     "averaged weights; the raw ones are saved as 'nca_raw'")
ap.add_argument("--warm-regs", type=float, default=0.0,
                help="fraction of fresh training samples whose registers come from running the "
                     "program in software for 0..--warm-cycles cycles first: long runs collapse "
                     "towards zeros, which uniform random registers never show")
ap.add_argument("--warm-cycles", type=int, default=200)
ap.add_argument("--instrs-set", default="",
                help="program lengths drawn from this list instead of --instrs min,max")
ap.add_argument("--local-span", type=int, default=0,
                help="> 0: every instruction's a, b, d lie in span + 1 consecutive registers "
                     "(what a local compiler emits); 0 = anywhere")
ap.add_argument("--ncc-frac", type=float, default=0.0,
                help="fraction of fresh samples cut from COMPILED programs (src/ncpu/ncc.py random "
                     "programs): a window of the real instruction stream on the real memory state")
ap.add_argument("--ncc-progs", type=int, default=300, help="random compiled programs to cut from")
ap.add_argument("--max-area", type=int, default=0,
                help="redraw (length, registers) until the picture has at most this many pixels")
ap.add_argument("--regs-set", default="",
                help="registers drawn from this list (e.g. 4,8,16,44) instead of --regs min,max; "
                     "few shapes keep the per-shape pools full")
ap.add_argument("--batch-ref-regs", type=int, default=0,
                help="pictures wider than this many registers get a proportionally smaller "
                     "batch (0 = off)")
ap.add_argument("--noise-uniform", type=int, default=0,
                help="1: each sample's noise std is drawn uniformly in [0, the given std]")
ap.add_argument("--batch-ref-instrs", type=int, default=0,
                help="programs longer than this get a proportionally smaller batch, so every "
                     "step holds about the same number of pixels (0 = fixed batch)")
ap.add_argument("--instrs-dist", default="uniform", choices=["uniform", "log"],
                help="log: program length log-uniform over --instrs, so long programs stay cheap")
ap.add_argument("--pool-gate", type=int, default=1,
                help="0 keep every state; 1 keep only states that landed exactly (memory and "
                     "every tape pixel); 2 keep states whose memory is right, drifted tape or not")
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
CYC_RANGE = tuple(int(v) for v in args.cycles_range.split(",")) if args.cycles_range else ()
REGS_SET = [int(v) for v in args.regs_set.split(",") if v.strip()]
INSTRS_SET = [int(v) for v in args.instrs_set.split(",") if v.strip()]
K = args.load_steps + args.exec_steps
P = args.band + args.gap                              # slot pitch
NCC = []
if args.ncc_frac:
    from ncpu.ncc import random_traces                # noqa: E402
    assert args.word_bits == 6, "ncc emits 6-bit code"
    NCC = random_traces(args.ncc_progs, seed=args.seed)
assert min(N_RANGE) >= 2, "a 1-instruction program never moves"

_lay = {}


def lay(nr):
    """The v12 register layout for ``nr`` registers (its width is the picture's)."""
    if nr not in _lay:
        L = TileLayoutV8(n_regs=nr, word_bits=args.word_bits, r=args.radius, pitch=args.pitch)
        if args.seamless:                             # the row wraps with the register pitch, like the tape
            L.col_x = [args.pitch // 2 + c * args.pitch for c in range(nr)]
            L.grid_x = [L.col_x]
            L.W = nr * args.pitch
            L._yx = np.ogrid[: L.H, : L.W]
        _lay[nr] = L
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
    S = args.local_span
    if S and nr > S:
        c = rng.randrange(nr - S)
        return (rng.choice(OPS), *(c + rng.randrange(S + 1) for _ in range(3)))
    return (rng.choice(OPS), *(rng.randrange(nr) for _ in range(3)))


def ncc_run(n, nr, rng):
    """n instructions of a compiled program's stream, from a random point of its run, on the
    machine state there, shifted into nr registers (None if the window spans more)."""
    tape_full, states = rng.choice(NCC)
    L, k = len(tape_full), rng.randrange(len(states) - 1)
    stream = [tape_full[(k + m) % L] for m in range(n)]
    used = [r for ins in stream for r in ins[1:]]
    lo, hi = min(used), max(used)
    if hi - lo + 1 > nr:
        return None
    sh = rng.randint(0, nr - (hi - lo + 1)) - lo
    st = states[k]
    regs = [st[j - sh] if 0 <= j - sh < len(st) else 0 for j in range(nr)]
    g = geom(n, n - 1 - min(args.below, n - 1))
    j = rng.randrange(n)                          # instruction j loaded; stream[0] loads next
    tape = [None] * n
    for m, (op, a, b, d) in enumerate(stream):
        tape[(j + 1 + m) % n] = (op, a + sh, b + sh, d + sh)
    ring = {(j + m) % n: (g.head - (m - 1)) % g.N for m in range(1, n)}
    return dict(g=g, nr=nr, tape=tape, ring=ring, inmem=j, regs=tuple(regs))


def fresh_run(n, nr, rng, warm=False, kind=None):
    """A random VALID boundary state: any instruction j loaded (and executed), the
    rest in program order behind it with j + 1 on the head, any register values.
    Memory is at the bottom; which band is where is what varies.
    kind: None = compiled with probability --ncc-frac, "ncc" = compiled, "random" = not."""
    if NCC and kind != "random" and (kind == "ncc" or rng.random() < args.ncc_frac):
        for _ in range(8):
            r = ncc_run(n, nr, rng)
            if r is not None:
                return r
    g = geom(n, n - 1 - min(args.below, n - 1))
    tape = [sample_instr(rng, nr) for _ in range(n)]
    j = rng.randrange(n)
    ring = {(j + m) % n: (g.head - (m - 1)) % g.N for m in range(1, n)}
    regs = [rng.randint(0, (1 << args.word_bits) - 1) for _ in range(nr)]
    if warm:                                      # the registers k instructions after random ones
        k = rng.randint(0, args.warm_cycles)
        for m in range(k - 1, -1, -1):            # ... ending with instruction j itself
            op, a, b, d = tape[(j - m) % n]
            regs[d] = apply_op(op, regs[a], regs[b], regs[d], args.word_bits)
    return dict(g=g, nr=nr, tape=tape, ring=ring, inmem=j, regs=tuple(regs))


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
        if args.clock:
            x[i, C - 2] = 1.0
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


def pool_push(key, x_end, runs, depth, ok, adv=None):
    p = _pool.setdefault(key, [])
    xs = x_end.detach().cpu().numpy().astype(np.float16 if args.pool_fp16 else np.float32)
    adv = args.cycles if adv is None else adv
    k = 0
    for i, run in enumerate(runs):
        if args.pool_gate and not ok[i]:
            continue
        p.append((xs[i].copy(), run, depth[i] + adv))  # a view would pin the whole batch
        k += 1
    cap = args.pool_size
    if args.pool_gb and len(xs):
        keys = (len(INSTRS_SET) or N_RANGE[1] - N_RANGE[0] + 1) * (len(REGS_SET) or R_RANGE[1] - R_RANGE[0] + 1)
        cap = max(8, min(cap, int(args.pool_gb * 2 ** 30 / keys / xs[0].nbytes)))
    if len(p) > cap:
        if args.pool_evict == "shallow":
            p.sort(key=lambda e: e[2])
        if args.pool_evict == "balanced":
            while len(p) > cap:
                bands = [[j for j, e in enumerate(p) if depth_band(e[2]) == b] for b in range(3)]
                full = max(bands, key=len)
                del p[full[0]]                                      # oldest of the fullest band
        else:
            del p[: len(p) - cap]
    return k


def depth_band(d):
    return 0 if d < 10 else 1 if d < 100 else 2


def pool_draw(key, bs, rng):
    p = _pool.get(key, [])
    if len(p) < bs:
        return None
    if args.pool_draw == "balanced":
        bands = [b for b in ([e for e in p if depth_band(e[2]) == k] for k in range(3)) if len(b) >= bs]
        if bands:
            p = bands[rng.randrange(len(bands))]
    ent = [p[rng.randrange(len(p))] for _ in range(bs)]
    return (torch.from_numpy(np.stack([e[0] for e in ent])).to(DEV).float(),
            [e[1] for e in ent], [e[2] for e in ent])


def noise_std(x, std):
    """std, or with --noise-uniform a per-sample std drawn in [0, std]."""
    if not args.noise_uniform:
        return std
    return torch.rand(x.shape[0], 1, 1, 1, device=x.device) * std


def batch_for(n, nr=0, exact=False):
    bs = args.batch_size
    ref = args.batch_ref_instrs
    if ref and n > ref:
        bs = bs * (3 * ref + 21) / (3 * n + 21)                    # H ~ 3n + 21
    ref = args.batch_ref_regs
    if ref and nr > ref:
        bs = bs * (args.pitch * ref + 7) / (args.pitch * nr + 7)   # W ~ pitch * regs
    return bs if exact else max(2, round(bs))


def draw_n(rng):
    if INSTRS_SET:
        return rng.choice(INSTRS_SET)
    lo, hi = N_RANGE
    if args.instrs_dist == "log":
        return min(hi, int(np.exp(rng.uniform(np.log(lo), np.log(hi + 1)))))
    return rng.randint(lo, hi)


class MeanFire(torch.nn.Module):
    """Deterministic stand-in for StochasticUpdate: the expected update."""

    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, delta):
        return delta * self.rate


def apply_fire(nca, cfg):
    if cfg.get("mean_fire"):
        nca.stochastic_update = MeanFire(cfg["fire_rate"])


# ── model ───────────────────────────────────────────────────────────────────

nca = NeuralCA(channels=C, hidden_channels=args.hidden, fire_rate=args.fire_rate,
               alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
               perception="static_sobel", padding_type="circular", read_only_dims=[],
               state_bound=10.0).to(DEV)
root = Path(__file__).resolve().parent.parent
if args.init_from:
    sd = torch.load(root / args.init_from, map_location=DEV, weights_only=False)["nca"]
    own = nca.state_dict()
    for k, v in sd.items():                       # more channels than the checkpoint: the new ones go
        if k in own and own[k].shape != v.shape:  # last, their weights start at zero (old dynamics kept)
            w = torch.zeros_like(own[k]) if "rule" in k else own[k].clone()
            w[tuple(slice(0, n) for n in v.shape)] = v
            sd[k] = w
            print("widened", k, tuple(v.shape), "->", tuple(w.shape))
    print("warm start:", args.init_from, nca.load_state_dict(sd, strict=False))
apply_fire(nca, vars(args))
opt = torch.optim.Adam(nca.parameters(), lr=args.lr)

run_dir = root / "runs" / f"TAPE_{time.strftime('%Y%m%d_%H%M%S')}_{args.tag}_s{args.seed}"
run_dir.mkdir(parents=True)
(run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))
print(f"run {run_dir.name}: {R_RANGE} registers, programs of {N_RANGE} instructions, "
      f"load {args.load_steps} + exec {args.exec_steps}")


def scored(seq, t, tail):
    return seq[:, t - tail + 1:t + 1]


def tape_err(a, b):
    if args.tape_loss == "l1":
        return (a - b).abs().flatten(1).mean(1) + topk_err(a, b)
    return per_sample(a, b)


def topk_err(a, b):
    if not args.topk:
        return 0.0
    e = ((a - b) ** 2).flatten(1)
    return e.topk(min(args.topk, e.shape[1]), dim=1).values.mean(1) * args.topk_weight


def per_sample(a, b):
    return ((a - b) ** 2).flatten(1).mean(1) + topk_err(a, b)


CLOCK_T = torch.stack([torch.cos(torch.arange(K + 1) * 2 * np.pi / K),
                       torch.sin(torch.arange(K + 1) * 2 * np.pi / K)], 1).to(DEV)    # (K+1, 2)


def clock_err(seq):
    """Per-sample MSE of the clock channels over the whole cycle against the phase."""
    return ((seq[:, :, C - 2:] - CLOCK_T[None, :, :, None, None]) ** 2).mean(dim=(1, 2, 3, 4))


def train_cycle(x, runs, nc=None):
    """``nc`` (default ``--cycles``) cycles in one graph, every checkpoint scored; landed only if every
    cycle landed. Also returns per-op stats of the batch for the log."""
    nca.gaussian_noise.std, nca.gaussian_noise.prob = args.step_noise, 1.0
    if args.boundary_noise:
        x = x.clone()
        x[:, 4:] = x[:, 4:] + torch.randn_like(x[:, 4:]) * noise_std(x, args.boundary_noise)
    if args.visible_noise:
        x = x.clone()
        x[:, :1] = x[:, :1] + torch.randn_like(x[:, :1]) * noise_std(x, args.visible_noise)
    tape_noise = args.visible_noise if args.tape_noise < 0 else args.tape_noise
    if tape_noise:
        x = x.clone()
        x[:, 1:4] = x[:, 1:4] + torch.randn_like(x[:, 1:4]) * noise_std(x, tape_noise)
    ll = ml = tl = 0.0
    ok = [True] * len(runs)
    mem_ok = torch.ones(len(runs), dtype=torch.bool, device=DEV)
    st = {op: {"instr": [], "mem": [], "bit": [], "reg": []} for op in OPS}
    nc = nc or args.cycles
    lagged = torch.zeros(len(runs), dtype=torch.bool, device=DEV)
    for _ in range(nc):
        before = runs
        runs, t_load, t_mem, t_tape = cycle_targets(runs)
        seq = nca(x, K)                      # (B, K+1, C, H, W)
        fl = scored(seq, args.load_steps, args.tail)
        fe = scored(seq, K, args.tail)
        ft = seq[:, args.load_steps:K + 1] if args.tape_hold else fe
        li = tape_err(fl[:, :, 1:4], t_load[:, None]) * args.load_weight
        lm = per_sample(fe[:, :, 0], t_mem[:, None])
        lt = tape_err(ft[:, :, 1:4], t_tape[:, None]) * args.tape_weight
        if args.time_free:                   # or: still the state before this cycle (one cycle late)
            p_mem = torch.from_numpy(np.stack([memory(r) for r in before])).to(DEV)
            p_tape = torch.from_numpy(np.stack([instr_channels(r) for r in before])).to(DEV)
            li0 = tape_err(fl[:, :, 1:4], p_tape[:, None]) * args.load_weight
            lm0 = per_sample(fe[:, :, 0], p_mem[:, None])
            lt0 = tape_err(ft[:, :, 1:4], p_tape[:, None]) * args.tape_weight
            lag = ((li0 + lm0 + lt0) < (li + lm + lt)).detach() & ~lagged
            li, lm, lt = torch.where(lag, li0, li), torch.where(lag, lm0, lm), torch.where(lag, lt0, lt)
            sel = lag[:, None, None, None]
            t_load, t_tape = torch.where(sel, p_tape, t_load), torch.where(sel, p_tape, t_tape)
            runs = [b if l else r for b, r, l in zip(before, runs, lag.tolist())]
            lagged = lag
        if args.clock:
            lt = lt + clock_err(seq) * args.clock
        ll, ml, tl = ll + li.mean(), ml + lm.mean(), tl + lt.mean()
        xe = seq[:, -1].detach()
        ok = [a and b for a, b in zip(ok, landed(seq[:, args.load_steps].detach(), xe, t_load, t_tape, runs))]
        hit = decode(runs, xe[:, 0]) == want_bits(runs)
        mem_ok &= hit.flatten(1).all(1)
        for i, r in enumerate(runs):
            op, _, _, d = r["tape"][r["inmem"]]
            st[op]["instr"].append(float(li[i] + lt[i])); st[op]["mem"].append(float(lm[i]))
            st[op]["bit"].append(float(hit[i, d].float().mean())); st[op]["reg"].append(float(hit[i, d].all()))
        x = seq[:, -1]
    nca.gaussian_noise.std = 0.0                     # eval and deep advances run clean
    ll, ml, tl = ll / nc, ml / nc, tl / nc
    loss = ll + ml + tl
    loss.backward()
    ops = {f"{k}_{op.lower()}": round(float(np.mean(v)), 6) for op in OPS for k, v in st[op].items() if v}
    ops["batch_mem_ok"] = round(float(mem_ok.float().mean()), 3)
    keep = mem_ok.tolist() if args.pool_gate == 2 else ok
    return float(loss), float(ll), float(ml), float(tl), x.detach(), runs, ok, keep, ops


REPAIR_BANDS = tuple(int(v) for v in args.repair_bands.split(","))


def damage(x, runs, rng):
    """--repair: damage a few bands of some samples, keeping them decodable."""
    x = x.clone()
    for i, run in enumerate(runs):
        if rng.random() >= args.repair:
            continue
        g, L = run["g"], lay(run["nr"])
        for k in rng.sample(sorted(run["ring"]), min(len(run["ring"]), rng.randint(*REPAIR_BANDS))):
            y = g.slot_row(run["ring"][k])
            ys = slice(y, y + args.band)
            f = rng.uniform(0.2, 0.8)
            x[i, 1:4, ys] *= f                                       # fade
            for ch, reg, v in cols(run["tape"][k]):
                for r in range(run["nr"]):
                    if r != reg and rng.random() < 0.5:              # ghost, weaker than the band
                        cx = L.col_x[r]
                        x[i, 1 + ch, ys, cx - L.r:cx + L.r + 1] += rng.uniform(-0.5, 0.5) * f * abs(v)
            x[i, 1:4, ys] += torch.randn_like(x[i, 1:4, ys]) * 0.05     # speckle
            x[i, 4:, ys] += torch.randn_like(x[i, 4:, ys]) * 0.1        # hidden
    return x


@torch.no_grad()
def evaluate(cycles, bs, kind="random"):
    """Fresh tapes run ``cycles`` cycles; exact = memory right at the end of every
    execute AND every instruction pixel right at both checkpoints. prog = memory
    alone right at the end of every execute (the program computes correctly)."""
    erng = random.Random(991 + cycles)
    runs = [fresh_run(args.eval_instrs, args.eval_regs, erng, kind=kind) for _ in range(bs)]
    x = start_state(runs)
    exact = torch.ones(bs, dtype=torch.bool, device=DEV)
    prog = exact.clone()
    per, per_prog = [], []
    lagged = [False] * bs
    for _ in range(cycles):
        before = runs
        runs, t_load, t_mem, t_tape = cycle_targets(runs)
        seq = nca(x, K)
        if args.time_free:                   # one cycle late is fine, never twice in a row
            now = (decode(runs, seq[:, -1, 0]) == want_bits(runs)).flatten(1).all(1).tolist()
            old = (decode(before, seq[:, -1, 0]) == want_bits(before)).flatten(1).all(1).tolist()
            lag = [not a and b and not l for a, b, l in zip(now, old, lagged)]
            runs = [b if l else r for b, r, l in zip(before, runs, lag)]
            p_tape = torch.from_numpy(np.stack([instr_channels(r) for r in before])).to(DEV)
            sel = torch.tensor(lag, device=DEV)[:, None, None, None]
            t_load, t_tape = torch.where(sel, p_tape, t_load), torch.where(sel, p_tape, t_tape)
            lagged = lag
        exact &= torch.tensor(landed(seq[:, args.load_steps], seq[:, -1], t_load, t_tape, runs),
                              device=DEV)
        prog &= (decode(runs, seq[:, -1, 0]) == want_bits(runs)).flatten(1).all(1)
        x = seq[:, -1]
        per.append(float(exact.float().mean()))
        per_prog.append(float(prog.float().mean()))
    return per, per_prog


DEEP = tuple(int(v) for v in args.deep_advance.split(","))
deep_log = {"n": 0, "kept": 0, "depth": 0.0}


@torch.no_grad()
def deep_advance(rng):
    """Run one pool batch on for many cycles without gradients; put back the
    memory-correct survivors, so the pool reaches states far deeper than a rollout."""
    keys = [k for k, p in _pool.items() if len(p) >= batch_for(*k)]
    if not keys:
        return
    key = keys[rng.randrange(len(keys))]
    x, runs, depth = pool_draw(key, batch_for(*key), rng)
    d = rng.randint(*DEEP)
    alive = torch.ones(len(runs), dtype=torch.bool, device=DEV)
    for _ in range(d):
        runs, _, _, _ = cycle_targets(runs)
        x = nca(x, K)[:, -1]
        alive &= (decode(runs, x[:, 0]) == want_bits(runs)).flatten(1).all(1)
    keep = alive.tolist()
    k = pool_push(key, x, runs, depth, keep, adv=d)
    deep_log.update(n=deep_log["n"] + 1, kept=k,
                    depth=float(np.mean([e[2] for e in _pool[key]])))


ema = {k: v.detach().clone() for k, v in nca.state_dict().items()} if args.ema else None


def ema_update():
    with torch.no_grad():
        for k, v in nca.state_dict().items():
            if v.dtype.is_floating_point:
                ema[k].lerp_(v, 1 - args.ema)


@torch.no_grad()
def desync(x, rng):
    k = rng.randint(1, args.desync_k)
    pick = [i for i in range(len(x)) if rng.random() < args.desync]
    if not pick:
        return x
    ahead = nca(x[pick], k)[:, -1]
    x = x.clone()
    H = x.shape[2]
    for j, i in enumerate(pick):
        h = rng.randint(H // 4, H // 2)
        rows = [(rng.randrange(H) + r) % H for r in range(h)]
        x[i, :, rows] = ahead[j, :, rows]
    return x


def snapshot(step):
    """The checkpoint dict; with --ema the averaged weights are 'nca'."""
    ck = {"nca": nca.state_dict(), "step": step, "config": vars(args)}
    if ema is not None:
        ck["nca_raw"], ck["nca"] = ck["nca"], ema
    return ck


tlog = (run_dir / "train.jsonl").open("a")
elog = (run_dir / "metrics.jsonl").open("a")
best = -1.0
bar = tqdm(range(args.steps), dynamic_ncols=True)
for step in bar:
    while True:
        n = draw_n(rng)
        nr = rng.choice(REGS_SET) if REGS_SET else rng.randint(*R_RANGE)
        if not args.max_area or geom(n, n - 1 - min(args.below, n - 1)).H * lay(nr).W <= args.max_area:
            break
    bs = batch_for(n, nr)
    nc = args.cycles
    if CYC_RANGE:                                     # same graph size as --cycles: fewer samples
        budget = batch_for(n, nr, exact=True) * args.cycles   # (sample x cycle) units
        nc = min(rng.randint(*CYC_RANGE), max(1, int(budget)))  # big pictures: fewer cycles
        bs = max(1, round(budget / nc))
    drawn = pool_draw((n, nr), bs, rng) if rng.random() < args.pool_frac else None
    if drawn:
        x, runs, depth = drawn
    else:
        runs = [fresh_run(n, nr, rng, bool(args.warm_regs) and rng.random() < args.warm_regs) for _ in range(bs)]
        x, depth = start_state(runs), [0] * bs

    if args.repair:
        x = damage(x, runs, rng)
    if args.desync:
        x = desync(x, rng)
    opt.zero_grad(set_to_none=True)
    loss, ll, ml, tl, x_end, done, ok, keep, ops = train_cycle(x, runs, nc)
    gn = torch.nn.utils.clip_grad_norm_(nca.parameters(), args.grad_clip or float("inf")).item()
    opt.step()
    if ema is not None:
        ema_update()
    pushed = pool_push((n, nr), x_end, done, depth, keep, adv=nc)
    if args.deep_every and step % args.deep_every == 0:
        deep_advance(rng)

    if step % args.log_every == 0:
        tlog.write(json.dumps({
            "step": step, "loss": round(loss, 6), "load": round(ll, 6), "mem": round(ml, 6),
            "tape": round(tl, 6), "grad_norm": round(gn, 4), "n": n, "nr": nr, "nc": nc, "from_pool": bool(drawn),
            "mean_depth": round(float(np.mean(depth)), 2), "deep_kept": deep_log["kept"], "batch_ok": round(float(np.mean(ok)), 3),
            "pool": sum(len(p) for p in _pool.values()), "pushed": pushed, **ops}) + "\n")
        if step % (args.log_every * 20) == 0:
            tlog.flush()

    if step % args.eval_every == 0:
        if ema is not None:                          # evaluate the averaged weights
            raw = {k: v.clone() for k, v in nca.state_dict().items()}
            nca.load_state_dict(ema)
        per, per_prog = evaluate(max(EVAL_CYCLES), args.eval_batch)
        nper = evaluate(max(EVAL_CYCLES), args.eval_batch, "ncc")[1] if NCC else None
        if ema is not None:
            nca.load_state_dict(raw)
        depths = [e[2] for p in _pool.values() for e in p]
        rec = {"step": step, "loss": round(loss, 6),
               "pool": sum(len(p) for p in _pool.values()),
               "pool_depth_mean": round(float(np.mean(depths)), 1) if depths else 0,
               "pool_depth_max": int(max(depths)) if depths else 0}
        for c in EVAL_CYCLES:
            rec[f"cyc{c}"] = round(per[c - 1], 4)
        for c in EVAL_CYCLES:
            rec[f"prog{c}"] = round(per_prog[c - 1], 4)
            if nper:
                rec[f"nprog{c}"] = round(nper[c - 1], 4)
        score = float(np.mean([rec[f"cyc{c}"] for c in EVAL_CYCLES]))
        rec["score"] = round(score, 4)
        elog.write(json.dumps(rec) + "\n"); elog.flush()
        ck = snapshot(step)
        torch.save(ck, run_dir / "nca_latest.pt")
        # best = the program still computing right at the longest eval horizon
        prog_long = rec[f"{'n' if nper else ''}prog{max(EVAL_CYCLES)}"]
        if prog_long > best:
            best = prog_long
            torch.save(ck, run_dir / "nca_best.pt")
        bar.set_description(f"loss {loss:.4f} pool {rec['pool']} depth {rec['pool_depth_mean']} "
                            + " ".join(f"c{c}:{rec[f'cyc{c}']:.2f}" for c in EVAL_CYCLES)
                            + " prog " + " ".join(f"{rec[f'prog{c}']:.2f}" for c in EVAL_CYCLES)
                            + (" ncc " + " ".join(f"{rec[f'nprog{c}']:.2f}" for c in EVAL_CYCLES) if nper else ""))

    if args.ckpt_every and step % args.ckpt_every == 0:
        torch.save(snapshot(step), run_dir / f"nca_{step:07d}.pt")
torch.save(snapshot(args.steps), run_dir / "nca_last.pt")
print("done:", run_dir)
