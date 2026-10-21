#!/usr/bin/env python3
"""Train the v9 rule: the program moves, the data stays.

A stack of 1 px instruction rows sits above one register grid.  Per
instruction the rule must LOAD (stack drops one pitch, bottom row expands into
the A / B / dst rectangles) and EXEC (dst <- A - B in place, SUBLEQ only if
<= 0).  Program length 0..N is sampled per batch; NOP rows are empty.

    uv run python scripts/train_stack.py --init-from runs/<wide ALU>/nca_latest.pt
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
ap.add_argument("--steps", type=int, default=300_000)
ap.add_argument("--device", default="cuda")
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--instr-min", type=int, default=0)
ap.add_argument("--instr-max", type=int, default=4)
ap.add_argument("--eval-instr", default="0,1,2,3,4,6", help="lengths to evaluate (beyond max = generalisation)")
ap.add_argument("--nop-frac", type=float, default=0.2)
ap.add_argument("--balance-subleq", action="store_true", default=True)
ap.add_argument("--channels", type=int, default=16)
ap.add_argument("--hidden", type=int, default=256)
ap.add_argument("--kernel", type=int, default=7)
ap.add_argument("--fire-rate", type=float, default=0.5)
ap.add_argument("--k-load", type=int, default=30, help="CA steps for a LOAD stage")
ap.add_argument("--k-exec", type=int, default=40, help="CA steps for an EXEC stage")
ap.add_argument("--load-tail", type=int, default=10, help="LOAD: supervise the last N frames only")
ap.add_argument("--mask-weight", type=float, default=1.0, help="weight of the instruction-channel loss")
ap.add_argument("--dst-weight", type=float, default=1.0)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--init-from", default=None)
ap.add_argument("--pad-channels", action="store_true",
                help="--init-from has fewer channels: keep the old ones in place with their "
                     "weights, give the new ones ZERO input weights (so the old function is "
                     "preserved exactly) and small random output weights (so they become "
                     "active and can receive gradient)")
ap.add_argument("--log-every", type=int, default=10)
ap.add_argument("--eval-every", type=int, default=500)
ap.add_argument("--eval-batch", type=int, default=16)
ap.add_argument("--ckpt-every", type=int, default=5000)
ap.add_argument("--snapshot-every", type=int, default=2500)
ap.add_argument("--curriculum-start", type=int, default=0,
                help="maximum program length the curriculum begins with")
ap.add_argument("--curriculum", type=int, default=0,
                help="N > 0: grow the maximum program length by 1 every N steps (0, then 0-1, "
                     "0-2, ...) up to --instr-max; 0 = sample the full range from the start")
ap.add_argument("--promote-at", type=float, default=0.0,
                help="with --curriculum: only grow the range once the CURRENT longest length "
                     "reaches this exact-match score (0 = grow purely on the step schedule)")
ap.add_argument("--no-clock", action="store_true",
                help="drop the supervised phase/ramp channels entirely: the rule gets only the "
                     "picture and must find its own way to time the stages")
ap.add_argument("--clock-weight", type=float, default=1.0,
                help="weight of the self-generated clock channel loss (ch 4: +1 EXEC, -1 LOAD)")
ap.add_argument("--teacher-frac", type=float, default=1.0,
                help="probability that a stage starts from the CORRECT visible state (data, "
                     "instruction, clock channels reset to target) instead of the rule's own")
ap.add_argument("--teacher-anneal", type=int, default=200_000,
                help="linearly anneal --teacher-frac to 0 over this many steps (0 = constant)")
ap.add_argument("--simple-loss", type=int, default=0,
                help="N > 0: replace all loss terms with ONE plain full-screen MSE over the "
                     "visible channels (data, A/B/R, phase, ramp) on the last N frames of each stage")
ap.add_argument("--no-detach", action="store_true",
                help="keep the whole program in one graph: gradient flows across stage "
                     "boundaries (LOAD can learn to set up the following EXEC)")
ap.add_argument("--preloaded", action="store_true",
                help="instruction 0 starts already expanded; stages EXEC, LOAD, EXEC, ...")
ap.add_argument("--max-stages", type=int, default=0,
                help="train only the FIRST N sub-stages of each program (0 = all). The ladder the "
                     "length curriculum cannot express: rung 1 = LOAD alone (stack drops, instr 0 "
                     "expands), rung 2 = +EXEC, rung 3 = +the next LOAD, ... one new behaviour per "
                     "rung instead of two.")
ap.add_argument("--stage-gate", type=float, default=0.8,
                help="with --max-stages: grow the stage depth by 1 once the CURRENT depth scores "
                     "this well (exact visible-state match after that many stages)")
ap.add_argument("--headroom-max", type=int, default=0,
                help="blank band-slots above the topmost instruction, sampled 0..N per batch. "
                     "Without it the top border sits flush against the last instruction, so the "
                     "picture encodes program length and nothing can be appended later.")
ap.add_argument("--window", type=int, default=0,
                help="fixed stack height: the program STREAMS through a window of N slots, so the "
                     "grid is the same size for a 2-instruction program and a 2000-instruction one")
ap.add_argument("--grad-clip", type=float, default=1.0,
                help="max grad norm (0 = off, measure only)")
ap.add_argument("--pool-size", type=int, default=0,
                help="replay pool of mid-program states, bucketed by stage (0 = off)")
ap.add_argument("--pool-frac", type=float, default=0.5,
                help="fraction of batches drawn from the pool rather than from t=0")
ap.add_argument("--pool-after-rung", type=int, default=4,
                help="only pool once the ladder reaches this rung: pooling stages the rule cannot "
                     "yet produce fills the buffer with unrecoverable states, which is how the "
                     "earlier pool attempt stalled at depth ~3 of 15")
ap.add_argument("--stage-tol-px", type=int, default=8,
                help="instruction-channel pixels allowed to be wrong by >0.5 in the stage gate. "
                     "A max-over-every-pixel test rejects on speckle: measured failures were a "
                     "median of 6 stray px out of ~2000, while a genuinely misplaced band is 30+.")
ap.add_argument("--drain", action="store_true",
                help="EXEC must END in the machine's REST form -- rectangles retracted and hidden "
                     "channels at zero -- so a cycle boundary is bit-for-bit the state at t=0 and "
                     "cycle N is the same problem as cycle 1. NOT a third sub-stage: a separate "
                     "DRAIN asked the rule to SIT in the rest state and then leave it, which is "
                     "one state with one (zero) hidden demanding two different behaviours.")
ap.add_argument("--boundary-tail", type=int, default=2,
                help="frames at the END of EXEC scored against the rest form -- picture, data AND "
                     "the hidden-to-zero term, one window for all three. Keep it SHORT: the rest "
                     "state is a point the trajectory passes through, not one it parks in, and "
                     "every extra frame is 'hold still' fighting 'start the next shift'. Measured: "
                     "from an exact rest state the rule leaves on frame 1 and never returns.")
ap.add_argument("--hidden-weight", type=float, default=1.0,
                help="weight of the hidden-channels-to-zero term at the boundary")
ap.add_argument("--exec-tail", type=int, default=0,
                help="EXEC: supervise only the last N frames (0 = every frame). Ignored with "
                     "--drain, where --boundary-tail is the EXEC window.")
ap.add_argument("--max-cycles", type=int, default=0,
                help="with --drain: train only the first N complete CYCLES (0 = all). The unit is "
                     "the cycle, not the sub-stage: a half cycle is a state the machine is never "
                     "in at rest, so a rung of LOAD-alone teaches nothing that composes.")
ap.add_argument("--extrap-cycles", default="1,2,4,8,16,32",
                help="cycle depths to probe during training: run this many cycles from t=0 and "
                     "report the deepest one still exactly right")
ap.add_argument("--boundary-noise", type=float, default=0.0,
                help="std of Gaussian noise added to the HIDDEN channels of a cycle-boundary "
                     "state during training. The boundary has to be an ATTRACTOR, not just a "
                     "target: trained only on clean boundaries, the map is fitted on a measure-"
                     "zero set and the residual that actually arrives may grow instead of shrink.")
ap.add_argument("--tag", default="stack")
args = ap.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed); rng = random.Random(args.seed)
CLOCK = 4    # phase: +1 while executing, -1 while loading
RAMP = 5     # countdown: +1 at the start of a stage, falls linearly to -1 at its end.
             # phase flips exactly when the ramp bottoms out -- a LOCAL condition on the
             # rule's own state, so the flip is learnable; neither channel is ever
             # teacher-forced, the rule must run its own clock from t=0 on.
DEV = args.device
# No NOP: it paints NOTHING in any channel, so on a NOP cycle end-of-LOAD and end-of-EXEC
# are pixel-identical (measured 3053/3053 ambiguous, against 0/6236 for real instructions)
# and the rule cannot tell "expand" from "retract". Blank slots still occur as the stream
# drains past the end of a program; that is the tail, where nothing is left to execute.
OPS = ("SUB", "SUBLEQ")
PHASE_K = {"LOAD": args.k_load, "EXEC": args.k_exec, "HOLD": args.k_exec}


def steps_for(L, k):
    return PHASE_K[L.phase_of(k)]


def cycle_steps():
    return args.k_load + args.k_exec
LENS = list(range(args.instr_min, args.instr_max + 1))
ap_cur_start = max(args.instr_min, args.curriculum_start)
_cur = {"max": ap_cur_start if args.curriculum else args.instr_max, "since": 0}


def active_lens(step):
    """Lengths the sampler may draw from right now (curriculum)."""
    if not args.curriculum:
        return LENS
    return [n for n in LENS if n <= _cur["max"]]


def maybe_promote(step, rec):
    """Grow the range on the step schedule, optionally gated on a score."""
    if not args.curriculum or _cur["max"] >= args.instr_max:
        return
    if step - _cur["since"] < args.curriculum:
        return
    if args.promote_at and rec.get(f"exact_L{_cur['max']}", 0.0) < args.promote_at:
        return
    _cur["max"] += 1; _cur["since"] = step
    print(f"\n[curriculum] step {step}: max program length -> {_cur['max']}")
EVAL_LENS = [int(x) for x in args.eval_instr.split(",") if x.strip()]

run = f"STACK_{datetime.now():%Y%m%d_%H%M%S}_{args.tag}_s{args.seed}"
run_dir = Path(__file__).resolve().parent.parent / "runs" / run
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

_layouts = {}


def layout(n, headroom=0):
    key = (n, headroom)
    if key not in _layouts:
        _layouts[key] = TileLayoutV9(n_instr=n, preloaded=args.preloaded, headroom=headroom,
                                     window=args.window, drain=args.drain)
    return _layouts[key]


nca = NeuralCA(channels=args.channels, hidden_channels=args.hidden, fire_rate=args.fire_rate,
               alive_threshold=0, zero_initialization=False, kernel_size=args.kernel,
               perception="static_sobel", padding_type="zeros", read_only_dims=[], state_bound=10.0).to(DEV)
def pad_channels(old, new, seed=0):
    """Transplant a checkpoint with fewer state channels.

    Perception is grouped per channel, so channel c occupies input indices
    3c..3c+2 in both models -- the old channels keep their slots. New channels
    get zero INPUT weights (nothing the old rule reads changes) and small random
    OUTPUT weights (they take values, so gradient reaches their input weights).
    """
    g = torch.Generator().manual_seed(seed); out = {}
    for k, v in new.items():
        if any(t in k for t in ("all_filters", "gaussian_noise", "read_only")):
            out[k] = v; continue          # fixed Sobel bank / channel config: keep the new ones
        o = old[k]
        if o.shape == v.shape:
            out[k] = o; continue
        t = torch.zeros_like(v)
        if k.endswith("rule.0.weight"):                       # (hidden, 3*C, 1, 1)
            t[:, : o.shape[1]] = o                            # rest stays ZERO
        elif k.endswith(".weight") and v.dim() == 4:          # final conv (C, hidden, 1, 1)
            t[: o.shape[0]] = o
            t[o.shape[0]:] = torch.randn(v.shape[0] - o.shape[0], *o.shape[1:], generator=g) * 0.01
        else:
            t[: o.shape[0]] = o
        out[k] = t
        print(f"  padded {k}: {tuple(o.shape)} -> {tuple(v.shape)}")
    return out


if args.init_from:
    sd = torch.load(args.init_from, map_location=DEV, weights_only=False)
    sd = sd["nca"] if "nca" in sd else sd.get("model", sd)
    nca.load_state_dict(pad_channels(sd, nca.state_dict()) if args.pad_channels else sd)
    print(f"warm-started from {args.init_from}{' (channel-padded)' if args.pad_channels else ''}")
opt = torch.optim.Adam(nca.parameters(), lr=args.lr)
print(f"run {run}: lengths {LENS}, K load/exec {args.k_load}/{args.k_exec}, "
      f"params {sum(p.numel() for p in nca.parameters() if p.requires_grad):,}")


# ── data ────────────────────────────────────────────────────────────────────

def sample_case(n, rng):
    prog = sample_program(n, 4, ops=OPS, copy_frac=0.0, rng=rng, nop_frac=args.nop_frac)
    regs = sample_regs(4, 4, rng=rng)
    if args.balance_subleq:
        trace = run_program(regs, prog, 4)
        # ~half of the SUBLEQs should refuse: resample register values a few times
        for _ in range(16):
            sl = [(i, ins) for i, ins in enumerate(prog) if ins.op == "SUBLEQ" and ins.a != ins.b]
            if not sl:
                break
            i, ins = rng.choice(sl)
            r = (trace[i][ins.a] - trace[i][ins.b]) & 15
            if (0 < r < 8) == (rng.random() < 0.5):
                break
            regs = sample_regs(4, 4, rng=rng); trace = run_program(regs, prog, 4)
    return prog, regs


def make_batch(bs, n, rng, headroom=None, cases=None):
    hr = rng.randint(0, args.headroom_max) if headroom is None and args.headroom_max else (headroom or 0)
    L = layout(n, hr)
    x0, tg_d, tg_m, ch_m, injs, progs, regs0 = [], [], [], [], [], [], []
    for i in range(bs):
        prog, regs = cases[i] if cases else sample_case(n, rng)
        x = np.zeros((args.channels, L.H, L.W), np.float32)
        x[0] = L.data_screen(regs); x[1:4] = L.mask_channels_at(prog, 0)
        if not args.no_clock:
            x[CLOCK] = 1.0 if L.is_exec(0) else -1.0
            x[RAMP] = 1.0
        x0.append(x)
        tg_d.append(np.stack([L.paint(prog, regs, k + 1) for k in range(L.n_stages)]))
        # a LOAD's target has the freed top slot EMPTY -- the rule only shifts; the feed
        # paints the next instruction there afterwards (blank slot = a NOP row)
        tg_m.append(np.stack([L.mask_channels_at(prog, k + 1, injected=True)
                              for k in range(L.n_stages)]))
        injs.append(np.stack([(L.top_slot_band(prog, k + 1) if L.phase_of(k) == "LOAD"
                               else np.zeros((3, L.H, L.W), np.float32))
                              for k in range(L.n_stages)]))
        ch_m.append(np.stack([L.stage_mask(prog, k + 1) for k in range(L.n_stages)]))
        progs.append(prog); regs0.append(regs)
    t = lambda a: torch.from_numpy(np.stack(a)).to(DEV)
    return L, t(x0), t(tg_d), t(tg_m), t(ch_m), t(injs), progs, regs0


# ── training step: backward per stage, detach at stage boundaries ───────────

def teacher_frac(step):
    if not args.teacher_anneal:
        return args.teacher_frac
    return args.teacher_frac * max(0.0, 1.0 - step / args.teacher_anneal)


def train_step(L, x, tg_d, tg_m, ch_m, p_teacher=0.0, max_stages=0, start=0, inj=None):
    """``start``: absolute stage index the state is at (nonzero for pooled states).
    ``inj``: (B, n_st, 3, H, W) incoming instruction bands, painted into the slot the
    shift frees at the end of each LOAD -- the rule shifts, the feed paints."""
    total = 0.0
    losses = []
    parts = {}                                # per-phase, per-term loss for the logs
    n_st = tg_d.shape[1] - start
    if max_stages:
        n_st = min(n_st, max_stages)
    for kk in range(n_st):
        k = start + kk
        phase = L.phase_of(k)
        is_exec = phase == "EXEC"
        K = steps_for(L, k)
        if inj is not None and phase == "LOAD" and args.load_tail < K:
            # Paint the incoming band BEFORE the supervised tail. If the LOAD target
            # asks for an empty top slot and the band is painted after scoring, the
            # rule is trained to clear exactly the slot the feed then fills -- and it
            # goes on clearing it through the next EXEC.
            head = nca(x, K - args.load_tail)[:, -1].clone()
            # ASSIGN, not add: the slot is not reliably empty when the feed arrives
            # (measured residue mean 0.038, p90 0.073), and adding on top of it pushed
            # pixels to 1.64 against a target of 1.0 -- an overshoot the rule is then
            # scored for. Paint only where the band is, leave the rest of the slot alone.
            band = (inj[:, k].abs() > 0.5).float()
            head[:, 1:4] = head[:, 1:4] * (1.0 - band) + inj[:, k] * band
            seq = nca(head, args.load_tail)
        else:
            seq = nca(x, K)                               # (B, K+1, C, H, W)
        if args.simple_loss:
            N = args.simple_loss
            nch = 4 if args.no_clock else 6
            tail = seq[:, -N:, :nch]                                      # data, A, B, R (+ phase, ramp)
            clk_t = 1.0 if is_exec else -1.0
            B = tail.shape[0]
            ramp_t = (1.0 - 2.0 * torch.arange(K - N + 1, K + 1, device=x.device) / K).view(1, N, 1, 1, 1)
            parts = [tg_d[:, k, None, None].expand(B, N, 1, L.H, L.W),
                     tg_m[:, k, None].expand(B, N, 3, L.H, L.W)]
            if not args.no_clock:
                parts += [torch.full((B, N, 1, L.H, L.W), clk_t, device=x.device),
                          ramp_t.expand(B, N, 1, L.H, L.W)]
            tgt = torch.cat(parts, 2)
            loss = ((tail - tgt) ** 2).mean() / n_st
            if args.no_detach:
                losses.append(loss); x = seq[:, -1]
            else:
                loss.backward(); x = seq[:, -1].detach()
            total += float(loss.detach())
            continue
        fr = seq[:, 1:]
        # Two windows per cycle, nothing else scored: the last --load-tail frames of
        # LOAD and the last --boundary-tail frames of EXEC. Inside a window every frame
        # is scored, every channel, every pixel, against the same static target.
        if phase == "LOAD":
            fr = fr[:, -args.load_tail:]
        elif phase == "EXEC":
            tail = args.boundary_tail if args.drain else args.exec_tail
            if tail:
                fr = fr[:, -tail:]
        d = fr[:, :, 0]; m = fr[:, :, 1:4]
        # the clock, every frame: phase held at +1 (EXEC) / -1 (LOAD); ramp runs +1 -> -1
        clk_t = 1.0 if is_exec else -1.0
        ramp_t = (1.0 - 2.0 * torch.arange(1, K + 1, device=x.device) / K).view(1, K, 1, 1)
        cl = 0.0 if args.no_clock else (((seq[:, 1:, CLOCK] - clk_t) ** 2).mean()
              + ((seq[:, 1:, RAMP] - ramp_t) ** 2).mean()) * args.clock_weight
        # Full-frame MSE, one term per channel group. There is no "background": a
        # pixel that should be empty is part of the target picture exactly like a
        # pixel that should be lit, and is scored the same.
        dl = ((d - tg_d[:, k, None]) ** 2).mean()
        tm = tg_m[:, k, None]
        ml = ((m - tm) ** 2).mean() * args.mask_weight
        # Hidden channels to zero, scoped STRICTLY to the boundary window -- zero is a
        # constant and constants are attractors; leaking this target into the compute
        # frames collapses the rule to writing zero everywhere. The hidden norm is the
        # rule's clock (measured: 0 -> 4.8 -> 0.03 over a cycle); it is flattened only
        # AT the boundary, never during the cycle.
        hl = 0.0
        if phase == "EXEC" and args.drain and args.hidden_weight:
            hl = args.hidden_weight * (seq[:, -args.boundary_tail:, 4:] ** 2).mean()
        loss = (dl + ml + cl + hl) / n_st
        pp = parts.setdefault(phase, {"data": 0.0, "mask": 0.0, "hid": 0.0,
                                      "tot": 0.0, "n": 0})
        pp["data"] += float(dl); pp["mask"] += float(ml)
        pp["hid"] += float(hl); pp["tot"] += float(dl + ml + cl + hl); pp["n"] += 1
        if args.no_detach:
            losses.append(loss)                          # one graph over the whole program
            x = seq[:, -1]
        else:
            loss.backward()                              # per stage, then cut the graph
            x = seq[:, -1].detach()
        total += float(loss.detach())
        # teacher forcing: with prob p, the next stage starts from the CORRECT visible
        # state (hidden channels are kept); otherwise from whatever the rule produced
        if k + 1 < n_st and p_teacher > 0 and random.random() < p_teacher:
            x = x.clone()
            x[:, 0] = tg_d[:, k]; x[:, 1:4] = tg_m[:, k]          # clocks are NOT reset
        # the boundary must be an ATTRACTOR: hand the next cycle a slightly dirty state
        # and still require it to land clean, or the map is only fitted where it is exact
        if phase == "EXEC" and args.drain and args.boundary_noise and k + 1 < n_st:
            x = x.clone()
            x[:, 4:] = x[:, 4:] + torch.randn_like(x[:, 4:]) * args.boundary_noise
    if losses:
        torch.stack(losses).sum().backward()
    for pp in parts.values():                 # mean over the stages of that phase in this batch
        n = max(pp.pop("n"), 1)
        for key in pp:
            pp[key] /= n
    return total, x.detach(), parts


# ── evaluation ──────────────────────────────────────────────────────────────

@torch.no_grad()
def run_program_nca(L, x, n_st, inj=None, start=0):
    for kk in range(n_st):
        k = start + kk
        K = steps_for(L, k)
        hit = (K - args.load_tail) if (inj is not None and L.phase_of(k) == "LOAD") else -1
        for i in range(K):
            x = nca(x, 1)[:, -1]
            if i + 1 == hit:                          # feed paints mid-stage, as in training
                x = x.clone()
                band = (inj[:, k].abs() > 0.5).float()
                x[:, 1:4] = x[:, 1:4] * (1.0 - band) + inj[:, k] * band
    return x


@torch.no_grad()
def evaluate(n, bs):
    L, x, tg_d, tg_m, ch_m, inj, progs, regs0 = make_batch(bs, n, random.Random(1234 + n))
    xf = run_program_nca(L, x, tg_d.shape[1], inj).cpu().numpy()
    ok_bits = ok_exact = 0
    for b in range(bs):
        want = np.array([[int(c) for c in format(v, "04b")] for v in run_program(regs0[b], progs[b], 4)[-1]])
        got = L.decode_bits(xf[b, 0])
        ok_bits += (got == want).mean(); ok_exact += float((got == want).all())
    return ok_bits / bs, ok_exact / bs


@torch.no_grad()
def evaluate_depth(depth, bs):
    """Exact match of the VISIBLE state (data grid + instruction channels) after
    exactly ``depth`` sub-stages -- the metric the stage ladder is gated on."""
    # DRAIN is merged into EXEC: two sub-stages per cycle, always
    ns = [n for n in LENS if n and 2 * n >= depth] or [max(LENS)]
    ok = 0.0
    for n in ns:
        L, x, tg_d, tg_m, _, inj, progs, regs0 = make_batch(bs, n, random.Random(4321 + n + depth))
        xf = run_program_nca(L, x, depth, inj).cpu().numpy()
        for b in range(bs):
            d_ok = (L.decode_bits(xf[b, 0]) ==
                    np.array([[int(c) for c in format(v, "04b")]
                              for v in run_program(regs0[b], progs[b], 4)[L.execs_done(depth)]])).all()
            # run_program_nca paints the incoming band after a LOAD, so compare against
            # the POST-injection picture; tg_m[depth-1] is the pre-injection target and
            # differs from it by exactly one band (~105 px)
            want_m = L.mask_channels_at(progs[b], depth, injected=True)
            bad = int((np.abs(xf[b, 1:4] - want_m) > 0.5).sum())
            m_ok = bad <= args.stage_tol_px
            ok += float(d_ok and m_ok)
    return ok / (bs * len(ns))


@torch.no_grad()
def evaluate_extrap(cycles, bs, hidden=False):
    """Run ``cycles`` complete cycles of a program LONGER than anything trained on and
    ask for an exact visible state at the final boundary.

    This is the only metric that distinguishes a position-independent step from a
    counted sequence: nothing in training ever showed the rule cycle 17.
    """
    n = cycles                                   # one instruction per cycle
    L = layout(n)
    depth = L.phases * cycles
    if depth > L.n_stages:
        return float("nan"), float("nan")
    Lb, x, tg_d, tg_m, _, inj, progs, regs0 = make_batch(bs, n, random.Random(777 + cycles))
    xf = run_program_nca(Lb, x, depth, inj)
    hid = float(xf[:, 4:].abs().mean()) if hidden else float("nan")
    xf = xf.cpu().numpy()
    ok = 0.0
    for b in range(bs):
        want_d = np.array([[int(c) for c in format(v, "04b")]
                           for v in run_program(regs0[b], progs[b], 4)[Lb.execs_done(depth)]])
        d_ok = (Lb.decode_bits(xf[b, 0]) == want_d).all()
        want_m = Lb.mask_channels_at(progs[b], depth, injected=True)
        m_ok = int((np.abs(xf[b, 1:4] - want_m) > 0.5).sum()) <= args.stage_tol_px
        ok += float(d_ok and m_ok)
    return ok / bs, hid


@torch.no_grad()
def snapshot(path, step):
    """RGB GIF: fixed 3-instruction example + 2 random (lengths 4 and 2); frames over all stages."""
    rows = []
    for n, seed in ((3, 9), (4, step + 1), (2, step + 2)):
        L, x, tg_d, tg_m, _, inj, progs, _ = make_batch(1, n, random.Random(seed))
        frames = [x[0, :6].cpu().numpy()]
        for k in range(tg_d.shape[1]):
            K = steps_for(L, k)
            hit = (K - args.load_tail) if L.phase_of(k) == "LOAD" else -1
            for i in range(K):
                x = nca(x, 1)[:, -1]
                if i + 1 == hit:                       # the feed paints, as in training
                    x = x.clone()
                    band = (inj[:, k].abs() > 0.5).float()
                    x[:, 1:4] = x[:, 1:4] * (1.0 - band) + inj[:, k] * band
                frames.append(x[0, :6].cpu().numpy())

        tgt = (tg_d[0, -1].cpu().numpy(), tg_m[0, -1].cpu().numpy())
        rows.append((frames, tgt, L))
    T = max(len(r[0]) for r in rows)
    imgs = []
    for t in range(0, T, 2):
        panel_rows = []
        for frames, (td, tm), L in rows:
            f = frames[min(t, len(frames) - 1)]
            sep = np.full((L.H, 3, 3), 40, np.uint8)
            row = np.concatenate([TileLayoutV9.to_rgb(frames[0][0], frames[0][1:4]), sep,
                                  TileLayoutV9.to_rgb(f[0], f[1:4]), sep, TileLayoutV9.to_rgb(td, tm)], 1)
            # clock strip under the row: green = +1 (EXEC), red = -1 (LOAD), by the rule's own mean value
            c = float(np.clip(f[CLOCK].mean(), -1, 1)); r_ = float(np.clip(f[RAMP].mean(), -1, 1))
            strip = np.zeros((4, row.shape[1], 3), np.uint8)
            strip[:2, :, 1] = int(255 * max(c, 0)); strip[:2, :, 0] = int(255 * max(-c, 0))     # phase: green/red
            strip[2:, : int(row.shape[1] * (r_ + 1) / 2), 2] = 255                              # ramp: blue bar length
            panel_rows.append(np.concatenate([row, strip], 0))
        W = max(r.shape[1] for r in panel_rows); Hs = [r.shape[0] for r in panel_rows]
        canvas = np.full((sum(Hs) + 3 * (len(Hs) - 1), W, 3), 40, np.uint8); y = 0
        for r in panel_rows:
            canvas[y: y + r.shape[0], : r.shape[1]] = r; y += r.shape[0] + 3
        im = Image.fromarray(canvas); imgs.append(im.resize((im.width * 4, im.height * 4), Image.NEAREST))
    imgs[0].save(str(path), save_all=True, append_images=imgs[1:], duration=70, loop=0)


# ── loop ────────────────────────────────────────────────────────────────────

tlog = (run_dir / "train.jsonl").open("a"); elog = (run_dir / "metrics.jsonl").open("a")
best = -1.0
bar = tqdm(range(args.steps), dynamic_ncols=True)
# ── replay pool: mid-program states, bucketed by (start stage, program length) ──
# Only populated once the ladder reaches --pool-after-rung: pooling stages the rule
# cannot yet produce fills the buffer with unrecoverable states, which is how the
# earlier pool attempt stalled at depth ~3 of 15.
_pool = {}


def pool_push(stage, xs, progs, regs0, n):
    if not args.pool_size:
        return
    cap = max(8, args.pool_size // 16)
    b = _pool.setdefault((stage, n), [])
    for i in range(len(progs)):
        if stage < 2 * n:                                          # still inside the program
            b.append((xs[i].cpu().numpy(), progs[i], regs0[i]))
    if len(b) > cap:
        del b[: len(b) - cap]


def pool_draw(rng, bs, depth):
    keys = [k for k, v in _pool.items()
            if len(v) >= bs and k[0] + depth <= 2 * k[1]]
    if not keys:
        return None
    st, n = keys[rng.randrange(len(keys))]
    ent = [_pool[(st, n)][rng.randrange(len(_pool[(st, n)]))] for _ in range(bs)]
    cases = [(e[1], e[2]) for e in ent]
    L, _, tg_d, tg_m, ch_m, inj, progs, regs0 = make_batch(bs, n, rng, cases=cases)
    x = torch.from_numpy(np.stack([e[0] for e in ent])).to(DEV)
    return L, x, tg_d, tg_m, ch_m, inj, progs, regs0, st, n


# the rung is a CYCLE, not a sub-stage: --max-cycles N trains the first N complete
# cycles. A half cycle (LOAD alone) is a state the machine is never in at rest, so a
# rung that stops there teaches nothing that composes into the next one.
PHASES = 2
_stage = {"depth": PHASES * args.max_cycles if args.max_cycles else args.max_stages}
EXTRAP = [int(x) for x in args.extrap_cycles.split(",") if x.strip()]
for step in bar:
    start = 0; drawn = None
    if (args.pool_size and _stage["depth"] >= args.pool_after_rung
            and rng.random() < args.pool_frac):
        drawn = pool_draw(rng, args.batch_size, _stage["depth"])
    lens = active_lens(step)
    if _stage["depth"]:                      # the stack must be deep enough to need that many stages
        lens = [m for m in lens if m and PHASES * m >= _stage["depth"]] or [max(LENS)]
    if drawn:
        L, x, tg_d, tg_m, ch_m, inj, progs_b, regs_b, start, n = drawn
    else:
        n = rng.choice(lens)
        L, x, tg_d, tg_m, ch_m, inj, progs_b, regs_b = make_batch(args.batch_size, n, rng)
    opt.zero_grad(set_to_none=True)
    loss, x_end, parts = train_step(L, x, tg_d, tg_m, ch_m, p_teacher=teacher_frac(step),
                                    max_stages=_stage["depth"], inj=inj, start=start)
    # Clip, do not merely measure. Unclipped, pool1 took a 9814-norm step at 137k and
    # the weights never recovered (cyc1 1.00 -> 0.00); the peak norm reached 8.2e6.
    gn = torch.nn.utils.clip_grad_norm_(nca.parameters(), args.grad_clip or float("inf")).item()
    opt.step()
    if args.pool_size and _stage["depth"] >= args.pool_after_rung:
        pool_push(start + _stage["depth"], x_end, progs_b, regs_b, n)
    if step % args.log_every == 0:
        rowt = {"step": step, "loss": loss, "grad_norm": gn, "n_instr": n,
                "teacher": round(teacher_frac(step), 3), "cur_max": _cur["max"]}
        for ph, pp in parts.items():          # e.g. LOAD_mask, EXEC_data, EXEC_hid, EXEC_tot
            for key, v in pp.items():
                rowt[f"{ph}_{key}"] = round(v, 6)
        tlog.write(json.dumps(rowt) + "\n")
        if step % (args.log_every * 20) == 0:
            tlog.flush()
    if step % args.eval_every == 0:
        rec = {"step": step, "loss": loss, "grad_norm": gn}
        for m in EVAL_LENS:
            b, e = evaluate(m, args.eval_batch); rec[f"bits_L{m}"] = round(b, 4); rec[f"exact_L{m}"] = round(e, 4)
        rec["cur_max"] = _cur["max"]
        if _stage["depth"]:
            ds = evaluate_depth(_stage["depth"], args.eval_batch)
            rec["stage_depth"] = _stage["depth"]; rec["stage_ok"] = round(ds, 4)
            if (ds >= args.stage_gate and not args.max_cycles
                    and _stage["depth"] < PHASES * max(LENS)):
                _stage["depth"] += 1; rec["stage_grew"] = _stage["depth"]
        # extrapolation: cycles far past anything trained on. If the boundary really is
        # canonical these should all hold; if the rule is counting, they fall off a cliff.
        if args.drain:
            deep = 0
            for c in EXTRAP:
                e, hid = evaluate_extrap(c, args.eval_batch, hidden=(c == EXTRAP[0]))
                rec[f"cyc{c}"] = None if e != e else round(e, 4)
                if c == EXTRAP[0] and hid == hid:
                    rec["hid_at_boundary"] = round(hid, 5)
                if e == e and e >= 0.5:
                    deep = c
            rec["deepest_cycle"] = deep
        # only the lengths actually evaluated: active_lens() is the SAMPLER's range and
        # need not be a subset of --eval-instr (KeyError when the two were set apart)
        scored = [rec[f"exact_L{m}"] for m in active_lens(step) if f"exact_L{m}" in rec]
        score = float(np.mean(scored)) if scored else 0.0
        maybe_promote(step, rec)
        rec["score"] = score; elog.write(json.dumps(rec) + "\n"); elog.flush()
        torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)}, run_dir / "nca_latest.pt")
        if score > best:
            best = score; torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)}, run_dir / "nca_best.pt")
        bar.set_description((f"d{_stage['depth']}:{rec.get('stage_ok',0):.2f} " if _stage["depth"] else "")
                            + f"loss {loss:.4f} "
                            + (f"deep{rec.get('deepest_cycle',0)} h{rec.get('hid_at_boundary',0):.3f} "
                               if args.drain else "")
                            + " ".join(f"L{m}:{rec[f'exact_L{m}']:.2f}" for m in EVAL_LENS))
    if args.ckpt_every and step % args.ckpt_every == 0:
        torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)}, run_dir / f"nca_{step:07d}.pt")
    if args.snapshot_every and step % args.snapshot_every == 0:
        snapshot(run_dir / f"snapshot_{step:07d}.gif", step)
torch.save({"nca": nca.state_dict(), "step": args.steps, "config": vars(args)}, run_dir / "nca_last.pt")
print("done:", run_dir)
