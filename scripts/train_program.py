#!/usr/bin/env python3
"""Train one program-agnostic NCA rule to execute painted straight-line programs.

The grid is a program listing: ``data | opcode | data | opcode | data``.  Only
data column 0 (the program input) is filled at t=0; every downstream data column
must be produced by the rule.  The rule sees a random program every step and is
never trained on the programs it will be evaluated on.

Supervision is staged: after ``K`` steps per instruction, data columns ``0..i``
must all be correct -- the new one must have appeared and the earlier ones must
still be holding their values.

    uv run python scripts/train_program.py --ops ADD,COPY --instr-min 1 --instr-max 2
    uv run python scripts/train_program.py --steps 30000 --device cuda
"""

import json
import random
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.gif import write_gif  # noqa: E402
from ncpu.nca import NeuralCA  # noqa: E402
from ncpu.program import (  # noqa: E402
    OPS,
    TileLayoutV2,
    TileLayoutV5,
    TileLayoutV7,
    TileLayoutV8,
    layout_for,
    layout_v2_for,
    layout_v5_for,
    layout_v7_for,
    run_program,
    sample_program,
    sample_regs,
)

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=30_000)
parser.add_argument("--device", default="cuda")
parser.add_argument("--batch-size", type=int, default=8)
# task
parser.add_argument("--ops", default="ADD,COPY", help="comma-separated subset of the ISA")
parser.add_argument("--layout", choices=["v1", "v2", "v5", "v7", "v8"], default="v1",
                    help="v1 = stacked data|opcode columns; v2 = registers side by side + 4 type "
                         "channels; v5 = MEM in (selA/selB above) | MEM out (dst below), op in the "
                         "dst sign (SUB/SUBLEQ), + glyph for unset bits, 3 type channels")
parser.add_argument("--stages-per-tile", type=int, default=2,
                    help="v5: 2 = COPY then STORE supervised separately, 1 = both at once")
parser.add_argument("--settle-move", type=int, default=0,
                    help="v7: P, CA steps for the MOVE stage (K is the COMPUTE stage); 0 = same as K")
parser.add_argument("--regs", type=int, default=4)
parser.add_argument("--word-bits", type=int, default=4)
parser.add_argument("--instr-min", type=int, default=1)
parser.add_argument("--instr-max", type=int, default=2)
parser.add_argument("--copy-frac", type=float, default=0.35, help="oversample COPY")
parser.add_argument("--balance-subleq", action="store_true",
                    help="resample register values so half of SUBLEQ tiles have a POSITIVE result "
                         "(refuse) and half <= 0 (write); uniform values give only ~30%% refuses")
parser.add_argument("--nop-frac", type=float, default=0.0,
                    help="v5: fraction of tiles with NO selectors painted (= copy-through NOP)")
parser.add_argument("--addressing", default="free",
                    choices=["identity", "fixed-dst", "fixed-src", "free"],
                    help="how much routing the rule must do; see sample_program")
# geometry
parser.add_argument("--radius", type=float, default=2, help="0 = one pixel per blob")
parser.add_argument("--pitch", type=int, default=6, help="vertical blob spacing")
parser.add_argument("--col-pitch", type=int, default=24, help="horizontal column spacing")
parser.add_argument("--unknown", choices=["bit0", "zero"], default="bit0",
                    help="value in unresolved data blobs; 'zero' restores the third state")
parser.add_argument("--ycoord", action="store_true",
                    help="read-only y-coordinate channel giving each blob row an identity. "
                         "Only y: an x channel would break generalisation to unseen program "
                         "lengths, but lanes running along x need to know which lane they are.")
parser.add_argument("--op-amp", type=float, default=1.0,
                    help="opcode blob amplitude; <1 makes program text locally distinct from data")
parser.add_argument("--loss", choices=["mse", "l1", "huber"], default="mse",
                    help="l1/huber shrink the abstain basin: hedging costs 1 and a wrong bit 2, "
                         "instead of 1 and 4 under mse")
parser.add_argument("--scaffold-cell", type=int, default=None,
                    help="carpet cell size in px; default keys columns to col-pitch (coarse)")
parser.add_argument("--no-scaffold", action="store_true",
                    help="v1: drop the carpet on the target background")
parser.add_argument("--scaffold", action="store_true",
                    help="v2: paint the carpet (off by default -- plain background)")
# model
parser.add_argument("--channels", type=int, default=16)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--kernel", type=int, default=5)
parser.add_argument("--fire-rate", type=float, default=0.5)
parser.add_argument("--state-bound", type=float, default=10.0)
# rollout / loss
parser.add_argument("--settle", type=int, default=0,
                    help="K, CA steps per instruction; 0 = 2x the travel-time floor")
parser.add_argument("--bg-last", type=int, default=0,
                    help="adder loss: also score the non-data pixels of channel 0 against 0 on the "
                         "LAST N frames of each stage (weight --bg-weight); 0 = off")
parser.add_argument("--dst-weight", type=float, default=1.0,
                    help="adder loss: extra weight on the bits that must CHANGE (the stage masks), "
                         "relative to the held bits")
parser.add_argument("--adder-loss", action="store_true",
                    help="the 8-bit adder recipe: MSE on the output-bit pixels only, every frame "
                         "of the window, normalised by the mask size (not the image size)")
parser.add_argument("--supervise-mask", choices=["all", "output"], default="all",
                    help="all = whole image, weighted; output = ONLY the data blobs the rule "
                         "must write (every data grid after the input), nothing else")
parser.add_argument("--bg-weight", type=float, default=0.0,
                    help="output mode: weight on holding the background at 0 (unsupervised "
                         "background was measured to blow up and swamp the bits)")
parser.add_argument("--sel-weight", type=float, default=0.0,
                    help="output mode: weight on keeping the painted selector blobs intact")
parser.add_argument("--data-weight", type=float, default=50.0)
parser.add_argument("--op-weight", type=float, default=10.0)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--grad-clip", type=float, default=10.0,
                    help="a clip that binds every step turns training into a random walk; 0 = off")
parser.add_argument("--pad-jitter", type=int, default=0,
                    help="randomise the left/right margin by up to N px per batch. Zero padding "
                         "makes the grid edge a free positional cue, and data column 0 is always "
                         "beside it -- a rule can anchor on the edge, which works for the first "
                         "gap at any width and cannot work for later gaps.")
parser.add_argument("--hold-weight", type=float, default=1.0,
                    help="weight on holding not-yet-computed columns BLANK across the whole "
                         "stage window. Staying blank is achievable from t=0, unlike having the "
                         "answer, so it is supervised over the full window rather than the tail. "
                         "0 disables; without it the rule may latch garbage into a downstream "
                         "column long before its input column exists.")
parser.add_argument("--no-detach-stages", action="store_true",
                    help="backprop across gap boundaries. Off by default: each stage carries "
                         "its own supervision at its own time, so truncating there costs "
                         "little and keeps memory flat in program length.")
parser.add_argument("--supervise-last", type=int, default=16,
                    help="supervise the last N frames of each stage window. NOT the whole "
                         "window: a frame earlier than the travel-time floor cannot hold the "
                         "answer, and penalising it drives the rule to emit the mean (grey).")
parser.add_argument("--zero-init", action="store_true",
                    help="zero the rule's last layer; measured to kill the run (ReLUs die)")
# bookkeeping
parser.add_argument("--eval-every", type=int, default=250)
parser.add_argument("--log-every", type=int, default=10,
                    help="write step/loss/grad_norm to train.jsonl this often (eval is separate)")
parser.add_argument("--eval-batch", type=int, default=8)
parser.add_argument("--eval-instr", default="", help="extra program lengths to eval, e.g. 3,4,8")
parser.add_argument("--snapshot-every", type=int, default=500,
                    help="save last-frame snapshot PNG + rollout GIF + nca_latest.pt this often")
parser.add_argument("--gif-stride", type=int, default=2, help="keep every Nth rollout frame in the GIF")
parser.add_argument("--gif-rows", type=int, default=4,
                    help="examples per GIF: the fixed one on top, then fresh random ones")
parser.add_argument("--ckpt-every", type=int, default=1000, help="save a numbered nca_<step>.pt this often")
parser.add_argument("--tag", default="v1")
parser.add_argument("--init-from", default=None,
                    help="checkpoint .pt to warm-start the rule from (next rung of the ladder)")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
rng = random.Random(args.seed)
DEVICE = args.device
OPSET = tuple(o.strip().upper() for o in args.ops.split(",") if o.strip())
if args.layout in ("v5", "v7", "v8"):
    assert all(o in TileLayoutV5.ISA for o in OPSET), f"ISA is {TileLayoutV5.ISA}, got {OPSET}"
else:
    assert all(o in OPS for o in OPSET), f"unknown op in {OPSET}, ISA is {OPS}"

_V8 = {}


def layout_v8_for(n_instr, **kw):
    key = tuple(sorted(kw.items()))
    if key not in _V8:
        _V8[key] = TileLayoutV8(**kw)
    return _V8[key]


if args.layout == "v8":
    GEOM = dict(n_regs=args.regs, word_bits=args.word_bits, r=args.radius,
                pitch=max(args.pitch, 2 * args.radius + 4))       # v8 wants 3 px between bits
    layout_for = layout_v8_for
    TYPE_CH = []                        # v8: instruction channels are painted, NOT read-only
    MASK_CH = [1, 2, 3]
elif args.layout == "v7":
    GEOM = dict(n_regs=args.regs, word_bits=args.word_bits, r=args.radius, pitch=args.pitch)
    layout_for = layout_v7_for
    TYPE_CH = list(range(1, 1 + len(TileLayoutV7.TYPES)))       # data / sel_in / sel_out
elif args.layout == "v5":
    GEOM = dict(n_regs=args.regs, word_bits=args.word_bits, r=args.radius, pitch=args.pitch,
                stages=args.stages_per_tile)
    layout_for = layout_v5_for
    TYPE_CH = list(range(1, 1 + len(TileLayoutV5.TYPES)))       # data / sel_in / sel_out
elif args.layout == "v2":
    GEOM = dict(n_regs=args.regs, word_bits=args.word_bits, r=args.radius, pitch=args.pitch,
                unknown=(0.0 if args.unknown == "zero" else -1.0),
                scaffold=args.scaffold, scaffold_cell=args.scaffold_cell or 6)
    layout_for = layout_v2_for
    # channels 1..4 are the fixed one-hot site-type map (data / sel_in / op / sel_out)
    TYPE_CH = list(range(1, 1 + len(TileLayoutV2.TYPES)))
else:
    GEOM = dict(n_regs=args.regs, word_bits=args.word_bits,
                r=args.radius, pitch=args.pitch, col_pitch=args.col_pitch,
                unknown=(0.0 if args.unknown == "zero" else -1.0),
                scaffold=not args.no_scaffold, scaffold_cell=args.scaffold_cell,
                op_amp=args.op_amp)
    TYPE_CH = []
BASE_PAD = {"v2": 8, "v5": 5, "v7": 5, "v8": 6}.get(args.layout, 6)
if args.layout != "v8":
    MASK_CH = []

INSTR_RANGE = list(range(args.instr_min, args.instr_max + 1))
EVAL_INSTR = sorted(set(INSTR_RANGE) |
                    {int(x) for x in args.eval_instr.split(",") if x.strip()})

# K is derived from the geometry, never hard-coded: one gap cannot finish faster
# than the time a bit needs to physically travel to its destination.
PX_PER_STEP = args.kernel // 2
K = args.settle or 2 * layout_for(1, **GEOM).min_rollout(PX_PER_STEP)
P_MOVE = args.settle_move or K


def stage_k(i):
    """CA steps for sub-stage ``i`` (0-based): v7 alternates COMPUTE=K, MOVE=P."""
    return P_MOVE if (args.layout == "v7" and i % 2 == 1) else K

run = f"PROG_{datetime.now():%Y%m%d_%H%M%S}_{args.tag}_{args.layout}_{args.addressing}_w{args.word_bits}_r{args.regs}_s{args.seed}"
run_dir = Path(__file__).resolve().parent.parent / "runs" / run
run_dir.mkdir(parents=True, exist_ok=True)
(run_dir / "config.json").write_text(json.dumps(
    {**vars(args), "opset": list(OPSET), "K": K, "px_per_step": PX_PER_STEP}, indent=2))

L1 = layout_for(1, **GEOM)
print(f"run {run}")
print(f"  ISA {OPSET}, {args.regs} regs x {args.word_bits} bits, program length {INSTR_RANGE}")
print(f"  grid (1 instr) {L1.H}x{L1.W}; travel floor {L1.min_rollout(PX_PER_STEP)} steps/gap -> K={K}"
      + (f", P(move)={P_MOVE}" if args.layout == "v7" else ""))
print(f"  deepest rollout {K * max(INSTR_RANGE)} steps")

# ── model ───────────────────────────────────────────────────────────────────
# No read-only channels and no coordinate channels: the rule must be position
# invariant, or it cannot span program lengths it was not trained on.
nca = NeuralCA(
    channels=args.channels,
    hidden_channels=args.hidden,
    fire_rate=args.fire_rate,
    alive_threshold=0,          # alive masking off: channel 0 is the value plane
    zero_initialization=args.zero_init,
    kernel_size=args.kernel,
    perception="static_sobel",
    padding_type="zeros",       # the grid has real edges; circular would wrap them
    read_only_dims=TYPE_CH + ([args.channels - 1] if args.ycoord else []),
    state_bound=args.state_bound,
).to(DEVICE)
def widen_state(old, new, seed=0):
    """Transplant a narrower rule into a wider one (net2net style).

    Old hidden units keep their weights exactly; new units get small random
    input weights and ZERO output weights, so the wider rule starts out
    computing the identical function and only diverges as training moves the
    new units off zero.
    """
    g = torch.Generator().manual_seed(seed)
    out = {}
    for k, v in new.items():
        o = old[k]
        if o.shape == v.shape:
            out[k] = o
            continue
        t = v.clone()
        if k == "rule.rule.0.weight":            # (hidden, in, 1, 1)
            t[: o.shape[0]] = o
            t[o.shape[0]:] = torch.randn(t.shape[0] - o.shape[0], *o.shape[1:], generator=g) * 0.01
        elif k == "rule.rule.0.bias":            # (hidden,)
            t[: o.shape[0]] = o
            t[o.shape[0]:] = 0.0
        elif k == "rule.rule.2.weight":          # (channels, hidden, 1, 1)
            t[:, : o.shape[1]] = o
            t[:, o.shape[1]:] = 0.0
        else:
            raise ValueError(f"cannot transplant {k}: {tuple(o.shape)} -> {tuple(v.shape)}")
        out[k] = t
        print(f"  widened {k}: {tuple(o.shape)} -> {tuple(v.shape)}")
    return out


if args.init_from:
    old = torch.load(args.init_from, map_location=DEVICE, weights_only=False)["nca"]
    nca.load_state_dict(widen_state(old, nca.state_dict()))
    print(f"  warm-started from {args.init_from}")
opt = torch.optim.Adam(nca.parameters(), lr=args.lr)
print(f"  params {sum(p.numel() for p in nca.parameters() if p.requires_grad):,}")

# ── batching ────────────────────────────────────────────────────────────────
_weight_cache = {}
_hold_cache = {}
_LAST_SMASK = None
_LAST_MCH = None
_aux_cache = {}


def aux_mask(layout, kind):
    key = (kind, layout.n_instr, layout.H, layout.W)
    if key not in _aux_cache:
        m = layout.program_mask() if kind == "sel" else (1.0 - layout.slot_mask())
        _aux_cache[key] = torch.from_numpy(m.astype(np.float32)).to(DEVICE)
    return _aux_cache[key]


def hold_mask(layout, stage):
    """Weight over the data columns that stage ``stage`` must still leave blank."""
    key = (layout.n_instr, stage, layout.H, layout.W)
    if key not in _hold_cache:
        future = range(stage + 2, layout.n_instr + 1)
        m = layout.data_mask(columns=future)
        _hold_cache[key] = torch.from_numpy(m).to(DEVICE)
    return _hold_cache[key]


def weight_map(layout):
    """Per-pixel loss weight: blobs matter far more than the empty background
    (a one-pixel blob is ~1% of the grid)."""
    key = (layout.n_instr, layout.H, layout.W)     # pad jitter changes the size
    if key not in _weight_cache:
        if args.supervise_mask == "output":
            w = layout.data_mask(columns=range(1, layout.n_instr + 1))
        else:
            w = np.ones((layout.H, layout.W), dtype=np.float32)
            w += args.data_weight * layout.data_mask()
            w += args.op_weight * layout.program_mask()
        _weight_cache[key] = torch.from_numpy(w).to(DEVICE)
    return _weight_cache[key]


def sample_batch(bs, n_instr, rng=rng, pad=None):
    geom = dict(GEOM)
    if pad is None and args.pad_jitter:
        pad = BASE_PAD + rng.randrange(args.pad_jitter + 1)
    if pad is not None:
        geom["pad"] = pad
    layout = layout_for(n_instr, **geom)
    if layout not in _LAYOUTS_SEEN:
        _LAYOUTS_SEEN.append(layout)
    inps, tgts, progs, regs0, smasks, mchs = [], [], [], [], [], []
    for _ in range(bs):
        prog = sample_program(n_instr, args.regs, ops=OPSET, copy_frac=args.copy_frac,
                              rng=rng, addressing=args.addressing, nop_frac=args.nop_frac)
        regs = sample_regs(args.regs, args.word_bits, rng=rng)
        if args.balance_subleq and prog[0].op == "SUBLEQ" and prog[0].a != prog[0].b:
            want_refuse = rng.random() < 0.5
            for _ in range(64):                      # rejection-sample the operand values
                r = (regs[prog[0].a] - regs[prog[0].b]) & ((1 << args.word_bits) - 1)
                positive = 0 < r < (1 << (args.word_bits - 1))
                if positive == want_refuse:
                    break
                regs = sample_regs(args.regs, args.word_bits, rng=rng)
        inps.append(layout.input_screen(prog, regs))
        tgts.append(np.stack(layout.stage_targets(prog, regs)))
        if args.supervise_mask == "output" and hasattr(layout, "stage_mask"):
            smasks.append(np.stack([layout.stage_mask(prog, k + 1)
                                    for k in range(layout.n_stages)]))
        if MASK_CH:
            mchs.append(layout.mask_channels(prog[0]))
        progs.append(prog)
        regs0.append(regs)
    inp = torch.from_numpy(np.stack(inps)).to(DEVICE)           # (B, H, W)
    tgt = torch.from_numpy(np.stack(tgts)).to(DEVICE)           # (B, n_stages, H, W)
    global _LAST_SMASK, _LAST_MCH
    _LAST_SMASK = torch.from_numpy(np.stack(smasks)).to(DEVICE) if smasks else None
    _LAST_MCH = torch.from_numpy(np.stack(mchs)).to(DEVICE) if mchs else None
    return layout, inp, tgt, progs, regs0


_type_cache = {}
_LAYOUTS_SEEN = []


def type_map(h, w):
    """Fixed (4, H, W) site-type one-hot for the layout of this grid size."""
    if (h, w) not in _type_cache:
        lay = next(l for l in _LAYOUTS_SEEN if (l.H, l.W) == (h, w))
        _type_cache[(h, w)] = torch.from_numpy(lay.type_channels()).to(DEVICE)
    return _type_cache[(h, w)]


def implant(inp):
    """State at t=0: the painted screen in channel 0, everything else zero.

    With ``--ycoord`` the LAST channel is a read-only row ramp in [-1, 1], so a
    cell knows which lane it is on.  Nothing carries the x position: the rule
    must stay translation invariant along the program.
    """
    b, h, w = inp.shape
    x = torch.zeros(b, args.channels, h, w, device=inp.device)
    x[:, 0] = inp
    if TYPE_CH:
        x[:, TYPE_CH] = type_map(h, w)
    if MASK_CH and _LAST_MCH is not None:
        x[:, MASK_CH] = _LAST_MCH[: b]        # the A / B / R rectangles, painted, free to evolve
    if args.ycoord:
        y = torch.linspace(-1, 1, h, device=inp.device)[None, :, None].expand(b, h, w)
        x[:, -1] = y
    return x


def advance(x, k, chunk=16):
    """Run ``k`` steps keeping only the final state.

    ``NeuralCA.forward`` stacks every frame it produces, which is 4.6 GB for one
    eval window at K=120; stepping in chunks keeps the peak flat.
    """
    done = 0
    while done < k:
        n = min(chunk, k - done)
        x = nca(x, n)[:, -1]
        done += n
    return x


def rollout(inp, n_stages, k=K, dense=0):
    """Run the CA through every stage window.

    Returns ``(B, n_stages, H, W)`` -- the value channel at each supervision
    point -- or, with ``dense``, a list of ``(B, k, H, W)`` holding every frame
    of each window, so the answer is asked to PERSIST rather than to appear at
    one instant.
    """
    x = implant(inp)
    out = []
    for si in range(n_stages):
        k = stage_k(si)
        if dense:
            head = k - dense
            if head > 0:
                x = advance(x, head)
            seq = nca(x, dense)
            x = seq[:, -1]
            out.append(seq[:, 1:, 0])
            if not args.no_detach_stages:
                x = x.detach()          # truncate BPTT at the gap boundary
        else:
            x = advance(x, k)
            out.append(x[:, 0])
    return out if dense else torch.stack(out, dim=1)


def train_stages(inp, tgt, layout, n_stages, k=K):
    """Walk the stages, backprop each one as soon as its loss is known.

    Holding every stage's graph until a single ``backward()`` makes memory grow
    with program length (two gaps already wanted 16 GB).  Backward per stage,
    then detach at the gap boundary, keeps it flat -- each stage still gets the
    full gradient through its own K steps.
    """
    dense = args.supervise_last
    x = implant(inp)
    total = 0.0
    n_stages = tgt.shape[1]
    for i in range(n_stages):
        k = stage_k(i)
        dense = min(args.supervise_last, k)
        head = k - dense
        headseq = None
        if head > 0:
            if args.hold_weight and i + 1 < n_stages and args.layout != "v5":
                headseq = nca(x, head)      # keep the head frames to supervise "stay blank"
                x = headseq[:, -1]
            else:
                x = advance(x, head)
        seq = nca(x, dense)
        if args.adder_loss:
            # every bit resolved so far: grid 0 (must be HELD while it is computed on)
            # plus whatever earlier stages produced; the target carries the current values
            changed = _LAST_SMASK[:, : i + 1].sum(1, keepdim=True).clamp(max=1.0)
            held = (torch.from_numpy(layout.data_mask(columns=[0])).to(DEVICE) - changed).clamp(min=0.0)
            m = held + args.dst_weight * changed          # result bits can be weighted up
            d2 = (seq[:, 1:, 0] - tgt[:, i, None]) ** 2 * m
            loss = d2.sum() / (m.sum() * d2.shape[1]) / n_stages
            if args.bg_last and args.bg_weight:
                bg = aux_mask(layout, "bg")                # everything that is not a data bit
                tail = seq[:, -args.bg_last:, 0]
                loss = loss + args.bg_weight * ((tail ** 2) * bg).sum() / (bg.sum() * tail.shape[0] * tail.shape[1]) / n_stages
            loss.backward(retain_graph=args.no_detach_stages)
            total += float(loss.detach())
            x = seq[:, -1]
            if not args.no_detach_stages:
                x = x.detach()
            continue
        if _LAST_SMASK is not None:
            # output-only for a staged layout: every blob resolved so far (cumulative stage masks)
            w = (_LAST_SMASK[:, : i + 1].sum(1, keepdim=True)
                 + torch.from_numpy(layout.data_mask(columns=[0])).to(DEVICE)).clamp(max=1.0)
            w = w + args.sel_weight * aux_mask(layout, "sel") + args.bg_weight * aux_mask(layout, "bg")
        else:
            w = weight_map(layout)
        loss = (w * err_fn(seq[:, 1:, 0], tgt[:, i, None])).mean() / n_stages
        if args.hold_weight and headseq is not None and args.layout != "v5":
            hm = hold_mask(layout, i)
            if float(hm.sum()) > 0:
                # the whole window, not just its tail: a column with no input yet
                # can stay blank from t=0, and latching early is what breaks chaining
                blank = torch.full_like(headseq[:, 1:, 0], layout.unknown)
                loss = loss + args.hold_weight * (
                    hm * err_fn(headseq[:, 1:, 0], blank)).mean() / n_stages
        loss.backward(retain_graph=args.no_detach_stages)
        total += float(loss.detach())
        x = seq[:, -1]
        if not args.no_detach_stages:
            x = x.detach()
    return total


def err_fn(p, t):
    d = p - t
    if args.loss == "l1":
        return d.abs()
    if args.loss == "huber":
        return F.huber_loss(p, t.expand_as(p), reduction="none", delta=0.5)
    return d ** 2


def staged_loss(pred, tgt, layout):
    """``pred`` is the dense list or the stacked last frames; ``tgt`` is
    ``(B, n_stages, H, W)``."""
    w = weight_map(layout)

    def err(p, t):
        d = p - t
        if args.loss == "l1":
            return d.abs()
        if args.loss == "huber":
            return F.huber_loss(p, t.expand_as(p), reduction="none", delta=0.5)
        return d ** 2

    if isinstance(pred, list):
        return torch.stack([(w * err(p, tgt[:, i, None])).mean()
                            for i, p in enumerate(pred)]).mean()
    return (w * err(pred, tgt)).mean()


# ── evaluation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(n_instr, bs):
    """Per-column bit accuracy and exact-match on freshly sampled programs."""
    layout, inp, tgt, progs, regs0 = sample_batch(
        bs, n_instr, rng=random.Random(1234 + n_instr), pad=BASE_PAD)   # canonical margin
    n_st = tgt.shape[1]
    pred = rollout(inp, n_st).cpu().numpy()
    per = n_st // n_instr                                   # stages per instruction
    bits_ok = np.zeros(n_instr)
    exact = np.zeros(n_instr)
    for b in range(bs):
        trace = run_program(regs0[b], progs[b], args.word_bits)
        for i in range(n_instr):
            got = layout.decode_bits(pred[b, (i + 1) * per - 1], 0 if args.layout == "v8" else i + 1)
            want = np.array([[int(c) for c in format(v, f"0{args.word_bits}b")]
                             for v in trace[i + 1]])
            bits_ok[i] += (got == want).mean()
            exact[i] += float((got == want).all())
    return bits_ok / bs, exact / bs


def snapshot(path, n_instr=None, step=0):
    global _LAST_MCH
    n_instr = n_instr or INSTR_RANGE[-1]
    layout, inp, tgt, progs, regs0 = sample_batch(1, n_instr, rng=random.Random(9), pad=BASE_PAD)   # A, B, dst on distinct columns
    mch0 = _LAST_MCH
    n_st = tgt.shape[1]
    with torch.no_grad():
        pred = rollout(inp, n_st).cpu().numpy()[0]
    tgt_t = tgt                                     # keep the tensor for the GIF panel
    tgt = tgt.cpu().numpy()[0]
    rows = [("input", inp.cpu().numpy()[0])]
    for i in range(n_st):
        rows += [(f"pred t={sum(stage_k(j) for j in range(i + 1))}", pred[i]), (f"target {i}", tgt[i])]
    fig, axes = plt.subplots(len(rows), 1, figsize=(0.09 * layout.W + 2, 1.1 * len(rows)))
    for ax, (title, img) in zip(np.atleast_1d(axes), rows):
        ax.imshow(img, cmap="gray", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_ylabel(title, fontsize=6, rotation=0, ha="right", va="center")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(" ; ".join(str(x) for x in progs[0]), fontsize=6)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    # GIF panel: the fixed example (row 0) plus fresh random ones, one row each
    _, inp_r, tgt_r, _, _ = sample_batch(args.gif_rows - 1, n_instr, rng=random.Random(step + 1),
                                         pad=BASE_PAD) if args.gif_rows > 1 else (None, None, None, None, None)
    inps = inp if inp_r is None else torch.cat([inp, inp_r])
    tgts = tgt_t[:, -1] if inp_r is None else torch.cat([tgt_t[:, -1], tgt_r[:, -1]])
    if mch0 is not None:
        _LAST_MCH = mch0 if inp_r is None else torch.cat([mch0, _LAST_MCH])
    rollout_gif(Path(path).with_suffix(".gif"), inps, tgts, n_st)


def rollout_gif_rgb(path, frames, final_target):
    """v8: (T, B, 4, H, W) -> RGB panel rows [input | rollout | target], blended
    with the instruction channels as they evolve."""
    from PIL import Image
    T, B, _, H, W = frames.shape
    tgt = final_target.cpu().numpy()
    imgs = []
    for t in range(T):
        rows = []
        for b in range(B):
            f0 = TileLayoutV8.to_rgb(frames[0, b, 0], frames[0, b, 1:])
            fr = TileLayoutV8.to_rgb(frames[t, b, 0], frames[t, b, 1:])
            ft = TileLayoutV8.to_rgb(tgt[b], frames[0, b, 1:])
            sep = np.full((H, 3, 3), 40, np.uint8)
            rows.append(np.concatenate([f0, sep, fr, sep, ft], 1))
        hsep = np.full((3, rows[0].shape[1], 3), 40, np.uint8)
        panel = rows[0]
        for r in rows[1:]:
            panel = np.concatenate([panel, hsep, r], 0)
        im = Image.fromarray(panel); im = im.resize((im.width * 4, im.height * 4), Image.NEAREST)
        imgs.append(im)
    imgs[0].save(str(path), save_all=True, append_images=imgs[1:], duration=70, loop=0)


@torch.no_grad()
def rollout_gif(path, inp, final_target, n_stages, k=K):
    """Frame-by-frame rollout, one row per example: input | rollout | target."""
    x = implant(inp)
    grab = (lambda x: x[:, [0] + MASK_CH].cpu().numpy()) if MASK_CH else (lambda x: x[:, 0].cpu().numpy())
    frames = [grab(x)]
    for _ in range(sum(stage_k(i) for i in range(n_stages))):
        x = nca(x, 1)[:, -1]
        frames.append(grab(x))
    frames = np.stack(frames)[:: args.gif_stride]
    if MASK_CH:
        return rollout_gif_rgb(path, frames, final_target)
    T, B, H, W = frames.shape
    vsep = np.full((T, B, H, 3), -1.0, dtype=np.float32)
    first = np.broadcast_to(frames[:1], frames.shape).copy()
    tgt = np.broadcast_to(final_target.cpu().numpy()[None], frames.shape).copy()
    rows = np.concatenate([first, vsep, frames, vsep, tgt], axis=3)   # (T, B, H, 3W+6)
    hsep = np.full((T, 3, rows.shape[3]), -1.0, dtype=np.float32)
    panel = rows[:, 0]
    for b in range(1, B):
        panel = np.concatenate([panel, hsep, rows[:, b]], axis=1)
    write_gif(path, panel, fps=15, cmap="gray", zoom=4)


# ── train ───────────────────────────────────────────────────────────────────
best = -1.0
log = (run_dir / "metrics.jsonl").open("a")
tlog = (run_dir / "train.jsonl").open("a")
bar = tqdm(range(args.steps), dynamic_ncols=True)
for step in bar:
    n_instr = rng.choice(INSTR_RANGE)
    layout, inp, tgt, _, _ = sample_batch(args.batch_size, n_instr)
    opt.zero_grad(set_to_none=True)
    loss = train_stages(inp, tgt, layout, n_instr)
    gnorm = torch.nn.utils.clip_grad_norm_(nca.parameters(), args.grad_clip or float("inf")).item()
    opt.step()

    if step % args.log_every == 0:
        tlog.write(json.dumps({"step": step, "loss": loss, "grad_norm": gnorm,
                               "n_instr": n_instr}) + "\n")
        if step % (args.log_every * 20) == 0:
            tlog.flush()

    if step % args.eval_every == 0:
        rec = {"step": step, "loss": loss, "grad_norm": gnorm, "K": K}
        for n in EVAL_INSTR:
            bits, exact = evaluate(n, args.eval_batch)
            rec[f"bits_L{n}"] = [round(float(v), 4) for v in bits]
            rec[f"exact_L{n}"] = [round(float(v), 4) for v in exact]
        score = float(np.mean(rec[f"exact_L{INSTR_RANGE[-1]}"]))
        rec["score"] = score
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if score > best:
            best = score
            torch.save({"nca": nca.state_dict(), "step": step, "score": score,
                        "config": vars(args)}, run_dir / "nca_best.pt")
        bar.set_description(
            f"loss {loss:.4f} gn {gnorm:.2f} "
            f"bits {rec[f'bits_L{INSTR_RANGE[-1]}']} exact {score:.3f}")

    if args.ckpt_every and step % args.ckpt_every == 0:
        torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)},
                   run_dir / f"nca_{step:07d}.pt")
    if args.snapshot_every and step % args.snapshot_every == 0:
        snapshot(run_dir / f"snapshot_{step:07d}.png", step=step)
        torch.save({"nca": nca.state_dict(), "step": step, "config": vars(args)},
                   run_dir / "nca_latest.pt")     # so a rollout of the CURRENT rule is always available

torch.save({"nca": nca.state_dict(), "step": args.steps, "config": vars(args)},
           run_dir / "nca_last.pt")
log.close()
tlog.close()
print(f"\ndone: {run_dir}")
