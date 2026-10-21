#!/usr/bin/env python3
"""v11: one rule, one run, the whole cycle in one graph.

The picture starts at t=0 (tape parked, nothing selected) and the rule runs the
entire program:  ADVANCE -> FETCH -> EXEC -> STORE -> ADVANCE -> ...
No detaching between phases, no teacher forcing, no harness writing state --
each phase just has its own supervision target, and the gradient flows back
through all of them.

Each phase is supervised on the TAIL of its window, so the target has to be a
fixed point the rule rests in, not a frame it passes through (the earlier runs
lost accuracy as K grew, which is exactly that failure).
"""
import argparse, json, random, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from ncpu.nca import NeuralCA
from ncpu.program import Instr, TileLayoutV11

ap = argparse.ArgumentParser()
ap.add_argument("--n-instr", type=int, default=4, help="tape capacity (fixes the grid)")
ap.add_argument("--cycle-start", type=int, default=1, help="program length to start the curriculum at")
ap.add_argument("--promote-at", type=float, default=0.9, help="accuracy gate to lengthen the program")
ap.add_argument("--op-weights", default="0.8,0.1,0.1",
                help="initial SUB,SUBLEQ,NOP sampling weights; uniform sampling makes ~50%% of "
                     "samples need no write at all, which rewards doing nothing")
ap.add_argument("--op-anneal", type=int, default=150000, help="steps to anneal op weights to uniform")
ap.add_argument("--reach-start", type=int, default=2,
                help="how many registers (nearest the slots outward) are addressable. Widening the "
                     "reach keeps every easier case in distribution; a discrete rung jump does not, "
                     "and wiped a solved FETCH in 5k steps.")
ap.add_argument("--addr-gate", type=float, default=0.85, help="FETCH accuracy needed to climb a rung")
ap.add_argument("--n-regs", type=int, default=4)
ap.add_argument("--word-bits", type=int, default=4)
ap.add_argument("--r", type=int, default=1, help="bit square radius (1 = 3x3 cell)")
ap.add_argument("--pitch", type=int, default=4)
ap.add_argument("--pad", type=int, default=3)
ap.add_argument("--sel-amp", type=float, default=1.0,
                help="amplitude of the painted selectors. At 1.0 they are as loud as the data they "
                     "gate, and the rule settles on copying a fixed register instead of reading them.")
ap.add_argument("--expand-selectors", action="store_true",
                help="the instruction at the line spans the register columns it selects")
ap.add_argument("--steps", type=int, default=1000000)
ap.add_argument("--batch-size", type=int, default=8)
ap.add_argument("--channels", type=int, default=16)
ap.add_argument("--hidden", default="256",
                help="rule width; comma-separated for depth (e.g. 128,128). Gating one channel's "
                     "flow on another is multiplicative, which one ReLU layer approximates badly.")
ap.add_argument("--fire-rate", type=float, default=0.5)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--grad-clip", type=float, default=0.0, help="0 disables (clipping every step random-walks)")
ap.add_argument("--k-scale", type=float, default=1.0)
ap.add_argument("--tail-frac", type=float, default=0.22)
ap.add_argument("--ckpt-chunk", type=int, default=8)
ap.add_argument("--active-weight", type=float, default=4.0)
ap.add_argument("--phase-weights", default="1,3,1,1",
                help="loss weight per phase ADVANCE,FETCH,EXEC,STORE. Three of the four phases are "
                     "predominantly hold-still targets and outvote the one phase that must move "
                     "data 3-to-1, which is why transport reaches 99% fill and then erodes.")
ap.add_argument("--commit-weight", type=float, default=0.5,
                help="penalty on |pred|<1 at cells whose target is +-1. Under L1 abstaining costs "
                     "1.0 and a random wrong answer also costs ~1.0, so abstaining is never worse; "
                     "this breaks that tie.")
ap.add_argument("--bg-weight", type=float, default=0.05,
                help="weight on background pixels; they are ~95% of the grid")
ap.add_argument("--chan-weights", default="4,1,1,1,1",
                help="relative loss weight of data,selA,selB,dst,op AFTER per-channel "
                     "normalisation. Flat weights give the 4 instruction channels 80% of the "
                     "loss and the data channel -- where the computation is -- only 20%.")
ap.add_argument("--tape-weight", type=float, default=1.0,
                help="extra weight on painted instruction cells, on top of per-channel normalisation")
ap.add_argument("--state-bound", type=float, default=4.0)
ap.add_argument("--hard-bound", action="store_true", help="clamp instead of tanh (kills gradient once saturated)")
ap.add_argument("--overflow-weight", type=float, default=1.0,
                help="penalise hidden-channel magnitude beyond 1; 0 disables")
ap.add_argument("--eval-batch", type=int, default=16)
ap.add_argument("--eval-every", type=int, default=2000)
ap.add_argument("--ckpt-every", type=int, default=2000)
ap.add_argument("--log-every", type=int, default=100)
ap.add_argument("--init-from", default="")
ap.add_argument("--deepen", action="store_true",
                help="insert an identity-initialised hidden layer into --init-from: function-preserving")
ap.add_argument("--widen", action="store_true",
                help="net2net-widen --init-from into --hidden: old units keep their weights, new "
                     "ones get 0.01*randn in and ZERO out, so the function is preserved exactly")
ap.add_argument("--tag", default="")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPS = ("SUB", "SUBLEQ", "NOP")
NVIS = 5                                              # data + selA/selB/dst/op
CW = np.array([float(v) for v in args.chan_weights.split(",")], dtype=np.float32)
assert len(CW) == NVIS, "--chan-weights needs one value per visible channel"
CW = CW / CW.sum()
PW = {p: float(v) for p, v in zip(("ADVANCE", "FETCH", "EXEC", "STORE"),
                                  args.phase_weights.split(","))}

L = TileLayoutV11(n_instr=args.n_instr, n_regs=args.n_regs, word_bits=args.word_bits,
                  r=args.r, pitch=args.pitch, pad=args.pad, expand=args.expand_selectors,
                  sel_amp=args.sel_amp)
K = {p: max(8, int(L.steps_for(p) * args.k_scale)) for p in L.PHASES}
TAIL = {p: max(4, int(K[p] * args.tail_frac)) for p in L.PHASES}
CYCLE = 4 * sum(K.values()) // 4                       # steps per instruction

run = f"V11_{time.strftime('%Y%m%d_%H%M%S')}_{args.tag or 'cycle'}_s{args.seed}"
out = Path("runs") / run; out.mkdir(parents=True, exist_ok=True)
(out / "config.json").write_text(json.dumps({**vars(args), "K": K, "tail": TAIL,
                                             "H": L.H, "W": L.W}, indent=2))
print(f"run {run}: grid {L.H}x{L.W}, tape holds {args.n_instr}")
for p in L.PHASES:
    print(f"  {p:8s} travel {L.travel(p):3d}px  K={K[p]:4d}  tail={TAIL[p]}")
print(f"  {sum(K.values())} CA steps per instruction; curriculum starts at {args.cycle_start}")


W0 = np.array([float(v) for v in args.op_weights.split(",")], dtype=np.float64)
W0 = W0 / W0.sum()
WUNI = np.full(len(OPS), 1.0 / len(OPS))


def op_weights(step):
    f = min(1.0, step / max(1, args.op_anneal))
    return W0 * (1 - f) + WUNI * f


def pick_regs(reach, distinct=False):
    """Registers an instruction may address: the ``reach`` nearest the slots.

    reach=2 -> {r2, r3}, reach=4 -> every register (free addressing).  Widening
    by one keeps everything already learned inside the sampling distribution,
    so the rule extends its transport instead of relearning it.
    """
    lo = max(0, args.n_regs - reach)
    pick = lambda: random.randrange(lo, args.n_regs)
    a, b = pick(), pick()
    if distinct and args.n_regs - lo > 1:
        while b == a:                       # a == b makes A-B trivially 0, which
            b = pick()                      # inflates exec_R without any arithmetic
    return a, b, pick()


def sample(cur, w=None, reach=4, distinct=False):
    w = WUNI if w is None else w
    prog = []
    for _ in range(cur):
        a, b, dst = pick_regs(reach, distinct)
        prog.append(Instr(OPS[int(np.searchsorted(np.cumsum(w), random.random()))], a, b, dst))
    regs = [random.randrange(1 << args.word_bits) for _ in range(args.n_regs)]
    traj = L.cycle(prog, regs)
    d, m = L.render(prog, traj[0][0])
    x = np.zeros((args.channels, L.H, L.W), dtype=np.float32)
    x[0] = d; x[1:5] = m
    tg, wg = [], []
    for st, ph, nx in traj:
        td, tm = L.render(prog, nx)
        t_all = np.concatenate([td[None], tm], 0)
        tg.append(t_all)
        # One weight map PER CHANNEL.  The data channel paints ~10x the pixels the
        # four instruction channels do; with a single shared map it drowns them out
        # and erasing the whole program costs <1% of the loss, so the rule does.
        wmap = np.empty((NVIS, L.H, L.W), dtype=np.float32)
        wmap[0] = (args.bg_weight + L.data_mask()
                   + args.active_weight * L.phase_mask(prog, st, ph))
        for c in range(1, NVIS):
            painted = (np.abs(t_all[c]) > 0.5).astype(np.float32)
            wmap[c] = args.bg_weight + args.tape_weight * painted
        wg.append(wmap)
    return x, np.stack(tg), np.stack(wg), [ph for _, ph, _ in traj], (prog, regs, traj)


def batch(n, cur, opw=None, reach=4, distinct=False):
    xs, ts, ws, metas = [], [], [], []
    for _ in range(n):
        x, t, w, phases, meta = sample(cur, opw, reach, distinct)
        xs.append(x); ts.append(t); ws.append(w); metas.append(meta)
    return (torch.from_numpy(np.stack(xs)).to(DEVICE),
            torch.from_numpy(np.stack(ts)).to(DEVICE),
            torch.from_numpy(np.stack(ws)).to(DEVICE), phases, metas)


HID = [int(v) for v in str(args.hidden).split(",")]
HID = HID[0] if len(HID) == 1 else HID
nca = NeuralCA(channels=args.channels, hidden_channels=HID, fire_rate=args.fire_rate,
               alive_threshold=0, zero_initialization=False, kernel_size=3,
               padding_type="zeros", state_bound=args.state_bound,
               soft_bound=not args.hard_bound).to(DEVICE)
def deepen_state(old, new, seed=0):
    """Insert a hidden layer initialised to the identity.

    The new layer sits directly after a ReLU, so its input is non-negative and
    an identity weight passes it through unchanged -- the following ReLU is then
    a no-op. The deepened rule computes exactly what the shallow one did.
    """
    out = {}
    ok = sorted(k for k in old if k.startswith("rule.rule."))
    nk = sorted(k for k in new if k.startswith("rule.rule."))
    # map the shallow layers onto the first and last conv of the deep stack
    first_w, first_b = "rule.rule.0.weight", "rule.rule.0.bias"
    last_old = ok[-1]
    last_new = nk[-1]
    for k, v in new.items():
        if k == first_w or k == first_b:
            out[k] = old[k]
        elif k == last_new:
            out[k] = old[last_old]
        elif k.endswith(".weight") and v.shape[0] == v.shape[1]:
            t = torch.zeros_like(v)
            t[:, :, 0, 0] = torch.eye(v.shape[0])          # identity 1x1 conv
            out[k] = t
            print(f"  identity-init {k}: {tuple(v.shape)}")
        elif k.endswith(".bias"):
            out[k] = torch.zeros_like(v)
        else:
            out[k] = old.get(k, v)
    return out


def widen_state(old, new, seed=0):
    """net2net: copy the old hidden units, give the new ones tiny random input
    weights and zero output weights -- the widened rule computes exactly what
    the old one did, so training continues rather than restarting."""
    g = torch.Generator().manual_seed(seed); out = {}
    for k, v in new.items():
        o = old[k]
        if o.shape == v.shape:
            out[k] = o; continue
        t = v.clone()
        if k == "rule.rule.0.weight":
            t[: o.shape[0]] = o
            t[o.shape[0]:] = torch.randn(t.shape[0] - o.shape[0], *o.shape[1:], generator=g) * 0.01
        elif k == "rule.rule.0.bias":
            t[: o.shape[0]] = o; t[o.shape[0]:] = 0.0
        elif k == "rule.rule.2.weight":
            t[:, : o.shape[1]] = o; t[:, o.shape[1]:] = 0.0
        else:
            raise ValueError(k)
        out[k] = t; print(f"  widened {k}: {tuple(o.shape)} -> {tuple(v.shape)}")
    return out


if args.init_from:
    sd = torch.load(args.init_from, map_location=DEVICE)
    old = sd["model"] if "model" in sd else sd
    if args.deepen:
        nca.load_state_dict(deepen_state(old, nca.state_dict()))
    elif args.widen:
        nca.load_state_dict(widen_state(old, nca.state_dict()))
    else:
        nca.load_state_dict(old)
    print(f"  warm start from {args.init_from}{' (widened)' if args.widen else ''}")
print(f"  params {sum(p.numel() for p in nca.parameters()):,}", flush=True)
opt = torch.optim.Adam(nca.parameters(), lr=args.lr)


def _run(z, n):
    return nca(z, n)[:, -1]


def _frames(z, n):
    return nca(z, n)[:, 1:]


def phase_rollout(x, k, tail):
    """K steps; returns the last ``tail`` frames.  Everything is checkpointed,
    so the graph spans the whole program without holding its activations."""
    done, head = 0, k - tail
    while done < head:
        n = min(args.ckpt_chunk, head - done)
        x = checkpoint(_run, x, n, use_reentrant=False)
        done += n
    frames, done = [], 0
    while done < tail:
        n = min(args.ckpt_chunk, tail - done)
        f = checkpoint(_frames, x, n, use_reentrant=False)
        frames.append(f); x = f[:, -1]; done += n
    return torch.cat(frames, 1)


@torch.no_grad()
def free_run(x, phases):
    for ph in phases:
        done = 0
        while done < K[ph]:
            n = min(32, K[ph] - done); x = nca(x, n)[:, -1]; done += n
    return x


@torch.no_grad()
def evaluate(n, cur, reach):
    """Per-phase accuracy.  The headline number is dominated by NOP and by
    SUBLEQs that correctly write nothing, so it hides whether the machine
    computes at all -- these do not."""
    nca.eval()
    h = {k: 0 for k in ("fetch_A", "fetch_B", "fetch_op", "exec_R", "store", "final")}
    tot = 0
    for _ in range(0, n, args.eval_batch):
        b = min(args.eval_batch, n)
        x, t, w, phases, metas = batch(b, cur, np.array([1.0, 0.0, 0.0]), reach, distinct=True)   # SUB only: must write
        for i, ph in enumerate(phases):
            done = 0
            while done < K[ph]:
                m = min(32, K[ph] - done); x = nca(x, m)[:, -1]; done += m
            for j in range(b):
                nx = metas[j][2][i][2]
                regs, sl = L.decode(x[j, 0].cpu().numpy())
                if ph == "FETCH":
                    h["fetch_A"] += sl["A"] == nx.A; h["fetch_B"] += sl["B"] == nx.B
                    h["fetch_op"] += sl["op"] == nx.op
                elif ph == "EXEC":
                    h["exec_R"] += sl["R"] == nx.R
                elif ph == "STORE":
                    h["store"] += regs == tuple(nx.regs)
        for j in range(b):
            want = metas[j][2][-1][2]
            regs, _ = L.decode(x[j, 0].cpu().numpy())
            h["final"] += regs == tuple(want.regs)
        tot += b
    nca.train()
    return {k: round(v / max(1, tot), 3) for k, v in h.items()}


cur = args.cycle_start
reach = args.reach_start
log = (out / "metrics.jsonl").open("a")
t0 = time.time()
for step in range(args.steps + 1):
    x, t, w, phases, _ = batch(args.batch_size, cur, op_weights(step), reach)
    loss, ov, per = 0.0, 0.0, {}
    for i, ph in enumerate(phases):
        seq = phase_rollout(x, K[ph], TAIL[ph])
        wi = w[:, i, None]                                   # (B, 1, NVIS, H, W)
        err = (seq[:, :, :NVIS] - t[:, i, None]).abs() * wi
        # normalise each channel by its own weight mass, then average the channels,
        # so a sparse instruction channel counts as much as the dense data channel
        per_chan = err.sum((0, 1, 3, 4)) / (wi.sum((0, 1, 3, 4)) * seq.shape[1])
        d = (per_chan * torch.from_numpy(CW).to(DEVICE)).sum()
        if args.commit_weight:
            # Weight the commitment penalty the same way as the main loss. A flat mean
            # dilutes it across ~288 definite cells, so abstaining on just the 8 slot
            # cells the phase must fill costs ~3% -- and the rule does exactly that.
            definite = (t[:, i].abs() > 0.5).float()[:, None] * wi
            short = (1.0 - seq[:, :, :NVIS].abs()).clamp(min=0) * definite
            d = d + args.commit_weight * short.sum() / (definite.sum() * seq.shape[1] + 1e-6)
        loss = loss + PW[ph] * d
        per[ph] = per.get(ph, 0.0) + float(d)
        if args.overflow_weight:
            ov = ov + (seq[:, :, NVIS:].abs() - 1.0).clamp(min=0).pow(2).mean()
        x = seq[:, -1]
    loss = loss / sum(PW[p] for p in phases)
    if args.overflow_weight:
        ov = ov / len(phases)
        loss = loss + args.overflow_weight * ov
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if args.grad_clip:
        torch.nn.utils.clip_grad_norm_(nca.parameters(), args.grad_clip)
    opt.step()

    if step % args.log_every == 0:
        rec = {"step": step, "cur": cur, "loss": round(float(loss), 4),
               "ov": round(float(ov), 4), "wsub": round(float(op_weights(step)[0]), 3),
               **{f"L_{k_}": round(v, 3) for k_, v in per.items()},
               "t": round(time.time() - t0, 1)}
        if step % args.eval_every == 0 and step:
            m = evaluate(args.eval_batch * 2, cur, reach)
            rec.update(m); rec["reach"] = reach
            fetch = min(m["fetch_A"], m["fetch_B"])
            if fetch >= args.addr_gate and reach < args.n_regs:
                reach += 1; rec["widened"] = reach
            elif reach == args.n_regs and m["final"] >= args.promote_at and cur < args.n_instr:
                cur += 1; rec["promoted"] = cur
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(" ".join(f"{k_}={v}" for k_, v in rec.items()), flush=True)
    if step % args.ckpt_every == 0:
        blob = {"model": nca.state_dict(), "step": step, "cur": cur, "reach": reach}
        torch.save(blob, out / "nca_latest.pt")
        torch.save(blob, out / f"nca_{step:07d}.pt")
