#!/usr/bin/env python3
"""Track M: instrumentation for the NCA depth-instability.

The claim under test
--------------------
Training the 3SAT verifier reaches verdict 1.000 and then destroys it, with
gradient norms rising roughly exponentially through training (mean 66 at 7k,
12,816 at 17k, against a clip of 1.0). The standing hypothesis was that
``NeuralCA.forward`` clips *state* to +-10 but bounds nothing about the
**per-step delta**, so a rule that grows its own activations hands BPTT a
diverging Jacobian.

This file measures that instead of assuming it. Everything here is read-only
with respect to the model: checkpoints are loaded, probed, and thrown away. No
training happens in this script except in ``gradsurface``, which takes a single
gradient (no optimiser step).

Subcommands
-----------
``sweep``      per-checkpoint scan of a finished run. For each archived
               checkpoint: per-step delta norm, state saturation against the
               +-10 clip, verdict/exact accuracy at the trained rollout AND at
               multiples of it (does the answer PERSIST?), settle drift, and
               the product of rule-layer spectral norms (an upper bound on the
               one-step Jacobian gain).

``jacobian``   power iteration on the true rollout Jacobian via forward-mode
               jvp, giving the measured per-step gain lambda. lambda > 1 is a
               diverging BPTT chain; lambda < 1 is a contraction. The stochastic
               fire mask is frozen by reseeding, so the map being differentiated
               is a fixed deterministic map.

``gradepth``   gradient norm as a function of BPTT depth k, holding the loss
               and the state fixed (truncated BPTT: T-k steps under no_grad,
               k steps with grad). If the norm is exponential in k, the depth-
               timescale coupling is a gradient-path-length effect and
               truncation is a real lever; if it is flat in k, it is not.

``persist``    the pool/persistence question in isolation: take a checkpoint,
               run 8x the trained rollout, and report the verdict accuracy at
               each multiple. "Arrives and stays" vs "arrives and leaves".

Everything writes its plot to the PROJECT ROOT, per house rules.

Usage:
    uv run python scripts/diagnose_stability.py sweep  --run runs/20260821_192943_SATVERIFY_n6_m8_r1_s0
    uv run python scripts/diagnose_stability.py jacobian --run <run> --ckpt nca_best.pt
    uv run python scripts/diagnose_stability.py gradepth --run <run>
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ncpu.nca import NeuralCA
from ncpu.sat import SATLayout

# ── args ─────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("mode", choices=["sweep", "jacobian", "gradepth", "persist",
                                     "levers", "window"])
parser.add_argument("--run", required=True, help="run directory")
parser.add_argument("--ckpt", default=None, help="single checkpoint filename (default: all archived)")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch", type=int, default=16)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--persist-mult", type=float, default=8.0,
                    help="run this many times the trained rollout when testing persistence")
parser.add_argument("--jac-iters", type=int, default=12, help="power iterations")
parser.add_argument("--jac-steps", type=int, default=None, help="rollout length for the Jacobian")
parser.add_argument("--fire-rate", type=float, default=None,
                    help="override fire_rate for the probe (1.0 = deterministic map)")
parser.add_argument("--tag", default="", help="suffix for output filenames")
parser.add_argument("--metric", default="verdict",
                    help="window mode: which eval.jsonl field to test")
parser.add_argument("--target", type=float, default=1.0,
                    help="window mode: the value that counts as solved")
# ── levers ───────────────────────────────────────────────────────────────────
parser.add_argument("--init-ckpt", default="nca_best.pt",
                    help="levers: checkpoint to warm-start every arm from, or 'none' to "
                         "train from a fresh init. Warm-starting from nca_best.pt is the "
                         "cheap form of the success criterion (does it STAY); 'none' is the "
                         "full confirmation (does it get there AND stay)")
parser.add_argument("--save-run", action="store_true",
                    help="levers: write a real run directory under --runs-dir, prefixed "
                         "STAB_, with log.jsonl / eval.jsonl / nca_best.pt / curves.png. "
                         "Use for anything longer than a probe")
parser.add_argument("--runs-dir", default="runs")
parser.add_argument("--lever-steps", type=int, default=2000)
parser.add_argument("--lever-batch", type=int, default=8)
parser.add_argument("--lever-eval-every", type=int, default=100)
parser.add_argument("--arm", default="control",
                    help="levers: control | lrdecay | wd | pool | tbptt | dbound | combo")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr-final", type=float, default=None)
parser.add_argument("--wd", type=float, default=0.0, help="AdamW decoupled weight decay")
parser.add_argument("--grad-clip", type=float, default=1.0)
parser.add_argument("--tbptt", type=int, default=0, help="backprop through only the last k steps")
parser.add_argument("--delta-bound", type=float, default=0.0,
                    help="soft-bound the per-step delta to +-b with b*tanh(d/b). 0 = off")
parser.add_argument("--pool-size", type=int, default=0,
                    help="levers: experience-replay pool size. 0 = off (the arm name\n                         is only a label; the pool is enabled by this flag alone)")
parser.add_argument("--pool-fresh", type=float, default=0.5)
parser.add_argument("--wd0", type=float, default=None, metavar="W",
                    help="weight decay BEFORE the trigger arms (default: 0 under a trigger, "
                         "--wd without one). Set equal to --wd to keep weight decay on from "
                         "step 0 and let the trigger gate the lr decay ALONE -- which is the "
                         "recommended recipe: decoupled weight decay does not shrink the "
                         "STEP, so it does not block the plateau crossing, while an lr decay "
                         "does exactly that")
parser.add_argument("--stab-trigger", type=float, default=0.0, metavar="V",
                    help="SCHEDULED stabilisation. While the held-out verdict has never "
                         "reached V, the run is left alone: constant lr, no weight decay. "
                         "The first eval at or above V ARMS the treatment -- weight decay "
                         "switches to --wd and the lr begins its decay to --lr-final over "
                         "the steps that remain. 0 = unscheduled (whatever --wd / "
                         "--lr-final say, from step 0). See docs/stability_findings.md: the "
                         "same constant-speed walk that crosses the plateau is the one that "
                         "wanders off it, so a GLOBAL damping cannot separate the two jobs")
parser.add_argument("--breakthrough", type=float, default=0.95, metavar="V",
                    help="held-out verdict counted as 'broken through', for reporting the "
                         "breakthrough STEP. A treatment that retains perfectly but doubles "
                         "time-to-breakthrough is expensive on a task with min_rollout 608")
args = parser.parse_args()

# ── window ───────────────────────────────────────────────────────────────────
# Re-emit the success-criterion metric from a finished run's eval.jsonl, without
# retraining anything. This exists because the metric shipped WRONG once: a run
# that held 11,749 steps was reported as 0 and nearly recorded as a miss. Any
# number this important needs a path to recompute it from the raw log.
#
# Handled here, before the SATLayout/config machinery below, so it works on any
# run directory regardless of which trainer wrote it.
if args.mode == "window":
    import sys as _sys

    def _perfect(rows, key, target):
        longest, best, start, wins = 0, (None, None), None, []
        for h in rows:
            if h.get(key, -1.0) >= target - 1e-9:
                if start is None:
                    start = h["step"]
                    wins.append([h["step"], h["step"]])
                else:
                    wins[-1][1] = h["step"]
                if h["step"] - start > longest:
                    longest, best = h["step"] - start, (start, h["step"])
            else:
                start = None
        return longest, best[0], best[1], [tuple(w) for w in wins]

    _p = Path(args.run)
    if not _p.is_absolute():
        _p = ROOT / _p
    if _p.is_dir():
        _p = _p / "eval.jsonl"
    _rows = [json.loads(l) for l in open(_p)]
    _key = args.metric if any(args.metric in r for r in _rows) else (
        "eval_solved" if any("eval_solved" in r for r in _rows) else
        "eval_verdict_acc")
    _n, _a, _b, _w = _perfect(_rows, _key, args.target)
    _hits = sum(1 for r in _rows if r.get(_key, -1.0) >= args.target - 1e-9)
    print(f"{_p}")
    print(f"  metric {_key!r}  target {args.target:g}  evals {len(_rows)} "
          f"(steps {_rows[0]['step']}..{_rows[-1]['step']})")
    print(f"  evaluations at target: {_hits}")
    print(f"  all windows: {_w}")
    print(f"  longest unbroken window: {_n} steps"
          + (f" (steps {_a}..{_b})" if _a is not None else "")
          + f"   [criterion: >= 10000]" + ("   PASS" if _n >= 10000 else "   MISS"))
    if _hits >= 2 and _n == 0:
        print("  !! INCONSISTENT: multiple evals at target but a zero-length window. "
              "The metric is broken, not the run.")
    _sys.exit(0)


DEVICE = args.device
RUN = Path(args.run)
if not RUN.is_absolute():
    RUN = ROOT / RUN
CFG = json.loads((RUN / "config.json").read_text())
LAY = CFG["layout"]
OPT = CFG["optim"]

layout = SATLayout(n_vars=LAY["n"], m_clauses=LAY["m"], k=LAY["k"],
                   r=LAY["r"], gap=LAY["gap"], pad=LAY["pad"])
DEPTH = layout.min_rollout()
ROLL_MIN, ROLL_MAX = OPT["rollout"]
EVAL_STEPS = (ROLL_MIN + ROLL_MAX) // 2
BIT_MASKS = layout.output_bit_masks("verify").to(DEVICE)
M = layout.m
CHANNELS = CFG["nca"]["channels"]
INPUT_DIMS = (0, 1)

print(f"run {RUN.name}")
print(f"  {layout.describe()}")
print(f"  min_rollout {DEPTH}  trained rollout {ROLL_MIN}..{ROLL_MAX}  eval at {EVAL_STEPS} "
      f"({EVAL_STEPS / DEPTH:.1f}x min_rollout)")


# ── shared machinery ─────────────────────────────────────────────────────────
def load_nca(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    nca = NeuralCA(**CFG["nca"]).to(DEVICE)
    nca.load_state_dict(ck["nca"])
    nca.eval()
    if args.fire_rate is not None:
        nca.stochastic_update.fire_rate = args.fire_rate
    return nca, ck.get("step", -1)


def checkpoints():
    d = RUN / "checkpoints"
    if args.ckpt:
        return [d / args.ckpt]
    named = sorted(p for p in d.glob("nca_[0-9]*.pt"))
    return named


def implant(inp):
    state = torch.zeros(inp.shape[0], CHANNELS, layout.H, layout.W, device=DEVICE)
    for d in INPUT_DIMS:
        state[:, d] = inp
    return state


def eval_batch():
    """The SAME held-out instances the trainer evaluated on: EVAL_RNG = seed+10000."""
    rng = random.Random(args.seed + 10_000)
    inp, tgt, bits, meta = layout.sample_batch(
        args.batch, mode="verify", device=DEVICE, rng=rng,
        sat_frac=OPT.get("sat_frac", 0.5), near_miss_frac=OPT.get("near_miss_frac", 0.5))
    return inp, tgt, bits


def acc(frame, bits):
    got = SATLayout.decode_bits(frame, BIT_MASKS)
    correct = got == bits
    return (correct[:, M].float().mean().item(),        # verdict
            correct.all(dim=1).float().mean().item(),   # exact
            correct[:, :M].float().mean().item())       # flags


def rule_spectral_product(nca):
    """Product over rule layers of the 1x1-conv weight spectral norms.

    A 1x1 conv is a per-pixel linear map, so its spectral norm is the matrix
    2-norm of the (out, in) weight. ReLU is 1-Lipschitz, and the fixed Sobel
    perception contributes a constant, so this product is an UPPER BOUND on the
    per-step delta's Lipschitz constant with respect to the local state. It is
    a bound, not the gain -- ``jacobian`` measures the gain.
    """
    prod = 1.0
    norms = []
    for mod in nca.rule.rule:
        w = getattr(mod, "weight", None)
        if w is None or w.dim() != 4:
            continue
        s = torch.linalg.matrix_norm(w.reshape(w.shape[0], -1), ord=2).item()
        norms.append(s)
        prod *= s
    return prod, norms



def perfect_window(hist, key="verdict", target=1.0):
    """Longest unbroken run of evaluations at ``target``, in STEPS.

    Returns ``(span, first_step, last_step, windows)``.

    Beware the sentinel that broke the first version of this: the running span
    is legitimately ZERO at the first evaluation of a window, so a `cur == 0`
    test for "no window is open" also fires on every window's opening
    evaluation, resets the start to the current step, and pins the answer at 0
    forever. It reported 0 for a run that held 11,749 steps. The open/closed
    state must be carried by something that cannot collide with a valid span --
    here ``start is None``.

    The span is measured between the first and last EVALUATION at target, so it
    understates the true window by up to one eval interval at each end. That is
    deliberate: it is the part of the window the run actually observed.
    """
    longest, best = 0, (None, None)
    start = None
    windows = []
    for h in hist:
        if h.get(key, -1.0) >= target - 1e-9:
            if start is None:
                start = h["step"]
                windows.append([h["step"], h["step"]])
            else:
                windows[-1][1] = h["step"]
            span = h["step"] - start
            if span > longest:
                longest, best = span, (start, h["step"])
        else:
            start = None
    return longest, best[0], best[1], [tuple(w) for w in windows]


def check_window_consistency(hist, longest, best_value, key="verdict", target=1.0):
    """Refuse to report a metric that contradicts the data it came from.

    ``longest_1000_window: 0`` sitting next to ``best: 1.0`` is not a result, it
    is a bug, and it shipped once. Returns a warning string or None.
    """
    n_at = sum(1 for h in hist if h.get(key, -1.0) >= target - 1e-9)
    if n_at >= 2 and longest == 0:
        return (f"INCONSISTENT: {n_at} evaluations reached {key} {target:g} but the longest "
                f"window computed to 0 steps. The window computation is wrong, not the run.")
    if best_value >= target - 1e-9 and n_at == 0:
        return (f"INCONSISTENT: best {key} is {best_value:.3f} but no evaluation is at "
                f"{target:g}.")
    return None


# ── sweep ────────────────────────────────────────────────────────────────────
@torch.no_grad()
def probe(nca, inp, bits, steps):
    """One rollout. Returns per-step delta stats and accuracy at each multiple."""
    torch.manual_seed(args.seed + 20_000)
    x = implant(inp)
    dnorm, xmax, sat = [], [], []
    marks = {}
    n_el = x[0].numel()
    for t in range(1, steps + 1):
        prev = x
        x = nca.forward(x, steps=1)[:, -1]
        d = (x - prev)
        dnorm.append((d.pow(2).sum(dim=(1, 2, 3)) / n_el).sqrt().mean().item())  # RMS per cell
        xmax.append(x.abs().max().item())
        sat.append((x.abs() >= 9.999).float().mean().item())
        if t % EVAL_STEPS == 0:
            marks[t] = acc(x[:, 0], bits)
    return np.array(dnorm), np.array(xmax), np.array(sat), marks, x


def cmd_sweep():
    inp, tgt, bits = eval_batch()
    steps = int(round(args.persist_mult * EVAL_STEPS))
    rows = []
    print(f"\nprobing {steps} steps ({args.persist_mult:g}x the trained rollout of {EVAL_STEPS})")
    hdr = (f"{'step':>7} {'vd@1x':>6} {'ex@1x':>6} {'vd@2x':>6} {'vd@4x':>6} {'vd@8x':>6} "
           f"{'d_rms_end':>10} {'d_rms_max':>10} {'|x|max':>7} {'sat%':>6} {'drift':>8} {'Wspec':>9}")
    print(hdr)
    for p in checkpoints():
        nca, tstep = load_nca(p)
        dn, xm, st, marks, final = probe(nca, inp, bits, steps)
        prod, norms = rule_spectral_product(nca)
        # settle drift over the last 8 frames, channel 0 (the relax.py criterion)
        drift = float(dn[-8:].max())
        mult = {}
        for k in (1, 2, 4, 8):
            t = k * EVAL_STEPS
            mult[k] = marks.get(t, (float("nan"),) * 3)
        rows.append(dict(step=tstep, name=p.name, vd=[mult[k][0] for k in (1, 2, 4, 8)],
                         ex=[mult[k][1] for k in (1, 2, 4, 8)],
                         d_end=float(dn[EVAL_STEPS - 1]), d_max=float(dn.max()),
                         d_tail=float(dn[-1]), xmax=float(xm.max()),
                         sat=float(st[EVAL_STEPS - 1]), drift=drift,
                         wspec=prod, wnorms=norms,
                         dcurve=dn[:steps].tolist()))
        print(f"{tstep:7d} {mult[1][0]:6.3f} {mult[1][1]:6.3f} {mult[2][0]:6.3f} "
              f"{mult[4][0]:6.3f} {mult[8][0]:6.3f} "
              f"{dn[EVAL_STEPS - 1]:10.4f} {dn.max():10.4f} {xm.max():7.2f} "
              f"{100 * st[EVAL_STEPS - 1]:6.2f} {drift:8.4f} {prod:9.2f}")
        del nca

    out = ROOT / f"stab_sweep_{RUN.name}{args.tag}.json"
    out.write_text(json.dumps(rows, indent=1))

    fig, axes = plt.subplots(1, 4, figsize=(22, 4))
    xs = [r["step"] for r in rows]
    a = axes[0]
    a.plot(xs, [r["vd"][0] for r in rows], "o-", label=f"verdict @{EVAL_STEPS} (1x)")
    a.plot(xs, [r["vd"][1] for r in rows], "s-", label="verdict @2x")
    a.plot(xs, [r["vd"][3] for r in rows], "^-", label="verdict @8x")
    a.plot(xs, [r["ex"][0] for r in rows], "o:", label="exact @1x", alpha=0.6)
    a.set_ylim(0, 1.02); a.set_title("does the answer PERSIST past the trained rollout?")
    a.set_xlabel("training step"); a.legend(fontsize=7)

    a = axes[1]
    a.plot(xs, [r["d_end"] for r in rows], "o-", label="per-step delta RMS at 1x")
    a.plot(xs, [r["d_max"] for r in rows], "s-", label="max over 8x rollout")
    a.plot(xs, [r["d_tail"] for r in rows], "^-", label="at 8x (settled?)")
    a.set_yscale("log"); a.set_title("per-step delta magnitude")
    a.set_xlabel("training step"); a.legend(fontsize=7)

    a = axes[2]
    a.plot(xs, [100 * r["sat"] for r in rows], "o-")
    a.set_title("% of cells pinned at the +-10 state clip"); a.set_xlabel("training step")

    a = axes[3]
    a.plot(xs, [r["wspec"] for r in rows], "o-", label="prod of rule spectral norms")
    a.set_yscale("log"); a.axhline(1.0, color="k", ls="--", lw=0.8)
    a.set_title("rule Lipschitz bound"); a.set_xlabel("training step"); a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(ROOT / f"stab_sweep_{RUN.name}{args.tag}.png", dpi=110)
    plt.close(fig)

    # delta-norm curves over the rollout, one line per checkpoint
    fig, a = plt.subplots(figsize=(9, 5))
    cm = plt.get_cmap("viridis")
    for i, r in enumerate(rows):
        a.plot(r["dcurve"], lw=0.9, color=cm(i / max(1, len(rows) - 1)), label=f"{r['step']}")
    a.axvline(EVAL_STEPS, color="k", ls="--", lw=0.8, label="trained rollout")
    a.set_yscale("log"); a.set_xlabel("rollout step t"); a.set_ylabel("per-step delta RMS")
    a.set_title(f"{RUN.name}: delta magnitude along the rollout")
    a.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(ROOT / f"stab_delta_{RUN.name}{args.tag}.png", dpi=110)
    plt.close(fig)
    print(f"\nwrote stab_sweep_{RUN.name}{args.tag}.png / .json and stab_delta_...png")


# ── jacobian ─────────────────────────────────────────────────────────────────
def cmd_jacobian():
    """Power-iterate the rollout Jacobian to get the per-step gain.

    The map differentiated is x_0 -> x_T with the fire mask FROZEN (the RNG is
    reseeded before each pass), so it is a genuine deterministic function and
    the power iteration is well posed. jvp is forward mode: cost is one extra
    forward pass per iteration and no BPTT memory.
    """
    inp, tgt, bits = eval_batch()
    T = args.jac_steps or EVAL_STEPS
    print(f"\npower iteration over a {T}-step rollout, {args.jac_iters} iters, "
          f"fire_rate {'frozen at ' + str(args.fire_rate) if args.fire_rate else 'frozen (reseeded)'}")
    print(f"{'step':>7} {'lambda_T':>12} {'per-step':>10} {'||J v||/||v||':>14}")
    rows = []
    for p in checkpoints():
        nca, tstep = load_nca(p)

        def f(x0):
            torch.manual_seed(args.seed + 20_000)   # freeze the fire masks
            x = x0
            for _ in range(T):
                x = nca.forward(x, steps=1)[:, -1]
            return x

        x0 = implant(inp)
        v = torch.randn_like(x0)
        v = v / v.norm()
        lam = float("nan")
        for it in range(args.jac_iters):
            _, jv = torch.func.jvp(f, (x0,), (v,))
            nrm = jv.norm().item()
            if not np.isfinite(nrm) or nrm == 0:
                break
            lam = nrm
            v = (jv / nrm).detach()
        per_step = lam ** (1.0 / T) if lam > 0 and np.isfinite(lam) else float("nan")
        rows.append(dict(step=tstep, lam=lam, per_step=per_step, T=T))
        print(f"{tstep:7d} {lam:12.4g} {per_step:10.4f} {lam:14.4g}")
        del nca

    (ROOT / f"stab_jacobian_{RUN.name}{args.tag}.json").write_text(json.dumps(rows, indent=1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    xs = [r["step"] for r in rows]
    a1.plot(xs, [r["lam"] for r in rows], "o-")
    a1.set_yscale("log"); a1.axhline(1.0, color="k", ls="--", lw=0.8)
    a1.set_title(f"rollout Jacobian spectral radius over {T} steps"); a1.set_xlabel("training step")
    a2.plot(xs, [r["per_step"] for r in rows], "o-")
    a2.axhline(1.0, color="k", ls="--", lw=0.8)
    a2.set_title("per-step gain lambda^(1/T)"); a2.set_xlabel("training step")
    fig.tight_layout(); fig.savefig(ROOT / f"stab_jacobian_{RUN.name}{args.tag}.png", dpi=110)
    plt.close(fig)
    print(f"wrote stab_jacobian_{RUN.name}{args.tag}.png")


# ── gradepth ─────────────────────────────────────────────────────────────────
def cmd_gradepth():
    """Gradient norm vs BPTT depth k at a FIXED loss and a fixed final state.

    T - k steps run under no_grad, the last k with grad, loss on the last frame.
    Every k sees the same trajectory and the same loss value; only the length of
    the gradient path changes. Exponential growth in k is the signature of a
    diverging BPTT chain, and it is exactly what truncated BPTT would cut.
    """
    inp, tgt, bits = eval_batch()
    T = EVAL_STEPS
    ks = [k for k in [1, 2, 4, 8, 16, 24, 32, 48, 64, 96, T] if k <= T]
    ks = sorted(set(ks))
    print(f"\ngradient norm vs BPTT depth k (T = {T}, loss on the final frame only)")
    print("  ".join(f"{k:>9}" for k in ["step"] + ks))
    rows = []
    for p in checkpoints():
        nca, tstep = load_nca(p)
        gs = []
        for k in ks:
            torch.manual_seed(args.seed + 20_000)
            with torch.no_grad():
                x = implant(inp)
                for _ in range(T - k):
                    x = nca.forward(x, steps=1)[:, -1]
            x = x.detach().requires_grad_(False)
            for pm in nca.parameters():
                pm.grad = None
            for _ in range(k):
                x = nca.forward(x, steps=1)[:, -1]
            loss = ((x[:, 0] - tgt) ** 2).mean()
            loss.backward()
            g = torch.sqrt(sum((pm.grad ** 2).sum() for pm in nca.parameters()
                               if pm.grad is not None)).item()
            gs.append(g)
        rows.append(dict(step=tstep, ks=ks, grads=gs))
        print(f"{tstep:9d}  " + "  ".join(f"{g:9.3g}" for g in gs))
        del nca

    (ROOT / f"stab_gradepth_{RUN.name}{args.tag}.json").write_text(json.dumps(rows, indent=1))
    fig, a = plt.subplots(figsize=(8, 5))
    cm = plt.get_cmap("viridis")
    for i, r in enumerate(rows):
        a.plot(r["ks"], r["grads"], "o-", color=cm(i / max(1, len(rows) - 1)),
               label=f"step {r['step']}", lw=1)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("BPTT depth k (steps carrying gradient)"); a.set_ylabel("||grad||")
    a.set_title(f"{RUN.name}: gradient norm vs gradient-path length")
    a.legend(fontsize=6, ncol=2)
    fig.tight_layout(); fig.savefig(ROOT / f"stab_gradepth_{RUN.name}{args.tag}.png", dpi=110)
    plt.close(fig)
    print(f"wrote stab_gradepth_{RUN.name}{args.tag}.png")


# ── persist ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def cmd_persist():
    inp, tgt, bits = eval_batch()
    steps = int(round(args.persist_mult * EVAL_STEPS))
    for p in checkpoints():
        nca, tstep = load_nca(p)
        torch.manual_seed(args.seed + 20_000)
        x = implant(inp)
        curve = []
        for t in range(1, steps + 1):
            x = nca.forward(x, steps=1)[:, -1]
            if t % 4 == 0:
                curve.append((t,) + acc(x[:, 0], bits))
        print(f"\n{p.name} (step {tstep}):")
        for t, vd, ex, fl in curve:
            if t % (EVAL_STEPS // 2) == 0 or t == steps:
                print(f"  t={t:5d} ({t / EVAL_STEPS:4.1f}x)  verdict {vd:.3f}  exact {ex:.3f}  flag {fl:.3f}")
        fig, a = plt.subplots(figsize=(9, 4))
        a.plot([c[0] for c in curve], [c[1] for c in curve], label="verdict")
        a.plot([c[0] for c in curve], [c[2] for c in curve], label="exact")
        a.axvline(EVAL_STEPS, color="k", ls="--", lw=0.8, label="trained rollout")
        a.set_ylim(0, 1.02); a.set_xlabel("rollout step"); a.legend(fontsize=8)
        a.set_title(f"{RUN.name} {p.name}: accuracy past the trained horizon")
        fig.tight_layout()
        fig.savefig(ROOT / f"stab_persist_{RUN.name}_{p.stem}{args.tag}.png", dpi=110)
        plt.close(fig)
        del nca



# ── levers ───────────────────────────────────────────────────────────────────
# A compact re-implementation of ``train_sat_verify.py``'s optimisation step,
# with the stability levers switchable. It exists so a lever can be measured
# WITHOUT editing the shared trainer or nca.py, and so the arms differ in
# exactly one thing. The loss, the weight masks and the RNG discipline are
# copied from that script verbatim so the control arm continues the original
# run's trajectory rather than a lookalike of it.
#
# Every arm WARM-STARTS from ``--init-ckpt`` (default nca_best.pt, the step at
# which held-out verdict hit 1.000). That is the honest short-budget form of
# the success criterion: the question is not "can it get there" -- we know it
# can -- but "does it STAY". A 2,000-step continuation from the peak answers
# that inside the per-probe GPU budget, where a from-scratch 25k run does not.

import torch.nn as nn


class BoundedDelta(nn.Module):
    """``b * tanh(delta / b)`` applied where the fire mask is applied.

    This is a PROTOTYPE of the change one might make to ``NeuralCA.forward``:
    the state is already clipped to +-10 there, but the per-step delta is not
    bounded at all. Wrapping ``nca.stochastic_update`` puts the bound at exactly
    the point the real change would sit -- the last thing done to ``delta``
    before ``x = x + delta`` -- so the prototype and the proposal are the same
    function, and nothing in ``nca.py`` has to be touched to measure it.
    """

    def __init__(self, inner, bound):
        super().__init__()
        self.inner = inner
        self.bound = float(bound)

    def forward(self, delta):
        d = self.inner(delta)
        return self.bound * torch.tanh(d / self.bound)


def build_loss_masks():
    """out_weight / verdict_weight / flag_neg_weight, exactly as the run used them."""
    bm = BIT_MASKS
    flag_mask = bm[:-1].sum(dim=0).clamp(0, 1)
    verdict_mask = bm[-1]
    return bm, flag_mask, verdict_mask


def cmd_levers():
    bm, flag_mask, verdict_mask = build_loss_masks()
    out_w = OPT["out_weight"]
    vw = OPT["verdict_weight"]
    fnw = OPT["flag_neg_weight"]
    last_frames = OPT.get("last_frames", 8)

    def weight_mask(bits):
        per_flag = torch.where(bits[:, :M].bool(), 1.0, fnw)
        return torch.einsum("bm,mhw->bhw", per_flag, bm[:-1]) + vw * verdict_mask

    def screen_loss(pred, tgt, wm):
        diff = (pred - tgt.unsqueeze(1)) ** 2
        T = diff.shape[1]
        out = (diff * wm.unsqueeze(1)).sum() / (wm.sum() * T)
        return diff.mean() + out_w * out

    if args.init_ckpt.lower() in ("none", "scratch", ""):
        nca = NeuralCA(**CFG["nca"]).to(DEVICE)
        if args.fire_rate is not None:
            nca.stochastic_update.fire_rate = args.fire_rate
        init_step = -1
    else:
        nca, init_step = load_nca(RUN / "checkpoints" / args.init_ckpt)
    nca.train()

    # ── run directory (STAB_ prefix; runs/ is never glob-deleted) ────────────
    run_out = ckpt_out = None
    if args.save_run:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rd = Path(args.runs_dir)
        if not rd.is_absolute():
            rd = ROOT / rd
        run_out = rd / (f"STAB_{stamp}_{args.arm}_n{layout.n}_m{layout.m}_"
                        f"r{layout.r}_s{args.seed}")
        ckpt_out = run_out / "checkpoints"
        ckpt_out.mkdir(parents=True)
        (run_out / "config.json").write_text(json.dumps(
            {"arm": args.arm, "source_run": RUN.name, "init_ckpt": args.init_ckpt,
             "init_step": init_step, "layout": LAY, "nca": CFG["nca"],
             "optim": OPT, "args": {k: v for k, v in vars(args).items()}}, indent=2))
        print(f"  run dir: {run_out.name}")
    if args.delta_bound > 0:
        nca.stochastic_update = BoundedDelta(nca.stochastic_update, args.delta_bound).to(DEVICE)

    # Always AdamW so weight decay can be switched ON mid-run by writing the
    # param group; AdamW at weight_decay=0 is exactly Adam.
    scheduled = args.stab_trigger > 0.0
    wd0 = args.wd0 if args.wd0 is not None else (0.0 if scheduled else args.wd)
    optim = torch.optim.AdamW(nca.parameters(), lr=args.lr, weight_decay=wd0)
    armed_step = None if scheduled else (0 if (args.wd > 0 or args.lr_final) else None)

    train_rng = random.Random(args.seed + 50_000)
    e_inp, e_tgt, e_bits = eval_batch()

    # experience-replay pool: slot -> (inp, tgt, bits, state|None)
    use_pool = args.pool_size > 0
    pool = [None] * args.pool_size if use_pool else None

    def fresh(bs):
        return layout.sample_batch(bs, mode="verify", device=DEVICE, rng=train_rng,
                                   sat_frac=OPT.get("sat_frac", 0.5),
                                   near_miss_frac=OPT.get("near_miss_frac", 0.5))

    @torch.no_grad()
    def held_out(steps):
        torch.manual_seed(args.seed + 10_000)
        x = implant(e_inp)
        for _ in range(steps):
            x = nca.forward(x, steps=1)[:, -1]
        return acc(x[:, 0], e_bits)

    print(f"\nLEVER ARM '{args.arm}'  init {args.init_ckpt} (step {init_step})")
    print(f"  lr {args.lr:g} -> {args.lr_final if args.lr_final else args.lr:g}   wd {args.wd:g}"
          + (f" (SCHEDULED: both held off until verdict >= {args.stab_trigger:g})" if scheduled else "") + "   "
          f"clip {args.grad_clip:g}   tbptt {args.tbptt or 'full'}   "
          f"delta_bound {args.delta_bound or 'off'}   pool {use_pool}")
    v0, e0, f0 = held_out(EVAL_STEPS)
    p0, _, _ = held_out(4 * EVAL_STEPS)
    prod0, _ = rule_spectral_product(nca)
    print(f"  start: verdict {v0:.3f}  exact {e0:.3f}  verdict@4x {p0:.3f}  Wspec {prod0:.2f}")

    hist = []
    best = (-1.0, 0)
    lo, hi = ROLL_MIN, ROLL_MAX
    from tqdm import tqdm
    pbar = tqdm(range(args.lever_steps), ncols=110, desc=args.arm)
    for step in pbar:
        # lr schedule. Unscheduled: decay across the whole run. Scheduled: hold
        # --lr at full size until the treatment arms, then decay across whatever
        # is left. The plateau needs the step size; the basin does not.
        if args.lr_final:
            if not scheduled:
                frac = step / max(1, args.lever_steps - 1)
                lr_now = args.lr * (args.lr_final / args.lr) ** frac
            elif armed_step is None:
                lr_now = args.lr
            else:
                span = max(1, args.lever_steps - 1 - armed_step)
                frac = min(1.0, (step - armed_step) / span)
                lr_now = args.lr * (args.lr_final / args.lr) ** frac
            for pg in optim.param_groups:
                pg["lr"] = lr_now
        steps = int(np.random.randint(lo, hi + 1))

        if use_pool:
            idx = [random.randrange(args.pool_size) for _ in range(args.lever_batch)]
            n_fresh = max(1, round(args.lever_batch * args.pool_fresh))
            need = [i for i, j in enumerate(idx) if pool[j] is None] 
            inp_l, tgt_l, bits_l, st_l = [], [], [], []
            f_inp, f_tgt, f_bits, _ = fresh(args.lever_batch)
            for i, j in enumerate(idx):
                if i < n_fresh or pool[j] is None:
                    pool[j] = (f_inp[i], f_tgt[i], f_bits[i], None)
                inp_, tgt_, bits_, st_ = pool[j]
                inp_l.append(inp_); tgt_l.append(tgt_); bits_l.append(bits_)
                st_l.append(st_ if (st_ is not None and i >= n_fresh) else None)
            inp = torch.stack(inp_l); tgt = torch.stack(tgt_l); bits = torch.stack(bits_l)
            base = implant(inp)
            state = torch.stack([base[i] if st_l[i] is None else st_l[i]
                                 for i in range(args.lever_batch)])
        else:
            inp, tgt, bits, _ = fresh(args.lever_batch)
            state = implant(inp)
            idx = None

        k = args.tbptt if args.tbptt > 0 else steps
        k = max(k, last_frames)
        if k < steps:
            with torch.no_grad():
                for _ in range(steps - k):
                    state = nca.forward(state, steps=1)[:, -1]
            state = state.detach()
        frames = []
        for _ in range(k):
            state = nca.forward(state, steps=1)[:, -1]
            frames.append(state[:, 0])
        seq = torch.stack(frames, dim=1)[:, -last_frames:]
        loss = screen_loss(seq, tgt, weight_mask(bits))

        optim.zero_grad()
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(
            nca.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")))
        optim.step()

        if use_pool:
            fin = state.detach().clone()
            for i, j in enumerate(idx):
                # idx can repeat a slot inside one batch, and a slot retired
                # earlier in this same loop is already None -- unpacking it
                # blindly is a crash that only fires once the reset probability
                # and a duplicate coincide, i.e. thousands of steps in.
                if pool[j] is None or random.random() < 0.05:
                    pool[j] = None
                else:
                    inp_, tgt_, bits_, _ = pool[j]
                    pool[j] = (inp_, tgt_, bits_, fin[i])

        if step % args.lever_eval_every == 0 or step == args.lever_steps - 1:
            nca.eval()
            v, e, f = held_out(EVAL_STEPS)
            p4, _, _ = held_out(4 * EVAL_STEPS)
            nca.train()
            prod, _ = rule_spectral_product(nca)
            tn = torch.sqrt(sum((q.detach() ** 2).sum() for q in nca.parameters()
                                if q.requires_grad)).item()
            hist.append(dict(step=step, loss=loss.item(), gnorm=gnorm, verdict=v, exact=e,
                             flag=f, verdict4x=p4, wspec=prod, theta=tn))
            if scheduled and armed_step is None and v >= args.stab_trigger - 1e-9:
                armed_step = step
                for pg in optim.param_groups:
                    pg["weight_decay"] = args.wd
                tqdm.write(f"  [step {step}] TREATMENT ARMED at verdict {v:.3f} "
                           f">= trigger {args.stab_trigger:.2f}: weight_decay -> {args.wd:g}, "
                           f"lr {args.lr:g} -> {args.lr_final if args.lr_final else args.lr:g} "
                           f"over the remaining {args.lever_steps - step} steps")
            if v > best[0] + 1e-9:
                best = (v, step)
                if ckpt_out is not None:
                    torch.save({"nca": nca.state_dict(), "optim": optim.state_dict(),
                                "step": step, "nca_config": CFG["nca"], "layout": LAY},
                               ckpt_out / "nca_best.pt")
            hist[-1]["best"] = best[0]
            hist[-1]["armed"] = armed_step
            hist[-1]["lr"] = optim.param_groups[0]["lr"]
            hist[-1]["wd"] = optim.param_groups[0]["weight_decay"]
            hist[-1]["steps_since_improve"] = step - best[1]
            if run_out is not None:
                with open(run_out / "eval.jsonl", "a") as fh:
                    fh.write(json.dumps(hist[-1]) + "\n")
                torch.save({"nca": nca.state_dict(), "optim": optim.state_dict(),
                            "step": step, "nca_config": CFG["nca"], "layout": LAY},
                           ckpt_out / "nca_last.pt")
        if run_out is not None:
            with open(run_out / "log.jsonl", "a") as fh:
                fh.write(json.dumps({"step": step, "loss": loss.item(), "gnorm": gnorm,
                                     "rollout": steps,
                                     "lr": optim.param_groups[0]["lr"]}) + "\n")
        pbar.set_description(f"{args.arm} l={loss.item():.3f} gn={gnorm:.1f} "
                             f"vd={hist[-1]['verdict']:.3f} W={hist[-1]['wspec']:.2f} "
                             f"stale={step - best[1]}")

    out = ROOT / f"stab_levers_{args.arm}{args.tag}.json"
    out.write_text(json.dumps(dict(arm=args.arm, run=RUN.name, init=args.init_ckpt,
                                   init_step=init_step, args=vars(args), hist=hist), indent=1))
    print(f"\n{'step':>6} {'loss':>8} {'gnorm':>10} {'verdict':>8} {'exact':>7} "
          f"{'vd@4x':>7} {'Wspec':>7} {'||theta||':>9}")
    for h in hist:
        print(f"{h['step']:6d} {h['loss']:8.4f} {h['gnorm']:10.2f} {h['verdict']:8.3f} "
              f"{h['exact']:7.3f} {h['verdict4x']:7.3f} {h['wspec']:7.2f} {h['theta']:9.3f}")
    vd = [h["verdict"] for h in hist]
    longest, start_of_best, end_of_best, windows = perfect_window(hist)
    print(f"\nARM {args.arm}: verdict start {vd[0]:.3f} end {vd[-1]:.3f} "
          f"min {min(vd):.3f} mean {np.mean(vd):.3f}   "
          f"Wspec {hist[0]['wspec']:.2f} -> {hist[-1]['wspec']:.2f}   "
          f"||theta|| {hist[0]['theta']:.3f} -> {hist[-1]['theta']:.3f}")
    bt = next((h["step"] for h in hist if h["verdict"] >= args.breakthrough - 1e-9), None)
    print(f"  breakthrough (verdict >= {args.breakthrough:g}) at continuation step "
          + (f"{bt}" if bt is not None else "NEVER")
          + (f"   treatment armed at {armed_step}" if scheduled else ""))
    print(f"  longest unbroken verdict-1.000 window: {longest} steps"
          + (f" (steps {start_of_best}..{end_of_best}, "
             f"{sum(1 for h in hist if start_of_best <= h['step'] <= end_of_best)} consecutive "
             f"evals)" if start_of_best is not None else "")
          + f"   [criterion: >= 10000]"
          + ("   PASS" if longest >= 10000 else ""))
    if len(windows) > 1:
        print(f"  all 1.000 windows: {windows}")
    _warn = check_window_consistency(hist, longest, best[0])
    if _warn:
        print(f"  !! {_warn}")
    if run_out is not None:
        fig, (b1, b2) = plt.subplots(1, 2, figsize=(13, 4))
        xs = [h["step"] for h in hist]
        b1.plot(xs, vd, "o-", ms=3, label="held-out verdict")
        b1.plot(xs, [h["exact"] for h in hist], "s-", ms=3, label="exact")
        b1.plot(xs, [h["verdict4x"] for h in hist], "^-", ms=3, alpha=0.6, label="verdict @4x rollout")
        b1.axhline(1.0, color="k", lw=0.8, ls="--")
        b1.set_ylim(0, 1.03); b1.set_xlabel("step"); b1.legend(fontsize=7)
        b1.set_title(f"{args.arm}: longest 1.000 window {longest} steps"
                     + (" (PASS)" if longest >= 10000 else ""))
        if start_of_best is not None:
            b1.axvspan(start_of_best, end_of_best, color="tab:green", alpha=0.12)
        b2.plot(xs, [h["theta"] for h in hist], label="||theta||")
        b2.plot(xs, [h["wspec"] for h in hist], label="rule Lipschitz")
        b2.set_xlabel("step"); b2.legend(fontsize=7); b2.set_title("optimiser drift")
        fig.tight_layout(); fig.savefig(run_out / "curves.png", dpi=110); plt.close(fig)
        (run_out / "summary.json").write_text(json.dumps(
            {"arm": args.arm, "longest_1000_window": longest,
             "window_first_step": start_of_best, "window_last_step": end_of_best,
             "all_1000_windows": windows, "criterion_met": bool(longest >= 10000),
             "breakthrough_step": bt, "armed_step": armed_step,
             "verdict_mean": float(np.mean(vd)), "verdict_min": float(min(vd)),
             "best": best[0], "best_step": best[1]}, indent=2))
    print(f"wrote {out.name}" + (f" and {run_out}" if run_out else ""))


{"sweep": cmd_sweep, "jacobian": cmd_jacobian, "gradepth": cmd_gradepth,
 "persist": cmd_persist, "levers": cmd_levers}[args.mode]()
