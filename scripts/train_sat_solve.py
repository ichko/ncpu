#!/usr/bin/env python3
"""Train an NCA to SOLVE 3SAT on the clause matrix — Track C.

The input screen carries the formula with the assignment row left blank; the
NCA has to fill the assignment discs in so that every clause is satisfied.  The
assignment row is seeded with noise, which is both the symmetry breaker (so
different restarts land in different basins) and the nondeterministic machine's
guess.  With ``fire_rate < 1`` the update is already asynchronous and local,
which is exactly the shape of a WalkSAT flip.

Two objectives (``--objective``)
--------------------------------
``energy`` (default, and the principled one)
    A differentiable satisfaction loss read straight off the assignment discs:
    soft literal satisfaction ``sigmoid(beta * sign * a)``, soft-OR into
    clauses, then aggregated over clauses in the LOG domain (see
    ``ncpu.satsolve.aggregate_energy``).  No target screen exists or is needed,
    so any solution is equally acceptable — which is the honest statement of
    the task.
``supervised``
    Per-pixel MSE against a solved screen.  Only coherent on *uniquely*
    satisfiable instances, because otherwise the MSE-optimal output is the
    average of the solution set, and that average is not a solution.  Unique
    instances only exist near the phase ratio (at m/n = 2 there are none at
    all), and rejection-sampling them gets exponentially more expensive with n
    — which is the practical argument for ``energy`` being the primary mode.

Rollout depth
-------------
``SATLayout.min_rollout()`` is a distance in PIXELS: a 3x3 kernel moves
information one pixel per step and the lattice pitch is ``layout.step``.  That
is the time for information to make the trip ONCE.  A local search has to make
it repeatedly — assignment down to the clause rows, verdict of that clause back
up to the assignment row, flip, again — so the eval rollout is
``--rollout-mult`` (default 3) light cones, i.e. roughly three search sweeps.
Nothing here is hard-coded; everything is derived from ``min_rollout()``.

Training does NOT backprop through a full rollout: at these depths that is tens
of GB.  Instead there is a persistent state pool (the Growing-NCA trick, and
``src/ncpu/pool.py`` in this repo).  Each optimiser step advances a batch of
in-progress searches by ``--iter-steps`` steps with gradient and writes the
states back, so states accumulate arbitrarily deep search time across training
while each step stays cheap.  Slots are recycled when they get old.

Metrics
-------
Solve rate is always checked EXACTLY with ``ncpu.sat.evaluate`` on the decoded
assignment — never inferred from the energy.  The primary number is the
assignment the rollout RETURNS (decoded from its final frame), whose floor is a
single random draw.  ``any_frame`` — did the rollout pass through a solution at
any point — is logged as a diagnostic only: it has no fixed floor, since a rule
that learns nothing but to jitter its discs visits many assignments and trips it
at a rate approaching 1.0.  So every eval also runs a random walk given the SAME
number of distinct assignments and the same restart count, and reports both
matched floors, plus the distinct-assignment count itself so "it searched" and
"it twitched" stay distinguishable.  WalkSAT at a matched budget is printed at
startup as the reference.

Usage:
    uv run python scripts/train_sat_solve.py
    uv run python scripts/train_sat_solve.py --objective supervised --n 6 --m 26
    uv run python scripts/train_sat_solve.py --steps 40000 --iter-steps 48 --seed 1
    uv run python scripts/train_sat_solve.py --resume last
"""

import json
import random
import shutil
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu import sat
from ncpu.gif import write_gif
from ncpu.nca import NeuralCA
from ncpu.sat import SATLayout
from ncpu.satsolve import (
    AGGREGATES,
    NCASolver,
    RandomWalkSolver,
    assignment_union_mask,
    clause_tensors,
    decisiveness_penalty,
    decode_assignment,
    hard_unsat,
    read_discs,
    satisfaction_loss,
    search_work,
    seed_noise,
    walksat,
)

# ── Args ─────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=20_000)
parser.add_argument("--device", default="auto",
                    help="auto = a GPU with most of its memory free, else cpu; "
                         "or an explicit device such as cuda:1 / cpu")
parser.add_argument("--batch-size", type=int, default=8)

# instance shape.  n=6, m=26 is ratio 4.33 -- at the phase transition, which is
# also the only place uniquely-satisfiable instances actually exist.
parser.add_argument("--n", type=int, default=6, help="variables")
parser.add_argument("--m", type=int, default=26, help="clauses")
parser.add_argument("--k", type=int, default=3)
parser.add_argument("--r", type=int, default=2, help="disc radius (sets the lattice pitch)")

parser.add_argument("--objective", choices=["energy", "supervised"], default="energy")
parser.add_argument("--beta", type=float, default=4.0, help="literal sigmoid sharpness")
parser.add_argument("--aggregate", choices=list(AGGREGATES), default="logprod")
parser.add_argument("--tau", type=float, default=8.0, help="softmin sharpness")
parser.add_argument("--recon-weight", type=float, default=1.0,
                    help="energy: keep the formula pixels intact")
parser.add_argument("--decisive-weight", type=float, default=0.1,
                    help="energy: push assignment discs out to +-1")
parser.add_argument("--out-weight", type=float, default=5.0,
                    help="supervised: extra weight on the output discs")

# rollout / pool
parser.add_argument("--rollout-mult", default="auto",
                    help="eval rollout in SEARCH SWEEPS (multiples of min_rollout()); "
                         "'auto' sizes it from the measured walksat flip count at this "
                         "instance size, which is the only budget that stays matched "
                         "as the instance grows")
parser.add_argument("--max-rollout", type=int, default=4000,
                    help="cap on the derived eval rollout, so eval stays affordable")
parser.add_argument("--rollout", type=int, default=None, help="override the eval rollout")
parser.add_argument("--allow-under-depth", action="store_true",
                    help="run even when the rollout is below the light cone (it makes the "
                         "task impossible, not hard — only for deliberate ablations)")
parser.add_argument("--iter-steps", type=int, default=32,
                    help="NCA steps per optimiser step (all with gradient)")
parser.add_argument("--pool", type=int, default=64, help="persistent state pool size")
parser.add_argument("--max-age-mult", type=float, default=6.0,
                    help="recycle a pool slot after this many light cones of search")
parser.add_argument("--reset-frac", type=float, default=0.03,
                    help="fraction of sampled slots reset to a fresh instance each step")
parser.add_argument("--noise-scale", type=float, default=0.5,
                    help="std of the noise seeded into the blank assignment discs")

# model
parser.add_argument("--channels", type=int, default=16)
parser.add_argument("--hidden", type=int, default=128)
parser.add_argument("--kernel", type=int, default=3)
parser.add_argument("--fire-rate", type=float, default=0.5,
                    help="stochastic update — this is what makes the rule a local search")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr-final", type=float, default=None)
parser.add_argument("--init-from", type=str, default=None)

# eval / logging
parser.add_argument("--restarts", type=int, default=4, help="eval restarts per instance")
parser.add_argument("--eval-instances", type=int, default=12)
parser.add_argument("--eval-every", type=int, default=500)
parser.add_argument("--check-every", type=int, default=4, help="eval: decode every k steps")
parser.add_argument("--extrap-m", type=int, default=None,
                    help="also eval on a taller clause matrix (same rule, bigger instance)")
parser.add_argument("--probe-instances", type=int, default=128,
                    help="instances used for the startup well-posedness/baseline probe")
parser.add_argument("--plot-every", type=int, default=500)
parser.add_argument("--media-every", type=int, default=5000)
parser.add_argument("--resume", type=str, default=None, help="run name, or 'last'")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
rng = random.Random(args.seed)


def pick_device(requested, need_free=0.5):
    """Never take a GPU somebody else is using.

    ``auto`` picks the visible CUDA device with the most free memory, and only
    if at least ``need_free`` of it is actually free — otherwise it falls back
    to CPU with a message rather than colliding with a running job.
    """
    if requested != "auto":
        return requested
    if not torch.cuda.is_available():
        return "cpu"
    best, best_free = None, 0.0
    for i in range(torch.cuda.device_count()):
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception as exc:                      # ECC-dead / inaccessible
            print(f"  cuda:{i} unavailable ({exc})")
            continue
        frac = free / total
        print(f"  cuda:{i} {free / 1e9:.1f}/{total / 1e9:.1f} GB free ({frac:.0%})")
        if frac >= need_free and free > best_free:
            best, best_free = i, free
    if best is None:
        print("  no GPU with enough free memory — falling back to cpu "
              "(pass --device cuda:N to override)")
        return "cpu"
    return f"cuda:{best}"


DEVICE = pick_device(args.device)
print(f"device: {DEVICE}")

# ── Geometry + rollout depth (derived, never hard-coded) ─────────────────────
layout = SATLayout(n_vars=args.n, m_clauses=args.m, k=args.k, r=args.r)
LIGHT_CONE = layout.min_rollout()
MAX_AGE = int(round(args.max_age_mult * LIGHT_CONE))

if args.rollout:
    EVAL_ROLLOUT, SWEEPS, budget_note = args.rollout, args.rollout / LIGHT_CONE, "fixed"
else:
    if args.rollout_mult == "auto":
        SWEEPS = max(2.0, search_work(args.n, args.m, args.k, instances=32))
        budget_note = f"auto (walksat p75 = {SWEEPS:.1f} flips at this size)"
    else:
        SWEEPS = float(args.rollout_mult)
        budget_note = "fixed sweeps"
    EVAL_ROLLOUT = int(round(SWEEPS * LIGHT_CONE))
    if EVAL_ROLLOUT > args.max_rollout:
        print(f"  derived rollout {EVAL_ROLLOUT} capped at --max-rollout "
              f"{args.max_rollout} ({args.max_rollout / LIGHT_CONE:.1f} sweeps) — "
              f"the eval is budget-limited and solve rates are a lower bound")
        EVAL_ROLLOUT = args.max_rollout
        budget_note += ", capped"

print(layout.describe())
print(f"light cone (min_rollout) = {LIGHT_CONE} steps   "
      f"eval rollout = {EVAL_ROLLOUT} ({EVAL_ROLLOUT / LIGHT_CONE:.1f} sweeps, {budget_note})   "
      f"pool slot lifetime = {MAX_AGE} ({args.max_age_mult:.1f}x)")

# One WalkSAT flip needs one full information round trip, so a rollout shorter
# than the light cone cannot express even a single search move.
if EVAL_ROLLOUT < LIGHT_CONE and not args.allow_under_depth:
    raise SystemExit(
        f"\nFATAL: eval rollout {EVAL_ROLLOUT} is below the light cone {LIGHT_CONE} "
        f"(n={args.n} m={args.m} step={layout.step}).  The answer cannot physically "
        f"reach the assignment discs in that many steps, so the task is impossible "
        f"rather than hard and the resulting numbers describe the budget, not the "
        f"model.  Raise --rollout / --rollout-mult, or pass --allow-under-depth if "
        f"this really is a deliberate ablation."
    )
if EVAL_ROLLOUT < 2 * LIGHT_CONE:
    print(f"  WARNING: the rollout affords only {EVAL_ROLLOUT / LIGHT_CONE:.1f} search "
          f"sweeps; solving needs several flips (walksat p75 here is "
          f"{search_work(args.n, args.m, args.k, instances=32):.0f})")
print(f"training: {args.iter_steps} steps/optimiser step through a persistent pool "
      f"of {args.pool} searches")

# ── Well-posedness gate + trivial baselines (measured, not assumed) ──────────
probe_rng = random.Random(args.seed + 7919)
t0 = time.time()
counts = []
flag_pos = []
for _ in range(args.probe_instances):
    f = sat.random_ksat(args.n, args.m, args.k, probe_rng)
    s = sat.solutions(f, args.n)
    counts.append(len(s))
    guess = tuple(probe_rng.randint(0, 1) for _ in range(args.n))
    flag_pos.append(float(np.mean(sat.clause_flags(f, guess))))
counts = np.array(counts)
p_sat = float((counts > 0).mean())
p_unique = float((counts == 1).mean())
median_sols = float(np.median(counts[counts > 0])) if p_sat else float("nan")
random_guess_rate = float(np.mean(counts / (2 ** args.n)))

print(f"\ninstance probe ({args.probe_instances} uniform draws at n={args.n} m={args.m}, "
      f"ratio {args.m / args.n:.2f}, {time.time() - t0:.1f}s):")
print(f"  satisfiable            {p_sat:.3f}")
print(f"  uniquely satisfiable   {p_unique:.3f}"
      + (f"  (~{1 / p_unique:.0f} draws per unique instance)" if p_unique else ""))
print(f"  median #solutions      {median_sols:.1f} of {2 ** args.n} assignments")
print("baselines to beat (a solve rate below these means nothing was learned):")
print(f"  random assignment solves   {random_guess_rate:.3f}   <- floor for the "
      f"PRIMARY metric (the assignment the rollout returns)")
print("  the any-frame diagnostic has NO fixed floor: a random walk over enough "
      "distinct\n  assignments reaches 1.000, so it is scored against a "
      "visit-matched random walk")
print(f"  constant 'clause satisfied' flag accuracy {float(np.mean(flag_pos)):.3f}"
      "   <- why flags are NOT the metric; solve rate is checked exactly")

if args.objective == "supervised" and p_unique == 0.0:
    raise SystemExit(
        f"\nFATAL: --objective supervised needs uniquely-satisfiable instances and there "
        f"are none at n={args.n} m={args.m} (ratio {args.m / args.n:.2f}) — "
        f"sample_instance(want='unique') would spin through 20,000 rejections and raise.\n"
        f"Move the ratio to the phase transition ({sat.PHASE_RATIO}), e.g. "
        f"--n {args.n} --m {int(round(args.n * 4.33))}, or use --objective energy."
    )
if args.objective == "supervised" and p_unique < 0.05:
    print(f"  WARNING: only {p_unique:.1%} of instances are uniquely satisfiable; batch "
          f"sampling will cost ~{1 / p_unique:.0f} brute-force draws per instance")

# WalkSAT at a matched step budget — the reference the NCA has to be read against
ws_solved, ws_flips = [], []
for _ in range(args.eval_instances):
    f, _s = sat.sample_instance(args.n, args.m, args.k, probe_rng, want="sat")
    out = walksat(f, args.n, max_flips=EVAL_ROLLOUT, restarts=args.restarts, rng=probe_rng)
    ws_solved.append(out.solved)
    if out.solved:
        ws_flips.append(out.work)
WALKSAT_RATE = float(np.mean(ws_solved))
WALKSAT_FLIPS = float(np.median(ws_flips)) if ws_flips else float("nan")
print(f"  walksat @ {EVAL_ROLLOUT} flips x {args.restarts} restarts: solve rate "
      f"{WALKSAT_RATE:.3f}, median {WALKSAT_FLIPS:.0f} flips\n")

WANT = "unique" if args.objective == "supervised" else "sat"

# ── Model config ─────────────────────────────────────────────────────────────
nca_config = dict(
    channels=args.channels,
    hidden_channels=[args.hidden],
    fire_rate=args.fire_rate,
    alive_threshold=0,                # every cell updates; nothing is "dead" here
    zero_initialization=True,
    kernel_size=args.kernel,
    perception="static_sobel",
    padding_type="zeros",
    read_only_dims=[1],               # channel 1 = frozen copy of the formula
)
INPUT_DIMS = (0, 1)                   # ch0 mutable (carries the answer), ch1 frozen

# ── Run directory ────────────────────────────────────────────────────────────
if args.resume == "last":
    runs = sorted(d.name for d in Path("runs").iterdir()
                  if "SATSOLVE" in d.name and (d / "checkpoints" / "nca_last.pt").exists())
    if not runs:
        raise RuntimeError("No resumable SATSOLVE runs found")
    args.resume = runs[-1]

if args.resume:
    run_dir = Path("runs") / args.resume
    ckpt_dir = run_dir / "checkpoints"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
else:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("runs") / (f"{stamp}_SATSOLVE_{args.objective}_"
                              f"n{args.n}m{args.m}_s{args.seed}")
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "rollouts").mkdir()
    (run_dir / "snapshots").mkdir()
    with open(run_dir / "config.json", "w") as f:
        json.dump({
            "run": {"name": run_dir.name, "seed": args.seed, "total_steps": args.steps,
                    "device": DEVICE},
            "layout": {"n": layout.n, "m": layout.m, "k": layout.k, "r": layout.r,
                       "W": layout.W, "H": layout.H, "step": layout.step,
                       "min_rollout": LIGHT_CONE},
            "rollout": {"eval": EVAL_ROLLOUT, "mult": args.rollout_mult,
                        "iter_steps": args.iter_steps, "max_age": MAX_AGE},
            "nca": nca_config,
            "optim": vars(args),
            "probe": {"p_sat": p_sat, "p_unique": p_unique, "median_solutions": median_sols,
                      "random_guess_solve_rate": random_guess_rate,
                      "constant_flag_baseline": float(np.mean(flag_pos)),
                      "walksat_rate": WALKSAT_RATE, "walksat_median_flips": WALKSAT_FLIPS},
            "env": {"python": sys.version, "torch": torch.__version__},
        }, f, indent=2)

print(f"satsolve  seed={args.seed}  {'(resume) ' if args.resume else ''}run: {run_dir.name}")

# ── Model / optimiser ────────────────────────────────────────────────────────
nca = NeuralCA(**nca_config).to(DEVICE)
optim = torch.optim.Adam(nca.parameters(), lr=args.lr)
start_step = 0
metrics = []

if args.init_from:
    init_ck = torch.load(args.init_from, map_location=DEVICE, weights_only=False)
    nca.load_state_dict(init_ck["nca"])
    print(f"warm-started nca from {args.init_from}")

if args.resume:
    ckpt = torch.load(ckpt_dir / "nca_last.pt", map_location=DEVICE, weights_only=False)
    nca.load_state_dict(ckpt["nca"])
    optim.load_state_dict(ckpt["optim"])
    start_step = ckpt["step"]
    metrics = ckpt.get("metrics", [])
    print(f"resumed from step {start_step}")

asn_masks = layout.assignment_masks().to(DEVICE)                 # (n, H, W)
asn_union = assignment_union_mask(layout).to(DEVICE)             # (H, W)
out_masks = layout.output_bit_masks(mode="solve").to(DEVICE)     # (n+m+1, H, W)
out_union = out_masks.sum(0).clamp(0, 1)
# pixels the solver may scribble on: its own answer row, plus the flag column as
# scratch space.  Everything else (the formula itself) has to survive.
free_mask = out_union
keep_mask = 1.0 - free_mask


# ── Persistent pool of in-progress searches ──────────────────────────────────

class SolvePool:
    """One slot = one instance, mid-search.  Slots age and get recycled.

    Keeping the state between optimiser steps is what lets a 32-step gradient
    window train behaviour that only pays off hundreds of steps in: the search
    continues where it left off, so the model sees late-stage states without
    anyone ever backpropagating through the whole rollout.
    """

    def __init__(self, size):
        self.size = size
        self.slots = [self._fresh() for _ in range(size)]

    def _fresh(self):
        formula, sols = sat.sample_instance(layout.n, layout.m, layout.k, rng, want=WANT)
        solution = tuple(int(b) for b in sols[rng.randrange(len(sols))])
        inp, tgt = layout.screens_for(formula, solution, mode="solve")
        inp = torch.from_numpy(inp).to(DEVICE)
        tgt = torch.from_numpy(tgt).to(DEVICE)
        seeded = seed_noise(inp.clone(), asn_union, args.noise_scale)
        state = torch.zeros(args.channels, layout.H, layout.W, device=DEVICE)
        for d in INPUT_DIMS:
            state[d] = seeded if d == 0 else inp
        return {"state": state, "formula": formula, "inp": inp, "tgt": tgt, "age": 0}

    def sample(self, batch_size):
        idx = rng.sample(range(self.size), batch_size)
        for i in idx:                                  # keep some churn in the pool
            if rng.random() < args.reset_frac:
                self.slots[i] = self._fresh()
        slots = [self.slots[i] for i in idx]
        state = torch.stack([s["state"] for s in slots])
        inp = torch.stack([s["inp"] for s in slots])
        tgt = torch.stack([s["tgt"] for s in slots])
        formulas = [s["formula"] for s in slots]
        var_idx, sign = clause_tensors(formulas, device=DEVICE)
        return idx, {"state": state, "inp": inp, "tgt": tgt, "formulas": formulas,
                     "var_idx": var_idx, "sign": sign}

    def write_back(self, idx, states, steps):
        for j, i in enumerate(idx):
            slot = self.slots[i]
            slot["state"] = states[j].detach()
            slot["age"] += steps
            if slot["age"] >= MAX_AGE:
                self.slots[i] = self._fresh()


# ── Losses ───────────────────────────────────────────────────────────────────

def energy_objective(frames, batch):
    """Target-free: satisfaction energy + keep-the-formula + decisive discs."""
    energy, extra = satisfaction_loss(frames, asn_masks, batch["var_idx"], batch["sign"],
                                      beta=args.beta, aggregate=args.aggregate, tau=args.tau)
    loss = energy.mean()
    parts = {"energy": loss.item()}
    if args.decisive_weight:
        d = decisiveness_penalty(extra["a"])
        loss = loss + args.decisive_weight * d
        parts["decisive"] = d.item()
    if args.recon_weight:
        diff = (frames - batch["inp"].unsqueeze(1)) ** 2
        recon = (diff * keep_mask).sum() / (keep_mask.sum() * frames.shape[0] * frames.shape[1])
        loss = loss + args.recon_weight * recon
        parts["recon"] = recon.item()
    return loss, parts, extra


def supervised_objective(frames, batch):
    """Per-pixel MSE against the solved screen; only sound on unique instances."""
    diff = (frames - batch["tgt"].unsqueeze(1)) ** 2
    B, T = frames.shape[0], frames.shape[1]
    loss = diff.mean() + args.out_weight * (diff * out_union).sum() / (out_union.sum() * B * T)
    with torch.no_grad():
        _e, extra = satisfaction_loss(frames, asn_masks, batch["var_idx"], batch["sign"],
                                      beta=args.beta, aggregate=args.aggregate, tau=args.tau)
    return loss, {"mse": loss.item()}, extra


OBJECTIVE = energy_objective if args.objective == "energy" else supervised_objective


# ── Eval: exact, restart-aware ───────────────────────────────────────────────

def make_solver(seed, rollout=EVAL_ROLLOUT):
    """rollout=None makes the depth scale with the instance's own light cone —
    which is what a size-extrapolation eval needs, since a taller clause matrix
    takes proportionally longer for information to cross."""
    return NCASolver(nca, rollout=rollout, rollout_mult=args.rollout_mult,
                     allow_under_depth=args.allow_under_depth, restarts=args.restarts,
                     noise_scale=args.noise_scale, check_every=args.check_every,
                     chunk=max(16, args.iter_steps), device=DEVICE,
                     layout_kwargs={"k": args.k, "r": args.r}, seed=seed,
                     input_dims=INPUT_DIMS, channels=args.channels)


@torch.no_grad()
def solve_eval(step, n=None, m=None):
    """Fresh instances, fresh noise, exact scoring.  Never uses the pool.

    The PRIMARY number is ``solve_rate`` — the assignment the rollout actually
    returns, decoded from its final frame.  Its floor is one random draw.
    ``any_frame`` is a diagnostic: it asks whether the rollout ever passed
    through a solution, and a model that merely jitters its discs scores
    arbitrarily well on it, so it is only meaningful next to
    ``random_any_matched`` — the same question asked of a random walk that
    visits the same number of distinct assignments.
    """
    n = n or layout.n
    m = m or layout.m
    scaled = (n, m) != (layout.n, layout.m)
    solver = make_solver(seed=args.seed * 1000 + step,
                         rollout=None if scaled else EVAL_ROLLOUT)
    eval_rng = random.Random(args.seed * 7 + step)
    final_1, final_k, stable_k, any_k, distinct = [], [], [], [], []
    steps_hit, restarts_hit, unsat, formulas = [], [], [], []
    for _ in range(args.eval_instances):
        formula, _s = sat.sample_instance(n, m, args.k, eval_rng, want="sat")
        out = solver(formula, n)
        formulas.append(formula)
        final_1.append(bool(out.meta["final"][0]))
        final_k.append(out.solved)
        stable_k.append(any(out.meta["stable"]))
        any_k.append(any(out.meta["any"]))
        distinct.append(float(np.median(out.meta["distinct"])))
        unsat.append(out.unsat)
        if out.solved:
            steps_hit.append(out.steps)
            restarts_hit.append(out.restarts)

    # matched baselines: a random walk given the SAME number of distinct visits
    # and the same restart count that the NCA actually used.
    visits = max(1, int(round(float(np.median(distinct)))))
    walker = RandomWalkSolver(visits=visits, restarts=args.restarts, mode="walk",
                              seed=args.seed + step)
    base_final, base_any = [], []
    for formula in formulas:
        out = walker(formula, n)
        base_final.append(out.solved)
        base_any.append(any(out.meta["any"]))

    return {
        "solve_rate": float(np.mean(final_k)),            # primary: returned answer
        "solve_rate_1": float(np.mean(final_1)),          # ... on the first restart
        "stable_rate": float(np.mean(stable_k)),          # ... and it stayed put
        "any_frame": float(np.mean(any_k)),               # diagnostic only
        "distinct": float(np.median(distinct)),
        "random_final_matched": float(np.mean(base_final)),
        "random_any_matched": float(np.mean(base_any)),
        "median_steps": float(np.median(steps_hit)) if steps_hit else float("nan"),
        "median_restarts": float(np.median(restarts_hit)) if restarts_hit else float("nan"),
        "mean_unsat": float(np.mean(unsat)),
    }


# ── Media ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def save_media(step, stride=6):
    """A fresh rollout on one instance: gif of channel 0 + a still figure."""
    media_rng = random.Random(args.seed + step)
    formula, sols = sat.sample_instance(layout.n, layout.m, layout.k, media_rng, want="sat")
    inp = torch.from_numpy(layout.input_screen(formula)).to(DEVICE)
    frozen = inp.expand(args.restarts, *inp.shape).contiguous()
    seeded = seed_noise(frozen.clone(), asn_union, args.noise_scale)
    state = torch.zeros(args.restarts, args.channels, layout.H, layout.W, device=DEVICE)
    for d in INPUT_DIMS:
        state[:, d] = seeded if d == 0 else frozen

    var_idx, sign = clause_tensors([formula] * args.restarts, device=DEVICE)
    frames, unsat_curve = [seeded[0].cpu().numpy()], []
    done = 0
    chunk = max(16, args.iter_steps)
    while done < EVAL_ROLLOUT:
        k = min(chunk, EVAL_ROLLOUT - done)
        seq = nca.forward(state, steps=k)
        state = seq[:, -1]
        ch0 = seq[:, 1:, 0]
        frames.extend(ch0[0, ::stride].cpu().numpy())
        a = read_discs(ch0, asn_masks)
        unsat_curve.extend(hard_unsat(a, var_idx, sign)[0].cpu().numpy().tolist())
        done += k

    write_gif(run_dir / "rollout_latest.gif", np.stack(frames), fps=20, zoom=3)

    final = state[:, 0]
    bits = decode_assignment(final, asn_masks)
    solved = [sat.evaluate(formula, tuple(int(v) for v in b.tolist())) for b in bits]
    fig, axes = plt.subplots(1, 4, figsize=(11, 4.2))
    for ax, (name, data) in zip(axes[:3], [("formula in", inp), ("seeded ch0", seeded[0]),
                                           ("final ch0", final[0])]):
        ax.imshow(data.cpu(), cmap="viridis", vmin=-1, vmax=1, interpolation="nearest")
        ax.set_title(name, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    axes[3].plot(unsat_curve, lw=0.9)
    axes[3].set_title(f"unsat clauses (restart 0)\nsolved restarts: {sum(solved)}/"
                      f"{args.restarts}", fontsize=9)
    axes[3].set_xlabel("rollout step"); axes[3].set_ylim(bottom=0)
    fig.suptitle(f"step {step}  —  {layout.describe().splitlines()[0]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(run_dir / "snapshot_latest.png", dpi=120)
    plt.close(fig)


def save_curves():
    xs = [m["step"] for m in metrics]
    ev = [m for m in metrics if "solve_rate" in m]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16, 4))
    a1.scatter(xs, [m["loss"] for m in metrics], s=1, alpha=0.35)
    a1.set_yscale("log"); a1.set_title("loss"); a1.set_xlabel("step")
    a2.plot(xs, [m["train_unsat"] for m in metrics], lw=0.7)
    a2.set_title(f"unsat clauses on pool states (of {layout.m})"); a2.set_xlabel("step")
    if ev:
        ex = [m["step"] for m in ev]
        a3.plot(ex, [m["solve_rate"] for m in ev], "-o", ms=3, color="#2a6f97",
                label=f"solve rate (returned answer, {args.restarts} restarts)")
        a3.plot(ex, [m["stable_rate"] for m in ev], "-o", ms=2.5, color="#468faf",
                alpha=0.7, label="... and settled")
        a3.plot(ex, [m["any_frame"] for m in ev], ":", lw=1, color="0.45",
                label="any-frame (diagnostic)")
        a3.plot(ex, [m["random_any_matched"] for m in ev], ":", lw=1, color="crimson",
                label="random walk, matched visits (any-frame floor)")
        a3.plot(ex, [m["random_final_matched"] for m in ev], "--", lw=1, color="crimson",
                label="random walk (returned-answer floor)")
        a3.axhline(WALKSAT_RATE, color="0.2", ls="--", lw=1,
                   label=f"walksat @ matched budget ({WALKSAT_RATE:.2f})")
    a3.set_ylim(-0.02, 1.02); a3.set_title("exact solve rate"); a3.set_xlabel("step")
    a3.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(run_dir / "curves.png", dpi=120); plt.close(fig)


def save_checkpoint(path, step):
    torch.save({"nca": nca.state_dict(), "optim": optim.state_dict(), "step": step,
                "metrics": metrics, "nca_config": nca_config,
                "layout": {"n": layout.n, "m": layout.m, "k": layout.k, "r": layout.r,
                           "W": layout.W, "H": layout.H, "min_rollout": LIGHT_CONE},
                "input_dims": INPUT_DIMS, "objective": args.objective,
                "eval_rollout": EVAL_ROLLOUT, "noise_scale": args.noise_scale},
               path)


# ── Training loop ────────────────────────────────────────────────────────────
pool = SolvePool(args.pool)
log_path = run_dir / "log.jsonl"
pbar = tqdm(range(start_step, start_step + args.steps), ncols=118,
            desc=f"satsolve s{args.seed}")
lr_final = args.lr_final if args.lr_final is not None else args.lr

for step in pbar:
    if lr_final != args.lr and args.steps > 1:
        frac = (step - start_step) / (args.steps - 1)
        for pg in optim.param_groups:
            pg["lr"] = args.lr * (lr_final / args.lr) ** frac

    idx, batch = pool.sample(args.batch_size)
    seq = nca.forward(batch["state"], steps=args.iter_steps)
    frames = seq[:, 1:, 0]                                   # (B, T, H, W)
    loss, parts, extra = OBJECTIVE(frames, batch)

    optim.zero_grad()
    loss.backward()
    optim.step()

    pool.write_back(idx, seq[:, -1], args.iter_steps)

    with torch.no_grad():
        train_unsat = float(extra["unsat"][:, -1].float().mean())
    row = {"step": step, "loss": loss.item(), "train_unsat": train_unsat, **parts}

    if step % args.eval_every == 0 or step == start_step + args.steps - 1:
        nca.eval()
        row.update(solve_eval(step))
        if args.extrap_m:
            ex = solve_eval(step, m=args.extrap_m)
            row.update({f"extrap_{key}": value for key, value in ex.items()})
        nca.train()

    metrics.append(row)
    with open(log_path, "a") as f:
        f.write(json.dumps(row) + "\n")

    last_eval = next((m for m in reversed(metrics) if "solve_rate" in m), None)
    pbar.set_description(
        f"satsolve s{args.seed} loss={loss.item():.4f} unsat={train_unsat:.2f}/{layout.m}"
        + (f" solve={last_eval['solve_rate']:.2f} (rand "
           f"{last_eval['random_final_matched']:.2f}) any={last_eval['any_frame']:.2f}"
           if last_eval else "")
    )

    if step % args.plot_every != 0:
        continue
    save_checkpoint(ckpt_dir / "nca_last.pt", step)
    save_curves()
    if step % args.media_every == 0:
        save_media(step)
        save_checkpoint(ckpt_dir / f"nca_{step:06d}.pt", step)
        shutil.copy(run_dir / "snapshot_latest.png",
                    run_dir / "snapshots" / f"snapshot_{step:07d}.png")
        shutil.copy(run_dir / "rollout_latest.gif",
                    run_dir / "rollouts" / f"rollout_{step:07d}.gif")

# the last optimiser step must be on disk: a run that stops at 19,999 with its
# newest checkpoint at 19,500 has silently thrown away its final weights.
save_checkpoint(ckpt_dir / "nca_last.pt", step)
save_checkpoint(ckpt_dir / f"nca_{step:06d}.pt", step)
save_curves()
save_media(step)

print(f"\nDone. Run: {run_dir}")
print("Hardness curve for this model:")
print(f"  uv run python scripts/eval_sat_hardness.py --solver nca "
      f"--checkpoint {ckpt_dir / 'nca_last.pt'} --n-list {args.n} "
      f"--scale-n-list {args.n} {args.n + 2} {args.n + 4}")
