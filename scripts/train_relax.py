#!/usr/bin/env python3
"""The shared relaxation trainer: one loop for every ``RelaxationTask``.

What this is
------------
``src/ncpu/relax.py`` defines the contract; this is the trainer that consumes
it. Point it at a task and it trains an NCA to be a dynamical system whose
attractors are that task's solutions. It is a direct descendant of
``scripts/train_sat_verify.py`` and keeps everything that script paid for:
grad-norm measurement and clipping, ``nca_best.pt``, plateau/staleness
instrumentation, RNG stream isolation, and a matched baseline printed beside
every rate.

What is different, and why
--------------------------
**Self-supervised energy, no targets.** The verifier had one right answer per
instance so a per-pixel MSE made sense. A solver does not: Wang tilings and
Sudoku grids generally have many solutions, and the per-pixel mean of a
solution set is not a solution (``docs/karp_notes.md``). The loss here is
``task.energy(state, instances)`` — differentiable soft constraint violation,
zero only at a solution. Nothing in this file ever looks at a target.

**The hold mask is re-implanted every step.** ``task.hold_mask`` marks the
givens plus any structural planes. Re-implanting them each step is what makes
them givens rather than suggestions, and it is also a load-bearing part of the
dynamics: a cell that keeps being told what it is becomes a fixed boundary
condition the rest of the grid relaxes against. ``clamp_mask`` — the strict
subset that is *only* givens — is what the audit checks, and it is checked on
the **scored** rollout. With the re-implant working it reads exactly 1.0; that
is the point. It is an assertion on the clamp path (right mask, right planes,
applied after the delta), not a behavioural score, and the run aborts if it
ever fails. Auditing a rollout with the hold *removed* instead — which this
file did at first — measures whether givens survive after deleting the thing
that preserves them, and reports 0.000 for a completely correct trainer.

**Energy is never an accuracy.** Reported rates all come from
``task.is_solved`` on the settled state, with ``relax.matched_baseline``
beside them.

Stability
---------
See ``docs/stability_findings.md``. The measured mechanism is NOT a diverging
BPTT Jacobian — per-step deltas stay bounded, and gradient norm does not grow
with rollout length. It is **optimiser drift under a permanently-binding
gradient clip**: with ``--grad-clip`` firing on ~100% of steps, Adam takes a
constant-size step forever, the weights random-walk out of the solution basin
(rule spectral-norm product 2.8 -> 13.0, monotone, on the 3SAT testbed), and
the solution is eventually lost. So the defaults here are ``--lr-final`` decay
and decoupled weight decay, both of which bound that drift, and the run logs
``wspec`` and ``theta_norm`` every eval so the drift is visible rather than
inferred.

Usage:
    uv run python scripts/train_relax.py --task wang --rows 6 --cols 6 --steps 25000
    uv run python scripts/train_relax.py --task sudoku --n 3 --difficulty naked_single
"""

import json
import random
import shutil
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ncpu.nca import NeuralCA
from ncpu.relax import (
    SolveReport,
    count_distinct,
    givens_preserved,
    matched_baseline,
    settled_state,
)
from ncpu.utils import git_info

# ── args ─────────────────────────────────────────────────────────────────────
parser = ArgumentParser()
parser.add_argument("--task", required=True, choices=["wang", "sudoku"])
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--steps", type=int, default=25_000)
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--batch-size", type=int, default=8)
# task geometry (only the flags its task reads are used)
parser.add_argument("--rows", type=int, default=6, help="wang")
parser.add_argument("--cols", type=int, default=6, help="wang")
parser.add_argument("--n-tiles", type=int, default=12, help="wang")
parser.add_argument("--n-colors", type=int, default=3, help="wang")
parser.add_argument("--tile-seed", type=int, default=0, help="wang tileset seed")
parser.add_argument("--n", type=int, default=3, help="sudoku box size")
parser.add_argument("--difficulty", default=None, help="task-defined preset name")
# rollout — ALWAYS a multiple of task.min_rollout(), never a constant
parser.add_argument("--rollout-scale-min", type=float, default=1.0)
parser.add_argument("--rollout-scale-max", type=float, default=1.5)
parser.add_argument("--rollout-cap", type=int, default=600,
                    help="hard ceiling on rollout steps; a task whose min_rollout "
                         "exceeds this is reported, not silently truncated")
# ── commitment / decimation ──────────────────────────────────────────────────
parser.add_argument("--determinacy-weight", type=float, default=0.0, metavar="W",
                    help="supervise the task's confidence plane K against per-cell "
                         "DETERMINACY from the reference solver's propagation-only pass. "
                         "K gets exactly zero gradient from every task energy (measured: "
                         "0.000e+00 against 5.6e-06 into the belief planes), so unless "
                         "something drives it the stiffening dynamic never develops and the "
                         "relaxation has no commitment step. Track I measured decimation as "
                         "the algorithm on its lattice: one-shot propagation 0.062 vs "
                         "SP-decimation 0.688. Needs a task that can supply the target.")
parser.add_argument("--stiffness-weight", type=float, default=0.0, metavar="W",
                    help="drive K with NO oracle: a cell whose beliefs are still moving pays "
                         "in proportion to how confident it claims to be. Learned decimation "
                         "rather than imitated. Needs --stiffness-reward or the trivial "
                         "optimum is K -> -inf everywhere.")
parser.add_argument("--stiffness-reward", type=float, default=0.1, metavar="R",
                    help="reward per unit of mean stiffness, opposing the above. The ratio "
                         "W:R sets how much stability a cell must show before committing.")
parser.add_argument("--stiffness-beta", type=float, default=1.0,
                    help="sharpness of sigmoid(beta*K) when reading K as a commitment gate")
parser.add_argument("--settle-weight", type=float, default=0.0, metavar="W",
                    help="penalise state change across the supervised tail window. "
                         "--last-frames already averages energy over the tail to ask for a "
                         "settled state; on Wang that left `settled` pinned at 0.00 for a "
                         "whole run, so this is the explicit version. Watch that it does not "
                         "simply freeze a bad state: compare violations, not just settled.")
parser.add_argument("--last-frames", type=int, default=8,
                    help="energy is averaged over this many FINAL frames, so the "
                         "answer has to be a settled state and not a transient")
# model
parser.add_argument("--hidden-channels", type=int, default=8,
                    help="NCA channels BEYOND the task's rendered planes (scratch space)")
parser.add_argument("--hidden", type=int, default=128, help="rule hidden width")
parser.add_argument("--kernel", type=int, default=3)
parser.add_argument("--fire-rate", type=float, default=0.5)
parser.add_argument("--padding", default=None, choices=["zeros", "circular"],
                    help="override the padding the task asks for. Padding SELECTS THE "
                         "CONSTRAINT GRAPH -- circular wraps the grid into a torus and makes "
                         "row 0 a neighbour of row H-1 -- so this is a task property, not a "
                         "style choice, and the trainer refuses an unsafe combination")
# optimisation
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--lr-final", type=float, default=None,
                    help="exponentially decay lr to this. OFF by default, deliberately: an "
                         "lr decay is a RETENTION treatment and an actively harmful "
                         "DISCOVERY treatment. On the 3SAT testbed, from a plateau "
                         "checkpoint, lr 1e-3 -> 1e-5 pinned the model at exactly its "
                         "baseline for 4,000 steps while the control ascended — the same "
                         "constant-speed walk that crosses the grokking plateau is the one "
                         "that later wanders off it, and an lr decay cannot tell the two "
                         "apart. Use it WITH --stab-trigger, which starts the decay only "
                         "once the model has arrived")
parser.add_argument("--weight-decay", type=float, default=0.0,
                    help="decoupled (AdamW) weight decay. Unlike an lr decay this does NOT "
                         "shrink the step, so it bounds the drift without blocking the "
                         "plateau crossing: at 0.1 it was the ONLY arm that both crossed "
                         "the plateau (breakthrough at 3,750 where the control never "
                         "reached 0.85) and retained (0.964 mean / 0.812 min over 5,000 "
                         "steps from the solved checkpoint), and from scratch it "
                         "reached 1.000 at 11.5k and held 11,749 steps. Off by default "
                         "because that is ONE TASK and ONE SEED — 0.1 is a starting value to "
                         "re-measure per task, not a constant to inherit. Applied from step 0 unless --stab-trigger is set, in which "
                         "case --wd-before applies until the trigger arms")
parser.add_argument("--wd-before", type=float, default=None, metavar="W",
                    help="weight decay before --stab-trigger arms (default: --weight-decay, "
                         "i.e. weight decay stays on throughout and the trigger gates the "
                         "lr decay ALONE — the recommended configuration)")
parser.add_argument("--stab-trigger", type=float, default=0.0, metavar="V",
                    help="SCHEDULED stabilisation, keyed on the held-out solve rate. While "
                         "it has never reached V the run is left to search: --wd-before and "
                         "a constant lr. The first eval at or above V arms the treatment — "
                         "weight decay becomes --weight-decay and the lr begins its decay to "
                         "--lr-final over the steps that remain. 0 = unscheduled.")
parser.add_argument("--grad-clip", type=float, default=1.0,
                    help="clip the global gradient norm. 0 disables clipping; the pre-clip "
                         "norm is measured and logged either way")
parser.add_argument("--grad-checkpoint", type=int, default=0, metavar="K",
                    help="recompute the rollout in K-step segments in backward: "
                         "~sqrt memory for ~1.3x compute. 0 = off")
parser.add_argument("--tbptt", type=int, default=0, metavar="K",
                    help="carry gradient through only the LAST K rollout steps (the rest "
                         "run under no_grad). 0 = full BPTT. Measured on the 3SAT testbed "
                         "as neutral for stability — it is here as a MEMORY lever")
parser.add_argument("--state-bound", type=float, default=0.0, metavar="B",
                    help="squash the whole state through B*tanh(x/B) after each NCA step. "
                         "REQUIRED for any task whose energy rewards saturated beliefs. "
                         "``NeuralCA.forward`` ends each step with torch.clip_(x, -10, 10), "
                         "a HARD clamp whose gradient is exactly zero outside the range; a "
                         "softmax energy with a small tau pays for ever-larger beliefs, so "
                         "the state is driven into that clamp and the BPTT product "
                         "underflows to zero. Measured on Wang 6x6: 64.3%% of the state "
                         "pinned at +-10 and the gradient norm at 5.4e-18 from step ~150 "
                         "onward, permanently. Set B strictly below 10 (3 is a good start) "
                         "so the hard clamp never engages and the bound stays differentiable. "
                         "Applied BEFORE the hold re-implant, so givens stay exact.")
parser.add_argument("--delta-bound", type=float, default=0.0,
                    help="soft-bound the NCA's per-step delta to +-b with b*tanh(d/b). "
                         "0 = off. Prototyped here rather than in nca.py; see the report "
                         "in docs/stability_findings.md")
# persistence / experience replay
parser.add_argument("--pool-size", type=int, default=0,
                    help="experience-replay pool of rollout end-states. 0 disables. When "
                         "on, a fraction of each batch continues from a stored state "
                         "instead of the seed, so the model is trained to STAY at a "
                         "solution rather than merely arrive at one")
parser.add_argument("--pool-fresh", type=float, default=0.5,
                    help="fraction of each batch that starts from the seed")
parser.add_argument("--pool-reset", type=float, default=0.05,
                    help="per-slot probability of being retired to a fresh instance")
# bookkeeping
parser.add_argument("--plot-every", type=int, default=500)
parser.add_argument("--media-every", type=int, default=5000)
parser.add_argument("--eval-batch", type=int, default=32)
parser.add_argument("--eval-scale", type=float, default=None,
                    help="rollout multiple used at eval (default: rollout-scale-max)")
parser.add_argument("--baseline-trials", type=int, default=16,
                    help="matched_baseline trials per instance; 0 skips the baseline "
                         "(only sane on a task where it is provably 0.000)")
parser.add_argument("--reference-budget", type=int, default=10_000)
parser.add_argument("--audit-freerun", action="store_true",
                    help="additionally roll out with the hold mask DROPPED and report "
                         "free_givens_ok: does the rule hold the givens unaided? A "
                         "diagnostic with no pass mark (it is ~0 for an untrained model), "
                         "reported separately from givens_ok, and it doubles eval cost")
parser.add_argument("--resume", type=str, default=None, help="run name, or 'last'")
parser.add_argument("--init-from", type=str, default=None)
parser.add_argument("--runs-dir", default="runs")
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)
# Separate streams, as in train_sat_verify.py: running an eval must never
# perturb the training instance sequence, or the run stops being reproducible.
TRAIN_RNG = random.Random(args.seed)
EVAL_RNG = random.Random(args.seed + 10_000)
BASE_RNG = random.Random(args.seed + 30_000)
DEVICE = args.device

if DEVICE.startswith("cuda"):
    if not torch.cuda.is_available():
        raise SystemExit(f"--device {DEVICE} but torch.cuda.is_available() is False")
    free, total = torch.cuda.mem_get_info(torch.device(DEVICE))
    print(f"GPU {torch.cuda.get_device_name(DEVICE)}: "
          f"{free / 1024 ** 3:.1f} of {total / 1024 ** 3:.1f} GB free")


# ── task ─────────────────────────────────────────────────────────────────────
def build_task():
    if args.task == "wang":
        from ncpu.tasks.wang import WangTiling
        return WangTiling(rows=args.rows, cols=args.cols, n_tiles=args.n_tiles,
                          n_colors=args.n_colors, seed=args.tile_seed,
                          difficulty=args.difficulty or "easy")
    from ncpu.tasks.sudoku import SudokuTask
    return SudokuTask(n=args.n, difficulty=args.difficulty or "search")


task = build_task()
P = task.n_planes
H, W = task.shape
DEPTH = task.min_rollout()
ROLL_MIN = min(args.rollout_cap, int(round(args.rollout_scale_min * DEPTH)))
ROLL_MAX = min(args.rollout_cap, max(ROLL_MIN, int(round(args.rollout_scale_max * DEPTH))))
EVAL_STEPS = min(args.rollout_cap,
                 int(round((args.eval_scale or args.rollout_scale_max) * DEPTH)))
CHANNELS = P + args.hidden_channels

print(f"task: {task.describe()}")
print(f"  planes {P}  grid {W}x{H}  nca channels {CHANNELS} ({P} rendered + "
      f"{args.hidden_channels} hidden)")
print(f"  min_rollout {DEPTH} -> train rollout {ROLL_MIN}..{ROLL_MAX}, eval at {EVAL_STEPS}")
if ROLL_MAX < args.rollout_scale_max * DEPTH:
    print(f"  WARNING: --rollout-cap {args.rollout_cap} is BELOW "
          f"{args.rollout_scale_max:g}x min_rollout ({args.rollout_scale_max * DEPTH:.0f}). "
          f"The rollout is truncated; treat a failure as a geometry result, not a "
          f"learning result (3sat_findings.md: exact 0.047 at 1.0x vs 0.906 at 2.0x).")

# ── two NeuralCA settings that a belief encoding REQUIRES ────────────────────
# Both are stated in relax.py's module docstring. They are resolved and asserted
# here rather than inherited, because both defaults are wrong for these tasks and
# both fail SILENTLY -- as a learning failure, not an error.

ALIVE_THRESHOLD = 0      # alive masking OFF. NOT a tuned default; see below.
"""``AliveMasking`` gates every cell's update on a max-pool of CHANNEL 0. In a
belief encoding channel 0 is an arbitrary member of the alphabet -- "belief in
tile 0", "belief in digit 1" -- so the gate is meaningless, and a cell that has
RULED THAT SYMBOL OUT sits strongly negative and is switched off entirely: the
cells that have made the most progress are the ones the mask kills. Every cell
in a relaxation task is alive by construction, so the mask must be off. There is
no relaxation task for which a non-zero value here is correct, which is why it is
a module constant and an assertion rather than a flag."""


def boundary_free_fraction(probe=8):
    """Fraction of BOUNDARY pixels that are NOT held, over a probe batch.

    This is the measurement the padding choice turns on. Zero means every
    boundary pixel is a given that gets re-implanted every step, so a circular
    wrap cannot reach a free cell and the torus seam is inert (Wang's clamped
    ring is the case that made this safe -- by luck, as its own docstring says).
    Anything above zero means a wrap would put two cells in each other's
    neighbourhood that the scorer does not treat as neighbours.
    """
    insts = task.sample(probe, random.Random(args.seed + 40_000))
    free = held = 0
    for inst in insts:
        m = np.asarray(task.hold_mask(inst))
        # Held only if held on EVERY plane. "Held on any plane" would be wrong:
        # Sudoku's hold mask covers the three structural planes everywhere, so
        # an ANY-test calls every boundary pixel held and the audit never fires
        # even though the digit planes there are completely free.
        m = m.min(axis=0) if m.ndim == 3 else m
        ring = np.concatenate([m[0, :], m[-1, :], m[1:-1, 0], m[1:-1, -1]])
        free += int((ring <= 0).sum())
        held += ring.size
    return free / max(1, held)


def resolve_padding():
    """Ask the task; never assume. Returns (padding_type, why)."""
    if args.padding:
        return args.padding, "--padding"
    declared = getattr(task, "padding_type", None)
    if declared:
        return declared, "task.padding_type"
    wrap = getattr(task, "wrap", None)
    if wrap is not None:
        return ("circular" if wrap else "zeros"), f"task.wrap={bool(wrap)}"
    return "zeros", "default (task declares neither padding_type nor wrap)"


PADDING, PAD_WHY = resolve_padding()
FREE_BOUNDARY = boundary_free_fraction()
print(f"  padding {PADDING!r} from {PAD_WHY}; free (un-held) boundary pixels "
      f"{FREE_BOUNDARY:.1%}")
if PADDING == "circular" and FREE_BOUNDARY > 0:
    if PAD_WHY.startswith("default"):
        raise SystemExit(
            f"REFUSING to run: circular padding with {FREE_BOUNDARY:.1%} of boundary "
            f"pixels free. A torus seam makes row 0 a neighbour of row H-1, so the rule "
            f"would learn a constraint graph the scorer does not use. Either give the task "
            f"a clamped boundary ring, declare `padding_type` / `wrap` on the task to opt "
            f"in deliberately, or pass --padding zeros.")
    print(f"  NOTE: circular padding with {FREE_BOUNDARY:.1%} free boundary pixels, "
          f"requested via {PAD_WHY}. The grid is a TORUS: row 0 neighbours row H-1. "
          f"min_rollout must already account for that ({DEPTH} steps).")

nca_config = dict(
    channels=CHANNELS,
    hidden_channels=[args.hidden],
    fire_rate=args.fire_rate,
    alive_threshold=ALIVE_THRESHOLD,
    zero_initialization=True,
    kernel_size=args.kernel,
    perception="static_sobel",
    padding_type=PADDING,
    read_only_dims=[],               # the hold mask does the clamping, per step
)
assert nca_config["alive_threshold"] == 0, (
    "alive masking must be OFF for every relaxation task: it gates on channel 0, "
    "which in a belief encoding is one arbitrary symbol of the alphabet")
assert nca_config["padding_type"] in ("zeros", "circular")


# ── run directory ────────────────────────────────────────────────────────────
RUNS_DIR = Path(args.runs_dir)
if not RUNS_DIR.is_absolute():
    RUNS_DIR = ROOT / RUNS_DIR
RUNS_DIR.mkdir(parents=True, exist_ok=True)

if args.resume == "last":
    cands = sorted(d.name for d in RUNS_DIR.iterdir()
                   if d.name.startswith("RELAX_") and args.task in d.name
                   and (d / "checkpoints" / "nca_last.pt").exists())
    if not cands:
        raise RuntimeError("no resumable RELAX runs found")
    args.resume = cands[-1]

if args.resume:
    run_dir = RUNS_DIR / args.resume
    ckpt_dir = run_dir / "checkpoints"
    if not run_dir.exists():
        raise FileNotFoundError(run_dir)
else:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (f"{args.rows}x{args.cols}" if args.task == "wang" else f"n{args.n}")
    run_dir = RUNS_DIR / f"RELAX_{stamp}_{args.task.upper()}_{tag}_s{args.seed}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)
    (run_dir / "snapshots").mkdir()

CONFIG = {
    "run": {"name": run_dir.name, "seed": args.seed, "total_steps": args.steps,
            "device": DEVICE, "task": args.task},
    "task": {"describe": task.describe(), "planes": P, "H": H, "W": W,
             "min_rollout": DEPTH, "rollout": [ROLL_MIN, ROLL_MAX],
             "eval_rollout": EVAL_STEPS, "difficulty": args.difficulty,
             "padding": PADDING, "padding_source": PAD_WHY,
             "free_boundary_fraction": FREE_BOUNDARY},
    "nca": nca_config,
    "optim": {k: getattr(args, k) for k in
              ["lr", "lr_final", "weight_decay", "wd_before", "stab_trigger", "state_bound",
               "determinacy_weight", "stiffness_weight", "stiffness_reward", "settle_weight",
               "grad_clip", "batch_size", "last_frames",
               "grad_checkpoint", "tbptt", "delta_bound", "pool_size", "pool_fresh",
               "pool_reset", "rollout_scale_min", "rollout_scale_max"]},
    "env": {"python": sys.version, "torch": torch.__version__, "git": git_info()},
}
if not args.resume:
    (run_dir / "config.json").write_text(json.dumps(CONFIG, indent=2))

print("NOTE ON PLATEAUS: these tasks grok. The 3SAT verifier sat at EXACTLY its baseline\n"
      "  for ~5,500 steps and then broke through to 1.000. Judge nothing before ~25k steps;\n"
      "  read steps_since_improve in eval.jsonl to tell 'never learned' from 'learned and\n"
      "  then lost it'. nca_best.pt holds the best-scoring weights either way.")
print(f"run: {run_dir.name}")


# ── model ────────────────────────────────────────────────────────────────────
class BoundedDelta(torch.nn.Module):
    """``b * tanh(delta / b)``, applied where the fire mask is applied.

    ``NeuralCA.forward`` clips the STATE to +-10 but bounds nothing about the
    per-step delta. This wraps ``nca.stochastic_update`` — the last thing done
    to ``delta`` before ``x = x + delta`` — so the option can be measured
    without editing ``nca.py``, which five tracks depend on. Off by default:
    on the 3SAT testbed the deltas were already bounded (RMS 0.03-0.3, never
    growing), so this was not the mechanism. Kept because a task with a much
    stiffer energy may yet need it.
    """

    def __init__(self, inner, bound):
        super().__init__()
        self.inner = inner
        self.bound = float(bound)

    def forward(self, delta):
        return self.bound * torch.tanh(self.inner(delta) / self.bound)


nca = NeuralCA(**nca_config).to(DEVICE)
if args.delta_bound > 0:
    nca.stochastic_update = BoundedDelta(nca.stochastic_update, args.delta_bound).to(DEVICE)

# Always AdamW so weight decay can be switched on mid-run by writing the param
# group; AdamW at weight_decay=0 is exactly Adam.
SCHEDULED = args.stab_trigger > 0.0
WD_BEFORE = args.wd_before if args.wd_before is not None else args.weight_decay
optim = torch.optim.AdamW(nca.parameters(), lr=args.lr,
                          weight_decay=(WD_BEFORE if SCHEDULED else args.weight_decay))
armed_step = None

if not SCHEDULED and args.lr_final and args.lr_final > 0:
    print("  WARNING: --lr-final without --stab-trigger decays the step size from step 0. "
          "That is a retention treatment applied during discovery, and it is what stopped a "
          "25k from-scratch 3SAT confirmation from EVER grokking. See "
          "docs/stability_findings.md.")
if args.weight_decay <= 0 and not args.lr_final:
    print("  NOTE: no stability treatment. These rollouts drift out of their solution — the "
          "3SAT verifier hit 1.000 at 9.5k and 0.469 by 16k. Recommended starting point:\n"
          "        --weight-decay 0.1 --lr-final 1e-5 --stab-trigger 0.9\n"
          "  (weight decay throughout: it bounds the drift without shrinking the step, so it "
          "does not block the plateau crossing; the lr decay arms only once the model has "
          "arrived.) From scratch on 3SAT this reached 1.000 at 11.5k and held it for the\n"
          "  remaining 11,749 steps. Watch wspec / theta_norm in eval.jsonl either way.\n"
          "  !! THE VALUE 0.1 IS EVIDENCED ON 3SAT VERIFICATION ONLY, one task and one seed.\n"
          "  Wang, Sudoku and Ising have differently-scaled energies — Ising's is an\n"
          "  optimisation objective, not a satisfaction one — so the right weight decay is a\n"
          "  per-task measurement, not an inherited constant. Re-measure it: two 4k probes,\n"
          "  one warm-started from a checkpoint sitting at its baseline (does the treatment\n"
          "  still cross the plateau?) and one from nca_best.pt (does it hold?). A lever\n"
          "  measured in only one of those two regimes will mislead you; that mistake cost\n"
          "  this track a 25k run. See docs/stability_findings.md.")

start_step, metrics = 0, []
if args.init_from:
    nca.load_state_dict(torch.load(args.init_from, map_location=DEVICE,
                                   weights_only=False)["nca"])
    print(f"warm-started from {args.init_from}")
if args.resume:
    ck = torch.load(ckpt_dir / "nca_last.pt", map_location=DEVICE, weights_only=False)
    nca.load_state_dict(ck["nca"])
    optim.load_state_dict(ck["optim"])
    start_step = ck["step"]
    metrics = ck.get("metrics", [])
    print(f"resumed from step {start_step}")


# ── batching ─────────────────────────────────────────────────────────────────
def render(instances):
    """(seed, hold, clamp) as (B, P, H, W) tensors on DEVICE."""
    seed = torch.from_numpy(np.stack([task.encode(i) for i in instances])).to(DEVICE)
    hold = torch.from_numpy(np.stack([task.hold_mask(i) for i in instances])).to(DEVICE)
    clamp = torch.from_numpy(np.stack([task.clamp_mask(i) for i in instances])).to(DEVICE)
    return seed.float(), hold.float(), clamp.float()


def implant(seed):
    """Seed the rendered planes; hidden channels start at zero (scratch space)."""
    x = torch.zeros(seed.shape[0], CHANNELS, H, W, device=DEVICE)
    x[:, :P] = seed
    return x


def reimplant(x, seed, hold):
    """Re-assert the givens and the structural planes.

    Done EVERY step, not just at the seed: that is what makes a given a
    boundary condition the rest of the grid relaxes against, rather than an
    initial suggestion the dynamics is free to forget. Only the rendered
    planes are touched; hidden channels are the rule's own scratch space.
    """
    keep = x[:, :P] * (1.0 - hold) + seed * hold
    return torch.cat([keep, x[:, P:]], dim=1)


def _segment(x, seed, hold, k):
    frames = []
    for _ in range(int(k)):
        x = nca.forward(x, steps=1)[:, -1]
        if args.state_bound > 0:
            # Keep the state off NeuralCA's hard +-10 clip, whose gradient is
            # zero outside the range. Applied BEFORE the re-implant so held
            # cells still land on exactly their seed value.
            x = args.state_bound * torch.tanh(x / args.state_bound)
        x = reimplant(x, seed, hold)
        frames.append(x[:, :P])
    return x, torch.stack(frames, dim=1)


def rollout(x, seed, hold, steps, grad_steps=None):
    """Step the NCA, returning (final_state, rendered frames (B, T, P, H, W)).

    ``grad_steps`` implements truncated BPTT: the first ``steps - grad_steps``
    run under ``no_grad``. ``--grad-checkpoint`` recomputes segments in backward
    instead of storing them (``checkpoint`` preserves RNG state, so the fire
    masks are reproduced exactly).
    """
    g = steps if not grad_steps else min(steps, max(grad_steps, args.last_frames))
    if g < steps:
        with torch.no_grad():
            x, _ = _segment(x, seed, hold, steps - g)
        x = x.detach()
    K = args.grad_checkpoint if args.grad_checkpoint > 0 else g
    out, done = [], 0
    while done < g:
        k = min(K, g - done)
        if args.grad_checkpoint > 0 and torch.is_grad_enabled():
            x, seg = checkpoint(_segment, x, seed, hold, k, use_reentrant=False)
        else:
            x, seg = _segment(x, seed, hold, k)
        out.append(seg)
        done += k
    return x, torch.cat(out, dim=1)


# ── experience-replay pool ───────────────────────────────────────────────────
# src/ncpu/pool.py exists but is wired to the old (inp, out) supervised dataset
# and to channel 1 as the input plane; neither matches this contract, so the
# same idea is re-expressed here against RelaxationTask. Slots hold the
# instance AND the state, because an instance's givens have to travel with the
# state they were re-implanted into.
POOL = [None] * args.pool_size if args.pool_size > 0 else None


def training_batch(bs):
    """Instances plus the state each rollout starts from.

    Without a pool this is always the seed. With one, ``--pool-fresh`` of the
    batch starts from the seed and the rest continue from a stored end-state,
    which is what turns 'reaches a solution' into 'stays at one'.
    """
    if POOL is None:
        insts = task.sample(bs, TRAIN_RNG)
        seed, hold, clamp = render(insts)
        return insts, seed, hold, clamp, implant(seed), None

    idx = [TRAIN_RNG.randrange(args.pool_size) for _ in range(bs)]
    n_fresh = max(1, round(bs * args.pool_fresh))
    need = sum(1 for i, j in enumerate(idx) if i < n_fresh or POOL[j] is None)
    new = task.sample(need, TRAIN_RNG) if need else []
    it = iter(new)
    insts, states = [], []
    for i, j in enumerate(idx):
        if i < n_fresh or POOL[j] is None:
            POOL[j] = (next(it), None)
        inst, st = POOL[j]
        insts.append(inst)
        states.append(st)
    seed, hold, clamp = render(insts)
    base = implant(seed)
    x = torch.stack([base[i] if states[i] is None else states[i].to(DEVICE)
                     for i in range(bs)])
    return insts, seed, hold, clamp, x, idx


def pool_update(idx, final):
    if POOL is None or idx is None:
        return
    fin = final.detach().cpu().clone()
    for i, j in enumerate(idx):
        # ``idx`` can name the same slot twice in one batch, and a slot retired
        # earlier in this loop is already None. Guard, or the crash waits for a
        # duplicate and a reset to coincide -- thousands of steps into a run.
        if POOL[j] is None or TRAIN_RNG.random() < args.pool_reset:
            POOL[j] = None
        else:
            POOL[j] = (POOL[j][0], fin[i])


# ── commitment machinery ─────────────────────────────────────────────────────
# The confidence plane is the task's, under whichever name it gave it.
CONF_PLANE = getattr(task, "conf_plane", None)
if CONF_PLANE is None:
    _cp = getattr(type(task), "confidence_plane", None)
    CONF_PLANE = task.confidence_plane if _cp is not None else None

_DET_CACHE = {}


def determinacy_target(instance):
    """Per-cell target for K: +1 where propagation alone fixes the cell, -1 where
    it does not. Cached per instance.

    This is the quantity survey propagation's *frozen* estimate is meant to
    carry, taken from ground truth instead. Track I measured SP's own estimate
    at rho = -0.09 against exact on its lattice, so supervising against the
    solver rather than against SP is the version where the NCA can beat the
    algorithm it imitates rather than inherit its blind spot.

    Caveat worth stating: this target is a function of the INSTANCE only, not of
    the model's current beliefs, so it teaches a static "which cells are easy"
    map rather than a dynamic frozen-variable estimate. That is the cheap first
    version, and it is the one whose target is exact.
    """
    key = id(instance)
    if key in _DET_CACHE:
        return _DET_CACHE[key]
    solver = getattr(task, "solver", None)
    if solver is None or not hasattr(solver, "propagate_only"):
        raise SystemExit(
            f"--determinacy-weight needs a reference solver exposing propagate_only(); "
            f"task {task.name!r} has none. Either add one or use --stiffness-weight, "
            f"which needs no oracle.")
    doms, ok, _ = solver.propagate_only(instance)
    t = np.array([1.0 if (d and (d & (d - 1)) == 0) else -1.0 for d in doms],
                 dtype=np.float32)
    _DET_CACHE[key] = t
    if len(_DET_CACHE) > 4096:
        _DET_CACHE.clear()
    return t


def commitment_terms(frames, instances, hold):
    """Extra loss terms that give K a gradient. Returns (loss, parts dict)."""
    parts = {}
    total = frames.new_zeros(())
    if CONF_PLANE is None or not (args.determinacy_weight or args.stiffness_weight):
        return total, parts
    tail = frames[:, -args.last_frames:]
    B = tail.shape[0]
    # free cells only: a held cell's K is re-implanted every step, so a loss on
    # it is a loss on a constant.
    freem = 1.0 - task.confidence(hold)                      # (B, cells)

    if args.determinacy_weight:
        tgt = torch.from_numpy(np.stack([determinacy_target(i) for i in instances]))
        tgt = tgt.to(frames.device)
        k = torch.stack([task.confidence(tail[:, t]) for t in range(tail.shape[1])], 1)
        err = ((k - tgt.unsqueeze(1)) ** 2 * freem.unsqueeze(1)).sum() / (
            freem.sum() * tail.shape[1] + 1e-8)
        total = total + args.determinacy_weight * err
        parts["determinacy"] = float(err)

    if args.stiffness_weight:
        k = torch.stack([task.confidence(tail[:, t]) for t in range(tail.shape[1])], 1)
        stiff = torch.sigmoid(args.stiffness_beta * k)       # (B, Tw, cells)
        bel = torch.stack([task.read_beliefs(tail[:, t]) for t in range(tail.shape[1])], 1)
        churn = (bel[:, 1:] - bel[:, :-1]).abs().mean(-1)    # (B, Tw-1, cells)
        pay = (stiff[:, 1:] * churn * freem.unsqueeze(1)).sum() / (
            freem.sum() * max(1, churn.shape[1]) + 1e-8)
        rew = (stiff * freem.unsqueeze(1)).sum() / (freem.sum() * stiff.shape[1] + 1e-8)
        term = args.stiffness_weight * pay - args.stiffness_reward * rew
        total = total + term
        parts["stiff_churn"] = float(pay)
        parts["stiff_mean"] = float(rew)
    return total, parts


# ── loss ─────────────────────────────────────────────────────────────────────
def energy_loss(frames, instances):
    """Mean ``task.energy`` over the final ``--last-frames`` frames.

    Averaging over a WINDOW rather than scoring the last frame is what asks for
    a settled state: a rule that oscillates through the right configuration
    pays, a rule that sits in it does not. Returns (scalar loss, per-sample).
    """
    tail = frames[:, -args.last_frames:]
    B, T = tail.shape[0], tail.shape[1]
    flat = tail.reshape(B * T, P, H, W)
    e = task.energy(flat, [instances[b] for b in range(B) for _ in range(T)])
    e = e.reshape(B, T)
    return e.mean(), e.mean(dim=1).detach()


def settle_penalty(frames):
    """Mean squared step-to-step change over the supervised tail."""
    if args.settle_weight <= 0:
        return frames.new_zeros(())
    tail = frames[:, -args.last_frames:]
    if tail.shape[1] < 2:
        return frames.new_zeros(())
    return ((tail[:, 1:] - tail[:, :-1]) ** 2).mean()


# ── evaluation ───────────────────────────────────────────────────────────────
class frozen_rng:
    """Run a block without advancing the training RNG, and with a fixed seed
    inside it, so the eval curve tracks learning and not fire-mask luck."""

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.seed)

    def __exit__(self, *exc):
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)


EVAL_SET = None


def eval_instances():
    """A FIXED held-out set, drawn once. Fixed so the curve moves when the model
    does rather than when the instances do."""
    global EVAL_SET
    if EVAL_SET is None:
        insts = task.sample(args.eval_batch, EVAL_RNG)
        ref = [task.reference_solve(i, budget=args.reference_budget) for i in insts]
        EVAL_SET = (insts, ref)
    return EVAL_SET


@torch.no_grad()
def evaluate(steps) -> SolveReport:
    """The relax.py scorecard. Every rate is on the SETTLED state, exactly
    checked, beside a budget-matched random-walk baseline."""
    insts, ref = eval_instances()
    seed, hold, clamp = render(insts)
    rep = SolveReport()
    solved = stayed = giv = viol = dist = anyf = settl = 0.0
    steps_to = []
    with frozen_rng(args.seed + 20_000):
        x, frames = rollout(implant(seed), seed, hold, steps)
        # ``--audit-freerun`` is a SEPARATE diagnostic, reported as
        # ``free_givens_ok`` and never as ``givens_ok``: it drops the re-implant
        # to ask whether the rule has internalised the boundary condition well
        # enough to hold it unaided. It is expected to be ~0 early and it has no
        # 1.0 requirement. Computing ``givens_ok`` from it -- which this file did
        # -- measures whether givens survive after deleting the mechanism that
        # preserves them, so it reads 0.000 on a perfectly correct trainer.
        af = None
        if args.audit_freerun:
            _, af = rollout(implant(seed), seed, torch.zeros_like(hold), steps)
    B = len(insts)
    free_giv = 0.0
    for b, inst in enumerate(insts):
        fr = frames[b]                                   # (T, P, H, W)
        final, ok_settled, drift = settled_state(fr)
        cand = task.decode(final)
        s = task.is_solved(inst, cand)
        solved += float(s)
        viol += task.violations(inst, cand)
        settl += float(ok_settled)
        stride = max(1, steps // 64)
        n_dist = count_distinct(fr, task, stride=stride)
        dist += n_dist
        anyf += float(any(task.is_solved(inst, task.decode(fr[t]))
                          for t in range(0, fr.shape[0], stride)))
        # The audit runs on the SCORED rollout. relax.py: "Anything below 1.0
        # means the solver rewrote the puzzle, and every other number in the
        # report is about a different problem than the one posed." With the
        # re-implant working this is exactly 1.0 -- it is an assertion on the
        # clamp path, not a behavioural score, and it is the only thing that
        # catches a mask that is the wrong shape, applied to the wrong planes,
        # or applied before the delta instead of after.
        giv += givens_preserved(fr, seed[b], clamp[b])
        if af is not None:
            free_giv += givens_preserved(af[b], seed[b], clamp[b])
        if s:
            for t in range(0, fr.shape[0], stride):
                if task.is_solved(inst, task.decode(fr[t])):
                    steps_to.append(t)
                    break
        if args.baseline_trials > 0:
            rep.solved_baseline += matched_baseline(task, inst, n_dist, BASE_RNG,
                                                    trials=args.baseline_trials) / B
    # "stayed solved": keep running for another half-rollout and re-check.
    with frozen_rng(args.seed + 20_001):
        x2, fr2 = rollout(x, seed, hold, max(8, steps // 2))
    for b, inst in enumerate(insts):
        stayed += float(task.is_solved(inst, task.decode(fr2[b, -1])))
    rep.solved = solved / B
    rep.stayed_solved = stayed / B
    rep.givens_ok = giv / B
    rep.violations = viol / B
    rep.distinct_visited = dist / B
    rep.any_frame = anyf / B
    rep.settled = settl / B
    rep.steps_to_solution = float(np.mean(steps_to)) if steps_to else None
    works = [w for _, w in ref if w is not None]
    rep.reference_work = float(np.mean(works)) if works else None
    # Fraction of the state pinned at NeuralCA's hard +-10 clip. Logged every
    # eval because it is the leading indicator of the dead-gradient failure:
    # it climbs long before the gradient underflows, and it is invisible in the
    # loss curve.
    rep.notes = {"ref_solved": float(np.mean([s is not None for s, _ in ref])),
                 "clip_saturation": float((x.abs() >= 9.999).float().mean())}
    # Is the confidence plane carrying anything? A channel can MOVE and still be
    # pure noise -- on Wang at 13k steps K had temporal variance 0.113 (larger
    # than the belief planes) while correlating -0.076 with determinacy and
    # -0.115 with being right. "Not frozen" is not "informative", so measure the
    # correlation and not the variance.
    if CONF_PLANE is not None:
        try:
            kf = task.confidence(frames[:, -1])
            freem = 1.0 - task.confidence(hold)
            m = freem.reshape(-1) > 0
            kv = kf.reshape(-1)[m]
            rep.notes["K_std"] = float(kv.std())
            tg = torch.from_numpy(np.stack([determinacy_target(i) for i in insts]))
            dv = tg.to(kv.device).reshape(-1)[m]
            if float(kv.std()) > 1e-6 and float(dv.std()) > 1e-6:
                kc = kv - kv.mean(); dc = dv - dv.mean()
                rep.notes["K_corr_determinacy"] = float(
                    (kc * dc).mean() / (kv.std() * dv.std() + 1e-12))
        except Exception:
            pass
    if af is not None:
        rep.notes["free_givens_ok"] = free_giv / B
    return rep


def rule_spectral_product():
    """Product of the rule's 1x1-conv spectral norms — the Lipschitz bound on
    one NCA step, and the quantity that grew monotonically 2.8 -> 13.0 while
    the 3SAT verifier drifted out of its solution. Logged so the drift is
    visible while a run is happening, not only in the post-mortem."""
    prod = 1.0
    for mod in nca.rule.rule:
        w = getattr(mod, "weight", None)
        if w is not None and w.dim() == 4:
            prod *= torch.linalg.matrix_norm(w.reshape(w.shape[0], -1), ord=2).item()
    return prod


def save_ckpt(path, step):
    torch.save({"nca": nca.state_dict(), "optim": optim.state_dict(), "step": step,
                "metrics": metrics, "nca_config": nca_config, "config": CONFIG}, path)


def save_snapshot(path, frames, instances, n=4):
    """Seed / final belief map / decoded grid, per instance."""
    n = min(n, frames.shape[0])
    fig, axes = plt.subplots(3, n, figsize=(2.4 * n, 7), squeeze=False)
    for c in range(n):
        fr = frames[c]
        axes[0][c].imshow(fr[0, :min(P, 3)].permute(1, 2, 0).cpu().clamp(-1, 1) * 0.5 + 0.5,
                          interpolation="nearest")
        axes[1][c].imshow(fr[-1, :min(P, 3)].permute(1, 2, 0).cpu().clamp(-1, 1) * 0.5 + 0.5,
                          interpolation="nearest")
        axes[2][c].imshow(np.asarray(task.decode(fr[-1])).reshape(-1, 1)
                          if np.asarray(task.decode(fr[-1])).ndim == 1
                          else np.asarray(task.decode(fr[-1])),
                          cmap="tab20", interpolation="nearest")
        for r in range(3):
            axes[r][c].set_xticks([]); axes[r][c].set_yticks([])
    for r, name in enumerate(["seed", "settled", "decoded"]):
        axes[r][0].set_ylabel(name, fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ── clamp self-check ─────────────────────────────────────────────────────────
def verify_clamp(steps=6, probe=4):
    """Abort NOW if the re-implantation path does not hold the givens.

    ``relax.py`` is unambiguous: a solver that overwrites the givens has solved
    a different puzzle, and every other number in the report is about that other
    puzzle. So this is a hard precondition, not a metric to watch — it runs
    before step 0, costs a six-step rollout, and raises rather than warns.

    It exists because the failure is SILENT: a broken clamp trains happily for
    25k steps and produces a full, plausible, entirely invalid scorecard. It
    also fails in a way that is easy to misattribute — the first version of this
    trainer audited a rollout with the hold mask removed and reported
    ``givens_ok 0.000`` on a clamp path that was working correctly.

    Checks the two things that actually break: whether held cells equal the seed
    after a step (mask applied at all, and applied AFTER the delta), and whether
    the mask spans the plane dimension rather than broadcasting over it.
    """
    insts = task.sample(probe, random.Random(args.seed + 60_000))
    seed, hold, clamp = render(insts)
    for name, m in (("hold_mask", hold), ("clamp_mask", clamp)):
        if m.shape[1:] != (P, H, W):
            raise SystemExit(
                f"CLAMP CHECK FAILED: {name} is {tuple(m.shape[1:])} but the state is "
                f"({P}, {H}, {W}). A spatial (H, W) mask would broadcast across the plane "
                f"dimension and clamp every plane at a given's pixel, which is a different "
                f"problem from the one the task posed.")
    if float(clamp.sum()) == 0:
        print("  clamp check: task exposes NO givens — givens_ok is vacuous for this task.")
        return
    # The rule is zero-initialised, so a fresh model emits an EXACTLY zero delta
    # and nothing can move -- which made the first version of this check pass a
    # deliberately broken clamp. Force a non-trivial delta by randomising the
    # rule's output layer for the duration of the check, then restore it bit for
    # bit. Without this the check is vacuous precisely when it is most needed:
    # at step 0, before any result depends on it.
    out_conv = [m for m in nca.rule.rule if getattr(m, "weight", None) is not None
                and m.weight.dim() == 4][-1]
    saved = out_conv.weight.detach().clone()
    try:
        with torch.no_grad(), frozen_rng(args.seed + 60_001):
            out_conv.weight.normal_(0.0, 0.5)
            _, frames = rollout(implant(seed), seed, hold, steps)
    finally:
        with torch.no_grad():
            out_conv.weight.copy_(saved)
    moved = float((frames[:, -1] - frames[:, 0]).abs().max())
    if moved == 0.0:
        raise SystemExit(
            "CLAMP CHECK INCONCLUSIVE: the state did not move at all even with a "
            "randomised rule, so the check cannot distinguish a working clamp from a "
            "broken one. Refusing to train on an unverified clamp path.")
    held = clamp > 0
    err = ((frames[:, -1] - seed).abs() * held.float()).max().item()
    ok = float(givens_preserved(frames[0], seed[0], clamp[0]))
    if err > 1e-5 or ok < 1.0 - 1e-9:
        bad = ((frames[:, -1] - seed).abs() > 0.25) & held
        raise SystemExit(
            f"CLAMP CHECK FAILED after {steps} steps: {int(bad.sum())} of {int(held.sum())} "
            f"clamped cells moved (max drift {err:.4g}, givens_preserved {ok:.4f}).\n"
            f"  The re-implant is not holding. Check, in this order:\n"
            f"    1. is `hold` applied AFTER the NCA delta, inside the per-step loop?\n"
            f"    2. does it cover the rendered planes 0..{P - 1} and leave the hidden\n"
            f"       channels alone?\n"
            f"    3. are the recorded frames taken after the re-implant, not before?\n"
            f"  Refusing to train: every number this run produced would be about a\n"
            f"  different instance than the one posed.")
    print(f"  clamp check: {int(held.sum())} clamped cells held exactly over {steps} steps "
          f"(max drift {err:.2g}) while free cells moved {moved:.3g} — "
          f"givens_ok will read 1.000 unless the path breaks")


verify_clamp()

# ── training loop ────────────────────────────────────────────────────────────
DEAD_GRAD = {"warned": False}
best_solved, best_step = -1.0, None
for _m in metrics:
    if _m.get("eval_solved", -1.0) > best_solved:
        best_solved, best_step = _m["eval_solved"], _m["step"]

log_path = run_dir / "log.jsonl"
pbar = tqdm(range(start_step, start_step + args.steps), ncols=130,
            desc=f"relax {args.task} s{args.seed}")
lr_final = args.lr_final if args.lr_final and args.lr_final > 0 else args.lr

for step in pbar:
    # lr schedule. Unscheduled: decay across the whole run. Scheduled: hold --lr
    # at full size until the treatment arms, then decay across whatever is left.
    # The plateau crossing needs the step size; the basin does not.
    if lr_final != args.lr and args.steps > 1:
        if not SCHEDULED:
            frac = (step - start_step) / (args.steps - 1)
            lr_now = args.lr * (lr_final / args.lr) ** frac
        elif armed_step is None:
            lr_now = args.lr
        else:
            span = max(1, start_step + args.steps - 1 - armed_step)
            frac = min(1.0, (step - armed_step) / span)
            lr_now = args.lr * (lr_final / args.lr) ** frac
        for pg in optim.param_groups:
            pg["lr"] = lr_now
    steps = int(np.random.randint(ROLL_MIN, ROLL_MAX + 1))
    insts, seed, hold, clamp, x0, idx = training_batch(args.batch_size)
    xN, frames = rollout(x0, seed, hold, steps, grad_steps=args.tbptt or None)
    loss, per_sample = energy_loss(frames, insts)
    energy_only = float(loss)
    commit, commit_parts = commitment_terms(frames, insts, hold)
    settle = settle_penalty(frames)
    loss = loss + commit + args.settle_weight * settle

    optim.zero_grad()
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(
        nca.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf")))
    clipped = bool(args.grad_clip > 0 and grad_norm > args.grad_clip)
    optim.step()
    pool_update(idx, xN)

    with torch.no_grad():
        final = frames[:, -1]
        train_solved = float(np.mean([task.is_solved(insts[b], task.decode(final[b]))
                                      for b in range(len(insts))]))
    metrics.append({"step": step, "loss": loss.item(), "rollout": steps,
                    "loss_max_sample": float(per_sample.max()),
                    "grad_norm": grad_norm, "grad_clipped": int(clipped),
                    "train_solved": train_solved,
                    "energy": energy_only, "settle": float(settle),
                    **{f"c_{k}": v for k, v in commit_parts.items()},
                    "lr": optim.param_groups[0]["lr"]})
    clip_rate = float(np.mean([m["grad_clipped"] for m in metrics[-200:]]))
    # Dead-gradient detector. The failure it catches is silent and total: the
    # loss keeps printing a healthy-looking number while nothing trains at all,
    # and every downstream metric freezes bit-for-bit. Warn once, loudly.
    if len(metrics) >= 100 and not DEAD_GRAD["warned"]:
        recent = [m["grad_norm"] for m in metrics[-100:]]
        if max(recent) < 1e-9:
            DEAD_GRAD["warned"] = True
            sat = float((xN.abs() >= 9.999).float().mean())
            tqdm.write(
                f"\n  !! GRADIENT IS DEAD: max |grad| over the last 100 steps is "
                f"{max(recent):.3g}. Nothing is training; the loss value is meaningless.\n"
                f"     {sat:.1%} of the final state sits at NeuralCA's hard +-10 clip, whose "
                f"gradient is exactly zero outside the range —\n"
                f"     over a {steps}-step rollout the BPTT product then underflows. If that "
                f"percentage is non-trivial, rerun with --state-bound 3.\n"
                f"     Note AdamW keeps shrinking ||theta|| on a dead gradient, so a falling "
                f"theta_norm here is NOT evidence about weight decay.\n")
    stale = "-" if best_step is None else f"{step - best_step}"
    pbar.set_description(
        f"relax {args.task} s{args.seed} E={loss.item():.4f} solved={train_solved:.2f} "
        f"gn={grad_norm:.2f}{'*' if clipped else ''} clip={clip_rate:.0%} "
        f"best={max(best_solved, 0):.2f} stale={stale}")
    with open(log_path, "a") as f:
        f.write(json.dumps(metrics[-1]) + "\n")

    if step % args.plot_every != 0:
        continue

    rep = evaluate(EVAL_STEPS)
    wspec = rule_spectral_product()
    theta = float(torch.sqrt(sum((q.detach() ** 2).sum()
                                 for q in nca.parameters() if q.requires_grad)))
    metrics[-1].update({"eval_solved": rep.solved, "eval_baseline": rep.solved_baseline,
                        "eval_lift": rep.lift(), "eval_stayed": rep.stayed_solved,
                        "eval_givens_ok": rep.givens_ok, "eval_violations": rep.violations,
                        "eval_distinct": rep.distinct_visited, "eval_any_frame": rep.any_frame,
                        "eval_settled": rep.settled, "eval_steps_to": rep.steps_to_solution,
                        "eval_reference_work": rep.reference_work,
                        "eval_clip_saturation": rep.notes.get("clip_saturation"),
                        "eval_K_std": rep.notes.get("K_std"),
                        "eval_K_corr": rep.notes.get("K_corr_determinacy"),
                        "wspec": wspec, "theta_norm": theta,
                        "armed_step": armed_step, "lr": optim.param_groups[0]["lr"],
                        "wd": optim.param_groups[0]["weight_decay"]})
    if SCHEDULED and armed_step is None and rep.solved >= args.stab_trigger - 1e-9:
        armed_step = step
        for pg in optim.param_groups:
            pg["weight_decay"] = args.weight_decay
        tqdm.write(f"  [step {step}] TREATMENT ARMED at solved {rep.solved:.3f} >= trigger "
                   f"{args.stab_trigger:.2f}: weight_decay {WD_BEFORE:g} -> "
                   f"{args.weight_decay:g}, lr {args.lr:g} -> "
                   f"{lr_final:g} over the remaining {args.steps + start_step - step} steps")
    if rep.solved > best_solved + 1e-9:
        best_solved, best_step = rep.solved, step
        save_ckpt(ckpt_dir / "nca_best.pt", step)
    metrics[-1].update({"eval_best": best_solved, "eval_best_step": best_step,
                        "steps_since_improve": step - best_step})
    with open(run_dir / "eval.jsonl", "a") as f:
        f.write(json.dumps({k: v for k, v in metrics[-1].items()
                            if k in ("step", "wspec", "theta_norm")
                            or k.startswith(("eval_", "steps_since"))}) + "\n")
    _kn = ("" if "K_corr_determinacy" not in rep.notes else
           f"  K std {rep.notes['K_std']:.3f} corr {rep.notes['K_corr_determinacy']:+.3f}")
    tqdm.write(f"[{step}] {rep}{_kn}   wspec {wspec:.2f}  |theta| {theta:.2f}  "
               f"stale {step - best_step}")
    if rep.givens_ok < 1.0 - 1e-9:
        save_ckpt(ckpt_dir / "nca_last.pt", step)
        raise SystemExit(
            f"\nCLAMP AUDIT FAILED at step {step}: givens_ok {rep.givens_ok:.4f} "
            f"({1 - rep.givens_ok:.1%} of clamped cells overwritten).\n"
            f"  This passed the startup check, so the clamp path broke DURING the run — "
            f"look at anything that rebuilds masks or states per batch (the pool stores "
            f"states, not masks; a stale state paired with a fresh instance would do it).\n"
            f"  Aborting rather than training on: every rate from here on would describe a "
            f"different instance than the one posed. Checkpoint saved to nca_last.pt.")
    save_ckpt(ckpt_dir / "nca_last.pt", step)

    xs = [m["step"] for m in metrics]
    ev = [m for m in metrics if "eval_solved" in m]
    exs = [m["step"] for m in ev]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 4))
    a1.scatter(xs, [m["loss"] for m in metrics], s=1, alpha=0.4, label="energy (batch)")
    a1.scatter(xs, [m["loss_max_sample"] for m in metrics], s=1, alpha=0.25,
               color="tab:red", label="worst sample")
    a1.set_yscale("log"); a1.set_title("energy"); a1.set_xlabel("step"); a1.legend(fontsize=7)
    a2.plot(exs, [m["eval_solved"] for m in ev], lw=1.4, color="tab:green", label="solved")
    a2.plot(exs, [m["eval_baseline"] for m in ev], lw=0.9, ls=":", color="tab:green",
            label="matched baseline")
    a2.plot(exs, [m["eval_stayed"] for m in ev], lw=0.9, color="tab:blue", label="stayed solved")
    a2.plot(exs, [m["eval_settled"] for m in ev], lw=0.9, color="tab:gray", label="settled")
    a2.plot(xs, [m["train_solved"] for m in metrics], lw=0.4, alpha=0.3,
            color="tab:olive", label="train solved")
    a2.set_ylim(0, 1.02); a2.set_title("solve rate vs matched baseline")
    a2.set_xlabel("step"); a2.legend(fontsize=7)
    if best_step is not None:
        a2.axvline(best_step, color="k", lw=0.8, ls="-.", alpha=0.5)
    a3.scatter(xs, [m["grad_norm"] for m in metrics], s=1, alpha=0.4, color="tab:purple",
               label="pre-clip grad norm")
    if args.grad_clip > 0:
        a3.axhline(args.grad_clip, color="k", lw=0.8, ls="--",
                   label=f"clip {args.grad_clip:g} (fires {clip_rate:.0%})")
    a3b = a3.twinx()
    a3b.plot(exs, [m["wspec"] for m in ev], color="tab:orange", lw=1.2, label="rule Lipschitz")
    a3b.plot(exs, [m["theta_norm"] for m in ev], color="tab:brown", lw=1.0, label="||theta||")
    a3b.legend(fontsize=7, loc="lower right")
    a3.set_yscale("log"); a3.set_title("gradient norm + optimiser drift")
    a3.set_xlabel("step"); a3.legend(fontsize=7, loc="upper left")
    fig.tight_layout(); fig.savefig(run_dir / "curves.png", dpi=110); plt.close(fig)

    try:
        save_snapshot(run_dir / "snapshot_latest.png", frames.detach(), insts)
    except Exception as exc:                    # a plot must never kill a run
        tqdm.write(f"  (snapshot failed: {exc})")
    if step % args.media_every == 0 and step > 0:
        save_ckpt(ckpt_dir / f"nca_{step:06d}.pt", step)
        if (run_dir / "snapshot_latest.png").exists():
            shutil.copy(run_dir / "snapshot_latest.png",
                        run_dir / "snapshots" / f"snapshot_{step:07d}.png")

if args.steps > 0:
    final_step = start_step + args.steps - 1
    save_ckpt(ckpt_dir / "nca_last.pt", final_step)
    save_ckpt(ckpt_dir / f"nca_{final_step:06d}.pt", final_step)
print(f"\nDone. Run: {run_dir}")
print(f"Best solved {best_solved:.3f} at step {best_step} -> {ckpt_dir / 'nca_best.pt'}")
