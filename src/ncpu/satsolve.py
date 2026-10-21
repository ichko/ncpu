"""3SAT *solving* on the grid — Track C, where the exponential lives.

Verification (Track A) is in P: broadcast the assignment down the variable
columns, AND the clause flags, done in O(n + m) steps.  Solving is not.  To go
from "check this assignment" to "find one" you need search, and the only kind of
search a cellular automaton can natively do is **stochastic local search**:

    WalkSAT: pick an unsatisfied clause, flip one of its three variables.

That is a local, asynchronous, message-passing update — exactly the shape of an
NCA step with ``fire_rate < 1``.  The noise seeded into the assignment row plays
the role of the nondeterministic machine's guess; rolling out for longer means
searching for longer.  So the hardness of the instance should show up as
**rollout steps** and **restarts**, and both are measurable.

What is in here
---------------
1. ``walksat`` — a plain reference solver (WalkSAT/SKC).  It is the baseline
   the learned NCA is measured against, and it is what
   ``scripts/eval_sat_hardness.py`` uses to draw the hardness curve.
2. The differentiable **satisfaction energy** — soft literal satisfaction from
   the assignment discs, soft-OR into clauses, aggregated over clauses.  This
   is a loss that needs *no target screen*, which is the point (see below).
3. Noise seeding, batch sampling and exact scoring helpers for the trainer.
4. ``WalkSATSolver`` / ``NCASolver`` / ``RandomWalkSolver`` — one
   ``SolveOutcome`` interface so the same hardness sweep can be run with any of
   them by a flag, including the degenerate baseline.

What an incomplete solver can and cannot show
---------------------------------------------
The textbook picture of random 3SAT is *easy-hard-easy* in the clause ratio: a
peak at the satisfiability threshold with a falling branch above it.  That
falling branch is **refutation getting cheap** — an over-constrained formula is
quickly proved UNSAT — and only a COMPLETE solver (DPLL/CDCL) can produce it.
WalkSAT is a stochastic local searcher: it can exhibit a solution but never
prove there is none, so on satisfiable instances its cost can only RISE as
solutions grow scarce.  Measured here at n=14 with 120 provably-satisfiable
instances per point and no timeouts, the curve is monotone (3 -> 26 flips over
m/n = 2.0 -> 5.5) with no peak.  The right claim for an NCA solver, which is
also incomplete, is therefore that its cost should track WalkSAT's **rising**
shape and the collapse of the solution count — not that it should reproduce a
peak.  Anything that does show a peak here is measurement noise.

Measuring a rollout solver without fooling yourself
---------------------------------------------------
Two traps, both of which produce a metric that climbs convincingly while the
model learns nothing:

* **Any-frame credit.**  Scoring "did any decoded frame satisfy" over a K-frame
  rollout with R restarts gives a random walk a hit probability that goes to 1
  as K grows.  The matched baseline for that metric is a random walk over the
  same number of DISTINCT assignments, which is what ``RandomWalkSolver``
  provides; the single-draw probability #solutions/2**n is not the floor.
* **Fixed budgets across instance sizes.**  Search work grows with n, so a
  rollout of fixed depth buys fewer and fewer search sweeps as the instance
  grows, and the resulting decline reads as a failure to extrapolate when it is
  really a shrinking budget.  ``NCASolver`` derives its depth from
  ``SATLayout.min_rollout()`` and can size it from the measured search work
  (``rollout_mult="auto"``); the sweep keeps a fixed-budget control curve so the
  two effects stay separable.

Why a target screen is the wrong objective (mostly)
---------------------------------------------------
A formula with many satisfying assignments has no single correct assignment
row.  A per-pixel MSE against one arbitrary solution is then not just weak, it
is actively wrong: the minimiser of the expected MSE is the *average* of the
solution set, and the average of two solutions is generally not a solution.
Two ways out, both selectable in the trainer:

``supervised``  train only on uniquely-satisfiable instances
                (``sat.sample_instance(want="unique")``) so the target is
                well-defined.  Honest, but it trains on a strange, atypical
                slice of instance space.
``energy``      score the assignment discs by how well they satisfy the
                formula, with no target at all.  Any solution is equally good,
                which is what we actually mean.

Aggregation over clauses matters
--------------------------------
The formula is an AND, so the natural relaxation is a product over clauses, not
a mean.  The default here is ``logprod``:

    L = mean_i [ -log s_i ]      (s_i = soft satisfaction of clause i)

the negative log-likelihood of the whole formula being satisfied under an
independent-bit relaxation, divided by m so the scale does not drift with the
instance size.  Two properties, both checked by ``scripts/preview_satsolve.py``
with no training involved:

* Once the assignment discs have COMMITTED to +-1, -log of a satisfied clause is
  a constant, so ``logprod`` is (unsat clause count) / m up to a factor — it is
  literally the WalkSAT objective.  Measured, it is then indistinguishable from
  ``mean`` (one broken clause costs 0.199 vs 0.198 of the energy range).  The
  often-repeated claim that a mean "lets the model abandon a clause" is not
  true in this regime, and this file does not rely on it.
* Where the two separate is the UNDECIDED regime — the discs early in a
  rollout, straight out of the noise seed.  d(-log s_i)/ds_i = -1/(m s_i)
  diverges as a clause collapses, while the mean's gradient is a constant
  -1/m.  So the worse a clause is doing, the harder logprod pulls on it, which
  is the behaviour a local search needs.  Measured rank agreement with the
  exact unsat count at quarter-magnitude discs: logprod +0.91, mean +0.88.

``softmin`` is the sharper worst-clause aggregate — it puts 0.73 of the energy
range on the FIRST broken clause (vs 0.20 for logprod) — but its gradient on
every other clause is nearly flat, so it is offered as an ablation rather than
the default.  ``min`` is rank-blind past one clause (rho +0.18: it cannot tell
one unsatisfied clause from five) and is there only to make that visible.

Value conventions follow ``ncpu.sat``: float32 screens, background 0.0, a false
disc at -1.0 and a true disc at +1.0.
"""

import random
from dataclasses import dataclass, field

import numpy as np
import torch

from ncpu.sat import SATLayout, evaluate, random_ksat, sample_instance

EPS = 1e-6


# ── outcome interface ────────────────────────────────────────────────────────

@dataclass
class SolveOutcome:
    """One attempt at one instance, from any solver.

    ``solved`` is the answer the solver ACTUALLY RETURNS — for a rollout solver
    that means the assignment decoded from the final frame, not "some frame
    somewhere satisfied the formula".  The distinction is not pedantic: a rule
    that only learns to jitter its assignment discs visits many assignments and
    will trip an any-frame test at a rate that rises to 1.0 with the number of
    distinct assignments visited, while contributing nothing.  Any-frame hits
    live in ``meta`` as a diagnostic and must be reported against a baseline
    matched on visits, never against the single-draw probability.

    solved:     the exact checker accepts the RETURNED assignment
    steps:      cost inside the winning attempt (WalkSAT flips / NCA steps)
    restarts:   number of attempts used, 1-based (so 1 == first try)
    work:       total cost across every attempt, including the failed ones
    assignment: the returned assignment
    meta:       per-attempt diagnostics — ``any``/``first`` (any-frame hits and
                when), ``final``/``stable`` (settled-answer hits), ``distinct``
                (assignments visited, i.e. how much searching really happened)
    """

    solved: bool
    steps: int
    restarts: int
    work: int
    assignment: tuple = ()
    unsat: int = -1          # unsatisfied clauses of the returned assignment
    meta: dict = field(default_factory=dict)


# ── WalkSAT/SKC reference solver ─────────────────────────────────────────────

def _occurrences(formula, n):
    """occ[v] = [(clause index, value that makes this literal true), ...]."""
    occ = [[] for _ in range(n)]
    for ci, clause in enumerate(formula):
        for lit in clause:
            occ[abs(lit) - 1].append((ci, 1 if lit > 0 else 0))
    return occ


def count_unsat(formula, assignment):
    """Number of clauses this assignment fails — the WalkSAT objective."""
    bad = 0
    for clause in formula:
        for lit in clause:
            if assignment[abs(lit) - 1] == (1 if lit > 0 else 0):
                break
        else:
            bad += 1
    return bad


def walksat_once(formula, n, max_flips=10_000, noise=0.5, rng=random,
                 assignment=None, trace=None):
    """One WalkSAT run from one random (or given) start.

    The rule (Selman-Kautz-Cohen): pick a random unsatisfied clause; if some
    variable in it can be flipped without breaking any currently-satisfied
    clause ("freebie"), flip that one; otherwise flip a random variable with
    probability ``noise`` and the least-breaking one otherwise.

    Args:
        trace: if a list is passed, every intermediate assignment is appended
               to it (used by the preview to animate the search).

    Returns ``(solved, assignment, flips, unsat)``.
    """
    a = list(assignment) if assignment is not None else [rng.randint(0, 1) for _ in range(n)]
    occ = _occurrences(formula, n)
    m = len(formula)

    true_count = [0] * m
    for ci, clause in enumerate(formula):
        true_count[ci] = sum(1 for lit in clause if a[abs(lit) - 1] == (1 if lit > 0 else 0))

    unsat = [ci for ci in range(m) if true_count[ci] == 0]
    where = [-1] * m                      # position of clause ci inside `unsat`
    for pos, ci in enumerate(unsat):
        where[ci] = pos

    def drop(ci):
        pos = where[ci]
        last = unsat.pop()
        if pos < len(unsat):
            unsat[pos] = last
            where[last] = pos
        where[ci] = -1

    def add(ci):
        where[ci] = len(unsat)
        unsat.append(ci)

    def break_count(v):
        """Clauses that v alone is currently keeping satisfied."""
        av, bad = a[v], 0
        for ci, req in occ[v]:
            if av == req and true_count[ci] == 1:
                bad += 1
        return bad

    if trace is not None:
        trace.append(tuple(a))

    flips = 0
    while unsat and flips < max_flips:
        clause = formula[unsat[rng.randrange(len(unsat))]]
        cvars = [abs(lit) - 1 for lit in clause]
        breaks = [break_count(v) for v in cvars]

        best = min(breaks)
        if best == 0:                                    # freebie move
            choices = [v for v, b in zip(cvars, breaks) if b == 0]
            v = choices[rng.randrange(len(choices))]
        elif rng.random() < noise:                       # random walk
            v = cvars[rng.randrange(len(cvars))]
        else:                                            # greedy
            v = cvars[breaks.index(best)]

        new = 1 - a[v]
        a[v] = new
        for ci, req in occ[v]:
            if new == req:
                true_count[ci] += 1
                if true_count[ci] == 1:
                    drop(ci)
            else:
                true_count[ci] -= 1
                if true_count[ci] == 0:
                    add(ci)
        flips += 1
        if trace is not None:
            trace.append(tuple(a))

    return (not unsat), tuple(a), flips, len(unsat)


def walksat(formula, n, max_flips=10_000, restarts=1, noise=0.5, rng=random):
    """WalkSAT with restarts.  Returns a :class:`SolveOutcome`.

    ``max_flips`` bounds *each* attempt, so the total work is bounded by
    ``restarts * max_flips``.  Note this is a one-sided procedure: failure is
    evidence of an UNSAT (or very hard) instance, never a proof.
    """
    work = 0
    best = None
    for attempt in range(1, restarts + 1):
        solved, a, flips, unsat = walksat_once(formula, n, max_flips, noise, rng)
        work += flips
        if best is None or unsat < best[1]:
            best = (a, unsat)
        if solved:
            return SolveOutcome(True, flips, attempt, work, a, 0)
    return SolveOutcome(False, max_flips, restarts, work, best[0], best[1])


class WalkSATSolver:
    """Callable solver front-end — the baseline plug for the hardness sweep."""

    name = "walksat"
    unit = "flips"

    def __init__(self, max_flips=10_000, restarts=1, noise=0.5, seed=0):
        self.max_flips = int(max_flips)
        self.restarts = int(restarts)
        self.noise = float(noise)
        self.rng = random.Random(seed)

    def __call__(self, formula, n):
        return walksat(formula, n, self.max_flips, self.restarts, self.noise, self.rng)

    def describe(self):
        return (f"walksat  max_flips={self.max_flips}  restarts={self.restarts}  "
                f"noise={self.noise}")


# ── clause tensors + differentiable satisfaction energy ──────────────────────

def clause_tensors(formulas, device=None):
    """Batch of formulas -> ``(var_idx, sign)``, both (B, m, k).

    ``var_idx`` is the 0-based variable of each literal; ``sign`` is +1 for a
    positive literal and -1 for a negated one.  A literal is satisfied iff
    ``sign * assignment_value > 0`` with the +-1 disc convention, which is what
    makes the whole energy a couple of elementwise ops.
    """
    idx = np.array([[[abs(lit) - 1 for lit in c] for c in f] for f in formulas], dtype=np.int64)
    sgn = np.array([[[1.0 if lit > 0 else -1.0 for lit in c] for c in f] for f in formulas],
                   dtype=np.float32)
    var_idx = torch.from_numpy(idx)
    sign = torch.from_numpy(sgn)
    if device is not None:
        var_idx, sign = var_idx.to(device), sign.to(device)
    return var_idx, sign


def read_discs(frames, masks):
    """Mask-average the discs.  frames (..., H, W), masks (D, H, W) -> (..., D)."""
    totals = masks.sum((-1, -2)).clamp(min=EPS)
    return torch.einsum("...hw,dhw->...d", frames, masks) / totals


def literal_values(a, var_idx, sign):
    """Signed literal values, (B, ..., n) + (B, m, k) -> (B, ..., m, k)."""
    B, n = a.shape[0], a.shape[-1]
    lead = a.shape[1:-1]
    m, k = var_idx.shape[1], var_idx.shape[2]
    flat = a.reshape(B, -1, n)                                   # (B, L, n)
    L = flat.shape[1]
    idx = var_idx.reshape(B, 1, m * k).expand(B, L, m * k)
    vals = flat.gather(2, idx).reshape(B, L, m, k)
    lit = vals * sign.unsqueeze(1)
    return lit.reshape(B, *lead, m, k)


def clause_satisfaction(a, var_idx, sign, beta=4.0):
    """Soft per-clause satisfaction in (0, 1).  Returns (B, ..., m).

    Literal:  sigmoid(beta * sign * a)  — 0.5 at an undecided disc, saturating
    as the disc commits to +-1.  Clause: probabilistic OR, 1 - prod(1 - s),
    which is the independent-bit relaxation of the disjunction.
    """
    lit = torch.sigmoid(beta * literal_values(a, var_idx, sign))
    return 1.0 - torch.prod(1.0 - lit, dim=-1)


def hard_unsat(a, var_idx, sign):
    """Number of clauses failed by the *thresholded* assignment.  (B, ...)."""
    lit = literal_values(a, var_idx, sign) > 0
    return (~lit.any(dim=-1)).sum(dim=-1)


AGGREGATES = ("logprod", "softmin", "mean", "min")


def aggregate_energy(clause_s, aggregate="logprod", tau=8.0):
    """Clause satisfactions (B, ..., m) -> an energy to MINIMISE, (B, ...).

    ``logprod`` (default): mean_i -log s_i.  The formula is an AND, so the
        relaxation should be a product, and in the log domain a product is a
        sum.  Dividing by m keeps the scale independent of instance size.  At
        committed +-1 discs this equals the unsatisfied-clause count over m —
        the WalkSAT objective itself; at undecided discs its gradient on a
        clause is 1/(m s_i), so a collapsing clause pulls without bound.
    ``softmin``: 1 - softmin_i s_i via log-sum-exp; a smooth "worst clause"
        objective.  Sharpest focus (0.73 of the energy range goes on the first
        broken clause) but a nearly flat gradient on every other clause.
    ``mean``: 1 - mean_i s_i.  The naive relaxation.  Identical to logprod once
        the discs commit; weaker only in the undecided regime, where its
        gradient per clause is the constant 1/m.
    ``min``: 1 - min_i s_i.  Non-smooth and rank-blind past one clause; kept so
        that failure mode is visible rather than assumed.
    """
    if aggregate == "logprod":
        return -torch.log(clause_s.clamp(min=EPS)).mean(dim=-1)
    if aggregate == "softmin":
        soft_min = -torch.logsumexp(-tau * clause_s, dim=-1) / tau
        soft_min = soft_min + np.log(clause_s.shape[-1]) / tau      # de-bias the sum
        return 1.0 - soft_min
    if aggregate == "mean":
        return 1.0 - clause_s.mean(dim=-1)
    if aggregate == "min":
        return 1.0 - clause_s.amin(dim=-1)
    raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")


def satisfaction_loss(frames, asn_masks, var_idx, sign, beta=4.0,
                      aggregate="logprod", tau=8.0):
    """The target-free objective.  frames (B, ..., H, W) -> (energy, extras).

    ``extras`` carries the soft clause satisfactions, the assignment disc
    values and the *exact* unsatisfied-clause count, so the trainer can log the
    honest number next to the differentiable one.  Never report the energy as
    if it were the solve rate.
    """
    a = read_discs(frames, asn_masks)
    clause_s = clause_satisfaction(a, var_idx, sign, beta)
    energy = aggregate_energy(clause_s, aggregate, tau)
    return energy, {"a": a, "clause_s": clause_s, "unsat": hard_unsat(a, var_idx, sign)}


def decisiveness_penalty(a, margin=1.0):
    """Push the assignment discs out to +-``margin`` so the readout is crisp.

    The energy alone is happy with tiny-but-correctly-signed discs, which decode
    fine but sit right on the threshold and are wrecked by the next update.
    """
    return torch.relu(margin - a.abs()).pow(2).mean()


# ── screens, seeding, batches ────────────────────────────────────────────────

def assignment_union_mask(layout):
    """(H, W) union of the assignment discs — where the answer gets written."""
    return layout.assignment_masks().sum(0).clamp(0, 1)


def seed_noise(inp, mask, scale=0.5, generator=None):
    """Symmetry breaking: fill the blank assignment discs with noise.

    Every restart is a different draw, which is the whole point — a
    deterministic rule on a deterministic seed gives the same basin every time,
    so "restarts to solution" would be meaningless.  This noise is the
    nondeterministic machine's guess, and the NCA's job is to repair it.
    """
    if scale <= 0:
        return inp
    noise = torch.randn(inp.shape, generator=generator, device=inp.device, dtype=inp.dtype)
    return inp + scale * noise * mask


def sample_solve_batch(layout, batch_size, want="unique", rng=random, device=None):
    """A solve-mode batch: blank assignment row in, formula tensors out.

    ``want="unique"`` (for the supervised objective) rejection-samples formulas
    with exactly one solution, so ``tgt`` is well defined — it is slow, and it
    is a biased slice of instance space.  ``want="sat"`` is the right choice for
    the energy objective, which needs no target.

    Returns a dict with ``inp`` (B, H, W) formula-only screens, ``tgt`` (B, H, W)
    fully-solved screens, the clause tensors, and the raw formulas.
    """
    inputs, targets, formulas, sols = [], [], [], []
    for _ in range(batch_size):
        formula, s = sample_instance(layout.n, layout.m, layout.k, rng, want=want)
        solution = tuple(int(b) for b in s[rng.randrange(len(s))])
        inp, tgt = layout.screens_for(formula, solution, mode="solve")
        inputs.append(inp)
        targets.append(tgt)
        formulas.append(formula)
        sols.append(solution)

    inp = torch.from_numpy(np.stack(inputs))
    tgt = torch.from_numpy(np.stack(targets))
    if device is not None:
        inp, tgt = inp.to(device), tgt.to(device)
    var_idx, sign = clause_tensors(formulas, device=device)
    return {"inp": inp, "tgt": tgt, "var_idx": var_idx, "sign": sign,
            "formulas": formulas, "solutions": sols}


# ── decoding + exact scoring ─────────────────────────────────────────────────

def decode_assignment(frames, asn_masks):
    """Threshold the assignment discs.  frames (..., H, W) -> (..., n) 0/1 int."""
    return (read_discs(frames, asn_masks) > 0).int()


def exact_solved(bits, formulas):
    """Ground truth, via ``ncpu.sat.evaluate`` — never via the loss.

    ``bits`` is (B, n) or (B, T, n); returns a bool array of the same leading
    shape.
    """
    arr = bits.detach().cpu().numpy()
    flat = arr.reshape(-1, arr.shape[-1])
    B = len(formulas)
    per = flat.shape[0] // B
    out = np.zeros(flat.shape[0], dtype=bool)
    for i in range(flat.shape[0]):
        out[i] = bool(evaluate(formulas[i // per], tuple(int(v) for v in flat[i])))
    return out.reshape(arr.shape[:-1])


def steps_to_solution(rollout_frames, asn_masks, formulas, check_every=1):
    """First rollout step whose decoded assignment actually satisfies.

    ``rollout_frames`` is (B, T, H, W) — the channel-0 sequence.  Returns
    ``(first_step, solved)``: an int array of step indices (-1 where never
    solved) and the bool array of whether it ever solved.  The step index is in
    original rollout units even when ``check_every > 1``.
    """
    frames = rollout_frames[:, ::check_every]
    bits = decode_assignment(frames, asn_masks)                # (B, T', n)
    ok = exact_solved(bits, formulas)                          # (B, T')
    solved = ok.any(axis=1)
    first = np.where(solved, ok.argmax(axis=1) * check_every, -1)
    return first, solved


# ── how much search does this instance size need? ────────────────────────────

_WORK_CACHE = {}


def search_work(n, m, k=3, instances=24, quantile=0.75, max_flips=20_000, seed=0):
    """Flips WalkSAT needs at this instance size — the natural budget unit.

    A rollout budget has to be stated in units of SEARCH, not of steps, or the
    measurement is confounded: flips-to-solution grows with n, so a fixed
    rollout buys fewer and fewer search sweeps as instances get bigger and the
    solver looks like it is failing to extrapolate when it is only running out
    of time.  This returns the ``quantile`` of the flip count over freshly drawn
    satisfiable instances at (n, m), cached per size.
    """
    key = (n, m, k, instances, quantile)
    if key in _WORK_CACHE:
        return _WORK_CACHE[key]
    rng = random.Random(seed * 1000 + n * 31 + m)
    # Uniform instances, not brute-force-filtered ones: the filter is limited to
    # n <= 24 and this has to work at any size.  WalkSAT solving an instance is
    # itself proof that it was satisfiable, so the quantile is taken over the
    # solved ones and unsolved draws are dropped rather than censored in.
    costs = []
    for _ in range(instances):
        formula = random_ksat(n, m, k, rng)
        out = walksat(formula, n, max_flips=max_flips, restarts=1, rng=rng)
        if out.solved:
            costs.append(out.work)
    value = float(np.quantile(costs, quantile)) if costs else float(max_flips)
    _WORK_CACHE[key] = value
    return value


# ── the matched degenerate baseline ──────────────────────────────────────────

class RandomWalkSolver:
    """Visit assignments without searching — the floor an any-frame metric has.

    ``mode="walk"`` flips one variable at a time from a random start, which is
    what an NCA that has learned nothing but to jitter its assignment discs
    does.  ``mode="iid"`` draws independent uniform assignments (a strictly
    stronger baseline for the same visit count, since a walk revisits).

    It reports exactly what ``NCASolver`` reports: ``solved`` is the FINAL
    assignment, and ``meta["any"]`` is the any-frame hit.  Feed it the number of
    distinct assignments a model actually visited and the comparison is honest.
    """

    name = "random"
    unit = "visits"

    def __init__(self, visits=100, restarts=1, mode="walk", seed=0):
        self.visits = int(visits)
        self.restarts = int(restarts)
        self.mode = mode
        self.rng = random.Random(seed)

    def _attempt(self, formula, n):
        a = [self.rng.randint(0, 1) for _ in range(n)]
        seen, first, any_hit = {tuple(a)}, -1, False
        if evaluate(formula, tuple(a)):
            first, any_hit = 0, True
        for t in range(1, self.visits):
            if self.mode == "walk":
                a[self.rng.randrange(n)] ^= 1
            else:
                a = [self.rng.randint(0, 1) for _ in range(n)]
            seen.add(tuple(a))
            if not any_hit and evaluate(formula, tuple(a)):
                first, any_hit = t, True
        final = tuple(a)
        return final, any_hit, first, len(seen)

    def __call__(self, formula, n):
        anys, firsts, distinct = [], [], []
        best = None
        for r in range(1, self.restarts + 1):
            final, any_hit, first, seen = self._attempt(formula, n)
            anys.append(any_hit)
            firsts.append(first)
            distinct.append(seen)
            if best is None:
                best = final
            if evaluate(formula, final):
                meta = {"any": anys, "first": firsts, "distinct": distinct,
                        "final": [True], "stable": [False]}
                return SolveOutcome(True, self.visits, r, r * self.visits, final, 0, meta)
        meta = {"any": anys, "first": firsts, "distinct": distinct,
                "final": [False], "stable": [False]}
        return SolveOutcome(False, self.visits, self.restarts, self.visits * self.restarts,
                            best, count_unsat(formula, best), meta)

    def describe(self):
        return f"random  mode={self.mode}  visits={self.visits}  restarts={self.restarts}"


# ── NCA as a solver ──────────────────────────────────────────────────────────

class UnderDepthError(ValueError):
    """Raised when a rollout is shorter than the task's light cone.

    Below ``min_rollout()`` the answer cannot physically reach the assignment
    discs, so the task is IMPOSSIBLE rather than hard, and any number measured
    there is a statement about the budget, not about the model.
    """


class NCASolver:
    """Run a trained NCA as a pluggable solver, for the same hardness sweep.

    The rule is convolutional, so the layout is rebuilt at the instance's own
    (n, m) — a rule trained at n=6, m=26 is *run* unchanged on a bigger clause
    matrix.  Restarts are batched into one rollout for speed, but costed as if
    they had been run one after another (attempt 1 to exhaustion, then attempt
    2, ...) so the numbers stay comparable with the sequential WalkSAT baseline.

    Three success criteria are recorded, and they are not the same thing:

    ``final``   the assignment decoded from the LAST frame satisfies.  This is
                the answer the solver returns, so it is ``solved`` and it is the
                number to report.  Its baseline is one random draw.
    ``stable``  that assignment has been constant and satisfying for the last
                ``settle_steps`` steps — the solver has actually halted on it.
    ``any``     some checked frame satisfied at some point.  A DIAGNOSTIC only:
                its baseline is a random walk over the same number of distinct
                assignments (see ``RandomWalkSolver``), which approaches 1.0.

    ``distinct`` (assignments visited per attempt) is reported alongside so that
    "it searched" and "it twitched" can be told apart.

    The rollout is consumed in chunks and only decoded bits are kept, so memory
    does not grow with rollout depth.
    """

    name = "nca"
    unit = "steps"

    def __init__(self, nca, rollout=None, rollout_mult=3.0, restarts=1, noise_scale=0.5,
                 check_every=4, chunk=32, device="cpu", layout_kwargs=None, seed=0,
                 input_dims=(0, 1), channels=None, settle_steps=None,
                 allow_under_depth=False, work_instances=16):
        self.nca = nca.eval()
        self.rollout = None if rollout is None else int(rollout)
        self.rollout_mult = rollout_mult          # float, or "auto" (measured work)
        self.restarts = int(restarts)
        self.noise_scale = float(noise_scale)
        self.check_every = int(check_every)
        self.chunk = int(chunk)
        self.device = device
        self.layout_kwargs = dict(layout_kwargs or {})
        self.input_dims = tuple(input_dims)
        self.channels = channels if channels is not None else nca.channels
        self.settle_steps = settle_steps
        self.allow_under_depth = bool(allow_under_depth)
        self.work_instances = int(work_instances)
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self._layouts = {}
        self._warned = set()

    def layout_for(self, n, m):
        key = (n, m)
        if key not in self._layouts:
            self._layouts[key] = SATLayout(n_vars=n, m_clauses=m, **self.layout_kwargs)
        return self._layouts[key]

    def sweeps_for(self, layout):
        """Search sweeps the budget affords — the size-invariant budget unit."""
        if isinstance(self.rollout_mult, str) and self.rollout_mult == "auto":
            work = search_work(layout.n, layout.m, layout.k,
                               instances=self.work_instances)
            return max(2.0, work)
        return float(self.rollout_mult)

    def steps_for(self, layout):
        """Rollout depth: an explicit budget, else sweeps x the light cone.

        ``layout.min_rollout()`` is a distance in PIXELS (a 3x3 kernel moves one
        pixel per step and the lattice pitch is ``layout.step``), and it is the
        time for information to make the trip *once*.  One WalkSAT flip needs
        that trip — assignment down to the clause rows, the verdict back up — so
        the multiplier is literally "how many flips can this rollout afford".
        With ``rollout_mult="auto"`` it is set from the measured flip count at
        this instance size, which is the only way the budget stays matched as
        the instance grows.
        """
        floor = layout.min_rollout()
        if self.rollout is not None:
            if self.rollout < floor and not self.allow_under_depth:
                raise UnderDepthError(
                    f"rollout={self.rollout} is below the light cone "
                    f"min_rollout()={floor} for n={layout.n} m={layout.m}: the answer "
                    f"cannot reach the assignment discs, so this measures the budget, "
                    f"not the model.  Raise it, drop --rollout to derive it, or pass "
                    f"allow_under_depth=True deliberately."
                )
            if self.rollout < 2 * floor and (layout.n, layout.m) not in self._warned:
                self._warned.add((layout.n, layout.m))
                print(f"  WARNING: rollout={self.rollout} is only "
                      f"{self.rollout / floor:.1f}x the light cone at n={layout.n} "
                      f"m={layout.m}; a search needs several sweeps, not one.")
            return self.rollout
        return int(round(self.sweeps_for(layout) * floor))

    def implant(self, seeded, frozen):
        state = torch.zeros(seeded.shape[0], self.channels, seeded.shape[-2],
                            seeded.shape[-1], device=seeded.device)
        for d in self.input_dims:
            state[:, d] = seeded if d == 0 else frozen
        return state

    @torch.no_grad()
    def __call__(self, formula, n):
        m = len(formula)
        layout = self.layout_for(n, m)
        rollout = self.steps_for(layout)
        settle = self.settle_steps if self.settle_steps is not None else layout.min_rollout() // 2
        masks = layout.assignment_masks().to(self.device)
        union = assignment_union_mask(layout).to(self.device)
        R = self.restarts

        blank = torch.from_numpy(layout.input_screen(formula)).to(self.device)
        frozen = blank.expand(R, *blank.shape).contiguous()
        seeded = seed_noise(frozen.clone(), union, self.noise_scale, self.gen)
        state = self.implant(seeded, frozen)

        bits_hist, step_hist = [], []
        done = 0
        while done < rollout:
            k = min(self.chunk, rollout - done)
            seq = self.nca.forward(state, steps=k)
            state = seq[:, -1]
            frames = seq[:, 1:, 0]                                  # (R, k, H, W)
            idx = np.arange(done + 1, done + k + 1)
            sel = idx % self.check_every == 0
            sel[-1] = True                        # always decode the final frame
            bits_hist.append(decode_assignment(frames[:, sel], masks).cpu())
            step_hist.append(idx[sel])
            done += k

        bits = torch.cat(bits_hist, dim=1)                          # (R, T, n)
        steps = np.concatenate(step_hist)                           # (T,)
        ok = exact_solved(bits, [formula] * R)                      # (R, T)

        any_hit = ok.any(axis=1)
        first = np.where(any_hit, steps[ok.argmax(axis=1)], -1)
        final = ok[:, -1].copy()
        arr = bits.numpy()
        distinct = [len(np.unique(arr[r], axis=0)) for r in range(R)]

        # "stable": constant and satisfying over the whole settle window
        stable = np.zeros(R, dtype=bool)
        commit = np.full(R, -1, dtype=np.int64)
        window = steps >= (rollout - settle)
        for r in range(R):
            if not final[r]:
                continue
            stable[r] = bool(ok[r, window].all()
                             and (arr[r, window] == arr[r, -1]).all())
            # first step of the maximal satisfied suffix == when it committed
            bad = np.flatnonzero(~ok[r])
            c = 0 if len(bad) == 0 else int(bad[-1]) + 1
            commit[r] = int(steps[c])

        meta = {"any": any_hit.tolist(), "first": first.tolist(), "final": final.tolist(),
                "stable": stable.tolist(), "distinct": distinct, "rollout": rollout,
                "settle_steps": settle, "checks": int(len(steps)),
                "sweeps": rollout / layout.min_rollout()}

        winners = np.flatnonzero(final)
        if len(winners) == 0:
            best = tuple(int(v) for v in arr[0, -1].tolist())
            return SolveOutcome(False, rollout, R, rollout * R, best,
                                count_unsat(formula, best), meta)

        r = int(winners[0])                                         # sequential winner
        answer = tuple(int(v) for v in arr[r, -1].tolist())
        return SolveOutcome(True, int(commit[r]), r + 1, r * rollout + int(commit[r]),
                            answer, count_unsat(formula, answer), meta)

    def describe(self):
        if self.rollout is not None:
            roll = f"{self.rollout} steps (fixed)"
        elif isinstance(self.rollout_mult, str):
            roll = "auto (measured walksat work) x light cone"
        else:
            roll = f"{self.rollout_mult:g}x light cone"
        return (f"nca  rollout={roll}  restarts={self.restarts}  "
                f"noise={self.noise_scale}  check_every={self.check_every}")


def load_nca_solver(checkpoint, device="cpu", **kwargs):
    """Rebuild an NCA from a ``train_sat_solve.py`` checkpoint as a solver."""
    from ncpu.nca import NeuralCA

    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    nca = NeuralCA(**ck["nca_config"]).to(device)
    nca.load_state_dict(ck["nca"])
    layout = ck.get("layout", {})
    kwargs.setdefault("layout_kwargs", {"k": layout.get("k", 3), "r": layout.get("r", 2)})
    kwargs.setdefault("input_dims", tuple(ck.get("input_dims", (0, 1))))
    return NCASolver(nca, device=device, channels=ck["nca_config"]["channels"], **kwargs)
