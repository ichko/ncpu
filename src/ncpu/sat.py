"""3SAT on a grid — shared substrate for the Karp-reduction exploration.

Why this file exists
--------------------
Everything downstream (verifier, reducer, solver) needs the same three things:
a way to make random 3SAT instances, a ground-truth reference to score against,
and *one* agreed pixel encoding so the pieces can be chained.  Divergent
encodings would make the tracks incomparable, so they all live here.

The encoding: a clause matrix
-----------------------------
A formula is drawn as its factor-graph adjacency matrix — ``n`` variable
columns by ``m`` clause rows::

        v0 v1 v2 v3 v4 v5   | flag
   a:    +  -  +  -  +  -   |        <- assignment row (+1 true / -1 false)
   c0:   +  .  .  -  .  .   |  s0
   c1:   .  +  .  .  .  +   |  s1
   ...
   cm:   .  .  -  +  .  .   |  sm
                            |  V     <- verdict, below the last clause row

* cell (i, j) = +1 if variable j occurs positively in clause i,
                -1 if it occurs negated,
                 0 (background) if it does not occur at all.
* the assignment row holds one +-1 disc per variable.
* the flag column holds one disc per clause: +1 iff that clause is satisfied.
* the verdict disc sits at the BOTTOM of the flag column: +1 iff every clause
  is satisfied.  It is below the clause rows rather than at the top corner for
  a depth reason, not a cosmetic one -- see ``SATLayout.min_rollout``.

The point of this layout is that the whole computation is *local and
directional*: broadcast each assignment down its column, multiply against the
literal sign in-place (satisfied literal <=> product > 0), OR along each clause
row into the flag column, AND down the flag column into the verdict.

Depth is the binding constraint and it is measured in PIXELS, not in lattice
cells: a 3x3 kernel moves information one pixel per step, so the requirement
scales with the lattice pitch ``step``.  ``min_rollout()`` returns
``(n + m + 1) * step`` -- 198 steps at n=6, m=26, r=2, and 99 at r=1.  A rollout
below that makes the task impossible rather than hard.

It is also size-agnostic.  The NCA is convolutional, so a rule learned at
n=6, m=26 can be *run* on a much larger clause matrix unchanged (with the
rollout rescaled to that layout's own ``min_rollout()``) -- instance-size
extrapolation comes free, and is the main thing worth measuring.

Value conventions follow ``ncpu.subleq``: float32 screens, neutral-grey
background at 0.0, a 0/false disc at -1.0 and a 1/true disc at +1.0.

Formula representation
----------------------
DIMACS-style: a formula is a tuple of clauses, a clause is a tuple of ints, and
a literal is ``+(j + 1)`` for variable ``j`` positive, ``-(j + 1)`` for negated.
An assignment is a tuple of ``n`` 0/1 ints indexed by variable.
"""

import random

import numpy as np
import torch

BACKGROUND = 0.0
BIT0_VALUE = -1.0
BIT1_VALUE = +1.0

# Satisfiability threshold for random 3SAT: instances near m/n = 4.267 are the
# hard ones (and are ~50% satisfiable).  Well below it almost everything is
# satisfiable and has many solutions; well above it almost everything is UNSAT.
PHASE_RATIO = 4.267


# ── instance generation ──────────────────────────────────────────────────────

def random_ksat(n, m, k=3, rng=random):
    """A uniform random k-SAT formula: ``m`` clauses over ``n`` variables.

    Each clause draws ``k`` *distinct* variables (so no clause is a tautology)
    with independent random signs; duplicate clauses are rejected.
    """
    if n < k:
        raise ValueError(f"need n >= k, got n={n} k={k}")
    seen = set()
    clauses = []
    while len(clauses) < m:
        variables = rng.sample(range(n), k)
        lits = [(v + 1) if rng.random() < 0.5 else -(v + 1) for v in variables]
        # Deduplicate on a canonical form, but STORE the literals unsorted.
        # Sorting signed ints would put the most-negative literal in slot 0 and
        # the most-positive in slot k-1, so slot index would carry sign
        # information. Invisible to any consumer that places a literal by its
        # variable (the clause matrix), but Track B's reduction indexes nodes by
        # SLOT, where it skewed per-cell edge rates from 0.789 to 0.980 and
        # handed a model a shortcut. rng.sample already returns its subset in
        # random order, so the stored tuple needs no further shuffling.
        key = tuple(sorted(lits))
        if key in seen:
            continue
        seen.add(key)
        clauses.append(tuple(lits))
    return tuple(clauses)


def planted_ksat(n, m, k=3, rng=random, assignment=None):
    """A random k-SAT formula guaranteed satisfiable by ``assignment``.

    Clauses are drawn uniformly but rejected unless the planted assignment
    satisfies them.  Note the planted distribution is *not* the uniform one —
    planted instances are easier than uniform instances at the same ratio —
    so use it for curriculum/warm-up, not for hardness claims.
    """
    if assignment is None:
        assignment = tuple(rng.randint(0, 1) for _ in range(n))
    seen = set()
    clauses = []
    while len(clauses) < m:
        variables = rng.sample(range(n), k)
        lits = [(v + 1) if rng.random() < 0.5 else -(v + 1) for v in variables]
        key = tuple(sorted(lits))          # dedup key only; see random_ksat
        if key in seen or not _clause_sat(lits, assignment):
            continue
        seen.add(key)
        clauses.append(tuple(lits))
    return tuple(clauses), tuple(assignment)


def _clause_sat(clause, assignment):
    for lit in clause:
        value = assignment[abs(lit) - 1]
        if (lit > 0 and value) or (lit < 0 and not value):
            return True
    return False


# ── reference semantics ──────────────────────────────────────────────────────

def clause_flags(formula, assignment):
    """Per-clause satisfaction, as 0/1 ints — the ground truth flag column."""
    return [1 if _clause_sat(c, assignment) else 0 for c in formula]


def evaluate(formula, assignment):
    """1 iff every clause is satisfied — the ground truth verdict."""
    return 1 if all(_clause_sat(c, assignment) for c in formula) else 0


def solutions(formula, n, cap=None):
    """All satisfying assignments, brute-forced.  Returns (S, n) uint8.

    Vectorised over the whole 2**n cube, so keep n <= 22 or so.  ``cap`` bounds
    the number of rows returned (the count itself is still exact).
    """
    if n > 24:
        raise ValueError(f"brute force refuses n={n} (2**n rows)")
    idx = np.arange(1 << n, dtype=np.int64)
    bits = np.empty((1 << n, n), dtype=bool)
    for v in range(n):
        bits[:, v] = (idx >> (n - 1 - v)) & 1
    ok = np.ones(1 << n, dtype=bool)
    for clause in formula:
        sat = np.zeros(1 << n, dtype=bool)
        for lit in clause:
            column = bits[:, abs(lit) - 1]
            sat |= column if lit > 0 else ~column
        ok &= sat
        if not ok.any():
            return np.zeros((0, n), dtype=np.uint8)
    found = bits[ok].astype(np.uint8)
    return found if cap is None else found[:cap]


def n_solutions(formula, n):
    return len(solutions(formula, n))


def sample_instance(n, m, k=3, rng=random, want="any", max_tries=20_000):
    """Rejection-sample a formula with a required solution structure.

    ``want`` is one of:
        "any"     no constraint
        "sat"     at least one satisfying assignment
        "unsat"   no satisfying assignment
        "unique"  exactly one satisfying assignment

    Returns ``(formula, sols)`` where ``sols`` is the (S, n) solution array.
    "unique" is the one to train a *solver* on: it removes the target ambiguity
    that would otherwise make a per-pixel loss incoherent (see docs/karp_notes).
    """
    for _ in range(max_tries):
        formula = random_ksat(n, m, k, rng)
        if want == "any":
            return formula, solutions(formula, n, cap=1)
        sols = solutions(formula, n)
        if want == "sat" and len(sols) >= 1:
            return formula, sols
        if want == "unsat" and len(sols) == 0:
            return formula, sols
        if want == "unique" and len(sols) == 1:
            return formula, sols
    raise RuntimeError(
        f"no want={want!r} instance in {max_tries} tries at n={n} m={m} "
        f"(ratio {m / n:.2f}); move the ratio toward {PHASE_RATIO} for 'unique'/'sat'"
    )


# ── geometry + rendering ─────────────────────────────────────────────────────

class SATLayout:
    """Clause-matrix screens for a fixed (n_vars, m_clauses) instance shape.

    The geometry is a plain lattice, so the same rule transfers to a larger
    lattice unchanged — build a second layout with bigger n/m to extrapolate.
    """

    def __init__(self, n_vars=6, m_clauses=12, k=3, r=2, gap=None, pad=None):
        self.n = n_vars
        self.m = m_clauses
        self.k = k
        self.r = int(r)
        self.gap = int(gap) if gap is not None else self.r        # half a disc
        self.pad = int(pad) if pad is not None else 2 * self.r    # one disc
        self.step = 2 * self.r + self.gap

        # columns: n variable columns, then the flag column
        x0 = self.pad + self.r
        self.var_x = [x0 + j * self.step for j in range(self.n)]
        self.flag_x = x0 + self.n * self.step

        # rows: the assignment row, then m clause rows, then the verdict
        y0 = self.pad + self.r
        self.asn_y = y0
        self.clause_y = [y0 + (i + 1) * self.step for i in range(self.m)]

        # The verdict sits at the BOTTOM of the flag column, not at the top
        # corner. Depth, not aesthetics, decides this: with the verdict on top
        # every flag has to travel the full height back up to it, adding an
        # m*step leg to a critical path that is already m*step + n*step. Putting
        # it below the last clause row lets the AND accumulate downward
        # alongside the flags as they are produced, cutting required depth by
        # ~43% (348 -> 198 steps at n=6, m=26, r=2). See min_rollout().
        self.verdict_y = self.clause_y[-1] + self.step

        self.W = self.flag_x + self.r + self.pad
        self.H = self.verdict_y + self.r + self.pad

        self.asn_centers = [(x, self.asn_y) for x in self.var_x]
        self.flag_centers = [(self.flag_x, y) for y in self.clause_y]
        self.verdict_center = (self.flag_x, self.verdict_y)

    # ── primitives ───────────────────────────────────────────────────────────

    def _disc(self, screen, center, value):
        cx, cy = center
        ys, xs = np.ogrid[: self.H, : self.W]
        screen[(xs - cx) ** 2 + (ys - cy) ** 2 <= self.r ** 2] = value

    def _mask(self, center):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        self._disc(m, center, 1.0)
        return m

    def _draw_formula(self, screen, formula):
        """Stamp the literal signs of every clause row."""
        for i, clause in enumerate(formula):
            for lit in clause:
                center = (self.var_x[abs(lit) - 1], self.clause_y[i])
                self._disc(screen, center, BIT1_VALUE if lit > 0 else BIT0_VALUE)

    def _draw_assignment(self, screen, assignment):
        for j, value in enumerate(assignment):
            self._disc(screen, self.asn_centers[j], BIT1_VALUE if value else BIT0_VALUE)

    # ── screens ──────────────────────────────────────────────────────────────

    def input_screen(self, formula, assignment=None):
        """Formula always drawn; assignment drawn only when it is *given*.

        ``assignment=None`` is the solver's input: the assignment row is left at
        background for the NCA to fill in.
        """
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        self._draw_formula(s, formula)
        if assignment is not None:
            self._draw_assignment(s, assignment)
        return s

    def target_screen(self, formula, assignment):
        """Input reproduced, plus the flag column and the verdict filled in."""
        s = self.input_screen(formula, assignment)
        for i, flag in enumerate(clause_flags(formula, assignment)):
            self._disc(s, self.flag_centers[i], BIT1_VALUE if flag else BIT0_VALUE)
        verdict = evaluate(formula, assignment)
        self._disc(s, self.verdict_center, BIT1_VALUE if verdict else BIT0_VALUE)
        return s

    def screens_for(self, formula, assignment, mode="verify"):
        """``mode="verify"``: assignment is given, flags+verdict are the answer.
        ``mode="solve"``:  assignment row is blank on input and is the answer."""
        given = assignment if mode == "verify" else None
        return self.input_screen(formula, given), self.target_screen(formula, assignment)

    # ── masks + decoding ─────────────────────────────────────────────────────

    def assignment_masks(self):
        return torch.from_numpy(np.stack([self._mask(c) for c in self.asn_centers]))

    def flag_masks(self):
        return torch.from_numpy(np.stack([self._mask(c) for c in self.flag_centers]))

    def verdict_mask(self):
        return torch.from_numpy(self._mask(self.verdict_center)[None])

    def output_bit_masks(self, mode="verify"):
        """Stacked masks over the bits the NCA is responsible for producing.

        verify -> (m + 1, H, W): the flags then the verdict.
        solve  -> (n + m + 1, H, W): the assignment, then the flags, verdict.
        """
        parts = [] if mode == "verify" else [self.assignment_masks()]
        parts += [self.flag_masks(), self.verdict_mask()]
        return torch.cat(parts, dim=0)

    @staticmethod
    def decode_bits(frames, masks):
        """Mask-average each output disc and threshold at 0.

        ``frames`` is (..., H, W); returns 0/1 ints of shape (..., n_masks).
        """
        totals = masks.sum((-1, -2))
        per = (frames.unsqueeze(-3) * masks).sum((-1, -2)) / totals
        return (per > 0).int()

    # ── batches ──────────────────────────────────────────────────────────────


    def _unsat_assignment(self, formula, solution, rng, near_miss_frac):
        """An assignment that genuinely FAILS the formula.

        Flipping one variable of a solution does not reliably break it: below
        the phase ratio a formula has many solutions, so a single-flip
        neighbour is often another solution and the "negative" is silently
        positive. That is how a nominally 50/50 batch drifts to 62% positive.
        So we search rather than assume, preferring near-miss negatives (one
        flip, so exactly one clause tends to fail) because they are the ones
        that force the verdict to depend on the AND rather than on a glance.
        """
        if rng.random() < near_miss_frac:
            for j in rng.sample(range(self.n), self.n):
                candidate = list(solution)
                candidate[j] ^= 1
                candidate = tuple(candidate)
                if not evaluate(formula, candidate):
                    return candidate
        for _ in range(64):
            candidate = tuple(rng.randint(0, 1) for _ in range(self.n))
            if not evaluate(formula, candidate):
                return candidate
        return solution      # near-tautological formula; label stays correct

    def sample_batch(self, batch_size, mode="verify", device=None, rng=random,
                     sat_frac=0.5, near_miss_frac=0.5):
        """A batch of (input, target) screens plus the ground-truth bits.

        The balance arguments matter more than they look.  At the phase ratio a
        *uniform random* assignment satisfies all m clauses with probability
        ~(7/8)**m, i.e. essentially never — so an unbalanced verify batch is
        solved perfectly by the constant rule "output UNSAT".  Two corrections:

        ``sat_frac``       fraction of samples whose assignment does satisfy
                           the formula (drawn from the true solution set).
        ``near_miss_frac`` of the remaining (unsatisfying) samples, the
                           fraction built by flipping ONE variable of a real
                           solution rather than drawing at random.  Those are
                           the informative negatives: exactly one clause fails,
                           so the verdict genuinely depends on the AND.

        Returns ``(inp, tgt, bits, meta)``; ``bits`` is (B, n_masks) int, and
        ``meta`` is the list of (formula, assignment) pairs for exact scoring.
        """
        masks = self.output_bit_masks(mode)
        inputs, targets, meta = [], [], []
        for _ in range(batch_size):
            want = "unique" if mode == "solve" else "sat"
            formula, sols = sample_instance(self.n, self.m, self.k, rng, want=want)
            solution = tuple(int(b) for b in sols[rng.randrange(len(sols))])

            if mode == "solve" or rng.random() < sat_frac:
                assignment = solution
            else:
                assignment = self._unsat_assignment(
                    formula, solution, rng, near_miss_frac
                )

            inp, tgt = self.screens_for(formula, assignment, mode)
            inputs.append(inp)
            targets.append(tgt)
            meta.append((formula, assignment))

        inp = torch.from_numpy(np.stack(inputs))
        tgt = torch.from_numpy(np.stack(targets))
        bits = self.decode_bits(tgt, masks)
        if device is not None:
            inp, tgt, bits = inp.to(device), tgt.to(device), bits.to(device)
        return inp, tgt, bits, meta

    def describe(self):
        return (f"SATLayout {self.W}x{self.H}  n={self.n} vars  m={self.m} clauses  "
                f"k={self.k}  ratio={self.m / self.n:.2f}  r={self.r} step={self.step}  "
                f"light cone ~{self.min_rollout()} px steps")

    def min_rollout(self):
        """Steps before the verdict disc can possibly be correct.

        A 3x3 kernel moves information ONE PIXEL per step, so this is a distance
        in pixels, not in lattice cells -- it scales with ``step``.

        The legs ADD rather than max, because they are sequentially dependent:
        the clause OR cannot begin before the assignment value has arrived, and
        the verdict AND cannot begin before the flags have. With the verdict at
        the bottom of the flag column there are two concurrent critical paths,
        and they happen to cost the same::

            last clause    assignment down to row m-1   m * step
                         + across to the flag column    n * step
                         + down one row to the verdict  1 * step

            first clause   assignment down to row 0     1 * step
                         + across to the flag column    n * step
                         + flag down the whole column   m * step

        so the bound is ``(n + m + 1) * step`` either way.

        Any rollout below this makes the task *impossible*, not hard, and must
        never be logged as a learning failure.
        """
        return (self.n + self.m + 1) * self.step
