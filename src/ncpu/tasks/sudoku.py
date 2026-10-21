"""Sudoku as a native relaxation task — Track S of the solving phase.

Why Sudoku is here
------------------
It is the control for one specific question: **does native relaxation survive
constraints that are not radius-1?**

3SAT had to have a layout imposed on it (its factor graph is not planar) and
paid ``(n + m + 1) * step`` of depth for the privilege.  Wang tiling is the
opposite extreme: every constraint is between touching neighbours, so a 3x3
kernel sees a whole constraint at once.  Sudoku sits between the two and is the
only one of the three that is *grid-native yet not neighbour-local*.  Nothing
has to be laid out — the puzzle is already a 2D array — but a cell must differ
from 20 peers, 16 of which are not in its 3x3 neighbourhood and up to 8 cells
away.  So the constraint graph is planar-free-of-charge and long-range at the
same time, and ``min_rollout`` ends up governed by *how far a constraint has to
travel*, not by the grid diameter (see ``SudokuTask.min_rollout``).

The encoding: one belief plane per digit
----------------------------------------
State is ``(n_planes, N, N)`` for an ``N x N`` puzzle (``N = n*n``), **one pixel
per Sudoku cell**::

    planes 0 .. N-1   digit beliefs.  plane d at cell (r, c) is how much the
                      cell believes it holds digit d+1:
                          +1.0  "it is this digit"        (pencil mark circled)
                          -1.0  "it is not this digit"    (pencil mark crossed)
                           0.0  undecided                 (mark still open)
    plane N           given/confidence seed: +1 at a clue, -1 at a free cell
    plane N+1         box phase x: 2*(c % n)/(n-1) - 1
    plane N+2         box phase y: 2*(r % n)/(n-1) - 1

A cell's channel vector *is* its pencil-mark set, so "belief, not commitment" is
the literal representation rather than something bolted on: a half-eliminated
cell is a real state of the encoding, not an artefact of a decoder.  The values
follow the house convention of ``ncpu.sat`` / ``ncpu.subleq`` (background 0.0,
false -1.0, true +1.0), so BACKGROUND already means "all digits still possible"
— the empty-cell seed is the zero screen, for free.

Note the beliefs are **not normalised**.  They are per-digit possibilities, the
way a survey-propagation message is, not a categorical distribution; a cell may
transiently believe two digits or none.  ``energy`` normalises with a softmax
only to score, never to constrain the state.

Why not a scalar digit per cell
-------------------------------
The obvious cheaper encoding is one channel holding the digit value, e.g.
``2*(d-1)/(N-1) - 1``.  It was rejected for three reasons, in order of weight:

1. **It has no way to be undecided.**  Every scalar in [-1, 1] decodes to some
   digit, so every intermediate state of the rollout is a full commitment to a
   grid.  The premise of this phase is that beliefs reconcile *before* anything
   commits; a scalar encoding forces commitment at every step and turns
   relaxation back into local search with a decoder attached.
2. **It makes an arbitrary metric on the digits.**  Digits are a set with no
   order, but a scalar channel makes 3 sit between 2 and 4, so an MSE-shaped
   pressure that cannot decide between 2 and 4 relaxes to 3 — a confidently
   wrong answer.  On one-hot planes the same ambiguity is a genuine tie between
   two planes and decodes as a coin flip, which is honest.
3. **Constraint arithmetic gets easier, not harder, with planes.**  "No two
   peers share a digit" is, per plane, "at most one +1 in this line" — a
   linear, per-plane statement a convolution can express directly.  On a scalar
   it is "no two peers are equal", which needs a difference and a comparison per
   pair.
   The cost is real (N times the channels, and N=9 planes plus the NCA's own
   hidden channels) but 9x9x12 is 972 floats, which is nothing next to the 3SAT
   verifier's 44x44 screens.

Box structure needs position, and that is not optional
------------------------------------------------------
A convolutional rule is translation-invariant; the box partition is not.
Without the two phase planes a cell cannot tell whether its right-hand neighbour
is in its own box or the next one, so the box constraint is *unlearnable* rather
than hard.  Three levels per axis (n=3) is the minimum that distinguishes the
left/middle/right position within a box, and it generalises: at n=4 the phase
planes carry four levels.  They cost 2 planes against a one-hot's 2n.

Room for the stiffening dynamic
-------------------------------
Decimation without an outer loop means a cell that has settled must start
*resisting* change instead of being frozen by Python.  Plane N is that hook: it
seeds at +1 on the clues and -1 on free cells, which is exactly "confidence in
my current belief".  Hold it read-only and it is a plain given-indicator; let
the rule write to it and it becomes the learned stiffening channel — the
expected dynamic is that a cell whose belief stops moving drives its confidence
up, and a high-confidence neighbour's message outweighs a low-confidence one, so
deductions crystallise outward from the clues without anything being frozen from
outside.  The trainer (Track M) owns whether it is writable; the encoding leaves
the room either way.

Clamping, and why it is audited separately from holding
-------------------------------------------------------
``clamp_mask`` marks *only* the digit planes of clue cells.  That is the set
``relax.givens_preserved`` audits, and it must contain nothing that is trivially
preserved, or the audit reports a number about the wrong thing.  The three
structural planes (given/confidence seed and the two box phases) must also be
held fixed during a rollout, but they are constants of the instance and would
score 1.0 in any audit, so they live in ``structure_mask`` instead.
``hold_mask`` is the union, which is what a trainer should actually re-implant
each step.  ``clamp_mask | structure_mask`` being two different things is the
one place this task does not fit ``relax.RelaxationTask`` cleanly.

Solutions are the zero set of the energy
----------------------------------------
Under ``p = softmax(beta * digit_planes)`` the energy pools three residual
families:

* **unit-digit**: for each of the 3N units and each digit, ``(sum_cells p - 1)^2``
  — the permutation-matrix relaxation.  It says both "no repeats" and "every
  digit appears" in one term.
* **purity**: per cell ``1 - sum_d p_d^2``.  This one is not decoration.  The
  unit-digit family alone is **exactly zero at the uniform state** (every cell
  believing every digit at 1/N makes every unit sum to exactly 1), so without a
  purity term the flattest possible non-answer is a global minimum.  That is the
  trap this encoding walks into by default, and it is measured in
  ``scripts/preview_sudoku.py --check``.
* **givens**: ``(1 - p_given)^2`` at clue cells, so the energy's zero set is the
  solution *of this puzzle* rather than of any puzzle with the same shape.

Zero on all three iff every cell is one-hot, every unit is a permutation, and
the clues are respected — i.e. iff the grid is the solution.  With a uniquely
solvable puzzle that infimum is attained (in the commitment limit) at exactly
one point, which is what makes the solution a fixed point and everything else
not.

The aggregate is a **smooth max**, ``(1/tau) * log mean_k exp(tau * e_k)``, not a
mean.  A mean over 324 residuals lets the model bank 320 easy cells and pay a
0.3% penalty for the four it cannot do; the smooth max puts most of the gradient
on the worst residual while leaving every other residual a nonzero share
(``softmin`` in ``ncpu.satsolve`` was rejected as the default for the mirror
reason: nearly flat gradients everywhere but the worst).  ``mean``, ``max`` and
``pnorm`` are kept as ablations so the difference stays measurable.

Instances
---------
Puzzles are carved from a random complete grid and are **proper**: exactly one
solution, verified by an exhaustive counter capped at 2.  Difficulty is the
propagation depth needed, measured by the reference solver, in three classes:

    "naked_single"   repeated "this cell has one candidate left" finishes it
    "hidden_single"  also needs "this digit fits one cell in this unit"
    "search"         propagation stalls; the solver must guess and backtrack

The first class is the trap this track has to avoid handing itself: a puzzle
solvable by naked singles alone needs no search at all, and a model scoring well
on those has demonstrated a local elimination rule, not a solver.  Measured over
400 minimally-carved 9x9 puzzles (``preview_sudoku.py --check --survey``, seed
1234, 77 ms per puzzle):

    class          share   deduction depth        search nodes    clues
    naked_single    1.5%   med 19  p90 23         0               med 25
    hidden_single  44.5%   med 23  p90 27         0               med 24
    search         54.0%   med 28  p90 35 max 45  med 3 p90 10 max 63   med 24

So minimal carving is already mostly non-trivial — but 1.5% of it needs no
search whatsoever, and that slice has to be excluded rather than averaged in.
``difficulty="search"`` is the honest default for a *solver* claim (its
reference work is strictly positive by construction); ``"hidden_single"`` is
the curriculum rung below it, and ``"naked_single"`` exists to be the negative
control — a model that scores on it has shown a local elimination rule and
nothing more.  Asking for a class carves under that class's constraint, which
costs 70 / 86 / 122 ms per puzzle respectively.

Generation costs ~40-200 ms per puzzle, far too slow for a per-step trainer
call, so ``sample`` draws from a pool of base puzzles and applies a random
element of the Sudoku symmetry group (digit relabelling, band/stack and
within-band row/column permutations, transposition).  Those are automorphisms of
the constraint structure, so the difficulty *class* is exactly preserved while
the pixels change completely (so is the deduction depth, except in the search
class, where it is a property of the accepted search path — see ``augment``).
They are *not* new instances: held-out evaluation must call
``sample(..., fresh=True)`` with its own RNG.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from ncpu.relax import BACKGROUND, HIGH_VALUE, LOW_VALUE, RelaxationTask

CLASSES = ("naked_single", "hidden_single", "search")

# Deduction depth (rounds of SYNCHRONOUS propagation on the accepted solve
# path) per difficulty class: the p90 of 60 puzzles carved under each class's
# constraint, plus the p90 over 400 unconstrained minimal carves for "any".
# min_rollout() is derived from these, so re-measure with
# `preview_sudoku.py --check --survey` whenever the generator changes.
DEPTH_P90 = {"naked_single": 24, "hidden_single": 26, "search": 38, "any": 33}

EPS = 1e-9


# ── instance ─────────────────────────────────────────────────────────────────

@dataclass
class SudokuInstance:
    """One proper puzzle, with the difficulty the reference solver measured.

    ``puzzle`` and ``solution`` are (N, N) uint8; 0 means empty in ``puzzle``.
    ``depth`` is the number of *synchronous* deduction rounds on the accepted
    solve path (see ``_solve``), which is what ``min_rollout`` is built on;
    ``nodes`` is the number of search branches taken, which is 0 for anything
    propagation alone can finish.
    """

    n: int
    puzzle: np.ndarray
    solution: np.ndarray
    difficulty: str = "any"
    depth: int = 0
    nodes: int = 0

    @property
    def N(self) -> int:
        return self.n * self.n

    @property
    def n_clues(self) -> int:
        return int((self.puzzle > 0).sum())

    def ascii(self, grid=None) -> str:
        """The puzzle (or any grid) as text, with box rules drawn."""
        g = self.puzzle if grid is None else np.asarray(grid)
        n, N = self.n, self.N
        rule = "+".join(["-" * (2 * n + 1)] * n)
        out = []
        for r in range(N):
            if r and r % n == 0:
                out.append(rule)
            cells = []
            for c in range(N):
                if c and c % n == 0:
                    cells.append("|")
                v = int(g[r, c])
                cells.append("." if v == 0 else str(v))
            out.append(" ".join(cells))
        return "\n".join(out)

    def __repr__(self) -> str:
        return (f"SudokuInstance(n={self.n}, clues={self.n_clues}, "
                f"difficulty={self.difficulty!r}, depth={self.depth}, "
                f"nodes={self.nodes})")


# ── bitmask core: units, propagation, search ─────────────────────────────────
#
# A grid is a flat list of N*N ints, 0 = empty.  Candidate sets are bitmasks
# (bit d-1 set means digit d is still possible).  Everything below is plain
# Python because it runs on the CPU during generation only; the tensors start
# at ``SudokuTask.encode``.

_UNIT_CACHE: dict = {}
_PEER_CACHE: dict = {}


def units(n: int):
    """The 3N unit index lists: N rows, then N columns, then N boxes."""
    if n in _UNIT_CACHE:
        return _UNIT_CACHE[n]
    N = n * n
    rows = [[r * N + c for c in range(N)] for r in range(N)]
    cols = [[r * N + c for r in range(N)] for c in range(N)]
    boxes = []
    for br in range(n):
        for bc in range(n):
            boxes.append([(br * n + dr) * N + (bc * n + dc)
                          for dr in range(n) for dc in range(n)])
    out = rows + cols + boxes
    _UNIT_CACHE[n] = out
    return out


def peers(n: int):
    """``peers(n)[i]`` is the sorted list of the 3(N-1)-|overlap| cells that
    cell ``i`` must differ from.  20 peers at n=3, each counted once even though
    a same-box same-row cell shares two units."""
    if n in _PEER_CACHE:
        return _PEER_CACHE[n]
    N = n * n
    sets = [set() for _ in range(N * N)]
    for unit in units(n):
        for i in unit:
            sets[i].update(unit)
    out = [sorted(s - {i}) for i, s in enumerate(sets)]
    _PEER_CACHE[n] = out
    return out


def flat_grid(grid) -> list:
    """Any (N, N) array or flat sequence -> a plain list of ints.

    ``propagate`` mutates its argument in place and wants a list, so every
    public entry point normalises here rather than trusting the caller.
    """
    a = np.asarray(grid, dtype=np.int64).reshape(-1)
    return [int(v) for v in a]


def _used_masks(grid, n):
    N = n * n
    rows, cols, boxes = [0] * N, [0] * N, [0] * N
    for i, v in enumerate(grid):
        if v:
            r, c = divmod(i, N)
            bit = 1 << (v - 1)
            rows[r] |= bit
            cols[c] |= bit
            boxes[(r // n) * n + (c // n)] |= bit
    return rows, cols, boxes


def _consistent(grid, n) -> bool:
    """No digit twice in any unit.  Empty cells ignored."""
    for unit in units(n):
        seen = 0
        for i in unit:
            v = grid[i]
            if v:
                bit = 1 << (v - 1)
                if seen & bit:
                    return False
                seen |= bit
    return True


def propagate(grid, n, hidden=True, max_rounds=1000):
    """Synchronous constraint propagation, in place.  -> (status, rounds).

    ``status`` is "solved", "stalled" or "contradiction".

    Rounds are **synchronous** on purpose: every placement in a round is derived
    from the state at the start of that round, then all are applied at once.
    A sequential sweep (later cells seeing earlier placements) finishes in fewer
    rounds and would flatter the depth measure — but the NCA updates every cell
    in parallel from the previous frame, so the synchronous count is the one
    that maps onto NCA steps, and ``min_rollout`` depends on it.

    ``hidden=False`` restricts the rule set to naked singles, which is what
    separates the trivial difficulty class from the rest.
    """
    N = n * n
    FULL = (1 << N) - 1
    rounds = 0
    while rounds < max_rounds:
        rounds += 1
        rows, cols, boxes = _used_masks(grid, n)
        cand = [0] * (N * N)
        empty = []
        for i, v in enumerate(grid):
            if v:
                continue
            r, c = divmod(i, N)
            m = FULL & ~(rows[r] | cols[c] | boxes[(r // n) * n + (c // n)])
            if m == 0:
                return "contradiction", rounds
            cand[i] = m
            empty.append(i)
        if not empty:
            return "solved", rounds

        placements = {}
        for i in empty:
            m = cand[i]
            if m & (m - 1) == 0:                       # exactly one candidate
                placements[i] = m.bit_length()
        if not placements and hidden:
            for unit in units(n):
                placed = 0
                for i in unit:
                    if grid[i]:
                        placed |= 1 << (grid[i] - 1)
                for d in range(1, N + 1):
                    bit = 1 << (d - 1)
                    if placed & bit:
                        continue
                    slots = [i for i in unit if not grid[i] and (cand[i] & bit)]
                    if len(slots) == 0:
                        return "contradiction", rounds
                    if len(slots) == 1:
                        placements[slots[0]] = d
        if not placements:
            return "stalled", rounds
        for i, d in placements.items():
            grid[i] = d
        if not _consistent(grid, n):
            # two cells in one unit were forced to the same digit this round
            return "contradiction", rounds
    return "stalled", rounds


def _mrv_cell(grid, n):
    """The empty cell with the fewest candidates -> (index, [digits])."""
    N = n * n
    FULL = (1 << N) - 1
    rows, cols, boxes = _used_masks(grid, n)
    best, best_cnt = None, N + 1
    for i, v in enumerate(grid):
        if v:
            continue
        r, c = divmod(i, N)
        m = FULL & ~(rows[r] | cols[c] | boxes[(r // n) * n + (c // n)])
        cnt = bin(m).count("1")
        if cnt < best_cnt:
            best, best_cnt, best_mask = i, cnt, m
            if cnt <= 1:
                break
    if best is None:
        return None, []
    return best, [d for d in range(1, N + 1) if best_mask & (1 << (d - 1))]


class _Budget:
    def __init__(self, limit):
        self.limit = limit
        self.nodes = 0

    def spend(self):
        self.nodes += 1
        return self.nodes <= self.limit


def _solve(grid, n, budget: _Budget, hidden=True):
    """Propagate-and-branch.  -> (solution|None, depth).

    ``depth`` accumulates propagation rounds plus one per guess **along the
    accepted path only** — the number of sequential deduction steps the answer
    actually rests on, which is the quantity ``min_rollout`` multiplies by the
    constraint travel distance.  Work (branches tried, including abandoned
    ones) is counted separately in ``budget.nodes``.
    """
    status, rounds = propagate(grid, n, hidden=hidden)
    if status == "contradiction":
        return None, rounds
    if status == "solved":
        return grid, rounds
    i, digits = _mrv_cell(grid, n)
    if i is None:
        return None, rounds
    for d in digits:
        if not budget.spend():
            return None, rounds
        child = list(grid)
        child[i] = d
        sol, sub = _solve(child, n, budget, hidden=hidden)
        if sol is not None:
            return sol, rounds + 1 + sub
    return None, rounds


def count_solutions(grid, n, cap=2, budget=200_000) -> int:
    """Exact solution count, stopping once ``cap`` have been found.

    This is what makes a puzzle *proper*: ``count_solutions(puzzle, n, cap=2)``
    must be 1.  A carve that skips this check produces puzzles with several
    solutions, and then a per-pixel target is not a function of the input —
    the trap that made the 3SAT solver train on ``want="unique"`` instances.
    """
    b = _Budget(budget)
    total = 0
    stack = [flat_grid(grid)]
    while stack:
        g = stack.pop()
        status, _ = propagate(g, n)
        if status == "contradiction":
            continue
        if status == "solved":
            total += 1
            if total >= cap:
                return total
            continue
        i, digits = _mrv_cell(g, n)
        if i is None:
            continue
        for d in digits:
            if not b.spend():
                return total
            child = list(g)
            child[i] = d
            stack.append(child)
    return total


def classify(grid, n, budget=200_000):
    """Difficulty of a puzzle -> ``(class, depth, nodes)``.

    ``class`` is the *weakest* rule set that finishes it, which is the only
    definition that makes "solvable by naked singles alone" a checkable
    statement.  ``depth`` is the accepted-path deduction depth under that rule
    set (for "search", under full propagation plus branching).
    """
    base = flat_grid(grid)
    g = list(base)
    status, rounds = propagate(g, n, hidden=False)
    if status == "solved":
        return "naked_single", rounds, 0
    g = list(base)
    status, rounds = propagate(g, n, hidden=True)
    if status == "solved":
        return "hidden_single", rounds, 0
    b = _Budget(budget)
    sol, depth = _solve(list(base), n, b)
    if sol is None:
        return "unsolvable", depth, b.nodes
    return "search", depth, b.nodes


# ── instance generation ──────────────────────────────────────────────────────

def random_solution(n, rng=random, tries=200):
    """A uniformly-shuffled complete grid, by randomised backtracking.

    Digit order is reshuffled at every cell, so no digit or position is
    privileged.  (Filling the first row with 1..N and permuting afterwards is
    the usual shortcut; it is fine for the grid distribution but it makes the
    band structure correlated with the digit values, which is exactly the kind
    of statistical shortcut ``docs/karp_eval_protocol.md`` Tier 0 looks for.)
    """
    N = n * n
    FULL = (1 << N) - 1
    for _ in range(tries):
        grid = [0] * (N * N)
        if _fill(grid, 0, n, FULL, rng):
            return np.array(grid, dtype=np.uint8).reshape(N, N)
    raise RuntimeError(f"no complete grid after {tries} tries at n={n}")


def _fill(grid, i, n, FULL, rng):
    N = n * n
    if i == N * N:
        return True
    r, c = divmod(i, N)
    used = 0
    for j in peers(n)[i]:
        if grid[j]:
            used |= 1 << (grid[j] - 1)
    options = [d for d in range(1, N + 1) if not used & (1 << (d - 1))]
    rng.shuffle(options)
    for d in options:
        grid[i] = d
        if _fill(grid, i + 1, n, FULL, rng):
            return True
        grid[i] = 0
    return False


_ALLOWED = {
    "naked_single": ("naked_single",),
    "hidden_single": ("naked_single", "hidden_single"),
    "search": CLASSES,
    "any": CLASSES,
}


def carve(solution, n, rng=random, keep_class="any", symmetric=False):
    """Remove clues while the puzzle stays proper (and stays in ``keep_class``).

    Removal order is random and every removal is checked with
    ``count_solutions(cap=2)``, so the result is a *minimal* puzzle for the
    constraint given: no further clue can come out.

    ``keep_class`` is how the difficulty knob is actually turned.  Carving with
    ``"naked_single"`` refuses any removal that would need a hidden single, so
    it produces the trivial class deliberately (for a curriculum rung, or as a
    negative control).  Carving with ``"any"`` produces the natural minimal
    distribution, which is the one reported in the module docstring.
    ``symmetric`` removes 180-degree pairs, which looks like a newspaper puzzle
    and leaves more clues; it is off by default because it biases difficulty
    down.
    """
    N = n * n
    allowed = _ALLOWED[keep_class]
    grid = flat_grid(solution)
    order = list(range(N * N))
    rng.shuffle(order)
    done = set()
    for i in order:
        if i in done or grid[i] == 0:
            continue
        group = [i, N * N - 1 - i] if symmetric else [i]
        saved = {j: grid[j] for j in group}
        for j in group:
            grid[j] = 0
            done.add(j)
        ok = count_solutions(grid, n, cap=2) == 1
        if ok and keep_class != "any":
            cls, _, _ = classify(grid, n)
            ok = cls in allowed
        if not ok:
            for j, v in saved.items():
                grid[j] = v
    return np.array(grid, dtype=np.uint8).reshape(N, N)


def generate(n=3, rng=random, difficulty="any", max_tries=64, symmetric=False):
    """One proper puzzle of the requested difficulty class.

    ``difficulty="search"`` is rejection-sampled from minimal carves (there is
    no local rule that forces a puzzle to need search), the others are carved
    under their class constraint directly.  Raises if the class does not turn
    up within ``max_tries`` grids.
    """
    if difficulty not in _ALLOWED:
        raise ValueError(f"difficulty must be one of {tuple(_ALLOWED)}, got {difficulty!r}")
    keep = "any" if difficulty in ("any", "search") else difficulty
    for _ in range(max_tries):
        sol = random_solution(n, rng)
        puzzle = carve(sol, n, rng, keep_class=keep, symmetric=symmetric)
        cls, depth, nodes = classify(puzzle, n)
        if difficulty in ("any", cls):
            return SudokuInstance(n=n, puzzle=puzzle, solution=np.asarray(sol, dtype=np.uint8),
                                  difficulty=cls, depth=depth, nodes=nodes)
    raise RuntimeError(
        f"no difficulty={difficulty!r} puzzle in {max_tries} tries at n={n}")


# ── symmetry augmentation ────────────────────────────────────────────────────

def augment(instance, rng=random, recompute=False):
    """A random element of the Sudoku symmetry group applied to an instance.

    Digit relabelling, band and stack permutation, row/column permutation
    within a band or stack, and transposition.  All are automorphisms of the
    constraint hypergraph, so the puzzle stays proper and its **difficulty
    class is exactly preserved**.  At n=3 the group has 9! * 3!^8 * 2 ~ 1.2e12
    elements.

    ``depth`` and ``nodes`` are copied, and that is exact for the two
    propagation-only classes — synchronous propagation is symmetric, so the
    round count cannot change.  It is **not** exact for the "search" class:
    depth and nodes there are properties of the *accepted search path*, and the
    path depends on the solver's tie-breaks (MRV by cell index, digits
    ascending), which a relabelling permutes.  Measured, 44 of 48 augmented
    search instances shift depth while none change class.  Pass
    ``recompute=True`` (~1-2 ms) when the exact numbers matter — for a
    difficulty *report*, say — and leave it off in the training loop, where the
    class is what selects the instance.

    This is what makes per-step training affordable (generation is 70-140 ms,
    an augmentation is ~50 us).  It is not a source of new instances: the orbit
    of a puzzle is one puzzle, so a held-out set must be carved from fresh
    grids.
    """
    n, N = instance.n, instance.N
    perm = list(range(1, N + 1))
    rng.shuffle(perm)
    relabel = np.zeros(N + 1, dtype=np.uint8)     # relabel[0] = 0 keeps empties
    for d, t in enumerate(perm, start=1):
        relabel[d] = t

    bands = list(range(n))
    rng.shuffle(bands)
    rows = []
    for b in bands:
        inner = list(range(n))
        rng.shuffle(inner)
        rows += [b * n + i for i in inner]
    stacks = list(range(n))
    rng.shuffle(stacks)
    cols = []
    for s in stacks:
        inner = list(range(n))
        rng.shuffle(inner)
        cols += [s * n + i for i in inner]

    transpose = rng.random() < 0.5

    def apply(g):
        out = relabel[g][np.ix_(rows, cols)]
        return out.T.copy() if transpose else out.copy()

    puzzle = apply(instance.puzzle)
    cls, depth, nodes = instance.difficulty, instance.depth, instance.nodes
    if recompute:
        cls, depth, nodes = classify(puzzle, n)
    return SudokuInstance(n=n, puzzle=puzzle, solution=apply(instance.solution),
                          difficulty=cls, depth=depth, nodes=nodes)


# ── the task ─────────────────────────────────────────────────────────────────

AGGREGATES = ("softmax", "mean", "max", "pnorm")


class SudokuTask(RelaxationTask):
    """``RelaxationTask`` over N x N Sudoku, one pixel per cell.

    Args:
        n:              box size; the grid is N = n*n on a side.  9x9 is n=3.
        difficulty:     class ``sample`` asks for — see ``CLASSES``, or "any".
        wrap:           does the NCA use circular padding?  It changes the
                        geometry (see ``min_rollout``), not the encoding.
        depth:          override the deduction depth ``min_rollout`` assumes;
                        default is the measured p90 for the difficulty class.
        beta:           belief -> probability sharpness in ``energy``.
        aggregate:      residual aggregate; "softmax" is the smooth max.
        tau:            sharpness of the smooth max.
        purity_weight:  weight on the per-cell one-hot residual.  **Do not set
                        this to 0**: the unit residuals alone are exactly zero
                        at the uniform state.
        given_weight:   weight on the clue residuals.
        pool_size:      base puzzles cached for ``sample``; augmentation
                        supplies the variety.  0 disables the pool (every call
                        carves fresh, ~40-200 ms per instance).
    """

    name = "sudoku"

    def __init__(self, n=3, difficulty="search", wrap=False, depth=None,
                 beta=4.0, aggregate="softmax", tau=8.0, purity_weight=1.0,
                 given_weight=1.0, pool_size=256, symmetric=False):
        if n < 2:
            raise ValueError(f"n >= 2 required (the box phase needs >1 level), got {n}")
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")
        if difficulty not in _ALLOWED:
            raise ValueError(f"difficulty must be one of {tuple(_ALLOWED)}, got {difficulty!r}")
        self.n = n
        self.N = n * n
        self.difficulty = difficulty
        self.wrap = bool(wrap)
        self.depth = int(depth) if depth is not None else DEPTH_P90[difficulty]
        self.beta = float(beta)
        self.aggregate = aggregate
        self.tau = float(tau)
        self.purity_weight = float(purity_weight)
        self.given_weight = float(given_weight)
        self.pool_size = int(pool_size)
        self.symmetric = bool(symmetric)
        self._pool: list = []

    # ── geometry ─────────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple:
        """(H, W) of the rendered grid — one pixel per Sudoku cell."""
        return (self.N, self.N)

    @property
    def n_planes(self) -> int:
        """Channels the encoding occupies: N digit planes + 3 structural."""
        return self.N + 3

    @property
    def digit_planes(self) -> slice:
        return slice(0, self.N)

    @property
    def confidence_plane(self) -> int:
        """Given indicator on the seed; the hook for a learned stiffening
        dynamic if the trainer lets the rule write to it."""
        return self.N

    @property
    def phase_planes(self) -> slice:
        return slice(self.N + 1, self.N + 3)

    def min_rollout(self) -> int:
        """Steps before the answer could physically be right, in PIXELS.

        Cells are pixels and a 3x3 kernel moves information one pixel per step,
        so distance here is the **Chebyshev** distance between two cells.

        What has to travel, and how far:

        * **box constraint**: peers are at Chebyshev distance <= n-1 (2 at
          n=3).  Radius-1-ish; not the binding term.
        * **row / column constraint**: peers are at distance up to N-1 (8 at
          n=3).  This is the term Wang tiling does not have, and it is what
          makes Sudoku the interesting contrast: the constraint graph is
          grid-native, so nothing needs laying out, but a single constraint
          still spans the whole line.
        * **one deduction** costs one traversal of the longest constraint,
          ``L = N - 1``: a naked single at a cell needs the values of all its
          peers to have arrived, and the farthest is L away.  A hidden single
          in a row needs the same.
        * **one repair** costs a round trip, ``2L``.  Two cells in a line both
          claiming digit d cannot see each other; their beliefs meet in the
          middle after L/2, but the resolution ("one of you drop d") has to
          travel back out to them.  A *relaxation* solver spends most of its
          time repairing, not deducing cleanly, so the round trip is the
          honest per-round cost.
        * **rounds are sequential**: a deduction that depends on the result of
          another cannot start earlier.  The reference solver measures how many
          synchronous rounds the puzzle needs (``SudokuInstance.depth``), and
          that number multiplies.

        So ``min_rollout = 2 * (N - 1) * depth``: **608** steps for the search
        class at n=3 (depth p90 38), 416 for hidden singles, 384 for naked
        singles.  **What dominates is the round count, not the grid.**  The 9x9
        grid's diameter is 8 pixels and the requirement is 76x that; halving the
        travel distance would only halve it, while a puzzle one deduction round
        harder costs another 16 steps regardless of size.  Sudoku is not deep
        because it is big — 81 pixels against the 3SAT verifier's 1936 — it is
        deep because deductions chain.  That is the whole difference between a
        long-range constraint graph and a nearest-neighbour one, and it is why
        this track is the one worth comparing against Wang tiling.

        Two honesty notes on the number:

        * The strict *impossibility* floor is only ``L`` (8 steps): below that a
          cell cannot yet have heard from its farthest peer, so nothing above
          chance is physically available.  ``2 * L * depth`` is the **critical
          path of the whole deduction**, which is the sense in which
          ``SATLayout.min_rollout`` used the term (it returned the full
          ``(n+m+1)*step`` dataflow depth, not one hop), and the sense a rollout
          should be sized to.
        * It is an over-estimate per round, not a bound: a given round's
          deduction may depend only on nearby peers, and ``L`` is its worst
          case.  Erring long is deliberate — the 3SAT work found depth sets the
          timescale but not the ceiling (2x later breakthrough for 2.5x depth),
          while under-depth makes the task impossible and reads as a learning
          failure.

        With ``wrap=True`` (circular padding, which is ``NeuralCA``'s default)
        the grid is a torus and the longest in-line distance falls to ``N//2``,
        so the bound roughly halves.  The default is the non-wrapping figure,
        because being over-deep costs training time while being under-depth
        makes the task impossible and reads as a learning failure.

        Puzzles in the "search" class have no honest finite bound at all: once
        propagation stalls the solver must guess, and depth then buys *search*
        rather than *propagation*, exactly as rollout length did for WalkSAT in
        ``ncpu.satsolve``.  The number here is the depth of the accepted path,
        i.e. the cost of the search that succeeds, not of finding it.
        """
        L = (self.N // 2) if self.wrap else (self.N - 1)
        return int(2 * L * self.depth)

    # ── instances ────────────────────────────────────────────────────────────

    def sample(self, batch_size: int, rng, difficulty: Optional[Any] = None,
               fresh: bool = False) -> list:
        """A batch of proper puzzles.

        Draws from a pool of base puzzles and returns a random symmetry image of
        each (see ``augment``), because carving costs ~40-200 ms and a trainer
        calls this every step.  ``fresh=True`` bypasses the pool and carves new
        grids — use it for every held-out evaluation, since an augmented puzzle
        is the *same* puzzle in the only sense that matters.
        """
        want = difficulty or self.difficulty
        out = []
        for _ in range(batch_size):
            if fresh or self.pool_size == 0:
                out.append(generate(self.n, rng, want, symmetric=self.symmetric))
                continue
            if len(self._pool) < self.pool_size:
                self._pool.append(generate(self.n, rng, want, symmetric=self.symmetric))
            base = self._pool[rng.randrange(len(self._pool))]
            out.append(augment(base, rng))
        return out

    # ── rendering ────────────────────────────────────────────────────────────

    def encode(self, instance) -> np.ndarray:
        """The seed state, ``(n_planes, N, N)`` float32.

        Free cells sit at BACKGROUND on every digit plane, which reads as "every
        digit still possible" — the empty pencil-mark grid.  Clue cells are
        fully committed: +1 on their digit, -1 on the other N-1.

        This departs from the letter of ``RelaxationTask.encode`` ("everything
        else at BACKGROUND") in the three structural planes, which carry the
        given indicator and the box phase.  They are constants of the instance,
        not part of the answer, and the box phase is not optional — without it
        a translation-invariant rule cannot locate a box boundary.
        """
        n, N = self.n, self.N
        s = np.full((self.n_planes, N, N), BACKGROUND, dtype=np.float32)
        p = np.asarray(instance.puzzle)
        given = p > 0
        s[self.confidence_plane] = np.where(given, HIGH_VALUE, LOW_VALUE)
        rows, cols = np.nonzero(given)
        if len(rows):
            s[self.digit_planes][:, rows, cols] = LOW_VALUE
            s[p[rows, cols] - 1, rows, cols] = HIGH_VALUE
        idx = np.arange(N, dtype=np.float32)
        phase = 2.0 * (idx % n) / (n - 1) - 1.0
        s[N + 1] = phase[None, :]                       # box phase x (column)
        s[N + 2] = phase[:, None]                       # box phase y (row)
        return s

    def clamp_mask(self, instance) -> np.ndarray:
        """1.0 on the digit planes of clue cells; 0.0 everywhere else.

        Deliberately **only** the clues.  ``relax.givens_preserved`` averages
        over whatever this marks, so putting the structural planes in here —
        which never move, since they are read-only constants — would pad the
        audit with guaranteed passes and hide a real overwrite.  Use
        ``hold_mask`` for what a trainer should re-implant.
        """
        m = np.zeros((self.n_planes, self.N, self.N), dtype=np.float32)
        given = np.asarray(instance.puzzle) > 0
        m[self.digit_planes] = given.astype(np.float32)[None]
        return m

    def structure_mask(self) -> np.ndarray:
        """1.0 on the three structural planes: instance constants, held fixed
        but never audited (they would always pass)."""
        m = np.zeros((self.n_planes, self.N, self.N), dtype=np.float32)
        m[self.N:self.n_planes] = 1.0
        return m

    def hold_mask(self, instance) -> np.ndarray:
        """What the trainer re-implants each step: clues + structure."""
        return np.maximum(self.clamp_mask(instance), self.structure_mask())

    def decode(self, frame: torch.Tensor):
        """Argmax over the digit planes -> an N x N tuple-of-tuples of digits.

        Takes the first ``N`` channels of whatever it is handed, so the trainer
        may pass the full NCA state (extra hidden channels and all) or just the
        encoded planes.

        It does **not** re-impose the clues.  A decoder that pastes the givens
        back in would make the clamp audit unfalsifiable and ``is_solved``
        would then be scoring a puzzle the model never had to respect.
        """
        f = torch.as_tensor(frame)
        if f.dim() != 3:
            raise ValueError(f"decode wants (C, H, W), got {tuple(f.shape)}")
        digits = f[: self.N].argmax(dim=0) + 1
        return tuple(tuple(int(v) for v in row) for row in digits)

    def beliefs(self, frame: torch.Tensor) -> torch.Tensor:
        """Digit planes as per-cell probabilities, (N, H, W).  Diagnostics only
        — nothing in the rollout is allowed to depend on a normalisation."""
        f = torch.as_tensor(frame)[: self.N]
        return torch.softmax(self.beta * f, dim=0)

    # ── exact semantics ──────────────────────────────────────────────────────

    @staticmethod
    def _flat(candidate, N):
        a = np.asarray(candidate, dtype=np.int64).reshape(N, N)
        return a

    def is_solved(self, instance, candidate) -> bool:
        """Ground truth: a complete valid grid that agrees with the clues.

        The clue check is not redundant with the clamp audit.  ``clamp_mask``
        is enforced on the *state*; this is enforced on the *answer*, and a
        solver that finds some other valid grid has solved a different puzzle
        even if no clamped pixel ever moved.
        """
        g = self._flat(candidate, self.N)
        p = np.asarray(instance.puzzle)
        given = p > 0
        if np.any(g[given] != p[given]):
            return False
        if g.min() < 1 or g.max() > self.N:
            return False
        flat = g.reshape(-1)
        for unit in units(self.n):
            if len(set(int(flat[i]) for i in unit)) != self.N:
                return False
        return True

    def violations(self, instance, candidate) -> int:
        """Conflicting peer pairs plus overwritten clues.  0 iff solved.

        Peer *pairs*, counted once each — not per-unit duplicate counts, which
        would charge a same-row same-box collision twice and quietly weight the
        box constraint differently from the line constraint.
        """
        g = self._flat(candidate, self.N).reshape(-1)
        bad = 0
        for i, plist in enumerate(peers(self.n)):
            for j in plist:
                if j > i and g[i] == g[j]:
                    bad += 1
        p = np.asarray(instance.puzzle).reshape(-1)
        bad += int(np.sum((p > 0) & (g != p)))
        return int(bad)

    def random_candidate(self, instance, rng, mode="uniform"):
        """A random candidate that respects the clues.

        ``mode="uniform"`` is the contract's floor: every free cell takes a
        uniform digit, independent of the others.  Be careful reading the
        resulting baseline — with ~55 free cells it is 9**-55, so
        ``relax.matched_baseline`` will return exactly 0.000 for any rollout
        budget that fits on a GPU, and a "lift" over it is not evidence of
        anything.  For Sudoku the informative comparisons are the *violation
        count* against this floor, and the solve rate against ``mode="propagate"``.

        ``mode="propagate"`` is the stronger baseline: fill cells in random
        order, each from the digits still consistent with what is already
        placed, falling back to a uniform digit when a cell paints itself into a
        corner.  That is a random DFS without backtracking — the closest
        non-learned analogue of what a relaxation is doing, and the number an
        NCA solver actually has to beat.
        """
        N = self.N
        g = flat_grid(instance.puzzle)
        free = [i for i, v in enumerate(g) if v == 0]
        if mode == "uniform":
            for i in free:
                g[i] = rng.randrange(1, N + 1)
        elif mode == "propagate":
            rng.shuffle(free)
            FULL = (1 << N) - 1
            for i in free:
                used = 0
                for j in peers(self.n)[i]:
                    if g[j]:
                        used |= 1 << (g[j] - 1)
                m = FULL & ~used
                opts = [d for d in range(1, N + 1) if m & (1 << (d - 1))]
                g[i] = rng.choice(opts) if opts else rng.randrange(1, N + 1)
        else:
            raise ValueError(f"mode must be 'uniform' or 'propagate', got {mode!r}")
        arr = np.array(g, dtype=np.int64).reshape(N, N)
        return tuple(tuple(int(v) for v in row) for row in arr)

    def reference_solve(self, instance, budget: int = 10_000):
        """Constraint propagation + MRV backtracking -> ``(solution, work)``.

        ``work`` is branches taken, which is 0 for anything propagation alone
        finishes — that is the difficulty scale, and the reason
        ``difficulty="naked_single"`` instances are worthless as a solver
        benchmark: their baseline work is zero.  Returns ``(None, work)`` if the
        budget runs out.
        """
        b = _Budget(budget)
        sol, _ = _solve(flat_grid(instance.puzzle), self.n, b)
        if sol is None:
            return None, b.nodes
        arr = np.array(sol, dtype=np.int64).reshape(self.N, self.N)
        return tuple(tuple(int(v) for v in row) for row in arr), b.nodes

    # ── differentiable energy ────────────────────────────────────────────────

    def _given_onehot(self, instance) -> np.ndarray:
        p = np.asarray(instance.puzzle)
        oh = np.zeros((self.N, self.N, self.N), dtype=np.float32)
        r, c = np.nonzero(p > 0)
        if len(r):
            oh[p[r, c] - 1, r, c] = 1.0
        return oh

    def energy_terms(self, state: torch.Tensor, instance):
        """The residual pool behind ``energy``, for diagnostics and ablations.

        Returns ``(residuals, valid, parts)`` with ``residuals`` (B, K) already
        scaled by their family weight, ``valid`` (B, K) a 0/1 mask (clue
        residuals are only defined at clues), and ``parts`` a dict of the three
        families kept separate so a preview can show which one is carrying the
        energy.
        """
        x = torch.as_tensor(state)
        if x.dim() == 3:
            x = x[None]
        if x.dim() != 4:
            raise ValueError(f"energy wants (C, H, W) or (B, C, H, W), got {tuple(x.shape)}")
        B, n, N = x.shape[0], self.n, self.N
        if x.shape[-3] < N:
            raise ValueError(f"need at least {N} digit planes, got {x.shape[-3]}")
        insts = [instance] * B if isinstance(instance, SudokuInstance) else list(instance)
        if len(insts) != B:
            raise ValueError(f"{len(insts)} instances for a batch of {B}")

        p = torch.softmax(self.beta * x[:, :N], dim=1)              # (B, D, H, W)
        oh = torch.from_numpy(np.stack([self._given_onehot(i) for i in insts]))
        oh = oh.to(device=p.device, dtype=p.dtype)

        rows = (p.sum(-1) - 1.0) ** 2                               # (B, D, N)
        cols = (p.sum(-2) - 1.0) ** 2                               # (B, D, N)
        boxes = (p.reshape(B, N, n, n, n, n).sum(dim=(3, 5)) - 1.0) ** 2
        purity = (1.0 - (p ** 2).sum(1)) * self.purity_weight       # (B, H, W)
        given_p = (p * oh).sum(1)                                   # (B, H, W)
        given = ((1.0 - given_p) ** 2) * self.given_weight
        given_valid = oh.sum(1)                                     # 1 at clues

        parts = {"row": rows, "col": cols, "box": boxes,
                 "purity": purity, "given": given}
        res = torch.cat([rows.flatten(1), cols.flatten(1), boxes.flatten(1),
                         purity.flatten(1), given.flatten(1)], dim=1)
        ones = torch.ones_like(rows.flatten(1))
        valid = torch.cat([ones, torch.ones_like(cols.flatten(1)),
                           torch.ones_like(boxes.flatten(1)),
                           torch.ones_like(purity.flatten(1)),
                           given_valid.flatten(1)], dim=1)
        return res, valid, parts

    def _aggregate(self, res, valid):
        """Pool residuals into one energy per instance.

        ``softmax`` (default) is the smooth max
        ``(1/tau) * log( sum_k v_k exp(tau e_k) / sum_k v_k )``.  Its gradient
        weight on residual k is ``softmax(tau e)_k``, so the worst constraint
        takes most of the pull while every other constraint keeps a strictly
        positive share — the property ``mean`` lacks in the direction that
        matters (it lets 320 easy residuals hide four impossible ones) and that
        ``max`` lacks in the other (zero gradient on everything but the worst).
        It interpolates between the two as ``tau`` goes 0 -> inf, so the
        ablation is one number.
        """
        w = valid.sum(dim=1).clamp(min=EPS)
        if self.aggregate == "mean":
            return (res * valid).sum(dim=1) / w
        if self.aggregate == "max":
            return res.masked_fill(valid <= 0, float("-inf")).amax(dim=1)
        if self.aggregate == "pnorm":
            q = 4.0
            return ((res.clamp(min=0) ** q * valid).sum(dim=1) / w) ** (1.0 / q)
        z = (self.tau * res).masked_fill(valid <= 0, float("-inf"))
        return (torch.logsumexp(z, dim=1) - torch.log(w)) / self.tau

    def energy(self, state: torch.Tensor, instance) -> torch.Tensor:
        """Differentiable constraint violation.  (C,H,W) -> (), (B,C,H,W) -> (B,).

        Zero exactly when every cell is one-hot, every unit is a permutation and
        the clues are respected — i.e. only at the solution, which is what makes
        the solution a fixed point of a dynamics that descends it.  It is a
        gradient signal and a diagnostic; it is **never** an accuracy.  Report
        ``is_solved``.
        """
        squeeze = torch.as_tensor(state).dim() == 3
        res, valid, _ = self.energy_terms(state, instance)
        out = self._aggregate(res, valid)
        return out[0] if squeeze else out

    # ── misc ─────────────────────────────────────────────────────────────────

    def describe(self) -> str:
        return (f"{self.name} {self.N}x{self.N} (n={self.n})  planes={self.n_planes}  "
                f"difficulty={self.difficulty}  depth={self.depth}  "
                f"wrap={self.wrap}  min_rollout={self.min_rollout()}")
