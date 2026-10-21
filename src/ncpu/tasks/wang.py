"""Wang tiling as an NCA *relaxation* problem — Track W.

Why Wang tiling
---------------
A Wang tile is a unit square with a colour on each of its four edges; no
rotation, no reflection.  Given a tile set and a region, place a tile in every
cell so that every pair of abutting edges carries the same colour.  Bounded
Wang tiling is NP-complete; the infinite-plane version is undecidable (Berger
1966), which is why aperiodic tile sets exist.

The constraint is **strictly nearest-neighbour**: cell (i, j) must agree with
its four neighbours and with nothing else.  That is radius 1 — exactly the
NCA's kernel — so there is no factor graph to embed, no broadcast, no imposed
layout.  3SAT paid ``(n + m + 1) * step`` in depth purely because its factor
graph is not planar (``docs/3sat_findings.md``); here the *instance is the
grid* and depth is the region diameter, nothing more.

The encoding: a belief simplex per cell, laid out spatially
-----------------------------------------------------------
The constraint is a relation between a cell's tile and its neighbour's tile,
so a cell must carry a **vector**: one belief per tile type.  ``relax.py``
offers ``n_planes``, so that vector lives in CHANNELS, not in space::

    n_planes = T + 1 = 13,   one pixel per tiling cell

        plane t (t < T)    belief in tile t at that cell, in [-1, +1]
                           -1 = ruled out, 0 = no opinion, +1 = it
        plane T            CONFIDENCE K, in [-1, +1]

The pixel at (i, j) therefore *is* the belief distribution over tile types for
cell (i, j) — the survey-propagation reading this phase is after — and the
whole 8x8 region is an 8x8 image.  Free cells start at BACKGROUND (0.0) on
every plane, which *is* the uniform belief, so "unseeded" and "maximally
uncertain" are the same state instead of two states that have to be told
apart.  A given cell is stamped saturated: +1 on its tile's plane, -1 on every
other, +1 on K.

Nothing decodes to a discrete tile during a rollout.  ``decode`` is an argmax
used by the *scorer*; the dynamics only ever see the continuous field.

Why planes and not a block of pixels (this file was rebuilt)
------------------------------------------------------------
The first version put the T beliefs in an ``s x s`` block of disc slots,
because the contract then rendered a single ``(H, W)`` screen and space was
the only place a vector could go.  ``n_planes`` removed that constraint, and
the rebuild is not a cosmetic saving — the spatial layout was fighting the
architecture:

* **A cell could not see its own belief vector.**  ``NeuralCA`` perceives with
  a 3x3 kernel per channel and then applies a **1x1** rule across channels, so
  at one pixel the rule sees every plane at once, for free.  With a 4x4 block
  the twelve beliefs of ONE cell were spread over twelve pixels and a 3x3
  kernel could not read them in a single step: the rule had to spend steps
  gathering its own state before it could begin to compare with a neighbour.
  In planes, "compare my distribution with my four neighbours' under the
  compatibility relation" is one 3x3-then-1x1 step.  That is the entire
  radius-1 claim of this track, and the block layout was throwing it away.
  Legibility — the argument for keeping blocks — is a rendering problem, and
  ``scripts/preview_wang.py`` solves it by plotting the plane stack and the
  decoded tiling side by side.
* **Depth.**  Pitch was ``s * step`` with ``s = ceil(sqrt(T + 1))``, so the
  lattice, and every distance in it, grew as ``sqrt(T)``: a T=40 tile set
  would have cost a 21 px pitch and a proportionally deeper rollout, making
  tile-set size unaffordable as an axis.  In planes the pitch is 1 and a tile
  type costs one channel.
* **Cost.**  The 8x8 region was a 96x96 screen (9,216 values); it is now
  8x8x13 (832), an 11x reduction that widens with T.  That reduction is what
  makes the honest, decimation-length rollout derived in ``min_rollout``
  affordable at all: 504 steps of BPTT over 8x8x13 is ~50 MB at batch 16,
  where the same rollout over the old screen would have been ~7 GB.

What was rejected
-----------------
* **Explicit edge colours (four colour slots per cell).**  Tempting, because
  the constraint is defined on edges, and if the colour is stored once per
  *edge* of the dual lattice then edge agreement is free by construction.  It
  loses on two counts.  (1) The per-cell constraint becomes "these four
  colours must form a tile that exists", a 4-ary disjunction over the tile set
  that does not factorise into pairwise terms; the natural relaxation of it is
  a ``min`` over tiles, which is exactly the flat, rank-blind objective
  ``ncpu.satsolve`` measured and rejected.  (2) Four independent colour
  marginals cannot represent the correlation *between* the four edges of one
  cell — and that correlation is the entire content of "a tile".  A cell that
  believes {north=red, east=blue} and {north=green, east=yellow} equally is
  indistinguishable, in the edge encoding, from one that also admits
  {north=red, east=yellow}.  Beliefs over tiles keep the joint.
* **Edge colours as *auxiliary* published planes** (belief planes plus four
  extra "port" planes carrying this cell's colour marginal on each side, so a
  neighbour reads a port directly).  Good idea, wrong place: as *rendered*
  planes the ports need their own supervision, are audited by
  ``givens_preserved`` when they should not be, and can desynchronise from the
  belief they are supposed to summarise.  The right home is hidden NCA
  channels, which cost nothing to add — and ``side_marginals()`` below
  computes exactly the quantity a port would hold, so the methods track can
  wire them up without touching this file.
* **The tile set drawn on the screen (a legend band).**  The compatibility
  relation would then have to be broadcast from the legend to every cell,
  which re-imports the whole ``(n + m)`` depth apparatus that made 3SAT
  expensive.  Instead the tile set is fixed per ``WangTiling`` object and
  baked into the weights: it is the *physics* the rule learns, and instances
  vary the boundary.  ``WangInstance`` still carries its tile set so that every
  exact check is self-contained, and a second ``WangTiling`` with a fresh tile
  set is the tile-set-generalisation experiment when someone wants it.

Confidence, and decimation without an outer loop
------------------------------------------------
Classic survey propagation freezes the most-confident variable and restarts.
That outer loop is ruled out here, so plane ``K`` is reserved for the
NCA-native replacement: a **learned stiffening dynamic**.  ``K`` is seeded at
0 for free cells and +1 for givens, and the intended reading is

    delta_cell <- (1 - relu(K)) * delta_cell

so a cell that has grown its own confidence resists further change and acts,
locally, like a given — decimation as a fixed point of the dynamics rather
than as a Python loop.  Negative ``K`` is deliberately left meaningful:
"contested", a cell that should stay soft because its neighbours disagree.
This module only *reserves* the plane and seeds it; the trainer (methods
track) owns the dynamics.  ``confidence()`` reads it back out.

Note that neither ``energy`` nor ``decisiveness`` touches ``K``, so it
receives **no gradient** from this file — measured: a backward pass through
``loss_terms`` puts gradient on the 12 belief planes and exactly zero on plane
12.  Whatever drives ``K`` has to come from the trainer, by construction
rather than by accident.  ``K`` is a rendered plane and so IS audited by
``givens_preserved`` on given cells, which is deliberate: a stiffening
dynamic that quietly un-commits a given is exactly the cheat the audit is for.

Two things the trainer must get right (consequences of the layout)
------------------------------------------------------------------
* **Alive masking must be off** (``alive_threshold=0``).  ``AliveMasking``
  gates updates on a max-pool of channel 0, which here is "belief in tile 0" —
  a meaningless quantity to gate on, and one that is exactly -1 on most given
  cells.  Every cell in this task is alive by construction; there is no growth
  process to mask.
* **Circular padding happens to be safe here, but only by luck.**
  ``NeuralCA`` defaults to ``padding_type="circular"``, so cell (0, j) sees
  cell (R-1, j) as a neighbour.  That is wrong for a bounded region — but the
  free interior is rows 1..R-2 by cols 1..C-2, so it is separated from the
  wrap seam by the clamped given ring, whose deltas are discarded anyway.  The
  wrap therefore cannot leak into the answer.  A variant with free cells ON
  the boundary (free-boundary regions, or a torus instance) must switch to
  zero padding or the constraint graph the model learns is not the one scored.

Difficulty
----------
Random tile sets are usually trivial in one direction or the other, so the
knob is a first-moment threshold rather than a guess.  For a random tile set
of ``T`` tiles over ``C`` colours on an ``n x n`` region with free boundary,

    E[#tilings] = T^(n^2) * C^(-2n(n-1))  ~  (T / C^2)^(n^2)

because each of the ``2n(n-1)`` internal adjacencies matches with probability
``1/C``.  So the order parameter is

    alpha = T / C^2

with the threshold at ``alpha = 1``: below it the expected number of tilings
vanishes, above it there are exponentially many.  This is the Wang analogue of
3SAT's clause ratio, and ``scripts/preview_wang.py --survey`` measures the
reference solver's work against it instead of asserting anything.

The second knob is the boundary.  A free-boundary region is easy for a greedy
row sweep; hardness in the NP-completeness proof comes from a *fixed*
boundary.  So an instance is a ring of given tiles around a free interior,
optionally plus interior clues, and instances are rejected unless the
reference solver actually has to search (``min_work``).  An instance that
arc-consistency alone decides is a warm-up, not an instance, and
``propagate_only()`` is the detector.

What that actually measures
---------------------------
``scripts/preview_wang.py --survey``, 48 instances per row over 6 random tile
sets, ``border="planted"``, ``min_work=0`` so that ``prop`` — the fraction
that arc consistency decides with NO search — stays informative.  ``work`` is
reference-solver node expansions.

    alpha sweep at 8x8              prop   q10  med  q90  max   uniq
      C=3 T=6    alpha 0.67         0.83     0    0    4   14   0.83
      C=3 T=8    alpha 0.89         0.56     0    0    5   10   0.56
      C=3 T=10   alpha 1.11         0.06     1    4    9   63   0.12
      C=3 T=12   alpha 1.33         0.00     3    8   13   37   0.02
      C=4 T=16   alpha 1.00         0.44     0    1    9   10   0.46
      C=4 T=20   alpha 1.25         0.23     0    5   12   30   0.29
      C=5 T=25   alpha 1.00         0.77     0    0    1    4   0.79

    region ladder at C=3 T=12       prop   q10  med  q90  max   rollout (floor)
      6x6                           0.29     0    2    5   15    160  (5)
      8x8    <- default             0.00     3    8   13   37    504  (7)
      10x10                         0.00     8   15   28   63   1152  (9)
      12x12                         0.00    16   30   49   82   2200 (11)
      16x16                         0.00    33   55   97  456   5880 (15)

Three things this says, and one it does not:

* **Below the threshold the instance is not an instance.**  At alpha <= 0.89
  more than half of all draws are decided by arc consistency alone — the tile
  set is so rigid that the border forces the interior, and a solver that only
  propagates scores 100% on them.  Those are warm-ups.  At alpha = 1.33 it is
  0.00 at every region size from 8x8 up, which is why that is the default.
* **alpha alone is not sufficient.**  At the same alpha = 1.00, C=4/T=16 gives
  prop 0.44 and C=5/T=25 gives 0.77: more colours at fixed alpha means each
  colour is rarer, each tile is more distinctive, and propagation gets
  stronger.  The first moment locates the interesting band; it does not pick
  the point inside it.  Measure, do not extrapolate the formula.
* **Region size is the strong axis** once alpha is in range: the median grows
  2 -> 8 -> 15 -> 30 -> 55 over 6x6 -> 16x16, with a heavy tail (a 16x16 draw
  cost 456 expansions; an off-default alpha=1.11 draw cost 23,056).  That is
  the curriculum ladder, and curriculum-by-size is the axis 3SAT said to climb.
  Note what it costs in depth: the critical path grows as
  ``(n-2)^2 * 2 * (n-1)``, i.e. roughly ``n^3``, so 16x16 asks for 5,880 steps.
  Planes are what make even the 8x8 rung affordable; the upper rungs will need
  truncated BPTT or a measured, smaller ``rounds`` rather than brute depth.
* It does NOT show a planting bias worth worrying about *here*.
  ``border="planted"`` conditions on a border some full tiling produced, which
  over-samples borders with many completions.  The unbiased control
  (``border="random"``: a ring consistent only along its own seam, kept only
  if the interior completes) at 8x8 gives q10/med/q90 = 3/9/14 against
  planted's 3/8/13 — indistinguishable — at an acceptance rate of 0.009
  against 1.000.  Train on planted; keep random as the audit, and re-run this
  comparison before claiming hardness at any other alpha or region.

Instances have MANY solutions (median >= 6 at 8x8; only 2% are unique).  A
per-pixel supervised target is therefore incoherent — its minimiser is the
mean of the solution set, which is not a solution — so this task is meant to
be trained on ``energy``, and ``target()`` exists only for the small-region
``unique`` preset.

Value conventions follow ``ncpu.sat`` / ``ncpu.subleq``: float32 screens,
background 0.0, ruled-out at -1.0, asserted at +1.0.
"""

import math
import random
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np
import torch

from ncpu.relax import BACKGROUND, HIGH_VALUE, LOW_VALUE, RelaxationTask

# Directions, in tile-tuple order: a tile is (north, east, south, west).
NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3
# (drow, dcol) for each direction; row index increases DOWNWARD, so south is +1.
OFFSETS = ((-1, 0), (0, 1), (1, 0), (0, -1))
OPPOSITE = (SOUTH, WEST, NORTH, EAST)
SIDE_NAMES = ("N", "E", "S", "W")

AGGREGATES = ("logprod", "softmax", "mean", "max")
EPS = 1e-6


# ── tile sets ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TileSet:
    """``tiles[t] = (north, east, south, west)`` colour indices.

    Frozen and hashable so the derived compatibility tables can be cached.
    """

    tiles: tuple
    n_colors: int
    label: str = ""

    @property
    def n_tiles(self) -> int:
        return len(self.tiles)

    @property
    def alpha(self) -> float:
        """``T / C^2`` — the first-moment order parameter; see module docstring."""
        return self.n_tiles / float(self.n_colors ** 2)

    def side(self, t: int, d: int) -> int:
        return self.tiles[t][d]

    def compatible(self, t: int, d: int, u: int) -> bool:
        """May tile ``u`` sit on side ``d`` of tile ``t``?"""
        return self.tiles[t][d] == self.tiles[u][OPPOSITE[d]]

    def describe(self) -> str:
        return (f"TileSet {self.label or ''} T={self.n_tiles} C={self.n_colors} "
                f"alpha={self.alpha:.2f}").strip()


def random_tileset(n_tiles=12, n_colors=3, rng=random, label="", max_tries=10_000):
    """``n_tiles`` distinct tiles with iid uniform edge colours.

    Distinctness is enforced (a duplicated tile is a smaller tile set wearing a
    disguise, and it would corrupt ``alpha``).  Nothing else is filtered: a
    tile that can never be placed is part of the honest random ensemble, and
    filtering those out would silently change the effective ``alpha``.
    """
    if n_tiles > n_colors ** 4:
        raise ValueError(f"only {n_colors ** 4} distinct tiles exist at C={n_colors}")
    seen, tiles = set(), []
    for _ in range(max_tries):
        t = tuple(rng.randrange(n_colors) for _ in range(4))
        if t in seen:
            continue
        seen.add(t)
        tiles.append(t)
        if len(tiles) == n_tiles:
            return TileSet(tuple(tiles), n_colors, label)
    raise RuntimeError(f"could not draw {n_tiles} distinct tiles in {max_tries} tries")


@lru_cache(maxsize=64)
def _supports(tileset: TileSet):
    """``supports[d][t]`` = bitmask of tiles that may sit on side ``d`` of ``t``."""
    T = tileset.n_tiles
    table = []
    for d in range(4):
        col = []
        for t in range(T):
            mask = 0
            for u in range(T):
                if tileset.compatible(t, d, u):
                    mask |= 1 << u
            col.append(mask)
        table.append(tuple(col))
    return tuple(table)


@lru_cache(maxsize=64)
def _side_onehot(tileset: TileSet):
    """``(4, T, C)`` float tensor: which colour each tile shows on each side."""
    T, C = tileset.n_tiles, tileset.n_colors
    m = torch.zeros(4, T, C)
    for d in range(4):
        for t in range(T):
            m[d, t, tileset.side(t, d)] = 1.0
    return m


def _bits(mask: int):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


# ── instances ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WangInstance:
    """One puzzle: a tile set, a region, and the cells that are given.

    ``givens`` is row-major, length ``rows * cols``; ``-1`` means free.
    ``solution`` is one known tiling (the generator always has one); it is
    kept for supervised targets and for reporting, never used by ``is_solved``.
    """

    tileset: TileSet
    rows: int
    cols: int
    givens: tuple
    solution: Optional[tuple] = None
    work: int = 0                     # reference-solver node expansions
    n_solutions: int = -1             # -1 = not counted
    meta: dict = field(default_factory=dict, compare=False)

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    @property
    def n_free(self) -> int:
        return sum(1 for g in self.givens if g < 0)

    def free_cells(self):
        return [k for k, g in enumerate(self.givens) if g < 0]


# ── exact semantics (no floats anywhere below this line) ─────────────────────

def violation_list(instance: WangInstance, candidate):
    """Every broken constraint, as ``(kind, a, b)`` tuples.

    ``("edge", cell, direction)`` for a mismatched abutting pair (counted once,
    for the east and south neighbour only), ``("given", cell, tile)`` for a
    given the candidate overwrote.  Overwritten givens are violations, not a
    separate audit: a candidate that changed the puzzle solved another puzzle.
    """
    ts, rows, cols = instance.tileset, instance.rows, instance.cols
    cand = tuple(int(x) for x in candidate)
    if len(cand) != rows * cols:
        raise ValueError(f"candidate has {len(cand)} cells, need {rows * cols}")
    bad = []
    for k, g in enumerate(instance.givens):
        if g >= 0 and cand[k] != g:
            bad.append(("given", k, g))
    for i in range(rows):
        for j in range(cols):
            t = cand[i * cols + j]
            if j + 1 < cols and ts.side(t, EAST) != ts.side(cand[i * cols + j + 1], WEST):
                bad.append(("edge", i * cols + j, EAST))
            if i + 1 < rows and ts.side(t, SOUTH) != ts.side(cand[(i + 1) * cols + j], NORTH):
                bad.append(("edge", i * cols + j, SOUTH))
    return bad


def n_internal_edges(rows, cols):
    return rows * (cols - 1) + cols * (rows - 1)


# ── reference solver: arc consistency + MRV backtracking ─────────────────────

class WangSolver:
    """Complete backtracking solver with arc consistency, over one tile set.

    Work is counted in **node expansions** (tile assignments tried), which is
    the number reported as the difficulty scale — the role WalkSAT flips played
    for 3SAT.  An instance that ``propagate_only`` already decides costs zero
    expansions and is a warm-up, not an instance.
    """

    def __init__(self, tileset: TileSet, rows: int, cols: int):
        self.ts = tileset
        self.rows, self.cols = rows, cols
        self.T = tileset.n_tiles
        self.full = (1 << self.T) - 1
        self.support = _supports(tileset)
        self._supp_cache = [{} for _ in range(4)]
        self.neighbours = []
        for i in range(rows):
            for j in range(cols):
                nb = []
                for d, (di, dj) in enumerate(OFFSETS):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        nb.append((d, ni * cols + nj))
                self.neighbours.append(tuple(nb))

    # ── domain algebra ───────────────────────────────────────────────────────

    def _supp(self, d: int, dom: int) -> int:
        """Tiles allowed on side ``d`` of *some* tile in the domain ``dom``."""
        cache = self._supp_cache[d]
        hit = cache.get(dom)
        if hit is None:
            table = self.support[d]
            hit = 0
            for t in _bits(dom):
                hit |= table[t]
            cache[dom] = hit
        return hit

    def _initial_domains(self, instance: WangInstance):
        doms = []
        for g in instance.givens:
            doms.append(self.full if g < 0 else (1 << g))
        return doms

    def _propagate(self, doms, queue):
        """AC-3 over the four-neighbour arcs.  False iff a domain empties."""
        while queue:
            k = queue.pop()
            dk = doms[k]
            for d, nk in self.neighbours[k]:
                allowed = self._supp(d, dk)
                nd = doms[nk] & allowed
                if nd == 0:
                    return False
                if nd != doms[nk]:
                    doms[nk] = nd
                    queue.append(nk)
        return True

    # ── entry points ─────────────────────────────────────────────────────────

    def propagate_only(self, instance: WangInstance):
        """Arc consistency from the givens alone, no search.

        Returns ``(domains, consistent, decided)``.  ``decided`` means every
        domain collapsed to one tile, i.e. the instance needs no search at all.
        """
        doms = self._initial_domains(instance)
        ok = self._propagate(doms, list(range(len(doms))))
        decided = ok and all(d and (d & (d - 1)) == 0 for d in doms)
        return doms, ok, decided

    def deduction_rounds(self, instance: WangInstance, cap=200):
        """Rounds of SYNCHRONOUS arc consistency to reach the fixpoint.

        Every cell updates from its neighbours' previous-round domains at once,
        which is what an NCA step actually does — the sequential AC-3 queue in
        ``_propagate`` is faster but is not the parallel schedule, and counting
        its revisions would overstate the depth an NCA needs.  This is the
        empirical grounding for the propagation part of ``min_rollout``.
        """
        doms = self._initial_domains(instance)
        for r in range(cap):
            new = list(doms)
            for k, dk in enumerate(doms):
                for d, nk in self.neighbours[k]:
                    new[nk] &= self._supp(d, dk)
            if any(d == 0 for d in new):
                return r + 1
            if new == doms:
                return r
            doms = new
        return cap

    def solve(self, instance: WangInstance, budget=10_000, rng=None):
        """``(solution | None, work, exhausted)``.

        ``exhausted`` is True when the search finished within budget, so
        ``solution is None and exhausted`` proves unsatisfiability and
        ``not exhausted`` means the budget ran out (an honest "don't know").
        """
        doms = self._initial_domains(instance)
        counter = [0]
        if not self._propagate(doms, list(range(len(doms)))):
            return None, 0, True
        try:
            sol = self._descend(doms, counter, budget, rng)
        except _BudgetExceeded:
            return None, counter[0], False
        return sol, counter[0], True

    def count(self, instance: WangInstance, cap=2, budget=200_000):
        """Number of solutions, capped at ``cap``.  ``cap=2`` tests uniqueness."""
        doms = self._initial_domains(instance)
        found = [0]
        if not self._propagate(doms, list(range(len(doms)))):
            return 0
        try:
            self._descend(doms, [0], budget, None, found=found, cap=cap)
        except _BudgetExceeded:
            pass
        return found[0]

    # ── search ───────────────────────────────────────────────────────────────

    def _descend(self, doms, counter, budget, rng, found=None, cap=1):
        k = self._select(doms)
        if k is None:
            if found is not None:
                found[0] += 1
                return None if found[0] < cap else [d.bit_length() - 1 for d in doms]
            return [d.bit_length() - 1 for d in doms]
        values = list(_bits(doms[k]))
        if rng is not None:
            rng.shuffle(values)
        for t in values:
            counter[0] += 1
            if counter[0] > budget:
                raise _BudgetExceeded()
            trial = list(doms)
            trial[k] = 1 << t
            if not self._propagate(trial, [k]):
                continue
            out = self._descend(trial, counter, budget, rng, found=found, cap=cap)
            if out is not None:
                return out
            if found is not None and found[0] >= cap:
                return None
        return None

    @staticmethod
    def _select(doms):
        """Minimum-remaining-values: the most constrained undecided cell."""
        best, best_size = None, None
        for k, d in enumerate(doms):
            size = bin(d).count("1")
            if size > 1 and (best_size is None or size < best_size):
                best, best_size = k, size
                if size == 2:
                    break
        return best


class _BudgetExceeded(Exception):
    pass


# ── the task ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WangDifficulty:
    """The instance-generation knob.  Every field is measured, not assumed.

    ``border``     ``"planted"`` takes the ring from a random full tiling (fast,
                   always satisfiable, and — like every planted ensemble — a
                   biased, easier distribution; see ``ncpu.sat.planted_ksat``).
                   ``"random"`` draws a ring that is only consistent along its
                   own seam and rejects it if the interior is unsatisfiable.
                   That is the uniform-conditioned-on-satisfiable ensemble and
                   it is the one hardness claims should use.
    ``clue_frac``  fraction of INTERIOR cells additionally revealed.  More
                   clues is strictly easier; 0.0 is the default.
    ``min_work``   reject instances the reference solver decides in fewer than
                   this many node expansions.  ``min_work=1`` alone throws out
                   every instance that arc consistency solves outright.
    ``unique``     require exactly one solution.  Needed only by a *supervised*
                   trainer: with many solutions a per-pixel target is the mean
                   of the solution set, which is generally not a solution
                   (``docs/karp_notes.md``).  An energy-trained solver should
                   leave this False and accept any solution.  Measured cost:
                   only 2% of 8x8 draws are unique and 0% of 10x10 draws, so
                   this preset is a small-region-only option, not a default.
    """

    border: str = "planted"
    clue_frac: float = 0.0
    min_work: int = 1
    unique: bool = False
    max_tries: int = 400


PRESETS = {
    # name        border      clue_frac min_work unique
    "warmup":  WangDifficulty("planted", 0.25, 0, False),
    "easy":    WangDifficulty("planted", 0.00, 1, False),
    "hard":    WangDifficulty("random", 0.00, 1, False),
    "unique":  WangDifficulty("planted", 0.00, 1, True, max_tries=4000),
}


class WangTiling(RelaxationTask):
    """Wang tiling on a ``rows x cols`` region, one tile set, given border ring.

    The tile set is fixed for the life of the object: it is the physics the
    rule is meant to internalise, not part of the per-instance input (see the
    module docstring for why putting it on the screen is a depth trap).
    """

    name = "wang"

    def __init__(self, rows=8, cols=8, tileset=None, n_tiles=12, n_colors=3,
                 pitch=1, seed=0, tau=0.25, beta=8.0, aggregate="softmax",
                 difficulty="easy", rounds=None):
        if rows < 3 or cols < 3:
            raise ValueError("need rows, cols >= 3: the border ring must enclose something")
        self.rows, self.cols = int(rows), int(cols)
        self.tileset = tileset or random_tileset(
            n_tiles, n_colors, random.Random(seed), label=f"seed{seed}")
        self.T = self.tileset.n_tiles
        self.C = self.tileset.n_colors

        # geometry: one PIXEL per tiling cell, one PLANE per tile type plus K
        self.pitch = int(pitch)
        if self.pitch < 1:
            raise ValueError("pitch must be >= 1 pixel per cell")
        self._n_planes = self.T + 1
        self.conf_plane = self.T
        self.H = self.rows * self.pitch
        self.W = self.cols * self.pitch
        self.rounds = int(rounds) if rounds is not None else (rows - 2) * (cols - 2)

        self.tau, self.beta, self.aggregate = float(tau), float(beta), aggregate
        self.difficulty = difficulty
        self.solver = WangSolver(self.tileset, self.rows, self.cols)

    # ── geometry ─────────────────────────────────────────────────────────────

    @property
    def shape(self):
        return (self.H, self.W)

    @property
    def n_planes(self):
        """``T`` belief planes plus the confidence plane ``K``.

        No structural planes: the task is translation-invariant, so there is
        nothing constant-across-instances to render, and therefore no reason to
        override ``hold_mask`` — the givens are the whole held set and the whole
        audited set, which keeps ``givens_preserved`` an undiluted cheat
        detector.  If a trainer wants coordinate planes it should add them as
        HIDDEN channels, where they are neither audited nor decoded.
        """
        return self._n_planes

    def cell_block(self, cell):
        """``(slice_y, slice_x)`` of the pixels owned by ``cell``."""
        i, j = divmod(cell, self.cols)
        return (slice(i * self.pitch, (i + 1) * self.pitch),
                slice(j * self.pitch, (j + 1) * self.pitch))

    def diameter(self):
        """Chebyshev pixel distance across the region — one one-way traverse."""
        return (max(self.rows, self.cols) - 1) * self.pitch

    def min_rollout_floor(self):
        """The one-way impossibility floor: 7 steps at the 8x8 default.

        Below this a given at one corner cannot have reached the opposite
        corner and the task is impossible rather than hard.  It is reported
        separately because it is 72x smaller than the critical path — with one
        pixel per cell and a 3x3 kernel, TRANSPORT is nearly free on this task
        and essentially all of the depth is reconciliation.  That is the point
        of choosing Wang tiling, and it is only visible once the two numbers
        are stated apart.
        """
        return self.diameter()

    def min_rollout(self):
        """The full sequential critical path, in PIXEL steps.  504 by default.

        A 3x3 kernel moves information one pixel per step, including
        diagonally, so distances are Chebyshev.  Unlike 3SAT there is no chain
        of layout-induced legs to add — the constraint graph IS the grid, and
        one traverse costs ``diameter = (max(rows, cols) - 1) * pitch``, which
        is 7 at the 8x8 default.  What is *not* free is that reconciliation is
        sequential::

            rounds          decimation is a sequence of commitments.  In the
                            worst case a round settles one cell, so the number
                            of sequential rounds is the number of free cells:
                            (rows - 2) * (cols - 2) = 36 at the default.

            2 * diameter    per round.  A commitment is only justified once the
                            consequences of every earlier commitment have
                            arrived, and the confirmation has to come back:
                            the farthest pair of cells is ``diameter`` apart,
                            so a round costs a ROUND TRIP, not a traverse.

            min_rollout  =  rounds * 2 * diameter  =  36 * 2 * 7  =  504

        This follows the Sudoku precedent (floor 8, critical path 608) and the
        ``SATLayout`` one: report the path the computation actually has to
        walk, not the light cone.

        Empirically the *propagation* part of that path is much shorter.
        ``deduction_rounds`` runs arc consistency on the SYNCHRONOUS schedule
        an NCA step actually uses; over 96 default instances it reaches its
        fixpoint in a median of 8 rounds (q10 6, q90 13, max 17), i.e. 112
        steps at the median and 238 at the worst.  And the round count barely
        grows with the region — measured medians 5 / 7 / 8 / 8 over 6x6, 8x8,
        10x10, 12x12 — so propagation depth scales with the DIAMETER while the
        worst-case decimation term above scales as ``n^3``.  The gap between
        the two is the search.

        Propagation alone decides 0.00 of these instances, so everything past
        ~240 of the 504 is search, and the search is the whole question.  If a
        run settles well inside 504 that is a *result*, not a licence: it would
        mean the learned dynamic decimates in parallel rather than one cell at
        a time.  Report it; do not assume it in advance by shortening the
        rollout.  ``scripts/preview_wang.py --survey`` reprints all of these.

        The 3SAT "run at 2.0x" lesson applies to a floor, not to a critical
        path that already contains the sequential term; 1.0x is the intended
        operating point here.  ``rounds=`` overrides the round count when a
        measurement justifies a smaller one.  Never hard-code a rollout.
        """
        return self.rounds * 2 * self.diameter()

    # ── instances ────────────────────────────────────────────────────────────

    def _resolve_difficulty(self, difficulty):
        difficulty = self.difficulty if difficulty is None else difficulty
        if isinstance(difficulty, WangDifficulty):
            return difficulty
        if isinstance(difficulty, str):
            if difficulty not in PRESETS:
                raise ValueError(f"unknown preset {difficulty!r}, have {sorted(PRESETS)}")
            return PRESETS[difficulty]
        if isinstance(difficulty, dict):
            return WangDifficulty(**difficulty)
        raise TypeError("difficulty must be a preset name, dict or WangDifficulty")

    def border_cells(self):
        """Ring cell indices, in cycle order, so consecutive entries abut."""
        R, C = self.rows, self.cols
        ring = [(0, j) for j in range(C)]
        ring += [(i, C - 1) for i in range(1, R)]
        ring += [(R - 1, j) for j in range(C - 2, -1, -1)]
        ring += [(i, 0) for i in range(R - 2, 0, -1)]
        return [i * C + j for i, j in ring]

    def interior_cells(self):
        ring = set(self.border_cells())
        return [k for k in range(self.rows * self.cols) if k not in ring]

    def _blank(self):
        return WangInstance(self.tileset, self.rows, self.cols,
                            tuple([-1] * (self.rows * self.cols)))

    def _random_full_tiling(self, rng, budget=200_000):
        """A tiling of the empty region, with randomised value order."""
        sol, _, _ = self.solver.solve(self._blank(), budget=budget, rng=rng)
        return sol

    def _random_ring(self, rng, budget=20_000):
        """A ring assignment consistent along the ring seam only.

        The ring is a cycle: consecutive cells abut, and the last abuts the
        first.  Nothing here says the interior can be completed — that is the
        point, and it is what makes this the uniform-conditioned-on-satisfiable
        ensemble once the caller rejects the unsatisfiable draws.

        One randomised, *complete* DFS with a node budget.  Restarting a
        complete search that failed is pure waste (it re-proves the same
        emptiness), so a failure here means no consistent ring exists for this
        tile set, and the caller must not retry.
        """
        ring = self.border_cells()
        n = len(ring)
        dirs = []
        for a in range(n):
            k, k2 = ring[a], ring[(a + 1) % n]
            i, j = divmod(k, self.cols)
            i2, j2 = divmod(k2, self.cols)
            dirs.append(OFFSETS.index((i2 - i, j2 - j)))
        assign = [None] * n
        if self._ring_descend(assign, 0, dirs, self.tileset, rng, [0], budget):
            return {ring[a]: assign[a] for a in range(n)}
        return None

    def _ring_descend(self, assign, a, dirs, ts, rng, counter, budget):
        n = len(assign)
        if a == n:
            return ts.compatible(assign[-1], dirs[n - 1], assign[0])
        values = list(range(self.T))
        rng.shuffle(values)
        for t in values:
            counter[0] += 1
            if counter[0] > budget:
                return False
            if a > 0 and not ts.compatible(assign[a - 1], dirs[a - 1], t):
                continue
            assign[a] = t
            if self._ring_descend(assign, a + 1, dirs, ts, rng, counter, budget):
                return True
            assign[a] = None
        return False

    def sample(self, batch_size, rng, difficulty=None):
        """A batch of instances at the requested difficulty.

        Every instance returned is satisfiable and carries the reference
        solver's node-expansion count in ``instance.work``, so a caller can
        report the difficulty distribution of the exact batch it trained on
        instead of the distribution it hoped for.
        """
        cfg = self._resolve_difficulty(difficulty)
        out, tries = [], 0
        while len(out) < batch_size:
            tries += 1
            if tries > cfg.max_tries * batch_size:
                raise RuntimeError(
                    f"only {len(out)}/{batch_size} instances met {cfg}; "
                    f"loosen min_work or move alpha={self.tileset.alpha:.2f} up")
            inst = self.draw_instance(rng, cfg)
            if inst is not None:
                out.append(inst)
        return out

    def draw_instance(self, rng, difficulty=None):
        """One draw, or None if it was rejected.  ``sample`` in the singular.

        Exposed so a survey can measure the ACCEPTANCE rate as well as the
        difficulty of what got through — a 2% acceptance rate is part of the
        description of an ensemble, not an implementation detail.
        """
        return self._one_instance(rng, self._resolve_difficulty(difficulty))

    def _one_instance(self, rng, cfg: WangDifficulty):
        givens = [-1] * (self.rows * self.cols)
        if cfg.border == "planted":
            base = self._random_full_tiling(rng)
            if base is None:
                return None
            for k in self.border_cells():
                givens[k] = base[k]
            if cfg.clue_frac > 0:
                interior = self.interior_cells()
                n_clue = int(round(cfg.clue_frac * len(interior)))
                for k in rng.sample(interior, n_clue):
                    givens[k] = base[k]
        elif cfg.border == "random":
            ring = self._random_ring(rng)
            if ring is None:
                return None
            for k, t in ring.items():
                givens[k] = t
        else:
            raise ValueError(f"border must be 'planted' or 'random', got {cfg.border!r}")

        probe = WangInstance(self.tileset, self.rows, self.cols, tuple(givens))
        sol, work, exhausted = self.solver.solve(probe, budget=200_000, rng=rng)
        if sol is None:
            return None                                   # unsatisfiable, or gave up
        if work < cfg.min_work:
            return None                                   # decided by propagation
        if cfg.clue_frac > 0 and cfg.border == "random":
            interior = self.interior_cells()
            n_clue = int(round(cfg.clue_frac * len(interior)))
            for k in rng.sample(interior, n_clue):
                givens[k] = sol[k]
            probe = WangInstance(self.tileset, self.rows, self.cols, tuple(givens))
            sol, work, exhausted = self.solver.solve(probe, budget=200_000, rng=rng)

        n_sol = -1
        if cfg.unique:
            n_sol = self.solver.count(probe, cap=2)
            if n_sol != 1:
                return None
        return WangInstance(self.tileset, self.rows, self.cols, tuple(givens),
                            solution=tuple(sol), work=work, n_solutions=n_sol,
                            meta={"border": cfg.border, "exhausted": exhausted})

    # ── rendering ────────────────────────────────────────────────────────────

    def _stamp_cell(self, state, cell, tile, confidence=HIGH_VALUE):
        """A committed cell: +1 on its tile's plane, -1 on every other, K set."""
        ys, xs = self.cell_block(cell)
        state[:self.T, ys, xs] = LOW_VALUE
        state[tile, ys, xs] = HIGH_VALUE
        state[self.conf_plane, ys, xs] = confidence

    def _blank_state(self):
        return np.full((self.n_planes, self.H, self.W), BACKGROUND, dtype=np.float32)

    def encode(self, instance):
        """The seed state ``(n_planes, H, W)``: givens stamped, the rest at
        BACKGROUND.

        BACKGROUND on the belief planes is not "empty", it is the **uniform
        belief** — ``softmax(0/tau)`` is flat — so a free cell starts in the
        state the relaxation is supposed to start in, with no separate "not yet
        seeded" value to disambiguate.  The confidence plane starts at 0 too:
        no confidence, neither committed nor contested.
        """
        state = self._blank_state()
        for k, g in enumerate(instance.givens):
            if g >= 0:
                self._stamp_cell(state, k, g)
        return state

    def target(self, instance, solution=None):
        """A fully committed state — every cell stamped, every K asserted.

        Only a *supervised* trainer wants this, and only on instances with a
        unique solution: with many solutions the per-pixel optimum is the mean
        of the solution set, which is generally not a solution.
        """
        solution = instance.solution if solution is None else solution
        if solution is None:
            raise ValueError("no solution to render")
        state = self._blank_state()
        for k, t in enumerate(solution):
            self._stamp_cell(state, k, int(t))
        return state

    def clamp_mask(self, instance):
        """1.0 on every plane of a GIVEN cell, 0.0 everywhere else.

        This is the audit set and nothing else is in it.  There are no
        structural planes to hold (see ``n_planes``), so ``hold_mask`` is left
        at its default of ``clamp_mask`` — every held pixel is a given, and
        ``givens_preserved`` is therefore all signal.  Padding this with
        constant scaffolding would fill the audit with guaranteed passes and
        stop it detecting anything.
        """
        m = np.zeros((self.n_planes, self.H, self.W), dtype=np.float32)
        for k, g in enumerate(instance.givens):
            if g >= 0:
                ys, xs = self.cell_block(k)
                m[:, ys, xs] = 1.0
        return m

    # ── reading the state back ───────────────────────────────────────────────

    def _as_tensor(self, state):
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        if state.shape[-3:] != (self.n_planes, self.H, self.W):
            raise ValueError(f"expected (..., {self.n_planes}, {self.H}, {self.W}), "
                             f"got {tuple(state.shape)}")
        return state

    def _pool(self, state):
        """``(..., P, H, W)`` -> ``(..., P, rows, cols)``, mean over each cell.

        A no-op reshape at the default ``pitch = 1``; it exists so a magnified
        rendering stays readable by exactly the same code.
        """
        x = self._as_tensor(state)
        x = x.reshape(*x.shape[:-2], self.rows, self.pitch, self.cols, self.pitch)
        return x.mean(dim=(-3, -1))

    def read_beliefs(self, state):
        """``(..., P, H, W)`` -> ``(..., cells, T)`` beliefs, row-major cells.

        Differentiable, and the only path from the state to beliefs — the
        energy and the decoder both go through it, so they can never disagree
        about what the state says.
        """
        pooled = self._pool(state)[..., :self.T, :, :]
        flat = pooled.reshape(*pooled.shape[:-2], self.rows * self.cols)
        return flat.transpose(-1, -2)

    def confidence(self, state):
        """``(..., cells)`` — the reserved stiffening plane, read back."""
        pooled = self._pool(state)[..., self.conf_plane, :, :]
        return pooled.reshape(*pooled.shape[:-2], self.rows * self.cols)

    def decode(self, state):
        """Argmax over the belief planes -> a tuple of tile indices.

        A tuple of plain ints, deliberately: it is hashable, it is exactly the
        type ``is_solved`` and ``random_candidate`` speak, and it survives
        ``relax._canonical`` unchanged.  Ties break to the lowest tile index,
        so a wholly undecided state decodes to all-zero — a real configuration
        that the matched baseline prices correctly rather than a sentinel.
        """
        b = self.read_beliefs(state)
        return tuple(int(x) for x in b.argmax(-1).reshape(-1).tolist())

    # ── exact checks ─────────────────────────────────────────────────────────

    def is_solved(self, instance, candidate):
        return not violation_list(instance, candidate)

    def violations(self, instance, candidate):
        return len(violation_list(instance, candidate))

    def edge_violations(self, instance, candidate):
        """Mismatched abutting pairs only — the givens audit reported apart."""
        return sum(1 for kind, _, _ in violation_list(instance, candidate) if kind == "edge")

    def random_candidate(self, instance, rng):
        """Uniform over tiles at every free cell; givens held.

        This is what ``matched_baseline`` samples, so it must respect the
        clamp: a baseline that is free to rewrite the border is solving an
        easier puzzle than the model and would flatter every lift.

        Be aware of what that floor is worth here.  The 8x8 default has 36
        free cells over 12 tiles, so the search space is 12**36 and a matched
        random walk over 1,000 visits measures 0.000 — a floor of exactly zero
        makes ``lift`` equal to ``solved`` and stops discriminating.  Print it
        anyway (it is a non-negotiable, and it is the number that catches a
        region so over-constrained that random guessing works), but the scale
        that actually carries information on this task is the reference
        solver's node-expansion count, not the random-walk baseline.
        """
        return tuple(g if g >= 0 else rng.randrange(self.T) for g in instance.givens)

    def reference_solve(self, instance, budget=10_000):
        sol, work, _ = self.solver.solve(instance, budget=budget)
        return (tuple(sol) if sol is not None else None), work

    def propagate_only(self, instance):
        """``(consistent, decided, total_domain_size)`` — the warm-up detector."""
        doms, ok, decided = self.solver.propagate_only(instance)
        return ok, decided, sum(bin(d).count("1") for d in doms)

    def deduction_rounds(self, instance):
        """Synchronous arc-consistency rounds to fixpoint — see ``min_rollout``."""
        return self.solver.deduction_rounds(instance)

    # ── differentiable energy ────────────────────────────────────────────────

    def beliefs_to_p(self, beliefs, tau=None):
        """Beliefs in [-1, 1] -> a simplex per cell.  ``tau`` sets the sharpness.

        At the default ``tau = 0.25`` a stamped cell (+1 against -1) is
        ``softmax(4, -4, ...)``, i.e. essentially one-hot, while an untouched
        cell at BACKGROUND is exactly uniform.
        """
        return torch.softmax(beliefs / (self.tau if tau is None else tau), dim=-1)

    def _batchify(self, state, instances):
        """Accept the batched contract, or a single state, and normalise.

        The contract is ``(B, n_planes, H, W)`` with a length-B sequence of
        instances -> ``(B,)``; an unbatched ``(n_planes, H, W)`` with one
        instance is accepted as a convenience and squeezed back on the way out.
        """
        state = self._as_tensor(state)
        if isinstance(instances, WangInstance):
            instances = [instances]
        if state.ndim == 3:
            return state.unsqueeze(0), list(instances), True
        if len(instances) == 1 and state.shape[0] != 1:
            instances = list(instances) * state.shape[0]
        if len(instances) != state.shape[0]:
            raise ValueError(f"{state.shape[0]} states vs {len(instances)} instances")
        return state, list(instances), False

    def side_marginals(self, p, instance=None):
        """``(..., cells, 4, C)`` — P(colour) on each side of each cell.

        This is the quantity an edge-colour "port" channel would carry, and
        the reason the port idea belongs in hidden channels rather than on the
        screen: it is a linear function of the belief, so nothing is lost by
        deriving it, and nothing can desynchronise.
        """
        ts = (instance.tileset if instance is not None else self.tileset)
        onehot = _side_onehot(ts).to(p.device, p.dtype)          # (4, T, C)
        return torch.einsum("...ct,dtk->...cdk", p, onehot)

    def edge_agreement(self, state, instances=None):
        """Per-adjacency soft agreement in [0, 1], ``(..., n_internal_edges)``.

        For the pair (cell, east neighbour) the agreement is
        ``sum_colour P_east(colour) * P_west'(colour)`` — the probability that
        the seam matches under an independent-cell relaxation.  Exactly 1 for a
        matching committed pair and 0 for a mismatching one, so the soft
        quantity degenerates to the exact one as beliefs saturate.
        """
        p = self.beliefs_to_p(self.read_beliefs(state))
        if instances is None or isinstance(instances, WangInstance):
            m = self.side_marginals(p, instances)
        else:
            tilesets = {inst.tileset for inst in instances}
            if len(tilesets) == 1:
                m = self.side_marginals(p, next(iter(instances)))
            else:                       # heterogeneous batch: per-instance
                m = torch.stack([self.side_marginals(p[b], inst)
                                 for b, inst in enumerate(instances)])
        R, C = self.rows, self.cols
        m = m.reshape(*m.shape[:-3], R, C, 4, self.C)
        horiz = (m[..., :, :-1, EAST, :] * m[..., :, 1:, WEST, :]).sum(-1)
        vert = (m[..., :-1, :, SOUTH, :] * m[..., 1:, :, NORTH, :]).sum(-1)
        return torch.cat([horiz.reshape(*horiz.shape[:-2], -1),
                          vert.reshape(*vert.shape[:-2], -1)], dim=-1)

    def aggregate_energy(self, agree, aggregate=None, beta=None):
        """Per-edge agreements ``(..., E)`` -> one energy to MINIMISE, ``(...)``.

        The choice was MEASURED, not argued (``preview_wang.py --check``), at
        the 8x8 default with 112 internal edges.  Two numbers per aggregate:
        the share of the energy range (solution -> uniformly random candidate)
        that ONE broken edge costs, and the Spearman rank agreement with the
        exact broken-edge count in the UNDECIDED regime (beliefs at a quarter
        magnitude, i.e. what a rollout actually looks like early on).

            aggregate   one-edge share   rho (undecided)
            softmax             45.0%             +0.938
            logprod              1.9%             +0.839
            mean                 1.7%             +0.958
            max                 99.9%             -0.011

        ``softmax`` (default): a smooth max via log-sum-exp, de-biased so a
            constant input returns itself.  It is the only aggregate that both
            keeps real pressure on the worst edge (45% of the range on one
            broken edge out of 112) and still ranks multi-violation states
            correctly.  With ``beta = 8`` a single fully broken edge among 112
            satisfied ones takes ~96% of the gradient, and the focus sharpens
            as an edge collapses rather than being fixed — which is the
            schedule a local search wants.
        ``logprod``: ``mean_e -log a_e``, the AND relaxation and the default in
            ``ncpu.satsolve``.  Its per-edge gradient ``1/(E * a_e)`` does
            diverge as an edge collapses, but on THIS task that buys nothing
            measurable: 1.9% of the range against ``mean``'s 1.7%, and worse
            rank agreement.  Reported rather than assumed — the folk claim
            that a log-domain aggregate beats a mean is not supported here.
        ``mean``: ``1 - mean_e a_e``.  The honest control, and it is not the
            straw man it is usually treated as: best rank agreement of the
            four.  What it lacks is the worst-edge pressure — 1.7% of the
            range, and a per-edge gradient pinned at ``1/112`` no matter how
            badly one edge is doing.
        ``max``: ``1 - min_e a_e``.  Non-smooth and rank-blind past one edge
            (rho ~ 0: it cannot tell one broken edge from forty); present so
            that failure mode is visible rather than assumed.
        """
        aggregate = self.aggregate if aggregate is None else aggregate
        beta = self.beta if beta is None else beta
        if aggregate == "logprod":
            return -torch.log(agree.clamp(min=EPS)).mean(dim=-1)
        if aggregate == "softmax":
            e = 1.0 - agree
            smax = torch.logsumexp(beta * e, dim=-1) / beta
            return smax - math.log(agree.shape[-1]) / beta
        if aggregate == "mean":
            return 1.0 - agree.mean(dim=-1)
        if aggregate == "max":
            return 1.0 - agree.amin(dim=-1)
        raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")

    def energy(self, state, instances):
        """``(B, n_planes, H, W)`` + B instances -> ``(B,)``.  The contract form.

        Constraint violation ONLY.  Decisiveness and given-agreement are
        separate terms below so a trainer composes them explicitly and no
        weight is hidden inside a number that gets logged as "the energy".

        The flat state is NOT a minimum, which is the thing to check before
        trusting any of this.  Uniform belief puts every colour marginal at
        1/C, so a seam agrees with probability 1/C and each edge costs
        ``1 - 1/C``; measured at the default (C=3), a wholly uniform state
        scores 0.660 (against the 0.667 the algebra predicts), the seed state
        0.621, and a correct tiling 0.005.  Flatness is penalised, not
        rewarded.  ``decisiveness`` then removes the shallow
        tiny-but-correctly-ordered optimum that would decode fine and be
        destroyed by the next update.
        """
        state, instances, squeeze = self._batchify(state, instances)
        e = self.aggregate_energy(self.edge_agreement(state, instances))
        return e[0] if squeeze else e

    def decisiveness(self, state, margin=1.0):
        """Penalty for beliefs that never leave the fence.

        The edge energy is content with tiny, correctly-ordered beliefs: they
        decode fine and are wrecked by the next update.  A solution has to be a
        FIXED POINT, and a fixed point needs saturation — this is the term that
        makes settling mean something, and it is the term the confidence slot's
        stiffening dynamic should co-evolve with.
        """
        b = self.read_beliefs(state)
        top = b.amax(-1)
        return torch.relu(margin - top).pow(2).mean(dim=(-1,))

    def given_penalty(self, state, instance):
        """Soft version of the clamp, for trainers that do not hard-clamp.

        Weight this at 0 when the clamp is hard — the clamp is the mechanism,
        this term is only a fallback, and double-counting it would let a model
        trade given-agreement against edge agreement.
        """
        p = self.beliefs_to_p(self.read_beliefs(state))
        idx = [(k, g) for k, g in enumerate(instance.givens) if g >= 0]
        if not idx:
            return torch.zeros(p.shape[:-2], device=p.device, dtype=p.dtype)
        cells = torch.tensor([k for k, _ in idx], device=p.device)
        tiles = torch.tensor([g for _, g in idx], device=p.device)
        picked = p[..., cells, :].gather(-1, tiles.expand(*p.shape[:-2], len(idx)).unsqueeze(-1))
        return -torch.log(picked.squeeze(-1).clamp(min=EPS)).mean(-1)

    def loss_terms(self, state, instances, w_decisive=0.05, w_given=0.0):
        """``(loss, extras)`` — the differentiable number and the honest one.

        Batched like ``energy``.  ``extras["exact_violations"]`` is the
        ALGORITHMIC count from the decoded state, per batch element.  Log both,
        and never report the energy as a solve rate.
        """
        state, instances, squeeze = self._batchify(state, instances)
        agree = self.edge_agreement(state, instances)
        energy = self.aggregate_energy(agree)
        loss = energy + w_decisive * self.decisiveness(state)
        if w_given:
            loss = loss + w_given * torch.stack(
                [self.given_penalty(state[b], inst) for b, inst in enumerate(instances)])
        exact = [self.violations(inst, self.decode(state[b]))
                 for b, inst in enumerate(instances)]
        extras = {
            "energy": energy.detach(),
            "agree": agree.detach(),
            "worst_edge": (1.0 - agree).amax(-1).detach(),
            "exact_violations": exact[0] if squeeze else exact,
        }
        return (loss[0] if squeeze else loss), extras

    # ── reporting ────────────────────────────────────────────────────────────

    def describe(self):
        return (f"WangTiling {self.rows}x{self.cols} cells  "
                f"state {self.n_planes}x{self.H}x{self.W}  {self.tileset.describe()}  "
                f"pitch={self.pitch}  free={self.rows * self.cols - len(self.border_cells())}  "
                f"min_rollout={self.min_rollout()} (floor {self.min_rollout_floor()}, "
                f"{self.rounds} rounds x 2x{self.diameter()})")
