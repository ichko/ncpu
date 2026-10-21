"""The lattice spin glass as a native relaxation task — Track I.

READ THIS FIRST: which variant, and is it actually hard
-------------------------------------------------------
The plain 2D Ising spin glass — nearest-neighbour couplings on a **planar**
lattice, **no external field** — has a ground state computable in POLYNOMIAL
time.  Frustrated plaquettes pair up along domain walls and the minimum-energy
pairing is a minimum-weight perfect matching (Bieche, Maynard, Rammal & Suchail
1980; Hadlock 1975 for planar max-cut; Barahona 1982 for the classification).
Building that and calling it NP-hard would burn GPU on a problem in P, so this
module makes the choice explicitly and names the complexity status of each:

    variant     bonds per site                    complexity of the ground state
    ---------------------------------------------------------------------------
    "planar"    N, E, S, W                        POLYNOMIAL (matching).  A
                                                  CONTROL, not a target.
    "nnn"       + SE, SW diagonals  <- default    non-planar; outside every known
                                                  polynomial class, and the
                                                  standard hard case
    any + field random h_i on top                 Barahona 1982: 2D + field is
                                                  NP-hard

Being careful about what is proved.  Ground-state search for a +-J glass is
max-cut on the interaction graph, which is NP-hard in general and polynomial
when the graph is planar.  Adding both diagonals to each square makes the
interaction graph non-planar (it contains K5 minors), so the planar matching
algorithm and every other known polynomial result stop applying, and the
problem sits in the class Barahona's hardness results are about (he proves
NP-hardness for the non-planar two-level grid and for the 2D lattice in a
field).  What this module does **not** claim is a reduction targeted at the
next-nearest-neighbour square lattice specifically.  If you need a citation
that is airtight for the exact family, use the field variant
(``field_frac > 0``, Barahona's 2D-with-field case) and read the caveat in
``IsingDifficulty.field_frac``.  ``variant="planar"`` is kept precisely so the
polynomial case can be run as a control: same encoding, same dynamics, same
scorecard, one is in P and one is not, and that comparison is available to no
other track here.

Why the two-level grid was NOT chosen, though its citation is the cleanest.
Barahona's canonical non-planar case is two stacked LxL layers, which would map
onto two belief planes at zero extra depth (the inter-layer bond is radius 0 in
space).  It loses on the thing this track exists for: the exact ground state.
The transfer matrix has to cut the graph, and the cut width doubles from ``C``
to ``2C``, so exact ground truth would be capped near a 7-wide lattice instead
of a 12-wide one.  Exact ground states are the yardstick for BP and SP here;
paying for a citation with the yardstick is the wrong trade.

Why a spin glass at all
-----------------------
Its constraints are already nearest-neighbour, so unlike 3SAT there is no
factor graph to embed and no layout to pay for in depth
(``docs/3sat_findings.md``).  With the diagonals, every interaction is inside a
single 3x3 kernel — the interaction range and the NCA's receptive field are the
same object, which is as close to free as this phase gets.  And the objective
is not a checklist of constraints to satisfy: it is an energy to minimise, some
of whose terms *must* be broken.  That distinction changes the right answer to
the energy-aggregate question (see ``IsingLattice.energy``) and makes this the
one track whose loss is the physics rather than a relaxation of a checker.

The encoding: one belief plane, and the couplings alongside it
--------------------------------------------------------------
State is ``(n_planes, R, C)``, **one pixel per spin**::

    plane 0        spin belief m in [-1, +1]
                       +1  "this spin is up"
                       -1  "this spin is down"
                        0  undecided
    plane 1        J to the EAST  neighbour, in {-1, 0, +1};  0 = no bond
    plane 2        J to the SOUTH neighbour
    plane 3        J to the SOUTH-EAST neighbour        (variant="nnn" only)
    plane 4        J to the SOUTH-WEST neighbour        (variant="nnn" only)
    plane 5        external field h                     (field_frac > 0 only)
    plane -1       CONFIDENCE K in [-1, +1]

A scalar belief is the *right* encoding here, and this is the one task where it
is.  Sudoku rejected a scalar digit channel because a scalar has no way to be
undecided and because it imposes an order on an unordered alphabet
(``ncpu.tasks.sudoku``).  Neither objection survives a binary, antipodal
alphabet: the two symbols really are opposite, the midpoint really is "no
opinion", and ``m`` is a magnetisation — the same object BP computes as a
marginal.  One plane per spin instead of two also means a cell's belief and its
neighbour's arrive in the same 3x3 read.

Each site owns the bonds to its *forward* neighbours only, so every bond is
stored exactly once; the backward bonds are read out of the neighbour's planes,
which a 3x3 kernel can see.  A missing bond is a hard ``0``, which is what makes
the boundary self-describing: the rule does not need a border plane to know it
is at an edge, because the coupling it would use is zero.

**Padding must be zero, not circular.**  Free cells sit on the boundary here
(``relax.py`` states the rule), so ``NeuralCA``'s default ``circular`` padding
would wrap row 0 onto row R-1 and hand the rule a torus that the scorer does
not score.  The zero couplings make that wrap *detectable* rather than harmless
— a wrapped neighbour is reached through a zero coupling — but "learnable to
ignore" is not "correct", and the trainer should pass ``padding_type="zeros"``.
Alive masking must also be off (``alive_threshold=0``): plane 0 is a signed
belief and gating on it would switch off every spin that believes "down".

What was rejected
-----------------
* **Two one-hot planes for the two spin states.**  Costs a plane, buys nothing:
  a binary alphabet's simplex is one-dimensional, so the second plane is
  ``-``the first plus noise, and the noise is a new way to be inconsistent.
* **Couplings baked into the weights, the way Wang bakes its tile set.**  For
  Wang the tile set is the physics and the instance is the boundary; here the
  couplings ARE the instance.  Hiding them in the weights would leave one
  instance per model.
* **Bonds rendered on their own dual-lattice sites** (a 2R x 2C screen with
  spins on even sites and couplings on odd ones).  It is the physically honest
  picture — couplings live on edges, not sites — and it doubles the lattice
  pitch, so every distance and therefore the whole rollout doubles, to buy
  nothing the forward-bond convention does not already give.
* **A supervised target: the exact ground state.**  Ground states come in
  degenerate sets (measured below: a median of 11.5 at 8x8, mod global flip,
  growing to 49.5 at 10x10), so a
  per-pixel target's minimiser is the mean of the set, which is not a ground
  state.  This is the same argument that makes ``energy`` the objective for Wang
  and for ``ncpu.satsolve``.  Here it is even cleaner, because the energy IS the
  objective rather than a relaxation of a checker.
* **Periodic boundaries.**  A torus is still (genus-1) polynomial in zero field,
  it costs the exact transfer matrix a factor ``2^C``, and it makes the
  padding question ambiguous.  Free boundaries throughout.

Belief propagation, survey propagation, and the exact answer
------------------------------------------------------------
This track owns the reference solvers (``ncpu.inference``) because a claim that
an NCA "learned something SP-like" is unfalsifiable without the SP curve.  All
numbers below are 16 8x8 instances at ``p_af = 0.5`` with the reference spin
clamped, against exact transfer-matrix ground states; "solved" means the decoded
configuration attains ``E0`` exactly::

    algorithm                       planar (girth 4)         nnn (girth 3)
                                 solved  excess  conv     solved  excess  conv
    BP  beta=1, no damping        0.000    59.0  0.00      0.000    98.0  0.00
    BP  beta=1, damping 0.5       0.125     6.0  0.00      0.000    16.0  0.00
    BP  beta=1, damping 0.9       0.125     8.0  0.00      0.062    12.0  0.00
    SP  y=0                       0.062     6.0  1.00      0.000    12.0  1.00
    SP  y=1                       0.125     5.0  0.06      0.000    10.0  0.00
    greedy quench, 64 restarts    0.000    19.0    -       0.125    21.9    -
    BP-decimation + polish        0.500     1.0    -       0.375     2.0    -
    SP-decimation + polish        0.750     0.0    -       0.688     0.0    -

Four findings, and the first two are the ones to keep.

1. **BP does not converge on this lattice, at any damping, ever.**  Not "often
   fails" -- 0 of 16 in every row above, at 500 iterations.  That is the price of
   short loops: BP assumes a locally tree-like factor graph, the square lattice
   has loops of length 4 everywhere, and the diagonals add loops of length 3.
   Reported as a finding, not filed as a bug: the same code converges in 8
   iterations on a chain and returns the exactly correct frozen set, and
   converges on 92% of locally tree-like random 3-regular graphs.
2. **The decimation loop is the algorithm.**  One-shot propagation solves
   0.000-0.125; add hard-fix-and-restart and the same messages solve 0.688.
   Everything interesting happens in the outer loop -- which is exactly the
   component this phase forbids the NCA, so "can a learned stiffening dynamic
   replace it" is not a detail of the method, it is the whole question.
3. **SP beats BP, and by more where BP is worse.**  0.750 vs 0.500 on the
   planar lattice, 0.688 vs 0.375 on the non-planar one.  SP also converges
   (1.00 at ``y=0``) where BP never does, so its per-round cost is lower: a
   median 3,626 message iterations against BP's 5,752.
4. **SP's frozen estimate is nevertheless useless here**, and that is the
   uncomfortable one.  Correlation of ``SP.frozen`` with the exact frozen mask
   over the same 16 instances is ``-0.09`` (``y=0``) and ``+0.09`` (``y=1``);
   ``|SP.bias|`` and ``|BP.marginal|`` are no better (``-0.09``, ``-0.04``).  On
   locally tree-like graphs the same quantity scores ``+0.52``.  So SP-decimation
   works on the lattice through the *ordering* of its biases, not through a
   calibrated frozenness -- and a confidence channel trained to imitate SP would
   be imitating noise.  The way out is that this track can supply the ground
   truth directly: ``exact_frozen`` is exact and costs one forward-backward pass
   of the transfer matrix, so the stiffening channel can be supervised against
   the true answer and then measured against SP, which is the one experiment in
   this phase where the NCA could beat the algorithm it is imitating.

The gauge, and why one spin is clamped
--------------------------------------
With no external field, ``H(s) = H(-s)``: every ground state comes with its
global flip, so "is spin i up" has no answer and BP, SP and any decoder all
return exactly zero polarisation.  The instance therefore clamps a **reference
spin** at the centre, taken from a ground state.  That single given is what
turns a coset into a configuration.  It is also the only genuinely auditable
given the task has, which is why ``clamp_mask`` marks it (plus any optional
clue spins) and nothing else: the coupling planes never move and would pad the
audit with guaranteed passes.

Confidence, and decimation without an outer loop
------------------------------------------------
This is the track where the target quantity for the stiffening channel can be
written down exactly.  Survey propagation's message alphabet is
``{-1, 0, +1}`` — "push down", "no opinion in this cluster", "push up" — and its
site survey gives ``P(H_i != 0)``: the probability that spin ``i`` is **frozen**
in a randomly drawn cluster of ground states.  SP-decimation then hard-fixes the
most strongly biased spin and restarts, which is exactly the outer loop this
phase forbids.  Plane ``K`` is the NCA-native replacement, and what it should
hold is not a vague confidence but a specific number::

    K_i  ~  P(spin i is frozen)  =  1 - SP_site(H_i = 0)

``sp_frozen(instance)`` returns SP's estimate and ``exact_frozen(instance)``
returns the ground truth (a spin is frozen iff it takes the same value in every
ground state, modulo the global flip), so a trained ``K`` can be scored against
both — and per finding 4 above it should be **supervised against the exact one**,
because SP's estimate of its own quantity is not usable on this lattice.

Nothing in this file puts gradient on ``K`` (measured: a backward pass through
``energy`` puts gradient on the belief plane and exactly zero everywhere else).
The trainer owns whether the rule may write to it, and the reading is Wang's,
``delta <- (1 - relu(K)) * delta``.

Difficulty, measured
--------------------
The knob is ``p_af``, the fraction of antiferromagnetic bonds.  The first-moment
*response* is frustration: a face is frustrated when the product of its
couplings is negative, so a 4-bond square is frustrated with probability
``(1 - (1-2p)^4)/2`` and a triangle with ``(1 - (1-2p)^3)/2``, both peaking at
``p = 0.5``.  Measured over 64 8x8 draws, the formula is right and that is the
problem with it::

    variant   p_af    frustration measured    formula
    planar    0.10        0.288 +- 0.074       0.295
    planar    0.25        0.456 +- 0.060       0.469
    planar    0.50        0.501 +- 0.076       0.500
    nnn       0.10        0.250 +- 0.053       0.244
    nnn       0.25        0.433 +- 0.038       0.438
    nnn       0.50        0.492 +- 0.036       0.500

**Frustration saturates and is therefore a bad difficulty axis.**  A tenth of
the bonds flipped already frustrates a quarter of the faces, and by ``p = 0.25``
the count is within 6% of its maximum.  The formula locates the band; it cannot
order instances inside it.  What does order them is measured per instance (48
8x8 draws each, degeneracy modulo the global flip, ``qhit`` the per-attempt
success rate of a random greedy quench, ``brk`` the fraction of bonds the ground
state itself leaves unsatisfied)::

    variant  p_af  frust     E0   deg  frozen    qhit  q-excess  brk
    planar   0.10  0.293  -92.8    12   0.620  0.0010     23.6   8.6%
    planar   0.25  0.480  -82.6    63   0.386  0.0010     18.5  13.1%
    planar   0.50  0.499  -81.8    59   0.394  0.0010     18.1  13.5%
    nnn      0.10  0.222 -171.8     1   0.995  0.0250     38.6   9.1%
    nnn      0.25  0.438 -124.4     8   0.701  0.0030     24.1  20.4%
    nnn      0.50  0.494 -112.6     9   0.594  0.0010     20.7  23.2%

``brk`` is the number that reframes the task: **the ground state of the default
instance leaves 23% of its bonds unsatisfied.**  There is no state that
satisfies everything, which is why ``violations`` counts excess energy rather
than broken bonds and why the energy aggregate is a sum rather than a smooth
max.

The size ladder is the strong axis, and it is also where an instance stops being
a warm-up (nnn, ``p_af = 0.5``; ``quench x64`` is the chance that any of 64
random greedy quenches lands on a ground state; SP/BP decimation fix ``n/32``
spins per round and finish with a greedy polish)::

    size    gen ms  bonds      E0  deg med  frozen  quench x64  SP-dec  BP-dec
    6x6        6.4    110   -59.6      5.5   0.604      0.812    0.750   0.688
    8x8  <-   19.2    210  -111.0     11.5   0.519      0.125    0.625   0.188
    10x10    268.3    342  -179.5     49.5   0.429      0.000    0.375   0.125
    12x12   8070.4    506  -273.5    137.0   0.736      0.000      -       -

  (unfiltered draws, ``difficulty="easy"``.  The default ``"hard"`` preset
  additionally rejects any instance one of 16 greedy quenches solves, which
  takes the 8x8 ``quench x64`` column from 0.188 to 0.125 for about 2 ms extra
  per instance -- it halves the easy tail and cannot remove it; see
  ``IsingDifficulty.max_quench_rate``.)

* **6x6 is not an instance.**  Sixty-four random quenches find a ground state
  81% of the time, so a model scoring on 6x6 has demonstrated local descent and
  nothing more.  It is the warm-up rung of a curriculum, never a result.
* **8x8 is the first honest rung** and is the default: 64 random quenches solve
  0.125 of filtered draws against 0.812 at 6x6, while SP-decimation reaches
  0.625-0.833 and BP-decimation 0.188-0.417.  Those three numbers -- quench,
  BP-decimation, SP-decimation -- are the ladder any learned solver is placed
  on.  (They move a few points with the instance sample; ``--survey --survey-n
  12`` reproduces the shape in under two minutes.)
* **Generation cost is a real constraint on the ladder.**  Exact ground truth is
  a ``2^C x 2^C`` transfer matrix per row, so it is 21 ms at 8x8, 270 ms at
  10x10 and 8 s at 12x12.  Fresh instances every training step are affordable
  through 8x8; past 10x10 a pool becomes necessary and the "is it memorising the
  pool" question comes back with it.
* ``deg`` and ``frozen`` are not monotone in size (the 12x12 row is 4 instances
  and is noise).  Degeneracy grows, so a per-pixel target gets worse as the
  lattice grows, not better.

Both variants are kept at ``p_af = 0.5``; the ``"ferro"`` preset (``p_af = 0.1``)
is the negative control, where the lattice is nearly a ferromagnet, the ground
state is unique, 99.5% of spins are frozen, and a quench succeeds 25x more
often.

Values follow the house convention (``ncpu.sat``, ``ncpu.subleq``): float32
planes, background 0.0, low -1.0, high +1.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from ncpu.inference import (
    IsingModel,
    belief_propagation,
    greedy_quench,
    quench_rate,
    sp_decimation,
    survey_propagation,
    warning_propagation,
)
from ncpu.relax import BACKGROUND, HIGH_VALUE, LOW_VALUE, RelaxationTask

VARIANTS = ("planar", "nnn")
AGGREGATES = ("sum", "mean", "softmax", "max", "logprod")
EPS = 1e-9

# Directions a site owns.  Every bond is stored once, on its upper/left end.
#   E  : (r, c) -- (r,   c+1)
#   S  : (r, c) -- (r+1, c  )
#   SE : (r, c) -- (r+1, c+1)
#   SW : (r, c) -- (r+1, c-1)
DIRS = ("E", "S", "SE", "SW")
DIR_OFFSET = {"E": (0, 1), "S": (1, 0), "SE": (1, 1), "SW": (1, -1)}

# Sequential reconciliation rounds.  MEASURED, not assumed: decimation is the
# reference algorithm and ``preview_ising.py --survey`` measures how many spins
# it can commit per round before the solve rate falls (8x8 nnn, SP-decimation
# with a greedy polish, 16 instances):
#
#     spins fixed per round      1     2     4     8    16    32    63
#     rounds                    63    32    16     8     4     2     1
#     solved                 0.688 0.688 0.562 0.438 0.375 0.188 0.125
#
# Two per round is free and four is nearly free, so the honest round count is
# half the free spins rather than all of them -- which is the one place this
# task is cheaper than Wang tiling, whose ``rounds`` is its full free-cell
# count.  ``ROUNDS_MEASURED`` overrides the rule where a measurement exists.
ROUNDS_MEASURED: dict = {("nnn", 8, 8): 32}
SPINS_PER_ROUND = 2


# ── instance ─────────────────────────────────────────────────────────────────

@dataclass
class IsingInstance:
    """One lattice, its couplings, and its exact ground state.

    ``J`` is ``(4, R, C)`` in ``DIRS`` order, zero where the bond does not exist
    (the lattice boundary, or a direction the variant does not use).  ``h`` is
    ``(R, C)`` integer-valued.  ``ground`` is one exact ground state and ``e0``
    its energy: both come from the transfer matrix at generation time, so
    ``is_solved`` is an exact check and never an energy threshold.

    ``clamped`` maps a flat site index to its clamped spin.  It always contains
    the reference spin that gauge-fixes the global flip.
    """

    rows: int
    cols: int
    variant: str
    J: np.ndarray
    h: np.ndarray
    ground: np.ndarray
    e0: float
    clamped: dict = field(default_factory=dict)
    degeneracy: float = 1.0
    quench_hit: float = float("nan")
    p_af: float = 0.5

    @property
    def n(self) -> int:
        return self.rows * self.cols

    def site(self, r, c) -> int:
        return int(r) * self.cols + int(c)

    def rc(self, i):
        return divmod(int(i), self.cols)

    @property
    def n_bonds(self) -> int:
        return int(np.count_nonzero(self.J))

    def __repr__(self) -> str:
        return (f"IsingInstance({self.rows}x{self.cols}, {self.variant}, "
                f"bonds={self.n_bonds}, E0={self.e0:.0f}, "
                f"deg={self.degeneracy:.0f}, clamped={len(self.clamped)})")


# ── lattice geometry ─────────────────────────────────────────────────────────

def bond_mask(rows: int, cols: int, variant: str) -> np.ndarray:
    """``(4, R, C)`` 0/1: which of the four owned bonds exist at each site."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    m = np.zeros((4, rows, cols), dtype=np.float32)
    m[0, :, :cols - 1] = 1.0                      # E
    m[1, :rows - 1, :] = 1.0                      # S
    if variant == "nnn":
        m[2, :rows - 1, :cols - 1] = 1.0          # SE
        m[3, :rows - 1, 1:] = 1.0                 # SW
    return m


def edge_list(rows: int, cols: int, J: np.ndarray):
    """``(edges, couplings)`` for the sites/bonds a ``(4, R, C)`` array carries."""
    edges, vals = [], []
    for d, name in enumerate(DIRS):
        dr, dc = DIR_OFFSET[name]
        rs, cs = np.nonzero(J[d])
        for r, c in zip(rs, cs):
            edges.append((r * cols + c, (r + dr) * cols + (c + dc)))
            vals.append(J[d, r, c])
    return np.array(edges, dtype=np.int64).reshape(-1, 2), np.array(vals, dtype=np.float64)


def to_model(instance: IsingInstance) -> IsingModel:
    """The instance as a graph-level ``inference.IsingModel``."""
    edges, vals = edge_list(instance.rows, instance.cols, instance.J)
    return IsingModel(instance.n, edges, vals, instance.h.reshape(-1))


def lattice_energy(instance: IsingInstance, s) -> float:
    """``H(s) = -sum J s s - sum h s`` straight off the ``(4, R, C)`` arrays."""
    s = np.asarray(s, dtype=np.float64).reshape(instance.rows, instance.cols)
    J, tot = instance.J, 0.0
    tot += float(np.sum(J[0, :, :-1] * s[:, :-1] * s[:, 1:]))
    tot += float(np.sum(J[1, :-1, :] * s[:-1, :] * s[1:, :]))
    tot += float(np.sum(J[2, :-1, :-1] * s[:-1, :-1] * s[1:, 1:]))
    tot += float(np.sum(J[3, :-1, 1:] * s[:-1, 1:] * s[1:, :-1]))
    return -tot - float(np.sum(instance.h * s))


# ── frustration ──────────────────────────────────────────────────────────────

def faces(rows: int, cols: int, variant: str):
    """The elementary cycles whose sign defines frustration.

    For ``"planar"`` these are the unit squares.  Adding the diagonals splits
    every square into four triangles, and the triangles are the elementary
    cycles then — which is the geometric statement of why ``"nnn"`` is harder
    for BP: the shortest loop drops from 4 to 3, so the locally-tree-like
    assumption is violated at the smallest possible scale.

    Returns a list of ``(kind, [(dir_index, r, c), ...])``.
    """
    out = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            if variant == "planar":
                out.append(("square", [(0, r, c), (1, r, c + 1),
                                       (0, r + 1, c), (1, r, c)]))
            else:
                # the two triangles cut by the SE diagonal, then the two cut by SW
                out.append(("tri", [(0, r, c), (1, r, c + 1), (2, r, c)]))
                out.append(("tri", [(1, r, c), (0, r + 1, c), (2, r, c)]))
                out.append(("tri", [(0, r, c), (3, r, c + 1), (1, r, c)]))
                out.append(("tri", [(1, r, c + 1), (0, r + 1, c), (3, r, c + 1)]))
    return out


def frustrated_faces(instance: IsingInstance) -> np.ndarray:
    """0/1 per elementary face: 1 when the product of its couplings is negative.

    A frustrated face cannot have all its bonds satisfied by any assignment.
    This is the *source* of hardness and the thing the difficulty knob moves,
    and it is a property of the couplings alone — no solving required.
    """
    fs = faces(instance.rows, instance.cols, instance.variant)
    out = np.zeros(len(fs), dtype=np.int8)
    for k, (_, bonds) in enumerate(fs):
        prod = 1.0
        for d, r, c in bonds:
            prod *= instance.J[d, r, c]
        out[k] = 1 if prod < 0 else 0
    return out


def frustration(instance: IsingInstance) -> float:
    """Fraction of elementary faces that are frustrated."""
    f = frustrated_faces(instance)
    return float(f.mean()) if len(f) else 0.0


def expected_frustration(p_af: float, variant: str) -> float:
    """First-moment prediction: ``(1 - (1-2p)^L)/2`` for a face of ``L`` bonds.

    Stated so that the measurement has something to disagree with.  ``L`` is 4
    for the planar squares and 3 for the ``nnn`` triangles, and both peak at
    ``p = 0.5``: 1/2 of all faces frustrated, which is the maximally glassy
    point of the +-J ensemble.
    """
    L = 4 if variant == "planar" else 3
    return (1.0 - (1.0 - 2.0 * p_af) ** L) / 2.0


# ── exact ground states: transfer matrix over rows ───────────────────────────
#
# The interaction range in the row direction is 1, so a row's spin
# configuration separates the lattice above from the lattice below.  Dynamic
# programming over rows with a state of ``2^C`` row configurations is therefore
# exact, at cost ``O(R * 4^C)`` — which reaches a 12-wide lattice, where brute
# force stops near 5x5.  Energies are integers (unit couplings, integer fields)
# so the DP runs in int32 and every comparison below is exact.

_ROWCFG_CACHE: dict = {}


def _row_spins(cols: int) -> np.ndarray:
    """``(2^C, C)`` int8 of every row configuration; bit c set means spin -1."""
    if cols in _ROWCFG_CACHE:
        return _ROWCFG_CACHE[cols]
    x = np.arange(1 << cols, dtype=np.int64)
    S = (1 - 2 * ((x[:, None] >> np.arange(cols)[None, :]) & 1)).astype(np.int8)
    _ROWCFG_CACHE[cols] = S
    return S


def _row_energy(instance, S, r):
    """Energy inside row ``r`` for every row configuration: ``(2^C,)`` int32."""
    Si = S.astype(np.int32)
    e = -(Si[:, :-1] * Si[:, 1:]) @ instance.J[0, r, :-1].astype(np.int32)
    e = e - Si @ instance.h[r].astype(np.int32)
    return e.astype(np.int32)


def _inter_energy(instance, S, r):
    """Energy between rows ``r`` and ``r+1``: ``(2^C, 2^C)`` int32, x=row r."""
    Si = S.astype(np.int32)
    J = instance.J
    M = -(Si * J[1, r].astype(np.int32)) @ Si.T
    if instance.variant == "nnn":
        M = M - (Si[:, :-1] * J[2, r, :-1].astype(np.int32)) @ Si[:, 1:].T
        M = M - (Si[:, 1:] * J[3, r, 1:].astype(np.int32)) @ Si[:, :-1].T
    return M


def exact_ground_state(instance: IsingInstance, gauge: Optional[bool] = None,
                       count: bool = True, max_cols: int = 14):
    """``(config, E0, degeneracy)`` by exact transfer matrix.  Ground truth.

    ``gauge`` restricts to configurations whose **top-left spin is +1**, so the
    reported degeneracy is the count *modulo the global flip*.  Without it every
    zero-field instance is trivially 2K-fold degenerate and the number says
    nothing.  ``None`` (the default) turns it on exactly when the field is zero,
    which is when the symmetry it removes actually exists.

    ``degeneracy`` is a float because the count is exponential; it saturates
    rather than overflowing, and it is a diagnostic, never a key.
    """
    R, C = instance.rows, instance.cols
    if gauge is None:
        gauge = not np.any(instance.h)
    if C > max_cols:
        raise ValueError(f"exact transfer matrix needs 2^{C} x 2^{C} per row; "
                         f"cap is {max_cols} columns (transpose the lattice, or "
                         "raise max_cols and wait)")
    S = _row_spins(C)
    BIG = np.int32(1 << 28)
    f = _row_energy(instance, S, 0).copy()
    if gauge:
        f = np.where(S[:, 0] > 0, f, BIG)
    cnt = (f < BIG).astype(np.float64) if count else None
    back = np.empty((max(R - 1, 0), 1 << C), dtype=np.int32)
    for r in range(R - 1):
        V = f[:, None] + _inter_energy(instance, S, r)
        arg = V.argmin(axis=0)
        mn = V[arg, np.arange(1 << C)]
        if count:
            cnt = ((V == mn[None, :]) * cnt[:, None]).sum(axis=0)
        back[r] = arg
        f = mn + _row_energy(instance, S, r + 1)
    e0 = int(f.min())
    deg = float(cnt[f == f.min()].sum()) if count else float("nan")
    # backtrack one optimal path
    cfg = np.empty((R, C), dtype=np.int8)
    x = int(np.argmin(f))
    for r in range(R - 1, -1, -1):
        cfg[r] = S[x]
        if r:
            x = int(back[r - 1][x])
    return cfg, float(e0), deg


def exact_frozen(instance: IsingInstance, gauge: Optional[bool] = None,
                 max_cols: int = 14):
    """``(frozen, can_up, can_down)``: which spins take the same value in EVERY
    ground state.

    Forward-backward on the same transfer matrix.  ``fwd[r][x] + bwd[r][x]`` is
    the best total energy of any configuration whose row ``r`` is ``x``, so a
    spin can be ``+1`` exactly when some row configuration with that bit set
    achieves the global minimum.  ``frozen = can_up XOR can_down``.

    This is the exact version of the quantity survey propagation estimates, and
    therefore the ground truth against which a learned confidence channel should
    be scored.

    **Read the gauge carefully.**  With no field a ground state and its global
    flip are both ground states, so *every* spin is free by symmetry and the
    unconditioned answer is vacuous.  Pinning the top-left spin to ``+1`` fixes
    that, and what the result then means is "frozen **relative to the top-left
    spin**": ``s_i * s_0`` is the same in every ground state.  A consequence
    that will bite anyone who does not expect it -- **site (0, 0) is reported
    frozen by construction**, whether or not it is physically frozen, because
    the enumeration never sees it take the other value.  Exclude it, or read it
    as the reference it is.  ``gauge=None`` enables this exactly when the field
    is zero.
    """
    R, C = instance.rows, instance.cols
    if gauge is None:
        gauge = not np.any(instance.h)
    if C > max_cols:
        raise ValueError(f"exact_frozen needs 2^{C} states per row; cap is {max_cols}")
    S = _row_spins(C)
    BIG = np.int32(1 << 28)
    n_cfg = 1 << C

    fwd = np.empty((R, n_cfg), dtype=np.int32)
    fwd[0] = _row_energy(instance, S, 0)
    if gauge:
        fwd[0] = np.where(S[:, 0] > 0, fwd[0], BIG)
    for r in range(R - 1):
        V = fwd[r][:, None] + _inter_energy(instance, S, r)
        fwd[r + 1] = V.min(axis=0) + _row_energy(instance, S, r + 1)

    bwd = np.zeros((R, n_cfg), dtype=np.int32)
    for r in range(R - 2, -1, -1):
        nxt = bwd[r + 1] + _row_energy(instance, S, r + 1)
        bwd[r] = (_inter_energy(instance, S, r) + nxt[None, :]).min(axis=1)

    total = fwd + bwd
    e0 = int(total.min())
    can_up = np.zeros((R, C), dtype=bool)
    can_dn = np.zeros((R, C), dtype=bool)
    for r in range(R):
        opt = total[r] == e0
        for c in range(C):
            up = S[:, c] > 0
            can_up[r, c] = bool(np.any(opt & up))
            can_dn[r, c] = bool(np.any(opt & ~up))
    return (can_up ^ can_dn), can_up, can_dn


def brute_force_ground_state(instance: IsingInstance, gauge: Optional[bool] = None):
    """Exhaustive ``2^n`` search.  Only for cross-checking the transfer matrix."""
    n = instance.n
    if gauge is None:
        gauge = not np.any(instance.h)
    if n > 22:
        raise ValueError(f"brute force over 2^{n} states is not a plan")
    best, best_e, deg = None, math.inf, 0
    for x in range(1 << n):
        s = np.array([1 - 2 * ((x >> k) & 1) for k in range(n)], dtype=np.int8)
        if gauge and s[0] < 0:
            continue
        e = lattice_energy(instance, s)
        if e < best_e - 1e-9:
            best, best_e, deg = s.reshape(instance.rows, instance.cols), e, 1
        elif abs(e - best_e) < 1e-9:
            deg += 1
    return best, best_e, float(deg)


# ── difficulty ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class IsingDifficulty:
    """The instance-generation knob.  Each field is measured, not assumed.

    ``p_af``        fraction of antiferromagnetic (``J = -1``) bonds.  0.5 is the
                    symmetric +-J glass and the frustration maximum; small
                    ``p_af`` is a dilute ferromagnet and a negative control.
    ``field_frac``  fraction of sites carrying a ``+-1`` external field.  Turning
                    this on is the second, better-cited route to NP-hardness
                    (Barahona 1982), with one caveat worth stating: his
                    construction uses fields of unbounded strength, so a
                    ``+-1`` field is the *family* he proves hard about but not
                    literally his instance.  It also destroys the global flip
                    symmetry, which makes BP and SP behave visibly better —
                    useful as a diagnostic, and a reason not to make it the
                    default when the question is whether the NCA can break the
                    symmetry itself.
    ``clue_frac``   fraction of sites revealed from a ground state, on top of the
                    reference spin.  Strictly easier; 0.0 is the default.
    ``max_quench_rate``
                    reject an instance if a random greedy quench finds a ground
                    state more often than this, over ``quench_trials`` restarts.
                    This is the "an instance a trivial rule solves is not an
                    instance" filter, and it is measured per instance rather
                    than argued from ``p_af``.  ``0.0`` means "not one of the
                    restarts may succeed", which is the setting that separates
                    an 8x8 lattice from a 6x6 one: at 6x6 the per-attempt rate
                    is 0.023 and 81% of instances fall to 64 restarts, at 8x8 it
                    is 0.000 and none do.  Read the per-attempt rate, not
                    any-of-K, when comparing across different ``quench_trials``.

                    **It is a screen, not a proof, and the measurement says so.**
                    At 8x8 the true per-attempt rate is already ~0.002, so a
                    64-restart sample is badly underpowered to detect it: over 48
                    filtered instances, re-measuring with 128 fresh restarts
                    still finds a hit on 12.5% of them, against 18.8% unfiltered.
                    The filter halves the easy tail and cannot do better at any
                    affordable number of restarts.  What is actually useful is
                    that every instance records its own measured rate in
                    ``IsingInstance.quench_hit``, so an evaluation can stratify
                    or exclude after the fact instead of trusting the screen.
                    The default is 16 restarts, which costs ~20 ms per 8x8
                    instance against 50 ms for 64 and screens nearly as well.
    ``min_frustration``
                    reject an instance with fewer frustrated faces than this
                    fraction.  Cheap pre-filter; the quench rate is the real one.
    """

    p_af: float = 0.5
    field_frac: float = 0.0
    clue_frac: float = 0.0
    max_quench_rate: float = 1.0
    min_frustration: float = 0.0
    quench_trials: int = 16
    max_tries: int = 200


PRESETS = {
    #  name         p_af  field  clues  max_quench  min_frust
    "warmup":  IsingDifficulty(0.50, 0.00, 0.25, 1.00, 0.00),
    "ferro":   IsingDifficulty(0.10, 0.00, 0.00, 1.00, 0.00),   # negative control
    "easy":    IsingDifficulty(0.50, 0.00, 0.00, 1.00, 0.00),   # no filter at all
    "hard":    IsingDifficulty(0.50, 0.00, 0.00, 0.00, 0.00),   # <- the default
    "field":   IsingDifficulty(0.50, 1.00, 0.00, 0.00, 0.00),
}


# ── the task ─────────────────────────────────────────────────────────────────

class IsingLattice(RelaxationTask):
    """Ground-state search on an ``R x C`` +-J lattice, one pixel per spin.

    Args:
        rows, cols:  lattice size.  Exact ground truth needs ``cols <= 14``.
        variant:     ``"nnn"`` (default, non-planar) or ``"planar"`` (in P, the
                     complexity control).  See the module docstring.
        difficulty:  preset name, dict, or ``IsingDifficulty``.
        rounds:      sequential reconciliation rounds ``min_rollout`` assumes.
                     Defaults to the measured p90 for this size and variant.
        gamma:       belief -> magnetisation sharpness in ``energy``
                     (``m = tanh(gamma * plane0)``).  3.0 puts a house-convention
                     ``+-1`` belief at ``tanh 3 = 0.995``, so a rendered ground
                     state scores 0.005 against a 0.57 range to a random one —
                     0.9% of the range, and exactly 0 once the beliefs saturate.
        field:       force the field plane on or off, overriding what the
                     difficulty preset implies.  The plane count is geometry and
                     must not change when ``sample`` is handed another preset.
        aggregate:   residual aggregate.  ``"sum"`` — the Hamiltonian itself — is
                     the default and the measurement says it should be; read
                     ``energy`` before changing it.
        beta:        sharpness of the smooth-max aggregates.
    """

    name = "ising"

    def __init__(self, rows=8, cols=8, variant="nnn", difficulty="hard",
                 rounds=None, gamma=3.0, aggregate="sum", beta=8.0,
                 exact_max_cols=14, field=None):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if aggregate not in AGGREGATES:
            raise ValueError(f"aggregate must be one of {AGGREGATES}, got {aggregate!r}")
        if rows < 2 or cols < 2:
            raise ValueError("need rows, cols >= 2 for a bond to exist")
        self.rows, self.cols = int(rows), int(cols)
        self.variant = variant
        self.difficulty = difficulty
        self.gamma = float(gamma)
        self.aggregate = aggregate
        self.beta = float(beta)
        self.exact_max_cols = int(exact_max_cols)
        self._bond_mask = bond_mask(self.rows, self.cols, self.variant)
        self.n_dirs = 4 if variant == "nnn" else 2
        cfg = self._resolve_difficulty(difficulty)
        # the field plane is part of the GEOMETRY, so it cannot follow a
        # per-call difficulty override: fix it at construction and say so.
        self.has_field = bool(cfg.field_frac > 0.0) if field is None else bool(field)
        n_free = self.rows * self.cols - 1
        self.rounds = int(rounds) if rounds is not None else ROUNDS_MEASURED.get(
            (variant, self.rows, self.cols),
            int(math.ceil(n_free / SPINS_PER_ROUND)))

    # ── geometry ─────────────────────────────────────────────────────────────

    @property
    def shape(self) -> tuple:
        return (self.rows, self.cols)

    @property
    def n_planes(self) -> int:
        """belief + owned couplings + optional field + confidence."""
        return 1 + self.n_dirs + (1 if self.has_field else 0) + 1

    @property
    def belief_plane(self) -> int:
        return 0

    @property
    def coupling_planes(self) -> slice:
        return slice(1, 1 + self.n_dirs)

    @property
    def field_plane(self) -> Optional[int]:
        return 1 + self.n_dirs if self.has_field else None

    @property
    def confidence_plane(self) -> int:
        return self.n_planes - 1

    def diameter(self) -> int:
        """Chebyshev diameter in pixels — the 8-neighbour kernel's own metric."""
        return max(self.rows, self.cols) - 1

    def min_rollout_floor(self) -> int:
        """The one-way impossibility floor: the lattice diameter.

        Below this a spin has not heard from the far corner.  That matters here
        in a way it does not for a checkable CSP: the ground state genuinely
        depends on distant couplings — flipping one coupling in a corner changes
        spins arbitrarily far away, which ``preview_ising.py --survey`` measures
        directly rather than asserting.  A frustrated pair of plaquettes is
        joined by a domain wall, and which pairing is cheapest is a *global*
        matching, so no finite-radius rule can decide a spin.

        Measured at 8x8 nnn, 48 single-coupling flips: flipping one bond moves
        7.6% of the lattice's spins on average, and the farthest moved spin sits
        at Chebyshev distance median 0, p90 3, **max 7 -- the full diameter**.
        Most flips are local; some are not, and a rule that cannot see the whole
        lattice gets those wrong every time.
        """
        return self.diameter()

    def min_rollout(self) -> int:
        """The full sequential critical path, in pixel steps.

        Two terms, and here — unlike Wang and Sudoku — the transport term is
        genuinely small:

            2 * diameter    per reconciliation round.  A spin flips only once
                            the consequences of the far side of the lattice have
                            arrived AND the confirmation has come back, so a
                            round is a round trip, not a traverse.

            rounds          how many such rounds are sequential.  MEASURED:
                            SP-decimation keeps its full solve rate while
                            committing two spins per round and loses only 0.13
                            at four, so the honest count is half the free spins,
                            not all of them (the table above ``ROUNDS_MEASURED``).
                            Committing every spin at once collapses the solve
                            rate from 0.688 to 0.125, so the sequential term is
                            real and not an artefact of the reference solver.

            min_rollout  =  rounds * 2 * diameter  =  32 * 2 * 7  =  448 at 8x8

        What dominates.  Both terms scale with the lattice, which is the
        difference from Sudoku (where the round count dominated a fixed 9x9
        grid) and from Wang (where the round count is the free-cell count).
        Here transport is one pixel per bond — the interaction range and the
        kernel are the same object, the cheapest geometry any track in this
        phase has — and the depth is bought entirely by *sequential
        reconciliation*, which is the honest statement of why a spin glass is
        hard: not that the information is far away, but that the decision keeps
        changing as it arrives.

        The reference cost says the same thing in another unit: BP-decimation
        spends a median of 5,752 message iterations at 8x8 and SP-decimation
        3,626, and one message iteration is one hop -- one NCA step.  So 448 is
        an order of magnitude *below* what the reference algorithm sequentially
        needs, and a model that solves inside it has done something the
        reference does not: commit in parallel.  That is the result to look for
        rather than a licence to shorten the rollout further.

        Never hard-code a rollout; derive from this and pass ``rounds=`` when a
        measurement justifies a different count.
        """
        return int(self.rounds * 2 * self.diameter())

    # ── instances ────────────────────────────────────────────────────────────

    def _resolve_difficulty(self, difficulty):
        difficulty = self.difficulty if difficulty is None else difficulty
        if isinstance(difficulty, IsingDifficulty):
            return difficulty
        if isinstance(difficulty, str):
            if difficulty not in PRESETS:
                raise ValueError(f"unknown preset {difficulty!r}, have {sorted(PRESETS)}")
            return PRESETS[difficulty]
        if isinstance(difficulty, dict):
            return IsingDifficulty(**difficulty)
        raise TypeError("difficulty must be a preset name, dict or IsingDifficulty")

    def _rng(self, rng):
        """Accept a ``random.Random``, a numpy Generator, or an int seed.

        The other tracks are handed ``random.Random`` by their trainers and this
        module is vectorised, so it normalises rather than forcing a choice on
        the caller.
        """
        if isinstance(rng, np.random.Generator):
            return rng
        if rng is None:
            return np.random.default_rng()
        if isinstance(rng, (int, np.integer)):
            return np.random.default_rng(int(rng))
        return np.random.default_rng(rng.randrange(1 << 62))

    def draw_couplings(self, rng, cfg: IsingDifficulty):
        J = np.where(rng.random((4, self.rows, self.cols)) < cfg.p_af, -1.0, 1.0)
        J = (J * self._bond_mask).astype(np.float32)
        h = np.zeros((self.rows, self.cols), dtype=np.float32)
        if cfg.field_frac > 0:
            on = rng.random((self.rows, self.cols)) < cfg.field_frac
            sign = np.where(rng.random((self.rows, self.cols)) < 0.5, -1.0, 1.0)
            h = (on * sign).astype(np.float32)
        return J, h

    def draw_instance(self, rng, difficulty=None) -> IsingInstance:
        """One accepted instance, with its exact ground state attached."""
        cfg = self._resolve_difficulty(difficulty)
        rng = self._rng(rng)
        for _ in range(cfg.max_tries):
            J, h = self.draw_couplings(rng, cfg)
            inst = IsingInstance(self.rows, self.cols, self.variant, J, h,
                                 ground=np.ones((self.rows, self.cols), np.int8),
                                 e0=0.0, p_af=cfg.p_af)
            if cfg.min_frustration > 0 and frustration(inst) < cfg.min_frustration:
                continue
            cfgnd, e0, deg = exact_ground_state(inst, max_cols=self.exact_max_cols)
            inst.ground, inst.e0, inst.degeneracy = cfgnd, e0, deg
            # the reference spin: gauge-fix the global flip with one given
            centre = (self.rows // 2) * self.cols + (self.cols // 2)
            clamped = {centre: int(cfgnd.reshape(-1)[centre])}
            if cfg.clue_frac > 0:
                extra = rng.random(inst.n) < cfg.clue_frac
                for i in np.nonzero(extra)[0]:
                    clamped[int(i)] = int(cfgnd.reshape(-1)[i])
            inst.clamped = clamped
            hit, _, _ = quench_rate(to_model(inst), e0, trials=cfg.quench_trials,
                                    rng=rng, clamped=clamped)
            inst.quench_hit = hit
            if hit > cfg.max_quench_rate:
                continue
            return inst
        raise RuntimeError(
            f"no {self.rows}x{self.cols} instance passed the difficulty filter "
            f"in {cfg.max_tries} tries ({cfg}).  Small lattices cannot pass "
            "max_quench_rate=0: a greedy quench solves essentially every 4x4 and "
            "80% of 6x6 draws.  Grow the lattice, or use difficulty='warmup'.")

    def sample(self, batch_size: int, rng, difficulty: Optional[Any] = None) -> list:
        """A batch of instances, each with its exact ground state.

        There is no instance pool and no augmentation.  Generation is a
        transfer matrix over ``2^C`` row states plus the quench filter, and the
        measured cost per instance is::

            size    difficulty="easy"   difficulty="hard" (quench filter)
            6x6            6.5 ms              ~30 ms  <- rejects most: see below
            8x8           19.5 ms              ~21 ms
            10x10        268   ms             ~270 ms
            12x12       8070   ms                -

        So at the 8x8 default a batch of 16 fresh instances costs about 0.3 s of
        CPU, which is affordable next to a 448-step rollout but is not free —
        draw it on a worker.  Fresh instances every step is what removes the "is
        it memorising the pool" question, and past 10x10 the question comes back
        because a pool becomes unavoidable.

        The filter is deliberately not tightened at 6x6.  It *could* be: 64
        quenches solve 79% of 6x6 draws and the filter would throw all of them
        away.  But conditioning that hard produces an atypical tail rather than
        a harder problem — the same planted-ensemble bias Wang tiling reports —
        so 6x6 belongs in a curriculum as ``"warmup"``, not as a filtered
        ``"hard"``.
        """
        rng = self._rng(rng)
        return [self.draw_instance(rng, difficulty) for _ in range(batch_size)]

    # ── rendering ────────────────────────────────────────────────────────────

    def encode(self, instance: IsingInstance) -> np.ndarray:
        """The seed state, ``(n_planes, R, C)`` float32.

        Free spins sit at BACKGROUND on the belief plane, which reads as "no
        opinion" — the honest starting state, and the same zero that means
        "unseeded".  Clamped spins are stamped saturated on the belief plane and
        get ``+1`` confidence.

        Like Sudoku, this departs from the letter of ``RelaxationTask.encode``
        ("everything else at BACKGROUND") in the structural planes: the coupling
        and field planes carry the instance.  They are not part of the answer —
        they *are* the question — and there is nowhere else for them to live,
        because unlike Wang's tile set the couplings change every instance.
        """
        s = np.zeros((self.n_planes, self.rows, self.cols), dtype=np.float32)
        s[self.belief_plane] = BACKGROUND
        s[self.coupling_planes] = instance.J[:self.n_dirs]
        if self.has_field:
            s[self.field_plane] = instance.h
        if not self.has_field and np.any(instance.h):
            raise ValueError("instance carries an external field but this task has "
                             "no field plane; build IsingLattice(..., field=True)")
        flat_b = s[self.belief_plane].reshape(-1)
        flat_k = s[self.confidence_plane].reshape(-1)
        for i, v in instance.clamped.items():
            flat_b[i] = HIGH_VALUE if v > 0 else LOW_VALUE
            flat_k[i] = HIGH_VALUE
        return s

    def target(self, instance: IsingInstance) -> np.ndarray:
        """The exact ground state as a belief plane.  Diagnostics and previews.

        NOT a training target.  Ground states are degenerate (see the module
        docstring), so per-pixel regression onto one of them has the mean of the
        set as its minimiser, and that mean is not a ground state.
        """
        s = self.encode(instance)
        s[self.belief_plane] = instance.ground.astype(np.float32)
        s[self.confidence_plane] = 1.0
        return s

    def clamp_mask(self, instance: IsingInstance) -> np.ndarray:
        """1.0 on the BELIEF plane at clamped spins; 0.0 everywhere else.

        Two deliberate exclusions.  The coupling and field planes are constants
        of the instance and would score 1.0 in any audit, so including them
        would pad the cheat detector with guaranteed passes — the failure mode
        ``relax.clamp_mask`` warns about.  The confidence plane is excluded for
        the opposite reason: it is *meant* to be written by the rule, so
        auditing it would report a working stiffening dynamic as an overwritten
        given.  (Wang audits its ``K`` at given cells; this track does not, and
        the difference is the point — here ``K`` has a numerical target,
        ``sp_frozen``, so a separate check on it is available and is better than
        a clamp.)
        """
        m = np.zeros((self.n_planes, self.rows, self.cols), dtype=np.float32)
        flat = m[self.belief_plane].reshape(-1)
        for i in instance.clamped:
            flat[i] = 1.0
        return m

    def structure_mask(self) -> np.ndarray:
        """1.0 on the coupling and field planes: the instance, held not audited."""
        m = np.zeros((self.n_planes, self.rows, self.cols), dtype=np.float32)
        m[self.coupling_planes] = 1.0
        if self.has_field:
            m[self.field_plane] = 1.0
        return m

    def hold_mask(self, instance) -> np.ndarray:
        """What the trainer re-implants each step: clamped spins + the couplings."""
        return np.maximum(self.clamp_mask(instance), self.structure_mask())

    def decode(self, frame: torch.Tensor) -> np.ndarray:
        """Sign of the belief plane -> an ``(R, C)`` int8 array of spins.

        A belief of exactly 0 decodes to ``+1``.  That is a coin flip dressed as
        a decision and it is deliberate: the alternative (a random tie-break)
        would make ``count_distinct`` and every replay non-deterministic.  It
        does not paste the clamped spins back in — a decoder that did would make
        the clamp audit unfalsifiable.
        """
        f = torch.as_tensor(frame)
        if f.dim() != 3:
            raise ValueError(f"decode wants (C, H, W), got {tuple(f.shape)}")
        m = f[self.belief_plane].detach().cpu().numpy()
        return np.where(m >= 0, 1, -1).astype(np.int8)

    def confidence(self, frame: torch.Tensor) -> np.ndarray:
        """The confidence plane, for comparison against ``sp_frozen``."""
        f = torch.as_tensor(frame)
        return f[self.confidence_plane].detach().cpu().numpy()

    # ── exact semantics ──────────────────────────────────────────────────────

    def is_ground_state(self, instance, candidate) -> bool:
        """Energy equals the exact ground-state energy.  Ignores the gauge."""
        return abs(lattice_energy(instance, candidate) - instance.e0) < 1e-9

    def is_solved(self, instance, candidate) -> bool:
        """Ground truth: a ground state that respects the clamped spins.

        The clamp check is not redundant.  With zero field the global flip of a
        ground state is also a ground state, so without it the task's answer is
        a coset rather than a configuration and ``givens_preserved`` would be
        auditing something the score did not require.  A model that returns the
        flipped ground state has found the physics and violated its given;
        ``is_ground_state`` reports the first, ``is_solved`` requires both.
        """
        s = np.asarray(candidate, dtype=np.int64).reshape(-1)
        for i, v in instance.clamped.items():
            if s[i] != v:
                return False
        return self.is_ground_state(instance, s)

    def violations(self, instance, candidate) -> int:
        """Excess energy above the ground state, in coupling quanta.  0 iff solved.

        NOT the number of unsatisfied bonds.  On a frustrated lattice every
        configuration breaks bonds — the ground state itself breaks a measured
        ~15% of them — so "bonds broken" is never zero and is not a violation
        count.  What is zero exactly at a solution is the *excess* over the
        minimum, and with unit couplings and integer fields that excess is an
        integer.  In zero field it is always even, because flipping any set of
        spins changes the energy by an even number, so ``excess / 2`` is the
        number of extra broken bonds; ``broken_bonds`` reports that directly.

        Clamp violations are added on top, so a candidate that overwrites a given
        cannot score zero.
        """
        s = np.asarray(candidate, dtype=np.int64).reshape(-1)
        e = lattice_energy(instance, s)
        bad = int(round(e - instance.e0))
        for i, v in instance.clamped.items():
            bad += int(s[i] != v)
        return bad

    def broken_bonds(self, instance, candidate) -> int:
        """Bonds with ``J s_i s_j < 0``.  Never zero on a frustrated instance."""
        return to_model(instance).unsatisfied(np.asarray(candidate).reshape(-1))

    def random_candidate(self, instance, rng, mode="uniform"):
        """A random configuration that respects the clamped spins.

        ``mode="uniform"`` is the contract's floor: an independent fair coin per
        free spin.  Read the resulting baseline with care — with 63 free spins at
        the 8x8 default the chance of hitting a ground state is around
        ``deg / 2^63``, so ``relax.matched_baseline`` returns exactly 0.000 for
        any rollout budget and a "lift" over it measures nothing.  Sudoku hit the
        same wall.

        ``mode="quench"`` is the baseline that means something: a random start
        followed by greedy single-spin descent until no flip helps.  That is the
        dumbest possible local relaxation, it uses exactly the locality an NCA
        has, and its success rate is this task's difficulty measure
        (``IsingDifficulty.max_quench_rate``).  It is the number a learned solver
        actually has to beat.
        """
        rng_np = self._rng(rng)
        s = np.where(rng_np.random(instance.n) < 0.5, -1, 1).astype(np.int8)
        for i, v in instance.clamped.items():
            s[i] = v
        if mode == "quench":
            s, _ = greedy_quench(to_model(instance), s, rng=rng_np,
                                 frozen=list(instance.clamped) or None)
        elif mode != "uniform":
            raise ValueError(f"mode must be 'uniform' or 'quench', got {mode!r}")
        return s.reshape(instance.rows, instance.cols)

    def exact_solve(self, instance):
        """``(ground state, E0, degeneracy)`` — the transfer matrix.  Ground truth."""
        return exact_ground_state(instance, max_cols=self.exact_max_cols)

    def reference_solve(self, instance, budget: int = 10_000):
        """SP-decimation plus a greedy polish -> ``(solution | None, work)``.

        The reference is deliberately **not** the exact solver.  The transfer
        matrix costs the same on every instance of a given width, so its work is
        a constant and says nothing about difficulty; it is ground truth, not a
        baseline.  Survey propagation with decimation is the algorithm this
        phase wants the NCA to have rediscovered, so its cost — total message
        iterations — is the comparator, and its failures are the interesting
        instances.  Returns ``None`` when it does not reach a ground state.
        """
        rng = np.random.default_rng(0)
        model = to_model(instance)
        res = sp_decimation(model, rng=rng, clamped=instance.clamped,
                            max_iters=min(budget, 300))
        cfg = res.config.reshape(instance.rows, instance.cols)
        ok = self.is_solved(instance, cfg)
        return (cfg if ok else None), float(res.total_iters)

    # ── reference propagation, exposed on the task for convenience ───────────

    def bp(self, instance, beta=1.0, **kw):
        """Sum-product BP on this instance, with the reference spin clamped."""
        return belief_propagation(to_model(instance), beta=beta,
                                  clamped=instance.clamped, **kw)

    def wp(self, instance, **kw):
        """Warning propagation (the ``beta -> inf`` limit)."""
        return warning_propagation(to_model(instance), clamped=instance.clamped, **kw)

    def sp(self, instance, y=0.0, **kw):
        """Survey propagation.  ``result.frozen`` is what plane ``K`` should hold.

        ``y=0`` is the measured working point; see ``inference.survey_propagation``
        for the sweep that rejects the larger values theory would suggest.
        """
        return survey_propagation(to_model(instance), y=y,
                                  clamped=instance.clamped, **kw)

    def sp_frozen(self, instance, y=0.0, **kw) -> np.ndarray:
        """SP's ``P(spin frozen)`` as an ``(R, C)`` array.

        It is what plane ``K`` is *for*, and on this lattice it is also a poor
        estimate of the truth (rho ~ 0 against ``exact_frozen``; see the module
        docstring).  Supervise ``K`` against ``exact_frozen`` and *measure* it
        against this.
        """
        return self.sp(instance, y=y, **kw).frozen.reshape(instance.rows, instance.cols)

    def exact_frozen(self, instance) -> np.ndarray:
        """Ground truth for ``K``: spins with the same value in every ground state."""
        fr, _, _ = exact_frozen(instance, max_cols=self.exact_max_cols)
        return fr

    # ── differentiable energy ────────────────────────────────────────────────

    def _stack(self, instances, attr):
        return torch.from_numpy(np.stack([getattr(i, attr) for i in instances]))

    def beliefs(self, state: torch.Tensor) -> torch.Tensor:
        """``m = tanh(gamma * plane0)`` — the belief plane as a magnetisation.

        ``tanh`` is not decoration.  ``NeuralCA`` clips its state to ``[-10, 10]``,
        and the Ising energy is *multilinear* in the spins, so an unsquashed
        belief of magnitude 10 would make the energy 100x more negative than any
        real configuration: the model would minimise the loss by shouting rather
        than by solving.  Squashing into the open cube restores the property this
        whole encoding rests on — see ``energy``.
        """
        x = torch.as_tensor(state)
        return torch.tanh(self.gamma * x[..., self.belief_plane, :, :])

    def soft_energy(self, state: torch.Tensor, instances) -> torch.Tensor:
        """The Hamiltonian evaluated on the relaxed beliefs, ``(B,)``.

        The key property, and the reason this track's energy needed no repair
        term while Sudoku's did: ``H`` is **multilinear** in ``m``, so its
        minimum over the cube ``[-1, 1]^n`` is attained at a *vertex*, and a
        vertex is a spin configuration.  The continuous relaxation is therefore
        exact — ``min_m H(m) = E0`` — and the zero set of ``energy`` is exactly
        the set of ground states.  No purity or one-hot term is needed to stop
        the beliefs pooling in the middle.
        """
        x = torch.as_tensor(state)
        if x.dim() == 3:
            x = x[None]
        insts = [instances] if isinstance(instances, IsingInstance) else list(instances)
        if len(insts) != x.shape[0]:
            raise ValueError(f"{len(insts)} instances for a batch of {x.shape[0]}")
        m = self.beliefs(x)
        J = self._stack(insts, "J").to(m.device, m.dtype)
        h = self._stack(insts, "h").to(m.device, m.dtype)
        bond = (J[:, 0, :, :-1] * m[:, :, :-1] * m[:, :, 1:]).flatten(1).sum(1)
        bond = bond + (J[:, 1, :-1] * m[:, :-1] * m[:, 1:]).flatten(1).sum(1)
        if self.variant == "nnn":
            bond = bond + (J[:, 2, :-1, :-1] * m[:, :-1, :-1] * m[:, 1:, 1:]).flatten(1).sum(1)
            bond = bond + (J[:, 3, :-1, 1:] * m[:, :-1, 1:] * m[:, 1:, :-1]).flatten(1).sum(1)
        return -bond - (h * m).flatten(1).sum(1)

    def residuals(self, state: torch.Tensor, instances):
        """Per-constraint unsatisfaction in ``[0, 1]``, ``(B, K)``, plus a mask.

        ``(1 - J m_i m_j) / 2`` per bond and ``(1 - sign(h) m_i) / 2`` per field
        term, weighted by ``|J|`` and ``|h|`` so that their weighted mean is an
        affine image of the Hamiltonian.  Only used by the aggregate ablations
        and by the diagnostics; ``energy`` goes through ``soft_energy``.
        """
        x = torch.as_tensor(state)
        if x.dim() == 3:
            x = x[None]
        insts = [instances] if isinstance(instances, IsingInstance) else list(instances)
        m = self.beliefs(x)
        J = self._stack(insts, "J").to(m.device, m.dtype)
        h = self._stack(insts, "h").to(m.device, m.dtype)
        pairs = [(J[:, 0, :, :-1], m[:, :, :-1] * m[:, :, 1:]),
                 (J[:, 1, :-1], m[:, :-1] * m[:, 1:])]
        if self.variant == "nnn":
            pairs.append((J[:, 2, :-1, :-1], m[:, :-1, :-1] * m[:, 1:, 1:]))
            pairs.append((J[:, 3, :-1, 1:], m[:, :-1, 1:] * m[:, 1:, :-1]))
        res, wts = [], []
        for Jd, prod in pairs:
            res.append(((1.0 - torch.sign(Jd) * prod) / 2.0).flatten(1))
            wts.append(Jd.abs().flatten(1))
        if self.has_field:
            res.append(((1.0 - torch.sign(h) * m) / 2.0).flatten(1))
            wts.append(h.abs().flatten(1))
        return torch.cat(res, dim=1), torch.cat(wts, dim=1)

    def energy(self, state: torch.Tensor, instances) -> torch.Tensor:
        """Excess energy above the exact ground state, per bond.  ``(B,)``, >= 0.

        ``(H(m) - E0) / n_bonds``, which is zero exactly at a ground state (in
        the saturation limit) and strictly positive everywhere else.

        **The aggregate, and why this track's answer differs from the others.**
        ``relax.py`` says to keep pressure on the WORST constraint and to measure
        the choice.  Wang measured it and chose a de-biased smooth max; Sudoku
        chose one too.  Measured here, the smooth max is *wrong*, and not
        marginally:

        Measured over 10 seeds at 8x8 (210 bonds, ``difficulty="hard"``), and
        reproduced by ``preview_ising.py --check``, which averages over 3
        instances because a single instance's ``rho`` moves by +-0.1.  ``rho`` is
        the Spearman agreement with the exact excess energy over 96
        clamp-respecting candidates in the *undecided* regime, each spin
        carrying its own random belief magnitude in ``[0.05, 0.65]`` -- which is
        what a rollout actually looks like early on::

            aggregate    rho, nnn (mean/min)     rho, planar (mean/min)
            logprod        +0.728 / +0.625         +0.781 / +0.725
            sum            +0.615 / +0.457         +0.658 / +0.490
            softmax        +0.361 / +0.220         +0.517 / +0.437
            max              ~0                      ~0

        ``sum`` beats ``softmax`` on 9 of 10 seeds in both variants, by +0.254 on
        average on ``nnn`` and +0.141 on ``planar``.  "One-flip share" -- the
        share of the saturated energy range that flipping one spin out of the
        ground state costs -- is ~5% for ``sum``, ~7% for ``softmax`` and exactly
        0% for ``max``: the smooth max buys about 1.5x the worst-bond pressure
        and pays roughly half the rank agreement for it, which is the trade this
        task specifically does not want.  At a *uniform* belief magnitude every
        monotone aggregate scores ``+1.000``, because the per-bond residuals then
        take only two values; the separation appears only once the beliefs carry
        different confidences, i.e. exactly when it matters.

        ``sum`` (default) is the Hamiltonian itself, normalised per bond, so its
        value is interpretable (excess energy per bond) and its zero set is the
        ground-state set.  ``mean`` is the same number by another route and is
        kept only to make that visible.

        ``logprod`` is the honest surprise: it ranks *better* than the sum on
        every seed, and it does not break the objective -- at saturation each
        residual is 0 or 1, so ``logprod`` degenerates to a weighted count of
        broken bonds, which is affine in the true energy.  Checked directly: over
        2,387 saturated non-ground states with strictly higher true energy, none
        of ``sum``, ``logprod`` or ``softmax`` ranked one at or below the ground
        state.  What ``logprod`` loses is legibility -- its value is not an
        energy -- and what it gains is extra weight on nearly-broken bonds while
        beliefs are still soft.  It is the ablation to try first, not a mistake.

        ``max`` is rank-blind (rho ~ 0), with a zero share because a saturated
        random configuration and a ground state both contain a fully broken bond.

        The reason is structural rather than numerical.  Wang and Sudoku are
        satisfaction problems: every constraint can be met at once, so a
        constraint that is still broken at the end is a constraint the model
        gave up on, and leaning on the worst one is right.  A frustrated spin
        glass has no such state — the ground state itself leaves a measured
        fraction of bonds unsatisfied, and *which* ones it leaves is the answer.
        Pressure on the worst bond is therefore pressure to satisfy a bond that
        must be broken, i.e. it optimises the wrong objective.  The sum is not a
        lazy default here; it is the physics, and the ablations are kept only so
        that this stays a measurement rather than a claim.

        The flat state.  ``m = 0`` gives ``H = 0``, which is *not* a minimum:
        every ground state has ``E0 < 0``.  But in zero field it IS an exact
        stationary point — ``dH/dm_i = -sum_j J_ij m_j - h_i`` is identically
        zero there — so a rollout seeded at BACKGROUND everywhere sits on a
        saddle with no gradient at all until something breaks the symmetry.  The
        clamped reference spin is what breaks it, and that is the second reason
        it is in the encoding rather than a convenience.  Measured at the 8x8
        nnn default, backward pass through ``energy``::

            state                                      max |dE/dm|   pixels with gradient
            flat, no clamp                               0.00e+00           0 / 64
            the actual seed (flat + one clamped spin)    1.42e-02           8 / 64
            flat, random field, no clamp                 1.43e-02          64 / 64

        Read the middle row carefully, because it is the operating condition:
        **the entire lattice has zero gradient except the eight pixels touching
        the one clamped spin.**  The solve has to nucleate from that single site
        and grow outward, which is a much harder start than Wang's given ring or
        Sudoku's 24 clues, and it is a strong argument for seeding noise on the
        belief plane or for training the field variant first.

        Energy landmarks at the 8x8 nnn default, everything rendered at the
        house magnitude ``+-1`` with ``gamma = 3``::

            ground state                     +0.0053   (exactly 0.0 once saturated)
            ground state, one spin flipped   +0.0336
            a greedy quench's answer         +0.0901
            random +-1                       +0.5711
            flat beliefs / the seed          +0.5333
        """
        squeeze = torch.as_tensor(state).dim() == 3
        insts = [instances] if isinstance(instances, IsingInstance) else list(instances)
        if self.aggregate in ("sum", "mean"):
            e = self.soft_energy(state, instances)
            e0 = torch.tensor([i.e0 for i in insts], device=e.device, dtype=e.dtype)
            nb = torch.tensor([max(i.n_bonds, 1) for i in insts],
                              device=e.device, dtype=e.dtype)
            out = (e - e0) / nb
        else:
            res, wts = self.residuals(state, instances)
            out = self._aggregate(res, wts)
        return out[0] if squeeze else out

    def _aggregate(self, res, wts):
        """The worst-constraint ablations.  See ``energy`` for why they lose."""
        w = wts.sum(dim=1).clamp(min=EPS)
        if self.aggregate == "max":
            return res.masked_fill(wts <= 0, float("-inf")).amax(dim=1)
        if self.aggregate == "logprod":
            a = (1.0 - res).clamp(min=1e-6)
            return (-torch.log(a) * wts).sum(dim=1) / w
        z = (self.beta * res).masked_fill(wts <= 0, float("-inf"))
        return (torch.logsumexp(z, dim=1) - torch.log(w)) / self.beta

    def decisiveness(self, state: torch.Tensor, margin: float = 1.0):
        """Penalty for beliefs that never leave the fence.

        A ground state is a vertex of the belief cube, so a solution has to
        saturate; a tiny, correctly-ordered belief decodes fine and is wrecked by
        the next update.  ``energy`` already rewards saturation (the multilinear
        minimum is at a vertex) but only weakly near ``m = 0``, which is exactly
        where a rollout starts.  This is the term that makes settling mean
        something, and it is what the confidence channel should co-evolve with.
        """
        x = torch.as_tensor(state)
        if x.dim() == 3:
            x = x[None]
        b = x[:, self.belief_plane]
        return torch.relu(margin - b.abs()).pow(2).flatten(1).mean(1)

    def loss_terms(self, state, instances, w_decisive=0.05):
        """``(loss, extras)`` — the differentiable number and the honest one."""
        energy = self.energy(state, instances)
        loss = energy + w_decisive * self.decisiveness(state)
        insts = [instances] if isinstance(instances, IsingInstance) else list(instances)
        x = torch.as_tensor(state)
        if x.dim() == 3:
            x = x[None]
        exact = [self.violations(i, self.decode(x[b])) for b, i in enumerate(insts)]
        return loss, {"energy": energy.detach(),
                      "exact_violations": exact,
                      "solved": [self.is_solved(i, self.decode(x[b]))
                                 for b, i in enumerate(insts)]}

    # ── reporting ────────────────────────────────────────────────────────────

    def describe(self) -> str:
        return (f"{self.name} {self.rows}x{self.cols} {self.variant}  "
                f"planes={self.n_planes}  difficulty={self.difficulty}  "
                f"rounds={self.rounds}  min_rollout={self.min_rollout()} "
                f"(floor {self.min_rollout_floor()})")
