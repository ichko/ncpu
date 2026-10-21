"""3SAT -> CLIQUE as morphogenesis — the Karp reduction drawn on a grid.

Why this is worth doing
-----------------------
A Karp reduction is a *gadget substitution*: every piece of the source instance
is replaced, in place, by a local widget, and the widgets are wired by rules
that only look at their immediate surroundings.  That is exactly what a cellular
automaton is.  So instead of asking an NCA to *solve* an NP-complete problem
(which it cannot do in a short rollout without collapsing P and NP), ask it to
*perform the reduction* — a polynomial-time, purely local job — and watch the
reduced instance grow out of the source instance.

The reduction (Karp 1972)
-------------------------
Given a 3SAT formula with ``m`` clauses, build a graph with ``3m`` nodes, one
per literal *occurrence* ``(i, a)`` = slot ``a`` of clause ``i``.  Put an edge
between ``(i, a)`` and ``(j, b)`` iff::

    i != j                    (different clauses)      <- geometry
    and lit_u != -lit_v       (not complementary)      <- content

The formula is satisfiable iff the graph has an ``m``-clique; because each
clause block is an independent set, an ``m``-clique must take exactly one node
per clause, and consistency of the chosen literals *is* a satisfying partial
assignment.  Both directions are constructive, so a clique is a certificate and
so is an assignment: see ``clique_to_assignment`` / ``assignment_to_clique``.

The biconditional ``sat  <=>  max_clique == m`` is the entire content of the
reduction, it is what every metric here rests on, and it is exact (checked on
600/600 instances across five shapes, including 174 unsatisfiable ones).  Note
that *nothing stronger* holds.  An unsatisfiable formula's maximum clique is
only guaranteed to be **smaller** than ``m``; it is *usually* ``m - 1`` but not
always -- measured 100% at n=5,m=22, 94.3% at n=6,m=27 and 97.0% at n=4,m=20,
with the remainder at ``m - 2`` or below.  Treat "m - 1" as a description of
the common case, never as a property; ``verify_reduction`` reports the measured
histogram rather than asserting one.

Why the adjacency matrix is the right picture
---------------------------------------------
Rendered as a ``3m x 3m`` adjacency matrix with the literal codes written along
the top and down the left, the reduction becomes a pure outer product::

        (header: code of occurrence v, n lanes deep)
        . . . .  c0a  c0b  c0c  c1a  c1b  c1c  ...
   c0a  . . . .   .    .    .    +    +    -
   c0b  . . . .   .    .    .    +    -    +
   ...  ^^^^^^^
        (header: code of occurrence u, n lanes wide)

Body cell ``(u, v)`` = +1 if the edge exists, -1 if not.  Broadcast the row
header right, broadcast the column header down, combine them where they meet.
That is a local, directional, size-agnostic dataflow — the same shape as the
verifier in ``ncpu.sat`` — and the result is checkable pixel-for-pixel against
``adjacency()``.

Depth, honestly
---------------
A 3x3 kernel moves information ONE PIXEL per step, so the required rollout is an
L1 distance in *pixels*, not in lattice cells: see ``CliqueLayout.min_rollout``.
Two things keep that affordable here.  First, the two broadcasts are orthogonal
and independent — the row code only travels along its row, the column code only
down its column — so the binding time is the max of the two journeys, about the
grid width, rather than their sum.  Second, a body cell carries a single bit, so
it needs far less pixel redundancy than the verifier's word discs: the default
lattice is r=1, step=3 (a 5-pixel plus per cell), which puts n=5, m=4 on a 54x54
grid with a 48-step light cone.  Keep m small and the lattice tight; the claim
that matters is not "trained big" but "the same rule, run unchanged on a bigger
instance", which is what the trainer's extrapolation eval measures.

Encoding: the signed one-hot literal code (chosen)
--------------------------------------------------
Each occurrence's header is a strip of ``n`` lanes carrying ``+1`` at the
literal's variable if the literal is positive, ``-1`` if negated, ``0``
elsewhere — literally one row of the ``ncpu.sat`` clause matrix restricted to a
single literal.  Then::

    complementary(u, v)  <=>  exists d:  rowcode[d] * colcode[d] < 0

a local product followed by an OR — the *same* primitive the verifier track uses
for "literal satisfied <=> sign x assignment > 0".  Sharing the primitive is the
point: the two tracks stay chainable and comparable.

Rejected alternatives, and why:

* **One grey level per variable** (variable j drawn at value j/n, sign as a
  separate lane).  Comparison becomes an equality test between analog values
  whose spacing shrinks as 1/n, so the margin degrades with instance size and
  the rule cannot extrapolate in n at all.  The whole appeal of a convolutional
  rule is that a rule learned at n=5 *runs* at n=20; an encoding that forbids
  that throws away the headline measurement.
* **Compact binary variable index** (``ceil(log2 n) + 1`` lanes).  Header depth
  grows like log n instead of n, which is a real saving for large n, but
  "same variable" becomes an AND over all index lanes rather than "one
  coincidence anywhere", i.e. a conjunction that must be *complete* to be
  correct — one wrong lane flips an edge.  The one-hot version has exactly one
  nonzero per strip, so the evidence is sparse, sign-carrying and unanimous.
  Kept as ``CompactCliqueLayout`` for ablation rather than discarded.

The ``i != j`` term: geometry, handed over
------------------------------------------
"Different clauses" carries no information about the formula — it depends only
on ``m`` — so the ``k x k`` diagonal blocks are *drawn* on the input screen
(``block_hint="diag"``, the default) exactly the way ``ncpu.orbit`` hands the
NCA a frozen map of the track.  Those cells are given, so they are excluded from
every score: per-cell accuracy and exact-adjacency accuracy are computed over
the off-block cells only, which are the ones that actually encode the formula.
``block_hint="none"`` blanks them instead and scores them, forcing the NCA to
derive block identity from the border/coordinates — a harder ablation, kept
available.  ``block_hint="wall"`` draws them at a third value so the picture
distinguishes "structurally absent" from "complementary".

Value conventions follow ``ncpu.sat`` / ``ncpu.subleq``: float32 screens,
background 0.0, false/no-edge -1.0, true/edge +1.0.  Formulas are DIMACS-style
tuples of tuples, literal ``+(j + 1)`` positive and ``-(j + 1)`` negated.

No scaffold carpet is needed here (unlike ``ncpu.subleq``): the body is dense
supervision covering the whole field, so there is already loss everywhere to
force long-range propagation.
"""

import math
import random

import numpy as np
import torch

from ncpu.sat import (
    BACKGROUND,
    BIT0_VALUE,
    BIT1_VALUE,
    random_ksat,
    sample_instance,
    solutions,
)

WALL_VALUE = 0.5   # block_hint="wall": structurally-absent (same clause) cells


# ── the reduction, as reference semantics ────────────────────────────────────

def occurrences(formula):
    """The literal-occurrence list ``[(clause_idx, slot, literal), ...]``.

    Node ``u`` of the reduced graph is entry ``u`` of this list, so with a
    fixed clause width ``k`` the node index is ``u = i * k + a``.
    """
    return [(i, a, lit)
            for i, clause in enumerate(formula)
            for a, lit in enumerate(clause)]


def adjacency(formula):
    """The ``(3m, 3m)`` boolean adjacency matrix of the reduced graph."""
    occ = occurrences(formula)
    N = len(occ)
    adj = np.zeros((N, N), dtype=bool)
    for u, (i, _, lu) in enumerate(occ):
        for v, (j, _, lv) in enumerate(occ):
            adj[u, v] = (i != j) and (lu != -lv)
    return adj


def sat_to_clique(formula):
    """The reduction itself: formula -> (occurrence list, adjacency matrix).

    The target clique size is ``len(formula)``; the formula is satisfiable iff
    the graph has a clique that big (``verify_reduction`` checks exactly this).
    """
    return occurrences(formula), adjacency(formula)


# ── exact max clique (small graphs) ──────────────────────────────────────────

def max_clique(adj, lower_bound=None):
    """An exact maximum clique, as a sorted list of node indices.

    Bitset branch-and-bound with a greedy-colouring bound (Tomita's MCQ): a
    clique can take at most one node per colour class, so the number of colours
    needed for the remaining candidates bounds how much they can still add.

    The colouring bound is what makes the *negative* side affordable.  Proving
    "no m-clique" means exhausting the search, and on these graphs the greedy
    colouring rediscovers the clause blocks (each block is an independent set),
    so the bound is essentially "how many clauses are still open" — the search
    collapses to a DPLL-shaped walk instead of a blind 3**m enumeration.

    ``lower_bound`` seeds ``best`` with a known clique, which prunes harder when
    you only care whether an ``m``-clique exists.
    """
    N = int(adj.shape[0])
    nbr = [0] * N
    for i in range(N):
        bits = 0
        for j in range(N):
            if adj[i, j] and i != j:
                bits |= 1 << j
        nbr[i] = bits

    best = list(lower_bound) if lower_bound else []

    def colour_sort(cand):
        """Greedy colouring of ``cand``: nodes in non-decreasing colour, with
        the colour number of each as its bound."""
        order, bounds = [], []
        colour = 0
        left = cand
        while left:
            colour += 1
            avail = left
            while avail:
                v = (avail & -avail).bit_length() - 1
                avail &= ~(1 << v) & ~nbr[v]      # v's neighbours can't share it
                left &= ~(1 << v)
                order.append(v)
                bounds.append(colour)
        return order, bounds

    def expand(chosen, cand):
        nonlocal best
        order, bounds = colour_sort(cand)
        for i in range(len(order) - 1, -1, -1):
            if len(chosen) + bounds[i] <= len(best):
                return                            # nothing left here can win
            v = order[i]
            chosen.append(v)
            nxt = cand & nbr[v]
            if nxt:
                expand(chosen, nxt)
            elif len(chosen) > len(best):
                best = list(chosen)
            chosen.pop()
            cand &= ~(1 << v)

    if N:
        expand([], (1 << N) - 1)
    return sorted(best)


def max_clique_size(adj):
    return len(max_clique(adj))


def is_clique(adj, nodes):
    """True iff ``nodes`` are pairwise adjacent (and distinct)."""
    nodes = list(nodes)
    if len(set(nodes)) != len(nodes):
        return False
    return all(adj[u, v] for i, u in enumerate(nodes) for v in nodes[i + 1:])


# ── certificate correspondence, both ways ────────────────────────────────────

def clique_to_assignment(occ, nodes, n_vars, fill=0):
    """Read a satisfying assignment off a clique.

    The clique's literals are pairwise non-complementary, so they are jointly
    satisfiable; variables no chosen literal mentions take ``fill``.  Returns a
    tuple of ``n_vars`` 0/1 ints, or ``None`` if the nodes are inconsistent
    (which cannot happen for a genuine clique).
    """
    values = {}
    for u in nodes:
        lit = occ[u][2]
        var, want = abs(lit) - 1, 1 if lit > 0 else 0
        if values.setdefault(var, want) != want:
            return None
    return tuple(values.get(v, fill) for v in range(n_vars))


def assignment_to_clique(formula, assignment, k=3):
    """Read a clique off a satisfying assignment: one satisfied literal per
    clause, taken as node ``i * k + a``.  ``None`` if some clause is unsatisfied.
    """
    nodes = []
    for i, clause in enumerate(formula):
        for a, lit in enumerate(clause):
            value = assignment[abs(lit) - 1]
            if (lit > 0) == bool(value):
                nodes.append(i * k + a)
                break
        else:
            return None
    return nodes


# ── answer preservation: the property that makes it a reduction ──────────────

def verify_reduction(trials=200, n_vars=5, m_clauses=4, k=3, rng=random, verbose=False):
    """Check on random instances that the reduction preserves the answer.

    For each formula: ``solutions()`` says satisfiable  <=>  ``max_clique_size``
    equals ``m``; and both certificates round-trip (a max clique decodes to a
    satisfying assignment, a solution encodes to a genuine ``m``-clique).

    That biconditional is the only thing asserted, because it is the only thing
    that is true.  The UNSAT side's clique sizes are *reported* -- as
    ``unsat_clique_hist`` plus the measured share that landed at ``m - 1`` --
    and never asserted: the common case is ``m - 1``, but a real fraction of
    unsatisfiable instances (a few percent at the shapes measured) sits lower.

    Returns a dict of counts; raises AssertionError on any mismatch.
    """
    stats = {"trials": 0, "sat": 0, "unsat": 0,
             "max_clique_hist": {}, "unsat_clique_hist": {}}
    for _ in range(trials):
        formula = random_ksat(n_vars, m_clauses, k, rng)
        occ, adj = sat_to_clique(formula)
        sat = len(solutions(formula, n_vars)) > 0
        clique = max_clique(adj)
        size = len(clique)

        assert size <= m_clauses, f"clique {size} > m={m_clauses}: {formula}"
        assert is_clique(adj, clique), f"max_clique returned a non-clique: {formula}"
        assert (size == m_clauses) == sat, (
            f"answer NOT preserved: sat={sat} but max clique={size}/{m_clauses} "
            f"for {formula}"
        )

        if sat:
            assignment = clique_to_assignment(occ, clique, n_vars)
            assert assignment is not None, f"clique decoded inconsistently: {formula}"
            from ncpu.sat import evaluate
            assert evaluate(formula, assignment) == 1, (
                f"clique -> assignment does not satisfy: {formula} {assignment}"
            )
            sols = solutions(formula, n_vars, cap=1)
            back = assignment_to_clique(formula, tuple(int(b) for b in sols[0]), k)
            assert back is not None and len(back) == m_clauses and is_clique(adj, back), (
                f"assignment -> clique is not an m-clique: {formula}"
            )

        stats["trials"] += 1
        stats["sat" if sat else "unsat"] += 1
        stats["max_clique_hist"][size] = stats["max_clique_hist"].get(size, 0) + 1
        if not sat:
            h = stats["unsat_clique_hist"]
            h[size] = h.get(size, 0) + 1
        if verbose:
            print(f"  sat={int(sat)}  max_clique={size}/{m_clauses}  {formula}")
    # measured, not asserted: how often UNSAT landed one short of m
    stats["unsat_at_m_minus_1"] = stats["unsat_clique_hist"].get(m_clauses - 1, 0)
    stats["unsat_at_m_minus_1_frac"] = (
        stats["unsat_at_m_minus_1"] / stats["unsat"] if stats["unsat"] else None)
    return stats


# ── geometry + rendering ─────────────────────────────────────────────────────

class CliqueLayout:
    """Screens for the 3SAT -> CLIQUE reduction at a fixed (n, m) shape.

    Lattice of ``(D + N)`` columns by ``(D + N)`` rows, where ``N = k * m`` is
    the node count and ``D`` is the header depth (``n`` for the signed one-hot
    code).  The top-left ``D x D`` corner is unused background::

        columns:  [ D header lanes ][ N body columns ]
        rows:     [ D header lanes ][ N body rows    ]

    The left header row ``u`` and the top header column ``u`` both carry the
    code of occurrence ``u``, so the picture is literally "row code x column
    code -> body cell".  Geometry is a plain lattice, so a rule trained here
    runs unchanged on a bigger layout: build a second ``CliqueLayout`` with
    larger n/m to measure instance-size extrapolation.
    """

    def __init__(self, n_vars=5, m_clauses=4, k=3, r=1, gap=None, pad=None,
                 block_hint="diag"):
        if block_hint not in ("diag", "wall", "none"):
            raise ValueError(f"block_hint must be diag|wall|none, got {block_hint!r}")
        self.n = int(n_vars)
        self.m = int(m_clauses)
        self.k = int(k)
        self.N = self.k * self.m                 # nodes in the reduced graph
        self.D = self.code_len()                 # header depth (lanes)
        self.block_hint = block_hint

        # A body cell carries ONE bit (edge / no edge), not a multi-bit word, so
        # it needs far less pixel redundancy than the verifier's discs -- and the
        # lattice step multiplies straight into the required rollout depth (see
        # min_rollout), so the default lattice is deliberately tight: r=1 draws a
        # 5-pixel plus, step=3.
        self.r = int(r)
        self.gap = int(gap) if gap is not None else max(1, self.r)
        self.pad = int(pad) if pad is not None else max(1, 2 * self.r)
        self.step = 2 * self.r + self.gap

        cells = self.D + self.N
        origin = self.pad + self.r
        axis = [origin + c * self.step for c in range(cells)]
        self.head_x, self.body_x = axis[: self.D], axis[self.D:]
        self.head_y, self.body_y = axis[: self.D], axis[self.D:]
        self.W = self.H = axis[-1] + self.r + self.pad

        # which body cells are the NCA's actual job (see the module docstring)
        u = np.arange(self.N)[:, None]
        v = np.arange(self.N)[None, :]
        self.same_block = (u // self.k) == (v // self.k)
        self.scored = np.ones((self.N, self.N), dtype=bool) if block_hint == "none" \
            else ~self.same_block

        self._decoders = {}                      # device -> (index, counts)

    # ── the literal code ─────────────────────────────────────────────────────

    def code_len(self):
        """Header depth: one lane per variable."""
        return self.n

    def code(self, lit):
        """Signed one-hot: +1 at the literal's variable if positive, -1 if
        negated, 0 elsewhere."""
        c = np.zeros(self.D, dtype=np.float32)
        c[abs(lit) - 1] = BIT1_VALUE if lit > 0 else BIT0_VALUE
        return c

    # ── primitives ───────────────────────────────────────────────────────────

    def _disc(self, screen, center, value):
        cx, cy = center
        ys, xs = np.ogrid[: self.H, : self.W]
        screen[(xs - cx) ** 2 + (ys - cy) ** 2 <= self.r ** 2] = value

    def _mask(self, center):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        self._disc(m, center, 1.0)
        return m

    def _draw_headers(self, screen, formula):
        for u, (_, _, lit) in enumerate(occurrences(formula)):
            c = self.code(lit)
            for d in range(self.D):
                if c[d] == 0.0:
                    continue                     # 0 == background, nothing to draw
                self._disc(screen, (self.head_x[d], self.body_y[u]), c[d])   # left
                self._disc(screen, (self.body_x[u], self.head_y[d]), c[d])   # top

    def _draw_blocks(self, screen):
        """The instance-independent ``i != j`` geometry, when it is handed over."""
        if self.block_hint == "none":
            return
        value = WALL_VALUE if self.block_hint == "wall" else BIT0_VALUE
        for u in range(self.N):
            for v in range(self.N):
                if self.same_block[u, v]:
                    self._disc(screen, (self.body_x[v], self.body_y[u]), value)

    def _draw_body(self, screen, adj):
        """Every body cell the NCA is responsible for: the off-block cells
        always, plus the diagonal blocks when they were not handed over."""
        for u in range(self.N):
            for v in range(self.N):
                if self.same_block[u, v] and self.block_hint != "none":
                    continue                     # already drawn by _draw_blocks
                self._disc(screen, (self.body_x[v], self.body_y[u]),
                           BIT1_VALUE if adj[u, v] else BIT0_VALUE)

    # ── screens ──────────────────────────────────────────────────────────────

    def input_screen(self, formula):
        """Headers drawn, body blank (bar the block geometry)."""
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        self._draw_headers(s, formula)
        self._draw_blocks(s)
        return s

    def target_screen(self, formula, adj=None):
        """Headers reproduced, body filled with the reduced instance."""
        s = self.input_screen(formula)
        self._draw_body(s, adjacency(formula) if adj is None else adj)
        return s

    def screens_for(self, formula):
        adj = adjacency(formula)
        return self.input_screen(formula), self.target_screen(formula, adj), adj

    # ── masks + decoding ─────────────────────────────────────────────────────

    def body_mask(self, scored_only=True):
        """(H, W) union of the body discs the NCA is scored/weighted on."""
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for u in range(self.N):
            for v in range(self.N):
                if scored_only and not self.scored[u, v]:
                    continue
                self._disc(m, (self.body_x[v], self.body_y[u]), 1.0)
        return torch.from_numpy(m)

    def header_mask(self):
        """(H, W) union of the header discs — what must be *preserved*."""
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for u in range(self.N):
            for d in range(self.D):
                self._disc(m, (self.head_x[d], self.body_y[u]), 1.0)
                self._disc(m, (self.body_x[u], self.head_y[d]), 1.0)
        return torch.from_numpy(m)

    def scored_mask(self):
        """(N, N) bool tensor: the body cells that count."""
        return torch.from_numpy(self.scored.copy())

    def cell_index(self):
        """(H, W) long: body-cell id ``u * N + v`` per pixel, ``N*N`` elsewhere.

        Decoding by index-add rather than by a stack of ``N**2`` masks keeps the
        cost O(H*W) per frame, which matters once N grows.
        """
        idx = np.full((self.H, self.W), self.N * self.N, dtype=np.int64)
        ys, xs = np.ogrid[: self.H, : self.W]
        for u in range(self.N):
            for v in range(self.N):
                cx, cy = self.body_x[v], self.body_y[u]
                idx[(xs - cx) ** 2 + (ys - cy) ** 2 <= self.r ** 2] = u * self.N + v
        return torch.from_numpy(idx)

    def _decoder(self, device):
        key = str(device)
        if key not in self._decoders:
            idx = self.cell_index().to(device).reshape(-1)
            counts = torch.bincount(idx, minlength=self.N * self.N + 1).float()
            self._decoders[key] = (idx, counts[:-1].clamp(min=1.0))
        return self._decoders[key]

    def decode_body(self, frames):
        """Mask-average every body cell.  (..., H, W) float -> (..., N, N)."""
        idx, counts = self._decoder(frames.device)
        flat = frames.reshape(-1, self.H * self.W).float()
        sums = torch.zeros(flat.shape[0], self.N * self.N + 1,
                           device=frames.device, dtype=flat.dtype)
        sums.index_add_(1, idx, flat)
        per = sums[:, :-1] / counts
        return per.reshape(*frames.shape[:-2], self.N, self.N)

    def decode_adjacency(self, frames):
        """(..., H, W) -> (..., N, N) bool: threshold each body cell at 0.

        Same-block cells are forced False when they are given geometry, so the
        result is always a legal adjacency matrix of the reduced graph.
        """
        adj = self.decode_body(frames) > 0
        if self.block_hint != "none":
            adj = adj & torch.from_numpy(~self.same_block).to(adj.device)
        return adj

    # ── batches ──────────────────────────────────────────────────────────────

    def sample_batch(self, batch_size, device=None, rng=random, want="any"):
        """(inp, tgt, adj, formulas).

        Note the reduction is defined for *every* formula, satisfiable or not —
        no balancing is needed, unlike the verifier's batches.  ``want`` is
        passed through to ``ncpu.sat.sample_instance`` only if you want to bias
        the training distribution (e.g. ``"sat"`` so the grown graphs really do
        contain an m-clique).
        """
        inputs, targets, adjs, formulas = [], [], [], []
        for _ in range(batch_size):
            if want == "any":
                formula = random_ksat(self.n, self.m, self.k, rng)
            else:
                formula, _ = sample_instance(self.n, self.m, self.k, rng, want=want)
            inp, tgt, adj = self.screens_for(formula)
            inputs.append(inp)
            targets.append(tgt)
            adjs.append(adj)
            formulas.append(formula)
        inp = torch.from_numpy(np.stack(inputs))
        tgt = torch.from_numpy(np.stack(targets))
        adj = torch.from_numpy(np.stack(adjs))
        if device is not None:
            inp, tgt, adj = inp.to(device), tgt.to(device), adj.to(device)
        return inp, tgt, adj, formulas

    # ── info ─────────────────────────────────────────────────────────────────

    def min_rollout(self):
        """Steps a body cell needs before both its headers can have reached it.

        A 3x3 kernel moves information ONE PIXEL per step, so this is an L1
        distance in pixels and it scales with ``step``, not with the cell count.

        The saving grace of the outer-product layout is that the two journeys
        are *orthogonal and independent*: the row code travels along its own row
        (pure dx) while the column code travels down its own column (pure dy),
        so the binding time is the max of the two, not their sum.  The worst
        cell is the far corner of the body, reached from the outermost header
        lane::

            (D + N - 1) * step   pixels

        which is about ``W`` -- not ``W + H``.  Anything below this makes the
        task physically impossible rather than merely hard.
        """
        return max(max(self.body_x) - min(self.head_x),
                   max(self.body_y) - min(self.head_y))

    def suggested_rollout(self, slack=1.25):
        """``min_rollout`` plus room for the combine and any indirect routing."""
        return int(math.ceil(slack * self.min_rollout()))

    def edge_baseline(self):
        """Per-cell accuracy of the constant rule "every off-block pair is an
        edge".  Two occurrences in different clauses are complementary with
        probability ``P(same variable) * P(opposite sign) = 1 / (2n)``, so this
        baseline is high -- which is exactly why exact-adjacency accuracy, not
        per-cell accuracy, is the metric that means "performed the reduction".
        """
        return 1.0 - 1.0 / (2.0 * self.n)

    def describe(self):
        scored = int(self.scored.sum())
        return (f"{type(self).__name__} {self.W}x{self.H}  n={self.n} vars  "
                f"m={self.m} clauses  k={self.k}  nodes={self.N}  header={self.D} lanes  "
                f"block_hint={self.block_hint}  scored cells={scored}/{self.N ** 2}  "
                f"r={self.r} step={self.step}  min_rollout={self.min_rollout()}px "
                f"(suggested {self.suggested_rollout()})  "
                f"all-edges baseline={self.edge_baseline():.3f}")


class CompactCliqueLayout(CliqueLayout):
    """Same geometry, compact header: variable index in binary + a sign lane.

    Header depth is ``ceil(log2 n) + 1`` instead of ``n``, so the screen stays
    small as n grows.  The price is the comparison primitive: "same variable"
    is now an AND over every index lane (all products positive) instead of "one
    nonzero coincidence anywhere", so a single mis-propagated lane flips an
    edge, and the code is dense rather than sparse.  Carried as the ablation
    that tests whether the choice of code is what makes the task learnable.

    Lanes ``0..b-1`` hold the variable index MSB-first as +-1; lane ``b`` holds
    the sign.  Complementary <=> all index lanes agree AND the sign lanes differ.
    """

    def code_len(self):
        return max(1, math.ceil(math.log2(max(self.n, 2)))) + 1

    def code(self, lit):
        bits = self.D - 1
        var = abs(lit) - 1
        c = np.empty(self.D, dtype=np.float32)
        for d in range(bits):
            c[d] = BIT1_VALUE if (var >> (bits - 1 - d)) & 1 else BIT0_VALUE
        c[bits] = BIT1_VALUE if lit > 0 else BIT0_VALUE
        return c


# ── degenerate baseline: what an input-independent rule already scores ───────

def baseline_scores(layout, trials=500, rng=random):
    """Score the best input-independent predictor, "every off-block pair is an
    edge", on random instances of ``layout``'s shape.

    Tier 0 of the eval protocol: every model number has to be read next to this
    one.  Per-cell accuracy starts around ``1 - 1/(2n)``, which is why it is a
    poor headline; exact-adjacency accuracy of the same rule is
    ``(1 - 1/(2n)) ** (#scored cells)``, i.e. essentially zero, which is why it
    is the right one.
    """
    cell, exact = [], []
    for _ in range(trials):
        formula = random_ksat(layout.n, layout.m, layout.k, rng)
        adj = adjacency(formula)[layout.scored]
        cell.append(float(adj.mean()))
        exact.append(float(adj.all()))
    return {"cell": float(np.mean(cell)),
            "exact": float(np.mean(exact)),
            "analytic_cell": layout.edge_baseline(),
            "scored_cells": int(layout.scored.sum()),
            "trials": trials}


# ── render round-trip: the picture really is the reduction ───────────────────

def verify_render(layout, trials=20, rng=random):
    """Decode the rendered target back and compare it to ``adjacency()``.

    Guards the thing that would silently invalidate every training metric: a
    layout whose pixels do not actually mean what the reference says.
    """
    for _ in range(trials):
        formula = random_ksat(layout.n, layout.m, layout.k, rng)
        inp, tgt, adj = layout.screens_for(formula)
        got = layout.decode_adjacency(torch.from_numpy(tgt)).numpy()
        ref = adj & ~layout.same_block if layout.block_hint != "none" else adj
        assert (got == ref).all(), f"render/decode mismatch for {formula}"
        # the input must NOT already contain the answer on the scored cells
        blank = layout.decode_body(torch.from_numpy(inp)).numpy()
        assert np.allclose(blank[layout.scored], BACKGROUND), \
            "input screen leaks body values on scored cells"
    return trials


# ── Tier 0 item 2: shortcut probes ───────────────────────────────────────────

def _summary_features(layout, formula):
    """Statistics a NON-computing model could read off the formula: positive
    literals per clause, occurrences and signed occurrences per variable, plus
    the totals.  Deliberately global — none of them says anything about one
    particular *pair* of occurrences, which is what an edge bit is."""
    pos_per_clause = [sum(1 for lit in c if lit > 0) for c in formula]
    occ_per_var, sign_per_var = [], []
    for v in range(layout.n):
        lits = [lit for c in formula for lit in c if abs(lit) - 1 == v]
        occ_per_var.append(len(lits))
        sign_per_var.append(sum(1 if lit > 0 else -1 for lit in lits))
    return (pos_per_clause + occ_per_var + sign_per_var
            + [sum(pos_per_clause), float(np.mean(occ_per_var))])


def shortcut_probe(layout, samples=1500, seed=0, split=0.7, n_cells=3,
                   max_iter=2000, verbose=True):
    """Tier 0 item 2 of ``docs/karp_eval_protocol.md``, for the reduction.

    Can a *linear* model read a scored edge bit straight off the raw input
    pixels, or off summary statistics of the formula?  If it can, the body of
    the target is partly inferable without performing the reduction, and any
    later NCA number would not be measuring the intended computation.

    **Every probe is scored against ITS OWN target's baseline.**  That matters
    more here than anywhere else in this repo.  Averaged over the scored cells
    an edge bit is positive with probability ``1 - 1/(2n)`` (``edge_baseline``,
    ~0.90 at n=5), so per-cell accuracy already carries a ~0.9 constant
    baseline — and the *per-cell* rate is not even uniform: ``random_ksat``
    emits each clause with its literals sorted, so slot 0 tends to hold the
    most negative literal and slot k-1 the most positive.  A same-slot pair is
    therefore far less likely to be complementary (~0.98 edge rate at n=5)
    than an opposite-end pair.  Scoring any probe against a single global
    number would manufacture a leak that is not there; only the lift over the
    same cell's own majority rate on the same held-out split means anything.

    Because of that skew the probed cells are chosen from the *measured*
    per-cell positive rates rather than by position: the most balanced scored
    cell (where a leak would be easiest to see and hardest to dismiss), the
    median cell, and the most skewed one.

    A **positive control** is included so a null result reads as "nothing to
    find" rather than "the probe is broken".  Given the ``D`` per-lane PRODUCTS
    of the two header codes the edge bit is linearly separable: with signed
    one-hot codes ``sum_d row_d * col_d`` is ``-1`` exactly when the two
    literals are complementary and ``0`` or ``+1`` otherwise.  That probe
    should reach ~1.00.  The multiplication is the only thing the raw-pixel
    probe lacks — and it is exactly the local computation the NCA has to learn.

    Returns a dict; ``leak`` lists any non-control probe whose lift exceeds
    0.10 on any probed cell.
    """
    from sklearn.linear_model import LogisticRegression

    rng = random.Random(seed)
    cells = [(int(u), int(v)) for u, v in zip(*np.nonzero(layout.scored))]
    if not cells:
        raise ValueError("layout has no scored cells to probe")

    pixels, summ, codes, labels = [], [], [], []
    for _ in range(samples):
        formula = random_ksat(layout.n, layout.m, layout.k, rng)
        pixels.append(layout.input_screen(formula).reshape(-1))
        summ.append(_summary_features(layout, formula))
        codes.append(np.stack([layout.code(lit) for _, _, lit in occurrences(formula)]))
        adj = adjacency(formula)
        labels.append([adj[u, v] for u, v in cells])

    X_pix = np.asarray(pixels, dtype=np.float32)
    X_sum = np.asarray(summ, dtype=np.float32)
    codes = np.asarray(codes, dtype=np.float32)           # (S, N, D)
    Y = np.asarray(labels, dtype=np.int64)                # (S, n_scored)
    cut = int(split * samples)

    # pick by measured balance, not by position (see the docstring)
    rate = Y.mean(0)
    order = np.argsort(np.abs(rate - 0.5))                # most balanced first
    tags = ["most balanced", "median", "most skewed"]
    take = [0, len(order) // 2, len(order) - 1][:max(1, n_cells)]
    picked = [(cells[order[i]], tags[j] if j < len(tags) else f"#{j}")
              for j, i in enumerate(take)]

    def one(features, y, label):
        """Accuracy next to the majority-class rate of the SAME held-out y."""
        y_tr, y_te = y[:cut], y[cut:]
        base = float(max(y_te.mean(), 1.0 - y_te.mean()))
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            if verbose:
                print(f"   {label:46s} degenerate (one class; baseline {base:.3f})")
            return {"acc": None, "baseline": base, "lift": None,
                    "degenerate": True, "pos_rate": float(y_te.mean())}
        clf = LogisticRegression(max_iter=max_iter, C=1.0)
        clf.fit(features[:cut], y_tr)
        acc = float(clf.score(features[cut:], y_te))
        if verbose:
            print(f"   {label:46s} {acc:.3f}   (own baseline {base:.3f}, "
                  f"lift {acc - base:+.3f})")
        return {"acc": acc, "baseline": base, "lift": acc - base,
                "degenerate": False, "pos_rate": float(y_te.mean())}

    if verbose:
        print(f"   {samples} instances, {cut}/{samples - cut} train/test split; "
              f"{len(cells)} scored cells, per-cell edge rate "
              f"{rate.min():.3f}..{rate.max():.3f} (mean {rate.mean():.3f}, "
              f"analytic {layout.edge_baseline():.3f})")
    results, leak = [], []
    for (u, v), tag in picked:
        c = cells.index((u, v))
        name = f"cell ({u},{v}) [{tag}]"
        entry = {"cell": [u, v], "which": tag}
        entry["pixels"] = one(X_pix, Y[:, c], f"linear on raw pixels -> {name}")
        entry["summary"] = one(X_sum, Y[:, c], f"summary stats -> {name}")
        entry["control_products"] = one(codes[:, u] * codes[:, v], Y[:, c],
                                        f"CONTROL row x col products -> {name}")
        for key in ("pixels", "summary"):
            if entry[key]["lift"] is not None and entry[key]["lift"] > 0.10:
                leak.append(f"{key} @ cell ({u},{v})")
        results.append(entry)

    control = [r["control_products"] for r in results]
    return {"samples": samples, "split": split, "n_cells": len(results),
            "scored_cells": int(layout.scored.sum()),
            "edge_baseline_analytic": layout.edge_baseline(),
            "edge_rate_per_cell": {"min": float(rate.min()), "max": float(rate.max()),
                                   "mean": float(rate.mean())},
            "probes": results, "leak": leak,
            "control_acc": [c["acc"] for c in control],
            # a cell whose own baseline is already ~0.98 cannot show the control
            # off, so the bar is "reaches its baseline and at least 0.97"
            "control_ok": all(c["acc"] is not None
                              and c["acc"] >= max(c["baseline"], 0.97) - 1e-9
                              for c in control)}
