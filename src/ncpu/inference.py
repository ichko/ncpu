"""Reference message-passing solvers: BP, warning propagation, survey propagation.

Why this file is not inside a task
----------------------------------
This phase asks the NCA to **rediscover survey propagation natively** — beliefs
that propagate and reconcile inside the dynamics, with SP's hard-fix-and-restart
decimation replaced by a learned stiffening dynamic.  You cannot claim a model
learned something SP-like without the SP curve to compare it against, and you
cannot tell "learned" from "memorised" without a non-learned solver on the same
instances.  So the references live here, one level below any single task, and
every track can call them.

What is in here
---------------
``IsingModel``            a pairwise Ising problem on an arbitrary graph:
                          ``H(s) = -sum_ij J_ij s_i s_j - sum_i h_i s_i``.
                          Track I builds one from a lattice; nothing here knows
                          about lattices.
``belief_propagation``    sum-product BP at inverse temperature ``beta``, in the
                          cavity-field parameterisation.  One iteration is one
                          hop, which is why one BP iteration is the natural unit
                          of NCA depth (see ``ising.IsingLattice.min_rollout``).
``warning_propagation``   the ``beta -> inf`` limit (min-sum / max-product).  With
                          unit couplings the messages are integers in
                          ``{-1, 0, +1}``: "push down", "no opinion", "push up".
                          That trit alphabet is what makes SP cheap below.
``survey_propagation``    1RSB survey propagation at reweighting parameter ``y``:
                          messages are *distributions over warnings*, so the
                          third symbol is literally "this bond exerts no field in
                          this cluster".  Site surveys give ``P(spin frozen)``.
``bp_decimation`` /       the classic outer loops: propagate, hard-fix the most
``sp_decimation``         polarised spin, repeat.  **This outer loop is exactly
                          what the phase forbids the NCA** — it is here as the
                          algorithm to beat, and as the definition of the
                          quantity a stiffening confidence channel should hold.
``greedy_quench``         synchronous/asynchronous local search.  The honest
                          "trivial rule" control and the polish step SP-style
                          algorithms normally end with.
``discrete_bp``           generic sum-product BP for a pairwise MRF over a finite
                          alphabet, for the tracks whose variables are not spins
                          (Wang tiling is a pairwise MRF over tile types; Sudoku
                          is one over digits with "not equal" potentials).

The equations, and where they come from
---------------------------------------
Cavity field ``u_{i->j}``: the effective field site ``i`` exerts on ``j`` when
``j`` is removed from the graph.  With ``H_{i->j} = h_i + sum_{k in di\\j} u_{k->i}``:

    sum-product   u_{i->j} = (1/b) atanh( tanh(b J_ij) tanh(b H_{i->j}) )
    min-sum       u_{i->j} = sign(J_ij) sign(H) min(|J_ij|, |H|)
    marginal      m_i = tanh( b (h_i + sum_k u_{k->i}) )

The min-sum form follows from the message recursion
``E_{i->j}(s_j) = -|J_ij s_j + H_{i->j}| + const``, and the same line gives the
**energy shift** that SP reweights by:

    edge:  dE_{i->j} = -max(|H_{i->j}|, |J_ij|)      weight exp(y * max(|H|,|J|))
    site:  dE_i      = -|H_i|                        weight exp(y * |H_i|)

SP then propagates ``Q_{i->j}(u)`` over ``u in {-1,0,+1}``: convolve the incoming
warning distributions to get the distribution of ``H``, reweight it by
``exp(y dE)``, and push it through the min-sum map.  ``y -> inf`` keeps only the
lowest-energy clusters, which is the ground-state limit and the default here.
This is the Mezard-Parisi zero-temperature cavity method; the SAT-flavoured
version everyone quotes is the same object with a clause alphabet.

Two things that bite, both measured in ``scripts/preview_ising.py --survey``
----------------------------------------------------------------------------
* **The spin-flip symmetry is a fixed point.**  With ``h = 0`` the all-zero
  warning assignment satisfies every WP equation, and the symmetric survey
  ``Q(-1) = Q(+1)`` satisfies every SP equation.  Both are the *correct* answer
  to the question asked ("is this spin up?" has no answer when both a state and
  its global flip are ground states) and a useless answer to the question meant.
  Every entry point therefore takes ``clamped``: fix one reference spin and the
  degeneracy is gauge-fixed away.  ``ising.py`` clamps a reference spin in the
  instance itself for the same reason.
* **Short loops.**  BP assumes a locally tree-like graph.  A square lattice has
  loops of length 4 everywhere; add the diagonals and it has loops of length 3.
  Expect BP to converge badly and to be *wrong when it converges*; that is a
  property of the problem, not a bug in the code, and it is the contrast this
  track exists to measure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

EPS = 1e-12
TRIT = (-1, 0, 1)          # the warning alphabet: down, undecided, up


# ── the model ────────────────────────────────────────────────────────────────

class IsingModel:
    """``H(s) = -sum_(i,j) J_ij s_i s_j - sum_i h_i s_i`` on an arbitrary graph.

    Args:
        n:      number of spins.
        edges:  ``(E, 2)`` integer array of undirected edges, each listed once.
        J:      ``(E,)`` couplings.  ``J > 0`` is ferromagnetic (the two spins
                want to agree).
        h:      ``(n,)`` external fields, or None for zero field.

    Directed-edge bookkeeping is built once and reused by every solver:
    directed index ``2e`` is ``edges[e,0] -> edges[e,1]``, ``2e+1`` the reverse,
    and ``rev[d] = d ^ 1``.
    """

    def __init__(self, n: int, edges, J, h=None):
        self.n = int(n)
        self.edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
        self.J = np.asarray(J, dtype=np.float64).reshape(-1)
        if len(self.J) != len(self.edges):
            raise ValueError(f"{len(self.J)} couplings for {len(self.edges)} edges")
        self.h = (np.zeros(self.n) if h is None
                  else np.asarray(h, dtype=np.float64).reshape(-1).copy())
        if len(self.h) != self.n:
            raise ValueError(f"{len(self.h)} fields for {self.n} spins")

        E = len(self.edges)
        self.src = np.empty(2 * E, dtype=np.int64)
        self.dst = np.empty(2 * E, dtype=np.int64)
        self.src[0::2], self.dst[0::2] = self.edges[:, 0], self.edges[:, 1]
        self.src[1::2], self.dst[1::2] = self.edges[:, 1], self.edges[:, 0]
        self.rev = np.arange(2 * E, dtype=np.int64) ^ 1
        self.Jd = np.repeat(self.J, 2)
        self.degree = np.bincount(self.src, minlength=self.n)
        order = np.argsort(self.src, kind="stable")
        self.adj_ptr = np.concatenate([[0], np.cumsum(self.degree)]).astype(np.int64)
        self.adj_nbr = self.dst[order]
        self.adj_J = self.Jd[order]

        # incoming-edge slots per site, padded with a sentinel that carries the
        # neutral message (delta at u = 0).  ``slot_of[d]`` is where directed
        # edge d sits in its destination's list, which is what a leave-one-out
        # product needs.
        self.max_degree = int(self.degree.max()) if self.n else 0
        self.in_slots = np.full((self.n, max(self.max_degree, 1)), 2 * E, dtype=np.int64)
        self.slot_of = np.full(2 * E, -1, dtype=np.int64)
        fill = np.zeros(self.n, dtype=np.int64)
        for d in range(2 * E):
            i = self.dst[d]
            self.in_slots[i, fill[i]] = d
            self.slot_of[d] = fill[i]
            fill[i] += 1

    # ── exact, cheap quantities ──────────────────────────────────────────────

    def energy(self, s) -> float:
        """Energy of a spin configuration ``s`` in ``{-1, +1}^n``."""
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        bond = float(np.sum(self.J * s[self.edges[:, 0]] * s[self.edges[:, 1]]))
        return -bond - float(np.dot(self.h, s))

    def local_fields(self, s) -> np.ndarray:
        """``H_i = h_i + sum_j J_ij s_j`` — the field each spin sits in."""
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        return self.h + np.bincount(self.src, weights=self.Jd * s[self.dst],
                                    minlength=self.n)

    def unsatisfied(self, s) -> int:
        """Bonds with ``J_ij s_i s_j < 0``.  Not the same as excess energy once
        a field is present, and never zero on a frustrated instance."""
        s = np.asarray(s, dtype=np.float64).reshape(-1)
        return int(np.sum(self.J * s[self.edges[:, 0]] * s[self.edges[:, 1]] < 0))

    def is_integral(self) -> bool:
        """True when every coupling is +-1 and every field is an integer.

        ``warning_propagation`` and ``survey_propagation`` need this: it is what
        confines warnings to ``{-1, 0, +1}`` and keeps a survey a 3-vector.
        """
        return (np.all(np.abs(np.abs(self.J) - 1.0) < 1e-12)
                and np.all(np.abs(self.h - np.round(self.h)) < 1e-12))

    def with_clamps(self, clamped: dict) -> np.ndarray:
        """Fields with clamped spins pinned by a field that outweighs their bonds.

        ``clamped`` maps site -> +-1.  The pin has to beat every incident
        coupling at once or a clamp can be argued out of by its neighbours, so
        it is ``1 + sum_j |J_ij|`` rather than a magic constant.
        """
        h = self.h.copy()
        if clamped:
            strength = 1.0 + np.bincount(self.src, weights=np.abs(self.Jd),
                                         minlength=self.n)
            for i, v in clamped.items():
                h[i] = float(v) * strength[i]
        return h


# ── belief propagation ───────────────────────────────────────────────────────

@dataclass
class BPResult:
    """One propagation run.  ``converged`` is the number that matters."""
    u: np.ndarray                 # (2E,) cavity fields, directed
    fields: np.ndarray            # (n,)  total local field h_i + sum_k u_{k->i}
    marginals: np.ndarray         # (n,)  m_i in [-1, 1]
    converged: bool
    iters: int
    residual: float
    config: np.ndarray            # (n,)  sign(marginal), the decoded state
    energy: float
    history: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def __str__(self) -> str:
        return (f"BP {'conv' if self.converged else 'STALL'} in {self.iters} it "
                f"(res {self.residual:.2e})  E={self.energy:.1f}  "
                f"|m| mean {np.abs(self.marginals).mean():.3f}")


def belief_propagation(model: IsingModel, beta: float = 1.0, max_iters: int = 2000,
                       tol: float = 1e-6, damping: float = 0.5, rng=None,
                       clamped: Optional[dict] = None, init: str = "random",
                       init_scale: float = 0.1) -> BPResult:
    """Sum-product BP in cavity fields.  One iteration = one hop of information.

    ``beta`` is the inverse temperature.  Large ``beta`` approaches min-sum but
    saturates ``tanh`` long before it gets there — use ``warning_propagation``
    for the actual limit.  ``beta`` around 1 is the usual working point for a
    +-J glass, whose ferro/glass transition sits near ``beta ~ 0.7-1.0``.

    ``damping`` is not optional on a lattice.  Undamped BP on a graph full of
    length-4 loops oscillates in most draws; the damped iteration
    ``u <- (1-lam) u_new + lam u_old`` converges far more often and to the same
    fixed point when it converges.  ``damping=0`` is available so that the
    difference stays measurable rather than assumed.

    ``init="random"`` breaks the spin-flip symmetry by hand.  With ``h = 0``,
    ``u = 0`` is an exact fixed point of these equations, so a zero start
    converges instantly to "every spin is exactly undecided" — a true statement
    about the +-symmetric ensemble and a useless one about the instance.  Clamp
    a reference spin (``clamped={0: +1}``) if you want the symmetry broken
    honestly rather than by noise.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    h = model.with_clamps(clamped or {})
    D = len(model.src)
    if init == "zero":
        u = np.zeros(D)
    elif init == "random":
        u = init_scale * rng.standard_normal(D)
    else:
        u = np.asarray(init, dtype=np.float64).copy()

    tJ = np.tanh(beta * model.Jd)
    hist = np.empty(max_iters)
    converged, residual, it = False, np.inf, 0
    for it in range(1, max_iters + 1):
        tot = np.bincount(model.dst, weights=u, minlength=model.n)
        H = h[model.src] + tot[model.src] - u[model.rev]
        prod = np.clip(tJ * np.tanh(beta * H), -1.0 + 1e-15, 1.0 - 1e-15)
        u_new = np.arctanh(prod) / beta
        residual = float(np.max(np.abs(u_new - u))) if D else 0.0
        hist[it - 1] = residual
        u = (1.0 - damping) * u_new + damping * u
        if residual < tol:
            converged = True
            break

    tot = np.bincount(model.dst, weights=u, minlength=model.n)
    fields = h + tot
    m = np.tanh(beta * fields)
    cfg = np.where(m >= 0, 1, -1).astype(np.int8)
    for i, v in (clamped or {}).items():
        cfg[i] = v
    return BPResult(u=u, fields=fields, marginals=m, converged=converged,
                    iters=it, residual=residual, config=cfg,
                    energy=model.energy(cfg), history=hist[:it])


def warning_propagation(model: IsingModel, max_iters: int = 2000, rng=None,
                        clamped: Optional[dict] = None, init: str = "random",
                        schedule: str = "parallel") -> BPResult:
    """The ``beta -> inf`` limit: min-sum, integer warnings.

    ``u_{i->j} = sign(J) sign(H) min(|J|, |H|)``.  With unit couplings this is a
    trit: ``-1`` "j should be down", ``+1`` "j should be up", ``0`` "this bond
    has no opinion" — the state SP promotes to a first-class symbol.

    ``schedule="parallel"`` is the synchronous update, which is what an NCA step
    does and therefore the schedule whose iteration count sets ``min_rollout``.
    ``"sequential"`` sweeps in a random order and converges more often; it is
    kept as the control, not as the default, precisely because it is *not* what
    the hardware being modelled can do.

    Returns a ``BPResult`` whose ``marginals`` are ``sign(H_i)`` and whose
    ``fields`` are the integer local fields, so ``fields == 0`` marks the spins
    that WP leaves genuinely undecided.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    h = model.with_clamps(clamped or {})
    D = len(model.src)
    u = (rng.integers(-1, 2, size=D).astype(np.float64) if init == "random"
         else np.zeros(D))

    def _map(H, Jd):
        return np.sign(Jd) * np.sign(H) * np.minimum(np.abs(Jd), np.abs(H))

    converged, it = False, 0
    hist = np.empty(max_iters)
    for it in range(1, max_iters + 1):
        if schedule == "parallel":
            tot = np.bincount(model.dst, weights=u, minlength=model.n)
            H = h[model.src] + tot[model.src] - u[model.rev]
            u_new = _map(H, model.Jd)
            residual = float(np.max(np.abs(u_new - u))) if D else 0.0
            u = u_new
        elif schedule == "sequential":
            order = rng.permutation(D)
            residual = 0.0
            for d in order:
                i = model.src[d]
                inc = model.in_slots[i]
                s = float(np.sum(u[inc[inc < D]])) - u[model.rev[d]]
                v = _map(h[i] + s, model.Jd[d])
                residual = max(residual, abs(v - u[d]))
                u[d] = v
        else:
            raise ValueError(f"schedule must be 'parallel' or 'sequential', got {schedule!r}")
        hist[it - 1] = residual
        if residual == 0.0:
            converged = True
            break

    tot = np.bincount(model.dst, weights=u, minlength=model.n)
    fields = h + tot
    cfg = np.where(fields >= 0, 1, -1).astype(np.int8)
    for i, v in (clamped or {}).items():
        cfg[i] = v
    return BPResult(u=u, fields=fields, marginals=np.sign(fields), converged=converged,
                    iters=it, residual=hist[it - 1] if it else 0.0, config=cfg,
                    energy=model.energy(cfg), history=hist[:it])


# ── survey propagation ───────────────────────────────────────────────────────

def _poly_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Batched polynomial product: ``(n, L) x (n, 3) -> (n, L + 2)``.

    Distributions over integer-valued warnings are polynomials in a formal
    variable; convolving them is how the distribution of a *sum* of warnings is
    built.  Batched over sites so a whole SP sweep is a handful of numpy ops.
    """
    n, L = A.shape
    out = np.zeros((n, L + 2))
    for k in range(3):
        out[:, k:k + L] += A * B[:, k:k + 1]
    return out


@dataclass
class SPResult:
    """One survey-propagation run.

    ``Q`` is ``(2E, 3)``: for each directed edge, the probability that the
    warning it carries is ``-1``, ``0``, ``+1`` **in a randomly chosen cluster**
    of the (re-weighted) solution space.

    ``frozen`` is the quantity this whole track is about.  ``frozen[i]`` is
    ``P(H_i != 0)``: the probability that, in a randomly drawn cluster, spin
    ``i`` is *forced* — that its value is the same in every state of the
    cluster.  A confidence channel in an NCA that learns something SP-like is
    tracking exactly this number, and ``1 - frozen`` is SP's "free / undecided
    in this cluster" symbol lifted to a site.
    """
    Q: np.ndarray                 # (2E, 3) over TRIT
    site: np.ndarray              # (n, 3) site survey over sign(H_i)
    frozen: np.ndarray            # (n,) P(H_i != 0)
    bias: np.ndarray              # (n,) P(H>0) - P(H<0)
    converged: bool
    iters: int
    residual: float
    config: np.ndarray
    energy: float

    def __str__(self) -> str:
        return (f"SP {'conv' if self.converged else 'STALL'} in {self.iters} it "
                f"(res {self.residual:.2e})  E={self.energy:.1f}  "
                f"frozen {self.frozen.mean():.3f}")


def survey_propagation(model: IsingModel, y: Optional[float] = 0.0,
                       max_iters: int = 1000, tol: float = 1e-6,
                       damping: float = 0.5, rng=None,
                       clamped: Optional[dict] = None) -> SPResult:
    """1RSB survey propagation for a +-J Ising model.  ``y=None`` means ``y=inf``.

    **Do not use ``y=None`` on this model family.**  It is implemented, it is the
    textbook ground-state limit, and it is degenerate here: the reweighting
    ``exp(y max(|H|,1))`` at ``y=inf`` keeps only the largest ``|H|`` in the
    support, and on a trit alphabet every combination of incoming warnings has
    strictly positive probability, so the hard limit always selects the fully
    polarised tail however improbable it is.  Measured on a 10-spin chain where
    the exact frozen set is ``[1,1,1,1,1,1,0,0,0,0]``: ``y=0`` and ``y=1``
    reproduce it exactly, ``y=None`` returns all-frozen.  Finite ``y`` is the
    working regime and the default; ``y=0`` weights every cluster equally.

    Requires ``model.is_integral()``: unit couplings and integer fields, so that
    a warning is a trit and a survey is a 3-vector.  That is not a limitation of
    the method, it is what makes it cheap enough to run inside a difficulty
    survey — the general case needs messages that are densities over the reals.

    The update, for the directed edge ``i -> j``:

    1. convolve the surveys arriving at ``i`` from every neighbour except ``j``,
       giving ``P(sum of incoming warnings)``;
    2. shift by ``h_i`` to get ``P(H_{i->j})``;
    3. reweight by ``exp(y * max(|H|, 1))`` — the 1RSB energy-shift factor.  At
       ``y = inf`` this keeps only the largest ``|H|`` in the support, i.e. only
       the lowest-energy clusters, which is the ground-state limit;
    4. push through the min-sum map ``H -> sign(J) sign(H) min(1, |H|)``.

    ``y = 0`` weights every cluster equally — the "count clusters" reading, which
    theory says is right for a satisfiable CSP and wrong for an optimisation
    problem with a frustrated ground state.  It is nevertheless the DEFAULT,
    because measurement disagrees with theory here and the measurement wins.
    On 16 8x8 lattices (``preview_ising.py --survey``):

        y        converges   solved (one-shot)   excess    rho(frozen, exact)
        0.0          1.00           0.000          12         -0.09
        0.5          0.00           0.000          12         +0.12
        1.0          0.00           0.000          10         +0.09
        2.0          0.00           0.000          10         +0.02
        inf          0.00              -            -          (degenerate)

    and on 12 locally tree-like random 3-regular graphs of 18 spins, where the
    exact answer is brute-forced:

        y        converges   excess above E0    rho(frozen, exact)
        0.0          0.67          1.33               +0.52
        1.0          0.42          1.67               +0.05

    Raising ``y`` makes the reweighting concentrate the survey on polarised
    incoming combinations, which drives ``P(frozen)`` towards 1 and destroys
    both convergence and the frozen estimate.  ``y = 0`` is what the decimation
    loop below is measured with.

    The **decimation loop is deliberately not here** — see ``sp_decimation``.
    A model that reproduces SP's answers without that loop is the result this
    phase is chasing, so the loop stays a clearly separable component.
    """
    if not model.is_integral():
        raise ValueError("survey_propagation needs +-1 couplings and integer fields; "
                         "use belief_propagation for the general case")
    rng = np.random.default_rng(0) if rng is None else rng
    h = np.rint(model.with_clamps(clamped or {})).astype(np.int64)
    D, n = len(model.src), model.n
    md = max(model.max_degree, 1)

    Q = rng.random((D, 3)) + 0.05
    Q /= Q.sum(axis=1, keepdims=True)

    hmax = int(np.abs(h).max()) if n else 0
    span = md - 1                      # |sum of cavity warnings| <= degree - 1
    off_c = span                       # index of value 0 in the cavity array
    Hspan = span + hmax
    Jd_sign = np.sign(model.Jd).astype(np.int64)
    src = model.src

    def _reweight_and_map(P, offset, Jsign):
        """``P`` is (rows, L) over integer H starting at ``-offset``.

        Returns the trit distribution after the 1RSB reweighting and the
        min-sum map, plus the log of the normalisation (unused but kept so the
        energy-shift bookkeeping is inspectable).
        """
        L = P.shape[1]
        Hval = np.arange(L) - offset
        w = np.maximum(np.abs(Hval), 1).astype(np.float64)
        if y is None:                                   # y = inf: hard selection
            mask = np.zeros_like(P)
            live = P > 1e-300
            best = np.where(live, w[None, :], -np.inf).max(axis=1, keepdims=True)
            mask[live & (w[None, :] >= best - 1e-12)] = 1.0
            W = P * mask
        else:
            lw = y * w
            W = P * np.exp(lw - lw.max())[None, :]
        Z = W.sum(axis=1, keepdims=True)
        W = W / np.maximum(Z, EPS)
        pos = W[:, Hval > 0].sum(axis=1)
        neg = W[:, Hval < 0].sum(axis=1)
        zer = W[:, Hval == 0].sum(axis=1)
        up = np.where(Jsign > 0, pos, neg)
        dn = np.where(Jsign > 0, neg, pos)
        return np.stack([dn, zer, up], axis=1)

    def _leave_one_out(Qcur):
        """(n, md, 2*span+1): distribution of the incoming warning sum at each
        site with one slot removed.  Prefix/suffix products, so a degree-8 site
        costs 2*8 tiny convolutions instead of 8 full ones."""
        Qpad = np.vstack([Qcur, np.array([[0.0, 1.0, 0.0]])])
        Qin = Qpad[model.in_slots]                            # (n, md, 3)
        pre = [np.tile(np.array([[1.0]]), (n, 1))]
        for k in range(md):
            pre.append(_poly_mul(pre[-1], Qin[:, k]))
        suf = [np.tile(np.array([[1.0]]), (n, 1))]
        for k in range(md - 1, -1, -1):
            suf.append(_poly_mul(suf[-1], Qin[:, k]))
        suf = suf[::-1]                                       # suf[k] = slots k..md-1
        loo = np.empty((n, md, 2 * span + 1))
        for k in range(md):
            a, b = pre[k], suf[k + 1]
            # conv(a, b): a is (n, 2k+1), b is (n, 2(md-1-k)+1) -> (n, 2md-1)
            out = np.zeros((n, a.shape[1] + b.shape[1] - 1))
            for t in range(b.shape[1]):
                out[:, t:t + a.shape[1]] += a * b[:, t:t + 1]
            loo[:, k] = out
        full = pre[md]                                        # (n, 2md+1)
        return loo, full

    converged, it, residual = False, 0, np.inf
    for it in range(1, max_iters + 1):
        loo, _ = _leave_one_out(Q)
        # P(H_{i->j}) = loo shifted by h_i.  Build on a common integer grid.
        P = np.zeros((n, md, 2 * Hspan + 1))
        base = Hspan - off_c
        for hv in np.unique(h):
            rows = np.nonzero(h == hv)[0]
            st = base + int(hv)
            P[rows, :, st:st + 2 * span + 1] = loo[rows]
        flatP = P[src, model.slot_of[model.rev]]              # (D, grid)
        Qnew = _reweight_and_map(flatP, Hspan, Jd_sign)
        residual = float(np.max(np.abs(Qnew - Q))) if D else 0.0
        Q = (1.0 - damping) * Qnew + damping * Q
        Q /= Q.sum(axis=1, keepdims=True)
        if residual < tol:
            converged = True
            break

    # site surveys: all incoming warnings, reweighted by the site shift exp(y|H|)
    _, full = _leave_one_out(Q)
    Ps = np.zeros((n, full.shape[1] + 2 * max(hmax, 0)))
    soff = md + hmax
    for hv in np.unique(h):
        rows = np.nonzero(h == hv)[0]
        st = hmax + int(hv)
        Ps[rows, st:st + full.shape[1]] = full[rows]
    Hval = np.arange(Ps.shape[1]) - soff
    if y is None:
        w = np.abs(Hval).astype(np.float64)
        live = Ps > 1e-300
        best = np.where(live, w[None, :], -np.inf).max(axis=1, keepdims=True)
        W = Ps * (live & (w[None, :] >= best - 1e-12))
    else:
        lw = y * np.abs(Hval)
        W = Ps * np.exp(lw - lw.max())[None, :]
    W = W / np.maximum(W.sum(axis=1, keepdims=True), EPS)
    pos = W[:, Hval > 0].sum(axis=1)
    neg = W[:, Hval < 0].sum(axis=1)
    zer = W[:, Hval == 0].sum(axis=1)
    site = np.stack([neg, zer, pos], axis=1)
    bias = pos - neg
    cfg = np.where(bias >= 0, 1, -1).astype(np.int8)
    for i, v in (clamped or {}).items():
        cfg[i] = v
    return SPResult(Q=Q, site=site, frozen=1.0 - zer, bias=bias,
                    converged=converged, iters=it, residual=residual,
                    config=cfg, energy=model.energy(cfg))


# ── decimation: the outer loop this phase is trying to make unnecessary ──────

@dataclass
class DecimationResult:
    config: np.ndarray
    energy: float
    rounds: int
    total_iters: int
    stalls: int                    # rounds where the propagation did not converge
    quench_gain: float = 0.0       # energy the final polish recovered

    def __str__(self) -> str:
        return (f"E={self.energy:.1f} in {self.rounds} rounds "
                f"({self.total_iters} msg iters, {self.stalls} stalls)")


def _decimate(model, propagate, polarisation, fix_per_round, clamped, polish, rng):
    clamped = dict(clamped or {})
    rounds = total = stalls = 0
    while len(clamped) < model.n:
        res = propagate(clamped)
        total += res.iters
        stalls += 0 if res.converged else 1
        pol = polarisation(res)
        pol[list(clamped)] = -np.inf
        free = np.nonzero(np.isfinite(pol))[0]
        k = min(max(1, fix_per_round), len(free))
        pick = free[np.argsort(-np.abs(pol[free]))[:k]]
        for i in pick:
            v = 1 if pol[i] >= 0 else -1
            # a completely unpolarised spin is a coin flip, and saying so is
            # more honest than letting the >= 0 tie-break masquerade as a call
            if pol[i] == 0.0:
                v = int(rng.integers(0, 2)) * 2 - 1
            clamped[int(i)] = v
        rounds += 1
    cfg = np.ones(model.n, dtype=np.int8)
    for i, v in clamped.items():
        cfg[i] = v
    e0 = model.energy(cfg)
    gain = 0.0
    if polish:
        cfg2, _ = greedy_quench(model, cfg, rng=rng, frozen=None)
        gain = e0 - model.energy(cfg2)
        cfg = cfg2
    return DecimationResult(config=cfg, energy=model.energy(cfg), rounds=rounds,
                            total_iters=total, stalls=stalls, quench_gain=gain)


def bp_decimation(model: IsingModel, beta: float = 1.0, fix_per_round: int = 1,
                  max_iters: int = 500, damping: float = 0.5, rng=None,
                  clamped: Optional[dict] = None, polish: bool = True,
                  **bp_kwargs) -> DecimationResult:
    """Propagate, freeze the most polarised free spin, repeat.

    The polarisation is ``|m_i|``.  When BP fails to converge the last iterate is
    used anyway — that is what the literature does, and refusing to decide would
    turn every non-convergent instance into a missing data point rather than a
    measured failure.  ``stalls`` counts those rounds so the failure stays
    visible in the result.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    return _decimate(
        model,
        lambda cl: belief_propagation(model, beta=beta, max_iters=max_iters,
                                      damping=damping, rng=rng, clamped=cl,
                                      **bp_kwargs),
        lambda r: r.marginals.copy(), fix_per_round, clamped, polish, rng)


def sp_decimation(model: IsingModel, y: Optional[float] = 0.0,
                  fix_per_round: int = 1, max_iters: int = 300,
                  damping: float = 0.5, rng=None, clamped: Optional[dict] = None,
                  polish: bool = True) -> DecimationResult:
    """The classic SP loop: survey, hard-fix the most *frozen* spin, repeat.

    The polarisation is the survey bias ``P(H>0) - P(H<0)``, so the spin that
    gets fixed is the one most often frozen the same way across clusters — not
    the one with the largest BP marginal.  That difference is the whole reason
    SP beats BP near a clustering transition, and it is why the NCA-native
    replacement has to track *frozenness*, not *belief*.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    return _decimate(
        model,
        lambda cl: survey_propagation(model, y=y, max_iters=max_iters,
                                      damping=damping, rng=rng, clamped=cl),
        lambda r: r.bias.copy(), fix_per_round, clamped, polish, rng)


# ── local search: the "trivial rule" control ─────────────────────────────────

def greedy_quench(model: IsingModel, s0=None, rng=None, max_sweeps: int = 1000,
                  frozen=None, schedule: str = "sequential"):
    """Flip any spin that lowers the energy until none does.  Returns (s, sweeps).

    This is the cheapest thing that could possibly work, so it is the control
    every solve rate has to clear: an instance a random quench solves outright is
    a warm-up, not an instance (``ising.IsingDifficulty.max_quench_rate``).  It
    is also the polish step SP-style algorithms normally end with, and the
    closest non-learned analogue of an NCA relaxation.

    ``schedule="parallel"`` flips every improving spin at once, which is what a
    synchronous NCA step does and what makes two-cycles possible; the sweep
    count it reports is therefore the honest depth unit for ``min_rollout``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    s = (np.where(rng.random(model.n) < 0.5, -1, 1).astype(np.int8)
         if s0 is None else np.asarray(s0, dtype=np.int8).copy())
    mask = np.ones(model.n, dtype=bool)
    if frozen is not None:
        mask[np.asarray(frozen, dtype=np.int64)] = False
    sweeps = 0
    for sweeps in range(1, max_sweeps + 1):
        if schedule == "parallel":
            H = model.local_fields(s)
            flip = (H * s < 0) & mask
            if not flip.any():
                return s, sweeps - 1
            s[flip] = -s[flip]
        else:
            moved = False
            for i in rng.permutation(model.n):
                if not mask[i]:
                    continue
                a, b = model.adj_ptr[i], model.adj_ptr[i + 1]
                Hi = model.h[i] + float(np.dot(model.adj_J[a:b],
                                               s[model.adj_nbr[a:b]]))
                if Hi * s[i] < 0:
                    s[i] = -s[i]
                    moved = True
            if not moved:
                return s, sweeps - 1
    return s, sweeps


def quench_rate(model: IsingModel, e_ground: float, trials: int = 32, rng=None,
                clamped: Optional[dict] = None, schedule: str = "sequential"):
    """``(hit rate, mean excess energy, mean sweeps)`` for random-start quenches.

    The difficulty measure with the best claim to being honest: it is the
    probability that the dumbest local rule stumbles onto a ground state within
    the same locality budget the NCA has.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    hits, excess, sweeps = 0, [], []
    for _ in range(trials):
        s = np.where(rng.random(model.n) < 0.5, -1, 1).astype(np.int8)
        for i, v in (clamped or {}).items():
            s[i] = v
        s, k = greedy_quench(model, s, rng=rng, schedule=schedule,
                             frozen=list(clamped) if clamped else None)
        e = model.energy(s)
        hits += int(abs(e - e_ground) < 1e-9)
        excess.append(e - e_ground)
        sweeps.append(k)
    return hits / trials, float(np.mean(excess)), float(np.mean(sweeps))


# ── generic pairwise BP, for the tracks whose variables are not spins ────────

def discrete_bp(n_vars: int, n_states: int, edges, log_pair, log_unary=None,
                max_iters: int = 500, tol: float = 1e-8, damping: float = 0.5,
                rng=None):
    """Sum-product BP for a pairwise MRF over a finite alphabet.

    Args:
        edges:      ``(E, 2)`` undirected edges.
        log_pair:   ``(E, S, S)`` log potentials, indexed ``[e, x_i, x_j]`` with
                    ``i = edges[e,0]``.  A hard constraint is ``-inf``-ish; use a
                    large negative number rather than ``-inf`` so a contradiction
                    produces a diagnosable message instead of a NaN.
        log_unary:  ``(n, S)`` or None.

    Wang tiling is exactly this object (variables = cells, alphabet = tile
    types, potential = the compatibility relation) and so is Sudoku (alphabet =
    digits, potential = "not equal" on every peer pair), which is why it lives
    here rather than in ``ising.py``.  Returns ``(log_marginals, converged,
    iters)``.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    E = len(edges)
    lp = np.asarray(log_pair, dtype=np.float64).reshape(E, n_states, n_states)
    lu = (np.zeros((n_vars, n_states)) if log_unary is None
          else np.asarray(log_unary, dtype=np.float64))

    src = np.empty(2 * E, dtype=np.int64)
    dst = np.empty(2 * E, dtype=np.int64)
    src[0::2], dst[0::2] = edges[:, 0], edges[:, 1]
    src[1::2], dst[1::2] = edges[:, 1], edges[:, 0]
    rev = np.arange(2 * E) ^ 1
    # potential seen from the directed edge's source
    pot = np.empty((2 * E, n_states, n_states))
    pot[0::2] = lp
    pot[1::2] = lp.transpose(0, 2, 1)

    msg = 0.01 * rng.standard_normal((2 * E, n_states))
    converged, it = False, 0
    for it in range(1, max_iters + 1):
        tot = np.zeros((n_vars, n_states))
        np.add.at(tot, dst, msg)
        cav = lu[src] + tot[src] - msg[rev]                    # (2E, S)
        new = np.logaddexp.reduce(pot + cav[:, :, None], axis=1)
        new -= new.max(axis=1, keepdims=True)
        res = float(np.max(np.abs(new - msg))) if E else 0.0
        msg = (1.0 - damping) * new + damping * msg
        if res < tol:
            converged = True
            break
    tot = np.zeros((n_vars, n_states))
    np.add.at(tot, dst, msg)
    lm = lu + tot
    lm -= np.logaddexp.reduce(lm, axis=1, keepdims=True)
    return lm, converged, it
