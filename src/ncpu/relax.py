"""Shared base for NCA *relaxation solvers* — solving, not checking.

Why this file exists
--------------------
The 3SAT verifier was feed-forward: implant an input, run a fixed number of
steps, read an answer. A solver is a different object. The NCA has to be a
dynamical system whose **attractors are solutions**: cells hold beliefs, the
beliefs propagate, contradictions reconcile locally, and the grid settles. No
outer Python loop decodes a value and re-seeds it -- reconciliation happens
inside the dynamics.

Every problem track implements ``RelaxationTask`` so the tracks stay comparable
and so one trainer and one scorecard serve all of them. The trainer itself is
NOT here (the methods track owns it); this file is the contract plus the eval
machinery that the contract makes possible.

Design notes that came out of the 3SAT work, and are not optional
----------------------------------------------------------------
* **Score the SETTLED state, never any-frame.** Asking "did any frame satisfy
  the constraints" is not a solve rate: a random walk over 512 states scored
  1.000 on that metric while its returned answer stayed at chance. Solve rate
  means the state the system settles on.
* **Baselines must be matched to the budget.** If the model visits K distinct
  configurations, the baseline is a random walk over K -- not one random draw.
  ``matched_baseline`` does this.
* **Exact algorithmic checking only.** ``is_solved`` is ground truth. A soft
  energy is for gradients and diagnostics; it is never reported as accuracy.
* **Givens must be clamped and the clamp must be audited.** A solver that
  quietly overwrites the puzzle's givens has solved a different puzzle.
  ``givens_preserved`` is the cheat detector.
* **Depth is a pixel distance.** Information moves one cell per step, so state
  ``min_rollout`` honestly and derive rollouts from it, never hard-code.
* **These tasks grok.** The 3SAT verifier sat at exactly its baseline for ~5.5k
  steps and then broke through to 1.000. A flat early plateau is not a dead run;
  judge nothing before ~25k steps.

Two ``NeuralCA`` settings that belief encodings REQUIRE
-------------------------------------------------------
Both bite every task in this phase, not just the one that found them.

* **``alive_threshold=0``, i.e. alive masking OFF.** ``AliveMasking`` gates
  updates on a max-pool of **channel 0**. In a belief encoding channel 0 is an
  arbitrary member of the alphabet -- "belief in tile 0", "belief in digit 1" --
  so gating on it is meaningless, and on cells that have *ruled that symbol out*
  it is strongly negative, which switches those cells off entirely. Every cell
  in a relaxation task is alive by construction.

* **Padding is a constraint-graph decision, not a detail.** ``NeuralCA``
  defaults to ``padding_type="circular"``, which wraps the grid into a torus and
  silently makes row 0 a neighbour of row H-1. That is safe only when the free
  cells are separated from the seam by a clamped ring whose deltas are
  discarded. **Any task with free cells on the boundary must use zero padding**,
  or the constraint graph the rule learns is not the one being scored.

Why one pixel per cell, not a block of pixels
---------------------------------------------
``NeuralCA`` perceives 3x3 **per channel** and then applies a **1x1** rule across
channels. So at one pixel per cell the rule reads a cell's whole belief vector in
a single step, for free. Spread that vector over an s x s block of pixels and a
3x3 kernel cannot even see a cell's own state in one step -- the rule must spend
steps gathering itself before it can compare with a neighbour, which throws away
the radius-1 property that made the task worth choosing. Put the alphabet in
``n_planes``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import torch

BACKGROUND = 0.0
LOW_VALUE = -1.0
HIGH_VALUE = +1.0


class RelaxationTask(ABC):
    """One constraint-satisfaction problem, rendered on a grid.

    Implement this and the shared trainer and scorecard both work. Keep the
    constraint structure as local as the problem allows: the 3SAT verifier paid
    ``(n+m+1)*step`` in depth purely because its factor graph is not planar and
    a layout had to be imposed. A problem whose constraints are already
    nearest-neighbour pays none of that.
    """

    name: str = "task"

    # ── geometry ─────────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def shape(self) -> tuple:
        """(H, W) of the rendered grid, per plane."""

    @property
    def n_planes(self) -> int:
        """How many *rendered* planes the state carries. Default 1.

        Belief-valued tasks need a vector per cell, and it can live in CHANNELS
        or in SPACE. Channels are strictly cheaper: putting a T-symbol alphabet
        into an s x s block of pixels makes the lattice pitch grow as sqrt(T),
        and depth grows with it, so a task pays for its alphabet in rollout
        length. Prefer planes; if you use space instead, say why and report the
        pitch you paid.
        """
        return 1

    @property
    def state_shape(self) -> tuple:
        """(n_planes, H, W) — what a trainer must implant and score."""
        h, w = self.shape
        return (self.n_planes, h, w)

    @abstractmethod
    def min_rollout(self) -> int:
        """Steps before the answer can be correct, as a PIXEL distance.

        This is the **full sequential critical path**, not the one-way
        impossibility floor -- the two can differ by orders of magnitude
        (Sudoku: floor 8, critical path 608, because each deduction round costs
        a round trip and rounds are sequential). Report the critical path here
        and expose the floor separately if the gap is interesting.

        A 3x3 kernel moves information one pixel per step, including diagonally,
        so distances are Chebyshev. Relaxation needs a MULTIPLE of this, not the
        bare value: the 3SAT verifier scored exact 0.047 at 1.0x and 0.906 at
        2.0x with identical weights.
        """

    # ── instances ────────────────────────────────────────────────────────────
    @abstractmethod
    def sample(self, batch_size: int, rng, difficulty: Optional[Any] = None) -> list:
        # ``rng`` is a numpy Generator: the shared trainer passes one, because
        # vectorised generators need it. A task preferring ``random.Random``
        # must normalise internally rather than assume its own type.
        """A batch of instances. ``difficulty`` is task-defined but must exist:
        instances that a trivial propagation rule solves outright make the whole
        measurement meaningless."""

    @abstractmethod
    def encode(self, instance) -> np.ndarray:
        """The seed screen: givens rendered, everything else at BACKGROUND."""

    @abstractmethod
    def clamp_mask(self, instance) -> np.ndarray:
        """1.0 exactly where a cell is a **given of the instance**, 0.0 elsewhere.

        This is the AUDIT set: ``givens_preserved`` checks these and nothing
        else. Keep structural scaffolding (coordinate planes, box-phase planes,
        anything constant across instances) OUT of it -- such cells are trivially
        preserved and would pad the audit with guaranteed passes, which is how a
        cheat detector stops detecting cheating.

        **A confidence / stiffening channel does NOT belong in the audit**, even
        on given cells. That channel is a mechanism the dynamics is supposed to
        write; auditing it flags a *working* stiffening dynamic as a cheat. The
        worry it was meant to catch -- silently un-committing a given -- is
        already caught, because the given's ANSWER planes are audited and a
        genuine un-commit has to move them. Put the confidence channel in
        ``hold_mask`` if it must persist; never in ``clamp_mask``.
        """

    def hold_mask(self, instance) -> np.ndarray:
        """Everything re-implanted each step: givens PLUS structure.

        Defaults to ``clamp_mask``. Override when the task has structural planes
        that must be held but must not be audited.
        """
        return self.clamp_mask(instance)

    @abstractmethod
    def decode(self, frame: torch.Tensor):
        """A rendered frame -> a candidate solution in the task's own terms."""

    # ── semantics: exact, algorithmic ────────────────────────────────────────
    @abstractmethod
    def is_solved(self, instance, candidate) -> bool:
        """Ground truth. Never approximate this, and never substitute energy."""

    @abstractmethod
    def violations(self, instance, candidate) -> float:
        """Distance from correct, 0 iff solved.

        For a *satisfaction* task this is the count of broken constraints. For an
        *optimisation* task, counting broken constraints is meaningless — an
        optimal spin-glass state breaks 23% of its bonds by construction — so
        return **excess cost above the optimum** instead. Either way 0 means
        solved and larger means worse; say in your docstring which it is."""

    @abstractmethod
    def energy(self, state: torch.Tensor, instances) -> torch.Tensor:
        """Differentiable soft constraint violation. Self-supervised: no target.

        Shape contract, so the shared trainer can be written once:
        ``state`` is ``(B, n_planes, H, W)`` and ``instances`` is a sequence of
        length B; returns ``(B,)``. Accept an unbatched ``(n_planes, H, W)`` with
        a single instance as a convenience if you like, but the batched form is
        the one that must work.

        MEASURE the aggregate; do not inherit it. Report the share of the energy
        range one violation commands and the rank correlation against the exact
        violation count in the *undecided* regime, where training starts.

        **Which aggregate is right depends on the task class**, and getting this
        backwards optimises the wrong objective:

        * *Satisfaction* tasks (every constraint must hold: tiling, Sudoku) want
          pressure on the WORST constraint. A plain mean lets a model bank the
          easy ones and abandon one; a plain max is rank-blind past the first
          violation. Wang measured a de-biased softmax at 45% one-violation
          share and rho +0.94, against logprod's 1.9% / +0.84.
        * *Optimisation* tasks (minimise a cost, some violation is CORRECT:
          spin-glass ground states) want the opposite. A frustrated ground state
          breaks 23% of its bonds **on purpose**, so worst-bond pressure chases a
          state that does not exist. Ising measured plain `sum` at +0.615 against
          softmax's +0.361, winning 9 of 10 seeds.

        Also check the flat state. If uniform belief is a global minimum, the
        flattest non-answer wins and nothing trains (Sudoku: every unit sums to
        exactly 1 at uniform belief, so a purity term is mandatory). And check
        whether it is a stationary POINT even when not a minimum -- Ising's flat
        state has identically zero gradient, so training must seed noise.

        Check the flat state. If uniform belief is a global minimum of your
        energy, the flattest possible non-answer wins and nothing will train.
        """

    @abstractmethod
    def random_candidate(self, instance, rng):
        """A uniformly random candidate consistent with the givens.

        This is what ``matched_baseline`` samples, so it defines the floor every
        solve rate is judged against. It MUST respect the clamped givens --
        otherwise the baseline is measuring an easier problem than the model is.
        """

    @abstractmethod
    def reference_solve(self, instance, budget: int = 10_000):
        """The algorithmic **baseline** — what the NCA is measured against, and
        the source of the difficulty signal. Returns ``(solution|None, work)``.

        Keep this separate from exact ground truth when the two differ. An exact
        solver whose cost is identical on every instance (a transfer matrix, say)
        is a fine oracle for ``is_solved`` but carries **no difficulty signal**,
        so it makes a useless baseline. In that case expose the oracle as its own
        method and let ``reference_solve`` be the search-based solver whose work
        actually varies with the instance."""

    def describe(self) -> str:
        h, w = self.shape
        return f"{self.name} {w}x{h} min_rollout={self.min_rollout()}"


# ── scorecard ────────────────────────────────────────────────────────────────

@dataclass
class SolveReport:
    """One evaluation. Every rate carries the baseline it must beat."""
    solved: float = 0.0             # settled state, exact check
    solved_baseline: float = 0.0    # matched random walk over the same budget
    steps_to_solution: Optional[float] = None
    stayed_solved: float = 0.0      # still solved after the hold window
    givens_ok: float = 1.0          # clamp audit; < 1.0 invalidates everything
    violations: float = 0.0
    distinct_visited: float = 0.0   # "searched" vs "twitched"
    any_frame: float = 0.0          # DIAGNOSTIC ONLY, has no fixed baseline
    reference_work: Optional[float] = None   # algorithmic solver's cost
    settled: float = 1.0            # fraction of rollouts that converged
    notes: dict = field(default_factory=dict)

    def lift(self) -> float:
        return self.solved - self.solved_baseline

    def __str__(self) -> str:
        s = (f"solved {self.solved:.3f}/{self.solved_baseline:.3f} "
             f"(lift {self.lift():+.3f})  stayed {self.stayed_solved:.3f}  "
             f"viol {self.violations:.2f}  distinct {self.distinct_visited:.1f}")
        if self.steps_to_solution is not None:
            s += f"  steps {self.steps_to_solution:.0f}"
        if self.reference_work is not None:
            s += f"  ref_work {self.reference_work:.1f}"
        if self.settled < 1.0:
            s += f"  settled {self.settled:.2f}"
        if self.givens_ok < 1.0:
            s += f"  !! GIVENS OVERWRITTEN {1 - self.givens_ok:.1%}"
        return s


def settled_state(frames: torch.Tensor, window: int = 8, tol: float = 1e-2):
    """The state the rollout settles on, plus whether it actually settled.

    ``frames`` is ``(T, ...)`` -- ``(T, H, W)`` for a single-plane task,
    ``(T, n_planes, H, W)`` for a belief encoding. Settled means the last
    ``window`` frames move less than ``tol``. A system that never settles has not solved anything, however
    good some intermediate frame looked.
    """
    tail = frames[-window:]
    drift = (tail[1:] - tail[:-1]).abs().max().item() if len(tail) > 1 else 0.0
    return frames[-1], drift <= tol, drift


def first_solved_step(frames: torch.Tensor, task: RelaxationTask, instance,
                      stride: int = 1) -> Optional[int]:
    """First step whose decoded state passes the EXACT check, or None."""
    for t in range(0, frames.shape[0], stride):
        if task.is_solved(instance, task.decode(frames[t])):
            return t
    return None


def _canonical(candidate) -> bytes:
    """A lossless key for a decoded candidate.

    NOT ``repr``: numpy truncates above 1000 elements with an ellipsis, so two
    different configurations collide, ``count_distinct`` UNDER-reports, and the
    matched baseline is computed against a budget smaller than the model
    actually used -- flattering the model. That is the same species of
    measurement error that made a random walk look like a solver on 3SAT.
    """
    arr = np.asarray(candidate)
    return arr.tobytes() + str(arr.shape).encode() + str(arr.dtype).encode()


def count_distinct(frames: torch.Tensor, task: RelaxationTask,
                   stride: int = 1) -> int:
    """Distinct decoded configurations visited -- separates a system that
    searched from one that twitched. Sets the matched baseline's budget."""
    seen = set()
    for t in range(0, frames.shape[0], stride):
        seen.add(_canonical(task.decode(frames[t])))
    return len(seen)


def matched_baseline(task: RelaxationTask, instance, n_visits: int, rng,
                     trials: int = 32) -> float:
    """P(a random walk over ``n_visits`` configurations lands on a solution).

    This is the number the model has to beat. Scoring a K-frame rollout against
    a single random draw is not a baseline, it is a licence -- on 3SAT that
    error made a jittering model look like a solver.

    Be aware it goes DEGENERATE on large configuration spaces: a 9x9 Sudoku has
    one uniform candidate in 9^56, so this returns exactly 0.000 for any
    realistic budget and ``lift`` collapses to ``solved``. When that happens it
    is still worth printing (it catches an over-constrained instance set), but
    the informative comparator becomes ``SolveReport.reference_work`` -- the
    algorithmic solver's cost on the same instance -- and a smarter
    ``random_candidate`` that respects propagation rather than sampling
    uniformly.
    """
    if n_visits <= 0:
        return 0.0
    hits = 0
    for _ in range(trials):
        for _ in range(max(1, n_visits)):
            if task.is_solved(instance, task.random_candidate(instance, rng)):
                hits += 1
                break
    return hits / trials


def givens_preserved(frames: torch.Tensor, seed: torch.Tensor,
                     clamp: torch.Tensor, tol: float = 0.25) -> float:
    """Fraction of clamped cells still holding their given value at the end.

    Anything below 1.0 means the solver rewrote the puzzle, and every other
    number in the report is about a different problem than the one posed.
    """
    if clamp.sum() == 0:
        return 1.0
    final = frames[-1]
    ok = ((final - seed).abs() <= tol).float() * clamp
    return (ok.sum() / clamp.sum()).item()
