"""Rolling-window sweep — size-agnostic NCA computation.

Motivation
----------
A conventional NCA run materialises the whole board and updates every cell for
``T`` steps, so autograd memory grows as ``O(T * H * W)``. Enlarging the board
to give the rule more room to compute therefore multiplies the memory budget by
the same factor. The knob that seems free — a bigger canvas — is the one knob
that is not.

The sweep removes the coupling. A fixed ``wh x ww`` window is placed on the
board, the local rule is applied inside it, and the window slides by ``wstride``
columns (or ``hstride`` rows at a band corner). Only the window is ever
materialised for the update, so the rule is *structurally* incapable of
learning anything about board size or absolute position: it sees a ``wh x ww``
crop, always. Two consequences follow directly:

* **Memory is ``O(window)`` for the update**, not ``O(board)``. Only the window
  is ever materialised for the update, plus one band (``wh x W``) to carry
  computed state across the corner; the board itself is reassembled from
  committed strips. Note the honest bound: the *output* board is unavoidably
  ``O(H * W)``, and so is the gradient reaching it. What is decoupled is the
  per-hop graph, which no longer scales with board size.
* **Length generalisation is free.** Training on a two-window board and
  evaluating on a two-hundred-window board exercises exactly the same
  computation, so the size gap that has to be *learned* in E5 is instead
  *built in* here.

Geometry
--------
Windows tile the board in a boustrophedon (serpentine) order: band 0 sweeps
left to right, band 1 right to left, and so on. Consecutive bands overlap:
band ``b+1`` starts ``hstride = wh - overlap_y`` rows below band ``b``, so the
bottom ``overlap_y`` rows of one band are the top ``overlap_y`` rows of the
next. That overlap is the *only* channel by which information crosses a band
boundary; a corner with ``hstride == wh`` (no overlap) is a hard seam and the
board silently splits into independent strips. The same argument applies
horizontally — ``overlap_x = ww - wstride`` is how much settled state survives
from one hop to the next, and it also sets the *settling budget*: a cell is
recomputed roughly ``overlap_x / wstride + 1`` times before it is committed, so
that ratio is the number of NCA updates the rule gets to finish a computation
at a given column. Raise ``ww`` or lower ``wstride`` to buy more.

Each hop *commits* the columns of its window that no later window in the band
will touch. Within a band the last window to touch a column is the same for
every row, so those commit rectangles tile the full width — which is what lets
each band be assembled with ``torch.cat`` rather than in-place slice assignment
(measured ~15x faster at 1300 hops). Bands then stack the same way, each
contributing only the rows no later band revisits, so the rectangles tile the
board exactly with no gaps.

Causality
---------
The sweep is causal in its direction of travel: information propagates from
window to window but never backwards. A single pass therefore computes correctly
only if the task's information flow agrees with the sweep direction. Since a
serpentine reverses direction at every band, ``n_passes=1`` is sound for a
single band (or for flow "down-then-left"). For anything bidirectional use
``n_passes >= 2``, which traverses the plan out and then back so both parities
of the boustrophedon are visited.

Nothing here feeds the rule any notion of position. Whether to hand it a
leading-edge marker channel is left to the training script; the default is to
let the rule infer the frontier from the blank cells ahead of it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

__all__ = ["Hop", "SweepPlan", "WindowSweep", "axis_positions"]


def axis_positions(extent: int, win: int, stride: int) -> List[int]:
    """Window start positions covering ``[0, extent)``, last one flush to the edge.

    Positions are ``0, stride, 2*stride, ...`` up to ``extent - win``, plus
    ``extent - win`` itself when the stride does not land on it, so the final
    window always reaches the boundary of the board.
    """
    if win > extent:
        raise ValueError(f"window {win} does not fit in extent {extent}")
    if not 1 <= stride <= win:
        raise ValueError(f"stride {stride} must satisfy 1 <= stride <= win {win}")
    last = extent - win
    if last == 0:
        return [0]
    pos = list(range(0, last + 1, stride))
    if pos[-1] != last:
        pos.append(last)
    return pos


def _band_hops(y0: int, r1: int, cols: List[int], ww: int, W: int) -> List[Hop]:
    """Commit rectangles for one band, given its x positions in traversal order.

    A hop commits the columns no *later* hop in this band will revisit: the
    columns up to where the next window starts, or the whole window if it is the
    last. ``abs`` and the edge clip together make this correct in both
    directions — the final window is clamped flush to the board edge, so without
    the clip the last two strips would overlap and the band would sum to more
    than ``W``.
    """
    out: List[Hop] = []
    for ci, x0 in enumerate(cols):
        if ci == len(cols) - 1:
            c0, c1 = 0, ww
        else:
            delta = cols[ci + 1] - x0
            room = (W - x0) if delta > 0 else (x0 + ww)
            gap = min(ww, abs(delta), room)
            c0, c1 = (0, gap) if delta > 0 else (ww - gap, ww)
        out.append(Hop(y0, x0, 0, 0, (0, r1, c0, c1)))
    return out


@dataclass(frozen=True)
class Hop:
    """One window placement on the board.

    Attributes:
        y0, x0:  top-left corner of the window, in board coordinates.
        dy, dx:  movement from the previous hop. Exactly one is non-zero;
                 both are 0 for the first hop of a pass.
        commit:  ``(r0, r1, c0, c1)`` in *window-local* coordinates — the
                 sub-rectangle no later window in this pass revisits, so its
                 value is final once written.
    """

    y0: int
    x0: int
    dy: int
    dx: int
    commit: Tuple[int, int, int, int]


@dataclass(frozen=True)
class SweepPlan:
    """An ordered list of window placements covering a board, split into passes.

    Args:
        H, W:              board size.
        wh, ww:            window size.
        hstride, wstride:  hop size per hop. ``wh - hstride`` (``ww - wstride``)
                           is the overlap carried across a band corner (an
                           in-band hop).
        n_passes:          1 = single boustrophedon; 2 = out and back; 4 = out
                           and back twice, and so on.
        start_direction:   ``"lr"`` or ``"rl"`` — direction of band 0; band
                           parity then alternates.
        row_order:         ``"tb"`` or ``"bt"`` — whether bands are visited
                           top-to-bottom or bottom-to-top.
    """

    H: int
    W: int
    wh: int
    ww: int
    hstride: int
    wstride: int
    n_passes: int = 1
    start_direction: str = "lr"
    row_order: str = "tb"

    def __post_init__(self) -> None:
        if self.wh > self.H or self.ww > self.W:
            raise ValueError(
                f"window {self.wh}x{self.ww} does not fit on board {self.H}x{self.W}"
            )
        if not 1 <= self.hstride <= self.wh:
            raise ValueError(f"hstride {self.hstride} must satisfy 1 <= hstride <= wh {self.wh}")
        if not 1 <= self.wstride <= self.ww:
            raise ValueError(f"wstride {self.wstride} must satisfy 1 <= wstride <= ww {self.ww}")
        if self.n_passes < 1:
            raise ValueError(f"n_passes must be >= 1, got {self.n_passes}")
        if self.start_direction not in ("lr", "rl"):
            raise ValueError(f"start_direction must be 'lr' or 'rl', got {self.start_direction!r}")
        if self.row_order not in ("tb", "bt"):
            raise ValueError(f"row_order must be 'tb' or 'bt', got {self.row_order!r}")

    # -- derived geometry -----------------------------------------------------

    @property
    def overlap_x(self) -> int:
        """Columns of settled state carried from one in-band hop to the next."""
        return self.ww - self.wstride

    @property
    def overlap_y(self) -> int:
        """Rows carried across a band corner. Zero makes the turn a hard seam."""
        return self.wh - self.hstride

    @property
    def settle_steps(self) -> float:
        """Expected NCA updates a cell receives before it is committed.

        This is the budget the rule has to finish a computation at a given
        column. If a task needs more settling than this, the window has to be
        wider, the stride smaller, or ``inner_steps`` raised.
        """
        return self.overlap_x / self.wstride + 1.0

    @property
    def band_rows(self) -> List[int]:
        rows = axis_positions(self.H, self.wh, self.hstride)
        return rows if self.row_order == "tb" else list(reversed(rows))

    @property
    def n_bands(self) -> int:
        return len(self.band_rows)

    def band_cols(self, band: int) -> List[int]:
        """Window x positions for ``band``, in traversal order."""
        cols = axis_positions(self.W, self.ww, self.wstride)
        forward = (band % 2 == 0) == (self.start_direction == "lr")
        return cols if forward else list(reversed(cols))

    def _r1_at(self, order: Sequence[int], i: int) -> int:
        """Rows committed by the band at *traversal* position ``i``.

        Keyed by traversal position, not by index into ``band_rows``, so a
        reversed pass commits the correct rows. ``abs`` because with
        ``row_order="bt"`` the bands descend, making the gap negative.
        """
        if i == len(order) - 1:
            return self.wh
        return min(self.wh, abs(self.band_rows[order[i + 1]] - self.band_rows[order[i]]))

    def _pass(self, reverse: bool) -> List[Hop]:
        """Hops for one pass.

        Reversing a pass reverses *both* the band order and each band's
        horizontal direction. Commit rectangles are then rebuilt from that
        order, because they depend on it: reversing traversal changes which
        window is the last one to cover a given cell, so a backward window
        commits the columns to its *right*, and the forward-pass rectangles
        would leave both a gap and a double-write.
        """
        rows = self.band_rows
        order = list(range(len(rows) - 1, -1, -1)) if reverse else list(range(len(rows)))
        raw: List[Hop] = []
        for i, b in enumerate(order):
            cols = self.band_cols(b)
            if reverse:
                cols = list(reversed(cols))
            raw.extend(_band_hops(rows[b], self._r1_at(order, i), cols, self.ww, self.W))

        out: List[Hop] = []
        for i, h in enumerate(raw):
            prev = raw[i - 1] if i else None
            dy = 0 if prev is None else h.y0 - prev.y0
            dx = 0 if prev is None else h.x0 - prev.x0
            out.append(Hop(h.y0, h.x0, dy, dx, h.commit))
        return out

    def segments(self) -> List[List[Hop]]:
        """The plan as one list of hops per pass.

        Pass ``p`` starts from the board state pass ``p-1`` produced, so passes
        chain. Every pass tiles the whole board, so every pass yields a complete
        board.
        """
        return [self._pass(reverse=p % 2 == 1) for p in range(self.n_passes)]

    @property
    def n_hops(self) -> int:
        return sum(len(s) for s in self.segments())

    def describe(self) -> str:
        return (
            f"{self.H}x{self.W} board | {self.wh}x{self.ww} window | "
            f"stride {self.wstride}x{self.hstride} | overlap {self.overlap_x}x{self.overlap_y} | "
            f"~{self.settle_steps:.1f} settle steps | "
            f"{self.n_bands} band(s) x {self.n_passes} pass(es) = {self.n_hops} hops"
        )

    def caveats(self) -> List[str]:
        """Human-readable warnings about this geometry, if any."""
        msgs = []
        if self.n_passes == 1 and self.n_bands > 1:
            msgs.append(
                f"n_passes=1 with {self.n_bands} bands reverses direction at each band, so "
                "information can only flow down-then-left (or down-then-right). If the task "
                "flows the other way, use n_passes>=2."
            )
        if self.overlap_y == 0:
            msgs.append(
                "overlap_y is 0, so band corners are a hard seam: bands compute "
                "independently and no information crosses between them."
            )
        if self.overlap_x == 0:
            msgs.append(
                "overlap_x is 0, so each column is computed by exactly one NCA step; "
                "there is no settling budget at all."
            )
        return msgs


class WindowSweep(nn.Module):
    """Applies a local NCA rule across a board through a rolling window.

    Args:
        rule: anything exposing ``step(x) -> x'`` for one update of a
              ``(B, C, h, w)`` state — ``NeuralCA``, ``GatedNCA`` and
              ``TemporalSobelNCA`` all do.
    """

    def __init__(self, rule: nn.Module) -> None:
        super().__init__()
        self.rule = rule

    @property
    def stateless(self) -> bool:
        """Whether re-running a window reproduces the same output.

        Rules carrying a temporal history buffer (``GatedNCA``,
        ``TemporalSobelNCA``) are not stateless, so gradient checkpointing them
        would advance the buffer a second time during recomputation and
        silently corrupt gradients.
        """
        return not getattr(self.rule, "is_stateful", False)

    def _step_window(self, win: torch.Tensor, inner_steps: int) -> torch.Tensor:
        for _ in range(inner_steps):
            win = self.rule.step(win)
        return win

    def forward(
        self,
        state: torch.Tensor,
        plan: SweepPlan,
        inner_steps: int = 1,
        grad_checkpoint: bool = False,
        return_frames: bool = False,
        return_bands: bool = False,
    ):
        """Sweep ``state`` according to ``plan``.

        Args:
            state:            ``(B, C, H, W)`` initial board state.
            plan:             the :class:`SweepPlan` to follow.
            inner_steps:      NCA updates per hop. 1 matches "one step, then
                              slide"; raising it buys settling budget at
                              proportional compute cost.
            grad_checkpoint:  recompute each window during backward rather than
                              storing it, trading ~30% compute for memory
                              independent of hop count. This is what makes full
                              backprop practical on long boards.
            return_frames:    also return the per-hop window states.
            return_bands:     also return each band's intermediate board. Every
                              band is a checkpoint where the board is already a
                              valid partial answer, so this is the natural
                              analogue of supervising every rollout frame.

        Returns:
            The ``(B, C, H, W)`` final board, plus the frames and band boards if
            requested (in that order).
        """
        _check_state(state, plan)
        board = state
        frames: List[torch.Tensor] = []
        bands: List[torch.Tensor] = []
        for hops in plan.segments():
            board, seg_frames, seg_bands = self._run_segment(
                board, plan, hops, inner_steps, grad_checkpoint, return_frames
            )
            frames.extend(seg_frames)
            bands.extend(seg_bands)
        extra: List[torch.Tensor] = []
        if return_frames:
            extra.append(frames)
        if return_bands:
            extra.append(bands)
        return (board, *extra) if extra else board

    def _run_segment(
        self,
        board: torch.Tensor,
        plan: SweepPlan,
        hops: Sequence[Hop],
        inner_steps: int,
        grad_checkpoint: bool,
        keep_frames: bool,
    ) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Sweep one pass; returns the new board, the frames, the band boards.

        ``board`` is the state at the start of the pass. It supplies the cells
        the frontier has not reached yet — which by definition still hold their
        pass-start values — and, for pass 2 onwards, is a node on the graph.

        A single sliding buffer is *not* enough to carry a band, and this is the
        subtlety the whole design turns on. Within a band, a window commits only
        the rows above the next band's start, so the rows *below* that line exit
        the window sideways while still owing updates to the next band. A buffer
        that drops them at the edge has silently forgotten them. So instead of
        one buffer we compose whole bands: each band assembles its commit strips
        into a full-height ``(B, C, wh, W)`` board and hands that to the next
        band, which is exactly what a band needs to know. That is sound because
        within a band the last window to touch a column is the same for every
        row, so the strips tile the full width, not just the committed rows.
        """
        wh, ww = plan.wh, plan.ww
        frames: List[torch.Tensor] = []
        spans: List[Tuple[int, torch.Tensor]] = []
        prev_band: Optional[torch.Tensor] = None
        use_ckpt = grad_checkpoint and self.stateless and torch.is_grad_enabled()

        for band in self._group_bands(hops):
            y0 = band[0].y0
            dy = band[0].dy  # distance from the previous band, 0 for the first
            r1 = band[0].commit[1]
            forwards = len(band) == 1 or band[1].dx > 0
            buf = self._read(prev_band, board, y0, band[0].x0, wh, ww, dy)

            strips: List[torch.Tensor] = []
            for i, hop in enumerate(band):
                if use_ckpt:
                    new = checkpoint(self._step_window, buf, inner_steps, use_reentrant=False)
                else:
                    new = self._step_window(buf, inner_steps)
                if keep_frames:
                    frames.append(new)

                c0, c1 = hop.commit[2], hop.commit[3]
                strips.append(new[:, :, :, c0:c1])

                if i + 1 == len(band):
                    break
                dx = band[i + 1].dx
                if dx > 0:
                    fresh = self._read(prev_band, board, y0, hop.x0 + ww, wh, dx, dy)
                    buf = torch.cat([new[:, :, :, dx:], fresh], dim=3)
                elif dx < 0:
                    n = -dx
                    fresh = self._read(prev_band, board, y0, hop.x0 - n, wh, n, dy)
                    buf = torch.cat([fresh, new[:, :, :, : ww - n]], dim=3)
                else:  # pragma: no cover - guarded by _group_bands
                    raise AssertionError(f"in-band hop has dy={hop.dy}, dx={hop.dx}")

            band_board = torch.cat(strips if forwards else strips[::-1], dim=3)
            prev_band = band_board
            spans.append((y0, band_board[:, :, :r1, :]))

        # Traversal order is not board order: descending bands and reversed
        # passes both visit rows bottom-up, but the row crops still tile [0, H).
        spans.sort(key=lambda s: s[0])
        rows = [t for _, t in spans]
        out = torch.cat(rows, dim=2) if len(rows) > 1 else rows[0]
        return out, frames, [t for _, t in spans]

    @staticmethod
    def _group_bands(hops: Sequence[Hop]) -> List[List[Hop]]:
        """Split a pass into bands; a ``dy`` on a hop means it starts one."""
        bands: List[List[Hop]] = [[]]
        for i, hop in enumerate(hops):
            if i > 0 and hop.dy != 0:
                bands.append([])
            bands[-1].append(hop)
        return bands

    @staticmethod
    def _read(
        prev_band: Optional[torch.Tensor],
        board: torch.Tensor,
        y0: int,
        x0: int,
        h: int,
        w: int,
        oh: int,
    ) -> torch.Tensor:
        """Contents of the window at ``(y0, x0)`` as the frontier reaches it.

        ``oh`` is how far this band sits from the one before it, i.e. the rows
        of ``prev_band`` this band inherits. It is *not* ``plan.overlap_y``:
        edge-clamping the last window flush to the board edge can pull a band
        closer than ``hstride``, so a plan can contain bands that overlap far
        more than the nominal stride implies.

        The inherited rows sit on the side facing the previous band — the top
        when this band is below it, the bottom when above. The rest are past
        the frontier and still hold ``board``'s pass-start values.
        """
        if prev_band is None or oh == 0:
            return board[:, :, y0 : y0 + h, x0 : x0 + w]
        gap = abs(oh)
        if oh > 0:
            top = prev_band[:, :, gap:, x0 : x0 + w]
            bottom = board[:, :, y0 + h - gap : y0 + h, x0 : x0 + w]
        else:
            top = board[:, :, y0 : y0 + gap, x0 : x0 + w]
            bottom = prev_band[:, :, : h - gap, x0 : x0 + w]
        return torch.cat([top, bottom], dim=2)


def _check_state(state: torch.Tensor, plan: SweepPlan) -> None:
    if state.dim() != 4:
        raise ValueError(f"state must be (B, C, H, W), got shape {tuple(state.shape)}")
    _, _, H, W = state.shape
    if (H, W) != (plan.H, plan.W):
        raise ValueError(f"state is {H}x{W} but the plan is for a {plan.H}x{plan.W} board")
