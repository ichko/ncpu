"""Tests for the rolling-window sweep.

The two properties worth protecting are:

1. the sliding buffer is *equivalent* to an explicit full-board sweep — i.e. the
   cheap implementation is the same computation as the obvious one;
2. autograd memory scales with the window, not with the board, which is the
   entire reason the sweep exists.
"""

import sys
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ncpu.sweep import Hop, SweepPlan, WindowSweep, axis_positions


class ShiftRule(nn.Module):
    """Deterministic local rule: shift channel 2 one cell to the right per step.

    Stands in for a real NCA — local, translation invariant, and completely
    blind to board size. A marker planted in channel 2 therefore has to be
    carried across the board by the sweep alone, which makes it a direct test
    that information survives the window boundary and the band corners.
    """

    is_stateful = False

    def __init__(self, channels: int = 3):
        super().__init__()
        self.channels = channels
        self.scale = nn.Parameter(torch.ones(1))

    def step(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, 2, :, 1:] = x[:, 2, :, :-1]
        return out * self.scale


class UpRule(nn.Module):
    """Shift channel 2 one cell down per step, to test vertical sweeps."""

    is_stateful = False

    def step(self, x: torch.Tensor) -> torch.Tensor:
        out = x.clone()
        out[:, 2, 1:, :] = x[:, 2, :-1, :]
        return out


class IdentityRule(nn.Module):
    """No-op rule, so every cell keeps its value and none is left unwritten."""

    is_stateful = False

    def step(self, x: torch.Tensor) -> torch.Tensor:
        return x * 1.0


def _reference_sweep(rule, state, plan, inner_steps=1):
    """Obvious implementation: keep a full board, overwrite each window in place.

    Every window overwrites its whole rectangle, so the board always holds the
    latest value of every cell. Writes are deliberately *not* limited to the
    commit rect: a cell is stepped once per hop its window covers it, and an
    oracle that wrote only the final value would drift differently from a real
    rolling window. Kept naive on purpose — it is the oracle the fast path is
    checked against.
    """
    board = state.clone()
    for hops in plan.segments():
        for hop in hops:
            y0, x0 = hop.y0, hop.x0
            win = board[:, :, y0 : y0 + plan.wh, x0 : x0 + plan.ww]
            for _ in range(inner_steps):
                win = rule.step(win)
            board[:, :, y0 : y0 + plan.wh, x0 : x0 + plan.ww] = win
    return board


class TestAxisPositions(unittest.TestCase):
    def test_covers_extent_and_ends_flush(self):
        pos = axis_positions(40, 32, 3)
        self.assertEqual(pos[0], 0)
        self.assertEqual(pos[-1], 8)  # 40 - 32, flush to the edge
        for a, b in zip(pos, pos[1:]):
            self.assertLess(b, a + 32, "consecutive windows must overlap")

    def test_exact_fit(self):
        self.assertEqual(axis_positions(32, 32, 1), [0])

    def test_rejects_bad_geometry(self):
        with self.assertRaises(ValueError):
            axis_positions(16, 32, 1)
        with self.assertRaises(ValueError):
            axis_positions(64, 32, 33)


class TestPlanGeometry(unittest.TestCase):
    def _plans(self):
        return [
            SweepPlan(H=64, W=160, wh=32, ww=32, hstride=30, wstride=3),
            SweepPlan(H=64, W=160, wh=32, ww=32, hstride=30, wstride=3, n_passes=2),
            SweepPlan(H=96, W=97, wh=32, ww=32, hstride=10, wstride=7),  # awkward sizes
            SweepPlan(H=64, W=160, wh=32, ww=32, hstride=30, wstride=3, row_order="bt"),
            SweepPlan(H=64, W=160, wh=32, ww=32, hstride=30, wstride=3, start_direction="rl"),
        ]

    def test_commit_rects_tile_the_board(self):
        for plan in self._plans():
            with self.subTest(plan=plan.describe()):
                covered = torch.zeros(plan.H, plan.W, dtype=torch.int32)
                for hops in plan.segments():
                    for hop in hops:
                        r0, r1, c0, c1 = hop.commit
                        covered[
                            hop.y0 + r0 : hop.y0 + r1, hop.x0 + c0 : hop.x0 + c1
                        ] += 1
                self.assertEqual(
                    int((covered == 0).sum()),
                    0,
                    "every cell must be committed by some window",
                )

    def test_hops_move_on_one_axis_only(self):
        for plan in self._plans():
            for hops in plan.segments():
                for hop in hops:
                    self.assertFalse(
                        hop.dy != 0 and hop.dx != 0,
                        "a hop must slide along a single axis",
                    )

    def test_consecutive_windows_overlap(self):
        plan = SweepPlan(H=64, W=160, wh=32, ww=32, hstride=30, wstride=3)
        for hops in plan.segments():
            for a, b in zip(hops, hops[1:]):
                if b.dx > 0:
                    self.assertLessEqual(b.x0, a.x0 + plan.ww, "in-band windows must overlap")
                if b.dy > 0:
                    self.assertLessEqual(b.y0, a.y0 + plan.wh, "bands must overlap")

    def test_corner_without_overlap_is_flagged(self):
        plan = SweepPlan(H=64, W=160, wh=32, ww=32, hstride=32, wstride=3)
        self.assertTrue(any("hard seam" in m for m in plan.caveats()))

    def test_single_pass_multiband_is_flagged(self):
        plan = SweepPlan(H=96, W=160, wh=32, ww=32, hstride=30, wstride=3, n_passes=1)
        self.assertTrue(any("n_passes=1" in m for m in plan.caveats()))


class TestSweepEquivalence(unittest.TestCase):
    def _plans(self):
        return [
            SweepPlan(H=64, W=96, wh=32, ww=32, hstride=30, wstride=3),
            SweepPlan(H=64, W=96, wh=32, ww=32, hstride=30, wstride=3, n_passes=2),
            SweepPlan(H=96, W=97, wh=32, ww=32, hstride=10, wstride=7),
            SweepPlan(H=64, W=96, wh=32, ww=32, hstride=30, wstride=3, row_order="bt"),
            SweepPlan(H=64, W=96, wh=32, ww=32, hstride=30, wstride=3, start_direction="rl"),
        ]

    def test_matches_explicit_full_board_sweep(self):
        torch.manual_seed(0)
        for plan in self._plans():
            with self.subTest(plan=plan.describe()):
                state = torch.zeros(1, 3, plan.H, plan.W)
                state[:, 2, plan.H // 2, 2] = 1.0  # marker near the top-left
                sweep = WindowSweep(ShiftRule())
                got = sweep(state, plan, inner_steps=2)
                want = _reference_sweep(ShiftRule(), state, plan, inner_steps=2)
                self.assertTrue(torch.equal(got, want))

    def test_vertical_transport(self):
        plan = SweepPlan(H=96, W=64, wh=32, ww=32, hstride=30, wstride=3)
        state = torch.zeros(1, 3, plan.H, plan.W)
        state[:, 2, 1, 10] = 1.0
        got = WindowSweep(UpRule())(state, plan, inner_steps=1)
        want = _reference_sweep(UpRule(), state, plan, inner_steps=1)
        self.assertTrue(torch.equal(got, want))

    def test_marker_crosses_a_band_corner(self):
        """A downward sweep must carry state through the corner overlap."""
        plan = SweepPlan(H=96, W=64, wh=32, ww=32, hstride=30, wstride=3, n_passes=1)
        state = torch.zeros(1, 3, plan.H, plan.W)
        state[:, 2, 0, 10] = 1.0
        out = WindowSweep(UpRule())(state, plan, inner_steps=1)
        self.assertGreater(float(out[:, 2].sum()), 0.0)
        self.assertGreater(float(out[:, 2, 60:, :].sum()), 0.0, "state never left band 0")


class TestSweepMechanics(unittest.TestCase):
    def test_grad_reaches_parameters(self):
        plan = SweepPlan(H=64, W=96, wh=32, ww=32, hstride=30, wstride=3)
        rule = ShiftRule()
        state = torch.zeros(2, 3, plan.H, plan.W)
        state[:, 2, 32, 2] = 1.0  # a marker, or the sum would be zero anyway
        WindowSweep(rule)(state, plan).sum().backward()
        self.assertIsNotNone(rule.scale.grad)
        self.assertGreater(float(rule.scale.grad.abs().sum()), 0.0)

    def test_checkpointing_matches_plain_forward(self):
        plan = SweepPlan(H=64, W=96, wh=32, ww=32, hstride=30, wstride=3, n_passes=2)
        state = torch.rand(2, 3, plan.H, plan.W)
        plain = WindowSweep(ShiftRule())(state, plan)
        ckpt = WindowSweep(ShiftRule())(state, plan, grad_checkpoint=True)
        self.assertTrue(torch.allclose(plain, ckpt))

    def test_rejects_mismatched_board(self):
        plan = SweepPlan(H=64, W=96, wh=32, ww=32, hstride=30, wstride=3)
        with self.assertRaises(ValueError):
            WindowSweep(ShiftRule())(torch.zeros(1, 3, 64, 128), plan)

    def test_every_board_cell_is_written(self):
        """The assembled board must not contain any leftover zero columns."""
        plan = SweepPlan(H=64, W=96, wh=32, ww=32, hstride=30, wstride=3, n_passes=2)
        state = torch.zeros(1, 3, plan.H, plan.W)
        state[:, 2] = 1.0
        # A shifting rule would legitimately blank the trailing column, so this
        # needs an identity rule: it isolates "was every cell written" from
        # "did the rule change it".
        out = WindowSweep(IdentityRule())(state, plan)
        self.assertTrue(torch.all(out == 1.0))


class TestMemoryScalesWithWindow(unittest.TestCase):
    """The central claim: activation memory tracks the window, not the board."""

    def _peak_bytes(self, plan, state):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        state = state.to(device)
        WindowSweep(ShiftRule()).to(device)(state, plan).sum().backward()
        if device == "cuda":
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated() - base
        return 0

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA to measure")
    def test_activation_memory_tracks_hops_not_cells(self):
        """Hold the output board fixed and grow the hop count 16x.

        Every extra hop adds one window to the graph, so this is the quantity
        that would blow up in a whole-board rollout. Growth should stay far
        below 16x because the per-hop window is the same size throughout; what
        does scale linearly is the output board, which is held fixed here.
        """
        H, W = 32, 512
        peaks = []
        for wstride in (16, 8, 4, 2, 1):
            plan = SweepPlan(H=H, W=W, wh=32, ww=32, hstride=30, wstride=wstride)
            state = torch.zeros(1, 3, plan.H, plan.W)
            peaks.append((plan.n_hops, self._peak_bytes(plan, state)))

        hops = [n for n, _ in peaks]
        growth = peaks[-1][1] / max(peaks[0][1], 1)
        self.assertLess(
            growth,
            hops[-1] / hops[0] / 2,
            f"activation memory grew {growth:.1f}x for {hops[-1] / hops[0]:.0f}x the hops "
            f"(hops={hops}, peaks={[b for _, b in peaks]})",
        )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA to measure")
    def test_long_board_beats_full_board_rollout(self):
        """Same NCA work, far less memory than the whole-board alternative."""
        H, W, T = 64, 2048, 120
        plan = SweepPlan(H=H, W=W, wh=32, ww=32, hstride=30, wstride=3)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        state = torch.zeros(1, 3, H, W, device="cuda")
        WindowSweep(ShiftRule()).to("cuda")(state, plan, inner_steps=1).sum().backward()
        torch.cuda.synchronize()
        sweep_peak = torch.cuda.max_memory_allocated() - base

        del state
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        state = torch.zeros(1, 3, H, W, device="cuda", requires_grad=True)
        x = state
        for _ in range(plan.n_hops):
            x = ShiftRule().to("cuda").step(x)
        x.sum().backward()
        torch.cuda.synchronize()
        full_peak = torch.cuda.max_memory_allocated() - base

        self.assertLess(
            sweep_peak, full_peak,
            f"sweep peak {sweep_peak} should beat whole-board peak {full_peak}",
        )


if __name__ == "__main__":
    unittest.main()
