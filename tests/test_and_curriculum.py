"""Verify displaced writeback and normalized compound loss."""

import importlib.util
from argparse import Namespace
from pathlib import Path
import random

import torch

spec = importlib.util.spec_from_file_location(
    "train_and", Path(__file__).resolve().parents[1] / "scripts/train/train_and.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class Rule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.tensor(0.01))

    def step(self, state):
        return torch.cat([state[:, :1] + self.delta, state[:, 1:]], dim=1)


def trainer():
    args = Namespace(
        seed=1,
        position_gradient=True,
        device="cpu",
        displacement=4,
        hops=3,
        hop=2,
        updates_per_hop=2,
        inner_steps=6,
        settle_steps=1,
        output_weight=1,
        input_weight=0.2,
        background_weight=0.05,
        stability_weight=0.1,
        hidden_weight=0.0001,
    )
    return module.ANDTrainer(Rule(), args)


def test_displaced_writeback_preserves_readonly_and_gradients():
    t = trainer()
    result = t.rollout(t.batch(4), True, random.Random(4))
    initial, _, state, _, mask = result
    assert state.shape == (4, 16, 40, 40)
    assert torch.equal(initial[:, 1:4], state[:, 1:4])
    assert torch.equal(initial[:, :, :2, :2], state[:, :, :2, :2])
    assert mask.sum() > 0
    loss, _, _ = t.loss(result)
    loss.backward()
    assert torch.isfinite(t.model.delta.grad) and t.model.delta.grad.abs() > 0


def test_compound_loss_does_not_change_with_batch_replication():
    t = trainer()
    result = t.rollout(t.batch(4), False, random.Random(0))
    original = t.loss(result)[0]
    repeated = t.loss(
        tuple(
            v.repeat(2, 1, 1, 1) if v.ndim == 4 else v.repeat(2, 1, 1) for v in result
        )
    )[0]
    torch.testing.assert_close(original, repeated)


def test_position_ramp_is_board_anchored_not_window_relative():
    t = trainer()
    state, _, _ = t.batch(4)
    ramp = state[0, 3]
    assert ramp[0, 0] == -1
    assert ramp[-1, -1] == 1
    torch.testing.assert_close(ramp[1:, :] - ramp[:-1, :], torch.full((31, 32), 1 / 31))
    padded = torch.nn.functional.pad(state, (4, 4, 4, 4))
    canonical = padded[0, 3, 4:36, 4:36]
    displaced = padded[0, 3, 4:36, 6:38]
    # The same world point appears two columns earlier after the window moves.
    torch.testing.assert_close(canonical[:, 2:], displaced[:, :30])
    assert not torch.equal(canonical, displaced)


def test_gaussian_head_scales_delta_without_fading_existing_state():
    torch.manual_seed(7)
    model = module.NeuralCA(
        channels=16,
        hidden_channels=[4],
        fire_rate=1,
        alive_threshold=0,
        zero_initialization=False,
        kernel_size=3,
        read_only_dims=[1, 2, 3],
        padding_type="zeros",
    )
    state = torch.randn(1, 16, 32, 32) * 0.1
    coordinates = torch.arange(32) - 15.5
    distance2 = coordinates[:, None].square() + coordinates[None, :].square()
    mask = torch.exp(-(distance2 - 0.5) / (2 * 8**2))[None, None]
    uniform = model.step(state)
    soft = model.step(state, update_mask=mask)
    torch.testing.assert_close(soft, state + (uniform - state) * mask)
    torch.testing.assert_close(
        model.step(state, update_mask=torch.zeros_like(mask)), state
    )
    assert mask.max() == 1 and mask[0, 0, 0, 0] < 0.03
    assert torch.equal(soft[:, 1:4], state[:, 1:4])


def test_context_halo_reads_neighbors_without_writing_outside_window():
    class NeighborRule(torch.nn.Module):
        def step(self, state):
            right = torch.nn.functional.pad(state[:, :1], (0, 1, 0, 0))[:, :, :, 1:]
            return torch.cat([right, state[:, 1:]], dim=1)

    t = trainer()
    t.model = NeighborRule()
    t.args.context_padding = 3
    state = torch.zeros(1, 16, 40, 40)
    state[:, 0, 4:36, 36] = 1  # Just outside the active window, inside its halo.
    result = t.advance_window(state, 4, 4, 2)
    assert torch.all(result[:, 0, 4:36, 34:36] == 1)
    assert torch.equal(result[:, :, :, 36:], state[:, :, :, 36:])
    assert torch.equal(result[:, :, :4], state[:, :, :4])


def test_blank_input_has_no_gate_and_fullscreen_loss_detects_noise():
    t = trainer()
    t.args.blank_examples = True
    state, target, mask = t.batch(4)
    assert state.shape[0] == 5
    assert torch.count_nonzero(state[-1, :3]) == 0
    assert torch.count_nonzero(state[-1, 3]) > 0
    result = (state, state, state, target, mask)
    _, terms, _ = t.loss(result)
    assert terms["blank"] == 0
    noisy = state.clone()
    noisy[-1, 0, 0, 0] = 0.8  # Outside the small output-circle mask.
    noisy.requires_grad_()
    _, terms, _ = t.loss((state, state, noisy, target, mask))
    torch.testing.assert_close(terms["blank"], torch.tensor(0.8**2 / (32 * 32 * 2)))
    terms["blank"].backward()
    assert noisy.grad[-1, 0, 0, 0] > 0


def test_random_hops_cover_both_axes_and_full_range_without_clamping():
    t = trainer()
    t.args.hop_min, t.args.hop_max = 2, 24
    t.args.displacement, t.args.hops = 24, 6
    distances, signs = set(), set()
    for seed in range(100):
        origins = t.displacement_origins(random.Random(seed))
        assert origins[0] == (24, 24)
        axes = set()
        for start, end in zip(origins, origins[1:]):
            dy, dx = end[0] - start[0], end[1] - start[1]
            assert (dy == 0) != (dx == 0)
            distance = abs(dy or dx)
            assert 2 <= distance <= 24
            assert all(0 <= coordinate <= 48 for coordinate in end)
            axes.add(0 if dy else 1)
            distances.add(distance)
            signs.add(1 if (dy or dx) > 0 else -1)
        assert axes == {0, 1}
    assert distances == set(range(2, 25))
    assert signs == {-1, 1}


def test_random_rollout_length_is_total_budget_not_budget_per_hop():
    t = trainer()
    t.args.rollout_steps_min = 64
    t.args.rollout_steps_max = 96
    for displaced in [False, True]:
        for seed in [1, 2, 3]:
            expected = random.Random(seed).randint(64, 96)
            result = t.rollout(t.batch(4), displaced, random.Random(seed))
            assert t.last_rollout_steps == expected
            if not displaced:
                initial, before, final, _, _ = result
                torch.testing.assert_close(
                    before[:, 0] - initial[:, 0],
                    torch.full_like(before[:, 0], expected * 0.01),
                )
                torch.testing.assert_close(
                    final[:, 0] - before[:, 0], torch.full_like(final[:, 0], 0.01)
                )
