from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.nca_utils import (
    AliveMasking,
    BorderMask,
    GaussianNoise,
    LazyLearnableInitialState,
    LearnableNCAPerception,
    NCARule,
    ReadOnlyChannels,
    SobelPerception,
    StochasticUpdate,
)


class NeuralCA(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        channels,
        hidden_channels,
        fire_rate,
        alive_threshold,
        zero_initialization,
        kernel_size,
        perception: Literal["static_sobel", "learnable"] = "static_sobel",
        num_perception_kernels=3,  # only used for learnable perception
        learnable_initial_state=False,
        padding_type="circular",
        read_only_dims=[],
        state_bound=10.0,
        soft_bound=False,
    ) -> None:
        """``state_bound`` / ``soft_bound`` control how cell state is limited.

        Defaults reproduce the historical behaviour exactly (hard clamp at
        +-10). Set ``soft_bound=True`` for a saturating ``b*tanh(x/b)`` instead.

        Why this exists: a HARD clamp has exactly zero gradient outside its
        range, so any cell pinned at the bound contributes nothing to backward.
        On the 3SAT tasks 0.00% of cells ever reached it and it was harmless.
        A *belief* encoding is different -- Wang tiling's ``softmax(beliefs/tau)``
        pays for ever-larger beliefs, so the dynamics is driven INTO the bound;
        at 64.3% saturation the BPTT product underflows and ``grad_norm`` goes
        to ~5e-18 permanently. That presents as a learning failure, not as a
        clipping artifact, so **check grad_norm and clip saturation before
        interpreting a flat relaxation run**. Sudoku and Ising are exposed for
        the same reason.
        """
        super().__init__()
        self.state_bound = float(state_bound)
        self.soft_bound = bool(soft_bound)
        self.learnable_initial_state = learnable_initial_state
        self.fire_rate = fire_rate
        self.read_only_dims = read_only_dims
        self.channels = channels

        if perception == "static_sobel":
            self.perception = SobelPerception(kernel_size, channels, padding_type)
        elif perception == "learnable":
            self.perception = LearnableNCAPerception(
                num_perception_kernels, channels, kernel_size, padding_type
            )

        self.rule = NCARule(
            num_perception_kernels * channels,
            hidden_channels,
            channels,
            zero_initialization,
        )
        self.alive_masking = AliveMasking(kernel_size, alive_threshold, padding_type)
        self.stochastic_update = StochasticUpdate(fire_rate)
        self.gaussian_noise = GaussianNoise(channels, read_only_dims)
        # self.border_mask = BorderMask()
        self.read_only_channels = ReadOnlyChannels(channels, read_only_dims)
        if self.learnable_initial_state:
            self.add_initial_state = LazyLearnableInitialState()

    def forward(self, x, steps):
        if self.learnable_initial_state:
            x = self.add_initial_state(x)
        seq = [x]

        for _ in range(steps):
            delta = self.perception(x)
            delta = self.rule(delta)
            delta = self.read_only_channels(delta)
            delta = self.alive_masking(x, delta)
            delta = self.stochastic_update(delta)

            x = x + delta
            # NOT in-place: clip_ mutates a tensor already recorded in the
            # graph, which can corrupt saved values for backward.
            if self.soft_bound:
                x = self.state_bound * torch.tanh(x / self.state_bound)
            else:
                x = torch.clamp(x, -self.state_bound, self.state_bound)
            x = self.gaussian_noise(x)
            seq.append(x)

        seq = torch.stack(seq)
        seq = seq.permute(1, 0, 2, 3, 4)

        return seq  # (batch, time, channels, height, width)
