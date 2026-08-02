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
        gaussian_noise: float = 0.0,
        gaussian_noise_fire_rate: float = 0.0,
    ) -> None:
        super().__init__()
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
        self.gaussian_noise.std = gaussian_noise
        self.gaussian_noise.prob = gaussian_noise_fire_rate
        # self.border_mask = BorderMask()
        self.read_only_channels = ReadOnlyChannels(channels, read_only_dims)
        if self.learnable_initial_state:
            self.add_initial_state = LazyLearnableInitialState()

    def forward(self, x, steps, noise_std=None, noise_fire_rate=None):
        if self.learnable_initial_state:
            x = self.add_initial_state(x)

        g = self.gaussian_noise
        saved_std, saved_prob = g.std, g.prob
        if noise_std is not None:
            g.std = noise_std
        if noise_fire_rate is not None:
            g.prob = noise_fire_rate

        seq = [x]
        for _ in range(steps):
            delta = self.perception(x)
            delta = self.rule(delta)
            delta = self.read_only_channels(delta)
            delta = self.alive_masking(x, delta)
            delta = self.stochastic_update(delta)

            x = x + delta
            torch.clip_(x, -10, 10)
            x = g(x)
            seq.append(x)

        g.std, g.prob = saved_std, saved_prob

        seq = torch.stack(seq)
        seq = seq.permute(1, 0, 2, 3, 4)

        return seq  # (batch, time, channels, height, width)
