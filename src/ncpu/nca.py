from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.nca_utils import (
    AliveMasking,
    BorderMask,
    LazyLearnableInitialState,
    LearnableNCAPerception,
    NCARule,
    ReadOnlyChannels,
    SobelPerception,
    StochasticUpdate,
)

class NeuralCAv1(nn.Module):
    def alive(self, x):
        # return F.max_pool2d(x[:, :1, :, :], kernel_size=3, stride=1, padding=0) > 0.1
        return F.max_pool2d(abs(0.5-x[:, :1, :, :]), kernel_size=3, stride=1, padding=0) > 0.1

    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        channels,
        hidden_channels,
        fire_rate,
        alive_masking_flag,
        zero_initialization,
        kernel_size = 5,
    ) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        all_filters = torch.stack((identity, sobel_x, sobel_y))
        all_filters_batch = all_filters.repeat(channels, 1, 1).unsqueeze(1)
        all_filters_batch = nn.Parameter(all_filters_batch, requires_grad=False)

        self.kernel_size = kernel_size
        self.channels = channels
        self.all_filters_batch = all_filters_batch
        self.rule = nn.Sequential(
            nn.Conv2d(3 * channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.fire_rate = fire_rate
        self.alive_masking_flag = alive_masking_flag

        if zero_initialization:
            nn.init.zeros_(self.rule[-1].weight)

        self.padding = (self.kernel_size - 1) // 2

    def perception(self, x, pad_type):
        delta = F.conv2d(
            F.pad(x, (self.padding+1,) * 4, pad_type),
            self.all_filters_batch,
            groups=self.channels,
        )
        return delta

    def alive(self, x):
        x = torch.abs(x)
        pooled = F.max_pool2d(
            x[:, :1, :, :],
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )
        pooled = pooled.amax(dim=1, keepdim=True) > 0.1
        return pooled

    def alive_masking(self, x, delta):
        if self.alive_masking_flag > 0:
            pre_life_mask = self.alive(x)
            post_life_mask = self.alive(x + delta)
            life_mask = (pre_life_mask & post_life_mask).to(x.dtype)
            print("after max pool: ",pre_life_mask.shape, post_life_mask.shape)
            delta = delta * life_mask
        return delta

    def stochastic_update(self, delta):
        if self.fire_rate >= 1.0:
            return delta
        fire_mask = (
            torch.rand(delta.shape[0], 1, delta.shape[2], delta.shape[3], device=delta.device)
            < self.fire_rate
        ).to(delta.dtype)
        return delta * fire_mask

    def forward(self, x, steps = 1):
        seq = [x]

        # pad_type = "circular"
        pad_type = "constant"
        for _ in range(steps):
            x_before = F.pad(x, (self.padding, ) * 4, pad_type)
            delta = self.perception(x, pad_type)
            print("1: ", delta.shape, x.shape, (self.padding, ) * 4)
            delta = self.rule(delta)
            print("2: ", delta.shape, x_before.shape)
            delta = self.alive_masking(x_before, delta)
            print("3: ", delta.shape)
            delta = self.stochastic_update(delta)
            print("4: ", delta.shape)
            x = x + delta[:, :, 1:-1, 1:-1]
            print(x)
            
            seq.append(x)

        seq = torch.stack(seq)
        seq = seq.permute(1, 0, 2, 3, 4)
        return seq

class NeuralCAv2(nn.Module):
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
            # delta = self.border_mask(delta)

            x = x + delta
            torch.clip_(x, -10, 10)
            seq.append(x)

        seq = torch.stack(seq)
        seq = seq.permute(1, 0, 2, 3, 4)

        return seq  # (batch, time, channels, height, width)


class NeuralCALegacy(nn.Module):
    def alive(self, x):
        # return F.max_pool2d(x[:, :1, :, :], kernel_size=3, stride=1, padding=0) > 0.1
        return (
            F.max_pool2d(abs(0.5 - x[:, :1, :, :]), kernel_size=3, stride=1, padding=0)
            > 0.1
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        channels,
        hidden_channels,
        fire_rate,
        alive_masking_flag,
        zero_initialization,
    ) -> None:
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]) / 8
        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        all_filters = torch.stack((identity, sobel_x, sobel_y))
        all_filters_batch = all_filters.repeat(channels, 1, 1).unsqueeze(1)
        all_filters_batch = nn.Parameter(all_filters_batch, requires_grad=False)

        self.channels = channels
        self.all_filters_batch = all_filters_batch
        self.rule = nn.Sequential(
            nn.Conv2d(3 * channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.fire_rate = fire_rate
        self.alive_masking_flag = alive_masking_flag

        if zero_initialization:
            nn.init.zeros_(self.rule[-1].weight)

    def perception(self, x, pad_type):
        delta = F.conv2d(
            F.pad(x, (1, 1, 1, 1), pad_type),
            self.all_filters_batch,
            groups=self.channels,
        )
        return delta

    def alive_masking(self, x, x_padded, pad_type):
        if self.alive_masking_flag:
            pre_life_mask = self.alive(x_padded)
            post_life_mask = self.alive(F.pad(x, (1, 1, 1, 1), pad_type))
            life_mask = (pre_life_mask & post_life_mask).to(x.dtype)
            x = x * life_mask
        return x

    def forward(self, x, steps=1):
        seq = [x]

        # pad_type = "circular"
        pad_type = "constant"

        for _ in range(steps):
            x_padded = F.pad(x, (1, 1, 1, 1), pad_type)
            delta = self.perception(x, pad_type)
            delta = self.rule(delta)
            x = x + delta
            x = self.alive_masking(x, x_padded, pad_type)

            seq.append(x)

        seq = torch.stack(seq)
        seq = seq.permute(1, 0, 2, 3, 4)
        return seq
