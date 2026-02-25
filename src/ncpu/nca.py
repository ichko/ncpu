import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.utils import make_sobel_kernels, mass_conserving_update


class NeuralCA(nn.Module):
    def alive(self, x, alive_threshold):
        return (
            F.max_pool2d(
                x[:, : self.visual_channels, :, :], kernel_size=3, stride=1, padding=0
            ).amax(dim=1, keepdim=True)
            > alive_threshold
        )

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
        mass_conserving,
        visual_channels,
        kernel_size=3,
        padding_type="circular",
    ) -> None:
        super().__init__()

        identity, sobel_x, sobel_y = make_sobel_kernels(kernel_size)
        self.kernel_size = kernel_size
        self.padding_size = kernel_size // 2  # same padding

        self.total_channels = channels
        all_filters = torch.stack((identity, sobel_x, sobel_y))
        all_filters_batch = all_filters.repeat(self.total_channels, 1, 1).unsqueeze(1)
        all_filters_batch = nn.Parameter(all_filters_batch, requires_grad=False)

        self.message_channels = channels
        self.visual_channels = visual_channels
        self.all_filters_batch = all_filters_batch
        self.rule = nn.Sequential(
            nn.Conv2d(3 * self.total_channels, hidden_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, self.total_channels, kernel_size=1, bias=False),
        )
        self.fire_rate = fire_rate
        self.alive_threshold = alive_threshold
        self.mass_conserving = mass_conserving
        self.padding_type = padding_type

        assert mass_conserving in [
            "no",
            "normal",
            "cross_channel",
        ], "Bad mass_conserving option"

        if zero_initialization:
            nn.init.zeros_(self.rule[-1].weight)

    def forward(self, x, steps):
        seq = [x]
        pad = self.kernel_size // 2

        for s in range(steps):
            x_padded = F.pad(x, (pad,) * 4, self.padding_type)
            pre_life_mask = self.alive(x_padded, self.alive_threshold)

            delta = F.conv2d(
                F.pad(x, (pad,) * 4, self.padding_type),
                self.all_filters_batch,
                groups=self.total_channels,
            )
            delta = self.rule(delta)
            if self.mass_conserving == "normal":
                affinity = delta[:, :1]
                q = x[:, :1]
                q_next = mass_conserving_update(
                    beta=self.beta,
                    q=q,
                    affinity=affinity,
                    padding_type=self.padding_type,
                    pad=pad,
                )
                x = torch.cat([q_next, x[:, 1:] + delta[:, 1:]], dim=1)
            else:
                x = x + delta

            torch.clip_(x, -2, 2)
            post_life_mask = self.alive(
                F.pad(x, (pad,) * 4, self.padding_type), self.alive_threshold
            )

            life_mask = (pre_life_mask & post_life_mask).to(x.dtype)
            if self.alive_threshold > 0 and (self.mass_conserving == "no"):
                x = x * life_mask

            seq.append(x)

        seq = torch.stack(seq)
        seq = seq.permute(1, 0, 2, 3, 4)

        return seq  # (batch, time, channels, height, width)
