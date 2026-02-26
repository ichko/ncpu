import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.utils import make_sobel_kernels


class NeuralCA(nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        fire_rate,
        alive_threshold,
        zero_initialization,
        mass_conserving,
        kernel_size,
        num_perception_kernels,
        learnable_kernels=False,
        learnable_initial_state=False,
        padding_type="circular",
        read_only_dims=[],
    ) -> None:
        super().__init__()

        # identity, sobel_x, sobel_y = make_sobel_kernels(kernel_size)
        # all_filters = torch.stack((identity, sobel_x, sobel_y))
        # self.num_perception_kernels = len(all_filters)
        # all_filters_batch = all_filters.repeat(self.channels, 1, 1).unsqueeze(1)
        # self.all_filters_batch = nn.Parameter(all_filters_batch, requires_grad=False)

        self.num_perception_kernels = num_perception_kernels
        self.kernel_size = kernel_size
        self.channels = channels
        self.learnable_initial_state = learnable_initial_state

        self.fire_rate = fire_rate
        self.alive_threshold = alive_threshold
        self.mass_conserving = mass_conserving
        self.padding_type = padding_type
        self.read_only_dims = read_only_dims

        self.read_only_mask = nn.Parameter(
            torch.ones(1, self.channels, 1, 1), requires_grad=False
        )
        for c in self.read_only_dims:
            if c < 0:
                c = self.channels + c  # -1 => 15
            self.read_only_mask[:, c] = 0

        self.perception = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels * num_perception_kernels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode=self.padding_type,
            groups=self.channels,
        )

        self.rule = nn.Sequential(
            nn.Conv2d(
                num_perception_kernels * self.channels, hidden_channels, kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, self.channels, kernel_size=1, bias=False),
        )
        if learnable_initial_state:
            self.register_parameter("init_state", None)

        if zero_initialization:
            nn.init.zeros_(self.rule[-1].weight)

    def forward(self, x, steps):
        if self.learnable_initial_state:
            if self.init_state == None:
                self.init_state = nn.Parameter(
                    torch.rand(1, *x.shape[1:], device=self.device)
                )
            x += self.init_state
        seq = [x]

        for s in range(steps):
            pre_life_mask = self.alive(x)

            # delta = F.conv2d(
            #     F.pad(x, (pad,) * 4, self.padding_type),
            #     self.all_filters_batch,
            #     groups=self.channels,
            # )
            delta = self.perception(x)
            delta = self.rule(delta)
            delta = delta * self.read_only_mask

            if self.alive_threshold > 0:
                post_life_mask = self.alive(x)
                life_mask = (pre_life_mask & post_life_mask).to(x.dtype)
                delta = delta * life_mask

            x = x + delta
            torch.clip_(x, -10, 10)

            seq.append(x)

        seq = torch.stack(seq)
        seq = seq.permute(1, 0, 2, 3, 4)

        return seq  # (batch, time, channels, height, width)

    def alive(self, x):
        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), self.padding_type)
        pooled = F.max_pool2d(
            x_padded[:, :1, :, :],
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )
        return pooled.amax(dim=1, keepdim=True) > self.alive_threshold

    @property
    def device(self):
        return next(self.parameters()).device
