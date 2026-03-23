import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.utils import conv_stack, make_sobel_kernels


class NCARule(nn.Module):
    def __init__(self, rule_input, hidden_channels, channels, zero_initialization):
        super().__init__()
        if type(hidden_channels) == int:
            hidden_channels = [hidden_channels]

        # rule_input = num_perception_kernels * channels
        layers = [rule_input, *hidden_channels]
        self.rule = nn.Sequential(
            *conv_stack(layers, activation=nn.ReLU, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels[-1], channels, kernel_size=1, bias=False),
        )

        if zero_initialization:
            nn.init.zeros_(self.rule[-1].weight)

    def forward(self, x):
        return self.rule(x)



class SobelPerception(nn.Module):
    def __init__(self, kernel_size, channels, padding_type):
        super().__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        # F.pad uses "constant" for zero-padding; nn.Conv2d calls it "zeros"
        self.padding_type = "constant" if padding_type == "zeros" else padding_type

        identity, sobel_x, sobel_y = make_sobel_kernels(kernel_size)
        all_filters = torch.stack((identity, sobel_x, sobel_y))
        all_filters_batch = all_filters.repeat(channels, 1, 1).unsqueeze(1)
        self.all_filters_batch = nn.Parameter(all_filters_batch, requires_grad=False)

    def forward(self, x):
        pad = self.kernel_size // 2
        delta = F.conv2d(
            F.pad(x, (pad,) * 4, self.padding_type),
            self.all_filters_batch,
            groups=self.channels,
        )
        return delta


class LearnableNCAPerception(nn.Module):
    def __init__(self, num_perception_kernels, channels, kernel_size, padding_type):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding_type = padding_type
        self.perception = nn.Conv2d(
            in_channels=self.channels,
            out_channels=self.channels * num_perception_kernels,
            kernel_size=self.kernel_size,
            padding=self.kernel_size // 2,
            padding_mode=self.padding_type,
            groups=self.channels // 2,
        )

    def forward(self, x):
        return self.perception(x)


class AliveMasking(nn.Module):
    def __init__(self, kernel_size, alive_threshold, padding_type):
        super().__init__()
        self.kernel_size = kernel_size
        self.alive_threshold = alive_threshold
        # F.pad uses "constant" for zero-padding; nn.Conv2d calls it "zeros"
        self.padding_type = "constant" if padding_type == "zeros" else padding_type

    def alive(self, x):
        x = torch.abs(x)
        pad = self.kernel_size // 2
        x_padded = F.pad(x, (pad, pad, pad, pad), self.padding_type)
        pooled = F.max_pool2d(
            x_padded[:, :1, :, :],
            kernel_size=self.kernel_size,
            stride=1,
            padding=0,
        )
        pooled = pooled.amax(dim=1, keepdim=True) > self.alive_threshold
        return pooled

    def forward(self, x, delta):
        if self.alive_threshold > 0:
            pre_life_mask = self.alive(x)
            post_life_mask = self.alive(x + delta)
            life_mask = (pre_life_mask & post_life_mask).to(x.dtype)
            delta = delta * life_mask
        return delta


class BorderMask(nn.Module):
    """Zeroes the delta on the 1-pixel border so border zero-padding never
    influences the interior computation."""

    def __init__(self):
        super().__init__()
        self.register_buffer("mask", None)

    def forward(self, delta):
        H, W = delta.shape[2], delta.shape[3]
        if self.mask is None or self.mask.shape[-2:] != (H, W):
            mask = torch.ones(1, 1, H, W, device=delta.device, dtype=delta.dtype)
            mask[:, :, 0, :] = 0    # top row
            mask[:, :, -1, :] = 0   # bottom row
            mask[:, :, :, 0] = 0    # left col
            mask[:, :, :, -1] = 0   # right col
            self.mask = mask
        return delta * self.mask


class StochasticUpdate(nn.Module):
    def __init__(self, fire_rate):
        super().__init__()
        self.fire_rate = fire_rate

    def forward(self, delta):
        if self.fire_rate >= 1.0:
            return delta
        fire_mask = (
            torch.rand(delta.shape[0], 1, delta.shape[2], delta.shape[3], device=delta.device)
            < self.fire_rate
        ).to(delta.dtype)
        return delta * fire_mask


class LazyLearnableInitialState(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_parameter("init_state", None)

    def forward(self, x):
        if self.init_state == None:
            self.init_state = nn.Parameter(
                torch.rand(1, *x.shape[1:], device=self.device)
            )
        x += self.init_state
        return x


class ReadOnlyChannels(nn.Module):
    def __init__(
        self, num_channels, read_only_dims
    ):  # specify channels as list of dims
        super().__init__()
        self.read_only_dims = read_only_dims
        self.read_only_mask = nn.Parameter(
            torch.ones(1, num_channels, 1, 1), requires_grad=False
        )
        for c in self.read_only_dims:
            self.read_only_mask[:, c] = 0

    def forward(self, x):
        x = x * self.read_only_mask
        return x


class GaussianNoise(nn.Module):
    """Adds Gaussian noise to all channels except read-only dims during training."""

    def __init__(self, channels: int, std: float = 0.1, fire_rate: float = 1.0, read_only_dims: list = []):
        super().__init__()
        self.std = std
        self.fire_rate = fire_rate
        # Build a mask: 1 for writable channels, 0 for read-only
        mask = torch.ones(channels)
        for d in read_only_dims:
            mask[d] = 0.0
        # shape (C,) → broadcastable to (B, C, H, W)
        self.register_buffer("mask", mask.view(1, -1, 1, 1))

    def forward(self, x):
        if self.std > 0:
            noise = torch.randn_like(x) * self.std
            noise = noise * self.mask
            if self.fire_rate < 1.0:
                # Stochastic spatial mask: (B, 1, H, W)
                fire_mask = (
                    torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
                    < self.fire_rate
                ).to(x.dtype)
                noise = noise * fire_mask
            x = x + noise
        return x
