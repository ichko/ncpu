import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralCA(nn.Module):
    def alive(self, x):
        return F.max_pool2d(x[:, :1, :, :], kernel_size=3, stride=1, padding=0) > 0.1

    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        channels,
        hidden_channels,
        fire_rate,
        alive_masking,
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
        self.alive_masking = alive_masking

        if zero_initialization:
            nn.init.zeros_(self.rule[-1].weight)

        self.symetry_break = None

    def forward(self, x, steps):
        seq = [x]
        if self.symetry_break is None:
            H, W = x.shape[-2:]
            self.symetry_break = torch.rand(1, self.channels, H, W) < 0.01
            self.symetry_break = self.symetry_break.float()
            self.symetry_break = self.symetry_break.to(x.device)

        # pad_type = "circular"
        pad_type = "constant"

        x += self.symetry_break
        for _ in range(steps):
            x_padded = F.pad(x, (1, 1, 1, 1), pad_type)
            pre_life_mask = self.alive(x_padded)

            delta = F.conv2d(
                F.pad(x, (1, 1, 1, 1), pad_type),
                self.all_filters_batch,
                groups=self.channels,
            )
            delta = self.rule(delta)
            x = x + delta

            post_life_mask = self.alive(F.pad(x, (1, 1, 1, 1), pad_type))
            life_mask = (pre_life_mask & post_life_mask).to(x.dtype)
            if self.alive_masking:
                x = x * life_mask

            seq.append(x)

        seq = torch.stack(seq)
        seq = seq.permute(1, 0, 2, 3, 4)
        return seq
