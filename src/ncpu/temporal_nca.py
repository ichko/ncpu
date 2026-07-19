"""TemporalSobelNCA — temporal-inception NCA with multi-scale Sobel perception.

At each update step, perception is computed by applying Sobel filters of
progressively larger kernel sizes to the corresponding past states:

    slice k=0 (x_t):     3×3 Sobel
    slice k=1 (x_{t-1}): 5×5 Sobel
    slice k=2 (x_{t-2}): 7×7 Sobel
    ...
    slice k     (x_{t-k}): (2k+3)×(2k+3) Sobel

Each Sobel perception emits identity + sobel_x + sobel_y per input channel
(3·C feature maps). Outputs from all history slices are concatenated along
the channel axis and fed into a 1×1-conv MLP rule (NCARule), so the rule
can mix spatial gradients across time scales.

The kernel-size progression matches the light cone of a 3×3 NCA: at depth
k, info from distance k can have arrived if it propagates 1 cell/step.

Forward signature mirrors ``NeuralCA`` and ``SpaceTimeAttentionNCA``:
``forward(x, steps)`` returns ``(B, steps+1, C, H, W)``. History is stateful
across forward() calls — call ``reset_history()`` between independent
rollouts. ``history=1`` collapses to a vanilla 3×3 Sobel NCA.
"""

from typing import Optional

import torch
import torch.nn as nn

from ncpu.nca_utils import (
    AliveMasking,
    NCARule,
    ReadOnlyChannels,
    SobelPerception,
    StochasticUpdate,
)


class TemporalSobelNCA(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        channels: int,
        hidden_channels,
        fire_rate: float = 0.5,
        alive_threshold: float = 0.0,
        alive_kernel_size: int = 3,
        zero_initialization: bool = False,
        padding_type: str = "zeros",
        read_only_dims: Optional[list] = None,
        history: int = 3,
        clip_value: float = 10.0,
        num_perception_kernels: int = 3,   # identity + sobel_x + sobel_y
        kernel_sizes: Optional[list] = None,
    ) -> None:
        super().__init__()
        self.channels       = channels
        self.fire_rate      = fire_rate
        self.history        = history
        self.read_only_dims = list(read_only_dims) if read_only_dims else []
        self.clip_value     = clip_value

        if kernel_sizes is None:
            kernel_sizes = [2 * k + 3 for k in range(history)]
        assert len(kernel_sizes) == history, (
            f"kernel_sizes ({len(kernel_sizes)}) must equal history ({history})"
        )
        self.kernel_sizes = list(kernel_sizes)

        # One Sobel perception per history slice; kernel grows with depth.
        self.perceptions = nn.ModuleList([
            SobelPerception(kernel_size=ks, channels=channels, padding_type=padding_type)
            for ks in self.kernel_sizes
        ])

        rule_input = history * num_perception_kernels * channels
        self.rule = NCARule(rule_input, hidden_channels, channels, zero_initialization)

        self.alive_masking      = AliveMasking(alive_kernel_size, alive_threshold, padding_type)
        self.stochastic_update  = StochasticUpdate(fire_rate)
        self.read_only_channels = ReadOnlyChannels(channels, self.read_only_dims)

        # Stateful history buffer; persists across forward() calls.
        self._history_state: Optional[list] = None

    def reset_history(self) -> None:
        self._history_state = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_history_state"] = None
        return state

    def _step(self, history_buf):
        # k=0 is the current state, k=1 is t-1, ..., k=history-1 is t-(history-1).
        feats = []
        for k, p in enumerate(self.perceptions):
            slice_state = history_buf[-(k + 1)]
            feats.append(p(slice_state))
        return self.rule(torch.cat(feats, dim=1))

    def forward(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        prev = self._history_state
        if prev is None or prev[-1].shape != x.shape:
            history_buf = [torch.zeros_like(x) for _ in range(self.history - 1)] + [x]
        else:
            history_buf = list(prev[:-1]) + [x]

        seq = [x]
        for _ in range(steps):
            delta = self._step(history_buf)
            delta = self.read_only_channels(delta)
            delta = self.alive_masking(history_buf[-1], delta)
            delta = self.stochastic_update(delta)

            x_new = history_buf[-1] + delta
            torch.clip_(x_new, -self.clip_value, self.clip_value)

            history_buf = history_buf[1:] + [x_new]
            seq.append(x_new)

        self._history_state = history_buf
        seq = torch.stack(seq, dim=1)
        return seq
