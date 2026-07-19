"""GatedNCA — NCA with a per-cell gated (GRU-style) state update.

Motivation: the residual ALU2 failures (carry_out, branch_taken, long ADD/SUB
carry chains) all require cells to *latch* intermediate state — flags and
running carries — across many stochastic update steps. The vanilla additive
rule must re-derive that state every step; a gated rule lets a cell decide
locally to hold its value.

Update per step (m = stochastic fire mask):

    feats = concat_k Sobel_{ks[k]}(x_{t-k})          # temporal-inception
    trunk = MLP_1x1(feats)
    z     = sigmoid(Conv1x1(trunk))                  # update gate, per channel
    h~    = tanh(Conv1x1(trunk)) * candidate_scale   # candidate state
    x'    = x + m * z * (h~ - x)                     # convex move toward h~

z -> 0 holds the current value exactly (a latch); z -> 1 rewrites the cell.
``history=1`` with ``kernel_sizes=[7]`` is the pure-gating control (champion
perception, gated update). ``history=3`` adds the multi-scale temporal
perception of ``TemporalSobelNCA`` (7/9/11 ladder by default).

Forward signature mirrors ``NeuralCA``/``TemporalSobelNCA``:
``forward(x, steps)`` returns ``(B, steps+1, C, H, W)``. History is stateful
across forward() calls — call ``reset_history()`` between independent rollouts.
"""

from typing import Optional

import torch
import torch.nn as nn

from ncpu.nca_utils import (
    AliveMasking,
    ReadOnlyChannels,
    SobelPerception,
    StochasticUpdate,
)


class GatedNCA(nn.Module):
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
        padding_type: str = "zeros",
        read_only_dims: Optional[list] = None,
        history: int = 1,
        kernel_sizes: Optional[list] = None,
        clip_value: float = 10.0,
        candidate_scale: float = 2.0,
        num_perception_kernels: int = 3,   # identity + sobel_x + sobel_y
    ) -> None:
        super().__init__()
        self.channels        = channels
        self.fire_rate       = fire_rate
        self.history         = history
        self.read_only_dims  = list(read_only_dims) if read_only_dims else []
        self.clip_value      = clip_value
        self.candidate_scale = candidate_scale

        if kernel_sizes is None:
            kernel_sizes = [2 * k + 7 for k in range(history)]
        assert len(kernel_sizes) == history, (
            f"kernel_sizes ({len(kernel_sizes)}) must equal history ({history})"
        )
        self.kernel_sizes = list(kernel_sizes)

        self.perceptions = nn.ModuleList([
            SobelPerception(kernel_size=ks, channels=channels, padding_type=padding_type)
            for ks in self.kernel_sizes
        ])

        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels]
        rule_input = history * num_perception_kernels * channels
        trunk_layers = []
        prev = rule_input
        for h in hidden_channels:
            trunk_layers += [nn.Conv2d(prev, h, kernel_size=1), nn.ReLU()]
            prev = h
        self.trunk     = nn.Sequential(*trunk_layers)
        self.gate_head = nn.Conv2d(prev, channels, kernel_size=1)
        self.cand_head = nn.Conv2d(prev, channels, kernel_size=1, bias=False)

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
        feats = []
        for k, p in enumerate(self.perceptions):
            feats.append(p(history_buf[-(k + 1)]))
        trunk = self.trunk(torch.cat(feats, dim=1))
        z     = torch.sigmoid(self.gate_head(trunk))
        cand  = torch.tanh(self.cand_head(trunk)) * self.candidate_scale
        return z * (cand - history_buf[-1])

    def forward(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        prev = self._history_state
        if prev is None or prev[-1].shape != x.shape:
            history_buf = [torch.zeros_like(x) for _ in range(self.history - 1)] + [x]
        else:
            history_buf = list(prev[:-1]) + [x]

        seq = [x]
        for _ in range(steps):
            change = self._step(history_buf)
            change = self.read_only_channels(change)
            change = self.alive_masking(history_buf[-1], change)
            change = self.stochastic_update(change)

            x_new = history_buf[-1] + change
            torch.clip_(x_new, -self.clip_value, self.clip_value)

            history_buf = history_buf[1:] + [x_new]
            seq.append(x_new)

        self._history_state = history_buf
        seq = torch.stack(seq, dim=1)
        return seq
