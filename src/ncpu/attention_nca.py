"""SpaceTimeAttentionNCA — light-cone attention NCA (ported from nca-lm).

Each cell update is computed via self-attention over a "light cone" of
positions in space-time:
  - 3x3 neighborhood at the current step  (9 positions)
  - 5x5 neighborhood at t-1                (25 positions)
  - 7x7 neighborhood at t-2                (49 positions)
  - ...(2k+3)^2 at t-k...

Spatial window grows linearly with temporal distance, matching the maximum
information-propagation speed of an NCA with 3x3 local kernels.

Forward signature matches NeuralCA: forward(x, steps) returns
(B, steps+1, C, H, W).  History is stateful across forward() calls — call
reset_history() between independent rollouts.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ncpu.nca_utils import AliveMasking, ReadOnlyChannels, StochasticUpdate


class SpaceTimeAttentionNCA(nn.Module):
    @property
    def device(self):
        return next(self.parameters()).device

    def __init__(
        self,
        channels: int,
        history: int = 3,
        num_heads: int = 1,
        fire_rate: float = 1.0,
        padding_type: str = "zeros",
        clip_value: float = 10.0,
        output_scale: float = 0.5,
        light_cone_mode: str = "perimeter",   # "full" or "perimeter"
        alive_threshold: float = 0.0,
        alive_kernel_size: int = 3,
        pe_mode: str = "additive",            # "additive" or "concat"
        temporal_pe_dim: int = 4,             # only used in concat mode
        spatial_pe_dim: int = 8,              # only used in concat mode
        read_only_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        assert light_cone_mode in ("full", "perimeter")
        assert pe_mode in ("additive", "concat")

        self.pe_mode         = pe_mode
        self.temporal_pe_dim = temporal_pe_dim if pe_mode == "concat" else 0
        self.spatial_pe_dim  = spatial_pe_dim  if pe_mode == "concat" else 0

        self.channels        = channels
        feature_dim          = channels + self.temporal_pe_dim + self.spatial_pe_dim
        assert feature_dim % num_heads == 0, (
            f"channels + PE ({feature_dim}) must be divisible by num_heads ({num_heads})"
        )

        self.history         = history
        self.num_heads       = num_heads
        self.feature_dim     = feature_dim
        self.head_dim        = feature_dim // num_heads
        self.fire_rate       = fire_rate
        self.padding_type    = "constant" if padding_type == "zeros" else padding_type
        self.clip_value      = clip_value
        self.light_cone_mode = light_cone_mode
        self.scale           = self.head_dim ** -0.5
        self.read_only_dims  = read_only_dims or []

        for k in range(history):
            window = 2 * k + 3
            mask = torch.zeros(window, window, dtype=torch.bool)
            if k == 0:
                mask[:] = True
            else:
                mask[0,  :] = True
                mask[-1, :] = True
                mask[:, 0]  = True
                mask[:, -1] = True
            idx = mask.flatten().nonzero(as_tuple=True)[0]
            self.register_buffer(f"perim_idx_{k}", idx, persistent=False)

        if pe_mode == "additive":
            self.time_embed = nn.Parameter(torch.zeros(history, channels))
            nn.init.normal_(self.time_embed, std=0.02)
            self.temporal_pe_concat = None
            self.spatial_pe_concat  = None
            self.q_pe_concat        = None
        else:
            self.temporal_pe_concat = nn.Parameter(torch.zeros(history, temporal_pe_dim))
            nn.init.normal_(self.temporal_pe_concat, std=0.02)
            sps = []
            for k in range(history):
                window = 2 * k + 3
                if light_cone_mode == "perimeter" and k > 0:
                    n_pos_k = window * window - (window - 2) * (window - 2)
                else:
                    n_pos_k = window * window
                sp = nn.Parameter(torch.zeros(n_pos_k, spatial_pe_dim))
                nn.init.normal_(sp, std=0.02)
                sps.append(sp)
            self.spatial_pe_concat = nn.ParameterList(sps)
            self.q_pe_concat = nn.Parameter(torch.zeros(temporal_pe_dim + spatial_pe_dim))
            nn.init.normal_(self.q_pe_concat, std=0.02)
            self.time_embed = None

        self.q_proj = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Conv2d(channels, channels, 1)
        self.v_proj = nn.Conv2d(channels, channels, 1)
        self.o_proj = nn.Conv2d(self.feature_dim, channels, 1, bias=False)

        if output_scale != 1.0:
            with torch.no_grad():
                self.o_proj.weight.mul_(output_scale)

        self.stochastic_update  = StochasticUpdate(fire_rate)
        self.alive_masking      = AliveMasking(alive_kernel_size, alive_threshold, padding_type)
        self.read_only_channels = ReadOnlyChannels(channels, self.read_only_dims)

        self._history_state: list[torch.Tensor] | None = None

    def reset_history(self) -> None:
        self._history_state = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_history_state"] = None
        return state

    def _attention_step(self, history_buf: list[torch.Tensor]) -> torch.Tensor:
        x = history_buf[-1]
        B, C, H, W = x.shape
        h         = self.num_heads
        feat_dim  = self.feature_dim
        head_dim  = self.head_dim
        HW        = H * W

        q = self.q_proj(x)
        if self.pe_mode == "concat":
            q_pe = self.q_pe_concat.view(1, -1, 1, 1).expand(B, -1, H, W)
            q = torch.cat([q, q_pe], dim=1)
        q = q.reshape(B, h, head_dim, HW)

        all_k, all_v = [], []
        for k_offset, hist_state in enumerate(reversed(history_buf)):
            window = 2 * k_offset + 3
            pad    = window // 2

            if self.pe_mode == "additive":
                t_add = self.time_embed[k_offset].view(1, C, 1, 1)
                k_feat = self.k_proj(hist_state) + t_add
                v_feat = self.v_proj(hist_state) + t_add
            else:
                k_feat = self.k_proj(hist_state)
                v_feat = self.v_proj(hist_state)

            k_padded = F.pad(k_feat, (pad,) * 4, self.padding_type)
            v_padded = F.pad(v_feat, (pad,) * 4, self.padding_type)

            n_pos = window * window
            k_unf = F.unfold(k_padded, window).view(B, C, n_pos, HW)
            v_unf = F.unfold(v_padded, window).view(B, C, n_pos, HW)

            if self.light_cone_mode == "perimeter":
                idx   = getattr(self, f"perim_idx_{k_offset}")
                k_unf = k_unf.index_select(2, idx)
                v_unf = v_unf.index_select(2, idx)
            n_pos_eff = k_unf.shape[2]

            if self.pe_mode == "concat":
                t_pe = self.temporal_pe_concat[k_offset].view(1, -1, 1, 1).expand(B, -1, n_pos_eff, HW)
                s_pe = self.spatial_pe_concat[k_offset].t().view(1, -1, n_pos_eff, 1).expand(B, -1, n_pos_eff, HW)
                k_unf = torch.cat([k_unf, t_pe, s_pe], dim=1)
                v_unf = torch.cat([v_unf, t_pe, s_pe], dim=1)

            k_unf = k_unf.reshape(B, h, head_dim, n_pos_eff, HW)
            v_unf = v_unf.reshape(B, h, head_dim, n_pos_eff, HW)

            all_k.append(k_unf)
            all_v.append(v_unf)

        k = torch.cat(all_k, dim=3)
        v = torch.cat(all_v, dim=3)

        scores = torch.einsum("bhdl,bhdpl->bhpl", q, k) * self.scale
        attn   = F.softmax(scores, dim=2)

        out = torch.einsum("bhpl,bhdpl->bhdl", attn, v)
        out = out.reshape(B, feat_dim, H, W)

        return self.o_proj(out)

    def forward(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        prev = getattr(self, "_history_state", None)
        if prev is None or prev[-1].shape != x.shape:
            history_buf = [torch.zeros_like(x) for _ in range(self.history - 1)] + [x]
        else:
            history_buf = list(prev[:-1]) + [x]

        seq = [x]
        for _ in range(steps):
            delta = self._attention_step(history_buf)
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
