"""Programmatic damages (data corruptions) for robustness research.

Two sides, mirroring the two places noise can enter the system:

- **input-side** damages corrupt the raw [0, 255] input screen before
  normalization. The trainer applies them in ``optim_step``/``evaluate``
  via :func:`apply_input_damage`. ``amplitude`` is in pixel units.
- **state-side** damages corrupt the NCA state during the rollout.
  ``NeuralCA.forward`` applies them each step (like ``GaussianNoise``)
  via :func:`apply_state_damage`, respecting read-only channels.
  ``amplitude`` is on the [-10, 10] state scale.

All generators are pure torch and draw from the global torch RNG, so
``torch.manual_seed`` (e.g. ``--seed``) makes runs reproducible.

A :class:`Damage` is a small config: ``kind``, ``amplitude``, ``prob``
(fire probability per application) and a free-form ``params`` dict for
kind-specific knobs (octaves, block_size, levels, ...). CLI specs look
like ``"perlin:40:0.5:octaves=3"`` → ``kind:amplitude[:prob[:k=v,...]]``.

Conventions / assumptions:
- Amplitude units differ by side: input damages are in [0, 255] pixel units
  on the raw screen; state damages are on the [-10, 10] NCA state scale.
- Fire semantics: input damages fire per batch element with ``prob``;
  state damages fire per rollout step with ``prob`` (matching GaussianNoise).
  ``NCPUTrainer(damage_mode="sample")`` draws one input and one state damage
  per step; ``damage_mode="all"`` stacks every configured damage.
- In ``NCPUTrainer.evaluate`` the legacy ``gaussian_noise`` input noise is
  still applied after normalization when > 0 (pre-existing behaviour); the
  damage pipeline itself always runs pre-normalization, like ``optim_step``.
"""

from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch
import torch.nn.functional as F

INPUT_MIN, INPUT_MAX = 0.0, 255.0
STATE_MIN, STATE_MAX = -10.0, 10.0


@dataclass
class Damage:
    kind: str
    amplitude: float
    prob: float = 1.0
    params: dict = field(default_factory=dict)

    @property
    def side(self) -> str:
        return DAMAGE_REGISTRY[self.kind]["side"]

    def label(self) -> str:
        s = f"{self.kind}:a{self.amplitude:g}"
        if self.prob < 1.0:
            s += f":p{self.prob:g}"
        if self.params:
            s += ":" + ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return s

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "amplitude": self.amplitude,
            "prob": self.prob,
            "params": dict(self.params),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Damage":
        return cls(
            kind=d["kind"],
            amplitude=float(d["amplitude"]),
            prob=float(d.get("prob", 1.0)),
            params=dict(d.get("params", {})),
        )

    @classmethod
    def parse(cls, spec: str) -> "Damage":
        """Parse ``kind:amplitude[:prob[:k=v,k=v]]``."""
        parts = spec.split(":")
        if len(parts) < 2 or not parts[0]:
            raise ValueError(
                f"damage spec must be 'kind:amplitude[:prob[:k=v,...]]', got {spec!r}"
            )
        kind = parts[0]
        if kind not in DAMAGE_REGISTRY:
            raise ValueError(
                f"unknown damage kind {kind!r}; available: {sorted(DAMAGE_REGISTRY)}"
            )
        amplitude = float(parts[1])
        prob = float(parts[2]) if len(parts) > 2 and parts[2] != "" else 1.0
        params = {}
        if len(parts) > 3 and parts[3]:
            for kv in parts[3].split(","):
                k, v = kv.split("=")
                params[k.strip()] = _parse_value(v.strip())
        return cls(kind=kind, amplitude=amplitude, prob=prob, params=params)


def _parse_value(v: str):
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def parse_damage_specs(specs: Optional[Sequence[str]]) -> list:
    return [Damage.parse(s) for s in (specs or [])]


# ── Shared generators ────────────────────────────────────────────────────────


def perlin_field_2d(B, H, W, octaves=4, base_scale=8, device=None, dtype=None):
    """Smooth value-noise field of shape (B, 1, H, W), roughly in [-1, 1].

    Random coarse grids are bilinearly upsampled per octave and summed with
    halving amplitude — no permutation tables, no new dependencies.
    """
    octaves = max(1, int(octaves))
    field = torch.zeros(B, 1, H, W, device=device, dtype=dtype)
    amp, total, size = 1.0, 0.0, max(2, int(base_scale))
    for _ in range(octaves):
        gh = max(2, H // size + 2)
        gw = max(2, W // size + 2)
        grid = torch.rand(B, 1, gh, gw, device=device, dtype=dtype) * 2 - 1
        up = F.interpolate(grid, size=(H, W), mode="bilinear", align_corners=False)
        field = field + amp * up
        total += amp
        amp *= 0.5
        size *= 2
    return field / max(total, 1e-6)


def _gaussian_kernel_1d(sigma, device, dtype):
    radius = max(1, int(3 * sigma))
    k = 2 * radius + 1
    xs = torch.arange(k, device=device, dtype=dtype) - radius
    ker = torch.exp(-(xs**2) / (2 * sigma**2))
    return ker / ker.sum(), radius


def _as_float_bchw(img):
    if not torch.is_floating_point(img):
        img = img.float()
    if img.dim() == 3:
        img = img.unsqueeze(1)
    return img


# ── Input-side damages (img: (B, 1, H, W) in [0, 255]) ───────────────────────


def _input_gaussian(x, d):
    return x + torch.randn_like(x) * d.amplitude


def _input_perlin(x, d):
    field = perlin_field_2d(
        x.shape[0],
        x.shape[2],
        x.shape[3],
        octaves=d.params.get("octaves", 4),
        base_scale=d.params.get("base_scale", 8),
        device=x.device,
        dtype=x.dtype,
    )
    return x + field * d.amplitude


def _input_salt_pepper(x, d):
    mask = torch.rand_like(x) < d.amplitude
    vals = torch.where(
        torch.rand_like(x) < 0.5,
        torch.full_like(x, INPUT_MAX),
        torch.full_like(x, INPUT_MIN),
    )
    return torch.where(mask, vals, x)


def _input_occlusion(x, d):
    if d.amplitude <= 0:
        return x
    B, _, H, W = x.shape
    n_blocks = max(1, int(round(d.amplitude)))
    bs = max(1, int(d.params.get("block_size", 8)))
    fill = float(d.params.get("fill", 0))
    out = x.clone()
    for b in range(B):
        for _ in range(n_blocks):
            y = int(torch.randint(0, max(1, H - bs + 1), (1,)).item())
            xx = int(torch.randint(0, max(1, W - bs + 1), (1,)).item())
            out[b, :, y : y + bs, xx : xx + bs] = fill
    return out


def _input_blur(x, d):
    sigma = d.amplitude
    if sigma <= 0:
        return x
    ker, radius = _gaussian_kernel_1d(sigma, x.device, x.dtype)
    C = x.shape[1]
    ky = ker.view(1, 1, -1, 1).expand(C, 1, -1, 1)
    kx = ker.view(1, 1, 1, -1).expand(C, 1, 1, -1)
    out = F.conv2d(F.pad(x, (0, 0, radius, radius), mode="replicate"), ky, groups=C)
    out = F.conv2d(F.pad(out, (radius, radius, 0, 0), mode="replicate"), kx, groups=C)
    return out


def _input_brightness_contrast(x, d):
    a = d.amplitude
    return (x - 128.0) * (1.0 + a) + 128.0 + a * 64.0


def _input_quantize(x, d):
    levels = int(d.params.get("levels", 4))
    if levels >= 256:
        return x
    step = INPUT_MAX / (levels - 1)
    return torch.round(x / step) * step


def _input_jitter(x, d):
    a = int(round(d.amplitude))
    if a <= 0:
        return x
    dy = int(torch.randint(-a, a + 1, (1,)).item())
    dx = int(torch.randint(-a, a + 1, (1,)).item())
    return torch.roll(x, shifts=(dy, dx), dims=(2, 3))


def _input_speckle(x, d):
    return x * (1.0 + torch.randn_like(x) * d.amplitude)


# ── State-side damages (x: (B, C, H, W) in [-10, 10]) ────────────────────────


def _mask_or_ones(x, channel_mask):
    if channel_mask is None:
        return torch.ones(1, x.shape[1], 1, 1, device=x.device, dtype=x.dtype)
    return channel_mask.to(device=x.device, dtype=x.dtype)


def _state_gaussian(x, d, channel_mask):
    return x + torch.randn_like(x) * d.amplitude * _mask_or_ones(x, channel_mask)


def _state_perlin(x, d, channel_mask):
    field = perlin_field_2d(
        x.shape[0],
        x.shape[2],
        x.shape[3],
        octaves=d.params.get("octaves", 4),
        base_scale=d.params.get("base_scale", 8),
        device=x.device,
        dtype=x.dtype,
    )
    return x + field * d.amplitude * _mask_or_ones(x, channel_mask)


def _state_channel_dropout(x, d, channel_mask):
    C = x.shape[1]
    keep = (torch.rand(x.shape[0], C, 1, 1, device=x.device) >= d.amplitude).to(x.dtype)
    mask = _mask_or_ones(x, channel_mask)
    keep = torch.where(mask.bool(), keep, torch.ones_like(keep))
    return x * keep


def _state_block_dropout(x, d, channel_mask):
    if d.amplitude <= 0:
        return x
    B, _, H, W = x.shape
    n_blocks = max(1, int(round(d.amplitude)))
    bs = max(1, int(d.params.get("block_size", 8)))
    block = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
    for b in range(B):
        for _ in range(n_blocks):
            y = int(torch.randint(0, max(1, H - bs + 1), (1,)).item())
            xx = int(torch.randint(0, max(1, W - bs + 1), (1,)).item())
            block[b, 0, y : y + bs, xx : xx + bs] = 1.0
    mask = _mask_or_ones(x, channel_mask)
    return x * (1.0 - block * mask)


DAMAGE_REGISTRY = {
    # input side
    "gaussian": {"side": "input", "fn": _input_gaussian},
    "perlin": {"side": "input", "fn": _input_perlin},
    "salt_pepper": {"side": "input", "fn": _input_salt_pepper},
    "occlusion": {"side": "input", "fn": _input_occlusion},
    "blur": {"side": "input", "fn": _input_blur},
    "brightness_contrast": {"side": "input", "fn": _input_brightness_contrast},
    "quantize": {"side": "input", "fn": _input_quantize},
    "jitter": {"side": "input", "fn": _input_jitter},
    "speckle": {"side": "input", "fn": _input_speckle},
    # state side
    "state_gaussian": {"side": "state", "fn": _state_gaussian},
    "state_perlin": {"side": "state", "fn": _state_perlin},
    "channel_dropout": {"side": "state", "fn": _state_channel_dropout},
    "block_dropout": {"side": "state", "fn": _state_block_dropout},
}

INPUT_DAMAGE_KINDS = sorted(
    k for k, v in DAMAGE_REGISTRY.items() if v["side"] == "input"
)
STATE_DAMAGE_KINDS = sorted(
    k for k, v in DAMAGE_REGISTRY.items() if v["side"] == "state"
)


def apply_input_damage(img, damage: Damage):
    """Apply an input-side damage to a raw screen batch.

    ``img`` is (B, H, W) or (B, 1, H, W) in [0, 255]. Each batch element is
    damaged independently with probability ``damage.prob``. Returns a
    float tensor clamped to [0, 255] with the same shape as the input.
    """
    if DAMAGE_REGISTRY[damage.kind]["side"] != "input":
        raise ValueError(f"{damage.kind} is a state-side damage, not input-side")
    squeeze = img.dim() == 3
    x = _as_float_bchw(img)
    out = DAMAGE_REGISTRY[damage.kind]["fn"](x, damage).clamp(INPUT_MIN, INPUT_MAX)
    if damage.prob < 1.0:
        fire = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < damage.prob
        out = torch.where(fire, out, x)
    return out.squeeze(1) if squeeze else out


def apply_state_damage(x, damage: Damage, channel_mask=None):
    """Apply a state-side damage to an NCA state (B, C, H, W) in [-10, 10].

    ``channel_mask`` (1, C, 1, 1) marks writable channels; read-only
    channels (mask 0) are never modified. Returns a tensor clamped to
    [-10, 10].
    """
    if DAMAGE_REGISTRY[damage.kind]["side"] != "state":
        raise ValueError(f"{damage.kind} is an input-side damage, not state-side")
    out = DAMAGE_REGISTRY[damage.kind]["fn"](x, damage, channel_mask)
    return out.clamp(STATE_MIN, STATE_MAX)
