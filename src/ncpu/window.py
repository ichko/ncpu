"""Local-gate tiles for the rolling-window sweep (infinite canvas).

A single NCA rule learns to compute several Boolean gates, one per 32×32 tile.
Each tile is fully self-contained: two input circles on the left, one output
circle on the right, and a read-only *selector* channel that tells the rule
which gate the tile should compute (or that the tile is empty). Because the
rule is translation-invariant and only ever sees a ``WINDOW_SIZE × WINDOW_SIZE``
crop, the same rule can be dropped onto a board of any size — the web demo
tiles a 1024×1024 canvas with these windows and slides over them.

Channel layout (``channels = 16``):
    0 — working / output channel (seeded with the input screen)
    1 — read-only input screen (marks the input circles)
    2 — read-only selector (constant gate code across the tile; 0 = empty)
    3…15 — zero
"""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

# ── Geometry ───────────────────────────────────────────────────────────────────

WINDOW_SIZE = 32
RADIUS = 3
INPUT_POS = [(6, 11), (6, 21)]  # (x, y) of the two input circles
OUTPUT_POS = (26, 16)  # (x, y) of the output circle

NCA_CHANNELS = 16
WORK_CHANNEL = 0
INPUT_CHANNEL = 1
SELECTOR_CHANNEL = 2
POSITION_CHANNEL = 3
READ_ONLY_DIMS = [INPUT_CHANNEL, SELECTOR_CHANNEL]  # Gradients and noise are writable

# ── Gate registry ──────────────────────────────────────────────────────────────
#
# Selector values are chosen non-zero and distinct from the empty background
# (0.0) so the rule can tell "no gate here" from "compute OR".

GATES = ["AND", "OR", "XOR"]
GATE_CODES = {"AND": -1.0, "OR": 0.5, "XOR": 1.0}
CODE_TO_GATE = {v: k for k, v in GATE_CODES.items()}
EMPTY_CODE = 0.0

# Normalised pixel values (raw screens are [0, 255], mapped to [-1, 1]).
BACKGROUND = 0.0
ONE = 1.0
ZERO = -1.0


def gate_fn(gate: str, a: int, b: int) -> int:
    if gate == "AND":
        return a & b
    if gate == "OR":
        return a | b
    if gate == "XOR":
        return a ^ b
    raise ValueError(f"unknown gate {gate!r}")


# ── Screen drawing ─────────────────────────────────────────────────────────────


def _draw_bit_circles(screen, bits, positions, radius=RADIUS):
    """Draw circle markers into ``screen`` (numpy float array)."""
    for bit, (x, y) in zip(bits, positions):
        cv2.circle(screen, (int(x), int(y)), int(radius), ONE if bit else ZERO, -1)


def make_gate_tile(gate: str | None, a_bit: int, b_bit: int):
    """One 32×32 tile: input screen, selector screen, output screen, output mask.

    ``gate=None`` yields an empty tile: no input circles, selector 0, target
    output is plain background, but the output position is still masked so the
    rule is supervised to emit *nothing* there.

    Returns four ``(WINDOW_SIZE, WINDOW_SIZE)`` float32 arrays:
        input_screen    — input circles on background
        selector_screen — constant gate code across the tile
        output_screen   — output circle on background (the target)
        out_mask        — 1.0 at the output-circle pixels, 0 elsewhere
    """
    H = W = WINDOW_SIZE
    input_screen = np.full((H, W), BACKGROUND, dtype=np.float32)
    output_screen = np.full((H, W), BACKGROUND, dtype=np.float32)
    selector_screen = np.full(
        (H, W), EMPTY_CODE if gate is None else GATE_CODES[gate], dtype=np.float32
    )

    if gate is not None:
        _draw_bit_circles(input_screen, [a_bit, b_bit], INPUT_POS)
        _draw_bit_circles(output_screen, [gate_fn(gate, a_bit, b_bit)], [OUTPUT_POS])

    # The output position is always masked (gate tiles supervise the answer,
    # empty tiles supervise background) so the rule learns to stay quiet.
    out_mask = np.zeros((H, W), dtype=np.float32)
    _draw_bit_circles(out_mask, [1], [OUTPUT_POS])

    return input_screen, selector_screen, output_screen, out_mask


def make_gate_board(
    n_rows: int, n_cols: int, rng: random.Random | None = None, gate_prob: float = 0.5
):
    """Tile an ``(n_rows * 32, n_cols * 32)`` board with random gates.

    Each tile independently holds a random gate (with probability ``gate_prob``)
    or is empty. Returns ``(input_screen, selector_screen, output_screen,
    out_mask)``, each a ``(H, W)`` float32 array in the normalised range.
    """
    rng = rng or random
    H = n_rows * WINDOW_SIZE
    W = n_cols * WINDOW_SIZE
    input_screen = np.full((H, W), BACKGROUND, dtype=np.float32)
    selector_screen = np.zeros((H, W), dtype=np.float32)
    output_screen = np.full((H, W), BACKGROUND, dtype=np.float32)
    out_mask = np.zeros((H, W), dtype=np.float32)

    for r in range(n_rows):
        for c in range(n_cols):
            gate = rng.choice(GATES) if rng.random() < gate_prob else None
            a_bit = rng.randint(0, 1)
            b_bit = rng.randint(0, 1)
            y0, x0 = r * WINDOW_SIZE, c * WINDOW_SIZE
            t_in, t_sel, t_out, t_mask = make_gate_tile(gate, a_bit, b_bit)
            input_screen[y0 : y0 + WINDOW_SIZE, x0 : x0 + WINDOW_SIZE] = t_in
            selector_screen[y0 : y0 + WINDOW_SIZE, x0 : x0 + WINDOW_SIZE] = t_sel
            output_screen[y0 : y0 + WINDOW_SIZE, x0 : x0 + WINDOW_SIZE] = t_out
            out_mask[y0 : y0 + WINDOW_SIZE, x0 : x0 + WINDOW_SIZE] = t_mask

    return input_screen, selector_screen, output_screen, out_mask


def compute_gradients(grid):
    """Compute horizontal and vertical gradients using Sobel filters."""
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
    grad_x = torch.nn.functional.conv2d(grid.unsqueeze(0).unsqueeze(0), sobel_x.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
    grad_y = torch.nn.functional.conv2d(grid.unsqueeze(0).unsqueeze(0), sobel_y.unsqueeze(0).unsqueeze(0), padding=1).squeeze()
    return grad_x, grad_y


def build_state(
    input_screen, selector_screen, channels=NCA_CHANNELS, position_gradient=False
):
    """Assemble a ``(channels, H, W)`` initial state from the two screens."""
    input_screen = torch.as_tensor(input_screen, dtype=torch.float32)
    selector_screen = torch.as_tensor(selector_screen, dtype=torch.float32)
    state = torch.zeros(channels, *input_screen.shape)
    
    # Channel 0: Original input
    state[WORK_CHANNEL] = input_screen
    state[INPUT_CHANNEL] = input_screen
    state[SELECTOR_CHANNEL] = selector_screen
    
    # Channels 1-2: Neighboring gradients (Sobel)
    grad_x, grad_y = compute_gradients(input_screen)
    state[1] = grad_x
    state[2] = grad_y
    
    # Channels 3+: Noise or synthetic neighbor data
    for c in range(3, channels):
        state[c] = torch.randn_like(input_screen) * 0.1  # Adjust noise scale

    if position_gradient:
        # Gate-local board coordinates, never recomputed relative to a crop.
        h, w = input_screen.shape
        y = torch.arange(h, dtype=torch.float32).remainder(WINDOW_SIZE)
        x = torch.arange(w, dtype=torch.float32).remainder(WINDOW_SIZE)
        state[POSITION_CHANNEL] = (y[:, None] + x[None, :]) / (WINDOW_SIZE - 1) - 1
    return state


# ── Dataset ────────────────────────────────────────────────────────────────────


class WindowGateDataset(IterableDataset):
    """Yields ``(state, target, out_mask)`` for a board tiled with random gates.

    ``state`` is ``(channels, H, W)``; ``target`` and ``out_mask`` are ``(H, W)``.
    The output-circle positions are identical across samples, so ``out_mask`` is
    effectively constant, but it is returned per-sample for convenience.
    """

    def __init__(
        self,
        n_rows: int = 1,
        n_cols: int = 4,
        seed: int | None = None,
        gate_prob: float = 0.5,
    ):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.gate_prob = gate_prob
        self.H = n_rows * WINDOW_SIZE
        self.W = n_cols * WINDOW_SIZE
        self._rng = random.Random(seed)

    def get_sample(self):
        input_screen, selector_screen, output_screen, out_mask = make_gate_board(
            self.n_rows, self.n_cols, self._rng, self.gate_prob
        )
        state = build_state(input_screen, selector_screen)
        return (
            state,
            torch.as_tensor(output_screen, dtype=torch.float32),
            torch.as_tensor(out_mask, dtype=torch.float32),
        )

    def __iter__(self):
        while True:
            yield self.get_sample()

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False)
