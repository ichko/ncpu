"""Subleq NCA encoding.

The NCA is used as the *ALU* of a one-instruction (subleq) machine.  Given two
operand words A and B drawn as circles on a black grid, it must produce, at the
last rollout step, a screen identical to the input plus the correctly-filled
output column::

    R      = (B - A) mod 2**word_bits      (the new value of mem[b])
    branch = 1 if signed(R) <= 0 else 0    (the subleq "<= 0" test)

Everything else (memory, program counter, fetch/branch) lives in the Python
harness that loops the NCA once per executed instruction.

Encoding conventions
--------------------
* Screens are float32 in [-1, 1] (no uint8 detour), so the faint scaffold grid
  can carry an exact value.
* Background is neutral grey = 0.0, so both bit values stay visible against it.
* A 0-bit circle is -1.0 (dark) and a 1-bit circle is +1.0 (bright).
* On the INPUT screen only the A and B operand columns are drawn; the output
  region is left as plain background, and there is no grid.
* On the TARGET (last-frame) screen the output circles carry their correct
  +-1 values and a faint 1px lattice at ``GRID_VALUE`` is stamped everywhere,
  phase-locked so the lines pass through the circle centres, to force the
  signal to propagate across the whole grid.

Layout (defaults: 100x100, word_bits=8)
---------------------------------------
Two operand register columns on the left (A then B), and one output column on
the right holding ``word_bits`` result circles (MSB-top) plus one branch circle
at the bottom.  Circles are sized as large as fit: spacing between circles is
half a circle (``gap = r``) and the padding around the state is one full circle
(``pad = 2r``); the tallest column (the output) sets the radius.
"""

import random

import numpy as np
import torch

BACKGROUND = 0.0
BIT0_VALUE = -1.0
BIT1_VALUE = +1.0
GRID_VALUE = 0.2
# Scaffold carpet stamped on the TARGET only. A stronger, deterministic (same
# every sample) pattern gives the NCA real loss weight to reproduce -> forces
# it to build long-range propagation. Pattern is cell-based, phase-locked to
# the circle lattice.
SCAFFOLD_PATTERN = "noise"   # "checker" | "vstripes" | "diag" | "noise"
SCAFFOLD_AMP = 0.5           # carpet amplitude in [-amp, amp]
SCAFFOLD_SEED = 1            # for the deterministic "noise" pattern


def int_to_bits_msb(n, width):
    """Return ``width`` bits of ``n``, most-significant first."""
    return [(n >> (width - 1 - i)) & 1 for i in range(width)]


def bits_msb_to_int(bits):
    n = 0
    for b in bits:
        n = (n << 1) | int(b)
    return n


def compute_subleq(a_int, b_int, word_bits=8):
    """subleq ALU: returns (result, branch).

    ``result`` = (B - A) wrapped to ``word_bits`` two's-complement bits.
    ``branch`` = 1 iff the signed result is <= 0 (i.e. result == 0 or its
    sign bit is set).
    """
    mask = (1 << word_bits) - 1
    result = (b_int - a_int) & mask
    sign_bit = 1 << (word_bits - 1)
    signed = result - (1 << word_bits) if (result & sign_bit) else result
    branch = 1 if signed <= 0 else 0
    return result, branch


def _largest_radius(H, n_tallest):
    """Largest integer radius so ``n_tallest`` circles fit vertically with
    gap=r and pad=2r: n*2r + (n-1)*r + 2*(2r) = (3n + 3) * r <= H."""
    return max(1, H // (3 * n_tallest + 3))


class SubleqLayout:
    """Geometry + rendering for the subleq input/target screens."""

    def __init__(self, W=100, H=100, word_bits=8, r=None, gap=None, pad=None):
        self.W = W
        self.H = H
        self.word_bits = word_bits
        self.out_bits = word_bits + 1  # result bits + 1 branch bit

        self.r = int(r) if r is not None else _largest_radius(H, self.out_bits)
        self.gap = int(gap) if gap is not None else self.r          # half a circle
        self.pad = int(pad) if pad is not None else 2 * self.r      # one circle
        self.step = 2 * self.r + self.gap

        # All columns share ONE lattice so the scaffold grid passes through
        # every circle centre. A, B on the left; the output column is the
        # right-most lattice line whose x shares A's phase.
        x_a = self.pad + self.r
        x_b = x_a + self.step
        x_o = x_a + ((self.W - self.pad - self.r - x_a) // self.step) * self.step

        # Lattice rows, centred for the tallest (output) column. The shorter
        # A/B columns take the top `word_bits` rows so result bit i lines up
        # with A_i / B_i (the branch bit is the extra row below).
        self.y0 = (self.H - (self.out_bits - 1) * self.step) // 2

        def rows(n):
            return [self.y0 + k * self.step for k in range(n)]

        self.A_centers = [(x_a, y) for y in rows(word_bits)]
        self.B_centers = [(x_b, y) for y in rows(word_bits)]
        self.O_centers = [(x_o, y) for y in rows(self.out_bits)]

        # Grid-line phases chosen so lattice lines pass through circle centres.
        self.grid_x_phase = x_a % self.step
        self.grid_y_phase = self.y0 % self.step

    # ── rendering ────────────────────────────────────────────────────────────

    def _disc(self, screen, center, value):
        cx, cy = center
        ys, xs = np.ogrid[: self.H, : self.W]
        m = (xs - cx) ** 2 + (ys - cy) ** 2 <= self.r ** 2
        screen[m] = value

    def _draw_bits(self, screen, centers, bits):
        for center, bit in zip(centers, bits):
            self._disc(screen, center, BIT1_VALUE if bit else BIT0_VALUE)

    def input_screen(self, a_bits, b_bits):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        self._draw_bits(s, self.A_centers, a_bits)
        self._draw_bits(s, self.B_centers, b_bits)
        # output column is left as plain background on the input (no output bits)
        return s

    def _carpet(self, pattern=None, amp=None, seed=None):
        """Full-field scaffold carpet in [-amp, amp], cell-based and phase-locked
        to the circle lattice. Deterministic (identical for every sample)."""
        pattern = pattern or SCAFFOLD_PATTERN
        amp = SCAFFOLD_AMP if amp is None else amp
        seed = SCAFFOLD_SEED if seed is None else seed
        col = (np.arange(self.W) - self.grid_x_phase) // self.step
        row = (np.arange(self.H) - self.grid_y_phase) // self.step
        R = row[:, None].astype(np.float64)
        C = col[None, :].astype(np.float64)
        if pattern == "checker":                      # symmetric ±1 by parity
            v = np.where(((row[:, None] + col[None, :]) % 2) == 0, 1.0, -1.0)
        elif pattern == "vstripes":                   # vertical stripes
            v = np.where((col[None, :] % 2) == 0, 1.0, -1.0) * np.ones_like(R)
        elif pattern == "diag":                       # asymmetric 3-phase diagonals
            m = (row[:, None] + col[None, :]) % 3
            v = np.select([m == 0, m == 1], [1.0, -0.3], default=-1.0)
        else:                                         # "noise": deterministic per-cell hash, asymmetric
            h = np.sin(R * 127.1 + C * 311.7 + seed * 13.7) * 43758.5453
            v = 2.0 * (h - np.floor(h)) - 1.0
        return (amp * v).astype(np.float32)

    def target_screen(self, a_bits, b_bits, out_bits):
        s = self._carpet()                            # scaffold carpet
        self._draw_bits(s, self.A_centers, a_bits)    # inputs reproduced
        self._draw_bits(s, self.B_centers, b_bits)
        self._draw_bits(s, self.O_centers, out_bits)  # outputs filled in
        return s

    def output_bit_masks(self):
        """(out_bits, H, W) float tensor — one disc mask per output circle."""
        masks = []
        for center in self.O_centers:
            m = np.zeros((self.H, self.W), dtype=np.float32)
            self._disc(m, center, 1.0)
            masks.append(m)
        return torch.from_numpy(np.stack(masks))

    # ── batch sampling ─────────────────────────────────────────────────────────

    def screens_for(self, a_int, b_int):
        a_bits = int_to_bits_msb(a_int, self.word_bits)
        b_bits = int_to_bits_msb(b_int, self.word_bits)
        result, branch = compute_subleq(a_int, b_int, self.word_bits)
        out_bits = int_to_bits_msb(result, self.word_bits) + [branch]
        return self.input_screen(a_bits, b_bits), self.target_screen(a_bits, b_bits, out_bits)

    def sample_batch(self, batch_size, device=None, rng=random, hard_frac=0.0):
        inps, tgts = [], []
        for _ in range(batch_size):
            a_int = rng.randint(0, (1 << self.word_bits) - 1)
            b_int = rng.randint(0, (1 << self.word_bits) - 1)
            inp, tgt = self.screens_for(a_int, b_int)
            inps.append(inp)
            tgts.append(tgt)
        inp = torch.from_numpy(np.stack(inps))
        tgt = torch.from_numpy(np.stack(tgts))
        if device is not None:
            inp = inp.to(device)
            tgt = tgt.to(device)
        return inp, tgt


# weighted deltas oversampling R near 0 (the ≤0 branch decision boundary, and
# the common loop-exit case) — 0 is over-represented so zero-detect is learned.
_HARD_DELTAS = [0, 0, 0, 0, 1, -1, 2, -2, 3, -3, 4, -4, 8, -8]


class DartboardLayout:
    """subleq as a space-filling disk dissection (dartboard), single channel.

    Outer band = operand A (8 angular sectors, MSB at top, clockwise).
    Inner band = operand B at t=0  ->  result R = B - A at the last step, IN
                 PLACE (same sectors).
    Centre hub = branch bit (a global sign/zero read).
    Outside the disk = background (0). No scaffold carpet — the disk is
    space-filling, so there is no dead background to prop up.

    Bit values: 1 -> +1 (bright), 0 -> -1 (dark). The output masks are the 8
    inner sectors (R) plus the hub (branch): out_bits = word_bits + 1.
    """

    def __init__(self, W=72, H=72, word_bits=8, hub_frac=0.30, mid_frac=0.63, pad=2):
        self.W = W
        self.H = H
        self.word_bits = word_bits
        self.out_bits = word_bits + 1

        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        R = min(W, H) / 2.0 - pad
        self.R, self.cx, self.cy = R, cx, cy

        ys, xs = np.mgrid[0:H, 0:W]
        dx = xs - cx
        dy = cy - ys                              # y up-positive
        rad = np.sqrt(dx * dx + dy * dy)
        ang = np.degrees(np.arctan2(dy, dx))      # 0 = right, 90 = up
        half = 180.0 / word_bits
        theta = (90.0 - ang + half) % 360.0       # sector 0 centred at the top
        sector = (theta // (360.0 / word_bits)).astype(int) % word_bits

        r_hub, r_mid = hub_frac * R, mid_frac * R
        region = np.zeros((H, W), np.int8)        # 0 = background (outside disk)
        region[rad <= R] = 3                      # outer band  -> A
        region[rad <= r_mid] = 2                  # inner band  -> B / R
        region[rad <= r_hub] = 1                  # hub         -> branch
        self.region, self.sector = region, sector
        self._outer = [(region == 3) & (sector == i) for i in range(word_bits)]
        self._inner = [(region == 2) & (sector == i) for i in range(word_bits)]
        self._hub = region == 1

    def _fill(self, a_bits, inner_bits, hub_val):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        for i in range(self.word_bits):
            s[self._outer[i]] = BIT1_VALUE if a_bits[i] else BIT0_VALUE
            s[self._inner[i]] = BIT1_VALUE if inner_bits[i] else BIT0_VALUE
        s[self._hub] = hub_val
        return s

    def input_screen(self, a_bits, b_bits):
        return self._fill(a_bits, b_bits, BACKGROUND)          # hub blank at t=0

    def target_screen(self, a_bits, r_bits, branch):
        return self._fill(a_bits, r_bits, BIT1_VALUE if branch else BIT0_VALUE)

    def output_bit_masks(self):
        """(out_bits, H, W): 8 inner sectors (R) + the hub (branch)."""
        masks = [m.astype(np.float32) for m in self._inner]
        masks.append(self._hub.astype(np.float32))
        return torch.from_numpy(np.stack(masks))

    def screens_for(self, a_int, b_int):
        a_bits = int_to_bits_msb(a_int, self.word_bits)
        b_bits = int_to_bits_msb(b_int, self.word_bits)
        result, branch = compute_subleq(a_int, b_int, self.word_bits)
        r_bits = int_to_bits_msb(result, self.word_bits)
        return self.input_screen(a_bits, b_bits), self.target_screen(a_bits, r_bits, branch)

    def sample_batch(self, batch_size, device=None, rng=random, hard_frac=0.0):
        mask = (1 << self.word_bits) - 1
        inps, tgts = [], []
        for _ in range(batch_size):
            a_int = rng.randint(0, mask)
            if rng.random() < hard_frac:                 # near-zero result (branch boundary)
                b_int = (a_int + rng.choice(_HARD_DELTAS)) & mask
            else:
                b_int = rng.randint(0, mask)
            inp, tgt = self.screens_for(a_int, b_int)
            inps.append(inp)
            tgts.append(tgt)
        inp = torch.from_numpy(np.stack(inps))
        tgt = torch.from_numpy(np.stack(tgts))
        if device is not None:
            inp = inp.to(device)
            tgt = tgt.to(device)
        return inp, tgt


class AddresserLayout:
    """Read-addresser: N cells (8-bit columns) + a one-hot select marker per cell;
    the NCA must gather the SELECTED cell's value across to an output column.

    This isolates the one capability the unrolled (option-B) stack needs and the
    in-place dartboard never did: content-addressable transport across the band.
    Single channel; interface matches the other layouts (W, H, word_bits,
    out_bits, output_bit_masks, sample_batch, screens_for) so the trainer reuses.
    """

    def __init__(self, W=108, H=64, n_cells=6, word_bits=8,
                 r=2, step=6, ybits0=16, ymark=6, x0=12, colstep=12, xout=98):
        self.W, self.H, self.n_cells, self.word_bits = W, H, n_cells, word_bits
        self.out_bits = word_bits
        self.r, self.step = r, step
        self.ymark = ymark
        self.cell_x = [x0 + i * colstep for i in range(n_cells)]
        self.xout = xout
        self.bit_y = [ybits0 + j * step for j in range(word_bits)]

    def _disc(self, s, cx, cy, val):
        ys, xs = np.ogrid[: self.H, : self.W]
        s[(xs - cx) ** 2 + (ys - cy) ** 2 <= self.r ** 2] = val

    def _fill(self, vals, sel, out_val):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        for i in range(self.n_cells):
            for j, b in enumerate(int_to_bits_msb(vals[i], self.word_bits)):
                self._disc(s, self.cell_x[i], self.bit_y[j], BIT1_VALUE if b else BIT0_VALUE)
            self._disc(s, self.cell_x[i], self.ymark, BIT1_VALUE if i == sel else BIT0_VALUE)
        if out_val is not None:                                   # output column
            for j, b in enumerate(int_to_bits_msb(out_val, self.word_bits)):
                self._disc(s, self.xout, self.bit_y[j], BIT1_VALUE if b else BIT0_VALUE)
        return s

    def input_screen(self, vals, sel):
        return self._fill(vals, sel, None)                        # output blank

    def target_screen(self, vals, sel):
        return self._fill(vals, sel, vals[sel])                   # output = selected cell

    def output_bit_masks(self):
        masks = []
        for j in range(self.word_bits):
            m = np.zeros((self.H, self.W), dtype=np.float32)
            self._disc(m, self.xout, self.bit_y[j], 1.0)
            masks.append(m)
        return torch.from_numpy(np.stack(masks))

    def screens_for(self, vals, sel):
        return self.input_screen(vals, sel), self.target_screen(vals, sel)

    def sample_batch(self, batch_size, device=None, rng=random, hard_frac=0.0):
        mask = (1 << self.word_bits) - 1
        inps, tgts = [], []
        for _ in range(batch_size):
            vals = [rng.randint(0, mask) for _ in range(self.n_cells)]
            sel = rng.randint(0, self.n_cells - 1)
            inp, tgt = self.screens_for(vals, sel)
            inps.append(inp); tgts.append(tgt)
        inp = torch.from_numpy(np.stack(inps)); tgt = torch.from_numpy(np.stack(tgts))
        if device is not None:
            inp = inp.to(device); tgt = tgt.to(device)
        return inp, tgt


class MemTileLayout:
    """Full option-B tile: a memory band of N cells (8-bit columns) with TWO
    one-hot selectors (selA, selB). The NCA must, in place, set
    cell[selB] = (mem[selB] - mem[selA]) mod 256 and leave every other cell
    unchanged (copy-through). This is gather x2 + subtract + scatter-to-selected
    + persistence — the whole tile minus the trigger.

    Single channel; interface matches the trainer (W, H, word_bits, out_bits,
    output_bit_masks, sample_batch, screens_for). out_bits = N*word_bits (the
    whole band is supervised so copy-through is learned).
    """

    def __init__(self, W=100, H=70, n_cells=6, word_bits=8,
                 r=2, step=6, ybits0=22, ymarkA=6, ymarkB=13, x0=13, colstep=14):
        self.W, self.H, self.n_cells, self.word_bits = W, H, n_cells, word_bits
        self.out_bits = n_cells * word_bits
        self.r, self.step = r, step
        self.ymarkA, self.ymarkB = ymarkA, ymarkB
        self.cell_x = [x0 + i * colstep for i in range(n_cells)]
        self.bit_y = [ybits0 + j * step for j in range(word_bits)]

    def _disc(self, s, cx, cy, val):
        ys, xs = np.ogrid[: self.H, : self.W]
        s[(xs - cx) ** 2 + (ys - cy) ** 2 <= self.r ** 2] = val

    def _fill(self, vals):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        for i in range(self.n_cells):
            for j, b in enumerate(int_to_bits_msb(vals[i], self.word_bits)):
                self._disc(s, self.cell_x[i], self.bit_y[j], BIT1_VALUE if b else BIT0_VALUE)
        return s

    def _markers(self, s, y, sel):
        for i in range(self.n_cells):
            self._disc(s, self.cell_x[i], y, BIT1_VALUE if i == sel else BIT0_VALUE)

    def input_screen(self, vals, sA, sB):
        s = self._fill(vals)
        self._markers(s, self.ymarkA, sA); self._markers(s, self.ymarkB, sB)
        return s

    def target_screen(self, vals, sA, sB):
        out = list(vals)
        out[sB] = (vals[sB] - vals[sA]) & ((1 << self.word_bits) - 1)   # cell[selB] = B - A
        s = self._fill(out)
        self._markers(s, self.ymarkA, sA); self._markers(s, self.ymarkB, sB)
        return s

    def output_bit_masks(self):                       # every band bit (loss over whole band)
        masks = []
        for i in range(self.n_cells):
            for j in range(self.word_bits):
                m = np.zeros((self.H, self.W), dtype=np.float32)
                self._disc(m, self.cell_x[i], self.bit_y[j], 1.0)
                masks.append(m)
        return torch.from_numpy(np.stack(masks))

    def cell_bit_masks(self, i):                      # one cell's 8 bits (for eval)
        masks = []
        for j in range(self.word_bits):
            m = np.zeros((self.H, self.W), dtype=np.float32)
            self._disc(m, self.cell_x[i], self.bit_y[j], 1.0)
            masks.append(m)
        return torch.from_numpy(np.stack(masks))

    def screens_for(self, vals, sA, sB):
        return self.input_screen(vals, sA, sB), self.target_screen(vals, sA, sB)

    def sample_batch(self, batch_size, device=None, rng=random, hard_frac=0.0):
        mask = (1 << self.word_bits) - 1
        inps, tgts = [], []
        for _ in range(batch_size):
            vals = [rng.randint(0, mask) for _ in range(self.n_cells)]
            sA = rng.randint(0, self.n_cells - 1)
            sB = rng.randint(0, self.n_cells - 1)
            inp, tgt = self.screens_for(vals, sA, sB)
            inps.append(inp); tgts.append(tgt)
        inp = torch.from_numpy(np.stack(inps)); tgt = torch.from_numpy(np.stack(tgts))
        if device is not None:
            inp = inp.to(device); tgt = tgt.to(device)
        return inp, tgt


class Gather2Layout:
    """Localizer for the decomposed tile: N cells (8-bit columns) + TWO one-hot
    selectors (selA, selB); the NCA must gather BOTH selected cells and produce
    R = mem[selB] - mem[selA] (+ the <=0 branch) at a FIXED output column.

    Isolates gather-2 + align + subtract (no scatter, no copy-through). If this
    trains, the decomposition works and scatter/copy/enable are known-easy adds.
    Interface matches the trainer.
    """

    def __init__(self, W=110, H=76, n_cells=6, word_bits=8,
                 r=2, step=6, ybits0=18, ymarkA=5, ymarkB=11, x0=13, colstep=14, xout=100):
        self.W, self.H, self.n_cells, self.word_bits = W, H, n_cells, word_bits
        self.out_bits = word_bits + 1                         # R bits + branch
        self.r, self.step = r, step
        self.ymarkA, self.ymarkB = ymarkA, ymarkB
        self.cell_x = [x0 + i * colstep for i in range(n_cells)]
        self.xout = xout
        self.bit_y = [ybits0 + j * step for j in range(self.out_bits)]

    def _disc(self, s, cx, cy, val):
        ys, xs = np.ogrid[: self.H, : self.W]
        s[(xs - cx) ** 2 + (ys - cy) ** 2 <= self.r ** 2] = val

    def _markers(self, s, y, sel):
        for i in range(self.n_cells):
            self._disc(s, self.cell_x[i], y, BIT1_VALUE if i == sel else BIT0_VALUE)

    def _fill(self, vals, sA, sB, out_bits):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        for i in range(self.n_cells):
            for j, b in enumerate(int_to_bits_msb(vals[i], self.word_bits)):
                self._disc(s, self.cell_x[i], self.bit_y[j], BIT1_VALUE if b else BIT0_VALUE)
        self._markers(s, self.ymarkA, sA); self._markers(s, self.ymarkB, sB)
        if out_bits is not None:
            for j, b in enumerate(out_bits):
                self._disc(s, self.xout, self.bit_y[j], BIT1_VALUE if b else BIT0_VALUE)
        return s

    def input_screen(self, vals, sA, sB):
        return self._fill(vals, sA, sB, None)

    def target_screen(self, vals, sA, sB):
        R, branch = compute_subleq(vals[sA], vals[sB], self.word_bits)   # mem[selB] - mem[selA]
        return self._fill(vals, sA, sB, int_to_bits_msb(R, self.word_bits) + [branch])

    def output_bit_masks(self):
        masks = []
        for j in range(self.out_bits):
            m = np.zeros((self.H, self.W), dtype=np.float32)
            self._disc(m, self.xout, self.bit_y[j], 1.0)
            masks.append(m)
        return torch.from_numpy(np.stack(masks))

    def screens_for(self, vals, sA, sB):
        return self.input_screen(vals, sA, sB), self.target_screen(vals, sA, sB)

    def sample_batch(self, batch_size, device=None, rng=random, hard_frac=0.0):
        mask = (1 << self.word_bits) - 1
        inps, tgts = [], []
        for _ in range(batch_size):
            vals = [rng.randint(0, mask) for _ in range(self.n_cells)]
            sA = rng.randint(0, self.n_cells - 1); sB = rng.randint(0, self.n_cells - 1)
            inp, tgt = self.screens_for(vals, sA, sB)
            inps.append(inp); tgts.append(tgt)
        inp = torch.from_numpy(np.stack(inps)); tgt = torch.from_numpy(np.stack(tgts))
        if device is not None:
            inp = inp.to(device); tgt = tgt.to(device)
        return inp, tgt


class GatherDiscLayout:
    """Decomposition test WITH the disc. Band of N cells (8-bit columns) + two
    one-hot selectors; a dartboard disc below is the aligned compute site. The
    NCA must gather mem[selA] -> outer ring, mem[selB] -> inner ring, subtract
    so the inner ring becomes R = mem[selB]-mem[selA], hub = branch.

    output masks (17): [inner R (8)] + [hub branch (1)] + [outer A (8)], so the
    trainer's result=first-8 / branch=index-8 metrics are meaningful; the outer
    ring (A) confirms the gather.
    """

    def __init__(self, W=100, H=104, n_cells=6, word_bits=8,
                 r=2, step=6, ybits0=16, ymarkA=5, ymarkB=11, x0=10, colstep=14,
                 disc_cx=50, disc_cy=80, r_out=20):
        self.W, self.H, self.n_cells, self.word_bits = W, H, n_cells, word_bits
        self.out_bits = 2 * word_bits + 1
        self.r, self.step = r, step
        self.ymarkA, self.ymarkB = ymarkA, ymarkB
        self.cell_x = [x0 + i * colstep for i in range(n_cells)]
        self.bit_y = [ybits0 + j * step for j in range(word_bits)]
        # disc geometry
        ys, xs = np.mgrid[0:H, 0:W]
        dx = xs - disc_cx; dy = disc_cy - ys
        rad = np.sqrt(dx * dx + dy * dy)
        ang = np.degrees(np.arctan2(dy, dx))
        half = 180.0 / word_bits
        sector = (((90.0 - ang + half) % 360.0) // (360.0 / word_bits)).astype(int) % word_bits
        r_mid, r_hub = 0.63 * r_out, 0.30 * r_out
        self._outer = [(rad <= r_out) & (rad > r_mid) & (sector == i) for i in range(word_bits)]
        self._inner = [(rad <= r_mid) & (rad > r_hub) & (sector == i) for i in range(word_bits)]
        self._hub = rad <= r_hub

    def _disc(self, s, cx, cy, val):
        ys, xs = np.ogrid[: self.H, : self.W]
        s[(xs - cx) ** 2 + (ys - cy) ** 2 <= self.r ** 2] = val

    def _band(self, s, vals, sA, sB):
        for i in range(self.n_cells):
            for j, b in enumerate(int_to_bits_msb(vals[i], self.word_bits)):
                self._disc(s, self.cell_x[i], self.bit_y[j], BIT1_VALUE if b else BIT0_VALUE)
            self._disc(s, self.cell_x[i], self.ymarkA, BIT1_VALUE if i == sA else BIT0_VALUE)
            self._disc(s, self.cell_x[i], self.ymarkB, BIT1_VALUE if i == sB else BIT0_VALUE)

    def input_screen(self, vals, sA, sB):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        self._band(s, vals, sA, sB)
        return s                                          # disc blank

    def target_screen(self, vals, sA, sB):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        self._band(s, vals, sA, sB)
        R, branch = compute_subleq(vals[sA], vals[sB], self.word_bits)
        for i, b in enumerate(int_to_bits_msb(vals[sA], self.word_bits)):
            s[self._outer[i]] = BIT1_VALUE if b else BIT0_VALUE          # A -> outer ring
        for i, b in enumerate(int_to_bits_msb(R, self.word_bits)):
            s[self._inner[i]] = BIT1_VALUE if b else BIT0_VALUE          # R -> inner ring
        s[self._hub] = BIT1_VALUE if branch else BIT0_VALUE             # branch -> hub
        return s

    def output_bit_masks(self):
        masks = []
        for i in range(self.word_bits):                   # inner R (0..7)
            masks.append(self._inner[i].astype(np.float32))
        masks.append(self._hub.astype(np.float32))        # hub branch (8)
        for i in range(self.word_bits):                   # outer A (9..16)
            masks.append(self._outer[i].astype(np.float32))
        return torch.from_numpy(np.stack(masks))

    def screens_for(self, vals, sA, sB):
        return self.input_screen(vals, sA, sB), self.target_screen(vals, sA, sB)

    def sample_batch(self, batch_size, device=None, rng=random, hard_frac=0.0):
        mask = (1 << self.word_bits) - 1
        inps, tgts = [], []
        for _ in range(batch_size):
            vals = [rng.randint(0, mask) for _ in range(self.n_cells)]
            sA = rng.randint(0, self.n_cells - 1); sB = rng.randint(0, self.n_cells - 1)
            inp, tgt = self.screens_for(vals, sA, sB)
            inps.append(inp); tgts.append(tgt)
        inp = torch.from_numpy(np.stack(inps)); tgt = torch.from_numpy(np.stack(tgts))
        if device is not None:
            inp = inp.to(device); tgt = tgt.to(device)
        return inp, tgt


class GatherColLayout:
    """Decomposition with a COLUMN compute site (not a disc). Band of N cells
    (8-bit columns) + two one-hot selectors. Two fixed compute columns on the
    right, at the SAME bit-rows as the band: gather mem[selA] -> A-slot and
    mem[selB] -> B-slot (straight column->column moves, like the addresser),
    then subtract the two adjacent aligned columns so B-slot becomes
    R = mem[selB]-mem[selA]; a branch cell below = (<=0).

    Everything is column-oriented and bit-aligned -> only uses capabilities
    already at ~100% (column gather + adjacent-column subtract).
    output masks (17): [B-slot R (8)] + [branch (1)] + [A-slot A (8)].
    """

    def __init__(self, W=112, H=76, n_cells=6, word_bits=8,
                 r=2, step=6, ybits0=16, ymarkA=5, ymarkB=11, x0=10, colstep=12,
                 xA=92, xB=100, ybr=68):
        self.W, self.H, self.n_cells, self.word_bits = W, H, n_cells, word_bits
        self.out_bits = 2 * word_bits + 1
        self.r, self.step = r, step
        self.ymarkA, self.ymarkB, self.xA, self.xB, self.ybr = ymarkA, ymarkB, xA, xB, ybr
        self.cell_x = [x0 + i * colstep for i in range(n_cells)]
        self.bit_y = [ybits0 + j * step for j in range(word_bits)]

    def _disc(self, s, cx, cy, val):
        ys, xs = np.ogrid[: self.H, : self.W]
        s[(xs - cx) ** 2 + (ys - cy) ** 2 <= self.r ** 2] = val

    def _band(self, s, vals, sA, sB):
        for i in range(self.n_cells):
            for j, b in enumerate(int_to_bits_msb(vals[i], self.word_bits)):
                self._disc(s, self.cell_x[i], self.bit_y[j], BIT1_VALUE if b else BIT0_VALUE)
            self._disc(s, self.cell_x[i], self.ymarkA, BIT1_VALUE if i == sA else BIT0_VALUE)
            self._disc(s, self.cell_x[i], self.ymarkB, BIT1_VALUE if i == sB else BIT0_VALUE)

    def _col(self, s, cx, bits):
        for j, b in enumerate(bits):
            self._disc(s, cx, self.bit_y[j], BIT1_VALUE if b else BIT0_VALUE)

    def input_screen(self, vals, sA, sB):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        self._band(s, vals, sA, sB)
        return s                                          # compute slots blank

    def target_screen(self, vals, sA, sB):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        self._band(s, vals, sA, sB)
        R, branch = compute_subleq(vals[sA], vals[sB], self.word_bits)
        self._col(s, self.xA, int_to_bits_msb(vals[sA], self.word_bits))    # A-slot
        self._col(s, self.xB, int_to_bits_msb(R, self.word_bits))           # B-slot -> R
        self._disc(s, (self.xA + self.xB) // 2, self.ybr, BIT1_VALUE if branch else BIT0_VALUE)
        return s

    def output_bit_masks(self):
        masks = []
        for j in range(self.word_bits):                   # R (B-slot) 0..7
            m = np.zeros((self.H, self.W), dtype=np.float32); self._disc(m, self.xB, self.bit_y[j], 1.0); masks.append(m)
        mb = np.zeros((self.H, self.W), dtype=np.float32); self._disc(mb, (self.xA + self.xB) // 2, self.ybr, 1.0); masks.append(mb)  # branch
        for j in range(self.word_bits):                   # A (A-slot) 9..16
            m = np.zeros((self.H, self.W), dtype=np.float32); self._disc(m, self.xA, self.bit_y[j], 1.0); masks.append(m)
        return torch.from_numpy(np.stack(masks))

    def screens_for(self, vals, sA, sB):
        return self.input_screen(vals, sA, sB), self.target_screen(vals, sA, sB)

    def sample_batch(self, batch_size, device=None, rng=random, hard_frac=0.0):
        mask = (1 << self.word_bits) - 1
        inps, tgts = [], []
        for _ in range(batch_size):
            vals = [rng.randint(0, mask) for _ in range(self.n_cells)]
            sA = rng.randint(0, self.n_cells - 1); sB = rng.randint(0, self.n_cells - 1)
            if rng.random() < hard_frac:                       # near-zero R (branch boundary)
                vals[sB] = (vals[sA] + rng.choice([0, 0, 1, -1, 2, -2, 3, -3])) & mask
            inp, tgt = self.screens_for(vals, sA, sB)
            inps.append(inp); tgts.append(tgt)
        inp = torch.from_numpy(np.stack(inps)); tgt = torch.from_numpy(np.stack(tgts))
        if device is not None:
            inp = inp.to(device); tgt = tgt.to(device)
        return inp, tgt


class GatherDartLayout:
    """Gather-into-dartboard-core tile: reuses the two proven capabilities and
    nothing that has failed.

    * SOURCE: a band of N column-cells (8-bit, MSB-top) + two one-hot selectors,
      exactly like the addresser / GatherColLayout. Gather is column->column
      (mem[selA] -> A-slot column, mem[selB] -> B-slot column), the transport
      that already trains to ~100%.
    * COMPUTE CORE: a small dartboard just right of the operand columns. The NCA
      subtracts the two aligned operand columns and renders the result into the
      core: R = mem[selB]-mem[selA] as an 8-sector RING (MSB top, clockwise) with
      the branch bit (<=0) at the central HUB. The ring+hub geometry is what let
      the standalone dartboard reach ~99% on the branch: the hub is adjacent to
      all 8 result sectors, so `<=0` is a *local* read, unlike a flat cell tacked
      below a column (which topped out ~63%).
    * The only new demand is a FIXED render of the computed result into the ring
      (always the same geometry, no content addressing) -- distinct from the
      variable column->ring GATHER that failed at 0.6%.

    output masks (3*word_bits+1 = 25):
        [ring R 0..7] + [hub branch 8] + [A-slot col 9..16] + [B-slot col 17..24]
    """

    def __init__(self, W=132, H=80, n_cells=6, word_bits=8,
                 r=2, step=6, ybits0=18, ymarkA=6, ymarkB=12, x0=8, colstep=10,
                 xA=74, xB=84, ccx=110, ccy=39, Rc=18.0, hub_frac=0.35):
        self.W, self.H, self.n_cells, self.word_bits = W, H, n_cells, word_bits
        self.out_bits = 3 * word_bits + 1
        self.r, self.step = r, step
        self.ymarkA, self.ymarkB, self.xA, self.xB = ymarkA, ymarkB, xA, xB
        self.cell_x = [x0 + i * colstep for i in range(n_cells)]
        self.bit_y = [ybits0 + j * step for j in range(word_bits)]

        ys, xs = np.mgrid[0:H, 0:W]
        dx = xs - ccx
        dy = ccy - ys                              # y up-positive
        rad = np.sqrt(dx * dx + dy * dy)
        ang = np.degrees(np.arctan2(dy, dx))
        half = 180.0 / word_bits
        theta = (90.0 - ang + half) % 360.0        # sector 0 centred at the top
        sector = (theta // (360.0 / word_bits)).astype(int) % word_bits
        r_hub = hub_frac * Rc
        self._ring = [(rad <= Rc) & (rad > r_hub) & (sector == j) for j in range(word_bits)]
        self._hub = rad <= r_hub

    def _disc(self, s, cx, cy, val):
        ys, xs = np.ogrid[: self.H, : self.W]
        s[(xs - cx) ** 2 + (ys - cy) ** 2 <= self.r ** 2] = val

    def _band(self, s, vals, sA, sB):
        for i in range(self.n_cells):
            for j, b in enumerate(int_to_bits_msb(vals[i], self.word_bits)):
                self._disc(s, self.cell_x[i], self.bit_y[j], BIT1_VALUE if b else BIT0_VALUE)
            self._disc(s, self.cell_x[i], self.ymarkA, BIT1_VALUE if i == sA else BIT0_VALUE)
            self._disc(s, self.cell_x[i], self.ymarkB, BIT1_VALUE if i == sB else BIT0_VALUE)

    def _col(self, s, cx, bits):
        for j, b in enumerate(bits):
            self._disc(s, cx, self.bit_y[j], BIT1_VALUE if b else BIT0_VALUE)

    def _core(self, s, r_bits, branch):
        for j in range(self.word_bits):
            s[self._ring[j]] = BIT1_VALUE if r_bits[j] else BIT0_VALUE
        s[self._hub] = BIT1_VALUE if branch else BIT0_VALUE

    def input_screen(self, vals, sA, sB):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        self._band(s, vals, sA, sB)
        return s                                          # slots + core blank

    def target_screen(self, vals, sA, sB):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        self._band(s, vals, sA, sB)
        R, branch = compute_subleq(vals[sA], vals[sB], self.word_bits)
        self._col(s, self.xA, int_to_bits_msb(vals[sA], self.word_bits))   # A-slot = A
        self._col(s, self.xB, int_to_bits_msb(vals[sB], self.word_bits))   # B-slot = B
        self._core(s, int_to_bits_msb(R, self.word_bits), branch)          # ring R + hub branch
        return s

    def output_bit_masks(self):
        masks = []
        for j in range(self.word_bits):                   # ring R 0..7
            masks.append(self._ring[j].astype(np.float32))
        masks.append(self._hub.astype(np.float32))        # hub branch 8
        for j in range(self.word_bits):                   # A-slot col 9..16
            m = np.zeros((self.H, self.W), dtype=np.float32); self._disc(m, self.xA, self.bit_y[j], 1.0); masks.append(m)
        for j in range(self.word_bits):                   # B-slot col 17..24
            m = np.zeros((self.H, self.W), dtype=np.float32); self._disc(m, self.xB, self.bit_y[j], 1.0); masks.append(m)
        return torch.from_numpy(np.stack(masks))

    def screens_for(self, vals, sA, sB):
        return self.input_screen(vals, sA, sB), self.target_screen(vals, sA, sB)

    def sample_batch(self, batch_size, device=None, rng=random, hard_frac=0.0):
        mask = (1 << self.word_bits) - 1
        inps, tgts = [], []
        for _ in range(batch_size):
            vals = [rng.randint(0, mask) for _ in range(self.n_cells)]
            sA = rng.randint(0, self.n_cells - 1); sB = rng.randint(0, self.n_cells - 1)
            if rng.random() < hard_frac:                       # near-zero R (branch boundary)
                vals[sB] = (vals[sA] + rng.choice([0, 0, 1, -1, 2, -2, 3, -3])) & mask
            inp, tgt = self.screens_for(vals, sA, sB)
            inps.append(inp); tgts.append(tgt)
        inp = torch.from_numpy(np.stack(inps)); tgt = torch.from_numpy(np.stack(tgts))
        if device is not None:
            inp = inp.to(device); tgt = tgt.to(device)
        return inp, tgt
