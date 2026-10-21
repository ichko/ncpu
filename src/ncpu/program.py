r"""Painted straight-line programs for an NCA that executes compiled code.

The grid is a program listing that runs left to right::

    [ data 0 ]   [ opcode 0 ]   [ data 1 ]   [ opcode 1 ]   [ data 2 ]
     all regs      instr 0       all regs      instr 1       all regs
                                = output of 0
                                = input to 1

* A **data column** is one snapshot of the machine state: ``n_regs`` registers
  of ``word_bits`` bits, painted as ``n_regs * word_bits`` blobs stacked
  vertically.  Register ``r`` owns the rows starting at ``r*word_bits``, MSB at
  the top, so the BOTTOM blob of a register is its LSB.
* An **opcode column** holds one instruction as ``len(OPS) + 3*n_regs`` blobs:
  a one-hot over the operation, then one-hot ``src_a``, ``src_b`` and ``dst``
  over the registers.  It is shorter than a data column and is simply centred;
  nothing has to line up with the data rows.
* The opcode column between data column ``i`` and ``i+1`` performs exactly that
  instruction -- read ``src_a`` and ``src_b`` from column ``i``, apply the op,
  write ``dst`` in column ``i+1``, and copy every other register through
  unchanged.

Values: background 0.0, bit 0 = -1.0 (dark), bit 1 = +1.0 (bright); a data blob
that is not yet resolved is 0.0.  Every opcode column is painted from the start;
only data column 0 -- the program input -- is filled.  Everything downstream
must be produced by the wavefront.
"""

import random
from dataclasses import dataclass

import numpy as np

BACKGROUND = 0.0
UNKNOWN = 0.0
BIT0 = -1.0
BIT1 = +1.0

OPS = ("ADD", "SUB", "AND", "XOR", "SHIFT", "CMP_LT", "SELECT", "COPY")
OP_INDEX = {name: i for i, name in enumerate(OPS)}

@dataclass(frozen=True)
class Instr:
    """One instruction: ``reg[dst] = op(reg[a], reg[b])``."""

    op: str
    a: int
    b: int
    dst: int

    def __str__(self):
        return f"{self.op:<7} r{self.a} r{self.b} -> r{self.dst}"


def apply_op(op, a, b, dst_old, word_bits):
    """Reference semantics for one instruction, on ``word_bits``-bit words."""
    mask = (1 << word_bits) - 1
    if op == "ADD":
        return (a + b) & mask
    if op == "SUB":
        return (a - b) & mask
    if op == "AND":
        return a & b
    if op == "XOR":
        return a ^ b
    if op == "SHIFT":                      # logical shift left by one
        return (a << 1) & mask
    if op == "CMP_LT":
        return 1 if a < b else 0
    if op == "SELECT":                     # predication: cond in a
        return b if a != 0 else dst_old
    if op == "COPY":
        return a
    if op == "NOP":                        # no selectors painted at all: pure copy-through
        return dst_old
    if op == "SUBLEQ":                     # conditional write: keep A-B only if it is <= 0
        r = (a - b) & mask
        signed = r - (1 << word_bits) if r & (1 << (word_bits - 1)) else r
        return r if signed <= 0 else dst_old
    raise ValueError(f"unknown op {op!r}")


def run_program(regs, program, word_bits):
    """Execute ``program`` from initial ``regs``; return the full column trace.

    Returns a list of ``len(program) + 1`` register tuples -- one per data
    column, so ``trace[i]`` is exactly what data column ``i`` must show.
    """
    trace = [tuple(regs)]
    cur = list(regs)
    for ins in program:
        nxt = list(cur)
        nxt[ins.dst] = apply_op(ins.op, cur[ins.a], cur[ins.b], cur[ins.dst], word_bits)
        trace.append(tuple(nxt))
        cur = nxt
    return trace


# COPY dominates real compiled programs, so it is oversampled during training.
def sample_program(n_instr, n_regs, ops=OPS, copy_frac=0.35, rng=random,
                   addressing="free", nop_frac=0.0):
    """Sample a random program.

    ``addressing`` sets how much routing the rule has to do -- the ladder that
    isolates the hard part:

    ``identity``   ``src_a == src_b == dst``.  No routing at all: the gap just
                   carries the state one column to the right.  Pure propagation
                   and persistence, the floor of the task.
    ``fixed-dst``  ``dst`` is always register 0; sources vary.  Gather only.
    ``fixed-src``  sources are registers 0 and 1; ``dst`` varies.  Scatter only.
    ``free``       everything varies.  Gather and scatter, the real task.
    """
    prog = []
    for _ in range(n_instr):
        if nop_frac and rng.random() < nop_frac:
            prog.append(Instr("NOP", 0, 0, 0))              # no selectors painted: copy-through
            continue
        op = "COPY" if (rng.random() < copy_frac and "COPY" in ops) else rng.choice(list(ops))
        if addressing == "identity":
            r = rng.randrange(n_regs)
            a = b = dst = r
        elif addressing == "fixed-dst":
            a, b, dst = rng.randrange(n_regs), rng.randrange(n_regs), 0
        elif addressing == "fixed-src":
            a, b, dst = 0, 1, rng.randrange(n_regs)
        else:
            a, b, dst = (rng.randrange(n_regs) for _ in range(3))
        if op == "NOP":
            a = b = dst = 0                                   # operands are meaningless
        prog.append(Instr(op, a, b, dst))
    return prog


def sample_regs(n_regs, word_bits, rng=random):
    hi = (1 << word_bits) - 1
    return tuple(rng.randint(0, hi) for _ in range(n_regs))


def int_to_bits_msb(n, width):
    return [(n >> (width - 1 - i)) & 1 for i in range(width)]


def bits_msb_to_int(bits):
    n = 0
    for b in bits:
        n = (n << 1) | int(b)
    return n


class ProgramLayout:
    """Geometry, compiler (paint) and decoder (read back) for painted programs.

    Columns alternate ``data, opcode, data, opcode, ..., data``, so ``n_instr``
    instructions give ``2*n_instr + 1`` columns.
    """

    def __init__(self, n_instr=1, n_regs=4, word_bits=4, r=2, pitch=6,
                 col_pitch=24, pad=6, unknown=BIT0, scaffold=True,
                 scaffold_amp=0.5, scaffold_seed=1, scaffold_cell=None,
                 op_amp=1.0):
        """``unknown`` is the value painted in unresolved data blobs.

        It defaults to ``BIT0`` (-1.0), NOT 0.0.  With unknown at 0 -- the
        midpoint of the two bit values -- MSE makes abstaining cost 1 while
        being wrong costs 4, so "output nothing" is a local optimum the rule
        never leaves (measured: every probe arm pinned at that floor).  Set
        ``unknown=0.0`` to get the third state back.

        ``scaffold`` paints a deterministic carpet on the TARGET background, as
        in ``subleq.py``: without it the background target equals the input and
        the whole field pulls towards "stay still".
        """
        self.n_instr = n_instr
        self.n_regs = n_regs
        self.word_bits = word_bits
        self.r = r
        self.pitch = pitch
        self.col_pitch = col_pitch
        self.pad = pad
        self.unknown = unknown
        self.scaffold = scaffold
        self.scaffold_amp = scaffold_amp
        self.scaffold_seed = scaffold_seed
        # carpet cell size; None keys the column axis to col_pitch, which makes
        # coarse low-frequency bands. subleq.py phase-locks to the blob lattice.
        self.scaffold_cell = scaffold_cell
        # amplitude of opcode blobs. At 1.0 they are indistinguishable from data
        # blobs locally -- same pitch, same +-1 -- so with no coordinate channels
        # a cell cannot tell program text from machine state. Below 1.0 it can.
        self.op_amp = op_amp

        self.data_rows = n_regs * word_bits
        self.op_rows = len(OPS) + 3 * n_regs

        self.n_cols = 2 * n_instr + 1
        self.data_col = [2 * i for i in range(n_instr + 1)]
        self.op_col = [2 * i + 1 for i in range(n_instr)]
        self.col_kind = {c: "data" for c in self.data_col}
        self.col_kind.update({c: "op" for c in self.op_col})

        self.col_x = [pad + c * col_pitch for c in range(self.n_cols)]
        self.tall_rows = max(self.data_rows, self.op_rows)
        self.H = 2 * pad + (self.tall_rows - 1) * pitch + 1
        self.W = self.col_x[-1] + pad + 1

        # each column has its own row lattice, vertically centred
        def centred(n):
            y0 = (self.H - (n - 1) * pitch) // 2
            return [y0 + k * pitch for k in range(n)]

        self.data_y = centred(self.data_rows)
        self.op_y = centred(self.op_rows)
        self.col_y = [self.data_y if self.col_kind[c] == "data" else self.op_y
                      for c in range(self.n_cols)]

        # precomputed disc index arrays, one per (column, row) blob site
        ys, xs = np.ogrid[: self.H, : self.W]
        self._disc_idx = {}
        for c, cx in enumerate(self.col_x):
            for j, cy in enumerate(self.col_y[c]):
                m = (xs - cx) ** 2 + (ys - cy) ** 2 <= r * r
                self._disc_idx[(c, j)] = np.nonzero(m)

    def min_rollout(self, px_per_step=1):
        """Lower bound on the CA steps ONE gap needs, from signal travel alone.

        Worst case is ``src_a`` in the register furthest from ``dst`` plus the
        hop to the next data column.  Does not include the carry chain or blob
        settling -- it is a floor, not a budget.
        """
        vspan = (self.n_regs - 1) * self.word_bits * self.pitch
        hspan = self.col_x[self.data_col[1]] - self.col_x[self.data_col[0]] \
            if self.n_instr else 0
        return int(np.ceil((vspan + hspan) / px_per_step))

    # ── geometry helpers ────────────────────────────────────────────────────

    def row_of(self, reg, bit):
        """Data-column row for ``bit`` (0 = MSB) of register ``reg``."""
        return reg * self.word_bits + bit

    def op_row(self, field, k):
        """Opcode-column row of entry ``k`` of ``field``.

        Fields, top to bottom: ``op`` (one-hot over OPS), then ``sel_a``,
        ``sel_b``, ``sel_dst`` (one-hot over registers).
        """
        base = {"op": 0, "sel_a": len(OPS), "sel_b": len(OPS) + self.n_regs,
                "sel_dst": len(OPS) + 2 * self.n_regs}[field]
        return base + k

    def _stamp(self, screen, col, row, value):
        yy, xx = self._disc_idx[(col, row)]
        screen[yy, xx] = value

    def _stamp_bit(self, screen, col, row, bit):
        self._stamp(screen, col, row, BIT1 if bit else BIT0)

    # ── compiler: program -> painted grid ───────────────────────────────────

    def carpet(self):
        """Deterministic background pattern, phase-locked to the blob lattice."""
        cw = self.scaffold_cell or self.col_pitch
        ch = self.scaffold_cell or self.pitch
        col = (np.arange(self.W) - self.pad) // cw
        row = (np.arange(self.H) - self.pad) // ch
        h = np.sin(row[:, None] * 127.1 + col[None, :] * 311.7
                   + self.scaffold_seed * 13.7) * 43758.5453
        return (self.scaffold_amp * (2.0 * (h - np.floor(h)) - 1.0)).astype(np.float32)

    def paint(self, program, regs, resolved=0, scaffold=False):
        """Paint the grid for ``program`` started from ``regs``.

        Opcode columns are always fully painted.  Data columns ``0..resolved``
        carry their true values; later ones are left UNKNOWN.  ``resolved=0`` is
        the input screen, ``resolved=n_instr`` the final target.
        """
        assert len(program) == self.n_instr, f"{len(program)} instrs, layout wants {self.n_instr}"
        s = (self.carpet() if (scaffold and self.scaffold)
             else np.full((self.H, self.W), BACKGROUND, dtype=np.float32))
        trace = run_program(regs, program, self.word_bits)

        for i, ins in enumerate(program):
            col = self.op_col[i]
            for k, name in enumerate(OPS):
                self._stamp(s, col, self.op_row("op", k),
                            self.op_amp if name == ins.op else -self.op_amp)
            for field, sel in (("sel_a", ins.a), ("sel_b", ins.b), ("sel_dst", ins.dst)):
                for k in range(self.n_regs):
                    self._stamp(s, col, self.op_row(field, k),
                                self.op_amp if k == sel else -self.op_amp)

        for i in range(self.n_instr + 1):
            col = self.data_col[i]
            if i > resolved:
                for row in range(self.data_rows):
                    self._stamp(s, col, row, self.unknown)
                continue
            for reg in range(self.n_regs):
                for bit, b in enumerate(int_to_bits_msb(trace[i][reg], self.word_bits)):
                    self._stamp_bit(s, col, self.row_of(reg, bit), b)
        return s

    def input_screen(self, program, regs):
        return self.paint(program, regs, resolved=0)

    def stage_targets(self, program, regs):
        """One target per instruction: ``targets[i]`` has data columns ``0..i+1``
        filled.  Supervising it at step ``(i+1)*K`` asks for column ``i+1`` to
        have appeared AND every earlier column to still be holding its value."""
        return [self.paint(program, regs, resolved=i + 1, scaffold=True)
                for i in range(self.n_instr)]

    # ── masks (loss weighting) ──────────────────────────────────────────────

    def _mask_over(self, sites):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for col, row in sites:
            yy, xx = self._disc_idx[(col, row)]
            m[yy, xx] = 1.0
        return m

    def slot_mask(self):
        """1.0 inside every blob disc (data and opcode alike)."""
        return self._mask_over([(c, j) for c in range(self.n_cols)
                                for j in range(len(self.col_y[c]))])

    def program_mask(self):
        """1.0 inside the opcode blobs -- the program text, which must persist."""
        return self._mask_over([(c, j) for c in self.op_col for j in range(self.op_rows)])

    def data_mask(self, columns=None):
        """1.0 inside the data blobs of the given data columns (default: all)."""
        cols = range(self.n_instr + 1) if columns is None else columns
        return self._mask_over([(self.data_col[i], j)
                                for i in cols for j in range(self.data_rows)])

    # ── decoder: painted grid -> register values ────────────────────────────

    def _blob_mean(self, screen, col, row):
        yy, xx = self._disc_idx[(col, row)]
        return float(np.asarray(screen)[yy, xx].mean())

    def decode_column(self, screen, i, unknown_tol=0.25):
        """Read data column ``i`` back.  Returns ``(regs, conf)``; ``regs[r]`` is
        None if any of its bits reads unresolved (|mean| < ``unknown_tol``)."""
        col = self.data_col[i]
        regs, conf = [], 1.0
        for reg in range(self.n_regs):
            bits, ok = [], True
            for bit in range(self.word_bits):
                v = self._blob_mean(screen, col, self.row_of(reg, bit))
                conf = min(conf, abs(v))
                if abs(v) < unknown_tol:
                    ok = False
                bits.append(1 if v > 0 else 0)
            regs.append(bits_msb_to_int(bits) if ok else None)
        return tuple(regs), conf

    def decode(self, screen, unknown_tol=0.25):
        """Read every data column back; a list of register tuples."""
        return [self.decode_column(screen, i, unknown_tol)[0]
                for i in range(self.n_instr + 1)]

    def decode_bits(self, screen, i):
        """Raw bit array (n_regs, word_bits) for data column ``i``, MSB first."""
        col = self.data_col[i]
        out = np.zeros((self.n_regs, self.word_bits), dtype=np.int64)
        for reg in range(self.n_regs):
            for bit in range(self.word_bits):
                out[reg, bit] = 1 if self._blob_mean(screen, col, self.row_of(reg, bit)) > 0 else 0
        return out

    def decode_instr(self, screen, i):
        """Read opcode column ``i`` back into an ``Instr`` (round-trip check)."""
        col = self.op_col[i]
        op = max(range(len(OPS)), key=lambda k: self._blob_mean(screen, col, self.op_row("op", k)))
        sel = {}
        for field in ("sel_a", "sel_b", "sel_dst"):
            sel[field] = max(range(self.n_regs),
                             key=lambda k: self._blob_mean(screen, col, self.op_row(field, k)))
        return Instr(OPS[op], sel["sel_a"], sel["sel_b"], sel["sel_dst"])


# ── held-out programs (never trained on) ────────────────────────────────────


def fib_program(n_instr, n_regs=4):
    """Fibonacci, one instruction per term.

    Alternating the destination turns the usual three-instruction shuffle into
    a single ADD::

        r0=1, r1=1
        ADD r0 r1 -> r0    # 2
        ADD r0 r1 -> r1    # 3
        ADD r0 r1 -> r0    # 5   ...

    After ``k`` instructions the newest value is ``fib(k+2)``.  Registers 2 and
    up stay 0 throughout and must be copied through untouched, so the program
    exercises copy-through as well as ADD.  Returns ``(program, init_regs)``.
    """
    regs = tuple([1, 1] + [0] * (n_regs - 2))
    prog = [Instr("ADD", 0, 1, i % 2) for i in range(n_instr)]
    return prog, regs


def fib_reference(n_instr):
    """The true Fibonacci values the program above should produce, in order."""
    a, b, out = 1, 1, []
    for _ in range(n_instr):
        a, b = b, a + b
        out.append(b)
    return out


_LAYOUT_CACHE = {}


def layout_for(n_instr, **kw):
    """Cached ``ProgramLayout`` for a given program length.

    Program length varies during training, and the grid width varies with it.
    The NCA is fully convolutional so one rule spans every width; batches are
    grouped by length so the tensors line up.
    """
    key = (n_instr,) + tuple(sorted(kw.items()))
    if key not in _LAYOUT_CACHE:
        _LAYOUT_CACHE[key] = ProgramLayout(n_instr=n_instr, **kw)
    return _LAYOUT_CACHE[key]


class TileLayoutV2:
    """Tile layout v2: registers side by side, selectors aligned to their column.

    One tile is ``[input grid] [op] [output grid]`` and the output grid of tile
    ``i`` IS the input grid of tile ``i+1``::

            A  o . . .              A  . o . .
            B  . o . .              B  . . . o
               ┌────────┐   ┌──┐    ┌────────┐   ┌──┐    ┌────────┐
        bits   │ r0..r3 │   │op│    │ r0..r3 │   │op│    │ r0..r3 │
               └────────┘   └──┘    └────────┘   └──┘    └────────┘
                                 dst  . . o .         dst  . . . o

    * A **register grid** is ``n_regs`` columns x ``word_bits`` rows, one
      column per register, MSB on the top row.
    * **src A / src B** are rows of ``n_regs`` blobs ABOVE the grid they read
      from, A on top so their order is geometric; the filled blob sits directly
      above the register column it names.
    * **dst** is a row of ``n_regs`` blobs BELOW the grid that gets written,
      filled blob directly under the destination column.
    * **op** is a one-hot block of ``len(OPS)`` blobs between the grids, laid
      out in columns of ``word_bits`` so its rows line up with the bit rows.

    Channel 0 carries every value at +-1 (0 = unresolved).  Four extra FIXED
    channels say what kind of site each blob is (``type_channels``): data,
    selector-in, op, selector-out.  So a cell reads "what am I" from its own
    state instead of decoding an amplitude, and ``is_data=1, value=0`` is
    unambiguously an unresolved bit rather than background.
    """

    TYPES = ("data", "sel_in", "op", "sel_out")

    def __init__(self, n_instr=1, n_regs=4, word_bits=4, r=2, pitch=6, gap=14,
                 pad=8, vgap=4, unknown=0.0, scaffold=True, scaffold_amp=0.5,
                 scaffold_seed=1, scaffold_cell=6):
        self.n_instr, self.n_regs, self.word_bits = n_instr, n_regs, word_bits
        self.r, self.pitch, self.gap, self.pad, self.vgap = r, pitch, gap, pad, vgap
        self.unknown = unknown
        self.scaffold, self.scaffold_amp = scaffold, scaffold_amp
        self.scaffold_seed, self.scaffold_cell = scaffold_seed, scaffold_cell

        P = pitch
        self.op_cols = -(-len(OPS) // word_bits)                # ceil
        grid_w = (n_regs - 1) * P
        op_w = (self.op_cols - 1) * P
        self.tile_w = grid_w + gap + op_w + gap + P              # grid start -> next grid start

        # x of every register column of every grid, and of every op block
        self.grid_x = [[pad + i * self.tile_w + c * P for c in range(n_regs)]
                       for i in range(n_instr + 1)]
        self.op_x = [[self.grid_x[i][-1] + gap + c * P for c in range(self.op_cols)]
                     for i in range(n_instr)]

        self.y_selA = pad
        self.y_selB = pad + P
        self.y_bits = [self.y_selB + P + vgap + b * P for b in range(word_bits)]
        self.y_dst = self.y_bits[-1] + P + vgap
        self.H = self.y_dst + pad + 1
        self.W = self.grid_x[-1][-1] + pad + 1

        # data columns kept for interface parity with ProgramLayout
        self.data_col = list(range(n_instr + 1))
        self.op_col = list(range(n_instr))

        ys, xs = np.ogrid[: self.H, : self.W]
        self._yx = (ys, xs)

    # ── sites ───────────────────────────────────────────────────────────────

    def _disc(self, x, y):
        ys, xs = self._yx
        return (xs - x) ** 2 + (ys - y) ** 2 <= self.r * self.r

    def _stamp(self, s, x, y, v):
        s[self._disc(x, y)] = v

    def sites(self):
        """Yield ``(kind, x, y, meta)`` for every blob site on the grid."""
        for i in range(self.n_instr + 1):
            for c in range(self.n_regs):
                for b in range(self.word_bits):
                    yield "data", self.grid_x[i][c], self.y_bits[b], (i, c, b)
        for i in range(self.n_instr):
            for c in range(self.n_regs):
                yield "sel_in", self.grid_x[i][c], self.y_selA, (i, "a", c)
                yield "sel_in", self.grid_x[i][c], self.y_selB, (i, "b", c)
                yield "sel_out", self.grid_x[i + 1][c], self.y_dst, (i, "dst", c)
            for k in range(len(OPS)):
                oc, orow = divmod(k, self.word_bits)
                yield "op", self.op_x[i][oc], self.y_bits[orow], (i, "op", k)

    def type_channels(self):
        """(4, H, W) one-hot: which kind of blob each pixel belongs to.  Fixed
        for the whole rollout -- program structure, not machine state."""
        t = np.zeros((len(self.TYPES), self.H, self.W), dtype=np.float32)
        for kind, x, y, _ in self.sites():
            t[self.TYPES.index(kind)][self._disc(x, y)] = 1.0
        return t

    # ── compiler ────────────────────────────────────────────────────────────

    def carpet(self):
        cell = self.scaffold_cell or self.pitch
        col = (np.arange(self.W) - self.pad) // cell
        row = (np.arange(self.H) - self.pad) // cell
        h = np.sin(row[:, None] * 127.1 + col[None, :] * 311.7 + self.scaffold_seed * 13.7) * 43758.5453
        return (self.scaffold_amp * (2.0 * (h - np.floor(h)) - 1.0)).astype(np.float32)

    def paint(self, program, regs, resolved=0, scaffold=False):
        assert len(program) == self.n_instr
        s = (self.carpet() if (scaffold and self.scaffold)
             else np.full((self.H, self.W), BACKGROUND, dtype=np.float32))
        trace = run_program(regs, program, self.word_bits)
        for kind, x, y, meta in self.sites():
            if kind == "data":
                i, c, b = meta
                if i > resolved:
                    v = self.unknown
                else:
                    v = BIT1 if (trace[i][c] >> (self.word_bits - 1 - b)) & 1 else BIT0
            elif kind == "sel_in":
                i, which, c = meta
                sel = program[i].a if which == "a" else program[i].b
                v = BIT1 if c == sel else BIT0
            elif kind == "sel_out":
                i, _, c = meta
                v = BIT1 if c == program[i].dst else BIT0
            else:
                i, _, k = meta
                v = BIT1 if OPS[k] == program[i].op else BIT0
            self._stamp(s, x, y, v)
        return s

    def input_screen(self, program, regs):
        return self.paint(program, regs, resolved=0)

    def stage_targets(self, program, regs):
        return [self.paint(program, regs, resolved=i + 1, scaffold=True)
                for i in range(self.n_instr)]

    # ── masks ───────────────────────────────────────────────────────────────

    def _mask(self, pred):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for kind, x, y, meta in self.sites():
            if pred(kind, meta):
                m[self._disc(x, y)] = 1.0
        return m

    def slot_mask(self):
        return self._mask(lambda k, m: True)

    def program_mask(self):
        return self._mask(lambda k, m: k != "data")

    def data_mask(self, columns=None):
        cols = set(range(self.n_instr + 1) if columns is None else columns)
        return self._mask(lambda k, m: k == "data" and m[0] in cols)

    def min_rollout(self, px_per_step=1):
        """Travel floor for one tile: across to the next grid, plus the
        vertical reach from a selector row to the far bit row."""
        h = self.grid_x[1][0] - self.grid_x[0][0] if self.n_instr else 0
        v = self.y_dst - self.y_selA
        return int(np.ceil((h + v) / px_per_step))

    # ── decoder ─────────────────────────────────────────────────────────────

    def _mean(self, screen, x, y):
        return float(np.asarray(screen)[self._disc(x, y)].mean())

    def decode_bits(self, screen, i):
        out = np.zeros((self.n_regs, self.word_bits), dtype=np.int64)
        for c in range(self.n_regs):
            for b in range(self.word_bits):
                out[c, b] = 1 if self._mean(screen, self.grid_x[i][c], self.y_bits[b]) > 0 else 0
        return out

    def decode_column(self, screen, i, unknown_tol=0.25):
        regs, conf = [], 1.0
        for c in range(self.n_regs):
            bits, ok = [], True
            for b in range(self.word_bits):
                v = self._mean(screen, self.grid_x[i][c], self.y_bits[b])
                conf = min(conf, abs(v))
                ok &= abs(v) >= unknown_tol
                bits.append(1 if v > 0 else 0)
            regs.append(bits_msb_to_int(bits) if ok else None)
        return tuple(regs), conf

    def decode(self, screen, unknown_tol=0.25):
        return [self.decode_column(screen, i, unknown_tol)[0] for i in range(self.n_instr + 1)]

    def decode_instr(self, screen, i):
        a = max(range(self.n_regs), key=lambda c: self._mean(screen, self.grid_x[i][c], self.y_selA))
        b = max(range(self.n_regs), key=lambda c: self._mean(screen, self.grid_x[i][c], self.y_selB))
        d = max(range(self.n_regs), key=lambda c: self._mean(screen, self.grid_x[i + 1][c], self.y_dst))
        op = max(range(len(OPS)), key=lambda k: self._mean(
            screen, self.op_x[i][k // self.word_bits], self.y_bits[k % self.word_bits]))
        return Instr(OPS[op], a, b, d)


def layout_v2_for(n_instr, **kw):
    key = ("v2", n_instr) + tuple(sorted(kw.items()))
    if key not in _LAYOUT_CACHE:
        _LAYOUT_CACHE[key] = TileLayoutV2(n_instr=n_instr, **kw)
    return _LAYOUT_CACHE[key]


class TileLayoutV4:
    """Tile v4: memory row on top, ALU strip underneath, four supervised stages.

    ::

        selA   o . . .                     . . . .   (next instr's selA)
        selB   . o . .                     . . . .   (next instr's selB)
              ┌────────┐                  ┌────────┐
              │ MEM in │ ──── copy ─────▶ │MEM out │
              └────────┘                  └────────┘
                                     dst   . . o .
                   [A][B] ─12px─ [op] ─12px─ [R]
                    load ↓        execute →   store ↑

    One instruction = one tile = ``MEM in | ALU | MEM out``; MEM out is the
    next tile's MEM in.  A register grid is ``n_regs`` columns x ``word_bits``
    rows, MSB on top.  Four supervision stages per tile, in time order:

        LOAD     A <- mem[selA],  B <- mem[selB]
        EXECUTE  R <- op(A, B)
        COPY     mem_out[c] <- mem_in[c]   for c != dst
        STORE    mem_out[dst] <- R

    ``paint(program, regs, stage)`` paints everything resolved after ``stage``
    completed sub-stages (``0`` = t=0 input, ``4*n_instr`` = final).  Site
    kinds (``TYPES``) are exposed as fixed one-hot channels.
    """

    TYPES = ("data", "sel_in", "sel_out", "op", "slot_a", "slot_b", "result")
    STAGES = ("LOAD", "EXECUTE", "COPY", "STORE")

    def __init__(self, n_instr=1, n_regs=4, word_bits=4, ops=("ADD", "COPY"),
                 r=2, pitch=6, pad=5, vgap=3, air=12, unknown=0.0):
        self.n_instr, self.n_regs, self.word_bits = n_instr, n_regs, word_bits
        self.ops = tuple(ops)
        self.r, self.pitch, self.pad, self.vgap, self.air = r, pitch, pad, vgap, air
        self.unknown = unknown
        P = pitch
        self.op_cols = -(-len(self.ops) // word_bits)
        gap = 2 * air + (self.op_cols - 1) * P                # B --air-- op --air-- R
        self.tile_w = (n_regs - 1) * P + gap

        self.grid_x = [[pad + i * self.tile_w + c * P for c in range(n_regs)]
                       for i in range(n_instr + 1)]
        self.op_x = [[self.grid_x[i][-1] + air + c * P for c in range(self.op_cols)]
                     for i in range(n_instr)]
        self.ab_x = [(self.grid_x[i][-2], self.grid_x[i][-1]) for i in range(n_instr)]
        self.r_x = [self.grid_x[i + 1][0] for i in range(n_instr)]

        self.y_selA = pad
        self.y_selB = pad + P
        self.y_mem = [self.y_selB + P + vgap + b * P for b in range(word_bits)]
        self.y_dst = self.y_mem[-1] + P + vgap
        self.y_alu = [self.y_dst + P + vgap + b * P for b in range(word_bits)]
        self.H = self.y_alu[-1] + pad + 1
        self.W = self.grid_x[-1][-1] + pad + 1
        self.n_stages = len(self.STAGES) * n_instr
        self._yx = np.ogrid[: self.H, : self.W]

    # ── sites ───────────────────────────────────────────────────────────────

    def _disc(self, x, y):
        ys, xs = self._yx
        return (xs - x) ** 2 + (ys - y) ** 2 <= self.r * self.r

    def sites(self):
        """Yield ``(kind, x, y, meta)`` for every blob site."""
        for i in range(self.n_instr + 1):
            for c in range(self.n_regs):
                for b in range(self.word_bits):
                    yield "data", self.grid_x[i][c], self.y_mem[b], ("mem", i, c, b)
        for i in range(self.n_instr):
            for c in range(self.n_regs):
                yield "sel_in", self.grid_x[i][c], self.y_selA, ("selA", i, c)
                yield "sel_in", self.grid_x[i][c], self.y_selB, ("selB", i, c)
                yield "sel_out", self.grid_x[i + 1][c], self.y_dst, ("dst", i, c)
            for k in range(len(self.ops)):
                oc, orow = divmod(k, self.word_bits)
                yield "op", self.op_x[i][oc], self.y_alu[orow], ("op", i, k)
            for b in range(self.word_bits):
                yield "slot_a", self.ab_x[i][0], self.y_alu[b], ("A", i, b)
                yield "slot_b", self.ab_x[i][1], self.y_alu[b], ("B", i, b)
                yield "result", self.r_x[i], self.y_alu[b], ("R", i, b)

    def type_channels(self):
        t = np.zeros((len(self.TYPES), self.H, self.W), dtype=np.float32)
        for kind, x, y, _ in self.sites():
            t[self.TYPES.index(kind)][self._disc(x, y)] = 1.0
        return t

    # ── which sites are resolved after `stage` completed sub-stages ─────────

    def _done(self, stage, i, sub):
        """Has sub-stage ``sub`` (0..3) of tile ``i`` completed after ``stage``?"""
        return stage > len(self.STAGES) * i + sub

    def _resolved(self, meta, stage, program):
        what = meta[0]
        if what == "mem":
            _, i, c, _ = meta
            if i == 0:
                return True
            j = i - 1                                        # written by tile i-1
            if c == program[j].dst:
                return self._done(stage, j, 3)               # STORE
            return self._done(stage, j, 2)                   # COPY
        if what in ("A", "B"):
            return self._done(stage, meta[1], 0)             # LOAD
        if what == "R":
            return self._done(stage, meta[1], 1)             # EXECUTE
        return True                                          # program text

    # ── values ──────────────────────────────────────────────────────────────

    def _value(self, meta, program, trace):
        what = meta[0]
        wb = self.word_bits
        bit = lambda v, b: BIT1 if (v >> (wb - 1 - b)) & 1 else BIT0
        if what == "mem":
            _, i, c, b = meta
            return bit(trace[i][c], b)
        if what == "A":
            _, i, b = meta
            return bit(trace[i][program[i].a], b)
        if what == "B":
            _, i, b = meta
            return bit(trace[i][program[i].b], b)
        if what == "R":
            _, i, b = meta
            return bit(trace[i + 1][program[i].dst], b)
        if what == "selA":
            return BIT1 if meta[2] == program[meta[1]].a else BIT0
        if what == "selB":
            return BIT1 if meta[2] == program[meta[1]].b else BIT0
        if what == "dst":
            return BIT1 if meta[2] == program[meta[1]].dst else BIT0
        if what == "op":
            return BIT1 if self.ops[meta[2]] == program[meta[1]].op else BIT0
        raise ValueError(meta)

    def paint(self, program, regs, stage=0):
        assert len(program) == self.n_instr
        assert all(ins.op in self.ops for ins in program), "op not in this layout's ISA"
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        trace = run_program(regs, program, self.word_bits)
        for kind, x, y, meta in self.sites():
            v = self._value(meta, program, trace) if self._resolved(meta, stage, program) else self.unknown
            s[self._disc(x, y)] = v
        return s

    def input_screen(self, program, regs):
        return self.paint(program, regs, 0)

    def stage_targets(self, program, regs):
        """``4 * n_instr`` targets: LOAD, EXECUTE, COPY, STORE for each tile."""
        return [self.paint(program, regs, k + 1) for k in range(self.n_stages)]

    def stage_mask(self, program, stage):
        """1.0 on the blobs that become resolved AT sub-stage ``stage`` (1-based)."""
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for kind, x, y, meta in self.sites():
            if self._resolved(meta, stage, program) and not self._resolved(meta, stage - 1, program):
                m[self._disc(x, y)] = 1.0
        return m

    def stage_name(self, stage):
        """Human name of 1-based sub-stage ``stage``: e.g. 'tile 0 LOAD'."""
        i, sub = divmod(stage - 1, len(self.STAGES))
        return f"tile {i} {self.STAGES[sub]}"

    def _mask(self, pred):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for kind, x, y, meta in self.sites():
            if pred(kind, meta):
                m[self._disc(x, y)] = 1.0
        return m

    def slot_mask(self):
        return self._mask(lambda k, m: True)

    def program_mask(self):
        return self._mask(lambda k, m: k in ("sel_in", "sel_out", "op"))

    def data_mask(self, columns=None):
        cols = set(range(self.n_instr + 1) if columns is None else columns)
        return self._mask(lambda k, m: k == "data" and m[1] in cols)

    def min_rollout(self, px_per_step=1):
        """Travel floor for the longest single hop (the memory copy)."""
        return int(np.ceil(self.tile_w / px_per_step))

    # ── decoder ─────────────────────────────────────────────────────────────

    def _mean(self, screen, x, y):
        return float(np.asarray(screen)[self._disc(x, y)].mean())


    def _word_at(self, screen, x, ys, unknown_tol=0.25):
        bits, ok = [], True
        for b, y in enumerate(ys):
            v = self._mean(screen, x, y)
            ok &= abs(v) >= unknown_tol
            bits.append(1 if v > 0 else 0)
        return bits_msb_to_int(bits) if ok else None

    def decode_bits(self, screen, i):
        out = np.zeros((self.n_regs, self.word_bits), dtype=np.int64)
        for c in range(self.n_regs):
            for b in range(self.word_bits):
                out[c, b] = 1 if self._mean(screen, self.grid_x[i][c], self.y_mem[b]) > 0 else 0
        return out

    def decode_column(self, screen, i, unknown_tol=0.25):
        regs = tuple(self._word_at(screen, self.grid_x[i][c], self.y_mem, unknown_tol)
                     for c in range(self.n_regs))
        return regs, None

    def decode(self, screen, unknown_tol=0.25):
        return [self.decode_column(screen, i, unknown_tol)[0] for i in range(self.n_instr + 1)]

    def decode_alu(self, screen, i, unknown_tol=0.25):
        """(A, B, R) words of tile ``i``; None where unresolved."""
        return (self._word_at(screen, self.ab_x[i][0], self.y_alu, unknown_tol),
                self._word_at(screen, self.ab_x[i][1], self.y_alu, unknown_tol),
                self._word_at(screen, self.r_x[i], self.y_alu, unknown_tol))

    def decode_instr(self, screen, i):
        pick = lambda xs, y: max(range(len(xs)), key=lambda c: self._mean(screen, xs[c], y))
        a = pick(self.grid_x[i], self.y_selA)
        b = pick(self.grid_x[i], self.y_selB)
        d = pick(self.grid_x[i + 1], self.y_dst)
        k = max(range(len(self.ops)), key=lambda k: self._mean(
            screen, self.op_x[i][k // self.word_bits], self.y_alu[k % self.word_bits]))
        return Instr(self.ops[k], a, b, d)


def layout_v4_for(n_instr, **kw):
    key = ("v4", n_instr) + tuple(sorted(kw.items()))
    if key not in _LAYOUT_CACHE:
        _LAYOUT_CACHE[key] = TileLayoutV4(n_instr=n_instr, **kw)
    return _LAYOUT_CACHE[key]


class TileLayoutV5:
    """Tile v5: the minimal tile.  Two ops, no op column, no ALU strip.

    ::

        selA   o . . .            . . . .   (next instr's selA)
        selB   . o . .            . . . .   (next instr's selB)
              ┌────────┐         ┌────────┐
              │ MEM in │ ──────▶ │MEM out │   MEM out = MEM in, except
              └────────┘         └────────┘   MEM out[dst] = mem[selA] - mem[selB]
                             dst  0 0 ± 0

    The op is encoded in the SIGN of the dst blob, no extra blob:
        dst = +1   SUB     dst <- A - B, always
        dst = -1   SUBLEQ  dst <- A - B only if the signed result is <= 0,
                           otherwise dst is copied through unchanged
        dst =  0   not the destination
    Selector rows use 0 for "not selected" (they are program text with their
    own type channel, so 0 never collides with an unknown data bit).  That is
    subleq's test applied to the write instead of to a jump: no jump and no
    flag ever travels, every decision is made inside the tile.

    Two supervised stages per tile: COPY (every column but dst), then STORE
    (the dst column).  Set ``stages=1`` to supervise both at once.
    """

    TYPES = ("data", "sel_in", "sel_out")
    STAGES = ("COPY", "STORE")
    ISA = ("SUB", "SUBLEQ", "NOP")          # NOP = no selector blobs at all, copy everything

    def __init__(self, n_instr=1, n_regs=4, word_bits=4, r=2, pitch=6, pad=5,
                 vgap=3, gap=12, unknown="pixel", stages=2, shape="square"):
        """``unknown``: how an unresolved data bit is drawn.  ``"plus"`` paints
        a small + of five -1 pixels (010/111/010) inside an otherwise blank
        disc, so the four states are visually distinct on one channel:
        white disc = 1, black disc = 0, small plus = not yet set, grey = not a
        bit at all.  A float paints the whole disc at that value."""
        self.n_instr, self.n_regs, self.word_bits = n_instr, n_regs, word_bits
        self.r, self.pitch, self.pad, self.vgap, self.gap = r, pitch, pad, vgap, gap
        self.unknown = unknown
        self.shape = shape          # "square": (2r+1)^2 block, pitch 2r+2 = 1 px between bits
        self.ops = self.ISA
        self.per_tile = stages                      # 2 = COPY then STORE, 1 = both at once
        P = pitch
        self.tile_w = (n_regs - 1) * P + gap
        self.grid_x = [[pad + i * self.tile_w + c * P for c in range(n_regs)]
                       for i in range(n_instr + 1)]
        self.y_selA = pad
        self.y_selB = pad + P
        self.y_mem = [self.y_selB + P + vgap + b * P for b in range(word_bits)]
        self.y_dst = self.y_mem[-1] + P + vgap
        self.H = self.y_dst + pad + 1
        self.W = self.grid_x[-1][-1] + pad + 1
        self.n_stages = self.per_tile * n_instr
        self._yx = np.ogrid[: self.H, : self.W]

    def _disc(self, x, y):
        ys, xs = self._yx
        if self.shape == "square":
            return (np.abs(xs - x) <= self.r) & (np.abs(ys - y) <= self.r)
        return (xs - x) ** 2 + (ys - y) ** 2 <= self.r * self.r

    def _plus(self, x, y):
        """The unresolved glyph: a single centre pixel ("pixel") or the
        five-pixel + (010/111/010, "plus")."""
        ys, xs = self._yx
        if self.unknown == "pixel":
            return (xs == x) & (ys == y)
        return (np.abs(xs - x) + np.abs(ys - y)) <= 1

    def _paint_unknown(self, s, x, y):
        if self.unknown in ("plus", "pixel"):
            s[self._disc(x, y)] = BACKGROUND
            s[self._plus(x, y)] = BIT0                # a -1 dot (or plus)
        else:
            s[self._disc(x, y)] = float(self.unknown)

    def sites(self):
        for i in range(self.n_instr + 1):
            for c in range(self.n_regs):
                for b in range(self.word_bits):
                    yield "data", self.grid_x[i][c], self.y_mem[b], ("mem", i, c, b)
        for i in range(self.n_instr):
            for c in range(self.n_regs):
                yield "sel_in", self.grid_x[i][c], self.y_selA, ("selA", i, c)
                yield "sel_in", self.grid_x[i][c], self.y_selB, ("selB", i, c)
                yield "sel_out", self.grid_x[i + 1][c], self.y_dst, ("dst", i, c)

    def type_channels(self):
        t = np.zeros((len(self.TYPES), self.H, self.W), dtype=np.float32)
        for kind, x, y, _ in self.sites():
            t[self.TYPES.index(kind)][self._disc(x, y)] = 1.0
        return t

    def _resolved(self, meta, stage, program):
        if meta[0] != "mem":
            return True
        _, i, c, _ = meta
        if i == 0:
            return True
        j = i - 1
        if self.per_tile == 1:
            return stage > j
        # STORE column resolves one sub-stage later; a NOP has no store column
        sub = 1 if (c == program[j].dst and program[j].op != "NOP") else 0
        return stage > 2 * j + sub

    def _value(self, meta, program, trace):
        wb = self.word_bits
        if meta[0] == "mem":
            _, i, c, b = meta
            return BIT1 if (trace[i][c] >> (wb - 1 - b)) & 1 else BIT0
        what, i, c = meta
        if program[i].op == "NOP":
            return 0.0                                              # nothing selected
        if what == "dst":
            if c != program[i].dst:
                return 0.0
            return BIT0 if program[i].op == "SUBLEQ" else BIT1     # sign = which op
        sel = program[i].a if what == "selA" else program[i].b
        return BIT1 if c == sel else 0.0

    def paint(self, program, regs, stage=0):
        assert len(program) == self.n_instr
        assert all(ins.op in self.ISA for ins in program), f"v5 ISA is {self.ISA}"
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        trace = run_program(regs, program, self.word_bits)
        for kind, x, y, meta in self.sites():
            if self._resolved(meta, stage, program):
                s[self._disc(x, y)] = self._value(meta, program, trace)
            else:
                self._paint_unknown(s, x, y)
        return s

    def input_screen(self, program, regs):
        return self.paint(program, regs, 0)

    def stage_targets(self, program, regs):
        return [self.paint(program, regs, k + 1) for k in range(self.n_stages)]

    def stage_mask(self, program, stage):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for kind, x, y, meta in self.sites():
            if self._resolved(meta, stage, program) and not self._resolved(meta, stage - 1, program):
                m[self._disc(x, y)] = 1.0
        return m

    def stage_name(self, stage):
        if self.per_tile == 1:
            return f"tile {stage - 1} COPY+STORE"
        i, sub = divmod(stage - 1, 2)
        return f"tile {i} {self.STAGES[sub]}"

    def _mask(self, pred):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for kind, x, y, meta in self.sites():
            if pred(kind, meta):
                m[self._disc(x, y)] = 1.0
        return m

    def slot_mask(self):
        return self._mask(lambda k, m: True)

    def program_mask(self):
        return self._mask(lambda k, m: k != "data")

    def data_mask(self, columns=None):
        cols = set(range(self.n_instr + 1) if columns is None else columns)
        return self._mask(lambda k, m: k == "data" and m[1] in cols)

    def min_rollout(self, px_per_step=1):
        return int(np.ceil((self.tile_w + (self.y_dst - self.y_selA)) / px_per_step))

    def _mean(self, screen, x, y):
        return float(np.asarray(screen)[self._disc(x, y)].mean())

    def _ring(self, screen, x, y):
        """Mean over the disc MINUS the + glyph: +-1 for a set bit, ~0 for an
        unresolved one (the plus only fills the centre), so the two never mix."""
        m = self._disc(x, y) & ~self._plus(x, y)
        return float(np.asarray(screen)[m].mean())

    def _word_at(self, screen, x, unknown_tol=0.25):
        bits, ok = [], True
        for y in self.y_mem:
            v = self._ring(screen, x, y)
            ok &= abs(v) >= unknown_tol
            bits.append(1 if v > 0 else 0)
        return bits_msb_to_int(bits) if ok else None

    def decode_bits(self, screen, i):
        out = np.zeros((self.n_regs, self.word_bits), dtype=np.int64)
        for c in range(self.n_regs):
            for b in range(self.word_bits):
                out[c, b] = 1 if self._ring(screen, self.grid_x[i][c], self.y_mem[b]) > 0 else 0
        return out

    def decode_column(self, screen, i, unknown_tol=0.25):
        return tuple(self._word_at(screen, self.grid_x[i][c], unknown_tol)
                     for c in range(self.n_regs)), None

    def decode(self, screen, unknown_tol=0.25):
        return [self.decode_column(screen, i, unknown_tol)[0] for i in range(self.n_instr + 1)]

    def decode_instr(self, screen, i):
        pick = lambda xs, y: max(range(len(xs)), key=lambda c: self._mean(screen, xs[c], y))
        dv = [self._mean(screen, x, self.y_dst) for x in self.grid_x[i + 1]]
        d = max(range(self.n_regs), key=lambda c: abs(dv[c]))
        if abs(dv[d]) < 0.25:
            return Instr("NOP", 0, 0, 0)
        op = "SUBLEQ" if dv[d] < 0 else "SUB"
        return Instr(op, pick(self.grid_x[i], self.y_selA), pick(self.grid_x[i], self.y_selB), d)


def layout_v5_for(n_instr, **kw):
    key = ("v5", n_instr) + tuple(sorted(kw.items()))
    if key not in _LAYOUT_CACHE:
        _LAYOUT_CACHE[key] = TileLayoutV5(n_instr=n_instr, **kw)
    return _LAYOUT_CACHE[key]


class TileLayoutV6(TileLayoutV5):
    """Tile v6: v5 plus an ALU strip underneath, three supervised stages.

    ::

        selA   o . . .            . . . .
        selB   . o . .            . . . .
              ┌────────┐         ┌────────┐
              │ MEM in │ ──────▶ │MEM out │
              └────────┘         └────────┘
                             dst  0 0 ± 0
                    [A][B]     [R]            ALU strip, under the grids

    Stage 1  LOAD          A <- mem[selA],  B <- mem[selB]      (down)
    Stage 2  EXEC + COPY   R <- A - B  and  mem_out[c] <- mem_in[c], c != dst
    Stage 3  STORE         mem_out[dst] <- R  (SUB always; SUBLEQ only if R <= 0,
                                              else mem_out[dst] <- mem_in[dst])

    A and B sit under the last two columns of MEM in; R sits one square plus
    two spacings to the right of B, which puts it directly under MEM out's
    first column.  A NOP tile has no selectors and its ALU slots stay unset.
    """

    TYPES = ("data", "sel_in", "sel_out", "slot_a", "slot_b", "result")
    STAGES = ("LOAD", "EXEC+COPY", "STORE")

    def __init__(self, n_instr=1, **kw):
        kw.setdefault("stages", 3)
        super().__init__(n_instr=n_instr, **kw)
        P = self.pitch
        self.per_tile = 3
        self.n_stages = 3 * n_instr
        self.ab_x = [(self.grid_x[i][-2], self.grid_x[i][-1]) for i in range(n_instr)]
        self.r_x = [self.grid_x[i][-1] + self.gap for i in range(n_instr)]       # = grid_x[i+1][0]
        self.y_alu = [self.y_dst + P + self.vgap + b * P for b in range(self.word_bits)]
        self.H = self.y_alu[-1] + self.pad + 1
        self._yx = np.ogrid[: self.H, : self.W]

    def sites(self):
        yield from super().sites()
        for i in range(self.n_instr):
            for b in range(self.word_bits):
                yield "slot_a", self.ab_x[i][0], self.y_alu[b], ("A", i, b)
                yield "slot_b", self.ab_x[i][1], self.y_alu[b], ("B", i, b)
                yield "result", self.r_x[i], self.y_alu[b], ("R", i, b)

    def _done(self, stage, i, sub):
        return stage > 3 * i + sub

    def _resolved(self, meta, stage, program):
        what = meta[0]
        if what == "mem":
            _, i, c, _ = meta
            if i == 0:
                return True
            j = i - 1
            if program[j].op != "NOP" and c == program[j].dst:
                return self._done(stage, j, 2)                # STORE
            return self._done(stage, j, 1)                    # COPY
        if what in ("A", "B", "R"):
            i = meta[1]
            if program[i].op == "NOP":
                return False                                  # nothing to load
            return self._done(stage, i, 0 if what in ("A", "B") else 1)
        return True

    def _value(self, meta, program, trace):
        what = meta[0]
        wb = self.word_bits
        bit = lambda v, b: BIT1 if (v >> (wb - 1 - b)) & 1 else BIT0
        if what == "A":
            _, i, b = meta; return bit(trace[i][program[i].a], b)
        if what == "B":
            _, i, b = meta; return bit(trace[i][program[i].b], b)
        if what == "R":
            _, i, b = meta
            r = (trace[i][program[i].a] - trace[i][program[i].b]) & ((1 << wb) - 1)
            return bit(r, b)                                  # raw A-B, even if SUBLEQ refuses it
        return super()._value(meta, program, trace)

    def stage_name(self, stage):
        i, sub = divmod(stage - 1, 3)
        return f"tile {i} {self.STAGES[sub]}"

    def program_mask(self):
        return self._mask(lambda k, m: k in ("sel_in", "sel_out"))

    def min_rollout(self, px_per_step=1):
        """Floor of the LONGEST single stage: the memory copy (tile_w across)."""
        hops = {
            "load": (self.y_alu[0] - self.y_mem[-1]) + 3 * self.pitch,     # down, up to 3 cols across
            "exec": self.r_x[0] - self.ab_x[0][0],                          # A/B -> R across
            "copy": self.tile_w,
            "store": (self.y_alu[0] - self.y_mem[-1]) + 3 * self.pitch,    # up, up to 3 cols across
        }
        return int(np.ceil(max(hops.values()) / px_per_step))

    def hop_floors(self, px_per_step=1):
        return {
            "load ↓": int(np.ceil(((self.y_alu[0] - self.y_mem[-1]) + 3 * self.pitch) / px_per_step)),
            "exec →": int(np.ceil((self.r_x[0] - self.ab_x[0][0]) / px_per_step)),
            "copy →": int(np.ceil(self.tile_w / px_per_step)),
            "store ↑": int(np.ceil(((self.y_alu[0] - self.y_mem[-1]) + 3 * self.pitch) / px_per_step)),
        }

    def decode_alu(self, screen, i, unknown_tol=0.25):
        def word(x):
            bits, ok = [], True
            for y in self.y_alu:
                v = float(np.asarray(screen)[self._disc(x, y) & ~self._plus(x, y)].mean())
                ok &= abs(v) >= unknown_tol; bits.append(1 if v > 0 else 0)
            return bits_msb_to_int(bits) if ok else None
        return word(self.ab_x[i][0]), word(self.ab_x[i][1]), word(self.r_x[i])


def layout_v6_for(n_instr, **kw):
    key = ("v6", n_instr) + tuple(sorted(kw.items()))
    if key not in _LAYOUT_CACHE:
        _LAYOUT_CACHE[key] = TileLayoutV6(n_instr=n_instr, **kw)
    return _LAYOUT_CACHE[key]


class TileLayoutV7(TileLayoutV5):
    """Tile v7: compute IN PLACE, then move.

    ::

        selA   o . . .            . . . .   (next instr's selA)
        selB   . o . .            . . . .
              ┌────────┐         ┌────────┐
              │ grid i │ ──────▶ │grid i+1│
              └────────┘         └────────┘
        dst    0 0 ± 0            . . . .   (next instr's dst)

    Selectors for instruction ``i`` sit ABOVE and BELOW grid ``i`` -- the same
    grid.  Two supervised stages per tile:

        COMPUTE (K steps)   grid i's dst column becomes A - B, in place
                            (SUBLEQ: only if A - B <= 0, else unchanged)
        MOVE    (P steps)   grid i+1 becomes a copy of the computed grid i

    The last grid carries no selectors.  A NOP tile has no selectors: its
    COMPUTE stage changes nothing and MOVE copies the grid as is.
    """

    TYPES = ("data", "sel_in", "sel_out")
    STAGES = ("COMPUTE", "MOVE")

    def __init__(self, n_instr=1, **kw):
        kw.setdefault("stages", 2)
        super().__init__(n_instr=n_instr, **kw)
        self.per_tile = 2
        self.n_stages = 2 * n_instr

    def sites(self):
        for i in range(self.n_instr + 1):
            for c in range(self.n_regs):
                for b in range(self.word_bits):
                    yield "data", self.grid_x[i][c], self.y_mem[b], ("mem", i, c, b)
        for i in range(self.n_instr):                       # selectors belong to grid i itself
            for c in range(self.n_regs):
                yield "sel_in", self.grid_x[i][c], self.y_selA, ("selA", i, c)
                yield "sel_in", self.grid_x[i][c], self.y_selB, ("selB", i, c)
                yield "sel_out", self.grid_x[i][c], self.y_dst, ("dst", i, c)

    # ── state of one site after `stage` completed sub-stages ─────────────────
    #   returns a float value, or None for "unresolved"

    def _site(self, meta, stage, program, trace):
        wb = self.word_bits
        bit = lambda v, b: BIT1 if (v >> (wb - 1 - b)) & 1 else BIT0
        if meta[0] != "mem":
            return self._value(meta, program, trace)
        _, i, c, b = meta
        computed = stage > 2 * i                            # COMPUTE of tile i done
        arrived = (i == 0) or stage > 2 * (i - 1) + 1       # MOVE of tile i-1 done
        if not arrived:
            return None
        if i < self.n_instr and computed:
            return bit(trace[i + 1][c], b)                  # in-place result
        return bit(trace[i][c], b)

    def paint(self, program, regs, stage=0):
        assert len(program) == self.n_instr
        assert all(ins.op in self.ISA for ins in program), f"ISA is {self.ISA}"
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        trace = run_program(regs, program, self.word_bits)
        for kind, x, y, meta in self.sites():
            v = self._site(meta, stage, program, trace)
            if v is None:
                self._paint_unknown(s, x, y)
            else:
                s[self._disc(x, y)] = v
        return s

    def stage_mask(self, program, stage, regs=None):
        """1.0 on the blobs whose target CHANGES at sub-stage ``stage``: the dst
        column at COMPUTE, the whole next grid at MOVE."""
        m = np.zeros((self.H, self.W), dtype=np.float32)
        i, sub = divmod(stage - 1, 2)
        for kind, x, y, meta in self.sites():
            if meta[0] != "mem":
                continue
            _, gi, c, _ = meta
            if sub == 0 and gi == i and program[i].op != "NOP" and c == program[i].dst:
                m[self._disc(x, y)] = 1.0
            elif sub == 1 and gi == i + 1:
                m[self._disc(x, y)] = 1.0
        return m

    def stage_name(self, stage):
        i, sub = divmod(stage - 1, 2)
        return f"tile {i} {self.STAGES[sub]}"

    def min_rollout(self, px_per_step=1):
        return int(np.ceil(self.tile_w / px_per_step))

    def hop_floors(self, px_per_step=1):
        v = self.y_dst - self.y_selA
        return {"compute (within grid, sel rows -> far bit)": int(np.ceil((v + 3 * self.pitch) / px_per_step)),
                "move (grid -> next grid)": int(np.ceil(self.tile_w / px_per_step))}

    def decode_instr(self, screen, i):
        pick = lambda xs, y: max(range(len(xs)), key=lambda c: self._mean(screen, xs[c], y))
        dv = [self._mean(screen, x, self.y_dst) for x in self.grid_x[i]]
        d = max(range(self.n_regs), key=lambda c: abs(dv[c]))
        if abs(dv[d]) < 0.25:
            return Instr("NOP", 0, 0, 0)
        op = "SUBLEQ" if dv[d] < 0 else "SUB"
        return Instr(op, pick(self.grid_x[i], self.y_selA), pick(self.grid_x[i], self.y_selB), d)


def layout_v7_for(n_instr, **kw):
    key = ("v7", n_instr) + tuple(sorted(kw.items()))
    if key not in _LAYOUT_CACHE:
        _LAYOUT_CACHE[key] = TileLayoutV7(n_instr=n_instr, **kw)
    return _LAYOUT_CACHE[key]


class TileLayoutV8:
    """v8: ONE data grid, in-place compute; the instruction lives in three
    extra channels, not in the picture.

    Channel 0     the register grid: ``n_regs`` columns x ``word_bits`` rows of
                  5x5 squares, 1 px apart (same bit encoding as v5).
    Channel A     a +1 rectangle over the column of register selA
    Channel B     a +1 rectangle over the column of register selB
    Channel R     a +1 rectangle over the destination column (dst);
                  +1 = SUB (always write), -1 = SUBLEQ (write only if <= 0)

    No read-only channels: the rectangles are painted at t=0 and the rule may
    do what it likes with them afterwards.  Task: after K steps the dst column
    holds A - B (in place); every other column is unchanged.  NOP = all three
    rectangle channels empty, nothing changes.
    """

    MASKS = ("A", "B", "R")
    ISA = ("SUB", "SUBLEQ", "NOP")

    def __init__(self, n_regs=4, word_bits=4, r=2, pitch=8, pad=6, unknown="pixel", shape="square"):
        """pitch 8 = 5 px squares with 3 px between them; a highlight is the
        squares plus a 1 px margin on every side (7 px), leaving 1 px between
        neighbouring highlights."""
        self.n_regs, self.word_bits, self.r, self.pitch, self.pad = n_regs, word_bits, r, pitch, pad
        self.unknown, self.shape = unknown, shape
        self.n_instr, self.n_stages = 1, 1
        self.col_x = [pad + c * pitch for c in range(n_regs)]
        self.row_y = [pad + b * pitch for b in range(word_bits)]
        self.W = self.col_x[-1] + pad + 1
        self.H = self.row_y[-1] + pad + 1
        self.grid_x = [self.col_x]
        self._yx = np.ogrid[: self.H, : self.W]

    def _disc(self, x, y):
        ys, xs = self._yx
        if self.shape == "square":
            return (np.abs(xs - x) <= self.r) & (np.abs(ys - y) <= self.r)
        return (xs - x) ** 2 + (ys - y) ** 2 <= self.r * self.r

    def _plus(self, x, y):
        ys, xs = self._yx
        return (xs == x) & (ys == y)

    def _column_rect(self, c):
        """Rectangle covering register column ``c``: exactly the squares' extent,
        so rectangles of neighbouring columns never touch (1 px between them)."""
        ys, xs = self._yx
        x = self.col_x[c]
        m = self.r + 1                                     # squares + 1 px margin, all sides
        return ((np.abs(xs - x) <= m) & (ys >= self.row_y[0] - m) & (ys <= self.row_y[-1] + m))

    # ── painting ────────────────────────────────────────────────────────────

    def data_screen(self, regs):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        for c in range(self.n_regs):
            for b in range(self.word_bits):
                v = BIT1 if (regs[c] >> (self.word_bits - 1 - b)) & 1 else BIT0
                s[self._disc(self.col_x[c], self.row_y[b])] = v
        return s

    def mask_channels(self, ins):
        """(3, H, W): the A / B / R rectangles for instruction ``ins``."""
        m = np.zeros((3, self.H, self.W), dtype=np.float32)
        if ins.op == "NOP":
            return m
        m[0][self._column_rect(ins.a)] = 1.0
        m[1][self._column_rect(ins.b)] = 1.0
        m[2][self._column_rect(ins.dst)] = BIT0 if ins.op == "SUBLEQ" else BIT1
        return m

    def input_screen(self, program, regs):
        return self.data_screen(regs)

    def paint(self, program, regs, stage=0):
        trace = run_program(regs, program, self.word_bits)
        return self.data_screen(trace[1] if stage >= 1 else trace[0])

    def stage_targets(self, program, regs):
        return [self.paint(program, regs, 1)]

    def stage_mask(self, program, stage):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        if program[0].op != "NOP":
            for b in range(self.word_bits):
                m[self._disc(self.col_x[program[0].dst], self.row_y[b])] = 1.0
        return m

    def data_mask(self, columns=None):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for c in range(self.n_regs):
            for b in range(self.word_bits):
                m[self._disc(self.col_x[c], self.row_y[b])] = 1.0
        return m

    slot_mask = data_mask

    def program_mask(self):
        return np.zeros((self.H, self.W), dtype=np.float32)

    def type_channels(self):
        return np.zeros((0, self.H, self.W), dtype=np.float32)

    def min_rollout(self, px_per_step=1):
        return int(np.ceil((self.col_x[-1] - self.col_x[0] + self.row_y[-1] - self.row_y[0]) / px_per_step))

    # ── decoding / display ──────────────────────────────────────────────────

    def _ring(self, screen, x, y):
        m = self._disc(x, y) & ~self._plus(x, y)
        return float(np.asarray(screen)[m].mean())

    def decode_bits(self, screen, i=0):
        out = np.zeros((self.n_regs, self.word_bits), dtype=np.int64)
        for c in range(self.n_regs):
            for b in range(self.word_bits):
                out[c, b] = 1 if self._ring(screen, self.col_x[c], self.row_y[b]) > 0 else 0
        return out

    def decode_column(self, screen, i=0, unknown_tol=0.25):
        regs = []
        for c in range(self.n_regs):
            bits, ok = [], True
            for b in range(self.word_bits):
                v = self._ring(screen, self.col_x[c], self.row_y[b]); ok &= abs(v) >= unknown_tol
                bits.append(1 if v > 0 else 0)
            regs.append(bits_msb_to_int(bits) if ok else None)
        return tuple(regs), None

    def decode(self, screen, unknown_tol=0.25):
        return [self.decode_column(screen, 0, unknown_tol)[0]]

    # display colours: the two inputs are neighbouring hues (blue / cyan) so they
    # read as "the operands"; the two output states are far apart (orange = SUB,
    # magenta = SUBLEQ) so the op is unmistakable.
    COLORS = {"A": (0.25, 0.45, 1.00), "B": (0.10, 0.85, 0.90),
              "R+": (1.00, 0.60, 0.05), "R-": (0.95, 0.10, 0.65)}

    @classmethod
    def to_rgb(cls, data, masks):
        """Blend: grayscale data (channel 0) tinted by the A / B / R rectangles."""
        g = np.clip((np.asarray(data) + 1) / 2, 0, 1)
        rgb = np.stack([g, g, g], -1)
        a, b, r = (np.clip(np.asarray(m), -1, 1) for m in masks)
        tint = np.zeros_like(rgb); w = np.zeros(rgb.shape[:2] + (1,), dtype=rgb.dtype)
        for m, key in ((np.clip(a, 0, 1), "A"), (np.clip(b, 0, 1), "B"),
                       (np.clip(r, 0, 1), "R+"), (np.clip(-r, 0, 1), "R-")):
            tint += m[..., None] * np.array(cls.COLORS[key], dtype=rgb.dtype)
            w = np.maximum(w, m[..., None])
        w = w * 0.5
        return (np.clip(rgb * (1 - w) + np.clip(tint, 0, 1) * w, 0, 1) * 255).astype(np.uint8)


class TileLayoutV9(TileLayoutV8):
    """v9: the DATA stays put, the PROGRAM moves.

    One register grid (as v8) at the bottom.  Above it a stack of instructions,
    each a 1 px tall row in the three instruction channels: a short segment
    over selA's column (channel A), over selB's (channel B), over dst's
    (channel R, sign = SUB / SUBLEQ).  Instruction 0 is the lowest row.

    Clocking, two stages per instruction:

        LOAD   the whole stack moves down one pitch; the bottom row enters the
               grid and EXPANDS into the full-height rectangles of v8
        EXEC   the rule computes: dst column <- A - B (in place).  With ``drain``
               it also RETRACTS the rectangles and zeroes the hidden channels, so
               the end of EXEC is exactly the state the machine starts in.

    Then the next row comes down and its rectangles REPLACE the old ones.  The
    stack is always ``n_instr`` rows at t=0 and shrinks by one row per LOAD,
    so program length is just how tall the stack is.
    """

    STAGES = ("LOAD", "EXEC")

    def __init__(self, n_instr=1, row_h=5, stack_pitch=None, stack_gap=4, preloaded=False,
                 headroom=0, window=0, drain=False, **kw):
        """``row_h``: height of one instruction band; ``stack_pitch`` defaults to
        ``row_h`` so bands touch.  Instruction 0 is the lowest band.

        ``preloaded``: instruction 0 starts already expanded over the grid (the
        picture the ALU was trained on) and the stack holds instructions 1..n-1;
        the stage sequence is then EXEC, LOAD, EXEC, LOAD, ... (2n-1 stages)."""
        super().__init__(**kw)
        self.n_instr = n_instr
        self.preloaded = preloaded
        # ``drain``: every cycle ends with a third sub-stage that returns the machine to
        # its REST form -- hidden channels at zero and the expanded rectangles retracted,
        # so the state at a cycle boundary is exactly the state at t=0. One cycle is then
        # the whole curriculum: cycle N is the same problem as cycle 1, by construction.
        self.drain = drain
        if drain and preloaded:
            raise ValueError("drain and preloaded are mutually exclusive")
        # Two phases either way. ``drain`` does NOT add a third stage: it changes what
        # EXEC must END in -- the rectangles retracted and the hidden channels at zero,
        # i.e. the REST form. A separate DRAIN stage asked the rule to SIT in the rest
        # state for a window and then leave it, which is the same state with the same
        # (zero) hidden demanding two different behaviours.
        self.phases = 2
        if not n_instr:
            self.n_stages = 1                                 # length 0: one HOLD stage
        else:
            self.n_stages = (2 * n_instr - 1 if preloaded
                             else self.phases * n_instr)
        self.row_h = row_h
        self.stack_pitch, self.stack_gap = (stack_pitch or row_h), stack_gap
        # push the grid down to make room for the stack above it
        # ``window``: the stack is a FIXED number of slots and the program streams
        # through it. Each LOAD drops the stack one slot and frees the top one, where
        # the next instruction (or a blank NOP row) is painted. The grid is then the
        # same size for a 2-instruction program and a 2000-instruction one.
        self.headroom, self.window = headroom, window
        slots = window if window else max(n_instr - (1 if preloaded else 0), 0) + headroom
        stack_h = slots * self.stack_pitch + stack_gap
        self.row_y = [y + stack_h for y in self.row_y]
        self.H = self.row_y[-1] + self.pad + 1
        self.y_stack0 = self.row_y[0] - self.r - 1 - stack_gap      # BOTTOM row of the next instruction
        self._yx = np.ogrid[: self.H, : self.W]

    # ── program painting ────────────────────────────────────────────────────

    def _row_segment(self, m, ch, c, y, value):
        """A band ``row_h`` tall whose BOTTOM row is ``y``, over column ``c``."""
        x = self.col_x[c]
        m[ch, y - self.row_h + 1: y + 1, x - self.r - 1: x + self.r + 2] = value

    def _instr_row(self, m, ins, y):
        if ins.op == "NOP":
            return
        self._row_segment(m, 0, ins.a, y, 1.0)
        self._row_segment(m, 1, ins.b, y, 1.0)
        self._row_segment(m, 2, ins.dst, y, BIT0 if ins.op == "SUBLEQ" else BIT1)

    def loads_done(self, stage):
        """How many instructions have been loaded after ``stage`` sub-stages."""
        if not self.n_instr:
            return 0
        if self.preloaded:
            return min(1 + stage // 2, self.n_instr)
        return min((stage + self.phases - 1) // self.phases, self.n_instr)

    def execs_done(self, stage):
        if not self.n_instr:
            return 0
        if self.preloaded:
            return (stage + 1) // 2
        return (stage + self.phases - 2) // self.phases

    def phase_of(self, k):
        """Name of 0-based sub-stage ``k``."""
        if not self.n_instr:
            return "HOLD"
        if self.preloaded:
            return "EXEC" if k % 2 == 0 else "LOAD"
        return ("LOAD", "EXEC")[k % self.phases]

    def is_exec(self, k):
        """Is 0-based sub-stage ``k`` an EXEC stage?"""
        if not self.n_instr:
            return True                                       # HOLD behaves like a compute window
        return self.phase_of(k) == "EXEC"

    def is_drain(self, k):
        return False                                          # merged into EXEC

    def at_rest(self, stage):
        """Is the machine in its REST form after ``stage`` completed sub-stages?
        Rest = instruction bands only, nothing expanded over the grid."""
        if not self.drain:
            return stage == 0
        return stage % self.phases == 0

    def top_slot_band(self, program, stage):
        """(3,H,W) holding ONLY the newest instruction, in the slot the shift frees.

        The rule's LOAD is a uniform downward shift, so the top slot is empty when it
        finishes; whatever feeds the program paints the next instruction there (or
        leaves it blank, which reads as a NOP).
        """
        m = np.zeros((3, self.H, self.W), dtype=np.float32)
        if not self.window:
            return m
        loads = self.loads_done(stage)
        j = loads + self.window - 1
        if j < self.n_instr:
            self._instr_row(m, program[j], self.y_stack0 - (self.window - 1) * self.stack_pitch)
        return m

    def mask_channels_at(self, program, stage, injected=True):
        """(3, H, W) instruction channels after ``stage`` completed sub-stages:
        the last loaded instruction expanded over the grid, the rest in the
        stack shifted down by the number of loads."""
        m = np.zeros((3, self.H, self.W), dtype=np.float32)
        loads = self.loads_done(stage)
        if loads and not self.at_rest(stage):
            m += self.mask_channels(program[loads - 1])           # v8 rectangles of the current instr
        # waiting instructions, bottom-up. With a window the stack is always `window`
        # slots tall; slots past the end of the program are simply blank (a NOP row).
        top = loads + self.window - (0 if injected else 1) if self.window else self.n_instr
        for j in range(loads, min(top, self.n_instr)):
            self._instr_row(m, program[j], self.y_stack0 - (j - loads) * self.stack_pitch)
        return m

    def input_masks(self, program):
        return self.mask_channels_at(program, 0)

    def paint(self, program, regs, stage=0):
        """Data channel after ``stage`` sub-stages."""
        trace = run_program(regs, program, self.word_bits)
        return self.data_screen(trace[self.execs_done(stage)])

    def stage_targets(self, program, regs):
        return [self.paint(program, regs, k + 1) for k in range(self.n_stages)]

    def stage_mask(self, program, stage):
        """Data bits that change at this sub-stage (EXEC of instruction i)."""
        m = np.zeros((self.H, self.W), dtype=np.float32)
        if self.n_instr and self.is_exec(stage - 1):
            i = self.execs_done(stage) - 1
            if program[i].op != "NOP":
                for b in range(self.word_bits):
                    m[self._disc(self.col_x[program[i].dst], self.row_y[b])] = 1.0
        return m

    def stage_name(self, stage):
        if not self.n_instr:
            return "HOLD (empty program)"
        ph = self.phase_of(stage - 1)
        if ph == "EXEC":
            return f"instr {self.execs_done(stage) - 1} EXEC"
        if ph == "DRAIN":
            return f"instr {self.execs_done(stage) - 1} DRAIN"
        return f"instr {self.loads_done(stage) - 1} LOAD"

    def decode(self, screen, unknown_tol=0.25):
        return [self.decode_column(screen, 0, unknown_tol)[0]]


def layout_v9_for(n_instr, **kw):
    key = ("v9", n_instr) + tuple(sorted(kw.items()))
    if key not in _LAYOUT_CACHE:
        _LAYOUT_CACHE[key] = TileLayoutV9(n_instr=n_instr, **kw)
    return _LAYOUT_CACHE[key]


class TileLayoutV10(TileLayoutV8):
    """v10: a conveyor.  No stages, no clock -- the program scrolls past the ALU.

    ::

            (random blank space -- the top border carries no information)
            ▓▓▓▓  instruction j+2        one 5 px band per instruction,
            ▓▓▓▓  instruction j+1        contiguous, scrolling DOWN
            ▓▓▓▓  instruction j          at 1 px every `scroll` steps
        ────────  the line              a band is consumed as it crosses
            ░░░░  the register grid     registers hold trace[bands consumed]

    The picture at any time is a pure function of the scroll offset ``o``:
    ``o // row_h`` instructions have been consumed, the rest are drawn shifted
    down by ``o`` and clipped at the line.  There is no stage boundary for the
    rule to time: an instruction is current when it is *there*.
    """

    def __init__(self, n_instr=1, row_h=5, gap=3, top_pad=6, scroll=8, **kw):
        super().__init__(**kw)
        self.n_instr, self.row_h, self.gap, self.top_pad = n_instr, row_h, gap, top_pad
        self.scroll = scroll                       # CA steps per pixel of belt travel
        stack_h = n_instr * row_h + gap + top_pad
        self.row_y = [y + stack_h for y in self.row_y]
        self.y_line = self.row_y[0] - self.r - 1 - gap          # bands live strictly above this
        self.H = self.row_y[-1] + self.pad + 1
        self._yx = np.ogrid[: self.H, : self.W]

    # ── the picture at a given belt offset ──────────────────────────────────

    def consumed(self, o):
        return min(o // self.row_h, self.n_instr)

    def total_travel(self):
        """Belt pixels needed to consume the whole program."""
        return self.n_instr * self.row_h

    def steps_for(self, o):
        return o * self.scroll

    def _band(self, m, ins, y_bot):
        """Draw one instruction band whose bottom edge is ``y_bot``, clipped at the line."""
        if ins.op == "NOP":
            return
        y0, y1 = y_bot - self.row_h + 1, y_bot
        y0, y1 = max(y0, 0), min(y1, self.y_line)               # clip: nothing below the line
        if y1 < y0:
            return
        for ch, c, v in ((0, ins.a, 1.0), (1, ins.b, 1.0),
                         (2, ins.dst, BIT0 if ins.op == "SUBLEQ" else BIT1)):
            x = self.col_x[c]
            m[ch, y0: y1 + 1, x - self.r - 1: x + self.r + 2] = v

    def mask_channels_at(self, program, o):
        m = np.zeros((3, self.H, self.W), dtype=np.float32)
        for j in range(self.consumed(o), self.n_instr):
            self._band(m, program[j], self.y_line - j * self.row_h + o)
        return m

    def data_at(self, program, regs, o):
        trace = run_program(regs, program, self.word_bits)
        return self.data_screen(trace[self.consumed(o)])

    def picture_at(self, program, regs, o):
        return self.data_at(program, regs, o), self.mask_channels_at(program, o)

    def settled(self, o):
        """True when the belt is far enough past a boundary that the register
        write for the instruction just consumed must already be complete."""
        return self.n_instr == 0 or o % self.row_h >= 2 or o >= self.total_travel()

    # ── interface parity ────────────────────────────────────────────────────

    def input_screen(self, program, regs):
        return self.data_at(program, regs, 0)

    def paint(self, program, regs, stage=0):
        return self.data_at(program, regs, stage)

    def decode(self, screen, unknown_tol=0.25):
        return [self.decode_column(screen, 0, unknown_tol)[0]]

    def min_rollout(self, px_per_step=1):
        return self.steps_for(self.total_travel())


def layout_v10_for(n_instr, **kw):
    key = ("v10", n_instr) + tuple(sorted(kw.items()))
    if key not in _LAYOUT_CACHE:
        _LAYOUT_CACHE[key] = TileLayoutV10(n_instr=n_instr, **kw)
    return _LAYOUT_CACHE[key]


# ─────────────────────────────────────────────────────────────────────────────
# v11: the machine split into four phases
# ─────────────────────────────────────────────────────────────────────────────

class V11State:
    """Everything the picture has to show at one moment of the v11 cycle."""

    __slots__ = ("regs", "A", "B", "R", "op", "consumed", "parked")

    def __init__(self, regs, A=None, B=None, R=None, op=None, consumed=0, parked=False):
        self.regs, self.A, self.B, self.R, self.op = list(regs), A, B, R, op
        self.consumed, self.parked = consumed, parked

    def copy(self):
        return V11State(self.regs, self.A, self.B, self.R, self.op, self.consumed, self.parked)


class TileLayoutV11:
    """One instruction is ONE band row; the cycle is four supervised phases.

    Lattice (pitch ``pitch``), left to right::

        r0  r1  r2  r3   ·   A   B   ·   op   R
        └── memory ──┘      operand      result
                             slots

    The tape sits above.  Each instruction is a single row in which the two
    input selectors and the return selector are three separate *channels*, so
    all three are aligned with each other and with the register columns they
    point at.  A fourth channel carries the op parameter over the ``op`` column
    (+1 SUB, −1 SUBLEQ).  A NOP row paints nothing in any channel.

    The four phases, each supervised on its own::

        FETCH     A <- mem[selA],  B <- mem[selB],  op cell <- op parameter
        EXEC      R <- A - B
        STORE     mem[dst] <- R, gated by op (SUBLEQ writes only if R <= 0)
        ADVANCE   tape drops one row; slots and op cell clear

    At t=0 the tape is parked ``park`` pixels clear of the grid, so nothing is
    selected until the first ADVANCE.
    """

    MASKS = ("selA", "selB", "dst", "op")
    ISA = ("SUB", "SUBLEQ", "NOP")
    PHASES = ("FETCH", "EXEC", "STORE", "ADVANCE")

    def __init__(self, n_instr=4, n_regs=4, word_bits=4, r=2, pitch=8, pad=6, noise=2, park=2,
                 expand=False, sel_amp=1.0):
        # selectors painted at the same amplitude as the data they gate are easy to
        # ignore: the rule settles on copying a fixed register and never reads them.
        self.sel_amp = sel_amp
        self.expand = expand          # selector at the line spans its register column (v8 geometry)
        self.n_instr, self.n_regs, self.word_bits = n_instr, n_regs, word_bits
        self.r, self.pitch, self.pad = r, pitch, pad
        self.park = park * pitch                       # t=0 clearance of the tape
        self.gap = pitch                               # current band -> first bit row

        # one lattice: registers, blank, A, B, blank, op, R
        # R sits next to its operands: EXEC is the hop that has to compute, so it
        # gets the shortest one.  op gates STORE, not EXEC, so it moves outboard.
        self.slots = [f"r{c}" for c in range(n_regs)] + [None, "A", "B", "R", None, "op"]
        self.lat_x = [pad + i * pitch for i in range(len(self.slots))]
        self.x = {s: self.lat_x[i] for i, s in enumerate(self.slots) if s}
        self.col_x = [self.x[f"r{c}"] for c in range(n_regs)]

        y0 = pad + noise * pitch + n_instr * pitch + self.park + self.gap
        self.row_y = [y0 + b * pitch for b in range(word_bits)]
        self.W = self.lat_x[-1] + pad + 1
        self.H = self.row_y[-1] + pad + 1
        self._yx = np.ogrid[: self.H, : self.W]
        self.mask = (1 << word_bits) - 1

    # ── geometry ────────────────────────────────────────────────────────────

    def band_y(self, k, parked=False):
        """Row of the k-th remaining band; k=0 is the one at the line."""
        return self.row_y[0] - self.gap - k * self.pitch - (self.park if parked else 0)

    def _disc(self, x, y):
        ys, xs = self._yx
        return (np.abs(xs - x) <= self.r) & (np.abs(ys - y) <= self.r)

    def _plus(self, x, y):
        ys, xs = self._yx
        return ((xs == x) & (np.abs(ys - y) <= 1)) | ((ys == y) & (np.abs(xs - x) <= 1))

    def _column_span(self, c):
        """The register column's own extent: the bit squares plus a 1 px margin."""
        ys, xs = self._yx
        mgn = self.r + 1
        return ((np.abs(xs - self.col_x[c]) <= mgn)
                & (ys >= self.row_y[0] - mgn) & (ys <= self.row_y[-1] + mgn))

    def _cell(self, s, x, y, v):
        """Paint one bit cell: ``v`` in {+1, -1} fills the square, None marks
        'unset' as a bare plus at -1 with a background ring (v8 convention)."""
        if v is None:
            s[self._disc(x, y)] = BACKGROUND
            s[self._plus(x, y)] = BIT0
        else:
            s[self._disc(x, y)] = BIT1 if v > 0 else BIT0

    # ── painting ────────────────────────────────────────────────────────────

    def data_screen(self, st):
        s = np.full((self.H, self.W), BACKGROUND, dtype=np.float32)
        for c in range(self.n_regs):
            for b in range(self.word_bits):
                self._cell(s, self.col_x[c], self.row_y[b],
                           1 if (st.regs[c] >> (self.word_bits - 1 - b)) & 1 else -1)
        for key in ("A", "B", "R"):
            v = getattr(st, key)
            for b in range(self.word_bits):
                self._cell(s, self.x[key], self.row_y[b],
                           None if v is None else (1 if (v >> (self.word_bits - 1 - b)) & 1 else -1))
        self._cell(s, self.x["op"], self.row_y[self.word_bits // 2], st.op)
        return s

    def mask_channels(self, program, st):
        """(4, H, W): selA / selB / dst / op, one row per remaining instruction."""
        m = np.zeros((4, self.H, self.W), dtype=np.float32)
        for k, ins in enumerate(program[st.consumed:]):
            if ins.op == "NOP":
                continue
            y = self.band_y(k, st.parked)
            if y - self.r < 0:
                continue
            amp = self.sel_amp
            m[0][self._disc(self.col_x[ins.a], y)] = BIT1 * amp
            m[1][self._disc(self.col_x[ins.b], y)] = BIT1 * amp
            m[2][self._disc(self.col_x[ins.dst], y)] = BIT1 * amp
            m[3][self._disc(self.x["op"], y)] = (BIT1 if ins.op == "SUB" else BIT0) * amp
            if self.expand and k == 0 and not st.parked:
                # the instruction at the line spans the columns it selects, so the
                # selector sits ON the data it gates rather than above it.  op is
                # already co-located with its cell, and op is the one copy that
                # has never failed.
                for ch, c in ((0, ins.a), (1, ins.b), (2, ins.dst)):
                    m[ch][self._column_span(c)] = BIT1 * amp
        return m

    def render(self, program, st):
        return self.data_screen(st), self.mask_channels(program, st)

    # ── the cycle ───────────────────────────────────────────────────────────

    def initial(self, regs):
        return V11State(regs, parked=True)

    def step_phase(self, program, st, phase):
        """Apply one phase; returns the new canonical state."""
        n = st.copy()
        ins = program[st.consumed] if st.consumed < len(program) else Instr("NOP", 0, 0, 0)
        if phase == "FETCH":
            if ins.op != "NOP":
                n.A, n.B, n.op = st.regs[ins.a], st.regs[ins.b], (1 if ins.op == "SUB" else -1)
        elif phase == "EXEC":
            if st.A is not None and st.B is not None:
                n.R = (st.A - st.B) & self.mask
        elif phase == "STORE":
            if st.R is not None and ins.op != "NOP":
                signed = st.R - (1 << self.word_bits) if st.R >> (self.word_bits - 1) else st.R
                if ins.op == "SUB" or signed <= 0:
                    n.regs[ins.dst] = st.R
        elif phase == "ADVANCE":
            n.A = n.B = n.R = n.op = None
            n.consumed = min(st.consumed + 1, len(program)) if not st.parked else st.consumed
            n.parked = False
        else:
            raise ValueError(phase)
        return n

    def cycle(self, program, regs):
        """The full canonical trajectory: [(state, phase, next_state), ...].

        The tape starts parked, so the first move is always an ADVANCE.
        """
        st, out = self.initial(regs), []
        seq = ["ADVANCE"] + [p for _ in range(len(program))
                             for p in ("FETCH", "EXEC", "STORE", "ADVANCE")][:-1]
        for phase in seq:
            nxt = self.step_phase(program, st, phase)
            out.append((st, phase, nxt))
            st = nxt
        return out

    # ── masks for the loss ──────────────────────────────────────────────────

    def _col_mask(self, x):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for b in range(self.word_bits):
            m[self._disc(x, self.row_y[b])] = 1.0
        return m

    def data_mask(self):
        m = np.zeros((self.H, self.W), dtype=np.float32)
        for c in range(self.n_regs):
            m += self._col_mask(self.col_x[c])
        for key in ("A", "B", "R"):
            m += self._col_mask(self.x[key])
        m[self._disc(self.x["op"], self.row_y[self.word_bits // 2])] = 1.0
        return np.clip(m, 0, 1)

    def cell_mask(self, program, st):
        """Every pixel that carries information: the data cells plus the cells
        of every band still on the tape.  The loss is normalised by this, not
        by the picture -- background is ~95% of the grid, so a picture-mean loss
        is minimised by painting nothing at all."""
        m = self.data_mask()
        for k, ins in enumerate(program[st.consumed:]):
            if ins.op == "NOP":
                continue
            y = self.band_y(k, st.parked)
            if y - self.r < 0:
                continue
            for x in (self.col_x[ins.a], self.col_x[ins.b], self.col_x[ins.dst], self.x["op"]):
                m[self._disc(x, y)] = 1.0
        return m

    def phase_mask(self, program, st, phase):
        """Where the phase is allowed to change things -- weighted higher in the
        loss than the rest, which must simply hold still."""
        m = np.zeros((self.H, self.W), dtype=np.float32)
        ins = program[st.consumed] if st.consumed < len(program) else Instr("NOP", 0, 0, 0)
        if phase == "FETCH":
            m += self._col_mask(self.x["A"]) + self._col_mask(self.x["B"])
            m[self._disc(self.x["op"], self.row_y[self.word_bits // 2])] = 1.0
        elif phase == "EXEC":
            m += self._col_mask(self.x["R"])
        elif phase == "STORE":
            for c in range(self.n_regs):
                m += self._col_mask(self.col_x[c])
        elif phase == "ADVANCE":
            for key in ("A", "B", "R"):
                m += self._col_mask(self.x[key])
            m[self._disc(self.x["op"], self.row_y[self.word_bits // 2])] = 1.0
        return np.clip(m, 0, 1)

    def travel(self, phase):
        """Longest distance information has to move in this phase, in pixels."""
        far = max(abs(self.x["B"] - c) for c in self.col_x)
        if phase == "FETCH":
            return far + self.gap
        if phase == "EXEC":
            return abs(self.x["R"] - self.x["A"])
        if phase == "STORE":
            return max(abs(self.x["R"] - c) for c in self.col_x)
        return self.pitch + self.gap

    def steps_for(self, phase, px_per_step=0.5, slack=1.25):
        return int(np.ceil(self.travel(phase) / px_per_step * slack))

    # ── decoding ────────────────────────────────────────────────────────────

    def _ring(self, screen, x, y):
        m = self._disc(x, y) & ~self._plus(x, y)
        return float(np.asarray(screen)[m].mean())

    def decode(self, screen, unknown_tol=0.25):
        """-> (regs, {A,B,R,op}); None where the cell reads as unset."""
        def word(x):
            bits, ok = [], True
            for b in range(self.word_bits):
                v = self._ring(screen, x, self.row_y[b])
                ok &= abs(v) >= unknown_tol
                bits.append(1 if v > 0 else 0)
            return bits_msb_to_int(bits) if ok else None
        regs = tuple(word(self.col_x[c]) for c in range(self.n_regs))
        slots = {k: word(self.x[k]) for k in ("A", "B", "R")}
        v = self._ring(screen, self.x["op"], self.row_y[self.word_bits // 2])
        slots["op"] = None if abs(v) < unknown_tol else (1 if v > 0 else -1)
        return regs, slots

    # ── display ─────────────────────────────────────────────────────────────

    COLORS = {"selA": (0.27, 0.45, 0.82), "selB": (0.09, 0.75, 0.81),
              "dst": (1.00, 0.62, 0.23), "op+": (0.69, 0.48, 0.63), "op-": (0.95, 0.10, 0.65)}

    @classmethod
    def to_rgb(cls, data, masks):
        g = np.clip((np.asarray(data) + 1) / 2, 0, 1)
        rgb = np.stack([g, g, g], -1)
        a, b, d, o = (np.clip(np.asarray(m), -1, 1) for m in masks)
        tint = np.zeros_like(rgb)
        w = np.zeros(rgb.shape[:2] + (1,), dtype=rgb.dtype)
        for m, key in ((np.clip(a, 0, 1), "selA"), (np.clip(b, 0, 1), "selB"),
                       (np.clip(d, 0, 1), "dst"), (np.clip(o, 0, 1), "op+"),
                       (np.clip(-o, 0, 1), "op-")):
            tint += m[..., None] * np.array(cls.COLORS[key], dtype=rgb.dtype)
            w = np.maximum(w, m[..., None])
        w = w * 0.6
        return (np.clip(rgb * (1 - w) + np.clip(tint, 0, 1) * w, 0, 1) * 255).astype(np.uint8)


_V11_CACHE = {}


def layout_v11_for(n_instr, **kw):
    key = (n_instr, tuple(sorted(kw.items())))
    if key not in _V11_CACHE:
        _V11_CACHE[key] = TileLayoutV11(n_instr=n_instr, **kw)
    return _V11_CACHE[key]
