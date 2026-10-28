"""Game of Life for the v15 tape machine with LOCAL operands (port of v12 ``life()``).

The tape machine only reaches a few registers per instruction (``gol_program`` reads
neighbours a whole board row away), so this follows v12's local compiler
(scripts/stamp_programs.py ``life``): every instruction touches registers within 2 of
each other.

* Every memory cell is a pair (C, T): C holds the board, T is scratch. Rows are
  n + 2 cells long: n board cells, a spare column the board drifts into, and a PAD
  holding a constant (one value on even rows, one on odd rows).
* Per board cell ONE accumulator walks the 3x3 neighbourhood a register at a time.
  Picking up a cell is one SUB, passing one costs two (zero the T slot, then copy).
  Cells are -1 (alive) / 0, so each pickup adds one. The cell itself is picked up
  ``s`` times and one or two pads add an offset:  X = N + s*self + delta.
* A short rule (found by exhaustive search, see ``RULES``) turns X into the new
  state with a constant K = +-1 in the neighbouring T slot. K is CARRIED along with
  the work, two instructions per register, flipping sign each step; each sign has
  its own rule and delta.
* No second buffer: cells go in raster order and the new value overwrites the
  top-left neighbour, which has no readers left. The board drifts one up-left; the
  next generation runs backwards and drifts it back.

v15 change: the program is a RING executed forever, so it is exactly two
generations followed by carrying K back to where it started (same register, same
sign). One pass of the ring = two generations.
"""

from dataclasses import dataclass

SUB, SUBLEQ = "SUB", "SUBLEQ"

# word bits -> {K: (s, delta, rule)}; rule over registers X, K, C (C <- new cell)
RULES = {
    4: {-1: (8, 5, [(SUB, "K", "X", "C"), (SUBLEQ, "C", "X", "X"), (SUBLEQ, "C", "C", "C"),
                    (SUBLEQ, "K", "X", "X"), (SUBLEQ, "K", "X", "C")]),
        +1: (8, 6, [(SUBLEQ, "K", "X", "X"), (SUB, "K", "X", "C"), (SUB, "K", "X", "X"),
                    (SUBLEQ, "X", "K", "X"), (SUB, "X", "C", "C")])},
    # the same rules with self on the sign bit (s = 32); nothing shorter exists at
    # s = 7 or 8 (the smallest s that separates the cases) up to 6 instructions
    6: {-1: (32, 29, [(SUB, "K", "X", "C"), (SUBLEQ, "C", "X", "X"), (SUB, "X", "X", "C"),
                      (SUBLEQ, "K", "X", "X"), (SUBLEQ, "K", "X", "C")]),
        +1: (32, 30, [(SUB, "X", "X", "C"), (SUBLEQ, "K", "X", "X"), (SUBLEQ, "C", "X", "C"),
                      (SUB, "C", "X", "X"), (SUBLEQ, "X", "K", "C")])},
}


def signed(v, wb):
    return v - (1 << wb) if v & (1 << (wb - 1)) else v


def apply(regs, ins, wb):
    op, a, b, d = ins
    r = (regs[a] - regs[b]) & ((1 << wb) - 1)
    if op == SUB or signed(r, wb) <= 0:
        regs[d] = r


@dataclass
class Life:
    tape: list          # (op, a, b, d) ring, two generations
    regs0: list         # start registers
    n: int              # board side
    word_bits: int
    gen_end: list       # instruction count at the end of generation 1 and 2
    cell_reg: list      # cell_reg[parity][y][x]: register of board cell after an even / odd gen

    @property
    def n_regs(self):
        return len(self.regs0)

    def board(self, regs, parity):
        return [[1 if regs[r] else 0 for r in row] for row in self.cell_reg[parity]]


def life(cells, n=5, word_bits=6):
    """Ring for an n x n board, dead outside; ``cells`` = alive (y, x)."""
    s_k, rules = {k: v[0] for k, v in RULES[word_bits].items()}, RULES[word_bits]
    assert s_k[-1] == s_k[+1], "both rules must use the same self count"
    S = s_k[-1]
    M = (1 << word_bits) - 1
    d1, d2 = rules[-1][1], rules[+1][1]          # K=-1 picks the even pad, K=+1 both
    pad_even, pad_odd = (-d1) & M, (-(d2 - d1)) & M
    NCOL, R0, NROW, OFF = n + 2, -1, n + 3, 3
    NM = OFF + NROW * NCOL + 3

    def idx(r, q):
        return OFF + (r - R0) * NCOL + q

    def rc(m):
        k = m - OFF
        return k // NCOL + R0, k % NCOL

    C = lambda m: 2 * m
    T = lambda m: 2 * m + 1
    regs0 = [0] * (2 * NM)
    for r in range(R0, R0 + NROW):
        regs0[C(idx(r, NCOL - 1))] = pad_even if r % 2 == 0 else pad_odd
    for y, x in cells:
        regs0[C(idx(1 + y, 1 + x))] = M
    tape = []
    emit = lambda op, a, b, d: tape.append((op, a, b, d))
    car = {"m": idx(0, 0) - 1, "s": -1}                       # K = -1 at T(m)
    start_m = car["m"]
    regs0[T(car["m"])] = M

    def zt(m):
        emit(SUBLEQ, T(m), T(m), T(m))

    def carry_to(m):
        while car["m"] != m:
            nx = car["m"] + (1 if m > car["m"] else -1)
            zt(nx)
            emit(SUB, T(nx), T(car["m"]), T(nx))
            car["m"], car["s"] = nx, -car["s"]

    def rule(Xr, Kr, Cr):
        reg = {"X": Xr, "K": Kr, "C": Cr}
        for op, a, b, d in rules[car["s"]][2]:
            emit(op, reg[a], reg[b], reg[d])

    gen_end, lo = [], 1
    for g in range(2):
        fwd = g == 0                                           # board at lo .. lo+n-1
        inb = lambda r, q: lo <= r < lo + n and lo <= q < lo + n
        order = [(y, x) for y in range(lo, lo + n) for x in range(lo, lo + n)]
        if not fwd:
            order.reverse()
        for y, x in order:
            if fwd:
                s, e = idx(y + 1, x + 1), idx(y - 1, x - 1)
                kpos, Xm, walk, start = e - 1, e, range(s, e, -1), s
            else:
                s, e = idx(y - 1, x - 1), idx(y + 1, x + 1)
                kpos, Xm, walk, start = e, e - 1, range(s, e), s - 1
            carry_to(kpos)
            pads = {r for r in (y, y - 1) if r % 2 == 0 or car["s"] > 0}
            nbrs = {(r, q) for r in (y - 1, y, y + 1) for q in (x - 1, x, x + 1)
                    if (r, q) != (y, x) and inb(r, q)}
            zt(start)
            prev = start
            for m in walk:
                land = m - 1 if fwd else m
                r, q = rc(m)
                if (r, q) in nbrs or (r, q) == (y, x) or (q == NCOL - 1 and r in pads):
                    emit(SUB, T(prev), C(m), T(land))
                else:
                    zt(land)
                    emit(SUB, T(prev), T(land), T(land))
                if (r, q) == (y, x):
                    for _ in range(S - 1):
                        emit(SUB, T(land), C(m), T(land))
                prev = land
            if rc(e) in nbrs:
                emit(SUB, T(Xm), C(e), T(Xm))
            rule(T(Xm), T(kpos), C(e))
        lo = lo - 1 if fwd else lo + 1
        if g == 1:
            carry_to(start_m)
        gen_end.append(len(tape))
    assert car["s"] == -1
    # trim unused registers at both ends
    used = sorted({r for ins in tape for r in ins[1:]} | {i for i, v in enumerate(regs0) if v})
    base, top = used[0], used[-1] + 1
    tape = [(op, a - base, b - base, d - base) for op, a, b, d in tape]
    cell_reg = [[[C(idx(lo0 + y, lo0 + x)) - base for x in range(n)] for y in range(n)]
                for lo0 in (1, 0)]
    out = Life(tape, regs0[base:top], n, word_bits, gen_end, cell_reg)
    span = max(max(ins[1:]) - min(ins[1:]) for ins in tape)
    assert span <= 2, span
    return out


def life_step(board):
    """Dead outside the board."""
    h, w = len(board), len(board[0])
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            k = sum(board[y + dy][x + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if (dy or dx) and 0 <= y + dy < h and 0 <= x + dx < w)
            row.append(1 if k == 3 or (k == 2 and board[y][x]) else 0)
        out.append(row)
    return out


def check(cells, n=5, word_bits=6, passes=20):
    """Run the ring in software against life_step; returns the number of generations right."""
    p = life(cells, n, word_bits)
    regs = list(p.regs0)
    b = [[1 if (y, x) in set(cells) else 0 for x in range(n)] for y in range(n)]
    assert p.board(regs, 0) == b
    for k in range(passes):
        for g in range(2):
            lo_i = 0 if g == 0 else p.gen_end[0]
            for ins in p.tape[lo_i:p.gen_end[g]]:
                apply(regs, ins, word_bits)
            b = life_step(b)
            if p.board(regs, 1 - g) != b:
                return 2 * k + g
    return 2 * passes
