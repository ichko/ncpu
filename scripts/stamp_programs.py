"""A library of programs for the v12 stamped machine.

The machine has two instructions on 4-bit registers:

    SUB    a b -> d     d = a - b                      (always writes)
    SUBLEQ a b -> d     d = a - b  if  a - b <= 0      (signed; otherwise d keeps its value)

There are no branches. A loop is unrolled and each pass is *predicated*: SUBLEQ
simply refuses to write once the loop's condition stops holding, so the tail of
an unrolled loop is a run of instructions that do nothing. That is how gcd and
mod below are written.

Locality is the one hard constraint. v12 was trained on a 4-register memory and
is translation invariant, so it runs on a memory of any width -- but only if the
three registers an instruction touches lie within a window of 4 (span <= 3).
Measured on a 24-register memory: span <= 2 gave 0 errors in 19,200
instructions, span 3 gave 5 (p = 2.6e-4), span 4 already 3.3%. ``Asm`` rejects
span > 3 and the programs keep span 3 to a handful of instructions.

Unused registers start at 0, which is what the ``z`` registers rely on.
"""

from dataclasses import dataclass, field

WORD_BITS = 4      # default; a program may use taller words (see ``sort``)
MAX_SPAN = 3


def signed(v, wb=WORD_BITS):
    return v - (1 << wb) if v & (1 << (wb - 1)) else v


def step(regs, op, a, b, d, wb=WORD_BITS):
    """Reference semantics (identical to ncpu.program.apply_op)."""
    r = (regs[a] - regs[b]) & ((1 << wb) - 1)
    out = list(regs)
    if op == "SUB" or signed(r, wb) <= 0:
        out[d] = r
    return out


@dataclass
class Program:
    id: str
    name: str
    desc: str
    labels: list
    regs0: list
    source: list
    inputs: list
    outputs: list
    signed: bool = False
    word_bits: int = WORD_BITS
    screen: dict = None       # how a small pixel screen reads this program's memory
    instrs: list = field(default_factory=list)

    @property
    def n_regs(self):
        return len(self.labels)

    def emit(self, op, a, b, d, note, src):
        ix = {n: i for i, n in enumerate(self.labels) if n}
        a, b, d = (ix[r] if isinstance(r, str) else r for r in (a, b, d))
        span = max(a, b, d) - min(a, b, d)
        if span > MAX_SPAN:
            raise ValueError(f"{self.id}: {op} {a} {b} -> {d} spans {span} registers")
        lab = self.labels
        self.instrs.append(dict(op=op, a=a, b=b, dst=d, note=note, src=src, span=span,
                                text=f"{op:<6} {lab[a]} {lab[b]} -> {lab[d]}"))

    def trace(self):
        regs, out = list(self.regs0), [list(self.regs0)]
        for i in self.instrs:
            regs = step(regs, i["op"], i["a"], i["b"], i["dst"], self.word_bits)
            out.append(regs)
        return out


def add_into(p, dst, src, t, line):
    """dst += src via subtraction: zero t, t = -src, dst -= t. Two of the three
    are genuine SUBLEQs -- their result is always <= 0 so they always fire."""
    p.emit("SUBLEQ", t, t, t, f"{t} = 0", line)
    p.emit("SUBLEQ", t, src, t, f"{t} = −{src}", line)
    p.emit("SUB", dst, t, dst, f"{dst} += {src}", line)


def fib(n_adds, target, word_bits=WORD_BITS):
    taller = "" if word_bits == WORD_BITS else (
        f" {target} does not fit in 4 bits, so the words here are {word_bits} bits tall — "
        f"v12 was only ever trained on 4.")
    p = Program(
        id=f"fib{n_adds + 2}", name=f"fib({n_adds + 2})",
        desc=f"Fibonacci by repeated addition; there is no ADD, so each += is three "
             f"subtractions through a scratch register. Leaves {target}.{taller}",
        labels=["a", "b", "t", "·"], regs0=[1, 1, 0, 0], word_bits=word_bits,
        source=["a, b = 1, 1"], inputs=[0, 1], outputs=[0])
    vals = [1, 1]
    for k in range(n_adds):
        dst, src = ("a", "b") if k % 2 == 0 else ("b", "a")
        vals.append(vals[-1] + vals[-2])
        p.source.append(f"{dst} += {src}          # {vals[-1]}")
        add_into(p, dst, src, "t", len(p.source) - 1)
    if k % 2 == 1:
        p.outputs = [1]
    return p


def gcd(a, b, rounds=7, word_bits=WORD_BITS):
    top = (1 << (word_bits - 1)) - 1
    p = Program(
        id=f"gcd{word_bits}", name=f"gcd({a}, {b})",
        desc=f"Euclid by subtraction. The loop is unrolled {rounds}× and predicated: SUBLEQ "
             f"only writes when its result is ≤ 0, so once a and b settle every remaining "
             f"pass does nothing. Correct for all inputs 1..{top}"
             + ("." if word_bits == WORD_BITS else f", on {word_bits}-bit words."),
        labels=["z", "a", "b"], regs0=[0, a, b], signed=True, word_bits=word_bits,
        source=[f"a, b = {a}, {b}",
                "a, b = −a, −b          # SUBLEQ tests ≤ 0",
                f"repeat {rounds}×:",
                "    if a ≥ b: a −= b",
                "    if b ≥ a: b −= a",
                "a = a + b              # one of them is 0"],
        inputs=[1, 2], outputs=[1])
    p.emit("SUB", "z", "a", "a", "a = −a", 1)
    p.emit("SUB", "z", "b", "b", "b = −b", 1)
    for _ in range(rounds):
        p.emit("SUBLEQ", "a", "b", "a", "if a ≥ b: a −= b", 3)
        p.emit("SUBLEQ", "b", "a", "b", "if b ≥ a: b −= a", 4)
    p.emit("SUB", "z", "a", "a", "a = −a", 5)
    p.emit("SUB", "a", "b", "a", "a += b", 5)
    return p


def mod(a, m, rounds=3):
    p = Program(
        id="mod", name=f"{a} mod {m}",
        desc="Remainder by predicated repeated subtraction: each pass is ONE "
             "instruction, which subtracts m only while that leaves a ≥ 0. "
             "Correct for a ≤ 7, m ≥ 2.",
        labels=["z", "a", "m"], regs0=[0, a, m], signed=True,
        source=[f"a, m = {a}, {m}",
                "a, m = −a, −m",
                f"repeat {rounds}×:",
                "    if a ≥ m: a −= m",
                "a = −a"],
        inputs=[1, 2], outputs=[1])
    p.emit("SUB", "z", "a", "a", "a = −a", 1)
    p.emit("SUB", "z", "m", "m", "m = −m", 1)
    for _ in range(rounds):
        p.emit("SUBLEQ", "a", "m", "a", "if a ≥ m: a −= m", 3)
    p.emit("SUB", "z", "a", "a", "a = −a", 4)
    return p


def max3(a, b, c):
    p = Program(
        id="max3", name=f"max({a}, {b}, {c})",
        desc="max(x, y) = x − min(x − y, 0), and min(·, 0) is exactly one SUBLEQ into "
             "a register holding 0. Two of those.",
        labels=["b", "a", "d", "c"], regs0=[b, a, 0, c],
        source=[f"a, b, c = {a}, {b}, {c}",
                "a = max(a, b)      # d = min(a − b, 0); a −= d",
                "a = max(a, c)"],
        inputs=[1, 0, 3], outputs=[1])
    p.emit("SUBLEQ", "a", "b", "d", "d = min(a − b, 0)", 1)
    p.emit("SUB", "a", "d", "a", "a = max(a, b)", 1)
    p.emit("SUBLEQ", "d", "d", "d", "d = 0", 2)
    p.emit("SUBLEQ", "a", "c", "d", "d = min(a − c, 0)", 2)
    p.emit("SUB", "a", "d", "a", "a = max(a, c)", 2)
    return p


def absdiff(a, b):
    p = Program(
        id="absdiff", name=f"|{a} − {b}|",
        desc="Absolute difference. n starts as −t; SUBLEQ overwrites it with t only when "
             "t ≤ 0, which leaves −|t| either way.",
        labels=["a", "b", "t", "n", "z"], regs0=[a, b, 0, 0, 0],
        source=[f"a, b = {a}, {b}",
                "t = a − b",
                "n = −t",
                "if t ≤ 0: n = t        # n = −|t|",
                "t = −n"],
        inputs=[0, 1], outputs=[2])
    p.emit("SUB", "a", "b", "t", "t = a − b", 1)
    p.emit("SUB", "n", "t", "n", "n = −t", 2)
    p.emit("SUBLEQ", "t", "z", "n", "if t ≤ 0: n = t", 3)
    p.emit("SUB", "z", "n", "t", "t = −n", 4)
    return p


def sort(vals, word_bits=WORD_BITS, binary=False):
    """Odd-even transposition sort, descending. Values sit in pairs with two
    scratch registers after each pair, so every compare-and-swap is local:

        v0 v1 d e | v2 v3 d e | v4 v5 d e ...
    """
    n = len(vals)
    assert n % 2 == 0
    labels, regs0, pos = [], [], []
    for k in range(0, n, 2):
        pos += [len(labels), len(labels) + 1]
        labels += [f"v{k}", f"v{k + 1}", "d", "e"]
        regs0 += [vals[k], vals[k + 1], 0, 0]
    shown = "[" + ", ".join(f"0b{v:b}" for v in vals) + "]" if binary else str(vals)
    taller = "" if word_bits == WORD_BITS else (
        f" The words are {word_bits} bits tall — the comparison is signed, so 0b1111 would "
        f"read as −1 in 4 bits — and v12 was only ever trained on 4.")
    p = Program(
        id=f"sort{n}" + ("b" if binary else ""),
        name=f"sort {shown.replace('0b', '')}" if binary else f"sort {n} numbers",
        desc=f"Odd–even transposition sort on {len(labels)} registers, {len(labels) / 4:g}× the "
             f"memory v12 was trained on. It still works because the network is translation "
             f"invariant and every compare-and-swap only touches neighbouring registers.{taller}",
        labels=labels, regs0=regs0, word_bits=word_bits,
        source=[f"v = {shown}          # sort descending"],
        inputs=pos, outputs=pos)

    def cswap(i, j, line):
        pi, pj = pos[i], pos[j]
        d = pos[i] + 2 if i % 2 == 0 else pos[i] + 1
        e = d + 1
        p.emit("SUBLEQ", pi, pj, d, f"d = min(v{i} − v{j}, 0)", line)
        p.emit("SUB", e, d, e, "e = −d", line)
        p.emit("SUB", pi, d, pi, f"v{i} = max", line)
        p.emit("SUB", pj, e, pj, f"v{j} = min", line)
        p.emit("SUBLEQ", d, d, d, "d = 0", line)
        p.emit("SUBLEQ", e, e, e, "e = 0", line)

    for rnd in range(n):
        pairs = [(i, i + 1) for i in range(rnd % 2, n - 1, 2)]
        p.source.append(f"round {rnd + 1}: " + "  ".join(f"cswap(v{i}, v{j})" for i, j in pairs))
        for i, j in pairs:
            cswap(i, j, len(p.source) - 1)
    p.source.append("# cswap(p, q): d = min(p − q, 0); p −= d; q += d")
    return p


def ladder(vals=(15, 1, 7, 3), word_bits=5):
    """Sort four numbers that are bit patterns 1, 11, 111, 1111 into a staircase.

    The four values sit in four ADJACENT registers so the finished memory reads as
    a ladder; the scratch pairs live at the two ends instead of between them:

        d e | v0 v1 v2 v3 | f g

    Ascending, so the ladder climbs to the right. 15 is -1 as a signed 4-bit word,
    so the words are 5 bits tall."""
    labels = ["d", "e", "v0", "v1", "v2", "v3", "f", "g"]
    regs0 = [0, 0, *vals, 0, 0]
    shown = ", ".join(f"{v:b}" for v in vals)
    p = Program(
        id="ladder", name="ladder",
        desc="Sort the bit patterns 1, 11, 111, 1111 so the memory itself becomes a staircase. "
             "The four values are kept in adjacent registers with the scratch pair at either "
             "end. 1111 does not fit a signed 4-bit comparison, so the words are 5 bits tall — "
             "v12 was only ever trained on 4.",
        labels=labels, regs0=regs0, word_bits=word_bits,
        source=[f"v = [{shown}]          # binary", "sort ascending:"],
        inputs=[2, 3, 4, 5], outputs=[2, 3, 4, 5])

    def cswap(i, j, X, Y, line):
        """min -> register i, max -> register j."""
        a, b = labels[i], labels[j]
        p.emit("SUBLEQ", j, i, X, f"{labels[X]} = min({b} − {a}, 0)", line)
        p.emit("SUB", Y, X, Y, f"{labels[Y]} = −{labels[X]}", line)
        p.emit("SUB", j, X, j, f"{b} = max", line)
        p.emit("SUB", i, Y, i, f"{a} = min", line)
        p.emit("SUBLEQ", X, X, X, f"{labels[X]} = 0", line)
        p.emit("SUBLEQ", Y, Y, Y, f"{labels[Y]} = 0", line)

    for rnd in range(4):
        if rnd % 2 == 0:
            p.source.append(f"round {rnd + 1}: cswap(v0, v1)  cswap(v2, v3)")
            cswap(2, 3, 1, 0, len(p.source) - 1)
            cswap(4, 5, 6, 7, len(p.source) - 1)
        else:
            p.source.append(f"round {rnd + 1}: cswap(v1, v2)")
            cswap(3, 4, 1, 0, len(p.source) - 1)
    p.source.append("# cswap(a, b): x = min(b − a, 0); b −= x; a += x")
    return p


def walksum(vals, word_bits=5):
    """A walking accumulator: one register that moves one step right per
    instruction, subtracting each value it passes. The same primitive Game of
    Life needs to carry a neighbour count between rows."""
    n = len(vals)
    labels, regs0 = [], []
    for k, v in enumerate(vals):
        labels += [f"v{k}", "t"]
        regs0 += [v, 0]
    labels += ["z", "sum"]
    regs0 += [0, 0]
    p = Program(
        id="walksum", name=f"sum of {n} numbers",
        desc="A walking accumulator: one running total that hops a register to the right "
             "per instruction, picking up each value as it passes. Its trail stays behind "
             "as prefix sums. This is the primitive that carries information along memory.",
        labels=labels, regs0=regs0, word_bits=word_bits,
        source=[f"v = {vals}", "acc = 0",
                "for each v: acc −= v       # walks right, one register per step",
                "sum = −acc"],
        inputs=list(range(0, 2 * n, 2)), outputs=[2 * n + 1])
    for k in range(n):
        src = 2 * k - 1 if k else 1
        p.emit("SUB", src, 2 * k, 2 * k + 1, f"acc −= v{k}", 2)
    p.emit("SUB", 2 * n, 2 * n - 1, 2 * n + 1, "sum = −acc", 3)
    return p


def ca1d(rule, n=16, gens=8):
    """A 1D cellular automaton on a line of n cells, dead outside.

    alive = 0b1000. Then 8 − 8 ≡ 0 and 0 − 8 ≡ 8 (mod 16): with the alive bit at
    the top of the word, SUB *is* exclusive-or, so rule 90 (left ⊕ right) is one
    instruction per cell and rule 150 (left ⊕ self ⊕ right) two.

    There is no second buffer. Generation g sweeps left to right and writes each
    new cell into its LEFT neighbour's register, which nothing needs any more;
    the next generation sweeps right to left and writes rightwards. The board
    drifts one register and back, and every instruction spans 2 registers.
    """
    ALIVE = 8
    M = n + 3                                   # slot 0 and n+2 are permanent zeros
    board = lambda g: list(range(2, n + 2)) if g % 2 == 0 else list(range(1, n + 1))
    regs0 = [0] * M
    regs0[board(0)[n // 2]] = ALIVE
    nm = {90: "left ⊕ right", 150: "left ⊕ self ⊕ right"}[rule]
    p = Program(
        id=f"rule{rule}", name=f"rule {rule}",
        desc=f"The 1D cellular automaton rule {rule} ({nm}) on {n} cells, from one live cell. "
             f"Alive is 0b1000, which makes subtraction exclusive-or: "
             f"{'one instruction' if rule == 90 else 'two instructions'} per cell per generation. "
             f"The screen stacks the generations.",
        labels=[f"r{k}" for k in range(M)], regs0=regs0,
        source=[f"cells = {n}, alive = 0b1000, one seed",
                f"c'[k] = {nm.replace('left', 'c[k−1]').replace('right', 'c[k+1]').replace('self', 'c[k]')}",
                "# ⊕ is a single SUB: 8 ± 8 ≡ 0 (mod 16)"],
        inputs=[board(0)[n // 2]], outputs=board(gens))
    rows = [{"at": -1, "regs": board(0)}]
    for g in range(gens):
        p.source.append(f"generation {g + 1}")
        line = len(p.source) - 1
        if g % 2 == 0:                          # sweep right, write left
            if g:
                p.emit("SUBLEQ", 1, 1, 1, "clear the left edge", line)
            for k, s_ in enumerate(board(g)):
                if rule == 90:
                    p.emit("SUB", s_ - 1, s_ + 1, s_ - 1, f"c{k}' = c{k - 1} ⊕ c{k + 1}", line)
                else:
                    p.emit("SUB", s_ - 1, s_, s_ - 1, f"c{k}' = c{k - 1} ⊕ c{k}", line)
                    p.emit("SUB", s_ - 1, s_ + 1, s_ - 1, f"c{k}' ⊕= c{k + 1}", line)
        else:                                   # sweep left, write right
            p.emit("SUBLEQ", n + 1, n + 1, n + 1, "clear the right edge", line)
            for k, s_ in reversed(list(enumerate(board(g)))):
                if rule == 90:
                    p.emit("SUB", s_ + 1, s_ - 1, s_ + 1, f"c{k}' = c{k + 1} ⊕ c{k - 1}", line)
                else:
                    p.emit("SUB", s_ + 1, s_, s_ + 1, f"c{k}' = c{k + 1} ⊕ c{k}", line)
                    p.emit("SUB", s_ + 1, s_ - 1, s_ + 1, f"c{k}' ⊕= c{k - 1}", line)
        rows.append({"at": len(p.instrs) - 1, "regs": board(g + 1)})
    p.screen = {"kind": "spacetime", "w": n, "h": gens + 1, "alive": ALIVE, "rows": rows}
    return p


def library():
    return [fib(3, 5), fib(6, 21, word_bits=5), gcd(6, 4), gcd(15, 6, rounds=15, word_bits=5),
            mod(7, 3), max3(3, 6, 5), absdiff(3, 7),
            ladder(), sort([2, 6, 4, 7, 1, 5]),
            sort([9, 2, 14, 5, 11, 0, 7, 12], word_bits=5),
            walksum([3, 1, 4, 1, 5, 2, 6, 3]), ca1d(90), ca1d(150), life(GLIDER, 8)]


# ── self-check: every program, and the generic ones over their whole domain ──

def _check():
    import itertools

    def run(p):
        return p.trace()[-1]

    for p in library():
        t = p.trace()
        print(f"{p.id:<8} {p.n_regs:>2} regs {len(p.instrs):>3} instr  "
              f"max span {max(i['span'] for i in p.instrs)}  "
              f"span-3: {sum(i['span'] == 3 for i in p.instrs):>2}   "
              f"{p.regs0} -> {t[-1]}")

    from math import gcd as g
    for a, b in itertools.product(range(1, 8), repeat=2):
        assert run(gcd(a, b))[1] == g(a, b), (a, b)
    for a, m in itertools.product(range(0, 8), range(2, 8)):
        assert run(mod(a, m))[1] == a % m, (a, m)
    for a, b, c in itertools.product(range(8), repeat=3):
        assert run(max3(a, b, c))[1] == max(a, b, c)
    for a, b in itertools.product(range(8), repeat=2):
        assert run(absdiff(a, b))[2] == abs(a - b)
    for vals in itertools.product(range(8), repeat=4):
        p = sort(list(vals))
        assert [run(p)[i] for i in p.outputs] == sorted(vals, reverse=True), vals
    for vals in itertools.product(range(16), repeat=4):
        p = sort(list(vals), word_bits=5)
        assert [run(p)[i] for i in p.outputs] == sorted(vals, reverse=True), vals
    for a, b in itertools.product(range(1, 16), repeat=2):
        assert run(gcd(a, b, rounds=15, word_bits=5))[1] == g(a, b), (a, b)
    assert run(fib(6, 21, word_bits=5))[1] == 21
    for vals in itertools.permutations([15, 1, 7, 3]):
        assert run(ladder(vals))[2:6] == [1, 3, 7, 15], vals
    for vals in itertools.product(range(16), repeat=4):
        assert run(ladder(vals))[2:6] == sorted(vals), vals
    import random
    rng = random.Random(0)
    for _ in range(300):
        vals = [rng.randrange(4) for _ in range(8)]
        assert run(walksum(vals))[-1] == sum(vals)
    for rule in (90, 150):                       # against a direct simulation
        p = ca1d(rule)
        tr = p.trace()
        cells = [0] * 16
        cells[8] = 1
        for row in p.screen["rows"][1:]:
            nxt = []
            for k in range(16):
                l = cells[k - 1] if k else 0
                r = cells[k + 1] if k < 15 else 0
                nxt.append(l ^ r if rule == 90 else l ^ cells[k] ^ r)
            cells = nxt
            got = [int(tr[row["at"] + 1][i] == 8) for i in row["regs"]]
            assert got == cells, (rule, got, cells)
    import numpy as np

    def gol(b):
        q = np.pad(b, 1)
        k = sum(q[1 + dy:6 + dy, 1 + dx:6 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        return ((k == 3) | ((b == 1) & (k == 2))).astype(int)

    for t in range(60):                          # Life against a direct simulation
        cells = sp_cells = [(y, x) for y in range(5) for x in range(5) if rng.random() < 0.4]
        if t == 0:
            cells = GLIDER
        gens = 8 if t == 0 else 2
        lp = life(cells, gens)
        tr = lp.trace()
        b = np.zeros((5, 5), int)
        for y, x in cells:
            b[y, x] = 1
        for gg in range(1, gens + 1):
            b = gol(b)
            got = [[int(tr[lp.screen["cells"][y][x][gg]["at"] + 1][lp.screen["cells"][y][x][gg]["reg"]] == 15)
                    for x in range(5)] for y in range(5)]
            assert (np.array(got) == b).all(), (t, gg)
    print("all programs correct over their full input domains")



# ── Conway's Game of Life ───────────────────────────────────────────────────

GLIDER = [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)]


def life(cells=GLIDER, gens=8, n=5):
    """Game of Life on an n x n board (dead outside) for the stamped machine.

    The memory is one long line, so a cell's neighbours above and below sit a
    whole row away -- and an instruction may only touch registers within 2 of
    each other. Everything below follows from that:

    * Every memory cell is a pair (C, T): C holds the board, T is scratch. Rows
      are 7 cells long: up to 5 board cells, a spare column the board drifts
      into, and a PAD holding a constant (-5 on even rows, -1 on odd ones).

    * Per board cell, ONE accumulator walks through its 3x3 neighbourhood, a
      register at a time (each step moves 2 slots, the most an instruction can
      span). Picking up a cell is one SUB; passing one costs two (the T slot it
      lands on must be zeroed first). Cells are stored as -1 (alive) / 0, so
      each pickup ADDS one. The cell itself is picked up 8 times, and one or
      two pads on the way add an offset:  X = N + 8*self + delta  (mod 16).

    * A 5-instruction rule, found by exhaustive search, turns X into the next
      state. It needs a constant K = +-1 in a neighbouring register, and there
      are no constants near a cell -- so K is CARRIED: it follows the work one
      register at a time, two instructions per step, flipping sign each step.
      Both signs have a rule (delta 5 for K = -1, 6 for K = +1).

    * No second buffer. Cells are processed in raster order and the walk runs
      from the bottom-right neighbour to the top-left one; by then that
      top-left cell has no readers left, so the new value goes into ITS
      register. The board drifts one row and one column up-left; the next
      generation runs in reverse order and drifts it back.
    """
    NCOL, R0, NROW, OFF = 7, -1, n + 3, 3
    M = OFF + NROW * NCOL + 3

    def idx(r, q):
        return OFF + (r - R0) * NCOL + q

    def rc(m):
        k = m - OFF
        return k // NCOL + R0, k % NCOL

    C = lambda m: 2 * m
    T = lambda m: 2 * m + 1
    ALIVE = 15
    labels = []
    for m in range(M):
        if OFF <= m < OFF + NROW * NCOL:
            r, q = rc(m)
            labels += [f"p{r + 1}" if q == NCOL - 1 else f"c{r + 1}{q}", f"t{r + 1}{q}"]
        else:
            labels += ["·", "·"]
    regs0 = [0] * (2 * M)
    for r in range(R0, R0 + NROW):
        regs0[C(idx(r, NCOL - 1))] = 11 if r % 2 == 0 else 15        # pads: -5 / -1
    for y, x in cells:
        regs0[C(idx(1 + y, 1 + x))] = ALIVE

    p = Program(
        id="life", name="game of life",
        desc=f"Conway's Game of Life on a {n}×{n} board, {gens} generations, starting from a "
             f"glider. One accumulator walks each cell's neighbourhood a register at a time, a "
             f"5-instruction rule found by exhaustive search decides life or death, and the new "
             f"value overwrites a neighbour nobody needs any more. The screen reads the board "
             f"straight out of memory.",
        labels=labels, regs0=regs0,
        source=[f"board {n}×{n}, alive = −1, dead = 0        # a glider",
                f"repeat {gens} generations:",
                "  for each cell, in raster order:",
                "    walk its 3×3:  X = N + 8·self + δ     # one register per step",
                "    rule(X, K)  →  write the dead cell up-left",
                "    carry the constant K one register on",
                "  next generation runs backwards and drifts the board back"],
        inputs=[C(idx(1 + y, 1 + x)) for y, x in cells], outputs=[])
    LW, LR, LK, LG = 3, 4, 5, 6

    carrier = {"m": idx(0, 0) - 1, "s": -1}                      # K = -1 at T(m)
    regs0[T(carrier["m"])] = 15

    def zt(m, line, note="clear"):
        p.emit("SUBLEQ", T(m), T(m), T(m), note, line)

    def carry_to(m):
        while carrier["m"] != m:
            nx = carrier["m"] + (1 if m > carrier["m"] else -1)
            zt(nx, LK)
            p.emit("SUB", T(nx), T(carrier["m"]), T(nx), "carry K", LK)
            carrier["m"], carrier["s"] = nx, -carrier["s"]

    def rule(Xs, Ks, Cs, line):
        if carrier["s"] < 0:                                     # K = -1, delta 5
            p.emit("SUB", Ks, Xs, Cs, "rule", line)
            p.emit("SUBLEQ", Cs, Xs, Xs, "rule", line)
            p.emit("SUBLEQ", Cs, Cs, Cs, "rule", line)
            p.emit("SUBLEQ", Ks, Xs, Xs, "rule", line)
            p.emit("SUBLEQ", Ks, Xs, Cs, "rule → new cell", line)
        else:                                                    # K = +1, delta 6
            p.emit("SUBLEQ", Ks, Xs, Xs, "rule", line)
            p.emit("SUB", Ks, Xs, Cs, "rule", line)
            p.emit("SUB", Ks, Xs, Xs, "rule", line)
            p.emit("SUBLEQ", Xs, Ks, Xs, "rule", line)
            p.emit("SUB", Xs, Cs, Cs, "rule → new cell", line)

    frames = [[{"at": -1, "reg": C(idx(1 + y, 1 + x))} for x in range(n)] for y in range(n)]
    lo = 1
    for g in range(gens):
        fwd = g % 2 == 0                                         # board at lo..lo+n-1
        inb = lambda r, q: lo <= r < lo + n and lo <= q < lo + n
        order = [(y, x) for y in range(lo, lo + n) for x in range(lo, lo + n)]
        if not fwd:
            order.reverse()
        for (y, x) in order:
            if fwd:
                s, e, kpos = idx(y + 1, x + 1), idx(y - 1, x - 1), idx(y - 1, x - 1) - 1
            else:
                s, e, kpos = idx(y - 1, x - 1), idx(y + 1, x + 1), idx(y + 1, x + 1)
            carry_to(kpos)
            delta = 5 if carrier["s"] < 0 else 6
            pads = {r for r in (y, y - 1) if (r % 2 == 0 or delta == 6)}
            nbrs = {(r, q) for r in (y - 1, y, y + 1) for q in (x - 1, x, x + 1)
                    if (r, q) != (y, x) and inb(r, q)}

            def wanted(m):
                r, q = rc(m)
                return (r, q) in nbrs or (r, q) == (y, x) or (q == NCOL - 1 and r in pads)

            step_ = -1 if fwd else 1
            zt(s - step_ * 0 if fwd else s - 1, LW, "start walk")
            prev = s if fwd else s - 1
            stop = e if fwd else e                               # walk lands on T(e) (fwd) / T(e-1) (bwd)
            ms = range(s, e, -1) if fwd else range(s, e)
            for m in ms:
                land = m - 1 if fwd else m
                if wanted(m):
                    rr, qq = rc(m)
                    tag = "self" if (rr, qq) == (y, x) else ("pad" if qq == NCOL - 1 else "nbr")
                    p.emit("SUB", T(prev), C(m), T(land), f"pick {tag}", LW)
                else:
                    zt(land, LW, "clear")
                    p.emit("SUB", T(prev), T(land), T(land), "step", LW)
                if rc(m) == (y, x):
                    for _ in range(7):
                        p.emit("SUB", T(land), C(m), T(land), "self again", LW)
                prev = land
            Xm = e if fwd else e - 1
            if rc(e) in nbrs:
                p.emit("SUB", T(Xm), C(e), T(Xm), "pick nbr", LW)
            rule(T(Xm), T(kpos), C(e), LR)
            frames[y - lo][x - lo] = None                        # placeholder, filled below
            p._last_write = (y - lo, x - lo, len(p.instrs) - 1, C(e))
            gi = len(p.screen_frames) if hasattr(p, "screen_frames") else None
            if not hasattr(p, "cellwrites"):
                p.cellwrites = []
            p.cellwrites.append((g + 1, y - lo, x - lo, len(p.instrs) - 1, C(e)))
        lo = lo - 1 if fwd else lo + 1

    # screen: for every cell, where its value lives in each generation
    cellsched = [[[{"at": -1, "reg": C(idx(1 + y, 1 + x))}] for x in range(n)] for y in range(n)]
    for (g, y, x, at, reg) in p.cellwrites:
        cellsched[y][x].append({"at": at, "reg": reg})
    p.screen = {"kind": "grid", "w": n, "h": n, "alive": ALIVE, "gens": gens, "cells": cellsched}
    final_lo = lo
    p.outputs = [C(idx(final_lo + y, final_lo + x)) for y in range(n) for x in range(n)]
    del p.cellwrites
    return p


if __name__ == "__main__":
    _check()
