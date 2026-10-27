"""Conway's Game of Life compiled to the v15 tape machine's SUB / SUBLEQ.

The machine has no jumps and no bit operations: the program is a ring of
``(op, a, b, d)`` instructions, ``d <- a - b`` (SUBLEQ: only if ``a - b <= 0`` as a
signed word), executed forever. One pass of the ring is two generations.

Memory map (registers):
    0 .. N-1         board A: cell (y, x) at y * w + x; alive = -1, dead = 0
    N .. 2N-1        board B (generation buffer; the ring alternates A -> B -> A)
    2N .. 2N+3       constants 0, -1, 5, 8
    2N+4 ..          temps

The screen is board A (or B): read register (y, x) as a pixel.

Per cell: n = sum of the 8 neighbours (each -c is +1 when alive), t = 2n + alive,
alive next iff 5 <= t <= 7 (n == 3, or n == 2 and alive). The interval test uses
min(v, 0) = SUBLEQ into a zeroed register:
    -clamp(v + 1, 0, 1) = min(-(min(v, 0) + 1), 0)
    new = -clamp(t - 4, 0, 1) - (-clamp(t - 7, 0, 1))
t reaches 17, so words need 6 bits.
"""

SUB, SUBLEQ = "SUB", "SUBLEQ"


def layout(h, w):
    N = h * w
    r = dict(A=0, B=N, Z=2 * N, M1=2 * N + 1, K5=2 * N + 2, K8=2 * N + 3)
    t = 2 * N + 4
    for k in ("acc", "neg", "tt", "a", "b", "e1", "e2"):
        r[k] = t
        t += 1
    r["n_regs"] = t
    return r


def _clamp_neg(r, out, a):
    """out <- -clamp(a + 1, 0, 1) for the value in register a (uses b)."""
    Z, M1, b = r["Z"], r["M1"], r["b"]
    return [(SUB, Z, Z, b),            # b = 0
            (SUBLEQ, a, Z, b),         # b = min(a, 0)
            (SUB, b, M1, b),           # b = min(a, 0) + 1 = min(a + 1, 1)
            (SUB, Z, Z, out),          # out = 0
            (SUBLEQ, Z, b, out)]       # out = min(-b, 0) = -clamp(a + 1, 0, 1)


def generation(h, w, src, dst, r):
    """Instructions computing board dst from board src."""
    Z, acc, neg, tt, a = r["Z"], r["acc"], r["neg"], r["tt"], r["a"]
    out = []
    for y in range(h):
        for x in range(w):
            nb = [src + ((y + dy) % h) * w + (x + dx) % w
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
            out.append((SUB, Z, nb[0], acc))                 # acc = n (cells are -1)
            out += [(SUB, acc, c, acc) for c in nb[1:]]
            out.append((SUB, Z, acc, neg))                   # neg = -n
            out.append((SUB, acc, neg, tt))                  # tt = 2n
            out.append((SUB, tt, src + y * w + x, tt))       # tt = 2n + alive
            out.append((SUB, tt, r["K5"], a))                # a = t - 5
            out += _clamp_neg(r, r["e1"], a)                 # e1 = -clamp(t - 4, 0, 1)
            out.append((SUB, tt, r["K8"], a))                # a = t - 8
            out += _clamp_neg(r, r["e2"], a)                 # e2 = -clamp(t - 7, 0, 1)
            out.append((SUB, r["e1"], r["e2"], dst + y * w + x))
    return out


def program(h, w):
    """The ring: A -> B, then B -> A. Returns (instructions, layout)."""
    r = layout(h, w)
    return generation(h, w, r["A"], r["B"], r) + generation(h, w, r["B"], r["A"], r), r


def initial_regs(board, word_bits, r):
    """Registers for a board (list of rows of 0/1): board A set, constants loaded."""
    mask = (1 << word_bits) - 1
    regs = [0] * r["n_regs"]
    w = len(board[0])
    for y, row in enumerate(board):
        for x, v in enumerate(row):
            regs[r["A"] + y * w + x] = (-v) & mask
    regs[r["M1"]], regs[r["K5"]], regs[r["K8"]] = mask, 5, 8
    return regs


def life_step(board):
    h, w = len(board), len(board[0])
    out = []
    for y in range(h):
        row = []
        for x in range(w):
            n = sum(board[(y + dy) % h][(x + dx) % w] for dy in (-1, 0, 1) for dx in (-1, 0, 1)
                    if dy or dx)
            row.append(1 if n == 3 or (n == 2 and board[y][x]) else 0)
        out.append(row)
    return out


def read_board(regs, base, h, w):
    return [[1 if regs[base + y * w + x] else 0 for x in range(w)] for y in range(h)]
