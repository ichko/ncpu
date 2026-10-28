"""ncc -- a C-like language compiled to the v15 tape machine (SUB / SUBLEQ ring, local operands).

The machine: registers of 6 bits, ``SUB a b d`` (d = a - b) and ``SUBLEQ a b d`` (d = a - b
only if that is <= 0, signed). The program is a RING executed forever, no jumps, and
every instruction must touch registers within 3 consecutive ones (the NCA's reach).

Language (one function, global ints):

    int a = 12, b = 18;            // ints are -16 .. 15 (5 bits; the 6th is headroom)
    while (a != b) {
        if (a > b) a = a - b; else b = b - a;
    }

    statements  int decl, x = e, x += e, x -= e, x++, x--, if/else, while, for, {...}, halt;
    expressions + - unary -, * and / by constants are folded, ( )
    conditions  == != < <= > >=, && || ! (short circuit), a bare expression means != 0

The result is read out of the variables' registers once the machine halts.

How it runs without jumps:
* Control flow is a ONE-HOT program counter: every basic block b owns a flag register
  F_b = 0 (active) / -16 (idle). The ring holds every block once, in order; each pass of
  the ring executes the active block (and, since a later block sees a flag set earlier in
  the same pass, falls straight through to its successor).
* A block first turns its flag into a guard G = 2 F_b (0 = on, 32 = off) and then does
  every write through a 6-instruction guarded write found by exhaustive search:
  V <- w if G == 0, else V unchanged (exact for w, V in -16 .. 15). An idle block runs
  all its instructions and changes nothing.
* Memory is pairs (V, T): V lane = variables / constants / flags, T lane = scratch.
  Values travel along the T lane two instructions per register (zero the next T, subtract:
  the value arrives negated, so the compiler tracks signs). An accumulator walks the
  terms of x = sum(+-v) and the guard G walks with it.
* Branches: t = lhs - rhs, a 4-5 instruction comparison (searched) turns t into
  q = 0 (true) / -16 (false), then F_true <- q and F_false <- -16 - F_true, both guarded.
  The machine halts when every flag is idle: the memory stops changing.
"""

import math
import re
from dataclasses import dataclass, field

SUB, SUBLEQ = "SUB", "SUBLEQ"
WB = 6
MASK = (1 << WB) - 1
LO, HI = -16, 15                 # int range
IDLE = -16

# V <- A if B == 0 (B = 32: unchanged); destroys A and B. Exact for A, V in -16 .. 15.
CMOV = [(SUB, "V", "A", "A"), (SUBLEQ, "A", "B", "A"), (SUB, "A", "V", "V"),
        (SUBLEQ, "B", "A", "A"), (SUBLEQ, "B", "A", "B"), (SUB, "B", "V", "V")]
# T = t in -31 .. 31, C = -16 (read-only), S scratch  ->  T = 0 if rel(t) else -16
CMP = {"lt": [(SUB, "T", "C", "S"), (SUB, "T", "C", "T"), (SUBLEQ, "T", "C", "T"), (SUB, "S", "T", "T")],
       "le": [(SUB, "T", "C", "S"), (SUBLEQ, "C", "S", "T"), (SUBLEQ, "C", "S", "S"), (SUB, "T", "S", "T")],
       "eq": [(SUB, "T", "C", "S"), (SUB, "T", "S", "T"), (SUBLEQ, "S", "T", "S"),
              (SUBLEQ, "S", "T", "S"), (SUBLEQ, "S", "T", "T")]}


class CompileError(Exception):
    pass


# ── parser ──────────────────────────────────────────────────────────────────

TOK = re.compile(r"\s*(?:(//[^\n]*|/\*.*?\*/)|(\d+)|([A-Za-z_]\w*)|"
                 r"(\+\+|--|\+=|-=|==|!=|<=|>=|&&|\|\||[-+*/%(){};,=<>!\[\]]))", re.S)
KEYWORDS = {"int", "if", "else", "while", "for", "halt"}


def lex(src):
    out, i = [], 0
    src = src.rstrip()
    while i < len(src):
        m = TOK.match(src, i)
        if not m:
            raise CompileError(f"bad character at {src[i:i + 20]!r}")
        i = m.end()
        if m.group(1):
            continue
        if m.group(2):
            out.append(("num", int(m.group(2))))
        elif m.group(3):
            out.append(("kw" if m.group(3) in KEYWORDS else "id", m.group(3)))
        else:
            out.append(("op", m.group(4)))
    out.append(("eof", None))
    return out


class Parser:
    """AST: statements are tuples; expressions are ('num', k) | ('var', x) | ('neg', e) |
    ('+'|'-'|'*'|'/', e1, e2); conditions are ('rel', op, e1, e2) | ('and'|'or', c1, c2) |
    ('not', c)."""

    def __init__(self, src):
        self.t, self.i = lex(src), 0

    def peek(self, k=0):
        return self.t[self.i + k]

    def next(self):
        tok = self.t[self.i]
        self.i += 1
        return tok

    def expect(self, v):
        tok = self.next()
        if tok[1] != v:
            raise CompileError(f"expected {v!r}, got {tok[1]!r}")
        return tok

    def accept(self, v):
        if self.peek()[1] == v:
            self.i += 1
            return True
        return False

    def program(self):
        out = []
        while self.peek()[0] != "eof":
            out.append(self.stmt())
        return ("block", out)

    def stmt(self):
        tok = self.peek()
        if tok[1] == "{":
            self.next()
            out = []
            while not self.accept("}"):
                out.append(self.stmt())
            return ("block", out)
        if tok[1] == ";":
            self.next()
            return ("block", [])
        if tok[1] == "int":
            self.next()
            decls = []
            while True:
                name = self.next()
                if name[0] != "id":
                    raise CompileError(f"expected a name, got {name[1]!r}")
                if self.accept("["):
                    size = self.next()
                    if size[0] != "num":
                        raise CompileError("array size must be a number")
                    self.expect("]")
                    vals = []
                    if self.accept("="):
                        self.expect("{")
                        while not self.accept("}"):
                            vals.append(self.expr())
                            self.accept(",")
                    decls.append(("adecl", name[1], size[1], vals))
                else:
                    init = self.expr() if self.accept("=") else ("num", 0)
                    decls.append(("decl", name[1], init))
                if not self.accept(","):
                    break
            self.expect(";")
            return ("block", decls)
        if tok[1] == "if":
            self.next()
            self.expect("(")
            c = self.cond()
            self.expect(")")
            then = self.stmt()
            other = self.stmt() if self.accept("else") else ("block", [])
            return ("if", c, then, other)
        if tok[1] == "while":
            self.next()
            self.expect("(")
            c = self.cond()
            self.expect(")")
            return ("while", c, self.stmt())
        if tok[1] == "for":
            self.next()
            self.expect("(")
            init = ("block", []) if self.peek()[1] == ";" else self.simple()
            self.expect(";")
            c = ("true",) if self.peek()[1] == ";" else self.cond()
            self.expect(";")
            step = ("block", []) if self.peek()[1] == ")" else self.simple()
            self.expect(")")
            body = self.stmt()
            return ("block", [init, ("while", c, ("block", [body, step]))])
        if tok[1] == "halt":
            self.next()
            self.expect(";")
            return ("halt",)
        s = self.simple()
        self.expect(";")
        return s

    def simple(self):
        name = self.next()
        if name[0] != "id":
            raise CompileError(f"expected a statement, got {name[1]!r}")
        lv = ("var", name[1])
        if self.accept("["):
            lv = ("idx", name[1], self.expr())
            self.expect("]")
        op = self.next()[1]
        if op == "=":
            return ("assign", lv, self.expr())
        if op in ("+=", "-="):
            return ("assign", lv, (op[0], lv, self.expr()))
        if op in ("++", "--"):
            return ("assign", lv, (op[0], lv, ("num", 1)))
        raise CompileError(f"unknown statement operator {op!r}")

    def expr(self):
        e = self.term()
        while self.peek()[1] in ("+", "-"):
            op = self.next()[1]
            e = (op, e, self.term())
        return e

    def term(self):
        e = self.unary()
        while self.peek()[1] in ("*", "/", "%"):
            op = self.next()[1]
            e = (op, e, self.unary())
        return e

    def unary(self):
        if self.accept("-"):
            return ("neg", self.unary())
        if self.accept("+"):
            return self.unary()
        tok = self.next()
        if tok[0] == "num":
            return ("num", tok[1])
        if tok[0] == "id":
            if self.accept("["):
                e = self.expr()
                self.expect("]")
                return ("idx", tok[1], e)
            return ("var", tok[1])
        if tok[1] == "(":
            e = self.expr()
            self.expect(")")
            return e
        raise CompileError(f"unexpected {tok[1]!r}")

    def cond(self):
        c = self.cand()
        while self.accept("||"):
            c = ("or", c, self.cand())
        return c

    def cand(self):
        c = self.catom()
        while self.accept("&&"):
            c = ("and", c, self.catom())
        return c

    def catom(self):
        if self.accept("!"):
            return ("not", self.catom())
        if self.peek()[1] == "(":                      # a parenthesised condition or expression
            save = self.i
            self.next()
            try:
                c = self.cond()
                self.expect(")")
                if self.peek()[1] in ("==", "!=", "<", "<=", ">", ">=", "+", "-", "*", "/", "%"):
                    raise CompileError("expression")
                return c
            except CompileError:
                self.i = save
        e = self.expr()
        if self.peek()[1] in ("==", "!=", "<", "<=", ">", ">="):
            op = self.next()[1]
            return ("rel", op, e, self.expr())
        return ("rel", "!=", e, ("num", 0))


# ── linear expressions ──────────────────────────────────────────────────────

def linear(e):
    """Expression -> ({var: coef}, const)."""
    k = e[0]
    if k == "num":
        return {}, e[1]
    if k == "var":
        return {e[1]: 1}, 0
    if k == "neg":
        v, c = linear(e[1])
        return {x: -a for x, a in v.items()}, -c
    if k in ("+", "-"):
        v1, c1 = linear(e[1])
        v2, c2 = linear(e[2])
        s = 1 if k == "+" else -1
        out = dict(v1)
        for x, a in v2.items():
            out[x] = out.get(x, 0) + s * a
        return {x: a for x, a in out.items() if a}, c1 + s * c2
    if k in ("*", "/", "%"):
        v1, c1 = linear(e[1])
        v2, c2 = linear(e[2])
        if k == "%":
            if v1 or v2:
                raise CompileError("internal: % not hoisted")
            return {}, int(math.fmod(c1, c2))
        if k == "*" and not v1:
            v1, c1, v2, c2 = v2, c2, v1, c1
        if v2:
            raise CompileError("only multiplication / division by constants (write a loop)")
        if k == "*":
            return {x: a * c2 for x, a in v1.items() if a * c2}, c1 * c2
        if v1:
            raise CompileError("division of a variable (write a loop)")
        return {}, int(c1 / c2)
    raise CompileError(f"bad expression {e}")


# ── control flow graph ──────────────────────────────────────────────────────

@dataclass
class Block:
    id: int
    stmts: list = field(default_factory=list)      # (x, {var: coef}, const)
    term: tuple = ("halt",)                          # goto L | br t rel Lt Lf | halt


class Lower:
    def __init__(self):
        self.blocks, self.vars, self.init, self.arrays, self.ntemp = [], [], {}, {}, 0

    def new(self):
        b = Block(len(self.blocks))
        self.blocks.append(b)
        return b

    def var(self, x, declare=False):
        if declare:
            if x in self.init:
                raise CompileError(f"{x} declared twice")
            self.vars.append(x)
            return
        if x not in self.vars:
            raise CompileError(f"{x} is not declared")

    def lin(self, e):
        v, c = linear(e)
        for x in v:
            self.var(x)
        return v, c

    def temp(self):
        self.ntemp += 1
        name = f"_t{self.ntemp}"
        self.var(name, declare=True)
        self.init[name] = 0
        return name

    def element(self, a, k):
        if a not in self.arrays:
            raise CompileError(f"{a} is not an array")
        if not 0 <= k < self.arrays[a]:
            raise CompileError(f"{a}[{k}] is out of bounds")
        return f"{a}[{k}]"

    def chain(self, it, n, body):
        """if (it == 0) body(0) else if (it == 1) body(1) ... (an out-of-range index does nothing)."""
        out = ("block", [])
        for k in reversed(range(n)):
            out = ("if", ("rel", "==", ("var", it), ("num", k)), body(k), out)
        return out

    def hoist(self, e, cur):
        """Rewrite e so that it is linear: array reads with a variable index and products /
        quotients of variables become code (in new blocks) that leaves the value in a temp.
        Returns (expression, the block control continues in)."""
        k = e[0]
        if k in ("num", "var"):
            return e, cur
        if k == "neg":
            a, cur = self.hoist(e[1], cur)
            return ("neg", a), cur
        if k in ("+", "-"):
            a, cur = self.hoist(e[1], cur)
            b, cur = self.hoist(e[2], cur)
            return (k, a, b), cur
        if k == "idx":
            ie, cur = self.hoist(e[2], cur)
            v, c = linear(ie)
            if not v:
                return ("var", self.element(e[1], c)), cur
            self.element(e[1], 0)
            it, t = self.temp(), self.temp()
            cur.stmts.append((it,) + self.lin(ie))
            cur = self.stmt(self.chain(it, self.arrays[e[1]],
                                       lambda j: ("assign", ("var", t), ("var", f"{e[1]}[{j}]"))), cur)
            return ("var", t), cur
        if k in ("*", "/", "%"):
            a, cur = self.hoist(e[1], cur)
            b, cur = self.hoist(e[2], cur)
            try:
                linear((k, a, b))
                return (k, a, b), cur
            except CompileError:
                pass
            x, y, r = self.temp(), self.temp(), self.temp()
            V = lambda n: ("var", n)
            setup = [("assign", V(x), a), ("assign", V(y), b), ("assign", V(r), ("num", 0))]
            inc = lambda n, d: ("assign", V(n), ("+" if d > 0 else "-", V(n), ("num", 1)))
            if k == "*":                             # r += x, y times (y counts to 0 from either side)
                code = setup + [
                    ("while", ("rel", ">", V(y), ("num", 0)),
                     ("block", [("assign", V(r), ("+", V(r), V(x))), inc(y, -1)])),
                    ("while", ("rel", "<", V(y), ("num", 0)),
                     ("block", [("assign", V(r), ("-", V(r), V(x))), inc(y, +1)]))]
                out = r
            else:                                    # C division: x ends as the remainder
                def loop(test, step, dq):
                    return ("while", test, ("block", [("assign", V(x), step), inc(r, dq)]))
                pos = ("if", ("rel", ">=", V(x), ("num", 0)),
                       loop(("rel", ">=", V(x), V(y)), ("-", V(x), V(y)), +1),
                       loop(("rel", "<=", ("+", V(x), V(y)), ("num", 0)), ("+", V(x), V(y)), -1))
                neg = ("if", ("rel", ">=", V(x), ("num", 0)),
                       loop(("rel", ">=", ("+", V(x), V(y)), ("num", 0)), ("+", V(x), V(y)), -1),
                       loop(("rel", "<=", ("-", V(x), V(y)), ("num", 0)), ("-", V(x), V(y)), +1))
                code = setup + [("if", ("rel", ">", V(y), ("num", 0)), pos,
                                 ("if", ("rel", "<", V(y), ("num", 0)), neg, ("block", [])))]
                out = r if k == "/" else x
            cur = self.stmt(("block", code), cur)
            return ("var", out), cur
        raise CompileError(f"bad expression {e}")

    def hoist_cond(self, c, cur):
        k = c[0]
        if k == "true":
            return c, cur
        if k == "not":
            a, cur = self.hoist_cond(c[1], cur)
            return ("not", a), cur
        if k in ("and", "or"):
            a, cur = self.hoist_cond(c[1], cur)
            b, cur = self.hoist_cond(c[2], cur)
            return (k, a, b), cur
        a, cur = self.hoist(c[2], cur)
        b, cur = self.hoist(c[3], cur)
        return ("rel", c[1], a, b), cur

    def stmt(self, s, cur):
        """Lower s starting in block cur; returns the block control continues in (or None)."""
        k = s[0]
        if cur is None:                              # unreachable code after halt
            cur = self.new()
        if k == "block":
            for t in s[1]:
                cur = self.stmt(t, cur)
                if cur is None:
                    cur = self.new()
            return cur
        if k == "adecl":                             # int a[n] = {...}: n variables a[0] .. a[n-1]
            _, a, n, vals = s
            if a in self.arrays or a in self.vars:
                raise CompileError(f"{a} declared twice")
            if len(vals) > n:
                raise CompileError(f"too many initial values for {a}")
            self.arrays[a] = n
            for j in range(n):
                cur = self.stmt(("decl", f"{a}[{j}]", vals[j] if j < len(vals) else ("num", 0)), cur)
            return cur
        if k == "decl":                              # entry-block constants are initial memory
            e, cur = self.hoist(s[2], cur)
            v, c = linear(e)
            self.var(s[1], declare=True)
            if v or cur.id != 0:
                cur.stmts.append((s[1],) + self.lin(e))
                self.init[s[1]] = 0
            else:
                check_range(c, f"initial value of {s[1]}")
                self.init[s[1]] = c
            return cur
        if k == "assign":
            lv = s[1]
            e, cur = self.hoist(s[2], cur)
            if lv[0] == "var":
                self.var(lv[1])
                cur.stmts.append((lv[1],) + self.lin(e))
                return cur
            ie, cur = self.hoist(lv[2], cur)
            v, c = linear(ie)
            if not v:
                cur.stmts.append((self.element(lv[1], c),) + self.lin(e))
                return cur
            self.element(lv[1], 0)
            tv, it = self.temp(), self.temp()
            cur.stmts.append((tv,) + self.lin(e))
            cur.stmts.append((it,) + self.lin(ie))
            return self.stmt(self.chain(it, self.arrays[lv[1]],
                                        lambda j: ("assign", ("var", f"{lv[1]}[{j}]"), ("var", tv))), cur)
        if k == "halt":
            cur.term = ("halt",)
            return None
        if k == "if":
            c, cur = self.hoist_cond(s[1], cur)
            then, other, after = self.new(), self.new(), self.new()
            self.cond(c, cur, then.id, other.id)
            for blk, body in ((then, s[2]), (other, s[3])):
                end = self.stmt(body, blk)
                if end is not None:
                    end.term = ("goto", after.id)
            return after
        if k == "while":
            head, body, after = self.new(), self.new(), self.new()
            cur.term = ("goto", head.id)
            c, test = self.hoist_cond(s[1], head)
            self.cond(c, test, body.id, after.id)
            end = self.stmt(s[2], body)
            if end is not None:
                end.term = ("goto", head.id)
            return after
        raise CompileError(f"bad statement {s}")

    def cond(self, c, cur, lt, lf):
        """Branch from block cur on condition c."""
        k = c[0]
        if k == "true":
            cur.term = ("goto", lt)
        elif k == "not":
            self.cond(c[1], cur, lf, lt)
        elif k in ("and", "or"):
            mid = self.new()
            if k == "and":
                self.cond(c[1], cur, mid.id, lf)
            else:
                self.cond(c[1], cur, lt, mid.id)
            self.cond(c[2], mid, lt, lf)
        elif k == "rel":
            op, a, b = c[1], c[2], c[3]
            if op in (">", ">="):                    # a > b  <=>  b < a
                op, a, b = {">": "<", ">=": "<="}[op], b, a
            if op == "!=":
                op, lt, lf = "==", lf, lt
            rel = {"<": "lt", "<=": "le", "==": "eq"}[op]
            cur.term = ("br", self.lin(("-", a, b)), rel, lt, lf)
        else:
            raise CompileError(f"bad condition {c}")


def check_range(v, what):
    if not LO <= v <= HI:
        raise CompileError(f"{what} = {v} is outside the int range {LO} .. {HI}")


def simplify(blocks):
    """Drop empty pass-through blocks (no statements, unconditional goto)."""
    fwd = {}
    for b in blocks:
        if not b.stmts and b.term[0] == "goto" and b.term[1] != b.id:
            fwd[b.id] = b.term[1]

    def res(i, seen=()):
        while i in fwd and i not in seen:
            seen = seen + (i,)
            i = fwd[i]
        return i
    keep = [b for b in blocks if b.id not in fwd or b.id == 0]
    for b in keep:
        if b.term[0] == "goto":
            b.term = ("goto", res(b.term[1]))
        elif b.term[0] == "br":
            b.term = ("br", b.term[1], b.term[2], res(b.term[3]), res(b.term[4]))
    # reachable only
    ids = {b.id: b for b in keep}
    seen, todo = set(), [0]
    while todo:
        i = todo.pop()
        if i in seen:
            continue
        seen.add(i)
        t = ids[i].term
        todo += [t[1]] if t[0] == "goto" else ([t[3], t[4]] if t[0] == "br" else [])
    return [b for b in keep if b.id in seen]


# ── code generation ─────────────────────────────────────────────────────────

class Gen:
    """Emits the ring. Slots: V_i = register 2i, T_i = 2i + 1. ``pos`` = T slot of each
    live train item (acc, G); ``sgn`` = stored / true value of the accumulator."""

    def __init__(self, slots, P):
        self.slot, self.P, self.tape = slots, P, []
        self.pos, self.sgn = {}, 1

    V = staticmethod(lambda i: 2 * i)
    T = staticmethod(lambda i: 2 * i + 1)

    def emit(self, op, a, b, d):
        assert max(a, b, d) - min(a, b, d) <= 2 and min(a, b, d) >= 0 and max(a, b, d) < 2 * self.P
        self.tape.append((op, a, b, d))

    def zt(self, i):
        self.emit(SUBLEQ, self.T(i), self.T(i), self.T(i))

    def holder(self, i):
        for k, p in self.pos.items():
            if p == i:
                return k
        return None

    def step(self, item, d):
        j = self.pos[item] + d
        if not 0 <= j < self.P:
            raise CompileError("train ran off the memory (internal)")
        other = self.holder(j)
        if other is not None:
            self.step(other, d)
        self.zt(j)
        self.emit(SUB, self.T(j), self.T(self.pos[item]), self.T(j))
        self.pos[item] = j
        if item == "acc":
            self.sgn = -self.sgn

    def walk(self, item, target):
        while self.pos[item] != target:
            self.step(item, 1 if target > self.pos[item] else -1)

    def negate(self):
        """acc <- -acc in place (a free neighbour T is zeroed and subtracted from)."""
        p = self.pos["acc"]
        for q in (p + 1, p - 1):
            if 0 <= q < self.P and self.holder(q) is None:
                self.zt(q)
                self.emit(SUB, self.T(q), self.T(p), self.T(p))
                self.sgn = -self.sgn
                return
        raise CompileError("no room to negate (internal)")

    def new_acc(self, near):
        """A zero accumulator in a free T slot next to ``near`` (on the side away from G)."""
        g = self.pos.get("G")
        cands = [near, near + 1, near - 1, near + 2, near - 2]
        if g is not None:
            cands.sort(key=lambda q: (q == g, (q - g) * (near - g) < 0, abs(q - near)))
        for q in cands:
            if 0 <= q < self.P and self.holder(q) is None:
                self.zt(q)
                self.pos["acc"], self.sgn = q, 1
                return
        raise CompileError("no room for the accumulator (internal)")

    def fold(self, v, coef):
        """acc += coef * V_v  (coef = +-1). The fold leaves E - s*v, so the accumulator must
        arrive with s = -coef: of the two T slots next to V_v pick the one with that parity."""
        p = self.pos["acc"]
        want = -coef
        sides = sorted((v - 1, v), key=lambda q: abs(q - p))
        side = next(q for q in sides if self.sgn * (-1) ** abs(q - p) == want)
        self.walk("acc", side)
        if self.sgn != want:
            self.negate()
        self.emit(SUB, self.T(side), self.V(v), self.T(side))

    def straddle(self, x):
        """acc and G on the two T slots around V_x (keeping their order); G's copy room is the
        next slot out. Returns (acc slot, G slot, outer slot)."""
        a, g = self.pos["acc"], self.pos["G"]
        if g < a:
            pa, pg, out = x, x - 1, x - 2
        else:
            pa, pg, out = x - 1, x, x + 1
        self.walk("acc", pa)
        self.walk("G", pg)
        if self.holder(out) is not None or not 0 <= out < self.P:
            raise CompileError("no room around a write (internal)")
        return pa, pg, out

    def gwrite(self, x):
        """V_x <- acc if the block is active (destroys acc; G survives in a copy)."""
        pa, pg, out = self.straddle(x)
        if self.sgn != 1:
            self.negate()
        self.zt(out)
        self.emit(SUB, self.T(out), self.T(pg), self.T(out))     # copy of G (-G == G)
        reg = {"A": self.T(pa), "B": self.T(pg), "V": self.V(x)}
        for op, a, b, d in CMOV:
            self.emit(op, reg[a], reg[b], reg[d])
        del self.pos["acc"]
        self.pos["G"] = out

    def compute(self, terms):
        """acc = sum coef * var (terms: list of (slot, coef) with coef = +-1), visited in an order
        that sweeps across the memory once."""
        near = terms[0][0] if terms else self.pos["G"]
        self.new_acc(near)
        p = self.pos["acc"]
        left = sorted([t for t in terms if t[0] <= p], key=lambda t: -t[0])
        right = sorted([t for t in terms if t[0] > p], key=lambda t: t[0])
        # go to the nearer end first
        first, second = (left, right) if (left and right and p - left[-1][0] < right[-1][0] - p) \
            else (right, left)
        for v, c in first + second:
            self.fold(v, c)

    def compare(self, rel, kslot):
        """acc (= t, sign +) -> 0 if rel(t) else -16, at the constant -16 in V_kslot."""
        a, g = self.pos["acc"], self.pos["G"]
        if g < a:
            pt, ps = kslot, kslot - 1
            gt = kslot - 2
        else:
            pt, ps = kslot - 1, kslot
            gt = kslot + 1
        self.walk("acc", pt)
        if (g < a and self.pos["G"] > gt) or (g > a and self.pos["G"] < gt):
            self.walk("G", gt)
        if self.sgn != 1:
            self.negate()
        reg = {"T": self.T(pt), "S": self.T(ps), "C": self.V(kslot)}
        for op, x, y, d in CMP[rel]:
            self.emit(op, reg[x], reg[y], reg[d])


def terms_of(lin, slot_of, consts):
    v, c = lin
    out = []
    for x, a in v.items():
        out += [(slot_of[x], 1 if a > 0 else -1)] * abs(a)
    c &= MASK
    if c:
        out.append((consts(c), 1))
    return out


@dataclass
class Compiled:
    tape: list
    regs0: list
    var_reg: dict           # variable -> register
    flag_reg: dict          # block id -> register
    blocks: list

    @property
    def n_regs(self):
        return len(self.regs0)

    def halted(self, regs):
        return all(regs[r] == IDLE & MASK for r in self.flag_reg.values())

    def read(self, regs):
        return {x: signed(regs[r]) for x, r in self.var_reg.items()}


def signed(v):
    v &= MASK
    return v - (1 << WB) if v & (1 << (WB - 1)) else v


def compile_src(src, layout_iters=2000, seed=0):
    """Source -> Compiled. The V-lane order starts as order of first use and is then improved
    by a local search (move one register elsewhere, keep it if the ring gets shorter): most
    instructions are carries along the T lane, so the layout sets the program length."""
    ast = Parser(src).program()
    low = Lower()
    entry = low.new()
    end = low.stmt(ast, entry)
    if end is not None:
        end.term = ("halt",)
    blocks = simplify(low.blocks)

    # ── layout: V lane in order of first use, block-local constants ──
    order, consts_needed = [], {}
    seen = set()

    def use(key):
        if key not in seen:
            seen.add(key)
            order.append(key)

    def lin_keys(b, lin):
        v, c = lin
        keys = [("v", x) for x in v]
        if c & MASK:
            keys.append(("c", b.id, c & MASK))
        return keys

    for b in blocks:
        use(("f", b.id))
        use(("c", b.id, IDLE & MASK))
        for x, v, c in b.stmts:
            for k in lin_keys(b, (v, c)):
                use(k)
            use(("v", x))
        t = b.term
        if t[0] == "br":
            for k in lin_keys(b, t[1]):
                use(k)
            use(("f", t[3]))
            use(("f", t[4]))
        elif t[0] == "goto":
            use(("f", t[1]))
    for x in low.vars:
        use(("v", x))
    best = _build(order, blocks, low)
    import random
    rng = random.Random(seed)
    for _ in range(layout_iters if len(order) > 2 else 0):
        o = list(order)
        i, j = rng.randrange(len(o)), rng.randrange(len(o))
        o.insert(j, o.pop(i))
        try:
            c = _build(o, blocks, low)
        except CompileError:
            continue
        if len(c.tape) <= len(best.tape):
            best, order = c, o
    return best


def _build(order, blocks, low):
    MARGIN = 3
    slot = {k: MARGIN + i for i, k in enumerate(order)}
    P = len(order) + 2 * MARGIN
    g = Gen(slot, P)
    slot_of = {k[1]: s for k, s in slot.items() if k[0] == "v"}

    for b in blocks:
        cslot = lambda val, b=b: slot[("c", b.id, val)]
        f = slot[("f", b.id)]
        g.pos = {}
        # G = 2 F_b in T_f (T_{f-1} scratch)
        g.zt(f - 1)
        g.emit(SUB, g.T(f - 1), g.V(f), g.T(f - 1))
        g.emit(SUB, g.V(f), g.T(f - 1), g.T(f))
        g.pos["G"] = f

        def assign(slot_x, terms):
            g.compute(terms)
            g.gwrite(slot_x)

        t = b.term
        targets = () if t[0] == "halt" else ((t[1],) if t[0] == "goto" else (t[3], t[4]))
        if b.id not in targets:                     # leave the block: F_b <- idle
            assign(f, [(cslot(IDLE & MASK), 1)])
        for x, v, c in b.stmts:
            assign(slot_of[x], terms_of((v, c), slot_of, cslot))
        if t[0] == "goto" and t[1] != b.id:
            assign(slot[("f", t[1])], [])            # F_target <- 0
        elif t[0] == "br":
            _, lin, rel, lt, lf = t
            g.compute(terms_of(lin, slot_of, cslot))
            g.compare(rel, cslot(IDLE & MASK))
            g.gwrite(slot[("f", lt)])                # F_true <- q
            if lf != lt:                             # F_false <- -16 - F_true
                assign(slot[("f", lf)], [(cslot(IDLE & MASK), 1), (slot[("f", lt)], -1)])

    regs0 = [0] * (2 * P)
    for k, s in slot.items():
        if k[0] == "v":
            regs0[2 * s] = low.init.get(k[1], 0) & MASK
        elif k[0] == "c":
            regs0[2 * s] = k[2]
        elif k[0] == "f":
            regs0[2 * s] = 0 if k[1] == blocks[0].id else IDLE & MASK
    return Compiled(g.tape, regs0, {x: 2 * s for x, s in slot_of.items()},
                    {k[1]: 2 * s for k, s in slot.items() if k[0] == "f"}, blocks)


# ── software machine and reference ──────────────────────────────────────────

def step(regs, ins):
    op, a, b, d = ins
    r = (regs[a] - regs[b]) & MASK
    if op == SUB or signed(r) <= 0:
        regs[d] = r


def run_machine(p, max_passes=10_000):
    """Run whole ring passes until halted; returns (variables, passes, instructions)."""
    regs = list(p.regs0)
    for k in range(1, max_passes + 1):
        for ins in p.tape:
            step(regs, ins)
        if p.halted(regs):
            return p.read(regs), k, k * len(p.tape)
    raise RuntimeError(f"no halt within {max_passes} passes")


def reference(src, max_steps=100_000):
    """Direct interpretation of the source (ints checked to stay in range)."""
    ast = Parser(src).program()
    env, arrays = {}, {}

    def ev(e):
        k = e[0]
        if k == "num":
            return e[1]
        if k == "var":
            return env[e[1]]
        if k == "neg":
            return -ev(e[1])
        if k == "idx":
            i = ev(e[2])
            if not 0 <= i < arrays[e[1]]:
                raise CompileError(f"{e[1]}[{i}] out of bounds")
            return env[f"{e[1]}[{i}]"]
        a, b = ev(e[1]), ev(e[2])
        if k == "+":
            return a + b
        if k == "-":
            return a - b
        const = not linear_ok(e[1]) or not linear_ok(e[2])
        if k == "*":
            r = a * b
        elif b == 0:
            raise CompileError("division by zero")
        else:
            r = int(a / b) if k == "/" else int(math.fmod(a, b))
        if const:                                    # the compiler keeps it in a register
            check_range(r, "product / quotient")
        return r

    def linear_ok(e):
        try:
            v, _ = linear(e)
            return not v
        except CompileError:
            return False

    def lval(lv):
        if lv[0] == "var":
            return lv[1]
        i = ev(lv[2])
        if not 0 <= i < arrays[lv[1]]:
            raise CompileError(f"{lv[1]}[{i}] out of bounds")
        return f"{lv[1]}[{i}]"

    def cond(c):
        k = c[0]
        if k == "true":
            return True
        if k == "not":
            return not cond(c[1])
        if k == "and":
            return cond(c[1]) and cond(c[2])
        if k == "or":
            return cond(c[1]) or cond(c[2])
        t = ev(("-", c[2], c[3]))
        if not -31 <= t <= 31:
            raise CompileError(f"comparison difference {t} overflows")
        return {"<": t < 0, "<=": t <= 0, "==": t == 0, "!=": t != 0, ">": t > 0, ">=": t >= 0}[c[1]]

    steps = [0]

    class Halt(Exception):
        pass

    def run(s):
        steps[0] += 1
        if steps[0] > max_steps:
            raise RuntimeError("reference: too many steps")
        k = s[0]
        if k == "block":
            for t in s[1]:
                run(t)
        elif k == "adecl":
            arrays[s[1]] = s[2]
            for j in range(s[2]):
                env[f"{s[1]}[{j}]"] = ev(s[3][j]) if j < len(s[3]) else 0
                check_range(env[f"{s[1]}[{j}]"], s[1])
        elif k in ("decl", "assign"):
            x = s[1] if k == "decl" else lval(s[1])
            env[x] = ev(s[2])
            check_range(env[x], x)
        elif k == "halt":
            raise Halt
        elif k == "if":
            run(s[2] if cond(s[1]) else s[3])
        elif k == "while":
            while cond(s[1]):
                run(s[2])
    try:
        run(ast)
    except Halt:
        pass
    return env


# ── random programs (fuzzing, training data) ────────────────────────────────

def random_program(rng, n_vars=4):
    """Random source with nested if / while / && / || over n_vars variables."""
    V = "abcdefgh"[:n_vars]

    def ex(d=0):
        if d > 1 or rng.random() < 0.4:
            return rng.choice(list(V) + [str(rng.randint(0, 5))])
        return f"({ex(d + 1)} {rng.choice('+-')} {ex(d + 1)})"

    def cond(d=0):
        k = rng.random()
        if d < 1 and k < 0.25:
            return f"({cond(d + 1)} {rng.choice(['&&', '||'])} {cond(d + 1)})"
        if d < 1 and k < 0.35:
            return f"!({cond(d + 1)})"
        return f"{ex(1)} {rng.choice(['<', '<=', '>', '>=', '==', '!='])} {ex(1)}"

    def st(d=0):
        k = rng.random()
        if d < 2 and k < 0.2:
            return f"if ({cond()}) {{ {blk(d + 1)} }} else {{ {blk(d + 1)} }}"
        if d < 2 and k < 0.3:
            i = f"i{d}{rng.randint(0, 99)}"
            return f"int {i} = {rng.randint(1, 3)}; while ({i} > 0) {{ {blk(d + 1)} {i}--; }}"
        return f"{rng.choice(V)} = {ex()};"

    def blk(d):
        return " ".join(st(d) for _ in range(rng.randint(1, 3)))

    return "int " + ", ".join(f"{v} = {rng.randint(-5, 5)}" for v in V) + "; " + blk(0)


def trace(p, max_cycles=4000):
    """Register states before every cycle until the program halts (or max_cycles)."""
    regs, out = list(p.regs0), []
    n = len(p.tape)
    for c in range(max_cycles):
        out.append(tuple(regs))
        step(regs, p.tape[c % n])
        if (c + 1) % n == 0 and p.halted(regs):
            break
    out.append(tuple(regs))
    return out


def random_traces(n_progs, seed=0, max_cycles=4000):
    """(tape, states) for n_progs random programs that compile and stay in range."""
    import random
    rng, out = random.Random(seed), []
    while len(out) < n_progs:
        src = random_program(rng)
        try:
            reference(src, 20000)
            p = compile_src(src)
        except (CompileError, RuntimeError, KeyError):
            continue
        out.append((p.tape, trace(p, max_cycles)))
    return out
