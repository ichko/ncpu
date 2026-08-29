#!/usr/bin/env python3
"""Minimal subleq assembler + VM for the NCA demo.

The ALU (compute `R = (B - A) mod 256` and `branch = signed(R) <= 0`) is a
pluggable callable, so we can run with a ground-truth ALU (to validate the
compiler/harness) or with the trained dartboard NCA (the actual demo).

subleq semantics per instruction `a b c`:
    mem[b] = (mem[b] - mem[a]) mod 256
    if signed(mem[b]) <= 0:  pc = c   else  pc += 3

Memory is a flat byte array: 3 bytes per instruction (a, b, c addresses) laid
out first, then the data cells. A jump to the `end` label lands the PC at the
start of the data region (>= code size) which halts the VM.
"""


class Program:
    def __init__(self):
        self.code = []          # list of (a, b, c) with symbolic operands (str)
        self.tags = []          # source-line label per instruction (for the demo)
        self.cur = ""           # current source line
        self.labels = {}        # name -> instruction index
        self.data = {}          # name -> initial value
        self.data_order = []

    def var(self, name, val=0):
        if name not in self.data:
            self.data[name] = val
            self.data_order.append(name)
        return name

    def label(self, name):
        self.labels[name] = len(self.code)

    def src(self, line):        # tag subsequent instructions with a source line
        self.cur = line

    def si(self, a, b, c=None):   # subleq a, b [, c]; c=None -> fall through
        self.code.append((a, b, c))
        self.tags.append(self.cur)


# ── subleq macros (each expands to subleq triples; Z is a scratch kept at 0) ──
def ZERO(p, x):          p.si(x, x)                    # x = 0
def SUB(p, dst, src):    p.si(src, dst)                # dst -= src
def ADD(p, dst, src):    p.si(src, "Z"); p.si("Z", dst); p.si("Z", "Z")   # dst += src
def MOV(p, dst, src):    ZERO(p, dst); ADD(p, dst, src)                    # dst = src
def DEC(p, x):           SUB(p, x, "one")              # x -= 1
def JMP(p, to):          p.si("Z", "Z", to)            # unconditional -> to
def BLEZ(p, x, to):      p.si("Z", x, to)              # if x <= 0 -> to (x unchanged)


def fib_program(n):
    """Iterative fib: after k loops a = fib(k); loops n times -> a = fib(n)."""
    p = Program()
    p.var("Z", 0); p.var("one", 1)
    p.var("n", n); p.var("a", 0); p.var("b", 1); p.var("t", 0)
    p.label("loop")
    p.src("while (n > 0) {");   BLEZ(p, "n", "end")
    p.src("t = a + b;");        MOV(p, "t", "a"); ADD(p, "t", "b")
    p.src("a = b;");            MOV(p, "a", "b")
    p.src("b = t;");            MOV(p, "b", "t")
    p.src("n = n - 1;");        DEC(p, "n")
    p.src("} // repeat");       JMP(p, "loop")
    p.label("end")
    return p, "a"           # result lives in cell `a`


def assemble(p):
    """Resolve symbols -> flat byte memory + an address map."""
    n_instr = len(p.code)
    code_bytes = n_instr * 3
    addr = {}                                   # symbol -> byte address
    for name in p.labels:                       # labels -> instruction byte offset
        addr[name] = p.labels[name] * 3
    for i, name in enumerate(p.data_order):     # data cells after the code
        addr[name] = code_bytes + i
    addr.setdefault("end", code_bytes)

    def resolve(op, i):
        if op is None:
            return (i + 1) * 3                  # fall through to next instruction
        return addr[op]

    mem = [0] * (code_bytes + len(p.data_order))
    for i, (a, b, c) in enumerate(p.code):
        mem[i * 3 + 0] = addr[a]
        mem[i * 3 + 1] = addr[b]
        mem[i * 3 + 2] = resolve(c, i)
    for name in p.data_order:
        mem[addr[name]] = p.data[name] & 0xFF
    return mem, addr, code_bytes


def ref_alu(A, B):
    R = (B - A) & 0xFF
    signed = R - 256 if R >= 128 else R
    return R, (1 if signed <= 0 else 0)


def run(mem, code_bytes, alu=ref_alu, max_steps=100_000):
    mem = list(mem)
    pc = 0
    trace = []
    steps = 0
    while 0 <= pc < code_bytes and steps < max_steps:
        a, b, c = mem[pc], mem[pc + 1], mem[pc + 2]
        opA, opB = mem[a], mem[b]
        R, branch = alu(opA, opB)
        mem[b] = R
        pc_after = c if branch else pc + 3
        trace.append({"step": steps, "pc": pc, "idx": pc // 3,
                      "a": a, "b": b, "c": c, "opA": opA, "opB": opB,
                      "result": R, "branch": branch, "pc_after": pc_after,
                      "mem": mem.copy()})
        pc = pc_after
        steps += 1
    return mem, trace, steps


if __name__ == "__main__":
    import sys
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    p, result_cell = fib_program(N)
    mem, addr, code_bytes = assemble(p)
    out, trace, steps = run(mem, code_bytes)
    expected = 0
    fa, fb = 0, 1
    for _ in range(N):
        fa, fb = fb, (fa + fb)
    print(f"fib({N}): VM result cell '{result_cell}'={out[addr[result_cell]]}  "
          f"expected={fa}  {'OK' if out[addr[result_cell]] == fa else 'MISMATCH'}")
    print(f"memory size: {len(mem)} bytes  ({code_bytes//3} instructions, "
          f"{len(mem)-code_bytes} data cells)")
    print(f"executed subleq instructions: {steps}")
