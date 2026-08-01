from collections import deque
from typing import List
import sys

import sys
import torch
from argparse import Namespace
from torch.utils.data import DataLoader, IterableDataset

import random

from ncpu.utils import make_alu2_screen, make_alu_screen, make_io_screen, make_io_screen_bottom_aligned, make_io_screen_cols1


def sample_8bit_adder(*args):
    a = torch.randint(0, 2, size=(8,))
    b = torch.randint(0, 2, size=(8,))
    a_int = int("".join(map(str, a.tolist())), 2)
    b_int = int("".join(map(str, b.tolist())), 2)
    s_int = a_int + b_int
    out = torch.tensor(list(map(int, f"{s_int:09b}")))
    return torch.cat([a, b]), out


def sample_8bit_subleq(*args):
    """subleq ALU with the SAME I/O shape as the 8-bit adder (16 in -> 9 out).

    input  = [A(8), B(8)]  (MSB-first, A in column 0, B in column 1)
    output = [R(8), branch] where R = (B - A) mod 256 and
             branch = 1 iff signed(R) <= 0  (i.e. R == 0 or R's sign bit set)
    """
    a = torch.randint(0, 2, size=(8,))
    b = torch.randint(0, 2, size=(8,))
    a_int = int("".join(map(str, a.tolist())), 2)
    b_int = int("".join(map(str, b.tolist())), 2)
    r_int = (b_int - a_int) & 0xFF
    branch = 1 if (r_int == 0 or r_int >= 128) else 0
    out = torch.tensor(list(map(int, f"{r_int:08b}")) + [branch])
    return torch.cat([a, b]), out


def sample_4bit_multiplier(*args):
    a = torch.randint(0, 2, size=(4,))
    b = torch.randint(0, 2, size=(4,))
    a_int = int("".join(map(str, a.tolist())), 2)
    b_int = int("".join(map(str, b.tolist())), 2)
    p_int = a_int * b_int  # 0..225, fits in 8 bits
    out = torch.tensor(list(map(int, f"{p_int:08b}")))
    return torch.cat([a, b]), out


def sample_4bit_adder(*args):
    a = torch.randint(0, 2, size=(4,))
    b = torch.randint(0, 2, size=(4,))

    a_int = int("".join(map(str, a.tolist())), 2)
    b_int = int("".join(map(str, b.tolist())), 2)

    s_int = a_int + b_int

    out = torch.tensor(list(map(int, f"{s_int:05b}")))
    inp = torch.cat([a, b])
    # Interlace bits
    # inp = torch.tensor([a[0], b[0], a[1], b[1], a[2], b[2], a[3], b[3]])

    return inp, out


def two_arg_sampler(op):
    inp = torch.randint(0, 2, size=(2,))
    a, b = inp
    out = op(a, b)
    return inp, torch.tensor([out])


def sample_AND_gate(*args):
    return two_arg_sampler(lambda a, b: a & b)


def sample_OR_gate(*args):
    return two_arg_sampler(lambda a, b: a | b)


def sample_NOR_gate(*args):
    return two_arg_sampler(lambda a, b: not (a | b))


def sample_NAND_gate(*args):
    return two_arg_sampler(lambda a, b: not (a & b))


def sample_XOR_gate(*args):
    return two_arg_sampler(lambda a, b: a != b)


def sample_XNOR_gate(*args):
    return two_arg_sampler(lambda a, b: not (a != b))


def sample_half_adder(*args):
    inp = torch.randint(0, 2, size=(2,))
    a, b = int(inp[0]), int(inp[1])
    return inp, torch.tensor([a ^ b, a & b])  # [sum, carry]


def sample_majority3(*args):
    inp = torch.randint(0, 2, size=(3,))
    a, b, c = int(inp[0]), int(inp[1]), int(inp[2])
    return inp, torch.tensor([(a + b + c) >= 2], dtype=torch.int64)


# def sample_8bit_adder(*args):
#     return two_arg_sampler(lambda a, b: a != b)


class ExtrapolationDataset(IterableDataset):
    """Adder dataset with variable bit width for length-generalisation experiments.

    Each batch_size consecutive samples share the same bit width k, sampled
    uniformly from [min_bits, max_bits].  The grid is fixed to accommodate
    max_bits inputs and max_bits+1 outputs, with bottom alignment so LSB is
    always at the same absolute y position.
    """

    def __init__(self, W, H, r, spacing, min_bits, max_bits, batch_size):
        self.W = W
        self.H = H
        self.r = r
        self.spacing = spacing
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.batch_size = batch_size
        self.current_k = torch.randint(min_bits, max_bits + 1, ()).item()
        self.samples_at_k = 0

    def _resample_k(self):
        self.current_k = torch.randint(self.min_bits, self.max_bits + 1, ()).item()
        self.samples_at_k = 0

    def _make_screen(self, a_bits, b_bits, output_bits):
        return make_io_screen_bottom_aligned(
            self.H, self.W, self.r, self.spacing, a_bits, b_bits, output_bits
        )

    def get_io_mask(self):
        inp = self._make_screen([1] * self.max_bits, [1] * self.max_bits, [])
        out = self._make_screen([], [], [1] * (self.max_bits + 1))
        return inp, out

    def get_output_bit_masks(self):
        n_out = self.max_bits + 1
        masks = []
        for i in range(n_out):
            one_hot = [1 if j == i else 0 for j in range(n_out)]
            s = self._make_screen([], [], one_hot)
            masks.append(torch.from_numpy(s > 200).float())
        return torch.stack(masks)  # (n_out, H, W)

    def get_sample(self):
        if self.samples_at_k >= self.batch_size:
            self._resample_k()

        k = self.current_k
        a = torch.randint(0, 2, (k,))
        b = torch.randint(0, 2, (k,))
        a_int = int("".join(map(str, a.tolist())), 2)
        b_int = int("".join(map(str, b.tolist())), 2)
        s_int = a_int + b_int
        out_bits = list(map(int, f"{s_int:0{k + 1}b}"))

        inp = self._make_screen(a.tolist(), b.tolist(), [])
        out = self._make_screen([], [], out_bits)

        self.samples_at_k += 1
        return (
            torch.from_numpy(inp).float(),
            torch.from_numpy(out).float(),
        )

    def __iter__(self):
        while True:
            yield self.get_sample()

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False)


class NCPUDataset(IterableDataset):
    def __init__(self, config):
        self.W = config.W
        self.H = config.H
        self.r = config.r
        self.spacing = config.spacing
        self.sampler = config.sampler
        self.screen_fn = getattr(config, "screen_fn", make_io_screen)

        self.balanced = config.balanced
        self.prev_class = 0
        self.class_neg = 0

    def get_io_mask(self):
        left, right = self.sampler()
        left_screen = self.screen_fn(
            W=self.W,
            H=self.H,
            r=self.r,
            spacing=self.spacing,
            left_input=torch.ones_like(left),
            right_input=[],
        )
        right_screen = self.screen_fn(
            W=self.W,
            H=self.H,
            r=self.r,
            spacing=self.spacing,
            left_input=[],
            right_input=torch.ones_like(right),
        )
        return left_screen, right_screen

    def get_output_bit_masks(self):
        """Returns (n_bits, H, W) float tensor — one binary mask per output circle."""
        _, right = self.sampler()
        n_bits = len(right)
        masks = []
        for i in range(n_bits):
            one_hot = [1 if j == i else 0 for j in range(n_bits)]
            screen = self.screen_fn(
                W=self.W,
                H=self.H,
                r=self.r,
                spacing=self.spacing,
                left_input=[],
                right_input=one_hot,
            )
            masks.append(torch.from_numpy(screen > 200).float())
        return torch.stack(masks)  # (n_bits, H, W)

    def get_sample(self):
        left, right = self.sampler()

        if self.balanced:
            # alternate between classes
            while self.prev_class == right:
                left, right = self.sampler()
            self.prev_class = right

        inp = self.screen_fn(
            W=self.W,
            H=self.H,
            r=self.r,
            spacing=self.spacing,
            left_input=left,
            right_input=[],
        )

        out = self.screen_fn(
            W=self.W,
            H=self.H,
            r=self.r,
            spacing=self.spacing,
            left_input=[],
            right_input=right,
        )

        return torch.from_numpy(inp).to(dtype=torch.float32), torch.from_numpy(out).to(
            dtype=torch.float32
        )

    def __iter__(self):
        while True:
            yield self.get_sample()

    def get_dataloader(self, batch_size):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,  # can't shuffle IterableDataset
        )

class DynamicDataset(IterableDataset):
    def __init__(
        self,
        config,
        update_y : int,
        update_x : int,
        steps: int,
        stages : int = sys.maxsize
    ):
        self.steps = steps
        self.counter = 0
        self.dataset = NCPUDataset(config)
        self.spacing = self.dataset.spacing
        self.W = self.dataset.W
        self.H = self.dataset.H
        self.r = self.dataset.r
        self.update_y = update_y
        self.update_x = update_x

        self.stage = 0
        self.stages = stages

    def get_io_mask(self):
        return self.dataset.get_io_mask()

    def get_sample(self):
        if self.counter >= self.steps and self.stage < self.stages:
            self.counter = 0
            spacing = self.spacing
            spacing_x = max(spacing[0] - self.update_x, self.r) # stop points from moving outside the board
            spacing_y = max(spacing[1] - self.update_y, self.r + 2) # stop points from moving outside the board
            self.spacing = (spacing_x, spacing_y)
            self.stage += 1

        self.dataset.W = self.W
        self.dataset.H = self.H
        self.dataset.r = self.r
        self.dataset.spacing = self.spacing
        ret = self.dataset.get_sample()
        self.counter += 1
        return ret

    def __iter__(self):
        while True:
            yield self.get_sample()

    def get_dataloader(self, batch_size):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,  # can't shuffle IterableDataset
        )

class ScheduledDataset(IterableDataset):
    def __init__(
        self,
        datasets: List[NCPUDataset],
        steps: int,
    ):
        self.steps = steps
        self.counter = 0
        self.ds_index = 0
        self.datasets = datasets
        self.W = datasets[self.ds_index].W
        self.H = datasets[self.ds_index].H
        self.r = datasets[self.ds_index].r

    def get_io_mask(self):
        return self.datasets[self.ds_index].get_io_mask()

    def get_sample(self):
        if self.counter >= self.steps:
            self.counter = 0
            self.ds_index = (
                self.ds_index + 1
                if self.ds_index < (len(self.datasets) - 1)
                else self.ds_index
            )

        self.W = self.datasets[self.ds_index].W
        self.H = self.datasets[self.ds_index].H
        self.r = self.datasets[self.ds_index].r
        ret = self.datasets[self.ds_index].get_sample()
        self.counter += 1
        return ret

    def get_io_mask(self):
        return self.datasets[self.ds_index].get_io_mask()

    def __iter__(self):
        while True:
            yield self.get_sample()

    def get_dataloader(self, batch_size):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,  # can't shuffle IterableDataset
        )


class PoolDataset(IterableDataset):
    def __init__(self, dataset, pool_size):
        self.dataset = dataset
        self.pool_size = pool_size
        self.pool = [None for _ in range(self.pool_size)]
        self._counter = 0
        self._counter_list = deque()

        # ugly hackety hack
        self.W = self.dataset.W
        self.H = self.dataset.H
        self.r = self.dataset.r
        self.dataset_iter = iter(self.dataset)

    def __iter__(self):
        while True:
            if not self.pool[self._counter]:
                self.pool[self._counter] = next(self.dataset_iter)
            val = self.pool[self._counter]
            self._counter_list.append(self._counter)
            self._counter += 1
            self._counter = self._counter % self.pool_size
            yield val[0].clone(), val[1].clone()

    def update(self, batch, losses):
        inp, out = batch[0].detach(), batch[1].detach()
        max_idx = torch.argmax(losses)
        min_idx = torch.argmin(losses)
        for batch_i, sample in enumerate(batch):
            pool_i = self._counter_list.popleft()
            if min_idx == batch_i and (inp[min_idx].mean() > 0.01):
                self.pool[pool_i] = (inp[min_idx].cpu(), self.pool[pool_i][1].cpu())
            else:
                self.pool[pool_i] = next(
                    self.dataset_iter
                )  # prune worst results by refreshing pool

    def get_dataloader(self, batch_size):
        return PoolLoader(
            self,
            batch_size=batch_size,
            shuffle=False,  # can't shuffle IterableDataset
        )


def _int_to_bits(n, width):
    return [int(b) for b in f"{n:0{width}b}"]


def _compute_alu(a_int, b_int, op):
    """Compute 8-bit ALU result (truncated to 8 bits).

    Opcodes:
        0: ADD   (A + B) & 0xFF
        1: SUB   (A - B) & 0xFF
        2: AND   A & B
        3: OR    A | B
        4: XOR   A ^ B
        5: NOT   ~A & 0xFF
        6: SHL   (A << 1) & 0xFF
        7: SHR   A >> 1
    """
    if   op == 0: return (a_int + b_int) & 0xFF
    elif op == 1: return (a_int - b_int) & 0xFF
    elif op == 2: return a_int & b_int
    elif op == 3: return a_int | b_int
    elif op == 4: return a_int ^ b_int
    elif op == 5: return (~a_int) & 0xFF
    elif op == 6: return (a_int << 1) & 0xFF
    else:         return a_int >> 1


class ALUDataset(IterableDataset):
    """8-bit ALU dataset — column layout (A/B left, opcode middle, result right)."""

    def __init__(self, config):
        self.W       = config.W
        self.H       = config.H
        self.r       = config.r
        self.spacing = config.spacing

    def _make_screen(self, a_bits=None, b_bits=None, opcode_bits=None, result_bits=None):
        return make_alu_screen(
            self.H, self.W, self.r, self.spacing,
            a_bits=a_bits, b_bits=b_bits,
            opcode_bits=opcode_bits, result_bits=result_bits,
        )

    def get_io_mask(self):
        inp = self._make_screen(a_bits=[1]*8, b_bits=[1]*8, opcode_bits=[1]*3)
        out = self._make_screen(result_bits=[1]*8)
        return inp, out

    def get_output_bit_masks(self):
        """Returns (8, H, W) tensor — one mask per output bit."""
        masks = []
        for i in range(8):
            s = self._make_screen(result_bits=[1 if j == i else 0 for j in range(8)])
            masks.append(torch.from_numpy(s > 200).float())
        return torch.stack(masks)  # (8, H, W)

    def get_sample(self):
        a_int = torch.randint(0, 256, ()).item()
        b_int = torch.randint(0, 256, ()).item()
        op    = torch.randint(0, 8, ()).item()

        result_int = _compute_alu(a_int, b_int, op)

        uses_b = op < 5  # ADD, SUB, AND, OR, XOR use B; NOT/SHL/SHR do not
        inp = self._make_screen(
            a_bits=_int_to_bits(a_int, 8),
            b_bits=_int_to_bits(b_int, 8) if uses_b else None,
            opcode_bits=_int_to_bits(op, 3),
        )
        out = self._make_screen(result_bits=_int_to_bits(result_int, 8))
        return (torch.from_numpy(inp).float(), torch.from_numpy(out).float())

    def __iter__(self):
        while True:
            yield self.get_sample()

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False)


# ── ALU v2 ────────────────────────────────────────────────────────────────────

#  Operations (3-bit opcode)
#   0: ADD   A + B + carry_in  → result, carry_out, Z, N, V
#   1: SUB   A - B - carry_in  → result, carry_out (=~borrow), Z, N, V
#   2: AND   A & B
#   3: OR    A | B
#   4: XOR   A ^ B
#   5: NOT   ~A
#   6: RCL   rotate-left  A through carry
#   7: RCR   rotate-right A through carry
#
#  CTRL column (7 bits, MSB-top): [op2, op1, op0, carry_in, cond2, cond1, cond0]
#  OUT  column (13 bits, MSB-top): [res7..res0, carry_out, zero, neg, overflow, branch_taken]
#
#  Condition codes (3-bit cond):
#   0=EQ(Z=1)  1=NE(Z=0)  2=CS(C=1)  3=CC(C=0)
#   4=MI(N=1)  5=PL(N=0)  6=AL(always)  7=NV(never)

ALU2_OP_NAMES = ["ADD", "SUB", "AND", "OR", "XOR", "NOT", "RCL", "RCR"]
ALU2_COND_NAMES = ["EQ", "NE", "CS", "CC", "MI", "PL", "AL", "NV"]

# Curriculum stages — each tuple is the set of op indices active in that stage
ALU2_CURRICULUM = [
    [2, 3, 4, 5],        # stage 0: bitwise only (AND/OR/XOR/NOT)
    [0, 1, 2, 3, 4, 5],  # stage 1: + arithmetic (ADD/SUB)
    list(range(8)),       # stage 2: + shifts/rotates (RCL/RCR) — full set
]


def _compute_alu2(a_int, b_int, carry_in, op, cond):
    """Full ALU v2: returns (result, carry_out, branch_taken).

    Flags (zero, neg) are computed internally only to resolve branch_taken;
    they are NOT separate outputs — the NCA must internalise them.

    Condition codes:
        0=EQ(Z)  1=NE(!Z)  2=CS(C)  3=CC(!C)
        4=MI(N)  5=PL(!N)  6=AL     7=NV
    """
    ci = int(carry_in) & 1

    if op == 0:   # ADD with carry
        full      = a_int + b_int + ci
        result    = full & 0xFF
        carry_out = (full >> 8) & 1
    elif op == 1: # SUB with borrow: A - B - carry_in
        full      = a_int - b_int - ci
        result    = full & 0xFF
        carry_out = 0 if full < 0 else 1   # NOT borrow
    elif op == 2: result = a_int & b_int; carry_out = 0   # AND
    elif op == 3: result = a_int | b_int; carry_out = 0   # OR
    elif op == 4: result = a_int ^ b_int; carry_out = 0   # XOR
    elif op == 5: result = (~b_int) & 0xFF; carry_out = 0 # NOT (operates on B)
    elif op == 6: # RCL — rotate left through carry (operates on B)
        carry_out = (b_int >> 7) & 1
        result    = ((b_int << 1) | ci) & 0xFF
    else:         # RCR — rotate right through carry (operates on B)
        carry_out = b_int & 1
        result    = ((b_int >> 1) | (ci << 7)) & 0xFF

    zero  = int(result == 0)
    neg   = (result >> 7) & 1
    cond_map     = [zero, 1-zero, carry_out, 1-carry_out, neg, 1-neg, 1, 0]
    branch_taken = cond_map[cond & 7]

    return result, carry_out, branch_taken


def _int_to_bits_msb(n, width):
    return [(n >> (width - 1 - i)) & 1 for i in range(width)]


class ALU2Dataset(IterableDataset):
    """Full 8-bit ALU with carry and branch output on a 160×112 grid.

    Layout (160×112, r=4, step=10px)
    ---------------------------------
    x_A    = 16  : operand A (8 bits, MSB-top)
    x_B    = 32  : operand B (8 bits, blank for NOT/RCL/RCR)
    x_CTRL = 60  : op[2:0] + carry_in + cond[2:0]  — 7 bits
    x_OUT  = 144 : result[7:0] + carry_out + branch_taken  — 10 bits

    The NCA must internalise zero and negative flags as intermediate state
    to correctly compute branch_taken — no external logic is used.
    """

    # Default geometry — mirrors the constants in train_alu2.py
    W        = 96
    H        = 112
    R        = 4
    AMONG_SP = 2
    X_A      = 12
    X_B      = 24
    X_CTRL   = 48
    X_OUT    = 82

    def __init__(self, config=None, curriculum_stage=None):
        if config is not None:
            self.W        = int(config.W)
            self.H        = int(config.H)
            self.R        = int(config.r)
            self.AMONG_SP = int(config.among_sp)
            self.X_A      = int(config.x_a)
            self.X_B      = int(config.x_b)
            self.X_CTRL   = int(config.x_ctrl)
            self.X_OUT    = int(config.x_out)
        self.curriculum_stage = curriculum_stage   # None → full set

    def _screen(self, a_bits=None, b_bits=None, ctrl_bits=None, out_bits=None):
        return make_alu2_screen(
            self.H, self.W, self.R, self.AMONG_SP,
            self.X_A, self.X_B, self.X_CTRL, self.X_OUT,
            a_bits=a_bits, b_bits=b_bits,
            ctrl_bits=ctrl_bits, out_bits=out_bits,
        )

    def _active_ops(self):
        s = self.curriculum_stage
        if s is None or s >= len(ALU2_CURRICULUM):
            return list(range(8))
        return ALU2_CURRICULUM[s]

    def get_io_mask(self):
        inp = self._screen(a_bits=[1]*8, b_bits=[1]*8, ctrl_bits=[1]*7)
        out = self._screen(out_bits=[1]*10)
        return inp, out

    def get_output_bit_masks(self):
        """Returns (10, H, W) tensor — one binary mask per output bit."""
        masks = []
        for i in range(10):
            bits = [1 if j == i else 0 for j in range(10)]
            s = self._screen(out_bits=bits)
            masks.append(torch.from_numpy(s > 200).float())
        return torch.stack(masks)   # (10, H, W)

    def get_sample(self):
        a_int    = random.randint(0, 255)
        b_int    = random.randint(0, 255)
        carry_in = random.randint(0, 1)
        op       = random.choice(self._active_ops())
        cond     = random.randint(0, 7)

        # NOT/RCL/RCR are single-operand ops; they operate on B (the column
        # closer to ctrl/output). A is forced to zero and drawn as all-zero
        # bits so the unused side is explicit, not blank.
        single_input_op = op in (5, 6, 7)
        if single_input_op:
            a_int = 0

        result, cout, branch = _compute_alu2(a_int, b_int, carry_in, op, cond)

        ctrl_bits = (_int_to_bits_msb(op, 3)
                     + [carry_in]
                     + _int_to_bits_msb(cond, 3))        # 7 bits
        out_bits  = (_int_to_bits_msb(result, 8)
                     + [cout, branch])                   # 10 bits

        inp = self._screen(
            a_bits    = _int_to_bits_msb(a_int, 8),
            b_bits    = _int_to_bits_msb(b_int, 8),
            ctrl_bits = ctrl_bits,
        )
        out = self._screen(out_bits=out_bits)
        return torch.from_numpy(inp).float(), torch.from_numpy(out).float()

    def __iter__(self):
        while True:
            yield self.get_sample()

    def get_dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, shuffle=False)


class PoolLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, batch, losses):
        self.dataset.update(batch, losses)
        self.dataset.update(batch, losses)


class DynamicDataset(IterableDataset):
    def __init__(
        self,
        config,
        update_y: int,
        update_x: int,
        steps: int,
        stages: int = sys.maxsize,
    ):
        self.steps = steps
        self.counter = 0
        self.dataset = NCPUDataset(config)
        self.spacing = self.dataset.spacing
        self.W = self.dataset.W
        self.H = self.dataset.H
        self.r = self.dataset.r
        self.update_y = update_y
        self.update_x = update_x

        self.stage = 0
        self.stages = stages

    def get_io_mask(self):
        return self.dataset.get_io_mask()

    def get_output_bit_masks(self):
        return self.dataset.get_output_bit_masks()

    def get_sample(self):
        if self.counter >= self.steps and self.stage < self.stages:
            self.counter = 0
            spacing = self.spacing
            spacing_x = max(
                spacing[0] - self.update_x, self.r
            )  # stop points from moving outside the board
            spacing_y = max(
                spacing[1] - self.update_y, self.r + 2
            )  # stop points from moving outside the board
            self.spacing = (spacing_x, spacing_y)
            self.stage += 1

        self.dataset.W = self.W
        self.dataset.H = self.H
        self.dataset.r = self.r
        self.dataset.spacing = self.spacing
        ret = self.dataset.get_sample()
        self.counter += 1
        return ret

    def __iter__(self):
        while True:
            yield self.get_sample()

    def get_dataloader(self, batch_size):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,  # can't shuffle IterableDataset
        )


class MultiGateDataset(IterableDataset):

    GATE_NAMES = ["AND", "OR", "XOR", "NAND"]

    def __init__(self, config, nca_channels):

        self.W = config.W
        self.H = config.H
        self.r = config.r
        self.spacing = config.spacing
        self.nca_channels = nca_channels

        self.AND_Dataset = NCPUDataset(
            Namespace(
                W=config.W,
                H=config.H,
                r=config.r,
                spacing=config.spacing,
                balanced=config.balanced,
                sampler=sample_AND_gate,
            )
        )
        self.OR_Dataset = NCPUDataset(
            Namespace(
                W=config.W,
                H=config.H,
                r=config.r,
                spacing=config.spacing,
                balanced=config.balanced,
                sampler=sample_OR_gate,
            )
        )
        self.XOR_Dataset = NCPUDataset(
            Namespace(
                W=config.W,
                H=config.H,
                r=config.r,
                spacing=config.spacing,
                balanced=config.balanced,
                sampler=sample_XOR_gate,
            )
        )
        self.NAND_Dataset = NCPUDataset(
            Namespace(
                W=config.W,
                H=config.H,
                r=config.r,
                spacing=config.spacing,
                balanced=config.balanced,
                sampler=sample_NAND_gate,
            )
        )

        self.datasets = [self.AND_Dataset, self.OR_Dataset, self.XOR_Dataset, self.NAND_Dataset]
        self.counter = 0

    def _return_dataset(self):
        return self.datasets[self.counter % 4]

    def _code_dataset(self, inp, gate_idx):
        """Write a one-hot gate code into the last 4 channels of inp.

        inp shape: (nca_channels, H, W)
        Channels [-4], [-3], [-2], [-1] encode AND, OR, XOR, NAND respectively.
        Only the left half (columns 0..W//2) is set to 1 for the active gate;
        the right half and all inactive gate channels remain 0.
        """
        mid = self.W // 2
        # Zero out all code channels
        inp[-4:] = 256 // 2 
        # Set the left half of the active gate's channel to 1
        inp[-(4 - gate_idx), :, :mid] = 256
        return inp

    def get_output_bit_masks(self):
        """Returns (n_bits, H, W) float tensor — one binary mask per output circle."""
        return self._return_dataset().get_output_bit_masks()

    def get_io_mask(self):
        return self._return_dataset().get_io_mask()

    def get_sample(self):
        gate_idx = self.counter % 4
        dataset = self.datasets[gate_idx]
        left, right = dataset.get_sample()

        # Expand single-channel (H, W) input to (nca_channels, H, W)
        inp = left.unsqueeze(0).expand(self.nca_channels, self.H, self.W).clone()

        # Encode which gate is active into the last 4 channels
        inp = self._code_dataset(inp, gate_idx)

        return inp, right

    def __iter__(self):
        while True:
            yield self.get_sample()
            self.counter += 1

    def get_dataloader(self, batch_size):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,  # can't shuffle IterableDataset
        )
