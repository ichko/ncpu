from collections import deque
from typing import List
import sys

import sys
import torch
from argparse import Namespace
from torch.utils.data import DataLoader, IterableDataset

from ncpu.utils import make_alu_screen, make_io_screen


def sample_8bit_adder(*args):
    a = torch.randint(0, 2, size=(8,))
    b = torch.randint(0, 2, size=(8,))
    a_int = int("".join(map(str, a.tolist())), 2)
    b_int = int("".join(map(str, b.tolist())), 2)
    s_int = a_int + b_int
    out = torch.tensor(list(map(int, f"{s_int:09b}")))
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


# def sample_8bit_adder(*args):
#     return two_arg_sampler(lambda a, b: a != b)


class NCPUDataset(IterableDataset):
    def __init__(self, config):
        self.W = config.W
        self.H = config.H
        self.r = config.r
        self.spacing = config.spacing
        self.sampler = config.sampler

        self.balanced = config.balanced
        self.prev_class = 0
        self.class_neg = 0

    def get_io_mask(self):
        left, right = self.sampler()
        left_screen = make_io_screen(
            W=self.W,
            H=self.H,
            r=self.r,
            spacing=self.spacing,
            left_input=[],
            right_input=torch.ones_like(left),
        )
        right_screen = make_io_screen(
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
            screen = make_io_screen(
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

        inp = make_io_screen(
            W=self.W,
            H=self.H,
            r=self.r,
            spacing=self.spacing,
            left_input=left,
            # right_input=torch.zeros_like(right),
            right_input=[],
        )

        out = make_io_screen(
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
        inp[-4:] = 0.0
        # Set the left half of the active gate's channel to 1
        inp[-(4 - gate_idx), :, :mid] = inp.max()
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


def _compute_alu(a_int, b_int, cin, op):
    """Compute 8-bit ALU result and flags.

    Opcodes:
        0: ADD   A + B + cin
        1: SUB   A - B - cin  (carry = NOT borrow)
        2: AND   A & B
        3: OR    A | B
        4: XOR   A ^ B
        5: NOT   ~A
        6: SHL   A << 1, cin → LSB
        7: SHR   A >> 1, cin → MSB

    Returns (result_int, [carry_out, overflow, zero, negative])
    """
    overflow = 0
    carry_out = 0

    if op == 0:  # ADD
        full = a_int + b_int + cin
        result = full & 0xFF
        carry_out = int(full > 0xFF)
        overflow = int(((a_int ^ result) & (b_int ^ result) & 0x80) != 0)
    elif op == 1:  # SUB
        full = a_int - b_int - cin
        result = full & 0xFF
        carry_out = int(full >= 0)  # carry = NOT borrow
        overflow = int(((a_int ^ b_int) & (a_int ^ result) & 0x80) != 0)
    elif op == 2:  # AND
        result = a_int & b_int
    elif op == 3:  # OR
        result = a_int | b_int
    elif op == 4:  # XOR
        result = a_int ^ b_int
    elif op == 5:  # NOT A
        result = (~a_int) & 0xFF
    elif op == 6:  # SHL
        carry_out = (a_int >> 7) & 1
        result = ((a_int << 1) | cin) & 0xFF
    else:  # SHR
        carry_out = a_int & 1
        result = ((cin << 7) | (a_int >> 1)) & 0xFF

    zero = int(result == 0)
    negative = (result >> 7) & 1
    return result, [carry_out, overflow, zero, negative]


class ALUDataset(IterableDataset):
    """8-bit ALU dataset using the make_alu_screen layout."""

    def __init__(self, config):
        self.W = config.W
        self.H = config.H
        self.r = config.r
        self.side = config.side
        self.among = config.among

    def _make_screen(
        self, a=None, b=None, carry_in=None, opcode=None, result=None, flags=None
    ):
        return make_alu_screen(
            self.H,
            self.W,
            self.r,
            self.side,
            self.among,
            a=a,
            b=b,
            carry_in=carry_in,
            opcode=opcode,
            result=result,
            flags=flags,
        )

    def get_io_mask(self):
        inp = self._make_screen(a=[1] * 8, b=[1] * 8, carry_in=[1], opcode=[1] * 3)
        out = self._make_screen(result=[1] * 8, flags=[1] * 4)
        return inp, out

    def get_output_bit_masks(self):
        """Returns (12, H, W) tensor — one mask per output bit (8 result + 4 flags)."""
        masks = []
        for i in range(8):
            s = self._make_screen(result=[1 if j == i else 0 for j in range(8)])
            masks.append(torch.from_numpy(s > 200).float())
        for i in range(4):
            s = self._make_screen(flags=[1 if j == i else 0 for j in range(4)])
            masks.append(torch.from_numpy(s > 200).float())
        return torch.stack(masks)  # (12, H, W)

    def get_sample(self):
        a_int = torch.randint(0, 256, ()).item()
        b_int = torch.randint(0, 256, ()).item()
        cin = torch.randint(0, 2, ()).item()
        op = torch.randint(0, 8, ()).item()

        result_int, flags = _compute_alu(a_int, b_int, cin, op)

        inp = self._make_screen(
            a=_int_to_bits(a_int, 8),
            b=_int_to_bits(b_int, 8),
            carry_in=[cin],
            opcode=_int_to_bits(op, 3),
        )
        out = self._make_screen(
            result=_int_to_bits(result_int, 8),
            flags=flags,
        )
        return (torch.from_numpy(inp).float(), torch.from_numpy(out).float())

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
