from collections import deque
from typing import List

import torch
from torch.utils.data import DataLoader, IterableDataset

from ncpu.utils import make_io_screen


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
                if self.ds_index < len(self.datasets)
                else self.ds_index
            )

        self.W = self.datasets[self.ds_index].W
        self.H = self.datasets[self.ds_index].H
        self.r = self.datasets[self.ds_index].r
        ret = self.datasets[self.ds_index].get_sample()
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


class PoolLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, batch, losses):
        self.dataset.update(batch, losses)
        self.dataset.update(batch, losses)
