from collections import deque

import torch
from torch.utils.data import DataLoader, IterableDataset

from ncpu.utils import make_io_screen


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


class NCPUDataset(IterableDataset):
    def __init__(
        self,
        W,
        H,
        r,
        spacing,
        margin,
        sampler,
        balanced=True,
    ):
        self.W = W
        self.H = H
        self.r = r
        self.spacing = spacing
        self.margin = margin
        self.sampler = sampler

        self.balanced = balanced
        self.prev_class = 0
        self.class_neg = 0

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
            margin=self.margin,
            left_input=left,
            right_input=torch.zeros_like(right),  # blank right input for input screen
        )

        out = make_io_screen(
            W=self.W,
            H=self.H,
            r=self.r,
            spacing=self.spacing,
            margin=self.margin,
            left_input=left,
            right_input=right,
        )

        return torch.from_numpy(inp), torch.from_numpy(out)

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
        inp, out = batch[0].detach(), batch[0].detach()
        max_idx = torch.argmax(losses)
        min_idx = torch.argmin(losses)
        for batch_i, sample in enumerate(batch):
            pool_i = self._counter_list.popleft()
            if min_idx == batch_i:
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
