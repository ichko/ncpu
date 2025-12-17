import torch
from ncpu.utils import make_io_screen
from torch.utils.data import IterableDataset, DataLoader
from collections import deque


def sample_AND_gate(*args):
    a = torch.randint(0, 2, size=(1,)).item()
    b = torch.randint(0, 2, size=(1,)).item()
    right = a & b
    left = a << 1 | b
    return left, right


def sample_OR_gate(*args):
    a = torch.randint(0, 2, size=(1,)).item()
    b = torch.randint(0, 2, size=(1,)).item()
    right = a | b
    left = a << 1 | b
    return left, right


def sample_NOR_gate(*args):
    a = torch.randint(0, 2, size=(1,)).item()
    b = torch.randint(0, 2, size=(1,)).item()
    right = not (a | b)
    left = a << 1 | b
    return left, right


def sample_NAND_gate(*args):
    a = torch.randint(0, 2, size=(1,)).item()
    b = torch.randint(0, 2, size=(1,)).item()
    right = not (a & b)
    left = a << 1 | b
    return left, right


def sample_XOR_gate(*args):
    a = torch.randint(0, 2, size=(1,)).item()
    b = torch.randint(0, 2, size=(1,)).item()
    right = a != b
    left = a << 1 | b
    return left, right


class NCPUDataset(IterableDataset):
    def __init__(
        self,
        W,
        H,
        r,
        small_r,
        spacing,
        margin,
        sampler,
        bit_length,
        balanced=True,
    ):
        self.W = W
        self.H = H
        self.r = r
        self.small_r = small_r
        self.spacing = spacing
        self.margin = margin
        self.sampler = sampler
        self.bit_length = bit_length

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
            small_r=self.small_r,
            spacing=self.spacing,
            margin=self.margin,
            left_input=left,
            right_input=0,  # intentionally left at 0
            bit_size_left=2,
            bit_size_right=1,
        )

        out = make_io_screen(
            W=self.W,
            H=self.H,
            r=self.r,
            small_r=self.small_r,
            spacing=self.spacing,
            margin=self.margin,
            left_input=left,
            right_input=right,
            bit_size_left=2,
            bit_size_right=1,
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
