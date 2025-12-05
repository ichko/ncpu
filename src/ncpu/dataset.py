import torch
from ncpu.utils import make_io_screen
from torch.utils.data import IterableDataset, DataLoader
from collections import deque
import numpy as np


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
    def __init__(self, W, H, r, spacing, margin, sampler):
        self.W = W
        self.H = H
        self.r = r
        self.spacing = spacing
        self.margin = margin
        self.sampler = sampler

    def getSample(self):
        left, right = self.sampler()

        inp = make_io_screen(
            W=self.W,
            H=self.H,
            r=self.r,
            spacing=self.spacing,
            margin=self.margin,
            left_input=left,
            right_input=0,
        )
        out = make_io_screen(
            W=self.W,
            H=self.H,
            r=self.r,
            spacing=self.spacing,
            margin=self.margin,
            left_input=0,
            right_input=right,
        )

        return torch.from_numpy(inp), torch.from_numpy(out) 

    def __iter__(self):
        while True:
            yield self.getSample()

    def __getitem__(self, idx):
        return self.getSample()

    def __len__(self):
        return self.length

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

    def __iter__(self):
        dataset_iter = iter(self.dataset)
        while True:
            if not self.pool[self._counter]:
                self.pool[self._counter] = next(dataset_iter)
            val = self.pool[self._counter]
            self._counter_list.append(val)
            self._counter += 1
            self._counter = self._counter % self.pool_size
            yield val

    def update(self, batch, losses):
        min_idx = torch.argmax(losses)
        for i ,sample in enumerate(batch):
            i = self._counter_list.popleft()
            if min_idx == i:
                self.pool[i] = next(dataset_iter) # prune worst results by re 
            else:
                self.pool[i] = sample

    def get_dataloader(self, batch_size):
        return Poolloader(
            self,
            batch_size=batch_size,
            shuffle=False,  # can't shuffle IterableDataset
        )

class Poolloader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, batch, losses):
        self.dataset.update(batch, losses)