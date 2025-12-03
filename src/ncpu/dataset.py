import torch
from ncpu.utils import make_io_screen
from torch.utils.data import IterableDataset, DataLoader
import numpy as np


def sample_conjunction_input_output(*args):
    a = torch.randint(0, 2, size=(1,)).item()
    b = torch.randint(0, 2, size=(1,)).item()
    right = a & b
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

    def __iter__(self):
        while True:
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

            yield inp, out

    def get_dataloader(self, batch_size):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,  # can't shuffle IterableDataset
        )
