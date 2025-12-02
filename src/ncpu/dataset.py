import torch
from ncpu.utils import make_io_screen
from torch.utils.data import IterableDataset, DataLoader
import numpy as np


def sample_and_io():
    a = torch.randint(0, 2, size=(1,)).item()
    b = torch.randint(0, 2, size=(1,)).item()
    c = a & b
    return a, b, c


class NCPUDataset(IterableDataset):
    def __init__(self, W, H, r, spacing, margin):
        self.W = W
        self.H = H
        self.r = r
        self.spacing = spacing
        self.margin = margin

    def __iter__(self):
        while True:
            left_input = np.random.randint(0, 2**16)
            right_input = np.random.randint(0, 2**16)
            a, b, right_input = sample_and_io()
            left_input = a << 1 | b

            inp = make_io_screen(
                W=self.W,
                H=self.H,
                r=self.r,
                spacing=self.spacing,
                margin=self.margin,
                left_input=left_input,
                right_input=0,
            )
            out = make_io_screen(
                W=self.W,
                H=self.H,
                r=self.r,
                spacing=self.spacing,
                margin=self.margin,
                left_input=0,
                right_input=right_input,
            )

            yield inp, out

    def get_dataloader(self, batch_size):
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,  # can't shuffle IterableDataset
        )
