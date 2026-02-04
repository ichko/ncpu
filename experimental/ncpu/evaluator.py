
import torch
from torch.nn import functional as F
from ncpu.model import NeuralCA
from ncpu.utils import add_gaussian_noise
import numpy as np
import os
from datetime import datetime

import mediapy as media
import numpy as np
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from torch.nn import functional as F

from ncpu.model import NeuralCA
from ncpu.utils import add_gaussian_noise, print_tensor, sequence_batch_to_html_gifs

CHECKPOINT_DIR = "./checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


#[WIP]
# This is all work in progress
class NCPUEval:

    #[WIP]
    def __init__(self, nca: NeuralCA, dataloader, lr, apply_gaussian_noise=False):
        super().__init__()
        self.nca = nca
        self.dataloader = dataloader
        self.ds = dataloader.dataset
        self.dataset_iter = iter(self.dataloader)
        self.history = []
        self.apply_gaussian_noise = apply_gaussian_noise

    #[WIP]
    def _inplant_input(self, inp):
        bs = inp.shape[0]
        first_state = torch.zeros(bs, self.nca.channels, self.ds.H, self.ds.W)
        first_state = first_state.to(self.nca.device)
        first_state[:, 0] = inp  # inplant in the first channel
        return first_state

    #[WIP]
    def eval(self, steps):
        batch = next(self.dataset_iter)

        inp, out = batch
        # only apply normalization and noise
        # if image is fresh and not reused from pool
        # Compute per-sample max
        inp = inp / 255.0
        out = out / 255.0
        # inp_max = inp.amax(dim=(1, 2), keepdim=True)
        # out_max = out.amax(dim=(1, 2), keepdim=True)
        # inp = inp / (inp_max + 1e-8)
        # out = out / (out_max + 1e-8)

        # this has to be applied during dataset creation, or it messes with pool
        if self.gaussian_noise > 0:
            inp = add_gaussian_noise(inp, 0, self.gaussian_noise)

        inp = inp.to(self.nca.device)
        out = out.to(self.nca.device)

        first_state = self._inplant_input(inp)
        if isinstance(steps, (tuple, list)):
            steps = np.random.randint(steps[0], steps[1])

        rollout = self.nca.forward(first_state, steps=steps)
        nca_out = rollout[:, -1, 0]

        white_mask = (out > 0.5).float()
        black_mask = 1 - white_mask

        batch_loss = F.mse_loss(nca_out, out, reduction="none")
        # batch_loss = F.binary_cross_entropy_with_logits(nca_out, out, reduction="none")
        white_loss = batch_loss * white_mask
        black_loss = batch_loss * black_mask

        masks_sum = white_mask.sum() + black_mask.sum()
        white_weight = black_mask.sum() / masks_sum
        black_weight = white_mask.sum() / masks_sum

        loss = (white_loss * white_weight + black_loss * black_weight).mean()

        track = {
            "loss": loss,
            "white_loss": white_loss,
            "black_loss": black_loss,
        }
        self.history.append(track)

        return {
            "loss": loss,
            "track": track,
            "inp": inp,
            "out": out,
            "nca_out": nca_out,
            "rollout": rollout,
        }
