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

class NCPUEval:
    def __init__(self, nca: NeuralCA, dataloader, lr, apply_gaussian_noise=False):
        super().__init__()
        self.nca = nca
        self.dataloader = dataloader
        self.ds = dataloader.dataset
        self.dataset_iter = iter(self.dataloader)
        self.history = []
        self.apply_gaussian_noise = apply_gaussian_noise

    def _inplant_input(self, inp):
        bs = inp.shape[0]
        first_state = torch.zeros(bs, self.nca.channels, self.ds.H, self.ds.W)
        first_state = first_state.to(self.nca.device)
        first_state[:, 0] = inp  # inplant in the first channel
        return first_state

    def eval(self, steps):
        batch = next(self.dataset_iter)

        inp, out = batch
        # only apply normalization and noise
        # if image is fresh and not reused from pool
        # Compute per-sample max
        inp_max = inp.amax(dim=(1, 2), keepdim=True)
        out_max = out.amax(dim=(1, 2), keepdim=True)
        inp = inp / (inp_max + 1e-8)
        out = out / (out_max + 1e-8)

        # this has to be applied during dataset creation, or it messes with pool
        if self.apply_gaussian_noise:
            inp = add_gaussian_noise(inp, 0, 0.2)

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

class NCPUTrainer:
    def __init__(
        self,
        nca: NeuralCA,
        dataloader,
        lr,
        gaussian_noise,
    ):
        self.nca = nca
        self.dataloader = dataloader
        self.ds = dataloader.dataset
        self.dataset_iter = iter(self.dataloader)
        self.optim = torch.optim.Adam(self.nca.parameters(), lr=lr)
        self.gaussian_noise = gaussian_noise

        self.metrics = []
        self.optim_steps = 0

    def sanity_check(self):
        print("Sanity check...")

        inp = torch.randn(
            2, self.nca.channels, self.ds.W, self.ds.H
        ).to(self.nca.device)
        out = self.nca.forward(inp, steps=10)

        print("  forward:", inp.shape, "->", out.shape)

        batch = next(iter(self.dataloader))
        print(f"  batch: {len(batch)}")
        inp, out = batch
        print("  dataloader:", inp.shape, "->", out.shape)

        first_state = self._implant_input(inp).to(self.nca.device)
        print("  first_state:", first_state.shape)

        rollout = self.nca.forward(first_state, steps=10)
        print("  rollout:", rollout.shape)

        with torch.no_grad():
            loss = self.optim_step(steps=10)
            print("  loss:", loss["loss"])

        print("Sanity check completed successfully")

    def _implant_input(self, inp):
        bs = inp.shape[0]
        first_state = torch.zeros(bs, self.nca.channels, self.ds.H, self.ds.W)
        first_state = first_state.to(self.nca.device)
        first_state[:, 0] = inp  # implant in the first channel
        return first_state

    # TODO: not yet sure if that works or not <- commenting out for now
    # please do not remove // Piotr
    #
    # def _adaptive_weights(self, out):
    #     s = out.sum(dim=(1, 2))
    #     white_weights = torch.where(s > 0.5, torch.tensor(0.7), torch.tensor(0.3))

    #     # Create opposite vector
    #     black_weights = 1.0 - white_weights
    #     return white_weights, black_weights

    def save_checkpoint(self):
        it = self.optim_steps
        t = datetime.now().strftime("%Y%m%d_%H%M%S")

        path = f"{CHECKPOINT_DIR}/ncpu_{it:06d}_{t}.pth"
        torch.save(self.nca.state_dict(), path)

        return path

    def optim_step(self, steps):
        self.optim_steps += 1
        batch = next(self.dataset_iter)

        inp, out = batch
        
        # only apply normalization and noise
        # if image is fresh and not reused from pool
        # Compute per-sample max
        inp_max = inp.amax(dim=(1, 2), keepdim=True)
        out_max = out.amax(dim=(1, 2), keepdim=True)
        inp = inp / (inp_max + 1e-8)
        out = out / (out_max + 1e-8)

        if self.gaussian_noise > 0:
            inp = add_gaussian_noise(inp, 0, self.gaussian_noise)

        inp = inp.to(self.nca.device)
        out = out.to(self.nca.device)

        first_state = self._implant_input(inp)
        if isinstance(steps, (tuple, list)):
            steps = np.random.randint(steps[0], steps[1])

        rollout = self.nca.forward(first_state, steps=steps)
        nca_out = rollout[:, -1, 0]

        white_mask = (out > 0.5).float()
        black_mask = 1 - white_mask

        batch_loss = F.mse_loss(nca_out, out, reduction="none")
        white_loss = batch_loss * white_mask
        black_loss = batch_loss * black_mask

        masks_sum = white_mask.sum() + black_mask.sum()
        white_weight = black_mask.sum() / masks_sum
        black_weight = white_mask.sum() / masks_sum

        loss = (white_loss * white_weight + black_loss * black_weight).mean()

        if torch.is_grad_enabled():
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        metrics = {
            "loss": loss.item(),
            "white_loss": white_loss.sum().item(),
            "black_loss": black_loss.sum().item(),
        }
        self.metrics.append(metrics)

        if hasattr(self.dataloader, "update"):
            self.dataloader.update((nca_out, out.detach()), loss)

        info = {
            "loss": loss.item(),
            "metrics": metrics,
            "inp": inp,
            "out": out,
            "nca_out": nca_out,
            "rollout": rollout,
        }

        return info

    def display_optim_step(self, info):
        fig, ax = plt.subplots(figsize=(8, 3))
        loss = [h["loss"] for h in self.metrics]
        ax.scatter(range(len(loss)), loss, s=1)
        ax.set_yscale("log")
        plt.close(fig)
        display(fig)

        print_tensor("inp    ", info["inp"])
        print_tensor("out    ", info["out"])
        print_tensor("nca_out", info["nca_out"])

        to_show = 8
        inp = info["inp"][:to_show]
        out = info["out"][:to_show]
        nca_out = info["nca_out"][:to_show]
        rollout = info["rollout"][:to_show]
        io = torch.cat([inp, out, nca_out], dim=0)

        media.show_images(io.detach().cpu(), columns=to_show, width=150, height=150)
        sequence_batch_to_html_gifs(
            rollout, columns=to_show, width=150, height=150, fps=10
        )
