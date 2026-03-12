from typing import Optional

import mediapy as media
import numpy as np
import panel as pn
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ncpu.base_trainer import BaseTrainer
from ncpu.loss import output_masked_rollout_loss
from ncpu.nca import NeuralCA
from ncpu.normalizers import normalize_neg1_to_1
from ncpu.utils import (
    add_gaussian_noise,
    print_tensor,
    sequence_batch_to_html_gifs,
    tensor_to_video_pane,
)

from typing import Optional, Callable, Sequence, Union


class NCPUTrainer(BaseTrainer):
    _exclude_from_pickle = {"dataloader", "ds", "dataset_iter", "optim"}

    def __init__(
        self,
        nca: NeuralCA,
        dataloader: DataLoader,
        lr: float,
        gaussian_noise: float,
        grad_clip: Optional[float] = 1.0,
        stop_loss: Optional[float] = None,
        checkpoint_pattern: Optional[str] = None,
        loss_fn=output_masked_rollout_loss,
        input_implant_type="first",
        normalize_fn=normalize_neg1_to_1,
    ):
        super().__init__(
            **(
                {}
                if checkpoint_pattern is None
                else {"checkpoint_pattern": checkpoint_pattern}
            )
        )
        self.nca = nca
        self.to(nca.device)
        self.dataloader = dataloader
        self.ds = dataloader.dataset
        self.dataset_iter = iter(self.dataloader)
        self.gaussian_noise = gaussian_noise
        self.stop_loss = stop_loss
        self.lr = lr
        self.grad_clip = grad_clip
        self.optim = torch.optim.Adam(self.nca.parameters(), lr=self.lr)

        self.optim = torch.optim.Adam(self.nca.parameters(), lr=self.lr)
        left_mask, right_mask = self.ds.get_io_mask()
        self.inp_mask = torch.tensor(left_mask).to(self.nca.device)
        self.out_mask = torch.tensor(right_mask).to(self.nca.device)
        # binary float mask: 1.0 at output circle pixels, 0.0 elsewhere
        self.out_mask_binary = (self.out_mask > 128).float()
        # per-bit masks: (n_bits, H, W)
        self.bit_masks = self.ds.get_output_bit_masks().to(self.nca.device)
        self.loss_fn = loss_fn
        self.input_implant_type = input_implant_type
        self.normalize_fn = normalize_fn

    def sanity_check(self):
        print("Sanity check...")

        inp = torch.randn(2, self.nca.channels, self.ds.W, self.ds.H).to(self.device)
        out = self.nca.forward(inp, steps=10)

        print("  forward:", inp.shape, "->", out.shape)

        batch = next(iter(self.dataloader))
        print(f"  batch: {len(batch)}")
        inp, out = batch
        print("  dataloader:", inp.shape, "->", out.shape)

        first_state = self._implant_input(inp).to(self.device)
        print("  first_state:", first_state.shape)

        rollout = self.nca.forward(first_state, steps=10)
        print("  rollout:", rollout.shape)

        with torch.no_grad():
            loss = self.optim_step(steps=10)
            print("  loss:", loss["loss"])

        print("Sanity check completed successfully")

    def _implant_input(self, inp):
        bs = inp.shape[0]
        if self.input_implant_type == "all":
            first_state = (
                inp.unsqueeze(1)
                .expand(bs, self.nca.channels, self.ds.H, self.ds.W)
                .clone()
            )
        else:
            first_state = torch.zeros(bs, self.nca.channels, self.ds.H, self.ds.W)
            first_state[:, 0] = inp
        first_state = first_state.to(self.nca.device)
        return first_state

    def optim_step(self, steps, return_rollout=False):
        batch = next(self.dataset_iter)
        inp, out = batch

        inp = self.normalize_fn(inp)
        out = self.normalize_fn(out)

        if self.gaussian_noise > 0:
            inp = add_gaussian_noise(inp, 0, self.gaussian_noise)

        inp = inp.to(self.device)
        out = out.to(self.device)

        forward_steps = steps
        if isinstance(steps, (tuple, list)):
            forward_steps = np.random.randint(steps[0], steps[1])

        first_state = self._implant_input(inp)
        rollout = self.nca.forward(first_state, steps=forward_steps)

        nca_out = rollout[:, -1, 0]
        loss = self.loss_fn(rollout, out, inp=inp, mask=self.out_mask_binary)

        # num_valid_bits: mean correct bits per sample (out of n_bits)
        with torch.no_grad():
            bm = self.bit_masks  # (n_bits, H, W)
            bm_sum = bm.sum(dim=(-1, -2))  # (n_bits,)
            # mean activation per bit circle: (B, n_bits)
            nca_bit = (nca_out.unsqueeze(1) * bm.unsqueeze(0)).sum(
                dim=(-1, -2)
            ) / bm_sum
            tgt_bit = (out.unsqueeze(1) * bm.unsqueeze(0)).sum(dim=(-1, -2)) / bm_sum
            num_valid_bits = (
                ((nca_bit > 0) == (tgt_bit > 0)).float().sum(dim=1).mean().item()
            )

        if torch.is_grad_enabled():
            self.learning_step += 1
            self.optim.zero_grad()
            loss.backward()
            grad_norm = (
                torch.nn.utils.clip_grad_norm_(self.nca.parameters(), self.grad_clip)
                if self.grad_clip is not None
                else None
            )
            self.optim.step()

            self.log_metrics(
                loss=loss.item(),
                grad_norm=grad_norm.item() if grad_norm is not None else None,
                num_valid_bits=num_valid_bits,
            )

        info = {
            "loss": loss.item(),
            "num_valid_bits": num_valid_bits,
            "inp": inp,
            "out": out,
            "nca_out": nca_out,
            "rollout": rollout.detach().cpu() if return_rollout else None,
        }

        return info

    def display_optim_step(self, info, display_size=64, to_show=8):
        fig, ax = plt.subplots(figsize=(8, 3))
        loss = [h["loss"] for h in self.metrics]
        ax.scatter(range(len(loss)), loss, s=1)
        ax.set_yscale("log")
        plt.close(fig)
        display(fig)

        print_tensor("inp    ", info["inp"])
        print_tensor("out    ", info["out"])
        print_tensor("nca_out", info["nca_out"])

        inp = info["inp"][:to_show]
        out = info["out"][:to_show]
        nca_out = info["nca_out"][:to_show]
        rollout = info["rollout"][:to_show]
        # read_only_channel = rollout[:to_show, -1, -1]
        io = torch.cat([inp, out, nca_out], dim=0)

        print(f"display size: {(display_size,display_size)}")
        media.show_images(
            io.detach().cpu(),
            columns=to_show,
            width=display_size,
            height=display_size,
            cmap="viridis",
            vmin=-1,
            vmax=1,
        )
        vid = tensor_to_video_pane(
            rollout, nrow=to_show, zoom=2, padding=1, fps=10, format="gif"
        )
        return pn.Column(vid)
