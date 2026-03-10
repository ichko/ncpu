from typing import Optional

import mediapy as media
import numpy as np
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ncpu.checkpoints import CheckpointTracker
from ncpu.base_trainer import BaseTrainer
from ncpu.nca import NeuralCAv2
from ncpu.utils import (
    add_gaussian_noise,
    meshgrid_xy,
    print_tensor,
    sequence_batch_to_html_gifs,
)

from typing import Optional, Callable, Sequence, Union


class NCPUTrainer(BaseTrainer):
    _exclude_from_pickle = {"dataloader", "ds", "dataset_iter"}

    def __init__(
        self,
        nca: NeuralCAv2,
        dataloader: DataLoader,
        lr: float,
        gaussian_noise: float,
        stop_loss: Optional[float] = None,
    ):
        super().__init__()
        
        self.nca = nca
        self.to(nca.device)
        self.dataloader = dataloader
        self.ds = dataloader.dataset
        self.dataset_iter = iter(self.dataloader)
        self.gaussian_noise = gaussian_noise
        self.stop_loss = stop_loss
        self.lr = lr

        self.optim = torch.optim.Adam(self.nca.parameters(), lr=self.lr)
        left_mask, right_mask = self.ds.get_io_mask()
        self.inp_mask = torch.tensor(left_mask).to(self.nca.device)
        self.out_mask = torch.tensor(right_mask).to(self.nca.device)

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
        first_state = torch.zeros(bs, self.nca.channels, self.ds.H, self.ds.W)
        first_state = first_state.to(self.nca.device)
        first_state[:, 0] = inp  # implant in the first channel
        xx, yy = meshgrid_xy(self.ds.H, self.ds.W, device=self.device)
        first_state[:, -2] = xx
        first_state[:, -1] = yy
        return first_state

    def optim_step(self, steps : Union[int, Sequence[int]], clip_max : int  = 128, clip_min : int = -128, loss : Callable = lambda rollout, target : F.mse_loss(rollout[:, -1, 0], target, reduction="none").mean()):
        batch = next(self.dataset_iter)
        inp, out = batch

        norm_mean = torch.round((inp/clip_max).max()) # Piotr: this is not the best, it assumes that we pass max value in input 
        print((inp / clip_max).max() , norm_mean, torch.floor(norm_mean / 2))
        inp = inp / clip_max - torch.floor(norm_mean / 2)
        out = out / clip_max - torch.floor(norm_mean / 2)

        if self.gaussian_noise > 0:
            inp = add_gaussian_noise(inp, 0, self.gaussian_noise, clip_min, clip_max)

        inp = inp.to(self.device)
        out = out.to(self.device)

        forward_steps = steps
        if isinstance(steps, (tuple, list)):
            forward_steps = np.random.randint(steps[0], steps[1])

        first_state = self._implant_input(inp)
        rollout = self.nca.forward(first_state, steps=forward_steps)

        # Piotr: if we are going to change loss, then lets do it in proper way
        #        so we can document different losses as lambdas/functions/classes 
        #        no strong feelings which one we should use, but definitelly we should 
        #        avoid changing trainer if not necessary
        #        LETS DO NOT WORK AGAINST THE CODE, BUT MAKE THE CODE WORK FOR US 
        loss = loss(rollout, out)

        nca_out = rollout[:, -1, 0]

        if torch.is_grad_enabled():
            self.learning_step += 1
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.log_metrics(
                loss=loss.item(),
                # white_loss=white_loss.sum().item(),
                # black_loss=black_loss.sum().item(),
            )

        info = {
            "loss": loss.item(),
            "inp": inp,
            "out": out,
            "nca_out": nca_out,
            "rollout": rollout,
        }

        return info

    def load_checkpoint(self, name : str = "") -> None:
        path = self.checkpointer.get(name)
        self.nca.load_state_dict(torch.load(path))

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
        sequence_batch_to_html_gifs(
            rollout, columns=to_show, width=display_size, height=display_size, fps=10
        )

    # piotr: can we remove this code? @iliya
    # def display_optim_step(self, info):
    #     fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    #     ax.scatter(
    #         range(len(self.history)), [h["loss"] for h in self.history], s=1, alpha=0.9
    #     )
    #     ax.set_yscale("log")
    #     plt.close()

    #     to_show = 5
    #     steps = info["rollout"].shape[1] - 1

    #     # rollout: (B, T, C, H, W)

    #     rollout = info["rollout"][:to_show, :, : self.config.visual_channels]
    #     # rollout = impact_frames(rollout, ts=[0, steps], ns=[5, 20])
    #     # rollout = rollout[:, :, :self.config.visual_channels]
    #     rollout = rollout[
    #         :, :, : self.config.visual_channels
    #     ]  # only show first 3 channels for visualization

    #     stats = f"""
    #         ```
    #         optim step: {self.learning_steps}
    #         frame  : {tensor_summary(info["rollout"])}
    #         weights: {tensor_summary(self.parameters())}
    #         grads  : {tensor_summary(info["grads"])}
    #         mass: {info['final_frame'].sum().item():.4f}
    #         ```
    #         """

    #     return pn.Row(
    #         pn.Column(
    #             pn.pane.Matplotlib(
    #                 fig, format="svg", width=500, height=250, tight=True
    #             ),
    #             pn.pane.HTML(
    #                 sequence_batch_to_html_gifs(
    #                     rollout,
    #                     columns=8,
    #                     width=120,
    #                     height=120,
    #                     fps=20,
    #                     return_html=True,
    #                 )
    #             ),
    #             image_row([f[:to_show] for f in info["frames"]], columns=to_show),
    #             image_row(
    #                 [f[:to_show] for f in info["noised_frames"]], columns=to_show
    #             ),
    #         ),
    #         pn.Column(
    #             stats,
    #             self.display_mass(info),
    #         ),
    #     )
