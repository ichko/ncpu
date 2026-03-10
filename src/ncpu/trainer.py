from typing import Optional

import mediapy as media
import numpy as np
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader

from ncpu.base_trainer import BaseTrainer
from ncpu.nca import NeuralCAv2
from ncpu.utils import (
    add_gaussian_noise,
    meshgrid_xy,
    print_tensor,
    sequence_batch_to_html_gifs,
)


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

    def save_checkpoint(self, name : str = "", steps = True, timestamp = True) -> str:
        if steps:
            name = f"{name}{self.optim_steps:06d}"
        else:
            name = f"{name}"
        path = self.checkpointer.make(custom_string = name, timestamp = timestamp)
        torch.save(self.nca.state_dict(), path)

    def optim_step_v2(self, steps):
        optim = torch.optim.Adam(self.nca.parameters(), lr=self.lr)
        batch = next(self.dataset_iter)

        inp, out = batch

        inp = inp / 128.0 - 1.0
        out = out / 128.0 - 1.0

        if self.gaussian_noise > 0:
            inp = add_gaussian_noise(inp, 0, self.gaussian_noise)

        inp = inp.to(self.device)
        out = out.to(self.device)

        first_state = self._implant_input(inp)

        forward_steps = steps
        if isinstance(steps, (tuple, list)):
            forward_steps = np.random.randint(steps[0], steps[1])

        rollout = self.nca.forward(first_state, steps=forward_steps)
        nca_out = rollout[:, -1, 0]

        # white_mask = (out > 0.5).float()
        # black_mask = 1 - white_mask

        # batch_loss = F.mse_loss(nca_out, out, reduction="none")
        # white_loss = batch_loss * white_mask
        # black_loss = batch_loss * black_mask

        # masks_sum = white_mask.sum() + black_mask.sum()
        # white_weight = black_mask.sum() / masks_sum
        # black_weight = white_mask.sum() / masks_sum

        # loss = (white_loss * white_weight + black_loss * black_weight).mean()

        # loss = F.mse_loss(nca_out * self.output_mask, out * self.output_mask)

        # loss = F.mse_loss(nca_out, out)

        N = min(5, rollout.shape[1])
        nca_outs = rollout[:, -N:, 0]
        out_rep = torch.unsqueeze(out, dim=1).repeat(1, N, 1, 1)
        loss = F.mse_loss(nca_outs, out_rep)

        if torch.is_grad_enabled():
            self.learning_step += 1
            optim.zero_grad()
            loss.backward()
            optim.step()

            self.log_metrics(
                loss=loss.item(),
                # white_loss=white_loss.sum().item(),
                # black_loss=black_loss.sum().item(),
            )

        if hasattr(self.dataloader, "update"):
            self.dataloader.update((nca_out, out.detach()), loss)

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

    def optim_step(self, steps):
        batch = next(self.dataset_iter)

        inp, out = batch

        inp = inp / 255.0
        out = out / 255.0

        if self.gaussian_noise > 0:
            inp = add_gaussian_noise(inp, 0, self.gaussian_noise)

        inp = inp.to(self.nca.device)
        out = out.to(self.nca.device)

        first_state = self._implant_input(inp)

        forward_steps = steps
        if isinstance(steps, (tuple, list)):
            forward_steps = np.random.randint(steps[0], steps[1])

        rollout = self.nca.forward(first_state, steps=forward_steps)
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
