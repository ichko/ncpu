import torch
from torch.nn import functional as F
from ncpu.model import NeuralCA
import numpy as np

from ncpu.persisted_list import PersistedList
from ncpu.utils import add_gaussian_noise


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

        self.optim_snapshots = PersistedList.new_list("./training_history")
        self.metrics = []

    def sanity_check(self):
        print("Sanity check...")

        inp = torch.randn(2, self.nca.channels, self.ds.W, self.ds.H).to(
            self.nca.device
        )
        out = self.nca.forward(inp, steps=10)

        print("  forward:", inp.shape, "->", out.shape)

        batch = next(iter(self.dataloader))
        print(f"  batch: {len(batch)}")
        inp, out = batch
        print("  dataloader:", inp.shape, "->", out.shape)

        first_state = self._inplant_input(inp).to(self.nca.device)
        print("  first_state:", first_state.shape)

        rollout = self.nca.forward(first_state, steps=10)
        print("  rollout:", rollout.shape)

        with torch.no_grad():
            loss = self.optim_step(steps=10, save_history=False)
            print("  loss:", loss["loss"].item())

        print("Sanity check completed successfully")

    def _inplant_input(self, inp):
        bs = inp.shape[0]
        first_state = torch.zeros(bs, self.nca.channels, self.ds.H, self.ds.W)
        first_state = first_state.to(self.nca.device)
        first_state[:, 0] = inp  # inplant in the first channel
        return first_state

    # TOOD: not yet sure if that works or not <- commenting out for now
    # please do not remove // Piotr
    #
    # def _adaptive_weights(self, out):
    #     s = out.sum(dim=(1, 2))
    #     white_weights = torch.where(s > 0.5, torch.tensor(0.7), torch.tensor(0.3))

    #     # Create opposite vector
    #     black_weights = 1.0 - white_weights
    #     return white_weights, black_weights

    def optim_step(self, steps, save_history=True):
        batch = next(self.dataset_iter)

        inp, out = batch
        inp = inp / 255.0
        out = out / 255.0

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

        track = {
            "loss": loss,
            "white_loss": white_loss,
            "black_loss": black_loss,
        }
        self.metrics.append(track)

        if hasattr(self.dataloader, "update"):
            self.dataloader.update((nca_out, out.detach()), loss)

        info = {
            "loss": loss,
            "track": track,
            "inp": inp,
            "out": out,
            "nca_out": nca_out,
            "rollout": rollout,
        }

        if save_history:
            self.optim_snapshots.append(info)

        return info
