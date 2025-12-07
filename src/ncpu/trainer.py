import torch
from torch.nn import functional as F
from ncpu.model import NeuralCA
import numpy as np


class NCPUTrainer:
    def __init__(self, nca: NeuralCA, dataloader, lr):
        super().__init__()
        self.nca = nca
        self.dataloader = dataloader 
        print(f"self.dataloader: {self.dataloader}")
        self.ds = dataloader.dataset
        self.dataset_iter = iter(self.dataloader)
        self.optim = torch.optim.Adam(self.nca.parameters(), lr=lr)
        self.history = []

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
            loss = self.optim_step()
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

    def optim_step(self):
        batch = next(self.dataset_iter)

        inp, out = batch
        inp = inp.to(self.nca.device)
        out = out.to(self.nca.device)
        inp = inp / 255.0
        # inp += torch.randn_like(inp) / 10.0
        out = out / 255.0

        first_state = self._inplant_input(inp)
        rollout = self.nca.forward(first_state, steps=np.random.randint(10, 20))
        nca_out = rollout[:, -1, 0]

        white_mask = (out > 0.5).float()

        white_loss = F.mse_loss(nca_out, out, reduction="none") * white_mask
        black_loss = F.mse_loss(nca_out, out, reduction="none") * (1 - white_mask)

        black_w = 0.5
        white_w = 0.5
        mean_losses = white_loss.mean(dim=(1, 2)) * white_w + black_loss.mean(dim=(1, 2)) * black_w # taking mean only from W, H 
        mean_total_loss = mean_losses.mean()

        if torch.is_grad_enabled():
            self.optim.zero_grad()
            mean_total_loss.backward()
            self.optim.step()

        self.history.append(mean_total_loss.item())

        if hasattr(self.dataloader, "update"):
            self.dataloader.update((nca_out, out.detach()), mean_losses)

        return {
            "loss": mean_total_loss,
            "inp": inp,
            "out": out,
            "nca_out": nca_out,
            "rollout": rollout,
        }