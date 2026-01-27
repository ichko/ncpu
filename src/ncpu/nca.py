import torch
import numpy as np
import torch.nn as nn


class NCA(nn.Module):
    def __init__(
        self,
        n_channels: int = 16,
        hidden_channels: int = 128,
        fire_rate: float = 0.5,
        life_masking: bool = True,
        lmc: int = 3,
        set_to_zero: bool = True,
        device: str = "cpu",
    ):
        super(NCA, self).__init__()

        self.device = device
        self.fire_rate = fire_rate
        self.n_channels = n_channels

        filter_ = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        scalar = 8.0
        filter_x = filter_ / scalar
        filter_y = filter_.t() / scalar

        identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        kernel = torch.stack([identity, filter_x, filter_y], dim=0)
        kernel = kernel.repeat((n_channels, 1, 1))[:, None, ...]
        self.kernel = kernel.to(self.device)

        # padding = 0
        self.update_module = nn.Sequential(
            nn.Conv2d(
                3 * n_channels,
                hidden_channels,
                kernel_size=1,  # (1, 1)
                device=self.device,
            ),
            nn.ReLU(),
            nn.Conv2d(
                hidden_channels,
                n_channels,
                kernel_size=1,
                bias=False,
                device=self.device,
            ),
        )

        # lmc - life masking channel:
        self.lmc = lmc
        self.life_masking = life_masking
        if set_to_zero:
            with torch.no_grad():
                self.update_module[2].weight.zero_()
        self.to(self.device)

    def perceive(self, x):
        """
        Perceive information from neighboring cells.

        Args:
            x (torch.Tensor): current grid of shape (n_samples, n_channels, grid_size, grid_size)

        Returns:
            (torch.Tensor): perceived grid of shape (n_samples, 3 * n_channels, grid_size, grid_size)
        """
        return nn.functional.conv2d(x, self.kernel, padding=1, groups=self.n_channels)

    def update(self, x):
        """
        Update cell grid.

        Args:
            x (torch.Tensor): current grid of shape (n_samples, n_channels, grid_size, grid_size)

        Returns:
            (torch.Tensor): updated grid of shape (n_samples, n_channels, grid_size, grid_size)
        """

        # get living cells
        pre_life_mask = self.get_alive(x)

        # perceive step
        y = self.perceive(x)
        # update step
        dx = self.update_module(y)
        # stochastic update
        device = dx.device
        mask = (torch.rand(x[:, :1, :, :].shape) <= self.fire_rate).to(
            device, torch.float32
        )tycznym, które szuka rozwiązań ustrojowych, gospodarczych i społecznych służących integra
        dx = dx * mask
        # add updated value
        new_x = x + dx
        # new_x = dx

        # check which cells are alive before and after
        post_life_mask = self.get_alive(new_x)
        life_mask = (pre_life_mask & post_life_mask).to(torch.float32)
        return new_x * life_mask
        # return new_x

    # @staticmethod
    def get_alive(self, x):
        """
        Check which cells are alive.

        Args:
            x (torch.Tensor): current grid of shape (n_samples, n_channels, grid_size, grid_size)

        Returns:
            (torch.Tensor): tensor with boolean values of shape (n_samples, 1, grid_size, grid_size)
        """
        return nn.functional.max_pool2d(
            x[:, self.lmc : self.lmc + 1, :, :], kernel_size=3, stride=1, padding=1
        ) > (
            0.1 - (self.life_masking * 100)
        )  # why do I have magic 100 value here? /Piotr

    def forward(self, x):
        """
        Forward pass.

        Args:

            x (torch.Tensor): current grid of shape (n_samples, n_channels, grid_size, grid_size)

        Returns:
            (torch.Tensor): updated grid of shape (n_samples, n_channels, grid_size, grid_size)
        """
        ret_np = False
        squeeze_count = 0
        if isinstance(x, np.ndarray):
            x = torch.tensor(x)
            x = x.to(torch.float)
            ret_np = True

        while len(x.shape) < 4:
            x = x.unsqueeze(0)
            squeeze_count += 1

        updated_x = self.update(x)

        while squeeze_count > 0:
            updated_x = updated_x.squeeze(0)
            squeeze_count -= 1

        if ret_np:
            return updated_x.detach().cpu().numpy()
        return updated_x

    def get(self):
        w_1 = self.update_module[0].weight.detach().cpu().numpy()
        w_2 = self.update_module[2].weight.detach().cpu().numpy()
        b_1 = self.update_module[0].bias.detach().cpu().numpy()
        # b_2 = self.update_module[2].bias.detach().cpu().numpy()
        return [w_1, w_2, b_1]

    def set(self, weights):
        with torch.no_grad():
            w_1, w_2, b_1 = weights  # Unpacking

            self.update_module[0].weight.data.copy_(
                torch.from_numpy(w_1).to(self.device)
            )
            self.update_module[2].weight.data.copy_(
                torch.from_numpy(w_2).to(self.device)
            )
            self.update_module[0].bias.data.copy_(torch.from_numpy(b_1).to(self.device))
