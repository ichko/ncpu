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
from ncpu.noise import Damage, apply_input_damage
from ncpu.normalizers import normalize_neg1_to_1
from ncpu.utils import (
    add_gaussian_noise,
    print_tensor,
    sequence_batch_to_html_gifs,
    tensor_to_video_pane,
)

from typing import Optional, Callable, Sequence, Union


class NCPUTrainer(BaseTrainer):
    _exclude_from_pickle = {"dataloader", "ds", "dataset_iter", "optim", "pool"}

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
        input_dims=(1,),
        input_implant_type: Optional[str] = None,
        normalize_fn=normalize_neg1_to_1,
        input_damages: Optional[Sequence[Damage]] = None,
        damage_mode: str = "all",
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
        self.inp_mask_binary = (self.inp_mask > 128).float()
        # per-bit masks: (n_bits, H, W)
        self.bit_masks = self.ds.get_output_bit_masks().to(self.nca.device)
        self.loss_fn = loss_fn
        self.input_dims = list(input_dims)
        self.input_implant_type = input_implant_type
        self.normalize_fn = normalize_fn
        self.input_damages = list(input_damages) if input_damages else []
        # "all": apply every input damage each step; "sample": pick one at random
        self.damage_mode = damage_mode
        self.pool = None  # set externally to enable state-pool training

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

        if self.input_implant_type == "disabled":
            # The dataset already produces the full multi-channel state
            # (e.g. MultiGateDataset): use it directly as the first state.
            if inp.dim() == 3:
                inp = inp.unsqueeze(1)
            return inp.expand(bs, self.nca.channels, self.ds.H, self.ds.W).clone().to(
                self.nca.device
            )

        if inp.dim() == 3:
            inp = inp.unsqueeze(1)  # (B, 1, H, W)

        if self.input_implant_type == "all":
            # Replicate the single-channel input across all NCA channels.
            return inp.expand(bs, self.nca.channels, self.ds.H, self.ds.W).clone().to(
                self.nca.device
            )

        # default / "first": zero state with the input implanted into input_dims
        first_state = torch.zeros(bs, self.nca.channels, self.ds.H, self.ds.W)
        for dim in self.input_dims:
            first_state[:, dim] = inp.squeeze(1)
        first_state = first_state.to(self.nca.device)
        return first_state

    def _apply_input_damages(self, inp):
        if not self.input_damages:
            return inp
        if self.damage_mode == "sample":
            idx = int(torch.randint(len(self.input_damages), (1,)).item())
            d = self.input_damages[idx]
            return apply_input_damage(inp, d)
        for d in self.input_damages:
            inp = apply_input_damage(inp, d)
        return inp

    def _resolve_state_damages(self, state_damages):
        if state_damages is None:
            state_damages = self.nca.state_damages
            if self.damage_mode == "sample" and state_damages:
                idx = int(torch.randint(len(state_damages), (1,)).item())
                state_damages = [state_damages[idx]]
        return state_damages

    def optim_step(self, steps, return_rollout=False, noise_std=None,
                   noise_fire_rate=None, state_damages=None):
        forward_steps = steps
        if isinstance(steps, (tuple, list)):
            forward_steps = np.random.randint(steps[0], steps[1])

        if noise_std is None:
            noise_std = self.nca.gaussian_noise.std
        if noise_fire_rate is None:
            noise_fire_rate = self.nca.gaussian_noise.prob
        state_damages = self._resolve_state_damages(state_damages)

        if self.pool is not None:
            inp, out, first_state, pool_indices = self.pool.sample(self.device)
        else:
            batch = next(self.dataset_iter)
            inp, out = batch
            if self.gaussian_noise > 0:
                inp = add_gaussian_noise(inp, 0, self.gaussian_noise)
            inp = self._apply_input_damages(inp)
            inp = self.normalize_fn(inp)
            out = self.normalize_fn(out)
            inp = inp.to(self.device)
            out = out.to(self.device)
            first_state = self._implant_input(inp)
            pool_indices = None

        rollout = self.nca.forward(
            first_state,
            steps=forward_steps,
            noise_std=noise_std,
            noise_fire_rate=noise_fire_rate,
            state_damages=state_damages,
        )

        if self.pool is not None:
            self.pool.update(pool_indices, rollout[:, -1].detach())

        nca_out = rollout[:, -1, 0]
        loss = self.loss_fn(
            rollout,
            out,
            inp=inp,
            mask=self.out_mask_binary,
            inp_mask=self.inp_mask_binary,
        )

        # num_valid_bits: mean correct bits per sample (out of n_bits)
        with torch.no_grad():
            num_valid_bits = self._compute_num_valid_bits(nca_out, out)

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

    def _compute_num_valid_bits(self, nca_out, out):
        """Mean fraction of output-bit circles whose sign matches the target."""
        bm = self.bit_masks  # (n_bits, H, W)
        bm_sum = bm.sum(dim=(-1, -2))  # (n_bits,)
        # mean activation per bit circle: (B, n_bits)
        nca_bit = (nca_out.unsqueeze(1) * bm.unsqueeze(0)).sum(dim=(-1, -2)) / bm_sum
        tgt_bit = (out.unsqueeze(1) * bm.unsqueeze(0)).sum(dim=(-1, -2)) / bm_sum
        return ((nca_bit > 0) == (tgt_bit > 0)).float().sum(dim=1).mean().item()

    def evaluate(self, eval_batch, noise_configs, n_rollouts=5, steps=80, damage_configs=None):
        """Evaluate the NCA on a fixed batch with stochastic noise/damages.

        RL-style evaluation: the NCA is rolled out `n_rollouts` times per
        config under torch.no_grad(), and the bit accuracy (num_valid_bits)
        is averaged over all rollouts to reduce the variance of stochastic
        runs. The model weights are untouched.

        Args:
            eval_batch:      (inp, out) batch exactly as produced by the dataset
                             (normalized inside, matching optim_step).
            noise_configs:   iterable of (std, fire_rate) pairs for the legacy
                             per-step GaussianNoise. Passing an empty list uses
                             the model's current gaussian_noise settings.
                             Ignored when `damage_configs` is given.
            n_rollouts:      stochastic rollouts per config.
            steps:           fixed rollout length for evaluation.
            damage_configs:  optional list of dicts, one per evaluation cell of
                             the robustness matrix. Each dict may contain:
                               - "label":         name of the eval cell
                               - "input_damages": list[Damage] applied to the raw
                                 input screen before normalization
                               - "state_damages": list[Damage] applied each
                                 rollout step
                               - "noise_std" / "noise_fire_rate": legacy
                                 GaussianNoise overrides for this cell

        Returns dict with eval_bits (mean accuracy), eval_loss (mean loss), and
        per_cfg (list of per-config means, including the damage labels).
        """
        inp_raw, out = eval_batch
        out = self.normalize_fn(out).to(self.device)

        if damage_configs is None:
            if not noise_configs:
                noise_configs = [(self.nca.gaussian_noise.std, self.nca.gaussian_noise.prob)]
            damage_configs = [
                {
                    "label": f"gauss_s{std:g}_p{fr:g}",
                    "noise_std": std,
                    "noise_fire_rate": fr,
                }
                for std, fr in noise_configs
            ]

        bits_all, loss_all, per_cfg = [], [], []
        with torch.no_grad():
            for cfg in damage_configs:
                inp = inp_raw.clone()
                for d in cfg.get("input_damages", []):
                    inp = apply_input_damage(inp, d)
                inp = self.normalize_fn(inp)
                if self.gaussian_noise > 0:
                    inp = add_gaussian_noise(inp, 0, self.gaussian_noise)
                inp = inp.to(self.device)
                first_state = self._implant_input(inp)

                cfg_bits, cfg_losses = [], []
                for _ in range(n_rollouts):
                    rollout = self.nca.forward(
                        first_state,
                        steps=steps,
                        noise_std=cfg.get("noise_std"),
                        noise_fire_rate=cfg.get("noise_fire_rate"),
                        state_damages=cfg.get("state_damages"),
                    )
                    nca_out = rollout[:, -1, 0]
                    loss = self.loss_fn(
                        rollout,
                        out,
                        inp=inp,
                        mask=self.out_mask_binary,
                        inp_mask=self.inp_mask_binary,
                    )
                    cfg_bits.append(self._compute_num_valid_bits(nca_out, out))
                    cfg_losses.append(loss.item())
                per_cfg.append(
                    {
                        "label": cfg.get("label", "default"),
                        "noise_std": cfg.get("noise_std"),
                        "noise_fire_rate": cfg.get("noise_fire_rate"),
                        "input_damages": [d.to_dict() for d in cfg.get("input_damages", [])],
                        "state_damages": [d.to_dict() for d in cfg.get("state_damages", [])],
                        "bits": sum(cfg_bits) / len(cfg_bits),
                        "loss": sum(cfg_losses) / len(cfg_losses),
                    }
                )
                bits_all.extend(cfg_bits)
                loss_all.extend(cfg_losses)

        return {
            "eval_bits": sum(bits_all) / len(bits_all),
            "eval_loss": sum(loss_all) / len(loss_all),
            "per_cfg": per_cfg,
        }

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
