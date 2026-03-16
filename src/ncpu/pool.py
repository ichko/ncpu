"""
Experience-replay state pool for NCA training (à la Growing NCA paper).

Each slot stores (inp_norm, out_norm, nca_state_or_None).
At each step the trainer asks the pool for a batch of initial states; fresh
slots get a zero-initialised state with the input implanted (channel 0).
After the forward pass the trainer hands the final states back so the pool
can store them for future steps.

A small fraction of slots is randomly reset to None (fresh) each update so
the pool never fully drifts away from the input signal.
"""

import random

import torch
from torch.utils.data import DataLoader


class StatePool:
    def __init__(
        self,
        pool_size: int,
        dataset,
        batch_size: int,
        normalize_fn,
        nca_channels: int,
        H: int,
        W: int,
        reset_prob: float = 0.05,
        fresh_ratio: float = 0.5,
    ):
        self.pool_size = pool_size
        self.batch_size = batch_size
        self.normalize_fn = normalize_fn
        self.nca_channels = nca_channels
        self.H = H
        self.W = W
        self.reset_prob = reset_prob
        self.fresh_ratio = fresh_ratio

        self._loader = iter(DataLoader(dataset, batch_size=1, shuffle=False))
        # each slot: (inp_norm, out_norm, nca_state) — state is None until first use
        self._pool = [None] * pool_size

    def _fresh_slot(self):
        inp, out = next(self._loader)
        inp = self.normalize_fn(inp.squeeze(0))  # (H, W)
        out = self.normalize_fn(out.squeeze(0))  # (H, W)
        return (inp, out, None)

    def _make_initial_state(self, inp):
        """Zero state with inp implanted into channel 1 (read-only input channel)."""
        state = torch.zeros(self.nca_channels, self.H, self.W)
        state[1] = inp
        return state

    def sample(self, device):
        """
        Sample a batch from the pool.

        Returns:
            inp     (B, H, W)  — normalised input screens
            out     (B, H, W)  — normalised output screens
            states  (B, C, H, W) — initial NCA states (stored or fresh)
            indices list[int]  — pool slot indices (needed for update)
        """
        indices = random.sample(range(self.pool_size), self.batch_size)
        n_fresh = round(self.batch_size * self.fresh_ratio)

        inps, outs, states = [], [], []
        for i, idx in enumerate(indices):
            if self._pool[idx] is None:
                self._pool[idx] = self._fresh_slot()
            inp, out, state = self._pool[idx]
            # force fresh state for the first n_fresh samples in the batch
            if i < n_fresh or state is None:
                state = self._make_initial_state(inp)
            inps.append(inp)
            outs.append(out)
            states.append(state)

        return (
            torch.stack(inps).to(device),
            torch.stack(outs).to(device),
            torch.stack(states).to(device),
            indices,
        )

    def update(self, indices, final_states):
        """
        Store final NCA states back into the pool.

        Args:
            indices     : list[int] returned by sample()
            final_states: (B, C, H, W) tensor — last frame of the rollout
        """
        cpu_states = final_states.detach().cpu()
        for i, idx in enumerate(indices):
            if random.random() < self.reset_prob:
                self._pool[idx] = None  # will get a fresh sample next time
            else:
                inp, out, _ = self._pool[idx]
                self._pool[idx] = (inp, out, cpu_states[i])
