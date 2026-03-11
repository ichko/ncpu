import pickle
from pathlib import Path

import torch
import torch.nn as nn

DEFAULT_CHECKPOINT_PATTERN = "checkpoints/nca_{step:06d}.pt"


class BaseTrainer(nn.Module):
    def __init__(self, checkpoint_pattern=DEFAULT_CHECKPOINT_PATTERN):
        super().__init__()
        self.learning_step = 0
        self.checkpoint_pattern = checkpoint_pattern
        self.metrics = []

    def __getstate__(self):
        return {
            k: v for k, v in self.__dict__.items() if k not in self._exclude_from_pickle
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
        for k in self._exclude_from_pickle:
            setattr(self, k, None)

    def log_metrics(self, **metrics_kwargs):
        self.metrics.append(metrics_kwargs)

    def _checkpoint_path(self, step):
        return Path(self.checkpoint_pattern.format(step=step))

    def save_checkpoint(self):
        path = self._checkpoint_path(self.learning_step)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.nca.state_dict(), path)
        with open(path.parent / "trainer.pkl", "wb") as f:
            pickle.dump(self, f)

    def load_checkpoint(self, step):
        path = self._checkpoint_path(step)
        self.nca.load_state_dict(torch.load(path))

    def load_last_checkpoint(self):
        ckpt_dir = self._checkpoint_path(0).parent
        nca_files = list(ckpt_dir.glob("nca_*.pt"))
        latest = max(nca_files, key=lambda p: int(p.stem.split("_")[1]))
        step = int(latest.stem.split("_")[1])
        self.load_checkpoint(step)

    @staticmethod
    def load_trainer(checkpoint_dir):
        with open(Path(checkpoint_dir) / "trainer.pkl", "rb") as f:
            return pickle.load(f)

    def model_checksum(model):
        return sum(p.abs().sum().item() for p in model.parameters())

    @property
    def device(self):
        return next(self.parameters()).device
