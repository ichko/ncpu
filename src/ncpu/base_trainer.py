import os
import pickle
import uuid
from datetime import datetime

import torch
import torch.nn as nn


class BaseTrainer(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.name = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
        )

    def save_checkpoint(self):
        root = os.path.join(self.checkpoint_path, self.name)
        os.makedirs(root, exist_ok=True)
        torch.save(
            self.nca.state_dict(),
            os.path.join(root, f"nca_{self.learning_steps:06d}.pt"),
        )
        torch.save(
            self.decoder.state_dict(),
            os.path.join(root, f"decoder_{self.learning_steps:06d}.pt"),
        )
        with open(os.path.join(root, "self.pkl"), "wb") as f:
            pickle.dump(self, f)

    def load_checkpoint(self, learning_step):
        root = os.path.join(self.checkpoint_path, self.name)
        self.nca.load_state_dict(
            torch.load(os.path.join(root, f"nca_{learning_step:06d}.pt"))
        )
        self.decoder.load_state_dict(
            torch.load(os.path.join(root, f"decoder_{learning_step:06d}.pt"))
        )

    def load_last_checkpoint(self):
        root = os.path.join(self.checkpoint_path, self.name)
        nca_files = [
            f for f in os.listdir(root) if f.startswith("nca_") and f.endswith(".pt")
        ]
        latest_file = max(
            nca_files,
            key=lambda x: int(x.split("_")[1].split(".")[0]),
        )
        learning_step = int(latest_file.split("_")[1].split(".")[0])
        self.load_checkpoint(learning_step)

    @staticmethod
    def load_trainer(root):
        with open(os.path.join(root, "self.pkl"), "rb") as f:
            trainer = pickle.load(f)
        # trainer.load_last_checkpoint()
        return trainer

    @classmethod
    def load_last_trainer(cls, checkpoint_path):
        nca_dirs = sorted([d for d in os.listdir(checkpoint_path)])
        latest_dir = max(
            nca_dirs,
            key=lambda x: os.path.getmtime(os.path.join(checkpoint_path, x)),
        )
        root = os.path.join(checkpoint_path, latest_dir)
        return cls.load_trainer(root)

    def model_checksum(model):
        return sum(p.abs().sum().item() for p in model.parameters())
