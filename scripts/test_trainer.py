import torch
import json
from datetime import datetime
from ncpu.nca import NeuralCA

from pathlib import Path
from matplotlib import pyplot as plt
from IPython.display import clear_output, display
from tqdm.auto import tqdm

from ncpu.loss import loss_mse_whole_seq, loss_white_black, fullscreen_rollout_loss, output_masked_rollout_loss, combined_loss
from ncpu.config import TINY_AND_FARAWAY_TRAINING_CONFIG
from ncpu.dataset import NCPUDataset, MultiGateDataset
from ncpu.trainer import NCPUTrainer
from ncpu.utils import freeze_frame, git_info, make_grid, save_grid_image

model_path = None

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())   # Should be True
print(torch.cuda.device_count())   # Should show 1
print(torch.cuda.get_device_name(0))  # Should show "NVIDIA GeForce GTX 1080 Ti"
torch.set_default_device('cuda')

LEARNING_RATE = 0.001
BATCH_SIZE = 8
GAUSSIAN_NOISE = 0.2
STEPS = 20_000
PLOT_EVERY = 1_000
NCA_CHANNELS = 8

dataset = MultiGateDataset(TINY_AND_FARAWAY_TRAINING_CONFIG, nca_channels=NCA_CHANNELS)

nca = NeuralCA(
    channels = NCA_CHANNELS,
    hidden_channels = [128],
    fire_rate = 0.99,
    alive_threshold = 0.1,
    zero_initialization = False,
    kernel_size = 5,
    read_only_dims = [-1]
)

trainer = NCPUTrainer(
    nca,
    dataset.get_dataloader(batch_size = BATCH_SIZE),
    lr = LEARNING_RATE,
    gaussian_noise = GAUSSIAN_NOISE,
    loss_fn = output_masked_rollout_loss,
    input_implant_type="disabled"
)

trainer.sanity_check()
info = trainer.optim_step(steps=(30, 80))
