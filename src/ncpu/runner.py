import time

from ncpu.dataset import NCPUDataset, PoolDataset
from ncpu.model import NeuralCA
from ncpu.trainer import NCPUTrainer

from matplotlib import pyplot as plt
from IPython.display import clear_output, display
from tqdm.auto import tqdm
from datetime import datetime
from ncpu.utils import print_tensor, sequence_batch_to_html_gifs
import mediapy as media
import torch
import panel as pn

pn.extension()


def setup_trainer(config):
    dataset = NCPUDataset(
        W=config.W,
        H=config.H,
        r=config.r,
        small_r=config.small_r,
        spacing=config.spacing,
        margin=config.margin,
        sampler=config.sampler,
        bit_length=config.bit_length,
        balanced=config.balanced,
    )
    if config.pool_size > 0:
        dataset = PoolDataset(dataset, pool_size=config.pool_size)

    nca = NeuralCA(
        channels=config.channels,
        hidden_channels=config.hidden_channels,
        fire_rate=config.fire_rate,
        alive_masking=config.alive_masking,
        zero_initialization=config.zero_initialization,
    ).to(config.device)

    trainer = NCPUTrainer(
        nca,
        dataset.get_dataloader(batch_size=config.batch_size),
        lr=config.lr,
        gaussian_noise=config.gaussian_noise,
    )
    trainer.sanity_check()

    return trainer
