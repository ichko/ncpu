from typing import List, Optional

from ncpu.dataset import NCPUDataset, PoolDataset, ScheduledDataset
from ncpu.model import NeuralCA
from ncpu.trainer import NCPUTrainer


def setup_scheduled_trainer(configs: List = [], steps=2_000):
    primary_config = configs[0]
    datasets = [NCPUDataset(config) for config in configs]
    dataset = ScheduledDataset(
        datasets,
        steps=steps
        * primary_config.batch_size,  # this will make shift after N optim_step instead of N/batch_size optim_steps
    )

    nca = NeuralCA(
        channels=primary_config.channels,
        hidden_channels=primary_config.hidden_channels,
        fire_rate=primary_config.fire_rate,
        alive_masking=primary_config.alive_masking,
        zero_initialization=primary_config.zero_initialization,
    ).to(primary_config.device)

    trainer = NCPUTrainer(
        nca,
        dataset.get_dataloader(batch_size=primary_config.batch_size),
        lr=primary_config.lr,
        gaussian_noise=primary_config.gaussian_noise,
    )
    trainer.sanity_check()

    return trainer


def setup_trainer(config):
    dataset = NCPUDataset(config)
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
