from typing import List, Optional

from ncpu.dataset import NCPUDataset, PoolDataset, ScheduledDataset
from ncpu.nca import NeuralCAv2
from ncpu.trainer import NCPUTrainer

def setup_trainer(base_model, config):
    dataset = NCPUDataset(config)
    if config.pool_size > 0:
        dataset = PoolDataset(dataset, pool_size=config.pool_size)

    nca = NeuralCAv2(
        channels=config.channels,
        hidden_channels=config.hidden_channels,
        fire_rate=config.fire_rate,
        alive_threshold=config.alive_threshold,
        zero_initialization=config.zero_initialization,
    ).to(config.device)

    nca = base_model
    trainer = NCPUTrainer(
        nca,
        dataset.get_dataloader(batch_size=config.batch_size),
        lr=config.lr,
        gaussian_noise=config.gaussian_noise,
    )
    trainer.sanity_check()

    return trainer
