from ncpu.dataset import NCPUDataset, PoolDataset
from ncpu.model import NeuralCA
from ncpu.trainer import NCPUTrainer


def setup_trainer(config):
    dataset = NCPUDataset(
        W=config.W,
        H=config.H,
        r=config.r,
        spacing=config.spacing,
        margin=config.margin,
        sampler=config.sampler,
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
