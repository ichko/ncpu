from ncpu.dataset import NCPUDataset, PoolDataset
from ncpu.model import NeuralCA
from ncpu.trainer import NCPUTrainer

#[WIP]
# This is all work in progress
def setup_evaluator(config, path):
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

    # 2. Load the state dict
    state_dict = torch.load(path, map_location="cpu")  # or "cuda"
    nca.load_state_dict(state_dict)

    evaluator = NCPUEval()
    return evaluator