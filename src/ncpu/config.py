from dataclasses import dataclass

from ncpu.dataset import sample_conjunction_input_output


@dataclass(frozen=True)
class _BaseTrainingConfig:
    name = "Base"
    device = "cuda"
    lr = 0.00001
    batch_size = 16
    channels = 16
    hidden_channels = 128
    fire_rate = 0.8
    alive_masking = True
    zero_initialization = True


@dataclass(frozen=True)
class _BigConjunctionTrainingConfig(_BaseTrainingConfig):
    name = "BigAnd"
    W = 117
    H = 117
    r = 25
    spacing = (55, 30)
    margin = 30
    sampler = sample_conjunction_input_output


BASE_TRAINING_CONFIG = _BaseTrainingConfig()
BIG_CONJUNCTION_TRAINING_CONFIG = _BigConjunctionTrainingConfig()
