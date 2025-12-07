from dataclasses import dataclass

from ncpu.dataset import sample_AND_gate, sample_NAND_gate


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
class _FastTrainingConfig:
    name = "Fast"
    device = "cuda"
    lr = 0.002
    batch_size = 16
    channels = 16
    hidden_channels = 128
    fire_rate = 0.8
    alive_masking = True
    zero_initialization = True


@dataclass(frozen=True)
class _TinyConjunctionTrainingConfig(_BaseTrainingConfig):
    name = "TinyAnd"
    W = 32
    H = 32
    r = 4
    spacing = (16, 8)
    margin = 8
    sampler = sample_AND_gate

@dataclass(frozen=True)
class _TinyNANDTrainingConfig(_BaseTrainingConfig):
    name = "TinyNAND"
    W = 32
    H = 32
    r = 4
    spacing = (16, 8)
    margin = 8
    sampler = sample_NAND_gate

@dataclass(frozen=True)
class _BigConjunctionTrainingConfig(_BaseTrainingConfig):
    name = "BigAnd"
    W = 117
    H = 117
    r = 25
    spacing = (55, 30)
    margin = 30
    sampler = sample_AND_gate


BASE_TRAINING_CONFIG = _BaseTrainingConfig()
BIG_CONJUNCTION_TRAINING_CONFIG = _BigConjunctionTrainingConfig()
TINY_CONJUNCTION_TRAINING_CONFIG = _TinyConjunctionTrainingConfig()
TINY_NAND_TRAINING_CONFIG = _TinyNANDTrainingConfig()
