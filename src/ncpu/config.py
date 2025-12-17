from dataclasses import dataclass

from ncpu.dataset import sample_AND_gate, sample_NAND_gate


@dataclass(frozen=True)
class _BaseTrainingConfig:
    name = "Base"
    device = "cuda"
    lr = 0.00001
    batch_size = 24
    channels = 16
    hidden_channels = 128
    fire_rate = 0.8
    alive_masking = True
    zero_initialization = True
    apply_gaussian_noise = False
    balanced = False
    pool_size = 0


@dataclass(frozen=True)
class _FastTrainingConfig:
    name = "Fast"
    device = "cuda"
    lr = 0.002
    batch_size = 16
    channels = 16
    hidden_channels = 128
    fire_rate = 0.99
    alive_masking = True
    zero_initialization = True
    apply_gaussian_noise = True
    balanced = True
    pool_size = 256


@dataclass(frozen=True)
class _TinyConjunctionTrainingConfig(_FastTrainingConfig):
    name = "TinyAnd"
    W = 32
    H = 32
    r = 4
    small_r = 1
    spacing = (16, 8)
    margin = 8
    sampler = sample_AND_gate
    bit_length = 2


@dataclass(frozen=True)
class _TinyNANDTrainingConfig(_BaseTrainingConfig):
    name = "TinyNAND"
    W = 32
    H = 32
    r = 4
    small_r = 1
    spacing = (16, 8)
    margin = 8
    sampler = sample_NAND_gate
    bit_length = 2


@dataclass(frozen=True)
class _BigConjunctionTrainingConfig(_BaseTrainingConfig):
    name = "BigAnd"
    W = 117
    H = 117
    r = 25
    small_r = 10
    spacing = (55, 30)
    margin = 30
    sampler = sample_AND_gate
    bit_length = 2


BASE_TRAINING_CONFIG = _BaseTrainingConfig()
BIG_AND_GATE_TRAINING_CONFIG = _BigConjunctionTrainingConfig()
TINY_AND_GATE_TRAINING_CONFIG = _TinyConjunctionTrainingConfig()
TINY_NAND_TRAINING_CONFIG = _TinyNANDTrainingConfig()
