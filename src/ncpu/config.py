from dataclasses import dataclass

from ncpu.dataset import sample_AND_gate, sample_NAND_gate, sample_XOR_gate


@dataclass(frozen=True)
class _TwoArgSmallGridMixin:
    W = 32
    H = 32
    r = 4
    spacing = (2, 16)


@dataclass(frozen=True)
class _TwoArgLargeGridMixin:
    W = 117
    H = 117
    r = 25
    spacing = (55, 30)


@dataclass(frozen=True)
class _DefaultNCAMixin:
    channels = 16
    hidden_channels = 128
    fire_rate = 0.99
    alive_masking = True
    zero_initialization = True


@dataclass(frozen=True)
class _OptimizationArgsMixin:
    lr = 0.0002
    batch_size = 16
    gaussian_noise = 0.01
    balanced = False
    pool_size = 0
    device = "cuda"


@dataclass(frozen=True)
class _TinyConjunctionTrainingConfig(
    _DefaultNCAMixin, _OptimizationArgsMixin, _TwoArgSmallGridMixin
):
    name = "TinyAnd"
    sampler = sample_AND_gate


@dataclass(frozen=True)
class _TinyNANDTrainingConfig(
    _DefaultNCAMixin, _OptimizationArgsMixin, _TwoArgSmallGridMixin
):
    name = "TinyNAND"
    sampler = sample_NAND_gate


@dataclass(frozen=True)
class _TinyNANDTrainingConfig(
    _DefaultNCAMixin, _OptimizationArgsMixin, _TwoArgSmallGridMixin
):
    name = "TinyXOR"
    sampler = sample_XOR_gate


@dataclass(frozen=True)
class _BigConjunctionTrainingConfig(
    _DefaultNCAMixin, _OptimizationArgsMixin, _TwoArgLargeGridMixin
):
    name = "BigAnd"
    sampler = sample_AND_gate


BIG_AND_GATE_TRAINING_CONFIG = _BigConjunctionTrainingConfig()
TINY_AND_GATE_TRAINING_CONFIG = _TinyConjunctionTrainingConfig()
TINY_NAND_TRAINING_CONFIG = _TinyNANDTrainingConfig()
TINY_XOR_TRAINING_CONFIG = _TinyNANDTrainingConfig()
