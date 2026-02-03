from dataclasses import dataclass

from ncpu.dataset import (
    sample_4bit_adder,
    sample_AND_gate,
    sample_NAND_gate,
    sample_XOR_gate,
)


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
    r = 4
    spacing = (2, 58)


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
    batch_size = 8
    gaussian_noise = 0.01
    balanced = False
    pool_size = 0
    device = "cuda"


@dataclass(frozen=True)
class _TinyANDTrainingConfig(
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
class _TinyXORTrainingConfig(
    _DefaultNCAMixin, _OptimizationArgsMixin, _TwoArgSmallGridMixin
):
    name = "TinyXOR"
    sampler = sample_XOR_gate


@dataclass(frozen=True)
class _TinyNORTrainingConfig(
    _DefaultNCAMixin, _OptimizationArgsMixin, _TwoArgSmallGridMixin
):
    name = "TinyNOR"
    sampler = sample_NOR_gate


@dataclass(frozen=True)
class _BigConjunctionTrainingConfig(
    _DefaultNCAMixin, _OptimizationArgsMixin, _TwoArgLargeGridMixin
):
    name = "BigAnd"
    sampler = sample_AND_gate


@dataclass(frozen=True)
class _Big4bitAdderTrainingConfig(
    _DefaultNCAMixin, _OptimizationArgsMixin, _TwoArgLargeGridMixin
):
    name = "Big4bitAdder"
    sampler = sample_4bit_adder


BIG_AND_GATE_TRAINING_CONFIG = _BigConjunctionTrainingConfig()
TINY_AND_GATE_TRAINING_CONFIG = _TinyANDTrainingConfig()
TINY_NAND_TRAINING_CONFIG = _TinyNANDTrainingConfig()
BIG_4BIT_ADDER_TRAINING_CONFIG = _Big4bitAdderTrainingConfig()
