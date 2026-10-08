"""Public benchmark registry."""

from robust_cfi_is.benchmarks.analytic import RareEventProblem, get_benchmark, g1_2d
from robust_cfi_is.benchmarks.resnet import G7ResNetEvent, get_g7_benchmark

__all__ = [
    "G7ResNetEvent", "RareEventProblem", "get_benchmark", "get_g7_benchmark", "g1_2d"
]
