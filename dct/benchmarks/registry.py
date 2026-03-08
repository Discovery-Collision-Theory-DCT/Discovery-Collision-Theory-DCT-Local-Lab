from __future__ import annotations

from dct.benchmarks.autonomy_generalization import AutonomyGeneralizationBenchmark
from dct.benchmarks.compression import CompressionBenchmark
from dct.benchmarks.dynamical import DynamicalBenchmark
from dct.benchmarks.open_world_noise import OpenWorldNoiseBenchmark
from dct.benchmarks.real_world_laws import RealWorldLawsBenchmark
from dct.benchmarks.symbolic import SymbolicBenchmark
from dct.schemas import BenchmarkTask


class BenchmarkRegistry:
    def __init__(self) -> None:
        self._families = {
            "symbolic": SymbolicBenchmark(),
            "dynamical": DynamicalBenchmark(),
            "compression": CompressionBenchmark(),
            "real_world_laws": RealWorldLawsBenchmark(),
            "autonomy_generalization": AutonomyGeneralizationBenchmark(),
            "open_world_noise": OpenWorldNoiseBenchmark(),
        }

    def generate(self, family: str, seed: int, n_train: int, n_heldout: int) -> BenchmarkTask:
        if family not in self._families:
            raise ValueError(f"Unknown benchmark family: {family}")
        return self._families[family].generate_task(seed=seed, n_train=n_train, n_heldout=n_heldout)

    def families(self) -> list[str]:
        return list(self._families.keys())
