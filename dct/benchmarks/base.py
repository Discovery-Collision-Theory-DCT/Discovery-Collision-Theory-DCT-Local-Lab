from __future__ import annotations

from abc import ABC, abstractmethod

from dct.schemas import BenchmarkTask


class BenchmarkFamily(ABC):
    family_name: str

    @abstractmethod
    def generate_task(self, seed: int, n_train: int, n_heldout: int) -> BenchmarkTask:
        raise NotImplementedError
