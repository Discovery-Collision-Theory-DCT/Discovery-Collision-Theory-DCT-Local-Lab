from __future__ import annotations

import random

from dct.benchmarks.base import BenchmarkFamily
from dct.schemas import BenchmarkTask, ObservationExample, SimulationCase


class CompressionBenchmark(BenchmarkFamily):
    family_name = "compression"

    def generate_task(self, seed: int, n_train: int, n_heldout: int) -> BenchmarkTask:
        rng = random.Random(seed)
        variant = rng.choice(["linear_noise", "quadratic_noise"])

        if variant == "linear_noise":
            expr = "4*x - 2"
            desc = "Noisy observations from compact linear rule"

            def clean(x: float) -> float:
                return 4 * x - 2

            noise_std = 1.2
            tolerance = 1.5
        else:
            expr = "x*x - x"
            desc = "Noisy observations from compact quadratic rule"

            def clean(x: float) -> float:
                return x * x - x

            noise_std = 2.5
            tolerance = 2.5

        train = []
        heldout = []

        for _ in range(n_train):
            x = rng.randint(-8, 8)
            noisy = clean(x) + rng.gauss(0, noise_std)
            if rng.random() < 0.08:
                noisy += rng.choice([-7.0, 7.0])
            train.append(ObservationExample(features={"x": x}, target=round(noisy, 4)))

        for _ in range(n_heldout):
            x = rng.randint(-12, 12)
            heldout.append(ObservationExample(features={"x": x}, target=round(clean(x), 4)))

        simulation_cases = [
            SimulationCase(features=example.features, expected_target=example.target)
            for example in heldout[: min(10, len(heldout))]
        ]

        return BenchmarkTask(
            family=self.family_name,
            task_id=f"compression_{variant}_{seed}",
            description=desc,
            feature_names=["x"],
            train=train,
            heldout=heldout,
            ground_truth_expression=expr,
            ground_truth_rule_text=desc,
            metadata={"target_tolerance": tolerance, "pass_threshold": 0.65},
            simulation_cases=simulation_cases,
        )
