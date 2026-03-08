from __future__ import annotations

import random

from dct.benchmarks.base import BenchmarkFamily
from dct.schemas import BenchmarkTask, ObservationExample, SimulationCase


class OpenWorldNoiseBenchmark(BenchmarkFamily):
    family_name = "open_world_noise"

    def generate_task(self, seed: int, n_train: int, n_heldout: int) -> BenchmarkTask:
        rng = random.Random(seed)
        variant = rng.choice(["heteroskedastic_linear", "nonlinear_outliers"])

        if variant == "heteroskedastic_linear":
            return self._heteroskedastic_linear_task(seed, rng, n_train, n_heldout)
        return self._nonlinear_outlier_task(seed, rng, n_train, n_heldout)

    def _heteroskedastic_linear_task(
        self,
        seed: int,
        rng: random.Random,
        n_train: int,
        n_heldout: int,
    ) -> BenchmarkTask:
        def clean(x: float, y: float) -> float:
            return 1.7 * x - 0.4 * y + 3.0

        train: list[ObservationExample] = []
        heldout: list[ObservationExample] = []
        ood: list[ObservationExample] = []
        stress: list[ObservationExample] = []

        for _ in range(n_train):
            x = rng.uniform(-8.0, 8.0)
            y = rng.uniform(-6.0, 6.0)
            drift = rng.uniform(-1.0, 1.0)
            base = clean(x, y)
            noise_std = 0.4 + 0.12 * abs(x)
            noisy = base + rng.gauss(0.0, noise_std)
            if rng.random() < 0.10:
                noisy += rng.choice([-9.0, 9.0])
            train.append(
                ObservationExample(
                    features={"x": round(x, 6), "y": round(y, 6), "sensor_drift": round(drift, 6)},
                    target=round(noisy, 6),
                )
            )

        for _ in range(n_heldout):
            x = rng.uniform(-10.0, 10.0)
            y = rng.uniform(-8.0, 8.0)
            heldout.append(
                ObservationExample(
                    features={"x": round(x, 6), "y": round(y, 6), "sensor_drift": 0.0},
                    target=round(clean(x, y), 6),
                )
            )

        for _ in range(max(10, n_heldout)):
            x = rng.uniform(-20.0, 20.0)
            y = rng.uniform(-20.0, 20.0)
            ood.append(
                ObservationExample(
                    features={"x": round(x, 6), "y": round(y, 6), "sensor_drift": rng.uniform(-3.0, 3.0)},
                    target=round(clean(x, y), 6),
                )
            )

        for _ in range(max(14, n_heldout)):
            x = rng.uniform(-20.0, 20.0)
            y = rng.uniform(-20.0, 20.0)
            heavy = clean(x, y) + rng.gauss(0.0, 2.2 + 0.15 * abs(x))
            if rng.random() < 0.15:
                heavy += rng.choice([-14.0, 14.0])
            stress.append(
                ObservationExample(
                    features={"x": round(x, 6), "y": round(y, 6), "sensor_drift": rng.uniform(-5.0, 5.0)},
                    target=round(heavy, 6),
                )
            )

        return BenchmarkTask(
            family=self.family_name,
            task_id=f"openworld_linear_{seed}",
            description="Infer stable rule under heteroskedastic noise and outliers",
            feature_names=["x", "y", "sensor_drift"],
            train=train,
            heldout=heldout,
            ground_truth_expression="1.7*x - 0.4*y + 3",
            ground_truth_rule_text="Linear mechanism under noisy open-world observations",
            metadata={
                "target_tolerance": 3.2,
                "pass_threshold": 0.65,
                "domain": "open_world_sensor_noise",
            },
            simulation_cases=self._simulation_from_examples(heldout),
            ood=ood,
            stress=stress,
        )

    def _nonlinear_outlier_task(
        self,
        seed: int,
        rng: random.Random,
        n_train: int,
        n_heldout: int,
    ) -> BenchmarkTask:
        def clean(x: float, y: float) -> float:
            return x * x + 0.5 * y - 1.0

        train: list[ObservationExample] = []
        heldout: list[ObservationExample] = []
        ood: list[ObservationExample] = []
        stress: list[ObservationExample] = []

        for _ in range(n_train):
            x = rng.uniform(-4.5, 4.5)
            y = rng.uniform(-10.0, 10.0)
            noise = rng.gauss(0.0, 0.8 + 0.08 * abs(y))
            target = clean(x, y) + noise
            if rng.random() < 0.12:
                target += rng.choice([-10.0, 10.0])
            train.append(ObservationExample(features={"x": round(x, 6), "y": round(y, 6)}, target=round(target, 6)))

        for _ in range(n_heldout):
            x = rng.uniform(-6.0, 6.0)
            y = rng.uniform(-12.0, 12.0)
            heldout.append(ObservationExample(features={"x": round(x, 6), "y": round(y, 6)}, target=round(clean(x, y), 6)))

        for _ in range(max(10, n_heldout)):
            x = rng.uniform(-12.0, 12.0)
            y = rng.uniform(-20.0, 20.0)
            ood.append(ObservationExample(features={"x": round(x, 6), "y": round(y, 6)}, target=round(clean(x, y), 6)))

        for _ in range(max(14, n_heldout)):
            x = rng.uniform(-12.0, 12.0)
            y = rng.uniform(-20.0, 20.0)
            noisy = clean(x, y) + rng.gauss(0.0, 3.0)
            if rng.random() < 0.18:
                noisy += rng.choice([-18.0, 18.0])
            stress.append(ObservationExample(features={"x": round(x, 6), "y": round(y, 6)}, target=round(noisy, 6)))

        return BenchmarkTask(
            family=self.family_name,
            task_id=f"openworld_nonlinear_{seed}",
            description="Recover nonlinear mechanism under heavy outliers and distribution drift",
            feature_names=["x", "y"],
            train=train,
            heldout=heldout,
            ground_truth_expression="x*x + 0.5*y - 1",
            ground_truth_rule_text="Nonlinear mechanism with additive component",
            metadata={
                "target_tolerance": 4.0,
                "pass_threshold": 0.62,
                "domain": "open_world_high_noise",
            },
            simulation_cases=self._simulation_from_examples(heldout),
            ood=ood,
            stress=stress,
        )

    @staticmethod
    def _simulation_from_examples(examples: list[ObservationExample]) -> list[SimulationCase]:
        return [
            SimulationCase(features=ex.features, expected_target=ex.target)
            for ex in examples[: min(12, len(examples))]
        ]
