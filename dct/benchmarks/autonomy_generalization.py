from __future__ import annotations

import random

from dct.benchmarks.base import BenchmarkFamily
from dct.schemas import BenchmarkTask, ObservationExample, SimulationCase


class AutonomyGeneralizationBenchmark(BenchmarkFamily):
    family_name = "autonomy_generalization"

    def generate_task(self, seed: int, n_train: int, n_heldout: int) -> BenchmarkTask:
        rng = random.Random(seed)
        variant = rng.choice(["spurious_shift", "unit_shift"])

        if variant == "spurious_shift":
            return self._spurious_shift_task(seed, rng, n_train, n_heldout)
        return self._unit_shift_task(seed, rng, n_train, n_heldout)

    def _spurious_shift_task(self, seed: int, rng: random.Random, n_train: int, n_heldout: int) -> BenchmarkTask:
        # True mechanism depends only on x. z is intentionally spurious and non-stationary.
        def truth(x: float) -> float:
            return 3.0 * x - 2.0

        train: list[ObservationExample] = []
        heldout: list[ObservationExample] = []
        ood: list[ObservationExample] = []
        stress: list[ObservationExample] = []

        for _ in range(n_train):
            x = rng.uniform(-4.0, 6.0)
            y = truth(x)
            z = y + rng.gauss(0.0, 0.25)  # Strongly correlated in train.
            train.append(
                ObservationExample(
                    features={"x": round(x, 6), "z": round(z, 6)},
                    target=round(y, 6),
                )
            )

        for _ in range(n_heldout):
            x = rng.uniform(6.0, 16.0)
            y = truth(x)
            z = rng.uniform(-50.0, 50.0)  # Correlation breaks.
            heldout.append(
                ObservationExample(
                    features={"x": round(x, 6), "z": round(z, 6)},
                    target=round(y, 6),
                )
            )

        for _ in range(max(10, n_heldout)):
            x = rng.uniform(-20.0, 30.0)
            y = truth(x)
            z = rng.uniform(-120.0, 120.0)
            ood.append(
                ObservationExample(
                    features={"x": round(x, 6), "z": round(z, 6)},
                    target=round(y, 6),
                )
            )

        for _ in range(max(12, n_heldout)):
            x = rng.uniform(-25.0, 35.0)
            y = truth(x)
            z = (0.5 * y) + rng.gauss(0.0, 30.0)  # Adversarially unstable nuisance feature.
            target = y + rng.gauss(0.0, 0.5)
            stress.append(
                ObservationExample(
                    features={"x": round(x, 6), "z": round(z, 6)},
                    target=round(target, 6),
                )
            )

        return BenchmarkTask(
            family=self.family_name,
            task_id=f"autonomy_spurious_{seed}",
            description="Generalize through spurious-correlation shift with invariant mechanism",
            feature_names=["x", "z"],
            train=train,
            heldout=heldout,
            ground_truth_expression="3*x - 2",
            ground_truth_rule_text="Invariant linear mechanism despite feature shift",
            metadata={
                "target_tolerance": 1.0,
                "pass_threshold": 0.75,
                "domain": "generalization_under_shift",
            },
            simulation_cases=self._simulation_from_examples(heldout),
            ood=ood,
            stress=stress,
        )

    def _unit_shift_task(self, seed: int, rng: random.Random, n_train: int, n_heldout: int) -> BenchmarkTask:
        # Mechanism: force = mass * accel. Training and heldout use different scales and nuisance shifts.
        def truth(mass_kg: float, accel_ms2: float) -> float:
            return mass_kg * accel_ms2

        train: list[ObservationExample] = []
        heldout: list[ObservationExample] = []
        ood: list[ObservationExample] = []
        stress: list[ObservationExample] = []

        for _ in range(n_train):
            mass = rng.uniform(0.5, 5.0)
            accel = rng.uniform(0.5, 4.0)
            drift = rng.uniform(-2.0, 2.0)
            train.append(
                ObservationExample(
                    features={"mass_kg": round(mass, 6), "accel_ms2": round(accel, 6), "sensor_drift": round(drift, 6)},
                    target=round(truth(mass, accel), 6),
                )
            )

        for _ in range(n_heldout):
            mass = rng.uniform(5.0, 15.0)
            accel = rng.uniform(0.2, 6.0)
            drift = rng.uniform(-25.0, 25.0)
            heldout.append(
                ObservationExample(
                    features={"mass_kg": round(mass, 6), "accel_ms2": round(accel, 6), "sensor_drift": round(drift, 6)},
                    target=round(truth(mass, accel), 6),
                )
            )

        for _ in range(max(10, n_heldout)):
            mass = rng.uniform(0.2, 50.0)
            accel = rng.uniform(0.1, 10.0)
            drift = rng.uniform(-100.0, 100.0)
            ood.append(
                ObservationExample(
                    features={"mass_kg": round(mass, 6), "accel_ms2": round(accel, 6), "sensor_drift": round(drift, 6)},
                    target=round(truth(mass, accel), 6),
                )
            )

        for _ in range(max(12, n_heldout)):
            mass = rng.uniform(0.2, 50.0)
            accel = rng.uniform(0.1, 10.0)
            drift = rng.uniform(-120.0, 120.0)
            target = truth(mass, accel) + rng.gauss(0.0, 2.0)
            stress.append(
                ObservationExample(
                    features={"mass_kg": round(mass, 6), "accel_ms2": round(accel, 6), "sensor_drift": round(drift, 6)},
                    target=round(target, 6),
                )
            )

        return BenchmarkTask(
            family=self.family_name,
            task_id=f"autonomy_units_{seed}",
            description="Generalize physical mechanism under scale/shift/unit-like changes",
            feature_names=["mass_kg", "accel_ms2", "sensor_drift"],
            train=train,
            heldout=heldout,
            ground_truth_expression="mass_kg*accel_ms2",
            ground_truth_rule_text="Force equals mass times acceleration",
            metadata={
                "target_tolerance": 2.5,
                "pass_threshold": 0.72,
                "domain": "mechanism_transfer",
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
