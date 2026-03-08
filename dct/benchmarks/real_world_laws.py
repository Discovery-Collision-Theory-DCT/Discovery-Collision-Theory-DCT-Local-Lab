from __future__ import annotations

import random

from dct.benchmarks.base import BenchmarkFamily
from dct.schemas import BenchmarkTask, ObservationExample, SimulationCase


class RealWorldLawsBenchmark(BenchmarkFamily):
    family_name = "real_world_laws"

    def generate_task(self, seed: int, n_train: int, n_heldout: int) -> BenchmarkTask:
        rng = random.Random(seed)
        variant = rng.choice(["kepler_third_law", "pendulum_period"])

        if variant == "kepler_third_law":
            return self._kepler_task(seed, rng, n_train, n_heldout)
        return self._pendulum_task(seed, rng, n_train, n_heldout)

    def _kepler_task(self, seed: int, rng: random.Random, n_train: int, n_heldout: int) -> BenchmarkTask:
        # Approximate orbital data (semi-major axis in AU, period in Earth years).
        measured = [
            {"a_au": 0.387, "period_years": 0.241},
            {"a_au": 0.723, "period_years": 0.615},
            {"a_au": 1.000, "period_years": 1.000},
            {"a_au": 1.524, "period_years": 1.881},
            {"a_au": 5.203, "period_years": 11.862},
            {"a_au": 9.537, "period_years": 29.447},
            {"a_au": 19.191, "period_years": 84.017},
            {"a_au": 30.070, "period_years": 164.79},
        ]
        challenge = [
            {"a_au": 2.77, "period_years": 4.61},   # Ceres-like
            {"a_au": 39.48, "period_years": 248.0},  # Pluto-like
            {"a_au": 43.1, "period_years": 283.0},   # Haumea-like
            {"a_au": 67.7, "period_years": 558.0},   # Eris-like
        ]

        train = self._sample_kepler_examples(rng=rng, rows=measured[:5], n=n_train, noise_scale=0.03)
        heldout = self._sample_kepler_examples(rng=rng, rows=measured[5:], n=n_heldout, noise_scale=0.01)
        ood = self._sample_kepler_examples(rng=rng, rows=challenge, n=max(8, n_heldout), noise_scale=0.02)
        stress = self._sample_kepler_examples(rng=rng, rows=challenge + measured, n=max(10, n_heldout), noise_scale=0.08)

        return BenchmarkTask(
            family=self.family_name,
            task_id=f"real_world_kepler_{seed}",
            description="Rediscover Kepler-like law from measured orbital observations",
            feature_names=["a_au"],
            train=train,
            heldout=heldout,
            ground_truth_expression="a_au**3",
            ground_truth_rule_text="Orbital period squared scales with semi-major axis cubed",
            metadata={
                "target_tolerance": 8.0,
                "pass_threshold": 0.70,
                "domain": "astronomy_observations",
                "scientific_law_candidate": "Kepler Third Law",
            },
            simulation_cases=self._simulation_from_examples(heldout),
            ood=ood,
            stress=stress,
        )

    def _pendulum_task(self, seed: int, rng: random.Random, n_train: int, n_heldout: int) -> BenchmarkTask:
        measured = [
            {"length_m": 0.10, "period_s": 0.64},
            {"length_m": 0.15, "period_s": 0.78},
            {"length_m": 0.20, "period_s": 0.90},
            {"length_m": 0.30, "period_s": 1.10},
            {"length_m": 0.40, "period_s": 1.27},
            {"length_m": 0.60, "period_s": 1.55},
            {"length_m": 0.80, "period_s": 1.79},
            {"length_m": 1.00, "period_s": 2.01},
            {"length_m": 1.20, "period_s": 2.20},
            {"length_m": 1.50, "period_s": 2.47},
        ]
        challenge = [
            {"length_m": 0.05, "period_s": 0.45},
            {"length_m": 1.80, "period_s": 2.70},
            {"length_m": 2.50, "period_s": 3.17},
            {"length_m": 3.00, "period_s": 3.48},
        ]

        train = self._sample_pendulum_examples(rng=rng, rows=measured[:6], n=n_train, noise_scale=0.03)
        heldout = self._sample_pendulum_examples(rng=rng, rows=measured[6:], n=n_heldout, noise_scale=0.01)
        ood = self._sample_pendulum_examples(rng=rng, rows=challenge, n=max(8, n_heldout), noise_scale=0.02)
        stress = self._sample_pendulum_examples(rng=rng, rows=challenge + measured, n=max(10, n_heldout), noise_scale=0.08)

        return BenchmarkTask(
            family=self.family_name,
            task_id=f"real_world_pendulum_{seed}",
            description="Rediscover pendulum period law from lab-style observations",
            feature_names=["length_m"],
            train=train,
            heldout=heldout,
            ground_truth_expression="2*math.pi*math.sqrt(length_m/9.81)",
            ground_truth_rule_text="Pendulum period scales with square root of length",
            metadata={
                "target_tolerance": 0.30,
                "pass_threshold": 0.75,
                "domain": "mechanics_observations",
                "scientific_law_candidate": "Pendulum Small-Angle Law",
            },
            simulation_cases=self._simulation_from_examples(heldout),
            ood=ood,
            stress=stress,
        )

    @staticmethod
    def _sample_kepler_examples(
        rng: random.Random,
        rows: list[dict[str, float]],
        n: int,
        noise_scale: float,
    ) -> list[ObservationExample]:
        out: list[ObservationExample] = []
        for i in range(max(1, n)):
            base = rows[i % len(rows)] if i < len(rows) else rng.choice(rows)
            a = max(0.02, base["a_au"] * (1.0 + rng.uniform(-noise_scale, noise_scale)))
            target = (base["period_years"] ** 2) * (1.0 + rng.uniform(-noise_scale, noise_scale))
            out.append(ObservationExample(features={"a_au": round(a, 6)}, target=round(target, 6)))
        return out[:n]

    @staticmethod
    def _sample_pendulum_examples(
        rng: random.Random,
        rows: list[dict[str, float]],
        n: int,
        noise_scale: float,
    ) -> list[ObservationExample]:
        out: list[ObservationExample] = []
        for i in range(max(1, n)):
            base = rows[i % len(rows)] if i < len(rows) else rng.choice(rows)
            length = max(0.01, base["length_m"] * (1.0 + rng.uniform(-noise_scale, noise_scale)))
            target = base["period_s"] * (1.0 + rng.uniform(-noise_scale, noise_scale))
            out.append(ObservationExample(features={"length_m": round(length, 6)}, target=round(target, 6)))
        return out[:n]

    @staticmethod
    def _simulation_from_examples(examples: list[ObservationExample]) -> list[SimulationCase]:
        return [
            SimulationCase(features=ex.features, expected_target=ex.target)
            for ex in examples[: min(12, len(examples))]
        ]
