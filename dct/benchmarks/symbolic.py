from __future__ import annotations

import random

from dct.benchmarks.base import BenchmarkFamily
from dct.schemas import BenchmarkTask, ObservationExample, SimulationCase


class SymbolicBenchmark(BenchmarkFamily):
    family_name = "symbolic"

    def generate_task(self, seed: int, n_train: int, n_heldout: int) -> BenchmarkTask:
        rng = random.Random(seed)
        variant = rng.choice(["mod_linear", "boolean_rule", "affine_pair"])

        if variant == "mod_linear":
            expr = "(3*x + 1) % 7"
            desc = "Integer mapping with modular arithmetic"
            feature_names = ["x"]
            train = [
                ObservationExample(features={"x": x}, target=(3 * x + 1) % 7)
                for x in self._sample_ints(rng, n_train, low=0, high=30)
            ]
            heldout = [
                ObservationExample(features={"x": x}, target=(3 * x + 1) % 7)
                for x in self._sample_ints(rng, n_heldout, low=31, high=60)
            ]
        elif variant == "boolean_rule":
            expr = "int(((a and (not b)) ^ c))"
            desc = "Boolean symbolic composition"
            feature_names = ["a", "b", "c"]
            pool = [{"a": a, "b": b, "c": c} for a in [0, 1] for b in [0, 1] for c in [0, 1]]
            rng.shuffle(pool)
            train_feats = pool[: min(n_train, len(pool))]
            held_feats = pool[min(n_train, len(pool)) : min(len(pool), n_train + n_heldout)]
            if len(held_feats) < n_heldout:
                held_feats.extend(pool[: n_heldout - len(held_feats)])
            train = [
                ObservationExample(
                    features=f,
                    target=int(((bool(f["a"]) and (not bool(f["b"]))) ^ bool(f["c"]))),
                )
                for f in train_feats
            ]
            heldout = [
                ObservationExample(
                    features=f,
                    target=int(((bool(f["a"]) and (not bool(f["b"]))) ^ bool(f["c"]))),
                )
                for f in held_feats[:n_heldout]
            ]
        else:
            expr = "2*x - y + 3"
            desc = "Two-variable algebraic rule"
            feature_names = ["x", "y"]
            train = []
            heldout = []
            for _ in range(n_train):
                x = rng.randint(-10, 10)
                y = rng.randint(-10, 10)
                train.append(ObservationExample(features={"x": x, "y": y}, target=2 * x - y + 3))
            for _ in range(n_heldout):
                x = rng.randint(11, 20)
                y = rng.randint(-20, 20)
                heldout.append(ObservationExample(features={"x": x, "y": y}, target=2 * x - y + 3))

        return BenchmarkTask(
            family=self.family_name,
            task_id=f"symbolic_{variant}_{seed}",
            description=desc,
            feature_names=feature_names,
            train=train,
            heldout=heldout,
            ground_truth_expression=expr,
            ground_truth_rule_text=desc,
            metadata={"target_tolerance": 0.0, "pass_threshold": 0.8},
            simulation_cases=self._simulation_from_heldout(heldout),
        )

    @staticmethod
    def _sample_ints(rng: random.Random, n: int, low: int, high: int) -> list[int]:
        space = list(range(low, high + 1))
        rng.shuffle(space)
        if n <= len(space):
            return space[:n]
        out = space.copy()
        while len(out) < n:
            out.append(rng.randint(low, high))
        return out

    @staticmethod
    def _simulation_from_heldout(heldout: list[ObservationExample]) -> list[SimulationCase]:
        return [
            SimulationCase(features=example.features, expected_target=example.target)
            for example in heldout[: min(10, len(heldout))]
        ]
