from __future__ import annotations

import random

from dct.benchmarks.base import BenchmarkFamily
from dct.schemas import BenchmarkTask, ObservationExample, SimulationCase


class DynamicalBenchmark(BenchmarkFamily):
    family_name = "dynamical"

    def generate_task(self, seed: int, n_train: int, n_heldout: int) -> BenchmarkTask:
        rng = random.Random(seed)
        variant = rng.choice(["cellular_local", "graph_like", "finite_state"])

        if variant == "cellular_local":
            expr = "int((l and not c) or r)"
            desc = "1D cellular local transition from neighborhood"
            feature_names = ["l", "c", "r"]

            def fn(f: dict[str, int]) -> int:
                return int((bool(f["l"]) and (not bool(f["c"]))) or bool(f["r"]))

            feature_space = [{"l": l, "c": c, "r": r} for l in [0, 1] for c in [0, 1] for r in [0, 1]]
            rng.shuffle(feature_space)
            train_feats = feature_space[: min(len(feature_space), n_train)]
            held_feats = feature_space[min(len(feature_space), n_train) :]
            if len(held_feats) < n_heldout:
                held_feats += feature_space[: n_heldout - len(held_feats)]
            train = [ObservationExample(features=f, target=fn(f)) for f in train_feats]
            heldout = [ObservationExample(features=f, target=fn(f)) for f in held_feats[:n_heldout]]

        elif variant == "graph_like":
            expr = "int(node_state or (active_neighbors >= 2))"
            desc = "Graph propagation with threshold activation"
            feature_names = ["node_state", "active_neighbors"]

            def fn(f: dict[str, int]) -> int:
                return int(bool(f["node_state"]) or f["active_neighbors"] >= 2)

            train = []
            heldout = []
            for _ in range(n_train):
                f = {"node_state": rng.randint(0, 1), "active_neighbors": rng.randint(0, 4)}
                train.append(ObservationExample(features=f, target=fn(f)))
            for _ in range(n_heldout):
                f = {"node_state": rng.randint(0, 1), "active_neighbors": rng.randint(0, 4)}
                heldout.append(ObservationExample(features=f, target=fn(f)))

        else:
            expr = "(state + signal + 1) % 3"
            desc = "Finite-state transition with hidden update arithmetic"
            feature_names = ["state", "signal"]

            def fn(f: dict[str, int]) -> int:
                return (f["state"] + f["signal"] + 1) % 3

            train = []
            heldout = []
            for _ in range(n_train):
                f = {"state": rng.randint(0, 2), "signal": rng.randint(0, 1)}
                train.append(ObservationExample(features=f, target=fn(f)))
            for _ in range(n_heldout):
                f = {"state": rng.randint(0, 2), "signal": rng.randint(0, 1)}
                heldout.append(ObservationExample(features=f, target=fn(f)))

        simulation_cases = [
            SimulationCase(features=example.features, expected_target=example.target)
            for example in heldout[: min(12, len(heldout))]
        ]

        return BenchmarkTask(
            family=self.family_name,
            task_id=f"dynamical_{variant}_{seed}",
            description=desc,
            feature_names=feature_names,
            train=train,
            heldout=heldout,
            ground_truth_expression=expr,
            ground_truth_rule_text=desc,
            metadata={"target_tolerance": 0.0, "pass_threshold": 0.75},
            simulation_cases=simulation_cases,
        )
