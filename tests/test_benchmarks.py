from dct.benchmarks import BenchmarkRegistry


def test_benchmark_registry_generates_all_families():
    registry = BenchmarkRegistry()
    assert set(registry.families()) == {
        "symbolic",
        "dynamical",
        "compression",
        "real_world_laws",
        "autonomy_generalization",
        "open_world_noise",
    }

    for idx, family in enumerate(registry.families()):
        task = registry.generate(family=family, seed=100 + idx, n_train=12, n_heldout=6)
        assert task.family == family
        assert len(task.train) == 12
        assert len(task.heldout) == 6
        assert task.ground_truth_expression
        assert task.feature_names
        assert isinstance(task.ood, list)
        assert isinstance(task.stress, list)
