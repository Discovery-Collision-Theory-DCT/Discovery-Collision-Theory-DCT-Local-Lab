from dct.agents.verifier import Verifier
from dct.benchmarks import BenchmarkRegistry
from dct.schemas import BenchmarkTask, Hypothesis, ObservationExample, SimulationCase
from dct.utils import new_id
from tests.fake_provider import FakeProvider


def test_verifier_modes_output_structured_results():
    task = BenchmarkRegistry().generate("symbolic", seed=123, n_train=12, n_heldout=8)
    verifier = Verifier(FakeProvider())

    hypothesis = Hypothesis(
        hypothesis_id=new_id("hyp"),
        source="trajectory_a",
        round_index=0,
        family=task.family,
        task_id=task.task_id,
        rule_text="Candidate",
        expression="(3*x + 1) % 7",
        rationale="",
        confidence=0.9,
    )

    verdict = verifier.verify(task, hypothesis, ["predictive", "symbolic", "simulation"])
    assert verdict.hypothesis_id == hypothesis.hypothesis_id
    assert isinstance(verdict.passed, bool)
    assert 0.0 <= verdict.confidence <= 1.0
    assert len(verdict.per_mode) == 3


def _build_small_task(
    *,
    ood_targets: list[float],
    stress_targets: list[float],
) -> BenchmarkTask:
    train = [
        ObservationExample(features={"x": 1.0}, target=1.0),
        ObservationExample(features={"x": 2.0}, target=2.0),
    ]
    heldout = [
        ObservationExample(features={"x": 3.0}, target=3.0),
        ObservationExample(features={"x": 4.0}, target=4.0),
    ]
    ood = [ObservationExample(features={"x": float(i + 5)}, target=t) for i, t in enumerate(ood_targets)]
    stress = [ObservationExample(features={"x": float(i + 7)}, target=t) for i, t in enumerate(stress_targets)]
    sim = [SimulationCase(features={"x": 5.0}, expected_target=5.0)]

    return BenchmarkTask(
        family="unit_test",
        task_id="verifier_robustness_gate",
        description="verifier robustness behavior",
        feature_names=["x"],
        train=train,
        heldout=heldout,
        ground_truth_expression="x",
        ground_truth_rule_text="identity",
        metadata={
            "target_tolerance": 0.0,
            "pass_threshold": 0.75,
            "ood_pass_threshold": 0.75,
            "stress_pass_threshold": 0.70,
        },
        simulation_cases=sim,
        ood=ood,
        stress=stress,
    )


def test_verifier_rejects_if_robustness_gate_fails():
    task = _build_small_task(
        ood_targets=[0.0, 0.0],      # expression=x -> ood accuracy should fail
        stress_targets=[0.0, 0.0],   # expression=x -> stress accuracy should fail
    )
    verifier = Verifier(FakeProvider())
    hypothesis = Hypothesis(
        hypothesis_id=new_id("hyp"),
        source="trajectory_a",
        round_index=0,
        family=task.family,
        task_id=task.task_id,
        rule_text="identity",
        expression="x",
        rationale="",
        confidence=0.9,
    )

    verdict = verifier.verify(task, hypothesis, ["predictive", "symbolic", "simulation"])
    assert verdict.passed is False
    assert "robustness:" in verdict.reason
    assert "ood_accuracy=" in verdict.reason
    assert "stress_accuracy=" in verdict.reason


def test_verifier_accepts_if_robustness_gate_passes():
    task = _build_small_task(
        ood_targets=[5.0, 6.0],      # expression=x -> ood accuracy should pass
        stress_targets=[7.0, 8.0],   # expression=x -> stress accuracy should pass
    )
    verifier = Verifier(FakeProvider())
    hypothesis = Hypothesis(
        hypothesis_id=new_id("hyp"),
        source="trajectory_b",
        round_index=0,
        family=task.family,
        task_id=task.task_id,
        rule_text="identity",
        expression="x",
        rationale="",
        confidence=0.9,
    )

    verdict = verifier.verify(task, hypothesis, ["predictive", "symbolic", "simulation"])
    assert verdict.passed is True
