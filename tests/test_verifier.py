from dct.agents.verifier import Verifier
from dct.benchmarks import BenchmarkRegistry
from dct.schemas import Hypothesis
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
