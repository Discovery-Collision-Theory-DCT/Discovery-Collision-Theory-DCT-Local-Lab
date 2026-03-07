from dct.agents.collision_engine import CollisionEngine
from dct.benchmarks import BenchmarkRegistry
from dct.schemas import Hypothesis
from dct.utils import new_id
from tests.fake_provider import FakeProvider


def test_collision_engine_returns_candidates():
    task = BenchmarkRegistry().generate("symbolic", seed=123, n_train=10, n_heldout=6)
    engine = CollisionEngine(FakeProvider())

    a = Hypothesis(
        hypothesis_id=new_id("a"),
        source="trajectory_a",
        round_index=0,
        family=task.family,
        task_id=task.task_id,
        rule_text="A",
        expression="(3*x + 1) % 7",
        rationale="",
        confidence=0.9,
    )
    b = Hypothesis(
        hypothesis_id=new_id("b"),
        source="trajectory_b",
        round_index=0,
        family=task.family,
        task_id=task.task_id,
        rule_text="B",
        expression="(x + 1) % 7",
        rationale="",
        confidence=0.8,
    )

    out = engine.collide(task, round_index=0, hypotheses_a=[a], hypotheses_b=[b], memory_expressions=[])
    assert out
    assert out[0].source == "collision"
    assert "novelty" in out[0].scores
