from dct.agents.collision_engine import CollisionEngine
from dct.benchmarks import BenchmarkRegistry
from dct.schemas import Hypothesis
from dct.utils import new_id, normalize_expr
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


def test_collision_engine_filters_duplicate_parent_expression_and_uses_novel_fallback():
    task = BenchmarkRegistry().generate("symbolic", seed=321, n_train=12, n_heldout=6)
    feature = task.feature_names[0]
    expr_a = f"({feature} + 1)"
    expr_b = f"({feature} + 2)"

    class DuplicateCollisionProvider(FakeProvider):
        def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 800):
            if "Collision Engine" in system_prompt:
                return {
                    "collision_hypotheses": [
                        {
                            "rule_text": "duplicate-a",
                            "expression": expr_a,
                            "rationale": "dup",
                            "confidence": 0.9,
                        }
                    ]
                }
            return super().generate_json(system_prompt, user_prompt, max_tokens=max_tokens)

    engine = CollisionEngine(DuplicateCollisionProvider())

    a = Hypothesis(
        hypothesis_id=new_id("a"),
        source="trajectory_a",
        round_index=0,
        family=task.family,
        task_id=task.task_id,
        rule_text="A",
        expression=expr_a,
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
        expression=expr_b,
        rationale="",
        confidence=0.8,
    )

    out = engine.collide(
        task=task,
        round_index=0,
        hypotheses_a=[a],
        hypotheses_b=[b],
        memory_expressions=[expr_b],
        max_to_generate=1,
    )

    assert out
    normalized_expr = normalize_expr(out[0].expression)
    assert normalized_expr != normalize_expr(a.expression)
    assert normalized_expr != normalize_expr(b.expression)
