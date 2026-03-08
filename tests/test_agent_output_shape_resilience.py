from dct.agents.collision_engine import CollisionEngine
from dct.agents.trajectory_a import TrajectoryAAgent
from dct.agents.trajectory_b import TrajectoryBAgent
from dct.benchmarks import BenchmarkRegistry
from dct.schemas import Hypothesis
from dct.utils import new_id


class MalformedProvider:
    def check_health(self):
        return True, "ok"

    def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 800):
        if "Trajectory A" in system_prompt:
            return {"hypotheses": "not-a-list"}
        if "Trajectory B" in system_prompt:
            return {"hypotheses": ["bad-item", {"expression": "x+1", "rule_text": "ok", "rationale": "r", "confidence": 0.5}]}
        if "Collision Engine" in system_prompt:
            return {"collision_hypotheses": "not-a-list"}
        return {}


def test_trajectory_agents_ignore_malformed_hypothesis_payloads():
    task = BenchmarkRegistry().generate("symbolic", seed=88, n_train=12, n_heldout=6)
    provider = MalformedProvider()

    a_agent = TrajectoryAAgent(provider)
    b_agent = TrajectoryBAgent(provider)

    a_hyps = a_agent.propose(task=task, round_index=0, memory_summaries=[], hypotheses_to_generate=2)
    b_hyps = b_agent.propose(task=task, round_index=0, memory_summaries=[], hypotheses_to_generate=2)

    assert a_hyps == []
    assert len(b_hyps) == 1
    assert b_hyps[0].expression == "x+1"


def test_collision_engine_ignores_malformed_collision_payload_and_falls_back():
    task = BenchmarkRegistry().generate("symbolic", seed=89, n_train=12, n_heldout=6)
    provider = MalformedProvider()
    engine = CollisionEngine(provider)

    hyp_a = Hypothesis(
        hypothesis_id=new_id("ha"),
        source="trajectory_a",
        round_index=0,
        family=task.family,
        task_id=task.task_id,
        rule_text="A",
        expression="x+1",
        rationale="",
        confidence=0.6,
    )
    hyp_b = Hypothesis(
        hypothesis_id=new_id("hb"),
        source="trajectory_b",
        round_index=0,
        family=task.family,
        task_id=task.task_id,
        rule_text="B",
        expression="x+2",
        rationale="",
        confidence=0.7,
    )

    out = engine.collide(
        task=task,
        round_index=0,
        hypotheses_a=[hyp_a],
        hypotheses_b=[hyp_b],
        memory_expressions=[],
        max_to_generate=1,
    )
    assert out
