from __future__ import annotations

from dct.agents.common import build_discovery_prompt
from dct.llm.prompts import load_prompt
from dct.llm.provider import LLMProvider
from dct.schemas import BenchmarkTask, Hypothesis
from dct.utils import clamp01, new_id


class TrajectoryAAgent:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.system_prompt = load_prompt("trajectory_a.txt")

    def propose(
        self,
        task: BenchmarkTask,
        round_index: int,
        memory_summaries: list[str],
        hypotheses_to_generate: int,
    ) -> list[Hypothesis]:
        user_prompt = build_discovery_prompt(task, memory_summaries, hypotheses_to_generate)
        data = self.provider.generate_json(self.system_prompt, user_prompt)
        raw = data.get("hypotheses", [])
        if not isinstance(raw, list):
            raw = []

        hypotheses: list[Hypothesis] = []
        for item in raw[: hypotheses_to_generate or len(raw)]:
            if not isinstance(item, dict):
                continue
            expression = str(item.get("expression", "")).strip()
            if not expression:
                continue
            hypotheses.append(
                Hypothesis(
                    hypothesis_id=new_id("hypa"),
                    source="trajectory_a",
                    round_index=round_index,
                    family=task.family,
                    task_id=task.task_id,
                    rule_text=str(item.get("rule_text", ""))[:300],
                    expression=expression,
                    rationale=str(item.get("rationale", ""))[:500],
                    confidence=clamp01(float(item.get("confidence", 0.5))),
                )
            )
        return hypotheses
