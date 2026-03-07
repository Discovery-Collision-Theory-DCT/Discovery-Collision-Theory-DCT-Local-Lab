from __future__ import annotations

import json
from dataclasses import dataclass

from dct.llm.prompts import load_prompt
from dct.llm.provider import LLMProvider
from dct.schemas import BenchmarkTask, CollisionScore, Hypothesis
from dct.utils import clamp01, jaccard_similarity, new_id, safe_eval_expression, token_set


@dataclass
class PairScore:
    a: Hypothesis
    b: Hypothesis
    score: CollisionScore


class CollisionEngine:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.system_prompt = load_prompt("collision.txt")

    def collide(
        self,
        task: BenchmarkTask,
        round_index: int,
        hypotheses_a: list[Hypothesis],
        hypotheses_b: list[Hypothesis],
        memory_expressions: list[str],
        max_to_generate: int = 2,
    ) -> list[Hypothesis]:
        if not hypotheses_a or not hypotheses_b:
            return []

        pair_scores = self._score_pairs(task, hypotheses_a, hypotheses_b, memory_expressions)
        top_pairs = sorted(pair_scores, key=lambda p: p.score.collision_strength, reverse=True)[:3]

        user_prompt = json.dumps(
            {
                "family": task.family,
                "task_id": task.task_id,
                "feature_names": task.feature_names,
                "examples": [
                    {"features": ex.features, "target": ex.target}
                    for ex in task.train[: min(20, len(task.train))]
                ],
                "top_pairs": [
                    {
                        "a": {
                            "id": p.a.hypothesis_id,
                            "expression": p.a.expression,
                            "rule_text": p.a.rule_text,
                            "confidence": p.a.confidence,
                        },
                        "b": {
                            "id": p.b.hypothesis_id,
                            "expression": p.b.expression,
                            "rule_text": p.b.rule_text,
                            "confidence": p.b.confidence,
                        },
                        "score": p.score.model_dump(),
                    }
                    for p in top_pairs
                ],
                "memory_expressions": memory_expressions[:20],
            },
            ensure_ascii=True,
            indent=2,
        )

        llm_candidates: list[Hypothesis] = []
        try:
            data = self.provider.generate_json(self.system_prompt, user_prompt)
            for item in data.get("collision_hypotheses", [])[:max_to_generate]:
                expression = str(item.get("expression", "")).strip()
                if not expression:
                    continue
                best_pair = top_pairs[0]
                llm_candidates.append(
                    Hypothesis(
                        hypothesis_id=new_id("hypc"),
                        source="collision",
                        round_index=round_index,
                        family=task.family,
                        task_id=task.task_id,
                        rule_text=str(item.get("rule_text", ""))[:300],
                        expression=expression,
                        rationale=str(item.get("rationale", ""))[:500],
                        confidence=clamp01(float(item.get("confidence", best_pair.score.collision_strength))),
                        parents=[best_pair.a.hypothesis_id, best_pair.b.hypothesis_id],
                        scores=best_pair.score.model_dump(),
                    )
                )
        except Exception:  # noqa: BLE001
            llm_candidates = []

        if llm_candidates:
            return llm_candidates
        return self._heuristic_collision(top_pairs, task, round_index, max_to_generate)

    def _score_pairs(
        self,
        task: BenchmarkTask,
        hypotheses_a: list[Hypothesis],
        hypotheses_b: list[Hypothesis],
        memory_expressions: list[str],
    ) -> list[PairScore]:
        scored: list[PairScore] = []
        memory_tokens = [token_set(expr) for expr in memory_expressions]

        for a in hypotheses_a:
            for b in hypotheses_b:
                tokens_a = token_set(a.expression)
                tokens_b = token_set(b.expression)
                sim = jaccard_similarity(tokens_a, tokens_b)
                structural = clamp01(1.0 - abs(sim - 0.5) / 0.5)

                predictive_overlap = self._predictive_overlap(task, a.expression, b.expression)
                explanatory_gain = clamp01((a.confidence + b.confidence) / 2.0)

                merged_tokens = tokens_a | tokens_b
                if memory_tokens:
                    max_mem_sim = max(jaccard_similarity(merged_tokens, mt) for mt in memory_tokens)
                    novelty = clamp01(1.0 - max_mem_sim)
                else:
                    novelty = 1.0

                strength = clamp01(
                    0.3 * structural + 0.3 * predictive_overlap + 0.2 * explanatory_gain + 0.2 * novelty
                )

                scored.append(
                    PairScore(
                        a=a,
                        b=b,
                        score=CollisionScore(
                            structural_complementarity=structural,
                            predictive_overlap=predictive_overlap,
                            explanatory_gain=explanatory_gain,
                            novelty=novelty,
                            collision_strength=strength,
                        ),
                    )
                )

        return scored

    @staticmethod
    def _predictive_overlap(task: BenchmarkTask, expr_a: str, expr_b: str) -> float:
        total = 0
        agree = 0
        tol = float(task.metadata.get("target_tolerance", 0.0))

        for sample in task.heldout[: min(16, len(task.heldout))]:
            try:
                pred_a = safe_eval_expression(expr_a, sample.features)
                pred_b = safe_eval_expression(expr_b, sample.features)
            except ValueError:
                continue

            total += 1
            if isinstance(pred_a, (int, float)) and isinstance(pred_b, (int, float)):
                if abs(float(pred_a) - float(pred_b)) <= max(tol, 1e-6):
                    agree += 1
            elif pred_a == pred_b:
                agree += 1

        if total == 0:
            return 0.0
        return agree / total

    def _heuristic_collision(
        self,
        top_pairs: list[PairScore],
        task: BenchmarkTask,
        round_index: int,
        max_to_generate: int,
    ) -> list[Hypothesis]:
        out: list[Hypothesis] = []
        for pair in top_pairs[:max_to_generate]:
            pick = pair.a if pair.a.confidence >= pair.b.confidence else pair.b
            out.append(
                Hypothesis(
                    hypothesis_id=new_id("hypc"),
                    source="collision",
                    round_index=round_index,
                    family=task.family,
                    task_id=task.task_id,
                    rule_text=f"Heuristic collision pick from {pair.a.hypothesis_id} + {pair.b.hypothesis_id}",
                    expression=pick.expression,
                    rationale=(
                        "Collision fallback selected higher-confidence parent while preserving "
                        "pairwise complementarity signal."
                    ),
                    confidence=pair.score.collision_strength,
                    parents=[pair.a.hypothesis_id, pair.b.hypothesis_id],
                    scores=pair.score.model_dump(),
                )
            )
        return out
