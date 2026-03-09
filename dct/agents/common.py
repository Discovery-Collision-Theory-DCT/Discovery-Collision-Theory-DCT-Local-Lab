from __future__ import annotations

import json
from typing import Any

from dct.schemas import BenchmarkTask
from dct.utils import safe_eval_expression


def build_discovery_prompt(
    task: BenchmarkTask,
    memory_summaries: list[str],
    hypotheses_to_generate: int,
) -> str:
    obs = [
        {"features": ex.features, "target": ex.target}
        for ex in task.train[: min(24, len(task.train))]
    ]
    payload = {
        "family": task.family,
        "task_id": task.task_id,
        "description": task.description,
        "feature_names": task.feature_names,
        "examples": obs,
        "hints": {
            "n_hypotheses": hypotheses_to_generate,
            "output_target_type": "numeric_or_boolean",
        },
        "memory_context": memory_summaries[:8],
    }
    return json.dumps(payload, ensure_ascii=True, indent=2)


def expression_is_executable(task: BenchmarkTask, expression: str, max_examples: int = 8) -> bool:
    expr = (expression or "").strip()
    if not expr:
        return False

    examples = task.train[: min(max_examples, len(task.train))]
    if not examples:
        return True

    for ex in examples:
        try:
            safe_eval_expression(expr, ex.features)
        except ValueError:
            return False
    return True


def safe_confidence(value: Any, default: float = 0.5) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)
