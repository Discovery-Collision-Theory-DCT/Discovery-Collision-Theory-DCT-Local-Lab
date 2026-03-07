from __future__ import annotations

import json

from dct.schemas import BenchmarkTask


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
