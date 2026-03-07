from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any


SAFE_GLOBALS = {
    "__builtins__": {},
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "int": int,
    "float": float,
    "bool": bool,
    "pow": pow,
    "math": math,
}


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def try_parse_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return json.loads(stripped[start : end + 1])
    raise ValueError("No JSON object found in model response")


def safe_eval_expression(expression: str, features: dict[str, Any]) -> Any:
    local_vars = {k: v for k, v in features.items()}
    try:
        return eval(expression, SAFE_GLOBALS, local_vars)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Expression evaluation failed: {exc}") from exc


def normalize_expr(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "")


def token_set(expr: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+|[+\-*/%<>=!&|^]+", expr or ""))


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))
