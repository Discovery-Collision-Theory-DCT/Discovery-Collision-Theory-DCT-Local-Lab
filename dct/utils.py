from __future__ import annotations

import json
import math
import re
import uuid
from ast import literal_eval
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
    stripped = _strip_fences(text.strip())

    candidates = _json_candidates(stripped)
    for candidate in candidates:
        parsed = _try_json_loads(candidate)
        if isinstance(parsed, dict):
            return parsed

        repaired = _repair_common_json_issues(candidate)
        parsed = _try_json_loads(repaired)
        if isinstance(parsed, dict):
            return parsed

        parsed = _try_literal_eval_dict(repaired)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("No valid JSON object found in model response")


def _strip_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    return cleaned.strip()


def _json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    if text:
        candidates.append(text)

    fence_matches = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    candidates.extend(fence_matches)

    brace_spans = _balanced_object_spans(text)
    for start, end in brace_spans:
        candidates.append(text[start : end + 1])

    unique = []
    seen = set()
    for c in candidates:
        normalized = c.strip()
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _balanced_object_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    stack: list[int] = []
    in_string = False
    escaped = False

    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "\"":
                in_string = False
            continue

        if ch == "\"":
            in_string = True
            continue

        if ch == "{":
            stack.append(i)
        elif ch == "}":
            if not stack:
                continue
            start = stack.pop()
            if not stack:
                spans.append((start, i))

    spans.sort(key=lambda t: (t[1] - t[0]), reverse=True)
    return spans


def _try_json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _repair_common_json_issues(text: str) -> str:
    repaired = text
    repaired = repaired.replace("“", "\"").replace("”", "\"").replace("’", "'")
    repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
    return repaired


def _try_literal_eval_dict(text: str) -> dict[str, Any] | None:
    try:
        value = literal_eval(text)
    except Exception:  # noqa: BLE001
        return None
    if isinstance(value, dict):
        return value
    return None


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
