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
        parsed = _parse_candidate(candidate)
        if parsed is not None:
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


def _parse_candidate(candidate: str) -> dict[str, Any] | None:
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

    parsed = _recover_partial_known_schema(repaired)
    if isinstance(parsed, dict):
        return parsed

    parsed = _recover_truncated_json_object(repaired)
    if isinstance(parsed, dict):
        return parsed
    return None


def _recover_truncated_json_object(text: str) -> dict[str, Any] | None:
    source = text.strip()
    if not source:
        return None

    # Trim to JSON-looking prefix.
    first_brace = source.find("{")
    if first_brace > 0:
        source = source[first_brace:]

    max_trim = min(280, max(0, len(source) - 1))
    for trim in range(0, max_trim + 1):
        attempt = source[:-trim] if trim else source
        attempt = _trim_incomplete_suffix(attempt)
        if not attempt:
            continue

        balanced = _append_missing_json_closers(attempt)
        balanced = _repair_common_json_issues(balanced)

        parsed = _try_json_loads(balanced)
        if isinstance(parsed, dict):
            return parsed

        parsed = _try_literal_eval_dict(balanced)
        if isinstance(parsed, dict):
            return parsed

    return None


def _trim_incomplete_suffix(text: str) -> str:
    out = text.rstrip()
    if not out:
        return out
    out = re.sub(r"```[a-zA-Z0-9_-]*\s*$", "", out).rstrip()
    while out and out[-1] in {":", ",", "\\", "`"}:
        out = out[:-1].rstrip()
    return out


def _append_missing_json_closers(text: str) -> str:
    stack: list[str] = []
    in_string = False
    escaped = False

    for ch in text:
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
        elif ch == "{":
            stack.append("}")
        elif ch == "[":
            stack.append("]")
        elif ch in {"}", "]"} and stack and ch == stack[-1]:
            stack.pop()

    out = text
    if in_string:
        if out.endswith("\\"):
            out = out[:-1]
        out += "\""
    if stack:
        out += "".join(reversed(stack))
    return out


def _recover_partial_known_schema(text: str) -> dict[str, Any] | None:
    lower = text.lower()

    if "\"hypotheses\"" in lower:
        hypotheses = _recover_hypothesis_list(text)
        if hypotheses:
            return {"hypotheses": hypotheses}

    if "\"collision_hypotheses\"" in lower:
        hypotheses = _recover_hypothesis_list(text)
        if hypotheses:
            return {"collision_hypotheses": hypotheses}

    if "\"pass\"" in lower or "\"confidence\"" in lower or "\"reason\"" in lower:
        verdict = _recover_verdict(text)
        if verdict:
            return verdict

    return None


def _recover_hypothesis_list(text: str) -> list[dict[str, Any]]:
    rule_texts = re.findall(r'"rule_text"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)
    expressions = re.findall(r'"expression"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)
    rationales = re.findall(r'"rationale"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)
    confidences = re.findall(r'"confidence"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', text)

    if not expressions:
        return []

    n = len(expressions)
    items: list[dict[str, Any]] = []
    for i in range(n):
        expression = _decode_json_escaped(expressions[i]).strip()
        if not expression:
            continue
        rule_text = _decode_json_escaped(rule_texts[i]).strip() if i < len(rule_texts) else "Recovered candidate"
        rationale = _decode_json_escaped(rationales[i]).strip() if i < len(rationales) else "Recovered from partial JSON"
        confidence = 0.5
        if i < len(confidences):
            try:
                confidence = float(confidences[i])
            except ValueError:
                confidence = 0.5
        items.append(
            {
                "rule_text": rule_text[:400],
                "expression": expression[:600],
                "rationale": rationale[:400],
                "confidence": confidence,
            }
        )
    return items


def _recover_verdict(text: str) -> dict[str, Any] | None:
    pass_match = re.search(r'"pass"\s*:\s*(true|false)', text, flags=re.IGNORECASE)
    conf_match = re.search(r'"confidence"\s*:\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', text)
    reason_match = re.search(r'"reason"\s*:\s*"((?:[^"\\]|\\.)*)"', text, flags=re.DOTALL)

    if not pass_match and not conf_match and not reason_match:
        return None

    verdict: dict[str, Any] = {}
    if pass_match:
        verdict["pass"] = pass_match.group(1).lower() == "true"
    if conf_match:
        try:
            verdict["confidence"] = float(conf_match.group(1))
        except ValueError:
            pass
    if reason_match:
        verdict["reason"] = _decode_json_escaped(reason_match.group(1))[:600]
    return verdict if verdict else None


def _decode_json_escaped(value: str) -> str:
    raw = value or ""
    try:
        return json.loads(f"\"{raw}\"")
    except Exception:  # noqa: BLE001
        return raw.replace('\\"', '"').replace("\\n", "\n")


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
