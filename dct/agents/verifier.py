from __future__ import annotations

import json
from statistics import mean
from typing import Any

from dct.llm.prompts import load_prompt
from dct.llm.provider import LLMProvider
from dct.schemas import BenchmarkTask, Hypothesis, VerificationResult, VerifierMode, VerifierVerdict
from dct.utils import clamp01, safe_eval_expression


class Verifier:
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.system_prompt = load_prompt("verifier.txt")

    def verify(
        self,
        task: BenchmarkTask,
        hypothesis: Hypothesis,
        modes: list[str],
    ) -> VerifierVerdict:
        selected_modes = [VerifierMode(m) for m in modes]
        results: list[VerificationResult] = []

        for mode in selected_modes:
            if mode == VerifierMode.predictive:
                results.append(self._predictive_verify(task, hypothesis))
            elif mode == VerifierMode.symbolic:
                results.append(self._symbolic_verify(task, hypothesis))
            elif mode == VerifierMode.simulation:
                results.append(self._simulation_verify(task, hypothesis))

        if not results:
            results.append(self._predictive_verify(task, hypothesis))

        modes_passed = all(r.passed for r in results)
        robustness = self._robustness_gate(task, hypothesis)
        deterministic_pass = modes_passed and robustness["passed"]

        confidence_parts = [r.confidence for r in results]
        if robustness["confidence"] is not None:
            confidence_parts.append(float(robustness["confidence"]))
        deterministic_confidence = clamp01(mean(confidence_parts)) if confidence_parts else 0.0

        mode_reason = "; ".join(f"{r.mode.value}: {r.reason}" for r in results)
        deterministic_reason = f"{mode_reason}; robustness: {robustness['reason']}"

        llm_reason, llm_conf = self._llm_verifier_note(hypothesis, task, results, robustness)
        final_reason = deterministic_reason if not llm_reason else f"{deterministic_reason}; llm_note: {llm_reason}"
        final_conf = deterministic_confidence if llm_conf is None else clamp01((deterministic_confidence + llm_conf) / 2.0)

        return VerifierVerdict(
            hypothesis_id=hypothesis.hypothesis_id,
            passed=deterministic_pass,
            confidence=final_conf,
            reason=final_reason,
            per_mode=results,
        )

    def _robustness_gate(self, task: BenchmarkTask, hypothesis: Hypothesis) -> dict[str, Any]:
        pass_threshold = float(task.metadata.get("pass_threshold", 0.75))
        ood_threshold = float(task.metadata.get("ood_pass_threshold", max(0.55, pass_threshold - 0.10)))
        stress_threshold = float(task.metadata.get("stress_pass_threshold", max(0.50, pass_threshold - 0.15)))

        ood_acc, ood_n = self._accuracy(task.ood, hypothesis.expression, task)
        stress_acc, stress_n = self._accuracy(task.stress, hypothesis.expression, task)

        checks = []
        confidence_parts: list[float] = []

        if ood_n > 0:
            ood_passed = ood_acc >= ood_threshold
            checks.append(ood_passed)
            confidence_parts.append(ood_acc)
            ood_reason = f"ood_accuracy={ood_acc:.3f} threshold={ood_threshold:.3f}"
        else:
            ood_passed = True
            ood_reason = "ood_n=0 skipped"

        if stress_n > 0:
            stress_passed = stress_acc >= stress_threshold
            checks.append(stress_passed)
            confidence_parts.append(stress_acc)
            stress_reason = f"stress_accuracy={stress_acc:.3f} threshold={stress_threshold:.3f}"
        else:
            stress_passed = True
            stress_reason = "stress_n=0 skipped"

        passed = all(checks) if checks else True
        confidence = mean(confidence_parts) if confidence_parts else None

        return {
            "passed": bool(passed),
            "confidence": confidence,
            "ood_passed": bool(ood_passed),
            "stress_passed": bool(stress_passed),
            "reason": f"{ood_reason}; {stress_reason}",
            "metrics": {
                "ood_accuracy": float(ood_acc),
                "ood_n": float(ood_n),
                "stress_accuracy": float(stress_acc),
                "stress_n": float(stress_n),
                "ood_threshold": float(ood_threshold),
                "stress_threshold": float(stress_threshold),
            },
        }

    def _predictive_verify(self, task: BenchmarkTask, hypothesis: Hypothesis) -> VerificationResult:
        acc, total = self._accuracy(task.heldout, hypothesis.expression, task)
        threshold = float(task.metadata.get("pass_threshold", 0.75))
        passed = acc >= threshold
        return VerificationResult(
            hypothesis_id=hypothesis.hypothesis_id,
            mode=VerifierMode.predictive,
            passed=passed,
            confidence=acc,
            reason=f"heldout_accuracy={acc:.3f} threshold={threshold:.3f}",
            metrics={"heldout_accuracy": acc, "n": float(total)},
        )

    def _symbolic_verify(self, task: BenchmarkTask, hypothesis: Hypothesis) -> VerificationResult:
        acc, total = self._accuracy(task.train, hypothesis.expression, task)

        if float(task.metadata.get("target_tolerance", 0.0)) == 0.0:
            threshold = 0.95
            reason = f"train_exact_consistency={acc:.3f}"
        else:
            threshold = 0.70
            reason = f"formal_approx_due_noise train_consistency={acc:.3f}"

        passed = acc >= threshold
        return VerificationResult(
            hypothesis_id=hypothesis.hypothesis_id,
            mode=VerifierMode.symbolic,
            passed=passed,
            confidence=acc,
            reason=f"{reason} threshold={threshold:.3f}",
            metrics={"train_accuracy": acc, "n": float(total)},
        )

    def _simulation_verify(self, task: BenchmarkTask, hypothesis: Hypothesis) -> VerificationResult:
        if task.simulation_cases:
            pseudo_examples = [
                type("Tmp", (), {"features": c.features, "target": c.expected_target}) for c in task.simulation_cases
            ]
            acc, total = self._accuracy(pseudo_examples, hypothesis.expression, task)
        else:
            acc, total = self._accuracy(task.heldout, hypothesis.expression, task)

        threshold = 0.70
        passed = acc >= threshold
        return VerificationResult(
            hypothesis_id=hypothesis.hypothesis_id,
            mode=VerifierMode.simulation,
            passed=passed,
            confidence=acc,
            reason=f"simulation_accuracy={acc:.3f} threshold={threshold:.3f}",
            metrics={"simulation_accuracy": acc, "n": float(total)},
        )

    def _llm_verifier_note(
        self,
        hypothesis: Hypothesis,
        task: BenchmarkTask,
        mode_results: list[VerificationResult],
        robustness: dict[str, Any],
    ) -> tuple[str | None, float | None]:
        payload = {
            "task": {
                "family": task.family,
                "task_id": task.task_id,
                "description": task.description,
            },
            "hypothesis": {
                "rule_text": hypothesis.rule_text,
                "expression": hypothesis.expression,
                "source": hypothesis.source,
            },
            "mode_results": [r.model_dump(mode="json") for r in mode_results],
            "robustness_gate": robustness,
        }
        try:
            data = self.provider.generate_json(self.system_prompt, json.dumps(payload, ensure_ascii=True))
        except Exception:  # noqa: BLE001
            return None, None

        reason = data.get("reason")
        conf = data.get("confidence")
        try:
            conf_float = None if conf is None else float(conf)
        except (TypeError, ValueError):
            conf_float = None
        return str(reason) if reason else None, conf_float

    @staticmethod
    def _accuracy(examples: list[Any], expression: str, task: BenchmarkTask) -> tuple[float, int]:
        if not examples:
            return 0.0, 0

        tol = float(task.metadata.get("target_tolerance", 0.0))
        correct = 0
        total = 0

        for sample in examples:
            total += 1
            try:
                pred = safe_eval_expression(expression, sample.features)
            except ValueError:
                continue

            target = sample.target
            if isinstance(pred, (int, float)) and isinstance(target, (int, float)):
                if abs(float(pred) - float(target)) <= max(1e-9, tol):
                    correct += 1
            else:
                if pred == target:
                    correct += 1

        if total == 0:
            return 0.0, 0
        return correct / total, total
