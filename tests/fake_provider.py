from __future__ import annotations


class FakeProvider:
    def check_health(self):
        return True, "ok"

    def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 800):
        if "Trajectory A" in system_prompt:
            return {
                "hypotheses": [
                    {
                        "rule_text": "Pattern candidate",
                        "expression": "(3*x + 1) % 7",
                        "rationale": "Compact symbolic fit",
                        "confidence": 0.9,
                    },
                    {
                        "rule_text": "Fallback candidate",
                        "expression": "x",
                        "rationale": "Simple baseline",
                        "confidence": 0.2,
                    },
                ]
            }

        if "Trajectory B" in system_prompt:
            return {
                "hypotheses": [
                    {
                        "rule_text": "Mechanistic candidate",
                        "expression": "(3*x + 1) % 7",
                        "rationale": "Counterfactual behavior aligns",
                        "confidence": 0.85,
                    }
                ]
            }

        if "Collision Engine" in system_prompt:
            return {
                "collision_hypotheses": [
                    {
                        "rule_text": "Merged rule",
                        "expression": "(3*x + 1) % 7",
                        "rationale": "Synthesized equivalent structure",
                        "confidence": 0.95,
                    }
                ]
            }

        if "strict verification reporter" in system_prompt:
            return {"pass": True, "confidence": 0.8, "reason": "metrics support acceptance"}

        return {}
