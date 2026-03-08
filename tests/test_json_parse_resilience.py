from dct.utils import try_parse_json


def test_try_parse_json_from_reasoning_and_markdown_block():
    text = """
<think>analyzing</think>
Here is output:
```json
{
  "hypotheses": [
    {"rule_text": "r", "expression": "x+1", "rationale": "fit", "confidence": 0.8}
  ]
}
```
"""
    data = try_parse_json(text)
    assert "hypotheses" in data
    assert data["hypotheses"][0]["expression"] == "x+1"


def test_try_parse_json_repairs_trailing_comma_and_smart_quotes():
    text = '{“pass”: true, “confidence”: 0.9, “reason”: “ok”,}'
    data = try_parse_json(text)
    assert data["pass"] is True
    assert data["confidence"] == 0.9


def test_try_parse_json_accepts_python_dict_style_fallback():
    text = "{'pass': True, 'confidence': 0.5, 'reason': 'fallback'}"
    data = try_parse_json(text)
    assert data["pass"] is True
    assert data["reason"] == "fallback"


def test_try_parse_json_recovers_truncated_hypotheses_payload():
    text = """```json
{
  "hypotheses": [
    {
      "rule_text": "Target equals 2x minus y plus 1",
      "expression": "2*x - y + 1",
      "rationale": "Checking examples quickly",
      "c
"""
    data = try_parse_json(text)
    assert "hypotheses" in data
    assert data["hypotheses"][0]["expression"] == "2*x - y + 1"
    assert data["hypotheses"][0]["rule_text"].startswith("Target equals")


def test_try_parse_json_recovers_partial_verifier_payload():
    text = '{"pass": true, "confidence": 0.92, "reason": "heldout metrics strong",'
    data = try_parse_json(text)
    assert data["pass"] is True
    assert data["confidence"] == 0.92
