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
