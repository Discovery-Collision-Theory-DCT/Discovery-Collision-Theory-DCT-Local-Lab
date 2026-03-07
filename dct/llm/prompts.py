from __future__ import annotations

from importlib.resources import files


PROMPT_PACKAGE = "dct.prompts"


def load_prompt(name: str) -> str:
    return files(PROMPT_PACKAGE).joinpath(name).read_text(encoding="utf-8")
