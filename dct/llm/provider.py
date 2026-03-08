from __future__ import annotations

from typing import Protocol

import httpx

from dct.config import OPENAI_COMPATIBLE_PROVIDERS, RuntimeSettings
from dct.utils import try_parse_json


class ModelUnavailableError(RuntimeError):
    pass


class LLMProvider(Protocol):
    def check_health(self) -> tuple[bool, str]:
        ...

    def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> dict:
        ...


class OpenAICompatibleProvider:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self.base_url = settings.openai_base_url.rstrip("/")
        self.client = httpx.Client(timeout=settings.model_timeout_seconds)

    def check_health(self) -> tuple[bool, str]:
        models_url = f"{self.base_url}/models"
        try:
            resp = self.client.get(models_url, headers=self._headers())
            if resp.status_code == 200:
                return True, f"Model endpoint reachable: {models_url}"
            return False, f"Model endpoint returned {resp.status_code}: {resp.text[:200]}"
        except httpx.HTTPError as exc:
            return False, f"Failed to reach model endpoint at {models_url}: {exc}"

    def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> dict:
        payload = {
            "model": self.settings.model_name,
            "temperature": self.settings.model_temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        url = f"{self.base_url}/chat/completions"
        try:
            response = self.client.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            if self.settings.is_remote_endpoint():
                message = (
                    "Could not query online model API endpoint. "
                    f"Base URL: {self.settings.openai_base_url}, Model: {self.settings.model_name}. "
                    "Check API key, endpoint compatibility, and model access permissions. "
                    f"Underlying error: {exc}"
                )
            else:
                message = (
                    "Could not query local model endpoint. "
                    f"Base URL: {self.settings.openai_base_url}, Model: {self.settings.model_name}. "
                    "If using Ollama, run `ollama serve` and `ollama pull <model>`, then retry. "
                    f"Underlying error: {exc}"
                )
            raise ModelUnavailableError(message) from exc

        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise ModelUnavailableError(f"Unexpected model response shape: {data}") from exc

        try:
            return try_parse_json(text)
        except Exception as exc:  # noqa: BLE001
            repaired = self._repair_json_text(text=text, max_tokens=max_tokens)
            if repaired is not None:
                try:
                    return try_parse_json(repaired)
                except Exception:  # noqa: BLE001
                    pass
            raise ModelUnavailableError(
                "Model response was not valid JSON. "
                "Adjust prompts/model or lower temperature. "
                "If using DeepSeek reasoning models, prefer deepseek-chat or reduce temperature to 0.1. "
                f"Raw snippet: {text[:300]}"
            ) from exc

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }

    def _repair_json_text(self, text: str, max_tokens: int) -> str | None:
        repair_payload = {
            "model": self.settings.model_name,
            "temperature": 0.0,
            "max_tokens": max(300, min(900, max_tokens)),
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You repair malformed JSON. "
                        "Return ONLY one valid JSON object. No markdown, no commentary."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Fix the following model output into a valid JSON object without changing field intent:\\n\\n"
                        f"{text[:4000]}"
                    ),
                },
            ],
        }
        url = f"{self.base_url}/chat/completions"
        try:
            resp = self.client.post(url, headers=self._headers(), json=repair_payload)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception:  # noqa: BLE001
            return None


class AnthropicProvider:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self.base_url = settings.anthropic_base_url.rstrip("/")
        self.client = httpx.Client(timeout=settings.model_timeout_seconds)

    def check_health(self) -> tuple[bool, str]:
        models_url = f"{self.base_url}/v1/models"
        try:
            resp = self.client.get(models_url, headers=self._headers())
            if resp.status_code == 200:
                return True, f"Anthropic endpoint reachable: {models_url}"
            return False, f"Anthropic endpoint returned {resp.status_code}: {resp.text[:200]}"
        except httpx.HTTPError as exc:
            return False, f"Failed to reach Anthropic endpoint at {models_url}: {exc}"

    def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> dict:
        payload = {
            "model": self.settings.model_name,
            "temperature": self.settings.model_temperature,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ],
        }
        url = f"{self.base_url}/v1/messages"
        try:
            response = self.client.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                "Could not query Anthropic API endpoint. "
                f"Base URL: {self.settings.anthropic_base_url}, Model: {self.settings.model_name}. "
                f"Underlying error: {exc}"
            ) from exc

        data = response.json()
        try:
            content_items = data.get("content", [])
            text = "\n".join(item.get("text", "") for item in content_items if item.get("type") == "text")
            if not text.strip():
                raise KeyError("No text content")
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(f"Unexpected Anthropic response shape: {data}") from exc

        try:
            return try_parse_json(text)
        except Exception as exc:  # noqa: BLE001
            repaired = self._repair_json_text(text=text, max_tokens=max_tokens)
            if repaired is not None:
                try:
                    return try_parse_json(repaired)
                except Exception:  # noqa: BLE001
                    pass
            raise ModelUnavailableError(
                "Anthropic response was not valid JSON. "
                "Adjust prompts/model or lower temperature. "
                f"Raw snippet: {text[:300]}"
            ) from exc

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def _repair_json_text(self, text: str, max_tokens: int) -> str | None:
        payload = {
            "model": self.settings.model_name,
            "temperature": 0.0,
            "max_tokens": max(300, min(900, max_tokens)),
            "system": "You repair malformed JSON and output only one valid JSON object.",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Fix this into one valid JSON object without changing field intent:\\n\\n"
                        f"{text[:4000]}"
                    ),
                }
            ],
        }
        url = f"{self.base_url}/v1/messages"
        try:
            response = self.client.post(url, headers=self._headers(), json=payload)
            response.raise_for_status()
            data = response.json()
            parts = data.get("content", [])
            repaired = "\n".join(item.get("text", "") for item in parts if item.get("type") == "text")
            return repaired if repaired.strip() else None
        except Exception:  # noqa: BLE001
            return None


class GeminiProvider:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self.base_url = settings.google_base_url.rstrip("/")
        self.client = httpx.Client(timeout=settings.model_timeout_seconds)

    def check_health(self) -> tuple[bool, str]:
        url = f"{self.base_url}/v1beta/models?key={self.settings.google_api_key}"
        try:
            resp = self.client.get(url)
            if resp.status_code == 200:
                return True, f"Gemini endpoint reachable: {self.base_url}/v1beta/models"
            return False, f"Gemini endpoint returned {resp.status_code}: {resp.text[:200]}"
        except httpx.HTTPError as exc:
            return False, f"Failed to reach Gemini endpoint at {self.base_url}: {exc}"

    def generate_json(self, system_prompt: str, user_prompt: str, max_tokens: int = 800) -> dict:
        model = self.settings.model_name
        url = f"{self.base_url}/v1beta/models/{model}:generateContent?key={self.settings.google_api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_prompt}],
                }
            ],
            "generationConfig": {
                "temperature": self.settings.model_temperature,
                "maxOutputTokens": max_tokens,
            },
        }

        try:
            response = self.client.post(url, json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise ModelUnavailableError(
                "Could not query Gemini API endpoint. "
                f"Base URL: {self.settings.google_base_url}, Model: {self.settings.model_name}. "
                f"Underlying error: {exc}"
            ) from exc

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "\n".join(p.get("text", "") for p in parts if "text" in p)
            if not text.strip():
                raise KeyError("No text in Gemini response")
        except Exception as exc:  # noqa: BLE001
            raise ModelUnavailableError(f"Unexpected Gemini response shape: {data}") from exc

        try:
            return try_parse_json(text)
        except Exception as exc:  # noqa: BLE001
            repaired = self._repair_json_text(text=text, max_tokens=max_tokens)
            if repaired is not None:
                try:
                    return try_parse_json(repaired)
                except Exception:  # noqa: BLE001
                    pass
            raise ModelUnavailableError(
                "Gemini response was not valid JSON. "
                "Adjust prompts/model or lower temperature. "
                f"Raw snippet: {text[:300]}"
            ) from exc

    def _repair_json_text(self, text: str, max_tokens: int) -> str | None:
        model = self.settings.model_name
        url = f"{self.base_url}/v1beta/models/{model}:generateContent?key={self.settings.google_api_key}"
        payload = {
            "systemInstruction": {
                "parts": [{"text": "Repair malformed JSON and return one valid JSON object only."}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": (
                                "Fix this into one valid JSON object without changing field intent:\\n\\n"
                                f"{text[:4000]}"
                            )
                        }
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.0, "maxOutputTokens": max(300, min(900, max_tokens))},
        }
        try:
            response = self.client.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            parts = data["candidates"][0]["content"]["parts"]
            repaired = "\n".join(p.get("text", "") for p in parts if "text" in p)
            return repaired if repaired.strip() else None
        except Exception:  # noqa: BLE001
            return None


def build_provider(settings: RuntimeSettings) -> LLMProvider:
    provider = settings.normalized_provider()

    if provider in OPENAI_COMPATIBLE_PROVIDERS:
        return OpenAICompatibleProvider(settings)
    if provider == "anthropic":
        return AnthropicProvider(settings)
    if provider == "gemini":
        return GeminiProvider(settings)

    supported = ", ".join(sorted(OPENAI_COMPATIBLE_PROVIDERS | {"anthropic", "gemini"}))
    raise ModelUnavailableError(f"Unsupported MODEL_PROVIDER={provider}. Supported: {supported}")
