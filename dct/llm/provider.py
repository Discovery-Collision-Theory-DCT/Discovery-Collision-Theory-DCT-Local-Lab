from __future__ import annotations

import json
from typing import Any, Callable, Protocol

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

    def set_debug_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        ...


class OpenAICompatibleProvider:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self.base_url = settings.openai_base_url.rstrip("/")
        self.client = httpx.Client(timeout=settings.model_timeout_seconds)
        self.debug_callback: Callable[[dict[str, Any]], None] | None = None

    def set_debug_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self.debug_callback = callback

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
        try:
            data = self._post_chat_completion(payload=payload, prefer_json_mode=True)
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

        try:
            content_text, reasoning_text = self._extract_chat_texts(data)
        except (KeyError, IndexError, TypeError) as exc:
            raise ModelUnavailableError(f"Unexpected model response shape: {data}") from exc
        primary_text = content_text.strip()
        if primary_text:
            self._emit_output(system_prompt=system_prompt, text=primary_text, phase="primary")
        if reasoning_text.strip():
            self._emit_output(system_prompt=system_prompt, text=reasoning_text, phase="reasoning")

        parse_candidates = []
        if primary_text:
            parse_candidates.append(primary_text)
        if reasoning_text.strip() and reasoning_text.strip() != primary_text:
            parse_candidates.append(reasoning_text.strip())

        last_parse_exc: Exception | None = None
        for candidate in parse_candidates:
            try:
                return try_parse_json(candidate)
            except Exception as exc:  # noqa: BLE001
                last_parse_exc = exc

        repair_source = parse_candidates[0] if parse_candidates else json.dumps(data)[:4000]
        repaired = self._repair_json_text(text=repair_source, max_tokens=max_tokens)
        if repaired is not None:
            self._emit_output(system_prompt=system_prompt, text=repaired, phase="repair")
            try:
                return try_parse_json(repaired)
            except Exception as exc:  # noqa: BLE001
                last_parse_exc = exc

        snippet = (repair_source or "")[:300]
        raise ModelUnavailableError(
            "Model response was not valid JSON. "
            "Adjust prompts/model or lower temperature. "
            "If using DeepSeek reasoning models, prefer deepseek-chat or reduce temperature to 0.1. "
            f"Raw snippet: {snippet}"
        ) from last_parse_exc

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
        try:
            data = self._post_chat_completion(payload=repair_payload, prefer_json_mode=True)
            repaired, reasoning = self._extract_chat_texts(data)
            if not repaired.strip():
                repaired = reasoning
            return repaired if repaired.strip() else None
        except Exception:  # noqa: BLE001
            return None

    def _post_chat_completion(self, payload: dict[str, Any], prefer_json_mode: bool) -> dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        attempts: list[dict[str, Any]] = []
        if prefer_json_mode:
            with_json_mode = dict(payload)
            with_json_mode["response_format"] = {"type": "json_object"}
            attempts.append(with_json_mode)
        attempts.append(payload)

        last_error: Exception | None = None
        for idx, attempt in enumerate(attempts):
            try:
                resp = self.client.post(url, headers=self._headers(), json=attempt)
                resp.raise_for_status()
                return resp.json()
            except httpx.HTTPStatusError as exc:
                last_error = exc
                if idx == 0 and prefer_json_mode and self._is_response_format_unsupported(exc.response):
                    continue
                raise
            except httpx.HTTPError as exc:
                last_error = exc
                raise

        if last_error is not None:
            raise last_error
        raise httpx.HTTPError("Unknown error while posting chat completion request")

    @staticmethod
    def _is_response_format_unsupported(response: httpx.Response | None) -> bool:
        if response is None:
            return False
        if response.status_code not in {400, 404, 415, 422}:
            return False
        body = response.text.lower()
        markers = [
            "response_format",
            "json_schema",
            "unsupported",
            "unknown field",
            "extra fields not permitted",
            "not supported",
        ]
        return any(marker in body for marker in markers)

    @staticmethod
    def _extract_chat_texts(data: dict[str, Any]) -> tuple[str, str]:
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise KeyError("Missing choices")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise KeyError("Missing message")

        content_text = OpenAICompatibleProvider._extract_text_from_message_content(message.get("content"))
        reasoning_text = OpenAICompatibleProvider._extract_reasoning_text(message)
        return content_text, reasoning_text

    @staticmethod
    def _extract_text_from_message_content(content: Any) -> str:
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                    continue
                if not isinstance(part, dict):
                    continue
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
                    continue
                if part.get("type") == "text" and isinstance(part.get("content"), str):
                    parts.append(part["content"])
            merged = "\n".join(p for p in parts if p).strip()
            if merged:
                return merged

        return ""

    @staticmethod
    def _extract_reasoning_text(message: dict[str, Any]) -> str:
        candidates = [
            message.get("reasoning_content"),
            message.get("reasoning"),
            message.get("reasoning_text"),
        ]

        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate
            if isinstance(candidate, list):
                parts: list[str] = []
                for item in candidate:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                merged = "\n".join(p for p in parts if p).strip()
                if merged:
                    return merged
        return ""

    def _emit_output(self, system_prompt: str, text: str, phase: str) -> None:
        if self.debug_callback is None:
            return
        try:
            agent_hint = (system_prompt.strip().splitlines()[0] if system_prompt else "unknown")[:120]
            self.debug_callback(
                {
                    "type": "model_output",
                    "provider": self.settings.normalized_provider(),
                    "model": self.settings.model_name,
                    "phase": phase,
                    "agent_hint": agent_hint,
                    "text": text,
                }
            )
        except Exception:  # noqa: BLE001
            return


class AnthropicProvider:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self.base_url = settings.anthropic_base_url.rstrip("/")
        self.client = httpx.Client(timeout=settings.model_timeout_seconds)
        self.debug_callback: Callable[[dict[str, Any]], None] | None = None

    def set_debug_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self.debug_callback = callback

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
        self._emit_output(system_prompt=system_prompt, text=text, phase="primary")

        try:
            return try_parse_json(text)
        except Exception as exc:  # noqa: BLE001
            repaired = self._repair_json_text(text=text, max_tokens=max_tokens)
            if repaired is not None:
                self._emit_output(system_prompt=system_prompt, text=repaired, phase="repair")
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

    def _emit_output(self, system_prompt: str, text: str, phase: str) -> None:
        if self.debug_callback is None:
            return
        try:
            agent_hint = (system_prompt.strip().splitlines()[0] if system_prompt else "unknown")[:120]
            self.debug_callback(
                {
                    "type": "model_output",
                    "provider": self.settings.normalized_provider(),
                    "model": self.settings.model_name,
                    "phase": phase,
                    "agent_hint": agent_hint,
                    "text": text,
                }
            )
        except Exception:  # noqa: BLE001
            return


class GeminiProvider:
    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self.base_url = settings.google_base_url.rstrip("/")
        self.client = httpx.Client(timeout=settings.model_timeout_seconds)
        self.debug_callback: Callable[[dict[str, Any]], None] | None = None

    def set_debug_callback(self, callback: Callable[[dict[str, Any]], None] | None) -> None:
        self.debug_callback = callback

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
        self._emit_output(system_prompt=system_prompt, text=text, phase="primary")

        try:
            return try_parse_json(text)
        except Exception as exc:  # noqa: BLE001
            repaired = self._repair_json_text(text=text, max_tokens=max_tokens)
            if repaired is not None:
                self._emit_output(system_prompt=system_prompt, text=repaired, phase="repair")
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

    def _emit_output(self, system_prompt: str, text: str, phase: str) -> None:
        if self.debug_callback is None:
            return
        try:
            agent_hint = (system_prompt.strip().splitlines()[0] if system_prompt else "unknown")[:120]
            self.debug_callback(
                {
                    "type": "model_output",
                    "provider": self.settings.normalized_provider(),
                    "model": self.settings.model_name,
                    "phase": phase,
                    "agent_hint": agent_hint,
                    "text": text,
                }
            )
        except Exception:  # noqa: BLE001
            return


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
