"""Audited Semantica LiteLLM adapter that retains provider response identity.

Semantica 0.6.6 deliberately returns only the parsed JSON from ``generate_structured``. That is
convenient for applications, but insufficient for a sealed model-facing cohort: two calls using
the same requested alias may resolve to different provider models. This subclass keeps the exact
response identity while retaining Semantica's LiteLLM provider abstraction.
"""
from __future__ import annotations

import json
import time
from typing import Any

from litellm import completion
from semantica.llms import LiteLLM


def _read(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump(mode="json"))
    return str(value)


class AuditedLiteLLM(LiteLLM):
    """Semantica LiteLLM with append-only metadata for each structured call."""

    def __init__(self, model: str, api_key: str | None = None, **kwargs: Any) -> None:
        self.max_transport_attempts = int(kwargs.pop("max_transport_attempts", 3))
        self.transport_retry_delay_s = float(kwargs.pop("transport_retry_delay_s", 1.0))
        if self.max_transport_attempts < 1:
            raise ValueError("max_transport_attempts must be >= 1")
        super().__init__(model=model, api_key=api_key, **kwargs)
        self.call_records: list[dict[str, Any]] = []

    @staticmethod
    def _transient(exc: Exception) -> bool:
        return type(exc).__name__ in {
            "APIConnectionError", "APIError", "RateLimitError", "ServiceUnavailableError",
            "Timeout", "TimeoutError", "InternalServerError",
        }

    def generate_structured(self, prompt: str, **kwargs: Any) -> dict[str, Any]:
        options = {**self.config, **kwargs}
        transport_failures: list[dict[str, Any]] = []
        response: Any = None
        parsed: dict[str, Any] | None = None
        for transport_attempt in range(1, self.max_transport_attempts + 1):
            try:
                response = completion(
                    model=self.model,
                    messages=[{"role": "user",
                               "content": f"{prompt}\n\nReturn the response as valid JSON only."}],
                    api_key=self.api_key,
                    **options,
                )
            except Exception as exc:
                transient = self._transient(exc)
                transport_failures.append({
                    "attempt": transport_attempt, "error_type": type(exc).__name__,
                    "error": str(exc)[:500], "transient": transient,
                })
                if not transient or transport_attempt == self.max_transport_attempts:
                    raise
                time.sleep(self.transport_retry_delay_s * transport_attempt)
                continue

            choices = _read(response, "choices") or []
            message = _read(choices[0], "message") or {} if choices else {}
            content = _read(message, "content")
            structured_error: str | None = None
            if isinstance(content, dict):
                parsed = content
            elif isinstance(content, str):
                try:
                    candidate = json.loads(content)
                except json.JSONDecodeError as exc:
                    structured_error = f"invalid JSON: {exc.msg} at char {exc.pos}"
                else:
                    if isinstance(candidate, dict):
                        parsed = candidate
                    else:
                        structured_error = "JSON root was not an object"
            elif not choices:
                structured_error = "provider returned no choices"
            else:
                structured_error = "provider returned no JSON content"

            if parsed is not None:
                break
            choice = choices[0] if choices else {}
            transport_failures.append({
                "attempt": transport_attempt,
                "error_type": "StructuredOutputError",
                "error": structured_error,
                "transient": True,
                "finish_reason": _read(choice, "finish_reason"),
                "response_id": _read(response, "id"),
                "resolved_model": _read(response, "model"),
                "content_chars": len(content) if isinstance(content, str) else None,
            })
            if transport_attempt == self.max_transport_attempts:
                raise ValueError(
                    "structured reconstruction failed after bounded provider retries")
            time.sleep(self.transport_retry_delay_s * transport_attempt)

        if parsed is None:  # defensive: every loop exit above either sets parsed or raises
            raise ValueError("structured reconstruction returned no accepted JSON object")

        hidden = _read(response, "_hidden_params") or {}
        provider = (_read(response, "provider") or _read(hidden, "custom_llm_provider")
                    or _read(hidden, "llm_provider"))
        requested_provider = self.model.split("/", 1)[0] if "/" in self.model else None
        record = {
            "call_index": len(self.call_records) + 1,
            "requested_model": self.model,
            "requested_provider": requested_provider,
            "resolved_model": _read(response, "model"),
            "response_provider": provider,
            "response_id": _read(response, "id"),
            "created": _read(response, "created"),
            "usage": _plain(_read(response, "usage")),
            "transport_attempts": transport_attempt,
            "transport_failures": transport_failures,
        }
        record["identity_status"] = (
            "RETURNED_BY_PROVIDER" if record["resolved_model"] and record["response_id"]
            else "PARTIAL_PROVIDER_RESPONSE")
        self.call_records.append(record)
        return parsed
