"""Model-facing reconstruction retains the actual provider response identity."""
from __future__ import annotations

import json

import pytest

reconstruction_llm = pytest.importorskip("acr.mvp.reconstruction_llm")


class _Message:
    content = json.dumps({"episodes": []})


class _Choice:
    message = _Message()
    finish_reason = "stop"


class _Response:
    id = "gen-123"
    model = "openai/gpt-5.6-terra-2026-08-20"
    provider = None
    created = 1787000000
    choices = [_Choice()]
    usage = {"prompt_tokens": 10, "completion_tokens": 3}
    _hidden_params = {"custom_llm_provider": "openrouter"}


def test_audited_semantica_litellm_keeps_resolved_model_provider_and_response(monkeypatch):
    monkeypatch.setattr(reconstruction_llm, "completion", lambda **_kwargs: _Response())
    llm = reconstruction_llm.AuditedLiteLLM(
        model="openrouter/openai/gpt-5.6-terra", api_key="not-a-real-key")

    assert llm.generate_structured("fixed cycles") == {"episodes": []}
    assert llm.call_records == [{
        "call_index": 1,
        "requested_model": "openrouter/openai/gpt-5.6-terra",
        "requested_provider": "openrouter",
        "resolved_model": "openai/gpt-5.6-terra-2026-08-20",
        "response_provider": "openrouter",
        "response_id": "gen-123",
        "created": 1787000000,
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        "transport_attempts": 1,
        "transport_failures": [],
        "identity_status": "RETURNED_BY_PROVIDER",
    }]


def test_transient_openrouter_transport_failure_is_retried_and_audited(monkeypatch):
    class APIError(Exception):
        pass

    outcomes = [APIError("incomplete chunked read"), _Response()]

    def complete(**_kwargs):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(reconstruction_llm, "completion", complete)
    monkeypatch.setattr(reconstruction_llm.time, "sleep", lambda _seconds: None)
    llm = reconstruction_llm.AuditedLiteLLM(
        model="openrouter/openai/gpt-5.6-luna", api_key="not-a-real-key")

    assert llm.generate_structured("fixed cycles") == {"episodes": []}
    assert llm.call_records[0]["transport_attempts"] == 2
    assert llm.call_records[0]["transport_failures"] == [{
        "attempt": 1, "error_type": "APIError", "error": "incomplete chunked read",
        "transient": True,
    }]


def test_truncated_structured_response_is_retried_without_recording_its_content(monkeypatch):
    class TruncatedMessage:
        content = '{"episodes": ['

    class TruncatedChoice:
        message = TruncatedMessage()
        finish_reason = "error"

    class TruncatedResponse:
        id = "gen-truncated"
        model = "openai/gpt-5.6-luna"
        choices = [TruncatedChoice()]

    outcomes = [TruncatedResponse(), _Response()]
    monkeypatch.setattr(reconstruction_llm, "completion", lambda **_kwargs: outcomes.pop(0))
    monkeypatch.setattr(reconstruction_llm.time, "sleep", lambda _seconds: None)
    llm = reconstruction_llm.AuditedLiteLLM(
        model="openrouter/openai/gpt-5.6-luna", api_key="not-a-real-key")

    assert llm.generate_structured("fixed cycles") == {"episodes": []}
    failure = llm.call_records[0]["transport_failures"][0]
    assert failure == {
        "attempt": 1,
        "error_type": "StructuredOutputError",
        "error": "invalid JSON: Expecting value at char 14",
        "transient": True,
        "finish_reason": "error",
        "response_id": "gen-truncated",
        "resolved_model": "openai/gpt-5.6-luna",
        "content_chars": 14,
    }
    assert llm.call_records[0]["transport_attempts"] == 2
    assert "content" not in failure


def test_nontransient_provider_failure_is_not_retried(monkeypatch):
    class AuthenticationError(Exception):
        pass

    calls = 0

    def fail(**_kwargs):
        nonlocal calls
        calls += 1
        raise AuthenticationError("bad key")

    monkeypatch.setattr(reconstruction_llm, "completion", fail)
    llm = reconstruction_llm.AuditedLiteLLM(
        model="openrouter/openai/gpt-5.6-luna", api_key="not-a-real-key")

    with pytest.raises(AuthenticationError, match="bad key"):
        llm.generate_structured("fixed cycles")
    assert calls == 1 and llm.call_records == []
