"""LLM access via LiteLLM — one interface, any provider.

LiteLLM normalises 100+ providers onto the OpenAI chat-completions shape, so the agent
never learns a vendor API. Model strings are passed straight through:

    ollama_chat/qwen3.6:35b          local Ollama (default here)
    ollama/qwen3.6:35b               Ollama /api/generate path
    hosted_vllm/Qwen/Qwen3.6-35B     vLLM OpenAI-compatible server
    openai/gpt-4.1                   OpenAI
    anthropic/claude-sonnet-4-5      Anthropic
    bedrock/...  azure/...  together_ai/...  etc.

Point at a self-hosted OpenAI-compatible server with `api_base`.

Only two things are required of a backend: chat with tools, and report token usage.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import litellm

# Local models frequently lack a public cost table; don't let that raise.
litellm.suppress_debug_info = True
litellm.drop_params = True  # silently drop params a given provider doesn't support


@dataclass
class LLMConfig:
    model: str = "ollama_chat/qwen3.6:35b"
    api_base: str | None = None          # e.g. http://localhost:11434 or a vLLM endpoint
    api_key: str | None = None
    temperature: float = 0.0
    max_tokens: int = 1536
    timeout: int = 600
    num_ctx: int | None = 32768          # Ollama-specific; ignored elsewhere
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides) -> "LLMConfig":
        cfg = cls(
            model=os.getenv("ACR_MODEL", cls.model),
            api_base=os.getenv("ACR_API_BASE") or None,
            api_key=os.getenv("ACR_API_KEY") or None,
            temperature=float(os.getenv("ACR_TEMPERATURE", "0.0")),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[dict]               # [{id, name, arguments(dict)}]
    raw: Any = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMClient:
    """Thin, provider-agnostic chat client with tool-calling."""

    def __init__(self, cfg: LLMConfig | None = None):
        self.cfg = cfg or LLMConfig()
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _kwargs(self) -> dict:
        kw: dict[str, Any] = {
            "model": self.cfg.model,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "timeout": self.cfg.timeout,
        }
        if self.cfg.api_base:
            kw["api_base"] = self.cfg.api_base
        if self.cfg.api_key:
            kw["api_key"] = self.cfg.api_key
        if self.cfg.num_ctx and self.cfg.model.startswith("ollama"):
            kw["num_ctx"] = self.cfg.num_ctx
        kw.update(self.cfg.extra)
        return kw

    def chat(self, messages: list[dict], tools: list[dict] | None = None) -> LLMResponse:
        kw = self._kwargs()
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = "auto"
        t0 = time.time()
        resp = litellm.completion(messages=messages, **kw)
        dt = time.time() - t0
        self.calls += 1

        msg = resp.choices[0].message
        calls: list[dict] = []
        for tc in (getattr(msg, "tool_calls", None) or []):
            fn = tc.function
            args = fn.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args or "{}")
                except json.JSONDecodeError:
                    args = {"__unparseable_arguments__": fn.arguments}
            calls.append({"id": getattr(tc, "id", f"call_{len(calls)}"), "name": fn.name, "arguments": args})

        usage = getattr(resp, "usage", None)
        pt = int(getattr(usage, "prompt_tokens", 0) or 0)
        ct = int(getattr(usage, "completion_tokens", 0) or 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct

        return LLMResponse(
            content=(msg.content or ""),
            tool_calls=calls,
            raw=resp,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_s=dt,
        )

    def json_chat(self, messages: list[dict], schema_hint: str = "") -> dict:
        """Ask for a JSON object. Tolerates fenced output and prose padding."""
        msgs = list(messages)
        if schema_hint:
            msgs.append({"role": "system", "content": f"Reply with JSON only: {schema_hint}"})
        txt = self.chat(msgs).content.strip()
        return extract_json(txt)

    def usage(self) -> dict:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


def extract_json(text: str) -> dict:
    """Best-effort JSON extraction from a model reply."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if len(t.split("```")) > 1 else t
        t = t[4:].lstrip() if t.lower().startswith("json") else t
    t = t.strip()
    try:
        v = json.loads(t)
        return v if isinstance(v, dict) else {"value": v}
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, ch in enumerate(t):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(t[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    return {"__unparsed__": text}
