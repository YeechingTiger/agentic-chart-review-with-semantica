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
    # Hybrid reasoning models spend most of a completion on thinking before they emit
    # anything. Measured here: qwen3.6:35b used 1394 completion tokens to produce a
    # 48-character JSON answer. At the old default of 1536 the thinking consumed the budget
    # and `content` came back empty — which the planner read as "no plan" and quietly
    # replaced with a one-line stub, in every run, undetected. Budget for the thinking.
    max_tokens: int = 4096
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
    used_reasoning_channel: bool = False

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
        self.reasoning_fallbacks = 0   # completions that arrived only in the thinking channel
        self.empty_completions = 0     # completions with no text AND no tool call

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

        # Hybrid reasoning models (qwen3.x among them) can spend the whole completion budget
        # on thinking and return `content: ""` with the reasoning in a side channel. That
        # looks exactly like "the model had nothing to say", and a caller that falls back on
        # empty output degrades silently — which is how this repo's planner ran on a one-line
        # fallback plan for every run without anything going red. Recover the side channel,
        # and when there is genuinely nothing, say so loudly rather than returning "".
        content = msg.content or ""
        reasoning = (getattr(msg, "reasoning_content", None)
                     or getattr(msg, "reasoning", None) or "")
        if not content.strip() and reasoning.strip():
            content = reasoning
            self.reasoning_fallbacks += 1
        if not content.strip() and not calls:
            self.empty_completions += 1

        return LLMResponse(
            content=content,
            tool_calls=calls,
            raw=resp,
            prompt_tokens=pt,
            completion_tokens=ct,
            latency_s=dt,
            used_reasoning_channel=bool(reasoning.strip() and not (msg.content or "").strip()),
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
            "reasoning_fallbacks": self.reasoning_fallbacks,
            "empty_completions": self.empty_completions,
        }


def extract_json(text: str, require: str | None = None) -> dict:
    """Best-effort JSON extraction from a model reply.

    `require` names a key the caller needs. When given, EVERY top-level {...} block is
    considered and the first one carrying that key wins; without it the first parseable
    block wins, which is the historical behaviour.

    Why this is not a cosmetic nicety: a reply may legitimately contain more than one JSON
    object, and then "first parseable" is simply the wrong one. Observed with
    gpt-5.6-luna, which leaks its tool-call channel into the text channel and emits a
    preamble object before the answer:

         to=record_evidence code
        {"search":"pathology OR biopsy ...","document_types":[...]}
        {"verdict":"CONTINUE","reason":"...","revised_plan":[]}

    The old code returned the {"search":...} object, so `.get("verdict")` was None and the
    reflect node recorded a degradation — while the correct verdict sat in the very next
    object, every time. That misreads a working supervisor as a broken one, which is the
    most expensive kind of wrong: it discredits runs that were actually fine.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.split("```")[1] if len(t.split("```")) > 1 else t
        t = t[4:].lstrip() if t.lower().startswith("json") else t
    t = t.strip()

    if require is None:
        try:
            v = json.loads(t)
            return v if isinstance(v, dict) else {"value": v}
        except json.JSONDecodeError:
            pass
    else:
        try:
            v = json.loads(t)
            if isinstance(v, dict) and require in v:
                return v
        except json.JSONDecodeError:
            pass

    first: dict | None = None
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
                    obj = json.loads(t[start : i + 1])
                except json.JSONDecodeError:
                    start = None
                    continue
                if isinstance(obj, dict):
                    if require is None:
                        return obj
                    if require in obj:
                        return obj
                    if first is None:
                        first = obj
                start = None

    if first is not None:
        return first
    return {"__unparsed__": text}
