"""Codex App Server is the MVP's agent harness.

This module is intentionally a thin adapter over OpenAI's open-source Python SDK.  It starts
the *explicit* Codex binary selected by the runner (0.150.0 or newer), creates an ephemeral
thread over the App Server JSON-RPC API, and archives the typed notification stream.  No agent
loop lives in ACR.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(slots=True)
class HarnessResult:
    returncode: int
    thread_id: str | None = None
    turn_id: str | None = None
    status: str = "failed"
    final_response: str | None = None
    error: str | None = None
    stderr: str = ""


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=True, mode="json")
    if hasattr(value, "root"):
        return _dump(value.root)
    if isinstance(value, dict):
        return {str(k): _dump(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(v) for v in value]
    if hasattr(value, "value"):
        return value.value
    return value


_ITEM_TYPES = {
    "agentMessage": "agent_message",
    "mcpToolCall": "mcp_tool_call",
    "userMessage": "user_message",
}


def normalise_notification(event: Any) -> dict[str, Any]:
    """Convert App Server notifications to the stable Layer-2 shape used by observe.py."""
    payload = _dump(event.payload)
    if not isinstance(payload, dict):
        payload = {"payload": payload}
    item = payload.get("item")
    if isinstance(item, dict):
        item = dict(item)
        item["type"] = _ITEM_TYPES.get(str(item.get("type")), item.get("type"))
        if item.get("type") == "reasoning" and not item.get("text"):
            words = item.get("summary") or item.get("content") or []
            item["text"] = "\n".join(str(x) for x in words if str(x).strip())
        payload["item"] = item
    return {"type": event.method.replace("/", "."), **payload}


class AppServerCodexHarness:
    """Run one Codex turn through the official App Server SDK."""

    def __init__(self, codex_bin: str, disabled_features: tuple[str, ...]) -> None:
        self.codex_bin = codex_bin
        self.disabled_features = disabled_features

    def run(
        self,
        prompt: str,
        *,
        model: str,
        workdir: Path,
        env: dict[str, str],
        layer2_path: Path,
        timeout_s: int,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> HarnessResult:
        return asyncio.run(self._run(
            prompt, model=model, workdir=workdir, env=env,
            layer2_path=layer2_path, timeout_s=timeout_s, on_event=on_event))

    async def _run(
        self,
        prompt: str,
        *,
        model: str,
        workdir: Path,
        env: dict[str, str],
        layer2_path: Path,
        timeout_s: int,
        on_event: Callable[[dict[str, Any]], None] | None,
    ) -> HarnessResult:
        from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox

        config = CodexConfig(
            codex_bin=self.codex_bin,
            cwd=str(workdir),
            env=env,
            config_overrides=tuple(f"features.{name}=false"
                                   for name in self.disabled_features),
            client_name="acr_chart_review",
            client_title="Agentic Chart Review with Semantica",
        )
        result = HarnessResult(returncode=1)
        client: Any = None
        final_messages: list[str] = []

        try:
            async with AsyncCodex(config) as client:
                thread = await client.thread_start(
                    approval_mode=ApprovalMode.deny_all,
                    cwd=str(workdir), ephemeral=True, model=model,
                    model_provider="acr", sandbox=Sandbox.read_only,
                    service_name="acr-chart-review",
                )
                result.thread_id = thread.id
                first = {"type": "thread.started", "thread_id": thread.id,
                         "source": "codex_app_server"}
                with layer2_path.open("w", encoding="utf-8") as stream_file:
                    stream_file.write(json.dumps(first, ensure_ascii=False) + "\n")
                    stream_file.flush()
                    if on_event:
                        on_event(first)

                    turn = await thread.turn(
                        prompt, approval_mode=ApprovalMode.deny_all,
                        model=model, sandbox=Sandbox.read_only)
                    result.turn_id = turn.id
                    stream = turn.stream()
                    try:
                        async with asyncio.timeout(timeout_s):
                            async for notification in stream:
                                event = normalise_notification(notification)
                                stream_file.write(json.dumps(event, ensure_ascii=False) + "\n")
                                stream_file.flush()
                                if on_event:
                                    on_event(event)
                                item = event.get("item") or {}
                                if (event.get("type") == "item.completed"
                                        and item.get("type") == "agent_message"
                                        and item.get("text")):
                                    final_messages.append(str(item["text"]))
                                if event.get("type") == "turn.completed":
                                    turn_data = event.get("turn") or {}
                                    result.status = str(turn_data.get("status") or "completed")
                                    # Codex 0.150.1 can keep the notification stream open after
                                    # emitting its terminal event. The turn is already sealed at
                                    # this boundary; waiting for transport EOF turns a successful
                                    # review into a timeout. Close the subscription explicitly.
                                    break
                    except TimeoutError:
                        await turn.interrupt()
                        result.status = "timed_out"
                        result.error = f"Codex turn exceeded {timeout_s}s"
                    finally:
                        await stream.aclose()

                result.final_response = final_messages[-1] if final_messages else None
                result.returncode = 0 if result.status == "completed" else 1
        except Exception as exc:  # operational failure becomes an honest NO_ANSWER run
            result.error = f"{type(exc).__name__}: {exc}"
        finally:
            try:
                sync = getattr(getattr(client, "_client", None), "_sync", None)
                if sync is not None:
                    result.stderr = sync._stderr_tail(200)  # SDK exposes no public stderr API
            except Exception:
                pass
        return result
