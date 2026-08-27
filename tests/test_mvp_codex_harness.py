"""Codex App Server stream lifecycle at the harness boundary."""
from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace

from acr.mvp.codex_harness import AppServerCodexHarness


def test_terminal_turn_notification_closes_a_stream_that_never_sends_eof(
        tmp_path, monkeypatch):
    class Stream:
        def __init__(self):
            self.calls = 0
            self.closed = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            self.calls += 1
            if self.calls == 1:
                return SimpleNamespace(
                    method="turn/completed",
                    payload={"turn": {"status": "completed"}},
                )
            await asyncio.Event().wait()  # Codex 0.150.1 transport can remain subscribed.
            raise StopAsyncIteration

        async def aclose(self):
            self.closed = True

    stream = Stream()

    class Turn:
        id = "turn-1"

        def stream(self):
            return stream

        async def interrupt(self):
            raise AssertionError("a completed turn must not be interrupted")

    class Thread:
        id = "thread-1"

        async def turn(self, *_args, **_kwargs):
            return Turn()

    class Client:
        _client = SimpleNamespace(_sync=None)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def thread_start(self, **_kwargs):
            return Thread()

    fake_sdk = SimpleNamespace(
        ApprovalMode=SimpleNamespace(deny_all="deny_all"),
        Sandbox=SimpleNamespace(read_only="read_only"),
        CodexConfig=lambda **kwargs: kwargs,
        AsyncCodex=lambda _config: Client(),
    )
    monkeypatch.setitem(sys.modules, "openai_codex", fake_sdk)

    result = AppServerCodexHarness("codex", ()).run(
        "review", model="luna", workdir=tmp_path, env={},
        layer2_path=tmp_path / "layer2.jsonl", timeout_s=1,
    )

    assert result.status == "completed" and result.returncode == 0
    assert stream.calls == 1
    assert stream.closed is True
