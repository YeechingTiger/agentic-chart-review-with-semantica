"""A deterministic Responses-API model for driving the codex harness without a network.

WHY THIS EXISTS. This environment's egress proxy blocks every real model endpoint, and the
v0.3.1 build brief independently requires that core tests use a fake model with recorded
fixtures rather than the network. Both problems have the same answer: a local HTTP server that
speaks just enough of the Responses API's SSE dialect for codex to run a full session against
it, replaying a scripted trajectory of tool calls.

THE WIRE FORMAT IS TAKEN FROM CODEX'S OWN PARSER, not guessed:
`codex-rs/codex-api/src/sse/responses.rs` (checked at codex-cli 0.149.1) accepts frames of
`event: <kind>\\ndata: <json>\\n\\n` and needs only three kinds for a turn —

    response.created            {"type":"response.created","response":{}}
    response.output_item.done   {"type":"response.output_item.done","item":<ResponseItem>}
    response.completed          {"type":"response.completed","response":{"id":...}}

where a tool call is the ResponseItem
    {"type":"function_call","name":"search","namespace":"mcp__chart","arguments":"{...}","call_id":"c1"}
and a final message is
    {"type":"message","role":"assistant","content":[{"type":"output_text","text":"..."}]}.

MCP tools arrive in the request's tool list as a NAMESPACE entry (observed against 0.149.1;
matches `ToolName::namespaced("mcp__chart", tool)` in `core/src/tools/handlers/mcp.rs`):
    {"type":"namespace","name":"mcp__chart","tools":[{"type":"function","name":"search",...}]}
and the function_call item must carry that namespace in its own `namespace` field — a bare or
prefix-joined name is rejected by the router ("unsupported call"). Script steps name bare tools
("search"); this server resolves (name, namespace) against the offered list, checking flat
function entries first and then each namespace's members.

The script is a JSON list of steps, each either
    {"tool": "search", "args": {...}}          -> one function_call turn
or  {"message": "text"}                        -> a final assistant message turn.
Every request body is appended to requests.jsonl beside the script, so a failing handshake
is diagnosable from disk.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class _State:
    def __init__(self, script: list[dict[str, Any]], log_path: Path) -> None:
        self.script = script
        self.log_path = log_path
        self.cursor = 0
        self.lock = threading.Lock()

    def next_step(self) -> dict[str, Any]:
        with self.lock:
            if self.cursor < len(self.script):
                step = self.script[self.cursor]
                self.cursor += 1
                return step
            return {"message": "No further scripted steps; ending the session."}


def _resolve_tool(bare: str, offered: list[dict[str, Any]]) -> tuple[str, str | None]:
    """(name, namespace) as the function_call item must carry them."""
    for t in offered:
        name = t.get("name", "")
        if t.get("type") == "namespace":
            for member in t.get("tools", []):
                if isinstance(member, dict) and member.get("name") == bare:
                    return bare, name
        elif name == bare or name.endswith(f"__{bare}"):
            return name, None
    return bare, None  # let the harness report the unknown tool; the trace will show it


def _sse(kind: str, payload: dict[str, Any]) -> bytes:
    return f"event: {kind}\ndata: {json.dumps(payload)}\n\n".encode()


class _Handler(BaseHTTPRequestHandler):
    state: _State  # injected by serve()

    def log_message(self, fmt: str, *args: Any) -> None:  # stderr, never stdout
        sys.stderr.write("fake_model: " + fmt % args + "\n")

    def do_POST(self) -> None:  # noqa: N802 - http.server API
        body = self.rfile.read(int(self.headers.get("Content-Length", "0") or 0))
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError:
            req = {"unparsed": body.decode(errors="replace")[:2000]}
        with self.state.lock:
            with self.state.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"path": self.path, "body": req}) + "\n")

        offered = [t for t in req.get("tools", []) if isinstance(t, dict)]
        step = self.state.next_step()
        n = self.state.cursor
        if "tool" in step:
            name, namespace = _resolve_tool(step["tool"], offered)
            item: dict[str, Any] = {
                "type": "function_call",
                "name": name,
                "arguments": json.dumps(step.get("args") or {}),
                "call_id": f"call_{n}",
            }
            if namespace is not None:
                item["namespace"] = namespace
        else:
            item = {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": step.get("message", "done")}],
            }

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(_sse("response.created", {"type": "response.created", "response": {}}))
        self.wfile.write(_sse("response.output_item.done",
                              {"type": "response.output_item.done", "item": item}))
        self.wfile.write(_sse("response.completed",
                              {"type": "response.completed", "response": {"id": f"resp_{n}"}}))
        self.wfile.flush()


def serve(script_path: Path, log_path: Path, port: int = 0) -> ThreadingHTTPServer:
    """Start the server on 127.0.0.1; returns it (serve_forever runs on a daemon thread)."""
    script = json.loads(script_path.read_text(encoding="utf-8"))
    handler = type("Bound", (_Handler,), {"state": _State(script, log_path)})
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd


def main() -> None:
    script = Path(os.environ["FAKE_MODEL_SCRIPT"])
    log = Path(os.environ.get("FAKE_MODEL_LOG", script.with_name("requests.jsonl")))
    port = int(os.environ.get("FAKE_MODEL_PORT", "0"))
    httpd = serve(script, log, port)
    print(f"fake_model listening on http://127.0.0.1:{httpd.server_address[1]}", flush=True)
    threading.Event().wait()  # foreground forever; the parent kills us


if __name__ == "__main__":
    main()
