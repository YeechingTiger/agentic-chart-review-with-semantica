"""One run, read back as a decision trace: think -> decide -> act -> observe -> ... -> verdict.

The spine is Layer 1 (the tool-boundary trace): every step that counts has a server-assigned
seq, so the order of decisions and actions is fact, not reconstruction. Layer 2 (the harness's
event stream) contributes the model's between-action thinking — reasoning summaries — which is
self-reported and clearly labeled as such. The two interleave without a shared clock: codex
emits each turn's reasoning items immediately before that turn's mcp_tool_call item, and the
k-th mcp_tool_call in Layer 2 IS the k-th tool_call in Layer 1 (both are server observations of
the same sequence), so thoughts attach to the call they preceded by position, not by timestamp.

Observation independence, inherited from the design doc: with Layer 2 deleted, the trace loses
only the "thought" steps — every decision, action, observation and verdict still reads in
order, because those live at the tool boundary. Nothing here scores anything; this is the
audit's reading view.

Step kinds:
    thought     (L2, self-reported)  a reasoning summary the model emitted
    decision    (L1) a note_decision call: facing / decision / because /
                     used / grounding / options — UNCLASSIFIED, by design
    action      (L1) search / read / list_documents, with objective and a result digest
    evidence    (L1) a recorded span, with the server-resolved quote
    submission  (L1) a submit_answer call
    verdict     (L1) the gate's answer to that submission
    remark      (L2, self-reported)  a final assistant message
    result      (L1) answer_accepted
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ACTIONS = ("search", "read", "list_documents")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _layer2_annotations(run_dir: Path) -> tuple[dict[int, list[str]], list[str], list[str]]:
    """thoughts before the k-th tool call, thoughts after the last one, final messages."""
    thoughts_before: dict[int, list[str]] = {}
    buffer: list[str] = []
    remarks: list[str] = []
    k = 0
    for e in _read_jsonl(run_dir / "layer2_codex.jsonl"):
        item = e.get("item") or {}
        if e.get("type") == "item.completed" and item.get("type") == "reasoning":
            if item.get("text"):
                buffer.append(item["text"])
        elif e.get("type") == "item.started" and item.get("type") == "mcp_tool_call":
            if buffer:
                thoughts_before[k] = buffer
                buffer = []
            k += 1
        elif e.get("type") == "item.completed" and item.get("type") == "agent_message":
            if item.get("text"):
                remarks.append(item["text"])
    return thoughts_before, buffer, remarks


def _digest(tool: str, result: dict[str, Any]) -> str:
    if "error" in result:
        return f"error: {result['error']}"
    if tool == "search":
        return f"{result.get('n', 0)} hit(s)"
    if tool == "list_documents":
        return f"{result.get('total', 0)} document(s), {len(result.get('types', []))} type(s)"
    if tool == "read":
        return (f"{result.get('note_id')} chars {result.get('offset', 0)}"
                f"+{result.get('returned_chars', 0)} of {result.get('total_chars', 0)}")
    return "ok"


def decision_trace(run_dir: Path) -> dict[str, Any]:
    """The ordered decision trace of one run. Works with Layer 2 absent (no thoughts)."""
    run_dir = Path(run_dir)
    events = _read_jsonl(run_dir / "trace.jsonl")
    meta = next((e for e in events if e.get("kind") == "run_meta"), {})
    thoughts_before, trailing, remarks = _layer2_annotations(run_dir)

    steps: list[dict[str, Any]] = []

    def add(kind: str, layer: int, seq: int | None = None, **fields: Any) -> None:
        channel = "server" if layer == 1 else "self_reported"
        steps.append({"kind": kind, "layer": layer, "channel": channel, "seq": seq, **fields})

    k = 0
    for e in events:
        if e.get("kind") == "answer_accepted":
            add("result", 1, e["seq"], status=e.get("status"), value=e.get("value"))
            continue
        if e.get("kind") != "tool_call":
            continue
        for text in thoughts_before.get(k, []):
            add("thought", 2, None, text=text)
        k += 1
        tool, args, result = e.get("tool"), e.get("args") or {}, e.get("result") or {}
        seq = e.get("seq")
        if tool == "note_decision":
            add("decision", 1, seq, facing=args.get("facing"),
                decision=args.get("decision"), because=args.get("because"),
                options=args.get("options"), context=result.get("context"),
                grounding=result.get("grounding") or args.get("grounding") or [],
                used=result.get("used") or [{"ref": r} for r in (args.get("used") or [])])
        elif tool in _ACTIONS:
            shown = {a: v for a, v in args.items() if a != "objective" and v is not None}
            add("action", 1, seq, tool=tool, objective=args.get("objective"),
                args=shown, observed=_digest(tool, result), context=result.get("context"))
        elif tool == "record_evidence":
            add("evidence", 1, seq, note_id=args.get("note_id"),
                span=[args.get("start"), args.get("end")], quote=result.get("quote"),
                supports=args.get("supports"), error=result.get("error"))
        elif tool == "submit_answer":
            add("submission", 1, seq, status=args.get("status"), value=args.get("value"),
                reasoning=args.get("reasoning"))
            add("verdict", 1, seq, accepted=bool(result.get("accepted")),
                why=result.get("why") or result.get("note"))
        else:
            add("action", 1, seq, tool=tool, objective=None, args=args,
                observed=_digest(tool or "", result))
    for text in trailing:
        add("thought", 2, None, text=text)
    for text in remarks:
        add("remark", 2, None, text=text)

    return {"run_id": run_dir.name, "patient_id": meta.get("patient_id"),
            "spec_id": meta.get("spec_id"), "spec_hash": meta.get("spec_hash"),
            "steps": steps}


_MARKS = {"thought": "~ thought   ", "decision": "* DECISION  ", "action": "> action    ",
          "evidence": "+ evidence  ", "submission": "! submit    ", "verdict": "= verdict   ",
          "result": "# RESULT    ", "remark": "~ remark    "}


def render(trace: dict[str, Any]) -> str:
    """The trace as text a person reads top to bottom. Self-reported lines are tagged."""
    out = [f"{trace['patient_id']} | {trace['spec_id']} | {trace['run_id']}"]
    for n, s in enumerate(trace["steps"], 1):
        mark = _MARKS.get(s["kind"], "  ")
        tag = "" if s["channel"] == "server" else " [self-reported]"
        head = f"{n:3d} {mark}"
        pad = " " * len(head)
        if s["kind"] in ("thought", "remark"):
            out.append(f"{head}{s['text']}{tag}")
        elif s["kind"] == "decision":
            out.append(f"{head}facing: {s['facing']}")
            out.append(f"{pad}decided: {s['decision']}")
            out.append(f"{pad}because: {s['because']}")
            if s.get("grounding"):
                out.append(f"{pad}grounding: {', '.join(s['grounding'])}")
            if s.get("used"):
                # An unverified citation is marked here rather than dropped: the claim IS
                # the finding, and hiding it would hide the falsest kind of warrant.
                marks = []
                for u in s["used"]:
                    v = u.get("verified")
                    tag = ("" if v is True else " (UNVERIFIED)" if v is False else " (claimed)")
                    depth = f" [{u['depth']}]" if u.get("depth") else ""
                    marks.append(f"{u.get('ref')}{depth}{tag}")
                out.append(f"{pad}used: {', '.join(marks)}")
            if s.get("options"):
                out.append(f"{pad}set aside: {'; '.join(s['options'])}")
            if s.get("context"):
                out.append(f"{pad}server state: {json.dumps(s['context'], ensure_ascii=False)}")
        elif s["kind"] == "action":
            obj = f' objective="{s["objective"]}"' if s.get("objective") else ""
            out.append(f"{head}{s['tool']} {json.dumps(s['args'], ensure_ascii=False)}{obj}"
                       f" -> {s['observed']}")
        elif s["kind"] == "evidence":
            if s.get("error"):
                out.append(f"{head}REFUSED: {s['error']}")
            else:
                out.append(f"{head}{s['note_id']} [{s['span'][0]},{s['span'][1]})"
                           f' "{s["quote"]}" — {s["supports"]}')
        elif s["kind"] == "submission":
            out.append(f"{head}{s['status']} {json.dumps(s.get('value'), ensure_ascii=False)}")
            if s.get("reasoning"):
                out.append(f"{pad}reasoning: {s['reasoning']}")
        elif s["kind"] == "verdict":
            word = "ACCEPTED" if s["accepted"] else "REFUSED"
            out.append(f"{head}{word} — {s['why']}")
        elif s["kind"] == "result":
            out.append(f"{head}{s['status']} {json.dumps(s.get('value'), ensure_ascii=False)}")
    return "\n".join(out)
