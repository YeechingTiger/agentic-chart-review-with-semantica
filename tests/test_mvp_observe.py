"""The decision trace: both layers interleaved in order, and readable with Layer 2 gone.

The fixtures are hand-written files in the exact shapes the toolserver (Layer 1) and codex
0.149 (Layer 2) write, so this test also pins those shapes: if either writer drifts, the
fixture stops matching reality and the integration test catches it there.
"""
from __future__ import annotations

import json
from pathlib import Path

from acr.mvp.observe import decision_trace, render


def _write_layers(run_dir: Path, with_layer2: bool = True) -> None:
    run_dir.mkdir(parents=True)
    layer1 = [
        {"seq": 1, "ts": "t", "kind": "run_meta", "spec_id": "STORE.390.date_of_initial_diagnosis",
         "spec_hash": "h", "patient_id": "SYN0001", "submittable": ["FOUND"]},
        {"seq": 2, "ts": "t", "kind": "tool_call", "tool": "search",
         "args": {"query": "adenocarcinoma", "objective": "find the diagnosing pathology"},
         "result": {"hits": [], "n": 2}, "ok": True},
        {"seq": 3, "ts": "t", "kind": "tool_call", "tool": "note_decision",
         "args": {"facing": "cytology 04-12 vs biopsy 04-27", "decision": "read the cytology",
                  "because": "earlier document governs if unambiguous",
                  "used": ["search:adenocarcinoma", "note:SPD_2023-04-12"],
                  "grounding": ["contract", "own_knowledge"],
                  "options": ["date by the biopsy unread"]},
         "result": {"noted": True, "n_decisions": 1,
                    "grounding": ["contract", "own_knowledge"],
                    "used": [{"ref": "search:adenocarcinoma", "kind": "search",
                              "verified": True},
                             {"ref": "note:SPD_2023-04-12", "kind": "note", "verified": False,
                              "why": "this run never read or surfaced it"}],
                    "context": {"n_searches": 1, "n_evidence": 0}}, "ok": True},
        {"seq": 4, "ts": "t", "kind": "tool_call", "tool": "record_evidence",
         "args": {"note_id": "SPD_2023-04-12", "start": 310, "end": 324, "supports": "histology"},
         "result": {"recorded": True, "n_evidence": 1, "quote": "adenocarcinoma"}, "ok": True},
        {"seq": 5, "ts": "t", "kind": "tool_call", "tool": "submit_answer",
         "args": {"status": "FOUND", "value": {"date_of_initial_diagnosis": "20230412"},
                  "reasoning": "cytology dates the case"},
         "result": {"accepted": True, "why": "obligations discharged"}, "ok": True},
        {"seq": 6, "ts": "t", "kind": "answer_accepted", "status": "FOUND",
         "value": {"date_of_initial_diagnosis": "20230412"}},
    ]
    (run_dir / "trace.jsonl").write_text(
        "\n".join(json.dumps(e) for e in layer1) + "\n", encoding="utf-8")
    if not with_layer2:
        return
    # codex --json shapes, as observed against 0.149.1: reasoning items complete BEFORE the
    # mcp_tool_call they precede starts.
    layer2 = [
        {"type": "thread.started", "thread_id": "x"},
        {"type": "item.completed", "item": {"id": "i0", "type": "reasoning",
                                            "text": "Pathology is the strongest source."}},
        {"type": "item.started", "item": {"id": "i1", "type": "mcp_tool_call",
                                          "server": "chart", "tool": "search"}},
        {"type": "item.completed", "item": {"id": "i1", "type": "mcp_tool_call",
                                            "server": "chart", "tool": "search"}},
        {"type": "item.completed", "item": {"id": "i2", "type": "reasoning",
                                            "text": "Two candidates; check ambiguity first."}},
        {"type": "item.started", "item": {"id": "i3", "type": "mcp_tool_call",
                                          "server": "chart", "tool": "note_decision"}},
        {"type": "item.completed", "item": {"id": "i3", "type": "mcp_tool_call",
                                            "server": "chart", "tool": "note_decision"}},
        {"type": "item.started", "item": {"id": "i4", "type": "mcp_tool_call",
                                          "server": "chart", "tool": "record_evidence"}},
        {"type": "item.started", "item": {"id": "i5", "type": "mcp_tool_call",
                                          "server": "chart", "tool": "submit_answer"}},
        {"type": "item.completed", "item": {"id": "i6", "type": "agent_message",
                                            "text": "Submitted 20230412."}},
        {"type": "turn.completed", "usage": {}},
    ]
    (run_dir / "layer2_codex.jsonl").write_text(
        "\n".join(json.dumps(e) for e in layer2) + "\n", encoding="utf-8")


def test_thoughts_interleave_before_the_calls_they_preceded(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_layers(run_dir)
    trace = decision_trace(run_dir)
    kinds = [(s["kind"], s["channel"]) for s in trace["steps"]]
    assert kinds == [
        ("thought", "self_reported"),      # before the search
        ("action", "server"),              # search
        ("thought", "self_reported"),      # before the decision point
        ("decision", "server"),            # note_decision
        ("evidence", "server"),
        ("submission", "server"),
        ("verdict", "server"),
        ("result", "server"),
        ("remark", "self_reported"),       # the final agent message
    ]
    decision = next(s for s in trace["steps"] if s["kind"] == "decision")
    assert decision["facing"] == "cytology 04-12 vs biopsy 04-27"
    assert decision["options"] == ["date by the biopsy unread"]
    assert "decision_type" not in decision   # the trace carries no taxonomy
    assert decision["grounding"] == ["contract", "own_knowledge"]
    assert decision["context"] == {"n_searches": 1, "n_evidence": 0}
    assert [u["ref"] for u in decision["used"]] == ["search:adenocarcinoma",
                                                    "note:SPD_2023-04-12"]
    action = next(s for s in trace["steps"] if s["kind"] == "action")
    assert action["objective"] == "find the diagnosing pathology"
    assert action["observed"] == "2 hit(s)"


def test_layer2_deleted_loses_only_the_self_reported_lines(tmp_path: Path):
    with_l2 = tmp_path / "a"
    without = tmp_path / "b"
    _write_layers(with_l2, with_layer2=True)
    _write_layers(without, with_layer2=False)
    full = decision_trace(with_l2)["steps"]
    bare = decision_trace(without)["steps"]
    assert [s for s in full if s["channel"] == "server"] == bare
    assert all(s["channel"] == "server" for s in bare)


def test_render_reads_top_to_bottom_and_tags_self_report(tmp_path: Path):
    run_dir = tmp_path / "run"
    _write_layers(run_dir)
    text = render(decision_trace(run_dir))
    assert text.splitlines()[0].startswith("SYN0001 | STORE.390")
    assert "[self-reported]" in text
    assert text.index("Pathology is the strongest source") < text.index("adenocarcinoma")
    assert "facing: cytology 04-12 vs biopsy 04-27" in text
    assert "ACCEPTED" in text and "FOUND" in text
    # The false citation is shown as false, not quietly dropped.
    assert "note:SPD_2023-04-12 (UNVERIFIED)" in text
    # own_knowledge on the page is the point of collecting it: it names where the contract
    # ran out, which is the question this instrument exists to raise.
    assert "grounding: contract, own_knowledge" in text
