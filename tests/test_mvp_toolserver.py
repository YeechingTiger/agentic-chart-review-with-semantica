"""The MVP toolserver: the gate's three refusals, the Layer-1 trace, and the stdio transport.

Everything the design doc claims about the tool boundary is asserted here without codex in the
room: the gate refuses exactly what it says it refuses, an accepted answer lands in result.json
with its evidence, and every call — including the refused ones — is in trace.jsonl in order.
The stdio test exercises the same server through the JSON-RPC framing an MCP client would use,
because the framing is the one part a unit call cannot cover.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from acr.mvp.toolserver import ChartToolServer

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"
PATIENT = ROOT / "corpus" / "patients" / "SYN0001"


@pytest.fixture()
def server(tmp_path: Path) -> ChartToolServer:
    return ChartToolServer(SPEC, PATIENT, tmp_path / "run")


def _trace(server: ChartToolServer) -> list[dict]:
    lines = server.trace_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(ln) for ln in lines if ln.strip()]


def _find_span(server: ChartToolServer) -> dict:
    payload, is_error = server.call("search", {"query": "adenocarcinoma"})
    assert not is_error and payload["n"] > 0
    return payload["hits"][0]


def test_the_gate_refuses_a_value_answer_with_no_evidence(server: ChartToolServer):
    payload, _ = server.call("submit_answer", {"status": "FOUND",
                                               "value": {"date_of_initial_diagnosis": "20230412"}})
    assert payload["accepted"] is False
    assert "evidence" in payload["why"]
    assert not (server.run_dir / "result.json").exists()


def test_the_gate_refuses_an_absence_claim_before_an_unfiltered_listing(server: ChartToolServer):
    # A filtered listing does not discharge the obligation — the claim is about the whole chart.
    server.call("list_documents", {"doc_type_contains": "Pathology"})
    payload, _ = server.call("submit_answer", {"status": "EVIDENCE_INSUFFICIENT"})
    assert payload["accepted"] is False
    server.call("list_documents", {})
    payload, _ = server.call("submit_answer", {"status": "EVIDENCE_INSUFFICIENT"})
    assert payload["accepted"] is True


def test_the_gate_refuses_a_status_the_contract_did_not_declare(server: ChartToolServer):
    payload, _ = server.call("submit_answer", {"status": "TECHNICAL_FAILURE"})
    assert payload["accepted"] is False
    assert "submittable" in payload["why"]


def test_out_of_bounds_spans_are_refused_not_clamped(server: ChartToolServer):
    hit = _find_span(server)
    payload, is_error = server.call("record_evidence", {
        "note_id": hit["note_id"], "start": 0, "end": 10_000_000, "supports": "x"})
    assert is_error and "outside the document" in payload["error"]
    payload, is_error = server.call("record_evidence", {
        "note_id": "no-such-note", "start": 0, "end": 5, "supports": "x"})
    assert is_error


def test_an_accepted_answer_writes_result_json_with_its_evidence(server: ChartToolServer):
    hit = _find_span(server)
    payload, is_error = server.call("record_evidence", {
        "note_id": hit["note_id"], "start": hit["start"], "end": hit["end"],
        "supports": "pathology names the histology", "field": "date_of_initial_diagnosis"})
    assert not is_error and payload["recorded"] and payload["quote"]
    payload, _ = server.call("submit_answer", {
        "status": "FOUND", "value": {"date_of_initial_diagnosis": "20230412"},
        "reasoning": "FNA pathology of 2023-04-12"})
    assert payload["accepted"] is True

    result = json.loads((server.run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "FOUND"
    assert result["value"] == {"date_of_initial_diagnosis": "20230412"}
    assert result["patient_id"] == "SYN0001"
    assert len(result["evidence"]) == 1
    assert result["evidence"][0]["quote"]  # server-resolved, not model-pasted


def test_the_trace_records_every_call_including_refusals_in_order(server: ChartToolServer):
    server.call("submit_answer", {"status": "FOUND", "value": {}})   # refused
    hit = _find_span(server)
    server.call("record_evidence", {"note_id": hit["note_id"], "start": hit["start"],
                                    "end": hit["end"], "supports": "s"})
    server.call("submit_answer", {"status": "FOUND", "value": {"date_of_initial_diagnosis": "20230412"}})
    events = _trace(server)
    assert events[0]["kind"] == "run_meta"
    assert events[0]["spec_id"] == "STORE.390.date_of_initial_diagnosis"
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    submits = [e for e in events if e.get("tool") == "submit_answer"]
    assert len(submits) == 2
    assert submits[0]["result"]["accepted"] is False   # the refusal is in the record
    assert submits[1]["result"]["accepted"] is True
    assert events[-1]["kind"] == "answer_accepted"


def test_objective_strings_land_in_the_trace_without_being_enforced(server: ChartToolServer):
    server.call("search", {"query": "adenocarcinoma", "objective": "establish histology"})
    calls = [e for e in _trace(server) if e.get("tool") == "search"]
    assert calls[0]["args"]["objective"] == "establish histology"


def test_stdio_transport_end_to_end(tmp_path: Path):
    """The same review through the JSON-RPC framing codex uses: initialize -> tools/list ->
    tools/call. One subprocess conversation, scripted requests, answers read back by id."""
    run_dir = tmp_path / "run"
    env = {"ACR_MVP_SPEC": str(SPEC), "ACR_MVP_PATIENT_DIR": str(PATIENT),
           "ACR_MVP_RUN_DIR": str(run_dir),
           "PATH": "/usr/bin:/bin", "PYTHONPATH": str(ROOT / "src")}

    def rpc(i, method, **params):
        return json.dumps({"jsonrpc": "2.0", "id": i, "method": method, "params": params})

    requests = [
        rpc(1, "initialize", protocolVersion="2025-06-18", capabilities={},
            clientInfo={"name": "test", "version": "0"}),
        json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
        rpc(2, "tools/list"),
        rpc(3, "tools/call", name="search", arguments={"query": "adenocarcinoma"}),
        rpc(4, "tools/call", name="submit_answer", arguments={"status": "FOUND", "value": {}}),
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "acr.mvp.toolserver"],
        input="\n".join(requests) + "\n", capture_output=True, text=True, env=env, timeout=60)
    by_id = {r["id"]: r for r in map(json.loads, proc.stdout.splitlines()) if "id" in r}

    assert by_id[1]["result"]["protocolVersion"] == "2025-06-18"
    names = {t["name"] for t in by_id[2]["result"]["tools"]}
    assert names == {"list_documents", "search", "read", "record_evidence", "submit_answer"}
    search = json.loads(by_id[3]["result"]["content"][0]["text"])
    assert search["n"] > 0 and by_id[3]["result"]["isError"] is False
    verdict = json.loads(by_id[4]["result"]["content"][0]["text"])
    assert verdict["accepted"] is False   # FOUND with no evidence, refused through the wire too
    assert (run_dir / "trace.jsonl").exists()
