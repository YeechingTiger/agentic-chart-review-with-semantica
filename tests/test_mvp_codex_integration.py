"""The whole MVP loop with the real codex binary and a scripted local model.

This is acceptance test 1 of docs/MVP_CODEX_DESIGN.md in miniature: codex exec drives the MCP
toolserver, the toolserver records Layer 1 and writes result.json, the runner archives codex's
event stream as Layer 2 — and then acceptance test 3: the run ingests into the ledger and the
audit chain reads back. The model is `fake_model` replaying a trajectory whose spans come from
the real SYN0001 chart (looked up at test time, so corpus edits cannot silently break the span).

Skipped when the codex binary is absent: the subject here is the harness seam itself, and a mock
of codex would test our assumptions about codex, which is exactly the class of error this test
exists to catch.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from acr.chartstore.corpus import PatientChart
from acr.mvp import fake_model
from acr.mvp.ledger import NullLedger, ingest_run
from acr.mvp.runner import run_patient

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"
PATIENT = ROOT / "corpus" / "patients" / "SYN0001"

pytestmark = pytest.mark.skipif(shutil.which("codex") is None,
                                reason="codex binary not on PATH")


@pytest.fixture(scope="module")
def scripted_steps() -> list[dict]:
    """A competent reviewer's trajectory, with the evidence span taken from the live chart."""
    hit = PatientChart(PATIENT).search("adenocarcinoma")[0]
    return [
        {"thought": "Survey the chart before searching.",
         "tool": "list_documents", "args": {}},
        {"tool": "search", "args": {"query": "adenocarcinoma",
                                    "objective": "establish the histologic diagnosis"}},
        {"tool": "note_decision", "args": {
            "facing": "cytology and biopsy both name the histology",
            "decision": "cite the cytology and date the case there",
            "because": "it is the earlier document and carries the same histology",
            "options": ["date by the later biopsy"]}},
        {"tool": "record_evidence", "args": {
            "note_id": hit.note_id, "start": hit.start, "end": hit.end,
            "supports": "pathology names the histology",
            "field": "date_of_initial_diagnosis"}},
        {"tool": "submit_answer", "args": {
            "status": "FOUND", "value": {"date_of_initial_diagnosis": "20230412"},
            "reasoning": "FNA pathology dated 2023-04-12 establishes the diagnosis"}},
        {"message": "Submitted: date_of_initial_diagnosis = 20230412."},
    ]


def test_codex_drives_the_toolserver_end_to_end(tmp_path: Path, scripted_steps: list[dict]):
    script = tmp_path / "script.json"
    script.write_text(json.dumps(scripted_steps), encoding="utf-8")
    httpd = fake_model.serve(script, tmp_path / "requests.jsonl")
    try:
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        run_dir = run_patient(SPEC, PATIENT, tmp_path / "runs",
                              model="scripted", base_url=base_url, api_key="local",
                              timeout_s=300)
    finally:
        httpd.shutdown()

    # The answer went through the gate and out to disk.
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "FOUND", (
        f"run did not reach an accepted answer: {result.get('why')}\n"
        f"stderr tail: {(run_dir / 'codex_stderr.log').read_text()[-2000:]}")
    assert result["value"] == {"date_of_initial_diagnosis": "20230412"}
    assert result["evidence"] and result["evidence"][0]["quote"] == "adenocarcinoma"

    # Layer 1 saw every scripted tool call, through codex, at the server side.
    events = [json.loads(ln) for ln in
              (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    tools = [e["tool"] for e in events if e["kind"] == "tool_call"]
    assert tools == ["list_documents", "search", "note_decision",
                     "record_evidence", "submit_answer"]
    search_call = next(e for e in events if e.get("tool") == "search")
    assert search_call["args"]["objective"] == "establish the histologic diagnosis"

    # Layer 2 was archived; Layer 1 never depends on it.
    assert (run_dir / "layer2_codex.jsonl").stat().st_size > 0

    # The decision trace reads in order: the scripted thought (Layer 2, self-reported)
    # lands before the survey it preceded, and the decision point sits where it was made.
    from acr.mvp.observe import decision_trace
    steps = decision_trace(run_dir)["steps"]
    kinds = [s["kind"] for s in steps]
    assert kinds.index("thought") < kinds.index("action")
    decision = next(s for s in steps if s["kind"] == "decision")
    assert decision["decision"] == "cite the cytology and date the case there"
    thought = next(s for s in steps if s["kind"] == "thought")
    assert thought["channel"] == "self_reported"
    assert thought["text"] == "Survey the chart before searching."

    # Acceptance 3 in miniature: the finished run distills into a ledger.
    summary = ingest_run(run_dir, NullLedger())
    assert summary["result"] == "FOUND"
    assert summary["n_evidence"] == 1 and summary["n_submissions"] == 1
    assert summary["n_steps"] == 1
