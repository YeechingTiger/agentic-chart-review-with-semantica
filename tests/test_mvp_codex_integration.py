"""The Codex App Server/MCP protocol seam with a real Codex and scripted endpoint.

This is deliberately a deterministic transport test, not model acceptance: the App Server
drives the MCP toolserver, the toolserver records Layer 1 and writes result.json, and the runner
archives the typed App Server stream as Layer 2. The trace sink is replaced in-memory because
Langtrace is tested by the live OpenRouter pilot, not by pretending an HTTP collector is the
real service. The scripted trajectory's evidence span comes from the live SYN0001 chart, so
corpus edits cannot silently break it.

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
from acr.mvp.runner import (
    CodexCompatibilityError,
    require_codex_tool_boundary,
    run_patient,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"
PATIENT = ROOT / "corpus" / "patients" / "SYN0001"

CODEX = shutil.which("codex")


def _codex_incompatibility() -> str | None:
    if CODEX is None:
        return "codex binary not on PATH"
    try:
        require_codex_tool_boundary(CODEX)
    except CodexCompatibilityError as exc:
        return str(exc)
    return None


CODEX_INCOMPATIBILITY = _codex_incompatibility()


def test_an_old_codex_fails_closed_before_a_run_directory_is_created(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    class Probe:
        returncode = 0
        stderr = ""

        def __init__(self, stdout: str):
            self.stdout = stdout

    def fake_run(cmd, **_kwargs):
        if cmd[1:] == ["features", "list"]:
            return Probe("shell_tool  stable  true\nunified_exec  stable  true\n")
        return Probe("codex-cli 0.145.0\n")

    monkeypatch.setattr("acr.mvp.runner._resolve_codex_bin", lambda _value: "old-codex")
    monkeypatch.setattr("acr.mvp.runner.subprocess.run", fake_run)
    with pytest.raises(CodexCompatibilityError, match=r"view_image.*0\.150\.0"):
        run_patient(SPEC, PATIENT, tmp_path / "runs", codex_bin="old-codex")
    assert not (tmp_path / "runs").exists()


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
            "cited_refs": ["search:adenocarcinoma", f"note:{hit.note_id}",
                           "decision_rule.1"],
            "basis_sources": ["task_contract", "chart"],
            "checked_discriminating_fact_refs": [],
            "rule_coverage_claim": "COVERED_WITH_INTERPRETATION",
            "alternatives": ["date by the later biopsy"]}},
        {"tool": "read", "args": {
            "note_id": hit.note_id, "objective": "judge whether this note can establish the field"}},
        {"tool": "record_finding", "args": {
            "note_id": hit.note_id, "field": "date_of_initial_diagnosis",
            "standing": "can_establish", "assertion_class": "pathology_diagnosis",
            "source_start": hit.start, "source_end": hit.end,
            "decision_testimony_ref": "decision:4"}},
        {"tool": "record_evidence", "args": {
            "note_id": hit.note_id, "start": hit.start, "end": hit.end,
            "supports": "pathology names the histology",
            "field": "date_of_initial_diagnosis"}},
        {"tool": "submit_answer", "args": {
            "status": "FOUND", "value": {"date_of_initial_diagnosis": "20230412"},
            "reasoning": "FNA pathology dated 2023-04-12 establishes the diagnosis"}},
        {"message": "Submitted: date_of_initial_diagnosis = 20230412."},
    ]


@pytest.mark.skipif(CODEX_INCOMPATIBILITY is not None,
                    reason=CODEX_INCOMPATIBILITY or "compatible Codex required")
def test_codex_app_server_drives_the_toolserver_end_to_end(
        tmp_path: Path, scripted_steps: list[dict], monkeypatch: pytest.MonkeyPatch):
    class TraceSink:
        trace_id = "deterministic-protocol-test"

        def __init__(self, **_kwargs):
            pass

        def codex_event(self, _event):
            pass

        def review_model_call(self, **_kwargs):
            pass

        def publish_run(self, _run_dir):
            pass

        def finish(self, **_kwargs):
            pass

    class TraceReader:
        project_id = "deterministic-protocol-test"

        def __init__(self, **_kwargs):
            pass

        def get_review(self, _trace_id):
            return object()

    monkeypatch.setattr("acr.mvp.runner.LangtraceRun", TraceSink)
    monkeypatch.setattr("acr.mvp.runner.LangtraceClient", TraceReader)
    script = tmp_path / "script.json"
    script.write_text(json.dumps(scripted_steps), encoding="utf-8")
    httpd = fake_model.serve(script, tmp_path / "requests.jsonl")
    try:
        base_url = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
        run_dir = run_patient(SPEC, PATIENT, tmp_path / "runs",
                              model="scripted", base_url=base_url, api_key="local",
                              timeout_s=300, langtrace_api_key="test",
                              langtrace_api_host="http://langtrace.invalid/api/trace")
    finally:
        httpd.shutdown()

    # The answer went through the gate and out to disk.
    result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "FOUND", (
        f"run did not reach an accepted answer: {result.get('why')}\n"
        f"stderr tail: {(run_dir / 'codex_stderr.log').read_text()[-2000:]}")
    assert result["value"] == {"date_of_initial_diagnosis": "20230412"}
    assert result["evidence"] and result["evidence"][0]["quote"] == "adenocarcinoma"

    # Only the chart namespace can reach patient data. Codex retains inert planning helpers and
    # MCP resource discovery (this server publishes no resources); pinning that complete list at
    # the wire seam makes every newly exposed capability a deliberate review.
    requests = [json.loads(ln) for ln in
                (tmp_path / "requests.jsonl").read_text(encoding="utf-8").splitlines()]
    offered = requests[0]["body"]["tools"]
    assert {(tool["type"], tool["name"]) for tool in offered} == {
        ("function", "list_mcp_resources"),
        ("function", "list_mcp_resource_templates"),
        ("function", "read_mcp_resource"),
        ("function", "update_plan"),
        ("function", "request_user_input"),
        ("namespace", "mcp__chart"),
    }
    chart = next(tool for tool in offered if tool["name"] == "mcp__chart")
    assert {member["name"] for member in chart["tools"]} == {
        "list_documents", "search", "read", "note_decision",
        "record_finding", "record_evidence", "submit_answer",
    }

    # Layer 1 saw every scripted tool call, through codex, at the server side.
    events = [json.loads(ln) for ln in
              (run_dir / "trace.jsonl").read_text(encoding="utf-8").splitlines()]
    tools = [e["tool"] for e in events if e["kind"] == "tool_call"]
    assert tools == ["list_documents", "search", "note_decision", "read",
                     "record_finding", "record_evidence", "submit_answer"]
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
    assert decision["basis_sources"] == ["task_contract", "chart"]
    assert decision["rule_coverage_claim"] == "COVERED_WITH_INTERPRETATION"
    assert decision["context"]["n_searches"] == 1   # server state at the moment of deciding
    thought = next(s for s in steps if s["kind"] == "thought")
    assert thought["channel"] == "self_reported"
    assert thought["text"] == "Survey the chart before searching."

    # The exact Task Presentation is immutable and runtime testimony is not projected live.
    presentation = json.loads((run_dir / "task_presentation.json").read_text(encoding="utf-8"))
    meta = json.loads((run_dir / "runner_meta.json").read_text(encoding="utf-8"))
    assert presentation["presentation_hash"] == meta["task_presentation_hash"]
    assert presentation["arm_id"] == "detailed"
    assert meta["review_model_call"]["requested_model"] == "scripted"
    assert meta["review_model_call"]["codex_thread_id"]
    assert meta["review_model_call"]["codex_turn_id"]
    assert meta["review_model_call"]["identity_status"] == "CODEX_HARNESS_IDS_ONLY"
