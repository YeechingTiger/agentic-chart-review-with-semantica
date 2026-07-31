from __future__ import annotations

import json

from typer.testing import CliRunner

from acr.commands.cli import app

runner = CliRunner()


def _run_artifacts(tmp_path):
    trace = tmp_path / "RUN-1.jsonl"
    trace.write_text(
        json.dumps({"seq": 1, "kind": "run_start"}) + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "RUN-1.manifest.json"
    manifest.write_text(
        json.dumps({
            "run_id": "RUN-1",
            "patient_id": "CASE-001",
            "spec_id": "STORE.synthetic",
            "spec_hash": "a" * 64,
            "spec_version": "1.0.0",
            "runtime_profile_id": "current-stratified-coverage",
            "runtime_profile_version": "1.0.0",
            "runtime_profile_hash": "b" * 64,
            "trace": str(trace),
            "answer": {
                "status": "FOUND",
                "value": {"histology": "8140"},
                "evidence": [{"note_id": "N1"}],
            },
            "evidence_ledger": {
                "proof_valid": True,
                "evidence": [{"note_id": "N1"}],
            },
            "gate_validated": True,
            "coverage_gate_validated": False,
            "spend": {"usd": 0.01, "priced": True},
        }),
        encoding="utf-8",
    )
    return manifest


def test_evaluation_cli_runs_only_the_v2_pipeline(tmp_path):
    manifest = _run_artifacts(tmp_path)
    result = runner.invoke(
        app,
        [
            "evaluation",
            "run",
            "--manifest",
            str(manifest),
            "--subject-id",
            "CASE-001",
            "--local-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = [
        json.loads(line)
        for line in (tmp_path / "evaluation" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert {row["producer_ref"]["asset_id"] for row in rows} == {
        "evidence-validity",
        "gate-effectiveness",
    }
    assert {row["signal_type"] for row in rows} == {"EVALUATION_RESULT"}
    summary = runner.invoke(
        app,
        [
            "evaluation",
            "summarize",
            "--local-root",
            str(tmp_path),
        ],
    )
    assert summary.exit_code == 0, summary.output
    assert set(json.loads(summary.output)["by_module"]) == {
        "evidence-validity@1.0.0",
        "gate-effectiveness@1.0.0",
    }


def test_evaluation_cli_has_no_v1_profile_or_truth_bag_options(tmp_path):
    manifest = _run_artifacts(tmp_path)
    result = runner.invoke(
        app,
        [
            "evaluation",
            "run",
            "--manifest",
            str(manifest),
            "--subject-id",
            "CASE-001",
            "--profile",
            "safety",
            "--local-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code != 0
    assert "No such option" in result.output


def test_gate_evaluator_rejects_a_runtime_provider_error(tmp_path):
    manifest = _run_artifacts(tmp_path)
    trace = tmp_path / "RUN-1.jsonl"
    trace.write_text(
        json.dumps({
            "seq": 1,
            "kind": "runtime_error",
            "error": "DeploymentNotFound",
        }) + "\n",
        encoding="utf-8",
    )
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    raw["answer"] = {"status": "NO_ANSWER", "value": {}}
    raw["gate_validated"] = False
    raw["degradation"] = {"runtime_or_provider_errors": 1}
    manifest.write_text(json.dumps(raw), encoding="utf-8")

    result = runner.invoke(
        app,
        [
            "evaluation",
            "run",
            "--manifest",
            str(manifest),
            "--subject-id",
            "CASE-001",
            "--local-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    rows = [
        json.loads(line)
        for line in (tmp_path / "evaluation" / "results.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    gate = next(
        row for row in rows
        if row["producer_ref"]["asset_id"] == "gate-effectiveness"
    )
    assert gate["status"] == "FAIL"
    assert "RUNTIME_OR_PROVIDER_ERROR" in gate["payload"]["payload"]["issue_kinds"]
