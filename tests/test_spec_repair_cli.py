"""The DEVELOP CLI keeps registry references local, validates patches, and is model-fenced."""
from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from acr.commands.cli import app

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "assets/specs/STORE.400_522_523.site_histology_behavior.yaml"
SPEC_ID = "STORE.400_522_523.site_histology_behavior"
runner = CliRunner()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def gold_document():
    return {
        "schema": "acr.chart_observable_gold/1",
        "cases": [{
            "case_id": "CASE1",
            "spec_id": SPEC_ID,
            "registry_value": {"primary_site": "C341"},
            "registry_source_version": "registry/2026",
            "chart_derivability": "DERIVABLE",
            "chart_answer": {
                "primary_site": {"status": "FOUND", "value": "C341"}
            },
            "gold_evidence": [{
                "note_id": "PATH1",
                "fields": ["primary_site"],
                "stance": "supports",
            }],
            "adjudication": "key_correct",
            "subgroups": ["pathology_present"],
        }],
    }


def manifest(value, run_id, note="PATH1"):
    evidence = [{"note_id": note, "stance": "supports", "fields": ["primary_site"]}]
    return {
        "run_id": run_id,
        "patient_id": "CASE1",
        "spec_id": SPEC_ID,
        "spec_hash": "hash",
        "answer": {
            "status": "FOUND",
            "value": {"primary_site": value},
            "evidence": evidence,
        },
        "evidence": evidence,
        "gate_validated": True,
        "open_threads": {},
        "rule_attribution": {
            "self_reported": {"accepted": ["decision_rule.1"]}
        },
        "degradation": {},
    }


def test_registry_reference_is_local_unresolved_and_never_called_silver_or_gold(tmp_path):
    key = write(tmp_path / "registry.json", {
        "CASE1": {"fields": {"primary_site": "C341"}}
    })
    reference = tmp_path / "registry-reference.json"
    staged = runner.invoke(app, [
        "gold", "stage-registry-reference",
        "--answer-key", str(key),
        "--spec-id", SPEC_ID,
        "--source-version", "registry/2026",
        "--out", str(reference),
        "--local-root", str(tmp_path),
    ])
    assert staged.exit_code == 0, staged.output
    doc = json.loads(reference.read_text(encoding="utf-8"))
    assert doc["schema"] == "acr.registry_reference/1"
    assert doc["contains_phi"] is True and doc["storage"] == "LOCAL_ONLY"
    assert doc["cases"][0]["adjudication"] == "UNRESOLVED"
    assert "chart_answer" not in doc["cases"][0]
    assert "silver" not in staged.output.lower()
    assert "de-identified" in staged.output


def test_cluster_and_diagnose_write_structured_artifacts_without_a_model(tmp_path):
    gold = write(tmp_path / "gold.json", gold_document())
    runs = tmp_path / "runs"
    write(runs / "good.manifest.json", manifest("C341", "good"))
    write(runs / "bad.manifest.json", manifest("C343", "bad", note="RAD1"))

    clustered = tmp_path / "clusters.json"
    result = runner.invoke(app, [
        "repair", "cluster", "--runs", str(runs), "--gold", str(gold),
        "--out", str(clustered), "--local-root", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(clustered.read_text())["summary"]["n_runs"] == 2

    packets = tmp_path / "packets.json"
    result = runner.invoke(app, [
        "repair", "diagnose", "--runs", str(runs), "--gold", str(gold),
        "--spec", str(SPEC), "--out", str(packets), "--local-root", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    packet = json.loads(packets.read_text())["packets"][0]
    assert packet["repair_permitted"] is True
    assert packet["selected"]["run_id"] == "good"
    assert packet["rejected"]["run_id"] == "bad"


def test_supplied_proposal_is_validated_but_never_applied(tmp_path):
    proposal = write(tmp_path / "proposal.json", {
        "case_id": "CASE1",
        "spec_id": SPEC_ID,
        "failure_class": "SPEC_AMBIGUITY",
        "parameter_id": "precedence_conflict_rule",
        "quoted_current_text": (
            "prefer the definitive resection over the initial biopsy"
        ),
        "selected_vs_rejected_difference": {
            "primary_site": {"selected": "C341", "rejected": "C343"}
        },
        "minimal_patch": "State which source wins when dates are equal.",
        "expected_behavior_change": "resolve the ambiguity",
        "change_class": "semantic",
        "source_basis": "STORE item 400",
        "cases_addressed": ["CASE1"],
        "blast_radius": {"computable": False, "basis": "paired replay required"},
        "requires_clinician_signoff": True,
    })
    packet = write(tmp_path / "packet.json", {
        "schema": "acr.contrastive_failure_packet/1",
        "case_id": "CASE1",
        "spec_id": SPEC_ID,
        "spec_hash": "hash",
        "disposition": "SPEC_AMBIGUITY",
        "selected": {},
        "rejected": {},
        "differences": {},
        "gold": {},
        "spec_sections": {},
        "repair_permitted": True,
        "why": "test",
    })
    before = SPEC.read_text(encoding="utf-8")
    out = tmp_path / "validated-proposal.json"
    result = runner.invoke(app, [
        "repair", "propose", "--packet", str(packet), "--spec", str(SPEC),
        "--proposal", str(proposal), "--max-usd", "1", "--out", str(out),
        "--local-root", str(tmp_path),
    ])
    assert result.exit_code == 0, result.output
    assert json.loads(out.read_text())["may_apply_automatically"] is False
    assert SPEC.read_text(encoding="utf-8") == before


def test_paired_validation_refuses_a_per_patient_regression(tmp_path):
    gold = write(tmp_path / "gold.json", gold_document())
    before = tmp_path / "before"
    after = tmp_path / "after"
    write(before / "run.manifest.json", manifest("C341", "before"))
    write(after / "run.manifest.json", manifest("C343", "after"))
    report = tmp_path / "validation.json"
    result = runner.invoke(app, [
        "repair", "validate",
        "--before", str(before), "--after", str(after), "--gold", str(gold),
        "--max-subgroup-drop", "0", "--out", str(report),
        "--local-root", str(tmp_path),
    ])
    assert result.exit_code == 1
    doc = json.loads(report.read_text())
    assert doc["accepted"] is False
    assert doc["regressions"] == ["CASE1"]


def test_compare_refinement_reports_compute_without_claiming_correctness(tmp_path):
    baseline = tmp_path / "baseline"
    row = manifest("C341", "baseline")
    row.update({"spend": {"usd": 0.2}, "steps": 4})
    write(baseline / "run.manifest.json", row)
    refined = write(tmp_path / "refined" / "conflict-refinement.json", {
        "schema": "acr.review.conflict_refinement/1",
        "enabled": True,
        "status": "CONVERGED",
        "n_rounds": 2,
        "rounds": [
            [{"gate_validated": True, "cost_usd": 0.2}],
            [{"gate_validated": True, "cost_usd": 0.3}],
        ],
        "conflicts": [],
        "selected_hypothesis_id": "h2",
        "selected_manifest": None,
        "reason": "test",
    })
    out = tmp_path / "comparison.json"
    result = runner.invoke(app, [
        "eval", "compare-refinement",
        "--baseline", str(baseline), "--refined", str(refined), "--out", str(out),
    ])
    assert result.exit_code == 0, result.output
    report = json.loads(out.read_text())
    assert report["scope"] == "operational_only"
    assert report["correctness_command"] == "acr repair validate"
    assert report["delta"]["deepagents_runs"] == 1
