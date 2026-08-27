"""The reviewer UI and stock Semantica API read the same persisted decision graph."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from acr.mvp.ledger import SemanticaLedger
from acr.mvp.review_ui import create_review_app, review_url, serve_review_ui
from acr.mvp.task_presentation import content_hash

pytest.importorskip("semantica")
fastapi = pytest.importorskip("fastapi")


def _write_analysis(tmp_path: Path) -> tuple[Path, Path]:
    run_id, analysis_id = "ui-run", "ui-analysis"
    cycle = {
        "cycle_id": f"{run_id}:cycle:2", "run_id": run_id,
        "source_event_ids": ["layer1:2"], "source_seq_range": [2, 2],
        "source_event_time": "2026-08-26T12:00:00Z",
        "state_before": {
            "observed_state": {"surfaced_notes": ["NOTE_A"], "read_notes": ["NOTE_A"]},
            "declared_state": {"findings": [], "uncertainties": ["standing unresolved"]},
        },
        "state_after": {
            "observed_state": {
                "surfaced_notes": ["NOTE_A"], "read_notes": ["NOTE_A"],
                "citation_resolutions": [{
                    "testimony_ref": "decision:2",
                    "cited": [{"ref": "decision_rule.1", "status": "CLAIMED_NOT_OFFERED"}],
                    "checked_facts": [],
                }],
            },
            "declared_state": {
                "findings": [{"finding_ref": "finding:1", "note_id": "NOTE_A",
                              "field": "diagnosis_date", "standing": "can_establish",
                              "assertion_class": "pathology_diagnosis"}],
                "uncertainties": [],
            },
        },
        "trigger_event_refs": ["layer1:1"],
        "declared_open_question": "Can this note establish the requested field?",
        "decision_testimony_refs": ["decision:2"], "has_decision_testimony": True,
        "structural_kind": "TOOL_INTERACTION",
        "actions": [{
            "event_ref": "layer1:2", "tool": "note_decision", "ok": True,
            "args": {
                "facing": "Can this note establish the requested field?",
                "decision": "Treat the pathology wording as definitive",
                "because": "The wording ordinarily denotes a confirmed diagnosis",
                "basis_sources": ["chart", "own_knowledge"],
                "cited_refs": ["decision_rule.1"],
                "checked_discriminating_fact_refs": [],
                "rule_coverage_claim": "NO_APPLICABLE_RULE",
                "provisional_inference": "Pathology wording is clinically definitive",
                "alternatives": ["The note merely mentions a suspected diagnosis"],
                "uncertainty": "The supplied arm contains no interpretation rule",
            },
        }],
        "observations": [{
            "event_ref": "layer1:2", "result": {
                "testimony_ref": "decision:2",
                "citation_resolutions": [
                    {"ref": "decision_rule.1", "status": "CLAIMED_NOT_OFFERED"}],
                "checked_fact_resolutions": [],
            },
        }],
    }
    submit_cycle = {
        "cycle_id": f"{run_id}:cycle:3", "run_id": run_id,
        "source_event_ids": ["layer1:3"], "source_seq_range": [3, 3],
        "source_event_time": "2026-08-26T12:00:01Z",
        "state_before": cycle["state_after"],
        "state_after": {
            **cycle["state_after"],
            "observed_state": {**cycle["state_after"]["observed_state"],
                               "result_status": "FOUND",
                               "gates": [{"name": "evidence", "passed": True}]},
        },
        "trigger_event_refs": ["layer1:2"], "declared_open_question": None,
        "decision_testimony_refs": [], "has_decision_testimony": False,
        "structural_kind": "SUBMISSION",
        "actions": [{"event_ref": "layer1:3", "tool": "submit_answer", "ok": True,
                     "args": {"status": "FOUND",
                              "value": {"diagnosis_date": "20230412"},
                              "reasoning": "The accepted pathology evidence establishes it"}}],
        "observations": [{"event_ref": "layer1:3", "result": {"accepted": True}}],
    }
    episode = {
        "episode_id": f"{analysis_id}:episode:1", "source_cycle_ids": [cycle["cycle_id"]],
        "source_event_ids": ["layer1:2"], "decision_function": "standing",
        "decision_subject": "evidence_item",
        "material_question": "Can this note establish the requested field?",
        "decision_rationale": "The wording ordinarily denotes a confirmed diagnosis",
        "scenario": "One read pathology note; whether it establishes the field is unresolved",
        "candidate_set": ["can_establish", "merely_mentions"],
        "decision": "can_establish", "model_interpretation": "A standing judgment",
        "claimed_basis_summary": ["chart", "own_knowledge"],
        "verified_reference_summary": ["decision_rule.1 was not offered"],
        "state_delta": "The note was declared establishing evidence",
        "observed_downstream_refs": [], "hypothesized_impact": [],
        "counterfactual_supported_impact": [],
        "field_provenance": {"decision": "SELF_REPORTED",
                             "scenario": "MODEL_RECONSTRUCTED"},
        "source_refs_by_field": {}, "reconstruction_provenance": "MODEL_RECONSTRUCTED",
        "stability_status": "STABLE_ACROSS_PASSES", "reconstruction_stability": 0.91,
    }
    answer_episode = {
        **episode, "episode_id": f"{analysis_id}:episode:2",
        "source_cycle_ids": [submit_cycle["cycle_id"]], "source_event_ids": ["layer1:3"],
        "decision_function": "what_to_answer", "scenario": "One establishing candidate",
        "decision_subject": "answer_selection",
        "material_question": "What answer should be returned?",
        "decision_rationale": "The accepted evidence establishes the requested value",
        "candidate_set": ["FOUND", "NOT_FOUND"], "decision": "FOUND",
        "model_interpretation": "Submit the accepted field value",
        "claimed_basis_summary": ["chart"], "verified_reference_summary": [],
        "state_delta": "The final answer passed the evidence gate",
    }
    artifact = {
        "schema": "acr.decision_episode_analysis.v1", "analysis_id": analysis_id,
        "run_id": run_id, "trace_id": "f" * 32, "trace_manifest_hash": "manifest",
        "trace_content_hash": "trace-content", "task_presentation_hash": "presentation",
        "reconstructor_identity": "openrouter/openai/gpt-5.6-terra", "pass_index": 1,
        "reconstructor_call": {
            "requested_model": "openrouter/openai/gpt-5.6-terra",
            "resolved_model": "openai/gpt-5.6-terra-2026-08-20",
            "response_provider": "openrouter", "response_id": "gen-ui",
            "identity_status": "RETURNED_BY_PROVIDER",
        },
        "cycles_hash": "cycles", "cycles": [cycle, submit_cycle],
        "episodes": [episode, answer_episode], "mechanical_cycle_ids": [],
        "cycle_annotations": {
            cycle["cycle_id"]: {
                "role": "DECISION_BEARING", "decision_function": "standing"},
            submit_cycle["cycle_id"]: {
                "role": "DECISION_BEARING", "decision_function": "what_to_answer"},
        }, "causal_assertions": [{
            "assertion_id": f"{analysis_id}:causal:1",
            "source_episode_id": episode["episode_id"],
            "target_episode_id": answer_episode["episode_id"],
            "relationship_type": "INFLUENCED", "evidence_refs": ["decision:2"],
            "reasoning": "The standing decision supplied the candidate that was submitted",
            "provenance": "MODEL_RECONSTRUCTED",
        }],
        "stability_status": "STABLE_ACROSS_PASSES", "reconstruction_stability": 0.91,
    }
    artifact["analysis_artifact_hash"] = content_hash(artifact)
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    artifact_path = analysis_dir / "analysis.json"
    artifact["artifact_ref"] = str(artifact_path)
    artifact_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")

    ledger_path = tmp_path / "ledger.json"
    ledger = SemanticaLedger(ledger_path)
    ledger.project_analysis(artifact)
    ledger.select_analysis(run_id, analysis_id, selected_by="test-reviewer",
                           reason="UI integration fixture")
    return ledger_path, analysis_dir


def test_review_app_is_semantica_decisions_and_saves_native_review_provenance(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from fastapi.testclient import TestClient

    ledger_path, run_dir = _write_analysis(tmp_path)
    monkeypatch.setenv("SEMANTICA_API_KEY", "test-review-key")
    monkeypatch.delenv("SEMANTICA_ALLOW_ANONYMOUS", raising=False)
    app = create_review_app(ledger_path, "ui-run", run_dir=run_dir)

    with TestClient(app) as client:
        assert client.get("/api/decision-review/bundles").status_code == 401
        headers = {"X-API-Key": "test-review-key"}
        bundles = client.get("/api/decision-review/bundles", headers=headers)
        bundle_id = bundles.json()[0]["bundle_id"]
        narrative = client.get(
            f"/api/decision-review/bundles/{bundle_id}", headers=headers)
        atomic_steps = [row for row in narrative.json()["steps"]
                        if not row.get("is_conclusion")]
        step_id = atomic_steps[0]["entity_id"]
        history_before = client.get(
            f"/api/decision-review/bundles/{bundle_id}/reviews", headers=headers)
        saved_review = client.post(
            f"/api/decision-review/bundles/{bundle_id}/reviews", headers=headers, json={
            "review_id": "ui-human-review", "reviewer": "test-domain-expert",
            "step_verdicts": [{
                "step_id": step_id, "verdict": "INCORRECT",
                "corrected_decision": "Treat this note as merely mentioning the target",
                "issue_source": "DECISION_JUDGMENT",
                "materiality": "DOES_NOT_CHANGE_FINAL", "note": "Fixture review",
            }],
            "final_answer_verdict": "CORRECT", "final_issue_source": "NONE",
            "final_rationale": "The later evidence still establishes the submitted answer.",
        })
        history_after = client.get(
            f"/api/decision-review/bundles/{bundle_id}/reviews", headers=headers)
        decisions = client.get("/api/decisions", headers=headers)
        old_acr_route = client.get("/api/acr/review", headers=headers)
        page = client.get("/")

    assert bundles.status_code == narrative.status_code == decisions.status_code == 200
    assert page.status_code == 200
    assert history_before.status_code == history_after.status_code == 200
    assert saved_review.status_code == 201
    assert len(atomic_steps) == 1
    assert atomic_steps[0]["audit_unit"] == "ATOMIC_DECISION"
    assert atomic_steps[0]["phase"] == "EVIDENCE_JUDGMENT"
    assert len(atomic_steps[0]["decision_ids"]) == 1
    assert "Recorded dependency: runtime decision 2" in \
        atomic_steps[0]["why_next_or_stop"]
    assert atomic_steps[0]["causal_links"][0]["evidence_refs"] == ["decision:2"]
    assert old_acr_route.status_code == 404
    assert len(decisions.json()) == 2
    stored = saved_review.json()["record"]
    assert stored["derived"]["outcome"] == "CORRECT_OUTCOME_FLAWED_REASONING"
    assert stored["derived"]["first_incorrect_step_id"] == step_id
    assert len(history_before.json()["reviews"]) == 0
    assert [row["review_id"] for row in history_after.json()["reviews"]] == [
        "ui-human-review",
    ]


def test_opening_an_old_run_appends_the_current_readable_projection(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from acr.mvp import semantica_audit

    ledger_path, run_dir = _write_analysis(tmp_path)
    ledger = SemanticaLedger(ledger_path)
    old_bundle = ledger.decision_narrative_bundle_id("ui-run", "ui-analysis")
    monkeypatch.setattr(
        semantica_audit, "NARRATIVE_PROJECTION_REVISION", "future-ui-test")

    app = create_review_app(ledger_path, "ui-run", run_dir=run_dir)

    new_bundle = ledger.decision_narrative_bundle_id("ui-run", "ui-analysis")
    assert new_bundle != old_bundle
    assert app.state.acr_decision_bundle_id == new_bundle
    rows = ledger.provenance_manager("ui-run", "ui-analysis").audit_log(format="json")
    assert any(row.get("entity_type") == "decision_narrative"
               and (row.get("metadata") or {}).get("bundle_id") == new_bundle
               for row in rows)
    assert (run_dir / "semantica-provenance.sqlite3").is_file()
    assert not (run_dir / "human_decision_reviews.jsonl").exists()


def test_mvp_cli_advertises_the_semantica_review_ui(capsys: pytest.CaptureFixture[str]):
    from acr.mvp.cli import main

    with pytest.raises(SystemExit) as exc:
        main(["--help"])

    assert exc.value.code == 0
    assert "review-ui" in capsys.readouterr().out


def test_review_url_opens_semantica_decisions_workspace():
    assert review_url("127.0.0.1", 8765, "bundle:a/b") == (
        "http://127.0.0.1:8765/?workspace=decisions&bundle_id=bundle%3Aa%2Fb"
    )


def test_review_server_refuses_anonymous_or_non_loopback_operation(
        monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SEMANTICA_API_KEY", "test-review-key")
    monkeypatch.setenv("SEMANTICA_ALLOW_ANONYMOUS", "true")
    with pytest.raises(ValueError, match="anonymous"):
        serve_review_ui(Path("unused.json"), "unused", open_browser=False)

    monkeypatch.setenv("SEMANTICA_ALLOW_ANONYMOUS", "false")
    with pytest.raises(ValueError, match="loopback"):
        serve_review_ui(Path("unused.json"), "unused", host="0.0.0.0", open_browser=False)


def test_review_app_refuses_a_run_directory_that_is_not_the_selected_provenance_store(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    ledger_path, run_dir = _write_analysis(tmp_path)
    monkeypatch.setenv("SEMANTICA_API_KEY", "test-review-key")

    with pytest.raises(ValueError, match="selected Semantica analysis"):
        create_review_app(ledger_path, "ui-run", run_dir=run_dir / "wrong")
