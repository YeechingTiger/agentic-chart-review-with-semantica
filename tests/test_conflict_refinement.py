"""The optional conflict loop reuses deepagents and never turns agreement into proof."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from acr.commands.cli import app
from acr.review import conflict_refinement as C

SPEC_ID = "STORE.400_522_523.site_histology_behavior"
ROOT = Path(__file__).resolve().parents[1]


def manifest(value="C341", *, gate=True, note="PATH1", run_id="r1",
             status="FOUND", open_threads=(), cost=0.1, entity=None, temporal=None):
    evidence = ([{"note_id": note, "stance": "supports",
                  "fields": ["primary_site"]}] if note else [])
    answer = {
        "status": status,
        "value": {"primary_site": value},
        "evidence": evidence,
    }
    if entity is not None:
        answer["entity_anchor"] = entity
    if temporal is not None:
        answer["temporal_anchor"] = temporal
    return {
        "run_id": run_id,
        "patient_id": "SYN0001",
        "spec_id": SPEC_ID,
        "spec_hash": "hash",
        "answer": answer,
        "evidence": evidence,
        "gate_validated": gate,
        "open_threads": {"open": list(open_threads)},
        "rule_attribution": {
            "self_reported": {"accepted": ["decision_rule.1"]}
        },
        "degradation": {},
        "spend": {"usd": cost},
    }


def hypothesis(row, candidate=0):
    return C.Hypothesis.from_manifest(
        row, round_index=0, candidate_index=candidate)


def test_structured_conflicts_include_value_evidence_entity_and_time():
    rows = [
        hypothesis(manifest(
            "C341", note="PATH1", entity={"tumor": "right-upper"},
            temporal={"diagnosis_date": "2020-01-01"}), 0),
        hypothesis(manifest(
            "C343", note="RAD1", entity={"tumor": "left-lower"},
            temporal={"diagnosis_date": "2020-02-01"}), 1),
    ]
    conflicts = C.detect_conflicts(rows)
    kinds = {row.kind for row in conflicts.conflicts}
    assert {
        C.VALUE_CONFLICT,
        C.EVIDENCE_CONFLICT,
        C.ENTITY_CONFLICT,
        C.TIME_CONFLICT,
    } <= kinds
    brief = conflicts.render_for_deepagents()
    assert "not as a vote" in brief
    assert "tumor/entity" in brief
    assert "event timeline" in brief


def test_no_conflict_selects_only_when_every_independent_run_passes_gate():
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        return manifest(run_id=kwargs["run_id"])

    result = C.run_conflict_refinement(
        runner=runner, candidates_per_round=3, max_rounds=2,
        runner_kwargs={"patient_id": "SYN0001"})
    assert result.status == C.NO_CONFLICT
    assert result.selected_manifest
    assert len(calls) == 3
    assert all(call["additional_task_context"] == "" for call in calls)


def test_conflict_brief_is_passed_back_to_the_same_runner_and_can_converge():
    calls = []

    def runner(**kwargs):
        calls.append(kwargs)
        context = kwargs["additional_task_context"]
        if not context:
            value = "C341" if kwargs["run_id"].endswith("c1") else "C343"
            return manifest(value, note="PATH1", run_id=kwargs["run_id"])
        return manifest("C341", note="PATH2", run_id=kwargs["run_id"])

    result = C.run_conflict_refinement(
        runner=runner, candidates_per_round=2, max_rounds=3,
        runner_kwargs={"patient_id": "SYN0001"})
    assert result.status == C.CONVERGED
    assert result.selected_manifest["answer"]["value"]["primary_site"] == "C341"
    assert len(calls) == 4
    assert all("OPTIONAL CONFLICT-REFINEMENT BRIEF" in c["additional_task_context"]
               for c in calls[2:])


def test_ungated_consensus_never_becomes_an_answer():
    def runner(**kwargs):
        return manifest(gate=False, run_id=kwargs["run_id"])

    result = C.run_conflict_refinement(
        runner=runner, candidates_per_round=2, max_rounds=2,
        runner_kwargs={})
    assert result.status == C.REVIEW_REQUIRED
    assert result.selected_manifest is None


def test_no_new_evidence_or_conflict_reduction_stops_early():
    def runner(**kwargs):
        value = "C341" if kwargs["run_id"].endswith("c1") else "C343"
        return manifest(value, note="PATH1", run_id=kwargs["run_id"])

    result = C.run_conflict_refinement(
        runner=runner, candidates_per_round=2, max_rounds=5,
        runner_kwargs={})
    assert result.status == C.REVIEW_REQUIRED
    assert len(result.rounds) == 2
    assert "no new evidence" in result.reason


def test_total_cost_ceiling_stops_without_modal_selection():
    def runner(**kwargs):
        value = "C341" if kwargs["run_id"].endswith("c1") else "C343"
        return manifest(value, run_id=kwargs["run_id"], cost=0.6)

    result = C.run_conflict_refinement(
        runner=runner, candidates_per_round=2, max_rounds=3,
        max_total_usd=1.0, runner_kwargs={})
    assert result.status == C.REVIEW_REQUIRED
    assert len(result.rounds) == 1
    assert result.selected_manifest is None
    assert "ceiling" in result.reason


def test_total_cost_ceiling_refuses_an_unpriced_run():
    def runner(**kwargs):
        value = "C341" if kwargs["run_id"].endswith("c1") else "C343"
        row = manifest(value, run_id=kwargs["run_id"])
        row["spend"]["usd"] = None
        return row

    result = C.run_conflict_refinement(
        runner=runner, candidates_per_round=2, max_rounds=3,
        max_total_usd=1.0, runner_kwargs={})
    assert result.status == C.REVIEW_REQUIRED
    assert result.selected_manifest is None
    assert "no priced USD cost" in result.reason


def test_run_without_feature_flag_calls_deepagents_once_and_directly(tmp_path, monkeypatch):
    calls = []

    def fake_run_patient(**kwargs):
        calls.append(kwargs)
        return {
            "answer": {"status": "EVIDENCE_INSUFFICIENT", "evidence": []},
            "steps": 1,
            "usage": {"total_tokens": 1},
            "rejections": [],
            "open_threads": {},
        }

    monkeypatch.setattr("acr.review.agent.run_patient", fake_run_patient)
    monkeypatch.setattr("acr.core.cli_common.chat_model", lambda *args, **kwargs: object())
    result = CliRunner().invoke(app, [
        "run", "SYN0001",
        "--spec", str(ROOT / "assets/specs/STORE.400_522_523.site_histology_behavior.yaml"),
        "--corpus", str(ROOT / "corpus/patients"),
        "--out", str(tmp_path / "baseline"),
    ])
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert "additional_task_context" not in calls[0]
    assert not list(tmp_path.rglob("conflict-refinement.json"))
