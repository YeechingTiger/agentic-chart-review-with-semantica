"""Ground-guided spec repair is deterministic, chart-conditional and never applies edits."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from acr import spec_repair as S
from acr.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs/STORE.400_522_523.site_histology_behavior.yaml"


def gold(*, case_id="CASE1", site="C341", derivability=S.DERIVABLE,
         adjudication=S.KEY_CORRECT, evidence=True, subgroup=("pathology_present",)):
    answer = ({"primary_site": S.GoldField(S.FOUND, site)}
              if derivability in (S.DERIVABLE, S.PARTIALLY_DERIVABLE) else {})
    return S.ChartObservableGold(
        case_id=case_id, spec_id="STORE.400_522_523.site_histology_behavior",
        registry_value={"primary_site": site}, registry_source_version="registry/2026",
        chart_derivability=derivability, chart_answer=answer,
        gold_evidence=(S.GoldEvidence("PATH1", ("primary_site",)),) if evidence else (),
        adjudication=adjudication, subgroups=subgroup)


def manifest(site="C341", *, patient="CASE1", gate=True, note="PATH1", status="FOUND",
             rules=("decision_rule.1",), run_id="r1", degradation=None):
    evidence = ([{"note_id": note, "stance": "supports", "fields": ["primary_site"],
                  "quote": "right upper lobe"}] if note else [])
    return {
        "run_id": run_id, "patient_id": patient,
        "spec_id": "STORE.400_522_523.site_histology_behavior", "spec_hash": "abc",
        "answer": {"status": status, "value": {"primary_site": site}, "evidence": evidence},
        "evidence": evidence, "gate_validated": gate, "open_threads": {},
        "rule_attribution": {"self_reported": {"accepted": list(rules)}},
        "degradation": degradation or {},
    }


def sig(site="C341", **kw):
    return S.BehaviorSignature.from_manifest(manifest(site, **kw))


def test_registry_value_is_not_repair_gold_until_chart_derivability_is_adjudicated():
    unresolved = gold(derivability=S.UNRESOLVED, adjudication=S.ADJUDICATION_UNRESOLVED,
                      evidence=False)
    assert not unresolved.usable_for_repair
    report = S.audit_gold([unresolved])
    assert report["repair_ready"] is False
    assert {f["severity"] for f in report["findings"]} == {"BLOCK"}


def test_registrar_corrected_chart_answer_can_guide_repair_when_registry_key_is_wrong():
    corrected = S.ChartObservableGold(
        case_id="CASE1", spec_id=gold().spec_id,
        registry_value={"primary_site": "C341"},
        registry_source_version="registry/2026",
        chart_derivability=S.DERIVABLE,
        chart_answer={"primary_site": S.GoldField(S.FOUND, "C343")},
        gold_evidence=(S.GoldEvidence("PATH1", ("primary_site",)),),
        adjudication=S.KEY_WRONG,
        adjudication_rationale="registrar confirmed the pathology supports C343")
    assert corrected.usable_for_repair
    dist = S.cluster_behaviors([sig("C343")], corrected)
    assert dist.gold_consistency == 1.0


def test_key_wrong_needs_registrar_rationale_and_empty_gold_is_not_ready():
    unowned = gold(adjudication=S.KEY_WRONG)
    assert not unowned.usable_for_repair
    assert S.audit_gold([unowned])["repair_ready"] is False
    assert S.audit_gold([])["repair_ready"] is False


def test_outside_chart_value_cannot_be_repaired_toward():
    outside = S.ChartObservableGold(
        case_id="CASE1", spec_id=gold().spec_id, registry_value={"primary_site": "C341"},
        registry_source_version="registry/2026", chart_derivability=S.NOT_DERIVABLE,
        chart_answer={}, gold_evidence=(), adjudication=S.OUTSIDE_CHART)
    dist = S.cluster_behaviors([sig("C349")], outside)
    packet = S.diagnose(dist, outside, load_spec(SPEC))
    assert packet.disposition == S.GOLD_NOT_CHART_OBSERVABLE
    assert packet.repair_permitted is False
    assert "guess" in packet.why


def test_behaviour_clusters_use_answer_evidence_rules_and_gate_not_reasoning():
    a = manifest("C341", run_id="a")
    b = manifest("C341", run_id="b")
    a["answer"]["reasoning"] = "one long explanation"
    b["answer"]["reasoning"] = "different prose"
    rows = [S.BehaviorSignature.from_manifest(x) for x in (a, b)]
    dist = S.cluster_behaviors(rows, gold())
    assert len(dist.clusters) == 1
    assert dist.gold_consistency == 1.0
    assert dist.grounded_consistency == 1.0
    assert dist.behavioral_entropy == 0.0


def test_portable_behaviour_artifacts_do_not_copy_manifest_paths():
    person_id = "1168" + ("0" * 12)
    source_manifest = manifest(run_id=f"{person_id}__run")
    row = S.BehaviorSignature.from_manifest(
        source_manifest, source=f"/secure/{person_id}/run.manifest.json",
        case_id="CASE1")
    assert person_id not in json.dumps(row.to_dict())
    distribution = S.cluster_behaviors([row], gold())
    assert person_id not in json.dumps(distribution.to_dict())


def test_consistent_wrong_is_low_entropy_but_not_grounded_correct():
    dist = S.cluster_behaviors(
        [sig("C343", run_id="a"), sig("C343", run_id="b")], gold())
    assert dist.behavioral_entropy == 0.0
    assert dist.gold_consistency == 0.0
    assert dist.grounded_consistency == 0.0


def test_correct_value_without_gate_is_not_selected_as_grounded():
    dist = S.cluster_behaviors([sig("C341", gate=False)], gold())
    assert dist.gold_consistency == 1.0
    assert dist.grounded_consistency == 0.0


def test_global_abstention_status_can_match_field_level_gold_without_an_asserted_value():
    abstain_gold = S.ChartObservableGold(
        case_id="CASE1", spec_id=gold().spec_id,
        registry_value={"primary_site": "C341"},
        registry_source_version="registry/2026",
        chart_derivability=S.PARTIALLY_DERIVABLE,
        chart_answer={
            "primary_site": S.GoldField(S.EVIDENCE_INSUFFICIENT)
        },
        gold_evidence=(),
        adjudication=S.KEY_CORRECT)
    row = manifest(status=S.EVIDENCE_INSUFFICIENT, note=None)
    row["answer"]["value"] = {}
    signature = S.BehaviorSignature.from_manifest(row)
    assert S.matches_gold(signature, abstain_gold)
    assert S.cluster_behaviors([signature], abstain_gold).gold_consistency == 1.0


def test_diagnosis_contrasts_selected_and_rejected_and_routes_missing_witness_to_retrieval():
    dist = S.cluster_behaviors([
        sig("C341", run_id="good"),
        sig("C343", note="RAD1", run_id="bad"),
    ], gold())
    packet = S.diagnose(dist, gold(), load_spec(SPEC))
    assert packet.repair_permitted
    assert packet.disposition == S.RETRIEVAL_FAILURE
    assert packet.selected["run_id"] == "good"
    assert packet.rejected["run_id"] == "bad"
    assert "field_results" in packet.differences


def test_no_correct_run_needs_an_adjudicated_witness_before_repair():
    with_witness = S.diagnose(
        S.cluster_behaviors([sig("C343")], gold()), gold(), load_spec(SPEC))
    assert with_witness.disposition == S.NO_CORRECT_BEHAVIOUR
    assert with_witness.repair_permitted
    without = gold(evidence=False)
    no_witness = S.diagnose(
        S.cluster_behaviors([sig("C343")], without), without, load_spec(SPEC))
    assert no_witness.repair_permitted is False


def proposal(**over):
    row = {
        "case_id": "CASE1", "spec_id": gold().spec_id,
        "failure_class": S.SPEC_AMBIGUITY,
        "parameter_id": "precedence_conflict_rule",
        "quoted_current_text": "prefer the definitive resection over the initial biopsy",
        "selected_vs_rejected_difference": {"primary_site": ["C341", "C343"]},
        "minimal_patch": "State which report wins when dates are equal.",
        "expected_behavior_change": "resolve the two readings",
        "change_class": S.SEMANTIC, "source_basis": "STORE item 400",
        "cases_addressed": ["CASE1"],
        "blast_radius": {"computable": False, "basis": "needs replay"},
        "requires_clinician_signoff": True,
    }
    row.update(over)
    return row


def test_semantic_proposal_is_cited_and_requires_signoff():
    text = SPEC.read_text(encoding="utf-8")
    got = S.SpecPatchProposal.from_dict(proposal(), spec_text=text)
    assert got.to_dict()["may_apply_automatically"] is False
    with pytest.raises(S.InvalidProposal, match="clinician"):
        S.SpecPatchProposal.from_dict(
            proposal(requires_clinician_signoff=False), spec_text=text)
    with pytest.raises(S.InvalidProposal, match="does not occur"):
        S.SpecPatchProposal.from_dict(
            proposal(quoted_current_text="this is not in the spec"), spec_text=text)


def test_retrieval_failure_cannot_edit_the_clinical_target():
    with pytest.raises(S.InvalidProposal, match="retrieval failure"):
        S.SpecPatchProposal.from_dict(proposal(failure_class=S.RETRIEVAL_FAILURE))
    with pytest.raises(S.InvalidProposal, match="asset change"):
        S.SpecPatchProposal.from_dict(proposal(
            failure_class=S.RETRIEVAL_FAILURE,
            parameter_id="keyword_retrieval_asset"))


def test_proposal_must_match_the_packet_and_asset_still_needs_certification():
    packet = S.ContrastiveFailurePacket(
        case_id="CASE1", spec_id=gold().spec_id, spec_hash="hash",
        disposition=S.RETRIEVAL_FAILURE, selected={}, rejected={},
        differences={}, gold={}, spec_sections={}, repair_permitted=True, why="test")
    asset = S.SpecPatchProposal.from_dict(proposal(
        failure_class=S.RETRIEVAL_FAILURE,
        parameter_id="keyword_retrieval_asset",
        change_class=S.ASSET,
        requires_clinician_signoff=False))
    assert S.validate_proposal_for_packet(asset, packet) is asset
    assert asset.to_dict()["may_apply_automatically"] is False
    assert asset.to_dict()["eligible_for_automatic_adoption_after_certification"] is True
    with pytest.raises(S.InvalidProposal, match="does not match"):
        S.validate_proposal_for_packet(
            S.SpecPatchProposal.from_dict(proposal(case_id="CASE2",
                                                   cases_addressed=["CASE2"])),
            packet)


def _dist(case_id, actual, expected, *, subgroup=("all",)):
    g = gold(case_id=case_id, site=expected, subgroup=subgroup)
    return S.cluster_behaviors([
        S.BehaviorSignature.from_manifest(
            manifest(actual, patient=case_id, run_id=case_id))
    ], g), g


def test_paired_validation_refuses_instance_regression_even_if_another_case_improves():
    b1, g1 = _dist("CASE1", "C341", "C341")
    a1, _ = _dist("CASE1", "C343", "C341")
    b2, g2 = _dist("CASE2", "C343", "C341")
    a2, _ = _dist("CASE2", "C341", "C341")
    report = S.paired_validate([b1, b2], [a1, a2], {"CASE1": g1, "CASE2": g2})
    assert report.mean_correct_delta == 0
    assert report.accepted is False
    assert report.regressions == ("CASE1",)
    assert report.metrics["before"]["field_exact_accuracy"] == 0.5
    assert report.metrics["after"]["field_exact_accuracy"] == 0.5
    assert report.metrics["after"]["found_precision"] == 0.5
    assert report.metrics["after"]["evidence_validity"] == 1.0


def test_paired_validation_requires_matching_model_budget_and_seed():
    before_row = manifest("C343", run_id="before")
    after_row = manifest("C341", run_id="after")
    before_row.update({
        "model": "model-a", "model_temperature": 1.0,
        "max_model_calls": 24, "sample_seed": 1234,
        "spend": {"max_usd": 5.0},
    })
    after_row.update({
        "model": "model-b", "model_temperature": 1.0,
        "max_model_calls": 24, "sample_seed": 1234,
        "spend": {"max_usd": 5.0},
    })
    g = gold()
    before = S.cluster_behaviors([S.BehaviorSignature.from_manifest(before_row)], g)
    after = S.cluster_behaviors([S.BehaviorSignature.from_manifest(after_row)], g)
    with pytest.raises(S.SpecRepairError, match="same model"):
        S.paired_validate([before], [after], {"CASE1": g})


def test_zero_error_sample_size_is_exact_and_close_to_rule_of_three():
    assert S.min_zero_error_n(0.05) == 59
    assert S.min_zero_error_n(0.02) == 149


def test_sealed_cohort_is_consumed_once():
    cert = S.SealedCertification("cohort", "bundle")
    used = cert.consume({"accepted": True})
    assert used.consumed and used.result_hash
    with pytest.raises(S.SealedSetReuse):
        used.consume({"accepted": True})


def test_gold_file_requires_explicit_schema_not_a_legacy_answer_key(tmp_path):
    p = tmp_path / "key.json"
    p.write_text(json.dumps({"CASE1": {"fields": {"primary_site": "C341"}}}))
    with pytest.raises(S.SpecRepairError, match="registry reference"):
        S.load_gold(p)
