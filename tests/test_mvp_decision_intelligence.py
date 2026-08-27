"""The Semantica adapter projects episodes, not runtime narration, and keeps analyses scoped."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from acr.contract.spec import load_spec
from acr.mvp.decision_receipts import make_runtime_decision_receipt
from acr.mvp.ledger import NullLedger, SemanticaLedger
from acr.mvp.semantica_audit import _basis_rows
from acr.mvp.task_presentation import build_task_presentation, content_hash

pytest.importorskip("semantica")


def _artifact(analysis_id="analysis-a", *, run_id="run-1", function="where_to_look",
              causal=True):
    cycle1 = {
        "cycle_id": f"{run_id}:cycle:2", "source_event_ids": ["layer1:2"],
        "source_seq_range": [2, 2], "source_event_time": "2026-01-01T00:00:00Z",
        "state_before": {"observed_state": {}, "declared_state": {}},
        "state_after": {"observed_state": {}, "declared_state": {}},
        "decision_testimony_refs": ["decision:2"], "structural_kind": "TOOL_INTERACTION",
        "actions": [{"tool": "note_decision", "args": {}}], "observations": [],
    }
    cycle2 = {
        "cycle_id": f"{run_id}:cycle:3", "source_event_ids": ["layer1:3"],
        "source_seq_range": [3, 3], "source_event_time": "2026-01-01T00:00:01Z",
        "state_before": {"observed_state": {}, "declared_state": {}},
        "state_after": {"observed_state": {}, "declared_state": {}},
        "decision_testimony_refs": [], "structural_kind": "SUBMISSION",
        "actions": [{"tool": "submit_answer", "args": {}}], "observations": [],
    }
    ep1 = {
        "episode_id": f"{analysis_id}:episode:1", "source_cycle_ids": [cycle1["cycle_id"]],
        "source_event_ids": ["layer1:2"], "decision_function": function,
        "decision_subject": ("retrieval_query_batch" if function == "where_to_look"
                             else "evidence_item"),
        "material_question": "Where should qualifying evidence be sought?",
        "scenario": "Patient SYN0001 note N1_2023-04-12 before decision",
        "candidate_set": ["candidate-a"], "decision": "choose 20230412",
        "decision_rationale": "The selected route can surface qualifying evidence.",
        "model_interpretation": "search route", "claimed_basis_summary": ["chart"],
        "verified_reference_summary": [], "state_delta": "candidate surfaced",
        "observed_downstream_refs": ["layer1:3"], "hypothesized_impact": [],
        "counterfactual_supported_impact": [],
        "field_provenance": {"decision": "SELF_REPORTED"},
        "source_refs_by_field": {}, "reconstruction_provenance": "MODEL_RECONSTRUCTED",
        "stability_status": "STABLE_ACROSS_PASSES", "reconstruction_stability": 0.92,
    }
    ep2 = {
        **ep1, "episode_id": f"{analysis_id}:episode:2",
        "source_cycle_ids": [cycle2["cycle_id"]], "source_event_ids": ["layer1:3"],
        "decision_function": "what_to_answer",
        "decision_subject": "answer_selection",
        "material_question": "What result should be submitted?",
        "scenario": "one candidate ready",
        "decision": "submit candidate", "observed_downstream_refs": [],
        "decision_rationale": "The candidate satisfies the answer contract.",
    }
    assertions = ([{
        "assertion_id": f"{analysis_id}:causal:1",
        "source_episode_id": ep1["episode_id"], "target_episode_id": ep2["episode_id"],
        "relationship_type": "INFLUENCED", "evidence_refs": ["decision:2"],
        "reasoning": "runtime testimony names the downstream choice",
        "provenance": "MODEL_RECONSTRUCTED",
    }] if causal else [])
    artifact = {
        "schema": "acr.decision_episode_analysis.v1", "analysis_id": analysis_id,
        "run_id": run_id, "trace_id": "t" * 32, "trace_manifest_hash": "manifest",
        "trace_content_hash": "tracehash", "task_presentation_hash": "presentation",
        "reconstructor_identity": "openrouter/openai/gpt-5.6-terra", "pass_index": 1,
        "reconstructor_call": {
            "requested_model": "openrouter/openai/gpt-5.6-terra",
            "resolved_model": "openai/gpt-5.6-terra-2026-08-20",
            "response_provider": "openrouter", "response_id": "gen-test",
            "identity_status": "RETURNED_BY_PROVIDER",
        },
        "cycles_hash": "cycles", "cycles": [cycle1, cycle2], "episodes": [ep1, ep2],
        "mechanical_cycle_ids": [], "cycle_annotations": {
            cycle1["cycle_id"]: {
                "role": "DECISION_BEARING", "decision_function": function},
            cycle2["cycle_id"]: {
                "role": "DECISION_BEARING", "decision_function": "what_to_answer"},
        },
        "causal_assertions": assertions, "stability_status": "STABLE_ACROSS_PASSES",
        "reconstruction_stability": 0.92,
    }
    artifact.pop("analysis_artifact_hash", None)
    artifact["analysis_artifact_hash"] = content_hash(artifact)
    return artifact


def _persist_artifact(tmp_path: Path, artifact: dict) -> Path:
    analysis_dir = tmp_path / artifact["run_id"] / "analyses"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / f"{artifact['analysis_id']}.json"
    artifact["artifact_ref"] = str(path)
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return analysis_dir.parent


def _write_task_presentation(run_dir: Path, artifact: dict, *, arm_id="detailed") -> dict:
    payload = {
        "schema": "acr.task_presentation.v1", "run_id": artifact["run_id"],
        "arm_id": arm_id,
        "task_contract_ref": {
            "id": "STORE.390.date_of_initial_diagnosis", "version": "0.1.0",
            "content_hash": "contract-hash",
        },
        "offered_clause_catalog": [{
            "rule_id": "decision_rule.1", "kind": "decision_rule",
            "text_sha": "rule-sha", "text": "Use the earliest qualifying date.",
            "rendered_locator": "task_contract:rule.1", "view_id": "rule.1",
            "enforced_path": None,
        }] if arm_id == "detailed" else [],
        "known_clause_index": [{
            "rule_id": "decision_rule.1", "kind": "decision_rule",
            "text_sha": "rule-sha",
        }],
        "method_card_refs": [], "operational_instruction_refs": [],
        "rendered_prompt_artifact_ref": "prompt.txt", "prompt_hash": "prompt-hash",
    }
    payload["presentation_hash"] = content_hash(payload)
    (run_dir / "task_presentation.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    artifact["task_presentation_hash"] = payload["presentation_hash"]
    artifact["analysis_artifact_hash"] = content_hash({
        key: value for key, value in artifact.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    path = Path(artifact["artifact_ref"])
    path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    return payload


def test_null_ledger_counts_only_episodes_as_decisions():
    ledger = NullLedger()
    ledger.project_analysis(_artifact())
    assert ledger.stats() == {"analyses": 1, "episodes": 2, "causal_edges": 1}


def test_readable_basis_merges_checked_facts_and_renders_structured_rules():
    rows = _basis_rows({
        "basis_sources": ["chart"],
        "guidelines": [
            {"rule_id": "conflict_rule.1", "text": json.dumps({
                "if": "an ambiguous cytology predates biopsy",
                "then": "use it only with a supporting clinical impression",
                "turns_on": ["impression_at_ambiguous_cytology"],
            })},
            {"rule_id": "discriminating_fact.impression_at_ambiguous_cytology",
             "text": "whether a clinical impression exists on that date"},
        ],
        "dependencies": {
            "citation_resolutions": [{
                "ref": "conflict_rule.1", "status": "CLAIMED_AND_VERIFIED"}],
            "checked_discriminating_facts": [{
                "ref": "discriminating_fact.impression_at_ambiguous_cytology",
                "status": "CLAIMED_AND_VERIFIED"}],
        },
    })

    facts = [row for row in rows if row["kind"] == "discriminating_fact"]
    conflict = next(row for row in rows if row.get("reference") == "conflict_rule.1")
    assert len(facts) == 1
    assert facts[0]["label"] == "whether a clinical impression exists on that date"
    assert facts[0]["reference_status"] == "CLAIMED_AND_VERIFIED"
    assert conflict["label"] == (
        "If an ambiguous cytology predates biopsy; then use it only with a supporting "
        "clinical impression; check: impression at ambiguous cytology."
    )


def test_projection_uses_the_analytics_profile_and_keeps_nondecisions_ordinary(tmp_path: Path):
    ledger = SemanticaLedger(tmp_path / "ledger.json")
    ledger.project_analysis(_artifact())
    assert ledger.graph.config == {
        "extract_entities": False, "extract_relationships": False,
        "advanced_analytics": True, "centrality_analysis": True,
        "community_detection": True, "node_embeddings": False,
    }
    decisions = ledger.graph.find_nodes(node_type="decision")
    assert len(decisions) == 2
    assert {row["metadata"]["confidence_semantics"] for row in decisions} == {
        "RECONSTRUCTION_STABILITY"}
    assert {row["metadata"]["confidence"] for row in decisions} == {0.92}
    first = next(row for row in decisions
                 if row["metadata"]["acr_episode_id"].endswith(":episode:1"))
    assert first["metadata"]["projection_schema"] == "acr.semantica_projection.v3"
    assert first["metadata"]["scenario"] == "Where should qualifying evidence be sought?"
    assert first["metadata"]["reasoning"] == (
        "The selected route can surface qualifying evidence. Basis used: chart evidence."
    )
    assert first["metadata"]["situation_signature"]["candidate_shape"] == "single"
    assert first["metadata"]["atomic_decision_identity"] == {
        "schema": "acr.atomic_decision_identity.v1",
        "identity_basis": "EXACT_QUESTION_AND_EVIDENCE",
        "decision_function": "where_to_look",
        "decision_subject": "retrieval_query_batch",
        "question_hash": first["metadata"]["atomic_decision_identity"]["question_hash"],
        "evidence_hash": first["metadata"]["atomic_decision_identity"]["evidence_hash"],
        "evidence_anchor_count": 1,
        "decision_point_hash": first["metadata"]["decision_point_hash"],
    }
    assert ledger.graph.find_nodes(node_type="ReActCycle")
    assert ledger.graph.find_nodes(node_type="DecisionTestimony")
    assert ledger.graph.find_nodes(node_type="Submission")
    assert ledger.graph.find_nodes(node_type="GateVerdict")
    analysis = ledger.graph.find_nodes(node_type="AnalysisArtifact")[0]
    assert analysis["metadata"]["reconstructor_resolved_model"] \
        == "openai/gpt-5.6-terra-2026-08-20"
    assert analysis["metadata"]["reconstructor_identity_status"] == "RETURNED_BY_PROVIDER"


def test_runtime_receipt_is_provenance_for_a_decision_not_an_extra_semantica_decision(
        tmp_path: Path):
    artifact = _artifact()
    cycle = artifact["cycles"][0]
    args = {
        "facing": "Where should qualifying evidence be sought?",
        "decision": "Search pathology", "because": "It can establish the field.",
        "alternatives": ["Search clinic notes"], "basis_sources": ["chart"],
        "cited_refs": [], "checked_discriminating_fact_refs": [],
        "rule_coverage_claim": "OPERATIONAL_DISCRETION",
        "provisional_inference": None, "uncertainty": None,
    }
    result = {"noted": True, "testimony_ref": "decision:2",
              "citation_resolutions": [], "checked_fact_resolutions": []}
    action = {"event_ref": "layer1:2", "tool": "note_decision", "args": args, "ok": True}
    receipt = make_runtime_decision_receipt(
        action["tool"], args, result, cycle["state_before"],
        source_event_ref=action["event_ref"])
    assert receipt is not None
    result["decision_receipt"] = receipt
    cycle["actions"] = [action]
    cycle["observations"] = [{"event_ref": "layer1:2", "result": result}]
    cycle["decision_receipt_refs"] = [receipt["receipt_id"]]
    cycle["has_decision_receipt"] = True
    artifact["episodes"][0].update({
        "runtime_testimony_ref": receipt["testimony_ref"],
        "runtime_receipt_ref": receipt["receipt_id"],
        "runtime_receipt_schema": receipt["schema"],
        "runtime_receipt_hash": receipt["receipt_hash"],
        "runtime_receipt_provenance": receipt["provenance"],
    })
    artifact["taxonomy_version"] = "acr.chart_review_decision_taxonomy.v1"
    artifact["supported_runtime_receipt_schema"] = receipt["schema"]
    artifact["runtime_receipt_manifest"] = {
        "mode": "MIXED", "sealed_receipt_count": 1, "witnessed_decision_count": 1,
        "receipt_schemas": [receipt["schema"]], "receipt_hashes": [receipt["receipt_hash"]],
        "provenance": "DETERMINISTIC_DERIVED",
    }
    artifact.pop("analysis_artifact_hash", None)
    artifact["analysis_artifact_hash"] = content_hash(artifact)

    ledger = SemanticaLedger(tmp_path / "ledger.json")
    ledger.project_analysis(artifact)

    assert len(ledger.graph.find_nodes(node_type="decision")) == 2
    assert ledger.graph.find_nodes(node_type="RuntimeDecisionReceipt") == []
    testimony = ledger.graph.find_nodes(node_type="DecisionTestimony")[0]["metadata"]
    assert testimony["receipt_ref"] == receipt["receipt_id"]
    assert testimony["receipt_hash"] == receipt["receipt_hash"]
    projected = next(row for row in ledger.graph.find_nodes(node_type="decision")
                     if row["metadata"]["acr_episode_id"].endswith(":episode:1"))
    assert projected["metadata"]["runtime_receipt_ref"] == receipt["receipt_id"]
    assert projected["metadata"]["taxonomy_version"] == artifact["taxonomy_version"]


def test_projection_records_a_complete_auditable_decision_episode_in_semantica(
        tmp_path: Path):
    artifact = _artifact()
    cycle = artifact["cycles"][0]
    cycle["state_before"] = {
        "observed_state": {"surfaced_notes": ["NOTE-A"], "read_notes": []},
        "declared_state": {"findings": [], "uncertainties": ["earliest date unknown"]},
    }
    cycle["state_after"] = {
        "observed_state": {"surfaced_notes": ["NOTE-A"], "read_notes": ["NOTE-A"]},
        "declared_state": {"findings": [{
            "finding_ref": "finding:1", "note_id": "NOTE-A",
            "field": "diagnosis_date", "standing": "can_establish",
            "assertion_class": "pathology_diagnosis",
        }], "uncertainties": []},
    }
    cycle["declared_open_question"] = "Can this pathology establish the field?"
    cycle["actions"] = [{
        "event_ref": "layer1:2", "tool": "note_decision", "ok": True,
        "args": {
            "facing": "Can this pathology establish the field?",
            "decision": "Treat it as establishing evidence",
            "because": "The supplied evidence rule covers a definitive pathology report",
            "basis_sources": ["task_contract", "chart"],
            "cited_refs": ["evidence_rule.counts_as_evidence.1"],
            "checked_discriminating_fact_refs": [],
            "rule_coverage_claim": "COVERED_EXACTLY",
            "provisional_inference": None,
            "alternatives": ["merely mentions"],
            "uncertainty": None,
        },
    }]
    cycle["observations"] = [{
        "event_ref": "layer1:2", "result": {
            "testimony_ref": "decision:2",
            "citation_resolutions": [{
                "ref": "evidence_rule.counts_as_evidence.1",
                "status": "CLAIMED_AND_VERIFIED",
            }],
            "checked_fact_resolutions": [],
        },
    }]
    episode = artifact["episodes"][0]
    episode.update({
        "candidate_set": ["can_establish", "merely_mentions"],
        "decision": "can_establish",
        "model_interpretation": "The pathology was judged answer-bearing.",
        "claimed_basis_summary": ["task_contract", "chart"],
        "verified_reference_summary": ["evidence rule verified"],
        "state_delta": "One answer-bearing finding was added.",
        "field_provenance": {
            "scenario": "MODEL_RECONSTRUCTED", "decision": "SELF_REPORTED",
        },
        "source_refs_by_field": {
            "decision": ["layer1:2"], "model_interpretation": ["layer1:2"],
        },
    })
    artifact["analysis_artifact_hash"] = content_hash({
        key: value for key, value in artifact.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    _persist_artifact(tmp_path, artifact)

    ledger = SemanticaLedger(tmp_path / "ledger.json")
    ledger.project_analysis(artifact)
    manager = ledger.provenance_manager("run-1", "analysis-a")
    rows = manager.audit_log(format="json")

    entity_types = {row["entity_type"] for row in rows}
    assert {
        "analysis_artifact", "state_snapshot", "agent_action", "tool_observation",
        "decision_testimony", "react_cycle", "decision", "decision_narrative",
        "relationship",
    } <= entity_types
    decision = next(row for row in rows if row["entity_type"] == "decision"
                    and row["metadata"].get("acr_episode_id") == episode["episode_id"])
    assert decision["metadata"]["episode"] == episode
    assert decision["metadata"]["component_map"] == {
        "time_boundary": "SERVER_FACT_AND_DETERMINISTIC_BOUNDARY",
        "state_before": "OBSERVED_AND_DECLARED",
        "choice_state_before": "OBSERVED_AND_DECLARED",
        "material_question": "SELF_REPORTED",
        "available_basis": "OBSERVED_AND_SELF_REPORTED",
        "alternatives_and_choice": "SELF_REPORTED_AND_MODEL_RECONSTRUCTED",
        "decision_rationale": "SELF_REPORTED",
        "actions_and_observations": "SERVER_FACT",
        "state_after": "OBSERVED_AND_DECLARED",
        "continue_or_stop": "DETERMINISTIC_DERIVED",
        "human_correctness": "NOT_YET_ADJUDICATED",
    }
    atomic = decision["metadata"]["atomic_decision"]
    assert atomic["schema"] == "acr.atomic_decision.v1"
    assert atomic["audit_unit"] == "ATOMIC_DECISION"
    assert atomic["decision_cardinality"] == 1
    assert atomic["decision_subject"] == "retrieval_query_batch"
    assert atomic["state"]["immediately_before_choice"] == cycle["state_before"]
    assert atomic["material_question"] == {
        "value": "Can this pathology establish the field?",
        "provenance": "SELF_REPORTED",
        "source_refs": ["decision:2"],
    }
    assert atomic["basis"]["used_own_knowledge"] is False
    assert atomic["transition"]["disposition"] == "CONTINUE"
    assert atomic["human_correctness"] == "NOT_YET_ADJUDICATED"
    assert atomic["human_correctness"] == "NOT_YET_ADJUDICATED"
    narrative = next(row for row in rows if row["entity_type"] == "decision_narrative"
                     and not row["metadata"].get("is_conclusion"))
    assert narrative["metadata"]["question"] == \
        "Can this pathology establish the field?"
    assert narrative["metadata"]["decision"] == "Treat it as establishing evidence"
    assert narrative["metadata"]["basis"][0]["kind"] == "task_contract"
    assert "evidence rule verified" not in narrative["metadata"]["observations"]
    assert narrative["metadata"]["domain_projection"]["dependencies"][
        "verified_reference_summary"] == ["evidence rule verified"]
    assert manager.check(strict=True)["valid"] is True
    assert manager.verify_chain()["valid"] is True


def test_repeated_basis_reference_is_recorded_per_testimony_occurrence(tmp_path: Path):
    artifact = _artifact(causal=False)
    for index, cycle in enumerate(artifact["cycles"], 2):
        event_ref = f"layer1:{index}"
        cycle["structural_kind"] = "TOOL_INTERACTION"
        cycle["decision_testimony_refs"] = [f"decision:{index}"]
        cycle["actions"] = [{
            "event_ref": event_ref, "tool": "note_decision", "ok": True,
            "args": {
                "facing": "Does the same supplied rule support this choice?",
                "decision": f"choice-{index}", "because": "Apply the offered rule",
                "basis_sources": ["task_contract"], "cited_refs": ["decision_rule.1"],
                "checked_discriminating_fact_refs": [],
                "rule_coverage_claim": "DIRECTLY_COVERED", "alternatives": [],
            },
        }]
        cycle["observations"] = [{
            "event_ref": event_ref, "result": {
                "testimony_ref": f"decision:{index}",
                "citation_resolutions": [{
                    "ref": "decision_rule.1", "status": "CLAIMED_AND_VERIFIED"}],
                "checked_fact_resolutions": [],
            },
        }]
    artifact["analysis_artifact_hash"] = content_hash({
        key: value for key, value in artifact.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    _persist_artifact(tmp_path, artifact)
    ledger = SemanticaLedger(tmp_path / "ledger.json")

    ledger.project_analysis(artifact)

    references = [
        row for row in ledger.provenance_manager("run-1", "analysis-a").audit_log(
            format="json")
        if row["entity_type"] == "decision_basis_reference"
        and row["metadata"]["reference"]["ref"] == "decision_rule.1"
    ]
    assert len(references) == 2
    assert len({row["entity_id"] for row in references}) == 2
    assert {row["metadata"]["testimony_ref"] for row in references} == {
        "decision:2", "decision:3"}


def test_atomic_finding_projects_as_testimony_with_a_human_scale_evidence_packet(
        tmp_path: Path):
    artifact = _artifact(causal=False, function="standing")
    cycle = artifact["cycles"][0]
    note_ids = [f"NOTE_{index:03d}" for index in range(200)]
    cycle["state_before"] = {
        "observed_state": {"surfaced_notes": note_ids, "read_notes": ["NOTE_007"]},
        "declared_state": {"findings": [], "uncertainties": []},
    }
    cycle["state_after"] = {
        "observed_state": {"surfaced_notes": note_ids, "read_notes": ["NOTE_007"]},
        "declared_state": {"findings": [{
            "finding_ref": "finding:1", "note_id": "NOTE_007",
            "field": "diagnosis_date", "event_time": "2023-04-12",
            "standing": "can_establish", "assertion_class": "clinical_diagnosis",
            "quote": "Assessment: newly diagnosed metastatic lung cancer.",
        }], "uncertainties": []},
    }
    cycle["decision_testimony_refs"] = ["decision:2"]
    cycle["actions"] = [{
        "event_ref": "layer1:2", "tool": "record_finding", "ok": True,
        "args": {
            "note_id": "NOTE_007", "field": "diagnosis_date",
            "standing": "can_establish", "assertion_class": "clinical_diagnosis",
            "event_time": "2023-04-12", "source_start": 11, "source_end": 65,
            "facing": "Can this clinical note establish the diagnosis date?",
            "because": "The assessment explicitly states a new diagnosis.",
            "basis_sources": ["chart"], "cited_refs": ["note:NOTE_007"],
            "checked_discriminating_fact_refs": [],
            "rule_coverage_claim": "COVERED_WITH_INTERPRETATION",
            "alternatives": ["merely_mentions"], "uncertainty": None,
        },
    }]
    cycle["observations"] = [{
        "event_ref": "layer1:2", "result": {
            "recorded": True, "noted": True, "finding_ref": "finding:1",
            "testimony_ref": "decision:2",
            "server_fact": {"span": [11, 65], "span_resolved": True},
            "quote": "Assessment: newly diagnosed metastatic lung cancer.",
            "self_reported": {
                "decision": "NOTE_007 is can_establish for diagnosis_date "
                            "(clinical_diagnosis)"},
            "citation_resolutions": [{"ref": "note:NOTE_007", "verified": True,
                                      "depth": "read"}],
            "checked_fact_resolutions": [],
        },
    }]
    episode = artifact["episodes"][0]
    episode.update({
        "decision_subject": "evidence_item",
        "material_question": "Can this clinical note establish the diagnosis date?",
        "decision": "NOTE_007 can establish the diagnosis date",
        "candidate_set": ["can_establish", "merely_mentions", "neither"],
        "state_delta": "One clinical finding was recorded.",
    })
    artifact["analysis_artifact_hash"] = content_hash({
        key: value for key, value in artifact.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    _persist_artifact(tmp_path, artifact)
    ledger = SemanticaLedger(tmp_path / "ledger.json")

    ledger.project_analysis(artifact)

    rows = ledger.provenance_manager("run-1", "analysis-a").audit_log(format="json")
    testimonies = [row for row in rows if row["entity_type"] == "decision_testimony"]
    assert testimonies[0]["metadata"]["testimony"]["tool"] == "record_finding"
    narrative = next(row for row in rows if row["entity_type"] == "decision_narrative"
                     and not row["metadata"].get("is_conclusion"))
    metadata = narrative["metadata"]
    assert len(metadata["known_before"]) <= 8
    assert "NOTE_199" not in str(metadata["known_before"])
    assert not any("Surfaced note sample" in line for line in metadata["known_before"])
    assert "NOTE_007" not in str(metadata["actions"])
    assert metadata["actions"][0]["name"] == "Judge NOTE 007"
    assert "establishes the requested field" in metadata["actions"][0]["observation"]
    assert any("newly diagnosed metastatic lung cancer" in line
               for line in metadata["observations"])
    assert "newly diagnosed metastatic lung cancer" in metadata["actions"][0]["observation"]


def test_observed_submission_is_a_conclusion_without_fabricating_a_decision(tmp_path: Path):
    artifact = _artifact(causal=False)
    submission_cycle = artifact["cycles"][1]
    submission_cycle["actions"][0]["args"] = {
        "status": "FOUND", "value": {"diagnosis_date": "20230412"},
        "reasoning": "The accepted evidence establishes the submitted date",
    }
    submission_cycle["observations"] = [{
        "event_ref": "layer1:3", "result": {"accepted": True},
    }]
    submission_cycle["actions"][0]["event_ref"] = "layer1:3"
    artifact["episodes"] = artifact["episodes"][:1]
    artifact["cycle_annotations"][submission_cycle["cycle_id"]] = {
        "role": "MECHANICAL", "decision_function": None}
    artifact["mechanical_cycle_ids"] = [submission_cycle["cycle_id"]]
    artifact["analysis_artifact_hash"] = content_hash({
        key: value for key, value in artifact.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    _persist_artifact(tmp_path, artifact)
    ledger = SemanticaLedger(tmp_path / "ledger.json")

    ledger.project_analysis(artifact)

    narratives = [
        row for row in ledger.provenance_manager("run-1", "analysis-a").audit_log(
            format="json")
        if row["entity_type"] == "decision_narrative"
    ]
    assert len(ledger.graph.find_nodes(node_type="decision")) == 1
    assert len(narratives) == 2
    conclusion = next(row for row in narratives if row["metadata"]["is_conclusion"])
    assert conclusion["metadata"]["audit_unit"] == "RUN_OUTCOME"
    assert conclusion["metadata"]["decision_ids"] == []
    assert conclusion["metadata"]["decision"] == {"diagnosis_date": "20230412"}
    assert conclusion["metadata"]["outcome_event_ref"] == "layer1:3"
    assert conclusion["metadata"]["known_before"] == []
    assert conclusion["metadata"]["known_before_boundary"] == \
        "NOT_APPLICABLE_RUN_OUTCOME"


def test_projection_rejects_a_compound_episode_before_writing_semantica(tmp_path: Path):
    artifact = _artifact(causal=False)
    first, second = artifact["cycles"]
    artifact["episodes"][0]["source_cycle_ids"] = [
        first["cycle_id"], second["cycle_id"]]
    artifact["episodes"] = artifact["episodes"][:1]
    artifact["cycle_annotations"][second["cycle_id"]] = {
        "role": "DECISION_BEARING", "decision_function": "where_to_look"}
    artifact["analysis_artifact_hash"] = content_hash({
        key: value for key, value in artifact.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    ledger = SemanticaLedger(tmp_path / "ledger.json")

    with pytest.raises(ValueError, match="exactly one atomic decision"):
        ledger.project_analysis(artifact)

    assert ledger.graph.find_nodes(node_type="decision") == []


def test_projection_hashes_long_source_cycle_lists_in_decision_metadata(tmp_path: Path):
    artifact = _artifact(causal=False)
    first = artifact["cycles"][0]
    support_cycles = []
    for index in range(12):
        cycle = json.loads(json.dumps(first))
        cycle["cycle_id"] = f"run-1:cycle:support-{index}-" + ("x" * 72)
        cycle["source_event_ids"] = [f"layer1:{10 + index}"]
        cycle["source_seq_range"] = [10 + index, 10 + index]
        cycle["actions"] = []
        cycle["decision_testimony_refs"] = []
        support_cycles.append(cycle)
        artifact["cycle_annotations"][cycle["cycle_id"]] = {
            "role": "DECISION_SUPPORT", "decision_function": None}
    artifact["cycles"] = [first, *support_cycles]
    artifact["episodes"] = artifact["episodes"][:1]
    artifact["episodes"][0]["source_cycle_ids"] = [
        first["cycle_id"], *(row["cycle_id"] for row in support_cycles)]
    artifact["analysis_artifact_hash"] = content_hash({
        key: value for key, value in artifact.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    _persist_artifact(tmp_path, artifact)
    ledger = SemanticaLedger(tmp_path / "ledger.json")

    ledger.project_analysis(artifact)

    metadata = ledger.graph.find_nodes(node_type="decision")[0]["metadata"]
    assert "source_cycle_ids" not in metadata
    assert metadata["source_cycle_count"] == 13
    assert len(metadata["source_cycle_ids_hash"]) == 64


def test_projection_rejects_a_recorded_finding_hidden_in_a_support_cycle(tmp_path: Path):
    artifact = _artifact(causal=False)
    cycle = artifact["cycles"][0]
    cycle["actions"] = [{
        "event_ref": "layer1:2", "tool": "record_finding", "ok": True,
        "args": {"field": "diagnosis_date", "standing": "can_establish"},
    }]
    artifact["cycle_annotations"][cycle["cycle_id"]] = {
        "role": "DECISION_SUPPORT", "decision_function": None}
    artifact["analysis_artifact_hash"] = content_hash({
        key: value for key, value in artifact.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    ledger = SemanticaLedger(tmp_path / "ledger.json")

    with pytest.raises(ValueError, match="separate standing decision"):
        ledger.project_analysis(artifact)

    assert ledger.graph.find_nodes(node_type="decision") == []


def test_core_projection_is_sanitized_and_causal_edges_have_assertion_provenance(tmp_path: Path):
    artifact = _artifact()
    artifact["episodes"][0]["material_question"] = (
        "Can patient SYN0001's NOTE_007 from 2023-04-12 establish the diagnosis date?"
    )
    artifact["episodes"][0]["decision_rationale"] = (
        "NOTE_007 on 2023-04-12 explicitly documents the new diagnosis for SYN0001."
    )
    artifact["analysis_artifact_hash"] = content_hash({
        key: value for key, value in artifact.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    ledger = SemanticaLedger(tmp_path / "ledger.json")
    ledger.project_analysis(artifact)
    assert ledger.validate_export() == []
    first = next(row for row in ledger.graph.find_nodes(node_type="decision")
                 if row["metadata"]["acr_episode_id"].endswith(":episode:1"))
    assert first["metadata"]["scenario"] == (
        "Can patient [patient]'s [note] from [date] establish the diagnosis date?"
    )
    assert first["metadata"]["reasoning"] == (
        "[note] on [date] explicitly documents the new diagnosis for [patient]. "
        "Basis used: chart evidence."
    )
    dump = " ".join(str(first["metadata"].get(key, ""))
                    for key in ("scenario", "reasoning", "outcome"))
    assert "SYN0001" not in dump and "20230412" not in dump and "N1_2023" not in dump
    assert "NOTE_007" not in dump and "2023-04-12" not in dump

    edges = [edge.to_dict() for edge in ledger.graph.edges]
    assert len([edge for edge in edges if edge.get("type") == "INFLUENCED"]) == 1
    assertions = ledger.graph.find_nodes(node_type="CausalAssertion")
    assert len(assertions) == 1
    assert assertions[0]["metadata"]["evidence_count"] == 1


def test_temporal_adjacency_alone_never_creates_a_causal_edge(tmp_path: Path):
    ledger = SemanticaLedger(tmp_path / "ledger.json")
    ledger.project_analysis(_artifact(causal=False))
    assert not [edge for edge in ledger.graph.edges
                if edge.to_dict().get("type") in {"CAUSED", "INFLUENCED", "PRECEDENT_FOR"}]
    suggested = ledger.chain("run-1", "analysis-a")
    assert suggested["causal_edges"] == []
    assert suggested["suggested_links"][0]["authority"] == "SUGGESTED_ONLY"
    assert suggested["suggested_links"][0]["semantica_relationship"] == "influences"


def test_projection_is_idempotent_and_queries_survive_save_reload(tmp_path: Path):
    path = tmp_path / "ledger.json"
    artifact = _artifact()
    ledger = SemanticaLedger(path)
    first_hash = ledger.project_analysis(artifact)
    manager = ledger.provenance_manager("run-1", "analysis-a")
    provenance_ids_before = {
        row["entity_id"] for row in manager.audit_log(format="json")}
    second_hash = ledger.project_analysis(artifact)
    assert second_hash == first_hash
    provenance_ids_after = {
        row["entity_id"] for row in manager.audit_log(format="json")}
    assert provenance_ids_after == provenance_ids_before
    assert not any(":v:" in entity_id for entity_id in provenance_ids_after)
    ledger.save()

    reloaded = SemanticaLedger(path)
    assert reloaded.projection_hash("run-1", "analysis-a") == first_hash
    episode_id = artifact["episodes"][0]["episode_id"]
    assert reloaded.similar_candidates(episode_id)["authority"] == "CANDIDATE_ONLY"
    assert reloaded.impact_candidates(episode_id)["authority"] == "CANDIDATE_ONLY"
    insights = reloaded.insights("run-1", "analysis-a")
    json.dumps(insights)
    assert insights["analysis_id"] == "analysis-a"
    assert insights["semantica_scoped_insights"]["total_decisions"] == 2
    assert insights["semantica_scoped_insights"]["confidence_stats"]["semantics"] \
        == "RECONSTRUCTION_STABILITY"


def test_semantica_similarity_then_acr_cross_run_identity_guards(tmp_path: Path):
    ledger = SemanticaLedger(tmp_path / "ledger.json")
    first = _artifact("analysis-a", run_id="run-1", causal=False)
    second = _artifact("analysis-b", run_id="run-2", causal=False)
    second["episodes"][0]["scenario"] = \
        "Different patient SYN9999 and note OTHER_2025-11-03"
    second["episodes"][0]["candidate_set"] = ["20251103"]
    second["episodes"][0]["decision"] = "choose 20251103"
    second["analysis_artifact_hash"] = content_hash({
        key: value for key, value in second.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    ledger.project_analysis(first)
    ledger.project_analysis(second)

    decisions = ledger.graph.find_nodes(node_type="decision")
    query = next(node for node in decisions
                 if (node.get("metadata") or {}).get("run_id") == "run-1")
    query_meta = query["metadata"]
    native = ledger.graph.find_similar_decisions(
        query_meta["scenario"], category=query_meta["category"],
        max_results=len(decisions), min_similarity=0.0,
    )
    assert query["id"] in {row["decision"]["id"] for row in native}

    result = ledger.similar_candidates(first["episodes"][0]["episode_id"], max_results=5)

    assert result["retrieval_engine"] == "semantica.ContextGraph.find_similar_decisions"
    assert [row["decision"]["metadata"]["run_id"] for row in result["candidates"]] \
        == ["run-2"]
    candidate = result["candidates"][0]
    assert candidate["similarity"] >= 0.3
    assert candidate["same_situation_signature"] is True
    assert candidate["same_decision_point"] is False
    assert candidate["decision"]["metadata"]["situation_signature_hash"] == \
        result["query"]["situation_signature_hash"]
    assert candidate["decision"]["metadata"]["decision_point_hash"] != \
        result["query"]["decision_point_hash"]
    core = " ".join(str(candidate["decision"].get(key) or "")
                    for key in ("scenario", "reasoning", "outcome"))
    assert "SYN9999" not in core and "20251103" not in core


def test_causal_trace_and_impact_delegate_to_semantica_native_analytics(tmp_path: Path):
    ledger = SemanticaLedger(tmp_path / "ledger.json")
    artifact = _artifact()
    ledger.project_analysis(artifact)
    source_episode = artifact["episodes"][0]["episode_id"]
    target_episode = artifact["episodes"][1]["episode_id"]

    trace = ledger.causal_trace(target_episode)
    impact = ledger.impact_candidates(source_episode)

    assert trace["retrieval_engine"] == "semantica.ContextGraph.trace_decision_chain"
    assert any(hop["type"] == "INFLUENCED"
               for chain in trace["chains"] for hop in chain.get("hops") or [])
    assert len(trace["chains"]) == 1
    assert all(hop["type"] in {"CAUSED", "INFLUENCED", "PRECEDENT_FOR"}
               for chain in trace["chains"] for hop in chain.get("hops") or [])
    assert all(hop.get("assertion_id") and hop.get("assertion_provenance")
               for chain in trace["chains"] for hop in chain.get("hops") or [])
    assert trace["excluded_from_audit_chain"] == ["influences", "temporal adjacency"]
    assert trace["truncated"] is False
    assert impact["retrieval_engine"] == \
        "semantica.ContextGraph.analyze_decision_impact"
    assert target_episode in {
        row["decision"]["metadata"]["acr_episode_id"]
        for row in impact["candidates"]["direct_influence"]
    }


def test_causal_trace_excludes_native_heuristics_when_no_assertion_exists(tmp_path: Path):
    ledger = SemanticaLedger(tmp_path / "ledger.json")
    artifact = _artifact(causal=False)
    ledger.project_analysis(artifact)

    trace = ledger.causal_trace(artifact["episodes"][1]["episode_id"])

    assert trace["chains"] == []
    assert trace["authority"] == "EXPLICIT_CAUSAL_ASSERTIONS"


def test_task_contract_is_a_semantica_policy_with_native_change_impact(tmp_path: Path):
    artifact = _artifact(causal=False)
    run_dir = _persist_artifact(tmp_path, artifact)
    presentation = _write_task_presentation(run_dir, artifact)
    ledger = SemanticaLedger(tmp_path / "ledger.json")

    ledger.project_analysis(artifact)
    report = ledger.affected_by_policy_change(
        "STORE.390.date_of_initial_diagnosis", from_version="0.1.0",
        to_version="0.2.0")

    policies = ledger.graph.find_nodes(node_type="Policy")
    assert len(policies) == 1
    assert policies[0]["metadata"]["metadata"]["automatic_compliance_supported"] is False
    applications = ledger.graph.find_edges(edge_type="APPLIED_POLICY")
    assert len(applications) == len(artifact["episodes"])
    assert {row["metadata"]["application_semantics"] for row in applications} == {
        "AUDIT_GOVERNANCE_NOT_AGENT_CLAIM"}
    assert {row["metadata"]["task_presentation_hash"] for row in applications} == {
        presentation["presentation_hash"]}
    assert report["retrieval_engine"] == "semantica.PolicyEngine.get_affected_decisions"
    assert {row["decision_id"] for row in report["affected_decisions"]} == {
        row["id"] for row in ledger.graph.find_nodes(node_type="decision")}


def test_chain_requires_explicit_append_only_selection_and_never_mixes_analyses(tmp_path: Path):
    ledger = SemanticaLedger(tmp_path / "ledger.json")
    ledger.project_analysis(_artifact("analysis-a"))
    ledger.project_analysis(_artifact("analysis-b", function="standing", causal=False))
    unselected = ledger.chain("run-1")
    assert unselected["status"] == "NO_ANALYSIS_SELECTION"
    assert set(unselected["available_analysis_ids"]) == {"analysis-a", "analysis-b"}

    ledger.select_analysis("run-1", "analysis-a", selected_by="reviewer-1",
                           reason="pass alignment adjudicated")
    chain = ledger.chain("run-1")
    assert chain["analysis_id"] == "analysis-a"
    assert {row["analysis_id"] for row in chain["episodes"]} == {"analysis-a"}
    assert all("analysis-b" not in str(row) for row in chain["causal_edges"])


def test_generic_semantica_rules_cannot_masquerade_as_task_contract_compliance(tmp_path: Path):
    ledger = SemanticaLedger(tmp_path / "ledger.json")
    with pytest.raises(ValueError, match="explicit ACR mechanical policy"):
        ledger.check_mechanical_policy({"category": "standing"}, rules=None)


def test_task_only_projection_does_not_invent_a_clinical_policy_binding(tmp_path: Path):
    artifact = _artifact("analysis-task-only", run_id="run-task-only",
                         function="standing", causal=False)
    artifact["cycles"][0]["actions"][0]["args"] = {
        "basis_sources": ["chart", "own_knowledge"],
        "cited_refs": [],
        "checked_discriminating_fact_refs": [],
        "rule_coverage_claim": "NO_APPLICABLE_RULE",
    }
    run_dir = _persist_artifact(tmp_path, artifact)
    spec = load_spec(Path(__file__).resolve().parents[1] / "assets" / "specs" /
                     "STORE.390.date_of_initial_diagnosis.yaml")
    _, presentation = build_task_presentation(
        spec, run_id=artifact["run_id"], arm_id="task_only",
        operational_preamble="Do the review through chart tools.")
    presentation.write(run_dir)
    artifact["task_presentation_hash"] = presentation.presentation_hash
    artifact["task_arm"] = "task_only"
    artifact["analysis_artifact_hash"] = content_hash({
        key: value for key, value in artifact.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    Path(artifact["artifact_ref"]).write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    ledger = SemanticaLedger(tmp_path / "ledger.json")
    ledger.project_analysis(artifact)

    assert ledger.graph.find_nodes(node_type="Policy") == []
    assert ledger.graph.find_nodes(node_type="PolicyBundle") == []
    assert ledger.graph.find_edges(edge_type="APPLIED_POLICY") == []
    assert ledger.chain(artifact["run_id"], artifact["analysis_id"])["episodes"][0][
        "policy_groundings"] == []


def test_semantica_similarity_finds_cross_run_ungrounded_outcome_divergence(tmp_path: Path):
    cohort = []
    ledger = SemanticaLedger(tmp_path / "ledger.json")
    for model, run_id, analysis_id, decision, facing, span in (
            ("openai/gpt-5.6-luna", "run-luna", "analysis-luna", "can_establish",
             "Does the physician assessment establish the requested diagnosis date?", (358, 523)),
            ("openai/gpt-5.6-terra", "run-terra", "analysis-terra", "merely_mentions",
             "This note was read; what standing does it have for this field?", (264, 432))):
        artifact = _artifact(analysis_id, run_id=run_id, function="standing", causal=False)
        artifact["task_arm"] = "task_only"
        artifact["review_model"] = model
        artifact["episodes"][0]["candidate_set"] = ["can_establish", "merely_mentions"]
        artifact["episodes"][0]["decision"] = decision
        artifact["cycles"][0]["actions"][0] = {
            "event_ref": "layer1:2", "tool": "record_finding", "ok": True,
            "args": {
            "note_id": "Onc-Med-MD-OP-Progress-Note_2023-04-12",
            "field": "date_of_initial_diagnosis", "source_start": span[0],
            "source_end": span[1], "assertion_class": (
                "clinical_diagnosis" if decision == "can_establish"
                else "suspected_or_clinical_malignancy"),
            "standing": decision, "facing": facing,
            "basis_sources": ["chart", "own_knowledge"],
            "cited_refs": [], "checked_discriminating_fact_refs": [],
            "rule_coverage_claim": "NO_APPLICABLE_RULE",
            }}
        artifact["analysis_artifact_hash"] = content_hash({
            key: value for key, value in artifact.items()
            if key not in {"analysis_artifact_hash", "artifact_ref"}
        })
        _persist_artifact(tmp_path, artifact)
        ledger.project_analysis(artifact)
        cohort.append({"run_id": run_id, "analysis_id": analysis_id})

    report = ledger.find_divergent_decision_points(cohort, min_similarity=0.5)

    assert report["retrieval_engine"] == "semantica.ContextGraph.find_similar_decisions"
    assert report["authority"] == "CANDIDATE_ONLY"
    assert len(report["divergences"]) == 1
    divergence = report["divergences"][0]
    assert divergence["decision_function"] == "standing"
    assert divergence["same_decision_point"] is True
    assert len({row["decision_point_hash"] for row in divergence["members"]}) == 1
    projected = [ledger.graph.find_node(row["decision_id"])["metadata"]
                 for row in divergence["members"]]
    assert {row["atomic_decision_identity"]["identity_basis"] for row in projected} == {
        "EXACT_EVIDENCE_AND_SEMANTIC_QUESTION"}
    assert len({row["scenario"] for row in projected}) == 2
    assert divergence["grounding_status"] == "UNGROUNDED_OUTCOME_DIVERGENCE"
    assert divergence["outcome_distribution"] == {
        "CAN_ESTABLISH": 1, "MERELY_MENTIONS": 1}
    assert divergence["review_model_distribution"] == {
        "openai/gpt-5.6-luna": 1, "openai/gpt-5.6-terra": 1}
    assert {row["basis_sources"] for row in divergence["members"]} == {
        ("chart", "own_knowledge")}


def test_policy_bundle_change_impact_is_limited_to_decisions_grounded_in_that_policy(
        tmp_path: Path):
    artifact = _artifact("analysis-policy", run_id="run-policy",
                         function="standing", causal=False)
    artifact["cycles"][0]["actions"][0]["args"] = {
        "basis_sources": ["task_contract", "chart"],
        "cited_refs": ["evidence_rule.counts_as_evidence.2"],
        "checked_discriminating_fact_refs": [],
        "rule_coverage_claim": "DIRECTLY_COVERED",
    }
    artifact["cycles"][1]["actions"] = [{
        "tool": "note_decision", "event_ref": "layer1:3", "ok": True,
        "args": {
            "basis_sources": ["task_contract", "chart"],
            "cited_refs": ["decision_rule.1"],
            "checked_discriminating_fact_refs": [],
            "rule_coverage_claim": "DIRECTLY_COVERED",
        },
    }]
    artifact["cycles"][1]["decision_testimony_refs"] = ["decision:3"]
    run_dir = _persist_artifact(tmp_path, artifact)
    spec = load_spec(Path(__file__).resolve().parents[1] / "assets" / "specs" /
                     "STORE.390.date_of_initial_diagnosis.yaml")
    _, presentation = build_task_presentation(
        spec, run_id=artifact["run_id"], arm_id="policy_bundle",
        operational_preamble="Do the review through chart tools.")
    presentation.write(run_dir)
    artifact["task_presentation_hash"] = presentation.presentation_hash
    artifact["task_arm"] = "policy_bundle"
    artifact["analysis_artifact_hash"] = content_hash({
        key: value for key, value in artifact.items()
        if key not in {"analysis_artifact_hash", "artifact_ref"}
    })
    Path(artifact["artifact_ref"]).write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    physician_policy = next(
        row for row in presentation.policy_bundle["policies"]
        if row["clause_refs"] == ["evidence_rule.counts_as_evidence.2"])
    ledger = SemanticaLedger(tmp_path / "ledger.json")
    ledger.project_analysis(artifact)
    grounded = ledger.chain(artifact["run_id"], artifact["analysis_id"])["episodes"]
    assert grounded[0]["policy_groundings"] == [{
        "policy_id": physician_policy["policy_id"],
        "version": physician_policy["version"],
        "cited_clause_refs": ["evidence_rule.counts_as_evidence.2"],
        "application_semantics": "AGENT_CLAIM_RESOLVED_TO_OFFERED_POLICY",
    }]
    assert grounded[1]["policy_groundings"][0]["policy_id"].endswith(".decision.1")
    revision = ledger.register_policy_revision(
        physician_policy["policy_id"], from_version=physician_policy["version"],
        rules={"clarification": "A documented physician clinical impression can establish."},
        change_reason="Clarify the clinical diagnosis boundary.")

    ledger = SemanticaLedger(tmp_path / "ledger.json")
    assert ledger.graph.find_node(
        f"{physician_policy['policy_id']}:{revision['version']}") is not None
    report = ledger.affected_by_policy_change(
        physician_policy["policy_id"], from_version=physician_policy["version"],
        to_version=revision["version"])

    assert revision["version"] != physician_policy["version"]
    assert revision["change_reason"] == "Clarify the clinical diagnosis boundary."
    assert len(report["affected_decisions"]) == 1
    assert report["affected_decisions"][0]["category"] == "standing"
    assert report["affected_decisions"][0]["run_id"] == "run-policy"
    assert report["affected_decisions"][0]["analysis_id"] == "analysis-policy"
    assert report["affected_cases"] == [{
        "run_id": "run-policy", "analysis_id": "analysis-policy",
        "affected_decision_count": 1, "decision_functions": ["standing"],
    }]
    assert report["scope"] == "DIRECT_POLICY_BINDINGS"
