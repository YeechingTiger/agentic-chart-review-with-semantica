"""Human review shows one selected analysis and mines questions without inventing policy."""
from __future__ import annotations

import json
from copy import deepcopy

import pytest

from acr.mvp.human_review import human_review_view
from acr.mvp.task_presentation import content_hash


def _artifact(run_id="run-a", analysis_id="analysis-a", *, coverage="AMBIGUOUS_RULE",
              function="standing", decision="can establish", note="NOTE_A",
              basis_sources=None, citation_status="CLAIMED_AND_VERIFIED"):
    basis_sources = basis_sources or ["task_contract", "chart"]
    cycle = {
        "cycle_id": f"{run_id}:cycle:2", "run_id": run_id,
        "source_event_ids": ["layer1:2"], "source_seq_range": [2, 2],
        "state_before": {
            "observed_state": {"surfaced_notes": [note], "read_notes": [note],
                               "citation_resolutions": []},
            "declared_state": {"findings": [], "uncertainties": ["standing unresolved"]}},
        "state_after": {
            "observed_state": {"surfaced_notes": [note], "read_notes": [note],
                               "citation_resolutions": [{
                                   "testimony_ref": "decision:2",
                                   "cited": [{"ref": "decision_rule.1",
                                              "status": citation_status}],
                                   "checked_facts": []}]},
            "declared_state": {"findings": [{
                "finding_ref": "finding:1", "note_id": note, "field": "diagnosis_date",
                "standing": "can_establish", "assertion_class": "pathology_diagnosis"}],
                "uncertainties": []}},
        "trigger_event_refs": ["layer1:1"],
        "declared_open_question": "can this note establish the field",
        "decision_testimony_refs": ["decision:2"], "has_decision_testimony": True,
        "structural_kind": "TOOL_INTERACTION",
        "actions": [{"event_ref": "layer1:2", "tool": "note_decision", "ok": True,
                     "args": {"facing": "can this note establish the field",
                              "decision": decision, "because": "rule needs interpretation",
                              "basis_sources": basis_sources,
                              "cited_refs": ["decision_rule.1"],
                              "checked_discriminating_fact_refs": [],
                              "rule_coverage_claim": coverage,
                              "provisional_inference": "pathology wording is definitive",
                              "alternatives": ["merely mentions"],
                              "uncertainty": "wording boundary"}}],
        "observations": [{"event_ref": "layer1:2", "result": {
            "testimony_ref": "decision:2",
            "citation_resolutions": [{"ref": "decision_rule.1",
                                      "status": citation_status}],
            "checked_fact_resolutions": []}}],
    }
    episode = {
        "episode_id": f"{analysis_id}:episode:1", "source_cycle_ids": [cycle["cycle_id"]],
        "source_event_ids": ["layer1:2"], "decision_function": function,
        "decision_subject": ("retrieval_query_batch" if function == "where_to_look"
                             else "evidence_item"),
        "material_question": "can this note establish the field",
        "decision_rationale": "rule needs interpretation",
        "scenario": "one read pathology note with unresolved standing",
        "candidate_set": ["can_establish", "merely_mentions"], "decision": decision,
        "model_interpretation": "standing judgment", "claimed_basis_summary": ["chart"],
        "verified_reference_summary": ["decision_rule.1 verified"],
        "state_delta": "standing declared", "observed_downstream_refs": [],
        "hypothesized_impact": [], "counterfactual_supported_impact": [],
        "field_provenance": {"decision": "SELF_REPORTED", "scenario": "MODEL_RECONSTRUCTED"},
        "source_refs_by_field": {}, "reconstruction_stability": 0.8,
        "stability_status": "PROVISIONAL_DRIFT",
    }
    return {"run_id": run_id, "analysis_id": analysis_id, "trace_id": "t" * 32,
            "task_presentation_hash": "presentation", "cycles": [cycle],
            "episodes": [episode], "mechanical_cycle_ids": [],
            "cycle_annotations": {
                cycle["cycle_id"]: {
                    "role": "DECISION_BEARING", "decision_function": function}},
            "stability_status": "PROVISIONAL_DRIFT", "reconstruction_stability": 0.8,
            "reconstructor_identity": "terra", "causal_assertions": []}


class StubLedger:
    def __init__(self, artifacts, selected=None):
        self.artifacts = {(row["run_id"], row["analysis_id"]): row for row in artifacts}
        self.selected = selected or {}

    def chain(self, run_id, analysis_id=None):
        chosen = analysis_id or self.selected.get(run_id)
        available = sorted(a for r, a in self.artifacts if r == run_id)
        if not chosen:
            return {"run_id": run_id, "status": "NO_ANALYSIS_SELECTION",
                    "available_analysis_ids": available, "episodes": [], "causal_edges": []}
        art = self.artifacts[(run_id, chosen)]
        return {"run_id": run_id, "analysis_id": chosen, "status": "OK",
                "episodes": art["episodes"],
                "causal_edges": art.get("causal_assertions") or [],
                "suggested_links": []}

    def load_analysis_artifact(self, run_id, analysis_id):
        return self.artifacts[(run_id, analysis_id)]

    def selected_analysis(self, run_id):
        return self.selected.get(run_id)


def test_review_refuses_to_mix_provisional_analyses_without_selection():
    ledger = StubLedger([_artifact(analysis_id="a"), _artifact(analysis_id="b")])
    view = human_review_view(ledger, "run-a")
    assert view["status"] == "NO_ANALYSIS_SELECTION"
    assert view["episodes"] == []
    assert view["available_analysis_ids"] == ["a", "b"]


def test_explicit_unselected_analysis_is_labeled_provisional_not_selected():
    ledger = StubLedger([_artifact(analysis_id="a"), _artifact(analysis_id="b")])

    view = human_review_view(ledger, "run-a", "a")

    assert view["status"] == "OK"
    assert view["analysis_id"] == "a"
    assert view["selected_analysis_id"] is None
    assert view["analysis_view_mode"] == "PROVISIONAL_EXPLICIT"


def test_stable_explicit_analysis_is_not_mislabeled_provisional():
    artifact = _artifact(analysis_id="a")
    artifact["stability_status"] = "STABLE_ACROSS_PASSES"
    artifact["reconstruction_stability"] = 1.0
    artifact["episodes"][0]["stability_status"] = "STABLE_ACROSS_PASSES"
    artifact["episodes"][0]["reconstruction_stability"] = 1.0

    view = human_review_view(StubLedger([artifact]), "run-a", "a")

    assert view["selected_analysis_id"] is None
    assert view["analysis_view_mode"] == "STABLE_EXPLICIT"


def test_recording_selected_evidence_is_shown_as_assembly_not_new_discovery():
    artifact = _artifact(function="where_to_look")
    artifact["episodes"][0]["decision_subject"] = "retrieval_document_set"
    artifact["cycles"][0]["actions"].append({
        "event_ref": "layer1:3", "tool": "record_evidence", "ok": True,
        "args": {"finding_ref": "finding:1"},
    })

    view = human_review_view(StubLedger([artifact]), "run-a", "analysis-a")
    step = view["review_chain"]["steps"][0]

    assert step["phase"] == "EVIDENCE_ASSEMBLY"
    assert step["phase_label"] == "Assemble the evidence record"


def test_nonadjacent_asserted_influence_stays_visible_in_the_human_chain():
    artifact = _artifact(function="standing")
    template_cycle = artifact["cycles"][0]
    template_episode = artifact["episodes"][0]
    for seq, decision_function, decision_subject in (
        (3, "standing", "evidence_item"),
        (4, "which_wins", "evidence_relationship"),
    ):
        cycle = deepcopy(template_cycle)
        cycle["cycle_id"] = f"run-a:cycle:{seq}"
        cycle["source_event_ids"] = [f"layer1:{seq}"]
        cycle["source_seq_range"] = [seq, seq]
        cycle["decision_testimony_refs"] = [f"decision:{seq}"]
        cycle["actions"][0]["event_ref"] = f"layer1:{seq}"
        cycle["actions"][0]["args"]["decision"] = f"choice {seq}"
        cycle["observations"][0]["event_ref"] = f"layer1:{seq}"
        cycle["observations"][0]["result"]["testimony_ref"] = f"decision:{seq}"
        episode = deepcopy(template_episode)
        episode["episode_id"] = f"analysis-a:episode:{seq - 1}"
        episode["source_cycle_ids"] = [cycle["cycle_id"]]
        episode["source_event_ids"] = [f"layer1:{seq}"]
        episode["decision_function"] = decision_function
        episode["decision_subject"] = decision_subject
        episode["decision"] = f"choice {seq}"
        artifact["cycles"].append(cycle)
        artifact["episodes"].append(episode)
        artifact["cycle_annotations"][cycle["cycle_id"]] = {
            "role": "DECISION_BEARING", "decision_function": decision_function}
    artifact["causal_assertions"] = [{
        "assertion_id": "analysis-a:runtime-dependency:1",
        "source_episode_id": "analysis-a:episode:1",
        "target_episode_id": "analysis-a:episode:3",
        "relationship_type": "INFLUENCED",
        "evidence_refs": ["note:N1"],
        "reasoning": "The later decision explicitly used note:N1.",
        "provenance": "RUNTIME_REFERENCE_DERIVED",
    }]

    view = human_review_view(StubLedger([artifact]), "run-a", "analysis-a")
    first = view["review_chain"]["steps"][0]

    assert first["link_to_next"]["kind"] == "TEMPORAL_ONLY"
    assert first["causal_links"] == [{
        "relationship_type": "INFLUENCED",
        "target_step_id": "analysis-a:episode:3",
        "target_title": "Resolve the evidence",
        "target_question": "can this note establish the field",
        "target_decision": "choice 4",
        "assertion_id": "analysis-a:runtime-dependency:1",
        "evidence_refs": ["note:N1"],
        "reasoning": "The later decision explicitly used note:N1.",
        "provenance": "RUNTIME_REFERENCE_DERIVED",
    }]


def test_atomic_finding_follows_its_explicit_prior_testimony_without_calling_it_local():
    artifact = _artifact(function="standing")
    source_episode = artifact["episodes"][0]
    source_cycle = artifact["cycles"][0]
    finding_cycle = deepcopy(source_cycle)
    finding_cycle["cycle_id"] = "run-a:cycle:3"
    finding_cycle["source_event_ids"] = ["layer1:3"]
    finding_cycle["source_seq_range"] = [3, 3]
    finding_cycle["decision_testimony_refs"] = []
    finding_cycle["actions"] = [{
        "event_ref": "layer1:3", "tool": "record_finding", "ok": True,
        "args": {"decision_testimony_ref": "decision:2", "field": "diagnosis_date",
                 "standing": "can_establish"},
    }]
    finding_cycle["observations"] = []
    finding_episode = deepcopy(source_episode)
    finding_episode["episode_id"] = "analysis-a:episode:2"
    finding_episode["source_cycle_ids"] = [finding_cycle["cycle_id"]]
    finding_episode["source_event_ids"] = ["layer1:3"]
    finding_episode["decision"] = "record NOTE_A as can_establish"
    finding_episode["decision_rationale"] = "the finding call committed this standing"
    artifact["cycles"].append(finding_cycle)
    artifact["episodes"].append(finding_episode)
    artifact["cycle_annotations"][finding_cycle["cycle_id"]] = {
        "role": "DECISION_BEARING", "decision_function": "standing"}

    view = human_review_view(StubLedger([artifact]), "run-a", "analysis-a")
    finding = view["episodes"][1]

    assert finding["runtime_testimonies"][0]["testimony_ref"] == "decision:2"
    assert finding["runtime_testimonies"][0]["link_scope"] == \
        "EXPLICIT_ACTION_REFERENCE"
    assert view["review_chain"]["steps"][1]["decision"] == \
        "record NOTE_A as can_establish"
    assert view["review_chain"]["steps"][1]["reason"] == \
        "the finding call committed this standing"
    assert view["review_chain"]["steps"][1]["reason_provenance"] == \
        "MODEL_RECONSTRUCTED"
    assert [row["code"] for row in finding["review_attention"]] == [
        "PROVISIONAL_INFERENCE_USED", "RULE_APPLICATION_REVIEW",
        "CROSS_EPISODE_TESTIMONY",
    ]


def test_legacy_testimony_reused_for_multiple_findings_is_flagged_as_compound():
    artifact = _artifact(function="standing")
    template_episode = artifact["episodes"][0]
    for seq, note in ((3, "NOTE_A"), (4, "NOTE_B")):
        cycle = deepcopy(artifact["cycles"][0])
        cycle["cycle_id"] = f"run-a:cycle:{seq}"
        cycle["source_event_ids"] = [f"layer1:{seq}"]
        cycle["source_seq_range"] = [seq, seq]
        cycle["decision_testimony_refs"] = []
        cycle["actions"] = [{
            "event_ref": f"layer1:{seq}", "tool": "record_finding", "ok": True,
            "args": {"decision_testimony_ref": "decision:2", "note_id": note,
                     "field": "diagnosis_date", "standing": "can_establish"},
        }]
        cycle["observations"] = [{
            "event_ref": f"layer1:{seq}", "result": {
                "recorded": True, "finding_ref": f"finding:{seq - 2}"},
        }]
        episode = deepcopy(template_episode)
        episode["episode_id"] = f"analysis-a:episode:{seq - 1}"
        episode["source_cycle_ids"] = [cycle["cycle_id"]]
        episode["source_event_ids"] = [f"layer1:{seq}"]
        artifact["cycles"].append(cycle)
        artifact["episodes"].append(episode)
        artifact["cycle_annotations"][cycle["cycle_id"]] = {
            "role": "DECISION_BEARING", "decision_function": "standing"}

    view = human_review_view(StubLedger([artifact]), "run-a", "analysis-a")

    for episode in view["episodes"]:
        flag = next(row for row in episode["review_attention"]
                    if row["code"] == "COMPOUND_RUNTIME_TESTIMONY")
        assert flag["refs"] == ["decision:2"]
        assert flag["finding_event_refs"] == ["layer1:3", "layer1:4"]
        assert flag["route"] == "INSTRUMENTATION_QUESTION"


def test_review_exposes_rejected_terra_attempts_without_projecting_them_as_episodes():
    artifact = _artifact()
    artifact["reconstructor_attempts"] = [
        {"attempt_index": 1, "validation_status": "REJECTED",
         "validation_error": "observed downstream refs must occur after the episode"},
        {"attempt_index": 2, "validation_status": "ACCEPTED", "validation_error": None},
    ]

    view = human_review_view(StubLedger([artifact]), "run-a", "analysis-a")

    assert view["reconstructor_attempts"] == artifact["reconstructor_attempts"]
    assert len(view["episodes"]) == 1


def test_selected_review_puts_testimony_beside_terra_and_links_raw_events():
    artifact = _artifact()
    ledger = StubLedger([artifact], {"run-a": "analysis-a"})
    view = human_review_view(ledger, "run-a")
    episode = view["episodes"][0]
    assert view["analysis_view_mode"] == "SELECTED"
    assert episode["runtime_testimonies"][0]["provenance"] == "SELF_REPORTED"
    assert episode["runtime_receipt_status"] == "LEGACY_TRACE_DERIVED"
    assert episode["choice_boundary_provenance"] == "MODEL_RECONSTRUCTED_FROM_LEGACY_TRACE"
    assert view["runtime_receipt_manifest"]["mode"] == "LEGACY"
    assert view["decision_receipt_coverage"]["status"] == "NO_SEALED_DECISIONS"
    assert episode["reconstruction"]["provenance"] == "MODEL_RECONSTRUCTED"
    assert episode["runtime_testimonies"][0]["citation_resolutions"][0]["status"] \
        == "CLAIMED_AND_VERIFIED"
    assert episode["state_before"] != episode["state_after"]
    assert episode["raw_langtrace_links"] == [{
        "trace_id": "t" * 32, "event_ref": "layer1:2",
        "href": f"langtrace://trace/{'t' * 32}#layer1:2"}]
    assert episode["disposition"] == "RULE_APPLICATION_QUESTION"


def test_human_chain_keeps_only_judgment_relevant_attention_in_the_default_view():
    artifact = _artifact()
    view = human_review_view(
        StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    chain = view["review_chain"]
    assert [row["kind"] for row in chain["steps"]] == ["judgment"]
    assert chain["steps"][0]["decision"] == "can establish"
    assert chain["steps"][0]["reason"] == "rule needs interpretation"
    assert [row["code"] for row in chain["steps"][0]["review_attention"]] == [
        "PROVISIONAL_INFERENCE_USED", "RULE_APPLICATION_REVIEW",
    ]
    assert chain["priority_review_count"] == 2
    assert chain["technical_attention_count"] == 0


def test_human_chain_groups_but_never_merges_consecutive_retrieval_decisions():
    artifact = _artifact(function="where_to_look", decision="Search pathology")
    second_cycle = deepcopy(artifact["cycles"][0])
    second_cycle["cycle_id"] = "run-a:cycle:3"
    second_cycle["source_event_ids"] = ["layer1:3"]
    second_cycle["actions"][0]["event_ref"] = "layer1:3"
    second_cycle["actions"][0]["args"]["decision"] = "Read the earliest pathology note"
    second_cycle["observations"][0]["event_ref"] = "layer1:3"
    artifact["cycles"].append(second_cycle)
    artifact["cycle_annotations"][second_cycle["cycle_id"]] = {
        "role": "DECISION_BEARING", "decision_function": "where_to_look"}
    second_episode = deepcopy(artifact["episodes"][0])
    second_episode["episode_id"] = "analysis-a:episode:2"
    second_episode["source_cycle_ids"] = ["run-a:cycle:3"]
    second_episode["source_event_ids"] = ["layer1:3"]
    second_episode["decision"] = "Read the earliest pathology note"
    artifact["episodes"].append(second_episode)

    view = human_review_view(
        StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    steps = view["review_chain"]["steps"]
    assert len(steps) == 2
    assert [row["kind"] for row in steps] == ["retrieval", "retrieval"]
    assert [row["episode_count"] for row in steps] == [1, 1]
    assert [row["decisions"] for row in steps] == [
        ["Search pathology"], ["Read the earliest pathology note"],
    ]
    assert view["review_chain"]["phases"] == [{
        "phase": "EVIDENCE_DISCOVERY",
        "label": "Find the evidence",
        "step_ids": [steps[0]["step_id"], steps[1]["step_id"]],
        "decision_count": 2,
    }]
    assert all(row["audit_unit"] == "ATOMIC_DECISION" for row in steps)


def test_retrieval_step_exposes_actual_coverage_without_claiming_it_is_complete():
    artifact = _artifact(function="where_to_look", decision="Search pathology")
    cycle = artifact["cycles"][0]
    cycle["actions"].extend([
        {"event_ref": "layer1:3", "tool": "list_documents", "ok": True,
         "args": {"objective": "establish the complete chart inventory"}},
        {"event_ref": "layer1:4", "tool": "search", "ok": True,
         "args": {"query": "cancer pathology", "objective": "find diagnosis evidence"}},
        {"event_ref": "layer1:5", "tool": "read", "ok": True,
         "args": {"note_id": "PATH_2023", "objective": "test whether it establishes"}},
    ])
    cycle["observations"].extend([
        {"event_ref": "layer1:3", "result": {
            "documents": [{"note_id": "PATH_2023"}, {"note_id": "ONC_2023"}],
            "returned": 200, "total": 321, "offset": 0, "limit": 200,
            "page_complete": False, "unreturned": 121,
            "types": [{"doc_type": "Progress-Note", "count": 100}]}},
        {"event_ref": "layer1:4", "result": {"hits": [
            {"note_id": "PATH_2023"}, {"note_id": "PATH_2023"}]}},
        {"event_ref": "layer1:5", "result": {
            "note_id": "PATH_2023", "truncated": False}},
    ])
    cycle["state_after"]["observed_state"].update({
        "surfaced_notes": ["PATH_2023", "ONC_2023"],
        "read_notes": ["PATH_2023"],
    })

    view = human_review_view(
        StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    coverage = view["review_chain"]["steps"][0]["coverage"]
    assert coverage["assessment"] == "HUMAN_TO_ADJUDICATE"
    assert coverage["surfaced_count"] == 2
    assert coverage["read_count"] == 1
    assert coverage["unfiltered_listing_done"] is True
    assert coverage["listings"][0]["documents_returned"] == 200
    assert coverage["listings"][0]["documents_total"] == 321
    assert coverage["listings"][0]["page_complete"] is False
    assert coverage["listings"][0]["unreturned_count"] == 121
    assert coverage["inventory_page_complete"] is False
    assert coverage["searches"][0]["query"] == "cancer pathology"
    assert coverage["searches"][0]["results_returned"] == 2
    assert coverage["searches"][0]["unique_notes_returned"] == 1
    assert coverage["searches"][0]["duplicate_hits"] == 1
    assert coverage["read_note_ids"] == ["PATH_2023"]
    assert view["review_chain"]["steps"][0]["review_attention"][-1]["code"] == \
        "INVENTORY_PAGE_PARTIAL"


def test_standing_step_places_server_resolved_quote_beside_the_judgment():
    artifact = _artifact()
    cycle = artifact["cycles"][0]
    cycle["actions"].append({
        "event_ref": "layer1:3", "tool": "record_finding", "ok": True,
        "args": {"note_id": "NOTE_A", "field": "diagnosis_date",
                 "standing": "can_establish", "assertion_class": "clinical_diagnosis",
                 "event_time": "2023-04-12"},
    })
    cycle["observations"].append({
        "event_ref": "layer1:3", "result": {
            "recorded": True, "finding_ref": "finding:1",
            "server_fact": {"span": [10, 70]},
            "quote": "Assessment: newly diagnosed metastatic lung cancer.",
        },
    })

    view = human_review_view(
        StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")
    evidence = view["review_chain"]["steps"][0]["evidence"]

    assert evidence == [{
        "kind": "NOTE_FINDING", "event_ref": "layer1:3", "finding_ref": "finding:1",
        "note_id": "NOTE_A", "field": "diagnosis_date",
        "standing": "can_establish", "assertion_class": "clinical_diagnosis",
        "event_time": "2023-04-12", "record_time": None, "carried_forward": None,
        "span": [10, 70],
        "quote": "Assessment: newly diagnosed metastatic lung cancer.",
        "claim_provenance": "SELF_REPORTED", "quote_provenance": "SERVER_RESOLVED",
    }]


def test_decision_step_exposes_conditions_and_keeps_reconstruction_provenance_distinct():
    artifact = _artifact()
    view = human_review_view(
        StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    step = view["review_chain"]["steps"][0]
    dependencies = step["dependencies"]
    assert step["basis_sources"] == ["task_contract", "chart"]
    assert dependencies["candidate_set"] == ["can_establish", "merely_mentions"]
    assert dependencies["alternatives_considered"] == ["merely mentions"]
    assert dependencies["provisional_inferences"] == [
        "pathology wording is definitive",
    ]
    assert dependencies["unresolved_uncertainties"] == ["wording boundary"]
    assert dependencies["citation_resolutions"][0]["ref"] == "decision_rule.1"
    assert dependencies["trace_reconstructed_basis"] == ["chart"]
    assert dependencies["verified_reference_summary"] == [
        "decision_rule.1 verified",
    ]


def test_task_brief_reads_only_the_question_and_contract_identity(tmp_path):
    artifact = _artifact()
    payload = {
        "schema": "acr.task_presentation.v1", "run_id": "run-a",
        "arm_id": "detailed",
        "task_contract_ref": {"id": "STORE.390", "version": "0.1.0",
                              "content_hash": "contract"},
        "offered_clause_catalog": [], "known_clause_index": [],
        "method_card_refs": [], "operational_instruction_refs": [],
        "rendered_prompt_artifact_ref": "prompt.txt", "prompt_hash": "prompt",
    }
    payload["presentation_hash"] = content_hash(payload)
    artifact["task_presentation_hash"] = payload["presentation_hash"]
    (tmp_path / "task_presentation.json").write_text(
        json.dumps(payload), encoding="utf-8")
    (tmp_path / "prompt.txt").write_text(
        "private preamble\nQUESTION: What is the first diagnosis date?\nsecret detail\n",
        encoding="utf-8")

    view = human_review_view(
        StubLedger([artifact], {"run-a": "analysis-a"}), "run-a", run_dir=tmp_path)

    assert view["task"] == {
        "question": "What is the first diagnosis date?",
        "contract_id": "STORE.390", "contract_version": "0.1.0",
        "arm_id": "detailed",
    }
    assert "private preamble" not in str(view["task"])


def test_self_hosted_langtrace_link_targets_the_real_project_trace_page(tmp_path):
    artifact = _artifact()
    (tmp_path / "runner_meta.json").write_text(json.dumps({
        "langtrace_project_id": "project-local",
    }), encoding="utf-8")

    view = human_review_view(
        StubLedger([artifact], {"run-a": "analysis-a"}), "run-a", run_dir=tmp_path,
        langtrace_ui_base="http://127.0.0.1:3100")

    assert view["episodes"][0]["raw_langtrace_links"][0]["href"] == (
        "http://127.0.0.1:3100/project/project-local/traces#layer1%3A2")


def test_review_surfaces_self_reported_model_knowledge_without_calling_it_an_error():
    artifact = _artifact(basis_sources=["chart", "own_knowledge"],
                         coverage="NO_APPLICABLE_RULE")
    view = human_review_view(StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    episode = view["episodes"][0]
    own_knowledge = next(row for row in episode["review_attention"]
                         if row["code"] == "MODEL_KNOWLEDGE_USED")
    assert own_knowledge == {
        "code": "MODEL_KNOWLEDGE_USED",
        "severity": "REVIEW",
        "title": "Model used knowledge outside the supplied material",
        "detail": "The agent self-reported own_knowledge as a basis for this decision.",
        "provenance": "SELF_REPORTED",
        "is_error": False,
        "route": "NEEDS_HUMAN_ADJUDICATION",
    }
    assert episode["uses_model_knowledge"] is True
    assert episode["grounding_assessment"]["judgment_mode"] == \
        "OUTSIDE_MATERIAL_MODEL_JUDGMENT"
    assert episode["grounding_assessment"]["semantic_entailment_status"] == \
        "NOT_ESTABLISHED_BY_REFERENCE_CHECK"


def test_review_does_not_confuse_resolved_policy_reference_with_semantic_entailment():
    artifact = _artifact(function="where_to_look", coverage="DIRECTLY_COVERED")
    view = human_review_view(StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    grounding = view["episodes"][0]["grounding_assessment"]
    assert grounding["reference_resolution_status"] == "ALL_REFERENCES_RESOLVED"
    assert grounding["semantic_entailment_status"] == \
        "NOT_ESTABLISHED_BY_REFERENCE_CHECK"
    assert grounding["judgment_mode"] == "POLICY_GUIDED_OPERATIONAL_JUDGMENT"
    assert grounding["uses_model_judgment"] is True


def test_review_flags_rule_claim_that_was_not_offered_to_this_run():
    artifact = _artifact(citation_status="CLAIMED_NOT_OFFERED")
    view = human_review_view(StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    attention = view["episodes"][0]["review_attention"]
    flag = next(row for row in attention if row["code"] == "RULE_NOT_OFFERED")
    assert flag["refs"] == ["decision_rule.1"]
    assert flag["provenance"] == "DETERMINISTIC_DERIVED"
    assert flag["is_error"] is False
    assert flag["route"] == "NEEDS_ATTRIBUTION"


def test_review_surfaces_provisional_inference_as_a_review_target_not_an_error():
    artifact = _artifact()
    view = human_review_view(StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    flag = next(row for row in view["episodes"][0]["review_attention"]
                if row["code"] == "PROVISIONAL_INFERENCE_USED")
    assert flag["inferences"] == ["pathology wording is definitive"]
    assert flag["provenance"] == "SELF_REPORTED"
    assert flag["is_error"] is False


def test_review_routes_no_applicable_rule_claim_to_attribution_before_guideline_change():
    artifact = _artifact(coverage="NO_APPLICABLE_RULE")
    view = human_review_view(StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    flag = next(row for row in view["episodes"][0]["review_attention"]
                if row["code"] == "NO_APPLICABLE_RULE_CLAIMED")
    assert flag["route"] == "NEEDS_ATTRIBUTION"
    assert flag["is_error"] is False
    assert "does not prove a guideline gap" in flag["detail"]


@pytest.mark.parametrize("coverage", [
    "COVERED_WITH_INTERPRETATION", "AMBIGUOUS_RULE", "CONFLICTING_RULES",
])
def test_review_routes_non_direct_rule_application_for_human_review(coverage: str):
    artifact = _artifact(coverage=coverage)
    view = human_review_view(StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    flag = next(row for row in view["episodes"][0]["review_attention"]
                if row["code"] == "RULE_APPLICATION_REVIEW")
    assert flag["claims"] == [coverage]
    assert flag["route"] == "RULE_APPLICATION_QUESTION"
    assert flag["is_error"] is False


def test_review_flags_unknown_reference_as_unverified_not_as_a_confirmed_error():
    artifact = _artifact(citation_status="CLAIMED_UNKNOWN")
    view = human_review_view(StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    flag = next(row for row in view["episodes"][0]["review_attention"]
                if row["code"] == "REFERENCE_UNVERIFIED")
    assert flag["refs"] == ["decision_rule.1"]
    assert flag["provenance"] == "DETERMINISTIC_DERIVED"
    assert flag["route"] == "NEEDS_ATTRIBUTION"
    assert flag["is_error"] is False


def test_verified_duplicate_reference_suppresses_an_unknown_channel_copy():
    artifact = _artifact(citation_status="CLAIMED_UNKNOWN")
    result = artifact["cycles"][0]["observations"][0]["result"]
    result["checked_fact_resolutions"] = [{
        "ref": "decision_rule.1", "status": "CLAIMED_AND_VERIFIED"}]

    view = human_review_view(StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    attention = view["episodes"][0]["review_attention"]
    assert not any(row["code"] == "REFERENCE_UNVERIFIED" for row in attention)
    assert view["episodes"][0]["disposition"] != "INSTRUMENTATION_QUESTION"


def test_review_routes_missing_runtime_testimony_to_instrumentation():
    artifact = _artifact()
    cycle = artifact["cycles"][0]
    cycle["actions"], cycle["observations"] = [], []
    cycle["decision_testimony_refs"], cycle["has_decision_testimony"] = [], False
    view = human_review_view(StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    flag = next(row for row in view["episodes"][0]["review_attention"]
                if row["code"] == "MISSING_RUNTIME_TESTIMONY")
    assert flag["provenance"] == "SERVER_FACT"
    assert flag["route"] == "INSTRUMENTATION_QUESTION"


def test_trace_cited_terra_reconstruction_is_not_called_an_unknown_reason():
    artifact = _artifact()
    cycle = artifact["cycles"][0]
    cycle["actions"] = [{
        "event_ref": "layer1:2", "tool": "record_finding", "ok": True,
        "args": {"field": "diagnosis_date", "standing": "can_establish"},
    }]
    cycle["observations"] = [{
        "event_ref": "layer1:2", "result": {"recorded": True},
    }]
    cycle["decision_testimony_refs"], cycle["has_decision_testimony"] = [], False
    artifact["episodes"][0]["source_refs_by_field"] = {
        "decision": ["layer1:2"],
        "model_interpretation": ["layer1:2"],
        "claimed_basis_summary": ["layer1:2"],
    }

    view = human_review_view(
        StubLedger([artifact], {"run-a": "analysis-a"}), "run-a")

    # The episode still records the instrumentation distinction, but the human-first chain
    # does not turn "no note_decision" into "the reason is unknowable" when Terra cited trace.
    assert any(row["code"] == "MISSING_RUNTIME_TESTIMONY"
               for row in view["episodes"][0]["review_attention"])
    step = view["review_chain"]["steps"][0]
    assert step["review_attention"] == []
    assert [row["code"] for row in step["technical_attention"]] == [
        "MISSING_RUNTIME_TESTIMONY",
    ]
    assert step["reason_provenance"] == "TRACE_RECONSTRUCTED"
    assert step["trace_support"] == {
        "status": "TRACE_CITED",
        "ref_count": 1,
        "refs": ["layer1:2"],
    }
