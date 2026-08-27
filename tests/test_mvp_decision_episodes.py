"""Terra annotates a fixed cycle skeleton; deterministic code owns every invariant."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from acr.mvp.decision_receipts import make_runtime_decision_receipt
from acr.mvp.decision_types import DECISION_SUBJECTS, DECISION_TAXONOMY_SCHEMA
from acr.mvp.langtrace_io import LangtraceReviewTrace
from acr.mvp.reconstruct import (
    ReconstructionError,
    _canonical_annotations,
    _retrieval_decision,
    _retrieval_subject,
    _runtime_reference_dependencies,
    build_prompt,
    extraction_response_format,
    model_input_projection,
    reconstruct_run,
    verify_extraction,
)
from acr.mvp.task_presentation import content_hash
from acr.mvp.timeline import build_react_cycles, build_trace_completeness


def _review() -> LangtraceReviewTrace:
    events = [
        {"seq": 1, "ts": "t1", "kind": "run_meta", "spec_id": "S", "patient_id": "P"},
        {"seq": 2, "ts": "t2", "kind": "tool_call", "tool": "note_decision", "ok": True,
         "args": {"facing": "no candidate has been surfaced", "decision": "search pathology",
                  "because": "pathology can establish the field", "basis_sources": ["chart"],
                  "cited_refs": [], "checked_discriminating_fact_refs": [],
                  "rule_coverage_claim": "OPERATIONAL_DISCRETION"},
         "result": {"noted": True, "testimony_ref": "decision:2",
                    "citation_resolutions": [], "checked_fact_resolutions": []}},
        {"seq": 3, "ts": "t3", "kind": "tool_call", "tool": "search", "ok": True,
         "args": {"query": "pathology", "objective": "find a candidate"},
         "result": {"n": 1, "hits": [{"note_id": "N1"}]}},
        {"seq": 4, "ts": "t4", "kind": "tool_call", "tool": "submit_answer", "ok": True,
         "args": {"status": "FOUND", "value": {"field": "redacted"}},
         "result": {"accepted": True, "why": "ok"}},
        {"seq": 5, "ts": "t5", "kind": "answer_accepted", "status": "FOUND"},
    ]
    root = {"acr.trace.schema": "acr.langtrace.v2", "acr.export.status": "COMPLETE",
            "acr.layer1.event_count": len(events),
            "acr.layer1.content_hash": content_hash(events)}
    spans = [{"spanId": "root", "attributes": root}]
    spans += [{"spanId": f"s{e['seq']}", "parentSpanId": "root",
               "attributes": {"acr.layer1.seq": e["seq"],
                              "acr.layer1.event_json": json.dumps(e)}} for e in events]
    return LangtraceReviewTrace(
        trace_id="b" * 32, run_id="run-episode", patient_id="P", spec_id="S",
        steps=[], layer1_events=events, spans=spans, spec_hash="sh", task_arm="detailed",
        task_presentation_hash="ph")


def _raw(cycles, *, function="where_to_look") -> dict[str, Any]:
    ids = [cycle["cycle_id"] for cycle in cycles]
    fields = {
        "material_question": ["decision:2"],
        "decision_subject": [f"cycle:{ids[0]}"],
        "scenario": [f"state_before:{ids[0]}"],
        "candidate_set": [f"state_before:{ids[0]}"],
        "decision": ["decision:2"],
        "decision_rationale": ["decision:2"],
        "model_interpretation": [f"cycle:{ids[0]}"],
        "claimed_basis_summary": ["decision:2"],
        "verified_reference_summary": [f"cycle:{ids[0]}"],
        "state_delta": [f"state_after:{ids[1]}"],
        "observed_downstream_refs": ["layer1:4"],
        "hypothesized_impact": [f"cycle:{ids[1]}"],
        "counterfactual_supported_impact": [],
    }
    return {
        "cycle_annotations": {
            ids[0]: {"role": "DECISION_BEARING", "decision_function": function},
            ids[1]: {"role": "DECISION_SUPPORT", "decision_function": None},
            ids[2]: {"role": "DECISION_BEARING", "decision_function": "what_to_answer"},
            ids[3]: {"role": "MECHANICAL", "decision_function": None},
        },
        "episodes": [
            {"source_cycle_ids": ids[:2], "decision_function": function,
             "decision_subject": ("retrieval_query_batch" if function == "where_to_look"
                                  else "evidence_item"),
             "material_question": "Where should I look for qualifying evidence?",
             "scenario": "No candidate had yet been surfaced.",
             "candidate_set": ["search pathology", "search clinic notes"],
             "decision": "search pathology",
             "decision_rationale": "Pathology is a permitted evidence source.",
             "model_interpretation": "retrieval routing",
             "claimed_basis_summary": ["chart"], "verified_reference_summary": [],
             "state_delta": "one candidate surfaced", "observed_downstream_refs": ["layer1:4"],
             "hypothesized_impact": ["narrowed the candidate set"],
             "counterfactual_supported_impact": [], "source_refs_by_field": fields},
            {"source_cycle_ids": [ids[2]], "decision_function": "what_to_answer",
             "decision_subject": "answer_selection",
             "material_question": "What result should be submitted?",
             "scenario": "A candidate was available for submission.", "candidate_set": ["FOUND"],
             "decision": "submit FOUND",
             "decision_rationale": "The observed candidate satisfies the submission contract.",
             "model_interpretation": "answer commitment",
             "claimed_basis_summary": [], "verified_reference_summary": [],
             "state_delta": "submission accepted", "observed_downstream_refs": ["layer1:5"],
             "hypothesized_impact": [], "counterfactual_supported_impact": [],
             "source_refs_by_field": {
                 **fields, "material_question": [f"cycle:{ids[2]}"],
                 "decision_subject": [f"cycle:{ids[2]}"],
                 "scenario": [f"state_before:{ids[2]}"],
                 "candidate_set": [f"state_before:{ids[2]}"],
                 "decision": [f"cycle:{ids[2]}"],
                 "decision_rationale": [f"cycle:{ids[2]}"],
                 "observed_downstream_refs": ["layer1:5"],
             }},
        ],
        "mechanical_cycle_ids": [ids[3]],
        "causal_links": [{"source_episode_index": 0, "target_episode_index": 1,
                          "relationship_type": "INFLUENCED", "evidence_refs": ["decision:2"],
                          "reasoning": "the declared search choice supplied the candidate"}],
    }


def _cycles():
    review = _review()
    return review, build_react_cycles(review, build_trace_completeness(review))


def _seal_first_receipt(cycles):
    cycle = cycles[0]
    action = cycle["actions"][0]
    result = cycle["observations"][0]["result"]
    receipt = make_runtime_decision_receipt(
        action["tool"], action["args"], result, cycle["state_before"],
        source_event_ref=action["event_ref"])
    assert receipt is not None
    result["decision_receipt"] = receipt
    cycle["decision_receipt_refs"] = [receipt["receipt_id"]]
    cycle["has_decision_receipt"] = True
    return receipt


def test_wire_schema_fixes_every_cycle_annotation_key():
    _, cycles = _cycles()
    schema = extraction_response_format(cycles)["json_schema"]["schema"]
    annotations = schema["properties"]["cycle_annotations"]
    assert set(annotations["properties"]) == {cycle["cycle_id"] for cycle in cycles}
    assert set(annotations["required"]) == set(annotations["properties"])
    assert annotations["additionalProperties"] is False
    episode = schema["properties"]["episodes"]["items"]
    assert set(episode["properties"]["decision_subject"]["enum"]) == \
        set(DECISION_SUBJECTS)
    assert "decision_subject" in episode["required"]
    ref_items = episode["properties"]["source_refs_by_field"]["properties"][
        "hypothesized_impact"]["items"]
    assert ref_items == {"type": "string"}
    # Stable refs are supplied once in the model projection and checked by deterministic code;
    # they are not copied into thirteen identical provider-schema enums.
    projection = model_input_projection(cycles)
    assert "decision:2" in projection["available_source_refs"]
    assert "search:pathology" in projection["available_source_refs"]


def test_prompt_makes_cycle_role_and_function_coherence_explicit():
    _, cycles = _cycles()

    prompt = build_prompt(cycles)

    assert "DECISION_SUPPORT -> decision_function MUST be JSON null" in prompt
    assert "MECHANICAL -> decision_function MUST be JSON null" in prompt
    assert "does not inherit the enclosing episode's function" in prompt
    assert "exactly one DECISION_BEARING cycle" in prompt
    assert "A precommitted keyword/query batch is one decision" in prompt
    assert "A new query chosen after observing results is a new decision" in prompt
    assert "Every successful record_finding call commits the Standing" in prompt
    assert "has_decision_receipt=true" in prompt
    assert "does not choose the decision_function" in prompt
    assert "offset>0 list_documents" in prompt
    assert "observable switch between inventory, search" in prompt
    assert "DECISION-RELEVANT FIXED CYCLE PROJECTION" in prompt


def test_model_projection_summarizes_repeated_inventory_and_receipt_payloads():
    cycle = {
        "cycle_id": "c1", "source_event_ids": ["layer1:1"],
        "source_seq_range": [1, 1], "actions": [{
            "event_ref": "layer1:1", "tool": "list_documents", "ok": True, "args": {}}],
        "state_before": {"declared_state": {"candidate_set": []}, "observed_state": {
            "surfaced_notes": ["N1", "N2"], "citation_resolutions": [{}, {}],
            "read_notes": [], "search_refs": []}},
        "state_after": {"declared_state": {"candidate_set": []}, "observed_state": {
            "surfaced_notes": ["N1", "N2", "N3"], "citation_resolutions": [{}, {}],
            "read_notes": [], "search_refs": []}},
        "observations": [{"event_ref": "layer1:1", "result": {
            "documents": [{"note_id": "N1", "date": "2020-01-01"},
                          {"note_id": "N2", "date": "2022-02-02"}],
            "citation_resolutions": [{"ref": "instruction:inventory",
                                      "status": "CLAIMED_AND_VERIFIED"}],
            "decision_receipt": {"schema": "receipt.v1", "receipt_id": "r1",
                                 "receipt_hash": "h1", "testimony_ref": "decision:1",
                                 "server_facts": {"verified_input_refs": ["policy:1"]}},
            "total": 2,
        }}],
    }

    projected = model_input_projection([cycle])["cycles"][0]

    assert projected["state_before"]["observed_state"]["surfaced_note_count"] == 2
    assert projected["state_after"]["observed_state"]["surfaced_note_count"] == 3
    result = projected["observations"][0]["result"]
    assert "documents" not in result and "decision_receipt" not in result
    assert result["documents_summary"] == {
        "count": 2, "first_date": "2020-01-01", "last_date": "2022-02-02"}
    assert result["decision_receipt_summary"]["verified_input_refs"] == ["policy:1"]
    assert "instruction:inventory" in model_input_projection([cycle])["available_source_refs"]


def test_retrieval_boundaries_are_deterministic_after_inventory_pagination():
    cycles = [
        {"cycle_id": "c1", "actions": [{"tool": "note_decision", "ok": True}]},
        {"cycle_id": "c2", "actions": [{"tool": "search", "ok": True, "args": {}}]},
        {"cycle_id": "c3", "actions": [{"tool": "list_documents", "ok": True,
                                           "args": {"offset": 200}}]},
        {"cycle_id": "c4", "actions": [{"tool": "read", "ok": True,
                                           "args": {"note_id": "N1"}}]},
        {"cycle_id": "c5", "actions": [{"tool": "read", "ok": True,
                                           "args": {"note_id": "N2"}}]},
    ]
    annotations = {
        "c1": {"role": "DECISION_BEARING", "decision_function": "where_to_look"},
        "c2": {"role": "DECISION_SUPPORT", "decision_function": None},
        "c3": {"role": "DECISION_SUPPORT", "decision_function": None},
        "c4": {"role": "DECISION_SUPPORT", "decision_function": None},
        "c5": {"role": "DECISION_SUPPORT", "decision_function": None},
    }

    canonical, changes = _canonical_annotations(cycles, annotations)

    assert canonical == {
        "c1": {"role": "DECISION_BEARING", "decision_function": "where_to_look"},
        "c2": {"role": "DECISION_SUPPORT", "decision_function": None},
        "c3": {"role": "MECHANICAL", "decision_function": None},
        "c4": {"role": "DECISION_BEARING", "decision_function": "where_to_look"},
        "c5": {"role": "DECISION_SUPPORT", "decision_function": None},
    }
    assert [(row["cycle_id"], row["field"], row["canonical_value"])
            for row in changes] == [
        ("c3", "role", "MECHANICAL"),
        ("c4", "role", "DECISION_BEARING"),
        ("c4", "decision_function", "where_to_look"),
    ]


def test_retrieval_subject_uses_the_first_executed_modality_not_any_later_one():
    by_id = {
        "search-first": {"actions": [
            {"tool": "note_decision", "args": {}},
            {"tool": "search", "args": {"query": "cancer"}},
            {"tool": "list_documents", "args": {"offset": 200}},
        ]},
        "inventory-first": {"actions": [
            {"tool": "list_documents", "args": {}},
            {"tool": "search", "args": {"query": "cancer"}},
        ]},
        "read-first": {"actions": [
            {"tool": "read", "args": {"note_id": "N1"}},
            {"tool": "search", "args": {"query": "cancer"}},
        ]},
    }

    assert _retrieval_subject(["search-first"], by_id) == "retrieval_query_batch"
    assert _retrieval_subject(["inventory-first"], by_id) == "retrieval_inventory"
    assert _retrieval_subject(["read-first"], by_id) == "retrieval_document_set"


def test_unwitnessed_retrieval_outcome_comes_from_execution_not_reader_wording():
    by_id = {
        "c1": {"actions": [{"tool": "read", "ok": True,
                              "event_ref": "layer1:10", "args": {"note_id": "N1"}}]},
        "c2": {"actions": [{"tool": "read", "ok": True,
                              "event_ref": "layer1:11", "args": {"note_id": "N2"}}]},
    }

    assert _retrieval_decision(["c1", "c2"], by_id, "retrieval_document_set") == (
        "Open selected document set: N1, N2", ["layer1:10", "layer1:11"])


def test_each_successful_record_finding_is_an_independent_standing_decision():
    _, cycles = _cycles()
    raw = _raw(cycles)
    cycles[1]["actions"] = [{
        "event_ref": "layer1:3", "tool": "record_finding", "ok": True,
        "args": {"field": "date", "standing": "can_establish"},
    }]

    with pytest.raises(ReconstructionError, match="separate standing decision"):
        verify_extraction(raw, cycles, analysis_id="analysis-a")


def test_explicit_finding_reference_becomes_a_deterministic_influenced_edge():
    finding_cycle = {
        "cycle_id": "c1", "actions": [{"event_ref": "layer1:1",
        "tool": "record_finding", "ok": True, "args": {}}],
        "observations": [{"event_ref": "layer1:1", "result": {
            "recorded": True, "finding_ref": "finding:1"}}],
    }
    resolution_cycle = {
        "cycle_id": "c2", "actions": [{"event_ref": "layer1:2",
        "tool": "note_decision", "ok": True,
        "args": {"cited_refs": ["finding:1"]}}],
        "observations": [{"event_ref": "layer1:2", "result": {
            "citation_resolutions": [{"ref": "finding:1", "verified": True}]}}],
    }
    episodes = [
        {"episode_id": "a:episode:1", "source_cycle_ids": ["c1"]},
        {"episode_id": "a:episode:2", "source_cycle_ids": ["c2"]},
    ]

    edges = _runtime_reference_dependencies(
        episodes, {"c1": finding_cycle, "c2": resolution_cycle}, analysis_id="a")

    assert edges == [{
        "assertion_id": "a:runtime-dependency:1",
        "source_episode_id": "a:episode:1",
        "target_episode_id": "a:episode:2",
        "relationship_type": "INFLUENCED",
        "evidence_refs": ["finding:1"],
        "reasoning": (
            "The target decision explicitly cited runtime reference(s) produced by the source "
            "decision's execution: finding:1."),
        "provenance": "DETERMINISTIC_DERIVED_FROM_RUNTIME_REFERENCE",
    }]


def test_explicit_decision_reference_closes_comparison_to_submit_dependency():
    comparison_cycle = {
        "cycle_id": "c1",
        "actions": [{"event_ref": "layer1:1", "tool": "note_decision", "ok": True,
                     "args": {"cited_refs": ["finding:1"]}}],
        "observations": [{"event_ref": "layer1:1", "result": {
            "noted": True, "testimony_ref": "decision:1",
            "citation_resolutions": [{"ref": "finding:1", "verified": True}]}}],
    }
    submit_choice_cycle = {
        "cycle_id": "c2",
        "actions": [{"event_ref": "layer1:2", "tool": "note_decision", "ok": True,
                     "args": {"cited_refs": ["decision:1"]}}],
        "observations": [{"event_ref": "layer1:2", "result": {
            "noted": True, "testimony_ref": "decision:2",
            "citation_resolutions": [{"ref": "decision:1", "verified": True}]}}],
    }
    episodes = [
        {"episode_id": "a:episode:1", "source_cycle_ids": ["c1"]},
        {"episode_id": "a:episode:2", "source_cycle_ids": ["c2"]},
    ]

    edges = _runtime_reference_dependencies(
        episodes, {"c1": comparison_cycle, "c2": submit_choice_cycle}, analysis_id="a")

    assert [(row["source_episode_id"], row["target_episode_id"], row["evidence_refs"])
            for row in edges] == [("a:episode:1", "a:episode:2", ["decision:1"])]


def test_runtime_references_form_retrieval_to_standing_to_resolution_chain():
    search_cycle = {
        "cycle_id": "c1",
        "actions": [
            {"event_ref": "layer1:1", "tool": "search", "ok": True,
             "args": {"query": "malignan"}},
            {"event_ref": "layer1:2", "tool": "search", "ok": True,
             "args": {"query": "carcinoma"}},
        ],
        "observations": [
            {"event_ref": "layer1:1", "result": {"hits": [{"note_id": "N1"}], "n": 1}},
            {"event_ref": "layer1:2", "result": {"hits": [{"note_id": "N1"}], "n": 1}},
        ],
    }
    select_cycle = {
        "cycle_id": "c2",
        "actions": [
            {"event_ref": "layer1:3", "tool": "note_decision", "ok": True,
             "args": {"cited_refs": ["search:malignan", "search:carcinoma"]}},
            {"event_ref": "layer1:4", "tool": "read", "ok": True,
             "args": {"note_id": "N1"}},
        ],
        "observations": [
            {"event_ref": "layer1:3", "result": {"citation_resolutions": [
                {"ref": "search:malignan", "verified": True},
                {"ref": "search:carcinoma", "verified": True},
            ]}},
            {"event_ref": "layer1:4", "result": {"note_id": "N1", "text": "evidence"}},
        ],
    }
    standing_cycle = {
        "cycle_id": "c3",
        "actions": [{"event_ref": "layer1:5", "tool": "record_finding", "ok": True,
                     "args": {"cited_refs": ["note:N1"]}}],
        "observations": [{"event_ref": "layer1:5", "result": {
            "recorded": True, "finding_ref": "finding:1",
            "citation_resolutions": [{"ref": "note:N1", "verified": True}]}}],
    }
    resolution_cycle = {
        "cycle_id": "c4",
        "actions": [{"event_ref": "layer1:6", "tool": "note_decision", "ok": True,
                     "args": {"cited_refs": ["finding:1"]}}],
        "observations": [{"event_ref": "layer1:6", "result": {
            "citation_resolutions": [{"ref": "finding:1", "verified": True}]}}],
    }
    episodes = [
        {"episode_id": "a:episode:1", "source_cycle_ids": ["c1"]},
        {"episode_id": "a:episode:2", "source_cycle_ids": ["c2"]},
        {"episode_id": "a:episode:3", "source_cycle_ids": ["c3"]},
        {"episode_id": "a:episode:4", "source_cycle_ids": ["c4"]},
    ]

    edges = _runtime_reference_dependencies(
        episodes, {cycle["cycle_id"]: cycle for cycle in (
            search_cycle, select_cycle, standing_cycle, resolution_cycle)},
        analysis_id="a")

    assert [(row["source_episode_id"], row["target_episode_id"], row["evidence_refs"])
            for row in edges] == [
        ("a:episode:1", "a:episode:2", ["search:carcinoma", "search:malignan"]),
        ("a:episode:2", "a:episode:3", ["note:N1"]),
        ("a:episode:3", "a:episode:4", ["finding:1"]),
    ]
    assert all(row["relationship_type"] == "INFLUENCED" for row in edges)
    assert all(row["provenance"] == "DETERMINISTIC_DERIVED_FROM_RUNTIME_REFERENCE"
               for row in edges)


def test_temporal_adjacency_without_a_runtime_reference_creates_no_dependency():
    cycles = {
        "c1": {"cycle_id": "c1", "actions": [
            {"event_ref": "layer1:1", "tool": "search", "ok": True,
             "args": {"query": "pathology"}}],
            "observations": [{"event_ref": "layer1:1", "result": {"hits": [], "n": 0}}]},
        "c2": {"cycle_id": "c2", "actions": [
            {"event_ref": "layer1:2", "tool": "note_decision", "ok": True,
             "args": {"cited_refs": []}}], "observations": []},
    }
    episodes = [
        {"episode_id": "a:episode:1", "source_cycle_ids": ["c1"]},
        {"episode_id": "a:episode:2", "source_cycle_ids": ["c2"]},
    ]

    assert _runtime_reference_dependencies(
        episodes, cycles, analysis_id="a") == []


def test_episode_rejects_two_material_choices_even_when_their_function_matches():
    _, cycles = _cycles()
    raw = _raw(cycles)
    second = cycles[1]["cycle_id"]
    raw["cycle_annotations"][second] = {
        "role": "DECISION_BEARING", "decision_function": "where_to_look"}

    with pytest.raises(ReconstructionError, match="one row per decision-bearing cycle"):
        verify_extraction(raw, cycles, analysis_id="analysis-a")


def test_episode_begins_at_its_material_choice_not_at_a_support_action():
    _, cycles = _cycles()
    raw = _raw(cycles)
    first, second = cycles[0]["cycle_id"], cycles[1]["cycle_id"]
    raw["cycle_annotations"][first] = {
        "role": "DECISION_SUPPORT", "decision_function": None}
    raw["cycle_annotations"][second] = {
        "role": "DECISION_BEARING", "decision_function": "where_to_look"}

    with pytest.raises(ReconstructionError, match="has no preceding decision-bearing cycle"):
        verify_extraction(raw, cycles, analysis_id="analysis-a")


def test_execution_derives_retrieval_subject_instead_of_trusting_reader_label():
    _, cycles = _cycles()
    raw = _raw(cycles)
    raw["episodes"][0]["decision_subject"] = "answer_selection"

    verified = verify_extraction(raw, cycles, analysis_id="analysis-a")

    episode = verified["episodes"][0]
    assert episode["decision_subject"] == "retrieval_query_batch"
    assert episode["field_provenance"]["decision_subject"] == \
        "DETERMINISTIC_DERIVED_FROM_EXECUTION"


def test_singleton_subject_is_derived_from_the_decision_function():
    _, cycles = _cycles()
    raw = _raw(cycles, function="standing")
    raw["episodes"][0]["decision_subject"] = "answer_selection"

    verified = verify_extraction(raw, cycles, analysis_id="analysis-a")

    episode = verified["episodes"][0]
    assert episode["decision_subject"] == "evidence_item"
    assert episode["field_provenance"]["decision_subject"] == \
        "DETERMINISTIC_DERIVED_FROM_DECISION_FUNCTION"


def test_retrieving_one_specific_note_can_act_on_an_evidence_item():
    _, cycles = _cycles()
    cycles[1]["actions"] = []
    raw = _raw(cycles)
    raw["episodes"][0]["decision_subject"] = "evidence_item"

    verified = verify_extraction(raw, cycles, analysis_id="analysis-a")

    assert verified["episodes"][0]["decision_function"] == "where_to_look"
    assert verified["episodes"][0]["decision_subject"] == "evidence_item"


def test_other_function_keeps_a_known_subject_instead_of_discarding_it():
    _, cycles = _cycles()
    raw = _raw(cycles)
    first = cycles[0]["cycle_id"]
    raw["cycle_annotations"][first]["decision_function"] = "other"
    raw["episodes"][0]["decision_function"] = "other"
    raw["episodes"][0]["decision_subject"] = "evidence_item"

    verified = verify_extraction(raw, cycles, analysis_id="analysis-a")

    assert verified["episodes"][0]["decision_function"] == "other"
    assert verified["episodes"][0]["decision_subject"] == "evidence_item"


def test_verified_episodes_keep_runtime_testimony_and_reconstruction_separate():
    _, cycles = _cycles()
    verified = verify_extraction(_raw(cycles), cycles, analysis_id="analysis-a")
    first = verified["episodes"][0]
    assert first["episode_id"] == "analysis-a:episode:1"
    assert first["field_provenance"]["decision"] == "SELF_REPORTED"
    assert first["field_provenance"]["scenario"] == "MODEL_RECONSTRUCTED"
    assert first["field_provenance"]["observed_downstream_refs"] == "DETERMINISTIC_DERIVED"
    assert verified["mechanical_cycle_ids"] == [cycles[-1]["cycle_id"]]
    assert "big_points" not in verified and "small_points" not in verified
    assert verified["causal_assertions"][0]["relationship_type"] == "INFLUENCED"


def test_runtime_testimony_anchors_fields_the_reader_must_not_rewrite():
    _, cycles = _cycles()
    raw = _raw(cycles)
    raw["episodes"][0].update({
        "material_question": "reader paraphrase",
        "candidate_set": ["reader candidate"],
        "decision": "reader changed the choice",
        "decision_rationale": "reader invented a rationale",
        "claimed_basis_summary": ["reader guess"],
        "verified_reference_summary": ["reader guess"],
    })

    verified = verify_extraction(raw, cycles, analysis_id="analysis-a")
    episode = verified["episodes"][0]

    assert episode["material_question"] == "no candidate has been surfaced"
    assert episode["candidate_set"] == ["search pathology"]
    assert episode["decision"] == "search pathology"
    assert episode["decision_rationale"] == "pathology can establish the field"
    assert episode["claimed_basis_summary"] == ["chart"]
    assert episode["verified_reference_summary"] == []
    assert episode["runtime_testimony_ref"] == "decision:2"
    assert all(episode["field_provenance"][field] == "SELF_REPORTED" for field in (
        "material_question", "candidate_set", "decision", "decision_rationale",
        "claimed_basis_summary"))
    assert episode["field_provenance"]["verified_reference_summary"] == "SERVER_FACT"
    assert "decision:2" in episode["source_refs_by_field"]["decision"]


def test_sealed_receipt_fixes_decision_boundary_but_not_post_run_taxonomy():
    _, cycles = _cycles()
    receipt = _seal_first_receipt(cycles)
    raw = _raw(cycles)
    first = cycles[0]["cycle_id"]
    raw["cycle_annotations"][first] = {
        "role": "DECISION_SUPPORT", "decision_function": None}

    with pytest.raises(ReconstructionError, match="sealed runtime Decision Receipt"):
        verify_extraction(raw, cycles, analysis_id="analysis-a")

    raw = _raw(cycles)
    verified = verify_extraction(raw, cycles, analysis_id="analysis-a")
    episode = verified["episodes"][0]
    assert episode["decision_function"] == "where_to_look"  # post-run projection
    assert episode["runtime_receipt_ref"] == receipt["receipt_id"]
    assert episode["runtime_testimony_ref"] == receipt["testimony_ref"]
    assert episode["runtime_receipt_hash"] == receipt["receipt_hash"]
    assert episode["runtime_receipt_provenance"] == "SERVER_SEALED_RUNTIME_RECEIPT"
    assert "decision_function" not in receipt and "category" not in receipt


def test_non_bearing_submit_is_canonical_support_for_the_preceding_commitment():
    _, cycles = _cycles()
    raw = _raw(cycles)
    submit_id = cycles[2]["cycle_id"]
    raw["cycle_annotations"][submit_id] = {
        "role": "MECHANICAL", "decision_function": None}
    raw["episodes"] = raw["episodes"][:1]
    raw["episodes"][0]["source_cycle_ids"].append(submit_id)
    raw["causal_links"] = []

    verified = verify_extraction(raw, cycles, analysis_id="analysis-a")

    assert verified["episodes"][0]["source_cycle_ids"] == [
        cycles[0]["cycle_id"], cycles[1]["cycle_id"], submit_id]
    assert verified["mechanical_cycle_ids"] == [cycles[3]["cycle_id"]]
    assert verified["annotation_normalizations"] == [{
        "cycle_id": submit_id,
        "field": "role",
        "reader_value": "MECHANICAL",
        "canonical_value": "DECISION_SUPPORT",
        "reason": "successful submit_answer executes the preceding commitment",
        "provenance": "DETERMINISTIC_DERIVED",
    }]


def test_downstream_field_source_index_is_canonicalized_deterministically():
    _, cycles = _cycles()
    raw = _raw(cycles)
    raw["episodes"][0]["source_refs_by_field"]["observed_downstream_refs"] = []

    verified = verify_extraction(raw, cycles, analysis_id="analysis-a")

    assert verified["episodes"][0]["source_refs_by_field"][
        "observed_downstream_refs"] == ["layer1:4"]
    assert verified["episodes"][0]["field_provenance"][
        "observed_downstream_refs"] == "DETERMINISTIC_DERIVED"


def test_current_or_prior_downstream_refs_are_discarded_and_audited():
    _, cycles = _cycles()
    raw = _raw(cycles)
    raw["episodes"][0]["observed_downstream_refs"] = ["layer1:3", "layer1:4"]

    verified = verify_extraction(raw, cycles, analysis_id="analysis-a")
    episode = verified["episodes"][0]

    assert episode["observed_downstream_refs"] == ["layer1:4"]
    assert episode["source_refs_by_field"]["observed_downstream_refs"] == ["layer1:4"]
    assert episode["discarded_reconstruction_claims"] == {
        "observed_downstream_refs": ["layer1:3"],
        "reason": "not a real event after this episode",
        "provenance": "DETERMINISTIC_DERIVED",
    }


def test_redundant_mechanical_and_support_ownership_is_canonicalized():
    _, cycles = _cycles()
    raw = _raw(cycles)
    raw["mechanical_cycle_ids"] = []

    verified = verify_extraction(raw, cycles, analysis_id="a")

    assert verified["mechanical_cycle_ids"] == [cycles[-1]["cycle_id"]]
    assert verified["episodes"][0]["source_cycle_ids"] == [
        cycles[0]["cycle_id"], cycles[1]["cycle_id"]]


def test_episode_must_still_name_its_corresponding_bearing_cycle_first():
    _, cycles = _cycles()

    raw = _raw(cycles)
    raw["episodes"][0]["source_cycle_ids"] = [
        cycles[0]["cycle_id"], cycles[2]["cycle_id"]]
    raw["episodes"][1]["source_cycle_ids"] = [cycles[1]["cycle_id"]]
    with pytest.raises(ReconstructionError, match="canonical contiguous envelope"):
        verify_extraction(raw, cycles, analysis_id="a")


def test_scenario_sources_cannot_include_the_episode_observation():
    _, cycles = _cycles()
    raw = _raw(cycles)
    raw["episodes"][0]["source_refs_by_field"]["scenario"] = ["layer1:3"]
    with pytest.raises(ReconstructionError, match="pre-decision state"):
        verify_extraction(raw, cycles, analysis_id="a")


class StubLLM:
    def __init__(self, replies):
        self.replies = replies
        self.calls = []

    def generate_structured(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.replies[len(self.calls) - 1]


class ProjectionSink:
    def __init__(self):
        self.artifacts = []

    def project_analysis(self, artifact):
        self.artifacts.append(artifact)

    def selected_analysis(self, _run_id):
        return None


def test_multiple_passes_are_append_only_and_drift_prevents_implicit_selection(tmp_path: Path):
    review, cycles = _cycles()
    first = _raw(cycles)
    second = _raw(cycles, function="standing")
    second["episodes"][0]["decision_function"] = "standing"
    sink, llm = ProjectionSink(), StubLLM([first, second])
    summary = reconstruct_run(review, sink, llm, passes=2, artifact_dir=tmp_path)

    assert len(summary["analyses"]) == 2 and len(sink.artifacts) == 2
    assert summary["drift"]["alignment_agrees"] is False
    assert summary["selected_analysis_id"] is None
    assert all(row["stability_status"] == "PROVISIONAL_DRIFT"
               for row in summary["analyses"])
    assert len(list(tmp_path.glob("*.json"))) == 2
    for path in tmp_path.glob("*.json"):
        stored = json.loads(path.read_text(encoding="utf-8"))
        assert stored["artifact_ref"] == str(path.resolve())
        hashed = {key: value for key, value in stored.items()
                  if key not in {"analysis_artifact_hash", "artifact_ref"}}
        assert stored["analysis_artifact_hash"] == content_hash(hashed)
    assert "fixed ReAct Cycle" in build_prompt(cycles)


def test_reconstruction_preserves_review_model_and_task_arm_for_cohort_comparison():
    review, cycles = _cycles()
    review.review_model = "openai/gpt-5.6-luna"
    review.task_arm = "task_only"
    sink, llm = ProjectionSink(), StubLLM([_raw(cycles)])

    reconstruct_run(review, sink, llm, passes=1)

    assert sink.artifacts[0]["review_model"] == "openai/gpt-5.6-luna"
    assert sink.artifacts[0]["task_arm"] == "task_only"
    assert sink.artifacts[0]["taxonomy_version"] == DECISION_TAXONOMY_SCHEMA
    assert sink.artifacts[0]["runtime_receipt_manifest"]["mode"] == "LEGACY"
    assert sink.artifacts[0]["episodes"][0]["runtime_receipt_ref"] is None
    assert sink.artifacts[0]["decision_receipt_coverage"] == {
        "status": "NO_SEALED_DECISIONS", "episode_count": 2,
        "sealed_episode_count": 0, "legacy_testimony_episode_count": 1,
        "reconstructed_without_testimony_count": 1,
        "sealed_episode_ids": [],
        "unsealed_episode_ids": [
            sink.artifacts[0]["episodes"][0]["episode_id"],
            sink.artifacts[0]["episodes"][1]["episode_id"],
        ],
        "provenance": "DETERMINISTIC_DERIVED",
    }


def test_pass_stability_uses_witnessed_choice_not_reader_paraphrase(tmp_path: Path):
    review, cycles = _cycles()
    first = _raw(cycles)
    second = json.loads(json.dumps(first))
    second["episodes"][0].update({
        "decision_subject": "retrieval_source",
        "material_question": "different reader wording",
        "decision": "different reader wording",
        "decision_rationale": "different reader wording",
        "candidate_set": ["different reader wording"],
    })
    sink, llm = ProjectionSink(), StubLLM([first, second])

    summary = reconstruct_run(review, sink, llm, passes=2, artifact_dir=tmp_path)

    assert summary["drift"]["alignment_agrees"] is True
    assert all(row["stability_status"] == "STABLE_ACROSS_PASSES"
               for row in summary["analyses"])
    assert summary["drift"]["alignments"][0] == summary["drift"]["alignments"][1]
    assert summary["drift"]["alignments"][0][0] == {
        "bearing_cycle_id": cycles[0]["cycle_id"],
        "decision_function": "where_to_look",
        "decision_subject": "retrieval_query_batch",
        "decision": "search pathology",
    }


def test_validator_feedback_retries_without_projecting_the_rejected_candidate():
    review, cycles = _cycles()
    rejected = _raw(cycles)
    rejected["episodes"][0]["source_refs_by_field"]["scenario"] = [
        f"state_before:{cycles[0]['cycle_id']}", "layer1:3"]
    accepted = _raw(cycles)
    sink, llm = ProjectionSink(), StubLLM([rejected, accepted])

    summary = reconstruct_run(
        review, sink, llm, passes=1, max_attempts_per_pass=2)

    assert len(llm.calls) == 2
    assert "CORRECTION ATTEMPT" in llm.calls[1][0]
    assert "scenario may cite only pre-decision state/events" in llm.calls[1][0]
    assert len(sink.artifacts) == 1
    attempts = sink.artifacts[0]["reconstructor_attempts"]
    assert [attempt["validation_status"] for attempt in attempts] == ["REJECTED", "ACCEPTED"]
    assert attempts[0]["raw_hash"] and "validation_error" in attempts[0]
    assert summary["analyses"][0]["n_attempts"] == 2
