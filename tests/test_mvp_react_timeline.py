"""Langtrace completeness and deterministic ReAct cycle replay."""
from __future__ import annotations

import json

import pytest

from acr.mvp.decision_receipts import (
    DecisionReceiptError,
    make_runtime_decision_receipt,
)
from acr.mvp.langtrace_io import LangtraceReviewTrace
from acr.mvp.task_presentation import content_hash
from acr.mvp.timeline import (
    IncompleteTraceError,
    build_react_cycles,
    build_trace_completeness,
)


def _events():
    return [
        {"seq": 1, "ts": "t1", "kind": "run_meta", "spec_id": "S", "patient_id": "P"},
        {"seq": 2, "ts": "t2", "kind": "tool_call", "tool": "note_decision", "ok": True,
         "args": {"facing": "where is the qualifying diagnosis", "decision": "search pathology",
                  "because": "no candidates are surfaced", "basis_sources": ["chart"],
                  "cited_refs": [], "checked_discriminating_fact_refs": [],
                  "rule_coverage_claim": "OPERATIONAL_DISCRETION"},
         "result": {"noted": True, "testimony_ref": "decision:2",
                    "citation_resolutions": [], "checked_fact_resolutions": [],
                    "context": {"n_searches": 0, "n_reads": 0, "n_findings": 0}}},
        {"seq": 3, "ts": "t3", "kind": "tool_call", "tool": "search", "ok": True,
         "args": {"query": "adenocarcinoma", "objective": "find pathology"},
         "result": {"n": 1, "hits": [{"note_id": "N1"}],
                    "context": {"n_searches": 1, "n_reads": 0, "n_findings": 0}}},
        {"seq": 4, "ts": "t4", "kind": "tool_call", "tool": "read", "ok": True,
         "args": {"note_id": "N1", "objective": "judge standing"},
         "result": {"note_id": "N1", "total_chars": 20, "returned_chars": 20,
                    "context": {"n_searches": 1, "n_reads": 1, "n_findings": 0}}},
        {"seq": 5, "ts": "t5", "kind": "tool_call", "tool": "record_finding", "ok": True,
         "args": {"note_id": "N1", "field": "diagnosis_date", "standing": "can_establish",
                  "assertion_class": "pathology_diagnosis", "source_start": 0, "source_end": 14,
                  "decision_testimony_ref": "decision:2"},
         "result": {"recorded": True, "finding_ref": "finding:1",
                    "server_fact": {"note_read": True, "span_resolved": True, "span": [0, 14]},
                    "self_reported": {"standing": "can_establish",
                                      "assertion_class": "pathology_diagnosis"}}},
        {"seq": 6, "ts": "t6", "kind": "tool_call", "tool": "submit_answer", "ok": True,
         "args": {"status": "FOUND", "value": {"diagnosis_date": "redacted"}},
         "result": {"accepted": True, "why": "obligations discharged"}},
        {"seq": 7, "ts": "t7", "kind": "answer_accepted", "status": "FOUND"},
    ]


def _review(events=None):
    events = list(events or _events())
    root_attrs = {
        "acr.trace.schema": "acr.langtrace.v2",
        "acr.export.status": "COMPLETE",
        "acr.layer1.event_count": len(_events()),
        "acr.layer1.content_hash": content_hash(_events()),
    }
    spans = [{"spanId": "root", "parentSpanId": None, "attributes": root_attrs}]
    spans += [{"spanId": f"s{event['seq']}", "parentSpanId": "root",
               "attributes": {"acr.layer1.seq": event["seq"],
                              "acr.layer1.event_json": json.dumps(event)}}
              for event in events]
    return LangtraceReviewTrace(
        trace_id="a" * 32, run_id="run-1", patient_id="P", spec_id="S",
        steps=[], layer1_events=events, spans=spans,
        spec_hash="spec-hash", task_arm="detailed", task_presentation_hash="presentation-hash",
    )


def test_complete_export_has_a_reproducible_manifest_and_parent_links():
    review = _review()
    manifest = build_trace_completeness(review)
    assert manifest.export_status == "COMPLETE"
    assert manifest.event_count == 7 and manifest.content_hash == content_hash(_events())
    assert manifest.ordered_event_ids == tuple(f"layer1:{n}" for n in range(1, 8))
    assert ("s2", "root") in manifest.parent_links
    assert build_trace_completeness(review).manifest_hash == manifest.manifest_hash


def test_complete_export_accepts_self_hosted_langtrace_parent_id_shape():
    review = _review()
    for span in review.spans:
        span["span_id"] = span.pop("spanId")
        span["parent_id"] = span.pop("parentSpanId")

    manifest = build_trace_completeness(review)

    assert manifest.export_status == "COMPLETE"
    assert ("s2", "root") in manifest.parent_links


def test_reconstruction_fails_closed_when_langtrace_omits_one_event():
    review = _review(_events()[:-1])
    manifest = build_trace_completeness(review)
    assert manifest.export_status == "INCOMPLETE"
    assert any("event count" in issue for issue in manifest.issues)
    with pytest.raises(IncompleteTraceError, match="event count"):
        build_react_cycles(review, manifest)


def test_each_action_and_observation_occurs_once_and_state_does_not_look_ahead():
    review = _review()
    cycles = build_react_cycles(review, build_trace_completeness(review))
    assert [cycle["source_seq_range"] for cycle in cycles] == [
        [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7]]
    assert [cycle["structural_kind"] for cycle in cycles][-2:] == [
        "SUBMISSION", "TERMINATION"]
    assert cycles[0]["has_decision_testimony"] is True
    assert cycles[0]["declared_open_question"] == "where is the qualifying diagnosis"

    search = cycles[1]
    assert search["state_before"]["observed_state"]["surfaced_notes"] == []
    assert search["observations"][0]["result"]["hits"][0]["note_id"] == "N1"
    assert search["state_after"]["observed_state"]["surfaced_notes"] == ["N1"]

    finding = cycles[3]
    assert finding["state_after"]["observed_state"]["finding_call_refs"] == ["finding:1"]
    assert finding["state_after"]["declared_state"]["findings"][0]["standing"] == "can_establish"
    owned = [ref for cycle in cycles for ref in cycle["source_event_ids"]]
    assert owned == [f"layer1:{n}" for n in range(2, 8)]


def test_atomic_record_finding_is_itself_runtime_testimony():
    events = _events()
    finding = events[4]
    finding["args"].update({
        "facing": "Can N1 establish this field?", "because": "diagnostic language",
        "basis_sources": ["chart"], "cited_refs": ["note:N1"],
        "checked_discriminating_fact_refs": [],
        "rule_coverage_claim": "COVERED_WITH_INTERPRETATION",
        "alternatives": ["merely_mentions"], "uncertainty": None,
    })
    finding["args"].pop("decision_testimony_ref")
    finding["result"].update({
        "testimony_ref": "decision:5", "citation_resolutions": [],
        "checked_fact_resolutions": [], "quote": "adenocarcinoma",
        "self_reported": {
            "decision": "N1 is can_establish for diagnosis_date (pathology_diagnosis)",
            "standing": "can_establish", "assertion_class": "pathology_diagnosis",
        },
    })
    review = _review(events)
    root = review.spans[0]["attributes"]
    root["acr.layer1.event_count"] = len(events)
    root["acr.layer1.content_hash"] = content_hash(events)

    cycles = build_react_cycles(review, build_trace_completeness(review))
    cycle = next(row for row in cycles if row["source_seq_range"] == [5, 5])

    assert cycle["has_decision_testimony"] is True
    assert cycle["decision_testimony_refs"] == ["decision:5"]
    assert cycle["declared_open_question"] == "Can N1 establish this field?"
    assert cycle["state_after"]["declared_state"]["findings"][0]["quote"] == \
        "adenocarcinoma"


def test_timeline_replays_a_valid_receipt_and_fails_closed_on_a_broken_seal():
    events = _events()
    event = events[1]
    receipt = make_runtime_decision_receipt(
        event["tool"], event["args"], event["result"],
        {"n_searches": 0, "n_reads": 0, "n_findings": 0},
        source_event_ref="layer1:2")
    assert receipt is not None
    event["result"]["decision_receipt"] = receipt
    review = _review(events)
    review.spans[0]["attributes"]["acr.layer1.content_hash"] = content_hash(events)

    cycles = build_react_cycles(review, build_trace_completeness(review))
    assert cycles[0]["has_decision_receipt"] is True
    assert cycles[0]["decision_receipt_refs"] == [receipt["receipt_id"]]
    assert cycles[0]["state_after"]["declared_state"]["decision_receipt_refs"] == [
        receipt["receipt_id"]]

    event["result"]["decision_receipt"]["testimony"]["selected"] = "tampered"
    review = _review(events)
    review.spans[0]["attributes"]["acr.layer1.content_hash"] = content_hash(events)
    manifest = build_trace_completeness(review)
    assert manifest.export_status == "COMPLETE"
    with pytest.raises(DecisionReceiptError, match="seal"):
        build_react_cycles(review, manifest)
