"""Runtime receipts preserve facts now while leaving semantic taxonomy for later."""
from __future__ import annotations

import copy

import pytest

from acr.mvp.decision_receipts import (
    DecisionReceiptError,
    make_runtime_decision_receipt,
    receipt_from_action,
    validate_runtime_decision_receipt,
)


def _interaction():
    action = {
        "event_ref": "layer1:7", "tool": "note_decision",
        "args": {
            "facing": "Which evidence should be opened next?",
            "decision": "Open the oncology note",
            "because": "It may contain the earlier physician impression",
            "alternatives": ["Open only pathology"],
            "basis_sources": ["chart"],
            "cited_refs": ["search:malignan"],
            "checked_discriminating_fact_refs": [],
            "rule_coverage_claim": "OPERATIONAL_DISCRETION",
            "provisional_inference": None,
            "uncertainty": "The note may only repeat pathology",
        },
    }
    result = {
        "noted": True, "testimony_ref": "decision:7",
        "citation_resolutions": [{
            "ref": "search:malignan", "verified": True,
            "status": "CLAIMED_AND_VERIFIED",
        }],
        "checked_fact_resolutions": [],
    }
    receipt = make_runtime_decision_receipt(
        action["tool"], action["args"], result,
        {"n_searches": 1, "searches_run": ["malignan"]},
        source_event_ref=action["event_ref"],
    )
    assert receipt is not None
    result["decision_receipt"] = receipt
    return action, result, receipt


def test_receipt_is_taxonomy_neutral_and_binds_testimony_to_server_facts():
    action, result, receipt = _interaction()

    assert receipt_from_action(action, result) == receipt
    assert receipt["receipt_id"] == "decision-receipt:7"
    assert receipt["testimony_ref"] == "decision:7"
    assert receipt["receipt_id"] != receipt["testimony_ref"]
    assert receipt["server_facts"]["state_at_recording"]["n_searches"] == 1
    assert receipt["server_facts"]["verified_input_refs"] == ["search:malignan"]
    assert not ({"category", "decision_function", "decision_subject", "taxonomy"}
                & set(receipt))


def test_receipt_rejects_taxonomy_leak_and_hash_tampering():
    _, _, receipt = _interaction()
    leaked = copy.deepcopy(receipt)
    leaked["category"] = "where_to_look"
    with pytest.raises(DecisionReceiptError, match="projection taxonomy"):
        validate_runtime_decision_receipt(leaked)

    tampered = copy.deepcopy(receipt)
    tampered["testimony"]["selected"] = "Open pathology instead"
    with pytest.raises(DecisionReceiptError, match="seal"):
        validate_runtime_decision_receipt(tampered)


@pytest.mark.parametrize("mutation, message", [
    (lambda action, result: action.update(event_ref="layer1:8"), "different event"),
    (lambda action, result: action.update(tool="record_finding"), "different source tool"),
    (lambda action, result: result.update(testimony_ref="decision:8"),
     "testimony differs"),
])
def test_receipt_fails_closed_when_moved_to_another_interaction(mutation, message):
    action, result, _ = _interaction()
    mutation(action, result)
    with pytest.raises(DecisionReceiptError, match=message):
        receipt_from_action(action, result)


def test_finding_receipt_seals_the_structured_commitment_without_assigning_category():
    action = {
        "event_ref": "layer1:9", "tool": "record_finding",
        "args": {
            "note_id": "N1", "field": "diagnosis_date",
            "standing": "can_establish", "assertion_class": "physician_diagnosis",
            "event_time": "2023-04-12", "record_time": "2023-04-12",
            "carried_forward": False, "facing": "Can this note establish the date?",
            "because": "The physician gives a clinical diagnosis.",
            "alternatives": ["merely_mentions"], "basis_sources": ["chart"],
            "cited_refs": ["note:N1"], "checked_discriminating_fact_refs": [],
            "rule_coverage_claim": "NO_APPLICABLE_RULE",
            "provisional_inference": "Clinical diagnosis is sufficient.",
            "uncertainty": None,
        },
    }
    result = {
        "recorded": True, "finding_ref": "finding:1", "testimony_ref": "decision:9",
        "citation_resolutions": [{"ref": "note:N1", "verified": True}],
        "checked_fact_resolutions": [],
    }
    receipt = make_runtime_decision_receipt(
        action["tool"], action["args"], result, {"n_reads": 1},
        source_event_ref=action["event_ref"])
    assert receipt is not None
    result["decision_receipt"] = receipt

    checked = receipt_from_action(action, result)
    assert checked is not None
    assert checked["structured_commitment"] == {
        "schema": "acr.runtime_note_finding_commitment.v1",
        "finding_ref": "finding:1", "note_ref": "note:N1",
        "field": "diagnosis_date", "standing": "can_establish",
        "assertion_class": "physician_diagnosis",
        "event_time": "2023-04-12", "record_time": "2023-04-12",
        "carried_forward": False,
    }
