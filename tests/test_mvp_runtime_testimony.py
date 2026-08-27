"""Runtime testimony records claims; the server records whether their references exist."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from acr.contract.spec import load_spec
from acr.mvp.decision_receipts import receipt_from_action
from acr.mvp.runner import TASK_PREAMBLE
from acr.mvp.task_presentation import (
    CLAIMED_AND_VERIFIED,
    CLAIMED_NOT_OFFERED,
    build_task_presentation,
)
from acr.mvp.toolserver import TOOL_SCHEMA_INSTRUCTIONS, ChartToolServer
from acr.mvp.warrants import BASIS_SOURCES, RULE_COVERAGE_CLAIMS

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"
PATIENT = ROOT / "corpus" / "patients" / "SYN0001"


def _decision(**over):
    value = {
        "facing": "two candidate diagnosis dates remain",
        "decision": "check whether the earlier cytology can establish the field",
        "because": "the offered first-date rule makes its standing discriminating",
        "basis_sources": ["task_contract", "chart"],
        "cited_refs": ["decision_rule.1"],
        "checked_discriminating_fact_refs": [
            "discriminating_fact.impression_at_ambiguous_cytology"],
        "rule_coverage_claim": "COVERED_WITH_INTERPRETATION",
        "provisional_inference": "the ambiguous wording may need corroboration",
        "alternatives": ["use the later biopsy"],
        "uncertainty": "standing of the cytology is unresolved",
    }
    value.update(over)
    return value


@pytest.fixture
def server(tmp_path: Path) -> ChartToolServer:
    return ChartToolServer(SPEC, PATIENT, tmp_path / "run")


def test_testimony_schema_requires_the_auditable_claims(server: ChartToolServer):
    schema = next(row for row in server.schemas() if row["name"] == "note_decision")
    props = schema["inputSchema"]["properties"]
    assert set(schema["inputSchema"]["required"]) == {
        "facing", "decision", "because", "basis_sources", "cited_refs",
        "checked_discriminating_fact_refs", "rule_coverage_claim",
    }
    assert set(props["basis_sources"]["items"]["enum"]) == set(BASIS_SOURCES)
    assert set(props["rule_coverage_claim"]["enum"]) == set(RULE_COVERAGE_CLAIMS)
    assert "grounding" not in props and "used" not in props

    finding = next(row for row in server.schemas() if row["name"] == "record_finding")
    assert set(finding["inputSchema"]["required"]) == {
        "note_id", "field", "standing", "assertion_class", "facing", "because",
        "basis_sources", "cited_refs", "checked_discriminating_fact_refs",
        "rule_coverage_claim",
    }
    assert "decision_testimony_ref" not in finding["inputSchema"]["properties"]


def test_runtime_prompt_records_choices_not_every_tool_call_or_combined_replans():
    assert "one independently auditable choice" in TASK_PREAMBLE
    assert "One precommitted batch of keywords" in TASK_PREAMBLE
    assert "a new Decision Testimony" in TASK_PREAMBLE
    assert "Opening every note in an already chosen read-all set is execution" in TASK_PREAMBLE
    assert "record_finding is itself that atomic" in TASK_PREAMBLE
    assert "Do not combine several notes' standings" in TASK_PREAMBLE
    assert "cite the exact finding:N" in TASK_PREAMBLE


def test_material_tool_schema_instruction_has_a_stable_citation_id(
        server: ChartToolServer):
    schema = next(row for row in server.schemas() if row["name"] == "list_documents")
    assert TOOL_SCHEMA_INSTRUCTIONS["list_documents_inventory_gate"] in schema["description"]
    assert "instruction:list_documents_inventory_gate" in schema["description"]
    resolved = server.task_presentation.resolve_asset(
        "instruction:list_documents_inventory_gate")
    assert resolved["status"] == CLAIMED_AND_VERIFIED


def test_testimony_keeps_claims_separate_from_exact_reference_resolution(
        server: ChartToolServer):
    payload, is_error = server.call("note_decision", _decision())
    assert not is_error and payload["noted"]
    assert payload["basis_sources"] == ["task_contract", "chart"]
    assert payload["citation_resolutions"][0]["status"] == CLAIMED_AND_VERIFIED
    assert payload["checked_fact_resolutions"][0]["status"] == CLAIMED_AND_VERIFIED
    assert payload["testimony_ref"].startswith("decision:")
    assert "decision_receipt" not in payload  # instrumentation is not echoed to the agent

    event = json.loads(server.trace_path.read_text(encoding="utf-8").splitlines()[-1])
    assert event["args"]["rule_coverage_claim"] == "COVERED_WITH_INTERPRETATION"
    assert event["result"]["citation_resolutions"] == payload["citation_resolutions"]
    receipt = receipt_from_action(
        {"event_ref": f"layer1:{event['seq']}", "tool": event["tool"],
         "args": event["args"]}, event["result"])
    assert receipt is not None
    assert receipt["receipt_id"] != receipt["testimony_ref"]
    assert receipt["testimony"]["selected"] == _decision()["decision"]
    assert receipt["server_facts"]["state_at_recording"]["n_searches"] == 0


def test_discriminating_fact_is_verified_in_cited_refs_as_well_as_checked_slot(
        server: ChartToolServer):
    ref = "discriminating_fact.impression_at_ambiguous_cytology"

    payload, is_error = server.call(
        "note_decision", _decision(cited_refs=[ref]))

    assert not is_error
    assert payload["citation_resolutions"][0]["ref"] == ref
    assert payload["citation_resolutions"][0]["status"] == CLAIMED_AND_VERIFIED
    assert payload["checked_fact_resolutions"][0]["status"] == CLAIMED_AND_VERIFIED


def test_requirements_only_marks_a_real_but_hidden_rule_not_offered(tmp_path: Path):
    spec = load_spec(SPEC)
    _, snapshot = build_task_presentation(
        spec, run_id="run", arm_id="requirements_only", operational_preamble="preamble")
    path = snapshot.write(tmp_path)
    server = ChartToolServer(SPEC, PATIENT, tmp_path / "run",
                             task_presentation_path=path)
    payload, is_error = server.call("note_decision", _decision())
    assert not is_error
    assert payload["citation_resolutions"][0]["status"] == CLAIMED_NOT_OFFERED
    assert payload["checked_fact_resolutions"][0]["status"] == CLAIMED_NOT_OFFERED


def test_runtime_note_finding_enforces_the_standing_span_contract(server: ChartToolServer):
    hit, _ = server.call("search", {"query": "adenocarcinoma"})
    note_id = hit["hits"][0]["note_id"]
    server.call("read", {"note_id": note_id})

    bad, is_error = server.call("record_finding", {
        "note_id": note_id, "field": "date_of_initial_diagnosis",
        "standing": "can_establish", "assertion_class": "pathology_diagnosis",
    })
    assert is_error and "source_start" in bad["error"]

    start, end = hit["hits"][0]["start"], hit["hits"][0]["end"]
    found, is_error = server.call("record_finding", {
        "note_id": note_id, "field": "date_of_initial_diagnosis",
        "standing": "can_establish", "assertion_class": "pathology_diagnosis",
        "source_start": start, "source_end": end,
        "decision_testimony_ref": "decision:3",
    })
    assert not is_error
    assert found["server_fact"]["span_resolved"] is True
    assert found["self_reported"]["standing"] == "can_establish"
    assert found["self_reported"]["assertion_class"] == "pathology_diagnosis"
    assert found["quote"] == "adenocarcinoma"
    assert found["instrumentation_status"] == "LEGACY_SHARED_TESTIMONY"


def test_record_finding_is_a_self_contained_atomic_testimony_and_dependency(
        server: ChartToolServer):
    hit, _ = server.call("search", {"query": "adenocarcinoma"})
    note_id = hit["hits"][0]["note_id"]
    server.call("read", {"note_id": note_id})
    start, end = hit["hits"][0]["start"], hit["hits"][0]["end"]

    found, is_error = server.call("record_finding", {
        "note_id": note_id, "field": "date_of_initial_diagnosis",
        "standing": "can_establish", "assertion_class": "pathology_diagnosis",
        "source_start": start, "source_end": end, "event_time": "2023-04-12",
        "facing": "Can this pathology note establish the diagnosis date?",
        "because": "The quoted diagnostic phrase is definitive under the offered rule.",
        "basis_sources": ["task_contract", "chart"],
        "cited_refs": [f"note:{note_id}", "decision_rule.1"],
        "checked_discriminating_fact_refs": [],
        "rule_coverage_claim": "DIRECTLY_COVERED",
        "provisional_inference": None, "alternatives": ["merely_mentions"],
        "uncertainty": None,
    })

    assert not is_error and found["recorded"] and found["noted"]
    assert found["finding_ref"] == "finding:1"
    assert found["testimony_ref"].startswith("decision:")
    assert found["self_reported"]["decision"].startswith(note_id)
    assert found["citation_resolutions"][0]["verified"] is True
    assert "decision_receipt" not in found

    finding_event = json.loads(
        server.trace_path.read_text(encoding="utf-8").splitlines()[-1])
    receipt = receipt_from_action(
        {"event_ref": f"layer1:{finding_event['seq']}", "tool": finding_event["tool"],
         "args": finding_event["args"]}, finding_event["result"])
    assert receipt is not None
    assert receipt["structured_commitment"]["finding_ref"] == "finding:1"
    # Pre-call state contains the read note, but not the finding being committed now.
    assert receipt["server_facts"]["state_at_recording"]["n_findings"] == 0

    followup, is_error = server.call("note_decision", _decision(
        decision="prefer this finding over a later candidate",
        cited_refs=["finding:1", found["testimony_ref"]],
    ))
    assert not is_error
    assert all(row["verified"] is True for row in followup["citation_resolutions"])
