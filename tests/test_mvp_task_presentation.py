"""The run can only claim rules that its actual prompt offered."""
from __future__ import annotations

import json
from pathlib import Path

from acr.contract.spec import load_spec
from acr.contract.trace import parse_rule_citations, rule_catalog
from acr.mvp.task_presentation import (
    CLAIMED_AND_VERIFIED,
    CLAIMED_NOT_OFFERED,
    CLAIMED_UNKNOWN,
    build_task_presentation,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"


def test_requirements_only_cannot_be_backed_by_hidden_clinical_rules(tmp_path: Path):
    spec = load_spec(SPEC)
    prompt, snapshot = build_task_presentation(
        spec, run_id="run-r", arm_id="requirements_only",
        operational_preamble="Do the review through chart tools.",
    )

    assert "No decision rules" in prompt
    assert snapshot.resolve_rule("decision_rule.1")["status"] == CLAIMED_NOT_OFFERED
    assert snapshot.resolve_rule("discriminating_fact.impression_at_ambiguous_cytology") == {
        "ref": "discriminating_fact.impression_at_ambiguous_cytology",
        "status": CLAIMED_NOT_OFFERED,
        "kind": "discriminating_fact",
    }
    assert snapshot.resolve_rule("decision_rule.999")["status"] == CLAIMED_UNKNOWN
    assert snapshot.prompt_hash and snapshot.presentation_hash

    path = snapshot.write(tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["presentation_hash"] == snapshot.presentation_hash
    assert loaded["rendered_prompt_artifact_ref"] == "prompt.txt"


def test_task_only_is_the_public_bare_task_arm_with_no_clinical_policy():
    spec = load_spec(SPEC)

    prompt, snapshot = build_task_presentation(
        spec, run_id="run-task-only", arm_id="task_only",
        operational_preamble="Do the review through chart tools.",
    )

    assert snapshot.arm_id == "task_only"
    assert "No decision rules" in prompt
    assert snapshot.offered_clause_catalog
    assert {row["kind"] for row in snapshot.offered_clause_catalog} <= {
        "field_format", "field_allowable_values",
    }
    assert snapshot.resolve_rule("decision_rule.1")["status"] == CLAIMED_NOT_OFFERED
    assert snapshot.resolve_rule("conflict_rule.1")["status"] == CLAIMED_NOT_OFFERED


def test_detailed_arm_verifies_exact_catalog_ids_and_content_hashes():
    spec = load_spec(SPEC)
    prompt, snapshot = build_task_presentation(
        spec, run_id="run-d", arm_id="detailed",
        operational_preamble="Do the review through chart tools.",
    )

    rule = snapshot.resolve_rule("rule:decision_rule.1")
    fact = snapshot.resolve_rule("discriminating_fact.impression_at_ambiguous_cytology")
    assert rule["status"] == CLAIMED_AND_VERIFIED
    assert rule["text_sha"]
    assert fact["status"] == CLAIMED_AND_VERIFIED
    assert "DECISION RULES" in prompt
    assert "EXACT GUIDELINE CITATION IDS" in prompt
    assert "EXACT METHOD / OPERATIONAL CITATION IDS" in prompt
    assert "instruction:chart_review_preamble" in prompt
    assert "conflict_rule.3:" in prompt
    assert "evidence_rule.counts_as_evidence.2:" in prompt
    assert snapshot.resolve_asset("instruction:chart_review_preamble")["status"] == \
        CLAIMED_AND_VERIFIED
    assert snapshot.task_contract_ref["content_hash"] == spec.spec_hash


def test_policy_bundle_arm_compiles_independently_versioned_decision_boundaries():
    spec = load_spec(SPEC)

    prompt, snapshot = build_task_presentation(
        spec, run_id="run-policy", arm_id="policy_bundle",
        operational_preamble="Do the review through chart tools.",
    )

    bundle = snapshot.policy_bundle
    assert bundle and bundle["schema"] == "acr.policy_bundle.v1"
    assert bundle["bundle_id"] == spec.spec_id
    assert bundle["bundle_hash"]
    policies = bundle["policies"]
    assert len(policies) > 10
    assert all(row["policy_id"] and row["version"] and row["content_hash"]
               for row in policies)

    cytology = next(row for row in policies
                    if row["policy_id"].endswith("conflict.impression_at_ambiguous_cytology"))
    assert cytology["category"] == "conflict_resolution"
    assert set(cytology["clause_refs"]) == {
        "discriminating_fact.impression_at_ambiguous_cytology",
        "conflict_rule.1", "conflict_rule.2",
    }

    owners = {}
    for policy in policies:
        for ref in policy["clause_refs"]:
            owners.setdefault(ref, []).append(policy["policy_id"])
    assert all(len(policy_ids) == 1 for policy_ids in owners.values())
    assert set(owners) == {row["rule_id"] for row in snapshot.offered_clause_catalog}
    assert "EXACT GUIDELINE CITATION IDS" in prompt


def test_prompt_or_catalog_tampering_changes_the_content_address():
    spec = load_spec(SPEC)
    _, first = build_task_presentation(
        spec, run_id="same", arm_id="detailed", operational_preamble="A")
    _, second = build_task_presentation(
        spec, run_id="same", arm_id="detailed", operational_preamble="B")
    assert first.prompt_hash != second.prompt_hash
    assert first.presentation_hash != second.presentation_hash


def test_discriminating_fact_ids_are_exactly_parseable_but_use_their_own_slot():
    spec = load_spec(SPEC)
    known = [row.rule_id for row in rule_catalog(spec)]
    good, bad = parse_rule_citations(
        "discriminating_fact.impression_at_ambiguous_cytology "
        "discriminating_fact.made_up", known)
    assert good == ["discriminating_fact.impression_at_ambiguous_cytology"]
    assert bad == ["discriminating_fact.made_up"]
