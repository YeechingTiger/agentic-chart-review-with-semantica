"""Tests that the gate can FAIL.

Every other gate test here asserts that a well-behaved run is allowed through. None of them
would notice if the gate quietly stopped checking anything — and that is exactly what
happened once already: restructuring the spec into strata emptied the three fields the old
flat checker read (`required_coverage`, `required_keywords`, `required_doc_types`), so it
looped over empty lists and returned `satisfied=True`. The run reported a satisfied proof
obligation having verified nothing, and no test went red.

A silent no-op looks identical to a clean pass. So the assertions below are all of the form
"this must be refused", which is the only shape that catches a checker that has stopped
working.

The same trap sits one level up: `graph.py` defaults an unset status to
EVIDENCE_INSUFFICIENT, and for a patient whose ground truth IS EVIDENCE_INSUFFICIENT a
completely inert agent scores correct. That is why SYN0002 is judged on the path it took,
not on its final label — see test_ablation_needs_a_found_case.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from acr.corpus import Corpus
from acr.coverage import (CoverageLedger, ForcedSampler, evaluate_gate, strata_from_spec)
from acr.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
STRATIFIED = ROOT / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"
UNSTRATIFIED = ROOT / "specs" / "ablation" / "STORE.400_522_523.unstratified.yaml"


def _fresh_ledger(spec, pid="SYN0002"):
    docs, _ = Corpus(ROOT / "corpus" / "patients").chart(pid).list_documents(limit=100_000)
    return CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(1234))


def test_a_gate_on_an_untouched_chart_must_refuse():
    """Nothing searched, nothing read, nothing sampled. Refusal is the only correct answer."""
    spec = load_spec(STRATIFIED)
    led = _fresh_ledger(spec)
    gate_spec = (spec.proof_obligation.for_negative or {}).get("gate") or {}

    g = evaluate_gate(gate_spec, led.stratum_results())
    assert g.verdict == "FAIL", "a gate that passes an untouched chart is checking nothing"
    assert g.missing


def test_the_gate_config_is_not_empty():
    """Guards the exact regression: restructuring the spec must not leave the gate blank.

    An empty gate spec makes evaluate_gate vacuously true, which is indistinguishable from
    a genuine pass in the output.
    """
    spec = load_spec(STRATIFIED)
    gate_spec = (spec.proof_obligation.for_negative or {}).get("gate") or {}
    assert gate_spec, "the stratified spec must declare a gate"
    assert len(gate_spec) >= 4


def test_forced_sampling_blocks_a_premature_negative():
    """The runtime draws validation samples; until they are inspected, no negative stands."""
    spec = load_spec(STRATIFIED)
    led = _fresh_ledger(spec)
    pending = led.pending_samples()
    assert pending, "there are unread strata, so samples must be owed"
    assert sum(len(v) for v in pending.values()) >= 25


def test_exclusion_declaration_is_overturned_by_a_single_hit():
    """One relevant document in the null sample falsifies the exclusion and must fail the gate."""
    spec = load_spec(STRATIFIED)
    led = _fresh_ledger(spec)
    drawn = led.pending_samples()["cannot_establish"]
    for d in drawn:
        led.record_sample_verdict("cannot_establish", d.note_id, relevant=False)
    clean = evaluate_gate({"exclusion_validated": True}, led.stratum_results())
    assert clean.checks["exclusion_validated"] is True

    led.record_sample_verdict("cannot_establish", drawn[0].note_id, relevant=True)
    dirty = evaluate_gate({"exclusion_validated": True}, led.stratum_results())
    assert dirty.checks["exclusion_validated"] is False
    assert any("promote" in m for m in dirty.missing), (
        "a falsified exclusion must say what to do next, not merely fail"
    )


def test_unstratified_arm_is_honestly_labelled_not_silently_passing():
    """The ablation arm declares no strata. That is legitimate — but the ledger has to say
    so, otherwise an unstratified run and a stratified one are indistinguishable afterwards
    and the comparison means nothing."""
    spec = load_spec(UNSTRATIFIED)
    assert strata_from_spec(spec) == []
    led = _fresh_ledger(spec)
    assert led.to_dict()["mode"] == "unstratified"


def test_ablation_needs_a_found_case():
    """SYN0002's ground truth equals the fallback status, so it cannot measure correctness.

    graph.py does `ans.setdefault("status", "EVIDENCE_INSUFFICIENT")`, and SYN0002's ground
    truth is EVIDENCE_INSUFFICIENT — so an agent that does nothing at all and exhausts its
    budget still "matches". Any ablation therefore needs at least one patient whose correct
    answer is FOUND, where matching cannot happen by default.
    """
    import json
    gt2 = json.loads((ROOT / "corpus" / "patients" / "SYN0002" / "_ground_truth.json").read_text())
    gt1 = json.loads((ROOT / "corpus" / "patients" / "SYN0001" / "_ground_truth.json").read_text())
    key = "STORE.400_522_523.site_histology_behavior"

    assert gt2["ground_truth"][key]["status"] == "EVIDENCE_INSUFFICIENT"
    assert gt1["ground_truth"][key]["status"] == "FOUND", (
        "SYN0001 is the arm's FOUND case; if this ever changes the ablation loses its only "
        "test of whether the agent actually succeeds"
    )
    assert gt1["ground_truth"][key]["histology"] == "8140"
