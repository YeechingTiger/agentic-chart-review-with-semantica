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
from acr.coverage import (CoverageLedger, ForcedSampler, evaluate_gate, keyword_was_searched,
                          strata_from_spec)
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


# ------------------------------------------------- doing less work must not be worth more
def _may_mention(led):
    return next(r for r in led.stratum_results() if r.name == "may_mention")


def test_zero_searches_does_not_validate_the_keyword_list():
    """The gate inversion, at the ledger rather than over MCP.

    `may_mention` is `search_then_read_hits_and_sample_misses`. With no search at all,
    `search_hit_notes` is empty, so every document in the stratum counts as a MISS; the
    sampler draws its 25 misses from the entire stratum; none is cited; and
    `keyword_list_validated` used to come back True. Skipping the search bought a cleaner
    sample than running it, which is the check paying out for the work it exists to demand.
    """
    spec = load_spec(STRATIFIED)
    led = _fresh_ledger(spec)
    led.listed_documents = True
    for docs in led.pending_samples().values():
        for d in docs:
            led.note_read(d.note_id, d.doc_type)
    led.resolve_sample_verdicts(cited=set())

    r = _may_mention(led)
    assert led.searched_terms == [], "precondition: nothing was searched"
    assert r.misses_sampled >= 25 and r.miss_sample_hits == 0, "precondition: a clean sample"
    assert r.keywords_unsearched == r.required_keywords != [], (
        "precondition: the stratum does declare a search obligation"
    )
    assert r.keyword_list_validated is False, (
        "no search ran, so there is no keyword list under test and nothing to validate"
    )

    gate_spec = (spec.proof_obligation.for_negative or {}).get("gate") or {}
    g = evaluate_gate(gate_spec, led.stratum_results())
    assert g.checks["required_keywords_all_searched"] is False
    assert g.checks["keyword_list_validated"] is False
    assert g.verdict == "FAIL"


def test_a_one_character_search_does_not_discharge_the_required_list():
    """`kw.lower() in t or t in kw.lower()` matched in BOTH directions, and the second half
    is free to satisfy: "t" is a substring of "pathology", "biopsy", "final diagnosis",
    "specimen" and "carcinoma", so a single one-character search discharged every required
    keyword the site spec declares."""
    spec = load_spec(STRATIFIED)
    led = _fresh_ledger(spec)
    led.note_search("t", [])
    r = _may_mention(led)
    assert r.required_keywords, "precondition: the stratum declares required keywords"
    assert r.keywords_unsearched == r.required_keywords, (
        "a search for 't' covers none of them"
    )
    assert not keyword_was_searched("carcinoma", led.searched_terms)
    # The direction that does mean something still works.
    assert keyword_was_searched("carcinoma", ["invasive ductal carcinoma"])


def test_an_unread_search_hit_is_reviewed_by_nothing():
    """A hit is removed from the miss-sampling frame, so if it is never read it is audited by
    nothing: not read, and not eligible to be drawn.

    That made searching pay for itself the wrong way round -- every extra search shrank the
    population the sample was drawn from without anyone opening the documents it removed.
    `hits_read` was computed for exactly this and then read by no one.
    """
    spec = load_spec(STRATIFIED)
    led = _fresh_ledger(spec)
    led.listed_documents = True
    chart = Corpus(ROOT / "corpus" / "patients").chart("SYN0002")
    for kw in ["pathology", "biopsy", "final diagnosis", "specimen", "carcinoma"]:
        led.note_search(kw, [h.note_id for h in chart.search(kw, max_hits=40)])
    for docs in led.pending_samples().values():
        for d in docs:
            led.note_read(d.note_id, d.doc_type)
    led.resolve_sample_verdicts(cited=set())

    r = _may_mention(led)
    assert not r.keywords_unsearched, "precondition: every required search ran"
    assert r.misses_sampled >= 25 and r.miss_sample_hits == 0, "precondition: a clean sample"
    assert r.hits_unread, "precondition: the search flagged a document nobody opened"
    assert r.hits_read < r.hits
    assert r.keyword_list_validated is False, (
        "the search flagged this document as relevant and it was never opened"
    )

    # Reading the hits it flagged is what discharges it -- the obligation is satisfiable.
    for nid in list(r.hits_unread):
        led.note_read(nid, "")
    done = _may_mention(led)
    assert done.hits_unread == [] and done.hits_read == done.hits
    assert done.keyword_list_validated is True
