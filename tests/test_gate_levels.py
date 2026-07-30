"""Spec-level checks must not be evaluated at instance level.

`require_can_establish_nonempty` exists to catch a specification in which no document type
could settle the question even in principle. That is a design fault and its honest output is
SPEC_INSUFFICIENT — someone has to go edit the spec.

It must not fire because a particular patient happens to own none of those documents. That
is the finding, often the answer, and its honest output is EVIDENCE_INSUFFICIENT — someone
has to adjudicate this one chart.

The two remedies have different owners and different scopes, so collapsing them costs real
work. This is the same inversion as returning an empty list both for a document type that
does not exist and for one this patient lacks: a level confusion, not a typo.
"""

# The calls below pass `enforce=True`. `evaluate_gate` is ADVISORY by default as of 2026-07-30:
# it still counts strata, samples and residual bounds identically, but routes its sentences to
# `advisories` instead of `missing` so they inform the model rather than refuse its answer.
# "Have I looked at enough of this chart?" is a clinical judgement and now lives in
# `skills/coverage-judgement/SKILL.md`; measured over every recorded trace, coverage obligations
# produced ~150 answer rejections and 27 of them refused a tuple that was exactly the registry's.
#
# These tests are about the ARITHMETIC, which is unchanged and still worth pinning: a bound that
# clears its cap only by inheriting a stale sampling frame is anti-conservative whether or not
# anybody is refused over it. `enforce=True` is how a test reaches the refusal wording.
from __future__ import annotations

from pathlib import Path

import pytest

from acr.corpus import Corpus
from acr.coverage import CoverageLedger, ForcedSampler, evaluate_gate, strata_from_spec
from acr.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC)


def _ledger(pid: str, spec):
    docs, _ = Corpus(ROOT / "corpus" / "patients").chart(pid).list_documents(limit=100_000)
    return CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(1234))


def test_empty_can_establish_is_a_finding_not_a_spec_fault(spec):
    """SYN0002's biopsy was done at an outside hospital, so it owns no pathology at all.

    The stratum is declared and simply empty. The gate must not report that as an illegal
    mode; the answer here is EVIDENCE_INSUFFICIENT.
    """
    led = _ledger("SYN0002", spec)
    assert led.by_stratum["can_establish"] == [], "precondition: this patient has no pathology"

    gate = evaluate_gate({"require_can_establish_nonempty": True}, led.stratum_results(), enforce=True)
    assert gate.checks["can_establish_declared"] is True
    assert not any("not a legal mode" in m for m in gate.missing)


def test_missing_can_establish_declaration_is_a_spec_fault(spec):
    """Strip the declaration and the gate should object — that is what the rule is for."""
    led = _ledger("SYN0002", spec)
    without = [r for r in led.stratum_results() if r.name != "can_establish"]

    gate = evaluate_gate({"require_can_establish_nonempty": True}, without, enforce=True)
    assert gate.checks["can_establish_declared"] is False
    assert any("SPEC_INSUFFICIENT" in m for m in gate.missing)


def test_an_empty_declared_stratum_is_complete_and_contributes_no_elusion(spec):
    """Nothing to read means nothing unread. Zero documents is exhaustively reviewed."""
    led = _ledger("SYN0002", spec)
    ce = next(r for r in led.stratum_results() if r.name == "can_establish")
    assert ce.N == 0 and ce.complete is True and ce.elusion_upper == 0.0


def test_a_patient_with_pathology_populates_the_stratum(spec):
    """Guards the contrast: if SYN0001 also came back empty the test above would be vacuous."""
    led = _ledger("SYN0001", spec)
    assert len(led.by_stratum["can_establish"]) > 0
