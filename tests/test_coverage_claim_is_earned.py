"""`coverage_attested` may travel with exactly one kind of answer.

The ledger asserts "I searched the defined universe". Only a negative that passed the proof
obligation has established that. A witness-proved positive never claimed it — one qualifying
document settles a FOUND, and the gate for FOUND checks only that — so attaching a coverage
ledger to it advertises a search that was never verified. A give-up and a budget exhaustion
never earned it either.

This is the same family as a check that records but cannot refuse: an artifact implying a
stronger claim than anything actually established. That one sat on the way in; this one sits
on the way out, and downstream it is indistinguishable from the real thing.
"""
from __future__ import annotations

import pytest

from acr.answer_contract import CoverageClaimError, assert_coverage_claim_is_earned


def test_gate_validated_negative_may_carry_the_ledger():
    assert_coverage_claim_is_earned({
        "status": "EVIDENCE_INSUFFICIENT",
        "negative_basis": "GATE_VALIDATED",
        "coverage_attested": {"mode": "stratified_exclusion"},
    })


def test_gate_validated_negative_must_carry_the_ledger():
    """A coverage claim without its evidence is unauditable, so the omission is also a bug."""
    with pytest.raises(CoverageClaimError, match="must carry"):
        assert_coverage_claim_is_earned({
            "status": "EVIDENCE_INSUFFICIENT",
            "negative_basis": "GATE_VALIDATED",
        })


def test_a_positive_finding_must_not_carry_a_coverage_claim():
    """FOUND is proved by witness. It never asserted that the universe was searched."""
    with pytest.raises(CoverageClaimError, match="only a gate-validated"):
        assert_coverage_claim_is_earned({
            "status": "FOUND",
            "proof_basis": "WITNESS",
            "coverage_attested": {"mode": "stratified_exclusion"},
        })


@pytest.mark.parametrize("basis", ["AGENT_GAVE_UP", "BUDGET_EXHAUSTED",
                                  "COVERAGE_UNREACHABLE"])
def test_an_unvalidated_negative_must_not_carry_a_coverage_claim(basis):
    with pytest.raises(CoverageClaimError):
        assert_coverage_claim_is_earned({
            "status": "EVIDENCE_INSUFFICIENT",
            "negative_basis": basis,
            "coverage_attested": {},
        })


def test_coverage_unreachable_is_a_negative_that_ended_without_earning_anything():
    """The status added on 2026-07-28, and the reason it is NOT `GATE_VALIDATED`.

    When the exclusion sample turns up a hit, or the elusion bound freezes over its cap, the
    coverage obligation becomes unmeetable for the rest of the run. The gate stops asking and the
    abstention stands -- otherwise the agent burns its budget on a demand that cannot be met and
    then invents an exit, which is what happened: five identical refusals and a SPEC_INSUFFICIENT
    claiming the specification was inadequate when the specification was fine.

    But "the gate gave up asking" is not "the search was verified". The two must stay
    distinguishable downstream or the weaker one gets read as the stronger, so this basis carries
    no ledger, routes to a human, and says what foreclosed it.
    """
    assert_coverage_claim_is_earned({
        "status": "EVIDENCE_INSUFFICIENT",
        "negative_basis": "COVERAGE_UNREACHABLE",
        "route_to_human": True,
        "coverage_note": "no coverage claim is made — the proof obligation cannot be met",
        "coverage_unreachable": ["exclusion not validated (sampled 25, hits 1)"],
    })


def test_clean_positives_and_give_ups_pass():
    assert_coverage_claim_is_earned({"status": "FOUND", "proof_basis": "WITNESS",
                                     "witness_count": 2})
    assert_coverage_claim_is_earned({"status": "FOUND", "proof_basis": "UNGATED",
                                     "route_to_human": True})
    assert_coverage_claim_is_earned({"status": "EVIDENCE_INSUFFICIENT",
                                     "negative_basis": "AGENT_GAVE_UP",
                                     "route_to_human": True})
    assert_coverage_claim_is_earned({"status": "SPEC_INSUFFICIENT",
                                     "remedy_class": "WRONG_DATA_SOURCE"})


def test_an_ungated_positive_is_distinguishable_from_a_witnessed_one():
    """A FOUND that left via budget exhaustion never met even the witness standard, and must
    not be filed alongside one that did."""
    witnessed = {"status": "FOUND", "proof_basis": "WITNESS", "witness_count": 3}
    ungated = {"status": "FOUND", "proof_basis": "UNGATED", "route_to_human": True}
    assert witnessed["proof_basis"] != ungated["proof_basis"]
    assert "route_to_human" not in witnessed
    assert ungated["route_to_human"] is True
