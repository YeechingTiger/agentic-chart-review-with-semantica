"""A conflict rule that was never considered must say so, not read as inapplicable.

This closes the hole that v1.1 of the design had, and the hole is worth stating precisely because it
is the same funnel twice:

    the candidate ledger:  candidate never formed
                             -> conflict never formed
                             -> discriminator never checked

    contract-derived, v1.1: applicability never recognised
                             -> obligation never created
                             -> the column reads applicable:false
                             -> THE RUN LOOKS COMPLETE

Deriving obligations from the contract instead of from a candidate set removes the first funnel. It
does nothing about the second unless every declared rule is *accounted for*. So every conflict rule
the contract declares starts at `not_considered`, and a run that assessed nothing reports four
unassessed rules rather than four inapplicable ones.

`not_considered` and `not_applicable` are the two states this whole file exists to keep apart. The
first is a gap in the review; the second is a judgement about the chart. They are the same bytes in
any representation that only records the judgements it received.

## This is a derived view, not a fifth ledger

`assess_rules` is a pure function of (contract, recorded assessments, resolved facts). The tree has
exactly one composition point for "what is still outstanding" — `outstanding_obligations` — and it
exists because *"when the old runtime had two ways to compute it they disagreed about whether a run
had finished"*. A stored assessment ledger would be a second such account.

## Why the shared fact pays off here

STORE.390's rules 1 and 2 turn on the same fact with opposite branches. Because the fact is declared
once at spec level and referenced by both, resolving it once moves BOTH rules to `applicable_checked`.
Had the fact been declared inside each rule — as the design first proposed — one resolution would
have left the other rule reading as unresolved forever.
"""

from __future__ import annotations

import pytest

from acr.contract.spec import load_spec
from acr.core import site
from acr.review.rule_assessment import (
    STATUSES,
    RuleAssessmentError,
    assess_rules,
)

SPEC_390 = site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml"
SPEC_610 = site.specs_root() / "STORE.610.class_of_case.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC_390)


def _by_id(rows):
    return {r.rule_id: r for r in rows}


# ------------------------------------------------------------------ the default state

def test_a_run_that_assessed_nothing_reports_every_rule_as_unconsidered(spec):
    """THE FIX. Not `not_applicable` — that is a judgement nobody made."""
    rows = assess_rules(spec)
    assert [r.rule_id for r in rows] == [f"conflict_rule.{i}" for i in range(1, 5)]
    assert {r.status for r in rows} == {"not_considered"}


def test_every_declared_rule_appears_exactly_once_in_catalog_order(spec):
    """The order is `conflict_rule.N`, which is what the prompt renders and what the model cites."""
    rows = assess_rules(spec)
    ids = [r.rule_id for r in rows]
    assert ids == sorted(ids, key=lambda s: int(s.rsplit(".", 1)[1]))
    assert len(ids) == len(set(ids)) == len(spec.conflict_rules)


def test_a_contract_with_no_conflict_rules_yields_nothing_and_does_not_raise():
    assert assess_rules(load_spec(SPEC_610)) == []


def test_the_status_vocabulary_is_closed():
    assert set(STATUSES) == {"not_considered", "not_applicable", "potentially_applicable",
                             "applicable_checked", "applicable_unresolved"}


# ------------------------------------------------------------------ recorded judgements

def test_a_declared_inapplicable_rule_is_distinguishable_from_an_unconsidered_one(spec):
    rows = _by_id(assess_rules(spec, declared={
        "conflict_rule.1": {"assessment": "not_applicable",
                            "evidence_ids": ["E1"],
                            "rationale": "no cytology in this chart"}}))
    assert rows["conflict_rule.1"].status == "not_applicable"
    assert rows["conflict_rule.1"].applicability_basis_evidence_ids == ("E1",)
    assert rows["conflict_rule.2"].status == "not_considered", "the others are untouched"


def test_an_applicable_rule_with_an_unresolved_fact_is_unresolved(spec):
    rows = _by_id(assess_rules(spec, declared={
        "conflict_rule.1": {"assessment": "applicable", "evidence_ids": ["E3"],
                            "rationale": "ambiguous cytology precedes the biopsy"}}))
    r = rows["conflict_rule.1"]
    assert r.status == "applicable_unresolved"
    assert r.unresolved_facts == ("impression_at_ambiguous_cytology",)


def test_resolving_the_shared_fact_once_moves_both_rules_that_turn_on_it(spec):
    """THE PAYOFF of declaring the fact at spec level. Rules 1 and 2 are one question with opposite
    branches; a fact declared inside each rule would have left one of them unresolved forever."""
    declared = {rid: {"assessment": "applicable", "evidence_ids": ["E3"], "rationale": "..."}
                for rid in ("conflict_rule.1", "conflict_rule.2")}
    rows = _by_id(assess_rules(spec, declared=declared,
                               resolved_facts={"impression_at_ambiguous_cytology"}))
    assert rows["conflict_rule.1"].status == "applicable_checked"
    assert rows["conflict_rule.2"].status == "applicable_checked"
    assert rows["conflict_rule.1"].unresolved_facts == ()


def test_a_tie_break_declared_applicable_is_checked_because_it_turns_on_nothing(spec):
    """Rule 4 turns on no fact and says so. Applying it requires checking nothing, so it must not sit
    at `applicable_unresolved` forever — which is what a design that assumed every rule has a fact
    would do."""
    rows = _by_id(assess_rules(spec, declared={
        "conflict_rule.4": {"assessment": "applicable", "evidence_ids": ["E1", "E2"],
                            "rationale": "two documents disagree"}}))
    r = rows["conflict_rule.4"]
    assert r.turns_on == ()
    assert r.status == "applicable_checked"


def test_potentially_applicable_is_kept_distinct_from_applicable(spec):
    """A run that has seen a hint but not established applicability is in a third state, and
    collapsing it into either neighbour loses the reason the obligation was opened."""
    rows = _by_id(assess_rules(spec, declared={
        "conflict_rule.1": {"assessment": "potentially_applicable", "evidence_ids": ["E3"],
                            "rationale": "a cytology report exists; ambiguity not yet read"}}))
    assert rows["conflict_rule.1"].status == "potentially_applicable"
    assert rows["conflict_rule.1"].unresolved_facts == ("impression_at_ambiguous_cytology",)


# ------------------------------------------------------------------ what it refuses

def test_an_assessment_of_a_rule_the_contract_does_not_declare_is_refused(spec):
    """A claim about `conflict_rule.9` on a four-rule contract is a claim about nothing, and silently
    dropping it would make the column's denominator depend on the model."""
    with pytest.raises(RuleAssessmentError, match="conflict_rule.9"):
        assess_rules(spec, declared={"conflict_rule.9": {"assessment": "applicable"}})


def test_an_unknown_assessment_value_is_refused(spec):
    with pytest.raises(RuleAssessmentError, match="sort_of"):
        assess_rules(spec, declared={"conflict_rule.1": {"assessment": "sort_of"}})


def test_a_resolved_fact_the_contract_does_not_declare_is_refused(spec):
    """A resolution against a fact nobody declared would silently move a rule to `checked`."""
    with pytest.raises(RuleAssessmentError, match="ghost_fact"):
        assess_rules(spec, resolved_facts={"ghost_fact"})


# ------------------------------------------------------------------ the evidence basis is reported

def test_an_assessment_with_no_evidence_basis_is_recorded_not_refused(spec):
    """The column asks `applicability_has_evidence_basis` as its own question, so this is reported
    rather than refused. Refusing it would make the model's only path to recording a judgement one
    that requires evidence it may not have cited yet — and an unrecorded judgement is the hole."""
    rows = _by_id(assess_rules(spec, declared={
        "conflict_rule.1": {"assessment": "not_applicable", "rationale": "no cytology"}}))
    r = rows["conflict_rule.1"]
    assert r.status == "not_applicable"
    assert r.applicability_basis_evidence_ids == ()
    assert r.has_evidence_basis is False


def test_the_rows_are_serialisable_for_the_manifest(spec):
    import json
    rows = assess_rules(spec, declared={
        "conflict_rule.1": {"assessment": "applicable", "evidence_ids": ["E3"], "rationale": "x"}})
    blob = json.dumps([r.to_dict() for r in rows])
    assert "conflict_rule.1" in blob and "applicable_unresolved" in blob
