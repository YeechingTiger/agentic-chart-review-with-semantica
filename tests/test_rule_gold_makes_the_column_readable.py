"""`not_considered` is not a finding until gold says the rule applied.

Four conflict rules over twenty-seven charts: most rules genuinely do not apply to most charts, so
`not_considered` is the COMMON state and a rate read off it alone measures nothing. The measurable is

    not_considered AND the gold annotation says this rule was applicable

which is why `rule_assessment`'s own docstring says the view "buys visibility with no way to tell a
miss from a correct silence" until this exists.

## The same distinction, one level up

The gold has to keep apart *annotated as inapplicable* and *not annotated*. An unannotated chart
counted as "the rule does not apply here" would manufacture correct silences by the dozen — the exact
`not_considered` / `not_applicable` confusion this whole line of work is about, reproduced in the
reference standard.

## Transcribed, not invented

The mirror pair's `why` fields already state the rule and the fact:

  SYN0001  "Cytology 2023-04-12 read as 'suspicious for adenocarcinoma' AND an oncology note of the
            same admission records a clinical impression of malignancy, so the cytology date is
            diagnostic per STORE [390]."
  SYNX03   "Cytology of 2022-02-14 is ambiguous ('suspicious for') and NO physician's clinical
            impression of cancer accompanies it, so STORE.390's second conflict_rule applies and the
            biopsy date governs."

SYNX03 names the rule outright. So the annotation records what the corpus already asserts; nothing
here is a fresh clinical judgement. Charts whose `why` does not state a rule are left unannotated
rather than guessed at.
"""

from __future__ import annotations

import pytest

from acr.contract.spec import load_spec
from acr.core import site
from acr.review.rule_assessment import assess_rules
from acr.review.rule_gold import (
    RuleGoldError,
    load_rule_gold,
    missed_rules,
)

SPEC = "STORE.390.date_of_initial_diagnosis"


@pytest.fixture(scope="module")
def spec():
    return load_spec(site.specs_root() / f"{SPEC}.yaml")


# ------------------------------------------------------------------ the mirror pair

def test_the_mirror_pair_is_annotated_with_opposite_fact_truths(spec):
    a = load_rule_gold("SYN0001", spec)
    b = load_rule_gold("SYNX03", spec)
    assert a["conflict_rule.1"].discriminating_fact_truth["impression_at_ambiguous_cytology"] is True
    assert b["conflict_rule.2"].discriminating_fact_truth["impression_at_ambiguous_cytology"] is False


def test_both_say_the_answer_turns_on_the_fact(spec):
    """That is what makes them a mirror pair, and what makes an unchecked discriminator a defect
    rather than a curiosity on these two charts."""
    for pid, rid in (("SYN0001", "conflict_rule.1"), ("SYNX03", "conflict_rule.2")):
        assert load_rule_gold(pid, spec)[rid].answer_changes_if_fact_flips is True


def test_the_annotation_carries_where_the_fact_is(spec):
    """A gold fact truth with no pointer to the evidence is unauditable — the reviewer cannot check
    the reference standard itself."""
    g = load_rule_gold("SYN0001", spec)["conflict_rule.1"]
    assert "oncology" in g.discriminator_evidence.lower()
    assert g.applicability_evidence


# ------------------------------------------------------------------ absent is not inapplicable

def test_an_unannotated_chart_yields_nothing_rather_than_inapplicable(spec):
    """THE DISTINCTION THIS FILE EXISTS FOR, one level up. A chart nobody annotated must not count as
    a chart where the rule does not apply, or the denominator fills with manufactured silences."""
    assert load_rule_gold("SYN0004", spec) == {}


def test_a_rule_annotated_inapplicable_is_a_positive_claim(spec, tmp_path):
    """`applicable: false` is a reviewer saying so. It is a different row from an absent one and both
    are needed: the first licenses a correct silence, the second licenses nothing."""
    from acr.review.rule_gold import RuleGoldAnnotation, parse_rule_gold
    rows = parse_rule_gold({SPEC: {"conflict_rule.1": {
        "applicable": False, "applicability_evidence": "no cytology anywhere in this chart"}}}, spec)
    r = rows["conflict_rule.1"]
    assert isinstance(r, RuleGoldAnnotation)
    assert r.applicable is False
    assert r.discriminating_fact_truth == {}, "an inapplicable rule has no fact to be true"


# ------------------------------------------------------------------ the join, which is the point

def test_an_unchecked_applicable_rule_is_a_miss_and_an_unchecked_inapplicable_one_is_not(spec):
    """The join. `not_considered` on a rule gold says applied is a MISS; on a rule gold says did not
    apply, or on a rule gold says nothing about, it is not."""
    rows = assess_rules(spec)                       # a run that assessed nothing
    assert [r.status for r in rows] == ["not_considered"] * 4

    misses = missed_rules(rows, load_rule_gold("SYN0001", spec))
    assert misses == ["conflict_rule.1"], "rule 1 applied on this chart and was never considered"

    assert missed_rules(rows, load_rule_gold("SYN0004", spec)) == [], (
        "an unannotated chart yields no misses, because nothing claims a rule applied")


def test_considering_the_rule_clears_the_miss_even_if_the_fact_is_unresolved(spec):
    """A miss is a rule nobody LOOKED AT. Whether its fact was then checked is the next column, and
    conflating the two would report one defect as two."""
    rows = assess_rules(spec, declared={
        "conflict_rule.1": {"assessment": "applicable", "evidence_ids": ["E1"], "rationale": "x"}})
    assert missed_rules(rows, load_rule_gold("SYN0001", spec)) == []
    r = next(x for x in rows if x.rule_id == "conflict_rule.1")
    assert r.status == "applicable_unresolved", "still open on the fact, which is a separate column"


def test_judging_an_applicable_rule_inapplicable_is_its_own_error(spec):
    """Distinct from a miss: the run looked and got it wrong. Stage 2 of the seven-stage attribution
    the annotation exists to make possible."""
    from acr.review.rule_gold import wrongly_dismissed
    rows = assess_rules(spec, declared={
        "conflict_rule.1": {"assessment": "not_applicable", "rationale": "no cytology"}})
    assert wrongly_dismissed(rows, load_rule_gold("SYN0001", spec)) == ["conflict_rule.1"]
    assert missed_rules(rows, load_rule_gold("SYN0001", spec)) == [], "it was considered"


# ------------------------------------------------------------------ refusals

def test_an_annotation_naming_a_rule_the_contract_does_not_declare_is_refused(spec):
    from acr.review.rule_gold import parse_rule_gold
    with pytest.raises(RuleGoldError, match="conflict_rule.9"):
        parse_rule_gold({SPEC: {"conflict_rule.9": {"applicable": True}}}, spec)


def test_a_fact_truth_against_an_undeclared_fact_is_refused(spec):
    from acr.review.rule_gold import parse_rule_gold
    with pytest.raises(RuleGoldError, match="ghost"):
        parse_rule_gold({SPEC: {"conflict_rule.1": {
            "applicable": True, "discriminating_fact_truth": {"ghost": True}}}}, spec)


def test_an_applicable_rule_must_state_the_truth_of_every_fact_it_turns_on(spec):
    """An applicable rule whose fact truth is unstated cannot score anything: the run's resolution has
    nothing to be right or wrong against."""
    from acr.review.rule_gold import parse_rule_gold
    with pytest.raises(RuleGoldError, match="impression_at_ambiguous_cytology"):
        parse_rule_gold({SPEC: {"conflict_rule.1": {"applicable": True}}}, spec)


def test_a_tie_break_annotated_applicable_needs_no_fact_truth(spec):
    """Rule 4 turns on nothing, so there is nothing to state."""
    from acr.review.rule_gold import parse_rule_gold
    rows = parse_rule_gold({SPEC: {"conflict_rule.4": {
        "applicable": True, "applicability_evidence": "two documents disagree"}}}, spec)
    assert rows["conflict_rule.4"].discriminating_fact_truth == {}
