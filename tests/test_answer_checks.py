"""What answer_checks still does, and what it must never do again.

This file used to specify five clinical checks that matched word lists against the model's own
cited quotes. They were removed on 2026-07-30 after being measured over every trace this
project has recorded. The tests that specified them are gone with them -- a test that pins a
rule in place is part of the rule -- and what is left here does two jobs:

  1. the surviving format/value-domain check still works, with the cases that justified it;
  2. the clinical kinds cannot come back by accident. A spec that declares one fails to load,
     and `check_answer_detail` returns nothing whatever it is handed.

THE MEASUREMENT, so that re-adding a word list has to argue with a number. Over 266 traces,
202 joinable to registry gold, 122 recorded firings:

    rule                       fires   rejected the registry's own value   ever helped
    not_less_specific             22                    22  (100%)                   0
    nos_requires_search           24                    21  ( 88%)                   0
    conflict_requires_nos         67                    18  ( 27%)                  15
    origin_not_specimen            2                     0                           0
    code_matches_cited_text        0                     -                           -

All 15 `conflict_requires_nos` "helps" were the same event -- a push to the NOS code, the only
remedy its message offered -- and the NOS code is the registry's answer for 9.6% of this corpus
against C341's 52.7%. Per submission: 60 of 254 recorded rejections refused a tuple that was
exactly the registry's, and 12 runs held the exact registry answer and shipped something else.

The negation bug this file was originally written to pin (`"small cell"` matching inside
`"non-small cell carcinoma"`, refusing the registry's 8046) is now impossible for the reason
that makes the whole class impossible: nothing matches phrases against quotes any more.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from acr.answer_checks import (
    ANSWER_CHECK_KINDS,
    answer_check_rule_id,
    check_answer,
    check_answer_detail,
    check_field_formats,
    check_field_formats_detail,
    field_rule_id,
)
from acr.spec import ProvenanceError, load_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"


class F:
    """The duck type `check_field_formats` reads: name, format, allowable_values."""

    def __init__(self, name, format=None, allowable_values=None):
        self.name, self.format, self.allowable_values = name, format, allowable_values


FIELDS = [F("primary_site", format=r"C\d{3}"),
          F("histology", format=r"\d{4}"),
          F("behavior", allowable_values=["0", "1", "2", "3"])]


# ------------------------------------------------------- the clinical checks are gone
def test_no_clinical_check_kind_is_implemented():
    assert ANSWER_CHECK_KINDS == frozenset()


@pytest.mark.parametrize("kind", [
    "not_less_specific", "nos_requires_search", "conflict_requires_nos",
    "origin_not_specimen", "code_matches_cited_text",
])
def test_a_spec_declaring_a_removed_kind_fails_to_load(tmp_path, kind):
    """Fail closed, not fail quiet.

    A check named in YAML, visible in the manifest's `rule_catalog`, and never firing is
    indistinguishable from a check that looked and found nothing. That is the exact confusion
    `ANSWER_CHECK_KINDS` was introduced to prevent, so emptying the set keeps the property
    rather than dropping it.
    """
    p = tmp_path / "S.1.yaml"
    p.write_text(
        "spec_id: S.1\nspec_version: 0.1.0\ndata_source: notes\n"
        "question: q\n"
        "fields:\n  - name: primary_site\n    type: string\n    format: 'C\\d{3}'\n"
        "decision_rule: [r]\n"
        "evidence_rules:\n  counts_as_evidence: [anything]\n"
        f"answer_checks:\n  - field: primary_site\n    kind: {kind}\n"
        "    nos_values: ['C349']\n", encoding="utf-8")
    with pytest.raises(ProvenanceError) as e:
        load_spec(p)
    assert kind in str(e.value)


def test_check_answer_returns_nothing_whatever_it_is_handed():
    checks = [{"field": "primary_site", "kind": "not_less_specific",
               "nos_values": ["C349"], "contradicted_by": ["upper lobe"]}]
    ev = [{"quote": "tumour in the right upper lobe", "supports": "primary_site"}]
    assert check_answer(checks, {"primary_site": "C349"}, ev, searched=[]) == []
    assert check_answer_detail(checks, {"primary_site": "C349"}, ev, searched=[]) == []


def test_the_registry_answers_the_removed_checks_destroyed_now_pass():
    """The three values that were refused in real runs, as a regression pin.

    C349 with a lobe in the evidence (`not_less_specific`, 16 firings), C341 with two lobes in
    the evidence (`conflict_requires_nos`, 13 firings), and 8046 cited off "non-small cell"
    (`not_less_specific` reading a negation as its opposite). Each was the registry's own
    answer; each was refused.
    """
    ev = [{"quote": "right upper lobe mass; also a left lower lobe nodule"},
          {"quote": "poorly differentiated non-small cell carcinoma"}]
    for value in ({"primary_site": "C349"}, {"primary_site": "C341"}, {"histology": "8046"}):
        assert check_answer([], value, ev, searched=[]) == []
        assert check_field_formats(FIELDS, value) == []


# ------------------------------------------------------- the format check still works
def test_a_four_digit_topography_code_is_refused():
    """C3412 shipped once, gate-validated, zero rejections. `C\\d{3}` is a contract."""
    v = check_field_formats_detail(FIELDS, {"primary_site": "C3412"})
    assert len(v) == 1
    assert v[0].rule_kind == "field_format"
    assert v[0].rule_id == "field_format.primary_site"
    assert v[0].coded_value == "C3412"


def test_the_punctuated_icdo3_form_is_still_refused_and_that_is_a_known_defect():
    """4 of this check's 7 useful firings rejected the form ICD-O-3 itself writes.

    Pinned as a FINDING, not as desired behaviour: the fix is deterministic normalisation
    (`C34.1` -> `C341`), an addition that was not made in the deletion pass. Whoever adds it
    should flip this test to assert the normalised value passes.
    """
    assert check_field_formats(FIELDS, {"primary_site": "C34.9"}) != []


def test_a_value_outside_the_declared_domain_is_refused():
    v = check_field_formats_detail(FIELDS, {"behavior": "9"})
    assert len(v) == 1
    assert v[0].rule_kind == "field_allowable_values"
    assert v[0].rule_id == "field_allowable_values.behavior"


def test_a_well_formed_answer_passes():
    assert check_field_formats(FIELDS, {"primary_site": "C341", "histology": "8140",
                                        "behavior": "3"}) == []


def test_an_absent_field_is_abstentions_business_not_the_format_checkers():
    assert check_field_formats(FIELDS, {"primary_site": None, "histology": "  "}) == []


def test_a_well_formed_but_invented_morphology_still_passes_and_that_is_the_open_gap():
    """`\\d{4}` cannot tell a real morphology from a well-formed invented one.

    Pinned so the limitation lives in a test and not only in a docstring. Closing it needs a
    real ICD-O-3 code table; a shape regex is not one.
    """
    assert check_field_formats(FIELDS, {"histology": "9999"}) == []


def test_a_broken_pattern_in_a_spec_cannot_block_a_run():
    assert check_field_formats([F("x", format="([unclosed")], {"x": "anything"}) == []


# ------------------------------------------------------- rule identity still resolves
def test_rule_ids_still_mint_for_traces_recorded_before_the_removal():
    """`acr.diagnosis.attribution` has to resolve ids in traces written while the checks existed."""
    rid = answer_check_rule_id({"field": "primary_site", "kind": "conflict_requires_nos",
                                "nos_value": "C349"})
    assert rid.startswith("answer_check.primary_site.conflict_requires_nos")
    assert answer_check_rule_id("not a dict", 3) == "answer_check.unparsed#3"
    assert field_rule_id("field_format", "histology") == "field_format.histology"


def test_the_shipped_spec_declares_no_answer_checks():
    spec = load_spec(SPEC)
    assert (getattr(spec, "answer_checks", []) or []) == []
