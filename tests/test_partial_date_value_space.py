"""A date field's value space, and what one boolean could not say.

TWO DEFECTS, ONE BLOCK OF YAML. Both were measured in the E4 floor run rather than reasoned
about, and both are about the contract's VALUE space rather than its rules.

1. `20999999` WAS FORMAT-VALID. The declared pattern was

       (19|20)\\d{2}(0[1-9]|1[0-2]|99)([0-2]\\d|3[01]|99)

   which is a shape, not a calendar. It admits day `00`, 31 April, 29 February in a common
   year — and `2099` as a year, which is what two runs actually submitted. They submitted it
   because `decision_rule[5]` orders the year approximated when it cannot be identified, and
   nothing in the field could record that the year was approximated; the model extended the
   "99 means unknown" convention into the year slot, where it does not hold.

   Validation is therefore two steps, not one: a COMPONENT pattern (is each of the three
   parts a legal token) and then a CALENDAR (do those three tokens name a day that exists).
   A regex cannot do the second — leap years are arithmetic — and pretending it can is how a
   shape check got described as a date check for a month.

2. ONE BOOLEAN FOR THREE QUESTIONS. `month_day_imputed` could not distinguish "the month came
   from 'the spring'" from "the day is simply not recorded" from "the year itself is an
   estimate". Those have different consequences downstream and different remedies, and a
   single flag forced the model to pick one meaning and leave the reader to guess which.

WHERE THE 99s ARE LEGAL, stated once here because it was previously implied by a regex:

    YYYYMMDD   fully known
    YYYYMM99   day unknown
    YYYY9999   month and day unknown
    YYYY99DD   NOT DECLARED — nobody has decided whether a known day under an unknown month
               is meaningful, and it is refused with that said rather than settled by regex.
               The year slot is never 99: decision_rule[5] says approximate the year, and
               `year_imputed` is where "this is an approximation" is recorded.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from acr.contract.answer_checks import check_field_formats_detail
from acr.contract.spec import ExtractionSpec, load_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC_390 = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"


@pytest.fixture(scope="module")
def spec390():
    return load_spec(SPEC_390)


def _violations(spec, value: str) -> list:
    return check_field_formats_detail(spec.fields, {"date_of_initial_diagnosis": value})


# --------------------------------------------------------------- the calendar step

@pytest.mark.parametrize("good", ["20181107", "20200229", "20180630", "20181231",
                                  "20180699", "20189999", "19700101"])
def test_dates_that_exist_are_accepted(spec390, good):
    assert _violations(spec390, good) == [], good


@pytest.mark.parametrize("bad,why", [
    ("20180229", "2018 is not a leap year"),
    ("21000229", "2100 is a century year and not a leap year"),
    ("20180431", "April has thirty days"),
    ("20180100", "there is no day zero"),
    ("20181301", "there is no thirteenth month"),
    ("20180000", "neither month nor day may be zero"),
])
def test_dates_that_do_not_exist_are_refused(spec390, bad, why):
    v = _violations(spec390, bad)
    assert v, f"{bad} accepted, but {why}"
    assert v[0].coded_value == bad


def test_a_known_day_under_an_unknown_month_is_refused_as_undeclared_not_as_invalid(spec390):
    """The honest answer is that nobody decided, so the message says nobody decided.

    Silently accepting it would let two runs mean different things by the same string;
    silently rejecting it as 'not a date' would claim a decision the contract has not made.
    """
    v = _violations(spec390, "20189915")
    assert v, "YYYY99DD must not pass unremarked"
    msg = v[0].message.lower()
    assert "not declared" in msg or "undeclared" in msg, v[0].message


def test_the_year_slot_is_never_unknown(spec390):
    """`9999` in the year is the notation that produced `20999999`.

    decision_rule[5] requires the year to be APPROXIMATED rather than left blank, so an
    unknown year has no representation and must not acquire one by convention.
    """
    v = _violations(spec390, "99991107")
    assert v, "9999 as a year must be refused"


def test_the_component_pattern_and_the_calendar_are_separate_checks(spec390):
    """A value can fail the shape, or fail the arithmetic, and a reader needs to know which."""
    shape = _violations(spec390, "2018-11-07")
    calendar = _violations(spec390, "20180229")
    assert shape and calendar
    assert {v.rule_kind for v in shape} == {"field_format"}
    assert {v.rule_kind for v in calendar} == {"field_calendar"}


def test_a_field_that_declares_no_calendar_is_untouched_by_the_calendar_step():
    """Every other contract in the tree codes ICD-O-3, not dates."""
    spec = ExtractionSpec.model_validate({
        "spec_id": "TEST.nocal", "question": "q?",
        "fields": [{"name": "primary_site", "type": "string", "format": "C\\d{3}"}]})
    assert check_field_formats_detail(spec.fields, {"primary_site": "C349"}) == []


# --------------------------------------------------------------- the three flags

def test_the_contract_can_say_which_component_was_imputed(spec390):
    names = {f.name for f in spec390.fields}
    assert {"year_imputed", "month_imputed", "day_imputed"} <= names
    assert "month_day_imputed" not in names, "one boolean answering three questions"


def test_each_flag_is_a_boolean_with_a_declared_domain(spec390):
    for n in ("year_imputed", "month_imputed", "day_imputed"):
        f = next(f for f in spec390.fields if f.name == n)
        assert f.type == "boolean"
        assert f.allowable_values == [True, False], n


def test_the_seasonal_boundary_case_now_says_which_components_it_imputed(spec390):
    """"diagnosed in the spring of 2010" imputes the month and the day, not the year."""
    bc = next(b for b in spec390.boundary_cases
              if isinstance(b, dict) and "spring" in str(b.get("case", "")).lower())
    assert bc.get("month_imputed") is True
    assert bc.get("day_imputed") is True
    assert bc.get("year_imputed") is not True, "the year was stated outright"


# --------------------------------------------------------------- lint

def test_speclint_refuses_a_date_regex_that_is_its_own_only_validator():
    """The defect this package exists to close, as a check rather than as a memory.

    The test is a fact about the pattern, not a guess about the author's intent: the regex is
    fed dates that do not exist. Accepting one while declaring no calendar means the contract
    calls a shape a date.
    """
    from acr.authoring.speclint import lint_spec
    spec = ExtractionSpec.model_validate({
        "spec_id": "TEST.dateonly", "question": "when?",
        "abstention": {"EVIDENCE_INSUFFICIENT": "x", "SPEC_INSUFFICIENT": "y"},
        "fields": [{"name": "d", "type": "string",
                    "format": "(19|20)\\d{2}(0[1-9]|1[0-2]|99)([0-2]\\d|3[01]|99)"}]})
    fails = [f for f in lint_spec(spec) if f.severity == "FAIL" and "20180229" in f.message]
    assert fails, [f"{f.check} {f.message}" for f in lint_spec(spec)]
