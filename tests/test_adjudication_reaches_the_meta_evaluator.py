"""The tree's only adjudication producer wrote a shape its only consumer cannot read.

`meta_evaluate_attributions` reads the human root-cause label from `row["primary_cause"]` or
`row["adjudication"]["primary_cause"]`. `AdjudicationEvent.to_dict()` — reached through
`acr attribute adjudicate`, the only writer — emitted `{case_id, decision, actor, actor_role,
rationale, evidence, created_at}` and neither key. `decision` is validated against `LIFECYCLE`
(OPEN / NEEDS_ADJUDICATION / ATTRIBUTED / …), and `LIFECYCLE ∩ CAUSES` is EMPTY: the two
vocabularies do not overlap at all, so even a lenient reader could not have salvaged one.

Consequence, reproduced before this fix with one exactly-matching case_id and `min_cases=1`:

    n_adjudicated_pairs: 0
    macro_f1: null
    reasons_not_certified: ["need at least 1 adjudicated cases", …]

Adjudicate two thousand cases and the report still says *need at least 30*. The repo shipped that
count as the explanation — `docs/NEW_TASK_NEW_DATA.md` and every generated repository's README say
"there are 2 records", diagnosing a data shortage where the defect was a format mismatch.

`scope_violations` is the same defect one level down: two of the five conditions gating
`CERTIFIED_SCREEN` read keys `AttributionReport.to_dict()` never writes, so they are permanently 0
and "patient-scope violations must be zero" could never appear in `reasons_not_certified`. A
precondition that cannot fail must not print as one that passed.
"""

from __future__ import annotations

import pytest

from acr.diagnosis import attribution as A


def _report(case_id="CASE-abc123", cause="RETRIEVAL", status="LIKELY"):
    """A prediction row of the shape `meta_evaluate_attributions` consumes."""
    return {"case_id": case_id, "primary_cause": {"cause": cause, "status": status}}


def test_the_two_vocabularies_still_do_not_overlap():
    """Guards the premise: if they ever did overlap, the reader could infer a cause from a
    decision and this whole file would be about nothing."""
    assert set(A.LIFECYCLE) & set(A.CAUSES) == set()


def test_an_adjudication_carries_a_root_cause():
    ev = A.AdjudicationEvent(
        case_id="CASE-abc123", decision="ATTRIBUTED", primary_cause="RETRIEVAL",
        actor="r", actor_role="registrar", rationale="the note was never opened")
    assert ev.to_dict()["primary_cause"] == "RETRIEVAL"


def test_the_cause_must_be_one_the_predictor_can_emit():
    """A free-text cause pairs with nothing: macro-F1 over labels only one side uses is 0."""
    with pytest.raises(A.AttributionError, match="primary_cause"):
        A.AdjudicationEvent(case_id="CASE-abc123", decision="ATTRIBUTED",
                            primary_cause="the model was lazy",
                            actor="r", actor_role="registrar", rationale="x")


def test_an_adjudication_may_decline_to_name_a_cause():
    """Not every adjudication is a root-cause label — WONT_FIX and OUTSIDE_CHART are decisions
    about what to DO. Requiring a cause would make those unrecordable."""
    ev = A.AdjudicationEvent(case_id="CASE-abc123", decision="WONT_FIX",
                             actor="r", actor_role="registrar", rationale="not chart-observable")
    assert ev.to_dict().get("primary_cause") in (None, "")


def test_a_written_adjudication_pairs_with_a_prediction():
    """The producer→consumer property. This is what returned 0 pairs for every input."""
    ev = A.AdjudicationEvent(
        case_id="CASE-abc123", decision="ATTRIBUTED", primary_cause="RETRIEVAL",
        actor="r", actor_role="registrar", rationale="never opened it")
    out = A.meta_evaluate_attributions(
        [_report(cause="RETRIEVAL")], [ev.to_dict()], min_cases=1, min_macro_f1=0.0)
    assert out["n_adjudicated_pairs"] == 1
    assert out["macro_f1"] == 1.0


def test_adjudications_with_no_readable_cause_refuse_rather_than_report_zero():
    """The failure that hid this for weeks: `n_adjudicated_pairs: 0` alongside "need at least N
    cases" reads as a data shortage. When rows EXIST and none carries a cause, that is a format
    problem and the report must say so."""
    rows = [A.AdjudicationEvent(case_id="CASE-abc123", decision="WONT_FIX", actor="r",
                                actor_role="registrar", rationale="x").to_dict()]
    out = A.meta_evaluate_attributions([_report()], rows, min_cases=1, min_macro_f1=0.0)
    reasons = " ".join(out["reasons_not_certified"])
    assert "primary_cause" in reasons, reasons
    assert out["status"] != "CERTIFIED_SCREEN"


def test_an_empty_adjudication_list_still_reports_a_shortage():
    """The genuine data shortage must keep reading as one."""
    out = A.meta_evaluate_attributions([_report()], [], min_cases=30, min_macro_f1=0.0)
    reasons = " ".join(out["reasons_not_certified"])
    assert "at least 30" in reasons
    assert "primary_cause" not in reasons


def test_scope_violations_is_reported_as_unmeasured_when_nothing_writes_it():
    """`AttributionReport.to_dict()` writes `gate_rejections` but never `scope_violations`, so the
    condition could never fail. An unmeasurable precondition must not print as a satisfied one."""
    out = A.meta_evaluate_attributions(
        [_report()], [], min_cases=1, min_macro_f1=0.0)
    assert out.get("scope_violations_measured") is False


def test_scope_violations_is_measured_when_a_row_carries_it():
    out = A.meta_evaluate_attributions(
        [{**_report(), "scope_violations": 0}], [], min_cases=1, min_macro_f1=0.0)
    assert out.get("scope_violations_measured") is True


def test_a_later_decision_does_not_erase_an_earlier_root_cause():
    """The ledger is APPEND-ONLY — `ErrorCaseLibrary.add_adjudication` never rewrites an event, and
    `LIFECYCLE` includes `REOPENED`, so a case legitimately accumulates several rows. Folding them
    with a dict comprehension made the LAST row win: a follow-up `VALIDATED_FIXED` or `REOPENED`
    carrying no cause (correctly, since it is a decision about what to DO) silently erased the
    root-cause label recorded earlier, dropping the pair AND counting the case as
    `n_adjudications_without_cause` — blocking certification on data that is complete.
    """
    labelled = A.AdjudicationEvent(
        case_id="CASE-abc123", decision="ATTRIBUTED", primary_cause="RETRIEVAL",
        actor="r", actor_role="registrar", rationale="never opened it").to_dict()
    later = A.AdjudicationEvent(
        case_id="CASE-abc123", decision="VALIDATED_FIXED",
        actor="e", actor_role="engineer", rationale="the fix holds").to_dict()

    out = A.meta_evaluate_attributions(
        [_report(cause="RETRIEVAL")], [labelled, later], min_cases=1, min_macro_f1=0.0)

    assert out["n_adjudicated_pairs"] == 1
    assert out["n_adjudications_without_cause"] == 0
    assert out["macro_f1"] == 1.0


def test_a_later_adjudication_may_correct_the_cause():
    """Append-only does not mean immutable-in-effect: a registrar who changes their mind appends a
    new row WITH a cause, and the latest cause is the one that counts."""
    first = A.AdjudicationEvent(
        case_id="CASE-abc123", decision="ATTRIBUTED", primary_cause="RETRIEVAL",
        actor="r", actor_role="registrar", rationale="first read").to_dict()
    revised = A.AdjudicationEvent(
        case_id="CASE-abc123", decision="ATTRIBUTED", primary_cause="EVIDENCE_INTERPRETATION",
        actor="r", actor_role="registrar", rationale="on review, it was read and misjudged").to_dict()

    out = A.meta_evaluate_attributions(
        [_report(cause="EVIDENCE_INTERPRETATION")], [first, revised],
        min_cases=1, min_macro_f1=0.0)
    assert out["n_adjudicated_pairs"] == 1
    assert out["macro_f1"] == 1.0


def test_a_case_whose_every_row_lacks_a_cause_is_still_counted_once():
    """`n_adjudications_without_cause` counted ROWS, so three causeless rows for one case read as
    three missing labels. It is a count of CASES that pair with nothing."""
    rows = [A.AdjudicationEvent(case_id="CASE-abc123", decision=d, actor="r",
                                actor_role="registrar", rationale="x").to_dict()
            for d in ("OPEN", "NEEDS_ADJUDICATION", "WONT_FIX")]
    out = A.meta_evaluate_attributions([_report()], rows, min_cases=1, min_macro_f1=0.0)
    assert out["n_adjudications_without_cause"] == 1
