"""There is ONE route to an answer, and `gate_validated` has ONE origin.

This file used to introspect `graph.py`'s edge table: SUFFICIENT must route to act and not to
finalize, exactly one node may reach END, and so on. That runtime is gone and the library owns
the edges now, so the questions change shape — but they do not go away, because what they were
really protecting is the property that made the gate worth having: an answer cannot become
`gate_validated` except by passing through the gate.

Asserted against the source rather than by running the agent, deliberately. A second
assignment to `gate_validated`, or a coverage ledger attached on the ungated branch, is a
defect that a passing run will not reveal — the old runtime shipped exactly that, accepting
FOUND answers unchecked and stamping them True.
"""
from __future__ import annotations

import inspect
import re

import pytest

pytest.importorskip("langchain.agents")

import acr.review.agent as A
from acr.contract.answer_contract import NO_COVERAGE_CLAIM, attach_coverage_claim

SRC = inspect.getsource(A)


def test_accepted_has_exactly_one_origin():
    """`ctx.accepted` is what the manifest reports as `gate_validated`."""
    assignments = re.findall(r"^\s*(?:self\.)?ctx\.accepted\s*=\s*(\S+)", SRC, re.MULTILINE)
    assert assignments == ["True"], (
        f"expected exactly one assignment, setting True; found {assignments}. Two places that "
        f"can validate an answer are two places that can validate a wrong one.")


def test_the_only_assignment_is_inside_the_gate():
    gate = inspect.getsource(A.AuditMiddleware._gate_answer)
    assert "ctx.accepted = True" in gate
    assert 'verdict.get("accepted")' in gate, "it must be the gate's verdict that sets it"


def test_submit_answer_cannot_reach_the_model_unjudged():
    """The toolbox returns a receipt for submit_answer; the gate's verdict must replace it."""
    hook = inspect.getsource(A.AuditMiddleware.wrap_tool_call)
    assert 'name == "submit_answer"' in hook
    assert "_gate_answer" in hook, "the receipt must not be what the model sees"


def test_an_unvalidated_negative_carries_no_coverage_claim():
    """The rule itself, exercised rather than grepped — it lives in answer_contract now."""
    gated, ungated = {}, {}
    attach_coverage_claim(gated, gate_validated=True, ledger={"mode": "stratified"},
                          ungated_basis="BUDGET_EXHAUSTED")
    attach_coverage_claim(ungated, gate_validated=False, ledger={"mode": "stratified"},
                          ungated_basis="BUDGET_EXHAUSTED")
    assert gated["coverage_attested"] == {"mode": "stratified"}
    assert gated["negative_basis"] == "GATE_VALIDATED"
    assert "coverage_attested" not in ungated, (
        "an unearned ledger is indistinguishable downstream from an earned one")
    assert ungated["route_to_human"] is True
    assert ungated["negative_basis"] == "BUDGET_EXHAUSTED"


def test_every_negative_declares_a_basis_that_exists():
    fin = inspect.getsource(A.run_chart_review)
    assert 'termination = "RUNTIME_ERROR"' in fin
    assert 'termination = "REJECTION_LOOP"' in fin
    assert 'termination = "SPEND_LIMIT"' in fin
    assert 'termination = "EXPANSION_LIMIT"' in fin
    assert 'termination = "MODEL_CALL_LIMIT"' in fin
    assert 'termination = "STOPPED_WITHOUT_ANSWER"' in fin
    assert 'termination = "BUDGET_EXHAUSTED"' not in fin, (
        "distinct runtime stop conditions must not be collapsed into one word")
    assert "attach_coverage_claim" in fin
    assert NO_COVERAGE_CLAIM in fin or "NO_COVERAGE_CLAIM" in fin


def test_a_positive_is_labelled_ungated_when_it_never_passed_the_gate():
    fin = inspect.getsource(A.run_chart_review)
    assert '"UNGATED"' in fin and "route_to_human" in fin, (
        "a FOUND answer that skipped the gate must say so; the old runtime stamped it True")


# ---------------------------------------------------- the wire that has been lost twice
# `_record_reads` is the route from "I read this document to its end" to "its `truncated`
# thread is discharged". It has now been deleted twice by accident: once when the runtime was
# ported (it lived in graph.py and was not carried over), and once by a range-based edit of
# this very file's neighbour whose slice happened to span the method. Both times the call site
# survived, so the failure was an AttributeError at the first read rather than anything
# resembling a missing rule.
#
# Pinned directly, not through a scripted run: the integration tests that caught it are exactly
# the ones a "clean up the old tests" pass is tempted to delete.

def test_the_read_recorder_exists_and_is_called_before_detection():
    import inspect

    assert hasattr(A.AuditMiddleware, "_record_reads"), (
        "the wire from a completed read to a discharged thread is gone; the call site will "
        "raise AttributeError on the first read of every run")
    hook = inspect.getsource(A.AuditMiddleware.wrap_tool_call)
    i, j = hook.index("_record_reads"), hook.index("_detect(")
    assert i < j, ("a read that completes a document must settle its thread BEFORE the same "
                   "result is rescanned for markers, or a window read can re-open what it closed")


def test_only_truncated_may_be_discharged_by_machine():
    """The recorder hands spans to the ledger; the ledger decides. `truncated` is the only
    marker whose predicate the runtime owns both sides of."""
    from acr.review.coverage_planner import MECHANICALLY_DISCHARGEABLE_MARKERS
    assert set(MECHANICALLY_DISCHARGEABLE_MARKERS) == {"truncated"}
