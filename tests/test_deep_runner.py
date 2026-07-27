"""The deepagents front end may not claim more than the gate gave it.

WHAT WAS WRONG
--------------
`coverage_attested` was written onto the MANIFEST — a top level `coverage.to_dict()` beside
`gate_validated`, computed on its own terms. Two consequences, and the second is the one that
matters:

  1. the claim did not have to follow the gate. A coverage ledger asserts "I searched the
     universe this spec defines", and the only thing that establishes that is the proof
     obligation `ChartReviewAgent._gate` evaluates. Anything else is a claim nothing checked.
  2. it sat OUTSIDE the answer, where `assert_answer_is_reportable` could not see it. The one
     rule that says who may claim coverage — `graph.assert_coverage_claim_is_earned` — was
     therefore never asked about a single deepagents run, while `evals.py` read that top-level
     key and `explain.py` read the answer's, so the two could disagree about the same run.

Downstream an unearned ledger is indistinguishable from an earned one. That is the whole
failure mode, and it is the same shape as the LangGraph defect where FOUND answers were
accepted unchecked and still stamped validated.

WHAT THESE TESTS HOLD
---------------------
  * every coverage claim this runtime makes is derived from the gate result;
  * the claim lands on the ANSWER, where the emission-time check can refuse it, and the
    manifest key is a copy of it and never a second computation;
  * there is still exactly ONE gate, borrowed by holding a `ChartReviewAgent`, and no second
    copy of the judgement has grown back beside it.

No provider is called anywhere in this file, no chart is read, and no manifest is written.
deepagents itself is not importable from this venv on purpose — `main` imports it lazily —
so the parts that can only be reached through a live agent loop are pinned structurally,
exactly as `tests/test_spec_insufficient.py` pins them.
"""
from __future__ import annotations

import inspect
import re

import pytest

import acr.deep_runner as D
from acr.answer_contract import CoverageClaimError, assert_answer_is_reportable

MAIN = inspect.getsource(D.main)
SRC = inspect.getsource(D)

LEDGER = {"mode": "stratified_exclusion", "n_read": 12, "searched_terms": ["carcinoma"]}


def _negative(**kw) -> dict:
    return {"status": "EVIDENCE_INSUFFICIENT", "value": {}, "reasoning": "", **kw}


# ==========================================================================================
# 1. THE CLAIM FOLLOWS THE GATE
# ==========================================================================================
def test_a_gated_negative_carries_its_ledger_and_says_which_gate_earned_it():
    ans = _negative()
    D.attach_coverage_claim(ans, gate_validated=True, ledger=LEDGER,
                            ungated_basis="BUDGET_EXHAUSTED")

    assert ans["negative_basis"] == "GATE_VALIDATED"
    assert ans["coverage_attested"] == LEDGER
    assert "route_to_human" not in ans
    assert_answer_is_reportable(ans)   # earned, and now checkable


def test_an_ungated_negative_carries_no_ledger_and_routes_to_a_human():
    """The defect, stated as the behaviour that replaces it: no gate, no claim, and the
    answer says so in words rather than by an absent key."""
    ans = _negative()
    D.attach_coverage_claim(ans, gate_validated=False, ledger=LEDGER,
                            ungated_basis="BUDGET_EXHAUSTED")

    assert "coverage_attested" not in ans, (
        "a ledger on an answer that never passed the proof obligation reads downstream "
        "exactly like one that did")
    assert ans["negative_basis"] == "BUDGET_EXHAUSTED"
    assert ans["route_to_human"] is True
    assert "no coverage claim" in ans["coverage_note"]
    assert_answer_is_reportable(ans)


def test_the_ledger_is_never_the_same_object_the_gate_did_not_see():
    """Same ledger in, same ledger out — the attestation is the run's own coverage state and
    not a summary assembled beside it."""
    ans = _negative()
    D.attach_coverage_claim(ans, gate_validated=True, ledger=LEDGER,
                            ungated_basis="BUDGET_EXHAUSTED")
    assert ans["coverage_attested"] is LEDGER


def test_how_an_ungated_run_ended_is_a_required_argument():
    """No default. "The agent ran out of room" and "the runtime fell over" have different
    owners, and a constant here would file every crash as the first one."""
    params = inspect.signature(D.attach_coverage_claim).parameters
    assert params["ungated_basis"].default is inspect.Parameter.empty
    assert params["gate_validated"].default is inspect.Parameter.empty
    assert "RUNTIME_ERROR" in MAIN and "crashed" in MAIN, (
        "main must distinguish a crashed loop from an exhausted one when it labels a "
        "negative that never passed the gate")


def test_the_pairing_is_enforced_and_not_merely_intended():
    """Both directions, because both were reachable: a claim without the gate, and — once
    the claim moved onto the answer — a gate-validated negative that quietly lost its ledger.
    """
    unearned = _negative(negative_basis="BUDGET_EXHAUSTED", coverage_attested=LEDGER)
    with pytest.raises(CoverageClaimError, match="only a gate-validated"):
        assert_answer_is_reportable(unearned)

    stripped = _negative()
    D.attach_coverage_claim(stripped, gate_validated=True, ledger=LEDGER,
                            ungated_basis="BUDGET_EXHAUSTED")
    stripped.pop("coverage_attested")
    with pytest.raises(CoverageClaimError, match="must carry"):
        assert_answer_is_reportable(stripped)


# ==========================================================================================
# 2. WHERE THE CLAIM IS WRITTEN, AND WHAT THE MANIFEST COPIES
# ==========================================================================================
def test_the_only_coverage_claim_is_derived_from_the_gate_result():
    """`gate_validated` is the run's own `state["accepted"]`, so the attestation has to be
    computed from that same value and from nothing else."""
    (call,) = re.findall(r"attach_coverage_claim\((.*?)\)\n", MAIN, re.DOTALL)
    assert 'gate_validated=state["accepted"]' in " ".join(call.split())
    assert MAIN.count("attach_coverage_claim(") == 1, (
        "one site, or the runtime has two opinions about what it searched")


def test_the_claim_is_attached_only_to_the_status_that_can_earn_it():
    i = MAIN.index('if answer.get("status") == "EVIDENCE_INSUFFICIENT":')
    j = MAIN.index("attach_coverage_claim(")
    assert 0 < j - i < 200, (
        "FOUND is proved by witness and SPEC_INSUFFICIENT is not a claim about this chart at "
        "all; neither may reach the coverage branch")


def test_the_check_that_can_refuse_the_claim_runs_after_it_is_written():
    """The old key was assembled inside the manifest literal, after the last check — which is
    why nothing ever refused it."""
    assert (MAIN.index("attach_coverage_claim(")
            < MAIN.index("assert_answer_is_reportable(answer)"))


def test_the_manifest_key_is_a_copy_of_the_answers_claim_never_a_second_computation():
    manifest_literal = MAIN[MAIN.index("manifest = {"):]
    assert '"coverage_attested"' not in manifest_literal, (
        "the manifest builds its own coverage key again — that is the defect, and it puts "
        "the claim back out of reach of assert_answer_is_reportable")
    assert "**coverage_claim," in manifest_literal
    (mirror,) = re.findall(r'coverage_claim = \((.*?)\)\n', MAIN, re.DOTALL)
    assert 'answer["coverage_attested"]' in mirror and '"coverage_attested" in answer' in mirror
    assert "coverage.to_dict()" not in mirror, (
        "recomputing the ledger here is how the manifest and the answer come to disagree "
        "about whether this run attested coverage")
    assert MAIN.count("coverage.to_dict()") == 1, (
        "the ledger is read once, and only where the gate said it was earned")


def test_a_spec_insufficient_answer_says_it_makes_no_coverage_claim():
    """Not an absent key. A reader filtering finished runs has to be able to tell "this run
    claimed nothing" from "this manifest predates the field"."""
    assert D.NO_COVERAGE_CLAIM.startswith("no coverage claim is made")
    i = MAIN.index('answer["proof_basis"] = "NOT_APPLICABLE"')
    assert 'answer["coverage_note"]' in MAIN[i:i + 400], (
        "SPEC_INSUFFICIENT is a statement about the specification; say that coverage of this "
        "chart is beside the point rather than leaving it to be inferred")


# ==========================================================================================
# 3. ONE GATE, AND NO SECOND COPY OF THE JUDGEMENT
# ==========================================================================================
def test_the_gate_is_borrowed_by_holding_the_agent_and_is_never_reimplemented():
    """A second copy of an audit rule is a liability: the two drift, and then a run's
    validation means whichever copy happened to execute."""
    assert SRC.count("gatekeeper._gate(") == 1
    assert "def _gate" not in SRC, "the gate has been copied into this runtime"
    assert "check_gate" not in SRC, "the gate's own helper is not this runtime's to call"
    assert not re.search(r"^\s*check_answer\(", SRC, re.MULTILINE), (
        "answer_checks is reached through the shared gate, never called beside it")
    assert "gatekeeper.run(" not in SRC and "gatekeeper.invoke(" not in SRC, (
        "the ChartReviewAgent is held for its gate; running its graph would be a second run")


def test_gate_validated_has_exactly_one_origin_in_this_runtime():
    """If this count ever exceeds one, the gate has grown a second door."""
    assignments = re.findall(r'state\["accepted"\]\s*=\s*True', MAIN)
    assert len(assignments) == 1
    assert MAIN.index('verdict = gatekeeper._gate(') < MAIN.index('state["accepted"] = True')
    assert '"gate_validated": state["accepted"]' in MAIN, (
        "the manifest's gate_validated must be the same value the coverage claim derives "
        "from, or the two can contradict each other")


def test_the_gate_judges_this_runs_own_ledgers():
    """A gate reading different ledgers from the ones the tools wrote would be validating a
    run that did not happen."""
    for line in ("gatekeeper.chart = chart", "gatekeeper.evidence = evidence",
                 "gatekeeper.coverage = coverage", "gatekeeper.toolbox = toolbox",
                 "gatekeeper.plan = plan", "gatekeeper.threads = threads"):
        assert line in MAIN, f"the borrowed gate does not share {line.split('=')[0].strip()}"


def test_submit_answer_cannot_bypass_the_gate():
    """The tool the model calls is replaced by the gated one, not added beside it."""
    assert 'tools = [t for t in tools if t.name != "submit_answer"]' in MAIN
    assert MAIN.count('func=_submit') == 1
