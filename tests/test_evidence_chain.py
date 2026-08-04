"""A causal chain has to be a **chain**, not a set of annotations.

Today `because` is prose, and the schema says "Recorded, never checked". So
`detect_uncaused_reads` can only count whether there is one, never check whether it is true — a
fabricated reason and a true one look exactly alike in the record.

A chain needs exactly one thing: **every label is a resolvable pointer at another artefact, by ID**.
Trace events already have `seq`, so the anchor already exists; what is missing is letting `because`
carry it, and a resolver that walks the pointer.

Once it walks, three things become countable that cannot be counted today:

  * the resolution-failure rate — deterministic, needing no model judgement at all;
  * **forward references** — a read claiming it was caused by an event that comes after it, which is
    provably impossible, and which is exactly the shape a model produces when it writes a reason
    after the fact for an action it has already taken;
  * grounding ratio and chain depth — numbers an evaluation can hang on.

Backwards compatibility is a hard requirement: `because` is a string in every run recorded so far,
and those runs must stay readable and **must not be recorded as failures** — a prose reason is not a
fabrication, it is merely uncheckable, and the difference between those two is what this whole file
is about.
"""
from __future__ import annotations

from acr.evaluation import evals as E
from acr.evaluation.evidence_chain import (
    FORWARD_REF,
    GROUNDED,
    PROSE_ONLY,
    UNRESOLVED_REF,
    UNSOURCED,
    chain_report,
)


def _run(*calls: dict) -> E.RunRecord:
    """A RunRecord whose trace is the given tool calls, numbered from 1."""
    trace = [{"seq": i, "kind": "tool", "tool": c.pop("tool", "read_document"), **c}
             for i, c in enumerate(calls, start=1)]
    rec = E.RunRecord({"patient_id": "SYN0001", "spec_id": "S"}, source="synthetic")
    rec.trace = trace
    return rec


def _status(rep: dict) -> list[str]:
    return [link["status"] for link in rep["links"]]


# ------------------------------------------------------------------ the four link states
def test_a_pointer_at_an_earlier_event_resolves():
    rep = chain_report(_run(
        {"tool": "search_notes", "args": {"q": "adenocarc"}},
        {"args": {"doc": "path-1"}, "because": {"why": "the search surfaced it",
                                                "from": {"event": 1}}},
    ))
    assert _status(rep) == [UNSOURCED, GROUNDED]
    assert rep["links"][1]["why"] == "the search surfaced it"


def test_a_pointer_at_an_event_that_does_not_exist_is_unresolved():
    rep = chain_report(_run(
        {"args": {"doc": "path-1"}, "because": {"why": "x", "from": {"event": 99}}},
    ))
    assert _status(rep) == [UNRESOLVED_REF]


def test_a_pointer_at_a_LATER_event_is_impossible_and_named_as_such():
    """A call cannot have been caused by something that had not happened yet.

    This is the one **purely deterministic** new detector this change buys, and what it catches is
    precisely the shape of a reason written after the fact: the action was already taken, the reason
    was written looking back, and so it points at something that did not exist at the time. Given
    its own state rather than folded into UNRESOLVED_REF, because "pointed at the wrong step" and
    "pointed at the future" are two different failures, and the second one cannot be a typo.
    """
    rep = chain_report(_run(
        {"args": {"doc": "path-1"}, "because": {"why": "x", "from": {"event": 2}}},
        {"tool": "search_notes", "args": {"q": "later"}},
    ))
    assert _status(rep) == [FORWARD_REF, UNSOURCED]


def test_a_pointer_at_itself_is_also_forward():
    """A self-reference is a cycle, not a chain. The same rule handles it: a target `seq` that is
    not smaller than the caller's own is not legal."""
    rep = chain_report(_run(
        {"args": {"doc": "d"}, "because": {"why": "x", "from": {"event": 1}}},
    ))
    assert _status(rep) == [FORWARD_REF]


# -------------------------------------------------------------- backwards compatibility
def test_a_plain_string_because_is_prose_not_a_failure():
    """Every recorded run has this shape. Prose cannot be checked, but it is not a fabrication.

    Split out as PROSE_ONLY rather than folded into GROUNDED or UNSOURCED: folding it into the first
    calls something checked that cannot be checked, and folding it into the second records every run
    that ever wrote a careful reason as having written none. Either one makes the number useless.
    """
    rep = chain_report(_run(
        {"args": {"doc": "d"}, "because": "the search that surfaced this document"},
    ))
    assert _status(rep) == [PROSE_ONLY]
    assert rep["links"][0]["why"] == "the search that surfaced this document"


def test_a_because_object_with_no_pointer_is_prose_too():
    rep = chain_report(_run(
        {"args": {"doc": "d"}, "because": {"why": "no pointer here"}},
    ))
    assert _status(rep) == [PROSE_ONLY]


def test_a_malformed_pointer_does_not_crash_and_is_unresolved():
    """A broken label has to become a record, not an exception — the evaluation runs over runs
    somebody else already produced."""
    for bad in ({"why": "x", "from": {"event": "not-a-number"}},
                {"why": "x", "from": "not-a-mapping"},
                {"why": "x", "from": {}}):
        rep = chain_report(_run({"args": {"doc": "d"}, "because": bad}))
        assert _status(rep) == [UNRESOLVED_REF], bad


# ------------------------------------------------- the three numbers an evaluation hangs on
def test_the_grounding_ratio_counts_only_resolvable_links():
    """The denominator is **every call that could carry a because**; the numerator is only the ones
    that actually resolved.

    Prose stays out of the numerator. The whole use of this ratio is to separate "a reason was
    written" from "the reason can be checked" — count prose in, and the number collapses back into
    what `detect_uncaused_reads` already counts.
    """
    rep = chain_report(_run(
        {"tool": "search_notes", "args": {"q": "a"}},
        {"args": {"doc": "d1"}, "because": {"why": "x", "from": {"event": 1}}},
        {"args": {"doc": "d2"}, "because": "prose"},
        {"args": {"doc": "d3"}},
    ))
    assert rep["n_links"] == 4
    assert rep["n_grounded"] == 1
    assert rep["grounding_ratio"] == 0.25
    assert rep["n_prose_only"] == 1 and rep["n_unsourced"] == 2   # the search and d3


def test_the_chain_is_walkable_and_its_depth_is_reported():
    """What makes a chain a chain: it can be walked. A depth of 1 says every step hangs off the root
    and nothing else, which is not the same thing as a real line of reasoning."""
    rep = chain_report(_run(
        {"tool": "list_documents", "args": {}},
        {"tool": "search_notes", "args": {"q": "a"},
         "because": {"why": "the inventory named this type", "from": {"event": 1}}},
        {"args": {"doc": "d"}, "because": {"why": "the search surfaced it",
                                           "from": {"event": 2}}},
    ))
    assert rep["max_depth"] == 2                       # 3 <- 2 <- 1, two hops
    assert rep["links"][2]["chain"] == [3, 2, 1]


def test_a_cycle_cannot_hang_the_walk():
    """Two events pointing at each other is impossible (the backwards-only rule already rules it
    out), but the resolver must not depend on that rule to avoid looping forever."""
    rec = _run({"args": {"doc": "a"}}, {"args": {"doc": "b"}})
    rec.trace[0]["because"] = {"why": "x", "from": {"event": 2}}
    rec.trace[1]["because"] = {"why": "y", "from": {"event": 1}}
    rep = chain_report(rec)                            # not hanging is the pass condition
    assert _status(rep) == [FORWARD_REF, GROUNDED]


def test_an_empty_run_reports_no_ratio_rather_than_zero():
    """0/0 reported as 0.0 reads as "grounded in nothing" when the fact is that there was no call to
    judge."""
    rep = chain_report(_run())
    assert rep["n_links"] == 0 and rep["grounding_ratio"] is None


# ================================================== the claim layer: every sentence in the prose
# The chain over tool calls answers "why this step was taken". The attribution report and L5 explain
# are **prose** artefacts, and every causal claim in them currently carries no label at all — that
# is the one place in this repo that produces prose and never checks its grounding.
#
# `claim_report` reuses the same pointer format and the same set of states, but **does not know the
# attribution report's schema**: it takes a generic list of claims, and the caller adapts. Because
# evidence_chain sits on the evaluation plane, making it import diagnosis's structures would weld
# the two planes together.
def test_a_claim_pointing_at_a_real_trace_event_is_grounded():
    from acr.evaluation.evidence_chain import claim_report
    run = _run({"tool": "search_notes", "args": {"q": "a"}},
               {"args": {"doc": "d"}})
    rep = claim_report([
        {"text": "the run never searched for the abbreviation",
         "because": {"why": "trace shows one search", "from": {"event": 1}}},
        {"text": "and it read the wrong document"},
    ], run)
    assert [c["status"] for c in rep["claims"]] == [GROUNDED, UNSOURCED]
    assert rep["grounding_ratio"] == 0.5


def test_a_claim_may_cite_an_evidence_entry_instead_of_an_event():
    """Attribution often has to point at **the evidence the answer cited**, not at a step that was
    taken. Both anchors have to be pointable."""
    from acr.evaluation.evidence_chain import claim_report
    run = _run({"args": {"doc": "d"}})
    run.manifest["evidence"] = [{"note_id": "path-1", "start": 10, "end": 20}]
    rep = claim_report([
        {"text": "the cited span does not mention behaviour",
         "because": {"why": "read the quote", "from": {"evidence": 0}}},
        {"text": "and neither does the second one",
         "because": {"why": "x", "from": {"evidence": 7}}},
    ], run)
    assert [c["status"] for c in rep["claims"]] == [GROUNDED, UNRESOLVED_REF]


def test_a_claim_with_prose_only_is_not_counted_as_grounded():
    """Same rule as the call layer: uncheckable does not mean fabricated, and it does not mean
    checked either."""
    from acr.evaluation.evidence_chain import claim_report
    rep = claim_report([{"text": "t", "because": "it seemed likely"}], _run())
    assert [c["status"] for c in rep["claims"]] == [PROSE_ONLY]
    assert rep["grounding_ratio"] == 0.0


def test_no_claims_reports_no_ratio():
    from acr.evaluation.evidence_chain import claim_report
    assert claim_report([], _run())["grounding_ratio"] is None
