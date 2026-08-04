"""The two implementations of "which terms did this run search" must not drift apart.

There were THREE. `evals.RunRecord.searched_terms` and `contract.behaviour`'s signature builder both
did `str(args["query"])`, which turned a batched `["bx", "adenocarcinoma"]` into one opaque token, and
`tools/analyze_arms.py` once had a third that disagreed with `RunRecord.searched_terms` after the
number it printed had already been written into a document.

Two remain, and they remain on purpose: `tests/test_layering.py` forbids `contract/` from importing
`evaluation/`, so the rule cannot be shared by import. What makes that acceptable rather than a
repeat of the same defect is this file — a duplicate with a test comparing it to the canonical one
is a different thing from a duplicate nobody checks.
"""

from __future__ import annotations

import pytest

from acr.contract.behaviour import _query_terms as behaviour_terms
from acr.evaluation.evals import _query_terms as evals_terms

CASES = [
    None,
    "adenocarcinoma",
    ["bx", "adenocarcinoma"],
    ("carcinoma",),
    [],
    ["", "carcinoma"],
    ["2023-04-12"],
    [1, 2],
    # THE CLASS THIS FILE ORIGINALLY OMITTED, and the two implementations diverged on exactly it:
    # `evals` dropped a whitespace-only term (`if str(t).strip() or t == ""`) and `behaviour` kept
    # it. A test that compares two implementations is only as good as its input classes.
    ["  "],
    ["\t"],
    ["  ", "carcinoma"],
    {"carcinoma"},
]


@pytest.mark.parametrize("query", CASES)
def test_both_implementations_return_the_same_terms(query):
    assert evals_terms(query) == behaviour_terms(query), query


@pytest.mark.parametrize("query", CASES)
def test_no_term_is_ever_a_stringified_collection(query):
    """The bug's signature. `"['bx', 'adeno']"` is not a term anybody searched for."""
    for t in evals_terms(query):
        assert not t.startswith(("[", "(", "{")), t


@pytest.mark.parametrize("query", [["  "], ["\t"], [""], ["  ", "carcinoma"]])
def test_a_blank_term_survives_to_the_detector(query):
    """Comparing two implementations to each other passes if both are wrong the same way. This pins
    the CORRECT answer: a blank search term is degenerate, and `detect_degenerate_search`'s first
    branch is `"empty" if not s` — so dropping it here hides the finding rather than making one."""
    terms = evals_terms(query)
    assert any(not t.strip() for t in terms), (
        f"{query!r} lost its blank term, which is exactly what detect_degenerate_search reports")
