"""A retrieval term derived from the development set must not be the answer itself.

WHY THE PERMUTATION CONTROL IS NOT ENOUGH
-----------------------------------------
`assetdev.certify` already has one statistical defence: a search is refused when it gains just as
much over 19 permuted labellings, which is an exact one-sided permutation test at level 1/20. What
that catches is "the overall gain is not about retrieval".

It does not catch this one: the spec's answer is `20230412`, the labelling pass hands the model the
true outcome as a hint (the ReMedi practice), and so the model proposes `2023-04-12` as a keyword.
On the development set that term genuinely points at the answer **for every patient**, so shuffling
the labels genuinely does destroy it — the permutation test lets it through. And on test it is
worth nothing, because at test time nobody hands you the answer.

NOTATION IS THE WHOLE DIFFICULTY OF THIS CHECK
----------------------------------------------
Comparing strings directly catches nothing. One day is `20230412` / `2023-04-12` / `04/12/2023` /
`4/12/23`; one code is `C187` / `C18.7`; one morphology is `8140` / `8140/3`. A filter that only
compares literals would miss every leak shape that actually occurs, and it would **look like it
was working**.

The parts already exist: `code_tables`'s `normalize()` folds a code's notation, and `corpus`'s date
tolerance knows `2019-03-12` and `3/12/19` are the same day.
"""
from __future__ import annotations

import pytest

from acr.improvement.answer_leak import (
    AnswerLeak,
    leaking_terms,
    looks_like_answer,
)


# ------------------------------------------------------------- how a date can be written
@pytest.mark.parametrize("term", [
    "20230412", "2023-04-12", "04/12/2023", "4/12/2023", "2023/04/12", "12 Apr 2023",
])
def test_every_rendering_of_the_gold_date_is_caught(term: str):
    """One day has many spellings, and a leak is a leak in whichever one it uses."""
    assert looks_like_answer(term, "20230412"), term


@pytest.mark.parametrize("term", ["adenocarc", "pathology", "2023", "04", "biopsy", "diagnosis"])
def test_ordinary_terms_and_bare_fragments_are_not_flagged(term: str):
    """`2023` on its own is not a leak — it is a year, and it appears in nearly every chart.

    Counting it as one would turn this check into a source of noise, and a filter that is forever
    firing on ordinary values gets switched off by the next person.
    """
    assert not looks_like_answer(term, "20230412"), term


# ------------------------------------------------------------- how a code can be written
@pytest.mark.parametrize("term,gold", [
    ("C187", "C187"), ("C18.7", "C187"), ("c18.7", "C187"),
    ("8140", "8140"), ("8140/3", "8140"),
])
def test_code_notation_variants_are_caught(term: str, gold: str):
    assert looks_like_answer(term, gold), (term, gold)


def test_a_code_that_merely_shares_a_prefix_is_not_a_leak():
    """`C18` is a coarser site, not that answer. Judging by prefix would wrongly kill the whole
    class of terms."""
    assert not looks_like_answer("C18", "C187")
    assert not looks_like_answer("814", "8140")


# ------------------------------------------------------------------ embedded in a phrase
def test_a_term_containing_the_answer_as_a_token_is_caught():
    """`diagnosed 2023-04-12` leaks exactly as much as the bare date."""
    assert looks_like_answer("diagnosed 2023-04-12", "20230412")
    assert looks_like_answer("histology 8140", "8140")


def test_a_longer_number_that_merely_contains_the_digits_is_not_a_leak():
    """`120230412` contains that string of digits, but it is not that date. Judging by substring
    would wrongly kill it."""
    assert not looks_like_answer("120230412", "20230412")


# ------------------------------------------------------ in bulk: filtering one term list
def test_leaking_terms_reports_which_term_leaked_which_case():
    """The report has to say **which term** leaked **which case's** answer — saying only "there
    is a leak" leaves nothing to fix."""
    found = leaking_terms(
        terms=["adenocarc", "2023-04-12", "pathology", "8140"],
        gold_values={"SYN0001": ["20230412", "C341"], "SYN0002": ["8140"]},
    )
    assert [f.term for f in found] == ["2023-04-12", "8140"]
    assert found[0].patient_id == "SYN0001" and found[1].patient_id == "SYN0002"
    assert isinstance(found[0], AnswerLeak)


def test_a_clean_term_list_passes():
    assert leaking_terms(terms=["adenocarc", "biopsy"],
                         gold_values={"SYN0001": ["20230412"]}) == []


def test_an_empty_gold_value_is_not_a_universal_match():
    """An empty gold value would make "every term contains it" true, condemning the whole term
    list."""
    assert not looks_like_answer("anything", "")
    assert leaking_terms(terms=["a", "b"], gold_values={"P": ["", None]}) == []


def test_certify_refuses_a_plan_whose_keywords_leak():
    """The wiring itself needs a test: `keywords` is a sequence of (stratum, terms) pairs, so
    passing it straight in would compare tuples and silently find nothing — exactly the shape of
    a check that cannot fail."""
    from acr.improvement.answer_leak import leaking_terms
    from acr.improvement.assetdev import AnswerLeaked
    plan_keywords = (("can_establish", ("adenocarc", "2023-04-12")),
                     ("may_mention", ("biopsy",)))
    flat = [t for _, group in plan_keywords for t in group]
    assert flat == ["adenocarc", "2023-04-12", "biopsy"]
    assert [f.term for f in leaking_terms(
        terms=flat, gold_values={"SYN0001": ["20230412"]})] == ["2023-04-12"]
    assert issubclass(AnswerLeaked, Exception)
