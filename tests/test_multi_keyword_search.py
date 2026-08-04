"""One search call takes several keywords, and each term's hits are reported separately.

WHY THIS IS THE TOOL'S PROBLEM AND NOT A POLICY'S
-------------------------------------------------
Measured in E2: `search-breadth-first` issued 506 searches over eighteen charts, eighty-six of them
for `biopsy` alone. That was read at the time as "this card teaches the model to do useless work",
but that is only half of it — the other half is that **the tool took one term per call**, so
"cover this chart with five terms" was necessarily five calls in the record, however the policy was
written.

Put the term list into one call and the same coverage costs 1 call instead of N. This changes no
policy; it turns "sweep wide" from a budget problem back into a retrieval problem — and the other
half of E2's finding was that breadth-first opened FEWER documents than the empty slot (2.6 against
3.3), because the budget was burned on issuing searches.

EACH TERM'S HITS MUST STAY SEPARATE
-----------------------------------
Merging them into one pool throws away the one piece of information that is any use: **which term
pulled this document out**. In E2's chains, `read ←9` and `read ←10` can be told apart as to which
search did the work only because each search is its own event. A multi-term call that flattens its
results throws that information straight back away.
"""
from __future__ import annotations

import pytest

from acr.chartstore.corpus import Corpus
from acr.core import site

CORPUS = str(site.corpus_root())


@pytest.fixture(scope="module")
def chart():
    return Corpus(CORPUS).chart("SYN0001")


def test_a_single_string_still_works(chart):
    """Backward compatibility: every run already recorded passes a string."""
    from acr.review.tools.toolbox import Toolbox
    out = Toolbox.search_many(chart, "adenocarc", max_hits=25)
    assert out["terms"] == ["adenocarc"]
    assert out["by_term"]["adenocarc"]["n_hits"] >= 1


def test_several_terms_in_one_call(chart):
    from acr.review.tools.toolbox import Toolbox
    out = Toolbox.search_many(chart, ["adenocarc", "biopsy", "zzznotathing"], max_hits=25)
    assert out["terms"] == ["adenocarc", "biopsy", "zzznotathing"]
    assert set(out["by_term"]) == {"adenocarc", "biopsy", "zzznotathing"}
    assert out["by_term"]["zzznotathing"]["n_hits"] == 0


def test_hits_stay_attributed_to_the_term_that_found_them(chart):
    """Merging into one pool loses "which term pulled this document out", and that is the one
    distinction in the causal chain that is any use."""
    from acr.review.tools.toolbox import Toolbox
    out = Toolbox.search_many(chart, ["adenocarc", "biopsy"], max_hits=25)
    for term, block in out["by_term"].items():
        for h in block["hits"]:
            assert "note_id" in h and "start" in h, term


def test_the_cap_is_per_term_not_shared(chart):
    """A shared cap would let the first term eat the budget and make every later term look like
    "not in this chart"."""
    from acr.review.tools.toolbox import Toolbox
    out = Toolbox.search_many(chart, ["a", "e"], max_hits=3)
    for block in out["by_term"].values():
        assert block["n_hits"] <= 3


def test_an_empty_term_list_is_refused_not_silently_empty(chart):
    """An empty term list returning "zero hits" reads as "this chart contains nothing"."""
    from acr.review.tools.toolbox import Toolbox
    out = Toolbox.search_many(chart, [], max_hits=5)
    assert out.get("error")
