"""The search matcher: notation-tolerant, not synonym-aware.

The model chooses the term; the matcher's only job is to not lose a hit because the corpus
spelled the same phrase a different way. Where that line sits is measured, not asserted by
taste, and both halves of the measurement are pinned below.

WHAT FOLDING SEPARATORS BUYS. Over the 12,190 diagnosis-bearing documents in this corpus,
documents found by a literal match against documents found by the folded one:

    non-small cell            2251 -> 2476   +225  (+10.0%)
    small cell carcinoma      1537 -> 1552    +15
    right upper lobe          1270 -> 1277     +7
    right lower lobe           876 ->  883     +7
    squamous cell carcinoma   2615 -> 2623     +8
    main bronchus               44 ->   45     +1
    invasive adenocarcinoma    147 ->  149     +2
    ten phrases, total       16797 -> 17064   +267   (+1.6%)

The hyphen case is real; the line-wrap case is small. Both are free and neither needs a word
list.

WHY SYNONYMS STAY OUT. Same corpus, share of diagnosis-bearing documents each single term
appears in:

    carcinoma 57.5%   malignan 50.1%   tumor 36.8%   cancer 32.7%
    adenocarcinoma 27.6%   neoplasm 4.3%   tumour 0.0%

677 documents (5.6%) contain `cancer` and not `carcinoma`, and 23.9% contain none of those
seven. No fixed list is close to complete, so a matcher that quietly expanded a term into a
synonym set would be guessing for the model and hiding its own miss rate inside a hit. Recall
comes from the model searching several times and reading what comes back.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from acr.corpus import Corpus, _notation_tolerant

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "patients"


def matches(query: str, text: str) -> bool:
    return re.search(_notation_tolerant(query), text, re.IGNORECASE) is not None


# ------------------------------------------------------------------ separators fold
@pytest.mark.parametrize("text", [
    "poorly differentiated non-small cell carcinoma",   # hyphen, as pathologists write it
    "poorly differentiated non small cell carcinoma",   # space, as clinicians write it
    "poorly differentiated non-small  cell carcinoma",  # doubled whitespace
    "poorly differentiated non-small\ncell carcinoma",  # hard line wrap
    "poorly differentiated NON-SMALL CELL carcinoma",   # case
])
def test_one_query_finds_every_spelling_of_the_same_phrase(text):
    """+225 documents on this one phrase, which is the whole measured gain."""
    assert matches("non-small cell", text)
    assert matches("non small cell", text), "the query's own notation must not matter either"


def test_a_phrase_broken_across_a_line_wrap_is_found():
    """Real quote shape from a recorded trace: 'bronchus of right upper\\nlobe'."""
    assert matches("right upper lobe", "bronchus of right upper\nlobe\nWill follow")


def test_a_single_token_query_is_unchanged_plain_substring_behaviour():
    """`lobe` matched inside `lobes` before this change and must still, or the fold has
    quietly become a word-boundary rule."""
    assert _notation_tolerant("lobe") == "lobe"
    assert matches("lobe", "two lobes were resected")
    assert matches("carcinoma", "adenocarcinoma, moderately differentiated")


# ------------------------------------------------------------------ synonyms do not
def test_the_matcher_does_not_fold_synonyms():
    """`cancer` must not find `carcinoma`. 677 documents hold one and not the other, and a
    matcher that bridged them would report a hit where the model's term genuinely missed."""
    assert not matches("cancer", "invasive carcinoma of the lung")
    assert not matches("carcinoma", "known lung cancer, on treatment")


def test_the_matcher_does_not_split_inside_a_token():
    """`nonsmall` -> `non-small` would need a word list, which is the thing being refused."""
    assert not matches("nonsmall cell", "non-small cell carcinoma")


def test_a_metacharacter_in_a_literal_query_is_not_a_pattern():
    """A model that searches for `C34.1` must not have `.` treated as any-character."""
    assert matches("C34.1", "coded C34.1 per ICD-O-3")
    assert not matches("C34.1", "coded C3451 per ICD-O-3")


# ------------------------------------------------------------------ through the real API
def test_regex_true_is_still_honoured_verbatim():
    chart = Corpus(CORPUS).chart("SYN0002")
    hits = chart.search(r"sigmoid|colon", regex=True, max_hits=3)
    assert hits, "a caller that wrote its own pattern gets it"


def test_a_hit_carries_the_offsets_a_read_can_use():
    chart = Corpus(CORPUS).chart("SYN0002")
    docs, _ = chart.list_documents(limit=1)
    word = next(w for w in chart.read(docs[0].note_id, 0, 2000)["text"].split()
                if len(w) > 5 and w.isalpha())
    hit = next(h for h in chart.search(word, max_hits=5))
    assert isinstance(hit.date, str) and hit.start < hit.end
    assert word.lower() in chart.quote(hit.note_id, hit.start, hit.end).lower()


def test_the_snippet_window_default_is_the_widened_one():
    """Widened from +/-160 to +/-250 characters: the point of a hit list is that the model can
    tell from the snippet whether the document is worth opening, without opening it.

    Asserted on the signature as well as on a document, because the fixture's longest note is
    657 characters -- shorter than two full windows -- so a document alone cannot distinguish
    160 from 250.
    """
    import inspect
    assert inspect.signature(Corpus.chart(Corpus(CORPUS), "SYN0002").search
                             ).parameters["context"].default == 250

    chart = Corpus(CORPUS).chart("SYN0002")
    note_id = max(((chart.read(d.note_id, 0, 10 ** 6)["total_chars"], d.note_id)
                   for d in chart.list_documents(limit=999)[0]))[1]
    text = chart.read(note_id, 0, 10 ** 6)["text"]
    word = next(w for w in text[300:].split() if len(w) > 5 and w.isalpha())
    hit = next(h for h in chart.search(word, max_hits=1) if h.note_id == note_id)

    # Pinned against the formula rather than against a length, because the window clips at both
    # document ends and the fixture's longest note is 657 characters -- shorter than two full
    # windows -- so a raw length cannot tell 160 from 250 here.
    expected = text[max(0, hit.start - 250):min(len(text), hit.end + 250)]
    assert hit.snippet == expected.replace("\n", " ").strip()


def test_dates_and_types_come_back_so_the_model_can_choose_reading_order():
    """The hit list IS the affordance: names, dates, and a window. Nothing ranks for it."""
    chart = Corpus(CORPUS).chart("SYN0002")
    hits = chart.search("colon", max_hits=10)
    assert hits
    for h in hits:
        assert h.doc_type and h.note_id
        date.fromisoformat(h.date)


# ======================================================================================
# NOTATION ADDED 2026-07-31: unicode dashes, the solidus, quote marks, date renderings.
#
# All four are aimed at text that has been through Word, a PDF or a dictation front end. This
# corpus is generated and writes ASCII hyphens, straight quotes and ISO dates by construction,
# so `tools/measure_matcher.py` reports "not exercised here" for most of them — which is not
# evidence they are worthless, and is the reason these tests assert on constructed strings
# rather than on corpus counts.
#
# The measurement earned its place on its first run anyway, in the negative direction: see
# `test_a_two_character_token_does_not_match_the_inside_of_other_words`.
# ======================================================================================

import re

from acr.corpus import _notation_tolerant


def found(query: str, text: str) -> bool:
    return bool(re.search(_notation_tolerant(query), text, re.IGNORECASE))


@pytest.mark.parametrize("dash", ["-", "‐", "‑", "–", "—", "−"])
def test_any_dash_is_the_same_dash(dash: str):
    """The class was `[\\s\\-]+`, ASCII only. Nothing in a generated corpus can show that gap:
    this generator writes hyphens. Word and most PDF extractors do not."""
    assert found("non-small cell", f"non{dash}small cell")


def test_the_solidus_relates_the_two_ways_an_abbreviation_is_written():
    assert found("c/w", "path c/w adenoCA")
    assert found("c/w", "path c w adenoCA")


def test_a_two_character_token_does_not_match_the_inside_of_other_words():
    """THE MEASUREMENT CAUGHT THIS, AND READING DID NOT.

    Putting the solidus in the separator class turned `s/p` into `s[sep]+p`, and
    `tools/measure_matcher.py` reported it finding 43 documents where a literal found one. Every
    one of the 43 was noise — "lungs Plan", "masses present" — so the gain column said +43 for a
    tolerance that had become strictly harmful.

    A token of one or two characters is a fragment, not a word, so a pattern whose shortest
    token is that short gets word boundaries.
    """
    assert found("s/p", "s/p right lobectomy")
    assert not found("s/p", "lungs Plan")
    assert not found("s/p", "no masses present")
    assert not found("c/w", "specific work")


def test_the_boundary_guard_does_not_touch_ordinary_substring_behaviour():
    """`lobe` must keep matching inside `lobes`. A trailing `\\b` would end that, so anchors go
    on only when the shortest token is a fragment."""
    assert found("lobe", "both lower lobes")
    assert found("carcinoma", "adenocarcinomatous")
    assert found("right upper lobe", "the right  upper\nlobe")


@pytest.mark.parametrize("written", [
    "3/12/19", "03/12/2019", "March 12, 2019", "Mar 12 2019", "12 March 2019", "20190312",
])
def test_one_calendar_day_is_found_however_it_is_written(written: str):
    """Closed, decidable, and every member denotes the same day — which is what separates this
    from a synonym set. STORE.390's answer IS a date: a run that reads "the 3/12/19 nodule" and
    then searches `2019-03-12` for the study it names gets nothing, and the miss looks like an
    absent document."""
    assert found("2019-03-12", f"reviewed the {written} study")


def test_a_date_query_does_not_match_a_different_day():
    assert not found("2019-03-12", "the 2019-03-13 study")
    assert not found("2019-03-12", "the 3/13/19 study")


@pytest.mark.parametrize("mark", ["'", "‘", "’", "ʼ", "´"])
def test_any_apostrophe_is_the_same_apostrophe(mark: str):
    assert found("patient's", f"the patient{mark}s wishes")


def test_widening_quotes_is_one_pass_not_one_per_character():
    """A loop of `str.replace` per quote character corrupts its own output: the first pass
    inserts a class CONTAINING the marks the later passes search for, so the second rewrites the
    inside of the class the first built and the result matches nothing."""
    pattern = _notation_tolerant("patient's")
    assert pattern.count("[") == 1, f"the quote class was rewritten by a later pass: {pattern}"


def test_the_refusal_to_fold_synonyms_survives_all_of_it():
    assert not found("cancer", "carcinoma")
    assert not found("adenocarcinoma", "adenoCA")
    assert not found("right upper lobe", "RUL")
    assert not found("nonsmall", "non-small")
