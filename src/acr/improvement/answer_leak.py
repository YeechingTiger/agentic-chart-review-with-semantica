"""A derived retrieval term must not be the answer it was derived from.

WHY THE PERMUTATION CONTROL IS NOT ENOUGH
-----------------------------------------
`assetdev.certify` already reruns a candidate search against 19 permuted labellings and refuses
it unless it beats every one — an exact one-sided permutation test at level 1/20. That catches a
gain that is not about retrieval at all.

It does NOT catch this: the spec's answer is `20230412`, the labelling pass is given the true
outcome as a hint, and the model proposes `2023-04-12` as a keyword. On the development set that
term genuinely points at the answer for every patient, so shuffling the labels genuinely destroys
it — the permutation test lets it through. On test it is worthless, because nobody hands you the
answer there.

The hint-at-construction-time pattern is legitimate and worth having; the DEVELOP plane is
allowed to see the key. This is the filter that keeps the key from walking across into RUN
disguised as a search term.

NOTATION IS THE WHOLE DIFFICULTY
--------------------------------
A literal string comparison catches nothing that actually happens. One day is `20230412` /
`2023-04-12` / `04/12/2023` / `4/12/23`; one code is `C187` / `C18.7`; one morphology is `8140` /
`8140/3`. A filter that only compares literals would miss every real leak while appearing to
work, which is worse than not having it.

Two failure directions, and both are refused here. Matching too loosely kills ordinary terms —
`2023` alone is a year that appears in nearly every chart, and `C18` is a coarser site, not the
answer. Matching too tightly is the literal comparison above. So: candidates are compared as
whole tokens after notation folding, never as substrings and never by prefix.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: Month names a date can be spelled with. Three letters is enough — `12 Apr 2023` and
#: `April 12, 2023` both reduce to the same fold.
_MONTHS = {m: i for i, m in enumerate(
    ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"), 1)}

#: A gold value shorter than this cannot be distinctive. Behaviour digits (`3`) and single
#: letters would otherwise flag every term containing them.
MIN_DISTINCTIVE = 3


@dataclass(frozen=True)
class AnswerLeak:
    """One derived term that renders one case's gold value."""

    term: str
    patient_id: str
    gold: str

    def __str__(self) -> str:
        return f"{self.term!r} renders {self.patient_id}'s answer {self.gold!r}"


def _fold_code(s: str) -> str:
    """Punctuation and case out; the behaviour suffix after `/` dropped.

    Same rule `code_tables` declares per table, restated over a bare string because this runs
    before any table is known — the caller has a gold value and a term, not a value domain.
    """
    s = re.sub(r"[.\s]", "", str(s or "")).upper()
    return s.split("/", 1)[0] if "/" in s else s


def _dates(s: str) -> set[str]:
    """Every YYYYMMDD a string could be spelling. Empty when it is not a date.

    Ambiguity is kept rather than resolved: `04/12/2023` is April 12 in the US and December 4
    elsewhere, and a leak filter that guessed wrong would let one of the two through. Both go in
    the set, because flagging a term for the wrong-but-adjacent reading still flags a leak.
    """
    t = str(s or "").strip().lower()
    out: set[str] = set()
    if re.fullmatch(r"\d{8}", t):
        out.add(t)
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", t)
    if m:
        out.add(f"{m[1]}{int(m[2]):02d}{int(m[3]):02d}")
    m = re.fullmatch(r"(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})", t)
    if m:
        y = int(m[3]) + (2000 if int(m[3]) < 100 else 0)
        out.add(f"{y}{int(m[1]):02d}{int(m[2]):02d}")     # US
        out.add(f"{y}{int(m[2]):02d}{int(m[1]):02d}")     # rest of the world
    m = re.fullmatch(r"(\d{1,2})\s+([a-z]{3,})\.?\s+(\d{4})", t)
    if m and m[2][:3] in _MONTHS:
        out.add(f"{m[3]}{_MONTHS[m[2][:3]]:02d}{int(m[1]):02d}")
    m = re.fullmatch(r"([a-z]{3,})\.?\s+(\d{1,2}),?\s+(\d{4})", t)
    if m and m[1][:3] in _MONTHS:
        out.add(f"{m[3]}{_MONTHS[m[1][:3]]:02d}{int(m[2]):02d}")
    return out


def _tokens(term: str) -> list[str]:
    """A phrase split into candidate values. `diagnosed 2023-04-12` leaks exactly as much as the
    bare date, so a term is checked token by token as well as whole."""
    t = str(term or "").strip()
    return [t, *re.split(r"[\s,;]+", t)] if t else []


def looks_like_answer(term: str, gold: str) -> bool:
    """Does `term` render `gold` under any notation this corpus uses?

    Whole tokens only. `120230412` contains the digits of `20230412` and is a different number;
    `C18` is a prefix of `C187` and is a different site. Substring or prefix matching would flag
    both, and a filter that fires on ordinary values gets switched off.
    """
    g = str(gold or "").strip()
    if len(g) < MIN_DISTINCTIVE:
        return False
    gold_dates = _dates(g)
    gold_code = _fold_code(g)
    for tok in _tokens(term):
        if not tok:
            continue
        if gold_dates and (_dates(tok) & gold_dates):
            return True
        if len(gold_code) >= MIN_DISTINCTIVE and _fold_code(tok) == gold_code:
            return True
    return False


def leaking_terms(*, terms: Sequence[str],
                  gold_values: Mapping[str, Sequence[str | None]]) -> list[AnswerLeak]:
    """Every (term, case) pair where the term renders that case's answer.

    Reports the pair rather than a count: "there is a leak" is not actionable, and the term has
    to be removed from the list while the case tells the author which hint produced it.
    """
    out: list[AnswerLeak] = []
    for term in terms:
        for patient_id, golds in gold_values.items():
            for gold in golds or ():
                if gold and looks_like_answer(term, str(gold)):
                    out.append(AnswerLeak(term=term, patient_id=patient_id, gold=str(gold)))
                    break
            else:
                continue
            break
    return out
