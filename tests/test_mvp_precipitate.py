"""The precipitate verb: what has settled, what diverges, and which kind of gap each is.

The two divergence classes are the whole point, so they are tested against hand-built ledgers
where the answer is known by construction: two runs that cited the same documents and decided
differently is a judgement the contract has not settled; two runs that cited different
documents is a retrieval failure wearing a judgement's clothes. Getting these backwards would
send a guideline author to write the wrong rule, which is worse than reporting nothing.
"""
from __future__ import annotations

from typing import Any

from acr.mvp.precipitate import render, survey


class FakeLedger:
    """Only the one method survey() uses. Rows in the shape SemanticaLedger.decisions returns."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def decisions(self, *, category_prefix: str | None = None,
                  case_id: str | None = None) -> list[dict[str, Any]]:
        out = self.rows
        if category_prefix:
            out = [r for r in out if str(r.get("category", "")).startswith(category_prefix)]
        if case_id:
            out = [r for r in out if r.get("case_id") == case_id]
        return out


def _d(dtype: str, case: str, facing: str, outcome: str, used: list[str],
       reasoning: str = "because", run: str | None = None, seq: int = 1,
       n_searches: int = 2, options: list[str] | None = None,
       unverified: list[str] | None = None) -> dict[str, Any]:
    from acr.mvp.decision_types import level_of
    lvl = level_of(dtype, on_action=True)
    return {"decision_id": f"{case}-{dtype}-{seq}", "category": f"{lvl}:{dtype}",
            "scenario": facing, "outcome": outcome, "reasoning": reasoning,
            "decision_maker": "model", "case_id": case, "run_id": run or f"run-{case}",
            "seq": seq, "used": used, "used_unverified": unverified or [],
            "options": options or [], "context": {"n_searches": n_searches, "n_evidence": 1}}


def test_same_information_different_call_is_a_judgement_divergence():
    """Both camps read the same two documents and still split — the contract owes a rule."""
    facing = "ambiguous cytology and a later confirmatory biopsy both date the case"
    shared = ["note:cytology_2022-02-14", "note:biopsy_2022-03-09"]
    ledger = FakeLedger([
        _d("which_wins", "A", facing, "the biopsy dates the case", shared),
        _d("which_wins", "B", facing, "the biopsy dates the case", shared),
        _d("which_wins", "C", facing, "the cytology dates the case", shared),
    ])
    report = survey(ledger)
    section = report["sections"][0]
    assert section["decision_type"] == "which_wins"
    situation = section["situations"][0]
    assert situation["status"] == "divergent"
    assert situation["divergence"]["kind"] == "judgement"
    assert situation["divergence"]["input_overlap"] == 1.0
    assert "Decision Rule" in situation["divergence"]["remedy"]
    assert [o["n"] for o in situation["outcomes"]] == [2, 1]   # majority first


def test_different_information_different_call_is_an_information_divergence():
    """The camps split because one had looked at the impression note and the other had not —
    the remedy is upstream of the judgement."""
    facing = "ambiguous cytology and a later confirmatory biopsy both date the case"
    ledger = FakeLedger([
        _d("which_wins", "A", facing, "the cytology dates the case",
           ["note:cytology_2023-04-12", "note:onc_impression_2023-04-12"]),
        _d("which_wins", "B", facing, "the cytology dates the case",
           ["note:cytology_2023-04-12", "note:onc_impression_2023-04-12"]),
        _d("which_wins", "C", facing, "the biopsy dates the case",
           ["note:biopsy_2023-04-27"]),
    ])
    situation = survey(ledger)["sections"][0]["situations"][0]
    assert situation["status"] == "divergent"
    d = situation["divergence"]
    assert d["kind"] == "information"
    assert d["input_overlap"] < 0.5
    assert "Coverage" in d["remedy"]
    assert "note:onc_impression_2023-04-12" in d["inputs_only_in_majority"]
    assert d["inputs_only_in_minority"] == ["note:biopsy_2023-04-27"]


def test_a_repeated_agreement_is_reported_as_settled():
    facing = "an absence claim is about to rest on two targeted searches"
    ledger = FakeLedger([
        _d("enough", c, facing, "do an unfiltered listing before claiming absence",
           ["search:impression"], seq=i)
        for i, c in enumerate(["A", "B", "C", "D"], start=1)
    ])
    situation = survey(ledger)["sections"][0]["situations"][0]
    assert situation["status"] == "settled"
    assert situation["n_decisions"] == 4
    assert situation["cases"] == ["A", "B", "C", "D"]
    assert "a clause could fix this" in situation["settled_note"]


def test_too_few_alike_decisions_are_thin_not_settled():
    facing = "an absence claim is about to rest on two targeted searches"
    ledger = FakeLedger([
        _d("enough", "A", facing, "list everything first", ["search:x"], seq=1),
        _d("enough", "B", facing, "list everything first", ["search:x"], seq=2),
    ])
    situation = survey(ledger)["sections"][0]["situations"][0]
    assert situation["status"] == "thin"
    assert "settled_note" not in situation
    # The threshold is the caller's to set — the same material settles at a lower bar.
    assert survey(ledger, settled_min=2)["sections"][0]["situations"][0]["status"] == "settled"


def test_unlike_situations_do_not_collapse_into_one():
    ledger = FakeLedger([
        _d("where_to_look", "A", "nothing examined yet; where does a diagnosis date live",
           "start from pathology terms", ["rule:field"]),
        _d("where_to_look", "B", "pathology searched and empty; broaden or stop",
           "broaden to clinician impressions", ["search:carcinoma"]),
    ])
    situations = survey(ledger)["sections"][0]["situations"]
    assert len(situations) == 2


def test_unverified_warrants_are_surfaced_per_type():
    ledger = FakeLedger([
        _d("enough", "A", "ready to submit", "submit FOUND", ["note:never_opened"],
           unverified=["note:never_opened"]),
    ])
    section = survey(ledger)["sections"][0]
    assert section["unverified_warrants"] == [
        {"case_id": "A", "seq": 1, "decided": "submit FOUND",
         "refs": ["note:never_opened"]}]


def test_filtering_by_type_and_the_empty_ledger():
    ledger = FakeLedger([
        _d("enough", "A", "f1", "o1", ["search:x"]),
        _d("which_wins", "A", "f2", "o2", ["search:y"]),
    ])
    only = survey(ledger, decision_type="enough")
    assert [s["decision_type"] for s in only["sections"]] == ["enough"]
    assert [s["level"] for s in only["sections"]] == ["big"]
    empty = survey(FakeLedger([]))
    assert empty["sections"] == [] and empty["n_decisions"] == 0
    assert "nothing recorded yet" in render(empty)


def test_render_puts_the_gap_and_its_remedy_in_front_of_the_reader():
    facing = "ambiguous cytology and a later confirmatory biopsy both date the case"
    shared = ["note:cytology", "note:biopsy"]
    ledger = FakeLedger([
        _d("which_wins", "A", facing, "the biopsy dates the case", shared,
           reasoning="no impression accompanies the cytology"),
        _d("which_wins", "B", facing, "the biopsy dates the case", shared),
        _d("which_wins", "C", facing, "the cytology dates the case", shared,
           options=["date by the biopsy"]),
    ])
    text = render(survey(ledger))
    assert "## [big] which_wins" in text
    assert "[DIVERGENT]" in text
    assert "JUDGEMENT divergence" in text
    assert "no impression accompanies the cytology" in text
    assert "used: note:biopsy, note:cytology" in text
    assert "searches when decided" in text
