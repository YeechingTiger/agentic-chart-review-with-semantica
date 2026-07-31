"""The six adversarial charts must keep being adversarial.

Each SYNX chart is built so that a straightforward pass reaches a specific WRONG date. That
property lives in the arrangement of documents, not in any assertion, so it can be destroyed by
an edit that looks harmless — adding the word "adenocarcinoma" to the shorthand note, emitting a
clinical impression on the cytology date, moving the addendum under the same document type as
the report it amends. Each of those would turn a trap into a giveaway, every arm would score it,
and nothing would say so: the chart would still generate, still carry a ground truth, still be
answerable. It would simply stop measuring anything.

So what is pinned here is the TRAP, not the answer. The answer is in `_ground_truth.json` and
`acr eval score` reads it. These tests assert the structural facts that make the answer hard to
reach, and each names the failure it prevents.

They read the corpus on disk rather than regenerating it, because the corpus on disk is what
runs are scored against.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

CORPUS = Path(__file__).resolve().parents[1] / "corpus" / "patients"
DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def chart(pid: str) -> list[Path]:
    d = CORPUS / pid
    if not d.is_dir():
        pytest.skip(f"{pid} not generated; run tools/generate_corpus.py")
    return sorted(d.glob("*.txt"), key=lambda f: DATE_RE.search(f.name).group(1))


def truth(pid: str) -> dict:
    return json.loads((CORPUS / pid / "_ground_truth.json").read_text(encoding="utf-8"))


def answer(pid: str) -> str:
    return truth(pid)["ground_truth"]["STORE.390.date_of_initial_diagnosis"]["value"]


def body(paths: list[Path]) -> str:
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


ALL_TRAPS = ["SYNX01", "SYNX02", "SYNX03", "SYNX04", "SYNX05", "SYNX06"]


@pytest.mark.parametrize("pid", ALL_TRAPS)
def test_every_trap_declares_the_wrong_answer_it_baits(pid: str):
    """`naive_answer` is what makes a failure countable.

    Without it a wrong run is just wrong; with it, a run that returns exactly this date is
    evidence the trap sprang rather than that the chart was hard in some general way.
    """
    expect = truth(pid)["expect"]
    assert expect.get("naive_answer"), f"{pid}: no naive_answer declared"
    assert expect["naive_answer"] != answer(pid), (
        f"{pid}: the bait equals the answer, so the trap cannot catch anything")


@pytest.mark.parametrize("pid", ALL_TRAPS)
def test_the_answer_is_not_simply_the_last_pathology_report(pid: str):
    """The shape every one of these exists to defeat.

    If the answer were the date of the most obvious pathology report, no traversal would need to
    do anything but find it, and the chart would rank arms no better than SYN0001 did.
    """
    paths = [p for p in chart(pid) if "Surgical-Pathology-Report" in p.name]
    dates = [DATE_RE.search(p.name).group(1).replace("-", "") for p in paths]
    assert answer(pid) not in dates or pid == "SYNX03", (
        f"{pid}: the answer IS a pathology report date; nothing here is hard")


def test_x01_the_retrospective_remark_and_the_scan_it_names_are_years_apart():
    """The two hops. Neither document alone yields the answer."""
    paths = chart("SYNX01")
    remark = [p for p in paths if "in retrospect" in p.read_text(encoding="utf-8")]
    assert len(remark) == 1, "exactly one note should carry the retrospective statement"
    remark_date = DATE_RE.search(remark[0].name).group(1)

    ans = answer("SYNX01")
    named = f"{ans[:4]}-{ans[4:6]}-{ans[6:]}"
    assert named in remark[0].read_text(encoding="utf-8"), (
        "the remark must NAME the earlier date, or the second hop has nowhere to land")
    assert [p for p in paths if named in p.name], "the scan the remark names must exist"
    assert remark_date > named, "the remark must come after the date it reaches back to"
    assert int(remark_date[:4]) - int(named[:4]) >= 2, (
        "a gap of under two years would let a single date-sorted read span both")


def test_x02_treatment_precedes_every_document_that_states_a_diagnosis():
    """`decision_rule`: treatment before a documented diagnosis dates the case.

    The infusion note must state NO diagnosis. If it did, the ordinary rule would date the case
    from it and there would be no trap — the answer would be reachable by searching diagnostic
    vocabulary, which is exactly what this chart is built to defeat.
    """
    paths = chart("SYNX02")
    ans = answer("SYNX02")
    infusion = [p for p in paths if DATE_RE.search(p.name).group(1).replace("-", "") == ans]
    assert infusion, "no document on the answer date"
    text = body(infusion).lower()
    for word in ("adenocarcinoma", "carcinoma", "malignan", "diagnos", "cancer"):
        assert word not in text, (
            f"the infusion record states {word!r}; it would then date the case by the ordinary "
            f"rule and the trap is gone")

    stated = [p for p in paths
              if "adenocarcinoma" in p.read_text(encoding="utf-8").lower()]
    earliest = min(DATE_RE.search(p.name).group(1).replace("-", "") for p in stated)
    assert ans < earliest, "treatment must precede every document that states the diagnosis"


def test_x03_differs_from_syn0001_by_exactly_one_absent_document():
    """The mirror branch of the cytology conflict rule.

    SYN0001: ambiguous cytology WITH a same-day clinical impression -> the cytology dates it.
    SYNX03: the same cytology WITHOUT one -> the biopsy dates it. The whole difference is a
    document that is not there, which no search can reveal — only noticing its absence does.
    """
    x03 = chart("SYNX03")
    cyto = [p for p in x03 if "Surgical-Pathology-Document" in p.name
            and "SUSPICIOUS FOR" in p.read_text(encoding="utf-8")]
    assert len(cyto) == 1, "SYNX03 needs exactly one ambiguous cytology"
    cyto_date = DATE_RE.search(cyto[0].name).group(1)

    same_day = [p for p in x03 if cyto_date in p.name]
    assert len(same_day) == 1, (
        f"a second document on {cyto_date} would supply the clinical impression this chart "
        f"exists to withhold")

    assert answer("SYNX03") != cyto_date.replace("-", ""), "the cytology must NOT date this case"

    syn1 = chart("SYN0001")
    s1_cyto = [p for p in syn1 if "SUSPICIOUS FOR" in p.read_text(encoding="utf-8")]
    s1_date = DATE_RE.search(s1_cyto[0].name).group(1)
    assert len([p for p in syn1 if s1_date in p.name]) > 1, (
        "SYN0001 must keep its same-day impression, or the two charts stop being a contrast")
    assert answer("SYN0001") == s1_date.replace("-", ""), (
        "SYN0001's answer IS its cytology date; that is the branch SYNX03 mirrors")


def test_x04_the_addendum_is_filed_under_a_different_type_than_the_report_it_settles():
    """Sweeping the pathology type finds a document that answers nothing."""
    paths = chart("SYNX04")
    deferred = [p for p in paths if "DEFERRED" in p.read_text(encoding="utf-8")]
    add = [p for p in paths if p.read_text(encoding="utf-8").startswith("=") or True]
    add = [p for p in paths if "ADDENDUM DIAGNOSIS" in p.read_text(encoding="utf-8")]
    assert len(deferred) == 1 and len(add) == 1

    def doctype(p: Path) -> str:
        return DATE_RE.split(p.name)[0].rstrip("_")

    assert doctype(deferred[0]) != doctype(add[0]), (
        "same document type: one type-filtered sweep would return both and the trap is gone")
    assert "SEE ADDENDUM" in deferred[0].read_text(encoding="utf-8"), (
        "without the pointer there is no thread to chase, only a gap to notice")
    assert answer("SYNX04") == DATE_RE.search(add[0].name).group(1).replace("-", "")


def test_x05_the_first_diagnosis_is_late_in_the_chart_and_in_an_unexpected_specialty():
    """A clinical impression in a note about diabetes, three-quarters of the way in."""
    paths = chart("SYNX05")
    buried = [p for p in paths if "malignant in my assessment" in p.read_text(encoding="utf-8")]
    assert len(buried) == 1
    note = buried[0]

    assert "Endo-Diab" in note.name, (
        "an oncology note would be the first place anyone looks; the point is that this is not")
    position = paths.index(note) + 1
    assert position / len(paths) > 0.5, (
        f"at {position}/{len(paths)} this is not buried; a chart-order read reaches it early")
    assert answer("SYNX05") == DATE_RE.search(note.name).group(1).replace("-", "")

    path_dates = [DATE_RE.search(p.name).group(1).replace("-", "")
                  for p in paths if "Surgical-Pathology-Report" in p.name]
    assert all(answer("SYNX05") < d for d in path_dates), (
        "the buried impression must PRECEDE the pathology, or the FIRST-date rule does not bite")


def test_x06_no_document_contains_a_full_diagnostic_word():
    """The chart's only statement of the diagnosis is dictation shorthand.

    A run searching the contract's own vocabulary comes back empty. `adeno` hits and
    `adenocarcinoma` does not — which is the shorter-stem move `search-native` advises, so the
    chart tests whether that advice is followed rather than merely rendered.
    """
    paths = chart("SYNX06")
    text = body(paths).lower()
    for word in ("adenocarcinoma", "carcinoma", "malignan", "diagnos", "cancer", "neoplas"):
        assert word not in text, (
            f"{word!r} appears somewhere in SYNX06; the contract's own vocabulary now finds the "
            f"answer and the chart no longer tests widening")

    assert "adenoca" in text, "the shorthand itself must be present, or nothing can be found"
    assert "adeno" in text
