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


# ======================================================================================
# CHARTS WHERE THE RECORDED ANSWER IS IN DISPUTE
#
# These test the EVALUATION, not the agent. `ground_truth` carries the registry's value —
# wrong on K02 — because that is what a deployment has, so an agent that reads the chart
# correctly scores MISMATCH and something downstream has to be the thing that notices.
# `key_dispute.kind` is the answer key for that downstream thing.
# ======================================================================================

DISPUTED = ["SYNK01", "SYNK02", "SYNK03"]
KINDS = {"OUTSIDE_EVIDENCE", "KEY_ERROR", "CHART_AMBIGUOUS"}


def dispute(pid: str) -> dict:
    d = truth(pid).get("key_dispute")
    assert d, f"{pid} carries no key_dispute block"
    return d


@pytest.mark.parametrize("pid", DISPUTED)
def test_the_scorer_sees_the_registry_value_not_the_chart_answer(pid: str):
    """The simulation only works if `acr eval score` reads what a deployment would read.

    If `ground_truth` carried the chart's answer instead, a correct run would score EXACT and
    there would be nothing for an evaluation to catch — the whole point of these charts is that
    the disagreement reaches the scorer.
    """
    d = dispute(pid)
    assert answer(pid) == d["registry_value"], (
        f"{pid}: the scorer's key must be the REGISTRY value, wrong or right")


@pytest.mark.parametrize("pid", DISPUTED)
def test_each_dispute_declares_a_kind_and_the_harm_of_getting_it_wrong(pid: str):
    d = dispute(pid)
    assert d["kind"] in KINDS, f"{pid}: unknown dispute kind {d['kind']!r}"
    assert d.get("correct_eval_verdict"), f"{pid}: no verdict for the eval to be scored against"
    assert d.get("harm_if_missed"), (
        f"{pid}: a dispute kind with no stated harm is a taxonomy, not a test — the three kinds "
        f"exist because they cause DIFFERENT damage")


def test_the_three_kinds_are_all_represented():
    """One of each. Two of a kind and an evaluation can score well by always guessing it."""
    assert {dispute(p)["kind"] for p in DISPUTED} == KINDS


def test_k01_the_chart_contains_no_establishing_document_at_all():
    """The key is right and unreachable: the report is at another hospital."""
    paths = chart("SYNK01")
    assert not [p for p in paths if "Surgical-Pathology" in p.name], (
        "a pathology report in this chart would make the key derivable and the case pointless")
    # Matched on a phrase that cannot be split by the note's wrapping. "outside facility" is
    # the obvious probe and it fails here for a reason that has nothing to do with the chart:
    # the words land on either side of a line break. A test that pins prose across a newline
    # pins the formatter, not the property.
    referring = [p for p in paths
                 if "NOT available for review" in p.read_text(encoding="utf-8")]
    assert referring, (
        "the chart must REFER to the evidence it lacks — that reference is the whole tell "
        "separating OUTSIDE_EVIDENCE from KEY_ERROR, and without it this chart is simply a "
        "record with a gap")
    assert dispute("SYNK01")["chart_supports"] is None


def test_k02_the_key_names_a_date_no_document_falls_on():
    """The cheapest dispute to detect, and the one that must never be missed.

    Decidable without reading a word of clinical text: list the chart, look for the date.
    """
    paths = chart("SYNK02")
    key = answer("SYNK02")
    key_iso = f"{key[:4]}-{key[4:6]}-{key[6:]}"
    assert not [p for p in paths if key_iso in p.name], (
        f"a document dated {key_iso} would give the key something to stand on")
    assert not [p for p in paths if key_iso in p.read_text(encoding="utf-8")], (
        f"{key_iso} appears in some document's text; the contradiction is no longer structural")

    supported = dispute("SYNK02")["chart_supports"]
    sup_iso = f"{supported[:4]}-{supported[4:6]}-{supported[6:]}"
    assert [p for p in paths if sup_iso in p.name], (
        "the date the chart DOES support must be present, or this is not a typo, it is a gap")


def test_k03_both_readings_are_present_in_the_chart():
    """Neither side erred. Without this control, an evaluation scores full marks by calling
    every disagreement a defect — which is the same failure as calling none of them one."""
    paths = chart("SYNK03")
    cyto = [p for p in paths if "POSITIVE FOR MALIGNANT CELLS" in p.read_text(encoding="utf-8")]
    assert len(cyto) == 1, "the unambiguous cytology is what makes both readings defensible"
    assert not [p for p in cyto if "SUSPICIOUS FOR" in p.read_text(encoding="utf-8")], (
        "'suspicious for' would make this the ordinary ambiguous-cytology case the spec DOES "
        "settle, and the ambiguity would be gone")

    both = dispute("SYNK03")["chart_supports"]
    assert isinstance(both, list) and len(both) == 2
    for candidate in both:
        iso = f"{candidate[:4]}-{candidate[4:6]}-{candidate[6:]}"
        assert [p for p in paths if iso in p.name], (
            f"{iso} is offered as a defensible reading but no document falls on it")


def test_the_key_challenge_skill_exists_and_a_named_mode_offers_it():
    """A card nobody loads is guidance nobody receives — the defect `acr.skills` was written
    to prevent, one level up.

    It is deliberately NOT in the default truth mode. These three charts are the reason the
    doubting posture exists, and the reason it is opt-in: on a chart whose key IS derivable, an
    agent holding this card can book every hard failure as a bad key. What must be true is that
    the card is reachable by naming a truth mode, not that every diagnosis receives it.

    `REGISTRY_REFERENCE` is the mode that licenses it, and that is not a coincidence: these keys
    ARE registry values, and attribution's boundary for that mode says a registry value is "an
    UNRESOLVED reference, not truth" whose disagreement may only be NEEDS_ADJUDICATION — which
    is exactly `key_dispute.correct_eval_verdict` on K03.
    """
    from acr.cli_signal import DEFAULT_TRUTH_MODE, EVAL_MODES
    from acr.skills import eval_skill_judges, skill_slot

    assert "eval-key-challenge" in EVAL_MODES["REGISTRY_REFERENCE"]
    assert "eval-key-challenge" not in EVAL_MODES[DEFAULT_TRUTH_MODE]
    assert skill_slot("eval-key-challenge") == "eval"
    assert "key_derivability" in eval_skill_judges("eval-key-challenge")


def test_no_unparseable_document_is_sitting_in_the_corpus():
    """A .txt whose name the loader cannot parse is invisible to every run.

    `corpus.FILENAME_RE` skips a stem it cannot read, silently — which is correct for iCloud's
    "Name 2.txt" conflict copies (1,929 appeared during one regeneration of this corpus, and
    none of them reached a chart) and dangerous for anything else. A genuinely misnamed clinical
    document would disappear the same way, and a chart missing a document still answers, still
    passes its gate, and still reports coverage over the documents it did see.

    So the skip stays silent and this makes the population visible instead.
    """
    from acr.corpus import FILENAME_RE

    unparseable = [p for p in CORPUS.rglob("*.txt") if not FILENAME_RE.match(p.stem)]
    assert not unparseable, (
        f"{len(unparseable)} document(s) the loader will skip without saying so, e.g. "
        f"{[p.name for p in unparseable[:3]]}. If these are iCloud conflict copies, delete "
        f"them: find corpus/patients -name '* [0-9].txt' -delete")
