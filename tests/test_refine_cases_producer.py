"""The bridge from recorded runs to `refine route`'s `--cases`, which nothing produced.

`acr refine route` REQUIRES `--cases`: a JSON list whose every member carries a pseudonymous
case_id, the disagreement with the answer key, and two facts the router's first two cuts read
before any model is consulted. Nothing in this tree wrote that file. Same shape as the term cache
and the case map: a consumer whose producer does not ship is a stage that cannot run, and it stays
invisible because reaching it costs money.

WHAT IS PINNED HERE is only what the router actually branches on. `establishing_evidence_surfaced`
decides cut 1 — not surfaced routes to RETRIEVAL_FAILURE without consulting the reflector at all —
so getting it wrong silently reclassifies every reading failure as a retrieval failure. And the
abstention case has a non-obvious right answer: when the key says abstaining was correct, NO
document establishes it, and encoding that as `False` would route a wrong answer to retrieval when
retrieval cannot possibly be the cause.
"""

from __future__ import annotations

import json

import pytest

from acr.commands.cli_refine import build_failure_cases
from acr.improvement import refine as R

SPEC = "STORE.390.date_of_initial_diagnosis"
FIELD = "date_of_initial_diagnosis"


def _manifest(tmp_path, patient, value, *, read_notes=(), name=None, trace=True,
              recorded_path=None):
    """A manifest and trace of the shape `run_patient` actually writes.

    `manifest["trace"]` is a PATH TO A JSONL FILE, not an inline list. The first version of this
    fixture inlined the list, every test passed, and the producer died on the first real manifest
    with `'str' object has no attribute 'get'`. A fixture that models a shape the system does not
    write tests nothing.

    `recorded_path` lets a test record a trace path that no longer resolves while the sibling
    `.jsonl` still sits beside the manifest — which is what `tools/archive_runs.sh` produces (49 of
    509 manifests in this tree are in that state).
    """
    d = tmp_path / "runs"
    d.mkdir(exist_ok=True)
    m = d / f"{name or patient}.manifest.json"
    tp = d / f"{name or patient}.jsonl"
    if trace:
        tp.write_text("".join(
            json.dumps({"seq": i, "kind": "tool", "tool": "read_document",
                        "args": {"note_id": n}}) + "\n"
            for i, n in enumerate(read_notes)), encoding="utf-8")
    m.write_text(json.dumps({
        "patient_id": patient,
        "spec_id": SPEC,
        "answer": ({"status": "SUBMITTED", "status_kind": "value", "value": {FIELD: value}}
                   if value else
                   {"status": "CORPUS_INSUFFICIENT", "status_kind": "abstention", "value": {}}),
        "trace": str(recorded_path if recorded_path is not None else tp),
    }), encoding="utf-8")
    return m


class FakeChart:
    """Only what the producer uses: which notes contain a literal."""

    def __init__(self, by_note):
        self._by_note = by_note

    def search(self, term, *a, **kw):
        return [type("H", (), {"note_id": n})()
                for n, text in self._by_note.items() if term in text]


class FakeCorpus:
    def __init__(self, charts):
        self._charts = charts

    def chart(self, pid):
        return self._charts[pid]


def _build(tmp_path, *, manifests, key, charts, case_map, adjudications=None):
    return build_failure_cases(
        manifests=manifests, answer_key=key, fields=[FIELD], spec_id=SPEC,
        corpus=FakeCorpus(charts), case_map=case_map, adjudications=adjudications or {})


def test_only_a_disagreement_becomes_a_case(tmp_path):
    right = _manifest(tmp_path, "P1", "20230412", name="right")
    wrong = _manifest(tmp_path, "P2", "20230501", name="wrong")
    cases = _build(
        tmp_path,
        manifests=[right, wrong],
        key={"P1": {"fields": {FIELD: "20230412"}}, "P2": {"fields": {FIELD: "20230412"}}},
        charts={"P1": FakeChart({}), "P2": FakeChart({})},
        case_map={"CASE-aaa": "P1", "CASE-bbb": "P2"})
    assert [c["case_id"] for c in cases] == ["CASE-bbb"]
    assert cases[0]["coded_value"] == "20230501"
    assert cases[0]["key_value"] == "20230412"


def test_surfaced_is_true_when_a_read_note_carries_the_key_value(tmp_path):
    m = _manifest(tmp_path, "P1", "20230501", read_notes=["N1"])
    cases = _build(
        tmp_path, manifests=[m],
        key={"P1": {"fields": {FIELD: "20230412"}}},
        charts={"P1": FakeChart({"N1": "impression 2023-04-12 adenocarcinoma"})},
        case_map={"CASE-aaa": "P1"})
    assert cases[0]["establishing_evidence_surfaced"] is True


def test_surfaced_is_false_when_the_carrying_note_was_never_read(tmp_path):
    """Cut 1's whole purpose: no sentence anywhere would have helped."""
    m = _manifest(tmp_path, "P1", "20230501", read_notes=["N2"])
    cases = _build(
        tmp_path, manifests=[m],
        key={"P1": {"fields": {FIELD: "20230412"}}},
        charts={"P1": FakeChart({"N1": "impression 2023-04-12", "N2": "nothing relevant"})},
        case_map={"CASE-aaa": "P1"})
    assert cases[0]["establishing_evidence_surfaced"] is False
    routed = R.GradientRouter({SPEC: "text"}, None).route(R.FailureCase(
        **{k: v for k, v in cases[0].items() if k != "key_value_in_corpus"}))
    assert routed.verdict == R.RETRIEVAL_FAILURE


def test_the_key_value_is_matched_in_every_notation_the_corpus_might_use(tmp_path):
    """`20230412` finds nothing in this corpus; the matcher is not date-tolerant across formats.

    Measured, not assumed: `chart.search('20230412')` returns 0 notes on SYN0001 while
    `chart.search('2023-04-12')` returns 2. A producer that searched only the key's own notation
    would report every date failure as a retrieval failure.
    """
    m = _manifest(tmp_path, "P1", "STATUS", read_notes=["N1"])
    cases = _build(
        tmp_path, manifests=[m],
        key={"P1": {"fields": {FIELD: "20230412"}}},
        charts={"P1": FakeChart({"N1": "diagnosed 4/12/2023"})},
        case_map={"CASE-aaa": "P1"})
    assert cases[0]["establishing_evidence_surfaced"] is True


def test_a_correct_abstention_key_counts_as_surfaced(tmp_path):
    """No document establishes an abstention, and `False` would blame retrieval for a misreading."""
    m = _manifest(tmp_path, "P1", "20230412")
    cases = _build(
        tmp_path, manifests=[m],
        key={"P1": {"fields": {FIELD: None}}},
        charts={"P1": FakeChart({})},
        case_map={"CASE-aaa": "P1"})
    assert cases[0]["key_value"] == ""
    assert cases[0]["establishing_evidence_surfaced"] is True


def test_a_correct_abstention_is_not_a_failure_case(tmp_path):
    """The bug that put a wrong number in the docs: a run that ABSTAINED, on a chart where the key
    says abstaining is correct, was emitted as a disagreement because the status string
    `CORPUS_INSUFFICIENT` != the empty key value. `eval score` calls the same run
    ABSTAINED_CORRECT — two scorers, one manifest, opposite verdicts. The published count of 8
    disagreements on the B0-base cohort was 7 for exactly this reason (CASE-976fc2f61c53).
    """
    m = _manifest(tmp_path, "P1", None)      # the run abstained
    cases = _build(
        tmp_path, manifests=[m],
        key={"P1": {"fields": {FIELD: None}}},   # and abstaining was correct
        charts={"P1": FakeChart({})},
        case_map={"CASE-aaa": "P1"})
    assert cases == []


def test_a_wrong_abstention_is_a_failure_with_an_empty_coded_value(tmp_path):
    """The other direction must survive: abstained where the key has a value IS a failure, and its
    coded_value is the empty string — the same convention the key uses — not a status string that
    can never equal anything."""
    m = _manifest(tmp_path, "P1", None, read_notes=["N1"])
    cases = _build(
        tmp_path, manifests=[m],
        key={"P1": {"fields": {FIELD: "20230412"}}},
        charts={"P1": FakeChart({"N1": "impression 2023-04-12"})},
        case_map={"CASE-aaa": "P1"})
    assert len(cases) == 1
    assert cases[0]["coded_value"] == ""
    assert cases[0]["establishing_evidence_surfaced"] is True


def test_a_99_partial_key_date_searches_for_nothing(tmp_path):
    """`_notations("20159999")` returned `["2015"]` — and a bare year is in EVERY note, so
    `key_value_in_corpus` came back True and cut 1 routed a constructed date to §6c retrieval,
    the exact inversion the docstring claims to avoid. Measured: SYNY02's year matched 27 notes.
    `tools/measure_controller_value.py` returns False for the same fact ("a constructed partial
    date is in no document"); this pins the producer to that semantics."""
    from acr.commands.cli_refine import _notations
    assert _notations("20159999") == []
    assert _notations("20191099") == []

    m = _manifest(tmp_path, "P1", "20150614", read_notes=["N1"])
    cases = _build(
        tmp_path, manifests=[m],
        key={"P1": {"fields": {FIELD: "20159999"}}},
        charts={"P1": FakeChart({"N1": "seen in 2015, imaging 2015-06-14"})},
        case_map={"CASE-aaa": "P1"})
    assert cases[0]["key_value_in_corpus"] is False
    assert cases[0]["establishing_evidence_surfaced"] is True   # unseedable, not retrieval


def test_a_moved_run_tree_still_reads_the_sibling_trace(tmp_path):
    """49 of 509 manifests in this tree record an absolute trace path that no longer resolves,
    every one with the sibling `.jsonl` present — `tools/archive_runs.sh` moves directories.
    `RunRecord.from_manifest` reads the sibling; the first version of this producer re-implemented
    the lookup from the recorded path and refused with a false statement."""
    m = _manifest(tmp_path, "P1", "20230501", read_notes=["N1"],
                  recorded_path="/nonexistent/moved/away.jsonl")
    cases = _build(
        tmp_path, manifests=[m],
        key={"P1": {"fields": {FIELD: "20230412"}}},
        charts={"P1": FakeChart({"N1": "impression 2023-04-12"})},
        case_map={"CASE-aaa": "P1"})
    assert cases[0]["establishing_evidence_surfaced"] is True


def test_adjudication_defaults_to_not_adjudicated_and_can_be_supplied(tmp_path):
    m = _manifest(tmp_path, "P1", "20230501")
    common = dict(
        manifests=[m], key={"P1": {"fields": {FIELD: "20230412"}}},
        charts={"P1": FakeChart({})}, case_map={"CASE-aaa": "P1"})
    assert _build(tmp_path, **common)[0]["answer_key_adjudication"] == R.NOT_ADJUDICATED
    supplied = _build(tmp_path, **common,
                      adjudications={"CASE-aaa": R.ADJUDICATED_KEY_WRONG})
    assert supplied[0]["answer_key_adjudication"] == R.ADJUDICATED_KEY_WRONG


def test_a_patient_absent_from_the_case_map_refuses(tmp_path):
    """The only way a real person_id could reach a develop artifact through this door."""
    m = _manifest(tmp_path, "P9", "20230501")
    with pytest.raises(R.RefineError, match="case map"):
        _build(tmp_path, manifests=[m],
               key={"P9": {"fields": {FIELD: "20230412"}}},
               charts={"P9": FakeChart({})}, case_map={})


def test_the_output_is_what_the_router_accepts(tmp_path):
    """The format coupling that broke the experience chain: two halves, two shapes."""
    m = _manifest(tmp_path, "P1", "20230501")
    cases = _build(tmp_path, manifests=[m],
                   key={"P1": {"fields": {FIELD: "20230412"}}},
                   charts={"P1": FakeChart({})}, case_map={"CASE-aaa": "P1"})
    for row in cases:
        # `key_value_in_corpus` is a REPORTING key, not a router input: the router branches on the
        # boolean, and adding a field to `FailureCase` would be a schema change for a diagnostic.
        R.FailureCase(**{k: v for k, v in row.items() if k != "key_value_in_corpus"})


def test_a_missing_trace_refuses_rather_than_reporting_nothing_was_read(tmp_path):
    """The failure that would be invisible: no trace means every case reads as a retrieval failure.

    `establishing_evidence_surfaced=False` sends a case to RETRIEVAL_FAILURE without consulting
    anything. An unreadable trace produces exactly that value while meaning "we do not know", and
    a whole cohort would be attributed to search when search was never the problem.
    """
    m = _manifest(tmp_path, "P1", "20230501", read_notes=["N1"], trace=False,
                  recorded_path="/nonexistent/nowhere.jsonl")
    with pytest.raises(R.RefineError, match="trace"):
        _build(tmp_path, manifests=[m],
               key={"P1": {"fields": {FIELD: "20230412"}}},
               charts={"P1": FakeChart({"N1": "impression 2023-04-12"})},
               case_map={"CASE-aaa": "P1"})


def test_a_key_value_no_document_carries_is_not_a_retrieval_failure(tmp_path):
    """The bug this test was written from, found on real runs and not by reasoning.

    SYNK01's key is `20210315` and SYNK02's is `20200714`, and NEITHER string appears in any
    notation in any document of those charts: the key value is CONSTRUCTED — imputed from a
    seasonal phrase or inferred across notes — which `tools/measure_controller_value.py` counts as
    its own class, `UNSEEDABLE`, precisely so it does not land in `NEVER_LOOKED`.

    The first version of this producer reported `surfaced: False` for both, which routes them to
    RETRIEVAL_FAILURE. Retrieval cannot surface a document that does not exist, and §6c would have
    been handed two cases no search could ever fix — while the two GENUINE retrieval failures in
    the same batch (SYNX02, SYNX06, where exactly one note carries the value and it was never
    opened) sat in the same bucket, indistinguishable.
    """
    m = _manifest(tmp_path, "P1", "20200614", read_notes=["N1"])
    cases = _build(
        tmp_path, manifests=[m],
        key={"P1": {"fields": {FIELD: "20200714"}}},
        charts={"P1": FakeChart({"N1": "no date in any notation here"})},
        case_map={"CASE-aaa": "P1"})
    assert cases[0]["establishing_evidence_surfaced"] is True
    assert cases[0]["key_value_in_corpus"] is False


def test_a_carried_but_unread_value_is_still_a_retrieval_failure(tmp_path):
    """The other side of the same cut: SYNX02's shape must not be swallowed by the fix above."""
    m = _manifest(tmp_path, "P1", "20200620", read_notes=["N2"])
    cases = _build(
        tmp_path, manifests=[m],
        key={"P1": {"fields": {FIELD: "20200510"}}},
        charts={"P1": FakeChart({"N1": "procedure 2020-05-10", "N2": "ct 2020-06-14"})},
        case_map={"CASE-aaa": "P1"})
    assert cases[0]["establishing_evidence_surfaced"] is False
    assert cases[0]["key_value_in_corpus"] is True


def test_a_patient_the_key_says_nothing_about_is_not_a_failure(tmp_path):
    """`_key_value` collapsed FOUR facts into `""`: a missing patient, a row that is not a mapping,
    a field absent from the row, and a genuine `None` meaning "abstaining is correct". So every run
    on a patient outside the key was emitted as a failure case and printed under "where abstaining
    was correct and the run answered anyway" — a claim the key never made.

    This is the same defect as the `n_unkeyed`-pinned-at-0 bug in `evals.score`, reintroduced in a
    producer that had no `n_unkeyed` concept at all. Measured: a STORE.390 key against five real
    STORE.400 manifests produced 6 such phantom cases.
    """
    m = _manifest(tmp_path, "P9", "20230501")
    cases = _build(tmp_path, manifests=[m],
                   key={"P1": {"fields": {FIELD: "20230412"}}},   # says nothing about P9
                   charts={"P9": FakeChart({})}, case_map={"CASE-zzz": "P9"})
    assert cases == []


def test_a_key_row_for_another_spec_is_not_a_failure(tmp_path):
    """`evals._key_row` refuses on a row whose `spec_id` differs; this producer ignored it entirely,
    a fresh divergence between the two scorers introduced in the same changeset."""
    m = _manifest(tmp_path, "P1", "20230501")
    cases = _build(tmp_path, manifests=[m],
                   key={"P1": {"fields": {FIELD: "20230412"}, "spec_id": "SOME.OTHER.spec"}},
                   charts={"P1": FakeChart({})}, case_map={"CASE-aaa": "P1"})
    assert cases == []


def test_a_malformed_key_row_is_not_a_correct_abstention(tmp_path):
    m = _manifest(tmp_path, "P1", "20230501")
    for bad in ([1, 2], "nope", {"fields": "not-a-mapping"}):
        cases = _build(tmp_path, manifests=[m], key={"P1": bad},
                       charts={"P1": FakeChart({})}, case_map={"CASE-aaa": "P1"})
        assert cases == [], bad


def test_a_short_non_date_key_value_searches_for_nothing(tmp_path):
    """`_notations` fell through to `return [value]` for any non-8-digit value, and
    `STORE.400_522_523`'s `behavior` field has allowable values `0`/`1`/`2`/`3`. Measured on SYN0001
    (321 documents): `'0'` matches 321 notes, `'2'` matches 321, `'3'` matches 172. So
    `key_value_in_corpus` was a meaningless True, `surfaced` was True for any run that opened any
    note, and cut 1 could never route a `behavior` failure to RETRIEVAL_FAILURE — the same routing
    inversion the 99-date guard exists to prevent, one field over.
    """
    from acr.commands.cli_refine import _notations
    assert _notations("0") == []
    assert _notations("3") == []
    assert _notations("C34.9") != [], "a real code must still be searched for"

    m = _manifest(tmp_path, "P1", "1")
    cases = _build(tmp_path, manifests=[m], key={"P1": {"fields": {FIELD: "3"}}},
                   charts={"P1": FakeChart({"N1": "grade 3 of 3, behavior 3"})},
                   case_map={"CASE-aaa": "P1"})
    assert cases[0]["key_value_in_corpus"] is False
    assert cases[0]["establishing_evidence_surfaced"] is True
