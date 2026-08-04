"""The evidence set can be defective AS A SET too.

Every existing check is per-item: does this quote re-read at its offsets, is this span non-empty.
DeepEvidence's evidence-graph audit reports set-level numbers instead — a 0.6% duplication rate,
≥99% relation correctness — and nothing here had ever counted even "the same text was recorded
twice".

There were three checks here. Two were deleted on 2026-07-31 after measuring them on twelve real
runs: `orphan_contradiction` and `single_witness_field` both grouped by `supports`, and `supports`
is not a field name — `record_evidence` declares it as "which field **or assertion** this backs",
and on real runs the model wrote a whole sentence every time. Grouped on prose, every group holds
exactly one row, so single_witness necessarily fired on 12 of 12 and orphan on 8 of 12, every one
of them a false positive by construction. A check that cannot come back clean measures nothing.

Deleted rather than softened: there is no machine-readable link between an evidence row and a spec
field, so neither question is computable at all from the manifest format as it stands, and guessing
at one is exactly what DETERMINISTIC_RULES_REMOVED.md already paid for.

The check that stayed groups by note_id and does not touch supports — two char ranges that overlap
inside one document are the same text recorded twice, whatever prose each row carries.
"""
from __future__ import annotations

from acr.evaluation.evals import RunRecord, audit_evidence_set


def _run(evidence: list[dict]) -> RunRecord:
    return RunRecord(manifest={"patient_id": "SYN01", "evidence": evidence}, trace=[])


def _ev(note="N1", start=0, end=10, supports="primary_site", stance="supports") -> dict:
    return {"note_id": note, "start": start, "end": end, "supports": supports,
            "stance": stance, "quote": "x" * (end - start)}


def test_overlapping_spans_for_one_field_are_reported():
    """The ledger de-duplicates identical spans by itself; overlapping ones it does not, and an
    overlap is the same sentence recorded twice.
    """
    f = audit_evidence_set(_run([_ev(start=0, end=40), _ev(start=10, end=50)]))
    hit = [x for x in f if x.detector == "evidence_span_overlap"]
    assert len(hit) == 1
    assert hit[0].evidence["n_overlapping_pairs"] == 1
    assert hit[0].evidence["overlap_rate"] == 0.5      # 1 of the 2 rows is redundant


def test_non_overlapping_spans_are_clean():
    f = audit_evidence_set(_run([_ev(start=0, end=10), _ev(start=20, end=30)]))
    assert not [x for x in f if x.detector == "evidence_span_overlap"]


def test_overlap_is_per_document_and_ignores_supports():
    """Grouped by note_id, not by supports.

    The old version grouped by supports, so it **under-fired** whenever the same text was described
    in two different sentences — and on real runs supports differs on every row, so in practice it
    never fired at all. An overlap is a fact about char ranges inside one document, independent of
    what that row wrote.
    """
    f = audit_evidence_set(_run([_ev(start=0, end=40, supports="one sentence"),
                                 _ev(start=10, end=50,
                                     supports="a completely different sentence")]))
    hit = [x for x in f if x.detector == "evidence_span_overlap"]
    assert len(hit) == 1 and hit[0].evidence["n_overlapping_pairs"] == 1


def test_spans_in_different_documents_never_overlap():
    f = audit_evidence_set(_run([_ev(note="N1", start=0, end=40),
                                 _ev(note="N2", start=10, end=50)]))
    assert not [x for x in f if x.detector == "evidence_span_overlap"]


def test_the_two_prose_grouped_detectors_are_gone():
    """Regression: the two detectors that grouped by supports must not come back.

    On twelve real runs they fired 12 times and 8 times respectively, every one a false positive,
    because supports is free text. Reintroducing any check that groups by supports reproduces the
    same failure.
    """
    import acr.evaluation.evals as E
    assert not hasattr(E, "detect_orphan_contradiction")
    ev = [_ev(supports="a one-of-a-kind sentence"), _ev(note="N2", supports="another sentence")]
    names = {f.detector for f in audit_evidence_set(_run(ev))}
    assert "orphan_contradiction" not in names
    assert "single_witness_field" not in names


def test_a_realistic_prose_supports_set_is_clean():
    """The real shape: every supports differs and each row sits in its own document — nothing
    should fire.
    """
    ev = [{"note_id": "Path-2023-04-27", "start": 100, "end": 180,
           "supports": "2023-04-27 biopsy definitively establishes adenocarcinoma",
           "stance": "supports"},
          {"note_id": "Cyto-2023-04-12", "start": 193, "end": 416,
           "supports": "2023-04-12 cytology was suspicious but recommended tissue",
           "stance": "supports"}]
    assert audit_evidence_set(_run(ev)) == []


def test_no_evidence_is_silent_here():
    """An empty ledger is the submission gate's case. This audit only describes an evidence set
    that already exists.
    """
    assert audit_evidence_set(_run([])) == []


def test_the_answer_copy_of_the_ledger_is_audited_too():
    """The manifest writes evidence in two places: once at the top level and once under
    `answer.evidence` (agent.py:987 and :1205).

    A manifest whose top level is empty and whose `answer` carries the rows is a shape that really
    occurs — `run_manifest.build_manifest` writes only the latter — and an audit reading only the
    top level would report "no evidence" for those runs. "No evidence" is the submission gate's
    conclusion, not one this audit is in a position to reach.
    """
    ev = [_ev(start=0, end=40), _ev(start=10, end=50)]
    run = RunRecord(manifest={"patient_id": "SYN01", "answer": {"evidence": ev}}, trace=[])
    assert [x.detector for x in audit_evidence_set(run) if
            x.detector == "evidence_span_overlap"] == ["evidence_span_overlap"]


def test_audit_is_wired_into_run_detectors():
    from acr.evaluation.evals import DetectorConfig, run_detectors
    cfg = DetectorConfig(min_term_chars=3, max_rejection_repeats=2,
                         token_band=(0, 10 ** 9), turn_band=(0, 10 ** 6))
    run = _run([_ev(start=0, end=40), _ev(start=10, end=50)])
    assert any(f.detector == "evidence_span_overlap" for f in run_detectors(run, config=cfg))
