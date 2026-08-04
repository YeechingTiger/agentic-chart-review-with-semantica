"""Which specimen a quote is about has to be writable down.

"Right document, wrong passage" — a real quote from the source that is nevertheless about another
specimen — is the failure mode eval-overconfidence names, and a flat list of spans cannot express it
structurally: there is nowhere to write "this row is about A and that one is about B".

There used to be an `entity_answer_mismatch` here as well, comparing each anchor against
`reported_lesion` by exact equality. Measured on twelve real runs on 2026-07-31: CRITICAL on all
twelve, and wrong all twelve times. The two sides are not the same kind of string at all — an anchor
is a short label, `reported_lesion` is a whole sentence of explanation the model wrote — so equality
can only fail, which means that check could not pass on a correct run either. A CRITICAL that is
always true is worse than no check: it teaches a reader to skip `patient_crossover` along with it,
since they share a severity class.

It became `multiple_anchored_entities`: count how many distinct anchor labels there are, and do not
judge whether they agree. That is a question the data can answer; "are these two phrasings the same
lesion" is clinical judgement, and writing it into Python is the mistake
DETERMINISTIC_RULES_REMOVED.md records.

The field is optional: no anchor is not a defect, it is a facility left unused. The check stays
silent when there is no anchor.
"""
from __future__ import annotations

from acr.core.state import Evidence, EvidenceLedger
from acr.evaluation.evals import DetectorConfig, RunRecord, run_detectors


def test_evidence_carries_an_optional_entity():
    e = Evidence("N1", "pathology", "2024-01-01", 0, 10, "x", "primary_site",
                 entity="specimen A")
    assert e.to_dict()["entity"] == "specimen A"


def test_entity_defaults_to_empty_so_old_records_still_load():
    e = Evidence("N1", "pathology", "2024-01-01", 0, 10, "x", "primary_site")
    assert e.to_dict()["entity"] == ""


def test_same_span_different_entity_is_not_a_duplicate():
    """The de-duplication key must carry the entity, or one of two quotes at the same offsets about
    two different specimens gets swallowed.
    """
    led = EvidenceLedger()
    led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology", entity="specimen A"))
    led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology", entity="specimen B"))
    assert len(led.items) == 2


def test_identical_entity_still_de_duplicates():
    led = EvidenceLedger()
    for _ in range(2):
        led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology", entity="specimen A"))
    assert len(led.items) == 1


def test_the_rendered_ledger_shows_the_anchor():
    """The ledger rendered back to the model has to carry the entity, or the model cannot see the
    distinction it just recorded.
    """
    led = EvidenceLedger()
    led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology", entity="specimen A"))
    assert "specimen A" in led.render()


def test_the_rendered_ledger_says_nothing_when_no_anchor_was_recorded():
    led = EvidenceLedger()
    led.add(Evidence("N1", "p", "2024-01-01", 0, 10, "x", "histology"))
    assert "entity" not in led.render()


def _run(evidence, reported_lesion="") -> RunRecord:
    return RunRecord(manifest={"patient_id": "SYN01", "evidence": evidence,
                               "answer": {"reported_lesion": reported_lesion}}, trace=[])


def test_no_detector_reads_the_anchor():
    """Regression: no detector may hang on `entity` again unless the tool contract first requires a
    stable label.

    Both were written, both were measured on the same batch of twelve runs, both were deleted.
    `entity_answer_mismatch` compared each anchor against `reported_lesion` (a whole sentence of
    prose) by exact equality: CRITICAL on 12 of 12, wrong all 12 times.
    `multiple_anchored_entities` counted distinct labels instead: fired on 12 of 12 and was right
    about 1 — the other four were one lesion under a name that had moved ("mass" → "carcinoma"),
    which is exactly how a chart is written.

    It measures phrasing drift, not entity count. Deciding that "sigmoid colon mass" and "sigmoid
    colon carcinoma" name one thing takes clinical judgement, and this tree has already paid once
    for writing clinical judgement into Python.
    """
    ev = [{"note_id": "N1", "start": 0, "end": 9, "entity": "left upper lobe"},
          {"note_id": "N2", "start": 0, "end": 9, "entity": "right lower lobe"}]
    cfg = DetectorConfig(min_term_chars=3, max_rejection_repeats=2,
                         token_band=(0, 10 ** 9), turn_band=(0, 10 ** 6))
    names = {f.detector for f in run_detectors(_run(ev), config=cfg)}
    assert "entity_answer_mismatch" not in names
    assert "multiple_anchored_entities" not in names

    import acr.evaluation.evals as E
    assert not hasattr(E, "detect_entity_answer_mismatch")
    assert not hasattr(E, "detect_multiple_anchored_entities")


def test_the_anchor_is_offered_and_never_required():
    from acr.review.tools.toolbox import TOOL_SCHEMAS
    schema = next(s for s in TOOL_SCHEMAS
                  if s["function"]["name"] == "record_evidence")["function"]["parameters"]
    assert "entity" in schema["properties"]
    assert "entity" not in schema["required"]


#: The corpus filename convention is `<DocType>_<YYYY-MM-DD>[__<n>].txt` and the note_id IS the
#: stem — see `acr.chartstore.corpus.FILENAME_RE`, and the same note in tests/test_read_causality.py.
_NOTE_ID = "pathology_2024-01-01"


def _toolbox(tmp_path):
    from acr.chartstore.corpus import Corpus
    from acr.review.coverage import CoverageLedger, ForcedSampler
    from acr.review.tools.toolbox import Toolbox
    d = tmp_path / "patients" / "SYN01"
    d.mkdir(parents=True)
    (d / f"{_NOTE_ID}.txt").write_text("final diagnosis: adenocarcinoma\n", encoding="utf-8")
    chart = Corpus(tmp_path / "patients").chart("SYN01")
    assert len(chart) == 1, "fixture built an empty chart; the filename does not parse"
    docs, _ = chart.list_documents(limit=100)
    return Toolbox(chart, EvidenceLedger(), CoverageLedger(docs, (), ForcedSampler(1)))


def test_the_tool_carries_the_anchor_through_to_the_ledger(tmp_path):
    """Having the parameter in the schema is not enough — `_t_record_evidence` has to actually pass
    it through into `Evidence`.
    """
    tb = _toolbox(tmp_path)
    out, _ = tb.dispatch("record_evidence",
                         {"note_id": _NOTE_ID, "start": 0, "end": 16,
                          "supports": "histology", "entity": "specimen A"})
    assert out.get("recorded") is True, out
    assert tb.evidence.items[0].entity == "specimen A"


def test_the_tool_still_works_without_an_anchor(tmp_path):
    tb = _toolbox(tmp_path)
    out, _ = tb.dispatch("record_evidence",
                         {"note_id": _NOTE_ID, "start": 0, "end": 16, "supports": "histology"})
    assert out.get("recorded") is True, out
    assert tb.evidence.items[0].entity == ""
