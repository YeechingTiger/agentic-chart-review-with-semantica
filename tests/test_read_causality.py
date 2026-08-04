"""A read should say why it happened, or attribution has nothing but adjacency to guess from.

A trace is a flat sequence of events with no links between them. The attribution agent is told to
"separate what the trace proves from what is inferred", but on a trace with no cause field, "step 9
read D12 because step 7's search returned it" is always an inference — and the report will not say
that it is one. This field turns it into a record.

Not filled in, not refused: a cause is a judgement, a judgement is allowed to be absent, and its
absence has to be countable.
"""
from __future__ import annotations

import pytest

from acr.evaluation.evals import RunRecord, detect_uncaused_reads, run_detectors
from acr.review.tools.toolbox import CAUSE_PARAM, TOOL_SCHEMAS

#: The corpus filename convention is `<DocType>_<YYYY-MM-DD>[__<n>].txt` and the note_id IS the
#: stem — see `acr.chartstore.corpus.FILENAME_RE`. A name that does not parse is skipped rather than
#: guessed at, so getting this wrong builds an EMPTY chart and every read below fails as
#: "unknown note_id" instead of as the thing under test.
NOTE_ID = "pathology_2024-01-01"


@pytest.fixture
def toolbox_with_one_doc(tmp_path):
    from acr.chartstore.corpus import Corpus
    from acr.core.state import EvidenceLedger
    from acr.review.coverage import CoverageLedger, ForcedSampler
    from acr.review.tools.toolbox import Toolbox
    d = tmp_path / "patients" / "SYN01"
    d.mkdir(parents=True)
    (d / f"{NOTE_ID}.txt").write_text("final diagnosis: adenocarcinoma\n", encoding="utf-8")
    chart = Corpus(tmp_path / "patients").chart("SYN01")
    assert len(chart) == 1, "fixture built an empty chart; the filename does not parse"
    docs, _ = chart.list_documents(limit=100)
    return Toolbox(chart, EvidenceLedger(), CoverageLedger(docs, (), ForcedSampler(1)))


@pytest.fixture
def tracer(tmp_path):
    from acr.contract.trace import Tracer
    return Tracer.create(tmp_path, "t1")


def _schema(name: str) -> dict:
    for s in TOOL_SCHEMAS:
        if s["function"]["name"] == name:
            return s["function"]["parameters"]["properties"]
    raise AssertionError(f"no tool {name!r}")


@pytest.mark.parametrize("tool", ["read_document", "read_documents_batch", "search_notes"])
def test_retrieval_tools_ask_for_a_cause(tool: str):
    assert CAUSE_PARAM in _schema(tool), f"{tool} never asks the model why"


@pytest.mark.parametrize("tool", ["read_document", "read_documents_batch", "search_notes"])
def test_cause_is_never_required(tool: str):
    """A record, not a gate. Making it required would turn a judgement into a ritual."""
    for s in TOOL_SCHEMAS:
        if s["function"]["name"] == tool:
            assert CAUSE_PARAM not in s["function"]["parameters"].get("required", [])


def test_dispatch_accepts_and_strips_the_cause(toolbox_with_one_doc):
    """The `_t_` methods need not know about this parameter — dispatch strips it, in one place
    rather than one place per tool."""
    tb = toolbox_with_one_doc
    out, _ms = tb.dispatch("read_document", {"note_id": NOTE_ID, CAUSE_PARAM: "search #7 hit"})
    assert "error" not in out
    assert tb.last_cause == "search #7 hit"


def test_dispatch_clears_the_cause_between_calls(toolbox_with_one_doc):
    """One call's cause must not stick to the next — that invents a causal link nobody wrote."""
    tb = toolbox_with_one_doc
    tb.dispatch("read_document", {"note_id": NOTE_ID, CAUSE_PARAM: "thread T3"})
    tb.dispatch("read_document", {"note_id": NOTE_ID})
    assert tb.last_cause == ""


def test_tracer_promotes_the_cause_to_a_top_level_field(tracer):
    ev = tracer.tool("read_document", {"note_id": NOTE_ID}, {"ok": True}, because="thread T3")
    assert ev["because"] == "thread T3"


def test_detector_counts_reads_with_no_cause():
    run = RunRecord(manifest={"patient_id": "SYN01"}, trace=[
        {"kind": "tool", "tool": "read_document", "because": "search #2 hit"},
        {"kind": "tool", "tool": "read_document", "because": ""},
        {"kind": "tool", "tool": "read_document"},
        {"kind": "tool", "tool": "submit_answer"},          # not a read, not in the denominator
    ])
    findings = detect_uncaused_reads(run)
    assert len(findings) == 1
    ev = findings[0].evidence
    assert (ev["n_reads"], ev["n_uncaused"]) == (3, 2)
    assert findings[0].detector == "uncaused_read"


def test_detector_is_silent_when_every_read_has_a_cause():
    run = RunRecord(manifest={"patient_id": "SYN01"}, trace=[
        {"kind": "tool", "tool": "read_document", "because": "search #2 hit"},
    ])
    assert detect_uncaused_reads(run) == []


def test_detector_is_silent_on_a_run_with_no_reads():
    """Zero reads is `detect_zero_document_read`'s case, not this detector's — two detectors
    reporting one fact reads as two problems."""
    assert detect_uncaused_reads(RunRecord(manifest={}, trace=[])) == []


def test_detector_is_wired_into_run_detectors():
    from acr.evaluation.evals import DetectorConfig
    run = RunRecord(manifest={"patient_id": "SYN01"}, trace=[
        {"kind": "tool", "tool": "read_document"},
    ])
    cfg = DetectorConfig(min_term_chars=3, max_rejection_repeats=2,
                         token_band=(0, 10 ** 9), turn_band=(0, 10 ** 6))
    assert any(f.detector == "uncaused_read" for f in run_detectors(run, config=cfg))
