"""The proof obligation has to be reachable by doing the work it asks for.

A gate nobody can pass is not a strict gate, it is a broken one, and it fails in a way that
looks like diligence: the agent submits, gets refused, tries again, gets refused identically,
and the trace fills with activity that cannot converge. That happened here — three rejections
in one run, each demanding the same fifty documents, because `record_sample_verdict` existed
and was never called from anything but tests, and because each check redrew a fresh sample
instead of reusing the outstanding one.

Two properties keep it satisfiable, and both are asserted below: the draw is stable, and work
against it counts.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from acr.chartstore.corpus import Corpus
from acr.contract.spec import load_spec
from acr.review.coverage import CoverageLedger, ForcedSampler, strata_from_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "assets" / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"


@pytest.fixture
def ledger():
    docs, _ = Corpus(ROOT / "corpus" / "patients").chart("SYN0002").list_documents(limit=100_000)
    return CoverageLedger(docs, strata_from_spec(load_spec(SPEC)), ForcedSampler(1234))


def test_the_draw_is_stable_across_checks(ledger):
    """Redrawing each time makes the debt permanent: the agent reads 25, is asked for a
    different 25, and never gets closer."""
    first = {k: [d.note_id for d in v] for k, v in ledger.pending_samples().items()}
    second = {k: [d.note_id for d in v] for k, v in ledger.pending_samples().items()}
    assert first == second and first, "the outstanding sample must be the same sample"


def test_reading_the_drawn_documents_clears_the_debt(ledger):
    pending = ledger.pending_samples()
    assert pending, "precondition: something is owed"
    for stratum, docs in pending.items():
        for d in docs:
            ledger.note_read(d.note_id, d.doc_type)

    resolved = ledger.resolve_sample_verdicts(cited=set())
    assert resolved == sum(len(v) for v in pending.values())
    assert ledger.pending_samples() == {}, "reading everything drawn must settle the obligation"


def test_partial_progress_reduces_what_is_owed(ledger):
    """The debt has to shrink monotonically, or an agent working steadily still never finishes."""
    before = sum(len(v) for v in ledger.pending_samples().values())
    some = ledger.pending_samples()["cannot_establish"][:10]
    for d in some:
        ledger.note_read(d.note_id, d.doc_type)
    ledger.resolve_sample_verdicts(cited=set())
    after = sum(len(v) for v in ledger.pending_samples().values())
    assert after == before - 10


def test_a_relevant_drawn_document_is_recorded_as_a_hit(ledger):
    """Reading a drawn document and citing it means the exclusion was wrong, and the ledger
    has to notice — otherwise sampling can only ever confirm what was assumed."""
    d = ledger.pending_samples()["cannot_establish"][0]
    ledger.note_read(d.note_id, d.doc_type)
    ledger.resolve_sample_verdicts(cited={d.note_id})
    r = next(x for x in ledger.stratum_results() if x.name == "cannot_establish")
    assert r.sample_hits == 1
    assert r.elusion_upper > 0.0


def test_the_obligation_fits_in_a_realistic_step_budget():
    """50 documents at one read per step cannot fit in a 20-step budget. The obligation is a
    statistical result and should not bend to tool granularity; the batch read is what makes
    it affordable."""
    from acr.review.tools import TOOL_SCHEMAS
    names = {t["function"]["name"] for t in TOOL_SCHEMAS}
    assert "read_documents_batch" in names, (
        "without a batched read the sampling obligation costs one step per document and "
        "cannot be met inside any sane max_steps"
    )
    batch = next(t for t in TOOL_SCHEMAS if t["function"]["name"] == "read_documents_batch")
    assert batch["function"]["parameters"]["properties"]["note_ids"]["type"] == "array"


def test_gate_validated_is_a_declared_state_channel():
    """An undeclared channel is dropped by LangGraph with no error, so a node can set it, the
    downstream read returns the falsy default, and an accepted answer comes out labelled
    ungated. That happened; this pins it."""
    from acr.core.state import RunState
    assert "gate_validated" in RunState.__annotations__
