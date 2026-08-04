"""Folding a corpus scan into a retrieval prior, and the four ways that arithmetic goes wrong.

The scan (`acr label scan`) reads every note of N subjects against one requirement and writes a
`NoteLabel` per note: `doc_type`, an `Admissibility` verdict PER FIELD, and the terms the reading
model proposed and that were verified present in the note. The prior is that, folded per variable:

    which document types carry the answer, and at what rate
    which terms surface an answer-bearing document, and what they drag in with them

The traps, each of which this file pins:

1. A RATE NEEDS ITS DENOMINATOR. One note of a type that established the answer is not a better
   prior than forty notes of a type that established it thirty times.
2. `merely_mentions` IS NOT `can_establish`. Ranking on "bears on the question" rewards retrieving
   the places that TALK about a thing over the places that SETTLE it — the failure the whole
   admissibility vocabulary was introduced to name.
3. A TERM NEEDS BOTH NUMBERS. Recall alone recommends the terms `derive.price_terms` refuses: a term
   matching every note has perfect recall and no value.
4. THE READER'S PROPOSALS ARE A LOWER BOUND. A scan capped at eight terms per note cannot propose a
   ninth, so a term absent from a note's list is not evidence it is absent from the note. That basis
   must travel with the count, and only `corpus_matched` may be used to say one term beats another.
"""

from __future__ import annotations

import json

import pytest

from acr.contract.retrieval_prior import (
    RetrievalPrior,
    RetrievalPriorError,
    prior_digest,
    to_experience_asset,
)
from acr.improvement.prior import build_prior

FIELD = "date_of_initial_diagnosis"
SPEC = "STORE.390.date_of_initial_diagnosis"


def _label(patient, note, doc_type, verdict, terms=(), field=FIELD):
    """One row of the shape `acr label scan` writes to `labels.jsonl`."""
    return {"patient_id": patient, "note_id": note, "doc_type": doc_type, "spec_id": SPEC,
            "admissibility": {"verdicts": {field: verdict}, "quote": "x", "quote_verified": True},
            "retrieval_terms": [{"term": t, "reason": "names_the_section"} for t in terms],
            "model": "test-model", "prompt_hash": "abc123"}


def _write(tmp_path, rows):
    p = tmp_path / "labels.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return p


# ------------------------------------------------------------------ 1. the denominator

def test_a_doc_type_rate_carries_its_denominator(tmp_path):
    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma"]),
            _label("P1", "n2", "Path", "neither"),
            _label("P2", "n3", "Path", "can_establish", ["carcinoma"]),
            _label("P2", "n4", "Imaging", "can_establish", ["mass"])]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2,
                        asset_id="test.prior")
    fp = prior.field_prior(FIELD)
    path = next(d for d in fp.doc_types if d.doc_type == "Path")
    imaging = next(d for d in fp.doc_types if d.doc_type == "Imaging")

    assert (path.n_scanned, path.n_can_establish, path.rate) == (3, 2, 0.6667)
    # Imaging is 1/1 = 100%, which OUT-RATES Path. The rate alone would rank it first; the
    # denominator is what tells a reader it rests on one note.
    assert (imaging.n_scanned, imaging.rate) == (1, 1.0)


def test_a_type_never_scanned_is_absent_not_zero(tmp_path):
    """A type with no scanned note must not appear at rate 0.0 — that reads as measured-and-useless
    when the truth is that nobody looked at one."""
    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma"]),
            _label("P2", "n2", "Path", "can_establish", ["carcinoma"])]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2,
                        asset_id="t")
    assert [d.doc_type for d in prior.field_prior(FIELD).doc_types] == ["Path"]


# ------------------------------------------------------------------ 2. merely_mentions

def test_merely_mentions_does_not_count_as_establishing(tmp_path):
    rows = [_label("P1", "n1", "Progress", "merely_mentions", ["cancer"]),
            _label("P1", "n2", "Progress", "merely_mentions", ["cancer"]),
            _label("P2", "n3", "Path", "can_establish", ["carcinoma"]),
            _label("P2", "n4", "Progress", "neither")]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")
    fp = prior.field_prior(FIELD)
    progress = next(d for d in fp.doc_types if d.doc_type == "Progress")

    assert progress.n_can_establish == 0, "a mention is not a witness"
    assert progress.n_merely_mentions == 2, "and it is still recorded — it is not nothing"
    assert fp.n_answer_bearing == 1


def test_a_term_only_on_mentioning_notes_scores_zero_recall(tmp_path):
    """`cancer` appears only where the answer is discussed, never where it is settled. Counting it
    as a win is how a prior comes to recommend the chart's chattiest documents."""
    rows = [_label("P1", "n1", "Progress", "merely_mentions", ["cancer"]),
            _label("P2", "n2", "Path", "can_establish", ["carcinoma"])]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")
    fp = prior.field_prior(FIELD)
    cancer = next(t for t in fp.terms if t.term == "cancer")
    carcinoma = next(t for t in fp.terms if t.term == "carcinoma")

    assert cancer.n_surfaced_answer_bearing == 0
    assert cancer.n_surfaced_other == 1
    assert carcinoma.recall(fp.n_answer_bearing) == 1.0


# ------------------------------------------------------------------ 3. both numbers per term

def test_a_term_that_surfaces_everything_records_its_cost(tmp_path):
    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma", "patient"]),
            _label("P1", "n2", "Imaging", "neither", ["patient"]),
            _label("P2", "n3", "Progress", "neither", ["patient"]),
            _label("P2", "n4", "Path", "can_establish", ["carcinoma", "patient"])]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")
    fp = prior.field_prior(FIELD)
    everything = next(t for t in fp.terms if t.term == "patient")
    precise = next(t for t in fp.terms if t.term == "carcinoma")

    # Identical recall. The cost column is the only thing that separates them.
    assert everything.recall(fp.n_answer_bearing) == precise.recall(fp.n_answer_bearing) == 1.0
    assert everything.n_surfaced_other == 2
    assert precise.n_surfaced_other == 0


def test_the_rendered_prompt_ranks_by_yield_then_cost(tmp_path):
    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma", "patient"]),
            _label("P1", "n2", "Imaging", "neither", ["patient"]),
            _label("P2", "n3", "Path", "can_establish", ["carcinoma", "patient"])]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")
    asset = to_experience_asset(prior)
    terms = asset["queries"][0]["terms"]
    assert terms.index("carcinoma") < terms.index("patient")


# ------------------------------------------------------------------ 4. the basis

def test_a_reader_proposed_count_is_labelled_as_a_lower_bound(tmp_path):
    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma"]),
            _label("P2", "n2", "Path", "can_establish", ["carcinoma"])]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")
    assert all(t.basis == "proposed_by_reader" for t in prior.field_prior(FIELD).terms)


def test_the_corpus_matcher_upgrades_the_basis_and_the_counts(tmp_path):
    """With the corpus in hand, a term is counted where it OCCURS rather than where it was
    proposed. `carcinoma` is proposed on n1 only, and present in both."""
    class FakeChart:
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

    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma"]),
            _label("P2", "n2", "Path", "can_establish", [])]
    corpus = FakeCorpus({"P1": FakeChart({"n1": "invasive carcinoma"}),
                         "P2": FakeChart({"n2": "invasive carcinoma"})})
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t",
                        corpus=corpus)
    t = next(x for x in prior.field_prior(FIELD).terms if x.term == "carcinoma")
    assert t.basis == "corpus_matched"
    assert t.n_surfaced_answer_bearing == 2, "present in both, proposed on one"


# ------------------------------------------------------------------ refusals

def test_too_few_subjects_refuses(tmp_path):
    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma"])]
    with pytest.raises(RetrievalPriorError, match="patient"):
        build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")


def test_min_patients_has_no_default():
    """A floor nobody typed is folklore — the same rule `DetectorConfig` states for its bands."""
    import inspect
    sig = inspect.signature(build_prior)
    assert sig.parameters["min_patients"].default is inspect.Parameter.empty


def test_a_field_nothing_established_is_reported_empty_not_dropped(tmp_path):
    """"The record is silent about this variable" and "nobody scanned it" must not look alike."""
    rows = [_label("P1", "n1", "Path", "neither"), _label("P2", "n2", "Path", "neither")]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")
    fp = prior.field_prior(FIELD)
    assert fp is not None and fp.is_empty and fp.n_notes == 2


def test_a_prior_is_never_born_certified(tmp_path):
    """`assets certify` grants that, on held-out subjects, by writing a certificate. A builder that
    could stamp `certified` would let an uncertified prior claim it."""
    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma"]),
            _label("P2", "n2", "Path", "can_establish", ["carcinoma"])]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")
    assert prior.status == "measured"


# ------------------------------------------------------------------ held-out discipline

def test_the_prior_knows_which_subjects_it_saw(tmp_path):
    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma"]),
            _label("P2", "n2", "Path", "can_establish", ["carcinoma"])]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")
    assert prior.informed_by("P1") and prior.informed_by("P2")
    assert not prior.informed_by("P9")


def test_the_asset_carries_no_subject_id(tmp_path):
    """It must be publishable. `tests/test_no_phi_in_tree.py` is the standing rule; this is the
    same rule stated where the asset is built."""
    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma"]),
            _label("P2", "n2", "Path", "can_establish", ["carcinoma"])]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")
    blob = json.dumps(prior.to_dict())
    assert "P1" not in blob and "P2" not in blob
    assert prior_digest("P1") in blob


# ------------------------------------------------------------------ the seam

def test_the_builders_output_is_what_the_renderer_consumes(tmp_path):
    """The producer→consumer property. `experience_block` has rendered this shape since it was
    written and had no producer; this is the test that keeps them attached."""
    from acr.review.document_concepts import experience_block
    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma"]),
            _label("P1", "n2", "Imaging", "neither", ["mass"]),
            _label("P2", "n3", "Path", "can_establish", ["carcinoma"])]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")

    block = experience_block(to_experience_asset(prior))

    assert block, "the renderer produced nothing from a real prior"
    assert "RETRIEVAL EXPERIENCE" in block
    assert "carcinoma" in block
    assert "Path" in block
    assert "2 patient(s)" in block


def test_a_round_trip_through_json_preserves_every_number(tmp_path):
    rows = [_label("P1", "n1", "Path", "can_establish", ["carcinoma", "patient"]),
            _label("P2", "n2", "Imaging", "merely_mentions", ["patient"])]
    prior = build_prior(_write(tmp_path, rows), fields=[FIELD], min_patients=2, asset_id="t")
    again = RetrievalPrior.from_dict(json.loads(json.dumps(prior.to_dict())))
    assert again.to_dict() == prior.to_dict()
    assert again.content_hash == prior.content_hash
