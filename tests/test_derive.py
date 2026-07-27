"""The first-order derivation, tested entirely offline against fixtures authored here.

NOTHING IN THIS FILE TOUCHES A REAL CHART, THE CACHED BITMAPS ON /N/project, A MODEL OR THE
NETWORK. The synthetic corpus below is thirty patients of pure Python, and it is shaped around
the cases that have actually gone wrong in this repo:

  * a term one patient's notes propose forty times — the shape that looks exactly like a
    corpus-wide term to any note-counting rank
  * "patho" and "pathology" matching the same documents, where the stem must win, and
    "biops" dragging a pile in behind "biopsy", where it must not
  * a rare document type that is the ONLY establishing source for a run of patients
  * one imaging type that is high-yield for primary_site and silent on histology — the C349
    coding error, in fixture form
  * an imaging type whose labels state the histology constantly and which still may not
    establish it, because the spec says so and the count does not get a vote
  * a document type no stratum speaks for, where guessing is the failure
  * two fields established by the very same notes and a third established by different ones,
    which is the question stage 5 answers: how many assets does this spec need?
  * a labelling that answers question 1 once per NOTE, which must be refused rather than
    spread across the fields
"""
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import types
from dataclasses import MISSING, fields
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from acr import derive as D
from acr.spec import load_spec

runner = CliRunner()

HISTO, SITE, BEHAV = "histology", "primary_site", "behavior"
#: The order the fields are counted in, and so the order the grouping walks them in.
THREE = (HISTO, BEHAV, SITE)
PATH, PROG, CT, TB, ODD = ("Surgical-Pathology-Report", "Progress-Note", "Chest-CT-W-Contr",
                           "Tumor-Board-Note", "Podiatry-Note")

#: Everything the imaginary cache indexed. A needle outside this list has no price, and
#: `price_terms` must say so rather than skip it.
NEEDLES = ("carcinoma", "patho", "pathology", "biops", "biopsy", "final diagnosis", "nod",
           "cough")


# ------------------------------------------------------------------ the synthetic cache
def bitmaps(root: Path) -> D.DocBitmaps:
    """Thirty patients of documents, written as a cache in the real cache's own format.

    Row layout is chosen so that the stage-3 story is unambiguous:
      patients  0-9   answer reachable by "carcinoma"        (the list already in the spec)
      patients 10-19  answer reachable only by "patho"/"pathology", which match identically
      patients 20-29  answer reachable only by "final diagnosis"
      everyone        a pile of noise carrying "nod" and "cough" and no answers
      "biops" matches every "biopsy" document plus 40 noise ones — a stem that costs extra
    """
    recs, ix = [], {}

    def add(pid, terms, answer):
        r = ix.setdefault(pid, {"pid": pid, "doc_type": [], "date": [], "hits": [], "oracle": []})
        mask = 0
        for t in terms:
            mask |= 1 << NEEDLES.index(t)
        r["doc_type"].append(PATH if answer else CT)
        r["date"].append("2021-01-01")
        r["hits"].append(mask)
        r["oracle"].append(answer)

    for i in range(30):
        pid = f"SYN{i:04d}"
        if i < 10:
            add(pid, ("carcinoma", "biopsy", "biops"), True)
        elif i < 20:
            add(pid, ("patho", "pathology"), True)
        else:
            add(pid, ("final diagnosis",), True)
        for _ in range(4):
            add(pid, ("nod", "cough"), False)
    for j in range(40):  # what "biops" drags in and "biopsy" does not
        add(f"SYN{j % 30:04d}", ("biops", "nod"), False)

    (root / "meta.json").write_text(json.dumps({"needles": list(NEEDLES)}))
    (root / "v1").mkdir(exist_ok=True)
    with gzip.open(root / "v1" / "chunk0000.pkl.gz", "wb") as fh:
        pickle.dump(list(ix.values()), fh, 4)
    return D.load_bitmaps(root)


@pytest.fixture
def bm(tmp_path) -> D.DocBitmaps:
    return bitmaps(tmp_path)


# ------------------------------------------------------------------ the synthetic labels
def label(pid, doc_type, *, can=(), mentions=(), terms=(), error=""):
    """One labelled note, in the shape `labelling.py` writes: ONE VERDICT PER FIELD.

    `can` and `mentions` are the fields this note can establish and the fields it only carries.
    Everything else is `neither`. There is deliberately no way to write a note-level verdict
    here — the fixture cannot express the shape the derivation refuses.
    """
    verdict = lambda f: ("can_establish" if f in can else
                         "merely_mentions" if f in mentions else "neither")
    return {"patient_id": pid, "note_id": f"{doc_type}_2021-01-01", "doc_type": doc_type,
            "error": error,
            "admissibility": {f: verdict(f) for f in THREE},
            D.TERMS_FIELD: [{"term": t, "reason": r} for t, r in terms]}


def labels() -> list[dict]:
    """Thirty patients. Read the doc_type column: it is what stage 4 turns into policy, and
    read the per-field verdicts: they are what stage 5 turns into a grouping.

    Behavior is established by exactly the notes that establish the histology and by no
    others; the site is established by a partly different set. That is this repo's own case —
    morphology fields travel together, the anatomical site does not travel with them.
    """
    out = []
    for i in range(30):
        pid = f"SYN{i:04d}"
        # Pathology: admissible and high-yield. For the first five patients it localises the
        # tumour and stops short of a morphology — which is what makes the rare type below
        # their ONLY establishing source for it. One doc_type, two answers, one note.
        out.append(label(pid, PATH, can=(SITE,) if i < 5 else (HISTO, BEHAV, SITE),
                         terms=[("carcinoma", "names_the_answer"),
                               ("final diagnosis", "names_the_section")]))
        # Imaging: carries the SITE nearly always and the histology never. Same type, two
        # different answers — this is the row the shipped `cannot_establish` name destroyed.
        for _ in range(3):
            out.append(label(pid, CT, mentions=(SITE,), terms=[("nod", "other")]))
        # A restatement machine: carries every field constantly, establishes nothing.
        out.append(label(pid, TB, mentions=THREE, terms=[("patho", "names_the_document")]))
        if i < 5:  # rare, and for these five patients the ONLY establishing source
            out.append(label(pid, PROG, can=(HISTO, BEHAV),
                             terms=[("pathology", "names_the_document")]))
    # One patient's private vocabulary: forty notes, one patient, a term nobody else uses.
    out += [label("SYN0000", PROG, terms=[("cough", "other")]) for _ in range(40)]
    return out


@pytest.fixture
def agg() -> D.Aggregate:
    return D.aggregate(labels(), (HISTO, SITE))


@pytest.fixture
def agg3() -> D.Aggregate:
    """The same labels counted over all three fields — the stage-5 input."""
    return D.aggregate(labels(), THREE)


CFG = D.DerivationConfig(max_extra_docs_per_answer=10.0, high_yield_rate=0.5,
                         min_patients_proposing=2, share_asset_jaccard=0.8)


# ------------------------------------------------------------------ the synthetic spec
def spec_doc(rest: bool = True) -> dict:
    def prov(el):
        return {"element": el, "origin": "model_authored", "status": "draft",
                "basis": "no external source; authored in a test fixture"}

    strata = [
        {"name": "can_establish", "policy": "exhaustive", "establishes": [HISTO, SITE],
         "match": {"doc_type_matches": ["Pathology", "Progress-Note"]}},
        {"name": "may_mention", "policy": "search_then_read_hits_and_sample_misses",
         "establishes": [HISTO, SITE], "match": {"doc_type_matches": ["Tumor-Board"]},
         "required_keywords": ["carcinoma"], "min_sample_of_misses": 25},
        # The whole point of the fixture: imaging may localise and may not diagnose.
        {"name": "imaging", "policy": "validate_by_sampling", "establishes": [SITE],
         "match": {"rest": True} if rest else {"doc_type_matches": ["CT"]}, "min_sample": 25},
    ]
    leaves = {"can_establish": ("match", "establishes"),
              "may_mention": ("match", "establishes", "required_keywords",
                              "min_sample_of_misses"),
              "imaging": ("match", "establishes", "min_sample")}
    return {"spec_id": "SYNTH.998.derive_fixture", "question": "Site and histology?",
            "fields": [{"name": HISTO, "type": "string"}, {"name": SITE, "type": "string"}],
            "proof_obligation": {"for_positive": "One pathology report.",
                                 "for_negative": {"mode": "stratified_exclusion",
                                                  "strata": strata}},
            "provenance": [prov(f"proof_obligation.for_negative.strata[{s['name']}].{leaf}")
                           for s in strata for leaf in leaves[s["name"]]]}


@pytest.fixture
def spec_path(tmp_path) -> Path:
    p = tmp_path / "SYNTH.998.derive.yaml"
    p.write_text(yaml.safe_dump(spec_doc(), sort_keys=False), encoding="utf-8")
    load_spec(p)  # the fixture is only useful if it is a spec
    return p


@pytest.fixture
def spec(spec_path):
    return load_spec(spec_path)


def digest(root: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode() + p.read_bytes())
    return h.hexdigest()


# ============================================================ no magic numbers
def test_the_config_has_no_default_for_anything():
    """A default here is a threshold chosen at import time by whoever typed it, and nothing
    downstream would ever say a choice had been made."""
    for f in fields(D.DerivationConfig):
        assert f.default is MISSING and f.default_factory is MISSING, \
            f"{f.name} has a default; it is a decision with an owner"
    with pytest.raises(TypeError):
        D.DerivationConfig()
    with pytest.raises(TypeError):
        D.DerivationConfig(max_extra_docs_per_answer=1.0, high_yield_rate=0.5)


@pytest.mark.parametrize("kw", [{"max_extra_docs_per_answer": -1.0},
                                {"high_yield_rate": 1.5},
                                {"min_patients_proposing": 0},
                                {"share_asset_jaccard": 1.5},
                                {"share_asset_jaccard": -0.1}])
def test_a_threshold_outside_its_domain_is_refused(kw):
    base = {"max_extra_docs_per_answer": 1.0, "high_yield_rate": 0.5,
            "min_patients_proposing": 1, "share_asset_jaccard": 0.8}
    with pytest.raises(ValueError):
        D.DerivationConfig(**{**base, **kw})


def test_a_stage_that_needs_a_threshold_refuses_to_run_without_one(agg, spec, bm):
    """Not "falls back to something sensible". There is nothing sensible to fall back to."""
    with pytest.raises(D.DerivationError):
        D.consolidate([], bm, None)
    with pytest.raises(D.DerivationError):
        D.derive_policy(agg, spec, None)
    with pytest.raises(D.DerivationError):
        D.suggest_grouping(agg, None)


def test_every_threshold_used_is_recorded_in_the_output(agg, spec, bm):
    con = D.consolidate(D.price_terms(["patho"], bm), bm, CFG)
    assert con.as_dict()["config"] == CFG.as_dict()
    assert D.derive_policy(agg, spec, CFG).as_dict()["config"] == CFG.as_dict()
    assert D.suggest_grouping(agg, CFG).as_dict()["config"] == CFG.as_dict()


# ============================================================ stage 1 — aggregate
def test_a_term_forty_notes_of_one_patient_proposed_is_that_patients_vocabulary(agg):
    """"cough" appears in forty notes and is the top term by note count. One patient wrote
    all forty. Ranking by notes would put it first; ranking by patients drops it."""
    assert agg.terms["cough"].n_notes == 40
    assert agg.terms["cough"].n_patients == 1
    assert agg.terms["cough"].n_notes > agg.terms["final diagnosis"].n_notes
    kept = [t.term for t in agg.ranked_terms(CFG)]
    assert "cough" not in kept and "final diagnosis" in kept
    assert kept[0] == max(kept, key=lambda t: agg.terms[t].n_patients)


def test_the_patient_floor_is_the_config_and_moving_it_moves_the_list(agg):
    lo = D.DerivationConfig(10.0, 0.5, 1, 0.8)
    assert "cough" in [t.term for t in agg.ranked_terms(lo)]
    hi = D.DerivationConfig(10.0, 0.5, 6, 0.8)
    assert "pathology" not in [t.term for t in agg.ranked_terms(hi)]  # only 5 patients


def test_reason_classes_are_counted_per_term(agg):
    assert agg.terms["carcinoma"].reasons == {"names_the_answer": 30}
    assert agg.terms["final diagnosis"].reasons == {"names_the_section": 30}


def test_an_unrecognised_reason_class_is_surfaced_and_not_folded_into_other():
    """Folding it away would make a change to the labelling prompt invisible here."""
    a = D.aggregate([label("P1", PATH, terms=[("nod", "names_a_measurement")])], (HISTO,))
    assert a.unknown_reason_classes == ("names_a_measurement",)
    assert a.terms["nod"].reasons == {"names_a_measurement": 1}


def test_a_label_with_no_terms_at_all_is_not_an_error():
    """A note the reader found no searchable term in is a real answer to question 2. The
    derivation then counts document types and no terms — smaller, not wrong. This is the one
    absence that is tolerated: a MISSING VERDICT is not, because it would be counted."""
    lab = label("P1", PATH, can=(HISTO,))
    lab.pop(D.TERMS_FIELD)
    a = D.aggregate([lab], (HISTO,))
    assert a.terms == {} and a.types[(PATH, HISTO)].n_can_establish == 1


def test_a_bare_string_candidate_is_accepted_and_marked_unclassified():
    a = D.aggregate([{"patient_id": "P1", "doc_type": PATH, "admissibility": {HISTO: "neither"},
                      D.TERMS_FIELD: ["  PaThO "]}], (HISTO,))
    assert a.terms["patho"].reasons == {D.UNCLASSIFIED: 1}


def test_a_label_that_failed_to_parse_is_not_counted_against_a_types_yield():
    """A note the reader could not read says nothing about a type, and padding the
    denominator with it understates every type it touches."""
    labs = [label("P1", CT, mentions=(SITE,)), label("P1", CT, error="timeout")]
    assert D.aggregate(labs, (SITE,)).types[(CT, SITE)].n == 1


def test_sole_source_counts_the_patients_a_type_alone_decided(agg):
    """The rare type is the number that matters. PROG appears for five patients out of thirty
    and is the only thing that establishes their histology; imaging never establishes
    anything, however much of the chart it is."""
    prog, ct = agg.types[(PROG, HISTO)], agg.types[(CT, SITE)]
    assert prog.n_patients_sole_source == 5
    assert ct.n_patients_sole_source == 0 and ct.n > prog.n
    # PATH establishes histology for every patient, but for those five it is not alone.
    assert agg.types[(PATH, HISTO)].n_patients_sole_source == 25


def test_a_mention_is_not_an_establishing_source(agg):
    """TB states the histology in every note and establishes it for nobody: `merely_mentions`
    plus a stated value is corroboration, not a witness."""
    tb = agg.types[(TB, HISTO)]
    assert tb.n_states == 30 and tb.n_merely_mentions == 30
    assert tb.n_patients_sole_source == 0


def test_aggregate_with_no_fields_refuses():
    with pytest.raises(D.DerivationError):
        D.aggregate(labels(), ())


# ================================================ the verdict is per FIELD, not per note
def test_one_note_carries_a_different_verdict_for_each_field(agg):
    """The whole of change 1. The same pathology report establishes the site for all thirty
    patients and the histology for twenty-five, and until the labels answered per field this
    row could only be one number applied to both."""
    assert agg.types[(PATH, SITE)].n_can_establish == 30
    assert agg.types[(PATH, HISTO)].n_can_establish == 25
    assert agg.types[(PATH, HISTO)].n_neither == 5


def test_yield_differs_between_two_fields_of_one_document_type(agg):
    """A note-level verdict copied onto every field makes these two numbers equal by
    construction. They are the numbers stage 4 turns into two different policies."""
    assert agg.types[(CT, SITE)].yield_rate == 1.0
    assert agg.types[(CT, HISTO)].yield_rate == 0.0


def test_a_labelling_with_one_verdict_per_note_is_refused_not_spread_over_the_fields():
    """The refusal that replaces the old fallback. A note-level verdict applied to every field
    produces a per-field matrix whose rows are all the same number: not a coarser answer, a
    wrong one, and nothing on any row would have said so."""
    old = {"patient_id": "P1", "note_id": "n1", "doc_type": PATH,
           "admissibility": {"verdict": "can_establish", "quote": "x"},
           D.TERMS_FIELD: []}
    with pytest.raises(D.StaleLabellingError) as e:
        D.aggregate([old], (HISTO, SITE))
    assert "per FIELD" in str(e.value) and "can_establish" in str(e.value)


def test_a_labelling_with_no_standing_at_all_is_refused():
    """An empty admissibility is not "neither" for every field: it is a row that never
    answered question 1, and counting it would move every denominator it touches."""
    with pytest.raises(D.StaleLabellingError):
        D.aggregate([{"patient_id": "P1", "doc_type": PATH, "admissibility": {}}], (HISTO,))


def test_a_field_the_labelling_never_answered_for_is_refused():
    """These labels were made against a different requirement. Reading the silence as
    "neither" would understate every document type that in fact carries the field."""
    lab = label("P1", PATH, can=(HISTO,))
    lab["admissibility"].pop(SITE)
    with pytest.raises(D.StaleLabellingError) as e:
        D.aggregate([lab], (HISTO, SITE))
    assert SITE in str(e.value)


def test_a_verdict_this_module_has_not_been_taught_raises():
    """A fourth class the prompt learned and this module did not would otherwise be counted
    into a column nothing reads, and every rate would quietly move."""
    lab = label("P1", PATH)
    lab["admissibility"][HISTO] = "probably"
    with pytest.raises(D.DerivationError) as e:
        D.aggregate([lab], (HISTO,))
    assert "probably" in str(e.value)


def test_a_per_field_verdict_may_arrive_as_an_object_with_its_own_quote():
    """The labeller may hang a quote off each field's verdict. What this module needs is the
    verdict; it reads it out of either shape and never reads the quote."""
    lab = label("P1", PATH)
    lab["admissibility"][HISTO] = {"verdict": "can_establish", "quote": "a sentence"}
    assert D.aggregate([lab], (HISTO,)).types[(PATH, HISTO)].n_can_establish == 1


# ================================================ the wire between the two modules
def test_the_key_the_labeller_writes_is_the_key_derive_reads():
    """A previous pass had this module reading a `candidate_terms` key `labelling.py` has
    never written. Nothing failed: the terms arrived as zero, every stage ran, and every
    number downstream was quietly a number about no terms at all. So the test does not compare
    two string constants — it builds a real `labelling.NoteLabel`, serialises it exactly as
    `LabelStore.append` does, and fails if the derivation cannot find what is in it."""
    from acr import labelling as L

    assert D.TERMS_FIELD in L.NoteLabel.__dataclass_fields__, \
        f"derive reads {D.TERMS_FIELD!r}; NoteLabel has no such field"
    assert D.ADMISSIBILITY_FIELD in L.NoteLabel.__dataclass_fields__, \
        f"derive reads {D.ADMISSIBILITY_FIELD!r}; NoteLabel has no such field"

    # Serialised by the labeller's own writer, which is what `LabelStore.append` puts on disk.
    row = L.NoteLabel(
        patient_id="P1", note_id="n1", doc_type=PATH,
        admissibility=L.Admissibility(verdicts={HISTO: "can_establish", SITE: "neither"},
                                      quote="a sentence"),
        retrieval_terms=(L.RetrievalTerm("Final Diagnosis", "names_the_section"),)).to_dict()
    a = D.aggregate([row], (HISTO, SITE))
    assert a.terms["final diagnosis"].n_notes == 1, \
        "the labeller's terms did not reach the derivation: the two modules disagree on the key"
    assert a.unknown_reason_classes == (), \
        "the labeller's reason classes are not the ones the derivation aggregates"
    # The two verdicts on one note are different, and the derivation must have BOTH of them.
    assert a.types[(PATH, HISTO)].n_can_establish == 1
    assert a.types[(PATH, SITE)].n_neither == 1


def test_the_collapsed_verdict_the_labeller_exports_is_not_mistaken_for_the_old_shape():
    """`NoteLabel.to_dict` writes a note-level `verdict` beside the per-field map, for readers
    that have not learned to ask per field. This one has. Reading the collapse — or refusing
    the row for carrying it — would undo the whole change while every row still looked right."""
    from acr import labelling as L

    row = L.NoteLabel(patient_id="P1", note_id="n1", doc_type=PATH,
                      admissibility=L.Admissibility(
                          verdicts={HISTO: "can_establish", SITE: "neither"},
                          quote="q")).to_dict()
    assert row["admissibility"]["verdict"] == "can_establish", "the projection has moved"
    assert D.field_verdicts(row, (HISTO, SITE)) == {HISTO: "can_establish", SITE: "neither"}
    # The same label as the object `LabelStore.load` hands back, not only as a JSON row.
    assert D.field_verdicts(L.NoteLabel.from_dict(row), (HISTO, SITE)) \
        == {HISTO: "can_establish", SITE: "neither"}


def test_a_labelling_the_derivation_can_read_is_one_the_labeller_can_read_back():
    """Both modules refuse the pre-per-field row, and neither may be the only one that does."""
    from acr import labelling as L

    old = {"patient_id": "P1", "note_id": "n1", "doc_type": PATH,
           "admissibility": {"verdict": "can_establish", "quote": "q"}}
    with pytest.raises(L.LabelShapeError):
        L.NoteLabel.from_dict(old)
    with pytest.raises(D.StaleLabellingError):
        D.aggregate([old], (HISTO,))


def test_the_derivation_reads_exactly_one_name_for_each_question():
    """No fallback key. A second name read when the first is missing means a rename in either
    module still produces a full run over zero terms, which is the failure this pair of
    constants exists to make loud."""
    assert [n for n in vars(D) if n.endswith("TERMS_FIELD")] == ["TERMS_FIELD"]
    lab = {"patient_id": "P1", "doc_type": PATH, "admissibility": {HISTO: "neither"},
           "candidate_terms": [{"term": "patho", "reason": "other"}]}
    assert D.aggregate([lab], (HISTO,)).terms == {}


# ============================================================ stage 2 — price
def test_a_candidate_the_cache_cannot_price_raises_rather_than_being_skipped(bm):
    """Skipping would silently turn the ranking into a ranking of the cached terms, which
    reads exactly like a ranking of the good ones."""
    with pytest.raises(D.UnpricedTermError) as e:
        D.price_terms(["carcinoma", "lobectomy"], bm)
    assert "lobectomy" in str(e.value)


def test_a_term_is_priced_against_the_list_already_in_the_spec(bm):
    """"carcinoma" matches ten answer-bearing documents. The spec already searches it, so it
    rescues nothing and reads nothing new. A raw-recall ranking would put it first."""
    (p,) = D.price_terms(["carcinoma"], bm, current=["carcinoma"])
    assert p.answer_bearing_matched == 10
    assert p.answers_rescued == 0 and p.extra_documents == 0
    (q,) = D.price_terms(["carcinoma"], bm, current=[])
    assert q.answers_rescued == 10


def test_price_reports_cost_and_yield_separately(bm):
    (nod,) = D.price_terms(["nod"], bm, current=["carcinoma"])
    assert nod.answers_rescued == 0 and nod.extra_documents == 160  # 120 noise + 40 biops rows
    (fd,) = D.price_terms(["final diagnosis"], bm, current=["carcinoma"])
    assert (fd.answers_rescued, fd.extra_documents) == (10, 10)


def test_the_bitmaps_keep_no_patient_id(tmp_path, bm):
    """The pid is dropped at load rather than carried and filtered later: a field that does
    not exist cannot leak into an artefact."""
    assert "SYN0000" not in json.dumps(bm.rows)
    assert not any("pid" in f or "patient_id" in f for f in D.DocBitmaps.__dataclass_fields__)
    assert bm.n_patients == 30


# ============================================================ stage 3 — consolidate
def test_a_stem_replaces_its_variants_when_it_subsumes_them_at_no_extra_cost(bm):
    """Dropping "pathology" for "patho" was a real, independently rediscovered win on 1,770
    patients. Here they match identically, so the stem carries the variant for free."""
    con = D.consolidate(D.price_terms(["pathology", "patho"], bm, ["carcinoma"]), bm, CFG,
                        ["carcinoma"])
    assert con.merged == {"pathology": "patho"}
    assert "patho" in con.keywords and "pathology" not in con.keywords


def test_a_stem_that_drags_a_pile_in_does_not_replace_its_variant(bm):
    """"biops" covers every "biopsy" document and forty more. Coverage alone would merge it;
    the no-extra-cost test is what keeps the cheaper variant."""
    con = D.consolidate(D.price_terms(["biopsy", "biops"], bm, []), bm, CFG, [])
    assert con.merged == {}
    assert "biopsy" in con.keywords


def test_duplicates_collapse_before_anything_is_ranked(bm):
    con = D.consolidate(D.price_terms(["patho", "patho"], bm, []), bm, CFG, [])
    assert [r.term for r in con.curve].count("patho") == 1


def test_the_whole_curve_is_emitted_beside_the_cut(bm):
    """The person choosing the threshold has to see what every other setting would buy, so
    the rejected rows are ranked and priced too, not dropped."""
    priced = D.price_terms(["patho", "final diagnosis", "nod", "cough"], bm, ["carcinoma"])
    con = D.consolidate(priced, bm, CFG, ["carcinoma"])
    assert {r.term for r in con.curve} == {"patho", "final diagnosis", "nod", "cough"}
    assert sorted(r.term for r in con.curve if r.in_cut) == ["final diagnosis", "patho"]
    assert [r.term for r in con.curve if not r.in_cut][0] in ("nod", "cough")
    assert con.curve[0].cum_answers_rescued == 10 and con.curve[1].cum_answers_rescued == 20
    assert all(r.docs_per_answer == float("inf") for r in con.curve if
               r.marginal_answers_rescued == 0)


def test_the_cut_is_a_prefix_of_the_curve(bm):
    """A term admitted after a rejected one would be admitted only because a cheaper term was
    refused first, and the resulting list is not one any single threshold would have chosen."""
    priced = D.price_terms(["patho", "nod", "final diagnosis"], bm, ["carcinoma"])
    flags = [r.in_cut for r in D.consolidate(priced, bm, CFG, ["carcinoma"]).curve]
    assert flags == sorted(flags, reverse=True)


def test_a_tighter_threshold_buys_fewer_terms(bm):
    priced = lambda: D.price_terms(["patho", "final diagnosis", "nod"], bm, ["carcinoma"])
    loose = D.consolidate(priced(), bm, D.DerivationConfig(1000.0, 0.5, 1, 0.8), CFG.as_dict() and
                          ["carcinoma"])
    tight = D.consolidate(priced(), bm, D.DerivationConfig(0.5, 0.5, 1, 0.8), ["carcinoma"])
    assert len(tight.keywords) < len(loose.keywords)
    assert set(tight.keywords) <= set(loose.keywords)


def test_the_list_starts_from_what_the_spec_already_searches(bm):
    con = D.consolidate(D.price_terms(["patho"], bm, ["carcinoma"]), bm, CFG, ["carcinoma"])
    assert con.keywords[0] == "carcinoma"


# ============================================================ stage 4 — policy
@pytest.mark.parametrize("admissible,high,expect", [
    (True, True, D.READ_ALL), (True, False, D.SEARCH),
    (False, True, D.SEARCH), (False, False, D.SAMPLE)])
def test_the_policy_matrix_is_exactly_the_declared_two_by_two(admissible, high, expect):
    assert D.policy_for(admissible, high) == expect


def test_one_imaging_type_is_read_all_for_the_site_and_sample_for_the_histology(agg, spec):
    """The C349 error, in fixture form. One doc_type, two fields, two policies — which is
    only expressible because yield is measured per FIELD and admissibility is read per field
    out of the stratum's `establishes`."""
    rows = {(r.doc_type, r.field): r for r in D.derive_policy(agg, spec, CFG).rows}
    site, histo = rows[(CT, SITE)], rows[(CT, HISTO)]
    assert (site.admissible, site.high_yield, site.policy) == (True, True, D.READ_ALL)
    assert (histo.admissible, histo.high_yield, histo.policy) == (False, False, D.SAMPLE)
    assert site.admissibility_source == histo.admissibility_source == "imaging"


def test_a_type_that_only_mentions_is_searched_and_never_read_as_a_witness(agg, spec):
    """`may_mention` establishes both fields in this spec, so TB is admissible; the axis the
    test pins is that high yield on an INADMISSIBLE type still yields SEARCH, not READ_ALL —
    corroboration and the absence proof are worth the read, the witness slot is not."""
    labs = [label(f"SYN{i:04d}", CT, mentions=(HISTO,))
            for i in range(30)]
    rows = {(r.doc_type, r.field): r
            for r in D.derive_policy(D.aggregate(labs, (HISTO,)), spec, CFG).rows}
    r = rows[(CT, HISTO)]
    assert (r.admissible, r.high_yield, r.yield_rate, r.policy) == (False, True, 1.0, D.SEARCH)


def test_admissibility_is_never_inferred_from_the_yield(agg, spec):
    """Imaging that states the histology in EVERY note is still not allowed to establish it.
    The count moves `high_yield`; only the spec moves `admissible`."""
    labs = [label(f"SYN{i:04d}", CT, can=(HISTO,)) for i in range(30)]
    row = [r for r in D.derive_policy(D.aggregate(labs, (HISTO,)), spec, CFG).rows][0]
    assert row.n_states == row.n == 30 and row.high_yield is True
    assert row.admissible is False and row.policy == D.SEARCH


def test_admissibility_tracks_the_spec_when_the_spec_changes(tmp_path, agg):
    """Edit `establishes` and the policy follows. Nothing else in the pipeline may move it."""
    doc = spec_doc()
    doc["proof_obligation"]["for_negative"]["strata"][2]["establishes"] = [HISTO, SITE]
    p = tmp_path / "widened.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    rows = {(r.doc_type, r.field): r for r in D.derive_policy(agg, load_spec(p), CFG).rows}
    assert rows[(CT, HISTO)].admissible is True and rows[(CT, HISTO)].policy == D.SEARCH


def test_a_document_type_no_stratum_speaks_for_raises(tmp_path, agg):
    """There is no `rest` stratum here and nothing matches a podiatry note. Whether it may
    establish a field is a clinician's answer; a frequency is not an answer to it."""
    p = tmp_path / "norest.yaml"
    p.write_text(yaml.safe_dump(spec_doc(rest=False), sort_keys=False), encoding="utf-8")
    labs = labels() + [label("SYN0000", ODD, mentions=(HISTO,))]
    with pytest.raises(D.UndeclaredAdmissibilityError) as e:
        D.derive_policy(D.aggregate(labs, (HISTO, SITE)), load_spec(p), CFG)
    assert ODD in str(e.value)


def test_yield_is_a_rate_over_the_types_own_notes(agg):
    ct = agg.types[(CT, SITE)]
    assert ct.n == 90 and ct.n_states == 90 and ct.yield_rate == 1.0
    assert agg.types[(CT, HISTO)].yield_rate == 0.0


def test_moving_the_yield_threshold_moves_the_policy_and_nothing_else(agg, spec):
    """Half the fixture's imaging notes state the site; at 0.4 that is high, at 0.6 it is
    not, and the same admissibility gives READ_ALL then SEARCH."""
    labs = ([label(f"SYN{i:04d}", CT, mentions=(SITE,)) for i in range(15)]
            + [label(f"SYN{i:04d}", CT) for i in range(15, 30)])
    a = D.aggregate(labs, (SITE,))
    lo = D.derive_policy(a, spec, D.DerivationConfig(10.0, 0.4, 1, 0.8)).rows[0]
    hi = D.derive_policy(a, spec, D.DerivationConfig(10.0, 0.6, 1, 0.8)).rows[0]
    assert (lo.high_yield, lo.policy) == (True, D.READ_ALL)
    assert (hi.high_yield, hi.policy) == (False, D.SEARCH)
    assert lo.admissible == hi.admissible is True


# ============================================================ stage 5 — how many assets?
def grouping(agg, jaccard):
    return D.suggest_grouping(agg, D.DerivationConfig(10.0, 0.5, 1, jaccard))


def test_the_overlap_matrix_is_measured_on_the_establishing_notes(agg3):
    """Behavior is established by exactly the notes that establish the histology: 30 notes,
    30 in common, coefficient 1. The site is established by 30 notes of which 25 also
    establish a morphology: 25/35. Nobody has to guess this — the labels say it."""
    m = {(o.field_a, o.field_b): o for o in D.overlap_matrix(agg3)}
    hb, hs = m[(HISTO, BEHAV)], m[(HISTO, SITE)]
    assert (hb.n_a, hb.n_b, hb.n_both, hb.jaccard) == (30, 30, 30, 1.0)
    assert (hs.n_a, hs.n_b, hs.n_both) == (30, 30, 25)
    assert hs.jaccard == pytest.approx(25 / 35)
    assert len(m) == 3, "every pair is measured, including the ones no threshold would merge"


def test_only_the_notes_that_can_establish_a_field_count_toward_its_overlap(agg3):
    """The tumour board note carries all three fields in every note and establishes none of
    them. If mentions counted, every pair here would coincide perfectly and the matrix would
    propose one asset for a spec that needs two."""
    assert agg3.n_establishing(HISTO) == 30 and agg3.types[(TB, HISTO)].n_states == 30
    # 30 pathology reports and 5 progress notes establish something; 150 imaging notes and 30
    # tumour board notes carry these fields constantly and establish nothing.
    assert sum(agg3.establishing_profiles.values()) == 35
    assert agg3.establishing_profiles == {(SITE,): 5, (HISTO, BEHAV, SITE): 25, (HISTO, BEHAV): 5}


def test_two_fields_nothing_establishes_do_not_read_as_perfectly_alike():
    """Empty sets are identical, and no evidence is not agreement. A coefficient of 1 here
    would propose one asset for two fields nothing has been measured about."""
    (o,) = D.overlap_matrix(D.aggregate([label("P1", CT, mentions=(HISTO, SITE))],
                                        (HISTO, SITE)))
    assert (o.n_a, o.n_b, o.n_both, o.jaccard) == (0, 0, 0, 0.0)
    assert grouping(D.aggregate([label("P1", CT)], (HISTO, SITE)), 0.5).n_assets == 2


def test_the_cut_is_the_config_and_moving_it_moves_which_fields_share_an_asset(agg3):
    """At 0.5 all three fields are one asset; at 0.8 the two morphology fields are one and the
    site is its own. Same labels, same matrix, two answers — which is exactly why the
    threshold is required and the whole matrix is printed beside it."""
    assert grouping(agg3, 0.5).groups == ((HISTO, BEHAV, SITE),)
    assert grouping(agg3, 0.8).groups == ((HISTO, BEHAV), (SITE,))
    assert grouping(agg3, 0.8).n_assets == 2


def test_a_grouping_never_loses_a_field_and_never_repeats_one(agg3):
    for j in (0.0, 0.3, 0.5, 0.71, 0.8, 1.0):
        flat = [f for g in grouping(agg3, j).groups for f in g]
        assert sorted(flat) == sorted(THREE), f"threshold {j} lost or duplicated a field"


def test_a_tighter_cut_never_merges_more(agg3):
    counts = [grouping(agg3, j).n_assets for j in (0.0, 0.5, 0.8, 1.0)]
    assert counts == sorted(counts)


def test_a_group_needs_every_pair_to_clear_the_cut_and_not_a_chain():
    """A and B overlap, B and C overlap, A and C not at all. Single linkage would put all
    three on one asset on the strength of a pair nothing measured; every pair inside a
    proposed group has to clear the cut on its own."""
    labs = ([label(f"P{i}", PATH, can=(HISTO, BEHAV)) for i in range(6)]
            + [label(f"Q{i}", PROG, can=(BEHAV, SITE)) for i in range(6)])
    a = D.aggregate(labs, THREE)
    m = {(o.field_a, o.field_b): o.jaccard for o in D.overlap_matrix(a)}
    assert m[(HISTO, BEHAV)] == m[(BEHAV, SITE)] == 0.5 and m[(HISTO, SITE)] == 0.0
    assert grouping(a, 0.5).groups == ((HISTO, BEHAV), (SITE,))


def test_the_grouping_carries_the_whole_matrix_not_only_the_pairs_it_merged(agg3):
    """The suggestion is unreadable without the rows it rejected: they are what tells the
    reader that 0.7 would have merged the site too."""
    prop = grouping(agg3, 0.8)
    assert {(o.field_a, o.field_b) for o in prop.overlaps} == \
        {(HISTO, BEHAV), (HISTO, SITE), (BEHAV, SITE)}
    body = prop.as_dict()
    assert len(body["overlaps"]) == 3 and len(body["groups"]) == 2
    assert all(0.0 <= o["jaccard"] <= 1.0 for o in body["overlaps"])


def test_a_grouping_is_a_proposal_and_the_spec_is_untouched(tmp_path, spec_path, agg3):
    """Merging two fields' assets changes what evidence is sought for each of them. That is
    semantic, so it is proposed and signed, exactly like the policy — and for a wider blast
    radius, since a policy proposal misreads one document type and this decides a field."""
    before = spec_path.read_bytes()
    prop = D.suggest_grouping(agg3, CFG, spec_id="SYNTH.998.derive_fixture")
    path = D.emit_grouping_proposal(prop, tmp_path / "proposals", today="2026-07-27")
    assert spec_path.read_bytes() == before, "emitting a grouping edited the spec"
    body = yaml.safe_load(path.read_text())
    assert body["STATUS"].startswith("PROPOSED")
    assert body["signature"]["decision"] is None
    assert body["groups"] == [[HISTO, BEHAV], [SITE]]
    assert len(body["overlaps"]) == 3, "the file must carry the rejected pairs too"
    assert "SYN0000" not in path.read_text()


def test_the_grouping_emitter_cannot_be_handed_a_spec_to_write_into():
    """Not a convention — the parameter does not exist, so there is nothing to pass."""
    params = set(D.emit_grouping_proposal.__annotations__) | \
        set(D.emit_grouping_proposal.__code__.co_varnames)
    assert not {"spec", "spec_path", "adopt", "install"} & params


# ============================================================ who may adopt what
def test_keywords_are_written_into_the_spec_with_provenance_that_verifies(spec_path, bm):
    """Retrieval-only, so this write is allowed — and value and record land together or not
    at all, because a new list under the old list's record loads and reads as measured."""
    el = "proof_obligation.for_negative.strata[may_mention].required_keywords"
    con = D.consolidate(D.price_terms(["patho", "final diagnosis"], bm, ["carcinoma"]), bm, CFG,
                        ["carcinoma"])
    out = D.write_keywords(spec_path, el, con, run="run-abc", today="2026-07-27")
    assert out["outcome"] == "adopted"
    reloaded = load_spec(spec_path)
    rec = reloaded.provenance_index[el]
    assert rec.origin == "corpus_derived" and rec.measured["run"] == "run-abc"
    assert rec.measured["n_patients"] == 30 and rec.measured["config"] == CFG.as_dict()
    assert rec.status == "measured" and rec.measured["verdict"] == "supports"
    strata = reloaded.proof_obligation.for_negative["strata"]
    assert [s for s in strata if s["name"] == "may_mention"][0]["required_keywords"] \
        == list(con.keywords)


def test_a_derivation_that_rescued_nothing_stays_draft(spec_path, bm):
    """The measurement is the reason to distrust the element; it must not rank above one
    nobody has looked at."""
    el = "proof_obligation.for_negative.strata[may_mention].required_keywords"
    con = D.consolidate(D.price_terms(["carcinoma"], bm, ["carcinoma"]), bm, CFG, ["carcinoma"])
    D.write_keywords(spec_path, el, con, run="r0")
    rec = load_spec(spec_path).provenance_index[el]
    assert rec.status == "draft" and rec.measured["verdict"] == "underpowered"


def test_writing_into_a_stratum_that_does_not_exist_writes_nothing(spec_path, bm):
    before = spec_path.read_bytes()
    con = D.consolidate(D.price_terms(["patho"], bm, []), bm, CFG, [])
    with pytest.raises(D.AdoptionAborted):
        D.write_keywords(spec_path, "proof_obligation.for_negative.strata[nope]"
                                    ".required_keywords", con, run="r")
    assert spec_path.read_bytes() == before


def test_a_non_keyword_element_is_not_writable_by_this_module(spec_path, bm):
    """`establishes` is the admissibility declaration. There is no path through this module
    that edits it."""
    con = D.consolidate([], bm, CFG, [])
    with pytest.raises(D.AdoptionAborted):
        D.write_keywords(spec_path, "proof_obligation.for_negative.strata[imaging].establishes",
                         con, run="r")


def test_a_policy_only_ever_becomes_a_proposal_and_the_spec_is_untouched(tmp_path, spec_path,
                                                                        agg, spec):
    before = spec_path.read_bytes()
    prop = D.derive_policy(agg, spec, CFG)
    path = D.emit_policy_proposal(prop, tmp_path / "proposals", today="2026-07-27")
    assert spec_path.read_bytes() == before, "emitting a proposal edited the spec"
    body = yaml.safe_load(path.read_text())
    assert body["STATUS"].startswith("PROPOSED")
    assert body["signature"] == {"reviewed_by": None, "reviewed_on": None, "decision": None,
                                 "note": "accept | reject | accept_with_changes"}
    assert any(r["policy"] == D.READ_ALL for r in body["rows"])
    assert "SYN0000" not in path.read_text()


def test_the_policy_emitter_cannot_be_handed_a_spec_to_write_into():
    """Not a convention — the parameter does not exist, so there is nothing to pass."""
    params = set(D.emit_policy_proposal.__annotations__) | \
        set(D.emit_policy_proposal.__code__.co_varnames)
    assert not {"spec", "spec_path", "adopt", "install"} & params


def test_the_guard_covers_the_new_asset_kind_too():
    """A grouping is as semantic as a policy, so the flag that would install one is refused by
    name before anybody writes it."""
    def adopt_grouping(proposal, *, write_grouping: bool = False):
        return None

    fake = types.ModuleType("fake")
    adopt_grouping.__module__ = D.__name__
    fake.adopt_grouping = adopt_grouping
    with pytest.raises(D.SemanticOverrideError):
        D.assert_no_semantic_override(fake)


def test_no_public_callable_offers_a_way_around_the_clinician():
    """This runs at import too. It is here as a test so the failure has a name when it comes
    back, and because the guard itself has to be shown to work."""
    D.assert_no_semantic_override(D)


def test_the_override_guard_actually_catches_one():
    def adopt_policy(proposal, *, force: bool = False):
        return None

    fake = types.ModuleType("fake")
    adopt_policy.__module__ = D.__name__
    fake.adopt_policy = adopt_policy
    with pytest.raises(D.SemanticOverrideError) as e:
        D.assert_no_semantic_override(fake)
    assert "force" in str(e.value)


# ============================================================ the CLI
def _args(labels_path, spec_path, cache):
    return ["--labels", str(labels_path), "--spec", str(spec_path), "--cache", str(cache),
            "--fields", ",".join(THREE), "--max-extra-docs-per-answer", "10",
            "--high-yield-rate", "0.5", "--min-patients-proposing", "2",
            "--share-asset-jaccard", "0.8"]


@pytest.fixture
def labels_path(tmp_path) -> Path:
    p = tmp_path / "labels.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in labels()) + "\n{tor", encoding="utf-8")
    return p  # the torn final line is the normal shape of a killed scan


def test_the_cli_exposes_every_command():
    from acr.cli import app
    out = runner.invoke(app, ["derive", "--help"]).output
    for cmd in ("terms", "policy", "show-curve", "groups"):
        assert cmd in out


@pytest.mark.parametrize("cmd", ["terms", "policy", "show-curve", "groups"])
def test_no_command_writes_anything_without_an_explicit_flag(cmd, tmp_path, labels_path,
                                                             spec_path, bm):
    args = _args(labels_path, spec_path, tmp_path)
    if cmd in ("policy", "groups"):
        args = [a for a in args if a not in ("--cache", str(tmp_path))]
    before = digest(tmp_path)
    r = runner.invoke(D.derive_app, [cmd] + args)
    assert r.exit_code == 0, r.output
    assert digest(tmp_path) == before, f"`derive {cmd}` wrote something"


@pytest.mark.parametrize("drop", ["--max-extra-docs-per-answer", "--high-yield-rate",
                                  "--min-patients-proposing", "--share-asset-jaccard"])
def test_the_cli_refuses_to_run_with_a_threshold_left_out(drop, tmp_path, labels_path,
                                                          spec_path, bm):
    args = _args(labels_path, spec_path, tmp_path)
    i = args.index(drop)
    r = runner.invoke(D.derive_app, ["terms"] + args[:i] + args[i + 2:])
    assert r.exit_code != 0


def test_show_curve_prints_the_rejected_rows_too(tmp_path, labels_path, spec_path, bm):
    r = runner.invoke(D.derive_app, ["show-curve"] + _args(labels_path, spec_path, tmp_path))
    assert r.exit_code == 0, r.output
    assert "KEEP" in r.output and "docs/ans" in r.output and "cut at 10.0 docs/answer" in r.output


def test_policy_emits_a_proposal_only_when_asked(tmp_path, labels_path, spec_path):
    args = [a for a in _args(labels_path, spec_path, tmp_path) if a not in
            ("--cache", str(tmp_path))]
    out = tmp_path / "proposals"
    r = runner.invoke(D.derive_app, ["policy"] + args + ["--emit-proposal", "--out-dir", str(out)])
    assert r.exit_code == 0, r.output
    assert list(out.glob("*_policy.yaml"))
    assert json.loads(r.output)["proposal_path"].endswith("_policy.yaml")


def _groups_args(labels_path, spec_path, tmp_path):
    return [a for a in _args(labels_path, spec_path, tmp_path) if a not in
            ("--cache", str(tmp_path))]


def test_groups_prints_the_whole_matrix_beside_the_suggestion(tmp_path, labels_path, spec_path):
    """The suggestion alone is a number to be taken on trust. Printed beside every pair and
    its coefficient, it is a decision the reader can second-guess without rerunning."""
    r = runner.invoke(D.derive_app, ["groups"] + _groups_args(labels_path, spec_path, tmp_path))
    assert r.exit_code == 0, r.output
    assert "jaccard" in r.output and r.output.count("MERGE") == 1
    for a, b in ((HISTO, BEHAV), (HISTO, SITE), (BEHAV, SITE)):
        assert f"{a}" in r.output and f"{b}" in r.output
    assert "0.714" in r.output, "the rejected pair's coefficient must be on the page"
    assert "cut at jaccard >= 0.8 -> 2 asset(s)" in r.output
    assert "PROPOSED, not in effect" in r.output


def test_groups_emits_a_proposal_only_when_asked(tmp_path, labels_path, spec_path):
    out = tmp_path / "proposals"
    r = runner.invoke(D.derive_app, ["groups"] + _groups_args(labels_path, spec_path, tmp_path)
                      + ["--emit-proposal", "--out-dir", str(out)])
    assert r.exit_code == 0, r.output
    (written,) = list(out.glob("*_field_groups.yaml"))
    body = yaml.safe_load(written.read_text())
    assert body["kind"] == "acr.derive.grouping_proposal/1"
    assert body["spec_id"] == "SYNTH.998.derive_fixture"
    assert body["groups"] == [[HISTO, BEHAV], [SITE]]
