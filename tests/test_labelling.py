"""Acceptance tests for the full-scan labeller. EVERY TEST RUNS OFFLINE, ON FIXTURES HERE.

No test reads the real corpus, no test reads `specs/*.yaml`, and no test makes a model call:
`no_network` is autouse and detonates `litellm.completion`, so a test that quietly acquires a
real client fails loudly instead of billing somebody.

The synthetic patients are named SYNL0x. The wording and dates were written for this file;
`tests/test_no_phi_in_tree.py` is the guard that keeps it that way.

What is asserted, in descending order of how expensive it would be to get wrong:

  1. THE MODULE KNOWS NOTHING ABOUT THE SUBJECT MATTER. Everything the model is told about what
     it is looking for comes from a spec. Two fixture requirements are used throughout — one
     medical, one about apartment leases — and the ONE clinical-vocabulary test is run against
     the lease requirement, so a clinical word appearing anywhere in the prompt-building path
     fails it. The previous version of this module hardcoded a disease and could not be pointed
     anywhere else; that is the defect these tests exist to keep out.
  2. A label is conditioned on the requirement and says so, and two requirements over one corpus
     cannot land in one file.
  3. STANDING IS ANSWERED PER FIELD OF THE REQUIREMENT, never once for the note: a document can
     establish one field and be unable to settle another, and the collapse of those two into one
     verdict is a mis-coding this project has already shipped once. Every field must come back —
     a missing one is a refused reply, not a "neither" — while the QUOTE stays at note level,
     because one span per field is the extraction that was deleted from this module. The axis
     grep cannot see (the same sentence establishing in one document and establishing nothing in
     a copy) is preserved, per field.
  4. Question 2's terms are checked against the note, so a proposed term that is not in the note
     is dropped and counted rather than becoming a needle that matches nothing; and the cap and
     the length floor that bound the question have no defaults anywhere.
  5. The scan resumes, and the cost ceiling is unforgettable and holds.
  6. The after-the-fact audit works with an answer key it has never heard of, in a shape this
     module does not define.
"""
from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import fields as dc_fields
from pathlib import Path
from types import SimpleNamespace

import pytest

from acr import labelling as L

# ============================================================================
# TWO REQUIREMENTS, NEITHER OF THEM THIS MODULE'S BUSINESS
# ============================================================================

def spec(spec_id, question, fields, evidence_rules):
    """The smallest thing `Requirement.from_spec` accepts: the four attributes it reads.

    Deliberately NOT an `ExtractionSpec`. This module must work for a requirement nobody has
    written a spec file for yet, so the coupling under test is four attribute names.
    """
    return SimpleNamespace(
        spec_id=spec_id, question=question, evidence_rules=evidence_rules,
        fields=[SimpleNamespace(name=n, description=d) for n, d in fields])


#: A clinical requirement, shaped like the one this module used to have welded into it.
#:
#: Its evidence rules are deliberately ASYMMETRIC ACROSS ITS FIELDS, because that asymmetry is
#: what the per-field verdict exists to carry: cross-sectional imaging is one of the richest
#: sources in a record for where something is and can never say what its cells are. A spec whose
#: rules said the same thing about every field would let a single note-level verdict pass these
#: tests, and the collapse of those two answers into one is the shipped mis-coding this design
#: is a response to.
MED_SPEC = spec(
    "SYN.400.site_histology",
    "For the tumour being reported, what is the primary site and the histology?",
    [("primary_site", "the anatomical site of origin"),
     ("histology", "the cell type as coded")],
    {"counts_as_evidence": ["A pathology report's FINAL DIAGNOSIS, for either field.",
                            "A cytology report, for the cell type only.",
                            "Cross-sectional imaging, for the site only."],
     "does_not_count": [("Imaging, for the cell type: it can localise a mass, it cannot say what "
                         "the cells are."),
                        "A problem-list entry restating a diagnosis established elsewhere."]})

#: A requirement with no medicine in it anywhere. Everything that must generalise is tested
#: against THIS one, so anything clinical surviving in the module fails rather than passes.
LEASE_SPEC = spec(
    "SYN.LEASE.rent_arrears",
    "For the tenancy being reviewed, how many months of rent are in arrears, and on what date "
    "did the arrears begin?",
    [("months_in_arrears", "whole months unpaid at the review date"),
     ("arrears_start", "the first month with a shortfall")],
    {"counts_as_evidence": ["A signed ledger statement issued by the managing agent."],
     "does_not_count": ["A tenant's or landlord's assertion in correspondence."]})

MED = L.Requirement.from_spec(MED_SPEC)
LEASE = L.Requirement.from_spec(LEASE_SPEC)

#: patient -> {filename stem: text}. Fabricated prose, none of it from any chart.
CORPUS_TEXT = {
    "SYNL01": {
        "Surgical-Pathology-Report_2021-03-02":
            "SPECIMEN: right upper lobe, wedge resection.\n"
            "FINAL DIAGNOSIS: Invasive adenocarcinoma arising in the right upper lobe.\n",
        "Progress-Note_2021-04-01":
            "ASSESSMENT: Known adenocarcinoma of the lung, status post wedge resection.\n",
        "Chest-CT-W-Contr_2021-02-10":
            "IMPRESSION: Spiculated 2.1 cm mass in the right upper lobe.\n",
    },
    # The motivating failure in miniature: the answer is stated ONLY in a progress note, in
    # wording containing none of the shipped keywords. Grep cannot find this note.
    "SYNL02": {
        "Progress-Note_2022-05-11":
            "ASSESSMENT: Newly diagnosed malignant pleural mesothelioma.\n",
        "Med-Reconciliation_2022-05-12": "Lisinopril 10 mg daily. No changes today.\n",
    },
    "SYNL03": {
        "Cytology-Report_2023-08-19":
            "FINAL DIAGNOSIS: Squamous cell carcinoma, in situ, no stromal invasion.\n",
    },
    # The drug-as-proxy case, which is the whole argument for asking a MODEL holding the
    # requirement rather than counting words: "etoposide" names no disease, appears on no
    # keyword list anyone has written by hand, and indicates this answer anyway.
    "SYNL04": {
        "Oncology-Consult_2024-01-15":
            "IMPRESSION: extensive-stage small cell lung cancer (SCLC).\n"
            "PLAN: begin carboplatin and etoposide.\n",
    },
}
N_NOTES = sum(len(v) for v in CORPUS_TEXT.values())
ALL_PATIENTS = sorted(CORPUS_TEXT)

#: The shipped list's weakness, copied in shape only: no "mesothelioma", no "lung cancer".
SHIPPED_KEYWORDS = ("pathology", "biopsy", "final diagnosis", "specimen", "carcinoma")

#: Bounds on question 2, chosen HERE, for this fixture, and nowhere in the module. The real ones
#: get chosen when the scan is first run for real.
TERMS = L.TermConfig(max_terms_per_note=4, min_term_chars=3)

#: An answer key in a shape this module has never heard of, for the audit at the end. Not a
#: registry export, not a CSV, no column names: a mapping to a tuple of surface forms. It holds
#: ONE of the requirement's two fields — the cell type — which is the ordinary case and the
#: reason `audit_relevance` takes the field it is scoring.
ANSWER_KEY = {
    "SYNL01": ("invasive adenocarcinoma",),
    "SYNL02": ("malignant pleural mesothelioma",),
    "SYNL03": ("squamous cell carcinoma",),
    "SYNL04": ("small cell lung cancer",),
}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Any real completion call fails the test that made it, by name."""
    import litellm

    def explode(*a, **k):
        raise AssertionError("a test called litellm.completion; this suite runs offline and free")

    monkeypatch.setattr(litellm, "completion", explode)


@pytest.fixture
def corpus(tmp_path: Path):
    from acr.corpus import Corpus

    for pid, docs in CORPUS_TEXT.items():
        (tmp_path / "patients" / pid).mkdir(parents=True)
        for stem, text in docs.items():
            (tmp_path / "patients" / pid / f"{stem}.txt").write_text(text, encoding="utf-8")
    return Corpus(tmp_path / "patients")


@pytest.fixture
def store(tmp_path: Path) -> L.LabelStore:
    return L.LabelStore(tmp_path / "devlabels", model="stub/gpt-5.6-luna", requirement=MED,
                        terms=TERMS)


#: What a model holding the requirement would offer, if it were reading these notes. Note what a
#: word count could never produce: "SCLC" is rare, "etoposide" is a drug, and neither is a word
#: anybody writing a keyword list from imagination puts on it.
TERM_SURFACES = (
    ("malignant pleural mesothelioma", "names_the_answer"),
    ("small cell lung cancer", "names_the_answer"),
    ("invasive adenocarcinoma", "names_the_answer"),
    ("squamous cell carcinoma", "names_the_answer"),
    ("SCLC", "names_the_answer"),
    ("etoposide", "other"),
    ("wedge resection", "names_the_document"),
    ("FINAL DIAGNOSIS", "names_the_section"),
)


def every_field(verdict: str, requirement: L.Requirement = MED) -> dict[str, str]:
    """One standing, repeated over every field — for the tests where the split is not the point.

    Written out rather than defaulted inside the module: there is no such thing as "the verdict"
    for a note, so a test that wants the same one everywhere has to say every field's name.
    """
    return {name: verdict for name in requirement.field_names}


def reply_json(*, verdicts: dict[str, str], quote: str = "", terms=()) -> str:
    """The exact shape of the two-question contract, so a test can vary one answer at a time.

    Note what has no per-field form here and must not grow one: the quote. ONE span for the
    document, however many fields it speaks to.
    """
    return json.dumps({"admissibility": {"verdicts": verdicts, "quote": quote},
                       "retrieval_terms": [{"term": t, "reason": r} for t, r in terms]})


def reply_for(blob: str) -> str:
    """A plausible PER-FIELD reading, derived only from the DOCUMENT — never from ANSWER_KEY.

    The stub is held to the rule the real model is: no test may prove the pipeline works using
    an oracle the pipeline would not have had. That covers question 2 too — every term it
    proposes is copied out of the note in front of it.

    It reads the document type as well as the text, because the two fields of MED_SPEC part
    company on exactly that: the same words in an imaging impression and in a rendered final
    diagnosis have different standing, and different standing PER FIELD.
    """
    doc_type = blob.split("DOCUMENT TYPE:")[1].split("DOCUMENT DATE:")[0].strip()
    text = blob.split("--- BEGIN DOCUMENT ---")[-1].split("--- END DOCUMENT ---")[0].strip("\n")
    low = text.lower()
    first = text.strip().splitlines()[0] if text.strip() else ""
    if "final diagnosis" in low:
        # The document that rendered it. A cytology report speaks for the cell type only — this
        # fixture's own evidence rules say so — so it cannot settle where the thing came from.
        verdicts = {"histology": "can_establish",
                    "primary_site": "merely_mentions" if "Cytology" in doc_type
                                    else "can_establish"}
    elif "-CT-" in doc_type:
        # THE MOTIVATING CASE, in one line: richest source in the chart for one field, mute on
        # the other. No single verdict for this note is true.
        verdicts = {"primary_site": "can_establish", "histology": "neither"}
    elif any(s.lower() in low for s, _ in TERM_SURFACES):
        verdicts = every_field("merely_mentions")
    else:
        verdicts = every_field("neither")
    bears = any(v != "neither" for v in verdicts.values())
    return reply_json(verdicts=verdicts, quote=first if bears else "",
                      terms=[(t, r) for t, r in TERM_SURFACES if t.lower() in low])


class StubClient:
    """Everything the runner needs of a model, plus a record of every character it was shown."""

    def __init__(self, *, behaviour=None, delay=0.0, prompt_tokens=1000, completion_tokens=100):
        self.prompts: list[str] = []
        self.calls = self.max_concurrent = self._live = 0
        self._lock = threading.Lock()
        self._behaviour, self._delay = behaviour, delay
        self._pt, self._ct = prompt_tokens, completion_tokens

    def chat(self, messages, tools=None):
        blob = "\n".join(m["content"] for m in messages)
        with self._lock:
            self.calls += 1
            self._live += 1
            self.max_concurrent = max(self.max_concurrent, self._live)
            self.prompts.append(blob)
            n = self.calls
        try:
            time.sleep(self._delay)
            if self._behaviour is not None:
                out = self._behaviour(n, blob)
                if isinstance(out, BaseException):
                    raise out
                content = out
            else:
                content = reply_for(blob)
            return SimpleNamespace(content=content, prompt_tokens=self._pt,
                                   completion_tokens=self._ct)
        finally:
            with self._lock:
                self._live -= 1


def make_runner(corpus, store, client, **cfg) -> L.FullScanRunner:
    return L.FullScanRunner(corpus=corpus, store=store, client=client, config=L.ScanConfig(
        **{"max_usd": 100.0, "requirement": store.requirement, "terms": store.terms,
           "concurrency": 2, **cfg}))


def scan(corpus, store, client=None, **cfg) -> dict:
    make_runner(corpus, store, client or StubClient(), **cfg).run(ALL_PATIENTS)
    return {key[1]: label for key, label in store.load().items()}


def a_note(text: str = "FINAL DIAGNOSIS: Invasive adenocarcinoma.") -> L.NoteForReading:
    return L.NoteForReading("SYNL01", "Path_2021-03-02", "Pathology", "2021-03-02", text)


# ============================================================================
# 1. THE MODULE KNOWS NOTHING ABOUT THE SUBJECT MATTER
# ============================================================================

#: Words that would betray a hardcoded requirement. The module was welded to one variable and
#: every one of these appeared in its prompt; if any comes back, it was welded again.
CLINICAL_WORDS = ("cancer", "tumour", "tumor", "histolog", "carcinoma", "patholog", "cytolog",
                  "icd", "lobe", "registry", "oncolog", "biopsy", "specimen", "malignan",
                  "abstractor", "diagnos", "disease", "clinician", "anatom", "drug")


def test_no_clinical_vocabulary_survives_anywhere_in_the_prompt_building_path():
    """The regression guard for the defect that motivated this rewrite.

    A prompt is built for a requirement about rent arrears over a note about rent arrears. If a
    single clinical word reaches the model, it did not come from the spec and it did not come
    from the note — it came from this module, and it is false for every requirement but one.
    """
    note = L.NoteForReading("SYNL99", "Ledger_2024-02-01", "Ledger-Statement", "2024-02-01",
                            "Managing agent ledger: February shortfall of 420.00, third month.")
    blob = "\n".join(m["content"] for m in L.build_note_prompt(note, requirement=LEASE,
                                                               terms=TERMS)).lower()
    assert not [w for w in CLINICAL_WORDS if w in blob]
    # And the constants themselves, so a word parked in a template that this fixture happens not
    # to render is caught too.
    static = " ".join([L.SYSTEM_PROMPT, L.NOTE_PROMPT_TEMPLATE, " ".join(L.TERM_REASONS),
                       " ".join(L.ADMISSIBILITY_VERDICTS)]).lower()
    assert not [w for w in CLINICAL_WORDS if w in static]


def test_the_source_of_the_prompt_building_path_carries_no_clinical_content_in_code():
    """Not the rendered prompt — the SOURCE. A clinical word in a default argument, a lookup
    table or a fallback string is one edit away from the prompt and grep is how it is found."""
    src = Path(L.__file__).read_text(encoding="utf-8")
    body = src.split("# THE PROMPT")[1].split("# AN AFTER-THE-FACT AUDIT")[0]
    offenders = []
    for i, line in enumerate(body.splitlines(), 1):
        code = line.split("#", 1)[0]
        if code.strip().startswith(("#", '"""', "'''")):
            continue
        offenders += [(i, w) for w in CLINICAL_WORDS if w in code.lower()]
    assert offenders == []


def test_the_question_the_fields_and_the_three_classes_all_come_from_the_spec():
    rendered = MED.render()
    assert MED_SPEC.question in rendered
    assert all(name in rendered for name in ("primary_site", "histology"))
    assert MED.field_names == ("primary_site", "histology")
    # The clause that DEFINES can_establish is the spec's own, verbatim, under the spec's own
    # name for it — not a sentence this module wrote about what evidence is.
    assert "counts_as_evidence" in rendered
    assert "A pathology report's FINAL DIAGNOSIS, for either field." in rendered
    assert "does_not_count" in rendered


def test_the_same_module_labels_a_requirement_with_no_medicine_in_it():
    """Retargeting is supplying a different spec, and nothing else."""
    rendered = LEASE.render()
    assert "months_in_arrears" in rendered and "managing agent" in rendered
    assert L.build_note_prompt(a_note(), requirement=LEASE, terms=TERMS) != \
        L.build_note_prompt(a_note(), requirement=MED, terms=TERMS)


def test_a_spec_that_does_not_say_what_counts_as_evidence_cannot_be_labelled():
    """The three classes are DEFINED from `evidence_rules`. With it empty, `can_establish` would
    mean whatever the model felt like, and a default supplied here would be this module inventing
    a clinical rule again — which is the whole defect."""
    for rules in ({}, {"counts_as_evidence": []}, {"counts_as_evidence": ""}):
        with pytest.raises(L.NotALabellableSpecError):
            L.Requirement.from_spec(spec("S1", "a question", [("f", "d")], rules))


def test_a_spec_with_no_question_or_no_id_cannot_be_labelled():
    rules = {"counts_as_evidence": ["something"]}
    with pytest.raises(L.NotALabellableSpecError):
        L.Requirement.from_spec(spec("S1", "  ", [("f", "d")], rules))
    with pytest.raises(L.NotALabellableSpecError):
        L.Requirement.from_spec(spec("", "a question", [("f", "d")], rules))


def test_evidence_rules_are_rendered_whatever_shape_the_spec_wrote_them_in():
    """Specs write these as strings, lists and nested mappings. Normalising a clinical rule into
    a shape its author did not choose is not this module's call, so every shape survives."""
    r = L.Requirement.from_spec(spec(
        "S1", "q", [("f", "d")], {"plain": "one statement",
                                 "nested": {"per_field": ["a", "b"]},
                                 "listed": ["c"]}))
    rendered = r.render()
    assert all(s in rendered for s in ("one statement", "per_field: a", "per_field: b", "c"))


# ============================================================================
# 2. A LABEL IS CONDITIONED ON THE REQUIREMENT AND SAYS SO
# ============================================================================

def test_every_label_carries_the_spec_id_and_the_prompt_hash(corpus, store):
    labels = scan(corpus, store)
    assert labels and all(lab.spec_id == MED.spec_id for lab in labels.values())
    assert all(lab.prompt_hash == L.prompt_hash(MED, TERMS) for lab in labels.values())
    assert all(lab.model == "stub/gpt-5.6-luna" for lab in labels.values())


def test_two_requirements_over_one_corpus_cannot_share_a_labelling(tmp_path, monkeypatch):
    """Change the requirement and both answers are about a different question. A file mixing
    them has nothing on any row to say which question it answered."""
    a = L.LabelStore(tmp_path, model="m", requirement=MED, terms=TERMS)
    assert L.LabelStore(tmp_path, model="m", requirement=LEASE, terms=TERMS).path != a.path
    # Same spec id, reworded evidence rules: still a different question, still a different file.
    reworded = L.Requirement.from_spec(spec(
        MED.spec_id, MED_SPEC.question, [("primary_site", "the anatomical site of origin"),
                                         ("histology", "the cell type as coded")],
        {"counts_as_evidence": ["A pathology report's FINAL DIAGNOSIS only."]}))
    assert L.LabelStore(tmp_path, model="m", requirement=reworded, terms=TERMS).path != a.path
    assert L.prompt_hash(reworded, TERMS) != L.prompt_hash(MED, TERMS)
    monkeypatch.setattr(L, "PROMPT_VERSION", "labelling/99")
    assert L.LabelStore(tmp_path, model="m", requirement=MED, terms=TERMS).path != a.path
    assert json.loads((a.dir / "manifest.json").read_text())["spec_id"] == MED.spec_id


def test_a_different_model_or_term_bound_cannot_land_in_the_same_file(tmp_path):
    a = L.LabelStore(tmp_path, model="stub/gpt-5.6-luna", requirement=MED, terms=TERMS)
    assert L.LabelStore(tmp_path, model="stub/cheaper", requirement=MED, terms=TERMS).path != a.path
    # The bounds on question 2 are rendered INTO the prompt, so they are prompt wording: a scan
    # capped at 4 terms and one capped at 9 must not append into one file and one manifest.
    for other in (L.TermConfig(9, TERMS.min_term_chars), L.TermConfig(TERMS.max_terms_per_note, 5)):
        assert L.LabelStore(tmp_path, model="stub/gpt-5.6-luna", requirement=MED,
                            terms=other).path != a.path


def test_a_store_and_a_scan_that_disagree_cannot_be_combined(corpus, store):
    """Two accounts of one number again: the file's manifest would name a requirement or bounds
    the run did not use, and no row would say so."""
    for bad in (L.ScanConfig(max_usd=1.0, requirement=LEASE, terms=TERMS),
                L.ScanConfig(max_usd=1.0, requirement=MED, terms=L.TermConfig(9, 3))):
        with pytest.raises(L.LabellingError):
            L.FullScanRunner(corpus=corpus, store=store, client=StubClient(), config=bad)


def test_the_prompt_builder_refuses_anything_that_merely_looks_like_a_note():
    """A duck-typed lookalike is how a field nobody reviewed reaches a prompt: `NoteForReading`
    is frozen and slotted and has exactly five fields, and a SimpleNamespace has as many as
    somebody felt like. The builder takes the type, not the shape."""
    with pytest.raises(L.LabellingError):
        L.build_note_prompt(SimpleNamespace(doc_type="Ledger", date="2024-02-01", text="x"),
                            requirement=LEASE, terms=TERMS)
    with pytest.raises(L.LabellingError):
        L.build_note_prompt(a_note(), requirement=SimpleNamespace(render=lambda: "anything"),
                            terms=TERMS)


def test_a_note_for_reading_has_nowhere_to_put_anything_else():
    note = a_note()
    with pytest.raises((AttributeError, TypeError)):
        note.the_answer = "8140"  # type: ignore[attr-defined]
    assert not hasattr(note, "__dict__")


# ============================================================================
# 3. STANDING, PER FIELD — the axis grep cannot see, on the axis one verdict cannot say
# ============================================================================

def test_a_document_can_establish_one_field_and_be_mute_on_another(corpus, store):
    """THE REASON THIS QUESTION IS ASKED PER FIELD, and a mis-coding this project shipped once.

    Cross-sectional imaging is among the richest sources in a chart for where something is, and
    it can never say what its cells are. Under one verdict per note this document is either an
    establishing document — and every downstream count then believes it settled the cell type —
    or it is not, and the best source for the site drops out of the policy entirely. Both are
    wrong, which is what tells you the question was the wrong shape.
    """
    adm = scan(corpus, store)["Chest-CT-W-Contr_2021-02-10"].admissibility
    assert adm.verdict_for("primary_site") == "can_establish"
    assert adm.verdict_for("histology") == "neither"
    assert adm.fields_where("can_establish") == ("primary_site",)
    # And a caller that has not learned to ask per field still gets the honest weaker claim.
    assert adm.verdict == "can_establish" and adm.bears_on_question


def test_the_same_sentence_establishes_in_one_document_and_not_in_its_copy(corpus, store):
    """The distinction the whole question exists for, and the reason a model is being paid.

    Both notes below say the patient has an adenocarcinoma. One is where that was decided; the
    other copied it forward. No keyword, count or filename separates them.
    """
    labels = scan(corpus, store)
    rendered = labels["Surgical-Pathology-Report_2021-03-02"].admissibility
    copied = labels["Progress-Note_2021-04-01"].admissibility
    for name in MED.field_names:
        assert rendered.verdict_for(name) == "can_establish"
        assert copied.verdict_for(name) == "merely_mentions"
    assert copied.bears_on_question
    none = labels["Med-Reconciliation_2022-05-12"].admissibility
    assert none.verdicts == every_field("neither") and not none.bears_on_question


def test_every_field_of_the_requirement_is_answered_on_every_label(corpus, store):
    """A label that answers three of a spec's fields is a label nothing can aggregate per field,
    and its silence on the fourth is unreadable: no answer and 'no' look identical."""
    labels = scan(corpus, store)
    assert labels and all(set(lab.admissibility.verdicts) == set(MED.field_names)
                          for lab in labels.values())


def test_a_missing_field_is_a_contract_violation_and_never_a_neither():
    """The single most tempting shortcut here, and the one that would quietly restore the bug:
    filling an unanswered field in with the cheapest verdict, in code, where nothing says so."""
    with pytest.raises(L.PromptContractError) as exc:
        L.parse_label_response(reply_json(verdicts={"primary_site": "can_establish"},
                                          quote="anything"), requirement=MED, terms=TERMS)
    assert "histology" in str(exc.value)
    # A reply that answered a field this requirement does not declare was not shaped by the list
    # it was shown, so the verdicts that DO line up are not evidence either.
    with pytest.raises(L.PromptContractError):
        L.parse_label_response(reply_json(verdicts=every_field("neither") | {"stage": "neither"}),
                               requirement=MED, terms=TERMS)


def test_one_verdict_for_the_whole_note_is_refused_outright():
    """The previous contract. Accepting it here — even by spreading it over the fields — would
    manufacture standing nobody read, on exactly the axis the fields disagree about."""
    for body in ({"admissibility": {"verdict": "can_establish", "quote": "q"},
                  "retrieval_terms": []},
                 {"admissibility": {"verdicts": "can_establish", "quote": "q"},
                  "retrieval_terms": []}):
        with pytest.raises(L.PromptContractError):
            L.parse_label_response(json.dumps(body), requirement=MED, terms=TERMS)


def test_the_parser_cannot_check_a_reply_without_the_field_list_it_was_asked_about():
    with pytest.raises(TypeError):
        L.parse_label_response(reply_json(verdicts=every_field("neither")),
                               terms=TERMS)  # type: ignore[call-arg]


def test_a_requirement_with_no_fields_or_a_repeated_field_cannot_be_labelled():
    """Standing is answered per field, so a requirement with nowhere to put an answer is not a
    requirement this module can read a note against."""
    rules = {"counts_as_evidence": ["something"]}
    for flds in ([], [("", "no name")], [("f", "d"), ("f", "again")]):
        with pytest.raises(L.NotALabellableSpecError):
            L.Requirement.from_spec(spec("S1", "a question", flds, rules))


def test_the_quote_stays_at_note_level_and_there_is_nowhere_to_put_one_per_field(corpus, store):
    """ONE span for the document, however many fields it speaks to. A quote per field is the
    per-field extraction that was deleted from this module, arriving through the evidence slot;
    the record has no room for it and the prompt does not ask for it."""
    adm = scan(corpus, store)["Surgical-Pathology-Report_2021-03-02"].admissibility
    assert isinstance(adm.quote, str) and adm.quote and adm.quote_verified
    assert L.verify_quote(adm.quote, CORPUS_TEXT["SYNL01"]["Surgical-Pathology-Report_2021-03-02"])
    assert [f.name for f in dc_fields(L.Admissibility)] == ["verdicts", "quote", "quote_verified"]
    prompt = "\n".join(m["content"] for m in
                       L.build_note_prompt(a_note(), requirement=MED, terms=TERMS))
    assert prompt.count('"quote"') == 1


def test_a_note_bearing_on_nothing_carries_no_quote_to_verify(corpus, store):
    none = scan(corpus, store)["Med-Reconciliation_2022-05-12"].admissibility
    assert none.quote == "" and not none.quote_verified


def test_a_fabricated_quote_is_recorded_as_unverified_not_silently_kept(corpus, store):
    invented = reply_json(verdicts=every_field("can_establish"),
                          quote="the reader was quite certain of this", terms=[])
    runner = make_runner(corpus, store, StubClient(behaviour=lambda n, b: invented))
    adm = runner.label_note(a_note()).admissibility
    assert adm.verdict == "can_establish" and adm.quote and not adm.quote_verified


def test_one_field_bearing_on_the_question_is_enough_to_require_the_quote():
    """The quote justifies whichever fields were not "neither" — so it is owed as soon as ONE
    of them is, and a note bearing on nothing must not carry one."""
    for verdict in ("can_establish", "merely_mentions"):
        with pytest.raises(L.PromptContractError):
            L.parse_label_response(
                reply_json(verdicts={"primary_site": verdict, "histology": "neither"}, quote=""),
                requirement=MED, terms=TERMS)
    reply = L.parse_label_response(
        reply_json(verdicts=every_field("neither"), quote="something"),
        requirement=MED, terms=TERMS)
    assert reply.admissibility.verdicts == every_field("neither")
    assert reply.admissibility.quote == ""


def test_an_unknown_or_missing_verdict_is_refused_rather_than_folded_into_neither():
    for body in ({"admissibility": {"verdicts": {"primary_site": "probably",
                                                 "histology": "neither"}},
                  "retrieval_terms": []},
                 {"retrieval_terms": []}):
        with pytest.raises(L.PromptContractError):
            L.parse_label_response(json.dumps(body), requirement=MED, terms=TERMS)
    # And nothing can hand-build a record out of one either.
    with pytest.raises(L.LabelShapeError):
        L.Admissibility({"primary_site": "probably"})


def test_quote_verification_tolerates_wrapping_but_not_different_words():
    assert L.verify_quote("Invasive\n  adenocarcinoma", "final: invasive adenocarcinoma here")
    assert not L.verify_quote("invasive carcinoma", "final: invasive adenocarcinoma here")
    assert not L.verify_quote("   ", "anything")


def test_a_reply_short_of_a_field_becomes_a_countable_error_not_a_quiet_label(corpus, store):
    """End to end: the refusal reaches the row, names the field, and costs what the call cost.

    This is what makes a model that cannot hold nine fields at once VISIBLE — a failure rate on
    the field contract is a number somebody can act on, where a silently completed "neither" is
    a measurement nobody can distinguish from a reading."""
    short = reply_json(verdicts={"histology": "can_establish"}, quote="anything")
    runner = make_runner(corpus, store, StubClient(behaviour=lambda n, b: short))
    label = runner.label_note(a_note())
    assert not label.ok and "primary_site" in label.error
    assert label.admissibility.verdicts == {} and label.cost_usd > 0
    store.append(label)  # errored labels are exempt from the field check, and countable
    assert not store.load()[("SYNL01", "Path_2021-03-02")].ok


def test_a_model_failure_becomes_a_label_carrying_its_error_not_an_exception(corpus, store):
    client = StubClient(behaviour=lambda n, b: RuntimeError("upstream 503"))
    label = make_runner(corpus, store, client).label_note(a_note())
    assert not label.ok and "upstream 503" in label.error and client.calls == 1


# ============================================================================
# 4. RETRIEVAL TERMS — verified in code, bounded with no defaults
# ============================================================================

def test_a_proposed_term_the_note_does_not_contain_is_dropped_and_counted(corpus, store):
    """A hallucinated term is a retrieval instruction that matches nothing. Stored, it would go
    into a keyword list, measure zero recall, and read as evidence that the question fails."""
    reply = reply_json(verdicts=every_field("neither"),
                       terms=[("ZZNOTHEREZZ", "names_the_answer"),
                              ("adenocarcinoma", "names_the_answer")])
    runner = make_runner(corpus, store, StubClient(behaviour=lambda n, b: reply))
    label = runner.label_note(a_note("FINAL DIAGNOSIS: Invasive adenocarcinoma."))
    assert [t.term for t in label.retrieval_terms] == ["adenocarcinoma"]
    assert label.n_terms_proposed == 2 and label.n_terms_hallucinated == 1


def test_the_terms_that_no_word_count_would_ever_surface_are_the_ones_kept(corpus, store):
    """The measured argument for the question: a rare abbreviation and a drug name, neither of
    which any frequency ranking would rank and neither of which is on the shipped list."""
    got = {t.term for t in scan(corpus, store)["Oncology-Consult_2024-01-15"].retrieval_terms}
    assert {"SCLC", "etoposide"} <= got
    assert not [k for k in SHIPPED_KEYWORDS if k in "\n".join(got).lower()]


def test_the_cap_bites_on_what_was_proposed_and_is_never_backfilled(corpus, store):
    """Backfilling would pay padding a dividend: offer forty, lose thirty-five, still land a full
    list. The cap exists so that proposing more costs the model its best slots."""
    proposed = [("adenocarcinoma", "names_the_answer"), ("ZZNOTHEREZZ", "names_the_answer"),
                ("FINAL DIAGNOSIS", "names_the_section"), ("Invasive", "names_the_answer"),
                ("resection", "names_the_document")]
    tight = L.TermConfig(max_terms_per_note=3, min_term_chars=3)
    reply = L.parse_label_response(reply_json(verdicts=every_field("neither"), terms=proposed),
                                   requirement=MED, terms=tight)
    assert len(reply.terms) == 3 and reply.n_terms_proposed == 5
    kept, dropped = L.verify_terms(reply.terms, "FINAL DIAGNOSIS: Invasive adenocarcinoma.")
    assert len(kept) == 2 and dropped == 1  # the fourth slot is NOT promoted into the gap


def test_a_term_below_the_length_floor_or_with_an_unknown_reason_never_lands():
    """A one-character term verifies as present in every note ever written — the exact failure
    the question exists to avoid, arriving through the verifier. And an unrecognised reason class
    is dropped rather than folded into "other", which would be a claim the model did not make."""
    reply = L.parse_label_response(reply_json(
        verdicts=every_field("neither"),
        terms=[("ad", "names_the_answer"), ("nodule", "names_a_measurement"),
               ("etoposide", "other")]), requirement=MED, terms=TERMS)
    assert [t.term for t in reply.terms] == ["etoposide"] and reply.n_terms_proposed == 3


def test_the_reason_classes_are_the_ones_derive_aggregates():
    """The two modules are coupled by this vocabulary and by nothing else. If they drift, every
    term the labeller emits lands in `Aggregate.unknown_reason_classes` and the derivation is
    silently a derivation of nothing."""
    from acr import derive as D

    assert set(L.TERM_REASONS) == set(D.REASON_CLASSES)


def test_the_bounds_on_question_two_have_no_defaults_anywhere():
    with pytest.raises(TypeError):
        L.TermConfig()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        L.TermConfig(max_terms_per_note=0, min_term_chars=3)
    with pytest.raises(TypeError):
        L.prompt_hash(MED)  # type: ignore[call-arg]


# ============================================================================
# 5. RESUME, CONCURRENCY, CEILING, PHI
# ============================================================================

def test_a_second_run_calls_the_model_for_nothing(corpus, store):
    first, second = StubClient(), StubClient()
    make_runner(corpus, store, first).run(ALL_PATIENTS)
    report = make_runner(corpus, store, second).run(ALL_PATIENTS)
    assert first.calls == N_NOTES and second.calls == 0
    assert report.n_already_labelled == N_NOTES and report.n_written == 0


def test_a_scan_killed_mid_flight_keeps_what_it_wrote_and_resumes_onto_it(corpus, tmp_path, store):
    """A torn final line is the normal shape of a killed job, not a reason to lose the run."""
    make_runner(corpus, store, StubClient(), max_usd=L.cost_usd(1000, 100) * 2.5,
                concurrency=1).run(ALL_PATIENTS)
    with open(store.path, "a", encoding="utf-8") as fh:
        fh.write('{"patient_id": "SYNL0')
    resumed = L.LabelStore(tmp_path / "devlabels", model=store.model,
                           requirement=MED, terms=store.terms)  # i.e. a new process
    assert len(resumed.load()) == 3
    rest = StubClient()
    make_runner(corpus, resumed, rest).run(ALL_PATIENTS)
    assert rest.calls == N_NOTES - 3 and len(resumed.load()) == N_NOTES


def test_the_same_key_written_twice_collapses_to_the_last_write(store):
    store.append(L.NoteLabel("SYNL01", "n1", error="first"))
    store.append(L.NoteLabel("SYNL01", "n1", doc_type="Pathology",
                             admissibility=L.Admissibility(every_field("neither"))))
    assert len(store.load()) == 1 and store.load()[("SYNL01", "n1")].ok


def test_a_completed_label_that_does_not_answer_this_labelling_s_fields_is_refused(store):
    """A row here whose verdicts name other fields did not come from a reading of this
    requirement, and every per-(type, field) number computed over this file would be built on it.
    An errored label is exempt: nothing was read, so there is nothing to have answered."""
    for verdicts in ({}, {"primary_site": "neither"},
                     every_field("neither") | {"stage": "neither"}):
        with pytest.raises(L.LabelShapeError):
            store.append(L.NoteLabel("SYNL01", "n2",
                                     admissibility=L.Admissibility(verdicts)))
    store.append(L.NoteLabel("SYNL01", "n2", error="upstream 503"))  # exempt, and countable


def test_a_per_note_labelling_on_disk_cannot_be_read_as_a_per_field_one(store, tmp_path):
    """Three guards, because this is a silent failure and the file outlives the code.

    The run key moves, so a new scan does not append into an old file at all. Should one be
    pointed at this reader anyway, the old row is refused BY NAME rather than dropped as a torn
    line — the difference between "this line was cut short" and "this line answers a differently
    shaped question" is the difference between resuming a run and assembling one out of two.
    """
    old_row = {"patient_id": "SYNL01", "note_id": "n1",
               "admissibility": {"verdict": "can_establish", "quote": "q", "quote_verified": True}}
    with pytest.raises(L.LabelShapeError) as exc:
        L.NoteLabel.from_dict(old_row)
    assert L.PROMPT_VERSION in str(exc.value)
    store.path.write_text(json.dumps(old_row) + "\n", encoding="utf-8")
    with pytest.raises(L.LabelShapeError):
        store.load()  # NOT swallowed by the torn-line handling
    assert L.PROMPT_VERSION == "labelling/4"  # bumped with the shape; see prompt_hash


def test_the_stored_row_carries_the_per_field_answer_and_a_collapse_that_is_never_read_back(store):
    """The collapsed `verdict` on the row is an export-only projection for readers that predate
    per-field standing. It is recomputed on every write, and `from_dict` ignores it — so a row
    whose projection was edited by hand cannot smuggle a verdict past the field-level answer."""
    label = L.NoteLabel("SYNL01", "n1", admissibility=L.Admissibility(
        {"primary_site": "can_establish", "histology": "neither"}, quote="q"))
    row = label.to_dict()
    assert row["admissibility"]["verdicts"] == {"primary_site": "can_establish",
                                                "histology": "neither"}
    assert row["admissibility"]["verdict"] == "can_establish"  # the upward collapse
    row["admissibility"]["verdict"] = "neither"
    back = L.NoteLabel.from_dict(row)
    assert back.admissibility == label.admissibility and back.admissibility.verdict == \
        "can_establish"


def test_asking_a_label_about_a_field_it_never_answered_raises_rather_than_saying_neither():
    adm = L.Admissibility({"primary_site": "can_establish", "histology": "neither"})
    with pytest.raises(L.LabelShapeError):
        adm.verdict_for("behavior")


def test_concurrency_is_bounded_by_the_config(corpus, store):
    client = StubClient(delay=0.02)
    make_runner(corpus, store, client, concurrency=2).run(ALL_PATIENTS)
    assert client.max_concurrent == 2


def test_a_scan_cannot_be_configured_without_a_ceiling_or_a_requirement():
    with pytest.raises(TypeError):
        L.ScanConfig()  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        L.ScanConfig(max_usd=1.0, terms=TERMS)  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        L.ScanConfig(max_usd=0.0, requirement=MED, terms=TERMS)


def test_the_ceiling_aborts_the_run_and_keeps_everything_already_paid_for(corpus, store):
    per_note = L.cost_usd(1000, 100)
    report = make_runner(corpus, store, StubClient(), max_usd=per_note * 2.5,
                         concurrency=1).run(ALL_PATIENTS)
    assert report.aborted and "raise the ceiling" in report.abort_reason
    assert report.n_written == 3 and len(store.load()) == 3
    assert report.spend_usd == pytest.approx(per_note * 3)


def test_a_resumed_run_counts_prior_spend_against_the_ceiling(corpus, store):
    per_note = L.cost_usd(1000, 100)
    make_runner(corpus, store, StubClient(), max_usd=per_note * 2.5,
                concurrency=1).run(ALL_PATIENTS)
    client = StubClient()
    report = make_runner(corpus, store, client, max_usd=per_note * 2.5).run(ALL_PATIENTS)
    assert client.calls == 0 and report.aborted and report.n_written == 0


def test_labels_root_refuses_a_path_inside_the_repository():
    with pytest.raises(L.LabellingError):
        L.labels_root(str(Path(__file__).resolve().parents[1] / "runs" / "labels"))


def test_labels_are_not_group_or_world_readable(corpus, store):
    scan(corpus, store)
    for p in (store.path, store.dir / "manifest.json"):
        assert p.stat().st_mode & 0o077 == 0, p
    assert json.loads((store.dir / "manifest.json").read_text())["model"] == store.model


# ============================================================================
# 6. THE AFTER-THE-FACT AUDIT — the only thing that ever checks the labeller
# ============================================================================

def carries(answer, label) -> bool:
    """The caller's decision procedure, over the caller's own key shape.

    This module supplies it; `labelling` never looks inside `ANSWER_KEY` and does not know that
    its values are tuples of strings. That is the point of the argument.
    """
    text = CORPUS_TEXT[label.patient_id][label.note_id].lower()
    return any(s.lower() in text for s in answer)


def test_the_audit_confirms_the_labeller_against_a_key_it_has_never_heard_of(corpus, store):
    """ANSWER_KEY holds one of the two fields — the cell type — so that is the field scored."""
    audit = L.audit_relevance(scan(corpus, store).values(), ANSWER_KEY, carries=carries,
                              field_name="histology")
    assert audit.field_scored == "histology"
    assert audit.n_labels == N_NOTES and audit.n_unscorable == 0
    assert audit.can_establish_precision == 1.0
    assert audit.n_missed == 0 and audit.disagreements == ()
    # Two of the four patients have their answer stated ONLY in a document that cannot establish
    # it, so no can_establish call was made for them at all. Per-note precision is a clean 1.0
    # and half the cohort is unreached; that gap is the finding, and it is the number a retrieval
    # plan lives or dies by. Precision alone would have reported this labelling as perfect.
    assert audit.n_patients_keyed == 4 and audit.n_patients_reached == 2
    assert audit.patient_reach == pytest.approx(0.5)


def test_scoring_the_wrong_field_against_a_key_makes_a_correct_labeller_look_broken(corpus,
                                                                                    store):
    """The cost of the collapse, measured — and the reason the audit records the field it scored.

    The imaging note establishes the site and says nothing about the cell type. Scored against a
    cell-type key at NOTE level, it is a false positive and the labeller looks 33% wrong; scored
    on the field the key is actually about, it is not in the numerator at all and the labeller is
    right. Same labels, same key. Only the question changed shape.
    """
    labels = list(scan(corpus, store).values())
    collapsed = L.audit_relevance(labels, ANSWER_KEY, carries=carries)
    assert collapsed.field_scored == ""
    assert collapsed.can_establish_precision == pytest.approx(2 / 3)
    assert [note for _, note, _ in collapsed.disagreements] == ["Chest-CT-W-Contr_2021-02-10"]
    per_field = L.audit_relevance(labels, ANSWER_KEY, carries=carries, field_name="histology")
    assert per_field.can_establish_precision == 1.0 and per_field.disagreements == ()


def test_the_audit_catches_a_labeller_that_calls_the_wrong_notes_establishing(corpus, store):
    """Without this, the model's judgement silently BECOMES the definition of relevant."""
    always = reply_json(verdicts=every_field("can_establish"),
                        quote="Lisinopril 10 mg daily.", terms=[])
    scan(corpus, store, StubClient(behaviour=lambda n, b: always))
    audit = L.audit_relevance(store.load().values(), ANSWER_KEY, carries=carries,
                              field_name="histology")
    assert audit.can_establish_precision == pytest.approx(4 / N_NOTES)
    assert len(audit.disagreements) == N_NOTES - 4
    assert all(v == "can_establish" for _, _, v in audit.disagreements)


def test_the_audit_catches_an_answer_bearing_note_filed_under_neither(corpus, store):
    """The more expensive direction, and the one precision alone cannot see: a note the key says
    carries the answer, which no keyword derived from these labels will ever be built to find."""
    scan(corpus, store, StubClient(behaviour=lambda n, b: reply_json(
        verdicts=every_field("neither"))))
    audit = L.audit_relevance(store.load().values(), ANSWER_KEY, carries=carries,
                              field_name="histology")
    assert audit.can_establish_precision is None  # nothing measured, NOT a precision of zero
    assert audit.n_missed == 4 and audit.n_patients_reached == 0
    assert {p for p, _, _ in audit.disagreements} == set(ALL_PATIENTS)


def test_a_note_that_merely_mentions_the_answer_is_not_an_audit_disagreement(corpus, store):
    """Restating an answer without standing to establish it is exactly what that class is for.
    Counting it as an error would train the labeller to call every restatement `neither`."""
    audit = L.audit_relevance(scan(corpus, store).values(), ANSWER_KEY, carries=carries,
                              field_name="histology")
    assert audit.table["merely_mentions"][L.CARRIES] >= 2
    assert not [d for d in audit.disagreements if d[2] == "merely_mentions"]


def test_a_patient_absent_from_the_key_and_a_failed_label_are_unscorable(corpus, store):
    """A padded denominator reports a number that is wrong in a direction nobody can sign."""
    labels = list(scan(corpus, store).values())
    partial = L.audit_relevance(labels, {"SYNL01": ANSWER_KEY["SYNL01"]}, carries=carries,
                                field_name="histology")
    assert partial.n_scorable == 3 and partial.n_unscorable == N_NOTES - 3
    assert partial.n_patients_keyed == 1
    broken = [L.NoteLabel("SYNL01", "x", error="timeout")]
    assert L.audit_relevance(broken, ANSWER_KEY, carries=carries).n_scorable == 0


def test_a_falsy_key_entry_is_an_answer_and_not_a_missing_one():
    """`0`, `""` and an empty record all mean "adjudicated, nothing there", which is scorable."""
    lab = L.NoteLabel("P1", "n1", admissibility=L.Admissibility({"f": "neither"}))
    audit = L.audit_relevance([lab], {"P1": ()}, carries=lambda a, l: False)
    assert audit.n_scorable == 1 and audit.n_unscorable == 0


def test_the_audit_is_not_reachable_from_the_labelling_call():
    """It is an after-the-fact check, and nothing on the reading path may take an answer key.

    Not a comment: the signatures are the enforcement. If somebody adds a `key` or `truth`
    argument to the runner so the labeller can 'check itself as it goes', the labelling stops
    being a measurement and becomes an opinion, and this fails.
    """
    import inspect

    for fn in (L.build_note_prompt, L.FullScanRunner.__init__, L.FullScanRunner.label_note,
               L.FullScanRunner.run, L.parse_label_response):
        params = set(inspect.signature(fn).parameters)
        assert not (params & {"answer_key", "key", "truth", "ground_truth", "answers", "carries",
                              "expected", "gold"}), fn
    audit_src = Path(L.__file__).read_text(encoding="utf-8").split("# AN AFTER-THE-FACT AUDIT")[1]
    assert not re.search(r"\bself\.client\b|litellm|\.chat\(", audit_src)


# ============================================================================
# 7. THE CLIENT IS WIRED AND NEVER CALLED
# ============================================================================

def test_the_credentials_file_is_parsed_as_data_never_sourced(tmp_path):
    env = tmp_path / ".azure_env"
    env.write_text("# comment\nexport ACR_API_BASE='https://example.invalid'\n"
                   f"export ACR_API_KEY=$(touch {tmp_path / 'pwned'})\n")
    assert L.parse_env_file(env)["ACR_API_BASE"] == "https://example.invalid"
    assert not (tmp_path / "pwned").exists()


def test_a_missing_or_incomplete_credentials_file_is_an_error_not_a_default(tmp_path):
    with pytest.raises(L.NotConfiguredError):
        L.azure_client(tmp_path / "absent")
    (tmp_path / "half").write_text("export ACR_API_BASE=https://example.invalid\n")
    with pytest.raises(L.NotConfiguredError):
        L.azure_client(tmp_path / "half")
