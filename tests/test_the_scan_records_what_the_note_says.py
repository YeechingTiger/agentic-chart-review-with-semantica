"""The scan measured *where* the answer lives and *whether* a note can settle it. Not *what it says*.

`acr label scan` reads every note in a development set with a model and records, per note: a standing
verdict per field, one verified quote, and the terms that would let a searcher find THIS note. Three
questions a downstream consumer needs are not asked, and each has exactly one consumer that cannot
work without it:

  * **the value the note asserts** -> the conflict base rate, the naive-answer baseline, and the
    distribution of how many notes per chart can establish the answer. That last one is the only
    stopping evidence available anywhere: the agent currently stops when its budget runs out, and
    `runtime_profiles`' whole `should_stop` / `revise` half is never called.
  * **what the note points at** -> indirect evidence that tells the agent where the DIRECT evidence
    is. `merely_mentions` already keeps indirect notes and refuses to fold them into
    `can_establish`, but it cannot tell a signpost ("see pathology report of 3/12") from a bare
    history mention ("history of lung cancer"). Question 2 is self-referential — it asks what finds
    THIS note, never what this note says about finding another.
  * **whether the content is original or carried forward** -> on a real record duplication is most
    of the corpus, and this is the honest denominator for any claim about how much reading is
    wasted. The prompt already reasons about it to answer question 1 ("the same sentence can
    establish an answer in the document that first rendered it and establish nothing in a document
    that copied it forward"); it just never reports the judgement.

## Everything the model claims to have copied is checked

That is this module's existing discipline and the additions keep it: `retrieval_terms` are verified
against the text and the discards counted in `n_terms_hallucinated`; the quote is verified and an
unverifiable one is recorded as such. So a pointer is a verbatim span or it is discarded, and an
asserted value carries the span it was read from.

The value itself is a MODEL CODING and cannot be verified against text — `March 12, 2019` and
`20190312` are the same fact in two notations. So it is recorded beside its verified span, and a
reader can tell a coding error from a fabrication. Any conflict rate derived from it is
lower-confidence than the admissibility counts, and it says so.

## This moves `prompt_hash`, which is the point

`PROMPT_VERSION` is bumped, so old labels land in a different directory and stay readable rather than
being silently mixed with answers to a different question. Adding a field after paying for a
100-patient scan means paying for it twice, which is why this lands before the scan and not after.
"""

from __future__ import annotations

import json

import pytest

from acr.improvement import labelling as L


def _requirement():
    from acr.contract.spec import load_spec
    from acr.core import site
    return L.Requirement.from_spec(
        load_spec(site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml"))


#: The field these tests are about. STORE.390 declares FOUR — the date and the three imputation
#: flags that replaced `month_day_imputed` — and a reply short of any of them is refused, so a
#: fixture reply has to answer all four while only the first is what is under test here.
FIELDS = ["date_of_initial_diagnosis"]
ALL_FIELDS = _requirement().field_names
#: The bounds on question 2, chosen for this fixture. `TermConfig` has no defaults anywhere, by
#: design: a cap typed into the module would be a decision about how much a model may pad made in
#: a commit nobody rereads.
TERMS = L.TermConfig(max_terms_per_note=8, min_term_chars=3)
NOTE = (
    "PROGRESS NOTE\n"
    "Patient seen in follow-up. Adenocarcinoma of the right upper lobe was first diagnosed on "
    "March 12, 2019 per the outside pathology report.\n"
    "See Surgical Pathology Report dated 2019-03-14, accession SP19-4471, for the full "
    "morphology.\n"
)


@pytest.fixture(scope="module")
def requirement():
    return _requirement()


def _verdicts(main: str = "can_establish") -> dict[str, str]:
    """One standing per field of the real spec — every field, because a reply missing one is
    refused rather than completed. Only `FIELDS[0]` varies; the imputation flags are not what any
    test here is about."""
    return {f: (main if f == FIELDS[0] else "neither") for f in ALL_FIELDS}


def _reply(**over) -> str:
    body = {
        "admissibility": {"verdicts": _verdicts(),
                          "quote": "Adenocarcinoma of the right upper lobe was first diagnosed on "
                                   "March 12, 2019 per the outside pathology report."},
        "retrieval_terms": [{"term": "Adenocarcinoma", "reason": "names_the_section"}],
        "asserted_values": {FIELDS[0]: {"value": "20190312", "as_written": "March 12, 2019"}},
        "pointers": ["Surgical Pathology Report dated 2019-03-14", "accession SP19-4471"],
        "copied_forward": True,
    }
    body.update(over)
    return json.dumps(body)


def parse(text, requirement):
    """The reply, read against the requirement AND the note it was written about.

    `note_text` is not optional in spirit: everything the model claims to have copied is checked
    against it, so a pointer parsed without it could only be discarded.
    """
    return L.parse_label_response(text, requirement=requirement, terms=TERMS, note_text=NOTE)


# ------------------------------------------------------------------ the value the note asserts

def test_the_asserted_value_is_recorded_per_field(requirement):
    r = parse(_reply(), requirement)
    got = r.asserted_values[FIELDS[0]]
    assert got.value == "20190312"
    assert got.as_written == "March 12, 2019"


def test_the_span_is_verified_and_the_coding_is_not(requirement):
    """`March 12, 2019` and `20190312` are one fact in two notations, so the value cannot be checked
    against the text. The span it was read from can be, and is — a reader has to be able to tell a
    coding error from a fabrication."""
    r = parse(_reply(), requirement)
    assert r.asserted_values[FIELDS[0]].verify(NOTE) is True

    bad = parse(_reply(asserted_values={FIELDS[0]: {"value": "20190312",
                                                    "as_written": "diagnosed in the spring"}}),
                requirement)
    assert bad.asserted_values[FIELDS[0]].verify(NOTE) is False, (
        "a span that is not in the note must not pass as verified")


def test_a_note_that_asserts_nothing_records_nothing(requirement):
    r = parse(_reply(asserted_values={}), requirement)
    assert r.asserted_values == {}


def test_a_field_the_requirement_does_not_declare_is_refused(requirement):
    """The same rule question 1 already applies: an answer about a field nobody asked for is an
    invented standing, and folding it in would make the scan's subject depend on the model."""
    with pytest.raises(L.LabelReplyError, match="not_a_field"):
        parse(_reply(asserted_values={"not_a_field": {"value": "x", "as_written": "x"}}),
              requirement)


def test_a_value_with_no_span_is_refused(requirement):
    """A coding with nothing behind it is the one shape this module refuses everywhere else."""
    with pytest.raises(L.LabelReplyError, match="as_written"):
        parse(_reply(asserted_values={FIELDS[0]: {"value": "20190312"}}), requirement)


# ------------------------------------------------------------------ what the note points at

def test_pointers_are_verbatim_spans_and_unverifiable_ones_are_discarded(requirement):
    """The signpost half. `merely_mentions` keeps indirect notes; it cannot tell one that names
    where the direct evidence is from one that names nothing."""
    r = parse(_reply(pointers=["Surgical Pathology Report dated 2019-03-14",
                               "see the tumour board note"]), requirement)
    assert "Surgical Pathology Report dated 2019-03-14" in r.pointers
    assert "see the tumour board note" not in r.pointers, "not in the note; discarded"
    assert r.n_pointers_proposed == 2 and r.n_pointers_hallucinated == 1


def test_a_note_that_points_nowhere_is_the_common_case(requirement):
    r = parse(_reply(pointers=[]), requirement)
    assert r.pointers == () and r.n_pointers_hallucinated == 0


# ------------------------------------------------------------------ original or carried forward

def test_copy_forward_is_reported_not_only_used(requirement):
    """The prompt already reasons about this to answer question 1. Reporting the judgement costs no
    extra reasoning and gives the cost model its denominator."""
    assert parse(_reply(copied_forward=True), requirement).copied_forward is True
    assert parse(_reply(copied_forward=False), requirement).copied_forward is False


def test_a_note_bearing_on_nothing_has_nothing_to_judge(requirement):
    """`None`, not `False`. There is no content to call original when the note is `neither` on every
    field, and `False` would assert that there is."""
    r = parse(_reply(admissibility={"verdicts": _verdicts("neither"), "quote": ""},
                     asserted_values={}, pointers=[], copied_forward=None), requirement)
    assert r.copied_forward is None


# ------------------------------------------------------------------ it survives the round trip

def test_every_addition_reaches_the_row_and_comes_back(requirement, tmp_path):
    """`NoteLabel.to_dict` -> JSONL -> `from_dict` is how the whole develop plane consumes this. A
    field that does not survive that trip is a field the aggregator cannot see."""
    r = parse(_reply(), requirement)
    label = L.NoteLabel(patient_id="P1", note_id="n1", doc_type="Progress-Note",
                        spec_id="S", admissibility=r.admissibility,
                        retrieval_terms=r.retrieval_terms,
                        asserted_values=r.asserted_values, pointers=r.pointers,
                        n_pointers_proposed=r.n_pointers_proposed,
                        n_pointers_hallucinated=r.n_pointers_hallucinated,
                        copied_forward=r.copied_forward)
    back = L.NoteLabel.from_dict(json.loads(json.dumps(label.to_dict())))
    assert back.asserted_values[FIELDS[0]].value == "20190312"
    assert back.pointers == r.pointers
    assert back.copied_forward is True


def test_a_row_written_before_these_fields_still_loads(requirement):
    """912 real labels already exist. They answer a different question and must stay readable, not
    be back-filled with invented values."""
    old = {"patient_id": "P1", "note_id": "n1", "doc_type": "X", "spec_id": "S",
           "admissibility": {"verdicts": {FIELDS[0]: "can_establish"}, "quote": "q"},
           "retrieval_terms": [{"term": "t", "reason": "names_the_section"}]}
    back = L.NoteLabel.from_dict(old)
    assert back.asserted_values == {} and back.pointers == ()
    assert back.copied_forward is None, "absent must read as unknown, never as False"


# ------------------------------------------------------------------ and the prompt moved

def test_the_prompt_version_moved_so_old_labels_do_not_mix(requirement):
    """`prompt_hash` is derived from `PROMPT_VERSION` and the prompt text, and the runner keys its
    output directory on it. Asking three more questions and keeping the version would put two
    different questions' answers in one file."""
    assert L.PROMPT_VERSION != "labelling/4"


def test_the_prompt_asks_all_three_and_says_what_is_checked(requirement):
    """The prompt is the interface. A field on the dataclass that the prompt never asks for is a
    column of empties, which reads as a model with nothing to say."""
    p = L.NOTE_PROMPT_TEMPLATE
    assert "QUESTION 3" in p and "QUESTION 4" in p and "QUESTION 5" in p
    for token in ("asserted_values", "as_written", "pointers", "copied_forward"):
        assert token in p, f"{token} is on the row and the prompt never asks for it"
    assert "checked against" in p, "the verification promise has to be stated to the model"
