"""The second variable family, held to the same standard as the first.

STORE.400_522_523 was the criterion every mechanism in this repo was designed against, so it
cannot tell us whether the mechanisms generalise or whether they were fitted to one variable.
This file is that question asked mechanically of STORE.700_880: does a spec written by
following the pattern -- and nothing else -- load, stratify, gate and self-check?

Four properties, in the order a run hits them:

  1. the value domains are enforceable        (formats are regexes, not registry notation)
  2. the strata route real document types     (substring matching is a known trap)
  3. the gate is SATISFIABLE                  (a gate nobody can pass is broken, not strict)
  4. the answer checks fire, and only fire when they should

Property 3 is the one that has actually gone wrong before: `format: "CCYYMMDD"` in STORE.390
rejects every valid date, and a `max_elusion_upper` below the Clopper-Pearson floor for the
declared sample size makes an obligation that no amount of work discharges. Both are silent.
"""

# The calls below pass `enforce=True`. `evaluate_gate` is ADVISORY by default as of 2026-07-30:
# it still counts strata, samples and residual bounds identically, but routes its sentences to
# `advisories` instead of `missing` so they inform the model rather than refuse its answer.
# "Have I looked at enough of this chart?" is a clinical judgement and now lives in
# `assets/skills/coverage-judgement/SKILL.md`; measured over every recorded trace, coverage obligations
# produced ~150 answer rejections and 27 of them refused a tuple that was exactly the registry's.
#
# These tests are about the ARITHMETIC, which is unchanged and still worth pinning: a bound that
# clears its cap only by inheriting a stale sampling frame is anti-conservative whether or not
# anybody is refused over it. `enforce=True` is how a test reaches the refusal wording.

# ---------------------------------------------------------------------------------------------
# TESTS REMOVED 2026-07-30, with the rules they specified.
#
# `answer_checks` carried five checks that decided clinical questions by matching word lists
# against the model's own cited quotes. Measured over every trace this project has recorded
# (266 traces, 202 joinable to registry gold, 122 firings):
#
#   not_less_specific        22 fires   22 rejected the registry's own value    0 ever helped
#   nos_requires_search      24 fires   21 rejected the registry's own value    0 ever helped
#   conflict_requires_nos    67 fires   18 rejected the registry's own value   15 "helped",
#                                       all 15 of them the same push to the NOS code
#   origin_not_specimen       2 fires    0                                      0
#   code_matches_cited_text   0 fires    -                                      -
#
# `fit_terms_to_budget` deleted 103 search terms the model had proposed for itself, and on
# CASE009 it deleted `lobe` and `bronchus` while `nos_requires_search` refused the answer for
# never having searched them. The required-keyword gate enforced a list measured at 87.4%
# recall over 276,054 documents.
#
# A test that pins a rule in place is part of the rule, so these went with them:
#   - test_a_clinical_stage_read_off_the_resection_is_rejected
#   - test_an_undivided_t_category_loses_to_a_documented_subcategory
#   - test_every_x_and_unknown_value_carries_the_proof_burden
#   - test_unknown_stage_is_falsified_by_a_stage_in_the_cited_evidence
#   - test_unknown_stage_requires_having_searched
#
# Nothing replaced them here. A wrong clinical value is an instruction-following failure and is
# measured as one. tests/test_answer_checks.py holds what survives: field `format` and
# `allowable_values`, the only check with a positive measured record.
# ---------------------------------------------------------------------------------------------
from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import pytest

from acr.chartstore.corpus import Corpus, DocMeta
from acr.contract.answer_checks import check_answer, check_field_formats
from acr.contract.spec import load_spec
from acr.review.coverage import (
    CoverageLedger,
    ForcedSampler,
    assign_strata,
    evaluate_gate,
    strata_from_spec,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "assets" / "specs" / "STORE.700_880.stage.yaml"
CORPUS = ROOT / "corpus" / "patients"


@pytest.fixture(scope="module")
def spec():
    # Loaded, not skipped. This fixture briefly caught UnprovenancedElementError and skipped
    # the module while the stage spec had no `provenance:` block; the block landed on
    # 2026-07-27 and the guard is removed with it. If the spec ever stops loading again, this
    # module must fail loudly rather than quietly report 30-odd passes it never ran.
    return load_spec(SPEC_PATH)


@pytest.fixture(scope="module")
def gate_spec(spec):
    return spec.proof_obligation.for_negative["gate"]


@pytest.fixture
def strata(spec):
    return strata_from_spec(spec)


def _doc(doc_type: str) -> DocMeta:
    """A DocMeta with only the field strata routing reads. No corpus, no PHI."""
    return DocMeta(note_id=f"{doc_type}_2020-01-01", doc_type=doc_type,
                   date=dt.date(2020, 1, 1), seq=1, n_chars=1000)


# --------------------------------------------------------------------------- 1. loading
def test_the_spec_loads_and_declares_an_identity(spec):
    assert spec.spec_id == "STORE.700_880.stage"
    assert set(spec.identity()) == {"spec_id", "spec_version", "spec_hash"}
    assert spec.data_source == "notes", "outside_notes forces SPEC_INSUFFICIENT unconditionally"


def test_clinical_and_pathologic_are_separate_fields(spec):
    """The conflation this criterion invites has to be REPRESENTABLE to be detectable.

    A single `stage` field would make "the run copied the pathologic group into the clinical
    one" a thing the output shape cannot express, and therefore a thing no check can catch.
    """
    names = [f.name for f in spec.fields]
    assert [n for n in names if n.startswith("clinical_")]
    assert [n for n in names if n.startswith("pathologic_")]
    assert "summary_stage" in names
    assert len(names) == len(set(names))


# ------------------------------------------------------------- 2. enforceable value domains
FORMAT_CASES = [
    # field,                value,      accepted?
    ("clinical_t",          "cT2a",     True),
    ("clinical_t",          "cT1mi",    True),
    ("clinical_t",          "cTX",      True),
    ("clinical_t",          "cTis",     True),
    ("clinical_t",          "ct2a",     False),   # case matters; re.fullmatch is not casefold
    ("clinical_t",          "T2a",      False),   # the c prefix IS the datum
    ("clinical_t",          "cT5",      False),
    ("clinical_n",          "cN2",      True),
    ("clinical_n",          "cN4",      False),
    ("clinical_m",          "cM1b",     True),
    ("clinical_m",          "cM0",      True),
    ("clinical_m",          "cMX",      False),   # AJCC 8th has no clinical MX
    ("pathologic_t",        "pT3",      True),
    ("pathologic_t",        "cT3",      False),   # a c-category in a p-field is the conflation
    ("pathologic_n",        "pN0",      True),
    ("pathologic_m",        "pM1c",     True),
    ("pathologic_m",        "pM0",      False),   # AJCC 8th has no pM0
    ("pathologic_m",        "pMX",      False),
    ("clinical_stage_group",   "IIIA",  True),
    ("clinical_stage_group",   "OCCULT", True),
    ("clinical_stage_group",   "99",    True),
    ("clinical_stage_group",   "IIIa",  False),
    ("clinical_stage_group",   "3A",    False),
    ("pathologic_stage_group", "IB",    True),
    ("pathologic_stage_group", "OCCULT", False),  # occult carcinoma is a clinical group only
    ("summary_stage",       "7",        True),
    ("summary_stage",       "5",        False),
]


@pytest.mark.parametrize("field,value,accepted", FORMAT_CASES)
def test_declared_value_domains_are_actually_enforced(spec, field, value, accepted):
    violations = check_field_formats(spec.fields, {field: value})
    assert (violations == []) is accepted, violations


def test_every_declared_format_is_a_python_regex_not_registry_notation(spec):
    """The STORE.390 defect, pinned so this spec cannot repeat it.

    `format: "CCYYMMDD"` compiles cleanly as a regex and then matches nothing, so
    check_field_formats rejects every valid value with no signal that the spec is at fault.
    A pattern is only enforcement if something it should accept is accepted.
    """
    accepted_somewhere = {f for f, _, ok in FORMAT_CASES if ok}
    for f in spec.fields:
        if not f.format:
            continue
        re.compile(f.format)                       # raises on a malformed pattern
        assert f.name in accepted_somewhere, (
            f"{f.name} declares a format that no test value exercises — an unexercised "
            f"pattern is indistinguishable from CCYYMMDD"
        )


def test_allowable_values_are_quoted_strings(spec):
    """check_field_formats compares str(v); an unquoted YAML 0 would still work, but an
    unquoted 99 next to a quoted "IIIA" is how a domain silently splits into two types."""
    for f in spec.fields:
        for v in (f.allowable_values or []):
            assert isinstance(v, str), f"{f.name}: {v!r} must be quoted in the YAML"


def test_a_null_field_is_abstention_not_a_format_violation(spec):
    """Nine fields and most charts document a handful. Leaving the rest null must be free,
    or the format checker becomes an argument for guessing."""
    assert check_field_formats(spec.fields, {"clinical_t": "cT2a"}) == []
    assert check_field_formats(spec.fields, {"clinical_t": None, "summary_stage": ""}) == []


# ------------------------------------------------------------------------ 3. strata routing
def test_three_strata_with_the_hardcoded_names(strata):
    """evaluate_gate looks up `can_establish`, `cannot_establish` and `may_mention` by literal
    name. A fourth stratum, or a rename, is invisible to every check but max_elusion_upper."""
    assert [s.name for s in strata] == ["can_establish", "cannot_establish", "may_mention"]
    assert [s.rest for s in strata].count(True) == 1
    assert strata[-1].rest, "the rest stratum must be declared last for assign_strata's order"


def test_every_policy_has_a_branch_in_stratum_results(strata):
    """`exhaustive_per_window` parses fine and falls into the sampling else-branch, where it
    can never be complete. Only these four are actually implemented."""
    implemented = {"exhaustive", "exhaustive_until_witness",
                   "validate_by_sampling", "search_then_read_hits_and_sample_misses"}
    assert {s.policy for s in strata} <= implemented


def test_establishes_names_only_real_fields(spec, strata):
    names = {f.name for f in spec.fields}
    for s in strata:
        assert set(s.establishes) <= names, f"{s.name} claims a field that does not exist"
    can = next(s for s in strata if s.name == "can_establish")
    assert set(can.establishes) == names, "the establishing stratum must speak to every field"


ROUTING = [
    # the pathology family, none of which contains the word "pathology"
    ("Fine-Needle-Report",              "can_establish"),
    ("Core-Needle-Biopsy",              "can_establish"),
    ("IMMUNOHISTOLOGY-RPT",             "can_establish"),
    ("Lung-Mediastinum-Perc-Needle-Bx", "can_establish"),
    ("Surgical-Pathology-Report",       "can_establish"),
    ("Cytology-Bronchial-Brushings",    "can_establish"),
    # where a clinical stage is actually assigned
    ("Tumor-Board-Recommendation-Note", "can_establish"),
    ("Thoracic-Surg-Initial-Eval-Note", "can_establish"),
    ("Hem-Onc-MD-OP-Progress-Note",     "can_establish"),
    ("Onc-Med-MD-OP-Progress-Note",     "can_establish"),
    ("Radiation-Oncology-Consult-Note", "can_establish"),
    # inert for all nine fields
    ("EKG",                             "cannot_establish"),
    ("Prescriptions-Filled-RxHub",      "cannot_establish"),
    ("Speech-Language-Pathology-Note",  "cannot_establish"),
    # everything else, including imaging, which supplies T/N/M inputs and restates stage
    ("Chest-CT-WWO-Contr",              "may_mention"),
    ("Body-Whole-PET-CT-Scan",          "may_mention"),
    ("Spine-Thoracic-2V-XR",            "may_mention"),
    ("Discharge-Summary",               "may_mention"),
    ("Pulm-MD-OP-Progress-Note",        "may_mention"),
    ("Some-Type-Nobody-Anticipated",    "may_mention"),
]


@pytest.mark.parametrize("doc_type,expected", ROUTING)
def test_document_types_route_to_the_intended_stratum(strata, doc_type, expected):
    """doc_type_matches is a case-insensitive SUBSTRING over a filename-derived type.

    That mechanism filed Fine-Needle-Report outside ["Pathology","Cytology"] and pulled
    Speech-Language-Pathology-Note in, on the real corpus on 2026-07-26. Enumerating the type
    names defensively is the only available fix, and enumeration is exactly the kind of thing
    that rots silently, so it is pinned here rather than trusted.
    """
    assigned = assign_strata([_doc(doc_type)], strata)
    assert [k for k, v in assigned.items() if v] == [expected]


def test_an_unrecognised_type_defaults_to_being_looked_at(strata):
    """The rest stratum is may_mention, not cannot_establish -- the reverse of
    STORE.400_522_523.

    Unmentioned means unjudged, and the safe default for unjudged is to look. Defaulting the
    other way also makes exclusion_validated brittle: it fails the moment a drawn document
    turns out to be citable, and under this criterion a spine film showing a bone metastasis
    is citable.
    """
    rest = next(s for s in strata if s.rest)
    assert rest.name == "may_mention"
    assert rest.policy == "search_then_read_hits_and_sample_misses"


# ------------------------------------------------------- 4. the gate is satisfiable at all
REQUIRED_KEYWORDS = ["stage", "tnm", "tumor size", "pleural", "lymph node", "metasta"]


def test_the_required_keywords_are_the_ones_the_gate_reads(spec):
    """Stratum-level required_keywords do not feed the gate; only this top-level list does.

    All four previously shipped specs left it empty, so their `required_keywords_all_searched`
    flags enforced nothing at all. This spec is the first to populate it, which means it is
    also the first for which the search obligation is real -- and the first that could be made
    unpassable by asking for a term the prompt never suggests.
    """
    assert spec.proof_obligation.required_keywords == REQUIRED_KEYWORDS


def test_no_required_keyword_silently_discharges_another(spec):
    """graph._check_gate matches bidirectionally (`kw in term or term in kw`), so a required
    "stage" would be satisfied by a search for "pathologic stage" AND vice versa. Overlapping
    terms therefore inflate the apparent obligation without adding a single search."""
    kws = spec.proof_obligation.required_keywords
    for a in kws:
        for b in kws:
            if a is not b:
                assert a not in b, f"required keyword {a!r} is subsumed by {b!r}"


def test_every_field_is_reachable_by_a_required_search(spec):
    """The measured omission on SYN0002: a keyword list built from the topic rather than from
    the fields. The site spec asked for pathology terms and none for laterality, so the list
    was falsified by a lobe sitting in an unsearched note."""
    cov = spec.model_extra["keyword_field_coverage"]
    declared = set(spec.proof_obligation.required_keywords)
    assert set(cov) == {f.name for f in spec.fields}, "every field must name its searches"
    used = {k for ks in cov.values() for k in ks}
    assert used <= declared, f"claims coverage by unrequired terms: {used - declared}"
    assert declared <= used, f"required search reaches no field: {declared - used}"
    for field, ks in cov.items():
        assert ks, f"{field} is covered by no search"


def test_the_search_hints_can_discharge_the_gated_keywords(spec):
    """The prompt has to contain a route to satisfying the gate.

    search_hints is rendered to the model; required_keywords is enforced against it and is
    not rendered. If the two drift apart the agent is asked to prove something it was never
    told to look for, and the trace fills with rejections that look like diligence.
    """
    hints = [h.lower() for h in spec.search_hints]
    for kw in spec.proof_obligation.required_keywords:
        assert any(kw in h or h in kw for h in hints), f"nothing in search_hints reaches {kw!r}"


def _work_the_obligation(ledger: CoverageLedger, chart) -> None:
    """Do exactly what the proof obligation asks, and nothing else."""
    ledger.listed_documents = True
    for d in ledger.by_stratum["can_establish"]:            # policy: exhaustive
        ledger.note_read(d.note_id, d.doc_type)
    # Both lists, because the gate now prices both. The top-level list is the one
    # graph._check_gate has always looped over; the STRATUM's list is what a
    # search_then_read_hits_and_sample_misses miss sample is measured against, and until the
    # keyword-list inversion was fixed it fed nothing but _keyword_hits_among_drawn. Running
    # only the top-level six left "staging", "ajcc", "mediastinal", "nodal" and "extent of
    # disease" unsearched while the ledger still called the keyword list validated.
    stratum_kws = [k for s in ledger.specs for k in s.required_keywords]
    for kw in list(dict.fromkeys([*REQUIRED_KEYWORDS, *stratum_kws])):
        ledger.note_search(kw, [h.note_id for h in chart.search(kw, max_hits=40)])
    # ...then READ what those searches turned up, which is the clause in the middle of
    # `search_then_read_hits_and_sample_misses` that nothing used to enforce. A hit is
    # excluded from the miss-sampling frame, so an unread hit is reviewed by nothing at all.
    by_id = {d.note_id: d for d in ledger.docs}
    for nid in sorted(ledger.search_hit_notes):
        if nid in by_id:
            ledger.note_read(nid, by_id[nid].doc_type)
    for docs in ledger.pending_samples().values():          # drawn by the runtime, not us
        for d in docs:
            ledger.note_read(d.note_id, d.doc_type)
    ledger.resolve_sample_verdicts(cited=set())


@pytest.fixture
def chart():
    return Corpus(CORPUS).chart("SYN0001")


@pytest.fixture
def ledger(spec, chart):
    docs, _ = chart.list_documents(limit=100_000)
    return CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(1234))


def test_the_gate_fails_before_the_work(ledger, gate_spec):
    """Guards every assertion below: if the gate passed on an empty ledger they prove nothing."""
    g = evaluate_gate(gate_spec, ledger.stratum_results(), enforce=True)
    assert g.verdict == "FAIL"
    assert not g.checks["exhaustive_strata_complete"]
    assert not g.checks["exclusion_validated"]


def test_the_gate_passes_once_the_work_is_done(ledger, chart, gate_spec):
    """Satisfiability. Nothing here is a judgement call -- read the establishing stratum, run
    the declared searches, inspect what the sampler drew."""
    _work_the_obligation(ledger, chart)
    g = evaluate_gate(gate_spec, ledger.stratum_results(), enforce=True)
    assert g.verdict == "PASS", g.missing


def test_the_elusion_cap_is_above_the_floor_its_sample_sizes_allow(ledger, chart, gate_spec):
    """max_elusion_upper has no code default, and 25 clean samples cannot bound elusion below
    0.1129. A cap of 0.10 with min_sample 25 is an obligation no work discharges."""
    _work_the_obligation(ledger, chart)
    worst = max(r.elusion_upper for r in ledger.stratum_results() if r.name != "can_establish")
    assert worst <= gate_spec["max_elusion_upper"]
    assert gate_spec["max_elusion_upper"] < 1.0, "an absent cap is not a cap"



def test_the_runtime_gate_no_longer_refuses_over_an_unsearched_term(spec, ledger, chart):
    """The required-keyword loop is gone from `check_gate`, and a dropped term proves it.

    It used to add two loops on top of `evaluate_gate`, one over the spec's `required_keywords`
    and one over the plan's expanded list, refusing the answer for any term the agent had not
    run. Both were removed on 2026-07-30. The baseline design gives the model no keyword list at
    all: it has `search`, it decides what to look for, and a term it chose not to run is a
    retrieval decision rather than a contract violation.

    The list that was being enforced here was also measured wrong -- 87.4% recall over 276,054
    documents, missing at least one answer-bearing document for 31.7% of patients, because it has
    `carcinoma` and not `cancer`. And `fit_terms_to_budget` was deleting the model's own
    proposals, `lobe` and `bronchus` among them, which `nos_requires_search` then refused the
    answer for not having searched.

    What `check_gate` still enforces is the one thing it can compute without a vocabulary: you
    must have listed the patient's documents before asserting absence.
    """
    from acr.review.answer_gate import check_gate

    _work_the_obligation(ledger, chart)
    assert check_gate(spec, ledger).verdict == "PASS"

    ledger.searched_terms.remove("pleural")
    g = check_gate(spec, ledger)
    assert g.verdict == "PASS", "an unsearched term is no longer a refusal"
    assert not any("pleural" in m for m in g.missing), g.missing

    ledger.listed_documents = False
    blocked = check_gate(spec, ledger)
    assert blocked.verdict == "FAIL"
    assert any("list the patient's documents" in m for m in blocked.missing)


def test_reading_only_the_pathology_does_not_pass(spec, chart):
    """The stratum deliberately spans pathology AND the oncology assessments, because a
    patient who was never resected has no pathologic stage and their clinical stage lives in
    an oncology note. Reading half the stratum must not be enough."""
    docs, _ = chart.list_documents(limit=100_000)
    led = CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(1234))
    pathology = [d for d in led.by_stratum["can_establish"] if "Pathology" in d.doc_type]
    assert pathology and len(pathology) < len(led.by_stratum["can_establish"])
    for d in pathology:
        led.note_read(d.note_id, d.doc_type)
    r = next(x for x in led.stratum_results() if x.name == "can_establish")
    assert not r.complete


# ---------------------------------------------------------------- 5. the answer checks fire
def _ev(*quotes: str) -> list[dict]:
    return [{"note_id": f"N{i}", "quote": q, "supports": "stage"} for i, q in enumerate(quotes)]


@pytest.fixture(scope="module")
def checks(spec):
    return spec.answer_checks


def test_a_well_supported_answer_passes_everything(spec, checks):
    """A check that always fires is a check that will be turned off. This is the control."""
    value = {"clinical_t": "cT2a", "clinical_n": "cN0", "clinical_m": "cM0",
             "clinical_stage_group": "IB", "summary_stage": "1"}
    evidence = _ev("3.2 cm right upper lobe mass with visceral pleural invasion",
                   "clinical stage IB (cT2a cN0 cM0) by thoracic oncology")
    searched = ["stage", "tnm", "tumor size", "pleural", "lymph node", "metasta"]
    assert check_field_formats(spec.fields, value) == []
    assert check_answer(checks, value, evidence, searched) == []


def test_a_correct_subcategory_is_not_punished_by_its_neighbour(checks):
    """Why the T checks are one entry per NOS value: contradicted_by is a flat list, so folding
    cT1 and cT2 together would make a documented cT1a reject a correctly coded cT2."""
    assert check_answer(checks, {"clinical_t": "cT2b"},
                        _ev("cT1a nodule in the left lower lobe, separate primary"),
                        ["tumor size"]) == []


def test_one_pre_treatment_quote_clears_the_conflation_check(checks):
    """all([]) is vacuously true and `all` over a mixed ledger is not: citing the workup as
    well as the specimen is the behaviour being asked for, so it must not be penalised."""
    assert check_answer(checks, {"clinical_stage_group": "IIB"},
                        _ev("SPECIMEN RECEIVED: right upper lobe, lobectomy",
                            "pre-treatment PET-CT: 4.5 cm mass, hilar node uptake, no distant disease"),
                        ["stage", "tnm"]) == []
