"""Document concepts are REFERENCE. The tests here are mostly about what they must not be.

`doc_type_matches` was this same knowledge written as a case-insensitive substring list that fed
the coverage gate and barred reads. Measured on this corpus it matched
`Speech-Language-Pathology-Note` and missed `Non-Gyn-Cyto-FNA` (1,285 documents),
`FN-Aspirate-Report` (881) and `SURG-PATH-RESULT` (231), and 107 of the 219 patients whose
`can_establish` count is zero hold one of those reports anyway.

So the properties worth pinning are negative: no local type name, no measurement claimed at
baseline, no ordering, and nothing that could be applied as a filter.
"""
from __future__ import annotations

import re

from acr.document_concepts import (
    BASELINE_CONCEPTS,
    DocumentConcept,
    baseline_block,
    experience_block,
)

#: Real type names from this corpus, and the count of documents each carries. Any of these
#: appearing in the baseline vocabulary would mean the portable layer had absorbed the local one.
LOCAL_TYPE_NAMES = ("Surgical-Pathology-Document", "Cytology-Report", "Non-Gyn-Cyto-FNA",
                    "FN-Aspirate-Report", "SURG-PATH-RESULT", "Chest-CT-WO-Contr",
                    "Hem-Onc-MD-OP-Progress-Note", "Prescriptions-Filled-RxHub",
                    "Speech-Language-Pathology-Note", "Lung-Bx-W-CT-Guid")


# ------------------------------------------------------------------ portable, not local
def test_no_local_type_name_appears_in_the_baseline_vocabulary():
    """The whole point of a portable concept is that it survives a move to another site."""
    blob = baseline_block()
    for name in LOCAL_TYPE_NAMES:
        assert name not in blob, f"{name!r} is a local type string and must not be in here"


def test_no_concept_name_is_a_local_type_string():
    for c in BASELINE_CONCEPTS:
        assert c.name.islower() and " " not in c.name
        assert "-" not in c.name, f"{c.name!r} looks like a local type name"


# ------------------------------------------------------------------ reference, not a rule
def test_the_block_says_it_restricts_nothing():
    blob = baseline_block().lower()
    assert "reference" in blob
    assert "can be\nopened" in blob or "can be opened" in blob.replace("\n", " ")
    assert "nothing here restricts you" in blob.replace("\n", " ")


def test_the_block_tells_the_model_to_judge_the_names_itself():
    """The replacement for substring matching is the model reading the list."""
    blob = " ".join(baseline_block().split())
    assert "document_type_summary" in blob
    assert "judge each name against these descriptions" in blob
    assert "not on whether the name contains a particular word" in blob


def test_the_block_carries_the_false_positive_and_false_negative_lesson():
    """Both directions of the measured failure, in words the model can act on."""
    blob = " ".join(baseline_block().split())
    assert "speech-language therapy note" in blob.lower()
    assert "never mentions pathology" in blob.lower()


def test_baseline_claims_no_measurement():
    """A prior that arrives unlabelled is indistinguishable from a finding."""
    blob = " ".join(baseline_block().split())
    assert "NOTHING BELOW IS MEASURED" in blob
    assert not re.search(r"\d+(\.\d+)?%", blob), "a percentage in the baseline block is a claim"


def test_no_concept_carries_a_priority_or_a_yield():
    """Priority depends on the field: for histology a CT is inert, for primary site it may be
    the only thing that names the lobe. This project already got that wrong in the other
    direction, coding lung-NOS while `right upper lobe` sat in seven other note types."""
    for c in BASELINE_CONCEPTS:
        assert not hasattr(c, "priority")
        assert not hasattr(c, "rank")
        assert not hasattr(c, "yield_")
    blob = " ".join(baseline_block().split())
    assert "Priority depends on the field" in blob


def test_no_keyword_list_is_supplied_at_baseline():
    """The baseline gives the model no search terms. It has `search` and works them out."""
    blob = baseline_block()
    for term in ("carcinoma", "biopsy", "final diagnosis", "specimen", "adenocarcinoma"):
        assert "terms" not in blob.lower() or term not in blob.lower().split("terms")[-1][:200]
    assert "keyword" not in blob.lower() or "no yields" in blob.lower()


# ------------------------------------------------------------------ what can establish what
def test_only_pathology_can_settle_histology_on_its_own():
    """The clinical contract's own statement, in the form the prompt renders."""
    settles = {c.name for c in BASELINE_CONCEPTS if "histology" in c.can_establish}
    assert settles == {"definitive_pathology"}


def test_imaging_may_support_primary_site_and_never_histology():
    ct = next(c for c in BASELINE_CONCEPTS if c.name == "cross_sectional_imaging")
    assert "primary_site" in ct.may_support
    assert "histology" not in ct.can_establish and "histology" not in ct.may_support


def test_specimen_acquisition_is_kept_apart_from_the_diagnosis():
    """Where tissue was taken is not where the tumour arose — the rule an answer_check used to
    enforce with a `specimen_markers` word list, now stated where the model can weigh it."""
    sa = next(c for c in BASELINE_CONCEPTS if c.name == "specimen_acquisition")
    assert not sa.can_establish
    assert "not the same claim as where the tumour arose" in " ".join(sa.means.split())


# ------------------------------------------------------------------ the experience tier
def test_there_is_no_certified_experience_asset_yet():
    """The honest baseline. The pilot that injected an UNCERTIFIED list scored 3/10 against
    native planning's 4/10, and that list had already been measured at 87.4% recall."""
    assert experience_block(None) == ""
    assert experience_block({}) == ""


def test_a_supplied_asset_renders_with_its_provenance_and_stays_declinable():
    asset = {
        "asset_id": "lung-site-histology-retrieval", "version": "0.1.0", "status": "draft",
        "measured": "held-out 200 patients, 2026-08-01",
        "queries": [{"id": "definitive-histology", "field": "histology",
                     "terms": ["final diagnosis", "addendum"],
                     "measured_yield": "witness in 91% of charts, median 2 reads"}],
    }
    blob = experience_block(asset)
    assert "lung-site-histology-retrieval" in blob and "0.1.0" in blob
    assert "held-out 200 patients" in blob, "the measurement must travel with the numbers"
    assert "91%" in blob
    assert "not a rule and not a checklist" in " ".join(blob.split())
    assert "depart from it and say in your reasoning" in " ".join(blob.split())


def test_an_empty_concept_list_renders_nothing_rather_than_an_empty_heading():
    assert baseline_block([]) == ""


def test_a_concept_renders_what_it_can_and_cannot_settle():
    c = DocumentConcept(name="x_concept", means="a  document   kind",
                        can_establish=("a",), may_support=("b",))
    out = c.render()
    assert "a document kind" in out, "whitespace is collapsed"
    assert "can establish on its own: a" in out
    assert "may support but not settle: b" in out
