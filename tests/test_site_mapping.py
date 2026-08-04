"""Site Mapping: the model classifies local type names, and the result is frozen.

The defect these tests fence in is measured, not hypothetical. `["Pathology", "Cytology"]` as
a case-insensitive substring over this corpus's 1,516 type names matched
Speech-Language-Pathology-Note and missed Non-Gyn-Cyto-FNA, FN-Aspirate-Report and
SURG-PATH-RESULT, so 107 patients holding a cytology diagnosis were stratified as holding
nothing that could establish histology. `test_substring_expression_that_was_replaced` pins
that measurement so nobody reintroduces the expression believing it worked.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from acr.chartstore.corpus import DocMeta
from acr.contract.site_mapping import (
    UNMAPPED,
    Concept,
    SiteMapping,
    SiteMappingError,
    TypeAssignment,
    build_site_mapping,
    concepts_from_strata,
    concepts_hash,
)
from acr.review.coverage import StratumSpec, assign_strata, unmapped_doc_types

PATH_MEANS = ("a report in which a pathologist or cytopathologist states a diagnosis "
              "from tissue or cells, including its addenda and amendments")
NOTE_MEANS = "a clinician's narrative note that may restate a diagnosis made elsewhere"


def strata(*, mapped: bool = True) -> list[StratumSpec]:
    can = ({"name": "can_establish", "policy": "exhaustive", "means": PATH_MEANS}
           if mapped else
           {"name": "can_establish", "policy": "exhaustive",
            "match": {"doc_type_matches": ["Pathology", "Cytology"]}})
    note = ({"name": "may_mention", "policy": "search_then_read_hits_and_sample_misses",
             "means": NOTE_MEANS}
            if mapped else
            {"name": "may_mention", "policy": "search_then_read_hits_and_sample_misses",
             "match": {"doc_type_matches": ["Progress-Note", "Consult"]}})
    return [StratumSpec.from_dict(can),
            StratumSpec.from_dict(note),
            StratumSpec.from_dict({"name": "cannot_establish", "policy": "validate_by_sampling",
                                   "match": {"rest": True}})]


def mapping_for(assignments: dict[str, str], specs=None) -> SiteMapping:
    specs = specs or strata()
    concepts = concepts_from_strata(specs)
    return SiteMapping(
        corpus_id="acr_real", concepts=tuple(concepts),
        bound_concepts_hash=concepts_hash(concepts),
        assignments={t: TypeAssignment(t, c, why="test", n_documents=1)
                     for t, c in assignments.items()},
        model="test-model", built_at="2026-07-30T00:00:00Z")


def docs(*types: str) -> list[DocMeta]:
    return [DocMeta(f"n{i}", t, date(2020, 1, 1 + i), 1, 100) for i, t in enumerate(types)]


# --------------------------------------------------------------- the measured defect
def test_substring_expression_that_was_replaced():
    """The real type names the retired expression got wrong, both directions.

    Not a style complaint. Each name below is a real document type in this corpus and the
    counts in `acr.contract.site_mapping` say how many documents each one carries.
    """
    pats = ["pathology", "cytology"]
    def hit(t):
        return any(p in t.lower() for p in pats)

    for missed in ("Non-Gyn-Cyto-FNA", "FN-Aspirate-Report", "SURG-PATH-RESULT",
                   "Fine-Needle-Aspiration", "Microscopic-Observation-ID-Cyto-Stain"):
        assert not hit(missed), f"{missed} would have matched; the measurement says it did not"
    assert hit("Speech-Language-Pathology-Note"), "the false positive is part of the record"


def test_mapping_places_the_documents_the_substring_missed():
    specs = strata()
    m = mapping_for({"Non-Gyn-Cyto-FNA": "can_establish",
                     "FN-Aspirate-Report": "can_establish",
                     "SURG-PATH-RESULT": "can_establish",
                     "Speech-Language-Pathology-Note": UNMAPPED,
                     "Hem-Onc-MD-OP-Progress-Note": "may_mention"}, specs)
    got = assign_strata(docs("Non-Gyn-Cyto-FNA", "FN-Aspirate-Report", "SURG-PATH-RESULT",
                             "Speech-Language-Pathology-Note", "Hem-Onc-MD-OP-Progress-Note"),
                        specs, m)
    assert [d.doc_type for d in got["can_establish"]] == [
        "Non-Gyn-Cyto-FNA", "FN-Aspirate-Report", "SURG-PATH-RESULT"]
    assert [d.doc_type for d in got["may_mention"]] == ["Hem-Onc-MD-OP-Progress-Note"]
    # The speech-language note is UNMAPPED, so it lands in the spec's declared fallback and
    # NOT in the stratum whose name a substring would have put it in.
    assert [d.doc_type for d in got["cannot_establish"]] == ["Speech-Language-Pathology-Note"]


# --------------------------------------------------------------- fail closed
def test_mapped_stratum_refuses_to_stratify_without_a_mapping():
    """No mapping is not "no strata". It is a coverage proof nobody performed.

    Without the refusal every document falls to `rest`, `can_establish` is empty, and
    `evaluate_gate` reads an empty exhaustive stratum as a completed one.
    """
    with pytest.raises(SiteMappingError) as e:
        assign_strata(docs("Non-Gyn-Cyto-FNA"), strata())
    assert "can_establish" in str(e.value)


def test_means_never_falls_back_to_doc_type_matches():
    """A stratum that declares both does NOT get the substring list as a safety net.

    A fallback that runs only when the mapping is missing or incomplete is the same defect at
    a lower firing rate, and it fires precisely when the mapping is broken.
    """
    s = StratumSpec.from_dict({"name": "can_establish", "policy": "exhaustive",
                               "means": PATH_MEANS,
                               "match": {"doc_type_matches": ["Pathology"]}})
    m = mapping_for({}, [s])
    assert s.is_mapped
    assert not s.matches(docs("Surgical-Pathology-Report")[0], m)


def test_legacy_substring_stratum_still_works_untouched():
    """The three unmigrated specs keep running; speclint is what flags them."""
    specs = strata(mapped=False)
    assert not specs[0].is_mapped
    got = assign_strata(docs("Surgical-Pathology-Report", "Non-Gyn-Cyto-FNA"), specs, None)
    assert [d.doc_type for d in got["can_establish"]] == ["Surgical-Pathology-Report"]


# --------------------------------------------------------------- unknown vs unmapped
def test_unknown_type_is_not_the_same_answer_as_unmapped():
    specs = strata()
    m = mapping_for({"Chest-CT-WO-Contr": UNMAPPED}, specs)
    assert m.concept_for("Chest-CT-WO-Contr") == UNMAPPED       # judged, no concept fits
    assert m.concept_for("Type-Invented-Last-Tuesday") is None  # never judged at all
    assert unmapped_doc_types(docs("Chest-CT-WO-Contr", "Type-Invented-Last-Tuesday"),
                              specs, m) == ["Type-Invented-Last-Tuesday"]


def test_unmapped_doc_types_is_empty_for_legacy_strata():
    """A legacy stratification has no mapping to be stale against; don't invent a finding."""
    assert unmapped_doc_types(docs("Whatever"), strata(mapped=False), None) == []


# --------------------------------------------------------------- staleness
def test_editing_a_means_invalidates_the_mapping():
    specs = strata()
    m = mapping_for({"FN-Aspirate-Report": "can_establish"}, specs)
    m.require_binds(concepts_from_strata(specs))          # built against these: fine

    edited = [StratumSpec.from_dict({"name": "can_establish", "policy": "exhaustive",
                                     "means": PATH_MEANS + " -- excluding cytology"}),
              specs[1], specs[2]]
    with pytest.raises(SiteMappingError) as e:
        m.require_binds(concepts_from_strata(edited))
    assert "no longer being asked" in str(e.value)


def test_reordering_strata_does_not_invalidate_the_mapping():
    specs = strata()
    m = mapping_for({"FN-Aspirate-Report": "can_establish"}, specs)
    m.require_binds(concepts_from_strata([specs[1], specs[0], specs[2]]))


def test_rest_stratum_contributes_no_concept():
    """`rest` is a destination, not something the model may file documents into directly."""
    names = {c.name for c in concepts_from_strata(strata())}
    assert names == {"can_establish", "may_mention"}


def test_stratum_without_means_or_matches_is_refused_as_a_concept():
    with pytest.raises(SiteMappingError) as e:
        concepts_from_strata([StratumSpec.from_dict({"name": "silent", "policy": "exhaustive"})])
    assert "means" in str(e.value)


# --------------------------------------------------------------- hand-edit detection
def test_hand_edited_mapping_file_is_refused_on_load():
    m = mapping_for({"FN-Aspirate-Report": "can_establish"})
    blob = m.to_dict()
    SiteMapping.from_dict(json.loads(json.dumps(blob)))          # round-trips

    blob["assignments"][0]["concept"] = "may_mention"            # a plausible "small fix"
    with pytest.raises(SiteMappingError) as e:
        SiteMapping.from_dict(blob)
    assert "edited after it was written" in str(e.value)


# --------------------------------------------------------------- the PHI boundary
class _Recorder:
    """An LLM stand-in that keeps every prompt it was handed."""

    def __init__(self, reply):
        self.prompts: list[list[dict]] = []
        self._reply = reply
        self.cfg = type("cfg", (), {"model": "recorder"})()

    def json_chat(self, messages, schema_hint=""):
        self.prompts.append(messages)
        return self._reply(messages)


def _assign_everything(concept: str):
    def reply(messages):
        listing = messages[-1]["content"]
        names = [ln.split("  (")[0].removeprefix("- ")
                 for ln in listing.splitlines() if ln.startswith("- ")]
        return {"assignments": [{"doc_type": n, "concept": concept, "why": "because"}
                                for n in names]}
    return reply


def test_builder_is_given_type_names_and_counts_and_nothing_else():
    """The module docstring claims no patient text crosses in. This is that claim.

    It is what makes one mapping reusable across 1,788 charts and safe to put in a review
    document, so it is asserted rather than intended.
    """
    llm = _Recorder(_assign_everything("can_establish"))
    build_site_mapping({"FN-Aspirate-Report": 881, "Chest-CT-WO-Contr": 900},
                       concepts_from_strata(strata()), llm,
                       corpus_id="acr_real", built_at="2026-07-30T00:00:00Z")
    blob = json.dumps(llm.prompts)
    # Synthetic stand-ins on purpose. `tests/test_no_phi_in_tree.py` refuses a real person_id
    # anywhere in the tree including this file, and it is right to: a leak-detection test that
    # ships the thing it detects is a leak. The shapes are what matter, not the values.
    for leak in ("P01",                              # a patient id
                 "FN-Aspirate-Report_2016-09-19",    # a note id
                 "2016-09-19",                       # a date
                 "Right Upper Lobe"):                # document text
        assert leak not in blob, f"{leak!r} reached the classification prompt"
    assert "FN-Aspirate-Report" in blob and "881" in blob


def test_builder_refuses_a_paraphrased_type_name():
    """A rewritten name maps no document and leaves the mapping looking complete."""
    def reply(messages):
        return {"assignments": [{"doc_type": "FN Aspirate Report",   # spaces, not hyphens
                                 "concept": "can_establish", "why": "x"}]}
    with pytest.raises(SiteMappingError) as e:
        build_site_mapping({"FN-Aspirate-Report": 881}, concepts_from_strata(strata()),
                           _Recorder(reply), corpus_id="c", built_at="t")
    assert "echoed" in str(e.value)


def test_builder_refuses_an_undeclared_concept():
    def reply(messages):
        return {"assignments": [{"doc_type": "FN-Aspirate-Report",
                                 "concept": "definitely_pathology", "why": "x"}]}
    with pytest.raises(SiteMappingError) as e:
        build_site_mapping({"FN-Aspirate-Report": 881}, concepts_from_strata(strata()),
                           _Recorder(reply), corpus_id="c", built_at="t")
    assert "does not declare" in str(e.value)


def test_builder_refuses_a_short_batch_instead_of_defaulting_it():
    """A truncated completion must not become 40 quiet UNMAPPEDs."""
    def reply(messages):
        return {"assignments": [{"doc_type": "A-Report", "concept": "can_establish", "why": "x"}]}
    with pytest.raises(SiteMappingError) as e:
        build_site_mapping({"A-Report": 1, "B-Report": 2}, concepts_from_strata(strata()),
                           _Recorder(reply), corpus_id="c", built_at="t")
    assert "do not default the remainder" in str(e.value)


def test_builder_batches_and_covers_every_name():
    llm = _Recorder(_assign_everything("may_mention"))
    counts = {f"Type-{i:04d}": i for i in range(250)}
    m = build_site_mapping(counts, concepts_from_strata(strata()), llm,
                           corpus_id="acr_real", built_at="t", batch_size=100)
    assert len(llm.prompts) == 3
    assert m.n_types == 250
    assert m.concept_for("Type-0249") == "may_mention"


# --------------------------------------------------------------- review affordance
def test_review_table_puts_unmapped_and_high_volume_rows_first():
    m = SiteMapping(
        corpus_id="c", concepts=(Concept("can_establish", PATH_MEANS),),
        bound_concepts_hash=concepts_hash([Concept("can_establish", PATH_MEANS)]),
        assignments={
            "Small-Type": TypeAssignment("Small-Type", "can_establish", "x", 3),
            "Huge-Type": TypeAssignment("Huge-Type", "can_establish", "x", 3849),
            "Odd-Type": TypeAssignment("Odd-Type", UNMAPPED, "no concept fits", 7),
        },
        model="m", built_at="t")
    assert [r["doc_type"] for r in m.review_table()] == ["Odd-Type", "Huge-Type", "Small-Type"]
