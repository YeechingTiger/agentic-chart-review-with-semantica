"""The corpus findings, and whether each one still describes the spec it is printed beside.

Section 7 is the only part of the review document that carries a number, which makes it the
most persuasive part and the easiest to turn into fiction. A finding is a property of the
exact configuration it was measured on; reprinting it beside a different one would be a
fabricated number wearing a measured number's clothes. Nothing here reads a spec beyond the
term list it is handed, so no other part of the renderer can quietly widen a finding's scope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Measurement:
    """One number from the 2026-07-26 corpus pass, with the configuration it was measured on.

    `applies_when` is what stops section 7 from becoming fiction. The 31.7% miss rate is a
    property of five specific search terms; reprinting it beside a sixth term would be a
    fabricated number wearing a measured number's clothes, and it would be the most
    persuasive sentence in the document.
    """
    key: str
    text: str
    #: Search terms whose presence in the spec makes this finding apply to it.
    needs_terms: tuple[str, ...] = ()
    #: spec_id this was measured on, when the finding is specific to one criterion.
    only_spec: str | None = None
    #: The exact term list measured, when the finding depends on the whole list.
    measured_list: tuple[str, ...] | None = None


CORPUS_HEADER = (
    "Measured on 2026-07-26 over the entire real corpus — 1,787 patients, 276,054 documents, "
    "read exhaustively with no sampling. The comparison was against each patient's known "
    "diagnosis, so a \"miss\" below means a document that states the answer in plain clinical "
    "prose and that this criterion's searches would never have opened."
)

MEASUREMENTS: tuple[Measurement, ...] = (
    Measurement(
        key="shb_keyword_miss",
        only_spec="STORE.400_522_523.site_histology_behavior",
        measured_list=("pathology", "biopsy", "final diagnosis", "specimen", "carcinoma"),
        text=(
            "**The required search terms miss the diagnosis on almost a third of patients.** "
            "Across the corpus, 4,005 progress notes, discharge summaries and consult notes "
            "state the patient's diagnosis in words none of the five required terms would "
            "find. That is 567 of 1,787 patients — **31.7%** — each holding at least one note "
            "that answers the question and that we would never have opened. In the group of "
            "documents we treat as inert, the same check finds 2,743 documents on 517 patients "
            "(28.9%).\n\n"
            "The cause is one missing word. The list has *carcinoma* and not *cancer*. "
            "Pathologists write carcinoma; almost nobody else does. Clinicians write \"small "
            "cell lung cancer\" (1,333 missed documents), \"squamous cell cancer\" (795), "
            "plain \"cancer\" (687), \"NSCLC\" (388), \"non-small cell\" (384), \"carcinoid\" "
            "(165). Adding the single term *cancer* recovers 3,605 of the 4,005.\n\n"
            "The notes carrying the missed answers are the ordinary ones: discharge summaries "
            "(631), general progress notes (624), haematology-oncology outpatient progress "
            "notes (558), primary-care progress notes (471), emergency department notes (447).\n\n"
            "Twelve of these documents were re-read by hand. All twelve stated the diagnosis "
            "and contained none of the five terms — \"scheduled to see oncology today for her "
            "small cell lung cancer\", \"stage iii squamous cell lung cancer\", \"residual "
            "typical carcinoid tumor\"."),
    ),
    Measurement(
        key="stem_pathology",
        needs_terms=("pathology",),
        text=(
            "**Searching for _pathology_ instead of _patholog_ loses 9,697 documents.** "
            "Corpus-wide the stem matches 37,721 documents and the full word 28,024. On 1,531 "
            "of 1,788 charts the stem returns strictly more; on no chart does it return fewer. "
            "The full word is the one written into the required list."),
    ),
    Measurement(
        key="resection_dead",
        needs_terms=("resection",),
        text=(
            "**_resection_ returns nothing at all on 43.1% of charts** — 770 of 1,788 — "
            "because most of these patients were never resected. It is still the fourth most "
            "useful term for rescuing the misses that remain after *cancer* is added."),
    ),
    Measurement(
        key="cytology_yield",
        needs_terms=("cytology",),
        text=("**_cytology_ matches 8,734 documents corpus-wide and none at all on 367 charts.**"),
    ),
    Measurement(
        key="carcinoma_yield",
        needs_terms=("carcinoma",),
        text=(
            "**_carcinoma_ is the single highest-yield term in the list** — on its own it is "
            "the only term that finds 8,325 of the answer-bearing documents — and it is also "
            "the term whose absence of a plain-English twin causes the miss rate above."),
    ),
    Measurement(
        key="final_diagnosis_dead",
        needs_terms=("final diagnosis",),
        text=(
            "**_final diagnosis_ is the only term that finds 6 documents out of 31,725.** It "
            "is a pathology-report heading, and it is required in a group of documents that "
            "contains no pathology reports. Removing it changes nothing."),
    ),
)

#: Where the 12% comes from, and why nothing below it is achievable. Applies to any criterion
#: that reads 25 documents and tolerates no surprises.
SAMPLING_ARITHMETIC = (
    "Reading 25 documents at random and finding nothing relevant supports a residual rate of "
    "**11.3%**, not zero — that is the strongest statement 25 documents can make. Any "
    "tolerance set below 11.3% cannot be met no matter how much work is done, so the number "
    "chosen and the number of documents read are one decision, not two."
)

UNMEASURED_NOTE = (
    "**Nothing in this criterion has been measured against the corpus.** One criterion has "
    "been — site, histology and behaviour — and its required search terms turned out to miss "
    "the stated diagnosis for 31.7% of patients. Silence here is not evidence that this "
    "criterion does better; it is evidence that nobody has looked."
)


def measurement_for(spec, terms: Sequence[str]) -> Measurement | None:
    for m in MEASUREMENTS:
        if m.only_spec == spec.spec_id and m.measured_list is not None:
            return m if tuple(terms) == m.measured_list else None
    return None
