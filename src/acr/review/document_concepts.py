"""Portable document concepts, offered to the model as reference. Never as a partition.

WHAT THIS IS AND WHAT IT REPLACED
---------------------------------
A stratum used to select its documents with `doc_type_matches: ["Pathology", "Cytology"]`,
evaluated as a case-insensitive substring over local type names, and the result was an INPUT TO
A GATE and a bar on what the agent could read. On this corpus that expression matched
`Speech-Language-Pathology-Note` and missed `Non-Gyn-Cyto-FNA` (1,285 documents),
`FN-Aspirate-Report` (881) and `SURG-PATH-RESULT` (231); 107 of the 219 patients whose
`can_establish` count is zero hold one of those reports anyway.

What is left is this: a small vocabulary of PORTABLE concepts, each with prose saying what the
document is, handed to the model beside the patient's own type list. The model reads both and
decides what to open. Nothing here filters a search, blocks a read, or conditions an answer.

TWO TIERS, AND THE LINE BETWEEN THEM IS PROVENANCE
--------------------------------------------------
`BASELINE_CONCEPTS` is what exists before any development work: standard names and definitions,
derived from what the clinical contract already says a document must do to establish a field.
Nothing in it is measured. It carries no yields, no orderings, no keyword lists, and it says so
out loud, because a prior that arrives unlabelled is indistinguishable from a finding.

Anything stronger -- which concepts actually pay off on this corpus, which terms find the answer
first, what the marginal yield of each is -- is a RETRIEVAL EXPERIENCE asset and exists only
after a development set has been scanned and the result certified on held-out patients. There is
no such asset in this tree yet. `docs/SEARCH_PLANNING_PILOT.md` records the attempt that was
made without one: the up-front plan it tested was the five spec-derived keywords
`pathology, biopsy, final diagnosis, specimen, carcinoma`, which had already been measured at
87.4% recall over 276,054 documents -- missing an answer-bearing document for 31.7% of patients
because the list has `carcinoma` and not `cancer`. That arm scored 3/10 against native
planning's 4/10, and the negative result says nothing about priors in general: it says an
uncertified list does not help.

So `experience_block()` returns nothing until such an asset is passed in, and when one is it is
rendered under its own heading with its provenance and measurement attached. A certified prior
is still reference: the model may decline it, and declining is recorded rather than refused.

WHY THE PRIORS ARE NOT AN ORDERING
----------------------------------
The obvious shape for this file is a ranked list -- read pathology first, then operative notes,
then imaging. Deliberately not that. Priority depends on the field being answered: for histology
a CT is inert, and for primary site the same CT localises the tumour, which is a distinction this
project already got wrong once in the other direction (a patient coded lung-NOS while "right
upper lobe" sat in seven imaging and oncology note types). Each concept therefore says what it
CAN ESTABLISH, and the ordering is the model's to derive per field.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentConcept:
    """One portable document concept: standard name, what it is, what it can settle."""

    name: str
    means: str
    #: Field names this kind of document can establish ON ITS OWN. Prose in the clinical
    #: contract is the authority; this is the same statement in a form the prompt can render.
    can_establish: tuple[str, ...] = ()
    #: Fields it can contribute to but not settle — a restatement, a localisation, a date.
    may_support: tuple[str, ...] = ()

    def render(self) -> str:
        bits = [f"- {self.name}: {' '.join(self.means.split())}"]
        if self.can_establish:
            bits.append(f"    can establish on its own: {', '.join(self.can_establish)}")
        if self.may_support:
            bits.append(f"    may support but not settle: {', '.join(self.may_support)}")
        return "\n".join(bits)


#: The baseline vocabulary. STANDARD NAMES ONLY — no local type string appears anywhere in it,
#: which is what makes it portable to another site, and no entry carries a measurement.
BASELINE_CONCEPTS: tuple[DocumentConcept, ...] = (
    DocumentConcept(
        name="definitive_pathology",
        means=("a report in which a pathologist or cytopathologist states a diagnosis from "
               "tissue or cells, together with its addenda and amendments — the later addendum "
               "supersedes an interim line"),
        can_establish=("histology", "behavior"),
        may_support=("primary_site",),
    ),
    DocumentConcept(
        name="specimen_acquisition",
        means=("a procedure or image-guided report describing how a specimen was obtained. It "
               "records where tissue was taken from, which is not the same claim as where the "
               "tumour arose, and it often does not carry the diagnosis at all"),
        may_support=("primary_site",),
    ),
    DocumentConcept(
        name="operative_localization",
        means=("an operative or endoscopic report describing what the operator found and where "
               "they were — usually the most explicit statement of anatomic origin in a chart"),
        may_support=("primary_site",),
    ),
    DocumentConcept(
        name="cross_sectional_imaging",
        means=("cross-sectional or functional imaging that localises a mass and describes its "
               "extent. It cannot say what a tumour is, and it frequently names the lobe when "
               "the pathology says only which lung"),
        may_support=("primary_site",),
    ),
    DocumentConcept(
        name="specialist_assessment",
        means=("an oncology, pulmonary or tumour-board note in which a clinician states the "
               "working diagnosis and stage. A restatement of a diagnosis made elsewhere, so it "
               "points at the source rather than being one"),
        may_support=("primary_site", "histology", "behavior"),
    ),
    DocumentConcept(
        name="molecular_or_ancillary",
        means=("immunohistochemistry, molecular and cytogenetic reports. Often the document that "
               "resolves a diagnosis a pathologist left open pending stains"),
        may_support=("histology",),
    ),
    DocumentConcept(
        name="administrative",
        means=("medication fills, tracings, scheduling and instruction documents. Usually inert "
               "for a diagnosis question — but 'usually' is a prior, not a rule, and nothing "
               "stops you opening one"),
    ),
)


_BASELINE_HEADER = """\
DOCUMENT CONCEPTS — REFERENCE, NOT INSTRUCTIONS

These are standard descriptions of the KINDS of document a chart holds. They are not this
site's type names, and nothing here restricts you: every document in this patient's chart can
be opened, searched and read whatever kind it is.

Use them like this. Call chart.document_type_summary to see the type names this patient
actually has, then judge each name against these descriptions — on what the document IS and who
wrote it, not on whether the name contains a particular word. A name containing "pathology" can
be a speech-language therapy note; a name that never mentions pathology can be a
cytopathologist's diagnosis. This site names diagnosis-bearing reports many ways.

Priority depends on the field you are answering, so it is not listed here: for histology an
imaging report is inert, and for primary site the same report may be the only thing that names
the lobe. Derive the order yourself, per field.

NOTHING BELOW IS MEASURED. It is a description of clinical document kinds, not a finding about
this corpus, and it carries no yields or orderings for that reason.
"""


def baseline_block(concepts: Sequence[DocumentConcept] | None = None) -> str:
    """The reference block for the prompt. Advisory text, and it says so in its own header."""
    cs = list(concepts if concepts is not None else BASELINE_CONCEPTS)
    if not cs:
        return ""
    return _BASELINE_HEADER + "\n" + "\n".join(c.render() for c in cs)


def experience_block(asset: dict | None) -> str:
    """A certified retrieval-experience asset, rendered with its provenance attached.

    Returns "" when there is no asset, which is the state this tree is in: no measured prior has
    been certified on held-out patients yet. An empty return is the honest baseline, not a gap to
    be filled with the spec's keyword list — that list is the one already measured at 87.4%
    recall, and the pilot that injected it scored below native planning.

    The provenance travels with the numbers on purpose. A prior whose measurement is not shown
    beside it reads exactly like a rule, and the model cannot weigh a rule.
    """
    if not asset:
        return ""
    lines = ["RETRIEVAL EXPERIENCE — MEASURED, AND STILL REFERENCE", ""]
    lines.append(f"asset: {asset.get('asset_id', '(unnamed)')} "
                 f"v{asset.get('version', '?')}  status: {asset.get('status', 'draft')}")
    if measured := asset.get("measured"):
        lines.append(f"measured on: {measured}")
    lines += ["",
              "This was derived from a development set and is offered because it paid off "
              "there. It is not a rule and not a checklist: if it does not fit this chart, "
              "depart from it and say in your reasoning that you did.", ""]
    for q in (asset.get("queries") or []):
        terms = ", ".join(q.get("terms") or [])
        lines.append(f"- {q.get('id', '?')} (field {q.get('field', '?')}): {terms}")
        if yield_ := q.get("measured_yield"):
            lines.append(f"    measured yield: {yield_}")
    for c in (asset.get("document_concepts") or []):
        lines.append(f"- concept {c.get('concept', '?')}: {c.get('note', '')}")
    return "\n".join(lines)


def concepts_manifest(concepts: Sequence[DocumentConcept] | None = None) -> dict:
    """The identity of the document-concept reference a run was shown.

    Hashed over the rendered block, not over the dataclass list, because the block is what the
    model read: its header carries the instruction that none of this restricts the model, and
    editing that header changes the run.
    """
    import hashlib
    cs = list(concepts if concepts is not None else BASELINE_CONCEPTS)
    block = baseline_block(cs)
    return {"n_concepts": len(cs), "names": [c.name for c in cs],
            "bytes": len(block.encode("utf-8")),
            "content_hash": hashlib.sha256(block.encode("utf-8")).hexdigest()[:16],
            "measured": False}


_ANCHOR_BLOCK = """\
WHICH TUMOUR THIS ANSWER IS ABOUT

The question asks about "the tumour being reported". A chart can document more than one
neoplasm: a second primary, a metastasis, a prior resected tumour, a benign finding, or the same
tumour described twice in different words. Nothing in the chart marks which one the registry is
asking about, so you have to decide, and you have to say what you decided.

Before you answer:

1. List every distinct neoplasm or mass the chart documents, with its site and laterality and
   the note that names it. Put that in `lesions_considered` -- one entry each, even when there
   is only one.
2. Say in `reported_lesion` which entry you answered for, and why each other entry is not the
   reportable primary.
3. If two entries cannot be resolved into one reportable tumour, say that in `reported_lesion`
   instead of picking one silently. Two lesions in different lobes may be one tumour crossing a
   boundary, two separate primaries, or a primary and its metastasis, and those code
   differently.

A benign finding is not a candidate. Neither is a metastasis: the reportable tumour is the
primary, coded to the organ it arose in.

These two are a record of your reasoning, not a scored answer. Nothing refuses an answer over
them.
"""


def anchor_block() -> str:
    """The reporting-unit instruction. See `_ANCHOR_BLOCK`.

    Separate from `baseline_block` because it is about WHICH tumour, not about which documents,
    and the two were being confused: document concepts tell the model where to look, and this
    tells it what it is looking for. Measured need -- three runs answered about the wrong
    neoplasm, and in none of them could the trace say why.
    """
    return _ANCHOR_BLOCK
