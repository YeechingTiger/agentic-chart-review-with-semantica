"""THE FULL SCAN — one cheap reading of every note in a development set, against ONE requirement.

WHAT THIS ANSWERS
-----------------
Given any development set of patients and any requirement, decide, per note:

  1. does this note bear on the requirement, and in what way (its STANDING) — asked and answered
     ONCE PER FIELD of the requirement, never once for the note;
  2. which terms in it would let a searcher find it for that requirement — ONE list per note, for
     the requirement as a whole.

Aggregated, those two answers become the keyword list and the document-type policy for that
requirement. That is the whole job. This module does not extract an answer, does not hold one,
and does not know what the requirement is about.

WHY STANDING IS PER FIELD AND NOT PER NOTE
------------------------------------------
Because a requirement is multi-field — three fields in one shipped spec, nine in another — and
one verdict for the whole note cannot express the thing that matters most: a document type can be
among the richest sources in the record for one field while being unable to settle another. The
first version of this module collapsed that into a single class per note, and the collapse is
exactly what produced a real mis-coding downstream: a document that could locate a site but not
establish a cell type was filed as establishing, once, for everything.

So the answer to question 1 is a mapping, one verdict per field, in the spec's own field names:

    {"primary_site": "can_establish", "histology": "neither", "behavior": "neither"}

EVERY field of the requirement must appear. A field the model left out is a contract violation
and the reply is refused — it is not read as "neither", because "the reader did not answer" and
"the reader answered no" are different facts and only one of them is a measurement.

THE QUOTE STAYS AT NOTE LEVEL. One verbatim sentence per note, justifying whichever fields were
not "neither" — not one quote per field. A quote per field is a per-field span extraction wearing
a different name, and the per-field extraction was deliberately deleted from this module (see
WHAT IS NOT HERE ANY MORE); it must not creep back in through the evidence slot.

Question 2 is NOT per field. Which fields a term serves is a question the aggregation answers by
looking at the verdicts of the notes that proposed it, and that is a measurement over the corpus;
asking the model to attribute each term to a field would be asking it to guess it.

WHY IT IS NOT WELDED TO A VARIABLE
----------------------------------
It was, once, and every line of it was false the moment the requirement moved: a system prompt
naming one professional role, a hardcoded block of three field descriptions, and a standing
class defined as one clinical sentence. That sentence was never a fact about medicine; it was
one spec's `counts_as_evidence` clause, typed into a prompt where nothing could revise it.

So the requirement now comes from a spec, the way `coverage_planner` already does it:

  * the question is `spec.question`
  * the fields are `spec.fields`
  * AND THE THREE STANDING CLASSES ARE DEFINED BY `spec.evidence_rules`. `can_establish` means
    "satisfies what THIS spec says counts as evidence"; `merely_mentions` means "bears on the
    question but this spec says it cannot establish it"; `neither` means "does not bear".

A spec with no `evidence_rules` therefore cannot be labelled at all, and `Requirement.from_spec`
refuses it rather than quietly falling back on a default that would be somebody's guess at a
clinical rule. Nothing in the prompt-building path names a disease, a document vocabulary, a
coding system or an institution; `tests/test_labelling.py` fails if one appears.

WHY THE STANDING QUESTION IS WORTH A MODEL CALL
-----------------------------------------------
Because it is invisible to grep. The same sentence can establish the answer in the document that
rendered it and establish nothing in the document that copied it forward, and no keyword, count
or filename can tell those two apart. Everything else about a note — its type, its date, which
needles it matches — is a pure function of text that costs nothing to compute. This is the one
axis that is not.

WHY THE RETRIEVAL-TERMS QUESTION EXISTS
---------------------------------------
A greedy optimiser that ranked candidate keywords by their FREQUENCY in answer-bearing notes was
run over 1,770 real patients. The first term it adopted was "patient", then "note", then "date",
then "with"; it reached recall 0.9998 by surfacing 99.7% of the corpus, which is to say it
rediscovered reading everything. Frequency cannot separate a word that OCCURS IN answer-bearing
notes from a word that INDICATES the answer, because that separation is not in the counts.

A model holding the requirement can make it. Shown one note and asked which of ITS terms would
let a searcher find it FOR THIS REQUIREMENT, it offered a drug name standing in for a diagnosis
— which no word count would ever surface and no human writing a keyword list from imagination
would think to include. (The observed run: "SCLC" and "etoposide", on a lung note. Recorded here
as the measurement that justifies the question, not as anything this module knows.)

Every term that comes back is verified against the note text in code: a term the note does not
contain is a needle that matches nothing, and it is dropped and counted rather than stored. The
one quote is checked the same way, for the same reason.

WHAT A LABEL IS CONDITIONED ON
------------------------------
The requirement and the prompt wording, both of which ride on every row. Point this at a
different requirement and both answers change — standing is standing ON something, and "which
words would find this note" is meaningless without saying find it FOR WHAT. So `spec_id` and
`prompt_hash` are fields on `NoteLabel` and both are hashed into the store's run key: two
requirements over one corpus land in two directories and can never silently share a labelling.

A note's stratum and its keyword hits are pure functions of `doc_type` and the note text under
whatever asset is being tested, so they are computed downstream by the experiment that cares
rather than frozen here.

WHAT IS NOT HERE ANY MORE
-------------------------
A per-field value extraction, and the apparatus that kept an answer key away from it. The
extraction was never wanted: aggregating it produced a second, worse copy of a number the
registry already holds, while the two questions above produce assets nothing else can. With the
extraction gone there is no answer in this module for a prompt to leak, so the truth-isolation
types went with it — see `audit_relevance` below, which is the one place an answer key still
meets these labels, deliberately after the fact and outside the labelling call.

PHI
---
A label carries a person_id, a note id (which is a document type and a DATE) and one verbatim
quote. `labels_root` therefore refuses any path inside the repository, and label files are
written 0600 inside 0700 directories.

MIXING TWO GENERATIONS OF LABEL
-------------------------------
A per-note labelling made before this change and a per-field labelling made after it are answers
to differently shaped questions, and averaging them would be silent. Three things prevent it:
`PROMPT_VERSION` is bumped, so `prompt_hash` moves and the two land in different directories;
`NoteLabel.from_dict` raises `LabelShapeError` — not one of the exceptions a torn line is
forgiven under — when it is handed a row carrying the old single `verdict`; and `LabelStore.load`
refuses a row whose verdicts do not name exactly the fields its own requirement declares.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..chartstore.corpus import Corpus, DocMeta
from ..core import site
from ..core.repo_paths import repo_root

#: Azure credentials for the reading model. Read as a FILE, never sourced as a shell script:
#: this module wires the deployment up and must not be able to execute what is in there.
AZURE_ENV_PATH = site.MODEL_ENV_FILE or ""

#: Where labellings live. Outside the tree, deliberately — see the PHI note above.
DEFAULT_LABELS_ROOT = str(site.LABELS_ROOT)
LABELS_ROOT_ENV = "ACR_DEVLABELS_ROOT"

#: The deployment this module was costed against: USD per 1M tokens, published rate.
DEPLOYMENT = "gpt-5.6-luna"
USD_PER_1M_INPUT = 1.0
USD_PER_1M_OUTPUT = 6.0

_REPO_ROOT = repo_root()


def cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * USD_PER_1M_INPUT + completion_tokens * USD_PER_1M_OUTPUT) / 1e6


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class LabellingError(Exception):
    """Base for everything this module refuses to do."""


class NotALabellableSpecError(LabellingError):
    """A spec cannot define the three standing classes, so nothing here can label against it."""


class PromptContractError(LabellingError):
    """The model's reply did not satisfy the response contract."""


class LabelShapeError(LabellingError):
    """A label on disk, or in hand, is not the shape this module writes.

    Deliberately NOT a `TypeError` or a `ValueError`: `LabelStore.load` forgives those, because a
    half-written final line is the normal shape of a killed job. A per-note labelling from before
    standing became per-field is not a torn line — it is a complete answer to a different
    question, and dropping it quietly would leave a run that had silently read half a file.
    """


class NotConfiguredError(LabellingError):
    """Azure credentials were asked for and are not present."""


# ============================================================================
# THE REQUIREMENT — everything the prompt says about the subject matter
# ============================================================================

@dataclass(frozen=True, slots=True)
class NoteForReading:
    """One note, as the reading model sees it.

    Frozen and slotted so that nothing can be attached to it on the way to a prompt: the only
    fields a prompt may interpolate are the five below, and a caller who wants to smuggle a
    sixth has to change this class in a diff somebody reads.
    """

    patient_id: str
    note_id: str
    doc_type: str
    date: str
    text: str


def _clauses(value: Any) -> tuple[str, ...]:
    """Any evidence-rule value -> lines, without inventing structure it does not have.

    Specs write these as strings, as lists of strings, and as nested mappings, and this module
    has no business normalising a clinical rule into a shape its author did not choose. So a
    leaf becomes one line and everything else is rendered as JSON, which is lossless and ugly
    rather than tidy and lossy.
    """
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, Mapping):
        return tuple(f"{k}: {c}" for k, v in value.items() for c in _clauses(v))
    if isinstance(value, (list, tuple, set)):
        return tuple(c for v in value for c in _clauses(v))
    return (json.dumps(value, sort_keys=True, default=str),) if value is not None else ()


@dataclass(frozen=True)
class Requirement:
    """The whole of what this module knows about the subject matter, taken from a spec.

    Held as a value object rather than read off a spec at prompt time for two reasons. It is
    HASHABLE, so a labelling can be keyed by the requirement it answers and two requirements
    over one corpus cannot share a file. And it is SMALL and inspectable, so a test can build
    one out of a fixture about anything at all — rent arrears, a shipping manifest — and prove
    that no clinical content is coming from this module.
    """

    spec_id: str
    question: str
    #: (name, description) per output field, in the spec's own order. NOT optional: standing is
    #: answered per field, so a requirement with no fields has nowhere to put an answer.
    fields: tuple[tuple[str, str], ...] = ()
    #: (clause name, statements) — `spec.evidence_rules`, flattened but never rewritten. The
    #: clause NAMES come from the spec too: `counts_as_evidence` is that spec's word, not ours.
    evidence_rules: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def __post_init__(self) -> None:
        if not self.spec_id.strip():
            raise NotALabellableSpecError(
                "a requirement needs a spec_id: it goes on every label, and labels whose "
                "requirement is unnamed cannot be told apart from another requirement's.")
        if not self.question.strip():
            raise NotALabellableSpecError(
                f"{self.spec_id} has no question. 'Does this note bear on the question' has no "
                "meaning without one, and neither does 'which terms would find it'.")
        names = [n.strip() for n, _ in self.fields]
        if not names or not all(names):
            raise NotALabellableSpecError(
                f"{self.spec_id} declares no usable fields ({[n for n, _ in self.fields]}). "
                "Standing is answered PER FIELD — one verdict for a whole note is the collapse "
                "this module exists to avoid — so a requirement with no field names has nowhere "
                "for an answer to land, and there is nothing to ask about it.")
        if len(set(names)) != len(names):
            raise NotALabellableSpecError(
                f"{self.spec_id} repeats a field name ({names}). Verdicts come back keyed by "
                "name, so two fields sharing one would silently collapse into a single answer.")
        if not any(stmts for _, stmts in self.evidence_rules):
            raise NotALabellableSpecError(
                f"{self.spec_id} declares no evidence_rules, so it does not say what would "
                "establish its answer. The three standing classes are defined FROM that clause; "
                "with it empty, 'can_establish' would mean whatever the model felt like, and "
                "this module will not supply a default clinical rule of its own.")

    @classmethod
    def from_spec(cls, spec: Any) -> Requirement:
        """Build from anything with `spec_id`, `question`, `fields` and `evidence_rules`.

        Duck-typed on purpose, exactly as `coverage_planner.plan_coverage` is: this module must
        work for a requirement nobody has written an `ExtractionSpec` for yet.
        """
        rules = getattr(spec, "evidence_rules", None) or {}
        return cls(
            spec_id=str(getattr(spec, "spec_id", "") or "").strip(),
            question=str(getattr(spec, "question", "") or "").strip(),
            fields=tuple((str(getattr(f, "name", "") or "").strip(),
                          str(getattr(f, "description", "") or "").strip())
                         for f in (getattr(spec, "fields", None) or ())),
            evidence_rules=tuple((str(k), _clauses(v)) for k, v in sorted(rules.items())
                                 if _clauses(v)),
        )

    @property
    def field_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.fields)

    @property
    def verdict_skeleton(self) -> str:
        """The shape question 1's answer must come back in, spelled with THIS spec's field names.

        Rendered rather than described because a field the model omits is a refused reply: the
        cheapest way to get every field back is to show every field's key in the template it is
        being asked to fill.
        """
        return "{" + ", ".join(f'"{n}": "<VERDICT>"' for n in self.field_names) + "}"

    def render(self) -> str:
        """The requirement as the model is shown it. Every line of it came from the spec."""
        out = [f"THE QUESTION THIS LABELLING SERVES (spec {self.spec_id}):", self.question]
        if self.fields:
            out += ["", ("THE FIELDS THAT QUESTION MUST ANSWER — question 1 below is asked "
                         "separately about EACH of them:")]
            out += [f"  - {n}: {d}" if d else f"  - {n}" for n, d in self.fields]
        out += ["", ("WHAT THIS QUESTION COUNTS AS EVIDENCE — the spec's own rules, verbatim, "
                     "and the ONLY definition of 'establish' in force here:")]
        for name, stmts in self.evidence_rules:
            out.append(f"  {name}:")
            out += [f"    - {s}" for s in stmts]
        return "\n".join(out)

    @property
    def hash(self) -> str:
        """Identity of the requirement, over exactly what is rendered into the prompt."""
        return hashlib.sha256(self.render().encode()).hexdigest()[:16]


# ============================================================================
# THE PROMPT — one note in, two answers out. NO CLINICAL CONTENT BELOW THIS LINE.
# ============================================================================

#: Bump when the wording changes. `prompt_hash()` moves on its own; this is the human-readable
#: version that appears in a write-up.
PROMPT_VERSION = "labelling/4"

SYSTEM_PROMPT = (
    "You are reading ONE document in isolation, on behalf of someone who holds the requirement "
    "stated below and has not read this document. You do not know the answer to that "
    "requirement for this patient and you must not guess it from outside the document. Judge "
    "only what THIS document is and what it says, in its own words. "
    "Reply with a single JSON object and nothing else."
)

#: The two questions. Every clinical word the model sees arrives through `{requirement}`, which
#: is rendered from the spec; nothing in this template names a disease, an organ, a document
#: vocabulary or a coding system, because the previous version did and every one of those words
#: was a lie the moment the requirement moved.
NOTE_PROMPT_TEMPLATE = """\
DOCUMENT TYPE: {doc_type}    DOCUMENT DATE: {date}

--- BEGIN DOCUMENT ---
{text}
--- END DOCUMENT ---

{requirement}

QUESTION 1 — WHAT STANDING DOES THIS DOCUMENT HAVE, FIELD BY FIELD?

Answer SEPARATELY FOR EVERY FIELD listed above, under that field's exact name. One verdict for the
whole document is not wanted and will not do: a document can be among the best sources in the
record for one field and unable to settle another, and a single answer cannot say so.

Standing is a property of the DOCUMENT, not only of its wording. The same sentence can establish
an answer in the document that first rendered it and establish nothing in a document that copied
it forward. Judge which of those you are holding, once per field:

  "can_establish"    FOR THIS FIELD, this document contains something that satisfies the evidence
                     rules above, rendered in THIS document by whoever was entitled to render it.
  "merely_mentions"  FOR THIS FIELD it bears on the question — it restates, carries forward,
                     refers to, plans around or argues against an answer — but by the rules above
                     THIS document cannot establish it.
  "neither"          FOR THIS FIELD the document does not bear on the question at all.

All {n_fields} field name(s) must appear, spelled exactly: {field_list}. Omitting one is not the
same as answering "neither" for it: a reply that leaves a field out is refused, not filled in.

THEN ONE QUOTE FOR THE WHOLE DOCUMENT, NOT ONE PER FIELD. A single sentence, copied character for
character out of the text above, showing why the fields you did not call "neither" are not
"neither". Quotes are checked against the document automatically and an unverifiable one is
recorded as such. If every field is "neither", leave the quote empty.

QUESTION 2 — WHICH TERMS IN THIS DOCUMENT WOULD LET A SEARCHER FIND IT? A retrieval question
about the requirement above. Someone who holds that requirement but has NOT read this document
must search every document of every patient by text. Which terms, AS THIS DOCUMENT SPELLS THEM,
would surface it?

  * Copy each term character for character out of the document; terms are checked against the
    text automatically, and one that is not there is discarded.
  * At most {max_terms} terms, each at least {min_chars} characters. Fewer is better; do not pad.
  * Nothing that would also match most documents of most patients — "patient", "note", "date",
    "history". A term that matches everything retrieves everything, which is not retrieval.
  * Offer a term because it INDICATES the answer this requirement asks for, not because it is
    frequent: a word that keeps company with the answer is worth less than a rare one that names
    it, or names the instrument that produced it.
  * Tag each term with exactly ONE reason class:
    {reason_classes}

Reply with exactly this JSON, where each TERM is {{"term": "", "reason": "<one reason class>"}}
and every <VERDICT> is one of: {verdict_classes}
{{"admissibility": {{"verdicts": {verdict_skeleton}, "quote": ""}},
 "retrieval_terms": [TERM, ...]}}"""

ADMISSIBILITY_VERDICTS = ("can_establish", "merely_mentions", "neither")
BEARS_ON_QUESTION = ("can_establish", "merely_mentions")

#: A reason class is demanded so that a term arrives as a reviewable CLAIM about why it works
#: rather than as a bare word: "<term>/names_the_section" is a proposition somebody can reject,
#: while the same term on a list is a number nobody can argue with.
#:
#: These four are deliberately about the ROLE a word plays in a document, not about what it
#: means. An earlier set classified terms by subject matter and could not be pointed at a
#: requirement outside one disease. They are also, on purpose, the same four `derive.py`
#: aggregates under `REASON_CLASSES`; the two modules are coupled by this vocabulary, and
#: `tests/test_labelling.py` fails if they drift apart.
TERM_REASONS = ("names_the_answer", "names_the_document", "names_the_section", "other")


@dataclass(frozen=True)
class TermConfig:
    """The bounds on question 2. Both are REQUIRED: `TermConfig()` is a TypeError.

    Neither gets a default, here or ever. A cap typed in today is a decision about how much a
    model may pad, made by whoever happened to write the line, in a commit nobody will reread;
    it belongs to the first real run and is recorded beside the labels it shaped.
    """

    #: Terms kept per note. The cap is what makes question 2 a RANKING task: a model allowed to
    #: list everything lists everything, and then term frequency has quietly won again.
    max_terms_per_note: int
    #: Shortest acceptable term. "a" occurs in every note ever written and would verify as
    #: present in all of them — the exact shape of the failure this whole question exists to
    #: avoid, arriving through the verifier instead of through a word count.
    min_term_chars: int

    def __post_init__(self) -> None:
        if self.max_terms_per_note < 1 or self.min_term_chars < 1:
            raise ValueError(f"need max_terms_per_note >= 1 and min_term_chars >= 1, got {self}")


def build_note_prompt(note: NoteForReading, *, requirement: Requirement,
                      terms: TermConfig) -> list[dict]:
    """One note and one requirement -> the chat messages that read it.

    The only prompt builder in this module, and the signature is the contract: a note, the
    requirement it is being read against, and the bounds on question 2. No `context` and no
    `hints` — a channel for free prose is a channel for somebody's clinical opinion to reach
    the model without passing through a spec that has provenance on it.
    """
    if type(note) is not NoteForReading:
        raise LabellingError(
            f"build_note_prompt takes a NoteForReading, got {type(note).__name__}: the only type "
            "on the reading path, and the only one whose fields are known not to carry anything "
            "but the note.")
    if type(requirement) is not Requirement:
        raise LabellingError(
            f"build_note_prompt takes a Requirement, got {type(requirement).__name__}: build one "
            "with Requirement.from_spec(spec) so that the refusals in its constructor run.")
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": NOTE_PROMPT_TEMPLATE.format(
            doc_type=note.doc_type, date=note.date, text=note.text,
            requirement=requirement.render(), max_terms=terms.max_terms_per_note,
            min_chars=terms.min_term_chars, reason_classes=" | ".join(TERM_REASONS),
            n_fields=len(requirement.field_names),
            field_list=", ".join(requirement.field_names),
            verdict_classes=" | ".join(ADMISSIBILITY_VERDICTS),
            verdict_skeleton=requirement.verdict_skeleton)},
    ]


def prompt_hash(requirement: Requirement, terms: TermConfig) -> str:
    """Identity of everything that conditions a label, independent of any note.

    Hashing a rendered prompt would give a different hash per note and identify nothing.
    Hashing the template, the requirement and the bounds identifies the three things that
    actually condition the answers, so two runs can be compared — or refused comparison — on
    the evidence.

    All three arguments are required and none has a default. `requirement` because both answers
    are answers ABOUT it. `terms` because the cap and the length floor are rendered into
    question 2 and change what comes back: leaving them out would let a scan capped at 3 terms
    and a scan capped at 30 share a file, and the resulting term list would be a mixture with
    nothing on any row to say so.
    """
    blob = "\0".join([PROMPT_VERSION, SYSTEM_PROMPT, NOTE_PROMPT_TEMPLATE,
                      "|".join(TERM_REASONS), requirement.spec_id, requirement.hash,
                      str(terms.max_terms_per_note), str(terms.min_term_chars)])
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ============================================================================
# THE RECORD
# ============================================================================

@dataclass(frozen=True)
class Admissibility:
    """Question 1's answer: this document's STANDING PER FIELD, and the one span for all of them.

    Its own object, so that it can never be confused with, or derived from, anything else on the
    label. The quote is EVIDENCE FOR THE CLASSIFICATION and nothing else — it is not a value, it
    is not parsed, and nothing downstream reads it as an answer. It exists so a human auditing a
    disputed verdict has the sentence the model was looking at. It is ONE quote for the note, not
    one per field: a span per field is the extraction that was deleted from this module, and it
    would come back through this slot first.
    """

    #: field name -> one of `ADMISSIBILITY_VERDICTS`. Empty ONLY on a label that carries an
    #: error: a reading that happened answered every field, because a reply missing one is
    #: refused in `parse_label_response` rather than completed here.
    verdicts: Mapping[str, str] = field(default_factory=dict)
    quote: str = ""
    #: Checked in code: does `quote` really occur in the note? A model that paraphrases a span it
    #: called verbatim is composing, not reporting, and the verdicts it composed for are the thing
    #: every later experiment counts.
    quote_verified: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.verdicts, str) or not isinstance(self.verdicts, Mapping):
            raise LabelShapeError(
                f"verdicts must be a mapping of field name -> verdict, got "
                f"{type(self.verdicts).__name__} {self.verdicts!r}. A bare string here is the "
                "per-note shape this record replaced; standing is answered once per field.")
        bad = {k: v for k, v in self.verdicts.items() if v not in ADMISSIBILITY_VERDICTS}
        if bad:
            raise LabelShapeError(f"not one of {list(ADMISSIBILITY_VERDICTS)}: {bad}")
        object.__setattr__(self, "verdicts", dict(self.verdicts))

    def verdict_for(self, name: str) -> str:
        """This document's standing on ONE field. Absence raises: it is not "neither".

        A caller asking about a field this reading does not hold is asking about a different
        requirement, or about a label that was never completed. Returning "neither" would answer
        it with a fact nobody established.
        """
        try:
            return self.verdicts[name]
        except KeyError:
            raise LabelShapeError(
                f"no verdict for {name!r}; this reading answered {sorted(self.verdicts)}. A "
                "field that was never answered is not a field answered 'neither'.") from None

    def fields_where(self, verdict: str) -> tuple[str, ...]:
        """The fields with this standing, in the order the reading holds them."""
        return tuple(n for n, v in self.verdicts.items() if v == verdict)

    @property
    def verdict(self) -> str:
        """The note-level COLLAPSE: the strongest standing any field got, or "" if none did.

        A projection, never a stored fact, and never the thing to reason with when a field is in
        hand — `verdict_for` is. It exists for two readers that are honestly note-level: the
        after-the-fact audit at the bottom of this file, whose answer keys are per patient and not
        per field, and anything downstream that has not yet learned to ask per field. Collapsing
        upward (any field that can establish makes the note an establishing note) is the only
        direction that does not invent standing the model did not give.
        """
        for v in ADMISSIBILITY_VERDICTS:  # strongest first, by construction
            if v in self.verdicts.values():
                return v
        return ""

    @property
    def bears_on_question(self) -> bool:
        """Does this document bear on ANY field of the requirement?"""
        return self.verdict in BEARS_ON_QUESTION


@dataclass(frozen=True)
class RetrievalTerm:
    """One term the model claims would surface this note for the requirement, and why.

    There is no `verified` flag, deliberately. Only terms found in the note text reach this
    type, so a reader cannot mistake an unverified proposal for a weak row: it is not a row.
    """

    term: str
    reason: str


@dataclass(frozen=True)
class LabelReply:
    """The two answers, parsed out of one model reply and not yet checked against the note.

    `n_terms_proposed` is kept although most of it may be discarded: after the cap has done its
    work, it is the only surviving evidence that a model tried to pad, and "the cap bites on 40%
    of notes" is the signal that the cap, or the question, needs rewriting.
    """

    admissibility: Admissibility
    terms: tuple[RetrievalTerm, ...] = ()
    n_terms_proposed: int = 0


@dataclass(frozen=True)
class NoteLabel:
    """One note, completely labelled. The unit the whole development plane consumes.

    `spec_id`, `prompt_hash` and `model` ride on every row because a label is conditional on all
    three. Change the requirement and these are answers to a different question about the same
    note; change the prompt and they are different answers to the same question; and the
    deployment behind a model name changes underneath us without a version bump. A row that
    records none of them is an unattributable claim about a note nobody will reread.
    """

    patient_id: str
    note_id: str
    doc_type: str = ""
    date: str = ""
    note_truncated: bool = False
    #: The requirement these answers are about. Not optional in practice — the runner always
    #: sets it — but defaulted so an old file still loads and reads as unattributed.
    spec_id: str = ""
    admissibility: Admissibility = field(default_factory=Admissibility)
    #: Question 2, after verification. Everything the model offered that is not in this tuple was
    #: refused; the two counters below are how much and why, because a silently shortened list
    #: would read as a model with little to say about a note it in fact invented terms for.
    retrieval_terms: tuple[RetrievalTerm, ...] = ()
    n_terms_proposed: int = 0
    n_terms_hallucinated: int = 0
    model: str = ""
    prompt_hash: str = ""
    scanned_at: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error

    def to_dict(self) -> dict[str, Any]:
        """The row as it is written. One addition to the dataclass: a collapsed `verdict`.

        That key is an EXPORT-ONLY PROJECTION of `admissibility.verdicts`, recomputed on every
        write and never read back — `from_dict` ignores it, and a test holds that. It is here
        because readers that predate per-field standing look for exactly that key, and without it
        they would find nothing, count every note as bearing on nothing, and report a confident
        table of zeroes. A wrong number nobody can see is worse than a redundant one somebody can
        recompute; the field-level answer beside it is the one to build on.
        """
        d = asdict(self)
        d["admissibility"] = {**d["admissibility"], "verdict": self.admissibility.verdict}
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> NoteLabel:
        """A row -> a label, refusing loudly anything that is not this generation's shape."""
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        kw["admissibility"] = _admissibility_from_dict(d.get("admissibility") or {})
        kw["retrieval_terms"] = tuple(
            RetrievalTerm(**t) for t in (d.get("retrieval_terms") or []))
        return cls(**kw)


def _admissibility_from_dict(raw: Any) -> Admissibility:
    """Question 1's stored answer -> `Admissibility`, or `LabelShapeError`.

    The one place the two generations of label meet. A row from before standing became per field
    carries a single `verdict` string and no `verdicts` map; it is a complete answer to a
    differently shaped question, and there is no honest way to widen it — "the note establishes
    the answer" says nothing about WHICH field, and spreading it over all of them would
    manufacture exactly the conflation this change removed. So it is refused by name.
    """
    if not isinstance(raw, Mapping):
        raise LabelShapeError(f"admissibility must be an object, got {type(raw).__name__}")
    if "verdicts" not in raw:
        if raw.keys() & {"verdict", "quote", "quote_verified"}:
            raise LabelShapeError(
                f"this row carries one verdict for the whole note ({dict(raw)!r}) and no "
                "per-field 'verdicts'. It was written by an older labelling, against a prompt "
                f"before {PROMPT_VERSION}; widening it to every field would invent standing "
                "nobody read. Rescan against this prompt, or read that file with the code that "
                "wrote it — do not mix them.")
        return Admissibility()
    return Admissibility(verdicts=raw["verdicts"], quote=str(raw.get("quote") or ""),
                         quote_verified=bool(raw.get("quote_verified")))


def _parse_terms(raw: Any, terms: TermConfig) -> tuple[tuple[RetrievalTerm, ...], int]:
    """Question 2's answer: capped, deduplicated, and structurally filtered. Not yet verified.

    The cap is applied to what was PROPOSED, before anything is checked against the note, and
    the slots that verification later empties are never backfilled. Backfilling would pay
    padding a dividend — offer forty, have thirty-five thrown away, still land a full list — and
    the cap exists precisely so that proposing more costs the model its best slots.
    """
    items = [t for t in (raw if isinstance(raw, list) else []) if isinstance(t, dict)]
    kept: dict[str, RetrievalTerm] = {}
    for item in items[:terms.max_terms_per_note]:
        term, reason = str(item.get("term") or "").strip(), str(item.get("reason") or "").strip()
        # An unrecognised reason class is dropped rather than folded into "other": "other" is a
        # claim the model did not make, and it would be indistinguishable from ones it did.
        if len(term) >= terms.min_term_chars and reason in TERM_REASONS:
            kept.setdefault(term.lower(), RetrievalTerm(term, reason))
    return tuple(kept.values()), len(items)


def _parse_verdicts(raw: Any, requirement: Requirement) -> dict[str, str]:
    """Question 1's answer: one verdict per field of THIS requirement, or `PromptContractError`.

    Every field is required and no field is invented. A reply short of a field is not completed
    with "neither" — the whole reason standing is asked per field is that "this document says
    nothing about that one" is a finding, and a finding this module wrote in itself would be
    indistinguishable from one the reader made. A reply with a name the requirement does not
    declare is refused too, and for a sharper reason: it means the reply was not shaped by the
    field list it was shown, so the verdicts that DO line up are not evidence of anything either.
    """
    if not isinstance(raw, Mapping):
        got = "a single note-level verdict" if isinstance(raw, str) else type(raw).__name__
        raise PromptContractError(
            f"'verdicts' must be an object keyed by field name, got {got} ({raw!r}). Standing is "
            f"answered once per field: {list(requirement.field_names)}.")
    wanted = requirement.field_names
    seen = {str(k).strip(): v for k, v in raw.items()}
    missing = [n for n in wanted if n not in seen]
    if missing:
        raise PromptContractError(
            f"no verdict for {missing}; the reply answered {sorted(seen)} of "
            f"{list(wanted)}. A field left out is a contract violation, not a 'neither'.")
    unknown = sorted(set(seen) - set(wanted))
    if unknown:
        raise PromptContractError(
            f"reply invented field(s) {unknown}; this requirement declares {list(wanted)}")
    out: dict[str, str] = {}
    for name in wanted:  # the spec's order, not the reply's
        verdict = str(seen[name] or "").strip()
        if verdict not in ADMISSIBILITY_VERDICTS:
            raise PromptContractError(
                f"verdict {verdict!r} for {name!r} is not one of {list(ADMISSIBILITY_VERDICTS)}")
        out[name] = verdict
    return out


def parse_label_response(text: str, *, requirement: Requirement, terms: TermConfig) -> LabelReply:
    """Model reply -> the two answers, or `PromptContractError`.

    Strict about the things that would corrupt a measurement, lenient about the rest. An unknown
    verdict is a hard error rather than being folded into "neither", because that would put a
    shrug in the standing column and nothing on the row would ever say so.

    `requirement` is required and has no default: question 1's answer is only checkable against
    the field list it was asked about, and a parser that did not hold that list would have to
    accept whatever set of keys came back — which is the missing-field bug with extra steps.
    """
    from ..core.llm import extract_json

    obj = extract_json(text or "", require="admissibility")
    adm = obj.get("admissibility") if isinstance(obj, dict) else None
    if not isinstance(adm, dict) or "__unparsed__" in obj:
        raise PromptContractError("reply has no 'admissibility' object")

    if "verdicts" not in adm and "verdict" in adm:
        raise PromptContractError(
            f"reply answered question 1 once for the whole document (verdict="
            f"{adm['verdict']!r}). It is asked once per field: {list(requirement.field_names)}. "
            "Copying one answer onto every field is the conflation this contract removed.")
    verdicts = _parse_verdicts(adm.get("verdicts"), requirement)
    bears = any(v in BEARS_ON_QUESTION for v in verdicts.values())
    quote = str(adm.get("quote") or "").strip()
    # A note said to bear on the question, with nothing shown for it, is a verdict with no
    # evidence: unauditable, and indistinguishable from a model that guessed at the class. ONE
    # quote covers however many fields were not "neither" — see `Admissibility`.
    if bears and not quote:
        raise PromptContractError(
            f"{[n for n, v in verdicts.items() if v in BEARS_ON_QUESTION]} were called "
            f"non-'neither' with no quote; the sentence that makes it so is what makes the "
            "classification reviewable, and there is no reading without it")

    # An absent key means question 2 was not answered; an empty list means it was answered "no
    # term here would find this note", which is the right answer for many notes and must not be
    # indistinguishable from a model that skipped the question.
    if not isinstance(obj.get("retrieval_terms"), list):
        raise PromptContractError("reply has no 'retrieval_terms' list; question 2 unanswered is "
                                  "not question 2 answered with nothing")
    proposed, n_proposed = _parse_terms(obj["retrieval_terms"], terms)
    # An all-"neither" note carries no quote by contract, so anything offered with one is
    # discarded rather than stored beside verdicts that do not license it.
    return LabelReply(Admissibility(verdicts, quote if bears else ""), proposed, n_proposed)


def verify_quote(quote: str, text: str) -> bool:
    """Is this span really in the note?

    Whitespace- and case-insensitive, and nothing more. Line wrapping and casing are transport
    damage; a different word is a different quote, and this returns False for it.
    """
    flat = lambda s: re.sub(r"\s+", " ", s).strip().lower()
    return bool(quote.strip()) and flat(quote) in flat(text)


def verify_terms(proposed: Sequence[RetrievalTerm],
                 text: str) -> tuple[tuple[RetrievalTerm, ...], int]:
    """Keep the proposed terms the note really contains; return the rest as a count.

    Exactly `verify_quote`'s check, for a stronger reason. A hallucinated quote misattributes a
    reading; a hallucinated TERM is a retrieval instruction that matches nothing, and it would
    go into a keyword list, be measured at zero recall, and look like evidence that asking a
    model for terms does not work. So it is dropped — and counted, because a hallucination rate
    that is silently swallowed is the one number that would have said the question is failing.
    """
    kept = tuple(t for t in proposed if verify_quote(t.term, text))
    return kept, len(proposed) - len(kept)


# ============================================================================
# THE STORE — a JSONL of labels, which is the whole resume mechanism
# ============================================================================

def labels_root(explicit: str | None = None) -> Path:
    """Where labellings may be written, refusing anywhere inside the repository.

    A label carries a person_id, a note date and one verbatim quote of the note.
    `tests/test_no_phi_in_tree.py` exists because that material got into the tree once already.
    Defaulting away from a bad path would hide the mistake; raising names it.
    """
    p = Path((explicit or os.environ.get(LABELS_ROOT_ENV) or DEFAULT_LABELS_ROOT).strip()).resolve()
    if p == _REPO_ROOT or _REPO_ROOT in p.parents:
        raise LabellingError(
            f"refusing to write labels to {p}: it is inside the repository, and a label carries a "
            f"person_id, a note date and verbatim note text. Use a PHI-approved path outside "
            f"the tree (default {DEFAULT_LABELS_ROOT}) or ${LABELS_ROOT_ENV}.")
    return p


class LabelStore:
    """Append-only JSONL keyed by (patient_id, note_id), one directory per labelling.

    Resumability is the file format, not a feature layered on top. A label is one line, flushed
    the moment the note comes back; a restart reads the lines and skips those keys. There is no
    checkpoint to go stale and no index to rebuild, and a half-written final line from a killed
    job is dropped on load rather than crashing the rerun on its own tail. The same key written
    twice is legal and the last line wins, so a deliberate re-scan is just running it again.

    Labels made under different models, prompt wordings or REQUIREMENTS must never share a file.
    The requirement is in the key for the same reason the prompt is: two requirements over one
    corpus produce two different answers per note, and a file mixing them has nothing on any row
    to say which question it answered. Rather than detect that collision, `run_key` prevents it.
    """

    def __init__(self, root: str | Path | None, *, model: str, requirement: Requirement,
                 terms: TermConfig):
        self.model, self.requirement, self.terms = model, requirement, terms
        self.key = hashlib.sha256("|".join(
            [model, PROMPT_VERSION, requirement.spec_id,
             prompt_hash(requirement, terms)]).encode()).hexdigest()[:12]
        self.dir = labels_root(str(root) if root is not None else None) / self.key
        self.path = self.dir / "labels.jsonl"
        self._lock = threading.Lock()
        self.dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.dir, 0o700)
        # A killed job leaves a torn final line. Terminate it now, or the first append of the
        # resumed run lands on the end of it and destroys a second, complete label as well.
        if self.path.is_file() and not self.path.read_bytes().endswith(b"\n"):
            with self.path.open("ab") as fh:
                fh.write(b"\n")
        manifest = self.dir / "manifest.json"
        if not manifest.exists():
            manifest.write_text(json.dumps(
                {"run_key": self.key, "model": model, "prompt_version": PROMPT_VERSION,
                 "spec_id": requirement.spec_id, "requirement_hash": requirement.hash,
                 # The fields every row in this file answers, so a reader can tell at the door
                 # whether the labelling covers the field it came for.
                 "fields": list(requirement.field_names),
                 "prompt_hash": prompt_hash(requirement, terms), "terms": asdict(terms),
                 "usd_per_1m": [USD_PER_1M_INPUT, USD_PER_1M_OUTPUT],
                 "created_at": _utc_now()}, indent=1))
            os.chmod(manifest, 0o600)

    def append(self, label: NoteLabel) -> None:
        """One line, flushed. Thread-safe: the pool has many readers and one writer."""
        self._check_fields(label)
        line = json.dumps(label.to_dict(), sort_keys=True, default=str)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)  # 0600 at creation
        with self._lock, os.fdopen(fd, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _check_fields(self, label: NoteLabel) -> None:
        """A completed reading in this file answers exactly this requirement's fields.

        Not a formality. The run key already keeps two requirements in two directories, so a row
        here whose verdicts name other fields — or none — did not come from a reading of this
        requirement, and every per-(type, field) number computed over this file would be built on
        it. An errored label is exempt: nothing was read, so there is nothing to have answered.
        """
        want, got = set(self.requirement.field_names), set(label.admissibility.verdicts)
        if label.ok and want != got:
            raise LabelShapeError(
                f"label {label.patient_id}/{label.note_id} answers {sorted(got)}; this labelling "
                f"is of {sorted(want)} (spec {self.requirement.spec_id}). Standing is per field "
                "and every field is required, so this row cannot be one of these.")

    def load(self) -> dict[tuple[str, str], NoteLabel]:
        """Every label, de-duplicated by key, last line wins. Torn final lines are dropped.

        `LabelShapeError` is deliberately NOT forgiven here: see the class docstring. A torn line
        is an incomplete write; a row of the wrong generation is a complete write of the wrong
        thing, and continuing past it would produce a run assembled from two labellings.
        """
        out: dict[tuple[str, str], NoteLabel] = {}
        for line in (self.path.read_text(encoding="utf-8").splitlines()
                     if self.path.is_file() else []):
            try:
                lab = NoteLabel.from_dict(json.loads(line))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue  # a torn final line is the normal shape of a killed job
            self._check_fields(lab)
            out[(lab.patient_id, lab.note_id)] = lab
        return out

    def spend(self) -> float:
        """Total cost, recomputed from the label file rather than kept in a second counter.

        Two accounts of one quantity disagree eventually and nothing raises when they do
        (`acr.core.state` documents making exactly that mistake), so there is only ever one.
        """
        return sum(lab.cost_usd for lab in self.load().values())


# ============================================================================
# THE CLIENT — wired to Azure, called by nobody in this module
# ============================================================================

def parse_env_file(path: str | Path) -> dict[str, str]:
    """`export KEY=value` lines -> a dict. Read as data; never executed.

    `.azure_env` is a shell file and sourcing it is the obvious thing to do. It is also how a
    credentials file becomes an execution vector, so this parses it instead.
    """
    p = Path(path)
    if not p.is_file():
        raise NotConfiguredError(f"no environment file at {p}")
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip().removeprefix("export ")
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip("\"'")
    return out


def azure_client(env_path: str | Path = AZURE_ENV_PATH, *, max_tokens: int = 1200,
                 timeout: int = 180):
    """Build the reading client. Nothing here calls it; spending is the operator's decision.

    Two things a caller reading `.azure_env` by hand would get wrong. TEMPERATURE: luna is a
    reasoning model and 400s on any value but its default, and `litellm.drop_params` drops
    unsupported parameters, not unsupported VALUES — so acr's usual 0.0 reaches the API and
    fails the first call. MAX_TOKENS: a reasoning model spends most of a completion thinking
    before it emits anything, and `acr.core.llm` records a run where the thinking consumed the
    budget and `content` came back empty. Budget for the thinking.
    """
    from ..core.llm import LLMClient, LLMConfig

    # A FILE IF ONE IS CONFIGURED, OTHERWISE THE ENVIRONMENT. This plane used to read only a
    # credentials file, whose path was an institutional absolute. De-institutionalising that on
    # 2026-08-03 left `site.MODEL_ENV_FILE` unset by default, so `env_path` became `""` -> `Path(".")`
    # -> "no environment file at ." and the entire experience plane could not start. Nothing caught
    # it: no test reaches this function, because reaching it costs money.
    #
    # Every other plane in this system reads `ACR_API_BASE` / `ACR_API_KEY` / `ACR_MODEL` from the
    # environment. This one had a second mechanism, and a second mechanism is a second thing that can
    # be unconfigured. The file remains an OVERRIDE, because reading credentials as data rather than
    # sourcing them as shell is worth keeping — but it is no longer the only way in.
    from_file = bool(env_path) and Path(str(env_path)).is_file()
    if from_file:
        env, source = parse_env_file(env_path), str(env_path)
    else:
        # ONLY THE `ACR_` NAMES from the environment, and that narrowing is not tidiness. A plain
        # `dict(os.environ)` fallback would let an `OPENAI_API_KEY` sitting in somebody's shell
        # profile silently become the credential this plane dials out with — a provider nobody in
        # this run chose, billed to an account nobody in this run named. A file may carry the vendor
        # spellings because someone wrote that file on purpose; an ambient variable did not.
        env = {k: v for k, v in os.environ.items() if k.startswith("ACR_")}
        source = "the environment (ACR_* only)"
    api_base = env.get("ACR_API_BASE") or (env.get("OPENAI_BASE_URL") if from_file else None)
    api_key = env.get("ACR_API_KEY") or (
        (env.get("OPENAI_API_KEY") or env.get("AZURE_API_KEY")) if from_file else None)
    if not api_base or not api_key:
        raise NotConfiguredError(
            f"{source} has no ACR_API_BASE/ACR_API_KEY; cannot address the deployment. Set them in "
            f"the environment, or point ACR_MODEL_ENV_FILE at a file of `KEY=value` lines.")
    return LLMClient(LLMConfig(
        model=env.get("ACR_MODEL") or f"openai/{DEPLOYMENT}", api_base=api_base, api_key=api_key,
        temperature=float(env.get("ACR_TEMPERATURE") or 1.0), max_tokens=max_tokens,
        timeout=timeout))


# ============================================================================
# THE RUNNER
# ============================================================================

@dataclass(frozen=True)
class ScanConfig:
    """How the scan may behave. `max_usd` is positional and has no default.

    That is the whole design of this dataclass: `ScanConfig()` is a TypeError, so there is no
    path to a scan that runs without a declared ceiling. A ceiling you can forget to set is not
    a ceiling, and what it guards is a five-figure call count against a per-token price.
    """

    max_usd: float
    #: What the notes are being read AGAINST. Required, and the store's must match it.
    requirement: Requirement
    #: Question 2's bounds, required for the same reason `max_usd` is: see `TermConfig`.
    terms: TermConfig
    concurrency: int = 8
    #: Notes longer than this are truncated and the label says so. Input tokens are what this
    #: run is buying, and the tail of a 200KB document is rarely where its standing is decided.
    max_note_chars: int = 24_000

    def __post_init__(self) -> None:
        if not self.max_usd > 0 or self.concurrency < 1:
            raise ValueError(f"need max_usd > 0 and concurrency >= 1, got {self}")


@dataclass(frozen=True)
class ScanReport:
    """What one invocation did. `spend_usd` is the label file's number, not a second account."""

    run_dir: str
    n_notes_in_scope: int
    n_already_labelled: int
    n_written: int
    n_errors: int
    spend_usd: float
    aborted: bool = False
    abort_reason: str = ""


class FullScanRunner:
    """Reads every note of every patient handed to it, once, against one requirement.

    `client` is anything with `.chat(messages)` returning an object carrying `.content` and
    token counts. Tests pass a stub; `azure_client()` builds the real one.
    """

    def __init__(self, *, corpus: Corpus, store: LabelStore, config: ScanConfig, client: Any):
        # The store's directory is keyed by the requirement and the bounds it was built with, so
        # a store built with one and a scan run under another would append answers to a different
        # question into a file whose manifest names the first. Two sources of one number, which
        # is the failure this module refuses everywhere else.
        if store.requirement != config.requirement:
            raise LabellingError(
                f"store keyed to spec {store.requirement.spec_id}/{store.requirement.hash}, scan "
                f"configured for {config.requirement.spec_id}/{config.requirement.hash}: these "
                "labels answer different questions and must not share a file.")
        if store.terms != config.terms:
            raise LabellingError(f"store keyed with {store.terms}, scan configured with "
                                 f"{config.terms}: question 2's bounds are part of the prompt, so "
                                 "these labels would not be comparable with that file's.")
        self.corpus = corpus
        self.store = store
        self.config = config
        self.client = client

    def scope(self, patient_ids: Sequence[str]) -> list[tuple[str, DocMeta]]:
        """Every note of every patient, in a deterministic order.

        Sorted so a resumed run does the same work in the same order, which is what makes "we
        scanned 12,000 of 39,000 notes" a statement somebody can act on.
        """
        items: list[tuple[str, DocMeta]] = []
        for pid in sorted(set(patient_ids)):
            try:
                docs, _ = self.corpus.chart(pid).list_documents(limit=1_000_000)
            except FileNotFoundError:
                continue
            items.extend((pid, meta) for meta in docs)
        return items

    def pending(self, items: Sequence[tuple[str, DocMeta]]) -> list[tuple[str, DocMeta]]:
        """Resume: a note already in the JSONL is done, error labels included.

        An error label is a completed reading of a note that does not work. Retrying it
        automatically would re-spend money on exactly those notes at every restart; deleting
        their lines is how you ask for them again, and that is a decision with a person on it.
        """
        done = set(self.store.load())
        return [it for it in items if (it[0], it[1].note_id) not in done]

    def label_note(self, note: NoteForReading, *, truncated: bool = False) -> NoteLabel:
        """One note -> one label. A model failure becomes a label carrying its error.

        That is not tolerance of failure, it is what makes the scan resumable and countable: a
        failure rate nobody can count is a failure rate nobody will fix. `NoteLabel.ok` is how
        the experiments exclude them. There is no retry here — a rerun is the retry, and it is
        free of the notes already done.
        """
        req = self.config.requirement
        base = NoteLabel(
            patient_id=note.patient_id, note_id=note.note_id, doc_type=note.doc_type,
            date=note.date, note_truncated=truncated, spec_id=req.spec_id,
            model=self.store.model, prompt_hash=prompt_hash(req, self.config.terms),
            scanned_at=_utc_now())
        pt = ct = 0
        try:
            resp = self.client.chat(
                build_note_prompt(note, requirement=req, terms=self.config.terms))
            pt = int(getattr(resp, "prompt_tokens", 0) or 0)
            ct = int(getattr(resp, "completion_tokens", 0) or 0)
            reply = parse_label_response(getattr(resp, "content", "") or "",
                                         requirement=req, terms=self.config.terms)
        except Exception as exc:  # noqa: BLE001 — recorded on the label rather than raised
            return replace(base, prompt_tokens=pt, completion_tokens=ct,
                           cost_usd=cost_usd(pt, ct), error=f"{type(exc).__name__}: {exc}"[:300])
        kept, n_hallucinated = verify_terms(reply.terms, note.text)
        adm = reply.admissibility
        return replace(
            base, prompt_tokens=pt, completion_tokens=ct, cost_usd=cost_usd(pt, ct),
            admissibility=replace(adm, quote_verified=verify_quote(adm.quote, note.text)),
            retrieval_terms=kept, n_terms_proposed=reply.n_terms_proposed,
            n_terms_hallucinated=n_hallucinated)

    def _label_item(self, item: tuple[str, DocMeta]) -> NoteLabel:
        """Read one note out of the corpus and label it. Never raises: the pool's only job is to
        hand back a label, and a corpus read that fails is a label with an error and no cost."""
        pid, meta = item
        limit = self.config.max_note_chars
        try:
            text = self.corpus.chart(pid).read(meta.note_id, limit=10_000_000)["text"]
        except Exception as exc:  # noqa: BLE001 — a corpus read, so nothing was spent
            return NoteLabel(pid, meta.note_id, meta.doc_type, meta.date.isoformat(),
                             spec_id=self.config.requirement.spec_id,
                             error=f"{type(exc).__name__}: {exc}"[:300])
        return self.label_note(
            NoteForReading(pid, meta.note_id, meta.doc_type, meta.date.isoformat(), text[:limit]),
            truncated=len(text) > limit)

    def run(self, patient_ids: Sequence[str]) -> ScanReport:
        """Scan every unlabelled note of these patients, or stop at the ceiling.

        One batch of `concurrency` notes is in flight at a time and the ceiling is consulted
        between batches. Submitting all 39,000 up front would put the ceiling behind a queue it
        cannot cancel and hold every note's text in memory at once; checking mid-batch would
        throw away calls already paid for. When the ceiling trips, the batch in flight lands,
        nothing further is dispatched, and the report says plainly that the run aborted.
        """
        items, n = self.scope(patient_ids), self.config.concurrency
        todo = self.pending(items)
        spent, n_written, n_errors, reason = self.store.spend(), 0, 0, ""
        with ThreadPoolExecutor(max_workers=n) as pool:
            for start in range(0, len(todo), n):
                if spent >= self.config.max_usd:
                    reason = (f"spend ${spent:.4f} reached the ceiling ${self.config.max_usd:.4f} "
                              f"with {len(todo) - start} note(s) unread. Every label written is "
                              "on disk; raise the ceiling and rerun to resume.")
                    break
                for label in pool.map(self._label_item, todo[start:start + n]):
                    self.store.append(label)
                    n_written += 1
                    n_errors += int(not label.ok)
                    spent += label.cost_usd
        return ScanReport(
            run_dir=str(self.store.dir), n_notes_in_scope=len(items),
            n_already_labelled=len(items) - len(todo), n_written=n_written, n_errors=n_errors,
            spend_usd=spent, aborted=bool(reason), abort_reason=reason)


# ============================================================================
# AN AFTER-THE-FACT AUDIT OF THE LABELLER — SEPARATE, OPTIONAL, OFFLINE
# ============================================================================
#
# NOT PART OF THE LABELLING CALL. Nothing above this line can reach anything below it: the
# runner never imports an answer key, the prompt builder takes a note and a requirement and
# nothing else, and these functions read labels that are already on disk.
#
# WHY IT HAS TO EXIST. Without it, the model's judgement silently BECOMES the definition of
# "relevant". Every downstream number — a keyword list's recall, a document type's yield, a
# read/search/sample policy — is computed against these labels, so if the labeller systematically
# calls the wrong notes `can_establish`, every one of those numbers is confidently wrong and
# nothing in the pipeline disagrees with anything else. This is the only place the LABELLER
# ITSELF is measured rather than believed.
#
# WHY IT TAKES THE ANSWER KEY AS AN ARGUMENT. Because there is no one answer key. An
# institutional export, one reviewer's adjudication of fifty charts, another site's extract and a
# synthetic fixture are all valid keys, and none of their column names belong in this module. The
# previous version of this section hardcoded one file and its columns and was useless to anyone
# who did not have that file. So the key is
# any mapping from patient id to anything at all, and `carries` is the caller's own decision
# procedure for "does this note carry that answer". Bake either one in and the audit works for
# one project.

#: The cells of the audit's cross-tab. Not a score: a table, so a disagreement can be read.
CARRIES, LACKS = "carries", "lacks"


@dataclass(frozen=True)
class RelevanceAudit:
    """How the labeller's standing calls line up with an external answer key.

    The headline is `can_establish_precision`, which answers the question this audit was asked
    for: of the notes the model said could establish the answer, how many really carry it. But
    the other direction is on the table too, because it is the more expensive failure — a note
    the key says carries the answer and the labeller filed under "neither" is a note no keyword
    derived from these labels will ever be built to find, and precision alone cannot see it.
    """

    #: verdict -> {CARRIES: n, LACKS: n}. Every scorable label appears in exactly one cell.
    table: dict[str, dict[str, int]]
    n_labels: int
    n_scorable: int
    n_unscorable: int
    #: Patients the audit could score at all, and how many of them the labeller gave at least one
    #: `can_establish` note that really carries the answer. The second is the number a retrieval
    #: plan lives or dies by: per-note precision can look fine while whole patients are unreached.
    n_patients_keyed: int
    n_patients_reached: int
    #: (patient_id, note_id, verdict) for every disagreement, so they can be read rather than
    #: averaged. A disagreement is a `can_establish` note that does NOT carry the answer, or a
    #: `neither` note that does; `merely_mentions` is never one, because a note that restates the
    #: answer without standing to establish it is exactly what that class is for. No note text
    #: and no answer value here: this object gets written into run directories, and
    #: `tests/test_no_phi_in_tree.py` is in this repo for a reason.
    disagreements: tuple[tuple[str, str, str], ...] = ()
    #: WHICH FIELD'S standing was scored, or "" for the note-level collapse. On the record because
    #: precision 0.62 for one field and precision 0.62 for a whole requirement are different
    #: claims, and a table that does not say which it is will be read as whichever suits.
    field_scored: str = ""

    def _n(self, verdict: str, cell: str) -> int:
        return self.table.get(verdict, {}).get(cell, 0)

    @property
    def can_establish_precision(self) -> float | None:
        """Of scorable `can_establish` notes, the fraction that carry the answer.

        `None`, not 0.0, when there are none to score. A precision of zero and a precision of
        nothing-was-measured are different facts, and reporting the second as the first is how
        an unrun audit becomes evidence of a broken labeller.
        """
        n = self._n("can_establish", CARRIES) + self._n("can_establish", LACKS)
        return self._n("can_establish", CARRIES) / n if n else None

    @property
    def n_missed(self) -> int:
        """Answer-carrying notes the labeller said do not bear on the question at all."""
        return self._n("neither", CARRIES)

    @property
    def patient_reach(self) -> float | None:
        """Fraction of keyed patients for whom a `can_establish` call was both made and right."""
        return self.n_patients_reached / self.n_patients_keyed if self.n_patients_keyed else None

    def as_dict(self) -> dict[str, Any]:
        return {"table": {v: dict(c) for v, c in sorted(self.table.items())},
                "field_scored": self.field_scored,
                "n_labels": self.n_labels, "n_scorable": self.n_scorable,
                "n_unscorable": self.n_unscorable, "n_patients_keyed": self.n_patients_keyed,
                "n_patients_reached": self.n_patients_reached,
                "can_establish_precision": self.can_establish_precision,
                "patient_reach": self.patient_reach, "n_missed": self.n_missed,
                "disagreements": [list(d) for d in self.disagreements]}


def audit_relevance(labels: Iterable[NoteLabel], answer_key: Mapping[str, Any], *,
                    carries: Callable[[Any, NoteLabel], bool],
                    field_name: str | None = None) -> RelevanceAudit:
    """Check the labeller's standing calls against ANY external answer key. Offline, afterwards.

    `answer_key` maps a patient id to whatever that key holds for the patient — a code, a row, a
    dataclass, a string. This module never looks inside it. A patient absent from the key is
    UNSCORABLE and stays out of every denominator; a padded denominator reports a number that is
    wrong in a direction nobody can sign.

    `carries(answer, label)` is the caller's decision procedure: given the key's entry for that
    patient and one label, does that NOTE carry that answer? The caller supplies it because only
    the caller knows what its key means and how to reach the note — it may compare codes, grep
    text it holds itself, or consult an adjudication table. It is called once per scorable label,
    it must not mutate anything, and an exception from it is the caller's bug and is not caught.

    Errored labels are unscorable: a note the reader could not read says nothing about the
    labeller's judgement, and counting it as a miss would blame the reader for a timeout.

    `field_name` says WHICH field's standing is being scored, and defaults to none — the
    note-level collapse (`Admissibility.verdict`: a note that can establish any field scores as
    an establishing note), which is the only claim an answer key holding no fields can
    adjudicate. Pass a field, with a `carries` that knows what the key holds for THAT field, and
    the table is about that field alone. Which one it was is on the returned record, because
    scoring the wrong field against a key is the way to make a labeller look broken: a document
    that establishes a site and is mute on a cell type is a false positive against a cell-type
    key and correct against a site key, and only the caller knows which key it brought.
    """
    table: dict[str, dict[str, int]] = {v: {CARRIES: 0, LACKS: 0} for v in ADMISSIBILITY_VERDICTS}
    n_labels = n_scorable = n_unscorable = 0
    reached: set[str] = set()
    keyed: set[str] = set()
    disagreements: list[tuple[str, str, str]] = []

    for lab in labels:
        n_labels += 1
        # `in`, not `.get(...) is None`: a key entry may legitimately be falsy — 0, "", an empty
        # record meaning "adjudicated, nothing there" — and that is a scorable answer, not a
        # missing one.
        if lab.patient_id not in answer_key or not lab.ok:
            n_unscorable += 1
            continue
        verdict = (lab.admissibility.verdict if field_name is None
                   else lab.admissibility.verdict_for(field_name))
        if verdict not in table:
            n_unscorable += 1
            continue
        keyed.add(lab.patient_id)
        n_scorable += 1
        got = bool(carries(answer_key[lab.patient_id], lab))
        table[verdict][CARRIES if got else LACKS] += 1
        if got and verdict == "can_establish":
            reached.add(lab.patient_id)
        elif (verdict == "can_establish") or (got and verdict == "neither"):
            disagreements.append((lab.patient_id, lab.note_id, verdict))

    return RelevanceAudit(
        table=table, n_labels=n_labels, n_scorable=n_scorable, n_unscorable=n_unscorable,
        n_patients_keyed=len(keyed), n_patients_reached=len(reached),
        disagreements=tuple(sorted(disagreements)), field_scored=field_name or "")
