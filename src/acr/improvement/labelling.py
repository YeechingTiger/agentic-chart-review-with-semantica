"""THE FULL SCAN — one cheap reading of every note in a development set, against ONE requirement.

WHAT THIS ANSWERS
-----------------
Given any development set of patients and any requirement, decide, per note:

  1. does this note bear on the requirement, and in what way (its STANDING) — asked and answered
     ONCE PER FIELD of the requirement, never once for the note;
  2. which terms in it would let a searcher find it for that requirement — ONE list per note, for
     the requirement as a whole;
  3. what value the note ASSERTS per field, each beside the verbatim span it was read from;
  4. what the note POINTS AT — verbatim spans naming another document, date or identifier, which
     is the indirect evidence that says where the direct evidence is;
  5. whether the content bearing on the requirement is ORIGINAL to this note or CARRIED FORWARD.

Aggregated, 1 and 2 become the keyword list and the document-type policy for that requirement; 3
becomes the conflict rate, the naive-answer baseline and the only stopping evidence anywhere (how
many notes per chart can establish an answer); 4 is what tells a reading agent where to go next
rather than only whether it has arrived; and 5 is the honest denominator for every claim about how
much reading is wasted, because on a real record duplication is most of the corpus.

That is the whole job. This module still does not hold an answer key and does not know what the
requirement is about — question 3 records what the NOTE says, which is a reading of one document
and not the answer for a patient; `audit_relevance` at the bottom of this file remains the only
place an answer key is ever in the room, deliberately after the fact.

THE QUESTIONS ARE A REGISTRY, NOT A TEMPLATE
--------------------------------------------
Each of the five is one `ScanQuestion` in `QUESTIONS`: its text, its slice of the reply contract,
the keys its answer arrives under, and the parser for those keys. `note_prompt` assembles the
selected ones and knows nothing about any of them; `selected_questions` is where one is dropped;
`prompt_hash` carries the selection, so a scan that asks four questions cannot land in the file of
a scan that asked five. That is the whole reason for the indirection: the useful experiment is
"what does this cost, and what does it answer, WITHOUT question 4", and a question typed into a
template is a question nobody can subtract without editing the template and moving every hash.
`standing` and `retrieval` cannot be dropped — a scan without standing is not a cheaper scan, it
is the same price per note for nothing to aggregate.

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
from dataclasses import fields as dc_fields
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


class LabelReplyError(LabellingError):
    """The model's reply did not satisfy the response contract.

    Raised for exactly one class of fault: the reply is not an answer to a question that was
    ASKED. A missing field, an invented field, a verdict outside the three classes, an absent key
    for a selected question — every one of them means the reply was not shaped by the contract it
    was shown, so the parts that do line up are not evidence of anything either.
    """


#: The name this exception carried while the scan asked two questions. Kept because `derive`,
#: the CLI and three test modules raise-check against it, and renaming it in nine places would be
#: a diff about nothing; there is one class here, under two names.
PromptContractError = LabelReplyError


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
# THE PROMPT — one note in, one answer per SELECTED question. NO CLINICAL CONTENT BELOW THIS LINE.
# ============================================================================

#: Bump when the wording changes. `prompt_hash()` moves on its own; this is the human-readable
#: version that appears in a write-up. `labelling/5` is the generation that also records what the
#: document ASSERTS, what it POINTS AT and whether its content is ORIGINAL: three more answers per
#: row, so the 912 rows written under `labelling/4` answer a different question and are kept in a
#: different directory rather than averaged with these.
PROMPT_VERSION = "labelling/5"

SYSTEM_PROMPT = (
    "You are reading ONE document in isolation, on behalf of someone who holds the requirement "
    "stated below and has not read this document. You do not know the answer to that "
    "requirement for this patient and you must not guess it from outside the document. Judge "
    "only what THIS document is and what it says, in its own words. "
    "Reply with a single JSON object and nothing else."
)

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

#: What every question is asked ABOUT, and the only part of the prompt that is not selectable: the
#: document, and the requirement rendered from the spec. Every clinical word the model sees arrives
#: through `{requirement}`; nothing in this file names a disease, an organ, a document vocabulary
#: or a coding system, because the first version did and every one of those words was a lie the
#: moment the requirement moved.
PROMPT_HEADER = """\
DOCUMENT TYPE: {doc_type}    DOCUMENT DATE: {date}

--- BEGIN DOCUMENT ---
{text}
--- END DOCUMENT ---

{requirement}"""

#: Where a question's position in the SELECTION is written into its text. Substituted by
#: `note_prompt` at assembly time rather than left to `str.format`, because the number depends on
#: which questions were selected: drop one and the ones after it renumber themselves, which is the
#: whole point of numbering them here instead of typing "QUESTION 4" into a template.
_NUMBER = "{number}"

REPLY_INSTRUCTION = "REPLY WITH EXACTLY THIS JSON OBJECT AND NOTHING ELSE:"


@dataclass(frozen=True)
class ScanQuestion:
    """ONE question the scan may put to the model, and everything needed to ask and read it.

    THE POINT OF THIS TYPE IS THAT THERE IS EXACTLY ONE PLACE A QUESTION IS INJECTED OR REMOVED.
    A question hard-coded into a prompt template is a question nobody can drop without editing the
    template, and editing the template moves `prompt_hash` for every scan — so the cheap experiment
    ("what does the scan cost, and what does it answer, without question 4?") is not cheap and is
    therefore never run. Here it is a selection, the selection rides on `prompt_hash`, and the two
    scans land in two directories with a manifest each naming what it asked.

    `prompt` is this question's text with `{placeholders}` the caller fills, plus `{number}` for
    its position. `reply_shape` is its slice of the single JSON object the reply must be — the
    shapes are joined into one object, so a question carries its own line of the contract rather
    than that contract living in a footer somebody has to remember to edit. `reply_keys` are the
    top-level keys its answer arrives under: they are what makes "this question was not asked"
    different from "this question was asked and not answered", which is the distinction the parser
    turns into an empty default in the first case and a refusal in the second.
    """

    name: str
    prompt: str
    reply_shape: str
    reply_keys: tuple[str, ...]
    #: Parses this question's keys out of the whole reply object and returns the `LabelReply`
    #: fields it contributes, by field name. It lives on the entry so that adding a question adds
    #: no branch anywhere: nothing outside this registry asks which question it is holding.
    parse: Callable[[Mapping[str, Any], "_ReplyContext"], dict[str, Any]]
    default_on: bool = True


@dataclass(frozen=True)
class _ReplyContext:
    """Everything a question's parser may look at besides the reply itself."""

    #: The field list the reply is checked against. A parser that did not hold it would have to
    #: accept whatever keys came back, which is the missing-field bug with extra steps.
    requirement: Requirement
    terms: TermConfig
    #: The note as it was sent. Anything the model claims to have COPIED is checked against this;
    #: with it empty nothing can be, and a span offered under it is discarded and counted rather
    #: than stored as though it had been checked.
    note_text: str
    #: The answers of the questions asked BEFORE this one, by `LabelReply` field name — the same
    #: dict the parse loop is filling. A later question sometimes has to be consistent with an
    #: earlier one: there is nothing to call carried-forward on a document that bears on no field,
    #: and only question 1's answer says whether that is the case.
    answers: dict[str, Any]


def _parse_standing(obj: Mapping[str, Any], ctx: _ReplyContext) -> dict[str, Any]:
    """Question 1: one verdict per field, and the one span that justifies whichever bear."""
    adm = obj.get("admissibility")
    if not isinstance(adm, Mapping):
        raise LabelReplyError(
            f"'admissibility' must be an object, got {type(adm).__name__} ({adm!r})")
    if "verdicts" not in adm and "verdict" in adm:
        raise LabelReplyError(
            f"reply answered question 1 once for the whole document (verdict="
            f"{adm['verdict']!r}). It is asked once per field: "
            f"{list(ctx.requirement.field_names)}. Copying one answer onto every field is the "
            "conflation this contract removed.")
    verdicts = _parse_verdicts(adm.get("verdicts"), ctx.requirement)
    bears = any(v in BEARS_ON_QUESTION for v in verdicts.values())
    quote = str(adm.get("quote") or "").strip()
    # A note said to bear on the question, with nothing shown for it, is a verdict with no
    # evidence: unauditable, and indistinguishable from a model that guessed at the class. ONE
    # quote covers however many fields were not "neither" — see `Admissibility`.
    if bears and not quote:
        raise LabelReplyError(
            f"{[n for n, v in verdicts.items() if v in BEARS_ON_QUESTION]} were called "
            f"non-'neither' with no quote; the sentence that makes it so is what makes the "
            "classification reviewable, and there is no reading without it")
    # An all-"neither" note carries no quote by contract, so anything offered with one is
    # discarded rather than stored beside verdicts that do not license it.
    return {"admissibility": Admissibility(verdicts, quote if bears else "")}


def _parse_retrieval(obj: Mapping[str, Any], ctx: _ReplyContext) -> dict[str, Any]:
    """Question 2: the terms, capped and structurally filtered. Verified later, against the note."""
    raw = obj.get("retrieval_terms")
    if not isinstance(raw, list):
        raise LabelReplyError(
            f"'retrieval_terms' must be a list, got {type(raw).__name__} ({raw!r})")
    proposed, n_proposed = _parse_terms(raw, ctx.terms)
    return {"terms": proposed, "n_terms_proposed": n_proposed}


def _parse_asserted_values(obj: Mapping[str, Any], ctx: _ReplyContext) -> dict[str, Any]:
    """Question 3: the value this document asserts per field, each beside the span it was read from.

    An empty object is an answer — most documents assert nothing for most fields — and it is not
    the same as the question having gone unasked, which is why the absence of the key is refused
    upstream rather than read as this.
    """
    raw = obj.get("asserted_values") or {}
    if not isinstance(raw, Mapping):
        raise LabelReplyError(
            f"'asserted_values' must be an object keyed by field name, got {type(raw).__name__} "
            f"({raw!r}); one entry per field this document asserts something for.")
    wanted = ctx.requirement.field_names
    out: dict[str, AssertedValue] = {}
    for key, item in raw.items():
        name = str(key).strip()
        # The rule question 1 already applies, for the same reason: a value for a field nobody
        # asked about was not shaped by the field list the reply was shown, so the entries that
        # DO line up are not evidence either — and folding it in would make the scan's subject
        # depend on the model.
        if name not in wanted:
            raise LabelReplyError(
                f"reply asserts a value for {name!r}, which this requirement does not declare "
                f"({list(wanted)}). A field nobody asked for is an invented answer.")
        if not isinstance(item, Mapping):
            raise LabelReplyError(
                f"the value asserted for {name!r} must be an object carrying 'value' and "
                f"'as_written', got {type(item).__name__} ({item!r})")
        span = str(item.get("as_written") or "").strip()
        if not span:
            raise LabelReplyError(
                f"the value asserted for {name!r} carries no 'as_written' span. The value is a "
                "coding and cannot be checked against the text; the span it was read from is the "
                "only thing that can be, and a coding with nothing behind it is unauditable.")
        out[name] = AssertedValue(value=_as_text(item.get("value")), as_written=span)
    return {"asserted_values": out}


def _parse_pointers(obj: Mapping[str, Any], ctx: _ReplyContext) -> dict[str, Any]:
    """Question 4: the spans naming where the answer is SOMEWHERE ELSE, verified against the note.

    Verified here and not later, unlike the terms: a term survives its own verification as a
    RANKED proposal whose cap must bite before anything is checked, while a pointer is nothing but
    a span — an unverifiable one is not a weaker pointer, it is not a pointer. Discarded, and
    counted, because a hallucination rate that is silently swallowed is the one number that would
    have said the question is failing.
    """
    raw = obj.get("pointers") or []
    if not isinstance(raw, list):
        raise LabelReplyError(
            f"'pointers' must be a list of spans copied out of the document, got "
            f"{type(raw).__name__} ({raw!r})")
    proposed = [s for s in (str(p).strip() for p in raw) if s]
    verified = [s for s in proposed if verify_quote(s, ctx.note_text)]
    return {"pointers": tuple(dict.fromkeys(verified)),
            "n_pointers_proposed": len(proposed),
            "n_pointers_hallucinated": len(proposed) - len(verified)}


def _parse_originality(obj: Mapping[str, Any], ctx: _ReplyContext) -> dict[str, Any]:
    """Question 5: was the content that bears on the question carried forward, or written here?

    `None` is an answer and never a default for "unknown": a document that bears on no field has
    no content whose originality could be judged, and `False` there would assert that it does.
    """
    raw = obj.get("copied_forward")
    if raw is not None and not isinstance(raw, bool):
        raise LabelReplyError(
            f"'copied_forward' must be true, false or null, got {type(raw).__name__} ({raw!r}). "
            "A string here would be a third value nothing downstream knows how to count.")
    adm = ctx.answers.get("admissibility")
    if adm is not None and not adm.bears_on_question:
        return {"copied_forward": None}
    return {"copied_forward": raw}


#: THE QUESTIONS, IN RENDER ORDER. Adding one to the scan is adding one entry here; removing one is
#: leaving its name out of the selection. Nothing outside this tuple branches on a question's name.
QUESTIONS: tuple[ScanQuestion, ...] = (
    ScanQuestion(
        name="standing",
        reply_keys=("admissibility",),
        parse=_parse_standing,
        reply_shape='"admissibility": {{"verdicts": {verdict_skeleton}, "quote": ""}}',
        prompt="""\
QUESTION """ + _NUMBER + """ — WHAT STANDING DOES THIS DOCUMENT HAVE, FIELD BY FIELD?

Answer SEPARATELY FOR EVERY FIELD listed above, under that field's exact name. One verdict for the
whole document is not wanted and will not do: a document can be among the best sources in the
record for one field and unable to settle another, and a single answer cannot say so.

Standing is a property of the DOCUMENT, not only of its wording. The same sentence can establish
an answer in the document that first rendered it and establish nothing in a document that copied
it forward. Judge which of those you are holding, once per field, with one of: {verdict_classes}

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
recorded as such. If every field is "neither", leave it empty.""",
    ),
    ScanQuestion(
        name="retrieval",
        reply_keys=("retrieval_terms",),
        parse=_parse_retrieval,
        reply_shape='"retrieval_terms": [{{"term": "", "reason": "<one reason class>"}}, ...]',
        prompt="""\
QUESTION """ + _NUMBER + """ — WHICH TERMS IN THIS DOCUMENT WOULD LET A SEARCHER FIND IT? A
retrieval question about the requirement above. Someone who holds that requirement but has NOT
read this document must search every document of every patient by text. Which terms, AS THIS
DOCUMENT SPELLS THEM, would surface it?

  * Copy each term character for character out of the document; terms are checked against the
    text automatically, and one that is not there is discarded.
  * At most {max_terms} terms, each at least {min_chars} characters. Fewer is better; do not pad.
  * Nothing that would also match most documents of most patients — "patient", "note", "date",
    "history". A term that matches everything retrieves everything, which is not retrieval.
  * Offer a term because it INDICATES the answer this requirement asks for, not because it is
    frequent: a word that keeps company with the answer is worth less than a rare one that names
    it, or names the instrument that produced it.
  * Tag each term with exactly ONE reason class:
    {reason_classes}""",
    ),
    ScanQuestion(
        name="value",
        reply_keys=("asserted_values",),
        parse=_parse_asserted_values,
        reply_shape='"asserted_values": {{"<FIELD NAME>": {{"value": "", "as_written": ""}}}}',
        prompt="""\
QUESTION """ + _NUMBER + """ — WHAT VALUE DOES THIS DOCUMENT ASSERT FOR EACH FIELD?

For every field you did NOT call "neither" above, and for no other field, give the value THIS
document asserts for it, in the notation that field's description asks for, together with the span
you read it from. Leave a "neither" field out entirely: a document that does not bear on a field
asserts nothing for it, and an empty entry is not an assertion.

  * "value" is YOUR reading, normalised. It is NOT checked against the document and cannot be —
    one fact is written many ways, and a normalised value often appears nowhere in the text.
  * "as_written" IS checked against the document, character for character, the same way the span
    above is. It is what makes a reading error tellable apart from an invention, so a value
    offered without one is refused and the reply with it.
  * Report what THIS document asserts, even where you judged that it may not settle the matter.
    Whether it settles it is the standing question above and was already answered there.""",
    ),
    ScanQuestion(
        name="pointers",
        reply_keys=("pointers",),
        parse=_parse_pointers,
        reply_shape='"pointers": ["<SPAN COPIED OUT OF THE DOCUMENT>", ...]',
        prompt="""\
QUESTION """ + _NUMBER + """ — WHAT DOES THIS DOCUMENT POINT AT?

Not what would find THIS document — that was asked above — but what this document says about where
the answer was settled or recorded SOMEWHERE ELSE: another document it names, a date it attributes
an answer to, an identifier, accession or number it cites for one. A signpost, not a mention: "see
the report of 3/12" says where to look next; "known history of X" says only that somebody knew.

  * Copy each span character for character out of the document. Pointers are checked against the
    document automatically and one that is not there is discarded and counted, as a term is.
  * Only spans that would let a reader ASK FOR something — a name, a date, an identifier. Not a
    span that merely repeats the answer: that was the question above.
  * A document that points nowhere is the ordinary case. Answer with an empty list; do not pad
    it.""",
    ),
    ScanQuestion(
        name="originality",
        reply_keys=("copied_forward",),
        parse=_parse_originality,
        reply_shape='"copied_forward": true | false | null',
        prompt="""\
QUESTION """ + _NUMBER + """ — IS THE CONTENT THAT BEARS ON THE QUESTION ORIGINAL TO THIS DOCUMENT?

You have judged this already, because the standing question above turns on it: the same sentence
can establish an answer in the document that first rendered it and establish nothing in a document
that copied it forward. This asks you to REPORT that judgement, which costs no further reading.

  true   the content bearing on the requirement was carried into this document from somewhere
         earlier — restated, imported, pasted or summarised out of another document.
  false  it is original here: this document is where that content was written down.
  null   there is nothing to judge, because this document bears on no field at all.""",
    ),
)

#: The two nobody may drop. Every downstream consumer reads standing — it is what the keyword list,
#: the document-type policy and the whole audit at the bottom of this file are computed from — and
#: `retrieval_terms` is the other asset the scan exists to produce. A run without them is not a
#: cheaper scan, it is a scan of nothing, at the same price per note.
MANDATORY_QUESTIONS = ("standing", "retrieval")


def _refuse_dropping_mandatory(names: Sequence[str]) -> None:
    """One message for one refusal, wherever a selection arrives from."""
    dropped = [n for n in MANDATORY_QUESTIONS if n not in set(names)]
    if dropped:
        raise LabellingError(
            f"{dropped} cannot be dropped from a scan. Every consumer of these labels reads "
            "standing — the keyword list, the document-type policy and the audit are all computed "
            "from it — and the terms are the other asset the scan is bought for. A reading without "
            "them costs the same per note and produces nothing to aggregate.")


def selected_questions(names: Sequence[str] | None = None) -> tuple[ScanQuestion, ...]:
    """The questions to ask, in render order. `None` means the `default_on` set.

    Refuses an unknown name rather than ignoring it: a typo silently dropping a question would
    move `prompt_hash`, write a directory the operator did not mean to write, and read as a
    completed scan of a question nobody answered.
    """
    by_name = {q.name: q for q in QUESTIONS}
    if names is None:
        return tuple(q for q in QUESTIONS if q.default_on)
    wanted = [str(n).strip() for n in names if str(n).strip()]
    unknown = [n for n in wanted if n not in by_name]
    if unknown:
        raise LabellingError(f"no such scan question {unknown}; this scan can ask "
                             f"{[q.name for q in QUESTIONS]}")
    _refuse_dropping_mandatory(wanted)
    return tuple(q for q in QUESTIONS if q.name in set(wanted))


def note_prompt(questions: Sequence[ScanQuestion]) -> str:
    """The selected questions -> the template `build_note_prompt` fills. THE ONE ASSEMBLY POINT.

    Deliberately generic over `QUESTIONS`: it numbers the questions in render order, lays them
    out under one header, and joins their reply shapes into the single JSON object the reply must
    be. Nothing here knows what any question is about, so there is no case to extend when one is
    added and no branch to delete when one is dropped.
    """
    blocks = [q.prompt.replace(_NUMBER, str(i)) for i, q in enumerate(questions, 1)]
    shapes = ",\n ".join(q.reply_shape for q in questions)
    return "\n\n".join([PROMPT_HEADER, *blocks, f"{REPLY_INSTRUCTION}\n" + "{{" + shapes + "}}"])


#: The DEFAULT assembly, and the thing to read when you want to know what the scan asks. Every
#: scan that does not select otherwise sends exactly this; `prompt_hash` hashes it, so rewording
#: any question moves every default scan's directory.
NOTE_PROMPT_TEMPLATE = note_prompt(selected_questions())

#: The default selection as a value, so a dataclass field and a keyword argument can default to
#: it without either of them deciding what the default is.
DEFAULT_QUESTIONS: tuple[ScanQuestion, ...] = selected_questions()


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


def build_note_prompt(note: NoteForReading, *, requirement: Requirement, terms: TermConfig,
                      questions: Sequence[ScanQuestion] | None = None) -> list[dict]:
    """One note and one requirement -> the chat messages that read it.

    The only prompt builder in this module, and the signature is the contract: a note, the
    requirement it is being read against, the bounds on question 2, and which questions to put.
    No `context` and no `hints` — a channel for free prose is a channel for somebody's clinical
    opinion to reach the model without passing through a spec that has provenance on it.

    Every placeholder is offered to `note_prompt`'s template whether the selected questions use it
    or not, which is what lets a question be dropped without a caller here knowing that it was.
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
        {"role": "user", "content": note_prompt(
            DEFAULT_QUESTIONS if questions is None else questions).format(
            doc_type=note.doc_type, date=note.date, text=note.text,
            requirement=requirement.render(), max_terms=terms.max_terms_per_note,
            min_chars=terms.min_term_chars, reason_classes=" | ".join(TERM_REASONS),
            n_fields=len(requirement.field_names),
            field_list=", ".join(requirement.field_names),
            verdict_classes=" | ".join(ADMISSIBILITY_VERDICTS),
            verdict_skeleton=requirement.verdict_skeleton)},
    ]


def prompt_hash(requirement: Requirement, terms: TermConfig,
                questions: Sequence[ScanQuestion] | None = None) -> str:
    """Identity of everything that conditions a label, independent of any note.

    Hashing a rendered prompt would give a different hash per note and identify nothing.
    Hashing the template, the requirement, the bounds and the SELECTION identifies the things
    that actually condition the answers, so two runs can be compared — or refused comparison — on
    the evidence.

    `requirement` and `terms` are required and neither has a default. `requirement` because every
    answer is an answer ABOUT it. `terms` because the cap and the length floor are rendered into
    question 2 and change what comes back: leaving them out would let a scan capped at 3 terms
    and a scan capped at 30 share a file, and the resulting term list would be a mixture with
    nothing on any row to say so.

    `questions` defaults to `DEFAULT_QUESTIONS`, whose assembly is `NOTE_PROMPT_TEMPLATE`. The
    SELECTED NAMES go into the blob as well as the assembled text, because two scans asking
    different question sets are answering different questions: a row from a scan that never asked
    what a document points at is not a row saying it points nowhere, and only the directory they
    land in can keep those two apart at the file level.
    """
    qs = DEFAULT_QUESTIONS if questions is None else tuple(questions)
    blob = "\0".join([PROMPT_VERSION, SYSTEM_PROMPT, note_prompt(qs),
                      "|".join(q.name for q in qs),
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
class AssertedValue:
    """Question 3's answer for ONE field: what this document says, and the span it was read from.

    THE VALUE IS A MODEL CODING AND IS NOT CHECKED AGAINST THE TEXT, because it cannot be: one
    fact has many notations, and a normalised value often appears nowhere in the document it is
    true of. Checking it would either fail on every correctly normalised answer or be reduced to
    checking nothing.

    `as_written` is what carries the weight instead. It is a verbatim span, it IS checkable
    (`verify`), and it is the difference between a reading error — a real span, coded wrong, which
    is a prompt problem — and an invention, which is a model problem. Any rate computed from
    `value` is therefore lower-confidence than the standing counts beside it, and the record says
    so here rather than in a write-up nobody reads.

    Nothing verifies `as_written` eagerly, unlike a pointer: a pointer that is not in the document
    has nothing left to keep, while a value survives an unverifiable span as evidence that the
    model read something and coded it. So the span rides on the row and `verify` is one call for
    whoever holds the note.
    """

    value: str
    as_written: str

    def __post_init__(self) -> None:
        if not str(self.as_written).strip():
            raise LabelShapeError(
                f"the asserted value {self.value!r} has no as_written span. The value is a coding "
                "and nothing can check it; the span is the only auditable half, so a value "
                "without one is a claim no reader could ever confirm or refute.")

    def verify(self, text: str) -> bool:
        """Is the span this value was read from really in the note? Not: is the value right."""
        return verify_quote(self.as_written, text)


@dataclass(frozen=True)
class LabelReply:
    """The answers to the questions ASKED, parsed out of one reply and not yet checked as a whole.

    A field left at its default is a question that was not asked OR a question answered with
    nothing, and those are told apart at the file level rather than here: the manifest and
    `prompt_hash` say which questions the scan put, so an empty `pointers` in a scan that asked
    for pointers means the document points nowhere, and the same empty tuple in a scan that did
    not ask means nobody looked.

    `n_terms_proposed` is kept although most of it may be discarded: after the cap has done its
    work, it is the only surviving evidence that a model tried to pad, and "the cap bites on 40%
    of notes" is the signal that the cap, or the question, needs rewriting. The pointer counters
    are the same measurement for question 4, and are already final here — see `_parse_pointers`.
    """

    admissibility: Admissibility
    terms: tuple[RetrievalTerm, ...] = ()
    n_terms_proposed: int = 0
    asserted_values: Mapping[str, AssertedValue] = field(default_factory=dict)
    pointers: tuple[str, ...] = ()
    n_pointers_proposed: int = 0
    n_pointers_hallucinated: int = 0
    copied_forward: bool | None = None

    @property
    def retrieval_terms(self) -> tuple[RetrievalTerm, ...]:
        """`terms` under the name `NoteLabel` gives them, for a caller assembling a row.

        Still UNVERIFIED: the name matches the row's, the state does not, and `verify_terms` is
        what closes that gap. It is a property and not a second field so there is one tuple.
        """
        return self.terms


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
    #: Question 3: field name -> the value this document asserts, beside the span it was read
    #: from. Empty on a document that asserts nothing, and on every row written before question 3
    #: was asked — `prompt_hash` is what tells those apart, and it moved.
    asserted_values: Mapping[str, AssertedValue] = field(default_factory=dict)
    #: Question 4, after verification: verbatim spans naming another document, date or identifier.
    #: The counters mirror question 2's, for the same reason — a silently shortened list would read
    #: as a document that points nowhere when it in fact pointed at something the model invented.
    pointers: tuple[str, ...] = ()
    n_pointers_proposed: int = 0
    n_pointers_hallucinated: int = 0
    #: Question 5. `None` is not "no": it means there was nothing to judge, because the document
    #: bears on no field or because the question was not asked. `False` would assert that there IS
    #: content here and that it is original, which on an all-"neither" document is a claim nobody
    #: made, and duplication is most of a real record — this is the denominator every statement
    #: about wasted reading is divided by, so a manufactured `False` inflates it silently.
    copied_forward: bool | None = None
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
        """A row -> a label, refusing loudly anything that is not this generation's shape.

        A ROW WRITTEN BEFORE A QUESTION EXISTED READS AS THE EMPTY DEFAULT AND NEVER AS AN ANSWER.
        912 labels answer questions 1 and 2 only; they stay readable here, with no `pointers`, no
        `asserted_values` and `copied_forward` at `None` — never at `False`, which would be this
        method inventing a judgement nobody made. What the row does NOT carry is that it was never
        asked, and `prompt_hash` on the row is what says so.
        """
        kw = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        kw["admissibility"] = _admissibility_from_dict(d.get("admissibility") or {})
        kw["retrieval_terms"] = tuple(
            RetrievalTerm(**t) for t in (d.get("retrieval_terms") or []))
        kw["asserted_values"] = _asserted_values_from_dict(d.get("asserted_values") or {})
        kw["pointers"] = tuple(str(p) for p in (d.get("pointers") or ()))
        cf = d.get("copied_forward")
        kw["copied_forward"] = cf if isinstance(cf, bool) else None
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


def _as_text(value: Any) -> str:
    """A coded value as it is stored: text, whatever notation the model replied in.

    A boolean field comes back as `true` and a numeric one as a number, and `str(True)` would put
    `"True"` on the row for a field whose own allowable values are `true` / `false`. JSON's
    spelling is the one the reply used and the one a reader of the row will compare against.
    """
    if isinstance(value, str):
        return value.strip()
    return "" if value is None else json.dumps(value, default=str)


def _asserted_values_from_dict(raw: Any) -> dict[str, AssertedValue]:
    """Question 3's stored answer -> the values, or `LabelShapeError`. ABSENT READS AS EMPTY.

    The second place two generations of label meet; `_admissibility_from_dict` is the first and
    the harder one. It is easier here because an added key has an honest empty reading that a
    reshaped key does not: a row with no `asserted_values` was written by a scan that did not ask
    question 3, and an empty mapping says exactly that much and no more. There is nothing to widen
    and therefore nothing to invent.

    A stored entry with no span still raises, through `AssertedValue` itself: this module never
    wrote one, so a row carrying one came from somewhere else and reading it as a value would put
    an unauditable coding into a rate that is reported as measured.
    """
    if not isinstance(raw, Mapping):
        raise LabelShapeError(
            f"'asserted_values' must be an object keyed by field name, got {type(raw).__name__} "
            f"{raw!r}")
    out: dict[str, AssertedValue] = {}
    for key, item in raw.items():
        if not isinstance(item, Mapping):
            raise LabelShapeError(
                f"the asserted value for {key!r} must be an object carrying 'value' and "
                f"'as_written', got {type(item).__name__} {item!r}")
        out[str(key)] = AssertedValue(value=_as_text(item.get("value")),
                                      as_written=str(item.get("as_written") or ""))
    return out


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


def parse_label_response(text: str, *, requirement: Requirement, terms: TermConfig,
                         note_text: str = "",
                         questions: Sequence[ScanQuestion] | None = None) -> LabelReply:
    """Model reply -> the answers to the questions ASKED, or `LabelReplyError`.

    Strict about the things that would corrupt a measurement, lenient about the rest. An unknown
    verdict is a hard error rather than being folded into "neither", because that would put a
    shrug in the standing column and nothing on the row would ever say so.

    ONLY THE SELECTED QUESTIONS ARE PARSED AND ONLY THEY ARE REQUIRED. A question that was not put
    cannot be unanswered, so its keys are neither looked for nor read if they happen to be there —
    its field stays at the empty default. A question that WAS put and whose key is absent is
    refused, because "question 4 unanswered" and "question 4 answered with nothing" are different
    facts and only the second is a measurement: an empty list, an empty object and `null` are all
    answers, and an absent key is not one.

    `requirement` is required and has no default: question 1's answer is only checkable against
    the field list it was asked about, and a parser that did not hold that list would have to
    accept whatever set of keys came back — which is the missing-field bug with extra steps.

    `note_text` is what the copied spans are checked against, and it defaults to empty for the
    callers that parse a reply with no note in hand. Under that default a pointer cannot be
    verified and is therefore discarded — visibly, in `n_pointers_proposed` against an empty
    `pointers`, never silently. The reading path always passes the note; see `label_note`.
    """
    from ..core.llm import extract_json

    qs = DEFAULT_QUESTIONS if questions is None else tuple(questions)
    _refuse_dropping_mandatory([q.name for q in qs])
    obj = extract_json(text or "", require="admissibility")
    if not isinstance(obj, dict) or "__unparsed__" in obj:
        raise LabelReplyError("reply is not the single JSON object the contract asks for")
    absent = [(q.name, k) for q in qs for k in q.reply_keys if k not in obj]
    if absent:
        raise LabelReplyError(
            f"reply has no {[k for _, k in absent]} and this scan asked "
            f"{sorted({n for n, _ in absent})}. A question left unanswered is not a question "
            "answered with nothing: an empty list, an empty object and null are answers, an "
            "absent key is a reply that was not shaped by the contract it was shown.")
    answers: dict[str, Any] = {}
    for q in qs:
        # THE ONLY DISPATCH. Each question parses its own keys, in render order, so a question
        # added to the registry is parsed here without a line being changed and a question left
        # out of the selection cannot raise, cannot fill a field, and cannot be told from one
        # answered with nothing by anything except `prompt_hash`.
        answers.update(q.parse(obj, _ReplyContext(requirement, terms, note_text, answers)))
    return LabelReply(**answers)


def verify_quote(quote: str, text: str) -> bool:
    """Is this span really in the note?

    Whitespace- and case-insensitive, and nothing more. Line wrapping and casing are transport
    damage; a different word is a different quote, and this returns False for it.
    """
    def flat(s):
        return re.sub(r"\s+", " ", s).strip().lower()
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
                 terms: TermConfig, questions: Sequence[ScanQuestion] | None = None):
        self.model, self.requirement, self.terms = model, requirement, terms
        #: WHICH QUESTIONS THIS FILE'S ROWS ANSWER. In the key through `prompt_hash`, because a
        #: row from a scan that never asked what a document points at is not a row saying it
        #: points nowhere, and one file holding both has nothing on either row to say which.
        self.questions = DEFAULT_QUESTIONS if questions is None else tuple(questions)
        self.key = hashlib.sha256("|".join(
            [model, PROMPT_VERSION, requirement.spec_id,
             prompt_hash(requirement, terms, self.questions)]).encode()).hexdigest()[:12]
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
                 # The questions every row in this file answers, for the same reason as `fields`:
                 # a reader can tell at the door whether the scan asked what it came for.
                 "questions": [q.name for q in self.questions],
                 "prompt_hash": prompt_hash(requirement, terms, self.questions),
                 "terms": asdict(terms),
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
    #: WHICH QUESTIONS TO PUT. Defaulted, unlike the two above, because there is a defensible
    #: default — the whole `default_on` set — where there is no defensible default ceiling. It is
    #: part of `prompt_hash`, so a scan that drops one lands in its own directory.
    questions: tuple[ScanQuestion, ...] = DEFAULT_QUESTIONS
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
        if tuple(store.questions) != tuple(config.questions):
            raise LabellingError(
                f"store keyed to questions {[q.name for q in store.questions]}, scan configured "
                f"for {[q.name for q in config.questions]}: those are different questions, and a "
                "row answering one set says nothing about the other.")
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
        req, qs = self.config.requirement, self.config.questions
        base = NoteLabel(
            patient_id=note.patient_id, note_id=note.note_id, doc_type=note.doc_type,
            date=note.date, note_truncated=truncated, spec_id=req.spec_id,
            model=self.store.model, prompt_hash=prompt_hash(req, self.config.terms, qs),
            scanned_at=_utc_now())
        pt = ct = 0
        try:
            resp = self.client.chat(
                build_note_prompt(note, requirement=req, terms=self.config.terms, questions=qs))
            pt = int(getattr(resp, "prompt_tokens", 0) or 0)
            ct = int(getattr(resp, "completion_tokens", 0) or 0)
            reply = parse_label_response(getattr(resp, "content", "") or "", requirement=req,
                                         terms=self.config.terms, note_text=note.text,
                                         questions=qs)
        except Exception as exc:  # noqa: BLE001 — recorded on the label rather than raised
            return replace(base, prompt_tokens=pt, completion_tokens=ct,
                           cost_usd=cost_usd(pt, ct), error=f"{type(exc).__name__}: {exc}"[:300])
        kept, n_hallucinated = verify_terms(reply.terms, note.text)
        adm = reply.admissibility
        # ANSWERED FIELDS ARE COPIED BY NAME, so a question added to the registry reaches the row
        # without a line here — the two records share a field name wherever they hold the same
        # answer. The three below are the ones verification against the note changes on the way,
        # and `terms` is deliberately not among the copied: on the row it is the VERIFIED tuple.
        answered = {f.name: getattr(reply, f.name) for f in dc_fields(reply)
                    if f.name in NoteLabel.__dataclass_fields__}
        answered["admissibility"] = replace(adm, quote_verified=verify_quote(adm.quote, note.text))
        answered["retrieval_terms"] = kept
        answered["n_terms_hallucinated"] = n_hallucinated
        return replace(base, **answered, prompt_tokens=pt, completion_tokens=ct,
                       cost_usd=cost_usd(pt, ct))

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
