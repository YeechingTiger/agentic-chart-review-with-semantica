"""What a corpus scan learned, in a form both planes can hold without importing each other.

    variable -> which document types tend to carry the answer, and at what rate
    variable -> which terms surface an answer-bearing document, and what they cost

THIS FILE IS THE DECOUPLING. `improvement` writes a prior; `review` renders one into a prompt; and
`tests/test_layering.py` forbids either from importing the other — work planes may share only
`core`/`contract` types and artifacts on disk. So the format lives in `contract`, at rank 1, and the
two planes meet through it and through a JSON file. Neither knows the other exists.

WHY IT DID NOT EXIST BEFORE. `review/document_concepts.experience_block` has rendered exactly this
shape since it was written — `queries: [{field, terms, measured_yield}]` plus `document_concepts` —
and had ZERO production callers: `agent.py` mentions it in a comment. `measured_yield` appeared once
in the whole tree, in the renderer that reads it. `assets/experience/` holds a hand-authored prior
whose own header says "Not loaded by anything yet", and `tools/make_review_pack.py` printed the gap
as a table row: *"prior experience (which note types, which keywords) — nothing certified exists
yet."* A complete receiver with no producer, which is this repository's most frequent defect.

## The three numbers, and why each is reported rather than collapsed into a score

A document-type prior is a RATE: of the notes of this type that were read, what fraction could
establish the answer. A term is TWO numbers — how many answer-bearing notes it surfaces and how many
other notes it drags in — because a term that matches everything has perfect recall and no value.
`improvement/derive.py` learned this the expensive way and prices candidates on marginal `cost/gain`;
a prior that reported recall alone would recommend exactly the terms `derive` refuses.

## Held-out discipline travels with the asset

`measured.patient_digests` carries a hash per subject the prior was measured on — never an id, so
the asset is publishable and `tests/test_no_phi_in_tree.py` stays satisfied. A run whose subject is
in that set is running against a prior that saw it, and `informed_by(patient_id)` says so. This
repository has the scar: six adversarial charts were designed by watching the agent fail, the search
cards were written from the same failures, and `analyze_arms.py` had to grow a refusal because the
contamination was only recorded in prose.

## What a prior is NOT

Not a rule, not a gate, and not evidence. `experience_block`'s header says so to the model, and
nothing here is read by the coverage gate or by any answer check. A prior that could refuse an
answer would be a spec element wearing a measurement's clothes — and this repository has already
removed five deterministic content checks after they destroyed 58 correct values against 21 helps.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "acr.retrieval_prior/1"

#: How a term's yield was counted. The distinction is load-bearing and the asset must carry it:
#:
#: `proposed_by_reader` — the scan's reading model proposed this term for this note, and the term
#:   was verified present in the note text. It is a LOWER BOUND on recall: a scan capped at eight
#:   terms per note cannot propose the ninth, so a term absent from a note's list is not evidence
#:   the term is absent from the note.
#: `corpus_matched` — the corpus's own matcher was asked, for every term, of every scanned note.
#:   The real number, and the only basis on which one term may be said to beat another.
YIELD_BASES = ("proposed_by_reader", "corpus_matched")

#: A prior is `draft` until the develop plane certifies it on held-out subjects. There is no path in
#: this module that writes `certified`: `improvement/assetdev.certify` is the only thing entitled to,
#: and it does so by writing a certificate beside the asset, never by editing the asset's own claim.
STATUSES = ("draft", "measured", "certified")


class RetrievalPriorError(ValueError):
    """A prior that cannot be trusted to mean what it says."""


def _digest(value: Any) -> str:
    """A short, stable, key-free digest. Used for subject ids, so an asset carries no person id.

    Not `core.site.fingerprint`: that is HMAC under a site secret, which makes a digest
    unreproducible by anyone who lacks the key — and the whole point here is that a READER of the
    published asset can check whether a subject was in the development set. Unkeyed is correct for a
    membership test over ids the reader already holds, and it is not a protection claim: the asset
    says `patient_digests`, not `patient_ids`, and nothing here reverses.
    """
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class TermYield:
    """One term, and what it bought on the development set.

    `n_surfaced_answer_bearing` / `n_surfaced_other` rather than a single score: the ratio is the
    reader's to form, and a stored score cannot be re-derived under a different cost tolerance.
    """

    term: str
    n_surfaced_answer_bearing: int
    n_surfaced_other: int
    basis: str

    def __post_init__(self) -> None:
        if not str(self.term).strip():
            raise RetrievalPriorError("a term must not be blank")
        if self.n_surfaced_answer_bearing < 0 or self.n_surfaced_other < 0:
            raise RetrievalPriorError(f"{self.term!r}: counts must not be negative")
        if self.basis not in YIELD_BASES:
            raise RetrievalPriorError(
                f"{self.term!r}: unknown yield basis {self.basis!r}; expected one of "
                f"{list(YIELD_BASES)}. A count whose basis is unstated cannot be compared with "
                f"one from a different basis, and `proposed_by_reader` is a lower bound.")

    @property
    def n_surfaced(self) -> int:
        return self.n_surfaced_answer_bearing + self.n_surfaced_other

    def recall(self, n_answer_bearing: int) -> float | None:
        """Of the answer-bearing notes, the share this term surfaces. `None` when there are none.

        `None`, not 0.0: with no answer-bearing note in the development set, the term's recall was
        not measured, and 0.0 would read as measured-and-useless.
        """
        return (round(self.n_surfaced_answer_bearing / n_answer_bearing, 4)
                if n_answer_bearing else None)

    def to_dict(self) -> dict:
        return {"term": self.term,
                "n_surfaced_answer_bearing": self.n_surfaced_answer_bearing,
                "n_surfaced_other": self.n_surfaced_other,
                "basis": self.basis}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TermYield:
        return cls(term=str(d.get("term") or ""),
                   n_surfaced_answer_bearing=int(d.get("n_surfaced_answer_bearing") or 0),
                   n_surfaced_other=int(d.get("n_surfaced_other") or 0),
                   basis=str(d.get("basis") or ""))


@dataclass(frozen=True)
class DocTypeYield:
    """One local document type, and how often a note of it could settle the question.

    `n_scanned` is the DENOMINATOR and it is not optional. A type with one note that established the
    answer is not a better prior than a type with forty that established it thirty times, and a rate
    without its denominator cannot tell them apart.
    """

    doc_type: str
    n_scanned: int
    n_can_establish: int
    n_merely_mentions: int = 0

    def __post_init__(self) -> None:
        if not str(self.doc_type).strip():
            raise RetrievalPriorError("a document type must not be blank")
        if self.n_scanned <= 0:
            raise RetrievalPriorError(f"{self.doc_type!r}: n_scanned must be positive")
        if self.n_can_establish > self.n_scanned:
            raise RetrievalPriorError(
                f"{self.doc_type!r}: {self.n_can_establish} establishing of {self.n_scanned} "
                f"scanned is impossible")

    @property
    def rate(self) -> float:
        return round(self.n_can_establish / self.n_scanned, 4)

    def to_dict(self) -> dict:
        return {"doc_type": self.doc_type, "n_scanned": self.n_scanned,
                "n_can_establish": self.n_can_establish,
                "n_merely_mentions": self.n_merely_mentions, "rate": self.rate}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> DocTypeYield:
        return cls(doc_type=str(d.get("doc_type") or ""),
                   n_scanned=int(d.get("n_scanned") or 0),
                   n_can_establish=int(d.get("n_can_establish") or 0),
                   n_merely_mentions=int(d.get("n_merely_mentions") or 0))


@dataclass(frozen=True)
class FieldPrior:
    """One variable's prior: the terms, and how many notes could settle it at all."""

    field_name: str
    n_answer_bearing: int
    n_notes: int
    terms: tuple[TermYield, ...] = ()
    doc_types: tuple[DocTypeYield, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.field_name).strip():
            raise RetrievalPriorError("a field prior must name its field")
        if self.n_answer_bearing > self.n_notes:
            raise RetrievalPriorError(
                f"{self.field_name}: {self.n_answer_bearing} answer-bearing of {self.n_notes} "
                f"notes is impossible")

    @property
    def is_empty(self) -> bool:
        """No note in the development set could establish this field.

        REPORTED, not omitted. A field the scan found nothing for is a finding about the corpus or
        the requirement — "the record is silent about this variable" — and dropping it from the asset
        would make it indistinguishable from a field nobody scanned.
        """
        return self.n_answer_bearing == 0

    def to_dict(self) -> dict:
        return {"field": self.field_name, "n_answer_bearing": self.n_answer_bearing,
                "n_notes": self.n_notes,
                "terms": [t.to_dict() for t in self.terms],
                "doc_types": [d.to_dict() for d in self.doc_types]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> FieldPrior:
        return cls(field_name=str(d.get("field") or ""),
                   n_answer_bearing=int(d.get("n_answer_bearing") or 0),
                   n_notes=int(d.get("n_notes") or 0),
                   terms=tuple(TermYield.from_dict(x) for x in (d.get("terms") or [])),
                   doc_types=tuple(DocTypeYield.from_dict(x) for x in (d.get("doc_types") or [])))


@dataclass(frozen=True)
class Measured:
    """WHERE the prior came from. Every field here is what makes it re-derivable or refutable."""

    n_patients: int
    n_notes: int
    #: One digest per subject scanned, never an id. Both the held-out check and the promise that
    #: this asset may be committed and shared.
    patient_digests: tuple[str, ...] = ()
    spec_id: str = ""
    #: The reading model and prompt the labels are conditional on. A label is an answer to a
    #: question asked in a particular way by a particular deployment; a prior that does not carry
    #: them is an unattributable claim, which is the reason `NoteLabel` carries them per row.
    model: str = ""
    prompt_hash: str = ""
    labelling_id: str = ""
    scanned_at: str = ""

    def __post_init__(self) -> None:
        if self.n_patients <= 0 or self.n_notes <= 0:
            raise RetrievalPriorError("a prior measured on nothing is not a prior")

    def to_dict(self) -> dict:
        return {"n_patients": self.n_patients, "n_notes": self.n_notes,
                "patient_digests": list(self.patient_digests), "spec_id": self.spec_id,
                "model": self.model, "prompt_hash": self.prompt_hash,
                "labelling_id": self.labelling_id, "scanned_at": self.scanned_at}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Measured:
        return cls(n_patients=int(d.get("n_patients") or 0),
                   n_notes=int(d.get("n_notes") or 0),
                   patient_digests=tuple(str(x) for x in (d.get("patient_digests") or [])),
                   spec_id=str(d.get("spec_id") or ""), model=str(d.get("model") or ""),
                   prompt_hash=str(d.get("prompt_hash") or ""),
                   labelling_id=str(d.get("labelling_id") or ""),
                   scanned_at=str(d.get("scanned_at") or ""))


@dataclass(frozen=True)
class RetrievalPrior:
    """One corpus scan, folded into what a run can be told about where answers live."""

    asset_id: str
    version: str
    status: str
    measured: Measured
    fields: tuple[FieldPrior, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not str(self.asset_id).strip():
            raise RetrievalPriorError("a prior must have an asset_id")
        if self.status not in STATUSES:
            raise RetrievalPriorError(
                f"unknown status {self.status!r}; expected one of {list(STATUSES)}")
        if not self.fields:
            raise RetrievalPriorError(
                "a prior with no field is a file, not a prior. A scan that established nothing "
                "should emit a field marked empty, so that 'the record is silent' and 'nobody "
                "looked' stay distinguishable.")

    # -- the two questions a consumer asks ----------------------------------------------------
    def field_prior(self, name: str) -> FieldPrior | None:
        return next((f for f in self.fields if f.field_name == name), None)

    def informed_by(self, patient_id: str) -> bool:
        """Was this subject in the development set this prior was measured on?

        The held-out check, and the reason `patient_digests` exists. A run on a subject the prior
        saw is not evidence that the prior generalises, and `analyze_arms.py` already refuses to
        fold a chart that informed a method into a headline number.
        """
        return _digest(patient_id) in set(self.measured.patient_digests)

    @property
    def content_hash(self) -> str:
        """Identity for the manifest. Over the CANONICAL DICT, so key order cannot change it."""
        return hashlib.sha256(
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
            .encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"schema": SCHEMA, "asset_id": self.asset_id, "version": self.version,
                "status": self.status, "measured": self.measured.to_dict(),
                "fields": [f.to_dict() for f in self.fields]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> RetrievalPrior:
        got = str(d.get("schema") or "")
        if got and got != SCHEMA:
            raise RetrievalPriorError(
                f"this is a {got!r} document and this reader is {SCHEMA!r}. Refusing rather than "
                f"reading a prior whose fields may mean something else.")
        return cls(asset_id=str(d.get("asset_id") or ""), version=str(d.get("version") or "0"),
                   status=str(d.get("status") or "draft"),
                   measured=Measured.from_dict(d.get("measured") or {}),
                   fields=tuple(FieldPrior.from_dict(x) for x in (d.get("fields") or [])))

    @classmethod
    def load(cls, path: str | Path) -> RetrievalPrior:
        p = Path(path).expanduser()
        if not p.is_file():
            raise RetrievalPriorError(f"no retrieval prior at {p}")
        try:
            return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))
        except json.JSONDecodeError as e:
            raise RetrievalPriorError(f"{p}: {e}") from e


def prior_digest(patient_id: str) -> str:
    """The subject digest a prior stores. Exposed so a caller can check membership itself."""
    return _digest(patient_id)


def to_experience_asset(prior: RetrievalPrior, *, fields: Sequence[str] | None = None,
                        max_terms: int = 12, min_doc_type_rate: float = 0.0,
                        min_doc_type_scanned: int = 1) -> dict:
    """Render a prior into the dict `review.document_concepts.experience_block` already consumes.

    A SEPARATE FUNCTION, and in `contract` rather than in either plane, because it is the shape of
    the seam: `experience_block` was written first and its keys (`queries`, `document_concepts`,
    `measured_yield`) are the published interface. Rewriting the renderer to match a new dataclass
    would break the one part of this chain that was already correct.

    `max_terms` and the two doc-type floors are the caller's policy, with no hidden default beyond
    "show everything": what belongs in a prompt is a budget decision, and a silently truncated list
    would let a prior claim it offered a term it never showed. The rendered dict records the cut.
    """
    want = list(fields) if fields else [f.field_name for f in prior.fields]
    queries: list[dict] = []
    concepts: dict[str, DocTypeYield] = {}
    dropped_terms = 0
    for name in want:
        fp = prior.field_prior(name)
        if fp is None:
            continue
        ranked = sorted(fp.terms,
                        key=lambda t: (-t.n_surfaced_answer_bearing, t.n_surfaced_other, t.term))
        keep = ranked[:max_terms] if max_terms > 0 else ranked
        dropped_terms += len(ranked) - len(keep)
        if keep:
            queries.append({
                "id": f"{name}.prior",
                "field": name,
                "terms": [t.term for t in keep],
                "measured_yield": (
                    f"{fp.n_answer_bearing} of {fp.n_notes} scanned notes could establish this "
                    f"field; the terms above surfaced "
                    f"{max(t.n_surfaced_answer_bearing for t in keep)} of them at best "
                    f"({keep[0].basis})"),
            })
        for d in fp.doc_types:
            if d.n_scanned < min_doc_type_scanned or d.rate < min_doc_type_rate:
                continue
            prev = concepts.get(d.doc_type)
            # Union across fields: a type that establishes ANY field under study is high-yield for
            # the scan as a whole, and the per-field detail is in `queries`.
            if prev is None or d.rate > prev.rate:
                concepts[d.doc_type] = d
    return {
        "asset_id": prior.asset_id,
        "version": prior.version,
        "status": prior.status,
        "measured": (f"{prior.measured.n_patients} patient(s), {prior.measured.n_notes} note(s)"
                     + (f", model {prior.measured.model}" if prior.measured.model else "")),
        "queries": queries,
        "document_concepts": [
            {"concept": d.doc_type,
             "note": (f"{d.n_can_establish} of {d.n_scanned} scanned notes of this type could "
                      f"establish the answer ({d.rate:.0%})")}
            for d in sorted(concepts.values(), key=lambda x: (-x.rate, x.doc_type))],
        #: What was NOT shown. A truncated prior that does not say so is a prior claiming credit
        #: for a term the model never saw.
        "n_terms_withheld": dropped_terms,
    }
