"""FIRST-ORDER DERIVATION — read the labelling, price the words, write the plan.

WHAT THIS REPLACES, AND WHY THE REPLACEMENT IS SMALLER
------------------------------------------------------
`assetdev.py` develops the same two assets by hill-climbing: propose neighbours, measure,
keep the best, repeat, then certify on a held-out split. That is a SECOND-ORDER refinement —
it improves a list that already exists. It cannot tell you what the list should have been,
and its search cost buys nothing on the first pass, because on the first pass the labels
already contain the answer. A model that has read every note has already said which words
indicate the field and which document types carried it; the derivation is to COUNT that and
PRICE it. Search is what you do afterwards, if the counted list is not good enough.

So this module is four counting stages and no search, no split, no negative control, and no
model call anywhere. Everything it consumes is already on disk.

THE DIVISION OF LABOUR, which is the one idea here
--------------------------------------------------
  the MODEL says what INDICATES the answer     (stage 1: read the labels)
  GREP says what it COSTS to look for          (stage 2: read the cached bitmaps)

Neither can do the other's job. A word count cannot recognise that "final diagnosis" is a
section heading that decides cases; a language model asked to guess how many of 276,054
documents contain the string "nod" will invent a number. Keeping the two apart is why the
ranking in stage 3 is trustworthy: every row of it is half measurement of meaning and half
measurement of cost, and neither half was imagined.

THE STRATUM BUG THIS MODULE EXISTS TO UNDO (stage 4)
----------------------------------------------------
`assets/specs/STORE.400_522_523` has a stratum named `cannot_establish`, and that one name bundles
two independent facts:

    MAY NOT ESTABLISH the field   — a clinical judgement about a document type's standing
    CARRIES NOTHING about it      — an empirical fact about what its text says

Cross-sectional imaging is the first and not the second: it may not establish histology, and
it is one of the highest-yield sources in the chart for primary_site. Filing it under a name
that asserts both taught the agent to skip it, and patient P03 was coded C349 (lung NOS) on
2026-07-26 while "right upper lobe" sat in seven imaging and oncology notes. So this module
derives on TWO AXES and never collapses them:

    ADMISSIBILITY   may this type establish this field?  CLINICAL, declared in the spec's
                    `strata[].establishes`, owned by a clinician, NEVER inferred from data.
                    An undeclared type raises rather than being guessed at.
    YIELD           how often does it in fact carry it?  EMPIRICAL, counted from labels,
                    measured PER FIELD — one CT type is high-yield for primary_site and
                    zero-yield for histology, and the policy must be free to differ.

              admissible + high yield   -> READ_ALL
              admissible + low  yield   -> SEARCH    (admissible but rare; reach by keyword)
              inadmissible + high yield -> SEARCH    (corroboration and the absence proof —
                                                      useful, never the witness)
              inadmissible + low  yield -> SAMPLE

Yield is PER FIELD because the labelling now answers per field. It did not always: a previous
labelling returned ONE standing verdict per note against the whole requirement, and this module
applied that single verdict to every field of it — every row of the matrix for a given doc_type
came out identical, and the per-field matrix was a per-field-shaped table of per-note numbers.
That fallback is gone. A labelling that carries a note-level verdict is REFUSED
(`StaleLabellingError`), because the alternative is a table that looks measured per field and is
not, and nothing on any row would say so.

HOW MANY ASSETS DOES A SPEC NEED? (stage 5)
-------------------------------------------
One keyword list and one policy per spec is a guess; one per field is a different guess. The
labels answer it. Two fields whose can_establish note sets largely COINCIDE are served by one
asset — the same documents establish both, so the same terms reach them and the same read policy
is right for both. Two fields whose sets DIVERGE need their own: in this repo's own specs,
histology and behavior come from much the same documents, while the anatomical site also comes
from imaging that can never establish a morphology, and in the staging spec the clinical group
and the pathologic group have systematically different sources.

So stage 5 emits the field-by-field overlap matrix — |A n B| over |A u B| on establishing notes,
measured, threshold-free — and a suggested grouping cut from it at a REQUIRED threshold, with the
WHOLE matrix printed beside the suggestion so the person choosing the cut sees which fields each
setting merges. The grouping is a PROPOSAL and there is no writer that installs one: merging two
fields' assets changes what evidence is sought for each, which is semantic, and a Jaccard
coefficient is not entitled to make that call.

WHO MAY ADOPT WHAT — enforced here, not just documented
-------------------------------------------------------
KEYWORDS are RETRIEVAL-ONLY. They change which text reaches the agent and nothing about what
an answer means, so a derived list may be written into a spec, with provenance, by
`write_keywords`. POLICY and STRATA are SEMANTIC, because admissibility is a clinical
judgement that no count can make. `emit_policy_proposal` writes a file a clinician signs and
never touches the spec. There is deliberately no flag that changes that, and
`assert_no_semantic_override` runs at import over this module's own public surface so that
adding one is an ImportError rather than a code review someone might lose.

NO DEFAULT THRESHOLDS
---------------------
Every cut-off is a required field of `DerivationConfig`, which has no defaults at all, and
the config is recorded in every result object. A default baked in today is a decision made
by whoever typed it, at a time when nobody had seen the curve; the curve is emitted whole
alongside the cut precisely so the person choosing gets to see what each setting buys.

PHI
---
Patients are counted, never named. `DocBitmaps` stores an integer index per patient and
discards the id at load, so no artefact this module writes can contain one.
"""
from __future__ import annotations

import gzip
import inspect
import json
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import date
from pathlib import Path
from typing import Any

import typer
import yaml

from ..chartstore.corpus import DocMeta
from ..contract.spec import ProvenanceRecord, _content_hash, load_spec
from ..contract.strata import assign_strata, strata_from_spec
from ..core import site

#: Where the cached document bitmaps live. Built once by `termcache/build_cache.py`; this
#: module reads them and never rescans a chart.
DEFAULT_CACHE_ROOT = str(site.TERMCACHE_ROOT)

#: The four policies stage 4 can assign. Closed on purpose: a fifth would be a new promise
#: to the run plane, and the run plane has to be taught it first.
READ_ALL, SEARCH, SAMPLE = "READ_ALL", "SEARCH", "SAMPLE"

#: The stage-1 input, as a record shape rather than an import:
#:
#:   NoteLabel.retrieval_terms : Sequence[RetrievalTerm]
#:   RetrievalTerm             : {"term": str, "reason": str}   (a bare str is also accepted)
#:   NoteLabel.admissibility   : {field: verdict}               — ONE VERDICT PER FIELD
#:
#: THESE TWO NAMES ARE THE WIRE between `labelling.py` and this module, and they are spelled
#: exactly once each, here. An earlier version of this module read a `candidate_terms` key the
#: labeller has never written, kept `retrieval_terms` as a fallback, and would have gone on
#: aggregating zero terms in silence if the fallback had ever been dropped. There is now one
#: name per question and `tests/test_derive.py::test_the_key_the_labeller_writes_is_the_key_
#: derive_reads` builds a real `labelling.NoteLabel` and fails if either drifts — an
#: integration mismatch that yields empty input is the worst kind, because the pipeline still
#: runs and every number it prints is quietly wrong.
TERMS_FIELD = "retrieval_terms"
ADMISSIBILITY_FIELD = "admissibility"

#: `reason` is a REASON CLASS: why the note's reader thinks the word points at the field —
#: e.g. it names the answer, the document, or a section heading. The classes are NOT
#: validated against a list here. Folding an unrecognised class into "other" would erase a
#: prompt change silently, so an unknown class is counted under its own name and surfaced on
#: `Aggregate.unknown_reason_classes` where a reader trips over it.
REASON_CLASSES = ("names_the_answer", "names_the_document", "names_the_section", "other")
UNCLASSIFIED = "unclassified"

#: Question 1's answers, per field. The same three the labeller offers; a fourth arriving here
#: is a prompt change this module has not been taught, and it raises rather than being counted
#: into a column nobody reads.
VERDICTS = ("can_establish", "merely_mentions", "neither")
BEARS_ON_QUESTION = ("can_establish", "merely_mentions")


class DerivationError(Exception):
    """Base for everything this module refuses to do."""


class UnpricedTermError(DerivationError):
    """A candidate term is not in the cached bitmaps, so nothing can say what it costs.

    Not a warning and not a skip. Dropping it would turn the stage-3 ranking into a ranking
    of the terms that happened to be cached, which reads identically to a ranking of the
    terms that are good. Rebuild the cache with these needles, or drop them deliberately.
    """


class StaleLabellingError(DerivationError):
    """A labelling that answers question 1 once per NOTE, not once per FIELD.

    Refused rather than spread across the fields. Spreading it is what this module used to do,
    and it produces a per-(doc_type, field) matrix whose every row for a doc_type is the same
    number — which is not a smaller result than the real one, it is a WRONG one wearing the
    right shape. Rescan with the current labelling prompt.
    """


class UndeclaredAdmissibilityError(DerivationError):
    """A document type no stratum in the spec speaks for.

    Admissibility is clinical. The honest answer to "may this type establish this field" is
    a clinician's, and this module will not manufacture one from a frequency.
    """


class SemanticOverrideError(DerivationError):
    """Something in this module's public surface offers to write a semantic asset anyway."""


class AdoptionAborted(DerivationError):
    """A keyword write did not verify, so nothing was written."""


# ============================================================================
# CONFIG — every cut-off, all required, recorded in every result
# ============================================================================

@dataclass(frozen=True)
class DerivationConfig:
    """The four numbers this derivation cannot compute for itself.

    NOT ONE OF THEM HAS A DEFAULT, and that is the point. Each is a trade-off with an owner:
    how many documents a rescued answer is worth, how often a document type has to carry a
    field before every one of them must be read, how many patients have to propose a word
    before it is the corpus's vocabulary rather than one patient's, and how far two fields'
    sources have to coincide before one asset serves both. A default would settle all four at
    import time, in a file nobody reads, before anyone had seen the curve — and afterwards
    nothing in the output would say a choice had been made.
    """

    #: Stage 3 cut. Extra documents the agent may be made to read per additional answer
    #: rescued. Above this, a term stops paying for itself.
    max_extra_docs_per_answer: float
    #: Stage 4 split. Fraction of a document type's notes that must state a field for that
    #: (type, field) to count as HIGH yield.
    high_yield_rate: float
    #: Stage 1 floor. How many distinct PATIENTS must propose a term before it is priced.
    #: Patients, not notes: a word proposed forty times for one patient is that patient's
    #: vocabulary, and forty is exactly the number that makes it look like the corpus's.
    min_patients_proposing: int
    #: Stage 5 cut. Jaccard overlap of two fields' establishing note sets at or above which the
    #: two are PROPOSED to share one keyword list and one policy. No default, and less
    #: defaulted than the others: this one decides how many assets a spec has, and the whole
    #: matrix is printed beside it because 0.6 and 0.8 merge different fields and only the
    #: person reading the matrix knows whether that is the same question or two.
    share_asset_jaccard: float

    def __post_init__(self) -> None:
        if self.max_extra_docs_per_answer < 0:
            raise ValueError("max_extra_docs_per_answer is documents per answer; it cannot be "
                             "negative")
        if not 0.0 <= self.high_yield_rate <= 1.0:
            raise ValueError(f"high_yield_rate is a rate in [0,1], got {self.high_yield_rate!r}")
        if self.min_patients_proposing < 1:
            raise ValueError("min_patients_proposing < 1 admits terms no patient proposed")
        if not 0.0 <= self.share_asset_jaccard <= 1.0:
            raise ValueError(f"share_asset_jaccard is a Jaccard coefficient in [0,1], got "
                             f"{self.share_asset_jaccard!r}")

    def as_dict(self) -> dict[str, Any]:
        return {"max_extra_docs_per_answer": self.max_extra_docs_per_answer,
                "high_yield_rate": self.high_yield_rate,
                "min_patients_proposing": self.min_patients_proposing,
                "share_asset_jaccard": self.share_asset_jaccard}


# ============================================================================
# STAGE 1 — AGGREGATE the per-note labels
# ============================================================================

@dataclass(frozen=True)
class TermEvidence:
    """One candidate term, as the labelling proposed it."""

    term: str
    n_notes: int
    n_patients: int
    reasons: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        return _d(self, reasons=dict(sorted(self.reasons.items())))


@dataclass(frozen=True)
class TypeEvidence:
    """One (doc_type, field), as the labelling found it.

    `n_patients_sole_source` is the number that decides policy. A type seen in 2% of notes
    that is the only establishing source for thirty patients outranks a type in 40% of notes
    that never decides anything, and no frequency ranking will ever show you that.
    """

    doc_type: str
    field: str
    n: int
    n_states: int
    n_can_establish: int
    n_merely_mentions: int
    n_neither: int
    n_patients_sole_source: int

    @property
    def yield_rate(self) -> float:
        """How often this type carries this field. Per FIELD — see the module docstring."""
        return self.n_states / self.n if self.n else 0.0

    def as_dict(self) -> dict[str, Any]:
        return _d(self, yield_rate=round(self.yield_rate, 6))


@dataclass(frozen=True)
class Aggregate:
    """Stage 1's whole output. Counts only: no patient id and no note text survives here."""

    terms: dict[str, TermEvidence]
    types: dict[tuple[str, str], TypeEvidence]
    n_notes: int
    n_patients: int
    fields: tuple[str, ...]
    unknown_reason_classes: tuple[str, ...] = ()
    #: The stage-5 input, and the smallest thing it can be: for each COMBINATION of fields a
    #: note can establish, how many notes establish exactly that combination. Every pairwise
    #: overlap follows from it, it is bounded by the number of combinations actually seen
    #: rather than by the corpus, and it holds no note id and no patient id — a set of note
    #: indices per field would answer the same question and would be a thing to leak.
    establishing_profiles: dict[tuple[str, ...], int] = field(default_factory=dict)

    def ranked_terms(self, cfg: DerivationConfig) -> list[TermEvidence]:
        """Terms at or above the patient floor, most-patients first. Notes break ties only."""
        keep = [t for t in self.terms.values() if t.n_patients >= cfg.min_patients_proposing]
        return sorted(keep, key=lambda t: (-t.n_patients, -t.n_notes, t.term))

    def n_establishing(self, *names: str) -> int:
        """Notes that can establish EVERY one of `names`. One name: that field's set size."""
        want = set(names)
        return sum(n for combo, n in self.establishing_profiles.items() if want <= set(combo))

    def as_dict(self) -> dict[str, Any]:
        return _d(self, fields=list(self.fields),
                  unknown_reason_classes=list(self.unknown_reason_classes),
                  terms=[t.as_dict() for t in
                         sorted(self.terms.values(), key=lambda t: (-t.n_patients, t.term))],
                  types=[t.as_dict() for t in
                         sorted(self.types.values(), key=lambda t: (t.field, t.doc_type))],
                  establishing_profiles=[{"fields": list(c), "n_notes": n} for c, n in
                                         sorted(self.establishing_profiles.items(),
                                                key=lambda kv: (-kv[1], kv[0]))])


def _d(obj: Any, **extra: Any) -> dict[str, Any]:
    """Any record here -> a plain dict, skipping the fields marked `repr=False`.

    Those are working state — `TermPrice.rows` is a set of document indices, and an artefact
    carrying one per term is megabytes of integers nobody reads.
    """
    return {f.name: getattr(obj, f.name) for f in fields(obj) if f.repr} | extra


def normalise_term(raw: str) -> str:
    """Case and whitespace only. A different word is a different term — see `_subsumes`."""
    return re.sub(r"\s+", " ", str(raw or "")).strip().lower()


def retrieval_terms(label: Any) -> list[tuple[str, str]]:
    """(term, reason_class) pairs off one label, tolerating the field's absence.

    Absence is not an error: a labelling written before this question existed returns [], which
    means stage 1 counts document types and no terms — a smaller derivation and not a wrong one.
    ONE key is read, and it is the key `labelling.NoteLabel` writes. A second name read as a
    fallback would mean a rename in either module still produced a full run over zero terms.
    """
    raw = _attr(label, TERMS_FIELD)
    out: list[tuple[str, str]] = []
    for item in (raw or ()):
        if isinstance(item, str):
            term, reason = item, UNCLASSIFIED
        elif isinstance(item, Mapping):
            term, reason = item.get("term", ""), item.get("reason") or UNCLASSIFIED
        else:
            term, reason = getattr(item, "term", ""), getattr(item, "reason", "") or UNCLASSIFIED
        term = normalise_term(term)
        if term:
            out.append((term, str(reason)))
    return out


#: Keys that belong to a NOTE-level answer. `labelling.NoteLabel.to_dict` writes a collapsed
#: `verdict` beside the per-field map, for readers that have not learned to ask per field; this
#: module has, so it takes the map and the collapse is never read. A row carrying ONLY these is
#: the old shape, and it is refused — the same refusal, by the same reasoning, that
#: `labelling._admissibility_from_dict` makes on the way in.
_NOTE_LEVEL_KEYS = frozenset({"verdict", "quote", "quote_verified"})


def _per_field_map(adm: Any) -> Mapping[str, Any] | None:
    """The {field: verdict} map, out of whichever wrapper it arrived in, or None if there is
    none. A mapping under `verdicts`, an object with a `verdicts` attribute, or the bare map."""
    inner = adm.get("verdicts") if isinstance(adm, Mapping) else getattr(adm, "verdicts", None)
    if isinstance(inner, Mapping):
        return inner
    if isinstance(adm, Mapping) and adm and not (adm.keys() & _NOTE_LEVEL_KEYS):
        return adm
    return None


def field_verdicts(label: Any, fields: Sequence[str]) -> dict[str, str]:
    """This note's standing ON EACH FIELD: {field: one of `VERDICTS`}.

    The whole of yield, and deliberately not the value: a value this module compared to an
    answer key would be scoring, not counting.

    Three refusals, all of them because the quiet alternative is a wrong number that looks
    right. A NOTE-LEVEL verdict — one `verdict` on the admissibility object rather than a map
    over fields — is `StaleLabellingError`: the old shape, and copying it onto every field is
    what this change exists to undo. A field the labelling does not answer for is the same
    refusal: it means these labels were made against a different requirement, and calling the
    silence "neither" would push a type's yield down with notes that were never asked. And a
    verdict outside `VERDICTS` raises, because a fourth class the prompt learned and this
    module did not would otherwise land in a column nothing reads.
    """
    adm = _attr(label, ADMISSIBILITY_FIELD)
    per_field = _per_field_map(adm)
    if per_field is None:
        scalar = adm.get("verdict") if isinstance(adm, Mapping) else getattr(adm, "verdict", None)
        raise StaleLabellingError(
            f"this labelling answers question 1 once per NOTE (verdict={scalar!r}), and this "
            f"module needs it once per FIELD: {{{', '.join(repr(f) for f in fields)}}} -> "
            f"can_establish | merely_mentions | neither. Applying one note-level verdict to "
            f"every field yields a per-field matrix whose rows are all the same number, which "
            f"is not a coarser answer but a wrong one. Rescan with the current prompt.")
    if not per_field:
        raise StaleLabellingError(
            "this label answered question 1 for no field at all. An empty standing is not "
            "'neither' for every field: it is a reading that did not happen, and counting it "
            "moves every denominator it touches.")
    out: dict[str, str] = {}
    for f in fields:
        v = per_field.get(f)
        if isinstance(v, Mapping):
            v = v.get("verdict")
        elif v is not None and not isinstance(v, str):
            v = getattr(v, "verdict", None)
        if v is None:
            raise StaleLabellingError(
                f"the labelling has no standing for field {f!r} (it answers for "
                f"{sorted(per_field)!r}). These labels were made against a different "
                f"requirement; counting the silence as 'neither' would understate every type "
                f"that carries the field.")
        if v not in VERDICTS:
            raise DerivationError(f"verdict {v!r} for field {f!r} is not one of {list(VERDICTS)}")
        out[f] = v
    return out


def _attr(label: Any, name: str) -> Any:
    return label.get(name) if isinstance(label, Mapping) else getattr(label, name, None)


def aggregate(labels: Iterable[Any], fields: Sequence[str]) -> Aggregate:
    """STAGE 1. Per-note labels -> per-term, per-(type, field) and per-field-combination evidence.

    A label that failed to parse is dropped, because a note the reader could not read says
    nothing about a document type's yield and padding the denominator with it understates
    every type it touches.
    """
    if not fields:
        raise DerivationError("aggregate needs the fields to count; with none it counts nothing")
    notes = 0
    patients: set[str] = set()
    term_pat: dict[str, set[str]] = defaultdict(set)
    term_n: Counter[str] = Counter()
    term_reason: dict[str, Counter[str]] = defaultdict(Counter)
    cells: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    #: (patient, field) -> the document types that ESTABLISHED it, per that field's own verdict.
    established: dict[tuple[str, str], set[str]] = defaultdict(set)
    #: The stage-5 tally: which fields this one note can establish, counted as a combination.
    profiles: Counter[tuple[str, ...]] = Counter()

    for lab in labels:
        if _attr(lab, "error"):
            continue
        pid, doc_type = str(_attr(lab, "patient_id") or ""), str(_attr(lab, "doc_type") or "")
        notes += 1
        patients.add(pid)
        verdicts = field_verdicts(lab, fields)
        for (term, reason) in retrieval_terms(lab):
            term_n[term] += 1
            term_pat[term].add(pid)
            term_reason[term][reason] += 1
        for f in fields:
            verdict = verdicts[f]
            c = cells[(doc_type, f)]
            c["n"] += 1
            c["n_states"] += int(verdict in BEARS_ON_QUESTION)
            c[f"n_{verdict}"] += 1
            if verdict == "can_establish":
                established[(pid, f)].add(doc_type)
        combo = tuple(f for f in fields if verdicts[f] == "can_establish")
        if combo:  # a note that establishes nothing is in no field's set and joins no overlap
            profiles[combo] += 1

    sole: Counter[tuple[str, str]] = Counter()
    for (pid, f), types in established.items():
        if len(types) == 1:  # ONLY establishing source for this patient — the number that matters
            sole[(next(iter(types)), f)] += 1

    terms = {t: TermEvidence(t, term_n[t], len(term_pat[t]), dict(term_reason[t])) for t in term_n}
    types = {k: TypeEvidence(k[0], k[1], c["n"], c["n_states"], c["n_can_establish"],
                             c["n_merely_mentions"], c["n_neither"], sole[k])
             for k, c in cells.items()}
    unknown = sorted({r for c in term_reason.values() for r in c} - set(REASON_CLASSES)
                     - {UNCLASSIFIED})
    return Aggregate(terms, types, notes, len(patients), tuple(fields), tuple(unknown),
                     dict(profiles))


# ============================================================================
# STAGE 2 — PRICE every candidate by grep, out of the cache
# ============================================================================

@dataclass(frozen=True)
class DocBitmaps:
    """The cached corpus, as bits. One row per document, patients as integers.

    The pid is dropped at load rather than carried and filtered later: a field that does not
    exist cannot leak into an artefact, and `tests/test_no_phi_in_tree.py` is in this repo
    because that material reached the tree once already.
    """

    needles: tuple[str, ...]
    rows: tuple[tuple[int, int, bool], ...]   # (patient_ix, needle bitmask, answer_bearing)
    n_patients: int

    def matched(self, term: str) -> frozenset[int]:
        """Row ids whose text contains `term`. Raises rather than rescanning the corpus."""
        j = {n: k for k, n in enumerate(self.needles)}.get(term)
        if j is None:
            raise UnpricedTermError(f"{term!r} is not in the cached bitmaps ({len(self.needles)} "
                                    f"needles). Rebuild the cache with it; this module does not "
                                    f"rescan charts.")
        bit = 1 << j
        return frozenset(i for i, (_, mask, _) in enumerate(self.rows) if mask & bit)


def load_bitmaps(root: str | Path = DEFAULT_CACHE_ROOT, *, chunks: str = "v1") -> DocBitmaps:
    """Read `meta.json` + the chunk pickles. No corpus read, no model call, no network."""
    root = Path(root)
    meta = json.loads((root / "meta.json").read_text(encoding="utf-8"))
    rows: list[tuple[int, int, bool]] = []
    ix, nxt = {}, 0
    for path in sorted((root / chunks).glob("chunk*.pkl.gz")):
        with gzip.open(path, "rb") as fh:
            for rec in pickle.load(fh):
                pid = rec["pid"]
                if pid not in ix:
                    ix[pid], nxt = nxt, nxt + 1
                p = ix[pid]
                rows.extend((p, int(m), bool(o)) for m, o in zip(rec["hits"], rec["oracle"]))
    return DocBitmaps(tuple(meta.get("needles") or ()), tuple(rows), len(ix))


@dataclass(frozen=True)
class TermPrice:
    """What one term costs and buys, measured against the list already in the spec."""

    term: str
    documents_matched: int
    answer_bearing_matched: int
    answers_rescued: int      # patients the CURRENT list surfaces nothing for, that this one does
    extra_documents: int      # documents the current list does not already open
    rows: frozenset[int] = field(default=frozenset(), repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return _d(self)


def _answered(bm: DocBitmaps, rows: frozenset[int]) -> frozenset[int]:
    """The patients a set of documents actually answers for. Patients, not documents: opening
    six answer-bearing notes for one patient answers one question."""
    return frozenset(bm.rows[i][0] for i in rows if bm.rows[i][2])


def price_terms(terms: Sequence[str], bm: DocBitmaps,
                current: Sequence[str] = ()) -> list[TermPrice]:
    """STAGE 2. Grep prices what the model proposed. The two never swap jobs.

    Priced against the CURRENT list, not against nothing: a term that matches ten thousand
    answer-bearing documents is worth zero if the shipped list already opens every one of
    them, and a raw recall number would rank it first.
    """
    have: frozenset[int] = frozenset()
    for t in current:
        have |= bm.matched(normalise_term(t))
    have_pat = _answered(bm, have)
    out = []
    for t in terms:
        rows = bm.matched(normalise_term(t))
        pats = _answered(bm, rows)
        out.append(TermPrice(normalise_term(t), len(rows),
                             sum(1 for i in rows if bm.rows[i][2]),
                             len(pats - have_pat), len(rows - have), rows))
    return out


# ============================================================================
# STAGE 3 — CONSOLIDATE into the list
# ============================================================================

@dataclass(frozen=True)
class CurveRow:
    """One step of the greedy ranking. The whole curve is emitted, cut or not."""

    rank: int
    term: str
    marginal_answers_rescued: int
    marginal_extra_documents: int
    docs_per_answer: float
    cum_answers_rescued: int
    cum_extra_documents: int
    in_cut: bool

    def as_dict(self) -> dict[str, Any]:
        return _d(self, docs_per_answer=round(self.docs_per_answer, 4))


@dataclass(frozen=True)
class Consolidation:
    """Stage 3's output: the list, the curve it was cut from, and what was merged away."""

    keywords: tuple[str, ...]
    curve: tuple[CurveRow, ...]
    merged: dict[str, str]           # variant -> the stem that subsumed it at no extra cost
    config: dict[str, Any]
    n_patients: int

    def as_dict(self) -> dict[str, Any]:
        return _d(self, keywords=list(self.keywords), merged=dict(sorted(self.merged.items())),
                  curve=[r.as_dict() for r in self.curve])


def _subsumes(stem: TermPrice, variant: TermPrice) -> bool:
    """Does the stem cover the variant at NO EXTRA COST?

    Substring matching makes a prefix's document set a superset of its variant's by
    construction, so `extra_documents` can only be >=; equality is exactly "no extra cost".
    Dropping "pathology", "biopsy" and "carcinoma" for "patho", "biops" and "carci" was a
    real win on 1,770 patients, rediscovered independently — this is the rule that finds it,
    and the equality is why it does not also merge away a stem that drags a pile in.
    """
    return (stem.term != variant.term and variant.term.startswith(stem.term)
            and stem.extra_documents == variant.extra_documents
            and stem.answers_rescued >= variant.answers_rescued)


def consolidate(priced: Sequence[TermPrice], bm: DocBitmaps, cfg: DerivationConfig,
                current: Sequence[str] = ()) -> Consolidation:
    """STAGE 3. Dedupe, merge variants onto stems, rank greedily, cut at the threshold."""
    if not isinstance(cfg, DerivationConfig):
        raise DerivationError("consolidate needs a DerivationConfig: the cut is a decision with "
                              "an owner, and there is no default for it")
    uniq: dict[str, TermPrice] = {}
    for p in priced:  # dedupe: same term twice is one term, keep the first pricing
        uniq.setdefault(p.term, p)
    merged: dict[str, str] = {}
    for stem in sorted(uniq.values(), key=lambda p: len(p.term)):
        if stem.term in merged:
            continue
        for var in list(uniq.values()):
            if var.term not in merged and _subsumes(stem, var):
                merged[var.term] = stem.term
    pool = [p for t, p in uniq.items() if t not in merged]

    have: frozenset[int] = frozenset()
    for t in current:
        have |= bm.matched(normalise_term(t))
    have_pat = _answered(bm, have)

    curve: list[CurveRow] = []
    keywords, cutting, cum_a, cum_d = list(dict.fromkeys(normalise_term(t) for t in current)), \
        True, 0, 0
    rank = 0
    while pool:
        best, best_key = None, None
        for p in pool:
            pats = _answered(bm, p.rows)
            gain, cost = len(pats - have_pat), len(p.rows - have)
            # No gain sorts last whatever it costs; among gainers, cheapest per answer wins.
            key = (0 if gain else 1, (cost / gain) if gain else 0.0, -gain, p.term)
            if best_key is None or key < best_key:
                best, best_key, best_gd = p, key, (gain, cost)
        pool.remove(best)
        gain, cost = best_gd
        ratio = (cost / gain) if gain else float("inf")
        rank += 1
        # The cut is a PREFIX of the curve, not a filter over it. A term admitted after a
        # rejected one would be admitted only because a cheaper term was refused first, and
        # the resulting list is not one any single threshold would have chosen.
        take = cutting and gain > 0 and ratio <= cfg.max_extra_docs_per_answer
        cutting = take
        if take:
            keywords.append(best.term)
            have, cum_a, cum_d = have | best.rows, cum_a + gain, cum_d + cost
            have_pat = _answered(bm, have)
        curve.append(CurveRow(rank, best.term, gain, cost, ratio, cum_a, cum_d, take))
    return Consolidation(tuple(keywords), tuple(curve), merged, cfg.as_dict(), bm.n_patients)


# ============================================================================
# STAGE 4 — POLICY per (doc_type, field), on two axes that never merge
# ============================================================================

@dataclass(frozen=True)
class PolicyRow:
    """One (doc_type, field) and the policy the two axes imply."""

    doc_type: str
    field: str
    admissible: bool
    admissibility_source: str    # the spec stratum that declared it — never a count
    n: int
    n_states: int
    yield_rate: float
    high_yield: bool
    n_patients_sole_source: int
    policy: str

    def as_dict(self) -> dict[str, Any]:
        return _d(self, yield_rate=round(self.yield_rate, 6))


def policy_for(admissible: bool, high_yield: bool) -> str:
    """The 2x2. Inadmissible+high is SEARCH and not SAMPLE, and that cell is the whole point:
    a type that mentions the field constantly is worth reading for corroboration and for the
    absence proof, and may still never be the witness."""
    if admissible:
        return READ_ALL if high_yield else SEARCH
    return SEARCH if high_yield else SAMPLE


def admissibility(spec: Any, doc_types: Sequence[str], fields: Sequence[str]
                  ) -> dict[tuple[str, str], tuple[bool, str]]:
    """(doc_type, field) -> (may establish, which stratum said so). CLINICAL, read from the
    spec and never from the labels.

    Routing goes through `coverage.assign_strata` rather than re-implementing the match, so
    this module and the run plane cannot drift into disagreeing about which stratum a
    document is in — which would make the derived policy a policy for a different corpus.
    """
    strata = strata_from_spec(spec)
    by_name = {s.name: s for s in strata}
    probe = [DocMeta(f"probe_{t}", t, date(2000, 1, 1), 1, 0) for t in doc_types]
    home: dict[str, str] = {}
    for name, docs in assign_strata(probe, strata).items():
        for d in docs:
            home[d.doc_type] = name
    out: dict[tuple[str, str], tuple[bool, str]] = {}
    for t in doc_types:
        name = home.get(t)
        if name is None:
            raise UndeclaredAdmissibilityError(
                f"no stratum in {getattr(spec, 'spec_id', spec)} matches document type {t!r}, and "
                f"there is no `rest` stratum to sweep it up. Whether it may establish a field is "
                f"a clinical judgement; this module will not infer one from how often it "
                f"mentions things.")
        s = by_name[name]
        for f in fields:
            # An empty `establishes` means the stratum speaks for every field — the spec
            # grammar's own default, honoured here rather than second-guessed.
            out[(t, f)] = (not s.establishes or f in s.establishes, name)
    return out


@dataclass(frozen=True)
class PolicyProposal:
    """Stage 4's output. A PROPOSAL — this object is never written into a spec."""

    spec_id: str
    rows: tuple[PolicyRow, ...]
    config: dict[str, Any]
    n_patients: int
    n_notes: int

    def as_dict(self) -> dict[str, Any]:
        return _d(self, rows=[r.as_dict() for r in self.rows])


def derive_policy(agg: Aggregate, spec: Any, cfg: DerivationConfig) -> PolicyProposal:
    """STAGE 4. Admissibility from the spec, yield from the labels, policy from the 2x2."""
    if not isinstance(cfg, DerivationConfig):
        raise DerivationError("derive_policy needs a DerivationConfig: high/low yield is a "
                              "threshold with an owner and has no default")
    types = sorted({dt for dt, _ in agg.types})
    adm = admissibility(spec, types, agg.fields)
    rows = []
    for (dt, f), ev in sorted(agg.types.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        ok, why = adm[(dt, f)]
        high = ev.yield_rate >= cfg.high_yield_rate
        rows.append(PolicyRow(dt, f, ok, why, ev.n, ev.n_states, ev.yield_rate, high,
                              ev.n_patients_sole_source, policy_for(ok, high)))
    return PolicyProposal(str(getattr(spec, "spec_id", "")), tuple(rows), cfg.as_dict(),
                          agg.n_patients, agg.n_notes)


# ============================================================================
# STAGE 5 — HOW MANY ASSETS? measured overlap, proposed grouping
# ============================================================================

@dataclass(frozen=True)
class FieldOverlap:
    """Two fields, and how far the documents that can establish them coincide.

    Jaccard and not "how much of A is in B": containment is asymmetric and would say a field
    established by four notes is served by the asset of a field established by four hundred,
    on the strength of those four. Sharing an asset is a symmetric bargain and the coefficient
    that scores it has to be symmetric too.
    """

    field_a: str
    field_b: str
    n_a: int
    n_b: int
    n_both: int

    @property
    def n_either(self) -> int:
        return self.n_a + self.n_b - self.n_both

    @property
    def jaccard(self) -> float:
        """|A n B| / |A u B|, and 0.0 when nothing establishes either field.

        No evidence is not agreement. Two fields no note establishes have identical (empty)
        sets, and a coefficient of 1.0 would propose merging their assets on the strength of
        having measured nothing about either.
        """
        return self.n_both / self.n_either if self.n_either else 0.0

    def as_dict(self) -> dict[str, Any]:
        return _d(self, n_either=self.n_either, jaccard=round(self.jaccard, 6))


def overlap_matrix(agg: Aggregate) -> tuple[FieldOverlap, ...]:
    """Every unordered pair of fields, measured. THRESHOLD-FREE on purpose.

    This is the measurement; `suggest_grouping` is the decision, and they are separate
    functions so the whole matrix can be printed beside any cut of it.
    """
    out = []
    for i, a in enumerate(agg.fields):
        for b in agg.fields[i + 1:]:
            out.append(FieldOverlap(a, b, agg.n_establishing(a), agg.n_establishing(b),
                                    agg.n_establishing(a, b)))
    return tuple(out)


@dataclass(frozen=True)
class GroupingProposal:
    """How many assets the labels say this spec needs. A PROPOSAL, like the policy and for a
    stronger reason: a policy proposal misreads one document type, and a grouping decides what
    evidence is sought for a whole field."""

    spec_id: str
    fields: tuple[str, ...]
    groups: tuple[tuple[str, ...], ...]
    overlaps: tuple[FieldOverlap, ...]
    config: dict[str, Any]
    n_notes: int
    n_patients: int

    @property
    def n_assets(self) -> int:
        return len(self.groups)

    def as_dict(self) -> dict[str, Any]:
        return _d(self, fields=list(self.fields), groups=[list(g) for g in self.groups],
                  overlaps=[o.as_dict() for o in self.overlaps], n_assets=self.n_assets,
                  STATUS="PROPOSED — NOT IN EFFECT. A grouping is a semantic claim; see "
                         "why_a_human_decides_this.",
                  why_a_human_decides_this=WHY_A_HUMAN_DECIDES_THE_GROUPING)


WHY_A_HUMAN_DECIDES_THE_GROUPING = (
    "The overlaps are measured; the grouping is not. Giving two fields one keyword list "
    "and one read policy changes what evidence is sought for each of them: the merged asset is "
    "tuned to the documents that establish both, and whatever establishes only one of them is "
    "afterwards reached, if at all, by terms chosen for the other. Two fields can coincide in "
    "this corpus and still be two questions — and the same two can diverge here only because "
    "this corpus is small. A coefficient cannot tell those apart, so it proposes and a person "
    "decides.")


def suggest_grouping(agg: Aggregate, cfg: DerivationConfig, *, spec_id: str = ""
                     ) -> GroupingProposal:
    """STAGE 5. Fields whose establishing sets coincide, grouped; cut at `share_asset_jaccard`.

    COMPLETE linkage: a group is proposed only if EVERY pair inside it clears the threshold.
    Single linkage would chain — A with B, B with C, and A with C never measured at all — and
    the resulting group would be one asset for two fields nothing said were alike. Merging is
    greedy on the strongest weakest-pair, which is deterministic given the field order, and
    ties go to the pair that comes first in that order.
    """
    if not isinstance(cfg, DerivationConfig):
        raise DerivationError("suggest_grouping needs a DerivationConfig: how much overlap makes "
                              "one asset out of two fields is a decision with an owner, and "
                              "there is no default for it")
    rows = overlap_matrix(agg)
    j = {(o.field_a, o.field_b): o.jaccard for o in rows}
    pair = lambda a, b: j.get((a, b), j.get((b, a), 0.0))
    groups = [[f] for f in agg.fields]
    while True:
        best = None
        for x in range(len(groups)):
            for y in range(x + 1, len(groups)):
                weakest = min(pair(a, b) for a in groups[x] for b in groups[y])
                if weakest >= cfg.share_asset_jaccard and (best is None or weakest > best[0]):
                    best = (weakest, x, y)
        if best is None:
            break
        _, x, y = best
        groups[x] = groups[x] + groups.pop(y)
    return GroupingProposal(spec_id, agg.fields, tuple(tuple(g) for g in groups), rows,
                            cfg.as_dict(), agg.n_notes, agg.n_patients)


# ============================================================================
# EMISSION — one writer per asset kind, and the kinds are not symmetric
# ============================================================================

#: Parameter names that would let a caller write a semantic asset anyway. Checked at import
#: over this module's public callables, by a check that runs at import rather than by review:
#: a rule enforced by a review is a rule that survives until the reviewer is busy.
_OVERRIDE_NAMES = frozenset({
    "force", "override", "adopt_strata", "write_strata", "write_policy", "apply_policy",
    "allow_semantic", "no_review", "skip_review", "auto_adopt", "yes",
    # Stage 5's asset kind, guarded before anyone offers a flag for it: which fields share an
    # asset decides what evidence is sought for each of them, and no count may install that.
    "write_grouping", "adopt_grouping", "apply_grouping", "write_groups", "merge_fields"})


def assert_no_semantic_override(module: Any) -> None:
    """Raise if any public callable here offers to bypass the clinician."""
    for name, obj in vars(module).items():
        if name.startswith("_") or not callable(obj) or getattr(obj, "__module__", "") != __name__:
            continue
        try:
            params = set(inspect.signature(obj).parameters)
        except (TypeError, ValueError):  # pragma: no cover - builtins have no signature
            continue
        bad = sorted(params & _OVERRIDE_NAMES)
        if bad:
            raise SemanticOverrideError(
                f"{name}{tuple(bad)} would let a caller adopt a semantic asset without a "
                f"clinician. Admissibility is a clinical judgement; there is no flag for it.")


_KW_ELEMENT_RE = re.compile(r"^proof_obligation\.for_negative(?:\.claims\[[^\]]+\])?"
                            r"\.strata\[(?P<stratum>[^\]]+)\]\.required_keywords$")


def write_keywords(spec_path: str | Path, element: str, con: Consolidation, *, run: str,
                   today: str | None = None) -> dict[str, Any]:
    """Write a derived keyword list into a spec, with its provenance, or write nothing.

    Legal because keywords are RETRIEVAL-ONLY: they change which text reaches the agent and
    nothing about what an answer means. Value and record go in one atomic replace and the
    result is reloaded and hash-checked first, because a spec carrying a new list under the
    old list's provenance record loads perfectly and reads as measured.
    """
    m = _KW_ELEMENT_RE.match(element)
    if not m:
        raise AdoptionAborted(f"{element!r} is not a required_keywords element")
    p = Path(spec_path)
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    value = list(con.keywords)
    holder = (doc.get("proof_obligation") or {}).get("for_negative") or {}
    hit = [s for s in (holder.get("strata") or []) if str(s.get("name")) == m.group("stratum")]
    if not hit:
        raise AdoptionAborted(f"{p}: no stratum {m.group('stratum')!r}; adopting into one that "
                              f"does not exist creates a list nothing reads")
    hit[0]["required_keywords"] = value
    rescued = con.curve[-1].cum_answers_rescued if con.curve else 0
    # A derivation that rescued nothing is `underpowered`, not support. `spec._validate_record`
    # keeps such an element at draft on purpose: a measurement is a reason to distrust an
    # element, and must never rank it above one nobody has looked at.
    verdict = "supports" if rescued > 0 else "underpowered"
    rec = ProvenanceRecord(
        element=element, origin="corpus_derived",
        basis=f"acr.improvement.derive stage 1-3 over run {run}: {len(value)} terms cut at "
              f"{con.config['max_extra_docs_per_answer']} extra documents per answer rescued",
        status="measured" if verdict == "supports" else "draft",
        measured={"run": run, "n_patients": con.n_patients, "verdict": verdict,
                  "answers_rescued": rescued, "config": con.config})
    block = doc.setdefault("provenance", [])
    blob = rec.model_dump(exclude_none=True, exclude={"element_hash", "element_kind",
                                                      "sign_off_voided_by_edit"})
    for i, ex in enumerate(block):
        if isinstance(ex, dict) and ex.get("element") == element:
            block[i] = blob
            break
    else:
        block.append(blob)
    tmp = p.with_name(f".{p.name}.derive-{os.getpid()}.tmp")
    try:
        tmp.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=96),
                       encoding="utf-8")
        reloaded = load_spec(tmp)
        got = reloaded.provenance_index.get(element)
        if got is None or got.element_hash != _content_hash(value):
            raise AdoptionAborted(f"{element}: the record on disk and the value on disk describe "
                                  f"different things; nothing was written to {p}")
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"outcome": "adopted", "element": element, "spec": str(p), "n_keywords": len(value),
            "adopted_on": today or date.today().isoformat()}


def emit_policy_proposal(proposal: PolicyProposal, out_dir: str | Path,
                         *, today: str | None = None) -> Path:
    """Write the policy where a clinician signs it, and NOWHERE ELSE.

    This function does not take a spec path and cannot be given one. Yield is measurable;
    admissibility is not, and a per-type read policy derived from a count and installed
    without a signature is the `cannot_establish` bug being made again by a tool instead of
    by a model.
    """
    stamp = today or date.today().isoformat()
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stamp}_{proposal.spec_id or 'spec'}_policy.yaml"
    body = yaml.safe_dump({
        "kind": "acr.improvement.derive.policy_proposal/1",
        "STATUS": "PROPOSED — NOT IN EFFECT. Requires a clinician signature.",
        "why_a_signature_is_required":
            "YIELD below is measured. ADMISSIBILITY is not measurable at all: it is a clinical "
            "judgement about what a document type may establish, and every row here read it out "
            "of the spec rather than deriving it. Changing a policy changes what may serve as a "
            "witness, so no count may install one.",
        "the_question_for_the_clinician":
            "For each row: the spec says this type may/may not establish this field, and the "
            "labelling says how often it carries it. Is the resulting read policy right?",
        **proposal.as_dict(),
        "signature": {"reviewed_by": None, "reviewed_on": None, "decision": None,
                      "note": "accept | reject | accept_with_changes"},
    }, sort_keys=False, allow_unicode=True, width=96)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return path


def emit_grouping_proposal(proposal: GroupingProposal, out_dir: str | Path,
                           *, today: str | None = None) -> Path:
    """Write the grouping where a person decides it, and NOWHERE ELSE.

    Same shape as `emit_policy_proposal` and for the same reason: this function takes no spec
    path and cannot be given one. The FULL matrix goes into the file, not only the pairs above
    the cut, so the reader can see what a different threshold would have merged without
    rerunning anything.
    """
    stamp = today or date.today().isoformat()
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{stamp}_{proposal.spec_id or 'spec'}_field_groups.yaml"
    rest = proposal.as_dict()   # STATUS and the reasoning go first, where they are read
    body = yaml.safe_dump({
        "kind": "acr.improvement.derive.grouping_proposal/1",
        "STATUS": rest.pop("STATUS"),
        "why_a_human_decides_this": rest.pop("why_a_human_decides_this"),
        "the_question_for_the_reviewer":
            f"These {len(proposal.fields)} fields are proposed as {proposal.n_assets} asset(s), "
            f"cut at Jaccard >= {proposal.config['share_asset_jaccard']}. For each proposed "
            f"group: is one keyword list and one read policy right for all of it? For each pair "
            f"left apart: would you have merged it anyway?",
        **rest,
        "signature": {"reviewed_by": None, "reviewed_on": None, "decision": None,
                      "note": "accept | reject | accept_with_changes"},
    }, sort_keys=False, allow_unicode=True, width=96)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(body, encoding="utf-8")
    os.replace(tmp, path)
    return path


# ============================================================================
# CLI — `acr derive`. Nothing writes without an explicit flag.
# ============================================================================

derive_app = typer.Typer(add_completion=False, help=(
    "Derive the retrieval assets from a completed labelling: aggregate the labels, price the "
    "candidate terms against the cached bitmaps, consolidate a keyword list, propose a read "
    "policy per (doc_type, field), and propose which fields can share one asset. No model calls "
    "and no chart reads. Every threshold is a required option — there are no defaults."))

_LABELS = typer.Option(..., "--labels", "-l", help="labels.jsonl from `labelling.LabelStore`")
_SPEC = typer.Option(..., "--spec", "-s", help="the spec whose strata declare admissibility")
_CACHE = typer.Option(DEFAULT_CACHE_ROOT, "--cache", help="cached document bitmaps")
_FIELDS = typer.Option(..., "--fields", "-f", help="comma-separated output fields to count")
_CUT = typer.Option(..., "--max-extra-docs-per-answer",
                    help="STAGE 3 CUT, required: documents per answer rescued")
_YIELD = typer.Option(..., "--high-yield-rate",
                      help="STAGE 4 SPLIT, required: rate at/above which a type is high-yield")
_MINPAT = typer.Option(..., "--min-patients-proposing",
                       help="STAGE 1 FLOOR, required: distinct patients that must propose a term")
_SHARE = typer.Option(..., "--share-asset-jaccard",
                      help="STAGE 5 CUT, required: overlap at/above which two fields are "
                           "PROPOSED to share one keyword list and one policy")


def _load_labels(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a torn final line is the normal shape of a killed scan
    return out


def _stage123(labels: str, spec: str, cache: str, fields: str, cfg: DerivationConfig):
    s = load_spec(spec)
    flds = [f.strip() for f in fields.split(",") if f.strip()]
    agg = aggregate(_load_labels(labels), flds)
    bm = load_bitmaps(cache)
    current = [k for st in strata_from_spec(s) for k in st.required_keywords]
    priced = price_terms([t.term for t in agg.ranked_terms(cfg)], bm, current)
    return s, agg, bm, consolidate(priced, bm, cfg, current)


@derive_app.command("terms")
def cmd_terms(labels: str = _LABELS, spec: str = _SPEC, cache: str = _CACHE, fields: str = _FIELDS,
              max_extra_docs_per_answer: float = _CUT, high_yield_rate: float = _YIELD,
              min_patients_proposing: int = _MINPAT, share_asset_jaccard: float = _SHARE,
              element: str = typer.Option(None, "--element",
                                          help="required_keywords element to write"),
              write: bool = typer.Option(False, "--write-spec",
                                         help="WRITE the derived list into the spec"),
              run: str = typer.Option("", "--run", help="run id recorded in the provenance")):
    """Stages 1-3: aggregate, price, consolidate. Prints; writes only with --write-spec."""
    cfg = DerivationConfig(max_extra_docs_per_answer, high_yield_rate, min_patients_proposing,
                           share_asset_jaccard)
    _, agg, _, con = _stage123(labels, spec, cache, fields, cfg)
    out: dict[str, Any] = {"aggregate": agg.as_dict(), "consolidation": con.as_dict()}
    if write:
        if not element:
            raise typer.BadParameter("--write-spec needs --element")
        out["write"] = write_keywords(spec, element, con, run=run)
    typer.echo(json.dumps(out, indent=1, sort_keys=False))


@derive_app.command("policy")
def cmd_policy(labels: str = _LABELS, spec: str = _SPEC, fields: str = _FIELDS,
               max_extra_docs_per_answer: float = _CUT, high_yield_rate: float = _YIELD,
               min_patients_proposing: int = _MINPAT, share_asset_jaccard: float = _SHARE,
               emit: bool = typer.Option(False, "--emit-proposal",
                                         help="write the proposal for a clinician to sign"),
               out_dir: str = typer.Option(None, "--out-dir", help="where the proposal goes")):
    """Stage 4. The spec is NEVER edited by this command — --emit-proposal writes a file a
    clinician signs, and there is no flag that installs it."""
    cfg = DerivationConfig(max_extra_docs_per_answer, high_yield_rate, min_patients_proposing,
                           share_asset_jaccard)
    s = load_spec(spec)
    flds = [f.strip() for f in fields.split(",") if f.strip()]
    prop = derive_policy(aggregate(_load_labels(labels), flds), s, cfg)
    body: dict[str, Any] = prop.as_dict()
    if emit:
        body["proposal_path"] = str(emit_policy_proposal(
            prop, out_dir or (Path(spec).parent / "proposals")))
    typer.echo(json.dumps(body, indent=1, sort_keys=False))


@derive_app.command("groups")
def cmd_groups(labels: str = _LABELS, spec: str = _SPEC, fields: str = _FIELDS,
               max_extra_docs_per_answer: float = _CUT, high_yield_rate: float = _YIELD,
               min_patients_proposing: int = _MINPAT, share_asset_jaccard: float = _SHARE,
               emit: bool = typer.Option(False, "--emit-proposal",
                                         help="write the proposal for a person to decide"),
               out_dir: str = typer.Option(None, "--out-dir", help="where the proposal goes")):
    """Stage 5. Which fields can share one keyword list and one policy?

    Prints the WHOLE overlap matrix beside the suggestion, so the person choosing the cut sees
    which pairs each setting would merge. Writes nothing without --emit-proposal, and what it
    writes then is a proposal: no flag here installs a grouping."""
    cfg = DerivationConfig(max_extra_docs_per_answer, high_yield_rate, min_patients_proposing,
                           share_asset_jaccard)
    s = load_spec(spec)
    flds = [f.strip() for f in fields.split(",") if f.strip()]
    prop = suggest_grouping(aggregate(_load_labels(labels), flds),
                            cfg, spec_id=str(getattr(s, "spec_id", "")))
    typer.echo(f"{'field A':<24} {'field B':<24} {'|A|':>7} {'|B|':>7} {'|A^B|':>7} "
               f"{'jaccard':>8}  share?")
    for o in sorted(prop.overlaps, key=lambda o: -o.jaccard):
        mark = "MERGE" if o.jaccard >= cfg.share_asset_jaccard else ""
        typer.echo(f"{o.field_a[:24]:<24} {o.field_b[:24]:<24} {o.n_a:>7} {o.n_b:>7} "
                   f"{o.n_both:>7} {o.jaccard:>8.3f}  {mark}")
    typer.echo(f"\ncut at jaccard >= {cfg.share_asset_jaccard} -> {prop.n_assets} asset(s) for "
               f"{len(prop.fields)} field(s):")
    for g in prop.groups:
        typer.echo(f"  - {', '.join(g)}")
    typer.echo("\nPROPOSED, not in effect. " + WHY_A_HUMAN_DECIDES_THE_GROUPING)
    if emit:
        typer.echo(str(emit_grouping_proposal(prop, out_dir or (Path(spec).parent / "proposals"))))


@derive_app.command("show-curve")
def cmd_show_curve(labels: str = _LABELS, spec: str = _SPEC, cache: str = _CACHE,
                   fields: str = _FIELDS, max_extra_docs_per_answer: float = _CUT,
                   high_yield_rate: float = _YIELD, min_patients_proposing: int = _MINPAT,
                   share_asset_jaccard: float = _SHARE):
    """The whole ranked curve beside the cut, so the person choosing the threshold sees what
    every other setting would have bought. Writes nothing."""
    cfg = DerivationConfig(max_extra_docs_per_answer, high_yield_rate, min_patients_proposing,
                           share_asset_jaccard)
    _, _, _, con = _stage123(labels, spec, cache, fields, cfg)
    typer.echo(f"{'cut':>4} {'#':>4} {'term':<28} {'+ans':>6} {'+docs':>8} {'docs/ans':>9} "
               f"{'cum ans':>8} {'cum docs':>9}")
    for r in con.curve:
        typer.echo(f"{'KEEP' if r.in_cut else '':>4} {r.rank:>4} {r.term[:28]:<28} "
                   f"{r.marginal_answers_rescued:>6} {r.marginal_extra_documents:>8} "
                   f"{r.docs_per_answer:>9.2f} {r.cum_answers_rescued:>8} "
                   f"{r.cum_extra_documents:>9}")
    typer.echo(f"\ncut at {cfg.max_extra_docs_per_answer} docs/answer -> "
               f"{len(con.keywords)} keywords; merged onto stems: {len(con.merged)}")


# Run the guard over this module's own surface, at import, after everything is defined. A
# rule that lives only in the docstring above is a rule that survives until someone needs the
# flag; an ImportError is a rule that survives the person who needs it.
assert_no_semantic_override(sys.modules[__name__])
