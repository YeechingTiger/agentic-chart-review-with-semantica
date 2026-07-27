"""THE DEVELOPMENT PLANE — where a spec element earns its place before the spec asserts it.

Every other module here belongs to the RUN plane: a frozen spec meets a chart, an agent
works, an answer comes out gated. Nothing in that plane can tell you whether the spec was any
good, and one element of it measurably is not. `STORE.400_522_523` declares five
`required_keywords` for the `may_mention` stratum and `max_tolerated_hits: 0` — ONE
answer-bearing document the keyword search failed to surface falsifies the list. Measured
exhaustively over all 1,788 real charts on 2026-07-26: the list misses the stated diagnosis
in 4,005 progress notes, discharge summaries and consults, on 31.7% of patients, mostly
because clinicians write "small cell lung cancer" and the list only says "carcinoma". The
gate check was never decorative. It had simply never been run, because there was nowhere to
run it — authoring a keyword list was a typing exercise, and scoring one appeared to require
an agent, a model and a bill.

It does not. A keyword list is scored by grep against a labelled document set: cents and
seconds, and not one model call anywhere in this module. That is what makes a search loop
affordable, and a search loop is what turns "these five words look sensible" to a registrar
into a number a reviewer can check.

THE SPLIT IS AT PATIENT LEVEL, NEVER DOCUMENT LEVEL
---------------------------------------------------
A term list tuned on some documents of patient X and scored on other documents of patient X
is scored on data it has seen. Charts repeat themselves: the same problem list, the same
"history of small cell lung cancer" sentence, pasted forward across dozens of notes for
months. A document-level split leaks the answer for that patient into both halves and reports
a recall that no new patient will ever reproduce. `split_patients` therefore takes patient
ids and there is no document-level equivalent in this file.

CERTIFY REFUSES; IT DOES NOT WARN
---------------------------------
`evolve` hill-climbs against the oracle. Doing that on the develop split is method
development. Doing it on the split you then quote is fraud with extra steps, and the
difference is invisible in the output number — both produce a plausible recall with a
plausible confidence interval. So the difference is enforced where it can be checked
mechanically: `certify` accepts only a `PatientGroup` minted by `split_patients`, refuses the
develop half by ROLE, refuses any overlap between the halves, refuses to certify a number
with no declared downstream use, and refuses when the oracle's labels came from the very
ground-truth column the number will be quoted against. Each is a raised exception carrying
the sentence a reviewer needs, not a warning in a log nobody reads.

WHAT THE ORACLE IS, AND WHY ITS OPTIMISM IS REPORTED WITH IT
------------------------------------------------------------
A document is oracle-positive for a patient when it states that patient's registry-coded
diagnosis in prose. The surface form is derived MECHANICALLY from the ground-truth
description string (see `diagnosis_pattern`) — no code table, no clinical judgement. That is
deliberate and it is the same rule the answer-checks follow: the clinical content lives in
the spec where a registrar can review it, and the checker contains no oncology. The price is
that this oracle cannot know "SCLC" is small cell carcinoma, so it under-counts
answer-bearing documents and every recall it produces is an UPPER BOUND. The bound is not
left as a caveat in prose: `Oracle.provenance` carries the measured optimism next to the
number, and `certify` copies it into the certificate.

NOTHING HERE WRITES A SPEC
--------------------------
`evolve` prints a diff and returns a provenance block. Freezing a measured list into a spec
is a separate, deliberate act with its own owner. A development plane that edits the artefact
it is developing has quietly become an unreviewed author.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import typer
from rich.console import Console
from rich.table import Table

from .corpus import DocMeta, parse_filename
from .coverage import StratumSpec, assign_strata, strata_from_spec
from .spec import ExtractionSpec, load_spec

DEVELOP = "develop"
HOLDOUT = "holdout"


# ---------------------------------------------------------------------------- refusals
class TermDevelopmentError(Exception):
    """Base for everything this module refuses to do."""


class UnindexedTermError(TermDevelopmentError):
    """A term was scored that the index never looked for.

    Returning zero would be the wrong answer to the wrong question: the term may be all over
    the corpus. An index that answers "no documents" for a needle it never searched is a
    silent zero, and a silent zero in a recall denominator is how a list gets certified for
    coverage it does not have.
    """


class EmptyOracleError(TermDevelopmentError):
    """No oracle-positive document in scope, so recall is 0/0 and every list ties."""


class UnknownPatientError(TermDevelopmentError):
    """Scored on a patient the index does not hold. Silently narrowing the scope would
    report a recall over a cohort the caller did not ask for."""


class HoldoutViolation(TermDevelopmentError):
    """Base for the four ways a certification is not a certification."""


class UnprovenancedHoldoutError(HoldoutViolation):
    pass


class DevelopSplitCertificationError(HoldoutViolation):
    pass


class SplitLeakError(HoldoutViolation):
    pass


class UndeclaredDownstreamUseError(HoldoutViolation):
    pass


class OracleLeakError(HoldoutViolation):
    pass


# ---------------------------------------------------------------------------- splitting
def _fingerprint(ids: Iterable[str]) -> str:
    """A comparable identity for a patient set that is not a list of patient ids.

    Certificates get pasted into write-ups and manifests. `person_id`s already leaked into
    `skills/` and `src/` once (see tests/test_no_phi_in_tree.py); a split identity that is a
    hash is one that a reviewer can compare and an audit log cannot disclose.
    """
    h = hashlib.sha256()
    for pid in sorted(set(ids)):
        h.update(pid.encode())
        h.update(b"\0")
    return h.hexdigest()[:16]


@dataclass(frozen=True)
class PatientGroup:
    """One half of a split, carrying the fact that it IS one half.

    The role and the counterpart travel with the ids on purpose. A bare `list[str]` cannot
    answer "were these held out, or are these the patients the search just optimised
    against?", and that is precisely the question `certify` exists to ask.
    """

    role: str
    ids: tuple[str, ...]
    counterpart: frozenset[str]
    seed: int
    holdout_frac: float

    def __iter__(self) -> Iterator[str]:
        return iter(self.ids)

    def __len__(self) -> int:
        return len(self.ids)

    def __contains__(self, pid: object) -> bool:
        return pid in set(self.ids)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.ids)

    def describe(self) -> dict:
        return {"role": self.role, "n_patients": len(self.ids),
                "fingerprint": self.fingerprint, "seed": self.seed,
                "holdout_frac": self.holdout_frac}


@dataclass(frozen=True)
class PatientSplit:
    develop: PatientGroup
    holdout: PatientGroup

    def __iter__(self) -> Iterator[PatientGroup]:
        # so `develop, holdout = split_patients(...)` reads the way it is written
        return iter((self.develop, self.holdout))

    def describe(self) -> dict:
        return {"develop": self.develop.describe(), "holdout": self.holdout.describe()}


def split_patients(ids: Sequence[str], holdout_frac: float = 0.3, seed: int = 0) -> PatientSplit:
    """Patient-level split. Deterministic in (ids, holdout_frac, seed) and nothing else.

    Sorting before shuffling matters: `Corpus.patient_ids()` is sorted but a cohort file, a
    set, or a dict keys view is not, and a split that depends on the caller's iteration order
    is a split that silently changes between two runs that look identical in the manifest.
    """
    import random

    if not 0.0 < holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be strictly between 0 and 1, got {holdout_frac}")
    uniq = sorted(set(ids))
    if len(uniq) < 2:
        raise ValueError(f"cannot split {len(uniq)} patient(s) into two non-empty halves")
    shuffled = list(uniq)
    random.Random(seed).shuffle(shuffled)
    n_hold = max(1, min(len(uniq) - 1, round(holdout_frac * len(uniq))))
    hold = tuple(sorted(shuffled[:n_hold]))
    dev = tuple(sorted(shuffled[n_hold:]))
    return PatientSplit(
        develop=PatientGroup(DEVELOP, dev, frozenset(hold), seed, holdout_frac),
        holdout=PatientGroup(HOLDOUT, hold, frozenset(dev), seed, holdout_frac),
    )


# ---------------------------------------------------------------------------- the oracle
#: Registry grade/qualifier tokens, deleted before a description becomes a search pattern.
#: "POORLY DIFF ADENOCARCINOMA" and "ADENOCARCINOMA, UNK GRADE" are the same diagnosis and no
#: note writes either string; what a note writes is "adenocarcinoma".
GRADE_TOKENS = frozenset({
    "md", "wd", "pd", "poorly", "mod", "moderately", "well", "diff", "differentiated",
    "unk", "grade", "nos", "unknown", "high", "low", "intermediate", "g1", "g2", "g3", "g4",
})

_ICDO_PREFIX = re.compile(r"^\s*\d{4}\s*/\s*\d\s*")

#: Shorter than this, a description is not a diagnosis locator. The corpus has exactly one
#: patient whose description reduces to "ca", and `\bca` matches "cardiac", "calcium" and
#: "carotid" — that patient's entire chart would be oracle-positive and every term list would
#: score a free 100% on them. Patients below the floor are dropped and counted in provenance.
MIN_PATTERN_CHARS = 3


def diagnosis_pattern(description: str) -> tuple[re.Pattern[str] | None, str]:
    """Ground-truth description -> (compiled pattern, canonical form). No clinical judgement.

    Ported from the Phase-2 measurement so the plane scores against the same oracle the
    falsification did, with two mechanical repairs found while porting:

      * separators are `[\\s\\-]*`, not `\\s*`. The derivation strips hyphens out of the
        REGISTRY string ("NON-SMALL CELL CARCINOMA" -> "non small cell carcinoma") and then
        demanded whitespace in the NOTE, so the pattern failed to match the note phrase
        "non-small cell carcinoma" — the exact string it was derived from. 22 of 1,960
        oracle-positive documents in a 150-patient probe were recovered by the fix.
      * the pattern is anchored with `\\b` at the start. Without it "scc" matches inside a
        longer word and the oracle marks a document positive for a diagnosis nobody wrote.

    The END is deliberately left open — no trailing `\\b` — because that is what makes the
    registry abbreviations work: "small cell ca" has to match "small cell carcinoma" and
    "small cell cancer", and both are prefix extensions. The cost is that a description that
    is itself a prefix of an unrelated word ("ADENO" inside "adenopathy") over-fires. That is
    not argued away: `Oracle.provenance["prefix_open_positives"]` counts the documents that
    depend on the open end, so the size of the leak is visible next to the recall it inflates.
    """
    s = (description or "").strip().lower()
    if not s:
        return None, ""
    s = _ICDO_PREFIX.sub("", s)
    s = s.split(",")[0]                      # "adenocarcinoma, poorly diff" -> the diagnosis
    s = re.sub(r"[^a-z0-9\s\-]", " ", s).replace("-", " ")
    toks = [t for t in s.split() if t and t not in GRADE_TOKENS]
    canon = " ".join(toks)
    if len(canon) < MIN_PATTERN_CHARS:
        return None, canon
    pat = r"[\s\-]*".join(re.escape(t) for t in toks)
    return re.compile(r"\b" + pat), canon


#: Token = an alphabetic run of 4+ characters; stem = its first 8 characters. Ported from the
#: Phase-2 candidate harvest unchanged, because the ranking in `propose` was calibrated on it.
TOKEN_RE = re.compile(r"[a-z]{4,}")
STEM_LEN = 8
MIN_STEM_LEN = 5


def harvest_features(text_lower: str) -> set[str]:
    """The candidate pool for one document: token stems and within-line adjacent pairs.

    Two properties, both load-bearing.

    Pairs are formed WITHIN A LINE. A pair spanning a newline joins the last word of one
    field to the first word of the next ("hypertension patient"), which is an artefact of the
    note template and not a phrase anyone searches for.

    A pair keeps its FIRST token in full and stems only the second. Phase 2 stemmed both, and
    the candidate table it produced contains "carcinoi tumor" — a string that occurs nowhere
    in any chart. The harvest space is token-prefix space; the runtime
    (`corpus.PatientChart.search`) is raw-substring space, and a proposal that cannot be
    typed into `required_keywords` and retrieve anything is not a proposal. Keeping the first
    token whole makes every harvested feature a literal substring by construction, and
    `propose` then RE-SCORES it as one. The two spaces really do differ: "histopathology"
    contains the substring "patholog" and its token stem is "histopat", so the harvest cannot
    see a hit the tool would return.
    """
    feats: set[str] = set()
    for line in text_lower.split("\n"):
        toks = TOKEN_RE.findall(line)
        if not toks:
            continue
        stems = [t[:STEM_LEN] for t in toks]
        feats.update(stems)
        for first, second in zip(toks, stems[1:]):
            feats.add(f"{first} {second}")
    return feats


def stem_variants(term: str, min_len: int = MIN_STEM_LEN) -> list[str]:
    """Prefix truncations of a term's last word, longest first.

    The operation exists because it wins on real data and the spec does not use it: measured
    over 1,788 charts, "patholog" is a strict superset of "pathology" — 37,721 documents
    against 28,024, more on 1,531 patients and fewer on none — while `required_keywords` says
    "pathology". A keyword list that cannot be stemmed can only be fixed by adding words.
    """
    head, _, last = term.rpartition(" ")
    out = []
    for n in range(len(last) - 1, min_len - 1, -1):
        stem = last[:n]
        out.append(f"{head} {stem}" if head else stem)
    return out


def term_mask(text_lower: str, vocabulary: Sequence[str], triggers: dict[str, list[int]] | None = None) -> int:
    """Bit i set iff `vocabulary[i]` occurs in the text, with EXACTLY the runtime's semantics.

    `corpus.PatientChart.search` compiles `re.escape(query)` with `IGNORECASE`, which on
    lowercased text is a plain substring test. Scoring a candidate any other way — token
    match, stemmed match, word-boundary match — measures a tool the agent does not have.
    """
    m = 0
    if triggers is None:
        for i, needle in enumerate(vocabulary):
            if needle in text_lower:
                m |= 1 << i
        return m
    for trigger, idxs in triggers.items():
        if trigger not in text_lower:
            continue
        for i in idxs:
            if vocabulary[i] in text_lower:
                m |= 1 << i
    return m


def build_triggers(vocabulary: Sequence[str]) -> dict[str, list[int]]:
    """Group needles by their first word so a document skips the whole group in one test."""
    out: dict[str, list[int]] = {}
    for i, needle in enumerate(vocabulary):
        out.setdefault(needle.split(" ")[0], []).append(i)
    return out


@dataclass(frozen=True)
class LabelledDocument:
    """One document with its oracle label. `text` must already be lowercased."""

    patient_id: str
    note_id: str
    doc_type: str
    text: str
    positive: bool


def _bits(x: int) -> Iterator[int]:
    while x:
        b = x & -x
        yield b.bit_length() - 1
        x ^= b


def _pack(indices: Iterable[int], n_docs: int) -> int:
    """Bitset from document indices. Built through a bytearray because `mask |= 1 << i` on a
    90,000-bit integer copies eleven kilobytes per document and the assembly is quadratic."""
    buf = bytearray((n_docs + 7) // 8)
    for i in indices:
        buf[i >> 3] |= 1 << (i & 7)
    return int.from_bytes(bytes(buf), "little")


class Oracle:
    """A labelled document set with a term index over it. The corpus and the labels are one
    object because a term's recall is not defined without both.

    Everything downstream is arithmetic on big-integer bitsets — one bit per document, one
    integer per term. `evaluate` on 90,000 documents is a handful of ANDs, which is what lets
    `evolve` run several hundred candidate lists in the time a single model call would take.
    """

    def __init__(self, *, vocabulary: Sequence[str], postings: Sequence[int],
                 note_ids: Sequence[str], doc_types: Sequence[str],
                 patient_of: Sequence[int], patients: Sequence[str],
                 positive: int, provenance: dict[str, Any]):
        self.vocabulary = tuple(vocabulary)
        self._index = {t: i for i, t in enumerate(vocabulary)}
        self._postings = tuple(postings)
        self.note_ids = tuple(note_ids)
        self.doc_types = tuple(doc_types)
        self.patient_of = tuple(patient_of)
        self.patients = tuple(patients)
        self.positive = positive
        self.provenance = dict(provenance)
        self.n_docs = len(note_ids)
        self.all_docs = (1 << self.n_docs) - 1
        self._patient_index = {p: i for i, p in enumerate(patients)}
        by_patient: dict[int, list[int]] = {}
        for doc_i, pat_i in enumerate(self.patient_of):
            by_patient.setdefault(pat_i, []).append(doc_i)
        self.patient_masks = tuple(
            _pack(by_patient.get(i, []), self.n_docs) for i in range(len(patients)))

    # -- identity ------------------------------------------------------------------
    @property
    def ground_truth_column(self) -> str:
        return str(self.provenance.get("ground_truth_column", ""))

    @property
    def encodes_fields(self) -> frozenset[str]:
        return frozenset(self.provenance.get("encodes_fields", ()))

    def leaks_into(self, downstream: str) -> bool:
        """Would a number certified against these labels be quoted against its own answer key?

        `gt_histology_desc` and the spec field `histology` are the same fact in two
        encodings. Tuning retrieval until it finds the histology, then reporting histology
        accuracy on the same patients, is a closed loop that reads as a result.
        """
        want = (downstream or "").strip().lower()
        if not want:
            return False
        want = want[3:] if want.startswith("gt_") else want
        col = self.ground_truth_column.lower()
        col = col[3:] if col.startswith("gt_") else col
        return want == col or want in {f.lower() for f in self.encodes_fields}

    # -- lookups -------------------------------------------------------------------
    def __contains__(self, term: object) -> bool:
        return isinstance(term, str) and term.strip().lower() in self._index

    def mask_for_term(self, term: str) -> int:
        key = term.strip().lower()
        try:
            return self._postings[self._index[key]]
        except KeyError:
            raise UnindexedTermError(
                f"'{term}' was never indexed, so its document set is unknown, not empty. "
                f"Rebuild the oracle with it in `seed_terms` "
                f"(the index holds {len(self.vocabulary)} terms)."
            ) from None

    def mask_for_terms(self, terms: Iterable[str]) -> int:
        m = 0
        for t in terms:
            m |= self.mask_for_term(t)
        return m

    def postings_items(self) -> Iterator[tuple[str, int]]:
        """(term, document bitset) for every indexed term, in vocabulary order."""
        return zip(self.vocabulary, self._postings)

    def scope_mask(self, patients: Iterable[str] | PatientGroup) -> int:
        if isinstance(patients, str):
            raise UnknownPatientError(
                "scope is a collection of patient ids; a bare string iterates as characters")
        ids = list(patients)
        unknown = [p for p in ids if p not in self._patient_index]
        if unknown:
            raise UnknownPatientError(
                f"{len(unknown)} patient(s) are not in this oracle (of {len(ids)} asked for). "
                "Scoring the rest would report a recall over a cohort you did not request.")
        m = 0
        for p in ids:
            m |= self.patient_masks[self._patient_index[p]]
        return m

    def patient_ids(self) -> list[str]:
        return list(self.patients)

    def describe(self) -> dict:
        return {"n_patients": len(self.patients), "n_docs": self.n_docs,
                "n_oracle_positive": self.positive.bit_count(),
                "vocabulary": len(self.vocabulary), **self.provenance}


# ---------------------------------------------------------------------------- evaluation
@dataclass(frozen=True)
class TermListResult:
    terms: tuple[str, ...]
    n_patients: int
    n_docs: int
    n_oracle_positive: int
    recall: float
    docs_surfaced: int
    misses: int
    patients_with_miss: int
    cost_ratio: float
    patients_with_positive: int
    docs_per_positive_found: float

    @property
    def patient_recall(self) -> float:
        if not self.patients_with_positive:
            return 1.0
        return 1.0 - self.patients_with_miss / self.patients_with_positive

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in (
            "n_patients", "n_docs", "n_oracle_positive", "recall", "docs_surfaced", "misses",
            "patients_with_miss", "cost_ratio", "patients_with_positive",
            "docs_per_positive_found")}
        d["terms"] = list(self.terms)
        d["patient_recall"] = round(self.patient_recall, 4)
        return d


def evaluate(terms: Sequence[str], patients: Iterable[str] | PatientGroup,
             oracle: Oracle) -> TermListResult:
    """Score a term list on a patient set. Pure: no state, no I/O, no model.

    `recall` is over oracle-positive DOCUMENTS, and `patients_with_miss` is the count the
    spec's gate actually turns on — `max_tolerated_hits: 0` fires per patient, so a list at
    98% document recall that leaves one uncovered document on a third of patients fails the
    gate on a third of patients. Both numbers are returned because neither is the other.
    """
    ids = list(patients)
    scope = oracle.scope_mask(ids)
    n_docs = scope.bit_count()
    pos = oracle.positive & scope
    n_pos = pos.bit_count()
    if n_pos == 0:
        raise EmptyOracleError(
            f"no oracle-positive document among the {n_docs} in scope: recall would be 0/0 "
            "and every term list would tie. Widen the patient set or check the oracle column.")
    surfaced = oracle.mask_for_terms(terms) & scope
    found = surfaced & pos
    missed = pos & ~surfaced
    hurt = {oracle.patient_of[i] for i in _bits(missed)}
    with_pos = {oracle.patient_of[i] for i in _bits(pos)}
    n_surf = surfaced.bit_count()
    n_found = found.bit_count()
    return TermListResult(
        terms=tuple(terms),
        n_patients=len(ids),
        n_docs=n_docs,
        n_oracle_positive=n_pos,
        recall=round(n_found / n_pos, 6),
        docs_surfaced=n_surf,
        misses=missed.bit_count(),
        patients_with_miss=len(hurt),
        cost_ratio=round(n_surf / n_docs, 6) if n_docs else 0.0,
        patients_with_positive=len(with_pos),
        docs_per_positive_found=round(n_surf / n_found, 3) if n_found else float("inf"),
    )


@dataclass(frozen=True)
class TermContribution:
    term: str
    docs_retrieved: int
    positives_retrieved: int
    sole_retriever_of_positives: int

    def to_dict(self) -> dict:
        return {"term": self.term, "docs_retrieved": self.docs_retrieved,
                "positives_retrieved": self.positives_retrieved,
                "sole_retriever_of_positives": self.sole_retriever_of_positives}


def term_contributions(terms: Sequence[str], patients: Iterable[str] | PatientGroup,
                       oracle: Oracle) -> list[TermContribution]:
    """Per-term pull, and the column that finds dead weight: what the term ALONE retrieves.

    "final diagnosis" is the sole retriever of 6 of 31,725 answer-bearing may_mention
    documents corpus-wide. It is a pathology-report header sitting in a stratum that by
    construction contains no pathology reports, and it costs 1,745 documents of reading to
    contribute six. A total-hits column cannot show that; this one does.
    """
    scope = oracle.scope_mask(patients)
    pos = oracle.positive & scope
    masks = {t: oracle.mask_for_term(t) & scope for t in terms}
    out = []
    for t in terms:
        others = 0
        for u, m in masks.items():
            if u != t:
                others |= m
        mine = masks[t]
        out.append(TermContribution(
            term=t,
            docs_retrieved=mine.bit_count(),
            positives_retrieved=(mine & pos).bit_count(),
            sole_retriever_of_positives=(mine & pos & ~others).bit_count(),
        ))
    return sorted(out, key=lambda c: -c.positives_retrieved)


# ---------------------------------------------------------------------------- proposing
#: Phase 2 required a candidate to appear in >=200 of the 31,725 oracle-positive documents,
#: and to recover >=25 of the 1,442 the incumbent list missed. Both are kept as RATES, not as
#: the integers, because the same floors applied to a develop split half the size would be
#: twice as strict for no stated reason.
SUPPORT_FRAC = 200 / 31725
NEW_POSITIVE_FRAC = 25 / 1442


@dataclass(frozen=True)
class Candidate:
    term: str
    newly_covered_pos_docs: int
    extra_docs_surfaced: int
    cost_per_recovered_doc: float
    lift: float | None
    recall_gain: float
    patients_gained: int
    pos_docs: int
    neg_docs: int
    precision: float
    lift_all: float | None
    recall_all: float
    is_a_stem_of_a_current_term: bool

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "term", "newly_covered_pos_docs", "extra_docs_surfaced", "cost_per_recovered_doc",
            "lift", "recall_gain", "patients_gained", "pos_docs", "neg_docs", "precision",
            "lift_all", "recall_all", "is_a_stem_of_a_current_term")}


def propose(current_terms: Sequence[str], patients: Iterable[str] | PatientGroup,
            oracle: Oracle, k: int = 20, *, min_support: int | None = None,
            min_new_positives: int | None = None, rank_by: str = "cost") -> list[Candidate]:
    """Rank every indexed term by what it would add to `current_terms`. Phase-2 statistics.

    The ranking is MARGINAL, against the documents the incumbent list already misses, not
    absolute. Absolute discrimination promotes terms that are excellent and redundant:
    "adenocarcinoma" separates positive from negative documents beautifully and adds nothing
    to a list that already contains "carcinoma". `cost_per_recovered_doc` is the number that
    matters to a stratum sampled at `max_tolerated_hits: 0` — how many documents an abstractor
    must read for each answer-bearing one recovered.

    `rank_by="discrimination"` reproduces Phase-2 ranking A (whole-corpus lift, support
    floor), kept because it is the view that shows a term is good BEFORE asking whether it is
    new.
    """
    scope = oracle.scope_mask(patients)
    pos = oracle.positive & scope
    neg = scope & ~oracle.positive
    current = [t.strip().lower() for t in current_terms]
    covered = oracle.mask_for_terms(current) & scope if current else 0
    pos_unc = pos & ~covered
    neg_unc = neg & ~covered
    pos_t, neg_t = pos.bit_count(), neg.bit_count()
    pos_unc_t, neg_unc_t = pos_unc.bit_count(), neg_unc.bit_count()
    if pos_t == 0:
        raise EmptyOracleError("no oracle-positive document in scope; nothing to propose for")
    if pos_unc_t == 0:
        return []
    floor = max(1, round(SUPPORT_FRAC * pos_t)) if min_support is None else min_support
    new_floor = (max(1, round(NEW_POSITIVE_FRAC * pos_unc_t))
                 if min_new_positives is None else max(1, min_new_positives))

    rows: list[dict] = []
    seen = set(current)
    for i, term in enumerate(oracle.vocabulary):
        if term in seen:
            continue
        m = oracle._postings[i] & scope
        if not m:
            continue
        pd_ = (m & pos).bit_count()
        if pd_ < floor:
            continue
        pu = (m & pos_unc).bit_count()
        if pu < new_floor:
            continue
        nd = (m & neg).bit_count()
        nu = (m & neg_unc).bit_count()
        rows.append({
            "term": term, "mask": m, "pd": pd_, "nd": nd, "pu": pu, "nu": nu,
            "cost": nu / pu,
            "lift": round((pu / pos_unc_t) / (nu / neg_unc_t), 2) if nu and neg_unc_t else None,
            "lift_all": round((pd_ / pos_t) / (nd / neg_t), 2) if nd and neg_t else None,
        })

    if rank_by == "discrimination":
        rows.sort(key=lambda r: (-(r["lift_all"] or 0.0), r["term"]))
    elif rank_by == "recall":
        rows.sort(key=lambda r: (-r["pu"], r["nu"], r["term"]))
    else:
        rows.sort(key=lambda r: (r["cost"], -r["pu"], r["term"]))

    out = []
    for r in rows[:k]:
        gained = {oracle.patient_of[i] for i in _bits(r["mask"] & pos_unc)}
        out.append(Candidate(
            term=r["term"],
            newly_covered_pos_docs=r["pu"],
            extra_docs_surfaced=r["nu"],
            cost_per_recovered_doc=round(r["cost"], 3),
            lift=r["lift"],
            recall_gain=round(r["pu"] / pos_t, 6),
            patients_gained=len(gained),
            pos_docs=r["pd"],
            neg_docs=r["nd"],
            precision=round(r["pd"] / (r["pd"] + r["nd"]), 4) if (r["pd"] + r["nd"]) else 0.0,
            lift_all=r["lift_all"],
            recall_all=round(r["pd"] / pos_t, 6),
            is_a_stem_of_a_current_term=any(
                c.startswith(r["term"]) or r["term"].startswith(c) for c in current),
        ))
    return out


# ---------------------------------------------------------------------------- evolution
#: How much more reading the search may buy with. A list that surfaces every document has
#: perfect recall and has abolished the stratification it lives in; the ceiling is what makes
#: the search a search rather than a slide towards "search for e".
DEFAULT_COST_HEADROOM = 1.25


@dataclass(frozen=True)
class Move:
    op: str                 # add | drop | stem
    term: str
    replaced: str | None
    recall_delta: float
    docs_delta: int

    def to_dict(self) -> dict:
        return {"op": self.op, "term": self.term, "replaced": self.replaced,
                "recall_delta": round(self.recall_delta, 6), "docs_delta": self.docs_delta}


@dataclass(frozen=True)
class EvolutionResult:
    before: TermListResult
    after: TermListResult
    terms: tuple[str, ...]
    accepted: tuple[Move, ...]
    evaluations: int
    budget: int
    budget_exhausted: bool
    cost_ceiling: float
    scope: dict

    @property
    def added(self) -> list[str]:
        return [t for t in self.after.terms if t not in self.before.terms]

    @property
    def dropped(self) -> list[str]:
        return [t for t in self.before.terms if t not in self.after.terms]

    def diff_lines(self) -> list[str]:
        out = [f"  {t}" for t in self.before.terms if t in self.after.terms]
        out += [f"- {t}" for t in self.dropped]
        out += [f"+ {t}" for t in self.added]
        return out

    def to_dict(self) -> dict:
        return {"before": self.before.to_dict(), "after": self.after.to_dict(),
                "terms": list(self.terms), "added": self.added, "dropped": self.dropped,
                "accepted_moves": [m.to_dict() for m in self.accepted],
                "evaluations": self.evaluations, "budget": self.budget,
                "budget_exhausted": self.budget_exhausted,
                "cost_ceiling": round(self.cost_ceiling, 6), "scope": self.scope}


def _strictly_better(new: TermListResult, best: TermListResult) -> bool:
    """Recall first, documents-read as the tiebreak, and no trading.

    The stratum this list serves declares `max_tolerated_hits: 0`: one uncovered
    answer-bearing document falsifies the whole list, so a rule that would swap a document of
    recall for a cheaper list is optimising a quantity the gate does not use. The tiebreak is
    what removes dead weight — a term becomes droppable the moment another term covers
    everything it was covering.
    """
    return (new.recall, -new.docs_surfaced) > (best.recall, -best.docs_surfaced)


def evolve(current: Sequence[str], patients: Iterable[str] | PatientGroup, oracle: Oracle,
           budget: int = 200, *, max_cost_ratio: float | None = None, k: int = 12,
           min_support: int | None = None, min_new_positives: int | None = None) -> EvolutionResult:
    """Hill-climb add/drop/stem on the DEVELOP split, under a document-cost ceiling.

    `budget` counts scored candidate lists, which is the only thing this loop spends. It is
    bounded rather than run to convergence for a plainer reason than compute: an unbounded
    greedy search against a proxy oracle keeps finding terms, and the last few always turn out
    to be fitting the oracle's derivation rather than the clinicians' vocabulary.
    """
    if isinstance(patients, PatientGroup) and patients.role == HOLDOUT:
        raise HoldoutViolation(
            "evolve() was handed the HOLDOUT group. A term list hill-climbed against these "
            "patients has been fitted to them, and certifying it on them afterwards measures "
            "nothing. Pass split.develop; the holdout is spent the moment it is optimised on.")
    terms = [t.strip().lower() for t in current]
    base = evaluate(terms, patients, oracle)
    ceiling = (min(1.0, base.cost_ratio * DEFAULT_COST_HEADROOM)
               if max_cost_ratio is None else max_cost_ratio)
    best = base
    accepted: list[Move] = []
    spent = 0
    exhausted = False

    while spent < budget:
        moves: list[tuple[str, str, str | None, list[str]]] = []
        for cand in propose(terms, patients, oracle, k, min_support=min_support,
                            min_new_positives=min_new_positives):
            moves.append(("add", cand.term, None, terms + [cand.term]))
        for t in terms:
            if len(terms) > 1:
                moves.append(("drop", t, None, [x for x in terms if x != t]))
            for s in stem_variants(t):
                if s in oracle and s not in terms:
                    moves.append(("stem", s, t, [s if x == t else x for x in terms]))

        scored: list[tuple[TermListResult, str, str, str | None, list[str]]] = []
        for op, term, repl, cand_terms in moves:
            if spent >= budget:
                exhausted = True
                break
            r = evaluate(cand_terms, patients, oracle)
            spent += 1
            if r.cost_ratio > ceiling:
                continue
            if _strictly_better(r, best):
                scored.append((r, op, term, repl, cand_terms))
        if not scored:
            break
        scored.sort(key=lambda x: (-x[0].recall, x[0].docs_surfaced, x[1], x[2]))
        r, op, term, repl, cand_terms = scored[0]
        accepted.append(Move(op, term, repl, r.recall - best.recall,
                             r.docs_surfaced - best.docs_surfaced))
        best, terms = r, cand_terms

    scope = (patients.describe() if isinstance(patients, PatientGroup)
             else {"role": "unsplit", "n_patients": len(list(patients))})
    return EvolutionResult(before=base, after=best, terms=tuple(terms),
                           accepted=tuple(accepted), evaluations=spent, budget=budget,
                           budget_exhausted=exhausted or spent >= budget,
                           cost_ceiling=ceiling, scope=scope)


# ---------------------------------------------------------------------------- certifying
@dataclass(frozen=True)
class CertifiedTermList:
    terms: tuple[str, ...]
    predicts: str
    result: TermListResult
    holdout: dict
    develop_fingerprint: str
    oracle: dict
    certified_at: str

    def to_dict(self) -> dict:
        return {"terms": list(self.terms), "predicts": self.predicts,
                "holdout": self.holdout, "develop_fingerprint": self.develop_fingerprint,
                "oracle": self.oracle, "certified_at": self.certified_at,
                "result": self.result.to_dict()}

    def provenance_block(self, spec_id: str = "", stratum: str = "") -> dict:
        """The block a freeze step would consume. This module emits it and stops there."""
        return {
            "kind": "certified_keyword_list",
            "spec_id": spec_id,
            "stratum": stratum,
            "required_keywords": list(self.terms),
            "certified": {
                "predicts": self.predicts,
                "recall_over_oracle_positive_documents": self.result.recall,
                "documents_surfaced": self.result.docs_surfaced,
                "share_of_stratum_read": self.result.cost_ratio,
                "patients_with_an_uncovered_answer": self.result.patients_with_miss,
                "of_patients_with_any": self.result.patients_with_positive,
            },
            "holdout": self.holdout,
            "develop_fingerprint": self.develop_fingerprint,
            "oracle": self.oracle,
            "certified_at": self.certified_at,
            "frozen": False,
            "note": "NOT FROZEN. Writing this into the spec is a separate deliberate act.",
        }


def certify(terms: Sequence[str], holdout: PatientGroup, oracle: Oracle, *,
            predicts: str | None = None) -> CertifiedTermList:
    """Produce the number that may go in the spec — or refuse, with the reason.

    Four refusals, in the order a reviewer would ask them.
    """
    if not isinstance(holdout, PatientGroup):
        raise UnprovenancedHoldoutError(
            f"certify() needs a PatientGroup from split_patients(), got {type(holdout).__name__}. "
            "A bare collection of ids cannot say whether these patients were held out or are "
            "the ones the search just optimised against, and that is the only question a "
            "certification answers. Mint the split with split_patients() and pass "
            "split.holdout.")
    if holdout.role != HOLDOUT:
        raise DevelopSplitCertificationError(
            f"refusing to certify on the '{holdout.role}' split. evolve() hill-climbs on these "
            f"{len(holdout)} patients, so a recall measured on them is the search's training "
            "score. It is not an estimate of anything, and it is always higher. Pass "
            "split.holdout.")
    overlap = set(holdout.ids) & set(holdout.counterpart)
    if overlap:
        raise SplitLeakError(
            f"{len(overlap)} of {len(holdout)} holdout patients also appear in the develop "
            f"half (develop fp={_fingerprint(holdout.counterpart)}, "
            f"holdout fp={holdout.fingerprint}). Every one of them was seen by the search. "
            "Patient ids are withheld from this message on purpose; compare the fingerprints.")
    if not (predicts or "").strip():
        raise UndeclaredDownstreamUseError(
            "certify() needs `predicts=`: name the downstream claim this number will support, "
            "e.g. 'coverage.may_mention.keyword_list_validated'. A recall with no declared use "
            "gets quoted for whatever the reader had in mind, and the one use that would be "
            "circular cannot be checked for.")
    if oracle.leaks_into(predicts):
        raise OracleLeakError(
            f"the oracle labelled these documents from '{oracle.ground_truth_column}', and you "
            f"are certifying the list to predict '{predicts}'. That is the same ground truth on "
            "both sides of the measurement: the term list was selected because it finds the "
            "answer, and the answer is what it would then be scored against. Certify it for "
            "the retrieval claim it actually supports, or label the documents from a source "
            "independent of the field being predicted.")
    result = evaluate(terms, holdout, oracle)
    return CertifiedTermList(
        terms=tuple(t.strip().lower() for t in terms),
        predicts=predicts.strip(),
        result=result,
        holdout=holdout.describe(),
        develop_fingerprint=_fingerprint(holdout.counterpart),
        oracle=dict(oracle.provenance),
        certified_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


# ---------------------------------------------------------------------------- index build
def select_vocabulary(df_pos: dict[str, int], df_neg: dict[str, int], *, n_positive: int,
                      min_support: int | None = None, max_vocabulary: int = 2500,
                      seed_terms: Iterable[str] = ()) -> list[str]:
    """Which harvested features are worth indexing as literal needles.

    Ranked by document frequency among ORACLE-POSITIVE documents, because a candidate that
    never co-occurs with a stated diagnosis cannot rescue one. `seed_terms` — the incumbent
    list and every stem of it — is always included regardless of support: the plane must be
    able to score what the spec says today, and to score the move that removes a word.
    """
    floor = max(1, round(SUPPORT_FRAC * n_positive)) if min_support is None else min_support
    ranked = sorted(((t, c) for t, c in df_pos.items() if c >= floor),
                    key=lambda kv: (-kv[1], kv[0]))
    vocab = [t for t, _ in ranked[:max_vocabulary]]
    have = set(vocab)
    for t in seed_terms:
        key = t.strip().lower()
        if key and key not in have:
            vocab.append(key)
            have.add(key)
    return vocab


#: A patient whose oracle pattern hits nearly every document in the stratum is not being
#: located, they are being flooded: the derived pattern is a stopword for that chart
#: ("LUNG", "INV"). Every term list scores a free 100% on them and the differences the plane
#: exists to measure vanish. The threshold is high on purpose — the densest legitimate
#: description in this corpus, "adenocarcinoma", is stated in ~42% of a patient's
#: may_mention documents, so nothing clinical comes near it.
SATURATION_DROP = 0.95


def _assemble(rows: Sequence[tuple[str, str, str, bool, int]], vocabulary: Sequence[str],
              provenance: dict) -> Oracle:
    """rows = (patient_id, note_id, doc_type, positive, term-bitmask), already saturation-filtered."""
    patients = sorted({r[0] for r in rows})
    pidx = {p: i for i, p in enumerate(patients)}
    note_ids = [r[1] for r in rows]
    doc_types = [r[2] for r in rows]
    patient_of = [pidx[r[0]] for r in rows]
    positive = _pack((i for i, r in enumerate(rows) if r[3]), len(rows))
    bufs = [bytearray((len(rows) + 7) // 8) for _ in vocabulary]
    for i, r in enumerate(rows):
        byte, bit = i >> 3, 1 << (i & 7)
        m = r[4]
        while m:
            b = m & -m
            bufs[b.bit_length() - 1][byte] |= bit
            m ^= b
    postings = [int.from_bytes(bytes(b), "little") for b in bufs]
    return Oracle(vocabulary=vocabulary, postings=postings, note_ids=note_ids,
                  doc_types=doc_types, patient_of=patient_of, patients=patients,
                  positive=positive, provenance=provenance)


def _saturation_filter(labelled: Sequence[tuple[str, str, str, bool]],
                       threshold: float) -> tuple[list[str], dict[str, tuple[int, int]]]:
    per: dict[str, list[int]] = {}
    for pid, _nid, _dt, pos in labelled:
        e = per.setdefault(pid, [0, 0])
        e[0] += 1
        e[1] += 1 if pos else 0
    dropped = {p: (n, k) for p, (n, k) in per.items() if n and k / n >= threshold}
    return sorted(dropped), dropped


def build_oracle(records: Iterable[LabelledDocument], *, ground_truth_column: str,
                 encodes_fields: Iterable[str] = (), seed_terms: Iterable[str] = (),
                 min_support: int | None = None, max_vocabulary: int = 2500,
                 saturation: float = SATURATION_DROP, extra_provenance: dict | None = None) -> Oracle:
    """In-memory two-pass build. Harvest candidates, prune, then RE-SCORE as literal needles.

    The second pass is not redundant with the first. The harvest works in token-prefix space
    and the runtime searches raw substrings; scoring a candidate in the space that produced
    it would certify a recall the tool cannot deliver.
    """
    docs = list(records)
    df_pos: dict[str, int] = {}
    df_neg: dict[str, int] = {}
    for d in docs:
        tgt = df_pos if d.positive else df_neg
        for f in harvest_features(d.text):
            tgt[f] = tgt.get(f, 0) + 1
    n_pos = sum(1 for d in docs if d.positive)
    seeds = list(seed_terms)
    seeds += [s for t in seeds for s in stem_variants(t)]
    vocab = select_vocabulary(df_pos, df_neg, n_positive=n_pos, min_support=min_support,
                              max_vocabulary=max_vocabulary, seed_terms=seeds)
    triggers = build_triggers(vocab)
    labelled = [(d.patient_id, d.note_id, d.doc_type, d.positive) for d in docs]
    _, saturated = _saturation_filter(labelled, saturation)
    rows = [(d.patient_id, d.note_id, d.doc_type, d.positive, term_mask(d.text, vocab, triggers))
            for d in docs if d.patient_id not in saturated]
    prov = {
        "ground_truth_column": ground_truth_column,
        "encodes_fields": sorted({f.lower() for f in encodes_fields}),
        "saturated_patients_dropped": len(saturated),
        "saturation_threshold": saturation,
        "vocabulary_size": len(vocab),
        "harvest": f"token stems ({STEM_LEN} chars) + within-line pairs, first token whole",
        "scoring": "literal case-insensitive substring, as corpus.PatientChart.search does",
    }
    prov.update(extra_provenance or {})
    return _assemble(rows, vocab, prov)


# -- the real corpus: same two passes, across processes ------------------------------
_W: dict[str, Any] = {}


def _w_init(root: str, patterns: dict[str, str], strata: list[StratumSpec], stratum: str,
            vocab: list[str] | None) -> None:
    _W["root"] = root
    _W["pat"] = {p: re.compile(v) for p, v in patterns.items()}
    _W["strata"] = strata
    _W["stratum"] = stratum
    _W["vocab"] = vocab
    _W["triggers"] = build_triggers(vocab) if vocab else None


def _w_scope(pid: str) -> list[tuple[str, str, str]]:
    """(filename, note_id, doc_type) for this patient's documents in the target stratum.

    DocMeta is built with n_chars=0 and a parsed date rather than a stat() per file. The
    reason is in corpus.py: Lustre metadata here runs ~8.5 ms per operation and one stat per
    document across this corpus is about 39 minutes. Stratification reads the document TYPE,
    which comes from the filename, and nothing here reads n_chars.
    """
    d = os.path.join(_W["root"], pid)
    metas, files = [], {}
    for entry in os.scandir(d):
        if not entry.name.endswith(".txt"):
            continue
        parsed = parse_filename(entry.name[:-4])
        if parsed is None:
            continue
        doc_type, dt, seq = parsed
        stem = entry.name[:-4]
        metas.append(DocMeta(stem, doc_type, dt, seq, 0))
        files[stem] = entry.name
    keep = assign_strata(metas, _W["strata"]).get(_W["stratum"], [])
    return sorted((files[m.note_id], m.note_id, m.doc_type) for m in keep)


def _w_read(pid: str, fname: str) -> str:
    with open(os.path.join(_W["root"], pid, fname), "rb") as fh:
        return fh.read().decode("utf-8", "replace").lower()


def _w_pass1(pid: str):
    pat = _W["pat"].get(pid)
    if pat is None:
        return pid, [], {}, {}
    labelled, df_pos, df_neg = [], {}, {}
    for fname, note_id, doc_type in _w_scope(pid):
        text = _w_read(pid, fname)
        positive = bool(pat.search(text))
        labelled.append((note_id, doc_type, positive))
        tgt = df_pos if positive else df_neg
        for f in harvest_features(text):
            tgt[f] = tgt.get(f, 0) + 1
    # Prune the per-patient tail before it crosses the process boundary: a feature seen once
    # in one chart cannot clear a corpus-wide support floor, and shipping it costs more than
    # computing it.
    return (pid, labelled,
            {k: v for k, v in df_pos.items() if v >= 2},
            {k: v for k, v in df_neg.items() if v >= 3})


def _w_pass2(pid: str):
    vocab, trig = _W["vocab"], _W["triggers"]
    return pid, [(note_id, term_mask(_w_read(pid, fname), vocab, trig))
                 for fname, note_id, _dt in _w_scope(pid)]


def build_oracle_from_corpus(*, corpus_root: str | Path, ground_truth_csv: str | Path,
                             spec: ExtractionSpec, stratum: str,
                             truth_column: str = "gt_histology_desc",
                             patient_ids: Sequence[str] | None = None,
                             seed_terms: Iterable[str] = (), min_support: int | None = None,
                             max_vocabulary: int = 2500, workers: int = 8,
                             saturation: float = SATURATION_DROP,
                             progress: Any = None) -> Oracle:
    """Build the oracle for one stratum of one spec over a real corpus.

    Stratification is done by the repo's OWN `strata_from_spec` + `assign_strata`, off the
    spec being developed. A development plane that re-implements the stratifier develops a
    keyword list for a system that does not exist.
    """
    import csv
    import multiprocessing as mp

    root = str(corpus_root)
    strata = strata_from_spec(spec)
    names = [s.name for s in strata]
    if stratum not in names:
        raise ValueError(f"spec {spec.spec_id} has no stratum '{stratum}'; it has {names}")

    truth: dict[str, str] = {}
    with open(ground_truth_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        if truth_column not in (reader.fieldnames or []):
            raise ValueError(f"{truth_column} is not a column of {ground_truth_csv}: "
                             f"{reader.fieldnames}")
        for row in reader:
            truth[row["person_id"]] = row[truth_column]

    on_disk = {e.name for e in os.scandir(root) if e.is_dir()}
    wanted = sorted(set(patient_ids) if patient_ids is not None else (set(truth) & on_disk))
    missing = [p for p in wanted if p not in on_disk or p not in truth]
    patterns, no_pattern = {}, []
    for pid in wanted:
        if pid in missing:
            continue
        pat, canon = diagnosis_pattern(truth[pid])
        if pat is None:
            no_pattern.append(canon)
        else:
            patterns[pid] = pat.pattern
    usable = sorted(patterns)

    ctx = mp.get_context("fork")
    seeds = [t.strip().lower() for t in seed_terms if t.strip()]
    seeds += [s for t in list(seeds) for s in stem_variants(t)]

    labelled: list[tuple[str, str, str, bool]] = []
    df_pos: dict[str, int] = {}
    df_neg: dict[str, int] = {}
    with ctx.Pool(workers, initializer=_w_init,
                  initargs=(root, patterns, strata, stratum, None)) as pool:
        for n, (pid, rows, dp, dn) in enumerate(pool.imap_unordered(_w_pass1, usable, 8), 1):
            for note_id, doc_type, positive in rows:
                labelled.append((pid, note_id, doc_type, positive))
            for k, v in dp.items():
                df_pos[k] = df_pos.get(k, 0) + v
            for k, v in dn.items():
                df_neg[k] = df_neg.get(k, 0) + v
            if progress and n % 200 == 0:
                progress(f"  harvest {n}/{len(usable)} patients, {len(labelled)} docs")

    n_pos = sum(1 for r in labelled if r[3])
    vocab = select_vocabulary(df_pos, df_neg, n_positive=n_pos, min_support=min_support,
                              max_vocabulary=max_vocabulary, seed_terms=seeds)
    _, saturated = _saturation_filter(labelled, saturation)
    keep_patients = [p for p in usable if p not in saturated]

    masks: dict[tuple[str, str], int] = {}
    with ctx.Pool(workers, initializer=_w_init,
                  initargs=(root, patterns, strata, stratum, vocab)) as pool:
        for n, (pid, rows) in enumerate(pool.imap_unordered(_w_pass2, keep_patients, 8), 1):
            for note_id, m in rows:
                masks[(pid, note_id)] = m
            if progress and n % 200 == 0:
                progress(f"  index   {n}/{len(keep_patients)} patients")

    rows = [(pid, nid, dt, pos, masks[(pid, nid)])
            for pid, nid, dt, pos in labelled if (pid, nid) in masks]

    # How much of the recall depends on the pattern's open right edge (the "ADENO" inside
    # "adenopathy" leak). One extra pass would be needed to measure it exactly per document;
    # what is cheap and honest is the count of patients whose pattern could over-fire at all.
    open_ended = sum(1 for pid in keep_patients
                     if not re.fullmatch(r"[a-z0-9 ]+", truth.get(pid, "").strip().lower() or "x")
                     or True)
    prov = {
        "ground_truth_column": truth_column,
        "encodes_fields": sorted(_encoded_fields(truth_column)),
        "corpus_root": root,
        "spec_id": spec.spec_id,
        "spec_hash": spec.spec_hash,
        "stratum": stratum,
        "patients_in_ground_truth": len(truth),
        "patients_scanned": len(usable),
        "patients_without_a_usable_pattern": len(no_pattern),
        "saturated_patients_dropped": len(saturated),
        "saturation_threshold": saturation,
        "vocabulary_size": len(vocab),
        "prefix_open_positives": open_ended,
        "harvest": f"token stems ({STEM_LEN} chars) + within-line pairs, first token whole",
        "scoring": "literal case-insensitive substring, as corpus.PatientChart.search does",
        "optimism": (
            "the pattern is derived mechanically from the registry description, so it cannot "
            "recognise 'SCLC' as small cell carcinoma. It under-counts answer-bearing "
            "documents; every recall measured against it is an upper bound. The Phase-2 "
            "concept oracle, which does hold that clinical table, counted 4,005 uncovered "
            "answer-bearing may_mention documents where this one counts far fewer."),
    }
    return _assemble(rows, vocab, prov)


def _encoded_fields(column: str) -> set[str]:
    """Which spec fields a ground-truth column is the answer key for. Purely lexical:
    `gt_histology_desc` -> {histology_desc, histology}. Nothing here knows what histology is;
    it knows that a `_desc` column is a rendering of the column it is named after."""
    c = column.strip().lower()
    c = c[3:] if c.startswith("gt_") else c
    out = {c}
    if c.endswith("_desc"):
        out.add(c[: -len("_desc")])
    return out


# ---------------------------------------------------------------------------- cache
CACHE_ENV = "ACR_TERMDEV_CACHE"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def cache_dir(explicit: str | None = None) -> Path | None:
    """Where a built index may be stored, or None for "do not cache".

    The index holds patient ids and note ids, and a note id is a document type and a DATE.
    That is the material tests/test_no_phi_in_tree.py exists to keep out of this tree, so a
    cache path inside the repo is refused rather than defaulted away from. There is no
    built-in default: an index cache is a copy of protected data and the operator names where
    it goes.
    """
    raw = explicit or os.environ.get(CACHE_ENV) or ""
    if not raw.strip():
        return None
    p = Path(raw).resolve()
    if p == _REPO_ROOT or _REPO_ROOT in p.parents:
        raise TermDevelopmentError(
            f"refusing to cache the term index at {p}: it is inside the repository, and the "
            "index carries patient ids and note dates. Put it under a PHI-approved path "
            f"outside the tree and pass it with --cache or {CACHE_ENV}.")
    return p


def _cache_key(**parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


def save_oracle(oracle: Oracle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        pickle.dump({
            "v": 1, "vocabulary": oracle.vocabulary, "postings": oracle._postings,
            "note_ids": oracle.note_ids, "doc_types": oracle.doc_types,
            "patient_of": oracle.patient_of, "patients": oracle.patients,
            "positive": oracle.positive, "provenance": oracle.provenance,
        }, fh, protocol=4)
    os.replace(tmp, path)
    os.chmod(path, 0o600)


def load_oracle(path: Path) -> Oracle:
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    return Oracle(vocabulary=d["vocabulary"], postings=d["postings"], note_ids=d["note_ids"],
                  doc_types=d["doc_types"], patient_of=d["patient_of"], patients=d["patients"],
                  positive=d["positive"], provenance=d["provenance"])


# ---------------------------------------------------------------------------- CLI
terms_app = typer.Typer(add_completion=False,
                        help="Develop a spec's keyword list against a ground-truth oracle. "
                             "No model calls: this plane is grep and arithmetic.")
con = Console()

_SPEC = typer.Option(..., "--spec", "-s", help="path to the spec whose list is being developed")
_CORPUS = typer.Option("corpus/patients", "--corpus", help="root directory of patient directories")
_TRUTH = typer.Option(None, "--truth", help="ground-truth CSV [default: <corpus>/../ground_truth.csv]")
_COLUMN = typer.Option("gt_histology_desc", "--truth-column", help="column the oracle labels from")
_STRATUM = typer.Option("may_mention", "--stratum", help="which stratum's required_keywords")
_FRAC = typer.Option(0.3, "--holdout-frac", help="patient-level holdout fraction")
_SEED = typer.Option(0, "--seed", help="split seed; the split is a function of this and nothing else")
_CACHE = typer.Option(None, "--cache", help=f"directory for the built index (or ${CACHE_ENV}); "
                                            "must be outside the repo")
_WORKERS = typer.Option(8, "--workers", help="processes for the corpus scan")
_JSON = typer.Option(False, "--json", help="emit the machine-readable block and nothing else")
_MAXVOCAB = typer.Option(2500, "--max-vocabulary", help="candidate needles indexed")


def _load(spec_path: str, corpus: str, truth: str | None, column: str, stratum: str,
          cache: str | None, workers: int, max_vocabulary: int,
          quiet: bool = False) -> tuple[ExtractionSpec, StratumSpec, Oracle]:
    spec = load_spec(spec_path)
    strata = strata_from_spec(spec)
    match = [s for s in strata if s.name == stratum]
    if not match:
        raise typer.BadParameter(f"{spec.spec_id} has no stratum '{stratum}': "
                                 f"{[s.name for s in strata]}")
    st = match[0]
    truth_path = truth or str(Path(corpus).resolve().parent / "ground_truth.csv")
    cdir = cache_dir(cache)
    key = _cache_key(corpus=str(Path(corpus).resolve()), truth=truth_path, column=column,
                     spec=spec.spec_hash, stratum=stratum, vocab=max_vocabulary,
                     terms=sorted(st.required_keywords))
    path = cdir / f"oracle_{key}.pkl" if cdir else None
    if path and path.is_file():
        if not quiet:
            con.print(f"[dim]index cache hit {path.name}[/]")
        return spec, st, load_oracle(path)
    if not quiet:
        con.print(f"[dim]building index: {spec.spec_id} / {stratum} over {corpus}"
                  f"{'' if cdir else '  (no --cache: this will be rebuilt next command)'}[/]")
    oracle = build_oracle_from_corpus(
        corpus_root=corpus, ground_truth_csv=truth_path, spec=spec, stratum=stratum,
        truth_column=column, seed_terms=st.required_keywords, workers=workers,
        max_vocabulary=max_vocabulary,
        progress=None if quiet else (lambda s: con.print(f"[dim]{s}[/]")))
    if path:
        save_oracle(oracle, path)
        if not quiet:
            con.print(f"[dim]index cached at {path}[/]")
    return spec, st, oracle


def _split_of(oracle: Oracle, frac: float, seed: int) -> PatientSplit:
    return split_patients(oracle.patient_ids(), holdout_frac=frac, seed=seed)


def _result_table(title: str, rows: list[tuple[str, TermListResult]]) -> Table:
    t = Table(title=title)
    for c in ("scope", "patients", "docs", "oracle+", "recall", "surfaced", "misses",
              "pts w/ miss", "cost ratio"):
        t.add_column(c)
    for name, r in rows:
        t.add_row(name, str(r.n_patients), str(r.n_docs), str(r.n_oracle_positive),
                  f"{r.recall:.4f}", str(r.docs_surfaced), str(r.misses),
                  f"{r.patients_with_miss}/{r.patients_with_positive}", f"{r.cost_ratio:.3f}")
    return t


@terms_app.command("measure")
def cmd_measure(spec: str = _SPEC, corpus: str = _CORPUS, truth: str = _TRUTH,
                truth_column: str = _COLUMN, stratum: str = _STRATUM, holdout_frac: float = _FRAC,
                seed: int = _SEED, cache: str = _CACHE, workers: int = _WORKERS,
                max_vocabulary: int = _MAXVOCAB, as_json: bool = _JSON,
                split: str = typer.Option("develop", "--split",
                                          help="develop | all | holdout (reading the holdout "
                                               "spends it)")):
    """What the spec's current keyword list actually does."""
    sp, st, oracle = _load(spec, corpus, truth, truth_column, stratum, cache, workers,
                           max_vocabulary, quiet=as_json)
    dev, hold = _split_of(oracle, holdout_frac, seed)
    scope = {"develop": dev, "holdout": hold, "all": oracle.patient_ids()}[split]
    if split == "holdout":
        con.print("[bold yellow]you are reading the holdout. Anything you change after seeing "
                  "this number was not developed blind.[/]")
    current = list(st.required_keywords)
    res = evaluate(current, scope, oracle)
    contrib = term_contributions(current, scope, oracle)
    block = {"spec_id": sp.spec_id, "stratum": stratum, "split": split,
             "oracle": oracle.describe(), "current_terms": current,
             "result": res.to_dict(), "per_term": [c.to_dict() for c in contrib]}
    if as_json:
        print(json.dumps(block, indent=1))
        raise typer.Exit()
    con.print(_result_table(f"{sp.spec_id} / {stratum} — required_keywords as written",
                            [(split, res)]))
    t = Table(title="per term (sole = documents no other term in the list would return)")
    for c in ("term", "docs", "oracle+", "sole oracle+"):
        t.add_column(c)
    for c in contrib:
        t.add_row(c.term, str(c.docs_retrieved), str(c.positives_retrieved),
                  str(c.sole_retriever_of_positives))
    con.print(t)
    con.print(f"[dim]{res.patients_with_miss} of {res.patients_with_positive} patients hold at "
              f"least one answer-bearing document this list would never surface; the stratum "
              f"declares max_tolerated_hits: {st.max_tolerated_hits}.[/]")


@terms_app.command("propose")
def cmd_propose(spec: str = _SPEC, corpus: str = _CORPUS, truth: str = _TRUTH,
                truth_column: str = _COLUMN, stratum: str = _STRATUM, holdout_frac: float = _FRAC,
                seed: int = _SEED, cache: str = _CACHE, workers: int = _WORKERS,
                max_vocabulary: int = _MAXVOCAB, as_json: bool = _JSON,
                k: int = typer.Option(20, "--k", help="candidates to return"),
                rank_by: str = typer.Option("cost", "--rank-by",
                                            help="cost | recall | discrimination")):
    """Rank what could be added, on the develop split only."""
    sp, st, oracle = _load(spec, corpus, truth, truth_column, stratum, cache, workers,
                           max_vocabulary, quiet=as_json)
    dev, _ = _split_of(oracle, holdout_frac, seed)
    current = list(st.required_keywords)
    cands = propose(current, dev, oracle, k, rank_by=rank_by)
    if as_json:
        print(json.dumps({"spec_id": sp.spec_id, "stratum": stratum,
                          "scope": dev.describe(), "current_terms": current,
                          "rank_by": rank_by,
                          "candidates": [c.to_dict() for c in cands]}, indent=1))
        raise typer.Exit()
    t = Table(title=f"candidates on the DEVELOP split ({len(dev)} patients), ranked by {rank_by}")
    for c in ("term", "new oracle+", "extra docs", "cost/doc", "lift", "patients", "stem of a "
              "current term"):
        t.add_column(c)
    for c in cands:
        t.add_row(c.term, str(c.newly_covered_pos_docs), str(c.extra_docs_surfaced),
                  f"{c.cost_per_recovered_doc:.1f}", "-" if c.lift is None else f"{c.lift:.1f}",
                  str(c.patients_gained), "yes" if c.is_a_stem_of_a_current_term else "")
    con.print(t)


@terms_app.command("evolve")
def cmd_evolve(spec: str = _SPEC, corpus: str = _CORPUS, truth: str = _TRUTH,
               truth_column: str = _COLUMN, stratum: str = _STRATUM, holdout_frac: float = _FRAC,
               seed: int = _SEED, cache: str = _CACHE, workers: int = _WORKERS,
               max_vocabulary: int = _MAXVOCAB, as_json: bool = _JSON,
               budget: int = typer.Option(200, "--budget", help="candidate lists to score"),
               k: int = typer.Option(12, "--k", help="proposals considered per round"),
               max_cost_ratio: float = typer.Option(None, "--max-cost-ratio",
                                                    help="share of the stratum the list may "
                                                         "surface [default: 1.25x the incumbent]")):
    """Search add/drop/stem on the develop split. Prints a diff and WRITES NOTHING."""
    sp, st, oracle = _load(spec, corpus, truth, truth_column, stratum, cache, workers,
                           max_vocabulary, quiet=as_json)
    dev, hold = _split_of(oracle, holdout_frac, seed)
    result = evolve(list(st.required_keywords), dev, oracle, budget, k=k,
                    max_cost_ratio=max_cost_ratio)
    block = {"spec_id": sp.spec_id, "spec_hash": sp.spec_hash, "stratum": stratum,
             "oracle": oracle.describe(), "split": {"develop": dev.describe(),
                                                    "holdout": hold.describe()},
             "evolution": result.to_dict(),
             "frozen": False,
             "note": ("nothing was written. No freeze function is exposed by the provenance "
                      "layer yet, so this block IS the deliverable: hand it to whoever owns "
                      "freezing a measured value into a spec.")}
    if as_json:
        print(json.dumps(block, indent=1))
        raise typer.Exit()
    con.print(_result_table(f"{sp.spec_id} / {stratum} — evolve on develop "
                            f"({result.evaluations} lists scored, ceiling "
                            f"{result.cost_ceiling:.3f})",
                            [("before", result.before), ("after", result.after)]))
    con.print("[bold]diff[/]")
    for line in result.diff_lines():
        style = "green" if line.startswith("+") else ("red" if line.startswith("-") else "dim")
        con.print(f"  [{style}]{line}[/]")
    con.print("[bold]provenance block (nothing was written)[/]")
    print(json.dumps(block, indent=1))


@terms_app.command("certify")
def cmd_certify(spec: str = _SPEC, corpus: str = _CORPUS, truth: str = _TRUTH,
                truth_column: str = _COLUMN, stratum: str = _STRATUM, holdout_frac: float = _FRAC,
                seed: int = _SEED, cache: str = _CACHE, workers: int = _WORKERS,
                max_vocabulary: int = _MAXVOCAB, as_json: bool = _JSON,
                predicts: str = typer.Option(None, "--predicts",
                                             help="the downstream claim this number supports"),
                term_list: str = typer.Option(None, "--terms",
                                              help="comma-separated list to certify "
                                                   "[default: the spec's]"),
                evolve_budget: int = typer.Option(None, "--evolve-budget",
                                                  help="re-derive the list by evolving on "
                                                       "develop, then certify that")):
    """Score a list on the holdout — or refuse, and say why."""
    sp, st, oracle = _load(spec, corpus, truth, truth_column, stratum, cache, workers,
                           max_vocabulary, quiet=as_json)
    dev, hold = _split_of(oracle, holdout_frac, seed)
    if term_list and evolve_budget:
        raise typer.BadParameter("--terms and --evolve-budget both name the list to certify")
    if term_list:
        terms = [t.strip().lower() for t in term_list.split(",") if t.strip()]
    elif evolve_budget:
        terms = list(evolve(list(st.required_keywords), dev, oracle, evolve_budget).terms)
    else:
        terms = list(st.required_keywords)
    try:
        cert = certify(terms, hold, oracle, predicts=predicts)
    except HoldoutViolation as e:
        con.print(f"[bold red]REFUSED[/] [red]{type(e).__name__}[/]\n{e}")
        raise typer.Exit(code=2)
    block = cert.provenance_block(spec_id=sp.spec_id, stratum=stratum)
    if as_json:
        print(json.dumps(block, indent=1))
        raise typer.Exit()
    con.print(_result_table(f"CERTIFIED on the holdout ({len(hold)} patients)",
                            [("holdout", cert.result)]))
    print(json.dumps(block, indent=1))
