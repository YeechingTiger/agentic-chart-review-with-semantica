"""Develop two retrieval assets on a dev set, certify on a held-out test set, apply at scale.

Two assets in every spec decide what text ever reaches the agent, and both were written by a
language model in one commit, from imagination:

  KEYWORDS  `strata[].required_keywords` — the needles a searched stratum is searched with
  STRATA    `strata[].match.doc_type_matches` — which document types are read exhaustively,
            which are searched, and which are written off as unable to establish anything

Development is affordable because the dev sample is small: every note of every dev patient is
read once by a cheap model, yielding a COMPLETE per-note labelling — for each note, which
fields it ESTABLISHES and which it merely MENTIONS. Complete is what removes the anticipation
blind spot: nothing was searched for, so nothing was missed for not having been thought of.
After that the model is out of the loop and every experiment is set arithmetic over the labels.
Nothing in this module opens a chart or calls a provider.

FOUR NUMBERS, NOT ONE. Everything measured in this project so far has been recall, and recall
is the cheap half: precision and notes_read_per_patient are what the bill is denominated in at
tens of thousands of patients, and patients_losing_the_answer is the only one of the four that
loses an answer rather than merely costing money.

ADOPT BRANCHES ON ASSET KIND and that branch is the point of the module. Keywords are
RETRIEVAL-ONLY — they change which text reaches the agent, never what an answer means — so a
certified list may auto-adopt. Strata are SEMANTIC: a stratum encodes ADMISSIBILITY, a clinical
judgement, so adopt only ever emits a proposal for signature. Data can say where the
information IS; only a clinician can say what may ESTABLISH a value.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import random
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields, replace
from datetime import date
from pathlib import Path
from typing import Any, Literal

import typer
import yaml

from ..contract.spec import ProvenanceRecord, load_spec

DEV, TEST = "dev", "test"
AssetKind = Literal["keywords", "strata"]
KEYWORDS: AssetKind = "keywords"
STRATA: AssetKind = "strata"

#: WHY each kind may or may not be written by a measurement. Quoted verbatim into every
#: adoption and every proposal: the reason is what has to survive, not the rule.
ADOPTION_RULE: dict[str, str] = {
    KEYWORDS: ("RETRIEVAL_ONLY: a keyword list changes which text reaches the agent and nothing "
               "about what an answer means. Certification is sufficient authority."),
    STRATA: ("SEMANTIC: a stratum encodes admissibility — what may ESTABLISH a value — which is a "
             "clinical judgement. A measurement may show where the information is; it may not "
             "decide what counts as evidence. Clinician signature required."),
}

#: How a stratum's policy turns into reads: the same three policies `coverage.py` runs at
#: inference time, so the development plane costs what the runtime costs.
READ_ALL, READ_HITS, READ_NONE = "read_all", "read_keyword_hits", "read_none"
POLICY_READS = {"exhaustive": READ_ALL, "validate_by_sampling": READ_NONE,
                "search_then_read_hits_and_sample_misses": READ_HITS}

#: Below this many test patients a result is `underpowered` whatever it found: one chart moves
#: every rate by more than the differences being compared.
MIN_PATIENTS_FOR_SUPPORT = 20

#: Digest domain. Public on purpose — it keeps patient ids out of the stored split, which is a
#: tracked artefact people paste into write-ups.
ID_DOMAIN = b"acr.improvement.assetdev.patient_digest/1"


class AssetDevelopmentError(Exception):
    """Base for every refusal here."""


class UnindexedTermError(AssetDevelopmentError):
    """A needle outside the labelling's vocabulary; scoring it zero would be a lie."""


class SplitLeakError(AssetDevelopmentError):
    """The dev and test halves are not disjoint."""


class ScoredOnDevError(AssetDevelopmentError):
    """A number the search optimised against, quoted as if it were held out."""


class NegativeControlFailed(AssetDevelopmentError):
    """The same search gained as much on permuted labels, so the gain is not about retrieval."""


class AnswerLeaked(AssetDevelopmentError):
    """A derived term renders a development case's own gold value.

    The permutation control cannot catch this one: such a term genuinely points at the answer on
    every dev case, so shuffling the labels genuinely destroys it and the test passes it. It is
    worthless on test, where nobody supplies the answer. See `acr.improvement.answer_leak`.
    """


class AdoptionAborted(AssetDevelopmentError):
    """The write did not verify; the spec is untouched."""


# ------------------------------------------------------------- the labels this module eats
@dataclass(frozen=True)
class NoteLabel:
    """One note read END TO END by the dev-time scan, and what it turned out to say.

    `establishes` and `mentions` are not nested: a pathology report stating the histology
    establishes it, a progress note saying "known adenocarcinoma" only mentions it — enough to
    break an absence proof, not enough to be a primary source. NOTE TEXT NEVER GOES HERE.
    """
    patient_id: str
    note_id: str
    doc_type: str
    establishes: frozenset[str] = frozenset()
    mentions: frozenset[str] = frozenset()
    terms: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Labelling:
    """The COMPLETE per-note labelling of a cohort — every note of every patient in scope.
    Redeclared here rather than imported from `acr.improvement.labelling` so the two modules are coupled by
    a file format, not a class. `indexed_vocabulary` is what the scan actually indexed; the
    default (every term seen) is right for a fixture and wrong for a scan that capped it."""
    model: str
    prompt_hash: str
    spec_hash: str
    notes: tuple[NoteLabel, ...]
    indexed_vocabulary: frozenset[str] | None = None

    @property
    def hash(self) -> str:
        return _hash("labelling", self.model, self.prompt_hash, self.spec_hash)

    @property
    def vocabulary(self) -> frozenset[str]:
        if self.indexed_vocabulary is not None:
            return self.indexed_vocabulary
        return frozenset().union(*(n.terms for n in self.notes)) if self.notes else frozenset()

    def patient_ids(self) -> list[str]:
        return sorted({n.patient_id for n in self.notes})

    @classmethod
    def load(cls, path: str | Path) -> Labelling:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        v = d.get("indexed_vocabulary")
        notes = tuple(NoteLabel(n["patient_id"], n["note_id"], n["doc_type"],
                                frozenset(n.get("establishes") or ()),
                                frozenset(n.get("mentions") or ()),
                                frozenset(n.get("terms") or ())) for n in d["notes"])
        return cls(d["model"], d["prompt_hash"], d["spec_hash"], notes,
                   None if v is None else frozenset(v))


def _hash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode() + b"\0")
    return h.hexdigest()[:16]


def _digest(patient_id: str) -> str:
    """A patient's identity in a stored artefact. Never the identifier itself."""
    return hmac.new(ID_DOMAIN, patient_id.encode(), hashlib.sha256).hexdigest()[:16]


def _key(note: NoteLabel) -> tuple[str, str]:
    """A note's identity ACROSS patients: `note_id` is a filename stem, unique only within one
    patient directory, so keying a read set by it alone would mark one patient's note as read
    because another patient's note of the same name was."""
    return (note.patient_id, note.note_id)


# ----------------------------------------------------------------------------------- split
@dataclass(frozen=True)
class Split:
    """A patient-level dev/test split, stored once, with its seed and hash attached.

    PATIENT LEVEL, NEVER NOTE LEVEL: charts repeat themselves, and a note-level split puts the
    same pasted-forward sentence in both halves. THE MEMBERS ARE DIGESTS, so the artefact cannot
    leak identifiers. THE SEED IS NOT A PARAMETER OF ANY MEASUREMENT — it is an argument to
    `make_split`, which writes the file once; a seed accepted later is one that can be tried
    again, and this repo already caught seed shopping once at the sampling layer.
    """
    dev: tuple[str, ...]
    test: tuple[str, ...]
    seed: int
    test_frac: float
    created_on: str
    #: Where it is stored. `None` means in memory only, which `certify` refuses.
    path: str | None = None

    @property
    def split_hash(self) -> str:
        return _hash("split", sorted(self.dev), sorted(self.test), self.seed, self.test_frac)

    @property
    def overlap(self) -> frozenset[str]:
        return frozenset(self.dev) & frozenset(self.test)

    def members(self, role: str) -> frozenset[str]:
        if role not in (DEV, TEST):
            raise ValueError(f"role must be {DEV!r} or {TEST!r}, got {role!r}")
        return frozenset(self.dev if role == DEV else self.test)

    def to_dict(self) -> dict:
        return {"kind": "acr.improvement.assetdev.split/1", "split_hash": self.split_hash, "seed": self.seed,
                "test_frac": self.test_frac, "created_on": self.created_on,
                "dev_digests": list(self.dev), "test_digests": list(self.test),
                "note": "Members are HMAC digests of patient ids, never the ids."}

    def save(self, path: str | Path) -> Split:
        """Write once. Overwriting a path with a DIFFERENT split is refused: whichever of the
        two you then quote, the reader cannot see that the other one existed."""
        p = Path(path)
        if p.exists():
            was = json.loads(p.read_text(encoding="utf-8")).get("split_hash")
            if was != self.split_hash:
                raise AssetDevelopmentError(f"{p} already holds split {was}; you are writing "
                                            f"{self.split_hash} (seed {self.seed}). Use a new "
                                            "path and let the history hold both.")
            return replace(self, path=str(p))
        p.parent.mkdir(parents=True, exist_ok=True)
        _write(p, json.dumps(self.to_dict(), indent=1) + "\n")
        return replace(self, path=str(p))

    @classmethod
    def load(cls, path: str | Path) -> Split:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(tuple(d["dev_digests"]), tuple(d["test_digests"]), int(d["seed"]),
                   float(d["test_frac"]), str(d["created_on"]), str(path))


def make_split(patient_ids: Sequence[str], *, seed: int, test_frac: float = 0.4,
               path: str | Path | None = None, today: str | None = None) -> Split:
    """Mint a patient-level split and store it. `seed` is accepted here and nowhere else.
    Sorting before shuffling is not cosmetic: a set has an iteration order the caller did not
    choose, and a split depending on it changes silently between two identical-looking runs."""
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be strictly between 0 and 1, got {test_frac}")
    uniq = sorted({p for p in patient_ids if p})
    if len(uniq) < 2:
        raise ValueError(f"cannot split {len(uniq)} patient(s) into two non-empty halves")
    shuffled = [_digest(p) for p in uniq]
    random.Random(seed).shuffle(shuffled)
    n_test = max(1, min(len(uniq) - 1, round(test_frac * len(uniq))))
    s = Split(tuple(sorted(shuffled[n_test:])), tuple(sorted(shuffled[:n_test])), seed,
              test_frac, today or date.today().isoformat())
    return s.save(path) if path else s


def scope(labelling: Labelling, split: Split, role: str) -> list[NoteLabel]:
    """The notes of one half. Patients the split does not know are dropped deliberately: a
    labelling may cover more patients than the split was minted over, and scoring them as dev
    would put unassigned patients on both sides of the wall."""
    want = split.members(role)
    return [n for n in labelling.notes if _digest(n.patient_id) in want]


# -------------------------------------------------------------------------- retrieval plan
@dataclass(frozen=True)
class RetrievalPlan:
    """What retrieval would do at scale: which stratum a doc type lands in, and what each
    stratum is searched with. `assignment` is ordered and the first substring match wins, the
    rest falling through to `fallback` — the rule `coverage.assign_strata` applies at inference
    time. A development plane that models retrieval differently from the runtime measures a
    system nobody is going to run."""
    field_name: str
    assignment: tuple[tuple[str, str], ...]
    keywords: tuple[tuple[str, tuple[str, ...]], ...]
    policies: tuple[tuple[str, str], ...]
    fallback: str = "cannot_establish"
    #: Set when this spec's strata sit under a claim; it is part of every element path.
    claim: str | None = None

    def stratum_of(self, doc_type: str) -> str:
        low = (doc_type or "").lower()
        return next((s for pat, s in self.assignment if pat.lower() in low), self.fallback)

    def keywords_for(self, stratum: str) -> tuple[str, ...]:
        return dict(self.keywords).get(stratum, ())

    def reads_for(self, stratum: str) -> str:
        default = "exhaustive" if stratum == "can_establish" else "validate_by_sampling"
        return POLICY_READS.get(dict(self.policies).get(stratum, default), READ_NONE)

    def element(self, stratum: str, leaf: str) -> str:
        base = "proof_obligation.for_negative" + (f".claims[{self.claim}]" if self.claim else "")
        return f"{base}.strata[{stratum}].{leaf}"

    @property
    def searched_stratum(self) -> str:
        for stratum, _ in self.policies:
            if self.reads_for(stratum) == READ_HITS:
                return stratum
        raise AssetDevelopmentError(f"plan for {self.field_name!r} has no searched stratum, so "
                                    f"there is no keyword list to develop: {list(self.policies)}")

    def with_keywords(self, stratum: str, terms: Sequence[str]) -> RetrievalPlan:
        kw = dict(self.keywords)
        kw[stratum] = tuple(dict.fromkeys(t.strip().lower() for t in terms if t.strip()))
        return replace(self, keywords=tuple(sorted(kw.items())))

    def with_assignment(self, doc_type: str, stratum: str) -> RetrievalPlan:
        """Move a document type. Prepended, so it wins over the incumbent patterns."""
        rest = tuple((p, s) for p, s in self.assignment if p.lower() != doc_type.lower())
        return replace(self, assignment=((doc_type, stratum),) + rest)

    @classmethod
    def from_spec(cls, spec: Any, field_name: str) -> RetrievalPlan:
        """The plan the spec ships today: the incumbent every candidate is scored against."""
        from ..contract.strata import strata_from_spec

        assignment, keywords, policies, fallback = [], [], [], "cannot_establish"
        for st in strata_from_spec(spec):
            policies.append((st.name, st.policy))
            if st.rest:
                fallback = st.name
            assignment.extend((pat, st.name) for pat in st.doc_type_matches)
            if st.required_keywords:
                keywords.append((st.name, tuple(k.strip().lower() for k in st.required_keywords)))
        fn = getattr(getattr(spec, "proof_obligation", None), "for_negative", {}) or {}
        claim = next((str(c["id"]) for c in (fn.get("claims") or []) if c.get("strata")), None)
        plan = cls(field_name, tuple(assignment), tuple(sorted(keywords)), tuple(policies),
                   fallback, claim)
        # `proof_obligation.required_keywords` is scoped to NO stratum, and the runtime unions
        # every declared term into one search list — so leaving them out made this plan a strict
        # subset of what a run searches, and every candidate was priced against the short list
        # while `adopt` deployed the long one. They belong on the stratum being developed, which is
        # the one whose keywords `evolve` and `certify` move.
        po_terms = [str(k).strip().lower()
                    for k in (getattr(getattr(spec, "proof_obligation", None),
                                      "required_keywords", []) or [])
                    if str(k).strip()]
        if not po_terms:
            return plan
        try:
            target = plan.searched_stratum
        except AssetDevelopmentError:
            # No searched stratum means no keyword list to develop at all; `from_spec`'s callers
            # already handle that refusal, and inventing a stratum here would be worse.
            return plan
        return plan.with_keywords(
            target, list(plan.keywords_for(target)) + po_terms)


@dataclass(frozen=True)
class Candidate:
    """A proposed change to one asset. `element` is the spec path it would land at, carried from
    the moment it is invented: a candidate that cannot name where it would be written is one
    nobody can adopt or audit. `value` is the whole proposed keyword list, or — for strata — a
    tuple of (doc_type, stratum) moves."""
    kind: AssetKind
    field_name: str
    stratum: str
    element: str
    value: tuple[Any, ...]
    rationale: str = ""

    @property
    def is_semantic(self) -> bool:
        """True when adopting this would change what may ESTABLISH a value."""
        return self.kind == STRATA

    @property
    def label(self) -> str:
        if self.kind == KEYWORDS:
            return f"{self.stratum}.required_keywords[{len(self.value)}]"
        return f"{self.stratum}.match[" + ", ".join(f"{d}->{s}" for d, s in self.value) + "]"

    def apply(self, plan: RetrievalPlan) -> RetrievalPlan:
        if self.kind == KEYWORDS:
            return plan.with_keywords(self.stratum, [str(v) for v in self.value])
        for doc_type, stratum in self.value:
            plan = plan.with_assignment(str(doc_type), str(stratum))
        return plan


# ---------------------------------------------------------------------------- the metrics
@dataclass(frozen=True)
class Metrics:
    """The four numbers and the counts behind them. Three are money and one is a lost answer;
    conflating "this costs more" with "this is wrong" is how a cheap plan gets rejected for a
    rounding error in recall and an expensive one gets adopted for the same."""
    n_patients: int
    n_notes: int
    n_read: int
    n_answer_bearing: int
    n_answer_bearing_read: int
    patients_with_an_answer: int
    #: The only failure here that loses an answer rather than costing money.
    patients_losing_the_answer: int

    recall = property(lambda s: _ratio(s.n_answer_bearing_read, s.n_answer_bearing))
    precision = property(lambda s: _ratio(s.n_answer_bearing_read, s.n_read))
    notes_read_per_patient = property(lambda s: _ratio(s.n_read, s.n_patients))
    read_share = property(lambda s: _ratio(s.n_read, s.n_notes))
    answer_coverage = property(lambda s: _ratio(
        s.patients_with_an_answer - s.patients_losing_the_answer, s.patients_with_an_answer))

    def to_dict(self) -> dict:
        return {**asdict(self), "recall": round(self.recall, 4),
                "precision": round(self.precision, 4),
                "notes_read_per_patient": round(self.notes_read_per_patient, 3)}


def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _expand(terms: Sequence[str], vocabulary: frozenset[str]) -> frozenset[str]:
    """Every indexed term a keyword would hit. Keywords match by PREFIX, so `carcinom` covers
    `carcinoma` and `carcinomatosis`, which is what makes a stem move mean something. A needle
    that prefixes nothing raises: a silent zero would land in a recall denominator and read as
    "the corpus does not say it"."""
    # CASE-INSENSITIVE, and returning the vocabulary's OWN spelling. Question 2 asks the reading
    # model to copy each term "as this document spells them", so the vocabulary carries `Endoscopy`
    # and `FILLED PRESCRIPTIONS` verbatim — while a keyword list is written lowercase. Comparing a
    # lowercased needle against an unlowercased vocabulary made `evolve` refuse its own first move:
    # `'endoscopy' prefixes nothing`, in a vocabulary that contained `Endoscopy`. Two spellings of one
    # word are not two terms, and treating them as such is what made `evolve`, `certify` and `adopt`
    # unreachable.
    out: set[str] = set()
    lowered = {w.lower(): w for w in vocabulary}
    for t in (x.strip().lower() for x in terms if x.strip()):
        hit = {orig for low, orig in lowered.items() if low.startswith(t)}
        if not hit:
            raise UnindexedTermError(f"{t!r} prefixes nothing in this labelling's {len(vocabulary)}"
                                     "-term vocabulary, so the labels cannot say what it would "
                                     "surface. Re-scan with it indexed, or drop it from the list.")
        out |= hit
    return frozenset(out)


def _read_set(plan: RetrievalPlan, pool: Sequence[NoteLabel],
              vocabulary: frozenset[str]) -> set[tuple[str, str]]:
    """Which notes retrieval would actually read, exactly as `coverage.py` runs it: exhaustive
    strata read everything, searched strata read their keyword hits, sampled strata read nothing
    — their sample is a validation cost, and counting it here would flatter every plan that
    dumps document types into them."""
    buckets: dict[str, list[NoteLabel]] = {}
    for n in pool:
        buckets.setdefault(plan.stratum_of(n.doc_type), []).append(n)
    read: set[tuple[str, str]] = set()
    for stratum, group in buckets.items():
        how = plan.reads_for(stratum)
        if how == READ_ALL:
            read |= {_key(n) for n in group}
        elif how == READ_HITS:
            hits = _expand(plan.keywords_for(stratum), vocabulary)
            read |= {_key(n) for n in group if n.terms & hits}
    return read


def measure(plan: RetrievalPlan, labelling: Labelling, split: Split, role: str = DEV,
            *, notes: Sequence[NoteLabel] | None = None) -> Metrics:
    """Replay a plan against the labels — recall, precision, notes read per patient, patients
    losing the answer. No model, no corpus, no network.

    A note is answer-bearing if it establishes the field or mentions it, since a mention is
    enough to break an absence proof. `patients_losing_the_answer` is chart-wide on purpose: a
    patient whose pathology was missed but whose progress note was read has not lost the answer,
    and a patient whose chart never held it is in no denominator at all.
    """
    pool = list(notes if notes is not None else scope(labelling, split, role))
    fld = plan.field_name
    read = _read_set(plan, pool, labelling.vocabulary)
    bearing = [n for n in pool if fld in n.establishes or fld in n.mentions]
    with_answer = {n.patient_id for n in pool if fld in n.establishes}
    kept = {n.patient_id for n in pool if fld in n.establishes and _key(n) in read}
    return Metrics(len({n.patient_id for n in pool}), len(pool), len(read), len(bearing),
                   sum(1 for n in bearing if _key(n) in read), len(with_answer),
                   len(with_answer - kept))


def objective(m: Metrics, cost_weight: float) -> float:
    """The scalar the hill-climb orders moves by, NOT a reported metric. Answer coverage minus
    the share of the corpus read to get it, so a term that buys one answer by doubling the reads
    is scored as what it is."""
    return m.answer_coverage - cost_weight * m.read_share


# --------------------------------------------------------------------- propose and evolve
def propose(plan: RetrievalPlan, labelling: Labelling, split: Split, *, kind: AssetKind,
            stratum: str | None = None, k: int = 8) -> list[Candidate]:
    """Generate candidates from the DEV labels alone; `propose` invents, `measure` judges. There
    is no `role` argument: proposing against the test half would make the candidates a function
    of the held-out data, and no certification against it would then mean anything."""
    pool = scope(labelling, split, DEV)
    if not pool:
        raise AssetDevelopmentError("no labelled notes for the dev half of this split")
    if kind == STRATA:
        return _propose_strata(plan, pool, k)
    return _propose_keywords(plan, labelling, pool, stratum, k)


def _propose_keywords(plan: RetrievalPlan, labelling: Labelling, pool: Sequence[NoteLabel],
                      stratum: str | None, k: int) -> list[Candidate]:
    """ADD a term, DROP a term, STEM a term — the whole move set.

    Adds come only from answer-bearing notes the current plan does NOT surface, ranked by how
    many of those they appear in rather than by recall: ranking by recall alone is how a list
    gets "improved" with a term like "patient", which surfaces every note ever written and every
    answer with them. Drop and stem let the search give ground back; without them a hill-climb
    can only grow the list.
    """
    target = stratum or plan.searched_stratum
    current = list(plan.keywords_for(target))
    fld, element = plan.field_name, plan.element(target, "required_keywords")
    group = [n for n in pool if plan.stratum_of(n.doc_type) == target]
    read = _read_set(plan, pool, labelling.vocabulary)

    def cand(value: Sequence[str], why: str) -> Candidate:
        return Candidate(KEYWORDS, fld, target, element, tuple(value), why)

    df: dict[str, int] = {}
    for n in group:
        if _key(n) not in read and (fld in n.establishes or fld in n.mentions):
            for t in n.terms & labelling.vocabulary:
                df[t] = df.get(t, 0) + 1
    adds = sorted((t for t in df if t not in current), key=lambda t: (-df[t], t))[:k]

    out = [cand(current + [t], f"add {t!r}: in {df[t]} unsurfaced answer-bearing dev note(s)")
           for t in adds]
    out += [cand([x for x in current if x != t], f"drop {t!r}: is it buying its reads?")
            for t in current]
    out += [cand([t[:-2] if x == t else x for x in current],
                 f"stem {t!r} -> {t[:-2]!r}: reach wordings nobody anticipated")
            for t in current if len(t) >= 6 and t[:-2] not in current]
    return out


def _propose_strata(plan: RetrievalPlan, pool: Sequence[NoteLabel], k: int) -> list[Candidate]:
    """Where the answers actually live, by document type. The labels say exactly one thing: this
    document type DOES carry the answer, or it never does. Whether it MAY establish the answer is
    a different sentence in a different language, which is why every candidate returned here is
    semantic and can only ever become a proposal."""
    fld = plan.field_name
    by_type: dict[str, list[NoteLabel]] = {}
    for n in pool:
        by_type.setdefault(n.doc_type, []).append(n)

    scored: list[tuple[float, Candidate]] = []
    for doc_type, notes in sorted(by_type.items()):
        here, n = plan.stratum_of(doc_type), len(notes)
        est = [x for x in notes if fld in x.establishes]
        pats = len({x.patient_id for x in est})
        if est and plan.reads_for(here) != READ_ALL:
            move, rank = "can_establish", _ratio(len(est), n) * pats
        elif not est and plan.reads_for(here) == READ_ALL and n >= 3:
            move, rank = "may_mention", float(n)
        else:
            continue
        scored.append((rank, Candidate(
            STRATA, fld, move, plan.element(move, "match"), ((doc_type, move),),
            f"{doc_type} sits in {here!r} and establishes the {fld} in {len(est)}/{n} dev note(s) "
            f"for {pats} patient(s). The labels say where the information is; only a clinician "
            f"can say whether this document type may establish it.")))
    scored.sort(key=lambda r: (-r[0], r[1].label))
    return [c for _, c in scored[:k]]


@dataclass(frozen=True)
class Evolution:
    """The dev-only search. WRITES NOTHING — not the spec, not a cache, not a log. Everything it
    learned is in this object, and turning that into a change takes `certify` and then `adopt`,
    both of which can refuse. A development plane that edits the artefact it is developing has
    quietly become an unreviewed author."""
    plan: RetrievalPlan
    labelling_hash: str
    split_hash: str
    baseline: Metrics
    final: Metrics
    candidate: Candidate | None
    log: tuple[str, ...]
    #: The plan the climb STARTED from and the exact arguments it ran with. Kept because the
    #: negative control has to rerun THIS search — same start, same move set, same cost weight —
    #: against permuted labels; a control that reruns a slightly different search compares two
    #: things that were never the same experiment.
    start_plan: RetrievalPlan
    settings: tuple[tuple[str, Any], ...]


def evolve(plan: RetrievalPlan, labelling: Labelling, split: Split, *, kind: AssetKind,
           stratum: str | None = None, rounds: int = 4, k: int = 8,
           cost_weight: float = 0.25) -> Evolution:
    """Greedy hill-climb on DEV ONLY, writing nothing.

    A move is accepted only if it improves the objective — which already prices the extra reads —
    AND does not lose a patient an answer they previously had. The two conditions are separate
    because they are different kinds of bad: there is no exchange rate at which a search may hand
    back an answer it already had.
    """
    dev = scope(labelling, split, DEV)
    if not dev:
        raise AssetDevelopmentError("no labelled notes for the dev half of this split")
    base = measure(plan, labelling, split, DEV, notes=dev)
    best_m, best_plan, best_o = base, plan, objective(base, cost_weight)
    best_c, log, accepted = None, [], 0

    for _ in range(max(0, rounds)):
        win: tuple[float, Candidate, Metrics] | None = None
        for c in propose(best_plan, labelling, split, kind=kind, stratum=stratum, k=k):
            m = measure(c.apply(best_plan), labelling, split, DEV, notes=dev)
            o = objective(m, cost_weight)
            if m.patients_losing_the_answer > best_m.patients_losing_the_answer:
                log.append(f"reject {c.rationale} — loses an answer already held")
            elif o <= best_o:
                log.append(f"reject {c.rationale} — costs more than the coverage it buys")
            elif win is None or o > win[0]:
                win = (o, c, m)
        if win is None:
            break
        best_o, best_c, best_m = win
        best_plan, accepted = best_c.apply(best_plan), accepted + 1
        log.append(f"ACCEPT {best_c.rationale} — objective {best_o:+.4f}")

    # The whole search as ONE candidate against the starting plan: a sequence of accepted moves
    # is a story, and the spec takes a value.
    if best_c is not None and kind == KEYWORDS:
        best_c = replace(best_c, value=tuple(best_plan.keywords_for(best_c.stratum)),
                         rationale=f"{accepted} accepted move(s) on dev, objective {best_o:+.4f}")
    return Evolution(best_plan, labelling.hash, split.split_hash, base, best_m, best_c,
                     tuple(log), plan, (("kind", kind), ("stratum", stratum), ("rounds", rounds),
                                        ("k", k), ("cost_weight", cost_weight)))


# --------------------------------------------------------------------------------- certify
# THE NEGATIVE CONTROL, AND THE RULE IT IS JUDGED BY — declared here, in numbers, because a
# threshold nobody can find is a threshold nobody can argue with. Hill-climbing on a dev set
# ALWAYS improves the dev objective; that is what hill-climbing is. So the only question a
# certification can answer is whether the identical search would have gained as much on labels
# that carry no signal. `certify` reruns it against CONTROL_SHUFFLES permuted labellings and
# compares held-out gains.
#
# WHY BEAT THE MAXIMUM OF 19: beating every one of 19 permutations is an exact one-sided
# permutation test at level 1/(19+1) = 0.05. Nineteen is the SMALLEST number of reruns that can
# reach the conventional threshold, and each rerun is a whole evolution, so more is not free.
# WHY A MARGIN AND NOT A BARE `>`: two evolutions on finite data always differ by something, and
# a rule that certifies a hairline win certifies noise. 0.02 objective points is two patients in
# a hundred keeping their answer, or an eight-point swing in the share of the corpus read — the
# smallest difference anyone would act on.
# WHY ALSO `> 0`: if every shuffle came out negative, a real gain of -0.01 would clear the
# margin while still being a search that made the held-out half worse.
CONTROL_SHUFFLES = 19
CONTROL_MARGIN = 0.02
CONTROL_RULE = (f"the real held-out gain must be positive AND exceed every one of "
                f"{CONTROL_SHUFFLES} shuffled-label gains by at least {CONTROL_MARGIN} objective "
                f"points (an exact one-sided permutation test at level 1/{CONTROL_SHUFFLES + 1}, "
                f"plus a margin so a hairline win cannot certify)")


@dataclass(frozen=True)
class NegativeControl:
    """What the same search achieved once the patient-answer link was permuted away.

    Part of every `Certification` and of every provenance record it writes: the rule, the margin
    and the seeds are all stored, so a reader can see exactly what was required and rerun it.
    """
    rule: str
    margin: float
    seeds: tuple[int, ...]
    real_gain: float
    shuffled_gains: tuple[float, ...]

    shuffled_max = property(lambda s: max(s.shuffled_gains))
    shuffled_mean = property(lambda s: sum(s.shuffled_gains) / len(s.shuffled_gains))

    @property
    def passed(self) -> bool:
        return self.real_gain > 0.0 and self.real_gain > self.shuffled_max + self.margin

    def __str__(self) -> str:
        return (f"real held-out gain {self.real_gain:+.4f} against {len(self.shuffled_gains)} "
                f"shuffled-label reruns of the same search (max {self.shuffled_max:+.4f}, mean "
                f"{self.shuffled_mean:+.4f}); rule: {self.rule}")

    def to_dict(self) -> dict:
        return {**asdict(self), "seeds": list(self.seeds),
                "shuffled_gains": list(self.shuffled_gains), "passed": self.passed,
                "shuffled_max": round(self.shuffled_max, 6),
                "shuffled_mean": round(self.shuffled_mean, 6)}


def _shuffle(labelling: Labelling, seed: int) -> Labelling:
    """Permute WHICH PATIENT OWNS WHICH ANSWER, and nothing else.

    Every note stays exactly where it is — same patient, same note id, same doc type, same terms
    — and only the (establishes, mentions) pair moves. It moves only to another note OF THE SAME
    DOC TYPE, which is what keeps this a control rather than a demolition: per-patient note
    counts are untouched, the multiset of answers is untouched, and so is the marginal
    distribution of answers over document types. Permuting answers between document types
    instead would leave the incumbent plan measuring a corpus that no plan is any good on, every
    candidate would look like a discovery against that, and the control would pass every time.

    The labelling's `hash` is deliberately unchanged: it identifies the scan — model, prompt,
    spec — not the labels, so the rerun still trips the split/evolution hash guard if it is on
    the wrong split.
    """
    by_type: dict[str, list[int]] = {}
    for i, n in enumerate(labelling.notes):
        by_type.setdefault(n.doc_type, []).append(i)
    notes, rng = list(labelling.notes), random.Random(seed)
    for _, idx in sorted(by_type.items()):
        answers = [(notes[i].establishes, notes[i].mentions) for i in idx]
        rng.shuffle(answers)
        for i, (est, men) in zip(idx, answers):
            notes[i] = replace(notes[i], establishes=est, mentions=men)
    return replace(labelling, notes=tuple(notes))


def _control_seeds(labelling: Labelling, split: Split) -> tuple[int, ...]:
    """Derived from what is being certified, and accepted from nobody: a seed a caller can pass
    is a seed a caller can shop for, and a control that can be reseeded until it passes is a
    report section. Stored in the Certification so the run is reproducible anyway."""
    return tuple(int(_hash("negative-control", labelling.hash, split.split_hash, i), 16) % 2**31
                 for i in range(CONTROL_SHUFFLES))


@dataclass(frozen=True)
class Certification:
    """A held-out number that may be quoted, and the arrangement it is only true under."""
    candidate: Candidate
    labelling_hash: str
    split_hash: str
    model: str
    certified_on: str
    dev: Metrics
    test: Metrics
    #: NOT OPTIONAL. `certify` is the only thing that builds a Certification and it refuses to
    #: build one whose control did not pass, so the existence of this object is the claim that
    #: the gain outran permuted labels.
    control: NegativeControl
    spec_id: str = ""

    @property
    def verdict(self) -> str:
        """`supports`, `underpowered` or `falsified`, in `spec.MeasuredVerdict`'s vocabulary. A
        candidate the held-out half found worse stays `draft`: the measurement is then the reason
        to distrust the element, and it must not rank above one nobody has looked at."""
        if (self.test.patients_losing_the_answer > self.dev.patients_losing_the_answer
                and self.test.answer_coverage < self.dev.answer_coverage):
            return "falsified"
        return "underpowered" if self.test.n_patients < MIN_PATIENTS_FOR_SUPPORT else "supports"

    def to_dict(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if k != "control"}
        return {**d, "kind": "acr.improvement.assetdev.certification/1", "verdict": self.verdict,
                "adoption_rule": ADOPTION_RULE[self.candidate.kind],
                "dev": self.dev.to_dict(), "test": self.test.to_dict(),
                "negative_control": self.control.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> Certification:
        c = dict(d["candidate"])
        c["value"] = tuple(tuple(v) if c["kind"] == STRATA else v for v in c["value"])
        keep = {f.name for f in fields(Metrics)}
        dev, test = ({k: v for k, v in d[h].items() if k in keep} for h in ("dev", "test"))
        n = d["negative_control"]
        ctl = NegativeControl(n["rule"], float(n["margin"]), tuple(n["seeds"]),
                              float(n["real_gain"]), tuple(n["shuffled_gains"]))
        return cls(Candidate(**c), d["labelling_hash"], d["split_hash"], d["model"],
                   d["certified_on"], Metrics(**dev), Metrics(**test), ctl, d.get("spec_id", ""))

    def provenance_record(self, run: str = "") -> ProvenanceRecord:
        """The `spec.ProvenanceRecord` that must land in the SAME write as the value. origin
        stays `corpus_derived` whatever the numbers say — that is a fact about where the content
        came from, not a verdict on it — while `status` is the verdict's business, and
        `spec._validate_record` refuses `measured` unless the verdict is `supports`."""
        v, t = self.verdict, self.test
        return ProvenanceRecord(
            element=self.candidate.element, origin="corpus_derived",
            status="measured" if v == "supports" else "draft",
            basis=(f"developed on the dev half of split {self.split_hash}, certified on its "
                   f"held-out test half against the complete per-note labelling "
                   f"{self.labelling_hash} from {self.model}: {t.n_patients} patients, "
                   f"{t.n_notes} notes, recall {t.recall:.3f}, precision {t.precision:.3f}, "
                   f"{t.notes_read_per_patient:.1f} notes read per patient, "
                   f"{t.patients_losing_the_answer}/{t.patients_with_an_answer} patients with an "
                   f"answer lose it. Controlled against shuffled labels: {self.control}."),
            measured={"run": run or f"assetdev:{self.split_hash}", "verdict": v,
                      "n_patients": t.n_patients, "recall": round(t.recall, 4),
                      "precision": round(t.precision, 4),
                      "notes_read_per_patient": round(t.notes_read_per_patient, 3),
                      "patients_losing_the_answer": t.patients_losing_the_answer,
                      "labelling_hash": self.labelling_hash, "split_hash": self.split_hash,
                      "negative_control": "passed", "control_rule": self.control.rule,
                      "control_real_gain": round(self.control.real_gain, 4),
                      "control_shuffled_max": round(self.control.shuffled_max, 4),
                      "control_seeds": list(self.control.seeds)})


def _guards(evolution: Evolution, labelling: Labelling, split: Split, on: str) -> list[NoteLabel]:
    """Every refusal standing between a search and a held-out number, AND THE ONLY WAY IN.

    Factored out of `certify` for one reason: the negative control's reruns are scored through
    this same function, on this same split, for this same half. A control reaching the test data
    down a second, unguarded path would be certifying precisely the arrangements these refusals
    exist to catch. Returns the held-out notes both of them then score against.
    """
    if not isinstance(split, Split) or not split.path:
        raise AssetDevelopmentError("certify() needs a Split written to disk: a certification "
                                    "quotes a split_hash the reader is meant to go and check.")
    if split.overlap:
        raise SplitLeakError(f"{len(split.overlap)} of {len(split.test)} test patients are also "
                             f"in the dev half of split {split.split_hash}; the search saw them, "
                             "so this would be partly a training score.")
    if on != TEST:
        raise ScoredOnDevError(f"refusing to certify on {on!r}: evolve() hill-climbed on the dev "
                               "half, so a number measured there is the search's training score. "
                               "It estimates nothing and it is always better.")
    if (evolution.split_hash, evolution.labelling_hash) != (split.split_hash, labelling.hash):
        raise AssetDevelopmentError(f"the evolution ran against split {evolution.split_hash} / "
                                    f"labelling {evolution.labelling_hash}, not "
                                    f"{split.split_hash} / {labelling.hash}. Rerun evolve().")
    test_notes = scope(labelling, split, TEST)
    if not test_notes:
        raise AssetDevelopmentError(f"the labelling holds no notes for any of the "
                                    f"{len(split.test)} test patients: scan the test half first.")
    return test_notes


def _held_out_gain(evolution: Evolution, labelling: Labelling, split: Split,
                   test_notes: Sequence[NoteLabel]) -> float:
    """What the search actually bought ON THE HELD-OUT HALF: the objective of the plan it ended
    with, minus the objective of the plan it started from, priced at the search's own cost
    weight. The dev gain is not a candidate for this comparison — it is the thing being doubted.
    """
    w = float(dict(evolution.settings)["cost_weight"])
    return (objective(measure(evolution.plan, labelling, split, TEST, notes=test_notes), w)
            - objective(measure(evolution.start_plan, labelling, split, TEST, notes=test_notes), w))


def _negative_control(evolution: Evolution, labelling: Labelling, split: Split, on: str,
                      test_notes: Sequence[NoteLabel]) -> NegativeControl:
    """Rerun the identical evolution against labellings whose patient-answer link is permuted
    away, and collect what each one gained on the same held-out half."""
    gains = []
    for seed in (seeds := _control_seeds(labelling, split)):
        shuffled = _shuffle(labelling, seed)
        rerun = evolve(evolution.start_plan, shuffled, split, **dict(evolution.settings))
        gains.append(_held_out_gain(rerun, shuffled, split,
                                    _guards(rerun, shuffled, split, on)))
    return NegativeControl(CONTROL_RULE, CONTROL_MARGIN, seeds,
                           _held_out_gain(evolution, labelling, split, test_notes), tuple(gains))


def certify(evolution: Evolution, labelling: Labelling, split: Split, *, model: str,
            on: str = TEST, spec_id: str = "", today: str | None = None,
            gold_values: dict[str, list] | None = None) -> Certification:
    """Turn a dev-set search into a number that may be quoted, or refuse with the reason. None of
    these refusals is a warning: a warning about a leaked split is read once and then becomes
    part of the output format.

    THE NEGATIVE CONTROL IS A PRECONDITION, not a section of the output, and there is no argument
    that turns it off. A hill-climb improves its dev objective by construction; whether that is
    signal or the search fitting this corpus's own structure is answered by rerunning the same
    search on permuted labels and comparing held-out gains. A caller who wants numbers without
    that comparison is not certifying anything, and `measure()` will give them the numbers.
    """
    # BEFORE the negative control, because it is the one failure that control cannot see: a term
    # that IS the answer survives permutation honestly. Optional only so that a caller without a
    # key can still certify; supplying one and having it leak is a refusal, not a note.
    if gold_values:
        from .answer_leak import leaking_terms
        # `keywords` is (stratum, terms) pairs, not a flat list — flattened here, because
        # passing the pairs would compare tuples against gold values and silently find
        # nothing, which is the shape of a check that cannot fail.
        terms = [t for _, group in evolution.final.keywords for t in group]
        leaks = leaking_terms(terms=terms, gold_values=gold_values)
        if leaks:
            raise AnswerLeaked(
                "derived terms render development answers, so their dev-set gain does not "
                "transfer: " + "; ".join(str(x) for x in leaks[:5])
                + (f" (+{len(leaks) - 5} more)" if len(leaks) > 5 else ""))

    test_notes = _guards(evolution, labelling, split, on)
    if evolution.candidate is None:
        raise AssetDevelopmentError("the evolution accepted no move, so there is nothing to "
                                    "certify. The incumbent asset stands — a real and reportable "
                                    "outcome. Record it; do not certify an empty change.")
    control = _negative_control(evolution, labelling, split, on, test_notes)
    if not control.passed:
        raise NegativeControlFailed(
            f"NEGATIVE CONTROL FAILED for {evolution.candidate.label}: {control}. The same search "
            f"on permuted labels gained {control.shuffled_max:+.4f} at best and "
            f"{control.shuffled_mean:+.4f} on average, against {control.real_gain:+.4f} for real "
            f"— which is this corpus's own structure being fitted, not a property of retrieval, "
            f"and no recall on the held-out half would have shown it. Nothing here may be "
            f"quoted, and there is no flag that makes it quotable. Shuffle seeds "
            f"{list(control.seeds)}; all gains {[round(g, 4) for g in control.shuffled_gains]}.")
    return Certification(evolution.candidate, labelling.hash, split.split_hash, model,
                         today or date.today().isoformat(), evolution.final,
                         measure(evolution.plan, labelling, split, TEST, notes=test_notes),
                         control, spec_id)


# ----------------------------------------------------------------------------------- adopt
@dataclass(frozen=True)
class Adoption:
    """What actually happened. `outcome` is load-bearing: `adopted` means the spec now holds the
    value and its provenance; `proposal_emitted` means the spec was not touched and a clinician
    has something to sign."""
    outcome: str
    kind: AssetKind
    element: str
    spec_path: str
    adopted_on: str
    rule: str
    proposal_path: str | None = None


def adopt(cert: Certification, spec_path: str | Path, *, proposals_dir: str | Path | None = None,
          today: str | None = None, run: str = "") -> Adoption:
    """The ONLY writer in this module, and it BRANCHES ON ASSET KIND.

    KEYWORDS are RETRIEVAL_ONLY — they decide which text reaches the agent and nothing about what
    an answer means — so a certified list is written here, value and provenance in one atomic
    operation. STRATA are SEMANTIC: a stratum says what MAY ESTABLISH a value. The measurement
    can show that pathology addenda carry the histology; it cannot decide they are admissible. So
    a stratum change becomes a PROPOSAL and the spec is left alone, with no flag to override it.
    """
    stamp, kind = today or date.today().isoformat(), cert.candidate.kind
    if kind == STRATA:
        path = _emit_proposal(cert, spec_path, proposals_dir, stamp)
        return Adoption("proposal_emitted", STRATA, cert.candidate.element, str(spec_path), stamp,
                        ADOPTION_RULE[STRATA], str(path))
    if kind != KEYWORDS:  # pragma: no cover - AssetKind is closed
        raise AssetDevelopmentError(f"unknown asset kind {kind!r}")
    _adopt_keywords(cert, Path(spec_path), run=run)
    return Adoption("adopted", KEYWORDS, cert.candidate.element, str(spec_path), stamp,
                    ADOPTION_RULE[KEYWORDS])


def _adopt_keywords(cert: Certification, spec_path: Path, *, run: str = "") -> None:
    """Value and provenance, or neither.

    The failure this prevents has already happened here: a value edited into a spec while its
    provenance record still describes the previous one — the spec loads, the element reads as
    measured, and the record is about something else. So the write goes to a temp file beside the
    spec, is reloaded through `spec.load_spec`, is verified, and only then does `os.replace` make
    it visible.
    """
    doc = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise AdoptionAborted(f"{spec_path} does not parse as a mapping")
    value = [str(v) for v in cert.candidate.value]
    _set_keywords(doc, cert.candidate.element, value, spec_path)
    _upsert_provenance(doc, cert.provenance_record(run=run))
    tmp = spec_path.with_name(f".{spec_path.name}.adopt-{os.getpid()}.tmp")
    try:
        tmp.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, width=96),
                       encoding="utf-8")
        try:
            reloaded = load_spec(tmp)
        except Exception as e:
            raise AdoptionAborted(f"the adopted spec does not load, so nothing was written to "
                                  f"{spec_path}: {type(e).__name__}: {e}") from e
        _verify(reloaded, cert.candidate.element, value)
        os.replace(tmp, spec_path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _verify(spec: Any, element: str, value: list[str]) -> None:
    """`load_spec` passing is not enough: an element that already had a provenance record loads
    perfectly well with a new value and a record describing the old one, and that state is
    invisible to every other check in the repo. `bind_provenance` sets `element_hash` from the
    element as it now stands, so one comparison proves both that the value landed and that the
    record is about it."""
    from ..contract.spec import _content_hash

    rec = spec.provenance_index.get(element)
    if rec is None:
        raise AdoptionAborted(f"{element} has no provenance record after the write; a value "
                              "without provenance reads as an invented one.")
    if rec.element_hash != _content_hash(value):
        raise AdoptionAborted(f"{element} hashes {rec.element_hash} in the spec but {value!r} "
                              f"hashes {_content_hash(value)}: the record and the value on disk "
                              "describe different things.")


_ELEMENT_RE = re.compile(r"^proof_obligation\.for_negative(?:\.claims\[(?P<claim>[^\]]+)\])?"
                         r"\.strata\[(?P<stratum>[^\]]+)\]\.required_keywords$")


def _set_keywords(doc: dict, element: str, value: list[str], where: Path) -> None:
    m = _ELEMENT_RE.match(element)
    if not m:
        raise AdoptionAborted(f"{element!r} is not a keyword element this module can write; "
                              "expected ...for_negative[.claims[<id>]].strata[<n>]"
                              ".required_keywords")
    holder = (doc.get("proof_obligation") or {}).get("for_negative") or {}
    if m.group("claim"):
        hits = [c for c in (holder.get("claims") or []) if str(c.get("id")) == m.group("claim")]
        if not hits:
            raise AdoptionAborted(f"{where}: no claim {m.group('claim')!r} for {element}")
        holder = hits[0]
    for s in (holder.get("strata") or []):
        if str(s.get("name")) == m.group("stratum"):
            s["required_keywords"] = list(value)
            return
    raise AdoptionAborted(f"{where}: no stratum {m.group('stratum')!r} to write {element} into; "
                          "adopting into one that does not exist creates a list nothing reads.")


def _upsert_provenance(doc: dict, record: ProvenanceRecord) -> None:
    """One record per element — replace, never append a second. `spec.bind_provenance` raises on
    two records for one element, so appending would produce a spec that does not load."""
    blob = record.model_dump(exclude_none=True,
                             exclude={"element_hash", "element_kind", "sign_off_voided_by_edit"})
    block = doc.setdefault("provenance", [])
    if not isinstance(block, list):
        raise AdoptionAborted("this spec's `provenance` is not a list; refusing to guess")
    for i, existing in enumerate(block):
        if isinstance(existing, dict) and existing.get("element") == record.element:
            block[i] = blob
            return
    block.append(blob)


def _emit_proposal(cert: Certification, spec_path: str | Path, proposals_dir: str | Path | None,
                   stamp: str) -> Path:
    """A stratum change, written where a clinician signs it and nowhere else. Everything the
    measurement found is here, including the numbers that would have justified an automatic
    adoption if this were a keyword list. What is missing is the one thing no measurement
    produces, and the file says so in the first field a reader sees."""
    d = Path(proposals_dir) if proposals_dir else Path(spec_path).parent / "proposals"
    d.mkdir(parents=True, exist_ok=True)
    c = cert.candidate
    path = d / f"{stamp}_{cert.spec_id or Path(spec_path).stem}_{c.stratum}.yaml"
    _write(path, yaml.safe_dump({
        "kind": "acr.improvement.assetdev.stratum_proposal/1",
        "STATUS": "PROPOSED — NOT IN EFFECT. Requires a clinician signature.",
        "why_a_signature_is_required": ADOPTION_RULE[STRATA],
        "the_question_for_the_clinician": (
            f"The measurement below shows WHERE the {c.field_name} is documented. It cannot say "
            f"whether these document types MAY ESTABLISH it. Proposed: {c.label}. Is that "
            "admissible evidence?"),
        "spec": str(spec_path), "element": c.element,
        "proposed_value": [list(v) for v in c.value], "rationale": c.rationale,
        "certification": cert.to_dict(),
        "signature": {"reviewed_by": None, "reviewed_on": None, "decision": None,
                      "note": "accept | reject | accept_with_changes"},
    }, sort_keys=False, allow_unicode=True, width=96))
    return path


def _write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------- CLI
assets_app = typer.Typer(add_completion=False, help=(
    "Develop the retrieval assets — keywords and strata — against a complete per-note labelling "
    "of a small dev set. No model calls, no corpus reads. `evolve` writes nothing; `adopt` is "
    "the only writer."))

_SPEC = typer.Option(..., "--spec", "-s", help="the spec being developed")
_LAB = typer.Option(..., "--labelling", "-l", help="per-note labelling JSON from the scan")
_SPLIT = typer.Option(..., "--split", help="stored split artefact")
_FIELD = typer.Option(..., "--field", "-f", help="which output field retrieval is for")
_KIND = typer.Option(KEYWORDS, "--kind", help=f"{KEYWORDS} | {STRATA}")
_STRAT = typer.Option(None, "--stratum", help="default: the searched stratum")
_WEIGHT = typer.Option(0.25, "--cost-weight", help="price of a read against an answer")


def _inputs(spec: str, field: str, labelling: str, split: str):
    s = load_spec(spec)
    return s, RetrievalPlan.from_spec(s, field), Labelling.load(labelling), Split.load(split)


def _line(name: str, m: Metrics) -> str:
    return (f"{name:16s} patients {m.n_patients:5d}  notes {m.n_notes:6d}  read {m.n_read:6d}  "
            f"recall {m.recall:.3f}  precision {m.precision:.3f}  "
            f"notes/patient {m.notes_read_per_patient:6.2f}  losing the answer "
            f"{m.patients_losing_the_answer}/{m.patients_with_an_answer}")


@assets_app.command("split")
def cmd_split(labelling: str = _LAB, out: str = typer.Option(..., "--out"),
              seed: int = typer.Option(0, "--seed", help="the ONLY place a seed is accepted; "
                                       "everything downstream reads the stored file"),
              test_frac: float = typer.Option(0.4, "--test-frac")):
    """Mint and STORE a patient-level dev/test split: seed, hash and date in one file."""
    s = make_split(Labelling.load(labelling).patient_ids(), seed=seed, test_frac=test_frac,
                   path=out)
    typer.echo(json.dumps(s.to_dict(), indent=1))


@assets_app.command("measure")
def cmd_measure(spec: str = _SPEC, field: str = _FIELD, labelling: str = _LAB,
                split: str = _SPLIT):
    """Score the incumbent plan on both halves, on all four numbers."""
    _, plan, lab, sp = _inputs(spec, field, labelling, split)
    for role in (DEV, TEST):
        typer.echo(_line(role, measure(plan, lab, sp, role)))


@assets_app.command("evolve")
def cmd_evolve(spec: str = _SPEC, field: str = _FIELD, labelling: str = _LAB, split: str = _SPLIT,
               kind: str = _KIND, stratum: str = _STRAT, cost_weight: float = _WEIGHT):
    """Hill-climb on DEV ONLY over add/drop/stem. Writes nothing, on purpose."""
    _, plan, lab, sp = _inputs(spec, field, labelling, split)
    ev = evolve(plan, lab, sp, kind=kind, stratum=stratum, cost_weight=cost_weight)
    typer.echo(_line("dev before", ev.baseline))
    typer.echo(_line("dev after", ev.final))
    for entry in ev.log:
        typer.echo(f"  {entry}")
    typer.echo("candidate: " + (ev.candidate.label if ev.candidate else "none, incumbent stands"))
    typer.echo("evolve wrote nothing; `assets certify` scores this on the held-out half.")


@assets_app.command("certify")
def cmd_certify(spec: str = _SPEC, field: str = _FIELD, labelling: str = _LAB,
                split: str = _SPLIT, kind: str = _KIND, stratum: str = _STRAT,
                cost_weight: float = _WEIGHT,
                out: str = typer.Option(..., "--out", help="certificate JSON to write")):
    """Rerun the dev search, control it against shuffled labels, score it ONCE on the test half."""
    s, plan, lab, sp = _inputs(spec, field, labelling, split)
    ev = evolve(plan, lab, sp, kind=kind, stratum=stratum, cost_weight=cost_weight)
    cert = certify(ev, lab, sp, model=lab.model, spec_id=s.spec_id)
    _write(Path(out), json.dumps(cert.to_dict(), indent=1) + "\n")
    typer.echo(_line("dev", cert.dev))
    typer.echo(_line("TEST (held out)", cert.test))
    typer.echo(f"negative control: {cert.control}")
    typer.echo(f"verdict: {cert.verdict}  ->  {out}")


@assets_app.command("adopt")
def cmd_adopt(spec: str = _SPEC,
              cert: str = typer.Option(..., "--cert", help="certificate from `certify`"),
              proposals_dir: str = typer.Option(None, "--proposals-dir"),
              run: str = typer.Option("", "--run")):
    """Adopt a certified keyword list, or emit a stratum PROPOSAL for a clinician."""
    path = Path(cert).expanduser()
    if not path.is_file():
        # `certify` writes NOTHING when it refuses, and refusing is the common case: the first real
        # certification on this corpus returned a negative held-out gain against nineteen
        # shuffled-label reruns. A missing certificate is that refusal, not a lost file, and a bare
        # FileNotFoundError makes a working pipeline look broken at its last stage.
        raise typer.BadParameter(
            f"no certificate at {path}. `acr assets certify` writes one only when the held-out gain "
            f"is positive AND beats every shuffled-label rerun by the required margin; when it "
            f"refuses it writes nothing and prints why. There is nothing to adopt.")
    c = Certification.from_dict(json.loads(path.read_text(encoding="utf-8")))
    a = adopt(c, spec, proposals_dir=proposals_dir, run=run)
    typer.echo(json.dumps(asdict(a), indent=1))
    if a.outcome == "proposal_emitted":
        typer.echo(f"The spec was NOT touched. {ADOPTION_RULE[STRATA]}")


@assets_app.command("prior")
def cmd_prior(labels: str = typer.Option(..., "--labels", help="labels.jsonl from a completed scan"),
              fields: str = typer.Option(..., "--fields", help="comma list, the variables to fold"),
              asset_id: str = typer.Option(..., "--asset-id",
                                           help="a name a later run can be told it was shown"),
              min_patients: int = typer.Option(..., "--min-patients", min=1,
                                               help="refuse below this. No default: what counts "
                                                    "as enough subjects to generalise from is a "
                                                    "policy choice about this corpus."),
              version: str = typer.Option("1", "--version"),
              corpus: str = typer.Option("", "--corpus",
                                         help="upgrade every term from `proposed_by_reader` (a "
                                              "LOWER BOUND — a capped scan cannot propose a ninth "
                                              "term) to `corpus_matched`, by asking the corpus's "
                                              "own matcher. Slower and the only basis on which one "
                                              "term may be said to beat another."),
              out: str = typer.Option(..., "--out", help="where to write the prior JSON")):
    """Fold a scan into a retrieval prior: which document types carry the answer, which terms find it.

    THE MISSING AGGREGATOR. `acr label scan` produced the raw material and
    `review.document_concepts.experience_block` had rendered the finished shape since the day it was
    written — with no producer and no caller. This is the producer. `acr run --prior` is the caller.

    WHY THIS AND NOT `assets adopt`. `adopt` writes a keyword list into the CONTRACT, which moves
    `spec_hash`, after which `tools/analyze_arms.py` correctly refuses to compare the arm against its
    own baseline — a changed contract is a changed question. A prior delivered as an ASSET leaves the
    contract alone, so `run_ladder --group experience` measures the prior instead of refusing.
    """
    from ..chartstore.corpus import Corpus
    from ..contract.retrieval_prior import RetrievalPriorError
    from .prior import build_prior

    flds = [f.strip() for f in fields.split(",") if f.strip()]
    try:
        prior = build_prior(labels, fields=flds, min_patients=min_patients, asset_id=asset_id,
                            version=version,
                            corpus=Corpus(Path(corpus)) if corpus else None)
    except RetrievalPriorError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(2) from e

    Path(out).expanduser().write_text(
        json.dumps(prior.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    m = prior.measured
    typer.echo(f"{prior.asset_id} v{prior.version} [{prior.status}] "
               f"content_hash={prior.content_hash}")
    typer.echo(f"  measured on {m.n_patients} patient(s), {m.n_notes} note(s), "
               f"model {m.model or '(unrecorded)'}")
    for fp in prior.fields:
        top = ", ".join(f"{d.doc_type} {d.rate:.0%} (n={d.n_scanned})" for d in fp.doc_types[:3])
        typer.echo(f"  {fp.field_name}: {fp.n_answer_bearing} of {fp.n_notes} decided notes can "
                   f"establish it; {len(fp.terms)} term(s)")
        typer.echo(f"      top types: {top or '—'}")
        best = sorted(fp.terms, key=lambda t: (-t.n_surfaced_answer_bearing, t.n_surfaced_other))[:5]
        for t in best:
            r = t.recall(fp.n_answer_bearing)
            typer.echo(f"      {t.term:<24} surfaced {t.n_surfaced_answer_bearing} answer-bearing"
                       f" + {t.n_surfaced_other} other"
                       + (f"  (recall {r:.0%})" if r is not None else "  (recall not measured)"))
        if fp.is_empty:
            # REPORTED, not omitted: "the record is silent about this variable" and "nobody
            # scanned it" are different findings and only one of them is about the corpus.
            typer.echo(f"      !! no scanned note could establish {fp.field_name}. That is a "
                       f"finding about this corpus or this requirement, not a missing measurement.")
    basis = {t.basis for fp in prior.fields for t in fp.terms}
    if basis == {"proposed_by_reader"}:
        typer.echo("  NOTE: term counts are a LOWER BOUND — they count where the reading model "
                   "PROPOSED a term, and a scan capped at N terms per note cannot propose the "
                   "N+1th. Pass --corpus to count where terms actually occur.")
    typer.echo(f"-> {out}")
    typer.echo("Pass it to a run with `acr run --prior <file>`. It does not touch the contract, "
               "so `spec_hash` does not move and the two arms stay comparable.")
