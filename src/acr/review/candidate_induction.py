"""Where candidates come from: extract mechanically, then let the model judge.

WHY THIS LAYER EXISTS, and it is a measurement rather than a preference. Phase A proved two
things at once. The candidate ledger mechanism holds — ten of ten runs populated it, zero
invented ids, zero failed calls, eleven of thirteen candidates grounded in a recorded span. And
a plain LLM reasoner does not reliably enumerate competing values: across thirteen candidates
not one pair was value-against-value, and every time a second candidate appeared it was an
abstention. On SYNY03 — built so three admissible sources give three different dates — it
declared one candidate with four supporting spans and nothing contradicting.

So the question stopped being the schema or the call timing and became: WHERE DO CANDIDATES
COME FROM. The reasoner was carrying two jobs — DISCOVER every value the evidence could support,
and REASON between them — and for a target like a date the first is close to mechanical.

THIS LAYER IS DELIBERATELY OVER-INCLUSIVE. Every value in the evidence that is type-compatible
with the target gets seeded. Pruning is the reasoner's job, and doing it here would make the
pruning invisible: "I saw this date and ruled it out because it is the document's own date" and
"I never considered it" are different facts, and only the first is auditable. Phase A produced
the second one three times without anyone being able to tell.

ABSTENTION IS NOT A CANDIDATE. "Date A versus EVIDENCE_INSUFFICIENT" and "date A versus date B"
are different disagreements, and putting them in one set makes `conflict` mean two things — which
is exactly what happened in Phase A, where all three multi-candidate charts were the first kind.
Answerability lives on the ledger as its own axis.
"""
from __future__ import annotations

import calendar as _calendar
import re
from dataclasses import dataclass, field

from ..core.state import ANSWERABILITY, SEED_SOURCES  # noqa: F401  — the ledger owns both

_MONTHS = {m.lower(): i for i, m in enumerate(_calendar.month_name) if m}
_MONTHS.update({m.lower(): i for i, m in enumerate(_calendar.month_abbr) if m})
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))

#: The notations this corpus actually writes. Ordered longest-first so `2010-05-17` is not
#: consumed as a bare year by the last pattern.
_DATE_PATTERNS = [
    re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),                       # 2010-05-17
    re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b"),                   # 05/17/2010
    re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}}),?\s+(\d{{4}})\b", re.IGNORECASE),   # May 17, 2010
    re.compile(rf"\b(\d{{1,2}})\s+({_MONTH_RE})\.?,?\s+(\d{{4}})\b", re.IGNORECASE),   # 17 May 2010
    re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{4}})\b", re.IGNORECASE),           # May 2010
]

#: A BARE FOUR-DIGIT NUMBER IS NOT A DATE. "cycle 2010", "qty 2010", a note id, a dose — the
#: corpus is full of them, and seeding one is a candidate nobody can rule out because it never
#: meant anything. A year has to be carried by a word that makes it a year.
_BARE_YEAR = re.compile(r"\b(?:in|of|since|during|around|circa|about|year)\s+((?:19|20)\d{2})\b",
                        re.IGNORECASE)


def normalise_date(text: str) -> str | None:
    """One notation -> CCYYMMDD with 99 for what is not stated, or None.

    None rather than a guess, everywhere. A two-digit year could be 2010 or 1910 and nothing in
    a span settles it; a date that does not exist is not a reading anybody holds. Either would
    enter the ledger as a candidate the reasoner then has to spend a rejection on, and would
    charge precision for a value nobody ever meant.
    """
    s = str(text or "").strip()
    for i, pat in enumerate(_DATE_PATTERNS):
        m = pat.search(s)
        if not m:
            continue
        g = m.groups()
        if i == 0:
            y, mo, d = int(g[0]), int(g[1]), int(g[2])
        elif i == 1:
            mo, d, y = int(g[0]), int(g[1]), int(g[2])
        elif i == 2:
            mo, d, y = _MONTHS[g[0].lower()], int(g[1]), int(g[2])
        elif i == 3:
            d, mo, y = int(g[0]), _MONTHS[g[1].lower()], int(g[2])
        else:
            mo, y, d = _MONTHS[g[0].lower()], int(g[1]), None
        if not 1000 <= y <= 9999 or not 1 <= mo <= 12:
            return None
        if d is None:
            return f"{y:04d}{mo:02d}99"
        if not 1 <= d <= _calendar.monthrange(y, mo)[1]:
            return None
        return f"{y:04d}{mo:02d}{d:02d}"
    m = _BARE_YEAR.search(s)
    return f"{int(m.group(1)):04d}9999" if m else None


def _all_dates(text: str) -> list[str]:
    """Every date in the span, in the order written, de-duplicated.

    One sentence often carries two — "cytology 2010-05-17 was suspicious; biopsy 2010-05-22
    confirmed" is the shape three of this corpus's conflict rules are about.
    """
    out: list[str] = []
    s = str(text or "")
    spans: list[tuple[int, int, str]] = []
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(s):
            if any(a < m.end() and m.start() < b for a, b, _ in spans):
                continue                              # already consumed by a longer pattern
            v = normalise_date(m.group(0))
            if v:
                spans.append((m.start(), m.end(), v))
    for m in _BARE_YEAR.finditer(s):
        if not any(a < m.end() and m.start() < b for a, b, _ in spans):
            v = normalise_date(m.group(0))
            if v:
                spans.append((m.start(), m.end(), v))
    for _, _, v in sorted(spans):
        if v not in out:
            out.append(v)
    return out


#: Extractor kinds, keyed the way a contract's fields already declare their value space. A field
#: whose `calendar` this layer does not implement yields nothing, which is the right answer: an
#: ICD-O topography column is not a date, and running a date extractor over it manufactures
#: candidates out of whatever four-digit numbers happen to be nearby.
EXTRACTORS: dict[str, object] = {"partial_date_ccyymmdd": _all_dates}


def target_fields(spec) -> list:
    """The contract's fields this layer knows how to extract for."""
    return [f for f in (getattr(spec, "fields", None) or [])
            if getattr(f, "calendar", None) in EXTRACTORS]


def extract_sources(spec, evidence) -> dict[str, list[tuple[str, str]]]:
    """evidence_id -> [(value, where it came from)]. Deterministic, no model.

    THREE PLACES A VALUE CAN COME FROM, and the third is the one a first version missed:

      SPAN_LITERAL   a date written inside the recorded span
      DOCUMENT_DATE  THE DOCUMENT'S OWN DATE. On a live SYNY03 run the extractor read only the
                     quotes, found one date in four spans, and the model went back to inventing
                     the other two — the exact behaviour this layer exists to stop. In a
                     clinical record the date that dates a diagnosis is usually in the header of
                     the note that states it, not in the sentence.
      EVENT_DATE     when the span says the event happened, where that differs from when the
                     note was written. A retrospective statement is this shape and
                     `decision_rule[2]` turns on it.

    Seeding the document date is over-inclusive on purpose: a note's own date is the single
    thing the reasoner most often has to reject. Recording WHICH source a value came from is
    what lets that rejection be counted separately from a wrong reading.

    INADMISSIBLE spans are skipped; UNJUDGED ones are not. Nobody having ruled on a span is a
    different fact from somebody having ruled it out, and treating the first as the second
    would silently shrink the candidate space before anyone looked at it.
    """
    fields = target_fields(spec)
    if not fields:
        return {}
    out: dict[str, list[tuple[str, str]]] = {}
    for i, e in enumerate(getattr(evidence, "items", []), 1):
        if getattr(e, "admissibility", "UNJUDGED") == "INADMISSIBLE":
            continue
        pairs: list[tuple[str, str]] = []

        def _add(value, where, _pairs=pairs):
            if value and not any(v == value for v, _ in _pairs):
                _pairs.append((value, where))

        for f in fields:
            for v in EXTRACTORS[f.calendar](e.quote):
                _add(v, "SPAN_LITERAL")
        _add(normalise_date(str(getattr(e, "date", "") or "")), "DOCUMENT_DATE")
        _add(normalise_date(str(getattr(e, "event_date", "") or "")), "EVENT_DATE")
        if pairs:
            out[e.evidence_id or f"E{i}"] = pairs
    return out


def extract_values(spec, evidence) -> dict[str, list[str]]:
    """The values only, in source order. `extract_sources` is the attributable form."""
    return {k: [v for v, _ in pairs] for k, pairs in extract_sources(spec, evidence).items()}


@dataclass
class InductionResult:
    """What one seeding pass did, so the metrics do not have to infer it from the ledger."""

    n_seeded: int = 0
    n_new: int = 0
    values: dict = field(default_factory=dict)
    conflict_sets: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"n_seeded": self.n_seeded, "n_new": self.n_new,
                "values": {k: list(v) for k, v in self.values.items()},
                "conflict_sets": list(self.conflict_sets)}


def seed_candidates(ledger, spec, evidence, *, step: int) -> InductionResult:
    """Turn extracted values into candidate objects. Idempotent and additive.

    Seeding NEVER changes a candidate's state. A value the reasoner already rejected must stay
    rejected when the same span is seen again on the next pass, or a second batch of evidence
    would silently revive every reading somebody had already ruled out — and the ledger's
    incremental-stability property is the one thing making it worth reading twice.
    """
    fields = target_fields(spec)
    if not fields:
        return InductionResult()
    per_span = extract_sources(spec, evidence)
    fname = fields[0].name
    res = InductionResult(values={k: [v for v, _ in p] for k, p in per_span.items()})

    by_value: dict[str, list[str]] = {}
    sources: dict[str, set[str]] = {}
    for eid, pairs in per_span.items():
        for v, where in pairs:
            by_value.setdefault(v, []).append(eid)
            sources.setdefault(v, set()).add(where)

    known = {c.candidate_id for c in ledger.candidates}
    for value, eids in by_value.items():
        c = ledger.declare({fname: value}, step=step, seeded_from=eids,
                           seed_sources=sorted(sources[value]),
                           seed_method="evidence_value_extraction")
        res.n_seeded += 1
        res.n_new += c.candidate_id not in known
        # The seeding link is `supports`: the span mentions the value. Whether it ESTABLISHES it
        # is the reasoner's call, and it may rerole the link to `contradicts` — which is
        # recorded as a rerole rather than as a fresh fact.
        for eid in eids:
            if eid not in c.contradicting_evidence_ids:
                ledger.link(c.candidate_id, eid, "supports", step=step)

    if by_value and ledger.answerability == "UNDETERMINED":
        # A MECHANICAL FACT, not a judgement: a value compatible with the target appears in the
        # evidence. Whether an absence is EVIDENCE_INSUFFICIENT or CORPUS_INSUFFICIENT is a
        # judgement, and this layer does not make it — silence stays UNDETERMINED.
        ledger.set_answerability("VALUE_AVAILABLE", step=step,
                                 reason="a target-compatible value appears in the evidence")
    ledger.rebuild_conflict_sets(step=step)
    res.conflict_sets = list(ledger.conflict_sets)
    return res


def controller_input(ledger, *, coverage_facts=None, budget=None) -> dict:
    """WHAT A STRATEGIC CONTROLLER MAY READ. Frozen before the Controller exists.

    Defined now, and deliberately, because the alternative is discovering later that the
    Controller reaches back into the raw evidence and redoes the candidate reasoning itself.
    If it does that, A1.5 stops being an architecture layer and becomes a middle step something
    routes around — and the reasoning that routed around it leaves no record. There is no chart
    here, no document inventory, no span text: only the state this layer is responsible for
    having made reliable.

    LIVE CANDIDATES ONLY, and UNRESOLVED discriminators only. A rejected candidate stays in the
    ledger forever — that is invariant 5, and it is what keeps "considered and ruled out"
    distinguishable from "never thought of" — but it is a settled question, and a Controller
    handed a settled question searches for an answer it already has.
    """
    live = [c for c in ledger.candidates if c.status in ("ACTIVE", "LEADING", "SELECTED")]
    return {
        "active_candidates": [
            {"candidate_id": c.candidate_id, "value": dict(c.value), "status": c.status,
             "supporting_evidence_ids": list(c.supporting_evidence_ids),
             "contradicting_evidence_ids": list(c.contradicting_evidence_ids)}
            for c in live],
        "conflict_sets": list(ledger.conflict_sets),
        "unresolved_discriminators": [d for d in ledger.discriminators
                                      if d.get("status", "UNRESOLVED") == "UNRESOLVED"],
        "answerability": {"status": ledger.answerability},
        "coverage_facts": dict(coverage_facts or {}),
        "remaining_budget": dict(budget or {}),
    }
