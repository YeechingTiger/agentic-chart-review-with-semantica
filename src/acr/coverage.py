"""Coverage: what has to be true before a negative answer is allowed.

Three things live here.

**Stratification.** Documents split into can_establish (this type could settle the question
on its own), may_mention (could carry a passing reference) and cannot_establish (declared
incapable). Exhaustive review of the first is affordable because it is small, and it
contributes exactly zero elusion. The other two are validated by sampling rather than read.

**Elusion.** A Clopper-Pearson upper bound on the relevance rate of whatever went unread.
Note what this is *not* used for: recall. With one to three qualifying documents per
criterion, `k / (k + elusion_upper * unreviewed)` collapses — k=2 over 400 unread documents
with 50 samples gives a recall bound near 0.08, and reaching 0.95 would need a sample larger
than the pool. So recall is reported and never gated. The gate asks a different, answerable
question: did the exclusion declaration and the keyword list survive being sampled?

**Windows.** For coverage claims the exhaustive stratum is partitioned by time, not by type.
Windows are generated from a surveillance schedule, then clipped at both ends by the
observable period, and only then judged. The clipping is not a detail: without it, every
patient lost to follow-up looks like an unbroken run of empty windows, and loss to follow-up
is the normal case. Only an *interior* gap — records either side, nothing between — supports
the inference that care happened somewhere else.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from datetime import date, timedelta
from typing import Any, Iterable, Literal, Sequence

from .corpus import DocMeta, PatientChart

Disposition = Literal[
    "covered", "interior_gap",
    "out_of_scope_before_anchor", "out_of_scope_after_observable_end",
]


# ---------------------------------------------------------------------------- statistics
def clopper_pearson_upper(hits: int, n: int, confidence: float = 0.95) -> float:
    """One-sided upper bound on a binomial rate. 1.0 when nothing was sampled.

    The zero-hit case has a closed form, 1 - alpha**(1/n), which is the one that matters:
    a null-set sample that turns up nothing is the evidence a clean exclusion rests on.
    """
    if n <= 0:
        return 1.0
    if hits <= 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / n)
    if hits >= n:
        return 1.0
    try:
        from scipy.stats import beta  # type: ignore
        return float(beta.ppf(confidence, hits + 1, n - hits))
    except Exception:
        # Bisection on the regularised incomplete beta via the binomial tail; no scipy needed.
        from math import comb
        def tail(p: float) -> float:
            return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, hits + 1))
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if tail(mid) > 1 - confidence:
                lo = mid
            else:
                hi = mid
        return hi


# ---------------------------------------------------------------------------- strata
@dataclass
class StratumSpec:
    name: str
    policy: str
    doc_type_matches: list[str] = field(default_factory=list)
    rest: bool = False
    required_keywords: list[str] = field(default_factory=list)
    min_sample: int = 25
    min_sample_of_misses: int = 25
    max_tolerated_hits: int = 0
    partition_by: str | None = None
    surveillance_schedule: Any = None
    qualifying_doc_types: list[str] = field(default_factory=list)
    empty_window_policy: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "StratumSpec":
        m = d.get("match") or {}
        return cls(
            name=d["name"], policy=d["policy"],
            doc_type_matches=list(m.get("doc_type_matches") or []),
            rest=bool(m.get("rest")),
            required_keywords=list(d.get("required_keywords") or []),
            min_sample=int(d.get("min_sample", 25)),
            min_sample_of_misses=int(d.get("min_sample_of_misses", 25)),
            max_tolerated_hits=int(d.get("max_tolerated_hits", 0)),
            partition_by=d.get("partition_by"),
            surveillance_schedule=d.get("surveillance_schedule"),
            qualifying_doc_types=list(d.get("qualifying_doc_types") or []),
            empty_window_policy=dict(d.get("empty_window_policy") or {}),
        )

    def matches(self, doc: DocMeta) -> bool:
        if self.rest:
            return True
        return any(pat.lower() in doc.doc_type.lower() for pat in self.doc_type_matches)


def assign_strata(docs: Sequence[DocMeta], specs: Sequence[StratumSpec]) -> dict[str, list[DocMeta]]:
    """First match wins; the `rest: true` stratum sweeps up whatever is left."""
    out: dict[str, list[DocMeta]] = {s.name: [] for s in specs}
    ordered = [s for s in specs if not s.rest] + [s for s in specs if s.rest]
    for d in docs:
        for s in ordered:
            if s.matches(d):
                out[s.name].append(d)
                break
    return out


# ---------------------------------------------------------------------------- sampling
class ForcedSampler:
    """Draws validation samples. The agent never chooses these.

    Letting the model pick which unread documents to check is the runtime form of the
    circularity an independent evidence-universe service exists to prevent — it would be
    validating its own judgement with its own judgement. The seed is recorded so an audit
    can reproduce exactly which documents were drawn.
    """

    def __init__(self, seed: int | None = None):
        self.seed = seed if seed is not None else random.randrange(2**31)
        self._rng = random.Random(self.seed)

    def draw(self, pool: Sequence[DocMeta], n: int) -> list[DocMeta]:
        if n >= len(pool):
            return list(pool)
        return self._rng.sample(list(pool), n)


# ---------------------------------------------------------------------------- windows
@dataclass
class Window:
    start: date
    end: date
    disposition: Disposition = "covered"
    qualifying_docs: list[str] = field(default_factory=list)
    n_documents_any_type: int = 0

    def to_dict(self) -> dict:
        return {
            "from": self.start.isoformat(), "to": self.end.isoformat(),
            "covered": self.disposition == "covered",
            "disposition": self.disposition,
            "qualifying_docs": self.qualifying_docs,
            "n_documents_any_type": self.n_documents_any_type,
        }


DEFAULT_SCHEDULE = [{"from_months": 0, "to_months": None, "interval_months": 6}]


def _parse_schedule(schedule: Any) -> list[dict]:
    if schedule in (None, "PLACEHOLDER_REQUIRES_CLINICAL_INPUT"):
        return DEFAULT_SCHEDULE
    out = []
    for seg in schedule:
        out.append({
            "from_months": _months(seg.get("from", "P0M")),
            "to_months": _months(seg["to"]) if seg.get("to") else None,
            "interval_months": _months(seg.get("interval", "P6M")) or 6,
        })
    return out or DEFAULT_SCHEDULE


def _months(iso: str | None) -> int | None:
    if not iso:
        return None
    s = str(iso).upper().lstrip("P")
    if s.endswith("Y"):
        return int(float(s[:-1]) * 12)
    if s.endswith("M"):
        return int(s[:-1])
    return None


def _add_months(d: date, m: int) -> date:
    return d + timedelta(days=int(round(m * 30.4375)))


def enumerate_windows(anchor: date, horizon: date, schedule: Any = None) -> list[Window]:
    segs = _parse_schedule(schedule)
    out: list[Window] = []
    cur = anchor
    guard = 0
    while cur < horizon and guard < 400:
        guard += 1
        elapsed = (cur - anchor).days / 30.4375
        step = 6
        for seg in segs:
            if elapsed >= seg["from_months"] and (seg["to_months"] is None or elapsed < seg["to_months"]):
                step = seg["interval_months"]
                break
        nxt = min(_add_months(cur, step), horizon)
        out.append(Window(cur, nxt))
        cur = nxt
    return out


def clip_and_judge(
    windows: list[Window],
    docs: Sequence[DocMeta],
    *,
    anchor: date,
    observable_start: date | None,
    observable_end: date | None,
    qualifying_doc_types: Sequence[str],
    policy: dict[str, str] | None = None,
) -> list[Window]:
    """Clip by observable period, then classify each surviving window.

    A window only counts as covered if it holds a document that WOULD HAVE CAUGHT the event
    had it occurred — a pharmacy fill does not qualify, a surveillance CT does. This is the
    can_establish idea applied at window granularity.

    Interior is defined structurally: an empty window with qualifying coverage both before
    and after it. Empty windows at the tail are truncation, not gaps.
    """
    policy = policy or {"interior": "reject",
                        "before_anchor": "out_of_scope",
                        "after_observable_end": "out_of_scope"}
    obs_end = observable_end or (max((d.date for d in docs), default=anchor))

    def qualifies(d: DocMeta) -> bool:
        if not qualifying_doc_types:
            return True
        return any(q.lower() in d.doc_type.lower() for q in qualifying_doc_types)

    for w in windows:
        inw = [d for d in docs if w.start <= d.date < w.end]
        w.n_documents_any_type = len(inw)
        w.qualifying_docs = [d.note_id for d in inw if qualifies(d)]

        if w.end <= anchor or (observable_start and w.end <= observable_start):
            w.disposition = "out_of_scope_before_anchor"
        elif w.start > obs_end:
            w.disposition = "out_of_scope_after_observable_end"
        elif w.qualifying_docs:
            w.disposition = "covered"
        else:
            w.disposition = "interior_gap"      # provisional; tail runs are demoted below

    # Demote the trailing run of empty windows: nothing after them means the patient
    # stopped being observed, which is a narrower scope, not a hole in the evidence.
    in_scope = [w for w in windows if not w.disposition.startswith("out_of_scope")]
    for w in reversed(in_scope):
        if w.disposition == "interior_gap":
            w.disposition = "out_of_scope_after_observable_end"
        else:
            break
    return windows


# ---------------------------------------------------------------------------- results
@dataclass
class StratumResult:
    name: str
    N: int
    reviewed: int = 0
    complete: bool = False
    keywords_searched: list[str] = field(default_factory=list)
    hits: int = 0
    hits_read: int = 0
    misses: int = 0
    misses_sampled: int = 0
    miss_sample_hits: int = 0
    keyword_list_validated: bool = False
    declared_types: list[str] = field(default_factory=list)
    sampled: int = 0
    sample_hits: int = 0
    elusion_upper: float = 1.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateResult:
    verdict: str = "FAIL"
    checks: dict[str, bool] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_gate(gate_spec: dict, strata: Sequence[StratumResult],
                  windows: Sequence[Window] | None = None) -> GateResult:
    g = GateResult()
    c, miss = g.checks, g.missing
    by = {s.name: s for s in strata}

    if gate_spec.get("require_can_establish_nonempty") or gate_spec.get("per_claim_can_establish_nonempty"):
        ok = "can_establish" in by and by["can_establish"].N > 0
        c["can_establish_nonempty"] = ok
        if not ok:
            miss.append("no can_establish stratum: stratified exclusion is not a legal mode here")

    if gate_spec.get("exhaustive_strata_complete", True):
        ce = by.get("can_establish")
        ok = ce is None or ce.complete
        c["exhaustive_strata_complete"] = ok
        if not ok and ce:
            miss.append(f"can_establish not fully reviewed ({ce.reviewed}/{ce.N})")

    if gate_spec.get("exclusion_validated", True):
        s = by.get("cannot_establish")
        ok = s is None or (s.sampled >= 1 and s.sample_hits == 0)
        c["exclusion_validated"] = ok
        if not ok and s:
            miss.append(
                f"exclusion not validated (sampled {s.sampled}, hits {s.sample_hits}) — "
                "a hit overturns the declaration; promote that type to may_mention and rerun"
            )

    if gate_spec.get("keyword_list_validated", True):
        s = by.get("may_mention")
        ok = s is None or (s.misses_sampled >= 1 and s.miss_sample_hits == 0)
        c["keyword_list_validated"] = ok
        if not ok and s:
            miss.append(
                f"keyword list not validated (misses sampled {s.misses_sampled}, "
                f"hits {s.miss_sample_hits}) — extend the keywords and re-search this stratum"
            )

    cap = gate_spec.get("max_elusion_upper")
    if cap is not None:
        worst = max((s.elusion_upper for s in strata if s.name != "can_establish"), default=0.0)
        ok = worst <= cap
        c["max_elusion_upper_ok"] = ok
        if not ok:
            miss.append(f"elusion upper bound {worst:.3f} exceeds cap {cap}")

    if windows is not None:
        gaps = [w for w in windows if w.disposition == "interior_gap"]
        c["no_interior_gaps"] = not gaps
        if gaps:
            days = sum((w.end - w.start).days for w in gaps)
            miss.append(
                f"{len(gaps)} interior follow-up gap(s) totalling {days} days — "
                "an empty interior window is an evidence gap, never an absence of events"
            )

    g.verdict = "PASS" if not miss else "FAIL"
    return g


def summarise_windows(windows: Sequence[Window], *, snapshot: date | None = None) -> dict:
    """through_date, the finality call, and the gap list a human would need."""
    covered = [w for w in windows if w.disposition == "covered"]
    gaps = [w for w in windows if w.disposition == "interior_gap"]
    truncated = [w for w in windows if w.disposition == "out_of_scope_after_observable_end"]
    through = max((w.end for w in covered), default=None)

    reasons: list[str] = []
    if truncated:
        reasons.append("OBSERVATION_TRUNCATED")
    lag = (snapshot - through).days if (snapshot and through) else None

    return {
        "through_date": through.isoformat() if through else None,
        "through_date_lag_days": lag,
        "finality": {"value": "Provisional" if reasons else "Final",
                     **({"reason": reasons} if reasons else {})},
        "windows": [w.to_dict() for w in windows],
        "windows_clipped": sum(1 for w in windows if w.disposition.startswith("out_of_scope")),
        "interior_gaps": [[w.start.isoformat(), w.end.isoformat()] for w in gaps],
        "total_uncovered_days": sum((w.end - w.start).days for w in gaps),
    }
