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

A SAMPLE IS TIED TO THE FRAME IT WAS DRAWN FROM, and that is enforced here rather than
assumed. The miss-sampling frame is "this stratum, minus whatever the searches hit", so it
MOVES whenever the term list grows — and a run may grow the term list mid-flight. Measured on
SYN0001 under STORE.400_522_523: 112 misses, 25 clean draws, elusion 0.1129, gate PASS
against a 0.12 cap; one added term turned 20 of those 25 draws into search hits, leaving 5
draws inside a 92-document frame and an earned bound of 0.4507, while the ledger went on
reporting 0.1129 and passing. No verdict had been overturned and nothing had been found — the
population the sample described had simply been replaced underneath it. So `sampling_frame`
is computed once and consulted by both `pending_samples` and `stratum_results`: draws that
left the frame are struck from the bound, counted in `draws_invalidated`, and replaced.

**Windows.** For coverage claims the exhaustive stratum is partitioned by time, not by type.
Windows are generated from a surveillance schedule, then clipped at both ends by the
observable period, and only then judged. The clipping is not a detail: without it, every
patient lost to follow-up looks like an unbroken run of empty windows, and loss to follow-up
is the normal case. Only an *interior* gap — records either side, nothing between — supports
the inference that care happened somewhere else.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import random
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

from ..chartstore.corpus import DocMeta
from ..contract.site_mapping import SiteMapping

# Re-exported, not re-implemented. `StratumSpec` / `assign_strata` / `strata_from_spec` moved to
# `acr.contract.strata` because they are DERIVED FROM THE SPEC and not a retrieval policy: anything that
# wanted spec-shaped strata had to import this runtime module to get them. `CoverageLedger` takes
# them as parameters, so keeping the names importable from here is a natural re-export rather than
# a compatibility shim — and `acr.contract.strata` is the one definition.
from ..contract.strata import StratumSpec, assign_strata, strata_from_spec  # noqa: F401

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
            return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(hits + 1))
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if tail(mid) > 1 - confidence:
                lo = mid
            else:
                hi = mid
        return hi

def unmapped_doc_types(docs: Sequence[DocMeta], specs: Sequence[StratumSpec],
                       mapping: SiteMapping | None) -> list[str]:
    """Type names in this chart that a `means:`-declaring stratification cannot speak for.

    Recorded in the manifest rather than inferred later. A document whose type the mapping
    never saw is not evidence that the document is irrelevant -- it is evidence that the
    mapping is older than the corpus -- and the difference is invisible once the run is over
    and everything has landed in `rest`.
    """
    if mapping is None or not any(s.is_mapped for s in specs):
        return []
    _, unknown = mapping.coverage_of(d.doc_type for d in docs)
    return unknown

# ---------------------------------------------------------------------------- sampling
#: Domain separator for the derived sampling seed. Public on purpose: on the CLI the seed
#: does not need to be unguessable, it needs to be NON-NEGOTIABLE. Set
#: ACR_SAMPLE_SEED_SECRET to make it unguessable as well; the MCP server does exactly that.
SEED_DOMAIN = b"acr.sample_seed/1"

#: Recorded next to every derived seed so a reader can tell a derivation from a draw.
SEED_DERIVED = "derived:hmac(patient,spec_id)"
SEED_CALLER = "caller_supplied"

def derive_sample_seed(patient_id: str, spec_id: str, secret: bytes | None = None) -> int:
    """The sampling seed for one (patient, spec) question, derived rather than drawn.

    `mcp_server` already worked this way and documented why: a caller that can supply a seed
    — or a runtime that draws a fresh random one per invocation — can rerun the same question
    until the validation draw looks convenient, which restores the exact circularity
    `ForcedSampler` exists to prevent. Deriving from (patient, spec_id) means asking the same
    question twice gets the same documents, so there is nothing to shop for.

    The agent front end was drawing `random.randrange(2**31)` for every run that did not pass
    `--seed` (graph.py:289 via ForcedSampler; `batch` and `consistency` never passed one), so
    the same patient and spec sampled different documents on every invocation and no run was
    reproducible. Same construction here as `ChartReviewService._seed_for`, in one place, so
    the two front ends cannot drift apart.
    """
    key = secret if secret is not None else (
        os.environ.get("ACR_SAMPLE_SEED_SECRET", "").encode() or SEED_DOMAIN)
    mac = hmac.new(key, f"{patient_id}|{spec_id}".encode(), hashlib.sha256)
    return int.from_bytes(mac.digest()[:4], "big") % (2**31)

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
def keyword_was_searched(keyword: str, searched_terms: Iterable[str]) -> bool:
    """Did a search that actually ran cover `keyword`?

    Containment in ONE direction: the required keyword must appear inside a term the agent
    searched. The test used to be bidirectional -- `kw.lower() in t or t in kw.lower()` -- and
    the second half is a hole rather than a convenience. `t` is chosen by the caller, so the
    single character "t" is a substring of "pathology", "biopsy", "specimen", "metasta" and
    "final diagnosis" at once: one search discharged an entire required list. The surviving
    direction is the one that means something, because it is the one where the search really
    did cover the term -- searching "invasive ductal carcinoma" does cover "carcinoma", while
    searching "carcinoma" does not establish that anybody looked for "final diagnosis".
    """
    k = (keyword or "").strip().lower()
    if not k:
        return False
    return any(k in (t or "").strip().lower() for t in searched_terms)

@dataclass
class StratumResult:
    name: str
    N: int
    reviewed: int = 0
    complete: bool = False
    keywords_searched: list[str] = field(default_factory=list)
    #: The stratum's declared search obligation, and the part of it nobody ran. Reported so a
    #: reader can tell "the keyword list survived a sample" from "no keyword list was tried".
    required_keywords: list[str] = field(default_factory=list)
    keywords_unsearched: list[str] = field(default_factory=list)
    hits: int = 0
    hits_read: int = 0
    #: Documents the agent's OWN search flagged and then never opened. A hit is removed from
    #: the miss-sampling frame, so an unread one is reviewed by nothing at all.
    hits_unread: list[str] = field(default_factory=list)
    misses: int = 0
    misses_sampled: int = 0
    miss_sample_hits: int = 0
    keyword_list_validated: bool = False
    declared_types: list[str] = field(default_factory=list)
    sampled: int = 0
    sample_hits: int = 0
    elusion_upper: float = 1.0
    establishes: list[str] = field(default_factory=list)
    #: Draws that were inside the sampling frame when they were made and are not any more,
    #: because a later search moved them out of it. Their verdicts are kept — the obligation
    #: to have looked at them is never cancelled — but they are struck from the bound, which
    #: is a statement about the population they have left.
    draws_invalidated: list[str] = field(default_factory=list)
    #: How many fresh draws the CURRENT frame still owes before n is restored. Non-zero after
    #: a frame revision is the run saying, in one number, that it has not earned its bound.
    replacement_draws_required: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

class CoverageLedger:
    """The single record of what the agent actually did.

    Deliberately the only one. An earlier draft kept a flat ledger in `state.py` beside this
    stratified one; two independent accounts of "how much was covered" can disagree, nothing
    would raise when they did, and you would be left with two numbers and no way to choose.
    So this replaces it rather than joining it.

    Written by the toolbox from real tool calls. The agent cannot address it.
    """

    def __init__(self, docs: Sequence[DocMeta], strata_specs: Sequence[StratumSpec],
                 sampler: ForcedSampler | None = None, confidence: float = 0.95,
                 mapping: SiteMapping | None = None):
        self.docs = list(docs)
        self.specs = list(strata_specs)
        self.confidence = confidence
        self.sampler = sampler or ForcedSampler()
        self.mapping = mapping
        self.by_stratum = assign_strata(self.docs, self.specs, mapping) if self.specs else {}
        #: Type names this chart carries that the Site Mapping never classified. Kept on the
        #: ledger so the manifest can report it: these documents are in `rest` because nobody
        #: judged them, which is a different fact from being judged irrelevant.
        self.unmapped_doc_types = unmapped_doc_types(self.docs, self.specs, mapping)
        self.total_documents = len(self.docs)

        self.listed_documents = False
        self.type_summary_seen = False
        self.searched_terms: list[str] = []
        self.read_notes: list[str] = []
        self.read_sections: list[str] = []
        self.doc_types_touched: list[str] = []
        self.search_hit_notes: set[str] = set()
        # Documents the runtime has DRAWN, persisted across calls. Redrawing a fresh sample
        # on every check makes the obligation unsatisfiable: the agent reads some, the next
        # check demands a different 25, and the debt never shrinks.
        self.drawn: dict[str, list[str]] = {}
        self.samples: dict[str, dict[str, bool]] = {}
        # Reported, never gated. See resolve_sample_verdicts.
        self.suspected_recognition_failures: list[dict] = []

    # -- written by the toolbox ---------------------------------------------------
    def note_search(self, term: str, hit_note_ids: Iterable[str] = ()) -> None:
        t = term.strip().lower()
        if t and t not in self.searched_terms:
            self.searched_terms.append(t)
        self.search_hit_notes.update(hit_note_ids)

    def note_read(self, note_id: str, doc_type: str) -> None:
        if note_id not in self.read_notes:
            self.read_notes.append(note_id)
        if doc_type and doc_type not in self.doc_types_touched:
            self.doc_types_touched.append(doc_type)

    # -- forced sampling ----------------------------------------------------------
    def sampling_frame(self, s: StratumSpec) -> tuple[list[DocMeta], int] | None:
        """The population a draw for this stratum is a sample OF, and how many are needed.

        None for the exhaustive policies, which sample nothing.

        Computed in ONE place because two callers need the same answer and used to derive it
        separately: `pending_samples` decided what to draw from the post-search miss
        universe, and `stratum_results` computed the bound from every verdict on record
        regardless of which universe it came from. That disagreement is the whole defect —
        the frame is recomputed on every call, so it silently tracks the term list, while the
        bound inherited a number earned against a frame that no longer exists.
        """
        pool = self.by_stratum.get(s.name, [])
        if s.policy == "validate_by_sampling":
            # Frame = the stratum. Fixed at construction, so no search can move a document
            # out of it and no draw here is ever invalidated.
            return list(pool), s.min_sample
        if s.policy == "search_then_read_hits_and_sample_misses":
            # Frame = the stratum MINUS the search hits, and it moves whenever a term is added.
            return ([d for d in pool if d.note_id not in self.search_hit_notes],
                    s.min_sample_of_misses)
        return None

    def pending_samples(self) -> dict[str, list[DocMeta]]:
        """Documents the runtime will make the agent inspect before a negative is allowed."""
        out: dict[str, list[DocMeta]] = {}
        for s in self.specs:
            frame = self.sampling_frame(s)
            if frame is None:
                continue
            universe, need = frame
            by_id = {d.note_id: d for d in universe}
            drawn = self.drawn.setdefault(s.name, [])
            # SURVIVING, not merely DRAWN. This test used to be `len(drawn) < need`, which
            # counts a draw the frame has since lost: adding a term that turns 20 of 25 draws
            # into search hits left `n_s >= need` true, so the runtime demanded no
            # replacement and the run kept a 25-draw bound over a 5-draw frame. A draw is
            # tied to the frame it came from, so a revision that changes the frame is a
            # revision that owes replacements.
            surviving = [n for n in drawn if n in by_id]
            if len(surviving) < need:
                # Never redraw something already drawn: an inspected document is not fresh
                # evidence about the frame it has left, and a redraw would look like one.
                fresh = [d for d in universe if d.note_id not in set(drawn)]
                drawn.extend(d.note_id for d in self.sampler.draw(fresh, need - len(surviving)))
            outstanding = [by_id[n] for n in drawn
                           if n in by_id and n not in self.samples.get(s.name, {})]
            if outstanding:
                out[s.name] = outstanding
        return out

    def resolve_sample_verdicts(self, cited: set[str], keyword_hits: set[str] | None = None) -> int:
        """Turn "the agent read a drawn document" into a recorded verdict.

        A drawn document that was read and yielded evidence is relevant; one read and unused
        is not. Without this bridge nothing populates `samples`, `pending_samples` recomputes
        the same debt forever, and no amount of work can satisfy the gate.

        WHAT THIS VALIDATES, AND WHAT IT DOES NOT
        -----------------------------------------
        Forced sampling tests the **stratum definition** — whether "documents of this type
        cannot establish the answer" is true. It catches a retrieval failure: a class of
        document the agent would never have looked at.

        It does NOT test the agent's reading comprehension. Because relevance is inferred
        from whether the agent cited the document, an agent that reads a relevant document
        and fails to recognise it registers the same as one that correctly judged it
        irrelevant — and that agent is exactly the one worth catching. Sampling would then be
        confirming the judgement it is supposed to be auditing.

        `keyword_hits` is a partial, independent counterweight: note_ids that matched the
        stratum's keywords without an LLM in the loop. A drawn document that matched but was
        not cited is counted as a SUSPECTED RECOGNITION FAILURE. It is reported and never
        gated, since keyword matching has its own false positives — but it means a run cannot
        report "0 hits" while quietly having walked past something.

        Measuring recognition properly belongs elsewhere: per-operation extraction metrics
        against known answers, not coverage accounting.
        """
        n = 0
        kw = keyword_hits or set()
        read = set(self.read_notes) | {k.split("#")[0] for k in self.read_sections}
        for stratum, ids in self.drawn.items():
            for nid in ids:
                if nid in read and nid not in self.samples.get(stratum, {}):
                    self.record_sample_verdict(stratum, nid, relevant=nid in cited)
                    if nid in kw and nid not in cited:
                        self.suspected_recognition_failures.append({"stratum": stratum, "note_id": nid})
                    n += 1
        return n

    def record_sample_verdict(self, stratum: str, note_id: str, relevant: bool) -> None:
        self.samples.setdefault(stratum, {})[note_id] = relevant

    # -- results ------------------------------------------------------------------
    def stratum_results(self) -> list[StratumResult]:
        out: list[StratumResult] = []
        for s in self.specs:
            pool = self.by_stratum.get(s.name, [])
            r = StratumResult(name=s.name, N=len(pool), establishes=list(s.establishes))
            r.reviewed = sum(1 for d in pool if d.note_id in self.read_notes
                             or any(k.startswith(d.note_id + "#") for k in self.read_sections))
            r.declared_types = sorted({d.doc_type for d in pool})[:12]
            verdicts = self.samples.get(s.name, {})
            # ONLY THE SURVIVING DRAWS. `n_s = len(verdicts)` credited every verdict ever
            # recorded to a bound over whatever the frame happens to be NOW, so an expansion
            # that emptied the frame of 20 of its 25 draws left the reported bound untouched.
            # A verdict is evidence about the population the document was drawn from; once
            # the document is no longer in that population the verdict is not evidence about
            # it. Struck from the bound, kept in `draws_invalidated`, and replaced by
            # `pending_samples`.
            frame = self.sampling_frame(s)
            if frame is None:
                surviving = dict(verdicts)
            else:
                frame_ids = {d.note_id for d in frame[0]}
                surviving = {k: v for k, v in verdicts.items() if k in frame_ids}
                r.draws_invalidated = sorted(k for k in verdicts if k not in frame_ids)
            n_s, n_hit = len(surviving), sum(1 for v in surviving.values() if v)

            if s.policy in ("exhaustive", "exhaustive_until_witness"):
                r.complete = (r.reviewed >= r.N) if s.policy == "exhaustive" else (r.reviewed > 0)
                r.elusion_upper = 0.0 if r.complete else 1.0
            elif s.policy == "search_then_read_hits_and_sample_misses":
                r.keywords_searched = list(self.searched_terms)
                r.required_keywords = list(s.required_keywords)
                r.keywords_unsearched = [k for k in s.required_keywords
                                         if not keyword_was_searched(k, self.searched_terms)]
                hits = [d for d in pool if d.note_id in self.search_hit_notes]
                r.hits = len(hits)
                # Sections count as having read the document, exactly as `reviewed` above
                # counts them; `read_notes` alone would re-demand a document already opened.
                seen = set(self.read_notes) | {k.split("#")[0] for k in self.read_sections}
                r.hits_read = sum(1 for d in hits if d.note_id in seen)
                r.hits_unread = [d.note_id for d in hits if d.note_id not in seen]
                r.misses = len(pool) - r.hits
                r.misses_sampled, r.miss_sample_hits = n_s, n_hit
                # A DOCUMENT IS ONLY A "MISS" RELATIVE TO A SEARCH THAT ACTUALLY RAN.
                #
                # This line used to read `n_s >= s.min_sample_of_misses and n_hit == 0`, and
                # that is the worst inversion the gate has had. Run no searches at all and
                # `search_hit_notes` stays empty, so every document in the stratum is a miss;
                # the sampler draws its 25 from the whole stratum; the agent reads them,
                # cites none, and the ledger announces that the keyword list is validated.
                # Doing LESS work made the gate EASIER to pass, which is precisely backwards
                # for a check whose whole purpose is to price the work.
                #
                # The searches are a precondition of the verdict, not a separate line item
                # somebody might forget to read: with an unsearched required keyword there is
                # no keyword list under test, so there is nothing for a clean sample to
                # validate.
                # `min(..., r.misses)` keeps the obligation satisfiable when the search left
                # fewer misses than the spec's sample size: inspecting all 4 of 4 remaining
                # misses is a census, which is strictly stronger than a sample of 25, and
                # demanding 25 anyway is an obligation no amount of work discharges.
                need = min(s.min_sample_of_misses, r.misses)
                r.replacement_draws_required = max(0, need - n_s)
                # ...AND THE HITS HAVE TO BE READ. The policy is spelled
                # `search_then_READ_HITS_and_sample_misses`, but only the searching and the
                # sampling were ever enforced: `hits_read` was computed here and read by
                # nothing. A hit is excluded from the miss frame, so an unread hit is
                # reviewed by nothing at all -- neither read, nor eligible to be sampled.
                # That makes searching HARDER to fail the more of it you do: every extra
                # search retires more documents from the audited population without anyone
                # opening them. Measured on SYN0002, the one document the required searches
                # flagged was the one document the passing run never read.
                r.keyword_list_validated = (
                    bool(r.required_keywords)
                    and not r.keywords_unsearched
                    and not r.hits_unread
                    and n_s >= need
                    and n_hit == 0)
                r.elusion_upper = clopper_pearson_upper(n_hit, n_s, self.confidence)
            else:
                r.sampled, r.sample_hits = n_s, n_hit
                r.replacement_draws_required = max(0, min(s.min_sample, r.N) - n_s)
                r.elusion_upper = clopper_pearson_upper(n_hit, n_s, self.confidence)
            out.append(r)
        return out

    def to_dict(self) -> dict:
        return {
            "mode": "stratified_exclusion" if self.specs else "unstratified",
            "sample_seed": self.sampler.seed,
            "universe": {"n_documents": self.total_documents,
                         "n_types": len({d.doc_type for d in self.docs})},
            "listed_documents": self.listed_documents,
            "searched_terms": self.searched_terms,
            "n_read": len(self.read_notes),
            "strata": [r.to_dict() for r in self.stratum_results()],
            "suspected_recognition_failures": self.suspected_recognition_failures,
            "sampling_validates": ("stratum definitions (retrieval), NOT the agent's reading "
                                   "comprehension — a 0-hit sample does not mean nothing was "
                                   "missed, only that this document class was not wrongly excluded"),
            "sample_frame_rule": ("a sample is tied to the frame it was drawn from. Adding a "
                                  "term moves the miss frame, so draws that became search "
                                  "hits are struck from the bound (draws_invalidated), the "
                                  "bound is recomputed over the surviving draws, and "
                                  "replacement_draws_required must reach 0 before the gate "
                                  "will accept it"),
        }

    def render(self) -> str:
        lines = [f"documents: {self.total_documents}   listed: {self.listed_documents}",
                 f"searches ({len(self.searched_terms)}): {', '.join(self.searched_terms) or '-'}",
                 f"documents read: {len(self.read_notes)}"]
        for r in self.stratum_results():
            bits = [f"  [{r.name}] N={r.N} reviewed={r.reviewed}"]
            if r.name == "can_establish":
                bits.append(f"complete={r.complete}")
            if r.sampled or r.misses_sampled:
                bits.append(f"sampled={r.sampled or r.misses_sampled} hits={r.sample_hits or r.miss_sample_hits}")
            bits.append(f"elusion<={r.elusion_upper:.3f}")
            if r.draws_invalidated:
                # Never silent. A bound that quietly loosened because the frame moved reads
                # like a bound that was always this loose, and the two need different fixes.
                bits.append(f"frame revised: {len(r.draws_invalidated)} draw(s) left it, "
                            f"{r.replacement_draws_required} replacement(s) owed")
            if r.establishes:
                bits.append("speaks to: " + ", ".join(r.establishes))
            lines.append("  ".join(bits))
        return "\n".join(lines)

# --------------------------------------------------------------- admissibility, written down
#: What `establishes` verdicts mean. Three values, not two, and the third is the point.
ADMITTED = "ADMITTED"
REFUSED = "REFUSED"
UNDECLARED = "UNDECLARED"

def admissibility_for_citations(spec, coverage: CoverageLedger, evidence: Sequence[dict],
                                fields: Sequence[str] = ()) -> list[dict]:
    """Which evidence rule admitted or refused each cited document, per field.

    THE GATE ALREADY KNOWS THIS AND HAS NEVER WRITTEN IT DOWN. Every document is assigned to
    a stratum the moment the ledger is built, and each stratum declares `establishes` — the
    spec's `evidence_rules.does_not_count` in the form code can read ("Radiology can localise
    a mass; it cannot establish histology or behaviour"). So at gate time the admissibility
    of every citation is a computed fact, and it was being discarded. Recovering it after the
    run means re-deriving the stratification from the spec and hoping it matches; recording it
    means the answer to "was the agent allowed to use that document for that field" is in the
    run, not reconstructed from one.

    The FIELD SCOPING is deliberately the answer's coded fields and NOT the evidence item's
    `supports` label. That label is model-authored free text, and scoping by it is the exact
    bug documented in `answer_checks._evidence_for`: three quotes labelled in three different
    styles, two silently dropped, and a check that then validated nothing.

    Three verdicts because `establishes: []` is genuinely ambiguous and must not be resolved
    here by fiat. The runtime's convention (`derive.py`) is that an empty list means the
    stratum speaks for EVERY field; STORE.700_880 declares `establishes: []` on a stratum it
    named `cannot_establish` and filled with EKGs and prescription lists, which means the
    opposite. Reporting UNDECLARED, with the convention the code would apply spelled out
    alongside, is how that authoring fault reaches a reader instead of being averaged away —
    and divergent readings of one passage is exactly the FORM evidence §6b permits.
    """
    want = [f for f in fields if f] or [getattr(f, "name", "") for f in
                                        (getattr(spec, "fields", []) or [])]
    want = [f for f in want if f]
    by_note: dict[str, str] = {}
    for name, docs in (coverage.by_stratum or {}).items():
        for d in docs:
            by_note[d.note_id] = name
    specs = {s.name: s for s in coverage.specs}
    witness = {}
    po = getattr(spec, "proof_obligation", None)
    if po is not None:
        witness = dict(getattr(po, "witness_strata", None) or {})

    out: list[dict] = []
    seen: set[str] = set()
    for item in evidence or []:
        nid = str(item.get("note_id", ""))
        if not nid or nid in seen:
            continue
        seen.add(nid)
        stratum = by_note.get(nid)
        s = specs.get(stratum) if stratum else None
        declared = list(s.establishes) if s else []
        rec: dict = {
            "note_id": nid,
            "doc_type": str(item.get("doc_type", "")),
            "stratum": stratum,
            "rule_id": (f"evidence_rule.stratum.{stratum}.establishes" if stratum
                        else "evidence_rule.stratum.UNMATCHED"),
            "declared_establishes": declared,
            # False on every record today, and it says so rather than being omitted. Nothing
            # in `gate_answer` refuses a citation for coming from a stratum that cannot
            # establish the field; a reader who assumed it did would conclude the ledger had
            # already been filtered and stop looking for the failure that is still in it.
            "enforced_by_gate": False,
            "by_field": {},
        }
        for f in want:
            if not stratum:
                verdict, why = UNDECLARED, "no stratum matched this document"
            elif f in declared:
                verdict, why = ADMITTED, f"stratum {stratum!r} declares it establishes {f}"
            elif declared:
                verdict, why = REFUSED, (f"stratum {stratum!r} establishes {declared} and not "
                                         f"{f}")
            else:
                verdict, why = UNDECLARED, (
                    f"stratum {stratum!r} declares no `establishes` list. The runtime "
                    f"convention reads an empty list as 'every field'; a stratum named for "
                    f"what it cannot establish means the opposite. Ambiguous as written.")
            row = {"verdict": verdict, "why": why}
            if f in witness:
                # The other admissibility rule: `for_positive.witness` names, per field, the
                # strata a citation may come from at all.
                row["witness_rule_id"] = f"evidence_rule.witness.{f}"
                row["witness_strata"] = list(witness[f])
                row["witness_satisfied"] = stratum in set(witness[f])
            rec["by_field"][f] = row
        out.append(rec)
    return out

@dataclass
class GateResult:
    verdict: str = "FAIL"
    checks: dict[str, bool] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    #: The subset of `missing` that NO AMOUNT OF FURTHER WORK IN THIS RUN can discharge.
    #:
    #: Every other entry in `missing` is an instruction: read these documents, run this search,
    #: settle that thread. These are not. A recorded hit in the exclusion sample cannot be
    #: un-hit, and the elusion bound cannot fall once the sampler has stopped drawing. Returning
    #: them as ordinary refusals is what produced the worst run of the 2026-07-28 batch: eight
    #: rejections, the last five identical, the ledgers frozen at [4, 3, 3], and an agent that
    #: escaped through SPEC_INSUFFICIENT -- a status meaning "the specification is inadequate"
    #: -- because nothing let it say "your coverage bar cannot be met on this chart".
    #:
    #: Kept separate from `missing` rather than removed from it: the reader still needs to see
    #: what failed. This says only that asking again will not change it.
    terminal: list[str] = field(default_factory=list)

    #: What the ledger OBSERVED, in the advisory mode that is now the default: the same
    #: sentences `missing` used to carry, addressed to the model as information instead of to
    #: the loop as a refusal. See `evaluate_gate`.
    advisories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

def evaluate_gate(gate_spec: dict, strata: Sequence[StratumResult],
                  windows: Sequence[Window] | None = None,
                  enforce: bool = False) -> GateResult:
    """What the coverage ledger observed, and — only if `enforce` — what it refuses.

    ADVISORY BY DEFAULT, AND THAT IS THE CHANGE. Every sentence this function produces still
    gets produced; `enforce=False` routes it to `advisories` instead of `missing`, so it reaches
    the model as information about its own run and stops being a condition on the answer. The
    counting is unchanged and still deterministic: how many `can_establish` documents exist, how
    many were reviewed, what residual bound the draws earn. Those are facts and the runtime is
    the right thing to compute them.

    "Have I looked at enough of this chart to say that something is absent?" is not a fact. It
    is a clinical judgement about this patient and this question, and the three-arm pilot
    measured what happens when code makes it instead of the model:

      - conditional coverage activated on 7 of 10 cases, completed its proof on 1, and lost 13
        populated field values across 5 cases to reach 2 recovered ones;
      - always-on coverage recovered nothing guideline-only lacked and lost 15 values across 6;
      - across every recorded trace, coverage obligations produced ~150 answer rejections, 27 of
        which refused a tuple that was exactly the registry's answer;
      - one run submitted the conflict-resolving gold answer ten times and was rejected into a
        model-call-limit failure.

    The obligation now lives in `assets/skills/coverage-judgement/SKILL.md` and reaches the model as
    prose it can act on and decline. What it may not do is decline silently: `advisories` is
    recorded in the manifest, so "the reviewer decided the chart was adequately searched" and
    "the reviewer never looked" stay distinguishable after the fact — which was always the real
    requirement, and it never needed a refusal.

    `enforce=True` keeps the old behaviour verbatim for the diagnostic arm.
    COVERAGE_THREE_ARM_PILOT.md already calls always-on coverage a diagnostic arm and not a
    candidate default; this is that sentence expressed in the signature.
    """
    g = GateResult()
    c = g.checks
    # One list to append to, and one decision about where it goes. Writing `miss` twice — once
    # for enforcement and once for advice — is how the two drift until they disagree about what
    # the run observed.
    miss = g.missing if enforce else g.advisories
    # `terminal` says "asking again will not change this", which is a statement ABOUT A REFUSAL.
    # In advisory mode nothing is refused, so there is nothing to declare undischargeable and it
    # stays empty — which also keeps `terminal` a subset of `missing` in both modes, the
    # invariant `test_gate_must_reject` asserts and the reason a reader can trust either list.
    terminal = g.terminal if enforce else []
    by = {s.name: s for s in strata}

    if gate_spec.get("require_can_establish_nonempty") or gate_spec.get("per_claim_can_establish_nonempty"):
        # SPEC-level, not instance-level. This rule exists to catch a specification in which
        # no document type could establish the answer even in principle — a design fault,
        # whose honest output is SPEC_INSUFFICIENT. It must NOT fire merely because this
        # patient has none of those documents: that is the finding, and its honest output is
        # EVIDENCE_INSUFFICIENT. Reading an empty stratum as a missing stratum collapses the
        # two — the same inversion as treating an unknown document type and an absent one as
        # the same empty list.
        ok = "can_establish" in by
        c["can_establish_declared"] = ok
        if not ok:
            miss.append("the spec declares no can_establish stratum — stratified exclusion is "
                        "not a legal mode for this criterion (that is SPEC_INSUFFICIENT, "
                        "not a finding about this patient)")

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
            line = (f"exclusion not validated (sampled {s.sampled}, hits {s.sample_hits}) — "
                    "a hit overturns the declaration; promote that type to may_mention and rerun")
            miss.append(line)
            # TERMINAL when a hit is on the record. `exclusion_validated` requires
            # `sample_hits == 0`, and a document drawn into the exclusion sample that turned out
            # to contain a keyword stays that document: no further reading, searching or
            # replanning makes the count zero. The remedy the line names -- promote the type and
            # RERUN -- is honest about needing a different run. Saying so here is what lets the
            # runtime stop instead of asking again. A shortfall in the DRAW (sampled == 0) is a
            # different thing and stays discharge-able: the sampler will draw.
            if s.sample_hits >= 1:
                terminal.append(line)

    if gate_spec.get("required_keywords_all_searched", True):
        # This key is declared by every stratified spec in the tree and, until now, was read
        # by nothing: `grep required_keywords_all_searched src/acr/coverage.py` returned
        # nothing at all, so the flag documented an obligation the gate never checked.
        unsearched = [(s.name, k) for s in strata for k in s.keywords_unsearched]
        ok = not unsearched
        c["required_keywords_all_searched"] = ok
        if not ok:
            for name, k in unsearched:
                miss.append(f"required search not performed for stratum {name!r}: {k!r}")

    # FRAME REVALIDATION. Deliberately NOT behind a `gate_spec` key: every other check here
    # asks whether enough work was done, and this one asks whether the arithmetic still
    # describes the population it claims to. A spec cannot opt out of that, and one that
    # could would be opting out of the meaning of its own cap. It can only fire on a run that
    # revised its frame after drawing, which is exactly the case that used to pass silently.
    revised = [s for s in strata if s.draws_invalidated and s.replacement_draws_required > 0]
    c["sample_frames_intact"] = not revised
    frame_reported = {s.name for s in revised}
    for s in revised:
        n_left = len(s.draws_invalidated)
        earned = s.misses_sampled or s.sampled
        miss.append(
            f"the sampling frame for stratum {s.name!r} was revised after the draw: "
            f"{n_left} of {n_left + earned} drawn document(s) left it when the term list "
            f"grew, so the reported bound (elusion <= {s.elusion_upper:.3f}) is earned by "
            f"{earned} surviving draw(s), not by {n_left + earned} — draw and inspect "
            f"{s.replacement_draws_required} replacement(s) to restore n. A sample is tied "
            "to the frame it was drawn from; inheriting the bound across a revision is the "
            "one way expansion can make an absence claim weaker while looking stronger"
        )

    if gate_spec.get("keyword_list_validated", True):
        s = by.get("may_mention")
        # Read the stratum's own verdict rather than recomputing a weaker one here. The
        # recomputation accepted `misses_sampled >= 1`, so a single inspected document
        # discharged a stratum whose spec asks for 25, and it ignored the searches entirely.
        ok = s is None or s.keyword_list_validated
        c["keyword_list_validated"] = ok
        if not ok and s:
            if s.keywords_unsearched:
                miss.append(
                    f"keyword list not validated: {len(s.keywords_unsearched)} of "
                    f"{len(s.required_keywords)} required searches never ran "
                    f"({', '.join(repr(k) for k in s.keywords_unsearched)}) — with no search "
                    "there are no misses to sample, so a clean sample validates nothing"
                )
            elif s.hits_unread:
                miss.append(
                    f"{len(s.hits_unread)} of {s.hits} search hit(s) in stratum {s.name!r} "
                    f"were never read ({', '.join(s.hits_unread[:5])}"
                    f"{', …' if len(s.hits_unread) > 5 else ''}) — a hit is excluded from the "
                    "miss sample, so an unread one is reviewed by nothing at all"
                )
            elif s.name in frame_reported:
                pass    # already refused above, in the vocabulary that fits the cause
            elif not s.required_keywords:
                miss.append(
                    f"stratum {s.name!r} is search-validated but declares no required_keywords "
                    "— there is no keyword list to validate (that is SPEC_INSUFFICIENT)"
                )
            else:
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
            # NOT MARKED TERMINAL, and the first attempt at this was wrong in a way worth
            # recording. It read: terminal when `all(replacement_draws_required <= 0)`, on the
            # reasoning that once every stratum has reached its sample size the sampler draws no
            # more and the bound is frozen. That is true of `cannot_establish`, whose frame is
            # fixed. It is FALSE of the miss frame, which this module's own docstring says MOVES
            # whenever the term list grows: the frame is "this stratum minus whatever the searches
            # hit", so extending the keywords changes N, the draws and the bound.
            #
            # Measured cost of getting that wrong: on the 2026-07-29 re-run one patient who had
            # been three-for-three correct came back EVIDENCE_INSUFFICIENT. The gate had walked it
            # down to a single remaining item whose own message read "keyword list not validated
            # (misses sampled 8, hits 1) -- extend the keywords and re-search this stratum" -- a
            # discharge-able instruction, and `revise_plan` exists to carry it out. CP upper for
            # 1 hit in 8 draws is 0.527, over the 0.12 cap, and the terminal test fired because
            # `replacement_draws_required` was 0 for a frame that had merely got SMALL. The run
            # stopped and its correct answer was lost. `eval compare` caught it as a per-instance
            # regression while all three headline rates were flat or better.
            #
            # An over-cap bound is a refusal, and the refusal names the remedy. The one condition
            # that genuinely forecloses is a hit in the fixed-frame exclusion sample, marked
            # above -- and the run that motivated all of this had that too, so nothing is lost.

    if windows is not None:
        gaps = [w for w in windows if w.disposition == "interior_gap"]
        c["no_interior_gaps"] = not gaps
        if gaps:
            days = sum((w.end - w.start).days for w in gaps)
            miss.append(
                f"{len(gaps)} interior follow-up gap(s) totalling {days} days — "
                "an empty interior window is an evidence gap, never an absence of events"
            )

    # The verdict is over `missing`, which is empty unless `enforce`. So an advisory run reports
    # PASS with a populated `advisories` list, and that pair is the honest description of what
    # happened: the runtime counted, it has something to say, and it is not refusing the answer.
    # Reading the verdict off `advisories` too would reinstate the gate under a new name.
    g.verdict = "PASS" if not g.missing else "FAIL"
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
