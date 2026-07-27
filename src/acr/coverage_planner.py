"""THE plan: what may be opened, what is searched with which terms, what is only sampled.

The problem this replaces
-------------------------
The spec currently declares strata as literal filename substrings:

    can_establish: {doc_type_matches: ["Pathology", "Cytology"]}

That cannot be enumerated correctly in advance and demonstrably was not. On 2026-07-26,
in this corpus of 1,516 distinct document types:

  * `Fine-Needle-Report`, `Core-Needle-Biopsy`, `IMMUNOHISTOLOGY-RPT` were filed under
    `cannot_establish` -- the stratum declared incapable of establishing histology.
  * `Speech-Language-Pathology-Note` was filed under `can_establish`.
  * For patient P01 the diagnosis text appears ONLY in `Discharge-Summary`
    and `Fine-Needle-Report`; no document matches Pathology or Cytology at all.

Asking a human to enumerate 1,516 types per criterion per site does not scale, and asking
the reviewing agent to decide for itself is circular -- the thing under audit would be
choosing its own scope.

WHY THIS FILE NOW OWNS "THE PLAN", SINGULAR
-------------------------------------------
Two greps used to settle the architecture of this repo:

    grep 's["plan"]'                 src/acr/graph.py   -> the prose plan, read by nobody
    grep 'CoveragePlan|policy_for'   src/acr/graph.py   -> nothing at all

So there were TWO plans. One was a list of {id, goal, rationale} that was rendered into
messages and read by no code; the other was this one, which governs retrieval and which the
agent loop never consulted. The revisable object was the one that did not matter. That is
why REPLAN fired 0 times in 291 actions across 37 runs: the supervisor was asked "does
something learned change what should be done next?" about goals like "find the pathology
report", and the honest answer is no -- the GOAL never changes. What changes is the
RETRIEVAL SCOPE, and the retrieval scope was not in the revisable plan.

There is now one plan and it is this one. The prose goals are gone (see graph.PLAN_PROMPT's
removal); what is rendered into the agent's messages is this object, and what the agent may
revise is this object.

WHY REVISION IS MONOTONE EXPANSION AND ONLY THAT
------------------------------------------------
The architecture forbids the audited party choosing its own audit scope -- that is the
circularity `ForcedSampler` exists to prevent, and letting an agent rewrite its own
retrieval plan would restore it wholesale.

A MONOTONE EXPANSION does not restore it. The agent may ADD a term, PROMOTE a type toward
more reading (sample -> search -> read_all) and OPEN a thread. It may never remove a term,
demote a type, or drop a type out of the plan. Under that restriction a revision can only add
to the EVIDENCE: reading is strictly stronger evidence than sampling, every added term
becomes a search the gate then REQUIRES to have been run (see `graph.check_gate`), and no
obligation already outstanding is ever cancelled. The agent cannot cheat by looking at less,
and looking at more is exactly what we want when the prior was wrong.

MONOTONE IN THE EVIDENCE IS NOT MONOTONE IN THE BOUND, and this file used to claim it was.
Expansion moves the sampling frame — the miss universe is "the stratum minus the search
hits", so a term added at step 9 can empty it of documents already drawn and inspected. The
evidence about those documents survives; the BOUND does not, because a bound is a statement
about a population and the population changed. `coverage.CoverageLedger` now recomputes it
over the post-expansion frame and demands replacement draws; see `MONOTONICITY_VS_LEDGER`
below, which states the arithmetic in code rather than leaving it as a comfortable assumption
— it was left as one, and a gate PASS was reached on a bound the run had not earned.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

# ==========================================================================================
#                       MONOTONE EXPANSION AGAINST THE LEDGER ARITHMETIC
# ==========================================================================================
#: Read this before believing that "expansion can only help". It is checked by
#: `tests/test_replan.py`, section 4, and by `tests/test_coverage.py`'s frame-revalidation
#: block. Bullet 2 was WRONG until 2026-07-27 and is written out at length because the wrong
#: version is how the defect survived review: it asserted a guarantee, the assertion read as
#: settled, and nobody re-derived it.
#:
#: 1. NO OBLIGATION IS EVER CANCELLED -- FOR THE DRAWN-BUT-UNREAD CASE. `CoverageLedger.drawn`
#:    persists across calls and `samples` verdicts are never deleted, so promoting a type
#:    cannot retire a draw. A drawn document that a newly added term turns into a search hit
#:    leaves the miss-sampling universe and lands in `hits_unread`, which
#:    `keyword_list_validated` refuses until it is READ: a weaker obligation traded for a
#:    stronger one.
#:
#:    That is the ONLY case this bullet ever covered, and it is the only case
#:    `test_expansion_never_cancels_a_drawn_obligation` exercises -- it searches a document
#:    that has been drawn and NOT yet inspected. Once a draw carries a verdict the trade is
#:    not available: `hits_read` is already satisfied, so nothing new is owed on that
#:    document, and what expansion took away is not an obligation but the draw's membership
#:    of the frame. Restoring THAT is a new obligation, and it comes from the frame check in
#:    `coverage.pending_samples`, not from monotonicity.
#:
#: 2. THE BOUND MOVES WITH THE FRAME, AND EXPANSION MOVES THE FRAME. The old text here read
#:    "the bound cannot be gamed downward ... expansion never decreases n_s", which is true
#:    of the NUMBER and false of the QUANTITY. n_s does not decrease numerically; the universe
#:    it was drawn from shrinks underneath it. `elusion_upper` bounds the relevance rate of
#:    the MISS FRAME, and that frame is recomputed from the current term list on every call,
#:    so an added term can delete a fifth of the frame's members -- including members already
#:    drawn and cleared -- without a single verdict changing.
#:
#:    Measured on SYN0001 under STORE.400_522_523: 112 misses, 25 draws, no hits, elusion
#:    0.1129, gate PASS against the spec's 0.12 cap. One added term, 92 misses, 20 of the 25
#:    draws now hits, 5 draws still inside the frame: earned bound 0.4507, reported bound
#:    0.1129, still PASS. Anti-conservative, on the mechanism whose entire job is to make an
#:    absence claim trustworthy.
#:
#:    What is true now: a sample is tied to the frame it was drawn from. `stratum_results`
#:    recomputes the bound over the post-expansion frame using ONLY the surviving draws and
#:    reports the rest in `draws_invalidated`; `pending_samples` counts survivors rather than
#:    draws and issues replacements until n is restored; `evaluate_gate` refuses while
#:    `replacement_draws_required` is non-zero. So expansion CAN raise the reported bound
#:    without anything being found -- that is the accounting telling the truth about a smaller
#:    frame, not the prior being falsified -- and a run that cannot restore n has not earned
#:    its bound and does not pass.
#:
#: 3. WHERE IT GENUINELY BREAKS, SAID LOUDLY. Reading extra documents in a SAMPLED stratum
#:    earns no credit at all: `stratum_results` computes `complete`/`elusion_upper = 0` only
#:    for the `exhaustive*` policies, and a sampled stratum's bound is computed from the
#:    drawn sample alone. Two consequences, opposite in sign:
#:      (a) CORRECT, keep it. Documents the AGENT chose to read are not a random sample.
#:          Crediting them would let the agent pick the reassuring ones and call it evidence,
#:          which is the circularity again. Refusing the credit is the right call.
#:      (b) A FALSE PENALTY, and it is a defect. Expand far enough that every document in a
#:          search-validated stratum becomes a search hit and `misses` falls to zero; the
#:          stratum has then been CENSUSED (every hit must be read), yet `elusion_upper`
#:          stays at `clopper_pearson_upper(0, 0) == 1.0` because no miss was ever drawn. A
#:          `max_elusion_upper` cap therefore becomes unpassable precisely for the run that
#:          did the most work. That is not a monotonicity violation -- it can only make the
#:          gate harder, never easier, so nothing unsafe passes -- but it does mean an
#:          honest exhaustive expansion can be forced into EVIDENCE_INSUFFICIENT.
#:          Fixing it means teaching `coverage.stratum_results` that a censused stratum has
#:          elusion 0, which is a change to the ledger and is NOT made here.
MONOTONICITY_VS_LEDGER = (
    "monotone expansion is monotone in the EVIDENCE, not in the BOUND. It adds obligations "
    "and cancels none, and for a drawn-but-unread document it trades the miss sample for the "
    "stronger obligation to read the hit. It does NOT protect the elusion bound: adding a "
    "term shrinks the miss frame, so draws that leave it are struck (draws_invalidated), the "
    "bound is recomputed over the surviving draws, and replacements are demanded until n is "
    "restored -- a run that cannot restore n has not earned its bound and does not pass. It "
    "earns no statistical credit for self-selected reading (correct: a chosen sample is not "
    "a random one), and a stratum expanded to a full census still reports elusion_upper 1.0 "
    "because no miss was drawn -- a conservative defect in coverage.stratum_results, "
    "harmless to soundness, reported not patched."
)

# ------------------------------------------------------------------------------- policies
#: The three things the plan can say about a document type, ordered by how much reading they
#: commit to. The ORDER is the whole enforcement mechanism: a revision is admissible iff no
#: type's rank goes down.
POLICIES = ("sample", "search", "read_all")
POLICY_RANK = {p: i for i, p in enumerate(POLICIES)}

#: A spec stratum policy -> the plan bucket it implies. Same table as `mcp_server`, and it
#: has to stay the same table: two mappings from stratum policy to reading policy would give
#: two different answers to "may the agent open this", one per front end.
POLICY_BUCKET = {
    "exhaustive": "read_all",
    "exhaustive_until_witness": "read_all",
    "search_then_read_hits_and_sample_misses": "search",
    "validate_by_sampling": "sample",
}

# ------------------------------------------------------------------------------- triggers
#: WHY TRIGGERS ARE DETECTED AND NOT ASKED.
#: The old reflect node posed an open question -- "does something learned change what should
#: be done next?" -- whose default answer is no, and which a model correctly answered no to
#: 291 times running. These are mechanical conditions over tool results and the gate. Each
#: one FORCES a typed response; the model chooses what to add, never whether it was asked.
TRIGGER_ZERO_HIT_SEARCH = "ZERO_HIT_SEARCH"
TRIGGER_UNLISTED_ANSWER_TERM = "UNLISTED_ANSWER_TERM"
TRIGGER_UNSETTLED_THREAD = "UNSETTLED_THREAD"
TRIGGER_GATE_OBLIGATION_UNREACHABLE = "GATE_OBLIGATION_UNREACHABLE"
TRIGGERS = (TRIGGER_ZERO_HIT_SEARCH, TRIGGER_UNLISTED_ANSWER_TERM, TRIGGER_UNSETTLED_THREAD,
            TRIGGER_GATE_OBLIGATION_UNREACHABLE)


@dataclass(frozen=True)
class Trigger:
    """One mechanically detected condition, with the observation that produced it.

    `observation` is kept verbatim and short. It is what makes an added term auditable
    afterwards -- "the agent added `mucinous` at step 9 because a read surfaced it" is a
    develop-plane candidate; "the agent added `mucinous`" is a shrug.
    """

    kind: str
    observation: str
    note_id: str = ""
    doc_type: str = ""
    marker: str = ""
    step: int = 0
    terms_proposed: tuple[str, ...] = ()
    types_proposed: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {"kind": self.kind, "observation": self.observation[:400],
                "note_id": self.note_id, "doc_type": self.doc_type, "marker": self.marker,
                "step": self.step, "terms_proposed": list(self.terms_proposed),
                "types_proposed": list(self.types_proposed)}


# ================================================================== the marker catalogue
#: The catalogue is NOT written here. It lives in `skills/thread-chasing/`, it was measured
#: over 7,965 real documents, and it already knows which markers are low precision. A second
#: hand-written list in src/ would drift from it within a week and the two would disagree
#: about what blocks an answer.
SKILL_DIR = Path(__file__).resolve().parents[2] / "skills" / "thread-chasing"
MARKER_OBLIGATION_TABLE = SKILL_DIR / "SKILL.md"
MARKER_BASE_RATE_TABLE = SKILL_DIR / "references" / "marker-catalogue.md"

#: `read_document` sets this itself. SKILL.md: "Treat `truncated: true` as an open thread in
#: its own right, whether or not you saw a marker word." It is the 8046 error's actual
#: mechanism -- the run stopped 353 characters short of the word that resolved the stain.
MARKER_TRUNCATED = "truncated"

_ROW = re.compile(r"^\|(?P<c1>[^|]*)\|(?P<c2>[^|]*)\|(?P<c3>[^|]*)\|\s*$")
_TICKED = re.compile(r"`([^`]+)`")
_LOW_PRECISION_SENTENCE = re.compile(
    r"most instances of (?P<markers>(?:`[^`]+`(?:\s*(?:,|and)\s*)?)+)[^.]*ordinary clinic chatter")


@dataclass(frozen=True)
class Marker:
    text: str
    obligation: str
    #: Low-precision markers fire only inside a document type the plan judged decisive. The
    #: catalogue is explicit about why: `pending` is the second most common string in the
    #: corpus and is overwhelmingly medication refills and appointments, while
    #: `special stains pending` -- the phrase that caused the 8046 error -- occurs in 6
    #: documents out of 7,965. Opening a thread on every `pending` would block every run and
    #: the block would carry no information.
    low_precision: bool = False

    def to_dict(self) -> dict:
        return {"marker": self.text, "obligation": self.obligation[:200],
                "low_precision": self.low_precision}


@dataclass(frozen=True)
class MarkerCatalogue:
    markers: tuple[Marker, ...]
    source: str
    #: "" when both files parsed. Non-empty means the runtime is operating with a marker set
    #: it cannot vouch for, and every consumer is expected to say so rather than proceed
    #: quietly -- a silently shortened marker list is a check that cannot refuse.
    degraded: str = ""

    def by_text(self) -> dict[str, Marker]:
        return {m.text: m for m in self.markers}

    def scan(self, text: str) -> list[Marker]:
        """Markers present in `text`, longest first so the specific one wins the report.

        Longest-first matters: `special stains pending` and `pending` both match the same
        line, and the catalogue's whole point is that those two are not the same finding.
        """
        low = (text or "").lower()
        found = [m for m in self.markers if m.text in low]
        found.sort(key=lambda m: -len(m.text))
        out: list[Marker] = []
        for m in found:
            if not any(m.text in kept.text for kept in out):
                out.append(m)
        return out

    def to_dict(self) -> dict:
        return {"source": self.source, "degraded": self.degraded,
                "n_markers": len(self.markers),
                "markers": [m.to_dict() for m in self.markers]}


def _clean(cell: str) -> str:
    return " ".join(cell.replace("**", "").split()).strip()


def load_marker_catalogue(skill_dir: str | Path | None = None) -> MarkerCatalogue:
    """Parse the thread-chasing skill's own tables into the marker set the runtime enforces.

    The obligation table in SKILL.md is the authority on WHICH markers create an obligation,
    and that is exactly the set wanted here. Note what it excludes: `final diagnosis` appears
    in the base-rate table (283 occurrences) but not in the obligation table, because it is
    the RESOLUTION, not the thread. Taking the base-rate table as the marker set would open a
    thread on every pathology report in the corpus.
    """
    d = Path(skill_dir) if skill_dir is not None else SKILL_DIR
    obligations = d / "SKILL.md"
    rates = d / "references" / "marker-catalogue.md"
    if not obligations.exists():
        return MarkerCatalogue(
            markers=(Marker(MARKER_TRUNCATED, "finish the document before reasoning about it"),),
            source=str(obligations),
            degraded=(f"{obligations} not found; only the runtime-detected `truncated` thread "
                      "is enforced and NO marker text is being scanned for"))

    body = obligations.read_text(encoding="utf-8")
    low_precision: set[str] = set()
    m = _LOW_PRECISION_SENTENCE.search(body)
    if m:
        low_precision = {t.strip().lower() for t in _TICKED.findall(m.group("markers"))}

    markers: list[Marker] = []
    seen: set[str] = set()
    in_table = False
    for line in body.splitlines():
        row = _ROW.match(line)
        if not row:
            in_table = False
            continue
        c1, c2 = _clean(row.group("c1")), _clean(row.group("c2"))
        if set(c1) <= {"-", ":"} and c1:
            in_table = True
            continue
        if not in_table:
            continue
        toks = [t.strip().lower() for t in _TICKED.findall(c1)]
        # Only rows whose first cell is entirely backticked markers. A row whose first cell
        # carries prose is a different table (the 8046 measurements, the resolution index),
        # and sweeping those in would enrol "the two Surgical-Pathology-Documents" as a
        # marker string.
        if not toks or _TICKED.sub("", c1).strip(" ,/"):
            continue
        for t in toks:
            if t and t not in seen:
                seen.add(t)
                markers.append(Marker(t, c2, low_precision=t in low_precision))

    markers.append(Marker(MARKER_TRUNCATED,
                          "the read stopped short of the end of the document; page to the end "
                          "before reasoning about it", low_precision=False))
    degraded = "" if markers[:-1] else (
        f"{obligations} parsed to zero markers; the obligation table has moved or changed shape")
    if not rates.exists():
        degraded = (degraded + "; " if degraded else "") + f"{rates} not found (base rates unknown)"
    return MarkerCatalogue(markers=tuple(markers),
                           source=f"{obligations}" + (f" + {rates}" if rates.exists() else ""),
                           degraded=degraded)


# ==================================================================== the open-thread ledger
@dataclass
class OpenThread:
    """One deferred conclusion, and whether anybody went back for it.

    THE 8046 ERROR. Patient P05 was coded 8046 (non-small cell carcinoma, NOS) off a line
    saying special stains were pending. The document that resolved those stains was in the
    same chart, in the same file, 353 characters past where the read stopped. Nothing in the
    runtime knew there was a question outstanding, so nothing could refuse the answer.
    """

    thread_id: str
    note_id: str
    doc_type: str
    marker: str
    obligation: str
    excerpt: str
    opened_at_step: int
    state: str = "open"                 # open | resolved | dismissed
    resolution: str = ""
    resolved_at_step: int | None = None

    def to_dict(self) -> dict:
        return {"thread_id": self.thread_id, "note_id": self.note_id,
                "doc_type": self.doc_type, "marker": self.marker,
                "obligation": self.obligation[:200], "excerpt": self.excerpt[:240],
                "opened_at_step": self.opened_at_step, "state": self.state,
                "resolution": self.resolution[:400],
                "resolved_at_step": self.resolved_at_step}


class OpenThreadLedger:
    """Open threads block submission. Dismissal is allowed, and it is recorded.

    Not a warning and not a prompt line. A marker that produces advice the model may decline
    to act on is the same shape of non-control as a coverage ledger nobody checks: it records
    and it cannot refuse. So this is wired into `graph.gate_answer`, and the only two ways
    past it are a resolution or a dismissal that states a reason -- both of which land in the
    manifest, so "the reviewer did not notice" and "the reviewer decided it did not matter"
    stay different facts.
    """

    def __init__(self) -> None:
        self.threads: list[OpenThread] = []
        self.refused_dismissals: list[dict] = []

    def _find(self, thread_id: str) -> OpenThread | None:
        return next((t for t in self.threads if t.thread_id == thread_id), None)

    def open_thread(self, *, note_id: str, doc_type: str, marker: str, obligation: str,
                    excerpt: str, step: int) -> OpenThread | None:
        """Idempotent on (note_id, marker): re-reading a document must not multiply the debt."""
        tid = f"{note_id}#{marker}"
        if self._find(tid):
            return None
        t = OpenThread(thread_id=tid, note_id=note_id, doc_type=doc_type, marker=marker,
                       obligation=obligation, excerpt=" ".join((excerpt or "").split())[:240],
                       opened_at_step=step)
        self.threads.append(t)
        return t

    def resolve(self, thread_id: str, how: str, *, step: int) -> dict:
        t = self._find(thread_id)
        if t is None:
            return {"ok": False, "why": f"no such thread {thread_id!r}",
                    "open": [x.thread_id for x in self.unresolved()]}
        if not str(how or "").strip():
            return {"ok": False, "why": "a resolution must say where the thread was settled"}
        t.state, t.resolution, t.resolved_at_step = "resolved", str(how), step
        return {"ok": True, "thread_id": thread_id, "state": "resolved"}

    def dismiss(self, thread_id: str, reason: str, *, step: int) -> dict:
        """A dismissal without a reason is refused, and the refusal is kept.

        "Dismissed" with no reason is indistinguishable in the manifest from "never noticed",
        which is the whole failure being fixed.
        """
        t = self._find(thread_id)
        if t is None:
            return {"ok": False, "why": f"no such thread {thread_id!r}",
                    "open": [x.thread_id for x in self.unresolved()]}
        if not str(reason or "").strip():
            self.refused_dismissals.append({"thread_id": thread_id, "why": "no reason given"})
            return {"ok": False, "why": ("a dismissal must carry a reason; an unreasoned "
                                         "dismissal reads exactly like an unnoticed thread")}
        t.state, t.resolution, t.resolved_at_step = "dismissed", str(reason), step
        return {"ok": True, "thread_id": thread_id, "state": "dismissed"}

    def unresolved(self) -> list[OpenThread]:
        return [t for t in self.threads if t.state == "open"]

    def render(self) -> str:
        if not self.threads:
            return "(no unsettled threads detected)"
        return "\n".join(
            f"  [{t.state.upper()}] {t.thread_id}  ({t.doc_type}) {t.marker!r}: "
            f"{t.obligation[:90]}" + (f"  -> {t.resolution[:80]}" if t.resolution else "")
            for t in self.threads)

    def to_dict(self) -> dict:
        return {
            "n_opened": len(self.threads),
            "n_unresolved": len(self.unresolved()),
            "n_dismissed": sum(1 for t in self.threads if t.state == "dismissed"),
            "n_resolved": sum(1 for t in self.threads if t.state == "resolved"),
            "refused_dismissals": list(self.refused_dismissals),
            "threads": [t.to_dict() for t in self.threads],
        }


# ======================================================================= the expansion budget
@dataclass(frozen=True)
class ExpansionBudget:
    """What expansion is allowed to cost. Every cap is a required parameter.

    Unbounded expansion is just reading everything, which is the outcome the stratified
    design exists to avoid paying for. Exhausting this budget with obligations still
    outstanding is an honest STUCK / EVIDENCE_INSUFFICIENT -- never a silent truncation and
    never a pass; see `graph.ChartReviewAgent._after_reflect`.
    """

    max_terms_added: int
    max_type_promotions: int
    max_documents_opened_by_promotion: int
    max_revisions: int

    @classmethod
    def priced_against(cls, plan: "CoveragePlan", n_docs_by_type: dict[str, int],
                       *, max_revisions: int) -> "ExpansionBudget":
        """Caps derived from the commitment the plan was already priced at.

        No literal appears here on purpose. Each cap is "no more than what has already been
        committed", which is a statement about this plan rather than a number somebody
        picked:

          * terms      -- at most as many additions as the SPEC declared terms. Needing to
                          more than double the spec's list is not a runtime rescue; it is a
                          spec that was wrong, and the honest output is an abstention plus a
                          develop-plane candidate. When the spec declares no terms at all
                          the floor is one per output field, the smallest coherent list.
          * types      -- at most as many promotions as the plan already puts above `sample`.
          * documents  -- at most as many newly opened documents as the plan already commits
                          to opening.
        """
        terms = len(plan.initial_keywords) or plan.n_fields
        already = list(plan.read_all) + list(plan.search)
        docs = sum(n_docs_by_type.get(t, 0) for t in already)
        return cls(max_terms_added=terms, max_type_promotions=len(already),
                   max_documents_opened_by_promotion=docs, max_revisions=max_revisions)

    def to_dict(self) -> dict:
        return {"max_terms_added": self.max_terms_added,
                "max_type_promotions": self.max_type_promotions,
                "max_documents_opened_by_promotion": self.max_documents_opened_by_promotion,
                "max_revisions": self.max_revisions}


# =========================================================================== the revision
@dataclass(frozen=True)
class PlanRevision:
    """The typed object reflection returns. Prose appended to messages is not a revision.

    IF THE RUNTIME DOES NOT APPLY IT, IT DID NOT HAPPEN. That sentence is the entire bug this
    file exists to fix: the old REPLAN verdict produced a revised prose plan, appended it to
    the message list, and changed nothing about what the agent could open or was told to
    search.
    """

    add_terms: tuple[str, ...] = ()
    #: (doc_type, target_policy) pairs.
    promote_types: tuple[tuple[str, str], ...] = ()
    #: (note_id, marker, why) -- an agent may open a thread the scanner did not catch.
    open_threads: tuple[tuple[str, str, str], ...] = ()
    #: (thread_id, how) for a settled thread, and (thread_id, reason) via `dismiss_threads`.
    resolve_threads: tuple[tuple[str, str], ...] = ()
    dismiss_threads: tuple[tuple[str, str], ...] = ()

    def is_empty(self) -> bool:
        return not (self.add_terms or self.promote_types or self.open_threads
                    or self.resolve_threads or self.dismiss_threads)

    def to_dict(self) -> dict:
        return {"add_terms": list(self.add_terms),
                "promote_types": [{"type": t, "to": p} for t, p in self.promote_types],
                "open_threads": [{"note_id": n, "marker": m, "why": w}
                                 for n, m, w in self.open_threads],
                "resolve_threads": [{"thread_id": t, "how": h} for t, h in self.resolve_threads],
                "dismiss_threads": [{"thread_id": t, "reason": r}
                                    for t, r in self.dismiss_threads]}

    @classmethod
    def from_json(cls, j: Any) -> "PlanRevision":
        """Parse the model's reply. Anything unrecognised is dropped, never guessed at."""
        j = j if isinstance(j, dict) else {}
        terms = tuple(str(t).strip().lower() for t in (j.get("add_terms") or [])
                      if str(t).strip())
        proms: list[tuple[str, str]] = []
        for p in (j.get("promote_types") or []):
            if isinstance(p, dict):
                t, to = str(p.get("type", "")).strip(), str(p.get("to", "")).strip()
            elif isinstance(p, (list, tuple)) and len(p) == 2:
                t, to = str(p[0]).strip(), str(p[1]).strip()
            else:
                continue
            if t and to in POLICIES:
                proms.append((t, to))
        opens: list[tuple[str, str, str]] = []
        for o in (j.get("open_threads") or []):
            if isinstance(o, dict) and str(o.get("note_id", "")).strip():
                opens.append((str(o["note_id"]).strip(),
                              str(o.get("marker", "") or "agent_reported").strip().lower(),
                              str(o.get("why", "")).strip()))
        def _pairs(key: str, second: str) -> tuple[tuple[str, str], ...]:
            out: list[tuple[str, str]] = []
            for r in (j.get(key) or []):
                if isinstance(r, dict) and str(r.get("thread_id", "")).strip():
                    out.append((str(r["thread_id"]).strip(), str(r.get(second, "")).strip()))
            return tuple(out)
        return cls(add_terms=terms, promote_types=tuple(proms), open_threads=tuple(opens),
                   resolve_threads=_pairs("resolve_threads", "how"),
                   dismiss_threads=_pairs("dismiss_threads", "reason"))


@dataclass(frozen=True)
class PlanSnapshot:
    """The plan reduced to what monotonicity is defined over."""

    policies: tuple[tuple[str, str], ...]      # (doc_type, policy), sorted
    keywords: frozenset[str]

    def as_map(self) -> dict[str, str]:
        return dict(self.policies)


def check_monotone(before: PlanSnapshot, after: PlanSnapshot) -> list[str]:
    """Every way a candidate plan fails to be a superset of the current one.

    Returns the violations, empty when the revision is admissible. Deliberately a free
    function over two snapshots rather than a guard inside the mutator: a mutator that only
    offers additive operations makes the check unreachable, and an unreachable check is one
    that stops being true the day somebody adds a `remove_term` helper "just for tests".
    """
    v: list[str] = []
    lost = sorted(before.keywords - after.keywords)
    if lost:
        v.append(f"terms removed: {lost} — a term the plan once carried is a search the gate "
                 f"already requires; dropping it retracts evidence")
    b, a = before.as_map(), after.as_map()
    for t, pol in sorted(b.items()):
        if t not in a:
            v.append(f"type dropped from the plan: {t!r} — an unplanned type is a type nobody "
                     f"audits, which shrinks the sampling frame")
        elif POLICY_RANK[a[t]] < POLICY_RANK[pol]:
            v.append(f"type demoted: {t!r} {pol} -> {a[t]} — the audited party may not choose "
                     f"to look at less")
    return v


@dataclass
class RevisionOutcome:
    applied: bool
    terms_added: list[str] = field(default_factory=list)
    types_promoted: list[dict] = field(default_factory=list)
    threads_opened: list[str] = field(default_factory=list)
    threads_resolved: list[str] = field(default_factory=list)
    threads_dismissed: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)
    refusal_class: str = ""

    def changed_retrieval(self) -> bool:
        """Did anything change what the agent may open or is told to search?

        Thread bookkeeping alone is not a replan. Counting it as one would inflate the replan
        rate with exactly the kind of no-op the old REPLAN verdict was.
        """
        return bool(self.terms_added or self.types_promoted)

    def to_dict(self) -> dict:
        return {"applied": self.applied, "terms_added": self.terms_added,
                "types_promoted": self.types_promoted,
                "threads_opened": self.threads_opened,
                "threads_resolved": self.threads_resolved,
                "threads_dismissed": self.threads_dismissed,
                "refused": self.refused, "refusal_class": self.refusal_class}


#: Refusal classes, so a directory of manifests can be counted without reading prose.
REFUSED_NOT_MONOTONE = "NOT_MONOTONE"
REFUSED_BUDGET = "BUDGET_EXHAUSTED"
REFUSED_UNKNOWN_TYPE = "UNKNOWN_TYPE"
#: Not inadmissible and not unaffordable -- it retrieves nothing. See `redundant_against`.
REFUSED_REDUNDANT_TERM = "REDUNDANT_TERM"


# ------------------------------------------------------------------ what makes two terms one
def normalise_term(raw: str) -> str:
    """Case, surrounding whitespace, and internal whitespace runs. Nothing else.

    THE SEARCH THE AGENT ACTUALLY HAS is `corpus.PatientChart.search`, which compiles
    `re.escape(query)` with `re.IGNORECASE` -- a plain case-insensitive substring test. So
    "CARCINOMA", " carcinoma " and "final  diagnosis" are not new searches, they are the
    same query written differently (the double space is in fact strictly narrower, since
    `re.escape` keeps both spaces and the note has one).

    `.lower()` and not `.casefold()` on purpose: `coverage.keyword_was_searched` lowers both
    sides, and casefolding would store "ß" as "ss" -- a term the corpus search would then
    fail to find. Stemming and punctuation stripping are deliberately absent: a different
    word is a different search, and folding "carcinoma" onto "carcinoid" would drop a term
    the agent is still charged for.
    """
    return re.sub(r"\s+", " ", str(raw or "")).strip().lower()


def redundant_against(candidate: str, existing: Iterable[str]) -> str | None:
    """The already-planned term that makes `candidate` retrieve nothing new, or None.

    GET THE DIRECTION RIGHT. Search is substring matching, so hits(candidate) is a subset of
    hits(e) exactly when e is a substring of candidate:

        plan has "carcin", agent asks for "carcinoma"  -> REDUNDANT. Every document
            "carcinoma" could return is already returned by the search the plan requires.
        plan has "carcinoma", agent asks for "carcin"  -> NOT redundant. It is BROADER; it
            can only find more, and it is exactly the widening expansion exists to allow.

    Inverting this would refuse the only kind of addition that is worth a budget slot, so it
    is tested both ways round in `tests/test_replan.py`.
    """
    c = normalise_term(candidate)
    if not c:
        return None
    for e in existing:
        en = normalise_term(e)
        if en and en in c:
            return e
    return None


def _redundant_why(term: str, covered_by: str) -> str:
    """Named terms, both of them: a refusal the agent cannot act on is one it repeats."""
    return (f"{REFUSED_REDUNDANT_TERM}: {term!r} was NOT added and cost no term budget — the "
            f"plan already searches {covered_by!r}, and search is case-insensitive substring "
            f"matching, so every document {term!r} could return is already returned. Ask for "
            f"a term the plan does not already cover, or a SHORTER one, which is broader.")


@dataclass
class CoveragePlan:
    read_all: list[str] = field(default_factory=list)
    search: list[str] = field(default_factory=list)
    sample: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)
    rationale: dict[str, str] = field(default_factory=dict)
    confidence: dict[str, float] = field(default_factory=dict)
    source: str = "llm_planner"          # never let a guess masquerade as a curated binding
    raw: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- the two term lists
    #: THE SPEC'S OWN LIST, frozen at construction and never touched again.
    #:
    #: Coverage is evaluated against the FINAL, expanded list -- you added the term, you must
    #: run the search. But the develop-plane falsification signal is evaluated against THIS
    #: one, because a runtime rescue that quietly repairs the spec's list erases the evidence
    #: that the spec's list was wrong, and that evidence is the whole input to §6c. On this
    #: corpus STORE.400's five declared terms miss the diagnosis for 31.7% of patients; if
    #: the agent adds the missing term at step 9 and coverage is scored against the expanded
    #: list, that 31.7% reads as 0% and the spec never gets fixed.
    initial_keywords: list[str] = field(default_factory=list)
    #: term -> {step, trigger, observation}. WHICH term, WHEN, and WHAT OBSERVATION CAUSED IT.
    term_provenance: list[dict] = field(default_factory=list)
    promotion_log: list[dict] = field(default_factory=list)
    refused_revisions: list[dict] = field(default_factory=list)
    revisions_applied: int = 0
    #: Only used to floor the term budget when a spec declares no keywords at all.
    n_fields: int = 1

    # ------------------------------------------------------------------------ reading it
    def policy_for(self, doc_type: str) -> str:
        for pol, names in (("read_all", self.read_all), ("search", self.search)):
            if doc_type in names:
                return pol
        return "sample"

    def known_types(self) -> list[str]:
        return sorted(set(self.read_all) | set(self.search) | set(self.sample))

    def snapshot(self) -> PlanSnapshot:
        return PlanSnapshot(
            policies=tuple(sorted((t, self.policy_for(t)) for t in self.known_types())),
            keywords=frozenset(self.keywords))

    def may_open(self, doc_type: str) -> bool:
        """`sample` types are the runtime's to draw, not the agent's to browse."""
        return POLICY_RANK[self.policy_for(doc_type)] >= POLICY_RANK["search"]

    def terms_added(self) -> list[str]:
        return [r["term"] for r in self.term_provenance]

    def n_documents_promoted(self, n_docs_by_type: dict[str, int]) -> int:
        return sum(n_docs_by_type.get(r["type"], 0) for r in self.promotion_log)

    # ------------------------------------------------------------------------ revising it
    def _screen_terms(self, requested: Sequence[str]) -> list[tuple[str, str | None]]:
        """Each requested term, normalised, paired with the planned term it duplicates.

        `None` in the second slot means the term is genuinely new and worth a budget slot.

        ORDER-INDEPENDENT WITHIN ONE REVISION. A candidate is screened against its siblings
        as well as against the plan, so asking for "carcin" and "carcinoma" together costs
        one slot whichever order the model emitted them in. Screening only against the plan
        would make the price of a revision depend on list order, which is not something an
        agent can reason about or a reviewer can reproduce.
        """
        wanted: list[str] = []
        for raw in requested:
            t = normalise_term(raw)
            if t and t not in wanted:      # the plain repeat, folded before anything is priced
                wanted.append(t)
        return [(t, redundant_against(t, list(self.keywords)
                                      + [s for j, s in enumerate(wanted) if j != i]))
                for i, t in enumerate(wanted)]

    def apply_revision(self, rev: PlanRevision, *, step: int, trigger: str, observation: str,
                       budget: ExpansionBudget, threads: OpenThreadLedger | None = None,
                       n_docs_by_type: dict[str, int] | None = None,
                       known_types: Sequence[str] | None = None) -> RevisionOutcome:
        """Apply a typed revision, or refuse it and say why. Nothing in between.

        All-or-nothing on the retrieval half, on purpose. Applying the admissible items of a
        revision that also asked to demote a type would hand back a plan the agent did not
        propose and cannot see, and the next revision would be computed against it.
        """
        out = RevisionOutcome(applied=False)
        n_docs = dict(n_docs_by_type or {})

        # ---- 1. build the candidate plan, INCLUDING anything inadmissible, so the
        #         monotonicity check has something to catch.
        #
        # TERM VARIANTS ARE DROPPED HERE, BEFORE ANYTHING IS PRICED. A duplicate that differs
        # only in case or spacing -- or one the plan already covers by a shorter term --
        # retrieves nothing, and letting it through spends one of `max_terms_added`, a cap
        # that is single digits. An agent can burn its whole allowance on variants and learn
        # nothing about the chart. The check sits in the MUTATOR and not in
        # `PlanRevision.from_json` because a revision can be built directly, and `graph` does
        # exactly that twice (`_fit_terms_to_budget`, `_salvage_thread_work`).
        cand_keywords = list(self.keywords)
        redundant: list[tuple[str, str]] = []
        for t, covered_by in self._screen_terms(rev.add_terms):
            if covered_by is None:
                cand_keywords.append(t)
            else:
                redundant.append((t, covered_by))
        cand_policy = {t: self.policy_for(t) for t in self.known_types()}
        vocabulary = set(known_types or self.known_types())
        unknown = [t for t, _ in rev.promote_types if t not in vocabulary]
        for t, to in rev.promote_types:
            if t in vocabulary:
                cand_policy[t] = to
        after = PlanSnapshot(policies=tuple(sorted(cand_policy.items())),
                             keywords=frozenset(cand_keywords))

        violations = check_monotone(self.snapshot(), after)
        if unknown:
            # A hallucinated document type must not enter the plan, exactly as in
            # `plan_coverage`. Refused rather than dropped: dropping it would leave the agent
            # believing it had widened a scope it had not.
            violations.append(f"unknown document type(s) {sorted(unknown)} — promote a type "
                              f"this chart actually has")
        if violations:
            out.refused = violations
            out.refusal_class = (REFUSED_UNKNOWN_TYPE if unknown and len(violations) == 1
                                 else REFUSED_NOT_MONOTONE)
            self._record_refusal(out, step=step, trigger=trigger, rev=rev)
            return out

        # ---- 2. price it.
        new_terms = [t for t in cand_keywords if t not in self.keywords]
        new_proms = [(t, to) for t, to in rev.promote_types
                     if POLICY_RANK[to] > POLICY_RANK[self.policy_for(t)]]
        over: list[str] = []
        if self.revisions_applied >= budget.max_revisions:
            over.append(f"revision budget exhausted ({budget.max_revisions} applied)")
        if len(self.terms_added()) + len(new_terms) > budget.max_terms_added:
            over.append(f"term budget exhausted: {len(self.terms_added())} added, "
                        f"{len(new_terms)} more requested, cap {budget.max_terms_added}")
        if len(self.promotion_log) + len(new_proms) > budget.max_type_promotions:
            over.append(f"promotion budget exhausted: {len(self.promotion_log)} promoted, "
                        f"{len(new_proms)} more requested, cap {budget.max_type_promotions}")
        added_docs = sum(n_docs.get(t, 0) for t, _ in new_proms)
        if (self.n_documents_promoted(n_docs) + added_docs
                > budget.max_documents_opened_by_promotion):
            over.append(f"document budget exhausted: promoting {[t for t, _ in new_proms]} "
                        f"would open {added_docs} more documents, cap "
                        f"{budget.max_documents_opened_by_promotion}")
        if over:
            out.refused = over
            out.refusal_class = REFUSED_BUDGET
            self._record_refusal(out, step=step, trigger=trigger, rev=rev)
            return out

        # A revision whose whole retrieval half was variants is a REFUSAL, not a silent no-op.
        # `graph._after_reflect` renders `outcome.refused` to the agent on the refusal branch
        # only; reporting "APPLIED" over a plan that did not move would leave the agent
        # believing it had widened its search, and it would send the variant again. The thread
        # half is not collateral — the caller re-applies it, see `_salvage_thread_work`.
        if redundant and not new_terms and not new_proms:
            out.refused = [_redundant_why(t, e) for t, e in redundant]
            out.refusal_class = REFUSED_REDUNDANT_TERM
            self._record_refusal(out, step=step, trigger=trigger, rev=rev)
            return out

        # ---- 3. commit.
        for t in new_terms:
            self.keywords.append(t)
            self.term_provenance.append({"term": t, "step": step, "trigger": trigger,
                                         "observation": (observation or "")[:400]})
            out.terms_added.append(t)
        for t, to in new_proms:
            frm = self.policy_for(t)
            for bucket in (self.read_all, self.search, self.sample):
                if t in bucket:
                    bucket.remove(t)
            {"read_all": self.read_all, "search": self.search, "sample": self.sample}[to].append(t)
            row = {"type": t, "from": frm, "to": to, "step": step, "trigger": trigger,
                   "observation": (observation or "")[:400],
                   "n_documents": n_docs.get(t, 0)}
            self.promotion_log.append(row)
            out.types_promoted.append(row)

        # Requests that were admissible but changed nothing are reported, not counted. An
        # agent re-promoting a type it already promoted is not a replan, and neither is a
        # term the plan already covers — but both travel back in `refused` and into the
        # manifest, because a request that vanished without a word is one nobody can audit.
        for t, e in redundant:
            out.refused.append(_redundant_why(t, e))
        for t, to in rev.promote_types:
            if (t, to) not in new_proms and t in vocabulary:
                out.refused.append(f"{t!r} is already at {self.policy_for(t)!r}; no change")

        if threads is not None:
            for note_id, marker, why in rev.open_threads:
                th = threads.open_thread(note_id=note_id, doc_type="", marker=marker or
                                         "agent_reported", obligation=why or "reported by the "
                                         "agent during reflection", excerpt=why, step=step)
                if th:
                    out.threads_opened.append(th.thread_id)
            for tid, how in rev.resolve_threads:
                r = threads.resolve(tid, how, step=step)
                (out.threads_resolved if r["ok"] else out.refused).append(
                    tid if r["ok"] else f"resolve {tid}: {r['why']}")
            for tid, reason in rev.dismiss_threads:
                r = threads.dismiss(tid, reason, step=step)
                (out.threads_dismissed if r["ok"] else out.refused).append(
                    tid if r["ok"] else f"dismiss {tid}: {r['why']}")

        out.applied = True
        if out.changed_retrieval():
            self.revisions_applied += 1
        return out

    def _record_refusal(self, out: RevisionOutcome, *, step: int, trigger: str,
                        rev: PlanRevision) -> None:
        self.refused_revisions.append({"step": step, "trigger": trigger,
                                       "refusal_class": out.refusal_class,
                                       "requested": rev.to_dict(), "why": out.refused})

    def budget_exhausted(self, budget: ExpansionBudget) -> bool:
        return any(r["refusal_class"] == REFUSED_BUDGET for r in self.refused_revisions)

    # ------------------------------------------------------------------------ rendering it
    def render(self, n_docs_by_type: dict[str, int] | None = None) -> str:
        n = dict(n_docs_by_type or {})
        def _row(t: str) -> str:
            return f"{t}" + (f" (n={n[t]})" if t in n else "")
        L = ["RETRIEVAL PLAN — this is the plan, and it governs what you may open.",
             "",
             "READ IN FULL (these can establish the answer on their own):",
             "  " + (", ".join(_row(t) for t in sorted(self.read_all)) or "(none)"),
             "SEARCH (may restate or localise; open the hits):",
             "  " + (", ".join(_row(t) for t in sorted(self.search)) or "(none)"),
             "SAMPLED BY THE RUNTIME — you may NOT open these directly. The sampler draws "
             "them and hands you the note_ids:",
             "  " + (", ".join(_row(t) for t in sorted(self.sample)) or "(none)"),
             "",
             "SEARCH TERMS (every one of these must actually be run before a negative is "
             "allowed):",
             "  " + (", ".join(sorted(self.keywords)) or "(none)")]
        if self.term_provenance:
            L += ["", "TERMS ADDED DURING THIS RUN:"]
            L += [f"  + {r['term']}  (step {r['step']}, {r['trigger']})"
                  for r in self.term_provenance]
        if self.promotion_log:
            L += ["", "TYPES PROMOTED DURING THIS RUN:"]
            L += [f"  ^ {r['type']}: {r['from']} -> {r['to']}  (step {r['step']}, {r['trigger']})"
                  for r in self.promotion_log]
        return "\n".join(L)

    def to_dict(self) -> dict:
        return {"source": self.source, "read_all": self.read_all, "search": self.search,
                "sample": self.sample,
                # BOTH LISTS, always, and never merged. See `initial_keywords`.
                "initial_keywords": self.initial_keywords,
                "initial_keywords_are": ("the SPEC-DECLARED list. The develop-plane "
                                         "falsification signal is scored against this one, "
                                         "not against the expanded list"),
                "keywords": self.keywords,
                "keywords_are": "the FINAL list; coverage is evaluated against it",
                "terms_added": list(self.term_provenance),
                "promotions": list(self.promotion_log),
                "refused_revisions": list(self.refused_revisions),
                "revisions_applied": self.revisions_applied,
                "uncertain": self.uncertain, "confidence": self.confidence,
                "rationale": self.rationale}


# ==================================================================== building the plan
def spec_declared_keywords(spec) -> list[str]:
    """The spec's own term list: the falsification baseline, and nothing else.

    Both places a spec can declare terms, because `strata_from_spec` descends into the
    strata and `proof_obligation.required_keywords` does not -- a baseline that saw only one
    of them would show the agent adding terms the spec had in fact declared.
    """
    po = getattr(spec, "proof_obligation", None)
    kws: list[str] = list(getattr(po, "required_keywords", []) or []) if po else []
    fn = (getattr(po, "for_negative", {}) or {}) if po else {}
    for st in (fn.get("strata") or []):
        kws.extend(st.get("required_keywords") or [])
    for claim in (fn.get("claims") or []):
        for st in (claim.get("strata") or []):
            kws.extend(st.get("required_keywords") or [])
    out: list[str] = []
    for k in kws:
        k = str(k).strip().lower()
        if k and k not in out:
            out.append(k)
    return out


def _blank_plan(spec) -> CoveragePlan:
    init = spec_declared_keywords(spec)
    return CoveragePlan(initial_keywords=list(init), keywords=list(init),
                        n_fields=max(1, len(getattr(spec, "fields", []) or [])))


def plan_from_spec(spec, chart) -> CoveragePlan:
    """The plan the runtime can always build: the spec's strata, projected onto this chart.

    Used when no model is available and as the floor under a degraded planner. It is not a
    fallback in the apologetic sense -- it is the spec's own declaration, and a run on it is
    a run whose retrieval scope is exactly what the specification says it should be. That is
    the arm the develop plane wants to falsify.
    """
    from .coverage import assign_strata, strata_from_spec

    p = _blank_plan(spec)
    p.source = "spec_strata"
    docs, _ = chart.list_documents(limit=100_000)
    strata = strata_from_spec(spec)
    types = sorted({d.doc_type for d in docs})
    if not strata:
        # No stratification declared. Unjudged is not junk: search everything, and say so.
        p.search.extend(types)
        for t in types:
            p.rationale[t] = "spec declares no strata; every type is searched"
        p.source = "unstratified"
        return p
    assigned = assign_strata(docs, strata)
    for s in strata:
        bucket = POLICY_BUCKET.get(s.policy, "search")
        for t in sorted({d.doc_type for d in assigned.get(s.name, [])}):
            if t in p.read_all or t in p.search or t in p.sample:
                continue
            getattr(p, bucket).append(t)
            p.rationale[t] = f"stratum {s.name!r} (policy {s.policy})"
    for t in types:
        if t not in p.read_all and t not in p.search and t not in p.sample:
            p.search.append(t)
            p.rationale[t] = "matched no stratum; unjudged defaults to search, not to junk"
    return p


PLAN_PROMPT = """You are planning the document coverage for one chart review.

THE QUESTION BEING ANSWERED:
{question}

WHAT COUNTS AS EVIDENCE FOR EACH FIELD:
{evidence_rules}

FIELDS:
{fields}

THIS PATIENT'S DOCUMENT TYPES (name, how many, date range):
{inventory}

Assign EVERY type to exactly one policy:

  read_all  This type can, on its own, establish one of the fields. Usually a
            tissue/cytology diagnosis authored by a pathologist. Small and decisive.
            Judge by what the type IS, not by what its name contains -- a
            "Fine-Needle-Report" or "Core-Needle-Biopsy" is a pathology report even
            though neither name contains the word "pathology".

  search    This type may restate or mention a finding, or may localise the tumour,
            without being able to establish the diagnosis itself. Progress notes,
            consults, discharge summaries, endoscopy, imaging.

  sample    This type is very unlikely to bear on the question at all. Medication fills,
            EKGs, scheduling, billing, unrelated labs. You are asserting it can be spot
            checked rather than read; a single relevant hit will overturn that.

This assignment is BINDING on the reviewer: it may not open a `sample` type at all except
when the runtime's sampler hands it the note_id. It may widen the plan later, never narrow
it, so err toward `search` when you are unsure.

Also propose the keyword list to run over the `search` types. It must cover EVERY field,
not only the diagnosis -- if a field is an anatomical site, include the terms that would
localise it (lobe names, laterality, organ subsites).

Reply with JSON only:
{{"assignments":[{{"type":"<exact type string>","policy":"read_all|search|sample",
                  "why":"<short>","confidence":0.0-1.0}}],
  "keywords":["..."],
  "uncertain":["<types you are least sure about>"]}}"""


def inventory(chart) -> list[dict]:
    """Type name, count and date span. Names and metadata only -- no document text."""
    by: dict[str, list] = {}
    docs, _ = chart.list_documents(limit=100_000)
    for d in docs:
        by.setdefault(d.doc_type, []).append(d.date)
    out = []
    for t, dates in sorted(by.items(), key=lambda kv: -len(kv[1])):
        ds = sorted(str(x) for x in dates if x)
        out.append({"type": t, "n": len(dates),
                    "from": ds[0] if ds else "?", "to": ds[-1] if ds else "?"})
    return out


def documents_by_type(chart) -> dict[str, int]:
    docs, _ = chart.list_documents(limit=100_000)
    out: dict[str, int] = {}
    for d in docs:
        out[d.doc_type] = out.get(d.doc_type, 0) + 1
    return out


def plan_coverage(spec, chart, llm) -> CoveragePlan:
    """Ask once, up front, what to read / search / sample for THIS patient.

    The planner's proposed keywords do NOT join `initial_keywords`. They are recorded as the
    run's first monotone addition, with trigger `planner_proposal`, because a term the model
    proposed is a term the SPEC did not declare -- and that gap is precisely the develop-plane
    signal. Folding it into the baseline would erase the evidence at the moment it is created.
    """
    inv = inventory(chart)
    ev = spec.evidence_rules or {}
    prompt = PLAN_PROMPT.format(
        question=spec.question,
        evidence_rules=json.dumps(ev, indent=1)[:2000],
        fields="\n".join(f"  - {f.name}: {f.description or ''}" for f in spec.fields),
        inventory="\n".join(f"  {d['type']}  (n={d['n']}, {d['from']}..{d['to']})" for d in inv),
    )
    from .llm import extract_json
    r = llm.chat([{"role": "user", "content": prompt}])
    j = extract_json(r.content, require="assignments")

    p = _blank_plan(spec)
    p.raw = j
    p.uncertain = [str(u) for u in (j.get("uncertain") or [])]
    known = {d["type"] for d in inv}
    seen: set[str] = set()
    for a in (j.get("assignments") or []):
        t, pol = str(a.get("type", "")), str(a.get("policy", "sample"))
        if t not in known:
            continue                      # a hallucinated type must not enter the plan
        seen.add(t)
        {"read_all": p.read_all, "search": p.search}.get(pol, p.sample).append(t)
        p.rationale[t] = str(a.get("why", ""))[:200]
        try:
            p.confidence[t] = float(a.get("confidence", 0.0))
        except (TypeError, ValueError):
            p.confidence[t] = 0.0
    # A type the planner forgot is NOT silently dropped to `sample`: unmentioned means
    # unjudged, and the safe default for unjudged is to search it, not to assume it is junk.
    for t in known - seen:
        p.search.append(t)
        p.rationale[t] = "not classified by the planner; defaulted to search"
    for k in (j.get("keywords") or []):
        k = str(k).strip().lower()
        if k and k not in p.keywords:
            p.keywords.append(k)
            p.term_provenance.append({"term": k, "step": 0, "trigger": "planner_proposal",
                                      "observation": "proposed by the up-front coverage "
                                                     "planner; not declared by the spec"})
    return p


# ==================================================================== trigger detection
_WORD = re.compile(r"[A-Za-z][A-Za-z\-]{3,}")
#: Words that carry no retrieval signal. Short and boring on purpose: this list only shapes
#: the SUGGESTIONS attached to a trigger. The model chooses what to add and the runtime
#: records what it chose, so a miss here costs a suggestion, never a term.
_STOP = {"the", "and", "with", "this", "that", "from", "were", "was", "have", "has", "for",
         "patient", "report", "note", "date", "name", "history", "clinical", "final",
         "impression", "diagnosis", "specimen", "left", "right", "there", "which", "been",
         "also", "into", "these", "those", "than", "then", "when", "will", "would", "should"}


def _candidate_terms(text: str, current: Iterable[str], limit: int) -> tuple[str, ...]:
    have = [c.lower() for c in current]
    out: list[str] = []
    for w in _WORD.findall(text or ""):
        lw = w.lower()
        if lw in _STOP or lw in out or any(lw in c or c in lw for c in have):
            continue
        out.append(lw)
        if len(out) >= limit:
            break
    return tuple(out)


#: How many candidate terms a single trigger may suggest. A parameter and not a literal in
#: the loop, because it is the difference between a readable trigger and a wall of tokens.
MAX_SUGGESTED_TERMS = 8


def triggers_from_tool_result(name: str, args: dict, result: dict, *, plan: CoveragePlan,
                              catalogue: MarkerCatalogue, step: int,
                              quote: str = "") -> list[Trigger]:
    """The mechanical conditions, read off one tool result. No model is consulted.

    Deliberately not "did anything interesting happen" -- these are four decidable facts:
    a search that returned nothing, a citation the term list would not have found, an
    unsettled-thread marker, and (computed elsewhere, in `graph`) an obligation the current
    plan cannot discharge.
    """
    out: list[Trigger] = []
    if not isinstance(result, dict) or result.get("error"):
        return out

    if name == "search_notes" and int(result.get("n_hits", 0) or 0) == 0:
        q = str(args.get("query", ""))
        out.append(Trigger(
            kind=TRIGGER_ZERO_HIT_SEARCH, step=step,
            observation=(f"search for {q!r} returned zero hits"
                         + (f" (restricted to {args['doc_type_contains']!r})"
                            if args.get("doc_type_contains") else "")),
            terms_proposed=()))

    docs: list[dict] = []
    if name in ("read_document", "read_section") and result.get("note_id"):
        docs = [result]
    elif name == "read_documents_batch":
        docs = list(result.get("documents") or [])
    for d in docs:
        nid, dtype = str(d.get("note_id", "")), str(d.get("doc_type", ""))
        text = str(d.get("text", ""))
        if d.get("truncated"):
            out.append(Trigger(kind=TRIGGER_UNSETTLED_THREAD, step=step, note_id=nid,
                               doc_type=dtype, marker=MARKER_TRUNCATED,
                               observation=(f"read of {nid} returned "
                                            f"{d.get('returned_chars')} of "
                                            f"{d.get('total_chars')} characters")))
        for m in catalogue.scan(text):
            # The catalogue's own precision rule, expressed in the plan's vocabulary rather
            # than as a hard-coded list of document types: a low-precision marker counts only
            # inside a type this plan judged capable of establishing the answer. "Inside a
            # Surgical-Pathology-Document it is a thread; inside a Visit-Note it is not."
            if m.low_precision and plan.policy_for(dtype) != "read_all":
                continue
            i = text.lower().find(m.text)
            out.append(Trigger(kind=TRIGGER_UNSETTLED_THREAD, step=step, note_id=nid,
                               doc_type=dtype, marker=m.text,
                               observation=" ".join(text[max(0, i - 80):i + 120].split())))

    if name == "record_evidence" and quote:
        # A CITATION THE TERM LIST WOULD NOT HAVE FOUND. Mechanical and model-free: if no
        # current search term occurs in the quote the agent just chose to rest its answer on,
        # then the spec's retrieval plan did not lead here, and the term that would have led
        # here is missing from the list. That is the develop-plane candidate, caught at the
        # exact moment it is created.
        low = quote.lower()
        if plan.keywords and not any(k in low for k in plan.keywords):
            out.append(Trigger(
                kind=TRIGGER_UNLISTED_ANSWER_TERM, step=step,
                note_id=str(args.get("note_id", "")),
                observation=f"cited quote matches no current search term: {quote[:200]!r}",
                terms_proposed=_candidate_terms(quote, plan.keywords, MAX_SUGGESTED_TERMS)))
    return out


def gate_obligation_triggers(missing: Sequence[str], *, plan: CoveragePlan,
                             unread_hit_types: Sequence[str] = (), step: int = 0) -> list[Trigger]:
    """The fourth trigger: an outstanding obligation the CURRENT PLAN forbids discharging.

    A gate that says "read these hits" while the plan says "you may not open that type" is
    not a rejection the agent can act on -- it is a deadlock, and the old loop would spend
    the rest of its budget in it. Detecting it forces the one move that breaks it: promote
    the type.
    """
    out: list[Trigger] = []
    blocked = sorted({t for t in unread_hit_types if not plan.may_open(t)})
    if blocked:
        out.append(Trigger(
            kind=TRIGGER_GATE_OBLIGATION_UNREACHABLE, step=step,
            observation=("the gate requires reading search hits in types the plan does not "
                         f"let you open: {blocked}"),
            types_proposed=tuple(blocked)))
    for msg in missing:
        m = re.search(r"required search not performed(?: for stratum [^:]*)?: '([^']+)'", msg)
        if m and m.group(1).lower() not in [k.lower() for k in plan.keywords]:
            out.append(Trigger(
                kind=TRIGGER_GATE_OBLIGATION_UNREACHABLE, step=step,
                observation=f"the gate requires a search for {m.group(1)!r}, which is not in "
                            f"the plan's term list",
                terms_proposed=(m.group(1).lower(),)))
    return out
