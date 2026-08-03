"""Run state: the plan, the evidence ledger, and the coverage ledger.

The two ledgers are the point of this module.

  EvidenceLedger  every claim the agent will finally make must be backed by a recorded
                  span in a specific note. The finalize step sees ONLY this ledger, not
                  the scratchpad, so the model cannot "remember" an uncited detail.

  CoverageLedger  an append-only record of what the agent actually did — which documents
                  it listed, which keywords it searched, which documents it read in full.
                  The spec's proof_obligation is checked against this, in code, before a
                  negative answer is allowed. Attestation is computed, never self-reported.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal, TypedDict


@dataclass
class Evidence:
    note_id: str
    doc_type: str
    date: str
    start: int
    end: int
    quote: str
    supports: str = ""          # which field / assertion this backs
    stance: Literal["supports", "contradicts"] = "supports"
    #: WHICH THING IN THE CHART this span is about — the specimen, the lesion, the procedure.
    #: Optional and empty by default, so every recorded run still loads. It exists because a
    #: flat span list cannot express "this quote is about specimen A and that one about
    #: specimen B", and "the right document, the wrong specimen" is a failure mode this repo
    #: has already named. `submit_answer` records `reported_lesion`; with both, the agreement
    #: between them is machine-checkable instead of something a reader has to notice.
    entity: str = ""
    #: A STABLE IDENTITY, assigned by the ledger, so a candidate has something to point at.
    #: `E1` / `E2` is already the numbering `render()` shows the model, so this is the
    #: identifier the model has been reading all along rather than a new one it has never seen
    #: — the `after_event` rule: a pointer to something unobservable gets invented, not resolved.
    #:
    #: Empty on an Evidence built by hand, which is every one in every recorded trace. Defaulted
    #: and last so those still load.
    evidence_id: str = ""

    # `event_date`, `polarity`, `certainty`, `target_entity_match` and `admissibility` were
    # added here for the candidate seeder and are gone with it. They were read by exactly one
    # caller and a field nobody reads is a field nobody maintains — which is the rule this
    # dataclass's own history is made of. `evidence_id` stays: `render()` has always shown the
    # model `[E1]`, `[E2]`, so the identifier is one it can cite.

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceLedger:
    items: list[Evidence] = field(default_factory=list)

    def add(self, e: Evidence) -> None:
        for x in self.items:  # de-duplicate identical spans
            # `entity` is IN the key: without it, one sentence quoted about two specimens
            # collapses to one item, and the collapse is silent — exactly the confusion the
            # field was added to make visible.
            if (x.note_id, x.start, x.end, x.supports, x.entity) == \
               (e.note_id, e.start, e.end, e.supports, e.entity):
                return
        # Numbered from the length AFTER the de-dup check, so a repeated span does not consume
        # an id. A gap in the numbering reads like a lost piece of evidence.
        e.evidence_id = f"E{len(self.items) + 1}"
        self.items.append(e)

    def to_list(self) -> list[dict]:
        return [e.to_dict() for e in self.items]

    def cited_notes(self) -> set[str]:
        return {e.note_id for e in self.items}

    def render(self) -> str:
        if not self.items:
            return "(no evidence recorded)"
        out = []
        for i, e in enumerate(self.items, 1):
            mark = "" if e.stance == "supports" else "  [CONTRADICTS]"
            # Omitted entirely when empty rather than rendered as "-": an absent anchor is the
            # field going unused, and a placeholder on every line would train the model to read
            # past the one line where an anchor was actually recorded.
            ent = f"\n      entity:   {e.entity}" if e.entity else ""
            out.append(
                f"[E{i}]{mark} {e.note_id} ({e.doc_type}, {e.date}) chars {e.start}-{e.end}\n"
                f'      supports: {e.supports or "-"}{ent}\n'
                f'      "{e.quote.strip()[:400]}"'
            )
        return "\n".join(out)


# THE CANDIDATE LEDGER LIVED HERE and is gone. Two attempts at making the competing-answer
# space a maintained structure were measured and both failed on their own terms:
#
#   as a SKILL CARD  `tactic-counterevidence` says the target thing in plain words. Paired over
#                    twelve charts: distinct values submitted per run 1.00 -> 1.00 (0 of 24 runs
#                    ever submitted two), and mentions of an alternative in the reasoning went
#                    DOWN, 4/12 -> 1/12, in 11% more words.
#   as MACHINERY     a typed ledger plus an independent structured reasoner call. It worked and
#                    was reliable — 83% gold retention, 100% rejection precision on competing
#                    charts, and all thirteen "already resolved" claims were truthful. What it
#                    never did was hold two competing VALUES; and the mechanical seeder added to
#                    make it do so functioned on dates only, cost SYN0002 its gold answer three
#                    times out of three, and left 40% of clear charts in false competition.
#
# The behaviour is now asked for in prose, in `assets/skills/controller-hypothesis-set/SKILL.md`:
# enumerate the ways the contract says an answer can be established, then confirm or exclude each.
# Judgement the model applies, recorded when it departs, enforced nowhere. What that gives up is
# the ability to COUNT whether it happened; what it stops paying for is a structure that shaped
# the answer more than it described it.
#
# The measurements are in `docs/CANDIDATE_LEDGER_REMOVED.md`. Anyone rebuilding this should read
# them first: both approaches are already falsified, and a third would need a different premise.

# CoverageLedger and the proof-obligation gate used to live here as a flat record
# (listed_documents / searched_terms / read_notes). They now live in `coverage.py`, where
# the stratified accounting, forced sampling and Clopper-Pearson bounds are — and there is
# deliberately only ONE of them. Two independent accounts of "how much was covered" can
# disagree, nothing raises when they do, and you are left with two numbers and no way to
# choose. Import from `acr.review.coverage`.


class PlanStep(TypedDict, total=False):
    id: str
    goal: str
    rationale: str
    status: Literal["pending", "active", "done", "dropped"]


class RunState(TypedDict, total=False):
    """LangGraph channel dict."""
    patient_id: str
    spec_id: str
    plan: list[PlanStep]
    plan_revisions: int
    messages: list[dict]
    step: int
    max_steps: int
    evidence: list[dict]
    coverage: dict
    reflection: dict
    answer: dict
    done: bool
    # Declared here or LangGraph drops it. A channel set by a node but absent from this
    # TypedDict is silently discarded, the downstream read returns the falsy default, and
    # nothing errors — which is how an accepted answer came out labelled UNGATED.
    gate_validated: bool
    rejections: list[dict]
    usage: dict


@dataclass
class Budget:
    max_steps: int = 24
    max_plan_revisions: int = 6
    max_tokens: int = 400_000
    max_seconds: int = 1200

    @property
    def report(self) -> dict:
        """The limits, for the manifest.

        `negative_basis: BUDGET_EXHAUSTED` without the numbers is not a finding a reader can
        act on: it does not say which of the three limits bound, and it does not say whether
        the limit was the default or something the operator chose. Seven real runs abstained
        on `max_tokens (400000)` while their step counts sat at 8-16 against a cap of 24, and
        the manifest recorded no budget at all — so the abstention read as a fact about the
        charts. It was a fact about this dataclass.
        """
        return {"max_steps": self.max_steps, "max_tokens": self.max_tokens,
                "max_seconds": self.max_seconds, "max_plan_revisions": self.max_plan_revisions,
                "is_library_default": self == Budget()}

    def exceeded(self, *, step: int, tokens: int, elapsed: float) -> str | None:
        if step >= self.max_steps:
            return f"max_steps ({self.max_steps}) reached"
        if tokens >= self.max_tokens:
            return f"max_tokens ({self.max_tokens}) reached"
        if elapsed >= self.max_seconds:
            return f"max_seconds ({self.max_seconds}) reached"
        return None
