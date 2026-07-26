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

from dataclasses import dataclass, field, asdict
from typing import Any, Literal, TypedDict


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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvidenceLedger:
    items: list[Evidence] = field(default_factory=list)

    def add(self, e: Evidence) -> None:
        for x in self.items:  # de-duplicate identical spans
            if (x.note_id, x.start, x.end, x.supports) == (e.note_id, e.start, e.end, e.supports):
                return
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
            out.append(
                f"[E{i}]{mark} {e.note_id} ({e.doc_type}, {e.date}) chars {e.start}-{e.end}\n"
                f'      supports: {e.supports or "-"}\n'
                f'      "{e.quote.strip()[:400]}"'
            )
        return "\n".join(out)


# CoverageLedger and the proof-obligation gate used to live here as a flat record
# (listed_documents / searched_terms / read_notes). They now live in `coverage.py`, where
# the stratified accounting, forced sampling and Clopper-Pearson bounds are — and there is
# deliberately only ONE of them. Two independent accounts of "how much was covered" can
# disagree, nothing raises when they do, and you are left with two numbers and no way to
# choose. Import from `acr.coverage`.


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

    def exceeded(self, *, step: int, tokens: int, elapsed: float) -> str | None:
        if step >= self.max_steps:
            return f"max_steps ({self.max_steps}) reached"
        if tokens >= self.max_tokens:
            return f"max_tokens ({self.max_tokens}) reached"
        if elapsed >= self.max_seconds:
            return f"max_seconds ({self.max_seconds}) reached"
        return None
