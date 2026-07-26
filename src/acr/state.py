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


@dataclass
class CoverageLedger:
    """Computed from real tool calls — the agent cannot write to it directly."""
    listed_documents: bool = False
    total_documents: int = 0
    type_summary_seen: bool = False
    searched_terms: list[str] = field(default_factory=list)
    read_notes: list[str] = field(default_factory=list)      # full or paginated reads
    read_sections: list[str] = field(default_factory=list)   # "note_id#SECTION"
    doc_types_touched: list[str] = field(default_factory=list)

    def note_search(self, term: str) -> None:
        t = term.strip().lower()
        if t and t not in self.searched_terms:
            self.searched_terms.append(t)

    def note_read(self, note_id: str, doc_type: str) -> None:
        if note_id not in self.read_notes:
            self.read_notes.append(note_id)
        if doc_type and doc_type not in self.doc_types_touched:
            self.doc_types_touched.append(doc_type)

    def note_section(self, note_id: str, section: str, doc_type: str = "") -> None:
        key = f"{note_id}#{section}"
        if key not in self.read_sections:
            self.read_sections.append(key)
        if doc_type and doc_type not in self.doc_types_touched:
            self.doc_types_touched.append(doc_type)

    def to_dict(self) -> dict:
        return asdict(self)

    def render(self) -> str:
        return (
            f"documents listed: {self.listed_documents} (total {self.total_documents})\n"
            f"type summary seen: {self.type_summary_seen}\n"
            f"searches run ({len(self.searched_terms)}): {', '.join(self.searched_terms) or '-'}\n"
            f"notes read ({len(self.read_notes)}): {', '.join(self.read_notes[:20]) or '-'}\n"
            f"sections read ({len(self.read_sections)}): {', '.join(self.read_sections[:20]) or '-'}\n"
            f"doc types touched: {', '.join(self.doc_types_touched) or '-'}"
        )


@dataclass
class ObligationCheck:
    satisfied: bool
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def check_proof_obligation(spec, coverage: CoverageLedger) -> ObligationCheck:
    """Code-enforced gate. Called before any negative/absent answer is accepted."""
    missing: list[str] = []
    po = spec.proof_obligation

    if po.required_coverage:
        if not coverage.listed_documents:
            missing.append("must list the patient's documents before asserting absence")
        for kw in po.required_keywords:
            if not any(kw.lower() in t or t in kw.lower() for t in coverage.searched_terms):
                missing.append(f"required search not performed: {kw!r}")
        for dt in po.required_doc_types:
            touched = any(dt.lower() in t.lower() for t in coverage.doc_types_touched)
            present = any(dt.lower() in n.lower() for n in coverage.read_notes)
            if not (touched or present):
                missing.append(f"required document type not reviewed: {dt!r}")
    return ObligationCheck(satisfied=not missing, missing=missing)


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
