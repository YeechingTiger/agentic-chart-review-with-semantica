"""The decision-point taxonomy: the identity that makes decisions comparable across runs.

A free-text note is readable but not queryable — "the same decision point, different runs,
different choices" needs the point to have a NAME. These types are that name. They come from
the operator discussions (corroboration / arbitration / dedup-ordering / derivation) plus the
review lifecycle's recurring choices (what to search, when coverage suffices, when to stop),
and they are deliberately about the REVIEW PROCESS, not about any one contract: a guideline is
per-contract, a decision type is cross-contract, and audit joins the two — a guideline rule
declares which types it governs, and every recorded decision of that type answers to it.

`other` is the escape valve: a forced taxonomy pollutes faster than an incomplete one. A type
the model claims that is not declared here is normalized to `other` with the claim preserved,
so a recurring unknown type is visible in the ledger and can be promoted into the list.
"""
from __future__ import annotations

#: type -> what a decision of this type settles
DECISION_TYPES: dict[str, str] = {
    "search_strategy": "what to look for next, and with which query or filter",
    "coverage": "whether the looking done so far suffices, and what remains unexamined",
    "source_selection": "which document to read or rely on among candidates",
    "credibility": "how much weight one source carries (document type, author, ambiguity)",
    "corroboration": "whether independent pieces of evidence reinforce one claim",
    "arbitration": "which of two conflicting candidate answers governs, and by what rule",
    "dedup_ordering": "whether two mentions are one event, and which date orders them",
    "derivation": "computing the answer value from established evidence (formats, calendars)",
    "inference": "concluding beyond what any document states, with premises and eliminations",
    "absence": "claiming the chart does not establish something",
    "sufficiency": "whether the recorded evidence discharges the answer's obligations",
    "scope": "whether the case or question falls inside this contract at all",
    "stopping": "ending the review: submit, abstain, or keep looking",
    "other": "a decision point the taxonomy does not name yet",
}


def normalize_type(claimed: str | None) -> tuple[str, str | None]:
    """(canonical type, preserved claim if it was not canonical)."""
    t = (claimed or "").strip()
    if t in DECISION_TYPES:
        return t, None
    return "other", (t or None)


def as_prompt_lines() -> str:
    return "\n".join(f"  - {name}: {what}" for name, what in DECISION_TYPES.items()
                     if name != "other")
