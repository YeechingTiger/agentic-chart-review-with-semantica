"""What a decision may say it rested on, and how it names it — the RUNTIME vocabulary.

Deliberately separate from [`decision_types`](decision_types.py), which is the taxonomy.
The two answer different questions and change at different rates:

  * **this module** asks *what did you use* and *where did you know it from*. Both are facts
    about the model's state at the moment it decided, so both must be collected AT RUNTIME —
    a later reconstruction sees only the trajectory and would mark every citation verified
    (it can see which documents were opened) and every judgment contract-grounded (it can see
    the contract would support the conclusion). The gap between what a run *consulted* and
    what it *looked at* is the finding, and only the model can report it.
  * `decision_types` asks *what KIND of judgment was this*. That vocabulary is still being
    grown from real runs, so it is applied afterwards, where a changed taxonomy costs one
    re-extraction rather than a re-run.

`acr.mvp.toolserver` imports this module and never imports `decision_types`. That is the
decoupling, stated so a test can check it.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

#: Orthogonal to rule coverage: this says where the agent claims its rationale came from.
BASIS_SOURCES: dict[str, str] = {
    "task_contract": "an exact clause in this run's Task Presentation",
    "method_card": "a named method card included in this run's Task Presentation",
    "operational_instruction": "a harness or chart-tool operating instruction",
    "precedent": "a precedent explicitly returned in this run",
    "chart": "a fact surfaced or read in this chart",
    "own_knowledge": "clinical or general knowledge outside the supplied material",
}

RULE_COVERAGE_CLAIMS = (
    "DIRECTLY_COVERED",
    "COVERED_WITH_INTERPRETATION",
    "NO_APPLICABLE_RULE",
    "AMBIGUOUS_RULE",
    "CONFLICTING_RULES",
    "OPERATIONAL_DISCRETION",
)

#: How a decision names the information it used. A **Warrant** can be articulate and false —
#: CONTEXT.md's example is a run stating a Discriminating Fact is absent having never searched
#: for it — so every claimed input is written in a form the server can check.
INPUT_KINDS: dict[str, str] = {
    "note": "one document, by note_id — checked against what this run read or surfaced",
    "search": "one search's results, by its query string verbatim",
    "evidence": "a recorded evidence span, by its 1-based index",
    "finding": "an earlier structured note finding, by its 1-based index",
    "rule": "a clause of the contract, by number or name",
    "card": "a method card from the prompt, by name",
    "precedent": "a precedent returned to you in this run, by id",
    "decision": "an earlier decision point of this run, by seq",
    "instruction": "an operational instruction from this run's Task Presentation",
}


def normalize_basis_sources(claimed: object) -> tuple[list[str], list[str]]:
    """(recognised kinds, preserved unrecognised claims). Never refuses — an unrecognised
    grounding claim is still the model telling us something about where it got this."""
    items = claimed if isinstance(claimed, list) else ([claimed] if claimed else [])
    good, bad = [], []
    for raw in items:
        s = str(raw).strip()
        (good if s in BASIS_SOURCES else bad).append(s)
    return good, bad


def basis_source_lines() -> str:
    return "\n".join(f"  - {k}: {v}" for k, v in BASIS_SOURCES.items())


def rule_coverage_lines() -> str:
    return "\n".join(f"  - {name}" for name in RULE_COVERAGE_CLAIMS)


# Old traces are still readable, but new runtime schemas never expose these names.  The aliases
# live only at the parsing seam so pre-upgrade audit artifacts do not become unreadable.
GROUNDING_KINDS = BASIS_SOURCES
normalize_grounding = normalize_basis_sources
grounding_lines = basis_source_lines


def input_prompt_lines() -> str:
    return "\n".join(f"  - {kind}:<...> — {what}" for kind, what in INPUT_KINDS.items())


@dataclass(slots=True)
class RunFacts:
    """What one run actually observed — the set of things a decision may legitimately cite.

    ONE implementation of "does this citation stand", built either from the live `ToolState`
    while a run happens or from a finished trace while it is read back. Two implementations
    that disagreed would be the worst failure this instrument can have: a warrant would count
    as false in the run and true in the report, or the reverse, and no reader could tell which
    number to believe.
    """

    documents_read: list[str]
    documents_seen: set[str]
    searches_run: list[str]
    n_evidence: int
    finding_refs: set[str] = field(default_factory=set)
    decision_refs: set[str] = field(default_factory=set)

    @classmethod
    def from_trace(cls, events: Iterable[dict[str, Any]]) -> RunFacts:
        """Replay a Layer-1 trace into the same facts the server held at the end of the run.

        Reads only what the SERVER recorded — the tool called and what came back — never what
        the model said about it, which is the whole point: this is the ruler, not the claim.
        """
        read: list[str] = []
        seen: set[str] = set()
        searches: list[str] = []
        n_evidence = 0
        finding_refs: set[str] = set()
        decision_refs: set[str] = set()
        for e in events:
            if e.get("kind") != "tool_call" or not e.get("ok"):
                continue
            tool, args = e.get("tool"), e.get("args") or {}
            result = e.get("result") or {}
            if tool == "read":
                note_id = args.get("note_id")
                if note_id:
                    if note_id not in read:
                        read.append(note_id)
                    seen.add(note_id)
            elif tool == "search":
                searches.append(args.get("query"))
                seen.update(h.get("note_id") for h in result.get("hits") or [])
            elif tool == "list_documents":
                seen.update(d.get("note_id") for d in result.get("documents") or [])
            elif tool == "record_evidence" and result.get("recorded"):
                n_evidence += 1
            if tool in {"note_decision", "record_finding"}:
                testimony_ref = result.get("testimony_ref")
                if testimony_ref:
                    decision_refs.add(str(testimony_ref))
            if tool == "record_finding" and result.get("recorded"):
                finding_ref = result.get("finding_ref")
                if finding_ref:
                    finding_refs.add(str(finding_ref))
        return cls(read, {s for s in seen if s}, [q for q in searches if q], n_evidence,
                   finding_refs, decision_refs)

    def resolve(self, raw: object) -> dict[str, Any]:
        """One claimed input, checked. Never refuses — an unverifiable citation is recorded as
        unverifiable, which is the useful outcome: a decision resting on a document the run
        never opened is a Warrant that cannot stand, and code can say so without reading a
        word of the reasoning."""
        kind, _, target = str(raw).partition(":")
        kind, target = kind.strip().lower(), target.strip()
        row: dict[str, Any] = {"ref": str(raw),
                               "kind": kind if kind in INPUT_KINDS else "unrecognised"}
        if kind == "note":
            if target in self.documents_read:
                row |= {"verified": True, "depth": "read"}
            elif target in self.documents_seen:
                row |= {"verified": True, "depth": "seen_in_results"}
            else:
                row |= {"verified": False, "why": "this run never read or surfaced it"}
        elif kind == "search":
            row |= ({"verified": True} if target in self.searches_run
                    else {"verified": False, "why": "no search with this query was run"})
        elif kind == "evidence":
            ok = target.isdigit() and 1 <= int(target) <= self.n_evidence
            row |= ({"verified": True} if ok
                    else {"verified": False, "why": "no evidence span with this index"})
        elif kind == "finding":
            ref = f"finding:{target}"
            row |= ({"verified": True} if ref in self.finding_refs
                    else {"verified": False, "why": "no earlier finding with this reference"})
        elif kind == "decision":
            ref = f"decision:{target}"
            row |= ({"verified": True} if ref in self.decision_refs
                    else {"verified": False, "why": "no earlier decision with this reference"})
        elif kind == "rule":
            # Rule availability is resolved against the immutable Task Presentation by the
            # tool server before a reference reaches this fallback resolver.
            row |= {"verified": None, "why": "recorded as claimed"}
        else:
            row |= {"verified": False, "why": f"unrecognised reference kind {kind!r}"}
        return row

    def resolve_all(self, used: object) -> list[dict[str, Any]]:
        return [self.resolve(r) for r in (used if isinstance(used, list) else [])]
