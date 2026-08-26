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

#: Where the reasoning came from. The first four are checked against what this run actually
#: had; `own_knowledge` cannot be checked, and an honest self-report of it is the most
#: valuable single fact this instrument collects — so it is never penalised.
GROUNDING_KINDS: dict[str, str] = {
    "contract": "a clause the contract states — name it in `used` as rule:<id>",
    "card": "a method card or preamble from the prompt — name it as card:<name>",
    "chart": "purely a fact read in this chart — name it as note:/evidence:",
    "precedent": "a precedent that was retrieved for you — name it as precedent:<id>",
    "own_knowledge": "your own clinical or general knowledge; not in the material we gave you",
}

#: How a decision names the information it used. A **Warrant** can be articulate and false —
#: CONTEXT.md's example is a run stating a Discriminating Fact is absent having never searched
#: for it — so every claimed input is written in a form the server can check.
INPUT_KINDS: dict[str, str] = {
    "note": "one document, by note_id — checked against what this run read or surfaced",
    "search": "one search's results, by its query string verbatim",
    "evidence": "a recorded evidence span, by its 1-based index",
    "rule": "a clause of the contract, by number or name",
    "card": "a method card from the prompt, by name",
    "precedent": "a precedent returned to you in this run, by id",
    "decision": "an earlier decision point of this run, by seq",
}


def normalize_grounding(claimed: object) -> tuple[list[str], list[str]]:
    """(recognised kinds, preserved unrecognised claims). Never refuses — an unrecognised
    grounding claim is still the model telling us something about where it got this."""
    items = claimed if isinstance(claimed, list) else ([claimed] if claimed else [])
    good, bad = [], []
    for raw in items:
        s = str(raw).strip()
        (good if s in GROUNDING_KINDS else bad).append(s)
    return good, bad


def grounding_lines() -> str:
    return "\n".join(f"  - {k}: {v}" for k, v in GROUNDING_KINDS.items())


def input_prompt_lines() -> str:
    return "\n".join(f"  - {kind}:<...> — {what}" for kind, what in INPUT_KINDS.items())
