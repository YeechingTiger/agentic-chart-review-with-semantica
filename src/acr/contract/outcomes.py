"""The set of things a run is allowed to conclude, and what each one obliges.

WHY THIS IS A MODULE AND NOT A LITERAL
--------------------------------------
It was a literal, in two places that could not see each other: an `enum` in
`toolbox.TOOL_SCHEMAS` (what the model is OFFERED) and a chain of `status == "..."` branches
in `answer_gate` (what the gate ACCEPTS). Neither was the contract's, and the contract is the
only place that knows what its own question can truthfully be answered with.

Two failures came out of that, and both are structural rather than a matter of tuning.

  A CONTRACT COULD NOT WIDEN ITS OWN OUTCOME SPACE. STORE.390's proof obligation tells the
  model to abstain "with the status that fits: no qualifying witness found, or the corpus
  itself insufficient". The second of those was a status the tool would not offer and no
  branch recognised. An instruction toward an unreachable outcome is not a weak prompt; it
  is a contract that cannot be complied with.

  AN UNDECLARED STATUS WAS THE MOST PERMISSIVE OUTCOME IN THE SYSTEM. `gate_answer` tested
  three literals and fell through to `accepted: True`, so a submission of `TOTALLY_MADE_UP`
  discharged nothing: not the evidence a value owes, not the coverage an absence owes, not
  the routing report a spec complaint owes. Nothing anywhere named that path, which is why
  it survived — a fall-through is invisible in a way an `else` is not.

WHY `kind` AND NOT JUST A LIST OF NAMES
---------------------------------------
Because the obligations are attached to what a status CLAIMS, not to its spelling. "The
chart does not establish a date" and "this record does not go back far enough" are both
claims about this chart, and both are only true if the chart was searched — so both owe the
coverage proof, and a gate branching on the literal `EVIDENCE_INSUFFICIENT` would hand the
second one a free pass. Every status a contract adds would arrive with zero obligations,
which turns a wider outcome space into a way around the gate.

Four kinds, and adding a fifth means writing the branch that implements it:

    value             carries a value; proved by WITNESS. Owes recorded evidence.
    abstain_evidence  a claim about THIS CHART that is also a claim about coverage. Owes
                      whatever proof of search the runtime profile requires.
    abstain_spec      a claim about the CONTRACT, not the chart. Exempt from coverage — no
                      amount of reading can make a silent spec speak — and owes instead the
                      routing report in `answer_contract.SPEC_SECTIONS`.
    failure           the run did not reach an answer. NOT submittable by a model: handing
                      one a give-up button measures the button. Declared anyway, because an
                      outcome space that omits the way runs actually end is not the space.

The `submittable: false` marking is how those last ones stay in the contract while staying
out of the model's enum.
"""
from __future__ import annotations

from typing import Any

KIND_VALUE = "value"
KIND_ABSTAIN_EVIDENCE = "abstain_evidence"
KIND_ABSTAIN_SPEC = "abstain_spec"
KIND_FAILURE = "failure"

#: Every kind some code path branches on. A contract may not invent one: a declared behaviour
#: that nothing implements is indistinguishable from one that ran and found nothing, which is
#: the same rule `answer_checks.ANSWER_CHECK_KINDS` holds for answer checks.
KINDS: tuple[str, ...] = (KIND_VALUE, KIND_ABSTAIN_EVIDENCE, KIND_ABSTAIN_SPEC, KIND_FAILURE)

#: What a contract that declares no `result:` block means. THE THREE STATUSES THIS REPO
#: SHIPPED WITH, unchanged, so that three of the four contracts in the tree keep their exact
#: behaviour and the default lives in one place instead of at every site that used to spell
#: it out. A contract overrides this by declaring its own space; it does not extend it.
DEFAULT_SPACE: dict[str, dict[str, Any]] = {
    "FOUND": {
        "kind": KIND_VALUE,
        "meaning": "a value is being returned and the evidence for it is in the ledger",
    },
    "EVIDENCE_INSUFFICIENT": {
        "kind": KIND_ABSTAIN_EVIDENCE,
        "meaning": "the specification is clear; this chart does not support an answer",
    },
    "SPEC_INSUFFICIENT": {
        "kind": KIND_ABSTAIN_SPEC,
        "meaning": ("the specification does not cover this case, or the variable is not "
                    "derivable from the notes at all"),
    },
}

class OutcomeSpaceError(ValueError):
    """A contract's `result.status` block does not describe a usable outcome space."""

def _block(spec: Any) -> dict[str, Any]:
    raw = getattr(spec, "result", None) or {}
    return raw if isinstance(raw, dict) else {}

def declared_statuses(spec: Any) -> dict[str, dict[str, Any]]:
    """status name -> its declaration, in the order the contract writes them.

    Order is kept because it reaches the model: the enum and the prose beneath it are
    rendered in this sequence, and an author who puts the value-carrying status first is
    saying something a sorted list would throw away.
    """
    declared = _block(spec).get("status")
    if not declared:
        return dict(DEFAULT_SPACE)
    if not isinstance(declared, dict):
        raise OutcomeSpaceError(
            f"{getattr(spec, 'spec_id', '?')}: result.status must be a MAPPING of status name "
            "to its declaration, not a bare list. The kind is the part code reads, and a list "
            "has nowhere to put it.")
    out: dict[str, dict[str, Any]] = {}
    for name, decl in declared.items():
        d = dict(decl) if isinstance(decl, dict) else {"meaning": str(decl)}
        kind = str(d.get("kind") or "")
        if kind not in KINDS:
            raise OutcomeSpaceError(
                f"{getattr(spec, 'spec_id', '?')}: result.status[{name}].kind is {kind!r}, which "
                f"nothing implements. One of: {', '.join(KINDS)}. A kind is not a label — it "
                "selects the obligations the gate applies, so an unknown one would mean an "
                "outcome that passes every check by not matching any of them.")
        if not str(d.get("meaning") or "").strip():
            raise OutcomeSpaceError(
                f"{getattr(spec, 'spec_id', '?')}: result.status[{name}] has no `meaning`. It is "
                "rendered to the model as the description of this outcome; a status code with "
                "no sentence under it is chosen by guesswork.")
        out[str(name)] = d
    return out

def submittable_statuses(spec: Any) -> tuple[str, ...]:
    """The enum `submit_answer` offers. Declaration order, `submittable: false` removed."""
    return tuple(n for n, d in declared_statuses(spec).items()
                 if d.get("submittable", True) is not False)

def status_kind(spec: Any, status: str) -> str | None:
    """The kind, or None for a status this contract does not declare.

    None rather than a default, and callers must treat it as a refusal. A status resolved to
    a plausible kind is exactly the silent acceptance this module was written to end.
    """
    d = declared_statuses(spec).get(str(status or ""))
    return str(d["kind"]) if d else None

def statuses_of_kind(spec: Any, *kinds: str) -> tuple[str, ...]:
    return tuple(n for n, d in declared_statuses(spec).items() if d.get("kind") in kinds)

def is_value(spec: Any, status: str) -> bool:
    return status_kind(spec, status) == KIND_VALUE

def abstention_statuses(spec: Any) -> tuple[str, ...]:
    return statuses_of_kind(spec, KIND_ABSTAIN_EVIDENCE, KIND_ABSTAIN_SPEC)

def default_evidence_abstention(spec: Any) -> str:
    """The status the RUNTIME writes when it downgrades an answer it will not let stand.

    A downgrade is the runtime's own conclusion, so it has to pick, and only the contract
    knows what there is to pick from. First-declared wins: declaration order is the author's
    statement of which abstention is the ordinary one, and for a contract that declares
    nothing this is `EVIDENCE_INSUFFICIENT` — exactly the literal it replaces.

    Falls back to the default space's name rather than raising: a runtime that could not
    finish an answer must not also fail to record that it could not.
    """
    found = statuses_of_kind(spec, KIND_ABSTAIN_EVIDENCE)
    return found[0] if found else "EVIDENCE_INSUFFICIENT"

def undeclared_status_message(spec: Any, status: str) -> list[str]:
    """What to tell an agent that submitted something outside the space.

    It names the whole space rather than saying "invalid", because this rejection is returned
    into a loop the agent has to recover in, and "that is not a status" without the list is
    an instruction to guess again.
    """
    space = declared_statuses(spec)
    lines = [(f"status={status!r} is not an outcome this specification declares. Resubmit "
              f"with one of: {', '.join(submittable_statuses(spec))}.")]
    lines += [f"  {n} — {d.get('meaning')}" for n, d in space.items()
              if d.get("submittable", True) is not False]
    return lines
