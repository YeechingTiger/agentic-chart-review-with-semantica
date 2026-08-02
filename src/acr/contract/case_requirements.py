"""What a contract requires of the CASE before a run of it means anything.

The counterpart to `acr.core.case_context`: that holds what was supplied, this reads what the
contract demands. Two declarations, both under a `case_context:` block:

    requires_target_entity   the question presupposes a choice among entities the chart may
                             hold more than one of ("this tumour", "the index admission").
                             When true and none is supplied, the run ends before it reads
                             anything.
    time_anchorable          whether a time window around an anchor date means anything for
                             this target. False for a target that IS a point in time: the only
                             anchor available would be the answer.

WHY THE REFUSAL IS BEFORE THE READ, and not a rejection at submit time. An unresolved referent
is not a retrieval problem. No amount of reading settles which tumour "this tumour" is, so a
run that searches first and then discovers it cannot say has spent a budget to arrive where it
started — and, worse, will usually have found A tumour and answered about that one. Three runs
did exactly that: one coded a sigmoid colon hyperplastic polyp in a lung-cancer chart, two
picked the wrong lung lesion in a chart documenting two. In none of them could the trace
distinguish "did not notice" from "noticed and judged" from "the gold row cannot hold two
tumours".

The refusal is `TARGET_ENTITY_UNCLEAR`, declared `kind: failure` in the outcome space: the run
did not reach an answer, and it is not an answer a model may claim. What the MODEL says when
it finds two lesions it cannot resolve is `reported_lesion`, which already exists.
"""
from __future__ import annotations

from typing import Any

#: Status for a run that could not start. See `acr.contract.outcomes` for why it is a
#: `failure` rather than an abstention: nothing about the chart or the contract is being
#: reported, and no coverage or spec-gap proof would make sense of it.
TARGET_ENTITY_UNCLEAR = "TARGET_ENTITY_UNCLEAR"


def _block(spec: Any) -> dict:
    raw = getattr(spec, "case_context", None) or {}
    return raw if isinstance(raw, dict) else {}


def requires_target_entity(spec: Any) -> bool:
    """Default FALSE. Silence means the question stands on its own."""
    return bool(_block(spec).get("requires_target_entity", False))


def is_time_anchorable(spec: Any) -> bool:
    """Default TRUE, and the asymmetry with the switch above is deliberate.

    An undeclared contract must not be quietly STRICTER than a declared one. Refusing a window
    by default would make every existing contract reject a scope it had never objected to,
    which is a behaviour change dressed as a default.
    """
    return bool(_block(spec).get("time_anchorable", True))


def refuse_before_reading(spec: Any, case: Any) -> dict | None:
    """The answer this run must return without reading anything, or None to proceed.

    Returns a whole answer rather than a boolean because the caller's job is then only to
    record it: a caller that had to assemble the refusal itself would be a second place where
    the shape of a failed run is decided.
    """
    if requires_target_entity(spec) and not (getattr(case, "target_entity", None) or "").strip():
        return {
            "status": TARGET_ENTITY_UNCLEAR,
            "value": {},
            "evidence": [],
            "reasoning": (
                f"{getattr(spec, 'spec_id', 'this contract')} declares "
                "`case_context.requires_target_entity`, and no target_entity was supplied for "
                f"{getattr(case, 'patient_id', 'this case')}. The question presupposes a "
                "choice this chart may not make on its own, and no amount of reading resolves "
                "a referent. Supply the entity, or use a contract whose question stands "
                "without one."),
            "proof_basis": "NOT_APPLICABLE",
            "coverage_note": ("no coverage claim is made — the run did not begin, so nothing "
                              "about this chart is being reported"),
        }
    return None
