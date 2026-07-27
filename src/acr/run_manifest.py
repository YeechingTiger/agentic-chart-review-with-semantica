"""Assemble the manifest a finished run leaves behind: the numbers a reader who was not
there has to be able to audit, in the shape a filter over a DIRECTORY of runs can read.

WHY THIS IS A MODULE AND NOT THE TAIL OF `run()`
------------------------------------------------
Every field here exists because a question was asked over a directory of finished runs and
the answer was only recoverable by replaying JSONL — which is to say, was never computed.
The replan rate, the seed provenance, the degradation block: each one is a number somebody
read wrongly once. Assembling them beside the loop that produces them invites the loop's
convenience to decide the record's shape; assembling them here forces every addition to
answer "what would a reader conclude from this six months from now".

It is a pure function of what the run finished with. It reads no ledger it was not handed
and it decides nothing — in particular it never recomputes a verdict, because a manifest
that can disagree with the gate is a second gate with better formatting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .coverage_planner import MONOTONICITY_VS_LEDGER


@dataclass
class RunCounters:
    """Everything one run counts about its own replanning and its own degradation.

    A dataclass and not loose attributes because the DEGRADATION block's contract is "any
    non-zero entry means a node degraded and the corresponding behaviour was NOT exercised",
    and a contract enforced by `getattr(self, "_name", 0)` at the reporting site is a
    contract that silently reports zero for a counter somebody renamed.
    """
    # --- replanning: how much the run had to correct the assets it was given
    reflections: int = 0
    revisions_applied: int = 0
    revisions_refused: int = 0
    plan_refused_opens: int = 0
    #: A supervisor still reaching for a verdict that no longer exists is a fact about the
    #: prompt, not about the chart.
    model_asserted_replan: int = 0
    #: null when the gate never passed; the manifest then says which condition blocked.
    steps_to_gate_pass: int | None = None

    # --- degradation: nodes that fell back, i.e. behaviour that was NOT exercised
    plan_fallbacks: int = 0
    reflect_fallbacks: int = 0
    finalize_defaults: int = 0
    act_no_tool_call: int = 0
    #: A SPEC_INSUFFICIENT nobody can route is the channel half-working, which historically
    #: reads as the channel working.
    spec_gaps_unroutable: int = 0


@dataclass
class SeedRecord:
    """The validation draw's seed and where it came from, kept together so neither travels
    without the other: a seed the caller chose is a seed the caller could have chosen again.
    """
    effective: int
    provenance: str
    caller_supplied: bool


@dataclass
class ExpansionRecord:
    """What the expansion budget was, and both halves of how it ended."""
    report: dict
    #: SPENT, not merely bumped into. A single refusal used to read as exhaustion forever;
    #: `terms_deferred` is the other half of the same fact now that a term overrun trims
    #: rather than refuses, and both are reported.
    exhausted: bool
    refused_at_least_once: bool
    terms_deferred: list[str] = field(default_factory=list)


def build_manifest(*, spec, patient_id: str, model: str, plan, coverage, threads, markers,
                   tracer, final: dict, counters: RunCounters, expansion: ExpansionRecord,
                   seed: SeedRecord, triggers_fired: dict[str, int], usage: dict,
                   elapsed_s: float) -> dict:
    """The finished run, as a record. Pure: it computes no verdict and touches no ledger."""
    answer = final.get("answer", {})
    return {
        "run_id": tracer.run_id,
        "patient_id": patient_id,
        **spec.identity(),
        "model": model,
        "answer": answer,
        # THE ONE PLAN, in the shape that governed retrieval. Both term lists are inside
        # it and they are never merged: `initial_keywords` is the spec's, and it is the
        # baseline the develop plane scores against; `keywords` is the final expanded
        # list, and it is what coverage was evaluated against.
        "plan": plan.to_dict(),
        "plan_revisions": final.get("plan_revisions", 0),
        # ============================================================================
        # THE REPLAN RATE, AND WHY IT IS IN THE MANIFEST
        # ============================================================================
        # This is the health metric for the PRIOR, not for the agent. High means the
        # assets are underdeveloped — the spec's term list and stratum declarations are
        # being corrected at run time, once per patient, at inference cost. Near zero
        # WITH good assets means the agentic layer has nothing left to do on this
        # criterion and the honest conclusion is to ship a workflow, not an agent.
        # It is here rather than only in the trace because that question is asked over a
        # DIRECTORY of finished runs, and a number recoverable only by replaying JSONL is
        # a number nobody computes. The old REPLAN counter answered it 0/291 times, which
        # read as "the agent is stable" and actually meant "the verdict did nothing".
        "replan": {
            "n_reflections": counters.reflections,
            "n_revisions_applied": counters.revisions_applied,
            "n_revisions_refused": counters.revisions_refused,
            # Applied revisions that CHANGED RETRIEVAL, over reflections. Thread
            # bookkeeping is deliberately not counted: resolving a thread is not a
            # replan, and counting it would reinflate the metric with no-ops.
            "replan_rate": (round(counters.revisions_applied / counters.reflections, 4)
                            if counters.reflections else None),
            # Split, because they are two different facts. Everything the SPEC did not
            # declare counts against the spec; only what a REFLECTION added counts as
            # replanning. The up-front planner's own proposals are in the first number
            # and not the second — a term the planner supplied before the run began is
            # evidence about the spec, not evidence that the agent learned something.
            "terms_added": len(plan.terms_added()),
            "terms_added_by_reflection": sum(
                1 for r in plan.term_provenance if r["trigger"] != "planner_proposal"),
            "types_promoted": len(plan.promotion_log),
            "triggers_fired": dict(triggers_fired),
            "plan_refused_opens": counters.plan_refused_opens,
            "model_asserted_replan": counters.model_asserted_replan,
        },
        # HARVESTED FROM PRODUCTION, with the trace that produced each one. Every term
        # and every promotion here is a candidate edit to the spec, carrying the step,
        # the trigger that forced it and the observation that caused it — which is the
        # difference between a develop-plane input and a list of words.
        "develop_plane_candidates": {
            "spec_declared_terms": list(plan.initial_keywords),
            "terms_added_at_runtime": list(plan.term_provenance),
            "types_promoted_at_runtime": list(plan.promotion_log),
            "refused_revisions": list(plan.refused_revisions),
            # A term the run ASKED FOR and the budget could not pay for is evidence about
            # the spec's list too, and partial application is exactly what stops it from
            # landing in `refused_revisions` (the revision was applied; only the tail of
            # its term list was not). It would otherwise disappear from the harvest.
            "terms_deferred_for_budget": list(expansion.terms_deferred),
            "what_this_is": ("candidate spec edits observed on a real chart. Score the "
                             "spec's list against spec_declared_terms, NEVER against the "
                             "expanded list — a runtime rescue that is folded back into "
                             "the baseline erases the evidence that the baseline was wrong"),
            "trace": str(tracer.path),
        },
        "open_threads": {**threads.to_dict(),
                         "marker_catalogue": markers.source,
                         "marker_catalogue_degraded": markers.degraded or None},
        "expansion_budget": {**expansion.report,
                             "exhausted": expansion.exhausted,
                             "refused_at_least_once": expansion.refused_at_least_once,
                             "terms_deferred": list(expansion.terms_deferred)},
        "monotonicity_vs_ledger": MONOTONICITY_VS_LEDGER,
        "steps": final.get("step", 0),
        "negative_basis": (final.get("answer") or {}).get("negative_basis"),
        "gate_validated": bool(final.get("gate_validated")),
        # What the spec's provenance permits this run to CLAIM, which is a separate
        # question from whether the gate passed. The gate proves the search was done; it
        # cannot prove the search terms were the right ones, and on this corpus STORE.400's
        # five terms miss the diagnosis for 31.7% of patients while the gate stays clean.
        # `reportable_as_validated` inside this block is the field a downstream filter
        # should read, never `gate_validated` alone.
        "provenance": spec.provenance_for_run(
            answer.get("value") or {}, str(answer.get("status") or ""),
            gate_validated=bool(final.get("gate_validated"))),
        # The seed and where it came from, in the manifest, always. A run whose
        # provenance is `caller_supplied` was sampled with a number the operator chose,
        # and a reader who cannot see that cannot tell a reproduced draw from a shopped
        # one. `seed_is_caller_supplied` is the flag; it is redundant with the string on
        # purpose, because a boolean is what a filter over a directory of manifests reads.
        "sample_seed": seed.effective,
        "seed_provenance": seed.provenance,
        "seed_is_caller_supplied": seed.caller_supplied,
        "steps_to_gate_pass": counters.steps_to_gate_pass,
        "suspected_recognition_failures": len(coverage.suspected_recognition_failures),
        "rejections": final.get("rejections", []),
        # WHICH SPEC RULE WAS IN PLAY. In the manifest as well as the trace, because the
        # §6b loop reads a directory of finished runs: attribution recoverable only by
        # replaying a JSONL file is attribution nobody computes. Two separately named
        # channels inside, never merged — a check that provably fired and a rule the
        # model says it applied must not be readable as the same kind of fact.
        "rule_attribution": tracer.rule_attribution(),
        "usage": usage,
        # Any non-zero entry here means a node degraded silently and the corresponding
        # behaviour was NOT exercised. Read every conclusion against this block first —
        # a non-zero `plan_fallbacks`, for instance, means the run never exercised the
        # planner and no conclusion about planning may be drawn from it.
        "degradation": {
            "plan_fallbacks": counters.plan_fallbacks,
            "reflect_fallbacks": counters.reflect_fallbacks,
            "finalize_defaults": counters.finalize_defaults,
            "act_no_tool_call": counters.act_no_tool_call,
            # A SPEC_INSUFFICIENT nobody can route belongs in the block a reader is told
            # to check before believing anything else in the manifest.
            "spec_gaps_unroutable": counters.spec_gaps_unroutable,
            # An INT, like every other entry here, because the block's contract is "any
            # non-zero value means a node degraded and the behaviour was not exercised" —
            # and a filter over a directory of manifests reads that contract, not prose.
            # The prose lives beside the thread ledger. Non-zero means the marker set was
            # incomplete, so thread detection — the control that exists to catch the 8046
            # error — was running blind in part.
            "marker_catalogue_incomplete": int(bool(markers.degraded)),
        },
        "elapsed_s": elapsed_s,
        "trace": str(tracer.path),
    }
