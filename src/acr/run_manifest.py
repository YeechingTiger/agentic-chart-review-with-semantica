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

A FIELD THAT SUMMARISES TRACE EVENTS IS DERIVED FROM THEM, BY ONE FUNCTION
-------------------------------------------------------------------------
The replan block used to be read off counters `graph.py` incremented as the run went. The
trace held the same quantities as events. Two counters for one quantity, and on the first
true end-to-end run they read differently in a way nobody could see: 14 `plan_revision`
events, 13 of them `applied: true`, against a manifest saying `n_revisions_applied: 0`.

Both were arithmetically right. `applied` on the event means the revision was ADMISSIBLE;
`n_revisions_applied` means it MOVED RETRIEVAL, and on that run all 13 admitted revisions
moved nothing — the agent kept re-promoting types already at `read_all`. The damage was not
the arithmetic. It was that the manifest had no field for "how many times did the agent
reach for this channel", so a run that reached 14 times and a run that never reached
rendered identically as `replan_rate: 0.0`, and 0.0 was written up as "the model ignores the
replanning channel".

So `replan_from_trace` below is the single definition of every one of those numbers, it
reads the events and nothing else, and `build_manifest` publishes exactly what it returns.
The runtime counters still exist — `graph.py` owns them — but they are no longer published:
they are CROSS-CHECKED against the derivation, and the comparison ships inside the block, so
the next divergence is visible to a reader who never runs the test suite.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .coverage_planner import MONOTONICITY_VS_LEDGER


@dataclass
class RunCounters:
    """Everything one run counts about its own replanning and its own degradation.

    A dataclass and not loose attributes because the DEGRADATION block's contract is "any
    non-zero entry means a node degraded and the corresponding behaviour was NOT exercised",
    and a contract enforced by `getattr(self, "_name", 0)` at the reporting site is a
    contract that silently reports zero for a counter somebody renamed.
    """
    # --- replanning: how much the run had to correct the assets it was given.
    # NOT PUBLISHED ANY MORE. `replan_from_trace` derives these from the trace events and the
    # manifest publishes that; these five are kept because `graph.py` owns them and because a
    # second, independently produced copy is a useful cross-check — but a cross-check is all
    # it is, and `_cross_check` reports any disagreement into the block rather than letting
    # one of the two quietly win. Two counters for one quantity is what produced the SYN0001
    # divergence; this is the pair kept honest by being compared instead of chosen between.
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


# ==========================================================================================
#                      THE REPLAN BLOCK, DERIVED FROM THE TRACE EVENTS
# ==========================================================================================
#: Event kinds the derivation reads. Named here so a rename in `graph.py` breaks loudly at
#: the one place that reads them rather than silently deriving zero from a kind that no
#: longer exists — which is the same failure mode, one layer down, as the counter drift.
EV_REFLECT = "reflect"
EV_REVISION = "plan_revision"
EV_TRIGGER = "trigger"
EV_REFUSED_OPEN = "plan_refused_open"
EV_ASSERTED_REPLAN = "model_asserted_replan"


def _moved_retrieval(outcome: Mapping[str, Any]) -> bool:
    """Did this revision change what the agent may open or is told to search?

    The same predicate as `coverage_planner.RevisionOutcome.changed_retrieval`, restated over
    the serialised event because THAT is what a reader of a finished run holds. Thread
    bookkeeping is deliberately excluded: resolving a thread is not a replan.
    """
    return bool(outcome.get("terms_added") or outcome.get("types_promoted"))


def replan_from_trace(events: Iterable[Mapping[str, Any]]) -> dict:
    """Every replan number of one run, recomputed from that run's own trace events.

    THE ONE DEFINITION. Nothing else in this tree may count these quantities; a second
    accumulator beside this one is the defect this function exists to make impossible.

    The counts are split because they answer different questions and a single "applied"
    could not carry all of them:

      n_revision_requests      the agent asked. THE NUMBER THAT WAS MISSING. A zero
                               `replan_rate` with a high request count means the plan was
                               already adequate for what the agent wanted; a zero rate with
                               a zero request count means the channel went unused. Those are
                               opposite diagnoses and they used to render identically.
      n_revisions_admitted     the request passed monotonicity, budget and redundancy — the
                               trace event's own `applied` flag, under a name that does not
                               promise the plan moved.
      n_revisions_applied      the retrieval scope actually moved. This is the numerator of
                               `replan_rate` and its meaning is unchanged.
      n_revisions_no_op        admitted and moved nothing. Re-promoting a type already at
                               `read_all`, re-opening an open thread. Invisible before.
      n_revisions_refused      refused whole.
      n_revisions_partly_refused
                               admitted, but carrying refusal prose on part of the request.
                               The agent was told "no" about something and no number said so.
    """
    evs = list(events)
    by_kind: dict[str, list[Mapping[str, Any]]] = {}
    for e in evs:
        by_kind.setdefault(str(e.get("kind", "")), []).append(e)

    reflections = len(by_kind.get(EV_REFLECT, []))
    revisions = by_kind.get(EV_REVISION, [])
    outcomes = [dict(e.get("outcome") or {}) for e in revisions]
    admitted = [o for e, o in zip(revisions, outcomes) if e.get("applied")]
    moved = [o for o in admitted if _moved_retrieval(o)]
    refused = [o for e, o in zip(revisions, outcomes) if not e.get("applied")]

    triggers: dict[str, int] = {}
    for e in by_kind.get(EV_TRIGGER, []):
        k = str(e.get("trigger", ""))
        triggers[k] = triggers.get(k, 0) + 1

    def _rate(n: int) -> float | None:
        return round(n / reflections, 4) if reflections else None

    return {
        "n_reflections": reflections,
        "n_revision_requests": len(revisions),
        "n_revisions_admitted": len(admitted),
        "n_revisions_applied": len(moved),
        "n_revisions_no_op": len(admitted) - len(moved),
        "n_revisions_refused": len(refused),
        "n_revisions_partly_refused": sum(1 for o in admitted if o.get("refused")),
        # Applied revisions that CHANGED RETRIEVAL, over reflections. Thread bookkeeping is
        # deliberately not counted: resolving a thread is not a replan, and counting it would
        # reinflate the metric with no-ops.
        "replan_rate": _rate(len(moved)),
        # NOT the replan rate and it must never be read as one. This one says the agent
        # REACHED for the channel; `replan_rate` says the plan MOVED. A design conclusion
        # about whether to ship an agent or a workflow needs both, and reading the second
        # while believing it was the first is exactly how a wrong conclusion got written.
        "request_rate": _rate(len(revisions)),
        "terms_added_by_reflection": sum(len(o.get("terms_added") or []) for o in outcomes),
        "types_promoted": sum(len(o.get("types_promoted") or []) for o in outcomes),
        "threads_opened_by_reflection": sum(len(o.get("threads_opened") or [])
                                            for o in outcomes),
        "triggers_fired": triggers,
        "plan_refused_opens": len(by_kind.get(EV_REFUSED_OPEN, [])),
        "model_asserted_replan": len(by_kind.get(EV_ASSERTED_REPLAN, [])),
        "derived_from": "trace_events",
    }


#: Derived key -> the `RunCounters` attribute that used to be published in its place. The
#: cross-check is by name so a counter somebody renames shows up as a missing attribute here
#: rather than as a silent zero — the same reason `RunCounters` is a dataclass.
_COUNTER_CROSS_CHECK = {
    "n_reflections": "reflections",
    "n_revisions_applied": "revisions_applied",
    "n_revisions_refused": "revisions_refused",
    "plan_refused_opens": "plan_refused_opens",
    "model_asserted_replan": "model_asserted_replan",
}


def _cross_check(derived: Mapping[str, Any], counters: RunCounters) -> dict:
    """Compare the derivation to the counters `graph.py` still keeps, and report it.

    Never raises and never corrects: a manifest builder that could take down a finished run
    would lose the very artifact the disagreement has to be read from. The disagreement goes
    in the block, where a directory-wide filter can find it.
    """
    bad = {k: {"trace": derived[k], "counter": getattr(counters, attr)}
           for k, attr in _COUNTER_CROSS_CHECK.items()
           if derived[k] != getattr(counters, attr)}
    return {"counters_agree": not bad, "counter_disagreements": bad}


def _cross_check_plan(derived: Mapping[str, Any], plan, cross: dict) -> dict:
    """Fold the PLAN OBJECT's own tallies into the same cross-check.

    `plan.term_provenance` and `plan.promotion_log` are the third copy of quantities the
    trace already holds — the ledger the revisions were applied against. They agree by
    construction today, which is exactly why a divergence would go unnoticed.
    """
    plan_side = {
        "terms_added_by_reflection": sum(
            1 for r in plan.term_provenance if r.get("trigger") != "planner_proposal"),
        "types_promoted": len(plan.promotion_log),
    }
    for k, v in plan_side.items():
        if derived[k] != v:
            cross["counters_agree"] = False
            cross["counter_disagreements"][k] = {"trace": derived[k], "plan_object": v}
    return cross


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
    # DERIVED, not accumulated. See the module docstring and `replan_from_trace`.
    replan = replan_from_trace(tracer.events)
    # The declared trigger key space comes from the runtime (every kind, zeroed), the VALUES
    # come from the trace. A kind that never fired must still appear — "it fired zero times"
    # and "this build does not have that trigger" are different findings — but the number
    # beside it is the one the events support.
    declared_triggers = {k: 0 for k in triggers_fired}
    replan["triggers_fired"] = {**declared_triggers, **replan["triggers_fired"]}
    cross = _cross_check(replan, counters)
    if dict(triggers_fired) != replan["triggers_fired"]:
        cross["counters_agree"] = False
        cross["counter_disagreements"]["triggers_fired"] = {
            "trace": replan["triggers_fired"], "counter": dict(triggers_fired)}
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
        # EVERY NUMBER BELOW IS RECOMPUTED FROM THIS RUN'S OWN TRACE by `replan_from_trace`,
        # which is the only place in the tree that counts them. Read `n_revision_requests`
        # BEFORE `replan_rate`: a rate of 0.0 over 14 requests says the plan was already
        # adequate for everything the agent asked for, and a rate of 0.0 over 0 requests says
        # the channel went unused. Those are opposite conclusions about whether to ship an
        # agent or a workflow, and until `n_revision_requests` existed they rendered the same.
        "replan": {
            **replan,
            # NOT a trace-event summary, so not derived above: this is the plan object's own
            # state, and the split is between two different facts. Everything the SPEC did
            # not declare counts against the spec; only what a REFLECTION added counts as
            # replanning. The up-front planner's own proposals are in this number and not in
            # `terms_added_by_reflection` — a term the planner supplied before the run began
            # is evidence about the spec, not evidence that the agent learned something. On
            # SYN0001 this read 84 against a `terms_added_by_reflection` of 0, which is the
            # whole of that distinction in two numbers.
            "terms_added": len(plan.terms_added()),
            # The plan object's own copies, kept beside the derived ones for the same reason
            # the counters are: so a divergence is a visible fact rather than a silent choice.
            **_cross_check_plan(replan, plan, cross),
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
