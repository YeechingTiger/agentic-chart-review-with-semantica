"""The arithmetic of the expansion budget: what a monotone widening of the one plan costs,
how much of a revision fits in what is left, and when widening is over.

Pure functions over a `CoveragePlan` and an `ExpansionBudget`, holding no run state, because
every number here has been wrong once by being computed in two places against two different
notions of what counts. The counting rule is stated once — `max_terms_added` is measured
against EVERY `term_provenance` row — and every caller is priced by the same rule.

`CoveragePlan.apply_revision` remains the only thing that MUTATES a plan; this module decides
only what may be afforded, so the semantics of a revision stay defined in one place.
"""
from __future__ import annotations

from dataclasses import replace as _dc_replace

from .coverage_planner import CoveragePlan, ExpansionBudget, PlanRevision


def price_expansion_budget(plan: CoveragePlan, docs_by_type: dict[str, int], *,
                           max_revisions: int, supplied: ExpansionBudget | None,
                           planner_terms: int) -> ExpansionBudget:
    """The caps, in the units the plan actually counts them in.

    THE PLANNER'S OWN TERMS ARE NOT THE AGENT'S ALLOWANCE. `priced_against` caps
    `max_terms_added` at the number of SPEC-declared terms — "needing to more than double
    the spec's list is a spec that was wrong" — but `plan.terms_added()` counts every
    `term_provenance` row, and the up-front planner's proposals are rows like any other
    (the planner is asked for a keyword list, and `plan_coverage` records what it gets as
    the run's first monotone addition, on purpose: a term the planner supplied is
    evidence about the SPEC). Charging those to the agent spent the whole allowance
    before the first reflection — the production path showed `EXPANSION REMAINING:
    terms -3` at step 1, refused the first single-term revision BUDGET_EXHAUSTED, and
    ended the run at step 1 of 12.

    The manifest already keeps `terms_added` and `terms_added_by_reflection` apart. This
    makes the BUDGET keep them apart too, by offsetting the cap by the rows that were
    already there. The alternative — teach the counter to skip `planner_proposal` rows —
    lives in `CoveragePlan.apply_revision`, which this module does not own.
    """
    priced = supplied or ExpansionBudget.priced_against(
        plan, docs_by_type, max_revisions=max_revisions)
    if not planner_terms:
        return priced
    return _dc_replace(priced, max_terms_added=priced.max_terms_added + planner_terms)


def headroom(plan: CoveragePlan, budget: ExpansionBudget) -> dict[str, int]:
    """What is left of each cap, counted exactly the way `apply_revision` prices it."""
    return {"terms": budget.max_terms_added - len(plan.terms_added()),
            "type_promotions": budget.max_type_promotions - len(plan.promotion_log),
            "revisions": budget.max_revisions - plan.revisions_applied}


def budget_report(plan: CoveragePlan, budget: ExpansionBudget, *, source: str,
                  planner_terms: int) -> dict:
    """The budget as a reader must see it: both halves of the term cap, named."""
    return {**budget.to_dict(),
            "source": source,
            "planner_proposed_terms": planner_terms,
            "max_terms_added_by_reflection": budget.max_terms_added - planner_terms,
            "terms_are_counted": (
                "`max_terms_added` is measured against EVERY term_provenance row, so the "
                "up-front planner's own proposals are added to the cap rather than "
                "charged against the agent's reflection allowance"),
            "headroom": headroom(plan, budget)}


def fit_terms_to_budget(rev: PlanRevision, plan: CoveragePlan,
                        budget: ExpansionBudget) -> tuple[PlanRevision, list[str]]:
    """Trim `add_terms` to the remaining allowance, keeping the supervisor's own order.

    WHY PARTIAL AND NOT ALL-OR-NOTHING. All-or-nothing is right for MONOTONICITY — a
    revision that also demoted a type must be refused whole, because applying its
    admissible half hands back a plan the agent did not propose. A budget overrun is a
    different kind of failure: nothing about the requested terms is inadmissible, there
    is simply not enough allowance for all of them, and every prefix of the list is a
    plan the agent DID propose. Refusing the whole thing taught the agent nothing, marked
    the plan exhausted forever, and ended the run — over one term.

    The order is the model's own priority order, and the truncation is reported back by
    name, so "you may have 5 of the 6" is a statement the agent can act on. Terms already
    in the plan are kept and cost nothing: `apply_revision` prices only what is new.
    """
    room = max(0, headroom(plan, budget)["terms"])
    kept: list[str] = []
    deferred: list[str] = []
    for t in rev.add_terms:
        if t in plan.keywords:
            kept.append(t)
        elif room > 0:
            kept.append(t)
            room -= 1
        else:
            deferred.append(t)
    if not deferred:
        return rev, []
    return _dc_replace(rev, add_terms=tuple(kept)), deferred


def expansion_is_spent(plan: CoveragePlan, budget: ExpansionBudget, *,
                       terms_deferred: list[str]) -> bool:
    """Widening is over: the agent has BUMPED INTO the cap and nothing is left to widen.

    The second half is what un-sticks the old rule. `plan.budget_exhausted` is true
    forever after a single budget refusal, so one term too many ended the run even when
    the plan could still promote a type — the widening move that was usually the one
    actually needed. Now a refusal (or a term the budget deferred: with partial
    application, a term overrun no longer records a refusal at all) only ARMS the check,
    and the run stops only when nothing can widen any further:

      * neither a term nor a promotion has any allowance left, or
      * no revisions remain, in which case nothing can be applied whatever the room.

    The first half is kept exactly as it was: a run that never tried to widen is not
    exhausted, however small its budget.
    """
    bumped = plan.budget_exhausted(budget) or bool(terms_deferred)
    if not bumped:
        return False
    h = headroom(plan, budget)
    return (h["terms"] <= 0 and h["type_promotions"] <= 0) or h["revisions"] <= 0


# ------------------------------------------------------------------ the plan in the prompt
#: Prefixes of a rendered plan block. Model-visible on purpose: the agent reads them as a
#: heading, and the runtime uses them to find the block it must replace.
#:
#: NOT the bare word "PLAN". `"PLANNING the next read".startswith("PLAN")` is true, and a
#: marker that matches ordinary prose deletes the agent's own words out of its transcript —
#: a silent amnesia that would look like the model forgetting rather than like a bug here.
#: Every header the runtime writes ends the word with `:` or ` (`.
PLAN_BLOCK_PREFIXES = ("PLAN:", "PLAN (")


def install_plan_block(msgs: list[dict], body: str) -> list[dict]:
    """Put the plan in the transcript ONCE, at the end, replacing any earlier copy.

    THE PLAN IS STATE, NOT HISTORY, and appending it was the single largest line item in the
    bill. Measured on a real 293-document chart: `plan.render()` is 6,310 characters, the run
    appended it eleven times — six from the plan node, five from `reflect` announcing an
    applied revision — and every copy was re-sent on all forty-nine subsequent calls. That is
    ~425,000 of the run's 1,030,179 prompt tokens, 41%, spent re-reading ten stale copies of
    a document whose current version was sitting at the bottom of the same prompt.

    Only the CURRENT plan is a fact about the run. What changed between revisions is history
    and stays in the transcript, as the one-line note `reflect` writes — a reader of the
    thread can still see that the scope widened and why, without carrying the full listing
    of every doc type and keyword for each step it was true.

    Position matters as much as uniqueness: the block goes at the END, because the plan
    governs the next tool call and a plan buried twenty messages up is a plan being recalled
    rather than read.
    """
    kept = [m for m in msgs if not is_plan_block(m)]
    return kept + [{"role": "user", "content": body}]


def is_plan_block(m: dict) -> bool:
    """A message the runtime wrote to carry the plan — never one the model wrote."""
    return (m.get("role") == "user"
            and str(m.get("content") or "").startswith(PLAN_BLOCK_PREFIXES))
