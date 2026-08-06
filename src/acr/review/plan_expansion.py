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

from .coverage_planner import CoveragePlan, ExpansionBudget


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

