"""Every conflict rule the contract declares must reach a recorded state.

A DERIVED VIEW, NOT A LEDGER. `assess_rules` is a pure function of the contract, the assessments a
run recorded, and the discriminating facts it resolved. This tree has exactly one composition point
for "what is still outstanding" — `outstanding_obligations` — and it exists because *"when the old
runtime had two ways to compute it they disagreed about whether a run had finished"*. A stored
assessment ledger would be a second such account, so there isn't one.

## The hole this closes

Deriving obligations from the contract rather than from a candidate set removes one funnel:

    candidate never formed -> conflict never formed -> discriminator never checked

It does nothing about the next one unless every declared rule is accounted for:

    applicability never recognised -> obligation never created
                                   -> the column reads applicable:false
                                   -> the run LOOKS complete

So the default is `not_considered`, and a run that assessed nothing reports four unassessed rules
rather than four inapplicable ones. `not_considered` and `not_applicable` are the two states this
module exists to keep apart: the first is a gap in the review, the second is a judgement about the
chart, and they are the same bytes in any representation that records only the judgements it received.

## `not_considered` is only interpretable against a gold annotation

Four rules over twenty-seven charts: most rules genuinely do not apply to most charts, so
`not_considered` will be the COMMON state and this column is mostly noise read on its own. The
measurable is `not_considered AND the gold annotation says this rule was applicable`. Until
`RuleGoldAnnotation` exists, this view buys visibility with no way to tell a miss from a correct
silence — which is worth knowing before anyone reads a rate off it.

## Why an assessment with no evidence basis is recorded rather than refused

The completeness column asks `applicability_has_evidence_basis` as its own question. Refusing an
unevidenced judgement would make the only path to recording one a path that requires evidence the run
may not have cited yet — and an unrecorded judgement is exactly the hole above. So it is recorded,
flagged, and counted.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

#: The closed vocabulary. `not_considered` first because it is the default and the one the fix is
#: about; `potentially_applicable` is kept distinct from `applicable` because collapsing it into
#: either neighbour loses the reason an obligation was opened.
STATUSES: tuple[str, ...] = (
    "not_considered",
    "not_applicable",
    "potentially_applicable",
    "applicable_checked",
    "applicable_unresolved",
)

#: What a run may record about a rule. `applicable_checked` / `applicable_unresolved` are DERIVED from
#: an `applicable` judgement plus the facts resolved, never stated directly: a run does not get to
#: declare its own work finished.
DECLARABLE: tuple[str, ...] = ("not_applicable", "potentially_applicable", "applicable")


class RuleAssessmentError(ValueError):
    """A recorded assessment names a rule or a fact this contract does not declare."""


@dataclass(frozen=True)
class RuleAssessment:
    """One conflict rule's standing in this run."""

    rule_id: str
    status: str
    #: The facts this rule turns on, from the CONTRACT — not from the run. A rule that turns on
    #: nothing (a residual tie-break) has an empty tuple and is `applicable_checked` the moment it
    #: is judged applicable, because applying it requires checking nothing.
    turns_on: tuple[str, ...] = ()
    #: Of those, the ones no resolution has closed.
    unresolved_facts: tuple[str, ...] = ()
    applicability_basis_evidence_ids: tuple[str, ...] = ()
    rationale: str = ""

    @property
    def has_evidence_basis(self) -> bool:
        """Reported, never enforced — see the module docstring."""
        return bool(self.applicability_basis_evidence_ids)

    @property
    def considered(self) -> bool:
        return self.status != "not_considered"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["has_evidence_basis"] = self.has_evidence_basis
        d["considered"] = self.considered
        return d


def declared_facts(spec) -> dict[str, dict]:
    """`fact_id -> the declared fact`, from the contract."""
    return {str(f.get("fact_id")): dict(f)
            for f in (getattr(spec, "discriminating_facts", None) or [])
            if isinstance(f, Mapping) and str(f.get("fact_id") or "").strip()}


def rule_ids(spec) -> list[str]:
    """`conflict_rule.N` in declaration order — the same ids `rule_catalog` mints and the prompt
    renders, so an assessment can be joined to a citation."""
    return [f"conflict_rule.{i}"
            for i, _ in enumerate(getattr(spec, "conflict_rules", None) or [], start=1)]


def _turns_on(rule: Any) -> tuple[str, ...]:
    return tuple(str(x) for x in ((rule or {}).get("turns_on") or [])) \
        if isinstance(rule, Mapping) else ()


def assess_rules(spec, *, declared: Mapping[str, Mapping[str, Any]] | None = None,
                 resolved_facts: Iterable[str] = ()) -> list[RuleAssessment]:
    """One row per declared conflict rule, in `conflict_rule.N` order.

    `declared` is what the run recorded, keyed by rule id, each `{assessment, evidence_ids?,
    rationale?}`. `resolved_facts` is the set of `fact_id`s a resolution has closed.

    Refuses an assessment or a resolution naming something the contract does not declare. Silently
    dropping either would make this column's denominator depend on what the model happened to say,
    which is the defect the column exists to measure.
    """
    facts = declared_facts(spec)
    rules = list(getattr(spec, "conflict_rules", None) or [])
    ids = rule_ids(spec)
    given = {str(k): dict(v) for k, v in (declared or {}).items()}

    if unknown := sorted(set(given) - set(ids)):
        raise RuleAssessmentError(
            f"assessment(s) for {unknown}, which {getattr(spec, 'spec_id', '?')} does not declare. "
            f"Declared: {ids or '(none)'}")
    resolved = {str(x) for x in resolved_facts}
    if ghosts := sorted(resolved - set(facts)):
        raise RuleAssessmentError(
            f"resolution(s) against discriminating fact(s) {ghosts}, which this contract does not "
            f"declare. A resolution against a fact nobody declared would move a rule to `checked` "
            f"without anything having been checked. Declared: {sorted(facts) or '(none)'}")

    out: list[RuleAssessment] = []
    for rid, rule in zip(ids, rules):
        turns = _turns_on(rule)
        unresolved = tuple(f for f in turns if f not in resolved)
        rec = given.get(rid)
        if rec is None:
            # THE DEFAULT, and the point of the whole module.
            out.append(RuleAssessment(rule_id=rid, status="not_considered", turns_on=turns,
                                      unresolved_facts=unresolved))
            continue
        said = str(rec.get("assessment") or "")
        if said not in DECLARABLE:
            raise RuleAssessmentError(
                f"{rid}: assessment {said!r} is not one of {list(DECLARABLE)}. "
                f"`applicable_checked` and `applicable_unresolved` are DERIVED from an `applicable` "
                f"judgement and the facts actually resolved; a run does not declare its own work "
                f"finished.")
        if said == "applicable":
            status = "applicable_unresolved" if unresolved else "applicable_checked"
        else:
            status = said
        out.append(RuleAssessment(
            rule_id=rid, status=status, turns_on=turns, unresolved_facts=unresolved,
            applicability_basis_evidence_ids=tuple(str(e) for e in (rec.get("evidence_ids") or [])),
            rationale=str(rec.get("rationale") or "")))
    return out


def unconsidered(rows: Iterable[RuleAssessment]) -> list[str]:
    """Rule ids nobody assessed. Read against `RuleGoldAnnotation.applicable`, never alone."""
    return [r.rule_id for r in rows if r.status == "not_considered"]


def unresolved_discriminators(rows: Iterable[RuleAssessment]) -> list[str]:
    """Every declared fact still open on a rule judged applicable or potentially applicable.

    Deduplicated, because a fact shared by two rules is ONE question and closure must not report it
    twice — that is why the contract declares it once.
    """
    seen: dict[str, None] = {}
    for r in rows:
        if r.status in ("applicable_unresolved", "potentially_applicable"):
            for f in r.unresolved_facts:
                seen.setdefault(f, None)
    return list(seen)

