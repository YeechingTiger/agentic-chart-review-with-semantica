"""Which conflict rules actually applied to a chart, per a human reader. The column's denominator.

`rule_assessment.assess_rules` reports every declared rule's state, and `not_considered` is the state
the whole design change exists to make visible. But four rules over twenty-seven charts means most
rules genuinely do not apply to most charts, so `not_considered` is the COMMON state and a rate read
off it alone measures nothing. The measurable is

    not_considered AND the gold annotation says this rule was applicable

## Absent is not inapplicable

The one thing this module must not do is let an unannotated chart count as a chart where the rule did
not apply. That would manufacture correct silences by the dozen — the same `not_considered` /
`not_applicable` confusion the run-side view exists to prevent, reproduced inside the reference
standard where nothing downstream could see it.

So `applicable: false` is a POSITIVE CLAIM by a reviewer, and an absent row is nothing at all.
`missed_rules` counts only against positive claims.

## Transcribed from what the corpus already asserts

The corpus's `why` fields on the mirror pair already state the rule and the fact — SYNX03's names
"STORE.390's second conflict_rule" outright. The annotations added with this module record that, and
charts whose `why` does not state a rule are left unannotated rather than guessed at. It extends the
`gold_candidates` / `gold_rejections` / `candidate_stratum` annotations that `docs/
CANDIDATE_LEDGER_REMOVED.md` deliberately kept, calling them *"hand-authored knowledge about which
readings are defensible and are the input any future approach needs"*.

## The seven stages this makes separable

Without it, a wrong answer says only that it was wrong. With it:

    1 never considered the rule            missed_rules
    2 considered it, judged it inapplicable when it applied   wrongly_dismissed
    3 judged it applicable, created no obligation
    4 created the obligation, never searched
    5 searched, did not find the fact that was there
    6 found it, misread it
    7 read it correctly, misapplied the then-clause

This module supplies stages 1 and 2. The rest need the obligation ledger.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rule_assessment import RuleAssessment, declared_facts, rule_ids


class RuleGoldError(ValueError):
    """An annotation names a rule or a fact the contract does not declare, or is unscoreable."""


@dataclass(frozen=True)
class RuleGoldAnnotation:
    """One conflict rule's standing on one chart, per a human reader."""

    rule_id: str
    applicable: bool
    #: Why the reviewer says it applied (or did not). Prose, and required for a positive claim: a
    #: fact truth with no pointer to the evidence leaves the reference standard itself unauditable.
    applicability_evidence: str = ""
    #: `fact_id -> truth`, for every fact the rule turns on. Empty for an inapplicable rule (there is
    #: no fact to be true) and for a tie-break (it turns on nothing).
    discriminating_fact_truth: dict[str, Any] = field(default_factory=dict)
    discriminator_evidence: str = ""
    #: Whether the chart's answer would differ if the fact's truth flipped. This is what makes an
    #: unchecked discriminator a defect rather than a curiosity, and it is true on both halves of the
    #: mirror pair by construction.
    answer_changes_if_fact_flips: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"rule_id": self.rule_id, "applicable": self.applicable,
                "applicability_evidence": self.applicability_evidence,
                "discriminating_fact_truth": dict(self.discriminating_fact_truth),
                "discriminator_evidence": self.discriminator_evidence,
                "answer_changes_if_fact_flips": self.answer_changes_if_fact_flips}


def parse_rule_gold(raw: Mapping[str, Any], spec) -> dict[str, RuleGoldAnnotation]:
    """`{spec_id: {rule_id: row}}` -> this contract's annotations, or refuse.

    Refuses a rule or a fact the contract does not declare, for the reason `assess_rules` refuses
    them: an annotation about a rule that does not exist would enter a denominator silently.

    Refuses an applicable rule that leaves any of its facts' truth unstated. Such a row cannot score
    anything — the run's resolution has nothing to be right or wrong against — and an unscoreable row
    in a reference standard reads as coverage.
    """
    spec_id = str(getattr(spec, "spec_id", ""))
    rows = (raw or {}).get(spec_id) or {}
    if not isinstance(rows, Mapping):
        raise RuleGoldError(f"{spec_id}: rule gold must be an object keyed by rule id")
    ids, facts = set(rule_ids(spec)), declared_facts(spec)
    by_rule = {f"conflict_rule.{i}": tuple(str(x) for x in ((r or {}).get("turns_on") or []))
               for i, r in enumerate(getattr(spec, "conflict_rules", None) or [], start=1)
               if isinstance(r, Mapping)}

    out: dict[str, RuleGoldAnnotation] = {}
    for rid, row in rows.items():
        rid = str(rid)
        if rid not in ids:
            raise RuleGoldError(
                f"{spec_id}: rule gold names {rid}, which this contract does not declare. "
                f"Declared: {sorted(ids) or '(none)'}")
        if not isinstance(row, Mapping):
            raise RuleGoldError(f"{spec_id}: {rid}: annotation is not an object")
        applicable = bool(row.get("applicable"))
        truth = dict(row.get("discriminating_fact_truth") or {})
        if ghosts := sorted(set(truth) - set(facts)):
            raise RuleGoldError(
                f"{spec_id}: {rid}: fact truth stated for {ghosts}, which this contract does not "
                f"declare. Declared: {sorted(facts) or '(none)'}")
        if applicable:
            if missing := sorted(set(by_rule.get(rid, ())) - set(truth)):
                raise RuleGoldError(
                    f"{spec_id}: {rid} is annotated applicable and states no truth for {missing}. "
                    f"An applicable rule whose fact truth is unstated scores nothing: the run's "
                    f"resolution has nothing to be right or wrong against.")
            if not str(row.get("applicability_evidence") or "").strip():
                raise RuleGoldError(
                    f"{spec_id}: {rid} is annotated applicable with no `applicability_evidence`. A "
                    f"positive claim about a chart has to say where in the chart it comes from, or "
                    f"the reference standard is itself unauditable.")
        else:
            # An inapplicable rule has no fact to be true. Keeping a truth here would let a reviewer
            # assert a fact about a rule they have just said does not apply.
            truth = {}
        out[rid] = RuleGoldAnnotation(
            rule_id=rid, applicable=applicable,
            applicability_evidence=str(row.get("applicability_evidence") or ""),
            discriminating_fact_truth=truth,
            discriminator_evidence=str(row.get("discriminator_evidence") or ""),
            answer_changes_if_fact_flips=bool(row.get("answer_changes_if_fact_flips")))
    return out


def load_rule_gold(patient_id: str, spec, corpus_root: str | Path | None = None
                   ) -> dict[str, RuleGoldAnnotation]:
    """This chart's annotations for this contract. `{}` when the chart carries none.

    `{}` is ABSENT, not "no rule applies". `missed_rules` counts only positive claims, so an
    unannotated chart contributes to no denominator.
    """
    from ..core import site
    root = Path(corpus_root) if corpus_root else site.corpus_root()
    path = root / str(patient_id) / "_ground_truth.json"
    if not path.is_file():
        return {}
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return parse_rule_gold(doc.get("rule_gold") or {}, spec)


def missed_rules(rows: Iterable[RuleAssessment],
                 gold: Mapping[str, RuleGoldAnnotation]) -> list[str]:
    """Rules gold says applied and the run never considered. Stage 1 of the attribution.

    Only positive claims count. A rule gold says nothing about, or says did not apply, is not a miss
    however the run treated it — otherwise every unannotated chart would report four misses.
    """
    return [r.rule_id for r in rows
            if r.status == "not_considered"
            and (g := gold.get(r.rule_id)) is not None and g.applicable]


def wrongly_dismissed(rows: Iterable[RuleAssessment],
                      gold: Mapping[str, RuleGoldAnnotation]) -> list[str]:
    """Rules the run judged inapplicable that gold says applied. Stage 2, and distinct from stage 1:
    the run looked and got it wrong, which is a different repair from never looking."""
    return [r.rule_id for r in rows
            if r.status == "not_applicable"
            and (g := gold.get(r.rule_id)) is not None and g.applicable]
