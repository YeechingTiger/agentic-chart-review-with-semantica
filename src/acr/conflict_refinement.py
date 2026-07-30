"""Optional conflict-driven refinement around, never instead of, the deepagents runtime.

The baseline runner remains the only chart-review implementation.  This module may call that
runner several times, compare the resulting structured hypotheses, and feed a concise conflict
brief into a later call.  It owns no chart tools, evidence ledger, coverage calculation or
answer gate.  Disabling the feature therefore removes this module from the execution path and
leaves baseline behaviour byte-for-byte under the existing runner's control.

Consensus is deliberately insufficient.  A round converges only when all usable hypotheses
agree on field results and each agreeing run passed the shared gate with no unresolved thread.
When that cannot be achieved inside the declared round/cost budget, the wrapper returns
REVIEW_REQUIRED and does not select the modal answer.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .spec_repair import BehaviorSignature

CONFLICT_SCHEMA = "acr.conflict_refinement/1"

VALUE_CONFLICT = "VALUE_CONFLICT"
STATUS_CONFLICT = "STATUS_CONFLICT"
EVIDENCE_CONFLICT = "EVIDENCE_CONFLICT"
RULE_CONFLICT = "RULE_CONFLICT"
ENTITY_CONFLICT = "ENTITY_CONFLICT"
TIME_CONFLICT = "TIME_CONFLICT"
COVERAGE_CONFLICT = "COVERAGE_CONFLICT"
OPEN_OBLIGATION = "OPEN_OBLIGATION"

CONVERGED = "CONVERGED"
NO_CONFLICT = "NO_CONFLICT"
REVIEW_REQUIRED = "REVIEW_REQUIRED"


class ConflictRefinementError(ValueError):
    """The optional wrapper was configured inconsistently."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()[:16]


def _semantic_result(signature: BehaviorSignature) -> dict:
    fields = {
        name: {"status": str(row.get("status") or ""), "value": row.get("value")}
        for name, row in sorted(signature.field_results.items())
    }
    return fields or {"__answer__": {"status": signature.answer_status, "value": None}}


@dataclass(frozen=True)
class Hypothesis:
    """One deepagents result represented without free-form chain of thought."""

    hypothesis_id: str
    round_index: int
    candidate_index: int
    manifest: Mapping[str, Any]
    signature: BehaviorSignature

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any], *, round_index: int,
                      candidate_index: int) -> Hypothesis:
        # Conflict refinement is per patient, so it needs no patient identifier in its
        # portable wrapper artifact.  The original manifest remains at its PHI-controlled
        # run path; using a constant here prevents the optional layer from copying it out.
        sig = BehaviorSignature.from_manifest(manifest, case_id="runtime-case")
        hid = _hash({"round": round_index, "candidate": candidate_index,
                     "run_id": manifest.get("run_id"), "signature": sig.signature_hash})
        return cls(hid, round_index, candidate_index, dict(manifest), sig)

    @property
    def field_results(self) -> dict:
        return _semantic_result(self.signature)

    @property
    def evidence_note_ids(self) -> tuple[str, ...]:
        return tuple(sorted(n for n, _, _ in self.signature.evidence_refs))

    @property
    def unresolved_count(self) -> int:
        return len(self.signature.proof_obligations)

    @property
    def degradation_count(self) -> int:
        return sum(int(v) for v in self.signature.degradation.values())

    @property
    def usable(self) -> bool:
        return bool(self.signature.gate_validated and not self.signature.proof_obligations
                    and self.degradation_count == 0)

    @property
    def rank(self) -> tuple[int, int, int, int, int]:
        """Deterministic evidence-first order; never a replacement answer gate."""
        return (
            int(self.signature.gate_validated),
            int(not self.signature.proof_obligations),
            int(bool(self.signature.evidence_refs)),
            len(self.signature.rules_applied),
            -self.degradation_count,
        )

    def to_dict(self, *, include_manifest: bool = False) -> dict:
        spend = self.manifest.get("spend") or {}
        usage = self.manifest.get("usage") or {}
        out = {
            "hypothesis_id": self.hypothesis_id, "round_index": self.round_index,
            "candidate_index": self.candidate_index, "field_results": self.field_results,
            "evidence_note_ids": list(self.evidence_note_ids),
            "rules_applied": list(self.signature.rules_applied),
            "gate_validated": self.signature.gate_validated,
            "entity_anchor": dict(self.signature.entity_anchor),
            "temporal_anchor": dict(self.signature.temporal_anchor),
            "coverage_result": dict(self.signature.coverage_result),
            "proof_obligations": list(self.signature.proof_obligations),
            "degradation_count": self.degradation_count, "usable": self.usable,
            "rank": list(self.rank), "run_id": self.signature.run_id,
            "cost_usd": spend.get("usd"), "total_tokens": usage.get("total_tokens"),
        }
        if include_manifest:
            out["manifest"] = dict(self.manifest)
        return out


@dataclass(frozen=True)
class Conflict:
    kind: str
    field: str
    alternatives: tuple[str, ...]
    hypothesis_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {"kind": self.kind, "field": self.field,
                "alternatives": list(self.alternatives),
                "hypothesis_ids": list(self.hypothesis_ids)}


@dataclass(frozen=True)
class ConflictSet:
    conflicts: tuple[Conflict, ...]
    targeted_queries: tuple[str, ...]

    @property
    def empty(self) -> bool:
        return not self.conflicts

    def to_dict(self) -> dict:
        return {"conflicts": [c.to_dict() for c in self.conflicts],
                "targeted_queries": list(self.targeted_queries)}

    def render_for_deepagents(self) -> str:
        """A task-context appendix. It grants no tool and changes no gate."""
        if self.empty:
            return ""
        lines = [
            "# OPTIONAL CONFLICT-REFINEMENT BRIEF",
            (
                "Earlier independent runs disagreed. Treat every item below as an unresolved "
                "thread, not as a vote. Re-open the primary chart evidence for every alternative, "
                "apply the extraction specification's precedence rules, record contradictions, "
                "and submit only through the ordinary evidence/coverage gate."
            ),
        ]
        for c in self.conflicts:
            label = c.field or "run"
            lines.append(f"- {c.kind} [{label}]: {' | '.join(c.alternatives)}")
        lines += ["Targeted chart queries suggested by the structured differences:"]
        lines += [f"- {q}" for q in self.targeted_queries]
        lines.append(
            "Agreement is not proof. If the chart cannot close these conflicts, abstain or "
            "leave the relevant obligation open; do not force a consensus.")
        return "\n".join(lines)


def detect_conflicts(hypotheses: Sequence[Hypothesis]) -> ConflictSet:
    if not hypotheses:
        raise ConflictRefinementError("conflict detection needs at least one hypothesis")
    conflicts: list[Conflict] = []
    fields = sorted({f for h in hypotheses for f in h.field_results})
    for field in fields:
        statuses: dict[str, list[str]] = {}
        values: dict[str, list[str]] = {}
        for h in hypotheses:
            row = h.field_results.get(field) or {}
            status = str(row.get("status") or "<missing>")
            statuses.setdefault(status, []).append(h.hypothesis_id)
            value = _canonical(row.get("value"))
            values.setdefault(value, []).append(h.hypothesis_id)
        if len(statuses) > 1:
            conflicts.append(Conflict(
                STATUS_CONFLICT, field, tuple(sorted(statuses)),
                tuple(sorted(h.hypothesis_id for h in hypotheses))))
        if len(values) > 1:
            conflicts.append(Conflict(
                VALUE_CONFLICT, field, tuple(sorted(values)),
                tuple(sorted(h.hypothesis_id for h in hypotheses))))

    evidence = {h.evidence_note_ids for h in hypotheses}
    if len(evidence) > 1:
        conflicts.append(Conflict(
            EVIDENCE_CONFLICT, "", tuple(_canonical(x) for x in sorted(evidence)),
            tuple(sorted(h.hypothesis_id for h in hypotheses))))
    rules = {h.signature.rules_applied for h in hypotheses}
    if len(rules) > 1:
        conflicts.append(Conflict(
            RULE_CONFLICT, "", tuple(_canonical(x) for x in sorted(rules)),
            tuple(sorted(h.hypothesis_id for h in hypotheses))))
    for kind, attr in (
        (ENTITY_CONFLICT, "entity_anchor"),
        (TIME_CONFLICT, "temporal_anchor"),
        (COVERAGE_CONFLICT, "coverage_result"),
    ):
        alternatives = {
            _canonical(getattr(h.signature, attr)): h.hypothesis_id for h in hypotheses
        }
        if len(alternatives) > 1:
            conflicts.append(Conflict(
                kind, "", tuple(sorted(alternatives)),
                tuple(sorted(h.hypothesis_id for h in hypotheses))))
    for h in hypotheses:
        if h.signature.proof_obligations:
            conflicts.append(Conflict(
                OPEN_OBLIGATION, "", h.signature.proof_obligations, (h.hypothesis_id,)))

    queries = []
    for c in conflicts:
        if c.kind in (VALUE_CONFLICT, STATUS_CONFLICT):
            queries.append(
                f"For field {c.field}, locate and compare the primary-source evidence for "
                f"each alternative: {', '.join(c.alternatives)}")
        elif c.kind == EVIDENCE_CONFLICT:
            queries.append(
                "Compare the authority, date, entity and provenance of the differing cited "
                "documents; prefer only what the spec permits")
        elif c.kind == RULE_CONFLICT:
            queries.append(
                "Re-read the spec's precedence and conflict rules and state which stable rule "
                "identifier resolves the alternatives")
        elif c.kind == ENTITY_CONFLICT:
            queries.append(
                "Resolve tumor/entity identity and distinguish tumor origin from specimen or "
                "procedure location using primary-source document headers and findings")
        elif c.kind == TIME_CONFLICT:
            queries.append(
                "Build a focused event timeline around the competing dates and apply the "
                "spec's temporal anchor and recurrence rules")
        elif c.kind == COVERAGE_CONFLICT:
            queries.append(
                "Reconcile which proof obligations and chart strata were actually completed; "
                "do not infer completion from agreement")
        else:
            queries.append(
                "Follow every unresolved marker to its completion or preserve it as an open "
                "obligation")
    return ConflictSet(tuple(conflicts), tuple(dict.fromkeys(queries)))


def _converged(hypotheses: Sequence[Hypothesis]) -> bool:
    """Every candidate in the latest round must be usable and semantically agree."""
    return bool(hypotheses and all(h.usable for h in hypotheses)
                and len({_canonical(h.field_results) for h in hypotheses}) == 1)


def _progress(previous: Sequence[Hypothesis], current: Sequence[Hypothesis],
              previous_conflicts: ConflictSet, current_conflicts: ConflictSet) -> bool:
    """Require new evidence or fewer structured conflicts before buying another round."""
    old_evidence = {note for h in previous for note in h.evidence_note_ids}
    new_evidence = {note for h in current for note in h.evidence_note_ids}
    return bool(new_evidence - old_evidence
                or len(current_conflicts.conflicts) < len(previous_conflicts.conflicts))


@dataclass(frozen=True)
class ConflictRefinementResult:
    status: str
    rounds: tuple[tuple[Hypothesis, ...], ...]
    conflicts: tuple[ConflictSet, ...]
    selected_hypothesis_id: str | None
    selected_manifest: Mapping[str, Any] | None
    reason: str

    def to_dict(self, *, include_manifests: bool = False) -> dict:
        return {
            "schema": CONFLICT_SCHEMA, "enabled": True, "status": self.status,
            "n_rounds": len(self.rounds),
            "rounds": [[h.to_dict(include_manifest=include_manifests) for h in row]
                       for row in self.rounds],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "selected_hypothesis_id": self.selected_hypothesis_id,
            "selected_manifest": dict(self.selected_manifest)
            if include_manifests and self.selected_manifest else None,
            "reason": self.reason,
        }


Runner = Callable[..., Mapping[str, Any]]


def run_conflict_refinement(*, runner: Runner, candidates_per_round: int,
                            max_rounds: int, runner_kwargs: Mapping[str, Any],
                            run_id_prefix: str = "conflict",
                            max_total_usd: float | None = None) -> ConflictRefinementResult:
    """Call the same deepagents runner in bounded rounds, adding only conflict task context."""
    if candidates_per_round < 2:
        raise ConflictRefinementError("conflict refinement needs at least two candidates per round")
    if max_rounds < 1:
        raise ConflictRefinementError("max_rounds must be at least one")
    if max_total_usd is not None and max_total_usd <= 0:
        raise ConflictRefinementError("max_total_usd must be positive when supplied")
    rounds: list[tuple[Hypothesis, ...]] = []
    conflict_history: list[ConflictSet] = []
    context = ""
    for round_index in range(max_rounds):
        current = []
        for candidate_index in range(candidates_per_round):
            kw = dict(runner_kwargs)
            kw["run_id"] = f"{run_id_prefix}__r{round_index + 1}c{candidate_index + 1}"
            kw["additional_task_context"] = context
            manifest = runner(**kw)
            current.append(Hypothesis.from_manifest(
                manifest, round_index=round_index, candidate_index=candidate_index))
        row = tuple(current)
        rounds.append(row)
        conflicts = detect_conflicts(row)
        conflict_history.append(conflicts)

        if round_index == 0 and conflicts.empty and _converged(row):
            selected = max(row, key=lambda h: h.rank)
            return ConflictRefinementResult(
                NO_CONFLICT, tuple(rounds), tuple(conflict_history),
                selected.hypothesis_id, selected.manifest,
                "independent baseline runs agreed and every run passed the shared gate")
        if _converged(row):
            selected = max(row, key=lambda h: h.rank)
            return ConflictRefinementResult(
                CONVERGED, tuple(rounds), tuple(conflict_history),
                selected.hypothesis_id, selected.manifest,
                "the latest conflict-informed round agreed; every candidate passed the "
                "shared gate with no open thread or degradation")
        if len(rounds) > 1 and not _progress(
                rounds[-2], row, conflict_history[-2], conflicts):
            return ConflictRefinementResult(
                REVIEW_REQUIRED, tuple(rounds), tuple(conflict_history), None, None,
                "the latest refinement round found no new evidence and reduced no structured "
                "conflict; no modal answer was selected")
        priced = [(h.manifest.get("spend") or {}).get("usd")
                  for old in rounds for h in old]
        if max_total_usd is not None and any(
                not isinstance(x, (int, float)) for x in priced):
            return ConflictRefinementResult(
                REVIEW_REQUIRED, tuple(rounds), tuple(conflict_history), None, None,
                "at least one optional run has no priced USD cost, so the total ceiling cannot "
                "be enforced; no answer was selected")
        total_usd = sum(float(x) for x in priced if isinstance(x, (int, float)))
        if max_total_usd is not None and total_usd >= max_total_usd:
            return ConflictRefinementResult(
                REVIEW_REQUIRED, tuple(rounds), tuple(conflict_history), None, None,
                f"the optional refinement spent ${total_usd:.4f} against its "
                f"${max_total_usd:.4f} total ceiling without gate-valid agreement")
        if round_index + 1 < max_rounds:
            context = conflicts.render_for_deepagents()

    return ConflictRefinementResult(
        REVIEW_REQUIRED, tuple(rounds), tuple(conflict_history), None, None,
        "the bounded deepagents rounds did not produce gate-valid agreement; no modal answer "
        "was selected")
