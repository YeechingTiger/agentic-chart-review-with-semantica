"""Develop-plane specification repair from chart-observable gold and recorded trajectories.

This module never runs an agent and never changes a specification.  It turns already-recorded
deepagents manifests into behavioural clusters, compares those clusters with adjudicated
chart-observable truth, and produces the contrastive packet from which a narrowly scoped spec
edit can be proposed.  Keeping those operations deterministic is the boundary: the answer key
may influence a proposal in DEVELOP, but it is never reachable from the RUN plane.

The analogy to program repair is intentionally exact but limited.  A trajectory is grouped by
what it *did* (answer, evidence, rules and coverage), not by its prose reasoning.  A cluster is
selected only when it both matches the chart-observable answer and earned the existing gate.
Agreement alone is not correctness, and a registry value that cannot be established from the
available chart is not an instruction to make the agent guess.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ------------------------------------------------------------------- moved to `behaviour.py`
# `BehaviorSignature`, `ChartObservableGold` and the gold/adjudication vocabulary moved out on
# 2026-08-03. Six imports reached them from THREE planes other than this one —
# `review/conflict_refinement`, `diagnosis/attribution`, `commands/cli_attribute`,
# `commands/cli_gold` — so when the tree was cut into four distributions, three of them imported a
# module assigned to the fourth and none declared the dependency. Every suite passed anyway, because
# the verification environment had all four installed. `tools/verify_structure.py` is the check that
# now says so out loud.
#
# The PRIVATE helpers come back too, and that is not sloppiness: both halves shared `_hash`,
# `_normalise`, `_canonical`, `_safe_case_id` and `_portable_source` before the split, and copying
# them would put two canonical-JSON serialisers and two path-portability rules in one package —
# which is how "is this artifact portable" becomes two answers.
#
# RE-EXPORTED rather than left for callers to chase: five importers use these names and a move that
# renames every call site has to be right everywhere at once, while a re-export is right immediately
# and can be narrowed later.
from .behaviour import (  # noqa: F401  (re-export: see above)
    ADJUDICATION_UNRESOLVED,
    ADJUDICATION_VALUES,
    DERIVABILITY_VALUES,
    DERIVABLE,
    EVIDENCE_INSUFFICIENT,
    FOUND,
    GOLD_SCHEMA,
    KEY_CORRECT,
    KEY_WRONG,
    NOT_APPLICABLE,
    NOT_DERIVABLE,
    OUTSIDE_CHART,
    PARTIALLY_DERIVABLE,
    SEMANTIC_STATUSES,
    SPEC_INSUFFICIENT,
    UNRESOLVED,
    BehaviorSignature,
    ChartObservableGold,
    GoldEvidence,
    GoldField,
    GoldNotUsable,
    SpecRepairError,
    _canonical,
    _evidence_refs,
    _field_results,
    _hash,
    _normalise,
    _portable_run_id,
    _portable_source,
    _rule_ids,
    _safe_case_id,
    artifact_hash,
    audit_gold,
    gold_document,
    load_gold,
    safe_case_id,
)

BEHAVIOUR_SCHEMA = "acr.behavior_distribution/1"
PACKET_SCHEMA = "acr.contrastive_failure_packet/1"
PROPOSAL_SCHEMA = "acr.spec_patch_proposal/1"
VALIDATION_SCHEMA = "acr.paired_validation/1"




RETRIEVAL_FAILURE = "RETRIEVAL_FAILURE"
SPEC_AMBIGUITY = "SPEC_AMBIGUITY"
NO_CORRECT_BEHAVIOUR = "NO_CORRECT_BEHAVIOUR"
GOLD_NOT_CHART_OBSERVABLE = "GOLD_NOT_CHART_OBSERVABLE"
GOLD_UNRESOLVED = "GOLD_UNRESOLVED"

SEMANTIC = "semantic"
ASSET = "asset"
CHANGE_CLASSES = (SEMANTIC, ASSET)

PARAMETERS = (
    "evidence_eligibility",
    "precedence_conflict_rule",
    "temporal_scope",
    "entity_association",
    "abstention_boundary",
    "document_type_policy",
    "keyword_retrieval_asset",
    "skill_instruction",
    "deterministic_answer_check",
)
RETRIEVAL_PARAMETERS = {
    "document_type_policy", "keyword_retrieval_asset", "skill_instruction",
}




class InvalidProposal(SpecRepairError):
    """A proposed edit violates the gradient-routing boundary."""

class SealedSetReuse(SpecRepairError):
    """A sealed cohort was used after its one permitted certification read."""



















def matches_gold(signature: BehaviorSignature, gold: ChartObservableGold) -> bool:
    if not gold.usable_for_repair or signature.spec_id != gold.spec_id:
        return False
    for field_name, expected in gold.chart_answer.items():
        actual = signature.field_results.get(field_name)
        actual_status = (
            str(actual.get("status") or "") if actual is not None
            else signature.answer_status
        )
        if actual_status != expected.status:
            return False
        if (expected.status == FOUND
                and (actual is None
                     or _normalise(actual.get("value")) != _normalise(expected.value))):
            return False
    return True

def overclaims(signature: BehaviorSignature, gold: ChartObservableGold) -> bool:
    if not gold.usable_for_repair:
        return False
    for field_name, expected in gold.chart_answer.items():
        actual = signature.field_results.get(field_name) or {}
        actual_status = str(actual.get("status") or signature.answer_status)
        if expected.status != FOUND and actual_status == FOUND:
            return True
    return False

@dataclass(frozen=True)
class BehaviorCluster:
    signature_hash: str
    count: int
    mass: float
    representative: BehaviorSignature
    sources: tuple[str, ...]
    run_conditions: tuple[Mapping[str, Any], ...]
    gold_correct: bool | None
    grounded_correct: bool | None
    overclaim: bool | None

    def to_dict(self) -> dict:
        return {
            "signature_hash": self.signature_hash, "count": self.count, "mass": self.mass,
            "representative": self.representative.to_dict(), "sources": list(self.sources),
            "run_conditions": [dict(x) for x in self.run_conditions],
            "gold_correct": self.gold_correct, "grounded_correct": self.grounded_correct,
            "overclaim": self.overclaim,
        }

@dataclass(frozen=True)
class BehaviorDistribution:
    case_id: str
    spec_id: str
    clusters: tuple[BehaviorCluster, ...]
    n_runs: int
    behavioral_entropy: float
    gold_consistency: float | None
    grounded_consistency: float | None
    overclaim_rate: float | None
    gold_usable: bool

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "spec_id": self.spec_id, "n_runs": self.n_runs,
            "behavioral_entropy": self.behavioral_entropy,
            "gold_consistency": self.gold_consistency,
            "grounded_consistency": self.grounded_consistency,
            "overclaim_rate": self.overclaim_rate, "gold_usable": self.gold_usable,
            "clusters": [c.to_dict() for c in self.clusters],
        }

def cluster_behaviors(signatures: Sequence[BehaviorSignature],
                      gold: ChartObservableGold | None = None) -> BehaviorDistribution:
    if not signatures:
        raise SpecRepairError("cannot cluster an empty trajectory set")
    case_ids = {s.case_id for s in signatures}
    spec_ids = {s.spec_id for s in signatures}
    if len(case_ids) != 1 or len(spec_ids) != 1:
        raise SpecRepairError(
            f"one distribution must contain one case/spec, got cases={sorted(case_ids)}, "
            f"specs={sorted(spec_ids)}")
    by: dict[str, list[BehaviorSignature]] = {}
    for s in signatures:
        by.setdefault(s.signature_hash, []).append(s)
    n = len(signatures)
    usable = bool(gold and gold.usable_for_repair)
    clusters = []
    for key, rows in by.items():
        rep = rows[0]
        correct = matches_gold(rep, gold) if usable and gold else None
        grounded = bool(correct and rep.is_grounded) if correct is not None else None
        over = overclaims(rep, gold) if usable and gold else None
        clusters.append(BehaviorCluster(
            signature_hash=key, count=len(rows), mass=len(rows) / n, representative=rep,
            sources=tuple(_portable_source(r.source) for r in rows),
            run_conditions=tuple(dict(r.run_conditions) for r in rows),
            gold_correct=correct,
            grounded_correct=grounded, overclaim=over))
    clusters.sort(key=lambda c: (-c.count, c.signature_hash))
    entropy = -sum(c.mass * math.log2(c.mass) for c in clusters if c.mass)
    return BehaviorDistribution(
        case_id=signatures[0].case_id, spec_id=signatures[0].spec_id,
        clusters=tuple(clusters), n_runs=n, behavioral_entropy=round(entropy, 6),
        gold_consistency=(round(sum(c.mass for c in clusters if c.gold_correct), 6)
                          if usable else None),
        grounded_consistency=(round(sum(c.mass for c in clusters if c.grounded_correct), 6)
                              if usable else None),
        overclaim_rate=(round(sum(c.mass for c in clusters if c.overclaim), 6)
                        if usable else None),
        gold_usable=usable,
    )

def behavior_document(distributions: Iterable[BehaviorDistribution]) -> dict:
    rows = list(distributions)
    return {
        "schema": BEHAVIOUR_SCHEMA, "distributions": [d.to_dict() for d in rows],
        "summary": {
            "n_cases": len(rows), "n_runs": sum(d.n_runs for d in rows),
            "mean_behavioral_entropy": (
                round(sum(d.behavioral_entropy for d in rows) / len(rows), 6) if rows else None),
            "mean_gold_consistency": _mean(
                [d.gold_consistency for d in rows if d.gold_consistency is not None]),
            "mean_grounded_consistency": _mean(
                [d.grounded_consistency for d in rows
                 if d.grounded_consistency is not None]),
            "mean_overclaim_rate": _mean(
                [d.overclaim_rate for d in rows if d.overclaim_rate is not None]),
        },
    }

def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None

def load_signatures(paths: Iterable[str | Path], *,
                    case_map: Mapping[str, str] | None = None) -> list[BehaviorSignature]:
    out = []
    mapping = dict(case_map or {})
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            found = sorted(p.rglob("*.manifest.json"))
            if not found:
                raise SpecRepairError(f"no *.manifest.json under {p}")
            for x in found:
                doc = json.loads(x.read_text(encoding="utf-8"))
                original = str(doc.get("patient_id") or doc.get("case_id") or "")
                out.append(BehaviorSignature.from_manifest(
                    doc, source=str(x), case_id=mapping.get(original)))
        else:
            doc = json.loads(p.read_text(encoding="utf-8"))
            original = str(doc.get("patient_id") or doc.get("case_id") or "")
            out.append(BehaviorSignature.from_manifest(
                doc, source=str(p), case_id=mapping.get(original)))
    return out

def _difference(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict:
    out = {}
    for key in sorted(set(a) | set(b)):
        av, bv = a.get(key), b.get(key)
        if _normalise(av) != _normalise(bv):
            out[key] = {"selected": av, "rejected": bv}
    return out

def _spec_sections(spec: Any) -> dict:
    return {
        "question": str(getattr(spec, "question", "") or ""),
        "decision_rule": list(getattr(spec, "decision_rule", ()) or ()),
        "evidence_rules": dict(getattr(spec, "evidence_rules", {}) or {}),
        "conflict_rules": list(getattr(spec, "conflict_rules", ()) or ()),
        "proof_obligation": getattr(
            getattr(spec, "proof_obligation", None), "model_dump", lambda **_: {})(
                mode="json"),
        "abstention": dict(getattr(spec, "abstention", {}) or {}),
        "boundary_cases": list(getattr(spec, "boundary_cases", ()) or ()),
        "search_hints": list(getattr(spec, "search_hints", ()) or ()),
        "answer_checks": list(getattr(spec, "answer_checks", ()) or ()),
    }

@dataclass(frozen=True)
class ContrastiveFailurePacket:
    case_id: str
    spec_id: str
    spec_hash: str
    disposition: str
    selected: Mapping[str, Any] | None
    rejected: Mapping[str, Any] | None
    differences: Mapping[str, Any]
    gold: Mapping[str, Any]
    spec_sections: Mapping[str, Any]
    repair_permitted: bool
    why: str

    def to_dict(self) -> dict:
        return {
            "schema": PACKET_SCHEMA, "case_id": self.case_id, "spec_id": self.spec_id,
            "spec_hash": self.spec_hash, "disposition": self.disposition,
            "selected": dict(self.selected) if self.selected else None,
            "rejected": dict(self.rejected) if self.rejected else None,
            "differences": dict(self.differences), "gold": dict(self.gold),
            "spec_sections": dict(self.spec_sections),
            "repair_permitted": self.repair_permitted, "why": self.why,
        }

def diagnose(distribution: BehaviorDistribution, gold: ChartObservableGold, spec: Any,
             ) -> ContrastiveFailurePacket:
    if distribution.case_id != gold.case_id or distribution.spec_id != gold.spec_id:
        raise SpecRepairError("distribution and gold identify different case/spec")
    sections = _spec_sections(spec)
    spec_hash = str(getattr(spec, "spec_hash", "") or "")
    if gold.chart_derivability == NOT_DERIVABLE or gold.adjudication == OUTSIDE_CHART:
        return ContrastiveFailurePacket(
            gold.case_id, gold.spec_id, spec_hash, GOLD_NOT_CHART_OBSERVABLE,
            None, None, {}, gold.to_dict(), sections, False,
            "the registry value is outside the available chart; changing the spec would train "
            "the agent to guess")
    if not gold.usable_for_repair:
        return ContrastiveFailurePacket(
            gold.case_id, gold.spec_id, spec_hash, GOLD_UNRESOLVED,
            None, None, {}, gold.to_dict(), sections, False,
            "chart derivability or registry adjudication is unresolved")

    selected = [c for c in distribution.clusters if c.grounded_correct]
    rejected = [c for c in distribution.clusters if not c.grounded_correct]
    sel = max(selected, key=lambda c: c.count, default=None)
    rej = max(rejected, key=lambda c: c.count, default=None)
    if sel and rej:
        missing_gold_notes = {
            e.note_id for e in gold.gold_evidence if e.stance == "supports"
        } - {n for n, _, _ in rej.representative.evidence_refs}
        disposition = RETRIEVAL_FAILURE if missing_gold_notes else SPEC_AMBIGUITY
        return ContrastiveFailurePacket(
            gold.case_id, gold.spec_id, spec_hash, disposition,
            sel.representative.to_dict(), rej.representative.to_dict(),
            _difference(sel.representative.behaviour, rej.representative.behaviour),
            gold.to_dict(), sections, True,
            ("the rejected behaviour did not surface adjudicated supporting evidence"
             if disposition == RETRIEVAL_FAILURE else
             "the same spec induced both a grounded-correct and a rejected interpretation"))
    if not sel:
        witness = [e.to_dict() for e in gold.gold_evidence if e.stance == "supports"]
        return ContrastiveFailurePacket(
            gold.case_id, gold.spec_id, spec_hash, NO_CORRECT_BEHAVIOUR,
            {"source": "gold_evidence", "evidence": witness,
             "chart_answer": {k: v.to_dict() for k, v in gold.chart_answer.items()}}
            if witness else None,
            rej.representative.to_dict() if rej else None,
            {}, gold.to_dict(), sections, bool(witness),
            ("no trajectory was grounded-correct; an adjudicated witness is available"
             if witness else
             "no trajectory was grounded-correct and no gold witness exists; locate evidence "
             "or adjudicate chart derivability before proposing text"))
    return ContrastiveFailurePacket(
        gold.case_id, gold.spec_id, spec_hash, "NO_REPAIR_NEEDED",
        sel.representative.to_dict(), None, {}, gold.to_dict(), sections, False,
        "all observed behaviour is grounded-correct")

@dataclass(frozen=True)
class SpecPatchProposal:
    """A proposed edit to one registered parameter; never an applied edit."""

    case_id: str
    spec_id: str
    failure_class: str
    parameter_id: str
    quoted_current_text: str
    selected_vs_rejected_difference: Mapping[str, Any]
    minimal_patch: str
    expected_behavior_change: str
    change_class: str
    source_basis: str
    cases_addressed: tuple[str, ...]
    blast_radius: Mapping[str, Any]
    requires_clinician_signoff: bool

    def __post_init__(self) -> None:
        _safe_case_id(self.case_id)
        if not self.spec_id.strip():
            raise InvalidProposal("spec_id is required")
        if not self.failure_class.strip():
            raise InvalidProposal("failure_class is required")
        if self.parameter_id not in PARAMETERS:
            raise InvalidProposal(
                f"parameter_id {self.parameter_id!r} is not registered; choose one of {PARAMETERS}")
        if self.change_class not in CHANGE_CLASSES:
            raise InvalidProposal(f"change_class must be one of {CHANGE_CLASSES}")
        if not self.minimal_patch.strip():
            raise InvalidProposal("minimal_patch is empty")
        if not self.expected_behavior_change.strip():
            raise InvalidProposal("expected_behavior_change is required")
        if not self.source_basis.strip():
            raise InvalidProposal("source_basis is required")
        if not self.quoted_current_text.strip():
            raise InvalidProposal("quoted_current_text is required for the citation mask")
        if not self.cases_addressed or self.case_id not in self.cases_addressed:
            raise InvalidProposal("cases_addressed must contain the proposal case_id")
        if not self.blast_radius:
            raise InvalidProposal("blast_radius must state the expected scope or its uncertainty")
        if self.failure_class == RETRIEVAL_FAILURE and self.parameter_id not in RETRIEVAL_PARAMETERS:
            raise InvalidProposal(
                f"a retrieval failure may only change {sorted(RETRIEVAL_PARAMETERS)}, not "
                f"{self.parameter_id}")
        if self.failure_class == RETRIEVAL_FAILURE and self.change_class != ASSET:
            raise InvalidProposal("a retrieval failure may produce only an asset change")
        if self.change_class == SEMANTIC and not self.requires_clinician_signoff:
            raise InvalidProposal("semantic patches always require clinician sign-off")
        if self.change_class == ASSET and self.parameter_id not in RETRIEVAL_PARAMETERS:
            raise InvalidProposal(
                f"asset change cannot target semantic parameter {self.parameter_id}")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], *, spec_text: str = "") -> SpecPatchProposal:
        proposal = cls(
            case_id=_safe_case_id(value.get("case_id")),
            spec_id=str(value.get("spec_id") or ""),
            failure_class=str(value.get("failure_class") or ""),
            parameter_id=str(value.get("parameter_id") or ""),
            quoted_current_text=str(value.get("quoted_current_text") or ""),
            selected_vs_rejected_difference=dict(
                value.get("selected_vs_rejected_difference") or {}),
            minimal_patch=str(value.get("minimal_patch") or ""),
            expected_behavior_change=str(value.get("expected_behavior_change") or ""),
            change_class=str(value.get("change_class") or ""),
            source_basis=str(value.get("source_basis") or ""),
            cases_addressed=tuple(str(x) for x in (value.get("cases_addressed") or ())),
            blast_radius=dict(value.get("blast_radius") or {}),
            requires_clinician_signoff=bool(value.get("requires_clinician_signoff")),
        )
        if spec_text and " ".join(proposal.quoted_current_text.split()) not in \
                " ".join(spec_text.split()):
            raise InvalidProposal(
                "quoted_current_text does not occur verbatim in the current spec")
        return proposal

    def to_dict(self) -> dict:
        return {
            "schema": PROPOSAL_SCHEMA, "case_id": self.case_id, "spec_id": self.spec_id,
            "failure_class": self.failure_class, "parameter_id": self.parameter_id,
            "quoted_current_text": self.quoted_current_text,
            "selected_vs_rejected_difference": dict(self.selected_vs_rejected_difference),
            "minimal_patch": self.minimal_patch,
            "expected_behavior_change": self.expected_behavior_change,
            "change_class": self.change_class, "source_basis": self.source_basis,
            "cases_addressed": list(self.cases_addressed),
            "blast_radius": dict(self.blast_radius),
            "requires_clinician_signoff": self.requires_clinician_signoff,
            # A proposal has not been certified merely because it is syntactically an asset
            # change. Adoption belongs to the held-out asset certification command.
            "may_apply_automatically": False,
            "eligible_for_automatic_adoption_after_certification": (
                self.change_class == ASSET and not self.requires_clinician_signoff),
        }

def validate_proposal_for_packet(
        proposal: SpecPatchProposal,
        packet: ContrastiveFailurePacket) -> SpecPatchProposal:
    """Bind a proposal to the exact failure packet it claims to repair."""
    if not packet.repair_permitted:
        raise InvalidProposal(f"{packet.case_id}: repair is not permitted: {packet.why}")
    expected = (packet.case_id, packet.spec_id, packet.disposition)
    got = (proposal.case_id, proposal.spec_id, proposal.failure_class)
    if got != expected:
        raise InvalidProposal(
            "proposal case/spec/failure does not match its packet: "
            f"expected {expected!r}, got {got!r}")
    return proposal

@dataclass(frozen=True)
class InstancePair:
    case_id: str
    subgroup: tuple[str, ...]
    before_correct: bool
    after_correct: bool
    before_grounded: bool
    after_grounded: bool
    before_overclaim: bool
    after_overclaim: bool

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id, "subgroups": list(self.subgroup),
            "before_correct": self.before_correct, "after_correct": self.after_correct,
            "before_grounded": self.before_grounded, "after_grounded": self.after_grounded,
            "before_overclaim": self.before_overclaim, "after_overclaim": self.after_overclaim,
        }

@dataclass(frozen=True)
class PairedValidationReport:
    pairs: tuple[InstancePair, ...]
    mean_correct_delta: float
    mean_grounded_delta: float
    overclaim_delta: float
    regressions: tuple[str, ...]
    subgroup_regressions: tuple[Mapping[str, Any], ...]
    accepted: bool
    refusal_reasons: tuple[str, ...]
    metrics: Mapping[str, Mapping[str, float | None]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "schema": VALIDATION_SCHEMA, "n_cases": len(self.pairs),
            "mean_correct_delta": self.mean_correct_delta,
            "mean_grounded_delta": self.mean_grounded_delta,
            "overclaim_delta": self.overclaim_delta,
            "regressions": list(self.regressions),
            "subgroup_regressions": [dict(x) for x in self.subgroup_regressions],
            "accepted": self.accepted, "refusal_reasons": list(self.refusal_reasons),
            "metrics": {arm: dict(values) for arm, values in self.metrics.items()},
            "per_instance": [p.to_dict() for p in self.pairs],
        }

def _best_distribution_value(d: BehaviorDistribution, attr: str) -> bool:
    """Read the modal cluster, with ties resolved against acceptance."""
    best = max(c.count for c in d.clusters)
    tied = [c for c in d.clusters if c.count == best]
    return bool(tied and all(bool(getattr(c, attr)) for c in tied))

def _validation_metrics(
        distributions: Sequence[BehaviorDistribution],
        gold: Mapping[str, ChartObservableGold]) -> dict[str, float | None]:
    """Score probability mass over fields; repeated trajectories remain visible."""
    field_total = status_correct = exact_correct = 0.0
    predicted_found = correct_found = expected_found = 0.0
    expected_abstain = correct_abstain = grounded_found = 0.0
    for distribution in distributions:
        expected = gold[distribution.case_id]
        for cluster in distribution.clusters:
            signature = cluster.representative
            mass = cluster.mass
            for name, target in expected.chart_answer.items():
                actual = signature.field_results.get(name) or {}
                status = str(actual.get("status") or signature.answer_status)
                status_match = status == target.status
                value_match = (
                    target.status != FOUND
                    or _normalise(actual.get("value")) == _normalise(target.value)
                )
                field_total += mass
                status_correct += mass * status_match
                exact_correct += mass * (status_match and value_match)
                if status == FOUND:
                    predicted_found += mass
                    if target.status == FOUND and value_match:
                        correct_found += mass
                    if signature.is_grounded:
                        grounded_found += mass
                if target.status == FOUND:
                    expected_found += mass
                else:
                    expected_abstain += mass
                    correct_abstain += mass * status_match
    n = len(distributions)

    def ratio(num: float, den: float) -> float | None:
        return round(num / den, 6) if den else None

    return {
        "field_exact_accuracy": ratio(exact_correct, field_total),
        "status_accuracy": ratio(status_correct, field_total),
        "found_precision": ratio(correct_found, predicted_found),
        "found_recall": ratio(correct_found, expected_found),
        "abstention_accuracy": ratio(correct_abstain, expected_abstain),
        "evidence_validity": ratio(grounded_found, predicted_found),
        "gate_valid_correct_mass": (
            round(sum(float(d.grounded_consistency or 0) for d in distributions) / n, 6)
            if n else None
        ),
        "critical_overclaim_mass": (
            round(sum(float(d.overclaim_rate or 0) for d in distributions) / n, 6)
            if n else None
        ),
        "mean_behavioral_entropy": (
            round(sum(d.behavioral_entropy for d in distributions) / n, 6)
            if n else None
        ),
    }

def paired_validate(before: Sequence[BehaviorDistribution],
                    after: Sequence[BehaviorDistribution],
                    gold: Mapping[str, ChartObservableGold], *,
                    max_subgroup_drop: float = 0.0,
                    require_positive_mean: bool = True) -> PairedValidationReport:
    """Compare frozen before/after distributions per case and subgroup."""
    bmap, amap = ({d.case_id: d for d in before}, {d.case_id: d for d in after})
    if set(bmap) != set(amap):
        raise SpecRepairError(
            f"paired validation needs identical case ids; before-only={sorted(set(bmap)-set(amap))}, "
            f"after-only={sorted(set(amap)-set(bmap))}")
    for case_id in sorted(bmap):
        before_conditions = Counter(
            _canonical(condition)
            for cluster in bmap[case_id].clusters
            for condition in cluster.run_conditions
        )
        after_conditions = Counter(
            _canonical(condition)
            for cluster in amap[case_id].clusters
            for condition in cluster.run_conditions
        )
        if before_conditions != after_conditions:
            raise SpecRepairError(
                f"{case_id}: paired validation requires the same model, temperature, "
                "max_model_calls, max_usd and preregistered seeds in both arms")
    pairs = []
    for case_id in sorted(bmap):
        if case_id not in gold or not gold[case_id].usable_for_repair:
            raise GoldNotUsable(f"{case_id}: missing usable chart-observable gold")
        pairs.append(InstancePair(
            case_id=case_id, subgroup=gold[case_id].subgroups,
            before_correct=_best_distribution_value(bmap[case_id], "gold_correct"),
            after_correct=_best_distribution_value(amap[case_id], "gold_correct"),
            before_grounded=_best_distribution_value(bmap[case_id], "grounded_correct"),
            after_grounded=_best_distribution_value(amap[case_id], "grounded_correct"),
            before_overclaim=bool(any(c.overclaim for c in bmap[case_id].clusters)),
            after_overclaim=bool(any(c.overclaim for c in amap[case_id].clusters)),
        ))
    n = len(pairs)
    if not n:
        raise SpecRepairError("paired validation has no cases")
    correct_delta = sum(p.after_correct - p.before_correct for p in pairs) / n
    grounded_delta = sum(p.after_grounded - p.before_grounded for p in pairs) / n
    overclaim_delta = sum(p.after_overclaim - p.before_overclaim for p in pairs) / n
    regressions = tuple(p.case_id for p in pairs
                        if (p.before_correct and not p.after_correct)
                        or (p.before_grounded and not p.after_grounded)
                        or (not p.before_overclaim and p.after_overclaim))
    subgroup_rows = []
    groups = sorted({g for p in pairs for g in p.subgroup})
    for group in groups:
        rows = [p for p in pairs if group in p.subgroup]
        delta = sum(p.after_correct - p.before_correct for p in rows) / len(rows)
        if delta < -max_subgroup_drop:
            subgroup_rows.append({"subgroup": group, "n": len(rows), "correct_delta": delta})
    reasons = []
    if regressions:
        reasons.append(f"{len(regressions)} per-instance regression(s)")
    if subgroup_rows:
        reasons.append(f"{len(subgroup_rows)} subgroup regression(s)")
    if overclaim_delta > 0:
        reasons.append("critical overclaim rate increased")
    if require_positive_mean and grounded_delta <= 0:
        reasons.append("grounded-correct rate did not improve")
    before_metrics = _validation_metrics(before, gold)
    after_metrics = _validation_metrics(after, gold)
    deltas = {
        name: (
            round(float(after_metrics[name]) - float(before_metrics[name]), 6)
            if before_metrics[name] is not None and after_metrics[name] is not None else None
        )
        for name in before_metrics
    }
    return PairedValidationReport(
        pairs=tuple(pairs), mean_correct_delta=round(correct_delta, 6),
        mean_grounded_delta=round(grounded_delta, 6),
        overclaim_delta=round(overclaim_delta, 6), regressions=regressions,
        subgroup_regressions=tuple(subgroup_rows), accepted=not reasons,
        refusal_reasons=tuple(reasons),
        metrics={"before": before_metrics, "after": after_metrics, "delta": deltas})

def min_zero_error_n(max_error_rate: float, confidence: float = 0.95) -> int:
    """Exact zero-event binomial sample size: (1-p)^n <= 1-confidence."""
    if not 0 < max_error_rate < 1:
        raise SpecRepairError("max_error_rate must be strictly between 0 and 1")
    if not 0 < confidence < 1:
        raise SpecRepairError("confidence must be strictly between 0 and 1")
    return math.ceil(math.log(1 - confidence) / math.log(1 - max_error_rate))

@dataclass(frozen=True)
class SealedCertification:
    """Write-once identity for the one permitted read of a sealed cohort."""

    cohort_hash: str
    bundle_hash: str
    consumed: bool = False
    result_hash: str = ""

    def consume(self, result: Mapping[str, Any]) -> SealedCertification:
        if self.consumed:
            raise SealedSetReuse(
                f"sealed cohort {self.cohort_hash} was already consumed for bundle "
                f"{self.bundle_hash}; mint a new sealed cohort")
        return SealedCertification(self.cohort_hash, self.bundle_hash, True, _hash(result))

    def to_dict(self) -> dict:
        return {"schema": "acr.sealed_certification/1", "cohort_hash": self.cohort_hash,
                "bundle_hash": self.bundle_hash, "consumed": self.consumed,
                "result_hash": self.result_hash}