"""Deterministic routing from typed analysis signals to repair obligations."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, ClassVar

from ..core.kernel import (
    AssetRef,
    SignalEnvelope,
    SignalEvidenceRef,
    TargetRef,
    digest,
)

REPAIR_KINDS = frozenset({
    "SECURITY_CONTROL",
    "SPEC_FORM",
    "CLINICIAN_QUESTION",
    "RETRIEVAL_ASSET",
    "SKILL_PROMPT",
    "ANSWER_CHECK",
    "RUNTIME_POLICY",
    "PROVIDER_RUNTIME",
    "ADJUDICATION_ONLY",
    "NO_PATCH",
})


class RepairContractError(ValueError):
    pass


@dataclass(frozen=True)
class RepairObligation:
    obligation_id: str
    trajectory_id: str
    target_ref: TargetRef
    repair_kind: str
    parameter_owner: str
    source_signal_ids: tuple[str, ...]
    truth_mode: str
    rationale: str
    semantic_change: bool = False
    human_adjudication_refs: tuple[str, ...] = ()
    status: str = "OPEN"

    def __post_init__(self) -> None:
        if self.repair_kind not in REPAIR_KINDS:
            raise RepairContractError(
                f"unknown repair kind {self.repair_kind!r}"
            )
        if not self.source_signal_ids:
            raise RepairContractError(
                "repair obligation needs at least one source signal"
            )
        if self.truth_mode not in {"BLIND", "REGISTRY_REFERENCE", "GOLD"}:
            raise RepairContractError(
                f"unknown repair truth mode {self.truth_mode!r}"
            )
        if self.semantic_change and (
            self.truth_mode != "GOLD" or not self.human_adjudication_refs
        ):
            raise RepairContractError(
                "semantic repair requires GOLD and human adjudication"
            )
        if self.repair_kind == "SECURITY_CONTROL" and self.semantic_change:
            raise RepairContractError(
                "security/control repair cannot be a semantic spec change"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "acr.repair_obligation/1",
            **asdict(self),
            "target_ref": self.target_ref.to_dict(),
            "source_signal_ids": list(self.source_signal_ids),
            "human_adjudication_refs": list(self.human_adjudication_refs),
        }

    def to_signal(self, producer: AssetRef) -> SignalEnvelope:
        return SignalEnvelope(
            signal_id=self.obligation_id,
            signal_type="REPAIR_OBLIGATION",
            producer_ref=producer,
            trajectory_ref=self.trajectory_id,
            target_ref=self.target_ref,
            status=self.status,
            severity="INFO",
            evidence_refs=tuple(
                SignalEvidenceRef("ANALYSIS_SIGNAL", signal_id)
                for signal_id in self.source_signal_ids
            ),
            payload_schema="acr.repair_obligation/1",
            payload=self.to_dict(),
        )


@dataclass(frozen=True)
class RepairProposal:
    proposal_id: str
    obligation_id: str
    repair_kind: str
    parameter_id: str
    current_asset_ref: str
    minimal_change: Mapping[str, Any]
    expected_behavior_change: str
    semantic_change: bool
    human_signoff_required: bool
    status: str = "CANDIDATE"

    def __post_init__(self) -> None:
        if self.repair_kind not in REPAIR_KINDS:
            raise RepairContractError(
                f"unknown proposal repair kind {self.repair_kind!r}"
            )
        if self.semantic_change and not self.human_signoff_required:
            raise RepairContractError(
                "semantic proposal must require human sign-off"
            )


@dataclass(frozen=True)
class ValidationReport:
    validation_id: str
    proposal_id: str
    baseline_asset_refs: tuple[str, ...]
    candidate_asset_refs: tuple[str, ...]
    patient_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    metrics: Mapping[str, Mapping[str, float]]
    per_case_regressions: tuple[str, ...]
    subgroup_regressions: tuple[str, ...]
    critical_error_increase: bool
    accepted: bool
    rationale: str

    def __post_init__(self) -> None:
        if not self.patient_ids or not self.seeds:
            raise RepairContractError(
                "paired validation needs patients and preregistered seeds"
            )
        if self.accepted and (
            self.critical_error_increase
            or self.per_case_regressions
            or self.subgroup_regressions
        ):
            raise RepairContractError(
                "validation cannot accept a candidate with declared regressions"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "acr.validation_report/1",
            **asdict(self),
            "baseline_asset_refs": list(self.baseline_asset_refs),
            "candidate_asset_refs": list(self.candidate_asset_refs),
            "patient_ids": list(self.patient_ids),
            "seeds": list(self.seeds),
            "per_case_regressions": list(self.per_case_regressions),
            "subgroup_regressions": list(self.subgroup_regressions),
        }


class RepairSignalRouter:
    """Route confirmed signal semantics; never invent clinical authority."""

    CAUSE_ROUTES: ClassVar[Mapping[str, tuple[str, str]]] = {
        "RETRIEVAL": ("RETRIEVAL_ASSET", "retrieval-owner"),
        "EVIDENCE_INTERPRETATION": ("SKILL_PROMPT", "agent-owner"),
        "ENTITY_ASSOCIATION": ("SKILL_PROMPT", "agent-owner"),
        "TEMPORAL_SCOPE": ("SPEC_FORM", "spec-owner"),
        "SPEC_FORM": ("SPEC_FORM", "spec-owner"),
        "SPEC_CONTENT": ("CLINICIAN_QUESTION", "clinical-owner"),
        "SKILL_OR_PROMPT": ("SKILL_PROMPT", "agent-owner"),
        "ANSWER_CHECK_OR_GATE": ("ANSWER_CHECK", "runtime-owner"),
        "RUNTIME_OR_PROVIDER": ("PROVIDER_RUNTIME", "platform-owner"),
        "EVIDENCE_GAP": ("NO_PATCH", "review-owner"),
        "REFERENCE_OR_GOLD": ("ADJUDICATION_ONLY", "registry-owner"),
        "UNRESOLVED": ("ADJUDICATION_ONLY", "review-owner"),
    }

    def route(
        self,
        signals: Sequence[SignalEnvelope],
        *,
        truth_mode: str,
        attribution: Mapping[str, Any] | None = None,
        human_adjudication_refs: Sequence[str] = (),
    ) -> tuple[RepairObligation, ...]:
        if truth_mode not in {"BLIND", "REGISTRY_REFERENCE", "GOLD"}:
            raise RepairContractError(f"unknown truth mode {truth_mode!r}")
        obligations: list[RepairObligation] = []
        audit = [
            signal for signal in signals
            if signal.signal_type == "AUDIT_INCIDENT"
        ]
        for signal in audit:
            obligations.append(self._obligation(
                signal,
                repair_kind="SECURITY_CONTROL",
                owner="privacy-or-platform-owner",
                truth_mode=truth_mode,
                rationale="audit incident routes only to security/control repair",
            ))

        evaluative = [
            signal for signal in signals
            if signal.signal_type in {
                "EVALUATION_RESULT",
                "ATTRIBUTION_REPORT",
            }
        ]
        if evaluative:
            raw_primary = (attribution or {}).get("primary_cause")
            if isinstance(raw_primary, Mapping):
                cause = str(raw_primary.get("cause") or "UNRESOLVED")
            else:
                cause = str(raw_primary or "UNRESOLVED")
            repair_kind, owner = self.CAUSE_ROUTES.get(
                cause, ("ADJUDICATION_ONLY", "review-owner")
            )
            source = evaluative[0]
            # A clinician question is an adjudication obligation, not a semantic
            # patch.  A later signed GOLD event may create a separate semantic
            # proposal through the existing SpecRepair path.
            semantic = False
            obligations.append(self._obligation(
                source,
                repair_kind=repair_kind,
                owner=owner,
                truth_mode=truth_mode,
                rationale=f"confirmed attribution cause routed as {cause}",
                semantic=semantic,
                human_refs=tuple(human_adjudication_refs),
                source_ids=tuple(row.signal_id for row in evaluative),
            ))
        return tuple(obligations)

    @staticmethod
    def _obligation(
        signal: SignalEnvelope,
        *,
        repair_kind: str,
        owner: str,
        truth_mode: str,
        rationale: str,
        semantic: bool = False,
        human_refs: tuple[str, ...] = (),
        source_ids: tuple[str, ...] = (),
    ) -> RepairObligation:
        ids = source_ids or (signal.signal_id,)
        obligation_id = "RO-" + digest([
            signal.trajectory_ref,
            signal.target_ref.to_dict(),
            repair_kind,
            ids,
        ])[:20]
        return RepairObligation(
            obligation_id=obligation_id,
            trajectory_id=signal.trajectory_ref,
            target_ref=signal.target_ref,
            repair_kind=repair_kind,
            parameter_owner=owner,
            source_signal_ids=ids,
            truth_mode=truth_mode,
            rationale=rationale,
            semantic_change=semantic,
            human_adjudication_refs=human_refs,
        )
