"""Stable, task-agnostic contracts for the ACR execution and analysis planes.

The kernel deliberately knows nothing about tumour registries or any particular
extraction field.  It gives runtime, audit, evaluation, and improvement modules a
small set of immutable objects to exchange.  Patient-derived source text remains
in the local run store; canonical trajectories carry references and hashes.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ASSET_TYPES = frozenset({
    "SPEC",
    "RUNTIME_PROFILE",
    "PROMPT",
    "SKILL",
    "TOOL",
    "EVALUATOR",
    "AUDIT_RULE",
    "REPAIR_STRATEGY",
})
ASSET_STATUSES = frozenset({"DRAFT", "CERTIFIED", "ACTIVE", "RETIRED"})
TARGET_KINDS = frozenset({
    "RUN",
    "FIELD",
    "ANSWER",
    "EVIDENCE",
    "TOOL_CALL",
    "RETRIEVAL",
    "GATE_DECISION",
    "TERMINATION",
    "SECURITY_BOUNDARY",
})
SIGNAL_TYPES = frozenset({
    "AUDIT_FINDING",
    "AUDIT_INCIDENT",
    "EVALUATION_RESULT",
    "ATTRIBUTION_REPORT",
    "HUMAN_ADJUDICATION",
    "REPAIR_OBLIGATION",
    "VALIDATION_REPORT",
})

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]*$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RAW_TEXT_KEYS = frozenset({
    "content",
    "document",
    "document_text",
    "note",
    "note_text",
    "raw_text",
    "text",
})


class KernelContractError(ValueError):
    """A stable-kernel object violated its public contract."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def normalized_hash(value: str | Mapping[str, Any] | Sequence[Any]) -> str:
    if isinstance(value, str) and _SHA256.fullmatch(value.lower()):
        return value.lower()
    return digest(value)


@dataclass(frozen=True)
class AssetRef:
    """A content-addressed reference to a versioned ACR asset."""

    asset_id: str
    asset_type: str
    version: str
    content_hash: str
    local_ref: str = ""
    status: str = "DRAFT"

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.asset_id):
            raise KernelContractError(f"invalid asset_id {self.asset_id!r}")
        if self.asset_type not in ASSET_TYPES:
            raise KernelContractError(
                f"{self.asset_id}: asset_type must be one of {sorted(ASSET_TYPES)}"
            )
        if not _VERSION.fullmatch(self.version):
            raise KernelContractError(
                f"{self.asset_id}: invalid asset version {self.version!r}"
            )
        if not _SHA256.fullmatch(self.content_hash):
            raise KernelContractError(
                f"{self.asset_id}: content_hash must be a lowercase sha256"
            )
        if self.status not in ASSET_STATUSES:
            raise KernelContractError(
                f"{self.asset_id}: status must be one of {sorted(ASSET_STATUSES)}"
            )

    @property
    def ref(self) -> str:
        return f"{self.asset_id}@{self.version}"

    @classmethod
    def from_value(
        cls,
        *,
        asset_id: str,
        asset_type: str,
        version: str,
        value: Any,
        local_ref: str = "",
        status: str = "DRAFT",
    ) -> AssetRef:
        return cls(
            asset_id=asset_id,
            asset_type=asset_type,
            version=version,
            content_hash=digest(value),
            local_ref=local_ref,
            status=status,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        asset_id: str,
        asset_type: str,
        version: str,
        status: str = "DRAFT",
    ) -> AssetRef:
        resolved = Path(path).resolve()
        return cls(
            asset_id=asset_id,
            asset_type=asset_type,
            version=version,
            content_hash=hashlib.sha256(resolved.read_bytes()).hexdigest(),
            local_ref=str(resolved),
            status=status,
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ArtifactRef:
    """A local artifact pointer; content is intentionally not embedded."""

    path: str
    sha256: str
    artifact_type: str = "RUN_ARTIFACT"

    def __post_init__(self) -> None:
        if not Path(self.path).is_absolute():
            raise KernelContractError("artifact path must be absolute")
        if not _SHA256.fullmatch(self.sha256):
            raise KernelContractError("artifact sha256 must be a lowercase sha256")
        if not _ID.fullmatch(self.artifact_type):
            raise KernelContractError(f"invalid artifact type {self.artifact_type!r}")

    @classmethod
    def from_path(
        cls, path: str | Path, *, artifact_type: str = "RUN_ARTIFACT"
    ) -> ArtifactRef:
        resolved = Path(path).resolve()
        return cls(
            path=str(resolved),
            sha256=hashlib.sha256(resolved.read_bytes()).hexdigest(),
            artifact_type=artifact_type,
        )

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class TargetRef:
    """The exact run object a signal describes."""

    kind: str
    target_id: str
    field: str = ""
    detail: Mapping[str, Any] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in TARGET_KINDS:
            raise KernelContractError(
                f"target kind must be one of {sorted(TARGET_KINDS)}"
            )
        if not str(self.target_id).strip():
            raise KernelContractError("target_id is required")
        if self.kind == "FIELD" and not self.field.strip():
            raise KernelContractError("FIELD target needs field")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_id": self.target_id,
            "field": self.field,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True)
class SignalEvidenceRef:
    """A non-copying pointer to evidence supporting an analysis signal."""

    source_type: str
    source_id: str
    sha256: str = ""
    locator: str = ""

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.source_type):
            raise KernelContractError(
                f"invalid signal evidence type {self.source_type!r}"
            )
        if not str(self.source_id).strip():
            raise KernelContractError("signal evidence source_id is required")
        if self.sha256 and not _SHA256.fullmatch(self.sha256):
            raise KernelContractError("signal evidence sha256 is invalid")

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class Trajectory:
    """Canonical, immutable semantic record of one complete agent execution."""

    trajectory_id: str
    case_ref: str
    task_ref: AssetRef
    runtime_profile_ref: AssetRef
    events: tuple[Mapping[str, Any], ...]
    evidence_state: Mapping[str, Any]
    coverage_state: Mapping[str, Any]
    gate_decisions: tuple[Mapping[str, Any], ...]
    output: Mapping[str, Any]
    termination: Mapping[str, Any]
    cost: Mapping[str, Any]
    asset_lineage: tuple[AssetRef, ...] = ()
    artifact_refs: tuple[ArtifactRef, ...] = ()
    created_at: str = dataclass_field(default_factory=_now)

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.trajectory_id):
            raise KernelContractError(
                f"invalid trajectory_id {self.trajectory_id!r}"
            )
        if not str(self.case_ref).strip():
            raise KernelContractError("trajectory case_ref is required")
        if self.task_ref.asset_type != "SPEC":
            raise KernelContractError("trajectory task_ref must reference a SPEC")
        if self.runtime_profile_ref.asset_type != "RUNTIME_PROFILE":
            raise KernelContractError(
                "trajectory runtime_profile_ref must reference a RUNTIME_PROFILE"
            )
        sequences = [
            int(event["seq"])
            for event in self.events
            if isinstance(event, Mapping) and "seq" in event
        ]
        if sequences != sorted(sequences) or len(sequences) != len(set(sequences)):
            raise KernelContractError(
                "trajectory event sequence must be unique and monotonic"
            )
        refs = [asset.ref for asset in self.asset_lineage]
        if len(refs) != len(set(refs)):
            raise KernelContractError("trajectory asset_lineage contains duplicates")

    @property
    def content_hash(self) -> str:
        return digest(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "schema": "acr.trajectory/1",
            "trajectory_id": self.trajectory_id,
            "case_ref": self.case_ref,
            "task_ref": self.task_ref.to_dict(),
            "runtime_profile_ref": self.runtime_profile_ref.to_dict(),
            "events": [dict(row) for row in self.events],
            "evidence_state": dict(self.evidence_state),
            "coverage_state": dict(self.coverage_state),
            "gate_decisions": [dict(row) for row in self.gate_decisions],
            "output": dict(self.output),
            "termination": dict(self.termination),
            "cost": dict(self.cost),
            "asset_lineage": [row.to_dict() for row in self.asset_lineage],
            "artifact_refs": [row.to_dict() for row in self.artifact_refs],
            "created_at": self.created_at,
        }
        if include_hash:
            value["content_hash"] = digest(value)
        return value


def _redacted_text(value: str) -> dict[str, Any]:
    return {
        "redacted": True,
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        "length": len(value),
    }


def _sanitize_event_value(value: Any, *, key: str = "") -> Any:
    """Remove source text while retaining stable structure for analysis."""
    if isinstance(value, Mapping):
        return {
            str(child_key): _sanitize_event_value(child, key=str(child_key).lower())
            for child_key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_event_value(child, key=key) for child in value]
    if isinstance(value, str) and key in _RAW_TEXT_KEYS:
        return _redacted_text(value)
    return value


class TrajectoryAdapter:
    """Build a canonical Trajectory from extraction manifest and trace artifacts."""

    @staticmethod
    def from_run_artifacts(
        *,
        manifest: Mapping[str, Any],
        trace: Sequence[Mapping[str, Any]],
        case_ref: str,
        spec_id: str,
        spec_hash: str,
        runtime_profile_id: str = "",
        runtime_profile_hash: str = "",
        trajectory_id: str = "",
        artifact_refs: Sequence[ArtifactRef] = (),
    ) -> Trajectory:
        task_ref = AssetRef(
            asset_id=spec_id,
            asset_type="SPEC",
            version=str(manifest.get("spec_version") or "unversioned"),
            content_hash=normalized_hash(spec_hash),
            local_ref=str(manifest.get("spec_path") or ""),
            status="ACTIVE",
        )
        resolved_runtime_profile_id = str(
            runtime_profile_id
            or manifest.get("runtime_profile_id")
            or "current-stratified-coverage"
        )
        runtime_ref = AssetRef(
            asset_id=resolved_runtime_profile_id,
            asset_type="RUNTIME_PROFILE",
            version=str(manifest.get("runtime_profile_version") or "1"),
            content_hash=normalized_hash(
                runtime_profile_hash
                or manifest.get("runtime_profile_hash")
                or {
                    "runtime": manifest.get("runtime"),
                    "coverage_plan": manifest.get("coverage_plan"),
                }
            ),
            status="ACTIVE",
        )
        safe_events_unsorted = tuple(
            _sanitize_event_value(dict(event)) for event in trace
        )
        # Tool calls may complete on different worker threads. Older Tracer versions assigned
        # a unique sequence before appending to JSONL, so the file could contain 11, 9, 12
        # even though ``seq`` still preserves the intended semantic order. Ingestion is the
        # boundary at which that application log becomes an analysis-ready Trajectory:
        # normalize a complete, uniquely sequenced trace; still fail closed on duplicate or
        # missing identities rather than inventing an order.
        sequenced = [
            event for event in safe_events_unsorted if "seq" in event
        ]
        if len(sequenced) == len(safe_events_unsorted):
            sequence_ids = [int(event["seq"]) for event in sequenced]
            if len(sequence_ids) != len(set(sequence_ids)):
                raise KernelContractError(
                    "trace contains duplicate event sequence identities"
                )
            safe_events = tuple(
                sorted(sequenced, key=lambda event: int(event["seq"]))
            )
        else:
            safe_events = safe_events_unsorted
        gate_decisions = tuple(
            event
            for event in safe_events
            if str(event.get("kind") or "").lower()
            in {
                "answer_gate",
                "answer_rejected",
                "evidence_admissibility",
                "gate_decision",
            }
        )
        explicit_trajectory_id = trajectory_id or str(
            manifest.get("trajectory_id") or ""
        )
        run_label = str(
            manifest.get("run_id")
            or f"trajectory-{digest([case_ref, spec_hash])[:20]}"
        )
        if explicit_trajectory_id:
            raw_id = explicit_trajectory_id
        else:
            # ``run_id`` is an operator label in the extraction plane. Cohort runs reuse
            # ``patient+spec`` across runtime profiles and reruns, so it cannot be the
            # analysis identity of one complete execution. Bind the readable label to the
            # immutable execution content; timestamps in the trace distinguish even
            # behaviourally identical reruns.
            execution_hash = digest({
                "run_label": run_label,
                "runtime_profile": runtime_ref.to_dict(),
                "events": safe_events,
                "output": manifest.get("answer") or manifest.get("output") or {},
                "artifact_hashes": [
                    artifact.sha256 for artifact in artifact_refs
                ],
            })[:16]
            raw_id = f"{run_label}--{execution_hash}"
        safe_id = re.sub(r"[^A-Za-z0-9._:/+-]", "-", raw_id)
        termination = manifest.get("termination")
        if not isinstance(termination, Mapping):
            termination = {
                "reason": termination
                or manifest.get("termination_reason")
                or "UNKNOWN"
            }
        output = manifest.get("answer") or manifest.get("output") or {}
        raw_evidence = manifest.get("evidence_ledger") or manifest.get("evidence") or {}
        if isinstance(raw_evidence, Mapping):
            evidence_state = dict(raw_evidence)
        elif isinstance(raw_evidence, Sequence) and not isinstance(
            raw_evidence, (str, bytes)
        ):
            evidence_state = {"evidence": [dict(row) for row in raw_evidence]}
        else:
            evidence_state = {}
        # Current extraction manifests carry the deterministic answer-gate verdict beside
        # the evidence list rather than inside an ``evidence_ledger`` object.  Preserve that
        # computed fact in the canonical evidence state; otherwise every gate-valid FOUND
        # imported from a real run is falsely classified as citation-present/proof-invalid.
        # This is a representation mapping, not a second proof calculation.
        if "proof_valid" not in evidence_state:
            evidence_state["proof_valid"] = bool(manifest.get("gate_validated"))
        raw_coverage = (
            manifest.get("coverage_state")
            or manifest.get("coverage_ledger")
            or manifest.get("coverage")
            or manifest.get("coverage_attested")
            or (
                output.get("coverage_attested")
                if isinstance(output, Mapping)
                else None
            )
            or {}
        )
        coverage_state = (
            dict(raw_coverage) if isinstance(raw_coverage, Mapping) else {}
        )
        created_at = str(
            manifest.get("created_at")
            or manifest.get("created_utc")
            or next(
                (
                    event.get("ts")
                    for event in safe_events
                    if str(event.get("ts") or "").strip()
                ),
                "",
            )
            or "UNRECORDED"
        )
        raw_cost = manifest.get("cost") or manifest.get("spend") or {}
        if isinstance(raw_cost, Mapping):
            cost = dict(raw_cost)
        else:
            cost = {}
        if not cost:
            usage = manifest.get("usage") or {}
            cost = {
                "usd": manifest.get("cost_usd"),
                "tokens": manifest.get("tokens"),
                "model_calls": manifest.get("model_calls"),
                **(dict(usage) if isinstance(usage, Mapping) else {}),
            }
        return Trajectory(
            trajectory_id=safe_id,
            case_ref=case_ref,
            task_ref=task_ref,
            runtime_profile_ref=runtime_ref,
            events=safe_events,
            evidence_state=evidence_state,
            coverage_state=coverage_state,
            gate_decisions=gate_decisions,
            output=dict(output),
            termination=dict(termination),
            cost=cost,
            artifact_refs=tuple(artifact_refs),
            created_at=created_at,
        )


@dataclass(frozen=True)
class SignalEnvelope:
    """Thin common envelope; domain payloads retain their own schemas."""

    signal_id: str
    signal_type: str
    producer_ref: AssetRef
    trajectory_ref: str
    target_ref: TargetRef
    status: str
    severity: str
    evidence_refs: tuple[SignalEvidenceRef, ...]
    payload_schema: str
    payload: Mapping[str, Any]
    created_at: str = dataclass_field(default_factory=_now)

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.signal_id):
            raise KernelContractError(f"invalid signal_id {self.signal_id!r}")
        if self.signal_type not in SIGNAL_TYPES:
            raise KernelContractError(
                f"signal_type must be one of {sorted(SIGNAL_TYPES)}"
            )
        if not str(self.trajectory_ref).strip():
            raise KernelContractError("signal trajectory_ref is required")
        if not str(self.status).strip() or not str(self.severity).strip():
            raise KernelContractError("signal status and severity are required")
        if not str(self.payload_schema).strip():
            raise KernelContractError("signal payload_schema is required")

    @property
    def content_hash(self) -> str:
        return digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "acr.signal/1",
            "signal_id": self.signal_id,
            "signal_type": self.signal_type,
            "producer_ref": self.producer_ref.to_dict(),
            "trajectory_ref": self.trajectory_ref,
            "target_ref": self.target_ref.to_dict(),
            "status": self.status,
            "severity": self.severity,
            "evidence_refs": [row.to_dict() for row in self.evidence_refs],
            "payload_schema": self.payload_schema,
            "payload": dict(self.payload),
            "created_at": self.created_at,
        }
