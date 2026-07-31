"""Application-level ACR audit plane.

Audit is truth-blind and deterministic.  It answers whether the recorded run
crossed an operational, privacy, or integrity boundary; it does not judge the
clinical answer.  Runtime/eBPF ingestion is intentionally left as a future
adapter behind ``runtime_evidence_refs``.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..core.kernel import (
    AssetRef,
    SignalEnvelope,
    SignalEvidenceRef,
    TargetRef,
    Trajectory,
    digest,
)
from ..core.local_artifacts import LocalArtifactStore
from ..core.modules import ModuleAsset, ModuleRegistry

AUDIT_SEVERITIES = frozenset({"INFO", "WARN", "CRITICAL", "IRB"})
AUDIT_STATUSES = frozenset({"PASS", "FINDING", "INCIDENT"})

_INSTITUTIONAL_PERSON = re.compile("1168" + r"\d{12}")
_EMAIL = re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)
_MRN = re.compile(
    r"(?i)\b(?:mrn|medical record(?: number)?)\s*[:#-]?\s*[A-Z0-9-]{5,20}\b"
)
_DOB = re.compile(
    r"(?i)\b(?:dob|date of birth)\s*[:#-]?\s*(?:\d{1,2}[/-]){2}\d{2,4}\b"
)
_URL = re.compile(r"https?://([^/\s:]+)")

PHI_PATTERNS = {
    "INSTITUTIONAL_PERSON_ID": _INSTITUTIONAL_PERSON,
    "EMAIL": _EMAIL,
    "PHONE": _PHONE,
    "MRN": _MRN,
    "DOB": _DOB,
}
OUTBOUND_TOOLS = frozenset({
    "web",
    "web_fetch",
    "http",
    "https",
    "curl",
    "send_email",
    "message",
    "upload",
    "post",
    "webhook",
})


class AuditContractError(ValueError):
    """An audit rule attempted to cross its truth or output boundary."""


def _fingerprint(value: str) -> str:
    key = os.environ.get("ACR_PHI_FINGERPRINT_KEY", "")
    if not key:
        return "<redacted:no-local-key>"
    return hmac.new(
        key.encode("utf-8"), value.encode("utf-8"), hashlib.sha256
    ).hexdigest()[:16]


def _walk(value: Any, path: str = "root"):
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from _walk(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def _private_or_local_host(host: str) -> bool:
    value = host.lower().strip("[]")
    return (
        value in {"localhost", "127.0.0.1", "::1"}
        or value.endswith((".local", ".internal"))
        or value.startswith((
            "10.",
            "192.168.",
            "172.16.",
            "172.17.",
            "172.18.",
            "172.19.",
            "172.2",
            "172.30.",
            "172.31.",
        ))
    )


def _outbound_event(event: Mapping[str, Any]) -> bool:
    tool = str(event.get("tool") or event.get("name") or "")
    tool = tool.lower().split(".")[-1]
    if tool in OUTBOUND_TOOLS:
        return True
    args = event.get("args") or {}
    for _, text in _walk(args, "args"):
        match = _URL.search(text)
        if match and not _private_or_local_host(match.group(1)):
            return True
    return False


@dataclass(frozen=True)
class AuditContext:
    trajectory: Trajectory
    application_events: tuple[Mapping[str, Any], ...]
    patient_scope: str
    provider_boundary: str = "UNKNOWN"
    declared_tools: tuple[str, ...] = ()
    local_root: str = ""
    git_root: str = ""
    runtime_evidence_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.patient_scope.strip():
            raise AuditContractError("audit patient_scope is required")
        if self.trajectory.case_ref != self.patient_scope:
            raise AuditContractError(
                "trajectory case_ref does not match audit patient scope"
            )
        if len(set(self.declared_tools)) != len(self.declared_tools):
            raise AuditContractError("declared_tools contains duplicates")
        for path in (self.local_root, self.git_root):
            if path and not Path(path).is_absolute():
                raise AuditContractError("audit boundary paths must be absolute")

    @property
    def input_hash(self) -> str:
        return digest({
            "trajectory_hash": self.trajectory.content_hash,
            "application_events": self.application_events,
            "patient_scope_hash": digest(self.patient_scope),
            "provider_boundary": self.provider_boundary,
            "declared_tools": self.declared_tools,
            "local_root": self.local_root,
            "git_root": self.git_root,
            "runtime_evidence_refs": self.runtime_evidence_refs,
        })


@dataclass(frozen=True)
class AuditFinding:
    finding_id: str
    rule_ref: str
    trajectory_id: str
    target_ref: TargetRef
    kind: str
    severity: str
    message: str
    evidence: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if self.severity not in AUDIT_SEVERITIES:
            raise AuditContractError(
                f"unknown audit severity {self.severity!r}"
            )
        if not self.kind.strip() or not self.message.strip():
            raise AuditContractError("audit finding kind and message are required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "acr.audit_finding/1",
            "finding_id": self.finding_id,
            "rule_ref": self.rule_ref,
            "trajectory_id": self.trajectory_id,
            "target_ref": self.target_ref.to_dict(),
            "kind": self.kind,
            "severity": self.severity,
            "message": self.message,
            "evidence": [dict(row) for row in self.evidence],
        }

    def to_signal(self, producer: AssetRef) -> SignalEnvelope:
        evidence_refs = tuple(
            SignalEvidenceRef(
                source_type=str(row.get("source_type") or "AUDIT_EVENT"),
                source_id=str(
                    row.get("source_id")
                    or row.get("path")
                    or row.get("seq")
                    or self.finding_id
                ),
                sha256=str(row.get("sha256") or ""),
                locator=str(row.get("path") or ""),
            )
            for row in self.evidence
        )
        return SignalEnvelope(
            signal_id=self.finding_id,
            signal_type="AUDIT_FINDING",
            producer_ref=producer,
            trajectory_ref=self.trajectory_id,
            target_ref=self.target_ref,
            status="FINDING",
            severity=self.severity,
            evidence_refs=evidence_refs,
            payload_schema="acr.audit_finding/1",
            payload=self.to_dict(),
        )


@dataclass(frozen=True)
class AuditIncident:
    incident_id: str
    rule_ref: str
    trajectory_id: str
    target_ref: TargetRef
    kind: str
    severity: str
    finding_ids: tuple[str, ...]
    rationale: str
    disposition: str = "QUARANTINE_RESULT"

    def __post_init__(self) -> None:
        if self.severity not in {"CRITICAL", "IRB"}:
            raise AuditContractError(
                "audit incidents require CRITICAL or IRB severity"
            )
        if not self.finding_ids:
            raise AuditContractError("audit incident needs supporting findings")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "acr.audit_incident/1",
            **asdict(self),
            "target_ref": self.target_ref.to_dict(),
            "finding_ids": list(self.finding_ids),
        }

    def to_signal(self, producer: AssetRef) -> SignalEnvelope:
        return SignalEnvelope(
            signal_id=self.incident_id,
            signal_type="AUDIT_INCIDENT",
            producer_ref=producer,
            trajectory_ref=self.trajectory_id,
            target_ref=self.target_ref,
            status="INCIDENT",
            severity=self.severity,
            evidence_refs=tuple(
                SignalEvidenceRef("AUDIT_FINDING", finding_id)
                for finding_id in self.finding_ids
            ),
            payload_schema="acr.audit_incident/1",
            payload=self.to_dict(),
        )


@dataclass(frozen=True)
class AuditReport:
    trajectory_id: str
    rule_refs: tuple[str, ...]
    findings: tuple[AuditFinding, ...]
    incidents: tuple[AuditIncident, ...]
    input_hash: str

    @property
    def status(self) -> str:
        if self.incidents:
            return "INCIDENT"
        if self.findings:
            return "FINDING"
        return "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "acr.audit_report/1",
            "trajectory_id": self.trajectory_id,
            "rule_refs": list(self.rule_refs),
            "status": self.status,
            "findings": [row.to_dict() for row in self.findings],
            "incidents": [row.to_dict() for row in self.incidents],
            "input_hash": self.input_hash,
        }


AuditImplementation = Callable[
    [ModuleAsset, AuditContext],
    tuple[tuple[AuditFinding, ...], tuple[AuditIncident, ...]],
]


class AuditRuleRegistry:
    """Truth-blind AuditRule assets backed by explicit CODE implementations."""

    def __init__(self) -> None:
        self.modules = ModuleRegistry()

    def register(
        self, asset: ModuleAsset, implementation: AuditImplementation
    ) -> None:
        if asset.module_kind != "AUDIT_RULE":
            raise AuditContractError(
                f"{asset.ref}: audit registry accepts only AUDIT_RULE assets"
            )
        self.modules.register_asset(asset)
        self.modules.register_implementation(
            asset.implementation_id, implementation
        )

    def resolve(self, ref: str) -> ModuleAsset:
        return self.modules.resolve(ref)

    def all(self) -> tuple[ModuleAsset, ...]:
        return self.modules.all_assets()


class AuditRunner:
    def __init__(
        self, registry: AuditRuleRegistry, store: AuditStore | None = None
    ):
        self.registry = registry
        self.store = store

    def run(
        self, context: AuditContext, rule_refs: Sequence[str] = ()
    ) -> AuditReport:
        assets = (
            tuple(self.registry.resolve(ref) for ref in rule_refs)
            if rule_refs
            else self.registry.all()
        )
        findings: list[AuditFinding] = []
        incidents: list[AuditIncident] = []
        for asset in assets:
            implementation = self.registry.modules.implementation(asset)
            module_findings, module_incidents = implementation(asset, context)
            findings.extend(module_findings)
            incidents.extend(module_incidents)
        report = AuditReport(
            trajectory_id=context.trajectory.trajectory_id,
            rule_refs=tuple(asset.ref for asset in assets),
            findings=tuple(findings),
            incidents=tuple(incidents),
            input_hash=context.input_hash,
        )
        if self.store is not None:
            self.store.add(report, assets)
        return report


class AuditStore:
    """Local append-only audit signal store, separate from evaluation results."""

    def __init__(self, local_store: LocalArtifactStore):
        self.local_store = local_store

    def add(
        self, report: AuditReport, assets: Sequence[ModuleAsset]
    ) -> None:
        by_ref = {asset.ref: asset for asset in assets}
        for finding in report.findings:
            asset = by_ref[finding.rule_ref]
            signal = finding.to_signal(asset.asset_ref)
            self.local_store.append_jsonl(
                "audit/findings.jsonl",
                signal.to_dict(),
                idempotency_key=signal.signal_id,
            )
        for incident in report.incidents:
            asset = by_ref[incident.rule_ref]
            signal = incident.to_signal(asset.asset_ref)
            self.local_store.append_jsonl(
                "audit/incidents.jsonl",
                signal.to_dict(),
                idempotency_key=signal.signal_id,
            )


def _finding_id(
    asset: ModuleAsset,
    context: AuditContext,
    kind: str,
    locator: str,
) -> str:
    return "AF-" + digest(
        [asset.ref, context.input_hash, kind, locator]
    )[:20]


def _incident_id(
    asset: ModuleAsset,
    context: AuditContext,
    kind: str,
    finding_ids: Iterable[str],
) -> str:
    return "AI-" + digest(
        [asset.ref, context.input_hash, kind, *sorted(finding_ids)]
    )[:20]


def _target(
    context: AuditContext, kind: str = "SECURITY_BOUNDARY", target_id: str = ""
) -> TargetRef:
    return TargetRef(
        kind=kind,
        target_id=target_id or context.trajectory.trajectory_id,
    )


def patient_boundary_audit(
    asset: ModuleAsset, context: AuditContext
) -> tuple[tuple[AuditFinding, ...], tuple[AuditIncident, ...]]:
    findings = []
    expected = {context.patient_scope, context.trajectory.case_ref}
    for index, event in enumerate(context.application_events):
        args = event.get("args") if isinstance(event.get("args"), Mapping) else {}
        for key in ("patient", "patient_id", "person_id", "subject_id"):
            if key not in args:
                continue
            observed = str(args[key])
            if not observed or observed in expected:
                continue
            seq = event.get("seq", index)
            path = f"events[{seq}].args.{key}"
            findings.append(AuditFinding(
                finding_id=_finding_id(
                    asset, context, "PATIENT_CROSSOVER", path
                ),
                rule_ref=asset.ref,
                trajectory_id=context.trajectory.trajectory_id,
                target_ref=_target(context),
                kind="PATIENT_CROSSOVER",
                severity="IRB",
                message="tool argument names a patient outside the run scope",
                evidence=({
                    "source_type": "APPLICATION_EVENT",
                    "source_id": str(seq),
                    "path": path,
                    "fingerprint": _fingerprint(observed),
                    "value_redacted": True,
                },),
            ))
    if not findings:
        return (), ()
    ids = tuple(row.finding_id for row in findings)
    return tuple(findings), (AuditIncident(
        incident_id=_incident_id(
            asset, context, "PATIENT_CROSSOVER", ids
        ),
        rule_ref=asset.ref,
        trajectory_id=context.trajectory.trajectory_id,
        target_ref=_target(context),
        kind="PATIENT_CROSSOVER",
        severity="IRB",
        finding_ids=ids,
        rationale="one or more tool calls crossed the declared patient boundary",
    ),)


def phi_provider_audit(
    asset: ModuleAsset, context: AuditContext
) -> tuple[tuple[AuditFinding, ...], tuple[AuditIncident, ...]]:
    findings: list[AuditFinding] = []
    metadata: dict[str, dict[str, Any]] = {}
    sources: list[tuple[str, Any, str, str]] = []
    for index, event in enumerate(context.application_events):
        kind = str(event.get("kind") or "").lower()
        seq = event.get("seq", index)
        locations = []
        if kind == "tool":
            if "args" in event:
                locations.append((f"events[{seq}].args", event["args"], "TOOL_ARGUMENT"))
            if "result" in event:
                locations.append((f"events[{seq}].result", event["result"], "MODEL_INPUT"))
        else:
            role = str(event.get("role") or "").lower()
            location = (
                "MODEL_OUTPUT"
                if kind == "model_output" or role == "assistant"
                else "MODEL_INPUT"
                if kind in {"llm", "model", "model_input"}
                else "TRACE"
            )
            locations.append((f"events[{seq}]", event, location))
        for root, value, location in locations:
            sources.append((root, value, location, str(seq)))
    sources.append((
        "trajectory.output",
        context.trajectory.output,
        "MODEL_OUTPUT",
        context.trajectory.trajectory_id,
    ))
    for index, artifact in enumerate(context.trajectory.artifact_refs):
        sources.append((
            f"artifacts[{index}].path",
            artifact.path,
            "ARTIFACT_PATH",
            str(index),
        ))

    for root, value, location, source_id in sources:
            for path, text in _walk(value, root):
                for phi_type, pattern in PHI_PATTERNS.items():
                    for match_index, match in enumerate(pattern.finditer(text), 1):
                        locator = f"{path}#{phi_type}:{match_index}"
                        finding_id = _finding_id(
                            asset, context, f"PHI_{phi_type}", locator
                        )
                        evidence = {
                            "source_type": (
                                "ARTIFACT_REF"
                                if location == "ARTIFACT_PATH"
                                else "APPLICATION_EVENT"
                            ),
                            "source_id": source_id,
                            "location": location,
                            "path": path,
                            "phi_type": phi_type,
                            "fingerprint": _fingerprint(match.group(0)),
                            "value_redacted": True,
                        }
                        findings.append(AuditFinding(
                            finding_id=finding_id,
                            rule_ref=asset.ref,
                            trajectory_id=context.trajectory.trajectory_id,
                            target_ref=_target(context),
                            kind=f"PHI_{phi_type}",
                            severity=(
                                "IRB"
                                if location in {
                                    "MODEL_OUTPUT", "ARTIFACT_PATH"
                                }
                                else "WARN"
                            ),
                            message=(
                                f"{phi_type} detected at {location}; value withheld"
                            ),
                            evidence=(evidence,),
                        ))
                        metadata[finding_id] = evidence

    incidents: list[AuditIncident] = []
    external = context.provider_boundary.upper() in {
        "EXTERNAL",
        "THIRD_PARTY",
        "CROSS_TRUST_BOUNDARY",
    }
    model_findings = tuple(
        row.finding_id
        for row in findings
        if metadata[row.finding_id]["location"]
        in {"MODEL_INPUT", "MODEL_OUTPUT"}
    )
    if external and model_findings:
        incidents.append(AuditIncident(
            incident_id=_incident_id(
                asset,
                context,
                "PHI_EXTERNAL_MODEL_BOUNDARY",
                model_findings,
            ),
            rule_ref=asset.ref,
            trajectory_id=context.trajectory.trajectory_id,
            target_ref=_target(context),
            kind="PHI_EXTERNAL_MODEL_BOUNDARY",
            severity="IRB",
            finding_ids=model_findings,
            rationale=(
                "PHI-bearing content crossed an external provider boundary; "
                "matched values remain withheld"
            ),
        ))
    for index, event in enumerate(context.application_events):
        if str(event.get("kind") or "").lower() != "tool":
            continue
        if not _outbound_event(event):
            continue
        prefix = f"events[{event.get('seq', index)}]"
        linked = tuple(
            row.finding_id
            for row in findings
            if str(metadata[row.finding_id]["path"]).startswith(prefix)
        )
        if linked:
            incidents.append(AuditIncident(
                incident_id=_incident_id(
                    asset, context, "PHI_OUTBOUND_ACTION", linked
                ),
                rule_ref=asset.ref,
                trajectory_id=context.trajectory.trajectory_id,
                target_ref=_target(context),
                kind="PHI_OUTBOUND_ACTION",
                severity="IRB",
                finding_ids=linked,
                rationale=(
                    "PHI and an external destination occur in the same tool event"
                ),
            ))
    return tuple(findings), tuple(incidents)


def undeclared_tool_audit(
    asset: ModuleAsset, context: AuditContext
) -> tuple[tuple[AuditFinding, ...], tuple[AuditIncident, ...]]:
    if not context.declared_tools:
        return (), ()
    allowed = set(context.declared_tools)
    findings = []
    for index, event in enumerate(context.application_events):
        if str(event.get("kind") or "").lower() != "tool":
            continue
        tool = str(event.get("tool") or event.get("name") or "")
        if tool in allowed:
            continue
        seq = event.get("seq", index)
        findings.append(AuditFinding(
            finding_id=_finding_id(
                asset, context, "UNDECLARED_TOOL", f"event:{seq}:{tool}"
            ),
            rule_ref=asset.ref,
            trajectory_id=context.trajectory.trajectory_id,
            target_ref=_target(context, "TOOL_CALL", str(seq)),
            kind="UNDECLARED_TOOL",
            severity="CRITICAL",
            message="run invoked a tool absent from its declared tool bundle",
            evidence=({
                "source_type": "APPLICATION_EVENT",
                "source_id": str(seq),
                "tool": tool,
            },),
        ))
    incidents = tuple(
        AuditIncident(
            incident_id=_incident_id(
                asset, context, row.kind, (row.finding_id,)
            ),
            rule_ref=asset.ref,
            trajectory_id=context.trajectory.trajectory_id,
            target_ref=row.target_ref,
            kind=row.kind,
            severity="CRITICAL",
            finding_ids=(row.finding_id,),
            rationale=row.message,
        )
        for row in findings
    )
    return tuple(findings), incidents


def local_artifact_audit(
    asset: ModuleAsset, context: AuditContext
) -> tuple[tuple[AuditFinding, ...], tuple[AuditIncident, ...]]:
    if not context.local_root:
        return (), ()
    local_root = Path(context.local_root).resolve()
    git_root = Path(context.git_root).resolve() if context.git_root else None
    findings = []
    for index, artifact in enumerate(context.trajectory.artifact_refs):
        path = Path(artifact.path).resolve(strict=False)
        try:
            path.relative_to(local_root)
            within_local = True
        except ValueError:
            within_local = False
        within_git = False
        if git_root is not None:
            try:
                path.relative_to(git_root)
                within_git = True
            except ValueError:
                pass
        if within_local and not within_git:
            continue
        locator = f"artifact:{index}"
        findings.append(AuditFinding(
            finding_id=_finding_id(
                asset, context, "ARTIFACT_BOUNDARY", locator
            ),
            rule_ref=asset.ref,
            trajectory_id=context.trajectory.trajectory_id,
            target_ref=_target(context),
            kind="ARTIFACT_BOUNDARY",
            severity="IRB",
            message="patient-derived artifact escaped the declared local store",
            evidence=({
                "source_type": "ARTIFACT_REF",
                "source_id": str(index),
                "path_fingerprint": _fingerprint(str(path)),
                "within_local_root": within_local,
                "within_git_root": within_git,
                "value_redacted": True,
            },),
        ))
    incidents = tuple(
        AuditIncident(
            incident_id=_incident_id(
                asset, context, row.kind, (row.finding_id,)
            ),
            rule_ref=asset.ref,
            trajectory_id=context.trajectory.trajectory_id,
            target_ref=row.target_ref,
            kind=row.kind,
            severity="IRB",
            finding_ids=(row.finding_id,),
            rationale=row.message,
        )
        for row in findings
    )
    return tuple(findings), incidents


def trajectory_integrity_audit(
    asset: ModuleAsset, context: AuditContext
) -> tuple[tuple[AuditFinding, ...], tuple[AuditIncident, ...]]:
    findings = []
    for index, artifact in enumerate(context.trajectory.artifact_refs):
        path = Path(artifact.path)
        if not path.is_file():
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if hmac.compare_digest(actual, artifact.sha256):
            continue
        findings.append(AuditFinding(
            finding_id=_finding_id(
                asset, context, "ARTIFACT_HASH_MISMATCH", f"artifact:{index}"
            ),
            rule_ref=asset.ref,
            trajectory_id=context.trajectory.trajectory_id,
            target_ref=_target(context),
            kind="ARTIFACT_HASH_MISMATCH",
            severity="CRITICAL",
            message="referenced run artifact no longer matches its recorded hash",
            evidence=({
                "source_type": "ARTIFACT_REF",
                "source_id": str(index),
                "expected_sha256": artifact.sha256,
                "actual_sha256": actual,
            },),
        ))
    incidents = tuple(
        AuditIncident(
            incident_id=_incident_id(
                asset, context, row.kind, (row.finding_id,)
            ),
            rule_ref=asset.ref,
            trajectory_id=context.trajectory.trajectory_id,
            target_ref=row.target_ref,
            kind=row.kind,
            severity="CRITICAL",
            finding_ids=(row.finding_id,),
            rationale=row.message,
        )
        for row in findings
    )
    return tuple(findings), incidents


def runtime_control_conformance_audit(
    asset: ModuleAsset, context: AuditContext
) -> tuple[tuple[AuditFinding, ...], tuple[AuditIncident, ...]]:
    """Detect a recorded hard-control denial that the final run bypassed.

    This rule does not ask whether the gate is clinically effective.  It only
    checks whether the implementation obeyed its own recorded deterministic
    decision.
    """
    output_status = str(context.trajectory.output.get("status") or "")
    output_values = context.trajectory.output.get("value") or {}
    output_fields = (
        set(output_values) if isinstance(output_values, Mapping) else set()
    )
    findings = []
    for index, event in enumerate(context.application_events):
        kind = str(event.get("kind") or "").lower()
        seq = event.get("seq", index)
        refused = set()
        for key in (
            "refused_fields",
            "inadmissible_fields",
            "rejected_fields",
            "disallowed_fields",
        ):
            value = event.get(key)
            if isinstance(value, str):
                refused.add(value)
            elif isinstance(value, (list, tuple, set)):
                refused.update(str(row) for row in value)
        verdict = event.get("verdict")
        if isinstance(verdict, Mapping):
            refused.update(
                str(field)
                for field, accepted in verdict.items()
                if accepted is False
            )
        denied = (
            kind in {"gate_decision", "answer_gate"}
            and str(
                event.get("decision")
                or event.get("outcome")
                or event.get("status")
                or ""
            ).upper()
            in {"DENY", "REJECT", "REJECTED", "FAIL"}
        )
        overlapping = sorted(refused & output_fields)
        if output_status != "FOUND" or (not denied and not overlapping):
            continue
        finding_kind = (
            "HARD_GATE_BYPASS"
            if denied
            else "ADMISSIBILITY_CONTROL_BYPASS"
        )
        findings.append(AuditFinding(
            finding_id=_finding_id(
                asset, context, finding_kind, f"event:{seq}"
            ),
            rule_ref=asset.ref,
            trajectory_id=context.trajectory.trajectory_id,
            target_ref=_target(context, "GATE_DECISION", str(seq)),
            kind=finding_kind,
            severity="CRITICAL",
            message=(
                "final FOUND output contradicts a recorded deterministic "
                "runtime-control denial"
            ),
            evidence=({
                "source_type": "APPLICATION_EVENT",
                "source_id": str(seq),
                "refused_output_fields": overlapping,
                "final_status": output_status,
            },),
        ))
    incidents = tuple(
        AuditIncident(
            incident_id=_incident_id(
                asset, context, row.kind, (row.finding_id,)
            ),
            rule_ref=asset.ref,
            trajectory_id=context.trajectory.trajectory_id,
            target_ref=row.target_ref,
            kind=row.kind,
            severity="CRITICAL",
            finding_ids=(row.finding_id,),
            rationale=row.message,
        )
        for row in findings
    )
    return tuple(findings), incidents


def _audit_asset(
    module_id: str, implementation_id: str, description: str
) -> ModuleAsset:
    return ModuleAsset(
        module_id=module_id,
        version="1.0.0",
        module_kind="AUDIT_RULE",
        runner_type="CODE",
        input_channels=("trajectory", "application_events"),
        output_schema="acr.audit_report/1",
        implementation_id=implementation_id,
        supported_truth_modes=("BLIND",),
        maximum_authority="QUARANTINE_RESULT",
        description=description,
        owner="platform-governance",
        tags=("audit", "application-events"),
    )


def builtin_audit_registry() -> AuditRuleRegistry:
    registry = AuditRuleRegistry()
    for asset, implementation in (
        (
            _audit_asset(
                "patient-boundary-audit",
                "audit.patient_boundary.v1",
                "Detect same-run tool calls naming another patient.",
            ),
            patient_boundary_audit,
        ),
        (
            _audit_asset(
                "phi-provider-audit",
                "audit.phi_provider.v1",
                "Detect PHI locations and correlated provider/outbound crossings.",
            ),
            phi_provider_audit,
        ),
        (
            _audit_asset(
                "undeclared-tool-audit",
                "audit.undeclared_tool.v1",
                "Detect tools outside the declared run bundle.",
            ),
            undeclared_tool_audit,
        ),
        (
            _audit_asset(
                "local-artifact-audit",
                "audit.local_artifact.v1",
                "Verify patient-derived artifacts remain in the local store.",
            ),
            local_artifact_audit,
        ),
        (
            _audit_asset(
                "trajectory-integrity-audit",
                "audit.trajectory_integrity.v1",
                "Verify content-addressed run artifacts remain reproducible.",
            ),
            trajectory_integrity_audit,
        ),
        (
            _audit_asset(
                "runtime-control-conformance-audit",
                "audit.runtime_control_conformance.v1",
                "Detect final results that bypass recorded hard-control denials.",
            ),
            runtime_control_conformance_audit,
        ),
    ):
        registry.register(asset, implementation)
    return registry
