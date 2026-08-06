"""Application-level ACR audit plane.

Audit is truth-blind and deterministic.  It answers whether the recorded run
crossed an operational, privacy, or integrity boundary; it does not judge the
clinical answer.  Runtime/eBPF ingestion is intentionally left as a future
adapter behind ``runtime_evidence_refs``.
"""
from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ..core import site
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

#: The site's person-id shape, from `core/site.py`. It used to be a literal here, written as a
#: string concatenation so that `tests/test_no_phi_in_tree.py`'s byte scan would not flag the file
#: containing it — which is a sign the literal did not belong in the tree at all.
#: May be `None` when this deployment has not declared an identifier shape; the rule
#: then contributes nothing, which is the honest state rather than a silent pass.
_INSTITUTIONAL_PERSON = site.PERSON_ID
_EMAIL = re.compile(r"(?<![\w.-])[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}(?![\w.-])")
_PHONE = re.compile(
    r"(?<!\d)(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}(?!\d)"
)
_MRN = re.compile(
    r"(?i)\b(?:mrn|medical record(?: number)?)\s*[:#-]?\s*[A-Z0-9-]{5,20}\b"
)
#: `DOB: 1/2/1956` AND `DOB: 1956-06-22`. The second form was missing, and what that cost is worth
#: recording: the first real run of this plane found an MRN in a submitted evidence quote whose text
#: was `MRN: SYN0009 / Patient: … / DOB: 1956-06-22` — a document header the agent had quoted whole.
#: The MRN rule fired. The DOB rule did not, because it required `\d{1,2}[/-]\d{1,2}[/-]\d{2,4}` and
#: the corpus writes ISO. One identifier reported out of the three that were sitting in one string.
_DOB = re.compile(
    r"(?i)\b(?:dob|date of birth)\s*[:#-]?\s*"
    r"(?:(?:\d{1,2}[/-]){2}\d{2,4}|\d{4}-\d{2}-\d{2})\b"
)
_URL = re.compile(r"https?://([^/\s:]+)")

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

#: Re-exported so a reader of this module sees which key it uses without chasing an import.
PSEUDONYM_KEY_ENV = site.PSEUDONYM_KEY_ENV


def _fingerprint(value: str) -> str:
    """`site.fingerprint`. See it for why this is not implemented here."""
    return site.fingerprint(value)

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
        if self.local_root and not Path(self.local_root).is_absolute():
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
    #: Per-rule: what that rule was able to examine. A rule that produced no findings may have
    #: found nothing OR looked at nothing, and `status: PASS` cannot tell them apart — which is how
    #: an IRB-severity rule reading four argument names no tool declares sat in every report's
    #: `rule_refs` for weeks reading as a satisfied check. A rule with no entry here made no claim
    #: about its own coverage; a rule with one did.
    basis: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

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
            "basis": {k: dict(v) for k, v in self.basis.items()},
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
        basis: dict[str, Mapping[str, Any]] = {}
        for asset in assets:
            implementation = self.registry.modules.implementation(asset)
            module_findings, module_incidents = implementation(asset, context)
            findings.extend(module_findings)
            incidents.extend(module_incidents)
            # A rule may declare what it examined by exposing `basis(context)`. Optional on
            # purpose: a rule that makes no claim about its own coverage says nothing here, which
            # is more honest than a default that would read as a claim.
            reporter = BASIS_REPORTERS.get(asset.implementation_id)
            if reporter is not None:
                basis[asset.ref] = reporter(context)
        report = AuditReport(
            trajectory_id=context.trajectory.trajectory_id,
            rule_refs=tuple(asset.ref for asset in assets),
            findings=tuple(findings),
            incidents=tuple(incidents),
            input_hash=context.input_hash,
            basis=basis,
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
                "assets/pricing/findings.jsonl",
                signal.to_dict(),
                idempotency_key=signal.signal_id,
            )
        for incident in report.incidents:
            asset = by_ref[incident.rule_ref]
            signal = incident.to_signal(asset.asset_ref)
            self.local_store.append_jsonl(
                "assets/pricing/incidents.jsonl",
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

#: The argument names this rule looks for. NO TOOL IN `TOOL_SCHEMAS` DECLARES ANY OF THEM, and a
#: scan of this tree's whole run history found 8,866 arg-bearing trace events with ZERO
#: intersection. Kept rather than deleted, because the reason is structural and worth asserting:
#: `Toolbox` binds exactly ONE `PatientChart`, so no tool can be asked about another subject and
#: the emptiness is TRUE. What was wrong was that a zero read as "no crossover occurred" when it
#: meant "nothing looked" — see `_boundary_basis`. If a tool ever takes a subject argument, this
#: fires without anyone remembering to re-enable it.
#:
#: The repo's precedent for the other case is a few hundred lines below: on 2026-08-03
#: `runtime_control_conformance_audit` was DELETED for reading four trace keys nothing writes.
#: The difference is that its emptiness was not structurally guaranteed by anything.
_SUBJECT_ARG_NAMES = ("patient", "patient_id", "person_id", "subject_id")


def _boundary_basis(context: AuditContext) -> dict[str, object]:
    """What this rule was able to examine, so a zero can be read.

    `AuditFinding` has no `examined` field and this rule may legitimately produce no findings, so
    the count rides on the INCIDENT this function feeds. An audit that cannot say "I had nothing to
    look at" reproduces, one level up, the defect that deleted the rule below.
    """
    args_seen = sum(1 for e in context.application_events
                    if isinstance(e.get("args"), Mapping) and e.get("args"))
    subject_args = sum(1 for e in context.application_events
                       if isinstance(e.get("args"), Mapping)
                       and set(e["args"]) & set(_SUBJECT_ARG_NAMES))
    return {"examined": subject_args,
            "arg_bearing_events": args_seen,
            "person_id_pattern_configured": site.PERSON_ID is not None,
            "why_zero": ("no tool in the declared surface accepts a subject argument, and "
                         "`Toolbox` binds exactly one chart, so an empty result is structural "
                         "rather than unexamined" if subject_args == 0 else "")}


def patient_boundary_audit(
    asset: ModuleAsset, context: AuditContext
) -> tuple[tuple[AuditFinding, ...], tuple[AuditIncident, ...]]:
    findings = []
    expected = {context.patient_scope, context.trajectory.case_ref}
    for index, event in enumerate(context.application_events):
        args = event.get("args") if isinstance(event.get("args"), Mapping) else {}
        for key in _SUBJECT_ARG_NAMES:
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

# 2026-08-03 `runtime_control_conformance_audit` used to sit here: a rule checking whether the final
# result had bypassed a hard-control refusal that was on the record. It imported nothing, so deleting
# the RuntimeControl protocol left it looking untouched — what it read was the trace events'
# `refused_fields` / `inadmissible_fields` / `rejected_fields` / `disallowed_fields`. Those four keys
# are written in ZERO places under `src/` and occur zero times across the 454 manifests and traces
# under runs/. Which is to say: from the day it was written it could never once have fired.
#
# This is the standard shape of what deleting a mechanism leaves behind, and the deletion that left
# this one behind was mine. A rule that reads events rather than importing a type is invisible on the
# dependency graph; the only way to judge it dead is to ask who else writes the keys it reads.

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

def _artifact_basis(context: AuditContext) -> dict[str, object]:
    """How many artifacts a rule that walks `artifact_refs` actually had to walk.

    Both `local_artifact_audit` and `trajectory_integrity_audit` iterate
    `context.trajectory.artifact_refs` and produce nothing when it is empty — which is the correct
    outcome for a clean run AND for a caller that passed no artifacts at all. Those are different
    facts. `tools/verify_mechanisms.py`'s M5 found this by building a context with `artifact_refs=()`
    and watching both rules report zero: exactly the reading a report gave with no way to tell.
    """
    refs = tuple(getattr(context.trajectory, "artifact_refs", ()) or ())
    on_disk = sum(1 for a in refs if Path(a.path).is_file())
    return {"examined": on_disk,
            "artifacts_declared": len(refs),
            "local_root_configured": bool(context.local_root),
            "why_zero": ("no artifact was declared for this trajectory, so this rule examined "
                         "nothing — that is a caller omission, not a clean result"
                         if not refs else
                         "every declared artifact is missing from disk" if on_disk == 0 else "")}


def _declared_tool_basis(context: AuditContext) -> dict[str, object]:
    """How many tool calls this rule compared against the declared surface.

    It returns early when `declared_tools` is empty, so a caller that forgot to pass the surface
    got zero findings and a clean-looking report. `tools/verify_mechanisms.py` M5 hit exactly that:
    it read the tool name from `t["name"]` where `TOOL_SCHEMAS` is OpenAI-style
    (`t["function"]["name"]`), passed `("",)`, and 1,688 spurious findings read as the rule working.
    Fixing the fixture made it produce nothing — correctly, since no run calls an undeclared tool —
    and that zero needs to be legible.
    """
    calls = [e for e in context.application_events
             if str(e.get("kind") or "").lower() == "tool"]
    return {"examined": len(calls) if context.declared_tools else 0,
            "declared_surface_size": len(context.declared_tools),
            "why_zero": ("no declared tool surface was supplied, so nothing could be compared"
                         if not context.declared_tools else
                         "the trace records no tool call" if not calls else "")}


#: `implementation_id` -> a callable saying what that rule EXAMINED, for the report's `basis`.
#: A registry rather than a protocol on the rule itself, because a rule is a plain function and
#: adding a required second return value would touch every one of them. Only rules whose emptiness
#: is ambiguous need an entry — the ones where "no findings" and "nothing looked" print the same.
BASIS_REPORTERS: dict[str, Callable[["AuditContext"], Mapping[str, Any]]] = {
    "audit.patient_boundary.v1": _boundary_basis,
    "audit.local_artifact.v1": _artifact_basis,
    "audit.trajectory_integrity.v1": _artifact_basis,
    "audit.undeclared_tool.v1": _declared_tool_basis,
}


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
                "undeclared-tool-audit",
                "audit.undeclared_tool.v1",
                "Detect tools outside the declared run bundle.",
            ),
            undeclared_tool_audit,
        ),
        (
            _audit_asset(
                "trajectory-integrity-audit",
                "audit.trajectory_integrity.v1",
                "Verify content-addressed run artifacts remain reproducible.",
            ),
            trajectory_integrity_audit,
        ),
    ):
        registry.register(asset, implementation)
    return registry
