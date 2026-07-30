"""Composable post-run evaluation over canonical trajectories.

Evaluator assets, pipeline bindings, task grants, and certification suites are
separate versioned contracts.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from functools import partial
from typing import Any

from .kernel import (
    SignalEnvelope,
    SignalEvidenceRef,
    TargetRef,
    Trajectory,
    digest,
)
from .local_artifacts import LocalArtifactStore
from .modules import (
    CapabilityRequest,
    ModuleAsset,
    ModuleContractError,
    ModuleRegistry,
    PipelineNode,
    PipelineRegistry,
    TaskBudget,
    effective_capabilities,
    narrowed_authority,
)

EVALUATION_STATUSES = frozenset({
    "PASS",
    "FAIL",
    "FLAG",
    "UNRESOLVED",
    "SKIPPED",
    "QUEUED",
})
_FORBIDDEN_CHANNEL_NAMES = frozenset({
    "answer_key",
    "chart_gold",
    "expected_output",
    "gold",
    "ground_truth",
    "registry_reference",
    "truth",
})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvaluationPipelineError(ValueError):
    """An evaluation asset, context, task, or execution crossed a boundary."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _contains_truth_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_CHANNEL_NAMES:
                return True
            if _contains_truth_key(child):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_truth_key(child) for child in value)
    return False


@dataclass(frozen=True)
class InputChannel:
    """A typed evaluator input that is not truth."""

    name: str
    schema: str
    value: Any = None
    artifact_ref: str = ""
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.schema.strip():
            raise EvaluationPipelineError(
                "input channel name and schema are required"
            )
        if self.name.strip().lower() in _FORBIDDEN_CHANNEL_NAMES:
            raise EvaluationPipelineError(
                f"truth must use TruthContext, not input channel {self.name!r}"
            )
        if self.value is not None and self.artifact_ref:
            raise EvaluationPipelineError(
                f"{self.name}: channel cannot embed a value and an artifact ref"
            )
        if self.value is None and not self.artifact_ref:
            raise EvaluationPipelineError(
                f"{self.name}: channel needs a value or artifact ref"
            )
        if self.value is not None and _contains_truth_key(self.value):
            raise EvaluationPipelineError(
                f"{self.name}: truth-like data is forbidden in ordinary channels"
            )
        if self.content_hash and not _SHA256.fullmatch(self.content_hash):
            raise EvaluationPipelineError(
                f"{self.name}: content_hash must be sha256"
            )

    @property
    def resolved_hash(self) -> str:
        return self.content_hash or digest(
            self.value if self.value is not None else self.artifact_ref
        )


@dataclass(frozen=True)
class TruthContext:
    """Truth is a separate capability, never a generic evaluator channel."""

    mode: str
    chart_gold: Mapping[str, Any] | None = None
    registry_reference: Mapping[str, Any] | None = None
    adjudication_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in {"BLIND", "REGISTRY_REFERENCE", "GOLD"}:
            raise EvaluationPipelineError(f"unknown truth mode {self.mode!r}")
        if self.mode == "BLIND":
            if self.chart_gold is not None or self.registry_reference is not None:
                raise EvaluationPipelineError(
                    "BLIND TruthContext cannot contain truth or registry values"
                )
        elif self.mode == "REGISTRY_REFERENCE":
            if self.registry_reference is None or self.chart_gold is not None:
                raise EvaluationPipelineError(
                    "REGISTRY_REFERENCE needs only an unresolved registry reference"
                )
            if str(
                self.registry_reference.get("adjudication") or ""
            ) != "UNRESOLVED":
                raise EvaluationPipelineError(
                    "registry reference must remain adjudication=UNRESOLVED"
                )
        elif self.chart_gold is None or self.registry_reference is not None:
            raise EvaluationPipelineError(
                "GOLD mode needs adjudicated chart-observable gold only"
            )
        if self.mode == "GOLD" and not self.adjudication_refs:
            raise EvaluationPipelineError(
                "GOLD TruthContext needs an accountable adjudication reference"
            )


@dataclass(frozen=True)
class EvaluationContext:
    trajectory: Trajectory
    spec_snapshot: InputChannel
    channels: tuple[InputChannel, ...] = ()
    truth: TruthContext = field(
        default_factory=lambda: TruthContext("BLIND")
    )
    audit_signal_refs: tuple[str, ...] = ()
    patient_scope: str = ""
    provider_boundary: str = "UNKNOWN"

    def __post_init__(self) -> None:
        if self.spec_snapshot.name != "spec":
            raise EvaluationPipelineError(
                "spec_snapshot must use the canonical 'spec' channel"
            )
        if self.patient_scope and self.patient_scope != self.trajectory.case_ref:
            raise EvaluationPipelineError(
                "evaluation patient scope differs from trajectory case"
            )
        names = [self.spec_snapshot.name, *(row.name for row in self.channels)]
        if len(names) != len(set(names)):
            raise EvaluationPipelineError(
                "evaluation context contains duplicate channels"
            )

    @property
    def truth_mode(self) -> str:
        return self.truth.mode

    @property
    def input_hash(self) -> str:
        return digest({
            "trajectory": self.trajectory.content_hash,
            "spec": self.spec_snapshot.resolved_hash,
            "channels": {
                row.name: row.resolved_hash for row in self.channels
            },
            "truth_mode": self.truth.mode,
            "truth_hash": digest(asdict(self.truth)),
            "audit_signal_refs": self.audit_signal_refs,
            "patient_scope_hash": digest(
                self.patient_scope or self.trajectory.case_ref
            ),
            "provider_boundary": self.provider_boundary,
        })

    def resolve_channel(self, name: str) -> InputChannel:
        if name == "trajectory":
            return InputChannel(
                name="trajectory",
                schema="acr.trajectory/1",
                value=self.trajectory.to_dict(),
            )
        if name == "output":
            return InputChannel(
                name="output",
                schema="acr.extraction_output/1",
                value=dict(self.trajectory.output),
            )
        if name == "spec":
            return self.spec_snapshot
        for row in self.channels:
            if row.name == name:
                return row
        raise EvaluationPipelineError(
            f"trajectory {self.trajectory.trajectory_id} lacks channel {name!r}"
        )


@dataclass(frozen=True)
class EvaluationResult:
    """Quality result only; audit findings and incidents are different payloads."""

    result_id: str
    module_ref: str
    trajectory_id: str
    target_ref: TargetRef
    status: str
    authority: str
    output_schema: str
    score: float | None = None
    reason: str = ""
    evidence_refs: tuple[SignalEvidenceRef, ...] = ()
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in EVALUATION_STATUSES:
            raise EvaluationPipelineError(
                f"unknown evaluation status {self.status!r}"
            )
        if not self.module_ref.strip() or not self.output_schema.strip():
            raise EvaluationPipelineError(
                "evaluation result needs module_ref and output_schema"
            )
        forbidden = {"findings", "incidents", "audit_incident"}
        if forbidden & set(self.payload):
            raise EvaluationPipelineError(
                "EvaluationResult cannot carry Audit Finding/Incident payloads"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "acr.evaluation_result/2",
            "result_id": self.result_id,
            "module_ref": self.module_ref,
            "trajectory_id": self.trajectory_id,
            "target_ref": self.target_ref.to_dict(),
            "status": self.status,
            "authority": self.authority,
            "output_schema": self.output_schema,
            "score": self.score,
            "reason": self.reason,
            "evidence_refs": [row.to_dict() for row in self.evidence_refs],
            "payload": dict(self.payload),
        }

    def to_signal(self, producer: ModuleAsset) -> SignalEnvelope:
        return SignalEnvelope(
            signal_id=self.result_id,
            signal_type="EVALUATION_RESULT",
            producer_ref=producer.asset_ref,
            trajectory_ref=self.trajectory_id,
            target_ref=self.target_ref,
            status=self.status,
            severity="ERROR" if self.status == "FAIL" else "INFO",
            evidence_refs=self.evidence_refs,
            payload_schema="acr.evaluation_result/2",
            payload=self.to_dict(),
        )


@dataclass(frozen=True)
class EvaluationInvocation:
    asset: ModuleAsset
    node: PipelineNode
    context: EvaluationContext
    inputs: Mapping[str, InputChannel]
    capabilities: tuple[CapabilityRequest, ...]
    budget: TaskBudget
    authority: str
    model_binding: str = ""
    seed: int | None = None
    prior_results: Mapping[str, EvaluationResult] = field(default_factory=dict)


EvaluatorImplementation = Callable[
    [EvaluationInvocation, Mapping[str, Callable[..., Any]]],
    EvaluationResult,
]
Condition = Callable[
    [EvaluationContext, Mapping[str, EvaluationResult]], bool
]


class CapabilityBroker:
    """Runtime enforcement for already-intersected evaluator capabilities."""

    def __init__(
        self,
        invocation: EvaluationInvocation,
        implementations: Mapping[str, Callable[..., Any]],
    ):
        self.invocation = invocation
        self._grants = {row.name: row for row in invocation.capabilities}
        self._implementations = dict(implementations)
        self.audit: list[dict[str, Any]] = []
        self.chart_reads = 0

    def call(self, capability: str, **kwargs: Any) -> Any:
        if capability not in self._grants:
            raise EvaluationPipelineError(
                f"{self.invocation.asset.ref}: capability {capability!r} "
                "was not effectively granted"
            )
        try:
            implementation = self._implementations[capability]
        except KeyError as exc:
            raise EvaluationPipelineError(
                f"{self.invocation.asset.ref}: no implementation for "
                f"capability {capability!r}"
            ) from exc
        grant = self._grants[capability]
        patient = str(
            kwargs.get("patient_id")
            or kwargs.get("subject_id")
            or kwargs.get("person_id")
            or ""
        )
        expected = (
            self.invocation.context.patient_scope
            or self.invocation.context.trajectory.case_ref
        )
        if grant.scope == "patient_under_review" and patient and patient != expected:
            raise EvaluationPipelineError(
                f"{capability}: cross-patient access denied"
            )
        if capability == "patient-chart-reader":
            self.chart_reads += 1
            maximum = self.invocation.budget.max_chart_reads
            if maximum >= 0 and self.chart_reads > maximum:
                raise EvaluationPipelineError(
                    f"{capability}: chart-read budget exhausted"
                )
        result = implementation(**kwargs)
        self.audit.append({
            "capability": capability,
            "scope": grant.scope,
            "status": "OK",
            "patient_scope_hash": digest(expected),
        })
        return result


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    pipeline_ref: str
    trajectory_ids: tuple[str, ...]
    truth_mode: str
    model_bindings: Mapping[str, str] = field(default_factory=dict)
    budgets: Mapping[str, TaskBudget] = field(default_factory=dict)
    capability_grants: Mapping[str, Mapping[str, str]] = field(
        default_factory=dict
    )
    authority_grants: Mapping[str, str] = field(default_factory=dict)
    local_output_root: str = ""
    seeds: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.pipeline_ref.strip():
            raise EvaluationPipelineError(
                "evaluation task needs task_id and pipeline_ref"
            )
        if not self.trajectory_ids:
            raise EvaluationPipelineError(
                "evaluation task needs at least one trajectory"
            )
        if self.truth_mode not in {"BLIND", "REGISTRY_REFERENCE", "GOLD"}:
            raise EvaluationPipelineError(
                f"unknown task truth mode {self.truth_mode!r}"
            )
        if self.local_output_root and not self.local_output_root.startswith("/"):
            raise EvaluationPipelineError(
                "evaluation local_output_root must be absolute"
            )


class EvaluationStore:
    def __init__(self, local_store: LocalArtifactStore):
        self.local_store = local_store

    def add(self, result: EvaluationResult, asset: ModuleAsset) -> None:
        signal = result.to_signal(asset)
        self.local_store.append_jsonl(
            "evaluation/results.jsonl",
            signal.to_dict(),
            idempotency_key=signal.signal_id,
        )

    def rows(self) -> list[dict[str, Any]]:
        path = self.local_store.path(
            "evaluation/results.jsonl", what="evaluation result ledger"
        )
        if not path.is_file():
            return []
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def summary(self) -> dict[str, Any]:
        rows = self.rows()
        by_status: dict[str, int] = {}
        by_module: dict[str, int] = {}
        trajectories: set[str] = set()
        for row in rows:
            payload = row.get("payload") or {}
            # The ledger stores a SignalEnvelope whose payload is the complete
            # EvaluationResult.  That result has its own evaluator-specific ``payload``.
            # Descending through both layers discards module_ref/trajectory_id and made every
            # real summary report module UNKNOWN.
            result = (
                payload
                if payload.get("schema") == "acr.evaluation_result/2"
                else payload.get("payload") or payload
            )
            status = str(result.get("status") or row.get("status") or "UNKNOWN")
            module = str(result.get("module_ref") or "UNKNOWN")
            trajectory = str(
                result.get("trajectory_id") or row.get("trajectory_ref") or ""
            )
            by_status[status] = by_status.get(status, 0) + 1
            by_module[module] = by_module.get(module, 0) + 1
            if trajectory:
                trajectories.add(trajectory)
        return {
            "schema": "acr.evaluation_summary/2",
            "n_results": len(rows),
            "n_trajectories": len(trajectories),
            "by_status": dict(sorted(by_status.items())),
            "by_module": dict(sorted(by_module.items())),
        }


class EvaluationPipelineRunner:
    """Execute evaluator modules without hard-coding evaluator IDs."""

    def __init__(
        self,
        module_registry: ModuleRegistry,
        pipeline_registry: PipelineRegistry,
        *,
        conditions: Mapping[str, Condition] | None = None,
        store: EvaluationStore | None = None,
    ):
        self.modules = module_registry
        self.pipelines = pipeline_registry
        self.conditions: dict[str, Condition] = {
            "always": lambda context, prior: True,
            "abnormal_or_disagreement": lambda context, prior: (
                any(row.status not in {"PASS", "SKIPPED"} for row in prior.values())
                or any(
                    channel.name == "disagreement"
                    for channel in context.channels
                )
            ),
            "cited_answer_and_high_yield_unread": lambda context, prior: (
                bool(context.trajectory.output)
                and any(
                    channel.name == "documents_not_read"
                    for channel in context.channels
                )
            ),
        }
        self.conditions.update(dict(conditions or {}))
        self.store = store

    def run(
        self,
        task: EvaluationTask,
        contexts: Sequence[EvaluationContext],
        *,
        capability_implementations: Mapping[
            str, Mapping[str, Callable[..., Any]]
        ] | None = None,
    ) -> Mapping[str, tuple[EvaluationResult, ...]]:
        profile = self.pipelines.resolve(task.pipeline_ref)
        profile.validate_modules(self.modules)
        wanted = set(task.trajectory_ids)
        supplied = {row.trajectory.trajectory_id for row in contexts}
        missing = sorted(wanted - supplied)
        if missing:
            raise EvaluationPipelineError(
                f"task trajectories are missing from contexts: {missing}"
            )
        capabilities = dict(capability_implementations or {})
        output: dict[str, tuple[EvaluationResult, ...]] = {}
        for context in contexts:
            trajectory_id = context.trajectory.trajectory_id
            if trajectory_id not in wanted:
                continue
            if context.truth_mode != task.truth_mode:
                raise EvaluationPipelineError(
                    f"{trajectory_id}: context/task truth modes differ"
                )
            prior: dict[str, EvaluationResult] = {}
            results: list[EvaluationResult] = []
            for node in profile.execution_order():
                asset = self.modules.resolve(node.module_ref)
                if asset.module_kind != "EVALUATOR":
                    raise EvaluationPipelineError(
                        f"{node.node_id}: audit modules run in the audit plane"
                    )
                if context.truth_mode not in asset.supported_truth_modes:
                    continue
                try:
                    condition = self.conditions[node.when]
                except KeyError as exc:
                    raise EvaluationPipelineError(
                        f"{node.node_id}: unregistered condition {node.when!r}"
                    ) from exc
                if not condition(context, prior):
                    continue
                requested_inputs = {}
                for channel_name in asset.input_channels:
                    source_name = node.input_mapping.get(
                        channel_name, channel_name
                    )
                    requested_inputs[channel_name] = context.resolve_channel(
                        source_name
                    )
                budget = task.budgets.get(node.node_id, node.budget)
                if node.node_id in task.budgets and not budget.narrows(node.budget):
                    raise EvaluationPipelineError(
                        f"{node.node_id}: task budget expands pipeline ceiling"
                    )
                task_caps = task.capability_grants.get(node.node_id, {})
                try:
                    granted = effective_capabilities(asset, node, task_caps)
                except ModuleContractError as exc:
                    raise EvaluationPipelineError(str(exc)) from exc
                task_authority = task.authority_grants.get(
                    node.node_id, node.authority
                )
                try:
                    authority = narrowed_authority(
                        asset, node, task_authority
                    )
                except ModuleContractError as exc:
                    raise EvaluationPipelineError(str(exc)) from exc
                model_binding = task.model_bindings.get(node.node_id, "")
                if asset.runner_type in {"LLM", "AGENT"} and not model_binding:
                    raise EvaluationPipelineError(
                        f"{node.node_id}: {asset.runner_type} runner needs a "
                        "task model binding"
                    )
                if asset.runner_type in {"CODE", "HUMAN"} and model_binding:
                    raise EvaluationPipelineError(
                        f"{node.node_id}: {asset.runner_type} runner cannot "
                        "receive a model binding"
                    )
                seed = task.seeds[0] if task.seeds else None
                invocation = EvaluationInvocation(
                    asset=asset,
                    node=node,
                    context=context,
                    inputs=requested_inputs,
                    capabilities=granted,
                    budget=budget,
                    authority=authority,
                    model_binding=model_binding,
                    seed=seed,
                    prior_results=dict(prior),
                )
                implementation = self.modules.implementation(asset)
                broker = CapabilityBroker(
                    invocation, capabilities.get(node.node_id, {})
                )
                result = implementation(invocation, {
                    name: partial(broker.call, name)
                    for name in (row.name for row in granted)
                })
                if not isinstance(result, EvaluationResult):
                    raise EvaluationPipelineError(
                        f"{asset.ref}: implementation returned "
                        f"{type(result).__name__}, expected EvaluationResult"
                    )
                if result.module_ref != asset.ref:
                    raise EvaluationPipelineError(
                        f"{asset.ref}: result claims {result.module_ref}"
                    )
                if result.authority != authority:
                    raise EvaluationPipelineError(
                        f"{asset.ref}: result authority differs from effective grant"
                    )
                prior[node.node_id] = result
                results.append(result)
                if self.store is not None:
                    self.store.add(result, asset)
            output[trajectory_id] = tuple(results)
        return output


def make_result(
    invocation: EvaluationInvocation,
    *,
    status: str,
    target_ref: TargetRef | None = None,
    score: float | None = None,
    reason: str = "",
    evidence_refs: Sequence[SignalEvidenceRef] = (),
    payload: Mapping[str, Any] | None = None,
) -> EvaluationResult:
    target = target_ref or TargetRef(
        "RUN", invocation.context.trajectory.trajectory_id
    )
    result_id = "ER-" + digest([
        invocation.asset.ref,
        invocation.context.input_hash,
        target.to_dict(),
        status,
        score,
        payload or {},
    ])[:20]
    return EvaluationResult(
        result_id=result_id,
        module_ref=invocation.asset.ref,
        trajectory_id=invocation.context.trajectory.trajectory_id,
        target_ref=target,
        status=status,
        authority=invocation.authority,
        output_schema=invocation.asset.output_schema,
        score=score,
        reason=reason,
        evidence_refs=tuple(evidence_refs),
        payload=dict(payload or {}),
    )
