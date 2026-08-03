"""Explicit, local-only module assets and composition profiles.

Only independently runnable units are modules.  Skills are referenced assets,
capabilities are grants, and stages remain private to a module implementation.
YAML assets may refer only to implementation IDs registered by Python code; this
module intentionally provides no dynamic import facility.
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import yaml

from .kernel import AssetRef, digest

#: 一个 kind 只有在既有 Protocol 又有实现时才留在这里。2026-08-03 删掉了 `RUNTIME_CONTROL`
#: 和 `REPAIR_STRATEGY`：两者的 Protocol 在同一天随各自唯一的实现一起删除
#: (`review/runtime_controls.py`、`improvement/repair_loop.py`，生产代码零引用)，而 kind 留了
#: 下来 —— 于是 `assets/module_catalog/runtime_controls/` 里五份 YAML 继续通过 `__post_init__`
#: 的 kind 校验并被 `from_directory` 正常加载，声明着没有任何代码能运行的资产。校验放行是因为
#: 名单里有这个名字，而名单是删除时唯一没人想起来改的地方。
MODULE_KINDS = frozenset({
    "RUNTIME_POLICY",
    "AUDIT_RULE",
    "EVALUATOR",
})
RUNNER_TYPES = frozenset({"CODE", "LLM", "AGENT", "HUMAN"})
TRUTH_MODES = frozenset({"BLIND", "REGISTRY_REFERENCE", "GOLD"})
AUTHORITIES = (
    "OBSERVE",
    "SCREEN_ONLY",
    "SCREEN_AND_ROUTE",
    "BLOCK_CURRENT_ANSWER",
    "QUARANTINE_RESULT",
    "BLOCK_RELEASE",
    "HUMAN_DECISION",
)
_AUTHORITY_RANK = {name: index for index, name in enumerate(AUTHORITIES)}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]*$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


class ModuleContractError(ValueError):
    """A module asset, profile, task grant, or implementation is invalid."""


@dataclass(frozen=True)
class CapabilityRequest:
    name: str
    scope: str = "patient_under_review"
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.name):
            raise ModuleContractError(f"invalid capability name {self.name!r}")
        if not _ID.fullmatch(self.scope):
            raise ModuleContractError(f"invalid capability scope {self.scope!r}")
        if not _VERSION.fullmatch(self.version):
            raise ModuleContractError(
                f"{self.name}: invalid capability version {self.version!r}"
            )

    @classmethod
    def from_value(
        cls, value: str | Mapping[str, Any]
    ) -> CapabilityRequest:
        if isinstance(value, str):
            return cls(value)
        if not isinstance(value, Mapping):
            raise ModuleContractError(
                "capability entries must be strings or mappings"
            )
        unknown = set(value) - {"name", "scope", "version"}
        if unknown:
            raise ModuleContractError(
                f"unknown capability fields {sorted(unknown)}"
            )
        return cls(
            name=str(value.get("name") or ""),
            scope=str(value.get("scope") or "patient_under_review"),
            version=str(value.get("version") or "1.0.0"),
        )


@dataclass(frozen=True)
class ModuleAsset:
    """Definition of one independently runnable and certifiable module."""

    module_id: str
    version: str
    module_kind: str
    input_channels: tuple[str, ...]
    output_schema: str
    implementation_id: str
    runner_type: str = "CODE"
    supported_truth_modes: tuple[str, ...] = ("BLIND",)
    requested_capabilities: tuple[CapabilityRequest, ...] = ()
    prompt_ref: AssetRef | None = None
    skill_refs: tuple[AssetRef, ...] = ()
    maximum_authority: str = "OBSERVE"
    description: str = ""
    owner: str = ""
    tags: tuple[str, ...] = ()
    source: str = ""

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.module_id):
            raise ModuleContractError(f"invalid module_id {self.module_id!r}")
        if not _VERSION.fullmatch(self.version):
            raise ModuleContractError(
                f"{self.module_id}: invalid version {self.version!r}"
            )
        if self.module_kind not in MODULE_KINDS:
            raise ModuleContractError(
                f"{self.module_id}: unknown module kind {self.module_kind!r}"
            )
        if self.runner_type not in RUNNER_TYPES:
            raise ModuleContractError(
                f"{self.module_id}: unknown runner type {self.runner_type!r}"
            )
        if (
            self.module_kind in {
                "RUNTIME_POLICY", "RUNTIME_CONTROL", "AUDIT_RULE"
            }
            and self.runner_type != "CODE"
        ):
            raise ModuleContractError(
                f"{self.module_kind} modules must use the CODE runner"
            )
        if not self.input_channels or any(not item for item in self.input_channels):
            raise ModuleContractError(
                f"{self.module_id}: input_channels cannot be empty"
            )
        if len(set(self.input_channels)) != len(self.input_channels):
            raise ModuleContractError(
                f"{self.module_id}: duplicate input channel"
            )
        if not self.output_schema.strip() or not _ID.fullmatch(self.implementation_id):
            raise ModuleContractError(
                f"{self.module_id}: output schema and implementation ID are required"
            )
        unknown_truth = set(self.supported_truth_modes) - TRUTH_MODES
        if unknown_truth:
            raise ModuleContractError(
                f"{self.module_id}: unknown truth modes {sorted(unknown_truth)}"
            )
        if self.module_kind == "AUDIT_RULE" and set(
            self.supported_truth_modes
        ) != {"BLIND"}:
            raise ModuleContractError(
                f"{self.module_id}: audit rules must be truth-blind"
            )
        if self.maximum_authority not in _AUTHORITY_RANK:
            raise ModuleContractError(
                f"{self.module_id}: unknown authority {self.maximum_authority!r}"
            )
        names = [row.name for row in self.requested_capabilities]
        if len(names) != len(set(names)):
            raise ModuleContractError(
                f"{self.module_id}: duplicate requested capability"
            )
        for ref in self.skill_refs:
            if ref.asset_type != "SKILL":
                raise ModuleContractError(
                    f"{self.module_id}: skill_refs may contain only SKILL assets"
                )
        if self.prompt_ref is not None and self.prompt_ref.asset_type != "PROMPT":
            raise ModuleContractError(
                f"{self.module_id}: prompt_ref must be a PROMPT asset"
            )

    @property
    def ref(self) -> str:
        return f"{self.module_id}@{self.version}"

    @property
    def asset_ref(self) -> AssetRef:
        asset_type = {
            "RUNTIME_POLICY": "RUNTIME_PROFILE",
            "RUNTIME_CONTROL": "TOOL",
            "AUDIT_RULE": "AUDIT_RULE",
            "EVALUATOR": "EVALUATOR",
            "REPAIR_STRATEGY": "REPAIR_STRATEGY",
        }[self.module_kind]
        return AssetRef(
            asset_id=self.module_id,
            asset_type=asset_type,
            version=self.version,
            content_hash=self.content_hash,
            local_ref=self.source,
        )

    @property
    def content_hash(self) -> str:
        return digest(self.to_dict(include_source=False))

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, source: str = ""
    ) -> ModuleAsset:
        allowed = {
            "schema",
            "module_id",
            "version",
            "kind",
            "runner",
            "inputs",
            "output_schema",
            "implementation",
            "truth_modes",
            "capabilities",
            "prompt_ref",
            "skills",
            "maximum_authority",
            "description",
            "owner",
            "tags",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ModuleContractError(
                f"{source or 'module'}: unknown fields {sorted(unknown)}"
            )

        def asset_from_mapping(raw: Any, expected_type: str) -> AssetRef | None:
            if raw in (None, ""):
                return None
            if not isinstance(raw, Mapping):
                raise ModuleContractError(
                    f"{source}: {expected_type.lower()} reference must be a mapping"
                )
            return AssetRef(
                asset_id=str(raw.get("asset_id") or ""),
                asset_type=expected_type,
                version=str(raw.get("version") or ""),
                content_hash=str(raw.get("content_hash") or ""),
                local_ref=str(raw.get("local_ref") or ""),
                status=str(raw.get("status") or "DRAFT"),
            )

        skills = []
        for raw in value.get("skills") or ():
            ref = asset_from_mapping(raw, "SKILL")
            if ref is not None:
                skills.append(ref)
        return cls(
            module_id=str(value.get("module_id") or ""),
            version=str(value.get("version") or ""),
            module_kind=str(value.get("kind") or ""),
            runner_type=str(value.get("runner") or "CODE"),
            input_channels=tuple(str(row) for row in value.get("inputs") or ()),
            output_schema=str(value.get("output_schema") or ""),
            implementation_id=str(value.get("implementation") or ""),
            supported_truth_modes=tuple(
                str(row) for row in value.get("truth_modes") or ("BLIND",)
            ),
            requested_capabilities=tuple(
                CapabilityRequest.from_value(row)
                for row in value.get("capabilities") or ()
            ),
            prompt_ref=asset_from_mapping(value.get("prompt_ref"), "PROMPT"),
            skill_refs=tuple(skills),
            maximum_authority=str(
                value.get("maximum_authority") or "OBSERVE"
            ),
            description=str(value.get("description") or ""),
            owner=str(value.get("owner") or ""),
            tags=tuple(str(row) for row in value.get("tags") or ()),
            source=source,
        )

    def to_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        value = {
            "schema": "acr.module_asset/1",
            "module_id": self.module_id,
            "version": self.version,
            "kind": self.module_kind,
            "runner": self.runner_type,
            "inputs": list(self.input_channels),
            "output_schema": self.output_schema,
            "implementation": self.implementation_id,
            "truth_modes": list(self.supported_truth_modes),
            "capabilities": [asdict(row) for row in self.requested_capabilities],
            "prompt_ref": self.prompt_ref.to_dict() if self.prompt_ref else None,
            "skills": [row.to_dict() for row in self.skill_refs],
            "maximum_authority": self.maximum_authority,
            "description": self.description,
            "owner": self.owner,
            "tags": list(self.tags),
        }
        if include_source:
            value["source"] = self.source
        return value


def load_module_asset(path: str | Path) -> ModuleAsset:
    source = Path(path).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ModuleContractError(f"{source}: expected a mapping")
    if value.get("schema") not in {None, "acr.module_asset/1"}:
        raise ModuleContractError(f"{source}: unsupported module schema")
    return ModuleAsset.from_dict(value, source=str(source))


Implementation = Callable[..., Any]


class ModuleRegistry:
    """Asset registry plus an explicit allowlist of local implementations."""

    def __init__(self) -> None:
        self._assets: dict[str, ModuleAsset] = {}
        self._versions: dict[str, list[ModuleAsset]] = defaultdict(list)
        self._implementations: dict[str, Implementation] = {}

    @classmethod
    def from_directory(
        cls,
        root: str | Path,
        *,
        implementations: Mapping[str, Implementation] | None = None,
    ) -> ModuleRegistry:
        directory = Path(root)
        if not directory.is_dir():
            raise ModuleContractError(
                f"module catalog is not a directory: {directory}"
            )
        registry = cls()
        for path in sorted(directory.rglob("*.yaml")):
            registry.register_asset(load_module_asset(path))
        if not registry._assets:
            raise ModuleContractError(f"module catalog is empty: {directory}")
        for implementation_id, implementation in dict(
            implementations or {}
        ).items():
            registry.register_implementation(
                implementation_id, implementation
            )
        return registry

    def register_asset(self, asset: ModuleAsset) -> None:
        if asset.ref in self._assets:
            raise ModuleContractError(f"duplicate module asset {asset.ref}")
        self._assets[asset.ref] = asset
        self._versions[asset.module_id].append(asset)
        self._versions[asset.module_id].sort(key=lambda row: row.version)

    def register_implementation(
        self, implementation_id: str, implementation: Implementation
    ) -> None:
        if not _ID.fullmatch(implementation_id):
            raise ModuleContractError(
                f"invalid implementation ID {implementation_id!r}"
            )
        if implementation_id in self._implementations:
            raise ModuleContractError(
                f"duplicate implementation {implementation_id}"
            )
        if not callable(implementation):
            raise ModuleContractError(
                f"implementation {implementation_id} is not callable"
            )
        self._implementations[implementation_id] = implementation

    def resolve(self, ref: str) -> ModuleAsset:
        if "@" in ref:
            try:
                return self._assets[ref]
            except KeyError as exc:
                raise ModuleContractError(f"unknown module ref {ref!r}") from exc
        versions = self._versions.get(ref, ())
        if not versions:
            raise ModuleContractError(f"unknown module {ref!r}")
        return versions[-1]

    def implementation(self, asset: ModuleAsset) -> Implementation:
        try:
            return self._implementations[asset.implementation_id]
        except KeyError as exc:
            raise ModuleContractError(
                f"{asset.ref}: implementation {asset.implementation_id!r} "
                "is not explicitly registered"
            ) from exc

    def validate(self) -> None:
        missing = sorted({
            asset.implementation_id
            for asset in self._assets.values()
            if asset.implementation_id not in self._implementations
        })
        if missing:
            raise ModuleContractError(
                f"unregistered module implementations: {missing}"
            )

    def all_assets(self) -> tuple[ModuleAsset, ...]:
        return tuple(
            sorted(self._assets.values(), key=lambda row: (row.module_id, row.version))
        )


@dataclass(frozen=True)
class TaskBudget:
    max_calls: int = 0
    max_chart_reads: int = 0
    max_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.max_calls < 0 or self.max_chart_reads < 0 or self.max_usd < 0:
            raise ModuleContractError("task budgets cannot be negative")

    def narrows(self, ceiling: TaskBudget) -> bool:
        return (
            self.max_calls <= ceiling.max_calls
            and self.max_chart_reads <= ceiling.max_chart_reads
            and self.max_usd <= ceiling.max_usd
        )


@dataclass(frozen=True)
class PipelineNode:
    node_id: str
    module_ref: str
    after: tuple[str, ...] = ()
    when: str = "always"
    input_mapping: Mapping[str, str] = field(default_factory=dict)
    allowed_capabilities: tuple[str, ...] = ()
    budget: TaskBudget = field(default_factory=TaskBudget)
    authority: str = "OBSERVE"

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.node_id):
            raise ModuleContractError(f"invalid pipeline node {self.node_id!r}")
        if not str(self.module_ref).strip() or not _ID.fullmatch(self.when):
            raise ModuleContractError(
                f"{self.node_id}: module_ref and when are required"
            )
        if self.node_id in self.after:
            raise ModuleContractError(f"{self.node_id}: cannot depend on itself")
        if self.authority not in _AUTHORITY_RANK:
            raise ModuleContractError(
                f"{self.node_id}: unknown authority {self.authority!r}"
            )
        if len(set(self.allowed_capabilities)) != len(
            self.allowed_capabilities
        ):
            raise ModuleContractError(
                f"{self.node_id}: duplicate allowed capability"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PipelineNode:
        budget = value.get("budget") or {}
        if not isinstance(budget, Mapping):
            raise ModuleContractError("pipeline node budget must be a mapping")
        return cls(
            node_id=str(value.get("id") or ""),
            module_ref=str(value.get("module") or ""),
            after=tuple(str(row) for row in value.get("after") or ()),
            when=str(value.get("when") or "always"),
            input_mapping={
                str(key): str(child)
                for key, child in dict(value.get("inputs") or {}).items()
            },
            allowed_capabilities=tuple(
                str(row) for row in value.get("capabilities") or ()
            ),
            budget=TaskBudget(
                max_calls=int(budget.get("max_calls", 0)),
                max_chart_reads=int(budget.get("max_chart_reads", 0)),
                max_usd=float(budget.get("max_usd", 0.0)),
            ),
            authority=str(value.get("authority") or "OBSERVE"),
        )


@dataclass(frozen=True)
class PipelineProfile:
    profile_id: str
    version: str
    nodes: tuple[PipelineNode, ...]
    description: str = ""
    source: str = ""

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.profile_id) or not _VERSION.fullmatch(self.version):
            raise ModuleContractError("pipeline profile needs valid ID and version")
        if not self.nodes:
            raise ModuleContractError(f"{self.profile_id}: pipeline is empty")
        ids = [node.node_id for node in self.nodes]
        if len(ids) != len(set(ids)):
            raise ModuleContractError(
                f"{self.profile_id}: duplicate pipeline node"
            )
        known = set(ids)
        missing = sorted({
            dependency
            for node in self.nodes
            for dependency in node.after
            if dependency not in known
        })
        if missing:
            raise ModuleContractError(
                f"{self.profile_id}: missing node dependencies {missing}"
            )
        self.execution_order()

    @property
    def ref(self) -> str:
        return f"{self.profile_id}@{self.version}"

    @property
    def content_hash(self) -> str:
        return digest(self.to_dict(include_source=False))

    def execution_order(self) -> tuple[PipelineNode, ...]:
        rows = {node.node_id: node for node in self.nodes}
        indegree = {node.node_id: len(node.after) for node in self.nodes}
        children: dict[str, list[str]] = defaultdict(list)
        for node in self.nodes:
            for dependency in node.after:
                children[dependency].append(node.node_id)
        queue = deque(sorted(key for key, value in indegree.items() if value == 0))
        ordered = []
        while queue:
            node_id = queue.popleft()
            ordered.append(rows[node_id])
            for child in sorted(children[node_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(ordered) != len(rows):
            cycle = sorted(key for key, value in indegree.items() if value)
            raise ModuleContractError(
                f"{self.profile_id}: pipeline dependency cycle {cycle}"
            )
        return tuple(ordered)

    def validate_modules(self, registry: ModuleRegistry) -> None:
        for node in self.nodes:
            asset = registry.resolve(node.module_ref)
            if asset.module_kind not in {"EVALUATOR", "AUDIT_RULE"}:
                raise ModuleContractError(
                    f"{node.node_id}: {asset.module_kind} cannot be an analysis "
                    "pipeline node"
                )
            if _AUTHORITY_RANK[node.authority] > _AUTHORITY_RANK[
                asset.maximum_authority
            ]:
                raise ModuleContractError(
                    f"{node.node_id}: pipeline authority exceeds {asset.ref} maximum"
                )
            requested = {row.name for row in asset.requested_capabilities}
            if not set(node.allowed_capabilities) <= requested:
                raise ModuleContractError(
                    f"{node.node_id}: pipeline grants undeclared capabilities "
                    f"{sorted(set(node.allowed_capabilities) - requested)}"
                )
            if asset.runner_type in {"CODE", "HUMAN"} and (
                node.budget.max_calls or node.budget.max_chart_reads
            ):
                raise ModuleContractError(
                    f"{node.node_id}: {asset.runner_type} runner cannot have "
                    "model or chart-read budget"
                )
            if asset.runner_type == "LLM" and (
                node.budget.max_calls != 1
                or node.budget.max_chart_reads
                or node.allowed_capabilities
            ):
                raise ModuleContractError(
                    f"{node.node_id}: LLM runner requires one model call and no tools"
                )
            if asset.runner_type == "AGENT" and node.budget.max_calls < 1:
                raise ModuleContractError(
                    f"{node.node_id}: AGENT runner requires max_calls >= 1"
                )
            if asset.runner_type != "AGENT" and node.allowed_capabilities:
                raise ModuleContractError(
                    f"{node.node_id}: only AGENT evaluators receive capabilities"
                )

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, source: str = ""
    ) -> PipelineProfile:
        return cls(
            profile_id=str(value.get("profile_id") or ""),
            version=str(value.get("version") or "1.0.0"),
            nodes=tuple(
                PipelineNode.from_dict(row) for row in value.get("nodes") or ()
            ),
            description=str(value.get("description") or ""),
            source=source,
        )

    def to_dict(self, *, include_source: bool = True) -> dict[str, Any]:
        value = {
            "schema": "acr.pipeline_profile/1",
            "profile_id": self.profile_id,
            "version": self.version,
            "description": self.description,
            "nodes": [
                {
                    "id": row.node_id,
                    "module": row.module_ref,
                    "after": list(row.after),
                    "when": row.when,
                    "inputs": dict(row.input_mapping),
                    "capabilities": list(row.allowed_capabilities),
                    "budget": asdict(row.budget),
                    "authority": row.authority,
                }
                for row in self.nodes
            ],
        }
        if include_source:
            value["source"] = self.source
        return value


def load_pipeline_profile(path: str | Path) -> PipelineProfile:
    source = Path(path).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ModuleContractError(f"{source}: expected a mapping")
    if value.get("schema") not in {None, "acr.pipeline_profile/1"}:
        raise ModuleContractError(f"{source}: unsupported pipeline schema")
    return PipelineProfile.from_dict(value, source=str(source))


class PipelineRegistry:
    def __init__(self) -> None:
        self._profiles: dict[str, PipelineProfile] = {}
        self._versions: dict[str, list[PipelineProfile]] = defaultdict(list)

    @classmethod
    def from_directory(cls, root: str | Path) -> PipelineRegistry:
        directory = Path(root)
        if not directory.is_dir():
            raise ModuleContractError(
                f"pipeline catalog is not a directory: {directory}"
            )
        registry = cls()
        for path in sorted(directory.glob("*.yaml")):
            registry.register(load_pipeline_profile(path))
        if not registry._profiles:
            raise ModuleContractError(f"pipeline catalog is empty: {directory}")
        return registry

    def register(self, profile: PipelineProfile) -> None:
        if profile.ref in self._profiles:
            raise ModuleContractError(f"duplicate pipeline {profile.ref}")
        self._profiles[profile.ref] = profile
        self._versions[profile.profile_id].append(profile)
        self._versions[profile.profile_id].sort(key=lambda row: row.version)

    def resolve(self, ref: str) -> PipelineProfile:
        if "@" in ref:
            try:
                return self._profiles[ref]
            except KeyError as exc:
                raise ModuleContractError(f"unknown pipeline {ref!r}") from exc
        versions = self._versions.get(ref, ())
        if not versions:
            raise ModuleContractError(f"unknown pipeline {ref!r}")
        return versions[-1]

    def all(self) -> tuple[PipelineProfile, ...]:
        return tuple(
            sorted(self._profiles.values(), key=lambda row: row.ref)
        )


@dataclass(frozen=True)
class CertificationSuite:
    suite_id: str
    version: str
    module_ref: str
    must_pass: tuple[str, ...]
    must_fail: tuple[str, ...]
    calibration_cohort_ref: str = ""
    metric_thresholds: Mapping[str, float] = field(default_factory=dict)
    max_cost_usd: float = 0.0
    required_citation_validity: float = 1.0
    source: str = ""

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.suite_id) or not _VERSION.fullmatch(self.version):
            raise ModuleContractError("certification suite needs valid ID and version")
        if not self.module_ref.strip():
            raise ModuleContractError(
                f"{self.suite_id}: module_ref is required"
            )
        if not self.must_pass or not self.must_fail:
            raise ModuleContractError(
                f"{self.suite_id}: must-pass and must-fail fixtures are required"
            )
        overlap = set(self.must_pass) & set(self.must_fail)
        if overlap:
            raise ModuleContractError(
                f"{self.suite_id}: certification fixtures overlap {sorted(overlap)}"
            )
        if self.max_cost_usd < 0:
            raise ModuleContractError("certification cost cannot be negative")
        if not 0 <= self.required_citation_validity <= 1:
            raise ModuleContractError(
                "required citation validity must be between zero and one"
            )

    @property
    def ref(self) -> str:
        return f"{self.suite_id}@{self.version}"

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any], *, source: str = ""
    ) -> CertificationSuite:
        return cls(
            suite_id=str(value.get("suite_id") or ""),
            version=str(value.get("version") or "1.0.0"),
            module_ref=str(value.get("module_ref") or ""),
            must_pass=tuple(str(row) for row in value.get("must_pass") or ()),
            must_fail=tuple(str(row) for row in value.get("must_fail") or ()),
            calibration_cohort_ref=str(
                value.get("calibration_cohort_ref") or ""
            ),
            metric_thresholds={
                str(key): float(child)
                for key, child in dict(
                    value.get("metric_thresholds") or {}
                ).items()
            },
            max_cost_usd=float(value.get("max_cost_usd", 0.0)),
            required_citation_validity=float(
                value.get("required_citation_validity", 1.0)
            ),
            source=source,
        )


def load_certification_suite(path: str | Path) -> CertificationSuite:
    source = Path(path).resolve()
    value = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ModuleContractError(f"{source}: expected a mapping")
    if value.get("schema") not in {None, "acr.certification_suite/1"}:
        raise ModuleContractError(
            f"{source}: unsupported certification schema"
        )
    return CertificationSuite.from_dict(value, source=str(source))


class CertificationRegistry:
    def __init__(self) -> None:
        self._suites: dict[str, CertificationSuite] = {}
        self._by_module: dict[str, list[CertificationSuite]] = defaultdict(list)

    @classmethod
    def from_directory(cls, root: str | Path) -> CertificationRegistry:
        directory = Path(root)
        if not directory.is_dir():
            raise ModuleContractError(
                f"certification catalog is not a directory: {directory}"
            )
        registry = cls()
        for path in sorted(directory.glob("*.yaml")):
            registry.register(load_certification_suite(path))
        if not registry._suites:
            raise ModuleContractError(
                f"certification catalog is empty: {directory}"
            )
        return registry

    def register(self, suite: CertificationSuite) -> None:
        if suite.ref in self._suites:
            raise ModuleContractError(
                f"duplicate certification suite {suite.ref}"
            )
        self._suites[suite.ref] = suite
        self._by_module[suite.module_ref].append(suite)

    def resolve(self, ref: str) -> CertificationSuite:
        try:
            return self._suites[ref]
        except KeyError as exc:
            raise ModuleContractError(
                f"unknown certification suite {ref!r}"
            ) from exc

    def for_module(self, module_ref: str) -> tuple[CertificationSuite, ...]:
        return tuple(self._by_module.get(module_ref, ()))

    def all(self) -> tuple[CertificationSuite, ...]:
        return tuple(
            sorted(self._suites.values(), key=lambda row: row.ref)
        )

    def validate_modules(self, modules: ModuleRegistry) -> None:
        for suite in self._suites.values():
            asset = modules.resolve(suite.module_ref)
            if asset.module_kind not in {"EVALUATOR", "AUDIT_RULE"}:
                raise ModuleContractError(
                    f"{suite.ref}: cannot certify {asset.module_kind} in the "
                    "analysis certification registry"
                )


def effective_capabilities(
    asset: ModuleAsset,
    node: PipelineNode,
    task_grants: Mapping[str, str],
) -> tuple[CapabilityRequest, ...]:
    """Intersect requested, pipeline-allowed, and task-granted capabilities."""
    requested = {row.name: row for row in asset.requested_capabilities}
    pipeline = set(node.allowed_capabilities)
    task = set(task_grants)
    undeclared = task - set(requested)
    disallowed = task - pipeline
    if undeclared or disallowed:
        raise ModuleContractError(
            f"{node.node_id}: task attempted to expand capabilities; "
            f"undeclared={sorted(undeclared)}, "
            f"outside_pipeline={sorted(disallowed)}"
        )
    granted = pipeline & task & set(requested)
    rows = []
    for name in sorted(granted):
        task_scope = str(task_grants[name])
        if task_scope != requested[name].scope:
            raise ModuleContractError(
                f"{node.node_id}: task scope {task_scope!r} differs from "
                f"declared {requested[name].scope!r} for {name}"
            )
        rows.append(
            CapabilityRequest(
                name=name,
                scope=task_scope,
                version=requested[name].version,
            )
        )
    return tuple(rows)


def narrowed_authority(asset: ModuleAsset, node: PipelineNode, task: str) -> str:
    """Return the narrowest authority granted by asset, profile, and task."""
    for value in (asset.maximum_authority, node.authority, task):
        if value not in _AUTHORITY_RANK:
            raise ModuleContractError(f"unknown authority {value!r}")
    ceiling = min(
        (asset.maximum_authority, node.authority),
        key=_AUTHORITY_RANK.__getitem__,
    )
    if _AUTHORITY_RANK[task] > _AUTHORITY_RANK[ceiling]:
        raise ModuleContractError(
            f"{node.node_id}: task authority {task} expands ceiling {ceiling}"
        )
    return task


@runtime_checkable
class RuntimePolicy(Protocol):
    def plan(self, run_context: Any) -> Any: ...
    def revise(self, state: Any, new_evidence: Any) -> Any: ...
    def should_stop(self, state: Any) -> Any: ...


@runtime_checkable
class AuditRule(Protocol):
    def inspect(self, trajectory: Any, application_events: Any) -> Any: ...
    def correlate(self, findings: Any) -> Any: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(self, context: Any) -> Any: ...


# 2026-08-03 这里曾有 `RuntimeControl` 和 `RepairStrategy` 两个 protocol，各自的唯一实现
# (`review/runtime_controls.py`、`improvement/repair_loop.py`) 在生产代码里零引用，删除时把
# protocol 一起带走：一个没有实现者的 protocol，是同一份死代码往上挪了一层，而且读起来像是
# 系统有这个能力。真正的强制在 `review/answer_gate.py`(答案义务)、`core/spend.py:74`
# (预算上限)、`review/agent.py` 的 `_undeclared`(工具白名单)、`core/local_artifacts.py`
# (病人衍生数据不出 worktree)，以及"一次运行只绑一个 PatientChart"这个对象图事实里。
