"""Dependency-checked attribution module registry."""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ModuleRegistryError(ValueError):
    pass


Validator = Callable[[Any, Any], list[str]]


@dataclass(frozen=True)
class AttributionStage:
    """One internal stage of the causal-attribution evaluator.

    A stage is intentionally not an independently runnable evaluator module: it
    has no standalone output or certification lifecycle.
    """
    module_id: str
    category: str
    description: str
    requires: tuple[str, ...] = ()
    tool_names: tuple[str, ...] = ()
    instructions: str = ""
    validate: Validator | None = None
    version: str = "1.0.0"

    def __post_init__(self) -> None:
        if not self.module_id or not self.category or not self.description:
            raise ModuleRegistryError("module_id, category, and description are required")
        if self.module_id in self.requires:
            raise ModuleRegistryError(f"{self.module_id}: cannot depend on itself")


@dataclass(frozen=True)
class AttributionStageProfile:
    name: str
    modules: tuple[str, ...]
    description: str


class AttributionStageRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, AttributionStage] = {}
        self._profiles: dict[str, AttributionStageProfile] = {}

    def register(self, module: AttributionStage) -> None:
        if module.module_id in self._modules:
            raise ModuleRegistryError(f"duplicate attribution module {module.module_id}")
        self._modules[module.module_id] = module

    def register_profile(self, profile: AttributionStageProfile) -> None:
        if profile.name in self._profiles:
            raise ModuleRegistryError(f"duplicate attribution profile {profile.name}")
        self._profiles[profile.name] = profile

    def module(self, module_id: str) -> AttributionStage:
        try:
            return self._modules[module_id]
        except KeyError as exc:
            raise ModuleRegistryError(f"unknown attribution module {module_id!r}") from exc

    def profile(self, name: str) -> tuple[AttributionStage, ...]:
        try:
            profile = self._profiles[name]
        except KeyError as exc:
            raise ModuleRegistryError(
                f"unknown attribution profile {name!r}; have {sorted(self._profiles)}"
            ) from exc
        return self.resolve(profile.modules)

    def resolve(self, requested: tuple[str, ...]) -> tuple[AttributionStage, ...]:
        selected: dict[str, AttributionStage] = {}

        def add(module_id: str) -> None:
            module = self.module(module_id)
            if module_id in selected:
                return
            selected[module_id] = module
            for dependency in module.requires:
                add(dependency)

        for module_id in requested:
            add(module_id)
        indegree = {module_id: 0 for module_id in selected}
        children: dict[str, list[str]] = defaultdict(list)
        for module in selected.values():
            for dependency in module.requires:
                indegree[module.module_id] += 1
                children[dependency].append(module.module_id)
        queue = deque(sorted(key for key, n in indegree.items() if n == 0))
        result: list[AttributionStage] = []
        while queue:
            module_id = queue.popleft()
            result.append(selected[module_id])
            for child in sorted(children[module_id]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    queue.append(child)
        if len(result) != len(selected):
            raise ModuleRegistryError(
                f"attribution module dependency cycle: "
                f"{sorted(key for key, n in indegree.items() if n)}"
            )
        return tuple(result)

    def all_modules(self) -> tuple[AttributionStage, ...]:
        return tuple(self._modules[key] for key in sorted(self._modules))

    def profiles(self) -> tuple[AttributionStageProfile, ...]:
        return tuple(self._profiles[key] for key in sorted(self._profiles))
