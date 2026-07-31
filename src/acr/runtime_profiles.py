"""Versioned runtime-policy profiles for coverage ablation.

These policies decide how far to search.  They do not judge clinical correctness
and they do not replace deterministic runtime controls such as patient scope or
the final evidence gate.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .modules import ModuleAsset, ModuleContractError, ModuleRegistry
from .skills import SkillStack

WITNESS_FIRST_PROFILE = "witness-first-baseline"
STRATIFIED_COVERAGE_PROFILE = "current-stratified-coverage"
GUIDELINE_ONLY_PROFILE = "guideline-only"
CONDITIONAL_COVERAGE_PROFILE = "conditional-negative-coverage"
ALWAYS_COVERAGE_PROFILE = "always-coverage"
DEFAULT_RUNTIME_PROFILE = STRATIFIED_COVERAGE_PROFILE

COVERAGE_NONE = "NONE"
COVERAGE_ON_NEGATIVE_OR_MISSING = "ON_NEGATIVE_OR_MISSING"
COVERAGE_ALWAYS = "ALWAYS"


@dataclass(frozen=True)
class RuntimePolicyContext:
    case_ref: str
    spec_snapshot: Mapping[str, Any]
    positive_terms: tuple[str, ...] = ()
    document_types: tuple[str, ...] = ()
    coverage_strata: tuple[str, ...] = ()
    max_rounds: int = 8
    max_documents: int = 100

    def __post_init__(self) -> None:
        if not self.case_ref.strip():
            raise ModuleContractError("runtime policy context needs case_ref")
        if self.max_rounds < 1 or self.max_documents < 1:
            raise ModuleContractError("runtime policy budgets must be positive")


@dataclass(frozen=True)
class SearchPlan:
    policy_ref: str
    strategy: str
    search_terms: tuple[str, ...]
    document_types: tuple[str, ...]
    required_strata: tuple[str, ...]
    proof_obligations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_ref": self.policy_ref,
            "strategy": self.strategy,
            "search_terms": list(self.search_terms),
            "document_types": list(self.document_types),
            "required_strata": list(self.required_strata),
            "proof_obligations": list(self.proof_obligations),
        }


@dataclass(frozen=True)
class RuntimePolicyState:
    witness_found: bool = False
    proof_valid: bool = False
    completed_strata: tuple[str, ...] = ()
    required_strata: tuple[str, ...] = ()
    open_conflicts: int = 0
    open_threads: int = 0
    rounds: int = 0
    documents_read: int = 0
    new_evidence_in_last_round: bool = True


@dataclass(frozen=True)
class SearchDecision:
    action: str
    reason: str
    added_terms: tuple[str, ...] = ()
    added_document_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class StopDecision:
    stop: bool
    reason: str
    answer_disposition: str


class WitnessFirstPolicy:
    """Stop after a grounded witness and closed discovered conflicts."""

    module_ref = f"{WITNESS_FIRST_PROFILE}@1.0.0"

    def plan(self, run_context: RuntimePolicyContext) -> SearchPlan:
        return SearchPlan(
            policy_ref=self.module_ref,
            strategy="WITNESS_FIRST",
            search_terms=run_context.positive_terms,
            document_types=run_context.document_types,
            required_strata=(),
            proof_obligations=("grounded_positive_witness",),
        )

    def revise(
        self, state: RuntimePolicyState, new_evidence: Mapping[str, Any]
    ) -> SearchDecision:
        if state.open_conflicts or state.open_threads:
            return SearchDecision(
                "EXPAND_TARGETED",
                "close already-discovered conflict or evidence thread",
                added_terms=tuple(
                    str(row) for row in new_evidence.get("targeted_terms") or ()
                ),
                added_document_types=tuple(
                    str(row)
                    for row in new_evidence.get("targeted_document_types") or ()
                ),
            )
        return SearchDecision(
            "CONTINUE_PRIMARY",
            "no grounded witness yet",
        )

    def should_stop(self, state: RuntimePolicyState) -> StopDecision:
        if (
            state.witness_found
            and state.proof_valid
            and not state.open_conflicts
            and not state.open_threads
        ):
            return StopDecision(
                True, "grounded witness and closed threads", "ANSWER"
            )
        if not state.new_evidence_in_last_round and state.rounds > 1:
            return StopDecision(
                True,
                "search saturated without a gate-valid witness",
                "EVIDENCE_INSUFFICIENT",
            )
        return StopDecision(False, "witness obligation remains open", "CONTINUE")


class StratifiedCoveragePolicy:
    """Preserve the current stronger coverage behavior as a separate profile."""

    module_ref = f"{STRATIFIED_COVERAGE_PROFILE}@1.0.0"

    def plan(self, run_context: RuntimePolicyContext) -> SearchPlan:
        return SearchPlan(
            policy_ref=self.module_ref,
            strategy="STRATIFIED_COVERAGE",
            search_terms=run_context.positive_terms,
            document_types=run_context.document_types,
            required_strata=run_context.coverage_strata,
            proof_obligations=(
                "grounded_positive_witness",
                "required_strata_reviewed",
                "discovered_conflicts_closed",
            ),
        )

    def revise(
        self, state: RuntimePolicyState, new_evidence: Mapping[str, Any]
    ) -> SearchDecision:
        missing = tuple(
            row
            for row in state.required_strata
            if row not in set(state.completed_strata)
        )
        if state.open_conflicts or state.open_threads:
            return SearchDecision(
                "EXPAND_TARGETED",
                "close discovered conflicts before broad coverage expansion",
                added_terms=tuple(
                    str(row) for row in new_evidence.get("targeted_terms") or ()
                ),
                added_document_types=tuple(
                    str(row)
                    for row in new_evidence.get("targeted_document_types") or ()
                ),
            )
        if missing:
            return SearchDecision(
                "COMPLETE_STRATA",
                f"required coverage strata remain: {', '.join(missing)}",
                added_document_types=missing,
            )
        return SearchDecision(
            "CONFIRM",
            "coverage obligations are complete; run final confirmation",
        )

    def should_stop(self, state: RuntimePolicyState) -> StopDecision:
        missing = set(state.required_strata) - set(state.completed_strata)
        if (
            state.proof_valid
            and not missing
            and not state.open_conflicts
            and not state.open_threads
        ):
            return StopDecision(
                True,
                "proof, coverage, and conflict obligations are closed",
                "ANSWER",
            )
        if not state.new_evidence_in_last_round and state.rounds > 1:
            return StopDecision(
                True,
                "coverage search saturated with open obligations",
                "EVIDENCE_INSUFFICIENT",
            )
        return StopDecision(False, "coverage obligations remain open", "CONTINUE")


class GuidelineOnlyPolicy(WitnessFirstPolicy):
    """Clinical contract plus unrestricted patient-inventory search, without task priors."""

    module_ref = f"{GUIDELINE_ONLY_PROFILE}@1.0.0"

    def plan(self, run_context: RuntimePolicyContext) -> SearchPlan:
        return SearchPlan(
            policy_ref=self.module_ref,
            strategy="GUIDELINE_ONLY",
            search_terms=(),
            document_types=run_context.document_types,
            required_strata=(),
            proof_obligations=(
                "grounded_positive_witness",
                "discovered_conflicts_closed",
            ),
        )


class ConditionalNegativeCoveragePolicy(StratifiedCoveragePolicy):
    """Start without retrieval priors; activate the proof asset for negative-shaped claims."""

    module_ref = f"{CONDITIONAL_COVERAGE_PROFILE}@1.0.0"

    def plan(self, run_context: RuntimePolicyContext) -> SearchPlan:
        return SearchPlan(
            policy_ref=self.module_ref,
            strategy="CONDITIONAL_NEGATIVE_COVERAGE",
            search_terms=(),
            document_types=run_context.document_types,
            required_strata=run_context.coverage_strata,
            proof_obligations=(
                "grounded_positive_witness",
                "discovered_conflicts_closed",
                "activate_coverage_for_negative_missing_or_nos_claim",
            ),
        )


class AlwaysCoveragePolicy(StratifiedCoveragePolicy):
    """Experimental upper-bound arm: coverage is active before any answer may stand."""

    module_ref = f"{ALWAYS_COVERAGE_PROFILE}@1.0.0"

    def plan(self, run_context: RuntimePolicyContext) -> SearchPlan:
        return SearchPlan(
            policy_ref=self.module_ref,
            strategy="ALWAYS_COVERAGE",
            search_terms=run_context.positive_terms,
            document_types=run_context.document_types,
            required_strata=run_context.coverage_strata,
            proof_obligations=(
                "grounded_positive_witness",
                "required_strata_reviewed_before_any_answer",
                "discovered_conflicts_closed",
            ),
        )


class RuntimePolicyRegistry:
    def __init__(self) -> None:
        self.modules = ModuleRegistry()

    def register(self, asset: ModuleAsset, implementation: Any) -> None:
        if asset.module_kind != "RUNTIME_POLICY":
            raise ModuleContractError(
                f"{asset.ref}: expected a RUNTIME_POLICY module"
            )
        self.modules.register_asset(asset)
        self.modules.register_implementation(
            asset.implementation_id, implementation
        )

    def resolve(self, ref: str):
        asset = self.modules.resolve(ref)
        factory = self.modules.implementation(asset)
        return asset, factory()


def builtin_runtime_policy_registry() -> RuntimePolicyRegistry:
    registry = RuntimePolicyRegistry()
    rows = (
        (
            ModuleAsset(
                module_id=GUIDELINE_ONLY_PROFILE,
                version="1.0.0",
                module_kind="RUNTIME_POLICY",
                runner_type="CODE",
                input_channels=("run_context",),
                output_schema="acr.search_plan/1",
                implementation_id="runtime.guideline_only.v1",
                maximum_authority="OBSERVE",
                description=(
                    "Clinical task contract only: no task keywords, note-type priors, "
                    "or negative coverage proof."
                ),
            ),
            GuidelineOnlyPolicy,
        ),
        (
            ModuleAsset(
                module_id=CONDITIONAL_COVERAGE_PROFILE,
                version="1.0.0",
                module_kind="RUNTIME_POLICY",
                runner_type="CODE",
                input_channels=("run_context",),
                output_schema="acr.search_plan/1",
                implementation_id="runtime.conditional_negative_coverage.v1",
                maximum_authority="BLOCK_CURRENT_ANSWER",
                description=(
                    "Begin with guideline-only retrieval and activate field-missing/NOS "
                    "coverage only when the proposed answer makes a negative-shaped claim."
                ),
            ),
            ConditionalNegativeCoveragePolicy,
        ),
        (
            ModuleAsset(
                module_id=ALWAYS_COVERAGE_PROFILE,
                version="1.0.0",
                module_kind="RUNTIME_POLICY",
                runner_type="CODE",
                input_channels=("run_context",),
                output_schema="acr.search_plan/1",
                implementation_id="runtime.always_coverage.v1",
                maximum_authority="BLOCK_CURRENT_ANSWER",
                description=(
                    "Experimental strong arm requiring the configured coverage proof "
                    "before positive or negative answers."
                ),
            ),
            AlwaysCoveragePolicy,
        ),
        (
            ModuleAsset(
                module_id=WITNESS_FIRST_PROFILE,
                version="1.0.0",
                module_kind="RUNTIME_POLICY",
                runner_type="CODE",
                input_channels=("run_context",),
                output_schema="acr.search_plan/1",
                implementation_id="runtime.witness_first.v1",
                maximum_authority="OBSERVE",
                description=(
                    "Low-cost baseline that stops after a grounded witness and "
                    "closed discovered conflicts."
                ),
            ),
            WitnessFirstPolicy,
        ),
        (
            ModuleAsset(
                module_id=STRATIFIED_COVERAGE_PROFILE,
                version="1.0.0",
                module_kind="RUNTIME_POLICY",
                runner_type="CODE",
                input_channels=("run_context",),
                output_schema="acr.search_plan/1",
                implementation_id="runtime.stratified_coverage.v1",
                maximum_authority="OBSERVE",
                description=(
                    "Current coverage behavior represented as a versioned policy "
                    "for paired ablation."
                ),
            ),
            StratifiedCoveragePolicy,
        ),
    )
    for asset, implementation in rows:
        # Explicit class registry, not dynamic import. `resolve` constructs one policy per run.
        registry.register(asset, implementation)
    return registry


def resolve_runtime_policy(
    ref: str = DEFAULT_RUNTIME_PROFILE,
) -> tuple[
    ModuleAsset,
    WitnessFirstPolicy
    | StratifiedCoveragePolicy
    | GuidelineOnlyPolicy
    | ConditionalNegativeCoveragePolicy
    | AlwaysCoveragePolicy,
]:
    """Resolve an explicitly registered policy; arbitrary imports are never accepted."""
    return builtin_runtime_policy_registry().resolve(ref)


def enforces_stratified_coverage(ref: str) -> bool:
    """Whether a profile may issue a coverage attestation for a negative answer."""
    asset, _ = resolve_runtime_policy(ref)
    return asset.module_id in {
        STRATIFIED_COVERAGE_PROFILE,
        CONDITIONAL_COVERAGE_PROFILE,
        ALWAYS_COVERAGE_PROFILE,
    }


def coverage_requirement(ref: str) -> str:
    """Return the profile's answer-level coverage requirement."""
    asset, _ = resolve_runtime_policy(ref)
    if asset.module_id == ALWAYS_COVERAGE_PROFILE:
        return COVERAGE_ALWAYS
    if asset.module_id in {
        CONDITIONAL_COVERAGE_PROFILE,
        STRATIFIED_COVERAGE_PROFILE,
    }:
        return COVERAGE_ON_NEGATIVE_OR_MISSING
    return COVERAGE_NONE


def starts_with_coverage_assets(ref: str) -> bool:
    """Whether task keywords and note-type strata are active from the first model call."""
    asset, _ = resolve_runtime_policy(ref)
    return asset.module_id in {
        ALWAYS_COVERAGE_PROFILE,
        STRATIFIED_COVERAGE_PROFILE,
        WITNESS_FIRST_PROFILE,
    }


def uses_clinical_contract_view(ref: str) -> bool:
    """Whether the model-facing spec must hide retrieval assets."""
    asset, _ = resolve_runtime_policy(ref)
    return asset.module_id in {
        GUIDELINE_ONLY_PROFILE,
        CONDITIONAL_COVERAGE_PROFILE,
        ALWAYS_COVERAGE_PROFILE,
    }


def targeted_negative_basis(ref: str) -> str:
    """Name the non-coverage basis emitted by profiles that accept targeted abstention."""
    asset, _ = resolve_runtime_policy(ref)
    if asset.module_id == GUIDELINE_ONLY_PROFILE:
        return "GUIDELINE_ONLY_TARGETED"
    if asset.module_id == WITNESS_FIRST_PROFILE:
        return "WITNESS_FIRST_BASELINE"
    return "TARGETED_SEARCH_ONLY"


#: WHICH METHOD SKILLS EACH PROFILE OFFERS THE MODEL, BY SLOT. A skill is judgement guidance,
#: so swapping one is exactly the kind of change an arm has to isolate — which is why this is a
#: property of the profile and not a default buried in the prompt builder.
#:
#: EVERY PROFILE BELOW RENDERS EXACTLY WHAT IT RENDERED BEFORE SLOTS EXISTED: `coverage-judgement`
#: and nothing else. That is deliberate. Every run ever recorded was made under that one skill,
#: and quietly adding a second here would make past and future runs incomparable while the
#: manifest went on looking the same. New search policies reach a run through `--skills` or
#: through a NEW profile, both of which are recorded as the change they are.
#:
#: `coverage-judgement` is in `general` and not `search`: it supplies no keywords, no note-type
#: prior and no strata, and it activates only when the answer is about to claim something is
#: absent. It is not the retrieval asset the arms compare. What it replaces is the refusal
#: `evaluate_gate` used to issue — the arms differ in whether coverage is PROVEN, not in whether
#: the model is told how to think about an absence claim.
_PROFILE_SKILLS: dict[str, SkillStack] = {
    GUIDELINE_ONLY_PROFILE: SkillStack(general=("coverage-judgement",)),
    CONDITIONAL_COVERAGE_PROFILE: SkillStack(general=("coverage-judgement",)),
    ALWAYS_COVERAGE_PROFILE: SkillStack(general=("coverage-judgement",)),
    WITNESS_FIRST_PROFILE: SkillStack(general=("coverage-judgement",)),
    STRATIFIED_COVERAGE_PROFILE: SkillStack(general=("coverage-judgement",)),
}

_FALLBACK_SKILLS = SkillStack(general=("coverage-judgement",))


def runtime_policy_skills(module_id: str) -> SkillStack:
    """The method skills this profile renders into the system prompt, by slot."""
    return _PROFILE_SKILLS.get(module_id, _FALLBACK_SKILLS)


def runtime_policy_instruction(module_id: str) -> str:
    """Short execution instruction that makes the selected profile observable to the agent."""
    if module_id == GUIDELINE_ONLY_PROFILE:
        return (
            "RUNTIME SEARCH PROFILE: guideline-only. Use the clinical contract and the "
            "patient's document inventory to choose your own searches and reading order. "
            "No task-specific keyword list or note-type prior is supplied. A grounded "
            "positive still needs admissible evidence. A read that stopped short of the end "
            "of a document is the one thing the runtime will refuse an answer over, because "
            "it can count the characters; conflicts and deferred results you discover are "
            "yours to weigh, and if you leave one open, say why in your reasoning."
        )
    if module_id == CONDITIONAL_COVERAGE_PROFILE:
        return (
            "RUNTIME SEARCH PROFILE: conditional-negative-coverage. Begin exactly like the "
            "guideline-only arm, without task-specific retrieval priors. If you propose an "
            "EVIDENCE_INSUFFICIENT, partial, NOS, or otherwise negative-shaped answer, the "
            "runtime will activate an independent coverage proof and return the remaining "
            "obligations. Complete those obligations or route the case to review."
        )
    if module_id == ALWAYS_COVERAGE_PROFILE:
        return (
            "RUNTIME SEARCH PROFILE: always-coverage. This experimental arm activates the "
            "configured keyword, document-stratum, sampling, and open-thread obligations "
            "from the first turn. No positive or negative answer is accepted until both its "
            "ordinary evidence checks and the coverage proof pass."
        )
    if module_id == WITNESS_FIRST_PROFILE:
        return (
            "RUNTIME SEARCH PROFILE: witness-first-baseline. Seek the smallest admissible "
            "positive witness and close every conflict or deferred thread you discover. "
            "This arm does not require stratified exclusion sampling. If targeted search "
            "finds no admissible witness, list the chart and submit EVIDENCE_INSUFFICIENT; "
            "do not claim that the full chart universe was covered."
        )
    if module_id == STRATIFIED_COVERAGE_PROFILE:
        return (
            "RUNTIME SEARCH PROFILE: current-stratified-coverage. Follow the declared "
            "strata, searches, forced sampling, and open-thread obligations before making "
            "a negative coverage claim."
        )
    raise ModuleContractError(f"unknown runtime policy {module_id!r}")


@dataclass(frozen=True)
class CoverageComparison:
    baseline_profile_ref: str
    treatment_profile_ref: str
    patient_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    metrics: Mapping[str, Mapping[str, float]]
    paired_deltas: Mapping[str, float]
    practical_improvement: bool
    regression_reasons: tuple[str, ...] = ()

    REQUIRED_METRICS = frozenset({
        "chart_observable_accuracy",
        "critical_miss_rate",
        "overclaim_rate",
        "abstention_accuracy",
        "evidence_validity",
        "documents_read",
        "cost_usd",
    })

    def __post_init__(self) -> None:
        if not self.patient_ids or not self.seeds:
            raise ModuleContractError(
                "coverage comparison needs paired patients and seeds"
            )
        for arm in ("baseline", "treatment"):
            missing = self.REQUIRED_METRICS - set(self.metrics.get(arm, {}))
            if missing:
                raise ModuleContractError(
                    f"coverage {arm} arm lacks metrics {sorted(missing)}"
                )
