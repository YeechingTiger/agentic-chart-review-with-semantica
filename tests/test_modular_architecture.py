from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from acr.audit.audit_loop import (
    AuditContext,
    AuditRunner,
    builtin_audit_registry,
)
from acr.contract.spec import load_spec
from acr.core.kernel import (
    KernelContractError,
    TargetRef,
    TrajectoryAdapter,
    digest,
)
from acr.core.modules import (
    CapabilityRequest,
    CertificationRegistry,
    CertificationSuite,
    ModuleAsset,
    ModuleContractError,
    ModuleRegistry,
    PipelineNode,
    PipelineProfile,
    PipelineRegistry,
    TaskBudget,
    effective_capabilities,
)
from acr.evaluation.evaluation_modules import builtin_evaluation_module_registry
from acr.evaluation.evaluation_pipeline import (
    CapabilityBroker,
    EvaluationContext,
    EvaluationInvocation,
    EvaluationPipelineError,
    EvaluationPipelineRunner,
    EvaluationResult,
    EvaluationTask,
    InputChannel,
    TruthContext,
    make_result,
)
from acr.review.answer_gate import gate_answer
from acr.review.runtime_profiles import (
    ALWAYS_COVERAGE_PROFILE,
    CONDITIONAL_COVERAGE_PROFILE,
    DEFAULT_RUNTIME_PROFILE,
    GUIDELINE_ONLY_PROFILE,
    RuntimePolicyContext,
    RuntimePolicyState,
    StratifiedCoveragePolicy,
    WitnessFirstPolicy,
    coverage_requirement,
    resolve_runtime_policy,
    starts_with_coverage_assets,
    uses_clinical_contract_view,
)

ROOT = Path(__file__).resolve().parents[1]


def _trajectory(*, output=None, trace=(), manifest=None):
    raw_manifest = {
        "answer": output or {
            "status": "FOUND",
            "value": {"histology": "8140"},
            "evidence": [{"note_id": "N1"}],
        },
        "evidence_ledger": {
            "proof_valid": True,
            "evidence": [{"note_id": "N1"}],
        },
        **dict(manifest or {}),
    }
    return TrajectoryAdapter.from_run_artifacts(
        manifest=raw_manifest,
        trace=trace,
        case_ref="CASE-001",
        spec_id="STORE.site-histology",
        spec_hash=digest({"spec": 1}),
    )


def _context(trajectory=None, truth=None, channels=()):
    return EvaluationContext(
        trajectory=trajectory or _trajectory(),
        spec_snapshot=InputChannel(
            "spec", "acr.extraction_spec/1", value={"spec_id": "STORE.x"}
        ),
        channels=tuple(channels),
        truth=truth or TruthContext("BLIND"),
        patient_scope="CASE-001",
    )


def test_trajectory_adapter_removes_raw_chart_text_and_keeps_hash():
    source = "patient has invasive adenocarcinoma"
    trajectory = _trajectory(trace=({
        "seq": 1,
        "kind": "tool",
        "tool": "read_document",
        "result": {"note_id": "N1", "text": source},
    },))
    rendered = str(trajectory.to_dict())
    assert source not in rendered
    assert hashlib.sha256(source.encode()).hexdigest() in rendered
    assert trajectory.task_ref.asset_type == "SPEC"
    assert trajectory.runtime_profile_ref.asset_type == "RUNTIME_PROFILE"


def test_runtime_policy_registry_resolves_only_registered_profiles():
    asset, policy = resolve_runtime_policy(DEFAULT_RUNTIME_PROFILE)
    assert asset.module_id == "current-stratified-coverage"
    assert isinstance(policy, StratifiedCoveragePolicy)
    with pytest.raises(ModuleContractError, match="unknown module"):
        resolve_runtime_policy("some.module:factory")


def test_trajectory_adapter_normalizes_manifest_list_ledgers():
    trajectory = TrajectoryAdapter.from_run_artifacts(
        manifest={
            "run_id": "RUN-1",
            "runtime_profile_id": "witness-first-baseline",
            "runtime_profile_version": "1.0.0",
            "answer": {
                "status": "EVIDENCE_INSUFFICIENT",
                "coverage_attested": {"n_read": 3},
            },
            "evidence": [{"note_id": "N1"}],
        },
        trace=(),
        case_ref="CASE-001",
        spec_id="STORE.x",
        spec_hash=digest({"spec": 1}),
    )
    assert trajectory.evidence_state == {
        "evidence": [{"note_id": "N1"}],
        "proof_valid": False,
    }
    assert trajectory.coverage_state == {"n_read": 3}
    assert trajectory.runtime_profile_ref.ref == "witness-first-baseline@1.0.0"


def test_trajectory_adapter_preserves_manifest_gate_proof_with_list_evidence():
    trajectory = TrajectoryAdapter.from_run_artifacts(
        manifest={
            "run_id": "RUN-1",
            "gate_validated": True,
            "answer": {
                "status": "FOUND",
                "proof_basis": "WITNESS",
            },
            "evidence": [{"note_id": "N1"}],
        },
        trace=(),
        case_ref="CASE-001",
        spec_id="STORE.x",
        spec_hash=digest({"spec": 1}),
    )
    assert trajectory.evidence_state["proof_valid"] is True


def test_trajectory_adapter_orders_uniquely_sequenced_application_events():
    trajectory = _trajectory(trace=(
        {"seq": 2, "kind": "tool"},
        {"seq": 0, "kind": "run_start"},
        {"seq": 1, "kind": "retrieval_plan"},
    ))
    assert [event["seq"] for event in trajectory.events] == [0, 1, 2]


def test_trajectory_adapter_refuses_duplicate_sequence_identities():
    with pytest.raises(KernelContractError, match="duplicate"):
        _trajectory(trace=(
            {"seq": 0, "kind": "run_start"},
            {"seq": 0, "kind": "tool"},
        ))


def test_trajectory_identity_distinguishes_profiles_and_reruns():
    current = TrajectoryAdapter.from_run_artifacts(
        manifest={
            "run_id": "CASE-001__STORE.x",
            "runtime_profile_id": "current-stratified-coverage",
            "answer": {"status": "FOUND"},
        },
        trace=({"seq": 0, "kind": "run_start", "ts": "t1"},),
        case_ref="CASE-001",
        spec_id="STORE.x",
        spec_hash=digest({"spec": 1}),
    )
    witness = TrajectoryAdapter.from_run_artifacts(
        manifest={
            "run_id": "CASE-001__STORE.x",
            "runtime_profile_id": "witness-first-baseline",
            "answer": {"status": "FOUND"},
        },
        trace=({"seq": 0, "kind": "run_start", "ts": "t2"},),
        case_ref="CASE-001",
        spec_id="STORE.x",
        spec_hash=digest({"spec": 1}),
    )
    assert current.trajectory_id != witness.trajectory_id
    assert current.trajectory_id.startswith("CASE-001__STORE.x--")


def test_trajectory_adapter_is_content_stable_across_repeated_ingestion():
    manifest = {
        "run_id": "CASE-001__STORE.x",
        "runtime_profile_id": "guideline-only",
        "runtime_profile_version": "1.0.0",
        "answer": {"status": "EVIDENCE_INSUFFICIENT"},
        "coverage_state": {"listed_documents": True, "n_read": 4},
        "spend": {"usd": 0.25, "prompt_tokens": 100},
    }
    trace = ({"seq": 0, "kind": "run_start", "ts": "2026-01-02T03:04:05Z"},)
    kwargs = {
        "manifest": manifest,
        "trace": trace,
        "case_ref": "CASE-001",
        "spec_id": "STORE.x",
        "spec_hash": digest({"spec": 1}),
    }
    first = TrajectoryAdapter.from_run_artifacts(**kwargs)
    second = TrajectoryAdapter.from_run_artifacts(**kwargs)
    assert first.trajectory_id == second.trajectory_id
    assert first.created_at == second.created_at == "2026-01-02T03:04:05Z"
    assert first.content_hash == second.content_hash
    assert first.coverage_state == {"listed_documents": True, "n_read": 4}
    assert first.cost == {"usd": 0.25, "prompt_tokens": 100}


def test_witness_first_negative_is_accepted_without_coverage_attestation():
    verdict = gate_answer(
        SimpleNamespace(),
        {"status": "EVIDENCE_INSUFFICIENT"},
        evidence=SimpleNamespace(items=[]),
        coverage=SimpleNamespace(listed_documents=True),
        chart=None,
        runtime_profile="witness-first-baseline",
    )
    assert verdict["accepted"] is True
    assert verdict["coverage_claim_earned"] is False
    assert verdict["negative_basis"] == "WITNESS_FIRST_BASELINE"


def test_witness_first_still_requires_chart_universe_to_be_listed():
    verdict = gate_answer(
        SimpleNamespace(),
        {"status": "EVIDENCE_INSUFFICIENT"},
        evidence=SimpleNamespace(items=[]),
        coverage=SimpleNamespace(listed_documents=False),
        chart=None,
        runtime_profile="witness-first-baseline",
    )
    assert verdict["accepted"] is False
    assert "list the patient's documents" in verdict["missing"][0]


def test_three_arm_profiles_have_distinct_coverage_activation_contracts():
    assert coverage_requirement(GUIDELINE_ONLY_PROFILE) == "NONE"
    assert coverage_requirement(CONDITIONAL_COVERAGE_PROFILE) == (
        "ON_NEGATIVE_OR_MISSING"
    )
    assert coverage_requirement(ALWAYS_COVERAGE_PROFILE) == "ALWAYS"
    assert starts_with_coverage_assets(GUIDELINE_ONLY_PROFILE) is False
    assert starts_with_coverage_assets(CONDITIONAL_COVERAGE_PROFILE) is False
    assert starts_with_coverage_assets(ALWAYS_COVERAGE_PROFILE) is True
    assert uses_clinical_contract_view(GUIDELINE_ONLY_PROFILE) is True
    assert uses_clinical_contract_view(CONDITIONAL_COVERAGE_PROFILE) is True
    assert uses_clinical_contract_view(ALWAYS_COVERAGE_PROFILE) is True


def test_guideline_only_accepts_targeted_abstention_without_coverage_asset():
    state = {}
    verdict = gate_answer(
        SimpleNamespace(),
        {"status": "EVIDENCE_INSUFFICIENT"},
        evidence=SimpleNamespace(items=[]),
        coverage=SimpleNamespace(listed_documents=True),
        chart=None,
        coverage_state=state,
        runtime_profile=GUIDELINE_ONLY_PROFILE,
    )
    assert verdict["accepted"] is True
    assert verdict["coverage_claim_earned"] is False
    assert verdict["negative_basis"] == "GUIDELINE_ONLY_TARGETED"
    assert state == {}, "the baseline must never activate a hidden coverage asset"


def test_partial_abstention_may_carry_a_malformed_populated_field_and_it_is_recorded():
    """    RECORDED, NOT REFUSED, as of 2026-07-30. `field_format` was the last check in `gate_answer`
    that judged an answer's CONTENT, and the constraint is already in the prompt:
    `as_prompt_block` renders every field's `format` and `allowable_values`, and STORE.400's own
    field description reads "no decimal point". A model that writes a malformed code against that
    has failed to follow an instruction rather than been under-informed, and 4 of that check's 6
    useful firings rejected `C34.9`/`C34.11`/`C34.2` -- the punctuated form ICD-O-3 itself writes
    -- so it was largely creating the round trips it then resolved.

    The measurement survives: `answer_shape_miss` carries which declared shape was missed, with
    `refused: False`, so the eval plane counts instruction-following failures instead of a gate
    absorbing them silently.
    """
    spec = load_spec(
        ROOT / "assets" / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"
    )
    verdict = gate_answer(
        spec,
        {
            "status": "EVIDENCE_INSUFFICIENT",
            "value": {
                "primary_site": "C3411",
                "histology": None,
                "behavior": None,
            },
        },
        evidence=SimpleNamespace(items=[]),
        coverage=SimpleNamespace(listed_documents=True),
        chart=None,
        runtime_profile=GUIDELINE_ONLY_PROFILE,
    )
    assert verdict["accepted"] is True
    assert "format rules" not in verdict["why"]


def test_conditional_profile_does_not_activate_for_complete_non_nos_positive(
    monkeypatch,
):
    spec = SimpleNamespace(
        fields=[SimpleNamespace(name="site"), SimpleNamespace(name="histology")],
        answer_checks=[
            {"field": "site", "kind": "not_less_specific", "nos_values": ["NOS"]}
        ],
    )
    evidence = SimpleNamespace(
        items=[object()],
        to_list=lambda: [
            {"field": "site", "note_id": "N1", "quote": "upper lobe"}
        ],
    )
    state = {}

    def forbidden(*args, **kwargs):
        raise AssertionError("coverage was activated for a wholly positive answer")

    monkeypatch.setattr("acr.review.answer_gate._coverage_verdict", forbidden)
    verdict = gate_answer(
        spec,
        {
            "status": "FOUND",
            "value": {"site": "C341", "histology": "8140"},
        },
        evidence=evidence,
        coverage=SimpleNamespace(searched_terms=[]),
        chart=None,
        coverage_state=state,
        runtime_profile=CONDITIONAL_COVERAGE_PROFILE,
    )
    assert verdict["accepted"] is True
    assert state == {}


@pytest.mark.parametrize(
    "submission, expected_reason",
    [
        (
            {"status": "EVIDENCE_INSUFFICIENT", "value": {}},
            "case_status:EVIDENCE_INSUFFICIENT",
        ),
        (
            {"status": "FOUND", "value": {"site": "C341", "histology": None}},
            "missing_field:histology",
        ),
        # REMOVED 2026-07-30: the `nos_or_unknown:site:NOS` case.
        #
        # `negative_claim_reasons` used to read a NOS-shaped VALUE as a claim that something was
        # not established, and activate coverage against it. A NOS code is a conclusion, not a
        # confession: 8000/8010/8046 are the registry's own answer for 10.8% of this corpus and
        # C349 for 9.6%, and `conflict_requires_nos` used to ORDER the agent toward exactly those
        # codes. One run submitted the conflict-resolving gold answer ten times and was rejected
        # into a model-call-limit failure. A missing field is still a negative claim -- that is a
        # fact about the submission, not an inference from a code table -- and that case remains
        # above.
        (
            {"status": "FOUND", "value": {"site": None, "histology": "8140"}},
            "missing_field:site",
        ),
    ],
)
def test_conditional_profile_activates_only_for_negative_shaped_claims(
    monkeypatch, submission, expected_reason
):
    spec = SimpleNamespace(
        fields=[SimpleNamespace(name="site"), SimpleNamespace(name="histology")],
        answer_checks=[
            {"field": "site", "kind": "not_less_specific", "nos_values": ["NOS"]}
        ],
    )
    evidence = SimpleNamespace(
        items=[] if submission["status"] != "FOUND" else [object()],
        to_list=list,
    )
    seen = {}

    def capture(*args, activation_reasons, **kwargs):
        seen["reasons"] = activation_reasons
        return {
            "accepted": False,
            "coverage_activated": True,
            "coverage_activation_reasons": activation_reasons,
            "missing": ["synthetic coverage obligation"],
        }

    monkeypatch.setattr("acr.review.answer_gate._coverage_verdict", capture)
    state = {}
    verdict = gate_answer(
        spec,
        submission,
        evidence=evidence,
        coverage=SimpleNamespace(searched_terms=[]),
        chart=None,
        coverage_state=state,
        runtime_profile=CONDITIONAL_COVERAGE_PROFILE,
    )
    assert verdict["accepted"] is False
    assert expected_reason in seen["reasons"]
    assert state["active"] is True
    assert expected_reason in state["reason"]


def test_always_coverage_activates_even_for_complete_positive(monkeypatch):
    spec = SimpleNamespace(
        fields=[SimpleNamespace(name="site")],
        answer_checks=[],
    )
    evidence = SimpleNamespace(
        items=[object()],
        to_list=lambda: [
            {"field": "site", "note_id": "N1", "quote": "upper lobe"}
        ],
    )
    seen = {}

    def capture(*args, activation_reasons, **kwargs):
        seen["reasons"] = activation_reasons
        return {
            "accepted": True,
            "coverage_activated": True,
            "coverage_claim_earned": True,
            "missing": [],
        }

    monkeypatch.setattr("acr.review.answer_gate._coverage_verdict", capture)
    state = {}
    verdict = gate_answer(
        spec,
        {"status": "FOUND", "value": {"site": "C341"}},
        evidence=evidence,
        coverage=SimpleNamespace(searched_terms=[]),
        chart=None,
        coverage_state=state,
        runtime_profile=ALWAYS_COVERAGE_PROFILE,
    )
    assert verdict["accepted"] is True
    assert verdict["coverage_claim_earned"] is True
    assert seen["reasons"] == [f"profile:{ALWAYS_COVERAGE_PROFILE}"]
    assert state["active"] is True


def test_clinical_contract_prompt_hides_retrieval_experience():
    spec = load_spec(
        ROOT / "assets" / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"
    )
    full = spec.as_prompt_block()
    clinical = spec.as_prompt_block(view="clinical_contract")
    assert "SEARCH HINTS" in full
    assert "SEARCH HINTS" not in clinical
    for keyword in spec.proof_obligation.required_keywords:
        assert keyword in full
        assert keyword not in clinical
    assert spec.question in clinical


def test_patient_inventory_plan_contains_no_task_retrieval_prior():
    from acr.review.coverage_planner import plan_from_patient_inventory

    chart = SimpleNamespace(
        list_documents=lambda limit: (
            [
                SimpleNamespace(doc_type="Local-Type-B"),
                SimpleNamespace(doc_type="Local-Type-A"),
                SimpleNamespace(doc_type="Local-Type-B"),
            ],
            None,
        )
    )
    plan = plan_from_patient_inventory(
        SimpleNamespace(fields=[SimpleNamespace(name="field")]), chart
    )
    assert plan.source == "patient_inventory_only"
    assert plan.keywords == []
    assert plan.read_all == []
    assert plan.sample == []
    assert plan.search == ["Local-Type-A", "Local-Type-B"]


def test_asset_pipeline_and_certification_are_independent_and_reusable():
    registry = builtin_evaluation_module_registry()
    first = PipelineProfile(
        "first", "1.0.0", (
            PipelineNode(
                "gate",
                "gate-effectiveness@1.0.0",
                authority="SCREEN_ONLY",
            ),
        ),
    )
    second = PipelineProfile(
        "second", "1.0.0", (
            PipelineNode(
                "gate-again",
                "gate-effectiveness@1.0.0",
                authority="OBSERVE",
            ),
        ),
    )
    first.validate_modules(registry)
    second.validate_modules(registry)
    suite = CertificationSuite(
        "gate-cert", "1.0.0", "gate-effectiveness@1.0.0",
        ("PASS-1",), ("FAIL-1",),
    )
    assert suite.module_ref == first.nodes[0].module_ref == second.nodes[0].module_ref


def test_catalogs_are_physically_separated_and_loadable():
    modules = ModuleRegistry.from_directory(ROOT / "assets" / "module_catalog")
    pipelines = PipelineRegistry.from_directory(ROOT / "assets" / "pipeline_catalog")
    suites = CertificationRegistry.from_directory(ROOT / "assets" / "certification_catalog")
    assert modules.resolve("phi-provider-audit").module_kind == "AUDIT_RULE"
    assert modules.resolve("gate-effectiveness").module_kind == "EVALUATOR"
    assert modules.resolve("causal-attribution").runner_type == "AGENT"
    profile = pipelines.resolve("chart-review-quality-v1")
    profile.validate_modules(modules)
    assert suites.for_module("gate-effectiveness@1.0.0")
    assert suites.for_module("causal-attribution@2.0.0")


def test_module_catalog_cannot_dynamic_import_unregistered_code():
    registry = ModuleRegistry.from_directory(ROOT / "assets" / "module_catalog")
    with pytest.raises(ModuleContractError, match="not explicitly registered"):
        registry.implementation(registry.resolve("gate-effectiveness"))


def test_pipeline_task_cannot_expand_budget_or_authority():
    modules = builtin_evaluation_module_registry()
    profiles = PipelineRegistry()
    profiles.register(PipelineProfile(
        "quality", "1.0.0", (
            PipelineNode(
                "gate", "gate-effectiveness@1.0.0",
                budget=TaskBudget(0, 0, 0),
                authority="SCREEN_ONLY",
            ),
        ),
    ))
    task = EvaluationTask(
        "TASK-1",
        "quality",
        (_trajectory().trajectory_id,),
        "BLIND",
        budgets={"gate": TaskBudget(1, 0, 0)},
        authority_grants={"gate": "BLOCK_RELEASE"},
    )
    with pytest.raises(EvaluationPipelineError, match="budget expands"):
        EvaluationPipelineRunner(modules, profiles).run(task, [_context()])

    authority_only = EvaluationTask(
        "TASK-1B",
        "quality",
        (_trajectory().trajectory_id,),
        "BLIND",
        authority_grants={"gate": "BLOCK_RELEASE"},
    )
    with pytest.raises(EvaluationPipelineError, match="authority"):
        EvaluationPipelineRunner(modules, profiles).run(
            authority_only, [_context()]
        )


def test_task_capability_grants_fail_closed_instead_of_silent_intersection():
    asset = ModuleAsset(
        "agent-probe", "1.0.0", "EVALUATOR",
        ("trajectory",), "synthetic/1", "synthetic.agent_probe",
        runner_type="AGENT",
        requested_capabilities=(
            CapabilityRequest("patient-chart-reader"),
        ),
        maximum_authority="SCREEN_ONLY",
    )
    node = PipelineNode(
        "probe", asset.ref,
        allowed_capabilities=("patient-chart-reader",),
        budget=TaskBudget(2, 2, 1),
        authority="SCREEN_ONLY",
    )
    with pytest.raises(ModuleContractError, match="expand capabilities"):
        effective_capabilities(
            asset,
            node,
            {
                "patient-chart-reader": "patient_under_review",
                "gate-replay": "patient_under_review",
            },
        )


def test_evaluation_pipeline_dispatches_registered_implementation_without_id_branch():
    modules = builtin_evaluation_module_registry()
    profiles = PipelineRegistry()
    profiles.register(PipelineProfile(
        "quality", "1.0.0", (
            PipelineNode(
                "evidence",
                "evidence-validity@1.0.0",
                authority="BLOCK_RELEASE",
            ),
            PipelineNode(
                "gate",
                "gate-effectiveness@1.0.0",
                after=("evidence",),
                authority="BLOCK_RELEASE",
            ),
        ),
    ))
    trajectory = _trajectory()
    results = EvaluationPipelineRunner(modules, profiles).run(
        EvaluationTask(
            "TASK-2", "quality", (trajectory.trajectory_id,), "BLIND"
        ),
        [_context(trajectory)],
    )[trajectory.trajectory_id]
    assert [row.module_ref for row in results] == [
        "evidence-validity@1.0.0",
        "gate-effectiveness@1.0.0",
    ]
    assert {row.status for row in results} == {"PASS"}


def test_truth_is_a_separate_context_and_blind_channels_reject_leakage():
    with pytest.raises(EvaluationPipelineError, match="TruthContext"):
        InputChannel("gold", "synthetic/1", value={"site": "C341"})
    with pytest.raises(EvaluationPipelineError, match="truth-like"):
        InputChannel(
            "extras", "synthetic/1",
            value={"nested": {"answer_key": "C341"}},
        )
    with pytest.raises(EvaluationPipelineError, match="BLIND"):
        TruthContext("BLIND", chart_gold={"site": "C341"})
    gold = TruthContext(
        "GOLD",
        chart_gold={"site": "C341"},
        adjudication_refs=("HUMAN-1",),
    )
    assert _context(truth=gold).truth_mode == "GOLD"


def test_evaluation_result_cannot_carry_audit_incident():
    with pytest.raises(EvaluationPipelineError, match="Audit"):
        EvaluationResult(
            "ER-1",
            "synthetic@1",
            "T-1",
            TargetRef("RUN", "T-1"),
            "FAIL",
            "SCREEN_ONLY",
            "synthetic/1",
            payload={"incidents": [{"kind": "PHI"}]},
        )


def test_application_audit_detects_cross_patient_and_external_phi_without_truth():
    secret = "1168" + "7" * 12
    events = (
        {
            "seq": 1,
            "kind": "tool",
            "tool": "read_document",
            "args": {"subject_id": "CASE-002"},
            "result": {"text": f"patient={secret}"},
        },
    )
    trajectory = _trajectory(trace=events)
    context = AuditContext(
        trajectory=trajectory,
        application_events=events,
        patient_scope="CASE-001",
        provider_boundary="EXTERNAL",
    )
    assert not hasattr(context, "truth")
    report = AuditRunner(builtin_audit_registry()).run(
        context,
        ("patient-boundary-audit", "phi-provider-audit"),
    )
    assert {row.kind for row in report.incidents} == {
        "PATIENT_CROSSOVER",
        "PHI_EXTERNAL_MODEL_BOUNDARY",
    }
    assert secret not in str(report.to_dict())


def test_clinical_quality_failure_is_not_an_audit_incident():
    trajectory = _trajectory(output={
        "status": "FOUND",
        "value": {"primary_site": "WRONG"},
        "evidence": [{"note_id": "N1"}],
    })
    audit = AuditRunner(builtin_audit_registry()).run(AuditContext(
        trajectory=trajectory,
        application_events=(),
        patient_scope="CASE-001",
        provider_boundary="LOCAL",
    ))
    assert audit.incidents == ()

    def site_correctness(invocation, capabilities):
        del capabilities
        expected = invocation.context.truth.chart_gold["primary_site"]
        observed = invocation.context.trajectory.output["value"]["primary_site"]
        return make_result(
            invocation,
            status="PASS" if observed == expected else "FAIL",
            target_ref=TargetRef(
                "FIELD", invocation.context.trajectory.trajectory_id,
                field="primary_site",
            ),
            reason="field comparison",
        )

    modules = ModuleRegistry()
    asset = ModuleAsset(
        "site-correctness", "1.0.0", "EVALUATOR",
        ("trajectory",), "acr.site_correctness/1",
        "evaluation.site_correctness.synthetic",
        runner_type="CODE",
        supported_truth_modes=("GOLD",),
        maximum_authority="BLOCK_RELEASE",
    )
    modules.register_asset(asset)
    modules.register_implementation(asset.implementation_id, site_correctness)
    pipelines = PipelineRegistry()
    pipelines.register(PipelineProfile(
        "gold-quality", "1.0.0", (
            PipelineNode(
                "site", asset.ref, authority="BLOCK_RELEASE"
            ),
        ),
    ))
    result = EvaluationPipelineRunner(modules, pipelines).run(
        EvaluationTask(
            "TASK-GOLD", "gold-quality",
            (trajectory.trajectory_id,), "GOLD",
        ),
        [_context(
            trajectory,
            TruthContext(
                "GOLD",
                chart_gold={"primary_site": "C341"},
                adjudication_refs=("HUMAN-1",),
            ),
        )],
    )[trajectory.trajectory_id][0]
    assert result.status == "FAIL"


def test_capability_broker_enforces_effective_patient_scope():
    asset = ModuleAsset(
        "probe", "1.0.0", "EVALUATOR",
        ("trajectory",), "synthetic/1", "synthetic.probe",
        runner_type="AGENT",
        requested_capabilities=(),
        maximum_authority="SCREEN_ONLY",
    )
    node = PipelineNode("probe", asset.ref, authority="SCREEN_ONLY")
    invocation = EvaluationInvocation(
        asset=asset,
        node=node,
        context=_context(),
        inputs={},
        capabilities=(),
        budget=TaskBudget(1, 1, 0),
        authority="SCREEN_ONLY",
    )
    broker = CapabilityBroker(invocation, {})
    with pytest.raises(EvaluationPipelineError, match="not effectively granted"):
        broker.call("patient-chart-reader", subject_id="CASE-002")


def test_audit_catches_a_tool_call_that_leaves_the_patient_under_review():
    """一次运行只应触及一个病人。这条测试钉住审计确实会发现越界,并且把发现变成一个可路由的信号。

    2026-08-03 之前它还断言 `RepairSignalRouter` 把这个信号路由成 `SECURITY_CONTROL` 且不改
    语义。那个路由器在生产代码里零引用,已删除;审计这一半是活的,所以断言退到信号本身 ——
    越界被发现了,而且带着产出它的资产引用。
    """
    events = ({
        "seq": 1,
        "kind": "tool",
        "args": {"subject_id": "CASE-002"},
    },)
    trajectory = _trajectory(trace=events)
    registry = builtin_audit_registry()
    report = AuditRunner(registry).run(
        AuditContext(
            trajectory, events, "CASE-001", provider_boundary="LOCAL"
        ),
        ("patient-boundary-audit",),
    )
    incident = report.incidents[0]
    asset = registry.resolve("patient-boundary-audit")
    signal = incident.to_signal(asset.asset_ref)
    assert signal.signal_type == "AUDIT_INCIDENT"
    assert signal.producer_ref == asset.asset_ref


def test_coverage_policies_have_distinct_stopping_obligations():
    context = RuntimePolicyContext(
        "CASE-001",
        {"field": "histology"},
        positive_terms=("adenocarcinoma",),
        document_types=("pathology",),
        coverage_strata=("pathology", "surgery"),
    )
    witness_plan = WitnessFirstPolicy().plan(context)
    coverage_plan = StratifiedCoveragePolicy().plan(context)
    assert witness_plan.required_strata == ()
    assert coverage_plan.required_strata == ("pathology", "surgery")
    state = RuntimePolicyState(
        witness_found=True,
        proof_valid=True,
        required_strata=("pathology", "surgery"),
        completed_strata=("pathology",),
    )
    assert WitnessFirstPolicy().should_stop(state).stop
    assert not StratifiedCoveragePolicy().should_stop(state).stop
