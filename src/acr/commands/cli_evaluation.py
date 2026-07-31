"""Run typed v2 evaluation pipelines over canonical local trajectories."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from ..core.cli_common import con
from ..core.kernel import ArtifactRef, TrajectoryAdapter, digest
from ..core.local_artifacts import (
    LOCAL_ROOT_ENV,
    LocalArtifactError,
    LocalArtifactStore,
    content_hash,
)
from ..core.modules import (
    CertificationRegistry,
    ModuleContractError,
    ModuleRegistry,
    PipelineRegistry,
)
from ..core.repo_paths import repo_root
from ..evaluation.evaluation_modules import builtin_evaluation_module_registry
from ..evaluation.evaluation_pipeline import (
    EvaluationContext,
    EvaluationPipelineError,
    EvaluationPipelineRunner,
    EvaluationStore,
    EvaluationTask,
    InputChannel,
    TruthContext,
)

evaluation_app = typer.Typer(
    add_completion=False,
    help="Typed local quality-evaluation pipelines over completed extraction trajectories.",
)

LOCAL_ROOT = typer.Option(None, "--local-root", envvar=LOCAL_ROOT_ENV)


def _store(root: str | None) -> LocalArtifactStore:
    try:
        return LocalArtifactStore(root)
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _registries() -> tuple[ModuleRegistry, PipelineRegistry, CertificationRegistry]:
    root = repo_root()
    display_modules = ModuleRegistry.from_directory(root / "assets" / "module_catalog")
    pipelines = PipelineRegistry.from_directory(root / "assets" / "pipeline_catalog")
    certifications = CertificationRegistry.from_directory(root / "assets" / "certification_catalog")
    profile = pipelines.resolve("chart-review-quality-v1")
    profile.validate_modules(display_modules)
    certifications.validate_modules(display_modules)
    return display_modules, pipelines, certifications


def _trace_path(store: LocalArtifactStore, manifest_path: Path, raw: dict) -> Path | None:
    declared = str(raw.get("trace") or "").strip()
    if declared:
        try:
            return store.require_input(declared, what="trajectory trace")
        except LocalArtifactError:
            # A moved local run directory may still have its trace beside the manifest.
            pass
    sibling = manifest_path.with_name(
        manifest_path.name.replace(".manifest.json", ".jsonl")
    )
    return sibling if sibling.is_file() else None


def _context(
    store: LocalArtifactStore,
    manifest: str | Path,
    *,
    subject_id: str,
    provider_boundary: str,
) -> EvaluationContext:
    manifest_path = store.require_input(manifest, what="evaluation manifest")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    trace_path = _trace_path(store, manifest_path, raw)
    trace = (
        tuple(
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if trace_path is not None
        else ()
    )
    artifacts = [
        ArtifactRef(
            str(manifest_path),
            content_hash(manifest_path),
            "RUN_MANIFEST",
        )
    ]
    if trace_path is not None:
        artifacts.append(
            ArtifactRef(str(trace_path), content_hash(trace_path), "RUN_TRACE")
        )
    case_ref = str(raw.get("patient_id") or subject_id)
    trajectory = TrajectoryAdapter.from_run_artifacts(
        manifest=raw,
        trace=trace,
        case_ref=case_ref,
        spec_id=str(raw.get("spec_id") or "unspecified-spec"),
        spec_hash=str(raw.get("spec_hash") or digest(raw.get("spec") or {})),
        artifact_refs=artifacts,
    )
    return EvaluationContext(
        trajectory=trajectory,
        spec_snapshot=InputChannel(
            "spec",
            "acr.extraction_spec/1",
            value={
                "spec_id": trajectory.task_ref.asset_id,
                "spec_hash": trajectory.task_ref.content_hash,
            },
        ),
        truth=TruthContext("BLIND"),
        patient_scope=case_ref,
        provider_boundary=provider_boundary,
    )


def _runner(
    store: LocalArtifactStore,
) -> tuple[EvaluationPipelineRunner, EvaluationStore]:
    _, pipelines, _ = _registries()
    result_store = EvaluationStore(store)
    return (
        EvaluationPipelineRunner(
            builtin_evaluation_module_registry(),
            pipelines,
            store=result_store,
        ),
        result_store,
    )


@evaluation_app.command("modules")
def modules():
    """Show only current module assets, pipelines, and certification suites."""
    module_registry, pipeline_registry, certification_registry = _registries()
    con.print_json(json.dumps({
        "architecture": "v2",
        "module_assets": [
            asset.to_dict() for asset in module_registry.all_assets()
        ],
        "pipelines": [
            profile.to_dict()
            for profile in pipeline_registry.all()
        ],
        "certification_suites": [
            suite.__dict__ for suite in certification_registry.all()
        ],
    }, ensure_ascii=False))


@evaluation_app.command("validate")
def validate():
    """Validate module contracts, pipeline DAGs, and certification bindings."""
    modules_registry, pipelines, certifications = _registries()
    for profile in pipelines.all():
        profile.validate_modules(modules_registry)
    certifications.validate_modules(modules_registry)
    con.print(
        f"[green]{len(modules_registry.all_assets())} modules, "
        f"{len(pipelines.all())} pipeline(s), and "
        f"{len(certifications.all())} certification suite(s) validated[/]"
    )


@evaluation_app.command("run")
def run(
    manifest: str = typer.Option(..., "--manifest"),
    subject_id: str = typer.Option(..., "--subject-id"),
    pipeline: str = typer.Option("chart-review-quality-v1", "--pipeline"),
    provider_boundary: str = typer.Option("UNKNOWN", "--provider-boundary"),
    local_root: str | None = LOCAL_ROOT,
):
    """Run one current CODE evaluation pipeline on one local manifest."""
    store = _store(local_root)
    try:
        context = _context(
            store,
            manifest,
            subject_id=subject_id,
            provider_boundary=provider_boundary,
        )
        runner, _ = _runner(store)
        task = EvaluationTask(
            task_id=f"evaluation-{context.trajectory.trajectory_id}",
            pipeline_ref=pipeline,
            trajectory_ids=(context.trajectory.trajectory_id,),
            truth_mode="BLIND",
            local_output_root=str(store.root),
        )
        result_map = runner.run(task, (context,))
    except (
        EvaluationPipelineError,
        LocalArtifactError,
        ModuleContractError,
        OSError,
        ValueError,
    ) as exc:
        raise typer.BadParameter(str(exc)) from exc
    table = Table("evaluator", "status", "score", "reason")
    for result in result_map[context.trajectory.trajectory_id]:
        table.add_row(
            result.module_ref,
            result.status,
            "—" if result.score is None else str(result.score),
            result.reason[:100],
        )
    con.print(table)
    con.print(f"→ {store.root / 'evaluation' / 'results.jsonl'}")


@evaluation_app.command("batch")
def batch(
    runs: str = typer.Option(..., "--runs"),
    pipeline: str = typer.Option("chart-review-quality-v1", "--pipeline"),
    provider_boundary: str = typer.Option("UNKNOWN", "--provider-boundary"),
    local_root: str | None = LOCAL_ROOT,
):
    """Run one current CODE pipeline over every local extraction manifest."""
    store = _store(local_root)
    try:
        root = store.path(runs, what="evaluation runs")
        manifests = [root] if root.is_file() else sorted(root.rglob("*.manifest.json"))
        if not manifests:
            raise typer.BadParameter(f"no manifests under {root}")
        contexts = tuple(
            _context(
                store,
                manifest,
                subject_id=f"CASE-{content_hash(manifest)[:12]}",
                provider_boundary=provider_boundary,
            )
            for manifest in manifests
        )
        runner, result_store = _runner(store)
        task = EvaluationTask(
            task_id=f"evaluation-batch-{digest([str(path) for path in manifests])[:16]}",
            pipeline_ref=pipeline,
            trajectory_ids=tuple(
                context.trajectory.trajectory_id for context in contexts
            ),
            truth_mode="BLIND",
            local_output_root=str(store.root),
        )
        runner.run(task, contexts)
    except (
        EvaluationPipelineError,
        LocalArtifactError,
        ModuleContractError,
        OSError,
        ValueError,
    ) as exc:
        raise typer.BadParameter(str(exc)) from exc
    con.print_json(json.dumps(result_store.summary(), ensure_ascii=False))


@evaluation_app.command("summarize")
def summarize(local_root: str | None = LOCAL_ROOT):
    """Summarize the current append-only evaluation-result ledger."""
    con.print_json(json.dumps(
        EvaluationStore(_store(local_root)).summary(),
        ensure_ascii=False,
    ))


@evaluation_app.command("compare")
def compare(
    before: str = typer.Option(..., "--before"),
    after: str = typer.Option(..., "--after"),
):
    """Compare two v2 evaluation summaries without mixing Audit signals."""
    left = json.loads(Path(before).read_text(encoding="utf-8"))
    right = json.loads(Path(after).read_text(encoding="utf-8"))
    fields = ("n_results", "n_trajectories")
    con.print_json(json.dumps({
        "before": {field: left.get(field) for field in fields},
        "after": {field: right.get(field) for field in fields},
        "delta": {
            field: int(right.get(field) or 0) - int(left.get(field) or 0)
            for field in fields
        },
    }, ensure_ascii=False))
