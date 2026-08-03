"""Truth-blind, application-level audit commands."""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from ..audit.audit_loop import (
    AuditContext,
    AuditRunner,
    AuditStore,
    builtin_audit_registry,
)
from ..core.cli_common import con
from ..core.kernel import ArtifactRef, TrajectoryAdapter, digest

#: A RUN RECORD is read with `require_run_artifact`, not through the store: `runs/` is inside
#: the worktree by design and the store's `require_input` proves the opposite, so every one of
#: these call sites was unreachable. Develop artifacts — gold, answer keys, case maps — keep
#: going through the store, where the outside-the-worktree rule is the right rule.
from ..core.local_artifacts import (
    LOCAL_ROOT_ENV,
    LocalArtifactError,
    LocalArtifactStore,
    require_run_artifact,
)

audit_app = typer.Typer(
    add_completion=False,
    help=(
        "Truth-blind application audit: patient/PHI/tool/artifact/integrity "
        "boundaries. This is separate from clinical evaluation."
    ),
)
LOCAL_ROOT = typer.Option(None, "--local-root", envvar=LOCAL_ROOT_ENV)
DECLARED_TOOL = typer.Option([], "--declared-tool")
RULE = typer.Option([], "--rule")


def _store(root: str | None) -> LocalArtifactStore:
    try:
        return LocalArtifactStore(root)
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _rows(path: Path) -> tuple[dict, ...]:
    if not path.is_file():
        return ()
    return tuple(
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


@audit_app.command("rules")
def rules():
    """List built-in AuditRule assets; no model or chart is accessed."""
    table = Table("audit rule", "runner", "authority", "description")
    for asset in builtin_audit_registry().all():
        table.add_row(
            asset.ref,
            asset.runner_type,
            asset.maximum_authority,
            asset.description,
        )
    con.print(table)


def audit_run_payload(*, manifest: str, subject_id: str = "",
                      provider_boundary: str = "UNKNOWN",
                      declared_tool: tuple[str, ...] = (),
                      rule: tuple[str, ...] = (),
                      local_root: str | None = None) -> dict:
    """Run the truth-blind audit over one manifest and return the report as a dict.

    Split out of the `run` command so `acr signal run --kind rule` reaches the same
    AuditContext construction rather than assembling a second one. Two places that build a
    trajectory from a manifest is two places that can disagree about what the run did.

    `subject_id` defaults to the manifest's own `patient_id`; the command still requires it
    explicitly because an operator naming the wrong subject is a boundary error, whereas a
    dispatcher reading it from the file it was handed is not.
    """
    store = _store(local_root)
    manifest_path = require_run_artifact(manifest, what="audit manifest")
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    trace_path = manifest_path.with_name(
        manifest_path.name.replace(".manifest.json", ".jsonl")
    )
    trace = _rows(trace_path)
    artifacts = [ArtifactRef.from_path(manifest_path)]
    if trace_path.is_file():
        artifacts.append(ArtifactRef.from_path(trace_path))
    patient_scope = str(raw.get("patient_id") or subject_id)
    trajectory = TrajectoryAdapter.from_run_artifacts(
        manifest=raw,
        trace=trace,
        case_ref=patient_scope,
        spec_id=str(raw.get("spec_id") or "unspecified-spec"),
        spec_hash=str(raw.get("spec_hash") or digest(raw.get("spec") or {})),
        runtime_profile_id=str(
            raw.get("runtime_profile_id") or "current-stratified-coverage"
        ),
        runtime_profile_hash=str(raw.get("runtime_profile_hash") or ""),
        artifact_refs=artifacts,
    )
    declared = tuple(
        declared_tool
        or raw.get("declared_tools")
        or raw.get("tool_bundle")
        or ()
    )
    report = AuditRunner(
        builtin_audit_registry(), AuditStore(store)
    ).run(
        AuditContext(
            trajectory=trajectory,
            application_events=trace,
            patient_scope=patient_scope,
            provider_boundary=provider_boundary,
            declared_tools=declared,
            local_root=str(store.root),
            git_root=str(store.git_root),
        ),
        rule_refs=tuple(rule),
    )
    return report.to_dict()


@audit_app.command("run")
def run(
    manifest: str = typer.Option(..., "--manifest"),
    subject_id: str = typer.Option(..., "--subject-id"),
    provider_boundary: str = typer.Option("UNKNOWN", "--provider-boundary"),
    declared_tool: list[str] = DECLARED_TOOL,
    rule: list[str] = RULE,
    local_root: str | None = LOCAL_ROOT,
):
    """Audit one completed local run without truth or clinical judgement."""
    report = audit_run_payload(
        manifest=manifest, subject_id=subject_id, provider_boundary=provider_boundary,
        declared_tool=tuple(declared_tool), rule=tuple(rule), local_root=local_root)
    con.print_json(json.dumps(report, ensure_ascii=False))
    if report.get("incidents"):
        raise typer.Exit(2)


@audit_app.command("summarize")
def summarize(local_root: str | None = LOCAL_ROOT):
    """Summarize separate audit ledgers."""
    store = _store(local_root)
    findings = _rows(store.root / "assets/pricing/findings.jsonl")
    incidents = _rows(store.root / "assets/pricing/incidents.jsonl")
    con.print_json(json.dumps({
        "n_findings": len(findings),
        "n_incidents": len(incidents),
        "incident_kinds": sorted({
            str(row.get("payload", {}).get("kind") or "")
            for row in incidents
            if row.get("payload")
        }),
    }, ensure_ascii=False))


@audit_app.command("incidents")
def incidents(local_root: str | None = LOCAL_ROOT):
    """Print high-fidelity audit incidents, never evaluation failures."""
    store = _store(local_root)
    rows = _rows(store.root / "assets/pricing/incidents.jsonl")
    con.print_json(json.dumps({
        "n_incidents": len(rows),
        "incidents": list(rows),
    }, ensure_ascii=False))
