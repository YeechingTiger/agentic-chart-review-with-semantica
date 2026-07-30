"""Manage local-only registry references and chart-observable gold."""
from __future__ import annotations

import typer
from rich.table import Table

from . import spec_repair as S
from .cli_common import con, read_json
from .local_artifacts import LOCAL_ROOT_ENV, LocalArtifactError, LocalArtifactStore

gold_app = typer.Typer(add_completion=False, help=(
    "DEVELOP only: stage registry values as LOCAL unresolved references, record chart "
    "derivability, and audit whether chart-observable gold may guide repair."))

LOCAL_ROOT = typer.Option(
    None, "--local-root", envvar=LOCAL_ROOT_ENV,
    help="absolute patient-artifact root outside Git; may also use ACR_LOCAL_ARTIFACT_ROOT")


def _store(root: str | None) -> LocalArtifactStore:
    try:
        return LocalArtifactStore(root)
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc


@gold_app.command("audit")
def audit(
    gold: str = typer.Option(..., "--gold", help=f"{S.GOLD_SCHEMA} JSON"),
    out: str = typer.Option("", "--out", help="root-relative local audit JSON"),
    local_root: str | None = LOCAL_ROOT,
):
    """Audit derivability, adjudication and witnesses; reads no chart and calls no model."""
    store = _store(local_root)
    try:
        gold_path = store.require_input(gold, what="chart-observable gold")
        report = S.audit_gold(S.load_gold(gold_path).values())
    except (S.SpecRepairError, LocalArtifactError) as e:
        raise typer.BadParameter(str(e)) from e
    t = Table("severity", "case", "finding")
    for row in report["findings"]:
        colour = "red" if row["severity"] == "BLOCK" else "yellow"
        t.add_row(f"[{colour}]{row['severity']}[/]", row["case_id"], row["finding"])
    con.print(t if report["findings"] else "[green]gold audit has no finding[/]")
    s = report["summary"]
    con.print(f"[bold]{s['n_repair_eligible']}/{s['n_cases']} case(s) repair-eligible[/]; "
              f"repair_ready={report['repair_ready']}")
    if out:
        try:
            path = store.write_json(out, report)
        except LocalArtifactError as exc:
            raise typer.BadParameter(str(exc)) from exc
        con.print(f"→ {path}")
    if not report["repair_ready"]:
        raise typer.Exit(1)


@gold_app.command("stage-registry-reference")
def stage_registry_reference(
    answer_key: str = typer.Option(
        ..., "--answer-key",
        help="LOCAL registry JSON {case_id: {fields: {...}}}; never promoted automatically"),
    spec_id: str = typer.Option(..., "--spec-id"),
    source_version: str = typer.Option(..., "--source-version"),
    out: str = typer.Option(..., "--out", help="root-relative LOCAL registry-reference JSON"),
    local_root: str | None = LOCAL_ROOT,
):
    """Stage registry values locally as unresolved references — not silver and not gold."""
    store = _store(local_root)
    try:
        answer_key_path = store.require_input(answer_key, what="registry answer key")
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc
    raw = read_json(answer_key_path, "registry answer key")
    if not isinstance(raw, dict):
        raise typer.BadParameter("registry answer key must be an object keyed by case id")
    cases = []
    for case_id, row in raw.items():
        values = (row or {}).get("fields") if isinstance(row, dict) else None
        if values is None and isinstance(row, dict):
            values = row
        if not isinstance(values, dict):
            raise typer.BadParameter(
                f"{answer_key}[{case_id!r}]: expected fields object")
        try:
            safe = S.safe_case_id(case_id)
        except S.SpecRepairError as e:
            raise typer.BadParameter(f"{case_id}: {e}") from e
        cases.append({
            "case_id": safe,
            "spec_id": spec_id,
            "registry_value": dict(values),
            "registry_source_version": source_version,
            "adjudication": "UNRESOLVED",
        })
    doc = {
        "schema": "acr.registry_reference/1",
        "contains_phi": True,
        "storage": "LOCAL_ONLY",
        "shareable": False,
        "cases": cases,
        "summary": {"n_cases": len(cases), "n_adjudicated": 0},
    }
    try:
        p = store.write_json(out, doc)
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc
    con.print(
        f"[yellow]{len(cases)} registry reference row(s) staged locally[/]: {p}\n"
        "No row was de-identified, imported into a dataset, or promoted to chart-observable "
        "gold. Every row remains UNRESOLVED until human chart adjudication.")
