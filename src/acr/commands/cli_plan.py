"""Answer what CAN be run before anything is run: route one question to a spec, and report what
a guideline reads and what a spec edit invalidates.

Neither command opens a chart. `ask` may spend one model call on a classifier that sees the
question and the specs' vocabulary; `deps` is pure rule. They sit together because they share
the one property that decides whether a command is safe to put in CI: no PHI is read, so the
worst outcome of running either is a wrong plan on the screen.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.table import Table

from ..authoring.intake import ModelClassifier, load_guidelines, route
from ..contract import deps as depsmod
from ..contract.concordance import GuidelineError
from ..contract.registry_catalog import VariableCatalog
from ..core import cli_common
from ..core.cli_common import API_BASE, MODEL, code_sha, con

plan_app = typer.Typer(add_completion=False)

ASK_SCHEMA = "acr.routing/1"


@plan_app.command("ask")
def ask(
    question: str = typer.Argument(..., help="a question, a variable name, a spec id, a STORE "
                                             "item, a guideline id or a recommendation id"),
    specs_dir: str = typer.Option("assets/specs", "--specs", help="directory scanned for specs"),
    guidelines_dir: str = typer.Option("assets/guidelines", "--guidelines",
                                       help="directory scanned for guideline YAMLs"),
    skills_dir: str = typer.Option("skills", "--skills-dir",
                                   help="checked so a route to a skill that is not there is "
                                        "reported instead of printed as advice"),
    model: str = MODEL,
    api_base: str = API_BASE,
    dry_run: bool = typer.Option(True, "--dry-run/--no-dry-run",
                                 help="plan only. There is no other mode; see below."),
    out: str = typer.Option("", "--out", help="write the routing report JSON here"),
    as_json: bool = typer.Option(False, "--json", help="print the report as JSON"),
):
    """L0: route one question to one of five outcomes, or to an explicit gap list.

    Without --model only exact names resolve — a variable, a spec id, a STORE item, a
    guideline id, a recommendation id. Prose needs a classifier and gets one model call,
    which sees the question and the specs' vocabulary and never a chart.

    --dry-run defaults on and --no-dry-run is refused rather than implemented. This layer
    decides what work to do; `acr extract` and `acr concord` do it. Wiring execution in here
    would put a command that reads PHI behind a flag whose default is the only safe value,
    and defaults get overridden in scripts.
    """
    catalog = VariableCatalog.from_directory(specs_dir)
    guidelines = load_guidelines(guidelines_dir)
    clf = None
    if model:
        # temperature=None, not 0.0: `LLMConfig.from_env` skips None overrides, so the
        # provider's own ACR_TEMPERATURE is used. gpt-5.6-luna 400s on temperature 0 — "only
        # the default (1) value is supported" — and acr's 0.0 default would kill the call
        # before it left the machine.
        clf = ModelClassifier(cli_common.llm_client(model, api_base, temperature=None))
    d = route(question, catalog, classifier=clf, guidelines=guidelines, skills_dir=skills_dir)

    if not dry_run:
        con.print("[red]`acr ask` never reads a chart, so there is nothing for --no-dry-run to "
                  "do. Run the plan below with `acr extract`.[/]")

    if as_json:
        con.print_json(json.dumps(_ask_doc(d, question, specs_dir, guidelines_dir)))
    else:
        _render_routing(d)
    if out:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_ask_doc(d, question, specs_dir, guidelines_dir), indent=2,
                                default=str), encoding="utf-8")
        con.print(f"→ {p}")
    if not dry_run or d.refused:
        raise typer.Exit(2)


def _ask_doc(d, question: str, specs_dir: str, guidelines_dir: str) -> dict:
    return {"schema": ASK_SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "code_sha": code_sha(), "question": question,
            "specs_dir": specs_dir, "guidelines_dir": guidelines_dir,
            "routing": d.to_dict()}


def _render_routing(d) -> None:
    """Print the decision, then the gap list at full length.

    The gap list is not truncated and is not summarised to a count. Routing the shipped NCCN
    subset produces twenty-three gaps against six resolved inputs, and a report that showed
    "6 resolved" and a rolled-up total would be read as progress. It is the opposite: nothing
    in that guideline can be scored yet, and the only useful output is the list of what a
    human has to go and do.
    """
    con.print(f"\n[bold]question[/]: {d.question}")
    verdict = d.outcome or "NO ROUTE"
    colour = "red" if d.refused else "green"
    con.print(f"[bold]routed as[/]: [{colour}]{verdict}[/]"
              + (f"   [dim](classifier said {d.classified_as}; the route is the code's)[/]"
                 if d.classified_as and d.classified_as != d.outcome else ""))
    con.print(f"[bold]classifier[/]: {d.classifier} — {d.rationale}")
    if d.guideline_id:
        con.print(f"[bold]guideline[/]: {d.guideline_id} "
                  f"({', '.join(d.recommendation_ids) or 'all recommendations'})")

    if d.resolved:
        t = Table("input", "outcome", "spec_id", "declared source", "note")
        for r in d.resolved:
            c = {"EXISTING_VARIABLE": "green", "WRONG_DATA_SOURCE": "yellow"}.get(r.outcome, "red")
            t.add_row(r.name, f"[{c}]{r.outcome}[/]", r.spec_id or "—",
                      r.declared_source or "—", (r.note or "")[:60])
        con.print(t)
        n_ok = sum(1 for r in d.resolved if r.outcome == "EXISTING_VARIABLE")
        con.print(f"[bold]{n_ok} of {len(d.resolved)} input(s) resolve to a shipped spec[/]")

    if d.predicate is not None:
        con.print("\n[bold]predicate[/] "
                  + ("[green]checkable[/]" if d.predicate.checkable else "[red]REFUSED[/]")
                  + ("" if d.predicate.complete else " [yellow]incomplete[/]"))
        con.print(f"  {d.predicate.expression()}")
        for term in d.predicate.terms:
            if not term.ok:
                con.print(f"  [red]✗ {json.dumps(term.condition)}: "
                          f"{'; '.join(term.problems)}[/]")
        con.print("[dim]  paste `conditions` into a guideline's applies_when; it is the same "
                  "grammar acr.contract.concordance executes[/]")

    if d.skeleton is not None:
        con.print(f"\n[bold]spec skeleton[/] for {d.skeleton.variable} "
                  f"→ route to {d.skeleton.route}")
        if d.skeleton.why_not_composable:
            con.print(f"  [dim]why not a composition: {d.skeleton.why_not_composable}[/]")
        con.print(f"[dim]{d.skeleton.yaml_text}[/]")
        con.print("[bold]a human must answer these before the skeleton is a spec:[/]")
        for i, q in enumerate(d.skeleton.open_questions, 1):
            con.print(f"  {i:2d}. [{q['key']}] {q['question']}")

    if d.gaps:
        con.print(f"\n[bold]{len(d.gaps)} GAP(S)[/] — nothing below is defaulted, guessed or "
                  f"quietly dropped")
        t = Table()
        # `kind` is a closed, machine-checked vocabulary (see intake.py's *_ constants) —
        # rich's default ellipsis overflow was chopping the longest of them (e.g.
        # "not_yet_extractable") to "not_yet_extracta…" whenever the console was narrow.
        # A gap's kind is exactly what a human greps for; it must never be shortened.
        t.add_column("kind", min_width=24, no_wrap=True)
        t.add_column("subject")
        t.add_column("detail")
        t.add_column("remedy")
        for g in d.gaps:
            t.add_row(f"[red]{g.kind}[/]" if g.refusing else f"[yellow]{g.kind}[/]",
                      g.subject[:44], g.detail, g.remedy)
        con.print(t)
    else:
        con.print("\n[green]no gaps[/]")
    con.print(f"[dim]{d.model_calls} model call(s); no chart was read[/]")


# ============================================================ L4.5: dependencies
def _render_deps(doc: dict, impact) -> None:
    """Print the graph, and the gap list AT FULL LENGTH.

    `typer.echo`, not `con.print`, and no table: rich wraps at the console width, and a
    wrapped gap list breaks the one thing a reader greps for — `date_of_first_adjuvant_
    systemic_therapy` is 41 characters and a narrow console splits it in half. The list is
    also never truncated to a head and a count. Twenty-one of the shipped guideline's
    thirty-eight declared inputs have no supply route; "21 gaps" plus a top-five would be
    summarising that into something more encouraging than it is.
    """
    g = doc["guideline"]
    typer.echo(f"guideline {g['guideline_id']} v{g['guideline_version']} "
               f"(hash {g['guideline_hash']})")
    fwd = doc["forward"]["per_recommendation"]
    for rid, rd in fwd.items():
        typer.echo(f"\n{rid}: {len(rd['resolved'])} of {len(rd['inputs'])} declared input(s) "
                   f"have a supply route")
        for i in rd["inputs"]:
            if not i["resolved"]:
                continue
            typer.echo(f"  OK   {i['name']}  [{','.join(i['predicate_classes'])}]  "
                       f"{i['spec_id'] or i['source']}")
        exc = doc["exceptions_declared_per_rec"].get(rid) or []
        reason = doc.get("exceptions_none_declared_reason", {}).get(rid, "")
        typer.echo(f"  exceptions: {', '.join(exc) if exc else 'NONE DECLARED — ' + reason}")

    total = doc["forward"]["totals"]
    typer.echo(f"\n{total['gaps']} GAP(S) of {total['inputs']} declared input(s) — nothing "
               f"below is defaulted, guessed or quietly dropped")
    for rid, rd in fwd.items():
        for gap in rd["gaps"]:
            typer.echo(f"  GAP  {rid}  {gap['name']}  kind={gap['kind']}  {gap['detail']}")

    if impact is not None:
        typer.echo(f"\nimpact of {impact.spec_id}: read by {len(impact.recommendations)} "
                   f"recommendation(s)")
        for rid in impact.recommendations:
            typer.echo(f"  reads it: {rid}")
        for a in impact.artifacts:
            typer.echo(f"  {a.verdict}  {a.path}  {a.reason}")
        if impact.stale_results:
            typer.echo(f"  {impact.stale_results} recorded result(s) were computed under a "
                       f"definition that has since changed")


@plan_app.command("deps")
def deps_cmd(
    guideline: str = typer.Option(..., "--guideline", help="path to a guideline YAML"),
    specs_dir: str = typer.Option("assets/specs", "--specs", help="directory scanned for specs"),
    spec: str = typer.Option("", "--spec",
                             help="report what editing this spec invalidates under --runs"),
    runs: str = typer.Option("runs", "--runs", help="tree scanned for concordance artifacts"),
    as_json: bool = typer.Option(False, "--json", help="print the whole graph as JSON"),
    out: str = typer.Option("", "--out", help="write the dependency manifest here"),
    fail_on_gap: bool = typer.Option(False, "--fail-on-gap",
                                     help="exit 1 if any declared input has no supply route"),
):
    """L4.5: what each recommendation reads, what cannot be supplied, and what a spec edit
    invalidates. A rule layer — no model is called and no chart is read.

    Exit codes are the point of the command in CI: 2 if the guideline is refused (an
    undeclared exception list is a wrong and damaging number waiting to happen), 1 if a
    recorded concordance result went stale or --fail-on-gap and gaps exist, 0 otherwise.
    """
    try:
        d = depsmod.load_guideline_deps(guideline, specs_dir=specs_dir)
    except (depsmod.UndeclaredExceptionsError, GuidelineError) as e:
        # typer.echo, not con.print: rich re-wraps at the console width and a wrapped
        # recommendation id is one a reader cannot grep for.
        typer.echo(f"REFUSED: {e}")
        raise typer.Exit(2) from e

    doc = d.manifest() | {"guideline_path": str(Path(guideline).resolve()),
                          "specs_dir": specs_dir}
    impact = None
    if spec:
        try:
            impact = depsmod.impact_of_spec(spec, d, runs)
        except KeyError as e:
            typer.echo(str(e.args[0]))
            raise typer.Exit(2) from e
        doc["impact"] = impact.to_dict()

    if out:
        # Written before anything is printed: with --json the only thing on stdout must be
        # the JSON document, so a "wrote it here" line cannot be allowed to follow it.
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
    if as_json:
        typer.echo(json.dumps(doc, indent=2, default=str))
    else:
        _render_deps(doc, impact)
        if out:
            typer.echo(f"→ {out}")

    if impact is not None and any(a.verdict == depsmod.STALE for a in impact.artifacts):
        raise typer.Exit(1)
    if fail_on_gap and doc["forward"]["totals"]["gaps"]:
        raise typer.Exit(1)
