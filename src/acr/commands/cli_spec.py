"""Put a specification in front of the clinician who owns its decisions, and record what they
said about it.

The only command group in the tree whose user is not an engineer. Every other group assumes
the reader can open a YAML file; these three exist because the person who owns the clinical
decisions in that file cannot, and has therefore never seen them.
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.table import Table

from ..authoring import speclint
from ..contract.spec import load_spec
from ..core.cli_common import con
from ..usecase import specview

spec_app = typer.Typer(add_completion=False,
                       help="Put a specification in front of the clinician who owns its decisions.")

SIGNOFFS = typer.Option("signoffs", "--signoffs",
                        help="directory of append-only sign-off ledgers, one file per spec")


@spec_app.command("lint")
def spec_lint(
    path: str = typer.Argument(..., help="a spec YAML, or a directory of them"),
    corpus: str = typer.Option(None, "--corpus", help="tier 2 only; without it tier 2 is NOT RUN"),
    max_patients: int = typer.Option(None, "--max-patients", help="tier 2 only; required with --corpus"),
    answer_key: str = typer.Option(None, "--answer-key", help="tier 3 only; refused without --tier3"),
    tier3: bool = typer.Option(False, "--tier3", help="opt in to answer-key checks"),
    bound: list[float] = typer.Option([], "--bound", help="extra elusion cap to price in the F8 table"),
    n: list[int] = typer.Option([], "--n", help="extra sample size to price in the F8 table"),
):
    """Four-tier completeness check. Exits non-zero on any TIER 1 failure.

    Tier 1 runs; tier 2 needs the charts and is not run without them; tier 3 is refused unless
    the caller asks for it by name AND supplies the key. The tiers are printed separately
    because they cost different things to run and mean different things when they pass, and a
    single PASS over all four is the sentence this command exists to make unsayable.
    """
    p = Path(path)
    # rglob, so `acr spec lint specs` covers assets/specs/ablation too: an arm nobody lints is an arm
    # that drifts, and the ablation spec is the one a result gets compared against.
    paths = sorted(p.rglob("*.yaml")) if p.is_dir() else [p]
    specs = [load_spec(f) for f in paths]
    typer.echo(speclint.render_report(specs, corpus=corpus, answer_key=answer_key,
                                      tier3_enabled=tier3, bounds=bound, sizes=n))
    if answer_key or tier3:
        try:
            speclint.tier3_checks(specs[0], answer_key=answer_key, enabled=tier3)
        except speclint.AnswerKeyRefused as e:
            con.print(f"[red]{e}[/]")
            raise typer.Exit(code=2)
        except NotImplementedError as e:
            con.print(f"[yellow]{e}[/]")
    if corpus:
        if max_patients is None:
            con.print("[red]--corpus requires --max-patients: how much of the corpus a lint "
                      "may read is the caller's decision, not a default.[/]")
            raise typer.Exit(code=2)
        for s in specs:
            for f in speclint.tier2_checks(s, corpus, max_patients):
                con.print(f"[red]{f.check}[/] {f.spec_id} {f.where}: {f.message}")
    failures = sum(1 for s in specs for f in speclint.lint_spec(s)
                   if f.severity == speclint.FAIL)
    if failures:
        raise typer.Exit(code=1)


@spec_app.command("review")
def spec_review(
    spec_path: str = typer.Argument(..., help="path to a spec YAML"),
    out: str = typer.Option(..., "--out", help="markdown file to write"),
    signoffs: str = SIGNOFFS,
):
    """Render a spec as a document a registrar can read in ten minutes and mark up."""
    s = load_spec(spec_path)
    recorded = specview.load_signoffs(signoffs, s.spec_id)
    doc = specview.render_review(s, source_path=spec_path, signoffs=recorded)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(doc, encoding="utf-8")

    els = specview.elements(s, source_path=spec_path)
    states = [specview.signoff_status(e, recorded)[0] for e in els]
    t = Table("spec", "elements", "decisions to confirm", "made up", "confirmed", "stale")
    t.add_row(s.spec_id, str(len(els)),
              str(len(specview.decisions(s, source_path=spec_path, els=els))),
              str(sum(1 for e in els if e.provenance == specview.MODEL_AUTHORED)),
              str(states.count(specview.SIGNED)), str(states.count(specview.STALE)))
    con.print(t)
    con.print(f"[dim]{out}[/]")


@spec_app.command("signoff")
def spec_signoff(
    spec_path: str = typer.Option(..., "--spec", help="path to a spec YAML"),
    reviewer: str = typer.Option(..., "--reviewer", help="who is confirming it"),
    element: str = typer.Option(..., "--element", help="element id, as printed in the review"),
    note: str = typer.Option("", "--note", help="anything the reviewer wants recorded"),
    signoffs: str = SIGNOFFS,
):
    """Record that a named reviewer confirmed one element, as it is worded today.

    The record carries the element's content hash, so the next render reports the approval as
    withdrawn the moment the wording changes. Clinical assent to a sentence is not assent to
    whatever that sentence is edited into.
    """
    s = load_spec(spec_path)
    try:
        rec = specview.record_signoff(signoffs, s, element, reviewer=reviewer,
                                      source_path=spec_path, note=note)
    except KeyError as e:
        con.print(f"[red]{e.args[0]}[/]")
        raise typer.Exit(code=2)
    con.print(f"[green]recorded[/] {rec['element_id']} ({rec['element_kind']}) "
              f"= {rec['element_hash']} by {rec['reviewer']} at {rec['signed_at']}")
