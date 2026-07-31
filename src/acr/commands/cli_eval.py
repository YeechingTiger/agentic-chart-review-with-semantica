"""Invoke the deterministic evaluation plane: what may be judged, what fired during a run, and
what changed between two baselines.

NOTHING IN THIS GROUP CALLS A MODEL, so nothing in it takes a cost ceiling — and that is a
property of `acr.evaluation.evals`, not a promise made here:
`tests/test_evals.py::test_no_model_is_reachable_from_this_module` walks the import closure and
fails if `acr.core.llm`, `acr.graph` or a provider SDK ever appears. What every command here DOES
take is its thresholds, because `DetectorConfig` gives none of them a default: a threshold
belongs where a reviewer reads it, not buried in a flag's default where it becomes folklore.

`eval dimensions` is the command to run before reaching for `acr judge`. It prints the fence.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.table import Table

from ..core.cli_common import con, dump, read_json
from ..evaluation import evals

eval_app = typer.Typer(add_completion=False, help=(
    "The deterministic evaluation plane: the precedence registry, the abnormal-behaviour "
    "detectors and the regression harness. No model is called by anything here."))

_MINTERM = typer.Option(..., "--min-term-chars",
                        help="a search term shorter than this cannot fail; no default")
_REPEATS = typer.Option(..., "--max-rejection-repeats",
                        help="the same answer_check firing this many times on a byte-identical "
                             "answer is a loop. No default, and the library floor is 2: "
                             "one rejection is not a loop.")
_TOKBAND = typer.Option(..., "--token-band", help="lo,hi total tokens a run may use; no default")
_TURNBAND = typer.Option(..., "--turn-band", help="lo,hi turns a run may take; no default")


def _band(raw: str, what: str) -> tuple[int, int]:
    try:
        lo, hi = (int(x) for x in str(raw).split(","))
    except ValueError as e:
        raise typer.BadParameter(f"{what} must be two integers, 'lo,hi'; got {raw!r}") from e
    if lo > hi:
        raise typer.BadParameter(f"{what}: lo {lo} is above hi {hi}")
    return lo, hi


def _detector_config(min_term_chars, max_rejection_repeats, token_band, turn_band):
    return evals.DetectorConfig(
        min_term_chars=min_term_chars, max_rejection_repeats=max_rejection_repeats,
        token_band=_band(token_band, "--token-band"), turn_band=_band(turn_band, "--turn-band"))


def _manifests(runs: str) -> list[Path]:
    p = Path(runs)
    if p.is_file():
        return [p]
    found = sorted(p.rglob("*.manifest.json"))
    if not found:
        raise typer.BadParameter(f"no *.manifest.json under {p}")
    return found


def _refinement_artifacts(runs: str) -> list[Path]:
    p = Path(runs)
    if p.is_file():
        found = [p]
    else:
        found = sorted(p.rglob("conflict-refinement.json"))
    if not found:
        raise typer.BadParameter(f"no conflict-refinement.json under {p}")
    return found


def _operational_baseline(runs: str) -> dict:
    rows = [read_json(path, "baseline manifest") for path in _manifests(runs)]
    costs = [
        float((row.get("spend") or {}).get("usd"))
        for row in rows
        if isinstance((row.get("spend") or {}).get("usd"), (int, float))
    ]
    return {
        "n_cases": len(rows),
        "n_deepagents_runs": len(rows),
        "n_gate_validated": sum(bool(row.get("gate_validated")) for row in rows),
        "n_review_required": sum(
            bool((row.get("answer") or {}).get("route_to_human")) for row in rows
        ),
        "total_cost_usd": round(sum(costs), 6),
        "mean_cost_usd": round(sum(costs) / len(costs), 6) if costs else None,
        "mean_steps": round(
            sum(float(row.get("steps") or 0) for row in rows) / len(rows), 6
        ) if rows else None,
    }


def _operational_refinement(runs: str) -> dict:
    artifacts = [read_json(path, "conflict-refinement artifact")
                 for path in _refinement_artifacts(runs)]
    for row in artifacts:
        if not isinstance(row, dict) or row.get("schema") != "acr.review.conflict_refinement/1":
            raise typer.BadParameter(
                "refinement input expected schema acr.review.conflict_refinement/1")
    hypotheses = [
        h for artifact in artifacts for round_rows in (artifact.get("rounds") or ())
        for h in round_rows
    ]
    costs = [
        float(h["cost_usd"]) for h in hypotheses
        if isinstance(h.get("cost_usd"), (int, float))
    ]
    statuses = {
        status: sum(a.get("status") == status for a in artifacts)
        for status in ("NO_CONFLICT", "CONVERGED", "REVIEW_REQUIRED")
    }
    return {
        "n_cases": len(artifacts),
        "n_deepagents_runs": len(hypotheses),
        "n_gate_validated": sum(bool(h.get("gate_validated")) for h in hypotheses),
        "n_review_required": statuses["REVIEW_REQUIRED"],
        "status_counts": statuses,
        "total_cost_usd": round(sum(costs), 6),
        "mean_cost_usd_per_case": (
            round(sum(costs) / len(artifacts), 6) if costs and artifacts else None
        ),
        "mean_rounds": (
            round(sum(float(a.get("n_rounds") or 0) for a in artifacts)
                  / len(artifacts), 6) if artifacts else None
        ),
    }


@eval_app.command("dimensions")
def dimensions(
    check: str = typer.Option("", "--check",
                              help="comma list of dimension names to rule on; exits 1 if any "
                                   "is unregistered or already decided by code"),
    out: str = typer.Option("", "--out", help="write the registry JSON here"),
):
    """Print THE FENCE: every evaluable dimension and whether a model judge is allowed on it.

    This is the one namespace. `acr.evaluation.judge` advertises dimensions and `assets/evaluators/*.yaml`
    declare one each; both are checked against these rows, and they once had zero names in
    common while every test on both sides passed.
    """
    # `no_wrap` with a min width on the identifier column, and never rich's default
    # ellipsis. A dimension name is exactly what a reader greps for, and rich silently chops the
    # longest of them on a narrow console — `acr deps` documents the same defect, where
    # a 41-character variable name came out split in half.
    t = Table()
    t.add_column("dimension", min_width=30, no_wrap=True)
    for col in ("judge?", "deterministic method", "verifier"):
        t.add_column(col)
    rows = []
    for name, d in evals.REGISTRY.items():
        if d.sub_questions:
            t.add_row(name, "[yellow]SPLIT[/]", f"names {', '.join(d.sub_questions)}", "—")
        else:
            t.add_row(name, "[green]allowed[/]" if not d.deterministic else "[red]FORBIDDEN[/]",
                      (d.method or "")[:56], d.verifier or "—")
        rows.append({"dimension": name, "deterministic": d.deterministic,
                     "sub_questions": list(d.sub_questions), "method": d.method,
                     "verifier": d.verifier, "replaces_judge_metric": d.replaces_judge_metric,
                     "why": d.why})
    con.print(t)
    con.print(f"[dim]{len(evals.judgeable_dimensions())} of {len(rows)} row(s) are judgeable; "
              f"the rest are already decided by code or are split parents[/]")

    problems: dict[str, str] = {}
    if check:
        problems = evals.unknown_dimensions([c.strip() for c in check.split(",") if c.strip()])
        for name, why in problems.items():
            con.print(f"[red]{name}[/]: {why}")
        if not problems:
            con.print("[green]every checked name is registered and judgeable[/]")
    dump({"registry": rows, "checked": problems}, out)
    if problems:
        raise typer.Exit(1)


@eval_app.command("detect")
def detect(
    runs: str = typer.Option(..., "--runs", help="a *.manifest.json, or a tree of them"),
    min_term_chars: int = _MINTERM,
    max_rejection_repeats: int = _REPEATS,
    token_band: str = _TOKBAND,
    turn_band: str = _TURNBAND,
    out: str = typer.Option("", "--out", help="write the findings JSON here"),
):
    """Run the abnormal-behaviour detectors over recorded runs. Exits 1 on IRB or CRITICAL.

    Each detector was first found by hand-auditing traces after the fact, which does not
    scale and did not happen twice. Person ids are masked on the way out, so the report is
    safe to write into the tree.
    """
    cfg = _detector_config(min_term_chars, max_rejection_repeats, token_band, turn_band)
    rows, worst = [], []
    t = Table("severity", "detector", "run", "message")
    for path in _manifests(runs):
        run = evals.RunRecord.from_manifest(path)
        for f in evals.run_detectors(run, config=cfg):
            d = f.to_dict()
            d["source"] = str(path)
            rows.append(d)
            colour = {evals.IRB: "red", evals.CRITICAL: "red"}.get(f.severity, "yellow")
            t.add_row(f"[{colour}]{f.severity}[/]", f.detector, path.name, d["message"][:70])
            if f.severity in (evals.IRB, evals.CRITICAL):
                worst.append(d)
    con.print(t if rows else "[green]no detector fired[/]")
    con.print(f"[bold]{len(rows)} finding(s)[/], {len(worst)} at IRB or CRITICAL")
    dump({"config": {"min_term_chars": cfg.min_term_chars,
                     "max_rejection_repeats": cfg.max_rejection_repeats,
                     "token_band": list(cfg.token_band), "turn_band": list(cfg.turn_band)},
          "n_findings": len(rows), "findings": rows}, out)
    if worst:
        raise typer.Exit(1)


@eval_app.command("score")
def score(
    runs: str = typer.Option(..., "--runs", help="a *.manifest.json, or a tree of them"),
    answer_key: str = typer.Option(..., "--answer-key",
                                   help="JSON {instance_id: {fields: {...}, subgroups: [...]}}; "
                                        "a field whose key value is null asserts that "
                                        "ABSTENTION is the correct answer"),
    fields: str = typer.Option(..., "--fields",
                               help="comma list. Required: scoring whatever keys the model "
                                    "happened to emit makes the denominator depend on the model"),
    commit: str = typer.Option(..., "--commit", help="part of the baseline key"),
    spec_hash: str = typer.Option(..., "--spec-hash", help="part of the baseline key"),
    model: str = typer.Option(..., "--model", help="part of the baseline key"),
    date: str = typer.Option(..., "--date", help="part of the baseline key"),
    baseline: str = typer.Option("", "--baseline", help="write the full report here"),
    min_term_chars: int = typer.Option(0, "--min-term-chars",
                                       help="with the other three bands, also runs the "
                                            "detectors and files their findings per instance"),
    max_rejection_repeats: int = typer.Option(0, "--max-rejection-repeats"),
    token_band: str = typer.Option("", "--token-band"),
    turn_band: str = typer.Option("", "--turn-band"),
):
    """Score recorded runs against an answer key. Reads only; runs nothing.

    Every part of the baseline key is required because a baseline is only comparable across
    all four: "accuracy fell" and "the question changed" look identical in the numbers, and
    the key is what lets `eval compare` tell the reader which one happened.
    """
    key = evals.BaselineKey(commit=commit, spec_hash=spec_hash, model=model, date=date)
    names = [f.strip() for f in fields.split(",") if f.strip()]
    if not names:
        raise typer.BadParameter("--fields resolved to nothing")
    akey = read_json(answer_key, "answer key")
    if not isinstance(akey, dict):
        raise typer.BadParameter(f"{answer_key}: expected an object keyed by instance id")
    # All four bands or none: a partial set would silently disable the detectors while the
    # report still carried a `findings` block reading zero, which is the shape of a channel
    # that is broken rather than clean.
    supplied = [min_term_chars, max_rejection_repeats, token_band, turn_band]
    cfg = None
    if any(supplied):
        if not all(supplied):
            raise typer.BadParameter("the detectors need all four of --min-term-chars, "
                                     "--max-rejection-repeats, --token-band and --turn-band, "
                                     "or none of them")
        cfg = _detector_config(min_term_chars, max_rejection_repeats, token_band, turn_band)

    records = [evals.RunRecord.from_manifest(p) for p in _manifests(runs)]
    report = evals.score(records, akey, fields=names, key=key, detector_config=cfg)
    typer.echo(report.table())
    if cfg is None:
        con.print("[dim]detectors NOT RUN: no thresholds were declared, so `findings` is "
                  "empty because nothing looked, not because nothing fired[/]")
    if baseline:
        con.print(f"→ {evals.save_baseline(report, baseline)}")


@eval_app.command("compare")
def compare(
    before: str = typer.Option(..., "--before", help="a baseline written by `eval score`"),
    after: str = typer.Option(..., "--after", help="a baseline written by `eval score`"),
    out: str = typer.Option("", "--out", help="write the delta JSON here"),
):
    """Two baselines into a delta. Exits 1 on REGRESSION, per instance AND per subgroup.

    The verdict is REGRESSION if any instance left a good outcome or any subgroup rate fell,
    even when every headline rate rose: the aggregate improvement is what gets shipped and
    the subgroup collapse is what reaches a patient.
    """
    d = evals.compare(evals.load_baseline(before), evals.load_baseline(after))
    for line in d["key_differences"]:
        con.print(f"[yellow]baseline key moved: {line}[/]")
    t = Table("field", "before", "after", "delta")
    for f, p in d["per_field"].items():
        t.add_row(f, str(p["before"]), str(p["after"]), str(p["delta"]))
    con.print(t)
    if nc := d.get("not_comparable"):
        # Printed BEFORE the field table would be tidier, but the field rates are still real
        # and worth seeing. What must not happen is a per-instance count appearing beneath
        # them, because a reader takes `0 regression(s)` as a result rather than as silence.
        con.print(f"[bold red]NOT_COMPARABLE[/] — {nc['reason']} "
                  f"({nc['n_colliding']} colliding id(s), basis={nc['pseudonym_basis']})")
        con.print(f"[dim]{nc['why']}[/]")
        con.print(f"[yellow]remedy: {nc['remedy']}[/]")
        dump(d, out)
        if not out:
            typer.echo(json.dumps({"verdict": d["verdict"]}))
        raise typer.Exit(2)
    for r in d["regressions"]:
        con.print(f"[red]REGRESSED[/] {r['instance_id']} / {r['field']}: "
                  f"{r['before']} -> {r['after']}")
    for s in d["subgroup_regressions"]:
        con.print(f"[red]SUBGROUP[/] {s['subgroup']} / {s['field']}: "
                  f"{s['before']} -> {s['after']} (n={s['n_after']})")
    colour = "red" if d["verdict"] == "REGRESSION" else "green"
    con.print(f"[bold {colour}]{d['verdict']}[/]  {len(d['improvements'])} improvement(s), "
              f"{len(d['regressions'])} regression(s), "
              f"{len(d['subgroup_regressions'])} subgroup regression(s)")
    dump(d, out)
    if not out:
        typer.echo(json.dumps({"verdict": d["verdict"]}))
    if d["verdict"] == "REGRESSION":
        raise typer.Exit(1)


@eval_app.command("compare-refinement")
def compare_refinement(
    baseline: str = typer.Option(
        ..., "--baseline", help="single-run baseline manifest or manifest tree"),
    refined: str = typer.Option(
        ..., "--refined", help="conflict-refinement.json file or artifact tree"),
    out: str = typer.Option("", "--out", help="write the operational A/B report here"),
):
    """Compare optional refinement compute and routing; correctness is scored separately.

    Conflict artifacts deliberately contain no patient identifier or gold label, so this
    command cannot manufacture a paired accuracy result. Use `repair validate` with
    chart-observable gold for correctness and this report for incremental compute, rounds and
    human-review routing.
    """
    before = _operational_baseline(baseline)
    after = _operational_refinement(refined)
    same_size = before["n_cases"] == after["n_cases"]
    report = {
        "schema": "acr.refinement_comparison/1",
        "scope": "operational_only",
        "correctness_command": "acr repair validate",
        "baseline": before,
        "refined": after,
        "comparable_case_count": same_size,
        "delta": {
            "deepagents_runs": after["n_deepagents_runs"] - before["n_deepagents_runs"],
            "total_cost_usd": round(
                after["total_cost_usd"] - before["total_cost_usd"], 6),
            "review_required": after["n_review_required"] - before["n_review_required"],
        },
    }
    t = Table("arm", "cases", "deepagents runs", "gate-valid runs", "review", "cost")
    t.add_row(
        "baseline", str(before["n_cases"]), str(before["n_deepagents_runs"]),
        str(before["n_gate_validated"]), str(before["n_review_required"]),
        f"${before['total_cost_usd']:.4f}")
    t.add_row(
        "refined", str(after["n_cases"]), str(after["n_deepagents_runs"]),
        str(after["n_gate_validated"]), str(after["n_review_required"]),
        f"${after['total_cost_usd']:.4f}")
    con.print(t)
    con.print(
        "[dim]Operational comparison only. Run `acr repair validate` on pseudonymous "
        "chart-observable gold for paired correctness and subgroup regression.[/]")
    if not same_size:
        con.print("[yellow]Case counts differ; the aggregate operational deltas are not a "
                  "paired estimate.[/]")
    dump(report, out)
