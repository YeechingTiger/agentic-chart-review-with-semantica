"""Invoke §6b's reflective optimizer over the text parameters an agent reads: what may be
edited and by whom, where a batch of failures routes, and what its acceptance test would cost.

NOTHING HERE CALLS A MODEL, so nothing here takes a cost ceiling. That is not restraint on
this module's part — `acr.improvement.refine` keeps the reflection call behind a seam whose only shipped
implementations are a stub and a `NotImplementedError`, and `FailureCase` refuses to hold a
real person_id, so a case assembled from the real corpus cannot be constructed at all. What
`route` therefore consumes is verdicts a caller ALREADY HAS on disk. Wiring a live reflector
in here would spend money on chart text, which is a decision this group does not get to make.

Every threshold is an argument with no default, exactly as the library demands: hard-coding
1.96 is how a power calculation stops being reviewable.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import typer
from rich.table import Table

from ..core.cli_common import con, dump, read_json
from ..improvement import refine as R

refine_app = typer.Typer(add_completion=False, help=(
    "§6b: route classified failures at the text parameter that caused them, size the "
    "validation run, and read its per-instance results. No model is called."))


class _FileReflector:
    """Verdicts the caller already has, keyed by case id.

    Not `StubReflector`, whose docstring reserves it for tests, and not `llm_reflector`, which
    is deliberately unbuilt. A verdict that arrived from a file still faces the citation mask
    in full: routing is not made easier by where the verdict came from.
    """

    def __init__(self, verdicts: Mapping[str, R.ReflectionVerdict]):
        self._by_case = dict(verdicts)

    def __call__(self, case: R.FailureCase, spec_text: str) -> R.ReflectionVerdict:
        try:
            return self._by_case[case.case_id]
        except KeyError:
            raise KeyError(f"no reflection verdict supplied for case {case.case_id!r}; "
                           f"--verdicts holds {sorted(self._by_case)}") from None


@refine_app.command("parameters")
def parameters(out: str = typer.Option("", "--out", help="write the registry JSON here")):
    """Print the text parameters the optimizer can reach, and WHO MAY UPDATE EACH. Free.

    `in_objective` is the asymmetry and exactly one row carries it: editing those sentences
    edits the definition of a correct answer, so loosening one raises agreement with the
    answer key and teaches us nothing.
    """
    R.registry_invariants()          # refuses at the door rather than mid-report
    # `no_wrap` with a min width on the identifier column, and never rich's default
    # ellipsis. A parameter id is exactly what a reader greps for, and rich silently chops the
    # longest of them on a narrow console — `acr deps` documents the same defect, where
    # a 41-character variable name came out split in half.
    t = Table()
    t.add_column("parameter", min_width=31, no_wrap=True)
    for col in ("file", "kind", "update policy", "in objective"):
        t.add_column(col)
    rows = []
    for p in R.PARAMETER_REGISTRY:
        t.add_row(p.id, p.file, p.kind, p.update_policy,
                  "[red]YES[/]" if p.in_objective else "no")
        rows.append({"id": p.id, "file": p.file, "path_within": p.path_within, "kind": p.kind,
                     "update_policy": p.update_policy, "in_objective": p.in_objective,
                     "mechanical": p.mechanical, "why": p.why})
    con.print(t)
    con.print(f"[dim]{len(R.DESIGN_TABLE_PARAMETER_IDS)} of {len(rows)} are the §6b design "
              f"table; the rest exist because the decision tree routes to them and a "
              f"destination the registry does not know is a gradient with nowhere to land[/]")
    dump({"parameters": rows,
          "design_table": list(R.DESIGN_TABLE_PARAMETER_IDS)}, out)


def _cases(path: str) -> list[R.FailureCase]:
    raw = read_json(path, "cases")
    if not isinstance(raw, list):
        raise typer.BadParameter(f"{path}: expected a JSON list of failure cases")
    out = []
    for i, d in enumerate(raw):
        try:
            out.append(R.FailureCase(
                case_id=str(d["case_id"]), spec_id=str(d["spec_id"]), field=str(d["field"]),
                coded_value=str(d.get("coded_value", "")), key_value=str(d.get("key_value", "")),
                establishing_evidence_surfaced=bool(d["establishing_evidence_surfaced"]),
                answer_key_adjudication=str(d["answer_key_adjudication"]),
                invoked_rules=tuple(d.get("invoked_rules") or ()),
                rejection_messages_seen=tuple(d.get("rejection_messages_seen") or ()),
                subgroup=str(d.get("subgroup", "unassigned"))))
        except KeyError as e:
            raise typer.BadParameter(f"{path}[{i}]: missing {e.args[0]!r}") from e
        except R.RefineError as e:
            # PhiInFailureCaseError lands here, and it must reach the shell as a refusal
            # rather than a traceback: the map from a real person_id lives outside this tree.
            con.print(f"[red]{path}[{i}]: {e}[/]")
            raise typer.Exit(2) from e
    return out


def _verdicts(path: str) -> dict[str, R.ReflectionVerdict]:
    raw = read_json(path, "verdicts")
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"{path}: expected an object {{case_id: verdict}}")
    return {str(cid): R.ReflectionVerdict(
        verdict=str(v["verdict"]), parameter_id=v.get("parameter_id"),
        rationale=str(v.get("rationale", "")), missing_sentence=v.get("missing_sentence"),
        quoted_passage=v.get("quoted_passage"), readings=tuple(v.get("readings") or ()),
        proposed_text=v.get("proposed_text")) for cid, v in raw.items()}


@refine_app.command("route")
def route(
    cases: str = typer.Option(..., "--cases", help="JSON list of failure cases"),
    verdicts: str = typer.Option(..., "--verdicts",
                                 help="JSON {case_id: reflection verdict}. Supplied, never "
                                      "generated: the reflection call is an unbuilt seam."),
    spec_text: list[str] = typer.Option(..., "--spec-text",
                                        help="spec_id=path, repeatable. Required, because "
                                             "without the text the citation mask can check "
                                             "that a quote is present but not that it is true."),
    out: str = typer.Option("", "--out", help="write batches, questions and leftovers here"),
):
    """Route each failure at the parameter that caused it. Acts on none of them.

    The two refusals that matter are visible in the output: a CONTENT gradient at a spec rule
    becomes a QUESTION and can never become an edit, and an uncited verdict returns UNRESOLVED
    rather than a guess.
    """
    texts: dict[str, str] = {}
    for pair in spec_text:
        sid, _, p = str(pair).partition("=")
        if not sid or not p:
            raise typer.BadParameter(f"--spec-text wants spec_id=path; got {pair!r}")
        if not Path(p).is_file():
            raise typer.BadParameter(f"--spec-text {sid}: no such file {p}")
        texts[sid] = Path(p).read_text(encoding="utf-8")

    router = R.GradientRouter(texts, _FileReflector(_verdicts(verdicts)))
    routings = []
    for case in _cases(cases):
        try:
            routings.append(router.route(case))
        except (R.RefineError, KeyError) as e:
            con.print(f"[red]{case.case_id}: {e}[/]")
            raise typer.Exit(2) from e

    try:
        # No mechanism is named, so a mechanical parameter reports NOT COMPUTABLE rather than
        # zero. Pricing a keyword by grep needs the note texts, and this command reads none.
        batches, questions, leftover = R.assemble(
            routings, lambda r: R.blast_radius_for(r.parameter_id or "skill"))
    except R.RefineError as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e

    t = Table("case", "verdict", "destination", "parameter", "class", "why not")
    for r in routings:
        t.add_row(r.case.case_id, r.verdict, r.destination, r.parameter_id or "—",
                  r.change_class or "—", (r.rejected_reason or "")[:44])
    con.print(t)
    con.print(f"[bold]{len(batches)} batch(es)[/], {len(questions)} clinician question(s), "
              f"{len(leftover)} unresolved; "
              f"{sum(1 for r in routings if not r.in_denominator)} out of the denominator")
    for q in questions:
        con.print(f"[yellow]QUESTION[/] {q.case_id} @ {q.parameter_id}: {q.question[:90]}")
    doc = {"n_routed": len(routings),
           "routings": [{"case_id": r.case.case_id, "verdict": r.verdict,
                         "destination": r.destination, "parameter_id": r.parameter_id,
                         "change_class": r.change_class, "citation": dict(r.citation),
                         "rejected_reason": r.rejected_reason,
                         "in_denominator": r.in_denominator} for r in routings],
           "batches": [b.to_dict() for b in batches],
           "questions": [q.to_dict() for q in questions],
           "unresolved": [{"case_id": r.case.case_id, "why": r.rejected_reason}
                          for r in leftover]}
    typer.echo(json.dumps({"n_batches": len(batches), "n_questions": len(questions),
                           "n_unresolved": len(leftover)}))
    dump(doc, out)


@refine_app.command("sample-size")
def sample_size(
    baseline_accuracy: float = typer.Option(..., "--baseline-accuracy"),
    detectable_regression_pp: float = typer.Option(..., "--detectable-regression-pp"),
    z_alpha: float = typer.Option(..., "--z-alpha",
                                  help="stated, never hard-coded: 1.96 typed into a function "
                                       "body is a power calculation nobody can review"),
    z_power: float = typer.Option(..., "--z-power"),
    cost_per_case_usd: float = typer.Option(..., "--cost-per-case-usd"),
    out: str = typer.Option("", "--out", help="write the sizing JSON here"),
):
    """Per-arm n for a two-proportion acceptance test, and what both arms would cost. Free.

    An underpowered run that shows no regression has not shown there is none, which is why
    `plan_validation` refuses a validation set smaller than this number.
    """
    try:
        n = R.required_per_arm_n(baseline_accuracy=baseline_accuracy,
                                 detectable_regression_pp=detectable_regression_pp,
                                 z_alpha=z_alpha, z_power=z_power)
    except R.RefineError as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e
    doc = {"per_arm_n": n, "arms": ["control", "candidate"],
           "baseline_accuracy": baseline_accuracy,
           "detectable_regression_pp": detectable_regression_pp,
           "z_alpha": z_alpha, "z_power": z_power,
           "cost_per_case_usd": cost_per_case_usd,
           "estimated_cost_usd": round(2 * n * cost_per_case_usd, 2),
           "status": "NOT RUN. This is a size and a price, not an experiment."}
    con.print(f"[bold]{n}[/] case(s) per arm to detect a {detectable_regression_pp}pp "
              f"regression at {baseline_accuracy} accuracy")
    con.print(f"[bold]${doc['estimated_cost_usd']}[/] for both arms at "
              f"${cost_per_case_usd}/case")
    typer.echo(json.dumps({"per_arm_n": n, "estimated_cost_usd": doc["estimated_cost_usd"]}))
    dump(doc, out)


@refine_app.command("read-results")
def read_results(
    results: str = typer.Option(..., "--results",
                                help="JSON list of {case_id, subgroup, control_correct, "
                                     "candidate_correct}"),
    max_tolerated_subgroup_drop_pp: float = typer.Option(
        ..., "--max-tolerated-subgroup-drop-pp",
        help="REQUIRED, no default. A positive mean does not carry a regressed subgroup."),
    out: str = typer.Option("", "--out", help="write the reading JSON here"),
):
    """Read a validation run PER INSTANCE. Exits 1 when the batch is not accepted. Free.

    A mean over these hides a revision that lifts the average while destroying one subgroup,
    which is the failure an average is built to hide.
    """
    raw = read_json(results, "results")
    if not isinstance(raw, list):
        raise typer.BadParameter(f"{results}: expected a JSON list; "
                                 f"see `per_instance_result_shape()`")
    try:
        rows = [R.PerInstanceResult(case_id=str(d["case_id"]),
                                    subgroup=str(d.get("subgroup", "unassigned")),
                                    control_correct=bool(d["control_correct"]),
                                    candidate_correct=bool(d["candidate_correct"])) for d in raw]
        reading = R.read_per_instance(
            rows, max_tolerated_subgroup_drop_pp=max_tolerated_subgroup_drop_pp)
    except KeyError as e:
        raise typer.BadParameter(f"{results}: a row is missing {e.args[0]!r}") from e
    except R.RefineError as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e

    t = Table("subgroup", "delta pp")
    for g, d in sorted(reading.per_subgroup_delta_pp.items()):
        colour = "red" if g in reading.regressed_subgroups else "green"
        t.add_row(g, f"[{colour}]{d:+.1f}[/]")
    con.print(t)
    con.print(f"[bold]mean[/] {reading.mean_delta_pp:+.1f}pp  "
              + ("[green]ACCEPT[/]" if reading.accept else "[red]DO NOT ACCEPT[/]")
              + (f"  regressed: {', '.join(reading.regressed_subgroups)}"
                 if reading.regressed_subgroups else ""))
    doc = {"mean_delta_pp": reading.mean_delta_pp,
           "per_subgroup_delta_pp": dict(reading.per_subgroup_delta_pp),
           "regressed_subgroups": list(reading.regressed_subgroups),
           "accept": reading.accept, "n_instances": len(rows),
           "max_tolerated_subgroup_drop_pp": max_tolerated_subgroup_drop_pp}
    typer.echo(json.dumps({"accept": reading.accept,
                           "mean_delta_pp": reading.mean_delta_pp}))
    dump(doc, out)
    if not reading.accept:
        raise typer.Exit(1)
