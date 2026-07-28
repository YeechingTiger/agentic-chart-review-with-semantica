"""Turn a cohort request into the L0-L5 artifact chain: extract, then concord, then explain.

The three commands hand each other JSON artifacts and nothing else, so any stage can be
rerun, diffed or audited without rerunning the model. That is why they belong in one module
and not three: the thing worth keeping honest is the seam between them — a variable that
quietly fails to arrive, a coverage ledger that quietly fails to travel, or an outcome
quietly folded into a denominator all produce a number that looks exactly like a real one.
"""
from __future__ import annotations

import csv
import io
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.table import Table

from . import cli_common
from .cli_common import (API_BASE, CONCORD_SCHEMA, CORPUS, EXPLAIN_SCHEMA, EXTRACT_SCHEMA, MODEL,
                         code_sha, con, load_artifact)
from .concordance import (ConcordanceInputError, GuidelineError, Recommendation, assess,
                          load_guideline, summarise, variables_from_answer)
from .corpus import Corpus
from .explain import (DEFAULT_MAX_ELUSION_UPPER, ArtifactBindingError, VariableResult,
                      mark_binding, resolve_bound_extract, scaffold_explanation,
                      side_input_record)
from .registry_catalog import (VariableCatalog, VariableResolutionError,
                               check_guideline_bindings)

pipeline_app = typer.Typer(add_completion=False)

#: Column names a cohort CSV may use for the patient identifier. Checked in this order.
COHORT_ID_COLUMNS = ("patient_id", "patient", "id", "mrn")


def read_cohort(path: str | Path) -> list[str]:
    """Patient ids from a .csv/.tsv/.txt/.json cohort file, in file order.

    Duplicates are collapsed and REPORTED rather than silently dropped: a cohort file with a
    repeated id changes every denominator computed from it, and a rate whose denominator
    moved without anyone being told is the failure the whole concordance layer is built to
    refuse.
    """
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"cohort file not found: {p}")
    raw = p.read_text(encoding="utf-8")

    ids: list[str] = []
    if p.suffix.lower() == ".json":
        j = json.loads(raw)
        rows = j.get("patients", j) if isinstance(j, dict) else j
        if not isinstance(rows, list):
            raise typer.BadParameter(f"{p}: expected a JSON list of ids or {{'patients': [...]}}")
        for r in rows:
            ids.append(str(r.get("patient_id", r.get("id", "")) if isinstance(r, dict) else r))
    else:
        delim = "\t" if p.suffix.lower() == ".tsv" else ","
        rows = list(csv.reader(io.StringIO(raw), delimiter=delim))
        rows = [r for r in rows if r and str(r[0]).strip()
                and not str(r[0]).lstrip().startswith("#")]
        if not rows:
            raise typer.BadParameter(f"{p}: no rows")
        header = [str(c).strip().lower() for c in rows[0]]
        col = next((header.index(c) for c in COHORT_ID_COLUMNS if c in header), None)
        body = rows[1:] if col is not None else rows
        col = col or 0
        ids = [str(r[col]).strip() for r in body if len(r) > col and str(r[col]).strip()]

    ids = [i for i in ids if i]
    if not ids:
        raise typer.BadParameter(f"{p}: no patient ids found")
    uniq = list(dict.fromkeys(ids))
    if len(uniq) != len(ids):
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        con.print(f"[yellow]cohort: {len(ids) - len(uniq)} duplicate id(s) collapsed: "
                  f"{', '.join(dupes[:10])}[/]")
    return uniq


def _resolve_or_die(specs_dir: str, variables: str):
    """L0. A resolution failure is a request the operator has to fix, so it exits non-zero
    with the vocabulary attached — never a shorter extract than was asked for."""
    try:
        catalog = VariableCatalog.from_directory(specs_dir)
        return catalog, catalog.resolve(variables)
    except VariableResolutionError as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e


def _variable_records(answer: dict, spec_id: str, fields: list[str],
                      gate_validated: bool) -> dict[str, dict]:
    """Flatten one answer into per-variable rows.

    `variables_from_answer` is the single flattening rule in the system — it is the thing
    that knows a populated field is FOUND even when the answer as a whole abstained, and
    that a field the model never mentioned is a silence rather than an established absence.
    Re-deriving that here would be a second copy free to disagree with L4's.
    """
    out: dict[str, dict] = {}
    for name, vv in variables_from_answer(answer, fields, source=spec_id).items():
        out[name] = {"status": vv.status, "value": vv.value,
                     "negative_basis": vv.negative_basis, "source": vv.source,
                     "spec_id": spec_id, "output_field": name,
                     "gate_validated": bool(gate_validated),
                     "proof_basis": answer.get("proof_basis")}
    return out


def _spec_gap_summary(patients: list[dict]) -> dict:
    """Roll the SPEC_INSUFFICIENT reports in one extract up into a countable block.

    Kept separate from `n_failed_runs`: a spec gap is a successful run reporting a real
    finding about the specification, and filing it beside the failures would teach a reader
    to skim past it. `unroutable` is broken out because a report nobody can act on is the
    shape this defect takes when it comes back.
    """
    # Iterating the run rows, not the gap blocks: `remedy_class` sits beside `spec_gap` on
    # the answer rather than inside it, because it is derived from the block and a second
    # copy is a second thing that can disagree.
    rows = [r for rec in patients for r in rec["runs"] if r.get("spec_gap")]
    gaps = [r["spec_gap"] for r in rows]
    return {
        "total": len(rows),
        "agent_reported": sum(1 for g in gaps if g.get("reported_by") == "agent"),
        "runtime_forced": sum(1 for g in gaps if g.get("reported_by") == "runtime"),
        "unroutable": sum(1 for g in gaps if not g.get("routable")),
        "by_section": {s: sum(1 for g in gaps if g.get("spec_section") == s)
                       for s in sorted({str(g.get("spec_section")) for g in gaps})},
        "by_remedy_class": {c: sum(1 for r in rows if r.get("remedy_class") == c)
                            for c in sorted({str(r.get("remedy_class")) for r in rows})},
    }


@pipeline_app.command("extract")
def extract(
    cohort: str = typer.Option(..., "--cohort", help="csv/tsv/txt/json of patient ids"),
    variables: str = typer.Option(..., "--variables",
                                  help="comma list of variable names, spec ids or STORE items"),
    specs_dir: str = typer.Option("specs", "--specs", help="directory scanned for specs"),
    corpus: str = CORPUS,
    model: str = MODEL,
    api_base: str = API_BASE,
    max_steps: int = cli_common.MAX_STEPS,
    temperature: float = typer.Option(0.0, "--temperature"),
    seed: int = typer.Option(None, "--seed", help="validation-sampling seed; share it across arms"),
    limit: int = typer.Option(0, "--limit", help="first N patients only; 0 = all"),
    out: str = typer.Option("runs", "--out"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="resolve, plan and cost the work without calling a model"),
):
    """L0-L3: extract a set of variables across a cohort, one gated run per patient x spec.

    The unit of work is the SPEC, not the variable: `--variables primary_site,histology`
    is one agent pass over the chart, because that spec answers both in one answer. Asking
    per variable would pay for the chart twice and could return two different sites for one
    patient.
    """
    catalog, res = _resolve_or_die(specs_dir, variables)
    pids = read_cohort(cohort)
    if limit:
        pids = pids[:limit]

    t = Table("requested", "variable", "spec_id", "matched on", "source")
    for v in res.variables:
        t.add_row(v.requested, v.name, v.spec_id, v.matched_on,
                  v.data_source if v.data_source == "notes" else f"[yellow]{v.data_source}[/]")
    con.print(t)
    for v in res.not_from_notes():
        # graph.py forces SPEC_INSUFFICIENT / WRONG_DATA_SOURCE for these at finalize. The
        # run still costs a full agent pass to arrive at a constant, so say so before
        # spending it across a cohort rather than after.
        con.print(f"[yellow]{v.name} ({v.spec_id}) has data_source={v.data_source}: every run "
                  f"will return SPEC_INSUFFICIENT / WRONG_DATA_SOURCE by design[/]")
    con.print(f"[bold]{len(pids)} patient(s) x {len(res.spec_ids)} spec(s) = "
              f"{len(pids) * len(res.spec_ids)} agent run(s)[/]")
    if dry_run:
        con.print("[dim]--dry-run: no model was called[/]")
        return

    c = Corpus(Path(corpus))
    vocab = c.doc_type_vocabulary()
    run_dir = cli_common.unique_run_dir(f"{out}/extract")
    specs = {sid: catalog.specs[sid] for sid in res.spec_ids}

    patients: list[dict] = []
    failed = 0
    for pid in pids:
        rec: dict = {"patient_id": pid, "runs": [], "answers": {}, "variables": {}, "errors": []}
        for sid in res.spec_ids:
            sp = specs[sid]
            con.print(f"[dim]— {pid} / {sid}[/]")
            try:
                from .agent import run_patient
                r = run_patient(
                    spec=sp, corpus=c, patient_id=pid, out_dir=run_dir,
                    model=cli_common.chat_model(model, api_base, temperature),
                    max_model_calls=max_steps, seed=seed or 1234,
                    run_id=f"{pid}__{sid}")
            except Exception as e:  # noqa: BLE001
                # One patient failing must not silently shrink the cohort: the row survives,
                # carries the error, and the command exits non-zero at the end.
                con.print(f"[red]{pid} / {sid} failed: {e}[/]")
                rec["errors"].append({"spec_id": sid, "error": f"{type(e).__name__}: {e}"})
                failed += 1
                # A crashed run and a run that never happened used to look identical in the
                # run directory: the trace stopped mid-stream and no manifest was written.
                # That is how a status that crashed 100% of the time got reported as a status
                # the model never chose. The stub is deliberately NOT named `.manifest.json`
                # — it is not an answer and must never be read as one.
                try:
                    (run_dir / f"{pid}__{sid}.failed.json").write_text(
                        json.dumps({"patient_id": pid, "spec_id": sid, "run_id": f"{pid}__{sid}",
                                    "outcome": "RUN_RAISED", "answer": None,
                                    "error": f"{type(e).__name__}: {e}",
                                    "note": ("no answer was produced. This file exists so an "
                                             "absent manifest cannot be mistaken for a run "
                                             "that was never attempted.")},
                                   indent=2, default=str), encoding="utf-8")
                except OSError:
                    pass          # a failed run must not be made worse by a failed write
                continue
            ans = r.get("answer", {})
            rec["answers"][sid] = ans
            rec["runs"].append({
                "spec_id": sid, "run_id": r.get("run_id"), "status": ans.get("status"),
                "gate_validated": r.get("gate_validated"),
                "steps_to_gate_pass": r.get("steps_to_gate_pass"),
                "negative_basis": ans.get("negative_basis"), "proof_basis": ans.get("proof_basis"),
                "remedy_class": ans.get("remedy_class"),
                # The §6b optimizer reads the extract artifact, not the per-run manifests.
                # Null rather than omitted, so "reported no gap" stays distinguishable from
                # "this artifact predates the channel" — the confusion that let a broken
                # channel be written up as a clean result.
                "spec_gap": ans.get("spec_gap"),
                "value_withheld": ans.get("value_withheld"),
                "proof_obligation": ans.get("proof_obligation"),
                "steps": r.get("steps"), "degradation": r.get("degradation"),
                "usage": r.get("usage"), "elapsed_s": r.get("elapsed_s"), "trace": r.get("trace"),
            })
            rec["variables"].update(
                _variable_records(ans, sid, res.fields_for(sid), r.get("gate_validated")))
        patients.append(rec)

    doc = {
        "schema": EXTRACT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha": code_sha(),
        "corpus": str(corpus), "specs_dir": specs_dir, "cohort": str(Path(cohort).resolve()),
        "model": (model or "default"), "sample_seed": seed,
        "resolution": res.to_dict(),
        "specs": {sid: specs[sid].identity() | {"question": specs[sid].question,
                                                "data_source": specs[sid].data_source}
                  for sid in res.spec_ids},
        "patients": patients,
        "n_failed_runs": failed,
        # Counted at the top so nobody has to walk the artifact to discover the channel is
        # silent — and so "zero" is a number someone had to look at, not an absence they
        # never noticed. `agent_reported` is the one that carries information: a
        # runtime-forced gap is a constant this spec returns for every chart.
        "spec_gaps": _spec_gap_summary(patients),
    }
    path = run_dir / "extract.json"
    path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")

    t = Table("patient", *res.names)
    for rec in patients:
        cells = []
        for n in res.names:
            v = rec["variables"].get(n)
            cells.append("[red]-[/]" if v is None else
                         (str(v["value"]) if v["status"] == "FOUND" and v["value"] is not None
                          else f"[yellow]{v['status']}[/]"))
        t.add_row(rec["patient_id"], *cells)
    con.print(t)
    sg = doc["spec_gaps"]
    if sg["agent_reported"]:
        # Surfaced at the end of every extract, because this is the highest-precision input
        # the spec-improvement loop receives and it spent 38 runs being invisible.
        con.print(f"[yellow]{sg['agent_reported']} agent-reported spec gap(s) "
                  f"({sg['runtime_forced']} runtime-forced, {sg['unroutable']} unroutable): "
                  f"{json.dumps(sg['by_section'])}[/]")
    con.print(f"→ {path}")
    if failed:
        con.print(f"[red]{failed} run(s) failed; the extract is incomplete[/]")
        raise typer.Exit(1)


# ============================================================ L4: concordance
def _merge_variables(extracted: dict, extra: dict, prefer: str, pid: str) -> dict:
    """Extraction output plus whatever the guideline needs that notes cannot supply.

    A name present in both is a conflict, not a merge: two sources disagreeing about one
    variable is the two-ledger failure, and picking one silently would make the concordance
    rate depend on dict ordering. The operator states the precedence or fixes the input.
    """
    both = sorted(set(extracted) & set(extra))
    if both and prefer == "error":
        raise ConcordanceInputError(
            f"patient {pid}: {', '.join(both)} arrive from both the extract and the extra "
            f"file. Pass --prefer extract|extra to say which source is authoritative, or "
            f"remove one. Merging silently would make the rate depend on dict order.")
    merged = dict(extracted)
    for k, v in extra.items():
        if k not in merged or prefer == "extra":
            merged[k] = v
    return merged


@pipeline_app.command("concord")
def concord(
    guideline: str = typer.Option(..., "--guideline", help="path to a guideline YAML"),
    input_path: str = typer.Option(..., "--input", "-i", help="an extract.json"),
    extra: str = typer.Option("", "--extra-variables",
                              help="JSON {patient_id: {variable: {status, value}}} for inputs "
                                   "the notes cannot supply (registry feed, treatment tables)"),
    prefer: str = typer.Option("error", "--prefer",
                               help="when a variable is in both sources: error|extract|extra"),
    recommendations: str = typer.Option("", "--recommendations",
                                        help="comma list; default every recommendation"),
    specs_dir: str = typer.Option("specs", "--specs"),
    out: str = typer.Option("", "--out", help="output JSON; default concord.json beside --input"),
):
    """L4: score a guideline over an extract. A rule engine — no model is called."""
    if prefer not in ("error", "extract", "extra"):
        raise typer.BadParameter("--prefer must be error, extract or extra")
    try:
        g = load_guideline(guideline)
    except GuidelineError as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e

    doc = load_artifact(input_path, EXTRACT_SCHEMA)
    extras = json.loads(Path(extra).read_text(encoding="utf-8")) if extra else {}
    wanted = [r.strip() for r in recommendations.split(",") if r.strip()] or None

    # A guideline input bound to a field name no spec produces never arrives, every case
    # comes back NOT_ASSESSABLE naming it, and the denominator quietly goes to zero. Loud,
    # but not fatal: an external feed may legitimately supply what this repo's specs do not.
    binding = check_guideline_bindings(VariableCatalog.from_directory(specs_dir), g)
    for b in binding:
        con.print(f"[yellow]guideline binding: {b}[/]")

    rows: list[dict] = []
    everything = []
    for rec in doc.get("patients", []):
        pid = rec["patient_id"]
        try:
            variables = _merge_variables(rec.get("variables", {}), extras.get(pid, {}), prefer, pid)
        except ConcordanceInputError as e:
            con.print(f"[red]{e}[/]")
            raise typer.Exit(2) from e
        results = assess(variables, g, recommendation_ids=wanted)
        everything += results
        rows.append({"patient_id": pid,
                     "results": [r.to_dict() for r in results],
                     "summary": summarise(results)})

    # Which variables each half of the rule reads. L5 holds exception variables to the
    # exception standard and driving variables to the coverage standard, and the split is
    # derived here from the guideline that was actually used rather than re-derived later.
    # The two lists OVERLAP on purpose: `date_of_definitive_surgery` is read by
    # satisfied_when (the adjuvant window) and by the died-before-the-window exception, and
    # it has to reach both. Assigning a shared variable to exceptions alone would drop a
    # driving variable from the absence set, which is exactly how cause B comes out looking
    # eliminated when the missing thing was the whole question.
    # `Recommendation(...).referenced_inputs` scans applies_when + satisfied_when +
    # exceptions, so instantiating one with only the half we want reads out that half.
    recs = {r.id: {
        "title": r.title,
        "rule_inputs": Recommendation(id=r.id, applies_when=r.applies_when,
                                      satisfied_when=r.satisfied_when).referenced_inputs,
        "exception_inputs": Recommendation(id=r.id, exceptions=r.exceptions).referenced_inputs,
    } for r in g.recommendations if wanted is None or r.id in wanted}

    outdoc = {
        "schema": CONCORD_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "code_sha": code_sha(), "engine": "acr.concordance/deterministic",
        "guideline": {"path": str(Path(guideline).resolve()), "guideline_id": g.guideline_id,
                      "guideline_version": g.guideline_version,
                      "guideline_hash": g.guideline_hash},
        "guideline_binding_warnings": binding,
        "extract_input": str(Path(input_path).resolve()),
        "extract_created_utc": doc.get("created_utc"),
        "extra_variables": str(Path(extra).resolve()) if extra else None,
        "prefer": prefer,
        "recommendations": recs,
        "patients": rows,
        "summary": summarise(everything),
    }
    path = Path(out) if out else Path(input_path).with_name("concord.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(outdoc, indent=2, default=str), encoding="utf-8")

    t = Table("patient", "recommendation", "outcome", "rule", "blocking")
    for row in rows:
        for r in row["results"]:
            colour = {"CONCORDANT": "green", "NON_CONCORDANT": "red"}.get(r["outcome"], "yellow")
            t.add_row(row["patient_id"], r["recommendation_id"],
                      f"[{colour}]{r['outcome']}[/]", r["rule_applied"],
                      ", ".join(r["blocking_inputs"])[:48])
    con.print(t)
    s = outdoc["summary"]
    rate = "n/a (nothing scorable)" if s["concordance_rate"] is None else f"{s['concordance_rate']:.1%}"
    con.print(f"[bold]concordance[/]: {rate} on a denominator of {s['denominator']}  "
              f"excluded {json.dumps(s['denominator_excludes'])}")
    con.print(f"[bold]assessable fraction[/]: {s['assessable_fraction']}")
    if s["blocking_inputs"]:
        con.print(f"[bold]blocked by[/]: {json.dumps(s['blocking_inputs'])}")
    con.print(f"→ {path}")


# ============================================================ L5: explanation
def _variable_result(name: str, use: dict, rec: dict) -> VariableResult:
    """Prefer the full L2 answer; fall back to what the concordance result recorded.

    The fallback matters and must not be quiet about what it is: a variable supplied by an
    external feed has no evidence and no coverage ledger, so B can never be proved for it
    and the scaffold will say so. That is the honest answer, not a degradation.
    """
    sid = (rec.get("variables", {}).get(name) or {}).get("spec_id")
    answer = (rec.get("answers", {}) or {}).get(sid) if sid else None
    if answer:
        return VariableResult.from_answer(name, answer, output_field=name)
    return VariableResult(name=name, status=str(use.get("status") or "NOT_EXTRACTED"),
                          value={name: use.get("value")},
                          negative_basis=use.get("negative_basis"), output_field=name)


def _truth_for(truth: dict, pid: str, names: list[str]) -> dict | None:
    """`{patient: {variable: value}}` or `{patient: {variable: {field: value}}}`.

    Returns None when this patient has no truth row at all, which keeps cause C OPEN — the
    registry covers 20% of the cohort, and eliminating C on the other 80% would turn a
    coverage limitation into a clean bill of health for the extraction layer.
    """
    row = truth.get(pid)
    if row is None:
        return None
    return {n: (row[n] if isinstance(row.get(n), dict) else {n: row[n]})
            for n in names if n in row}


@pipeline_app.command("explain")
def explain(
    input_path: str = typer.Option(..., "--input", "-i", help="a concord.json"),
    only: str = typer.Option("NON_CONCORDANT", "--only", help="comma list of outcomes"),
    extract_path: str = typer.Option("", "--extract",
                                     help="a relocated copy of the extract recorded in --input; "
                                          "verified by content digest, not by filename"),
    allow_unbound_extract: bool = typer.Option(
        False, "--allow-unbound-extract",
        help="run --extract even when its content is NOT the extract this concord.json was "
             "scored from. Every scaffold produced is stamped UNBOUND."),
    truth: str = typer.Option("", "--truth",
                              help="EVAL ONLY: {patient: {variable: value}} registry ground truth"),
    max_elusion_upper: float = typer.Option(DEFAULT_MAX_ELUSION_UPPER, "--max-elusion-upper"),
    out: str = typer.Option("", "--out", help="output JSON; default explain.json beside --input"),
):
    """L5: for each selected case, eliminate the causes the ledger eliminates.

    Produces a scaffold and a case packet for an adjudicator, never a chosen cause. Where the
    coverage proof does not separate a care gap from a documentation gap the verdict is
    CANNOT_DISTINGUISH and `assert_cause_is_earned` refuses either one downstream.
    """
    doc = load_artifact(input_path, CONCORD_SCHEMA)
    # The extract must BE the one this concord.json was scored from, by content digest. An
    # unbound artifact lets the four-cause verdict be moved by pointing at different inputs;
    # a missing one strips every coverage ledger and manufactures CANNOT_DISTINGUISH.
    try:
        ext, binding = resolve_bound_extract(doc, load=lambda p: load_artifact(p, EXTRACT_SCHEMA),
                                             override_path=extract_path,
                                             allow_unbound=allow_unbound_extract)
    except ArtifactBindingError as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e
    if not binding.bound:
        con.print(f"[red]UNBOUND inputs, running anyway on your say-so: "
                  f"{'. '.join(binding.because)}[/]")
    by_patient = {r["patient_id"]: r for r in ext.get("patients", [])}
    truths = json.loads(Path(truth).read_text(encoding="utf-8")) if truth else {}
    truth_record = side_input_record(truth, truths)
    wanted = {o.strip().upper() for o in only.split(",") if o.strip()}
    recs = doc.get("recommendations", {})

    cases: list[dict] = []
    for row in doc.get("patients", []):
        pid = row["patient_id"]
        erec = by_patient.get(pid, {})
        for r in row["results"]:
            if r["outcome"] not in wanted:
                continue
            if r["outcome"] != "NON_CONCORDANT":
                # NOT_ASSESSABLE is a first-class outcome, not a quiet non-concordance;
                # scaffolding one would fold a case into a rate it cannot be scored in.
                cases.append({"case_id": pid, "recommendation_id": r["recommendation_id"],
                              "outcome": r["outcome"], "scaffold": None,
                              "not_explainable": "explain runs only on NON_CONCORDANT; "
                                                 f"{r['outcome']} is an outcome in its own right"})
                continue
            spl = recs.get(r["recommendation_id"], {}) or {}
            rule_names = set(spl.get("rule_inputs", []))
            exc_names = set(spl.get("exception_inputs", []))
            uses = {u["variable"]: u for u in r["inputs_used"]}
            # A variable read by both halves belongs in both lists — see the comment where
            # `rule_inputs` is written. Membership, not exclusion.
            driving = [_variable_result(n, u, erec) for n, u in uses.items() if n in rule_names]
            exceptions = [_variable_result(n, u, erec) for n, u in uses.items() if n in exc_names]
            try:
                sc = scaffold_explanation(
                    case_id=pid, recommendation_id=r["recommendation_id"],
                    concordance=r["outcome"], driving_variables=driving,
                    exception_results=exceptions,
                    registry_truth=_truth_for(truths, pid, list(uses)),
                    max_elusion_upper=max_elusion_upper)
            except ValueError as e:
                con.print(f"[red]{pid} / {r['recommendation_id']}: {e}[/]")
                raise typer.Exit(2) from e
            cases.append({"case_id": pid, "recommendation_id": r["recommendation_id"],
                          "outcome": r["outcome"], "scaffold": sc.to_dict()})
            con.print(sc.render())

    outdoc = {"schema": EXPLAIN_SCHEMA,
              "created_utc": datetime.now(timezone.utc).isoformat(), "code_sha": code_sha(),
              "concord_input": str(Path(input_path).resolve()),
              "extract_input": binding.used_path,
              "registry_truth_supplied": bool(truth), "registry_truth": truth_record,
              "only": sorted(wanted), "max_elusion_upper": max_elusion_upper,
              "n_cases": len(cases), "cases": cases,
              "verdicts": dict(Counter(c["scaffold"]["verdict"] for c in cases if c["scaffold"]))}
    mark_binding(outdoc, binding)
    path = Path(out) if out else Path(input_path).with_name("explain.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(outdoc, indent=2, default=str), encoding="utf-8")
    if not cases:
        con.print(f"[green]no case matched --only {sorted(wanted)}[/]")
    con.print(f"[bold]verdicts[/]: {json.dumps(outdoc['verdicts'])}")
    con.print("[dim]" + "no ground truth exists for the 'why' layer; every scaffold is an "
              "input to human adjudication, not a finding[/]")
    con.print(f"→ {path}")
