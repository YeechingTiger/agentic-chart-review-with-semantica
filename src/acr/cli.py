"""Command line interface.

Two families of command live here. The first (`patients`, `chart`, `specs`, `run`, `batch`,
`consistency`, `trace`) drives one agent run at a time. The second (`extract`, `concord`,
`explain`) is the L0-L5 pipeline of the design doc: resolve a request to specs, run the gated
agent per patient x spec, score a guideline over the results, and scaffold why the misses
missed. The three pipeline commands hand each other JSON artifacts and nothing else, so any
stage can be rerun, diffed or audited without rerunning the model.
"""
from __future__ import annotations

import csv
import io
import json
from collections import Counter
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

import subprocess
from datetime import datetime, timezone

from .concordance import (ConcordanceInputError, GuidelineError, Recommendation, assess,
                          load_guideline, summarise, variables_from_answer)
from .corpus import Corpus
from .explain import (DEFAULT_MAX_ELUSION_UPPER, ArtifactBindingError, VariableResult,
                      mark_binding, resolve_bound_extract, scaffold_explanation,
                      side_input_record)
from .graph import ChartReviewAgent
from .intake import ModelClassifier, load_guidelines, route
from .llm import LLMClient, LLMConfig
from .registry_catalog import (VariableCatalog, VariableResolutionError,
                               check_guideline_bindings)
from . import deps as depsmod
from . import specview
from . import speclint
from .spec import load_spec, load_specs
from .state import Budget
from .trace import load_trace, plan_summary

app = typer.Typer(add_completion=False, help="Agentic EHR chart review.")
con = Console()

# ------------------------------------------------------------------------ development plane
# A third family, and it belongs to neither of the two above. `assets` does not run the agent
# and does not read a chart: it develops the retrieval assets a run depends on — the keyword
# lists and the strata — against a complete per-note labelling of a small dev set, and refuses
# to certify one on data the search has seen. Mounted here rather than kept as a private
# entry point because the loop it implements (measure -> propose -> evolve -> certify ->
# adopt) is the only way anything in `specs/` stops being a guess, and a development tool
# nobody can find is a development tool nobody runs.
from .assetdev import assets_app  # noqa: E402  (after `app`, so the group can attach)
# `derive` is the FIRST-ORDER member of that same family, and it comes before `assets` in the
# order anyone should use them: count what the labelling already says, price it by grep, cut
# the list, propose the read policy. `assets` hill-climbs, which only refines a list that
# already exists. Both are mounted because the search is still worth running afterwards; the
# derivation is what makes there be something to refine.
from .derive import derive_app  # noqa: E402

app.add_typer(derive_app, name="derive")
app.add_typer(assets_app, name="assets")

CORPUS = typer.Option("corpus/patients", "--corpus", help="root directory of patient directories")
MODEL = typer.Option(None, "--model", "-m", help="LiteLLM model string, e.g. ollama_chat/qwen3.6:35b")
API_BASE = typer.Option(None, "--api-base", help="override provider base URL (vLLM, proxy, …)")


def _code_sha() -> str:
    """Short git sha, or 'dirty'/'nogit'. A run is only reproducible against the code that
    produced it, so the code identity belongs in the run's name."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        return (sha or "nogit") + ("-dirty" if dirty else "")
    except Exception:
        return "nogit"


def _unique_run_dir(base: str) -> Path:
    """runs/<label>__<utc>__<sha>/ — never reused.

    Reusing a directory name across code versions silently replaces one experiment's record
    with another's, and nothing records that a substitution happened. The same configuration
    under different code is not the same experiment, so the sha is part of the identity.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    d = Path(f"{base}__{stamp}__{_code_sha()}")
    d.mkdir(parents=True, exist_ok=False)
    return d


def _llm(model, api_base, temperature=0.0) -> LLMClient:
    return LLMClient(LLMConfig.from_env(model=model, api_base=api_base, temperature=temperature))


@app.command("patients")
def patients(corpus: str = CORPUS):
    """List patients in the corpus."""
    c = Corpus(Path(corpus))
    t = Table("patient", "documents", "types", "earliest", "latest")
    for pid in c.patient_ids():
        ch = c.chart(pid)
        docs, _ = ch.list_documents(limit=10_000)
        t.add_row(pid, str(len(ch)), str(len(ch.doc_types)),
                  docs[0].date.isoformat() if docs else "-",
                  docs[-1].date.isoformat() if docs else "-")
    con.print(t)


@app.command("chart")
def chart(patient: str, corpus: str = CORPUS):
    """Show one patient's document-type summary — what the agent sees first."""
    ch = Corpus(Path(corpus)).chart(patient)
    t = Table("doc_type", "count", "earliest", "latest")
    for r in ch.type_summary():
        t.add_row(r["doc_type"], str(r["count"]), r["earliest"], r["latest"])
    con.print(f"[bold]{patient}[/] — {len(ch)} documents")
    con.print(t)


@app.command("specs")
def specs_cmd(directory: str = "specs"):
    """List available extraction specs with their freeze hashes."""
    t = Table("spec_id", "version", "hash", "source", "question")
    for s in load_specs(directory).values():
        t.add_row(s.spec_id, s.spec_version, s.spec_hash, s.data_source, s.question[:60])
    con.print(t)


# --------------------------------------------------------------------------- spec review
#: A third family, and it is the only one whose user is not an engineer. Every command above
#: assumes the reader can open a YAML file; these two exist because the person who owns the
#: clinical decisions in that file cannot, and has therefore never seen them.
spec_app = typer.Typer(add_completion=False,
                       help="Put a specification in front of the clinician who owns its decisions.")
app.add_typer(spec_app, name="spec")

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
    # rglob, so `acr spec lint specs` covers specs/ablation too: an arm nobody lints is an arm
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


@app.command("run")
def run(
    patient: str = typer.Argument(..., help="patient id"),
    spec: str = typer.Option(..., "--spec", "-s", help="path to a spec YAML"),
    corpus: str = CORPUS,
    model: str = MODEL,
    api_base: str = API_BASE,
    max_steps: int = typer.Option(24, "--max-steps"),
    reflect_every: int = typer.Option(2, "--reflect-every"),
    out: str = typer.Option("runs", "--out"),
    temperature: float = typer.Option(0.0, "--temperature"),
    seed: int = typer.Option(None, "--seed",
                             help="validation-sampling seed; fix it to make two runs comparable"),
):
    """Run the agent for one patient and one spec."""
    sp = load_spec(spec)
    c = Corpus(Path(corpus))
    ch = c.chart(patient)
    # Corpus-wide type vocabulary: without it, "this patient has none" and "no such type"
    # come back looking identical, and only the first of those is a finding.
    # Cached, name-only scan. The previous form built a PatientChart per patient, which
    # stats every file in the corpus (~276k stats, ~39 min on Lustre) to run one patient.
    vocab = c.doc_type_vocabulary()
    agent = ChartReviewAgent(sp, _llm(model, api_base, temperature),
                             budget=Budget(max_steps=max_steps),
                             reflect_every=reflect_every, out_dir=_unique_run_dir(out),
                             sample_seed=seed)
    con.print(f"[bold]{sp.spec_id}[/] v{sp.spec_version} (hash {sp.spec_hash}) "
              f"→ patient {patient} ({len(ch)} docs, {len(vocab)} types in corpus vocabulary)")
    res = agent.run(ch, known_doc_types=vocab)
    _show(res)


@app.command("batch")
def batch(
    spec: str = typer.Option(..., "--spec", "-s"),
    corpus: str = CORPUS,
    model: str = MODEL,
    api_base: str = API_BASE,
    patients_arg: str = typer.Option("", "--patients", help="comma list; default all"),
    max_steps: int = typer.Option(24, "--max-steps"),
    out: str = typer.Option("runs", "--out"),
):
    """Run one spec across many patients."""
    sp = load_spec(spec)
    c = Corpus(Path(corpus))
    pids = [p.strip() for p in patients_arg.split(",") if p.strip()] or c.patient_ids()
    results = []
    for pid in pids:
        agent = ChartReviewAgent(sp, _llm(model, api_base), budget=Budget(max_steps=max_steps), out_dir=out)
        con.print(f"[dim]— {pid}[/]")
        try:
            results.append(agent.run(c.chart(pid)))
        except Exception as e:  # noqa: BLE001
            con.print(f"[red]{pid} failed: {e}[/]")
            results.append({"patient_id": pid, "error": str(e)})
    Path(out).mkdir(parents=True, exist_ok=True)
    summ = Path(out) / f"batch-{sp.spec_id}.json"
    summ.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    t = Table("patient", "status", "steps", "replans", "rejected", "tokens")
    for r in results:
        a = r.get("answer", {})
        t.add_row(r.get("patient_id", "?"), str(a.get("status", r.get("error", "?"))),
                  str(r.get("steps", "-")), str(r.get("plan_revisions", "-")),
                  str(len(r.get("rejections", []))), str(r.get("usage", {}).get("total_tokens", "-")))
    con.print(t)
    con.print(f"→ {summ}")


@app.command("consistency")
def consistency(
    patient: str = typer.Argument(...),
    spec: str = typer.Option(..., "--spec", "-s"),
    n: int = typer.Option(3, "--n", help="independent runs"),
    temperature: float = typer.Option(0.7, "--temperature"),
    corpus: str = CORPUS, model: str = MODEL, api_base: str = API_BASE,
    out: str = typer.Option("runs", "--out"),
):
    """Run the same spec N times to measure SELF-consistency.

    High self-consistency is not validity: a model can settle on one wrong reading and
    repeat it. Report this next to accuracy, never instead of it.
    """
    sp = load_spec(spec)
    ch = Corpus(Path(corpus)).chart(patient)
    outs = []
    for i in range(n):
        agent = ChartReviewAgent(sp, _llm(model, api_base, temperature), out_dir=out)
        r = agent.run(ch, run_id=None)
        outs.append(r)
        con.print(f"  run {i+1}/{n}: {r['answer'].get('status')} "
                  f"{json.dumps(r['answer'].get('value', {}), ensure_ascii=False)}")
    keys = [json.dumps({"status": o["answer"].get("status"), "value": o["answer"].get("value")},
                       sort_keys=True, ensure_ascii=False) for o in outs]
    counts = Counter(keys)
    top, top_n = counts.most_common(1)[0]
    con.print(f"\n[bold]self-consistency[/]: {top_n}/{n} = {top_n/n:.0%} agreement on the modal answer")
    con.print(f"distinct answers: {len(counts)}")
    for k, v in counts.items():
        con.print(f"  {v}x  {k}")
    con.print("\n[yellow]Self-consistency measures stability, not correctness.[/]")


@app.command("trace")
def trace_cmd(path: str, capg: bool = typer.Option(False, "--capg", help="emit CAPG observation-tree JSON")):
    """Summarise a run trace."""
    evs = load_trace(path)
    if capg:
        obs = [e for e in evs]
        con.print_json(json.dumps({"n_events": len(obs)}))
        return
    t = Table("seq", "t(s)", "kind", "detail")
    for e in evs:
        d = ""
        if e["kind"] == "tool":
            d = f"{e['tool']}({json.dumps(e.get('args', {}), ensure_ascii=False)[:70]}) ok={e.get('ok')}"
        elif e["kind"] == "plan":
            d = f"rev{e.get('revision')} " + plan_summary(e.get("plan"))
        elif e["kind"] == "reflect":
            d = f"{e.get('verdict')} — {e.get('reason','')[:64]}"
        elif e["kind"] == "answer_rejected":
            d = f"REJECTED: {e.get('why','')[:70]}"
        elif e["kind"] == "llm":
            d = f"{e.get('role')} {str(e.get('tool_calls') or '')[:50]}"
        elif e["kind"] == "run_end":
            d = f"status={e.get('status')} steps={e.get('steps')}"
        t.add_row(str(e["seq"]), f"{e['elapsed_s']:.1f}", e["kind"], d)
    con.print(t)


# ============================================================ L0-L5 pipeline: extract
#: Column names a cohort CSV may use for the patient identifier. Checked in this order.
COHORT_ID_COLUMNS = ("patient_id", "patient", "id", "mrn")

EXTRACT_SCHEMA = "acr.extract/1"
CONCORD_SCHEMA = "acr.concord/1"
EXPLAIN_SCHEMA = "acr.explain/1"


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


@app.command("extract")
def extract(
    cohort: str = typer.Option(..., "--cohort", help="csv/tsv/txt/json of patient ids"),
    variables: str = typer.Option(..., "--variables",
                                  help="comma list of variable names, spec ids or STORE items"),
    specs_dir: str = typer.Option("specs", "--specs", help="directory scanned for specs"),
    corpus: str = CORPUS,
    model: str = MODEL,
    api_base: str = API_BASE,
    max_steps: int = typer.Option(24, "--max-steps"),
    reflect_every: int = typer.Option(2, "--reflect-every"),
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
    run_dir = _unique_run_dir(f"{out}/extract")
    specs = {sid: catalog.specs[sid] for sid in res.spec_ids}

    patients: list[dict] = []
    failed = 0
    for pid in pids:
        rec: dict = {"patient_id": pid, "runs": [], "answers": {}, "variables": {}, "errors": []}
        for sid in res.spec_ids:
            sp = specs[sid]
            con.print(f"[dim]— {pid} / {sid}[/]")
            try:
                agent = ChartReviewAgent(sp, _llm(model, api_base, temperature),
                                         budget=Budget(max_steps=max_steps),
                                         reflect_every=reflect_every, out_dir=run_dir,
                                         sample_seed=seed)
                r = agent.run(c.chart(pid), run_id=f"{pid}__{sid}", known_doc_types=vocab)
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
        "code_sha": _code_sha(),
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
def _load_artifact(path: str, schema: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"input not found: {p}")
    doc = json.loads(p.read_text(encoding="utf-8"))
    got = doc.get("schema")
    if got != schema:
        raise typer.BadParameter(f"{p}: expected schema {schema}, got {got!r}")
    return doc


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


@app.command("concord")
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

    doc = _load_artifact(input_path, EXTRACT_SCHEMA)
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
        "code_sha": _code_sha(), "engine": "acr.concordance/deterministic",
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


@app.command("explain")
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
    doc = _load_artifact(input_path, CONCORD_SCHEMA)
    # The extract must BE the one this concord.json was scored from, by content digest. An
    # unbound artifact lets the four-cause verdict be moved by pointing at different inputs;
    # a missing one strips every coverage ledger and manufactures CANNOT_DISTINGUISH.
    try:
        ext, binding = resolve_bound_extract(doc, load=lambda p: _load_artifact(p, EXTRACT_SCHEMA),
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
              "created_utc": datetime.now(timezone.utc).isoformat(), "code_sha": _code_sha(),
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


# ============================================================ L0: intake
ASK_SCHEMA = "acr.routing/1"


@app.command("ask")
def ask(
    question: str = typer.Argument(..., help="a question, a variable name, a spec id, a STORE "
                                             "item, a guideline id or a recommendation id"),
    specs_dir: str = typer.Option("specs", "--specs", help="directory scanned for specs"),
    guidelines_dir: str = typer.Option("guidelines", "--guidelines",
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
        clf = ModelClassifier(_llm(model, api_base, temperature=None))
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
            "code_sha": _code_sha(), "question": question,
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
                  "grammar acr.concordance executes[/]")

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


@app.command("deps")
def deps_cmd(
    guideline: str = typer.Option(..., "--guideline", help="path to a guideline YAML"),
    specs_dir: str = typer.Option("specs", "--specs", help="directory scanned for specs"),
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


def _show(res: dict) -> None:
    a = res.get("answer", {})
    con.print(f"\n[bold]status[/]: {a.get('status')}")
    if a.get("value"):
        con.print(f"[bold]value[/]: {json.dumps(a['value'], ensure_ascii=False)}")
    con.print(f"[bold]reasoning[/]: {a.get('reasoning','')[:600]}")
    gap = a.get("spec_gap")
    if gap:
        # Printed, not buried in the manifest. This is the agent telling a human that the
        # specification is at fault, which is the one output of a run that is worth acting
        # on immediately — and for 38 real runs it could not be produced at all.
        con.print(f"[bold yellow]spec gap[/]: {a.get('remedy_class')} in "
                  f"[bold]{gap.get('spec_section')}[/] "
                  f"(reported by {gap.get('reported_by')}, "
                  f"{'routable' if gap.get('routable') else 'NOT routable'})")
        if gap.get("uncovered_fields"):
            con.print(f"   fields: {', '.join(gap['uncovered_fields'])}")
        if gap.get("invoked_rules"):
            con.print(f"   rules cited: {', '.join(gap['invoked_rules'])}")
        if gap.get("spec_quote"):
            con.print(f"   quoting: “{gap['spec_quote'][:200]}”")
        if a.get("value_withheld"):
            con.print(f"   [red]value withheld[/]: {', '.join(a['value_withheld'])} — "
                      f"{a.get('value_withheld_why','')}")
    po = a.get("proof_obligation", {})
    con.print(f"[bold]proof obligation[/]: satisfied={po.get('satisfied')} "
              f"{('missing=' + '; '.join(po.get('missing', []))) if po.get('missing') else ''}")
    con.print(f"[bold]evidence[/]: {len(a.get('evidence', []))} quote(s)")
    for e in a.get("evidence", [])[:6]:
        con.print(f"   • {e['note_id']} [{e['date']}] {e['start']}-{e['end']} "
                  f"({e.get('stance','supports')}) “{e['quote'].strip()[:110]}”")
    # `plan_revisions` counts entries into the plan node, which is 1 on a run that never
    # replanned. The honest number is how many revisions actually changed the retrieval
    # scope; `replan_rate` is the health metric for the PRIOR — high means the spec's term
    # list and stratum declarations are being repaired at inference time, once per patient.
    rp = res.get("replan") or {}
    con.print(f"[bold]steps[/]: {res.get('steps')}  "
              f"replans: {rp.get('n_revisions_applied', 0)}"
              f"/{rp.get('n_reflections', 0)} reflections"
              + (f" (rate {rp['replan_rate']:.2f})" if rp.get("replan_rate") else "")
              + f"  refused: {rp.get('n_revisions_refused', 0)}  "
              f"rejected answers: {len(res.get('rejections', []))}  "
              f"tokens: {res.get('usage', {}).get('total_tokens')}  {res.get('elapsed_s')}s")
    ot = res.get("open_threads") or {}
    if ot.get("n_opened"):
        con.print(f"[bold]threads[/]: {ot['n_opened']} opened, {ot.get('n_resolved', 0)} "
                  f"resolved, {ot.get('n_dismissed', 0)} dismissed, "
                  + (f"[red]{ot['n_unresolved']} UNRESOLVED[/]" if ot.get("n_unresolved")
                     else "none outstanding"))
    added = (res.get("plan") or {}).get("terms_added") or []
    proms = (res.get("plan") or {}).get("promotions") or []
    if added or proms:
        # Printed because these are the develop-plane candidates, and a candidate nobody
        # sees at the end of a run is a candidate nobody harvests.
        moves = ["{}: {} -> {}".format(r["type"], r["from"], r["to"]) for r in proms]
        con.print(f"[bold]plan widened[/]: +{len(added)} term(s) "
                  f"{[r['term'] for r in added]}, {len(proms)} promotion(s) {moves}")
    con.print(f"[dim]trace: {res.get('trace')}[/]")


if __name__ == "__main__":
    app()
