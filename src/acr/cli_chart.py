"""Drive ONE agent run at a time: what is in the corpus, what a spec says, and what the agent
did with the two.

Every command here reads charts and most of them call a model. They are the oldest surface in
the tree and the only one whose unit of work is a single patient — the cohort-scale versions
live in `cli_pipeline`, which pays for a run directory and an artifact and is not what you
want when you are debugging one chart.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.table import Table

from . import cli_common
from .cli_common import API_BASE, CORPUS, MODEL, con
from .corpus import Corpus
from .spec import load_spec, load_specs
from .trace import load_trace, plan_summary

chart_app = typer.Typer(add_completion=False)


@chart_app.command("patients")
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


@chart_app.command("chart")
def chart(patient: str, corpus: str = CORPUS):
    """Show one patient's document-type summary — what the agent sees first."""
    ch = Corpus(Path(corpus)).chart(patient)
    t = Table("doc_type", "count", "earliest", "latest")
    for r in ch.type_summary():
        t.add_row(r["doc_type"], str(r["count"]), r["earliest"], r["latest"])
    con.print(f"[bold]{patient}[/] — {len(ch)} documents")
    con.print(t)


@chart_app.command("specs")
def specs_cmd(directory: str = "specs"):
    """List available extraction specs with their freeze hashes."""
    t = Table("spec_id", "version", "hash", "source", "question")
    for s in load_specs(directory).values():
        t.add_row(s.spec_id, s.spec_version, s.spec_hash, s.data_source, s.question[:60])
    con.print(t)


def _skill_stack(runtime_profile: str, skills: str):
    """Resolve the profile's skill assembly and apply the `--skills` override to it.

    Called BEFORE the model client is built and before the loop in `batch`, both deliberately.
    A misspelt slot is a typo, and a typo caught after the first call is a bill for a typo;
    inside `batch`'s per-patient `except` it would be reported once per chart as if ten charts
    had failed, when what failed was one string.
    """
    from .runtime_profiles import resolve_runtime_policy, runtime_policy_skills
    from .skills import parse_skill_stack

    profile_asset, _ = resolve_runtime_policy(runtime_profile)
    return parse_skill_stack(skills, runtime_policy_skills(profile_asset.module_id))


@chart_app.command("run")
def run(
    patient: str = typer.Argument(..., help="patient id"),
    spec: str = typer.Option(..., "--spec", "-s", help="path to a spec YAML"),
    corpus: str = CORPUS,
    model: str = MODEL,
    api_base: str = API_BASE,
    max_steps: int = cli_common.MAX_STEPS,
    max_usd: float = cli_common.MAX_USD,
    out: str = typer.Option("runs", "--out"),
    temperature: float = typer.Option(1.0, "--temperature"),
    seed: int = typer.Option(1234, "--seed",
                             help="validation-sampling seed; fix it to make two runs comparable"),
    runtime_profile: str = typer.Option(
        "current-stratified-coverage",
        "--runtime-profile",
        help=(
            "registered search policy. Experimental three-arm profiles are "
            "guideline-only, conditional-negative-coverage, and always-coverage; "
            "current-stratified-coverage and witness-first-baseline remain as "
            "legacy-compatible profiles"
        ),
    ),
    skills: str = typer.Option(
        "", "--skills",
        help="override the profile's skill assembly: comma-separated slot=value. "
             "`search=search-native` replaces the search policy, `general=+chart-triage` "
             "appends one, `general=chart-triage|thread-chasing` replaces the whole general "
             "list (`|` because comma already separates clauses), `search=` clears the slot. "
             "Validated before any model call."),
    conflict_refine: bool = typer.Option(
        False, "--conflict-refine",
        help="OPTIONAL: run bounded conflict-informed candidates around the same deepagents "
             "runtime. Off by default; the baseline path is unchanged."),
    conflict_candidates: int = typer.Option(
        3, "--conflict-candidates", min=2,
        help="deepagents candidates per optional refinement round"),
    conflict_rounds: int = typer.Option(
        2, "--conflict-rounds", min=1,
        help="maximum optional refinement rounds"),
    conflict_max_usd: float = typer.Option(
        15.0, "--conflict-max-usd", min=0.01,
        help="total priced ceiling across optional candidates; per-run --max-usd still applies"),
):
    """Run the agent for one patient and one spec; optional refinement never replaces it."""
    from .agent import run_patient

    sp = load_spec(spec)
    stack = _skill_stack(runtime_profile, skills)
    c = Corpus(Path(corpus))
    ch = c.chart(patient)
    con.print(f"[bold]{sp.spec_id}[/] v{sp.spec_version} (hash {sp.spec_hash}) "
              f"→ patient {patient} ({len(ch)} docs)")
    run_dir = cli_common.unique_run_dir(out)
    chat = cli_common.chat_model(model, api_base, temperature)
    if not conflict_refine:
        # THE BASELINE PATH. Keep this direct: an optional feature that wraps even its
        # disabled arm has already replaced the runtime in the only sense users can observe.
        show(run_patient(spec=sp, corpus=c, patient_id=patient, out_dir=run_dir,
                         model=chat, max_model_calls=max_steps, seed=seed,
                         max_usd=max_usd, runtime_profile=runtime_profile,
                         skill_stack=stack))
        return

    from .conflict_refinement import run_conflict_refinement

    result = run_conflict_refinement(
        runner=run_patient, candidates_per_round=conflict_candidates,
        max_rounds=conflict_rounds, max_total_usd=conflict_max_usd,
        runner_kwargs={
            # `run_conflict_refinement` forwards this dict verbatim to the same `run_patient`,
            # so the override rides here rather than through a new wrapper parameter. Every
            # candidate must run under the SAME assembly: a refinement round whose arms differ
            # in their search policy is not a re-examination of one question, and its
            # agreement would be read as convergence.
            "spec": sp, "corpus": c, "patient_id": patient, "out_dir": run_dir,
            "model": chat, "max_model_calls": max_steps, "seed": seed,
            "max_usd": max_usd, "runtime_profile": runtime_profile,
            "skill_stack": stack,
        })
    summary = result.to_dict(include_manifests=False)
    path = run_dir / "conflict-refinement.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if result.selected_manifest:
        con.print(f"[green]{result.status}[/]: selected a gate-valid deepagents run")
        show(dict(result.selected_manifest))
    else:
        con.print(f"[bold yellow]{result.status}[/]: {result.reason}")
        con.print("No modal answer was selected; route this case to review.")
    con.print(f"→ {path}")


@chart_app.command("batch")
def batch(
    spec: str = typer.Option(..., "--spec", "-s"),
    corpus: str = CORPUS,
    model: str = MODEL,
    api_base: str = API_BASE,
    patients_arg: str = typer.Option("", "--patients", help="comma list; default all"),
    max_steps: int = cli_common.MAX_STEPS,
    max_usd: float = cli_common.MAX_USD,
    temperature: float = typer.Option(1.0, "--temperature"),
    seed: int = typer.Option(1234, "--seed"),
    runtime_profile: str = typer.Option(
        "current-stratified-coverage", "--runtime-profile"
    ),
    skills: str = typer.Option(
        "", "--skills",
        help="override the profile's skill assembly: comma-separated slot=value. "
             "`search=search-native` replaces the search policy, `general=+chart-triage` "
             "appends one, `general=chart-triage|thread-chasing` replaces the whole general "
             "list (`|` because comma already separates clauses), `search=` clears the slot. "
             "Parsed once before the loop, so a typo cannot be charged per patient."),
    out: str = typer.Option("runs", "--out"),
):
    """Run one spec across many patients.

    `acr extract` is the cohort-scale command and writes the artifact chain; this one is for
    debugging a handful of charts and writes one JSON summary.
    """
    from .agent import run_patient

    sp = load_spec(spec)
    stack = _skill_stack(runtime_profile, skills)
    c = Corpus(Path(corpus))
    pids = [p.strip() for p in patients_arg.split(",") if p.strip()] or c.patient_ids()
    run_dir = cli_common.unique_run_dir(out)
    results = []
    for pid in pids:
        con.print(f"[dim]— {pid}[/]")
        try:
            results.append(run_patient(spec=sp, corpus=c, patient_id=pid, out_dir=run_dir,
                                       model=cli_common.chat_model(model, api_base, temperature),
                                       max_model_calls=max_steps, seed=seed, run_id=pid,
                                       max_usd=max_usd,
                                       runtime_profile=runtime_profile,
                                       skill_stack=stack))
        except Exception as e:  # noqa: BLE001
            con.print(f"[red]{pid} failed: {e}[/]")
            results.append({"patient_id": pid, "error": str(e)})
    summ = run_dir / f"batch-{sp.spec_id}.json"
    summ.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    t = Table("patient", "status", "calls", "revisions", "rejected")
    for r in results:
        a = r.get("answer", {})
        t.add_row(r.get("patient_id", "?"), str(a.get("status", r.get("error", "?"))),
                  str(r.get("n_model_calls", "-")),
                  str((r.get("replan") or {}).get("n_requests", "-")),
                  str(len(r.get("rejections", []))))
    con.print(t)
    con.print(f"→ {summ}")


@chart_app.command("consistency")
def consistency(
    patient: str = typer.Argument(...),
    spec: str = typer.Option(..., "--spec", "-s"),
    n: int = typer.Option(3, "--n", help="independent runs"),
    temperature: float = typer.Option(1.0, "--temperature"),
    corpus: str = CORPUS, model: str = MODEL, api_base: str = API_BASE,
    max_steps: int = cli_common.MAX_STEPS,
    max_usd: float = cli_common.MAX_USD,
    runtime_profile: str = typer.Option(
        "current-stratified-coverage", "--runtime-profile"
    ),
    out: str = typer.Option("runs", "--out"),
):
    """Run the same spec N times to measure SELF-consistency.

    High self-consistency is not validity: a model can settle on one wrong reading and repeat
    it. Report this next to accuracy, never instead of it.
    """
    from .agent import run_patient

    sp = load_spec(spec)
    c = Corpus(Path(corpus))
    run_dir = cli_common.unique_run_dir(out)
    outs = []
    for i in range(n):
        r = run_patient(spec=sp, corpus=c, patient_id=patient, out_dir=run_dir,
                        model=cli_common.chat_model(model, api_base, temperature),
                        max_model_calls=max_steps, seed=1234, run_id=f"{patient}__c{i}",
                        max_usd=max_usd, runtime_profile=runtime_profile)
        outs.append(r)
        con.print(f"  run {i+1}/{n}: {r['answer'].get('status')} "
                  f"{json.dumps(r['answer'].get('value', {}), ensure_ascii=False)}")
    counts = Counter(json.dumps({"status": o["answer"].get("status"),
                                 "value": o["answer"].get("value")},
                                sort_keys=True, ensure_ascii=False) for o in outs)
    top, top_n = counts.most_common(1)[0]
    con.print(f"\n[bold]self-consistency[/]: {top_n}/{n} = {top_n/n:.0%} agreement on the modal answer")
    con.print(f"distinct answers: {len(counts)}")
    for k, v in counts.items():
        con.print(f"  {v}x  {k}")
    con.print("\n[yellow]Self-consistency measures stability, not correctness.[/]")


@chart_app.command("trace")
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


def show(res: dict) -> None:
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
