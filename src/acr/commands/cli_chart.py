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

from ..chartstore.corpus import Corpus
from ..contract.spec import load_spec, load_specs
from ..contract.trace import load_trace, plan_summary
from ..core import cli_common, site
from ..core.cli_common import API_BASE, CORPUS, MODEL, con

chart_app = typer.Typer(add_completion=False)


#: The flag every run command takes, described once.
SITE_MAPPING = typer.Option(
    "", "--mapping",
    help="Site Mapping JSON from `acr site-mapping build`. Required when the spec has any "
         "`means:` stratum; ignored otherwise.")


def _require_mapping_for(strata, mapping_path: str) -> str | None:
    """The message to refuse with, or None when this spec needs no mapping.

    REFUSED AT THE DOOR, not inside the ledger. `StratumSpec.matches` already refuses correctly,
    but its message ends "pass it to assign_strata" — a function name, not something an operator
    can supply. This says `--mapping`, and it fires before the corpus is opened or a model reached.
    """
    mapped = [st.name for st in strata if getattr(st, "is_mapped", False)]
    if not mapped or mapping_path:
        return None
    return (f"this contract selects documents through a Site Mapping "
            f"(stratum {', '.join(sorted(mapped))}) and no --mapping was given. With no mapping "
            f"every document falls to the `rest` stratum, the gate counts an empty stratum as a "
            f"satisfied one, and the run would report a coverage proof it never performed. Build "
            f"one with `acr site-mapping build --out mapping.json` and pass `--mapping "
            f"mapping.json`.")


def _load_site_mapping(spec, mapping_path: str):
    """Resolve `--mapping` against a spec, refusing rather than silently stratifying by `rest`."""
    from ..contract.site_mapping import SiteMapping, SiteMappingError
    from ..contract.strata import strata_from_spec

    problem = _require_mapping_for(strata_from_spec(spec), mapping_path)
    if problem:
        raise typer.BadParameter(problem)
    if not mapping_path:
        return None
    path = Path(mapping_path).expanduser()
    if not path.is_file():
        raise typer.BadParameter(f"no Site Mapping at {path}")
    try:
        return SiteMapping.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (SiteMappingError, json.JSONDecodeError, KeyError) as e:
        raise typer.BadParameter(f"{path}: {e}") from e


#: The measured-prior flag, described once.
PRIOR = typer.Option(
    "", "--prior",
    help="a retrieval prior from `acr assets prior`: which document types carried the answer on "
         "OTHER patients and which terms surfaced them. Rendered into the prompt as REFERENCE, "
         "never as a rule, and recorded in the manifest so two arms can be told apart. Leaving it "
         "off is the baseline arm and produces a byte-identical prompt.")


PLANNER = typer.Option(
    "", "--planner",
    help="where the run STARTS looking: `spec-strata` (the contract's hand-written strata and "
         "declared keywords — a supplied prior) or `patient-inventory` (every type this patient has, "
         "no keywords — the absence of one). Empty means whatever --runtime-profile chooses, which "
         "is what every run recorded before 2026-08-04 used. Split out because the profile moved "
         "this AND whether coverage is enforced, so no arm could attribute a result to either.")


def _planner(value: str) -> str:
    """Validated here, before any model call, so a typo costs nothing.

    A bad `--planner` reaching `run_patient` would raise mid-run on the first patient of a batch,
    after the run directory exists and a trace has been opened — the shape of failure `preflight` was
    added to `tools/run_ladder.py` to end.
    """
    from ..review.agent import PLANNERS
    if value and value not in PLANNERS:
        raise typer.BadParameter(f"unknown planner {value!r}; one of {list(PLANNERS)}, "
                                 f"or leave it empty to take the runtime profile's choice")
    return value


def _load_prior(path: str):
    """A prior, or None. Refuses a path that does not load rather than running without one.

    An unreadable prior must not degrade to the baseline silently: the manifest would then say
    `retrieval_prior: null` for a run the operator believes was informed, and the arm would be
    mislabelled in every later comparison.
    """
    if not path:
        return None
    from ..contract.retrieval_prior import RetrievalPrior, RetrievalPriorError
    try:
        return RetrievalPrior.load(path)
    except RetrievalPriorError as e:
        raise typer.BadParameter(str(e)) from e


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
def specs_cmd(directory: str = str(site.specs_root())):
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
    from ..contract.skills import parse_skill_stack
    from ..review.runtime_profiles import resolve_runtime_policy, runtime_policy_skills

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
    site_mapping: str = SITE_MAPPING,
    prior: str = PRIOR,
    planner: str = PLANNER,
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
             "`policy=policy-information-gain` replaces the policy, "
             "`tactics=+tactic-counterevidence` and `general=+chart-triage` append one, "
             "`experience=experience-adapter` turns the develop-set prior on, "
             "`general=chart-triage|thread-chasing` replaces a whole list (`|` because "
             "comma already separates clauses), `policy=` clears the slot. "
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
    from ..review.agent import run_patient

    sp = load_spec(spec)
    mapping = _load_site_mapping(sp, site_mapping)
    prior_asset = _load_prior(prior)
    pl = _planner(planner)
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
                         skill_stack=stack, site_mapping=mapping,
                         retrieval_prior=prior_asset, planner=pl))
        return

    from ..review.conflict_refinement import run_conflict_refinement

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
            "skill_stack": stack, "site_mapping": mapping,
            "retrieval_prior": prior_asset,
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
             "`policy=policy-information-gain` replaces the policy, "
             "`tactics=+tactic-counterevidence` and `general=+chart-triage` append one, "
             "`experience=experience-adapter` turns the develop-set prior on, "
             "`general=chart-triage|thread-chasing` replaces a whole list (`|` because "
             "comma already separates clauses), `policy=` clears the slot. "
             "Parsed once before the loop, so a typo cannot be charged per patient."),
    out: str = typer.Option("runs", "--out"),
    site_mapping: str = SITE_MAPPING,
    prior: str = PRIOR,
    planner: str = PLANNER,
):
    """Run one spec across many patients.

    `acr extract` is the cohort-scale command and writes the artifact chain; this one is for
    debugging a handful of charts and writes one JSON summary.
    """
    from ..review.agent import run_patient

    sp = load_spec(spec)
    mapping = _load_site_mapping(sp, site_mapping)
    prior_asset = _load_prior(prior)
    pl = _planner(planner)
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
                                       skill_stack=stack, site_mapping=mapping,
                                       retrieval_prior=prior_asset, planner=pl))
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
    skills: str = typer.Option(
        "", "--skills",
        help="override the profile's skill assembly: comma-separated slot=value, "
             "same grammar as `acr chart run --skills`. Validated before any model call."),
    seed: int = typer.Option(
        1234, "--seed",
        help="validation-sampling seed, SHARED by all N runs. Self-consistency is about the "
             "model, so the runtime's own sampling must not vary between them"),
    out: str = typer.Option("runs", "--out"),
    site_mapping: str = SITE_MAPPING,
    prior: str = PRIOR,
    planner: str = PLANNER,
):
    """Run the same spec N times to measure SELF-consistency.

    High self-consistency is not validity: a model can settle on one wrong reading and repeat
    it. Report this next to accuracy, never instead of it.
    """
    from ..review.agent import run_patient

    sp = load_spec(spec)
    mapping = _load_site_mapping(sp, site_mapping)
    prior_asset = _load_prior(prior)
    pl = _planner(planner)
    c = Corpus(Path(corpus))
    stack = _skill_stack(runtime_profile, skills)
    run_dir = cli_common.unique_run_dir(out)
    outs = []
    for i in range(n):
        r = run_patient(spec=sp, corpus=c, patient_id=patient, out_dir=run_dir,
                        model=cli_common.chat_model(model, api_base, temperature),
                        max_model_calls=max_steps, seed=seed, run_id=f"{patient}__c{i}",
                        max_usd=max_usd, runtime_profile=runtime_profile,
                        skill_stack=stack, site_mapping=mapping,
                        retrieval_prior=prior_asset, planner=pl)
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


@chart_app.command("check-corpus")
def cmd_check_corpus(
    corpus: str = CORPUS,
    patients: str = typer.Option("", "--patients", help="comma list; default every patient"),
    strict: bool = typer.Option(False, "--strict",
                                help="exit non-zero when any filename is unreadable"),
):
    """Report every document the loader cannot see, BEFORE anything is spent reading the ones it can.

    `corpus.FILENAME_RE` takes `<Doc-Type>_<YYYY-MM-DD>[__<n>].txt`: the type and the date come from
    the NAME, and every date filter and every type sweep works off them. A stem it cannot parse is
    skipped — correctly, because guessing a date is worse than missing one — but skipped SILENTLY,
    and a silently missing document is the most expensive kind: the run still answers, still passes
    its gate, and still reports coverage over the documents it did see.

    On the synthetic corpus this finds nothing. On a real export it is the first command to run, and
    `--strict` is what belongs in a pipeline: a corpus that half-loaded is not a corpus.
    """
    c = Corpus(Path(corpus))
    ids = [p.strip() for p in patients.split(",") if p.strip()] or c.patient_ids()
    bad: dict[str, list[str]] = {}
    empty: list[str] = []
    n_ok = 0
    for pid in ids:
        chart = c.chart(pid)
        n_ok += len(chart)
        if chart.unreadable_filenames:
            bad[pid] = chart.unreadable_filenames
        if len(chart) == 0:
            # A zero-document patient is not "no findings" — it is a subject every run will answer
            # CORPUS_INSUFFICIENT about while the report reads clean. Listed by name, because an
            # aggregate hides exactly the patient someone is looking for.
            empty.append(pid)
    total_bad = sum(len(v) for v in bad.values())
    con.print(f"{len(ids)} patient(s), [bold]{n_ok}[/] document(s) the loader can see, "
              f"[bold]{total_bad}[/] it cannot, [bold]{len(empty)}[/] patient(s) with no "
              f"readable document at all")
    if bad:
        t = Table("patient", "unreadable", "examples")
        for pid, names in sorted(bad.items()):
            t.add_row(pid, str(len(names)), ", ".join(names[:2]))
        con.print(t)
        con.print("[yellow]Expected `<Doc-Type>_<YYYY-MM-DD>[__<n>].txt`. Rename them, or accept "
                  "that no run will ever read them.[/]")
    if empty:
        con.print(f"[yellow]No readable documents: {', '.join(sorted(empty))}[/]")
    if strict and (total_bad or empty):
        raise typer.Exit(code=1)
