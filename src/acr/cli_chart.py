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
from .graph import ChartReviewAgent
from .spec import load_spec, load_specs
from .state import Budget
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


@chart_app.command("run")
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
    agent = ChartReviewAgent(sp, cli_common.llm_client(model, api_base, temperature),
                             budget=Budget(max_steps=max_steps),
                             reflect_every=reflect_every,
                             out_dir=cli_common.unique_run_dir(out), sample_seed=seed)
    con.print(f"[bold]{sp.spec_id}[/] v{sp.spec_version} (hash {sp.spec_hash}) "
              f"→ patient {patient} ({len(ch)} docs, {len(vocab)} types in corpus vocabulary)")
    res = agent.run(ch, known_doc_types=vocab)
    show(res)


@chart_app.command("batch")
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
        agent = ChartReviewAgent(sp, cli_common.llm_client(model, api_base),
                                 budget=Budget(max_steps=max_steps), out_dir=out)
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


@chart_app.command("consistency")
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
        agent = ChartReviewAgent(sp, cli_common.llm_client(model, api_base, temperature),
                                 out_dir=out)
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
