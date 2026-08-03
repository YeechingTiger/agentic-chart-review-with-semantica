"""Invoke the full scan: one cheap reading of every note of a development set against one
spec's requirement, under a ceiling the caller states.

WHY THIS FILE EXISTS AT ALL
---------------------------
`acr.improvement.labelling` is 1292 lines with a test file beside it and, until this module, no way to
run it. A development tool nobody can invoke is a development tool nobody runs, and the
keyword lists and read policies in `assets/specs/` stay guesses for exactly as long as that is true.

NOTHING HERE RUNS BY DEFAULT AND NOTHING SPENDS BY DEFAULT
----------------------------------------------------------
`scan` is the only command in this group that can cost money and it cannot be invoked without
saying how much. `--max-usd` has NO DEFAULT, exactly as `ScanConfig.max_usd` has none: a
ceiling you can forget to set is not a ceiling, and what it guards here is a five-figure call
count against a per-token price. `--max-terms-per-note` and `--min-term-chars` have no default
for the same reason `TermConfig` refuses one — a cap typed in by whoever wrote the line is a
decision about how much a model may pad, made in a commit nobody will reread.

`--dry-run` resolves the requirement, opens the store, counts what is already done and prices
the input side WITHOUT constructing a client. It is not a mode of the scan; it is the scan's
plan, and it is the only way to see the scope before paying for it.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import typer
from rich.table import Table

from ..chartstore.corpus import Corpus
from ..contract.spec import load_spec
from ..core.cli_common import CORPUS, con, dump
from ..improvement import labelling as lab

label_app = typer.Typer(add_completion=False, help=(
    "THE FULL SCAN — read every note of a dev set once, against one spec's requirement. "
    "`scan` spends money per note and refuses to start without --max-usd."))

_SPEC = typer.Option(..., "--spec", "-s", help="the spec whose question the notes are read against")
#: The two halves of `TermConfig`, and neither may acquire a default here. They are part of the
#: prompt hash, so they also key the store: a scan run under different bounds must not append to
#: a file whose manifest names the first pair.
_MAXTERMS = typer.Option(..., "--max-terms-per-note",
                         help="cap on question 2's list. No default: the cap is what makes "
                              "question 2 a ranking task rather than a word count.")
_MINCHARS = typer.Option(..., "--min-term-chars",
                         help="shortest acceptable term. No default: 'a' verifies as present "
                              "in every note ever written.")
_ROOT = typer.Option(None, "--labels-root",
                     help=f"where labellings live; refuses anywhere inside the repository. "
                          f"Default ${lab.LABELS_ROOT_ENV} or {lab.DEFAULT_LABELS_ROOT}")


def _requirement(spec_path: str) -> lab.Requirement:
    """A spec -> the requirement, turning the module's refusals into an exit code.

    `NotALabellableSpecError` is the interesting one: a spec with no `evidence_rules` does not
    say what would establish its answer, and this group will not supply a default clinical rule
    on its behalf. That refusal has to reach the shell intact, not as a traceback.
    """
    try:
        return lab.Requirement.from_spec(load_spec(spec_path))
    except lab.LabellingError as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e


def _store(root, spec_path: str, terms: lab.TermConfig, model: str):
    req = _requirement(spec_path)
    try:
        return req, lab.LabelStore(root, model=model, requirement=req, terms=terms)
    except lab.LabellingError as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e


@label_app.command("requirement")
def requirement_cmd(spec: str = _SPEC,
                    out: str = typer.Option("", "--out", help="write the requirement JSON here")):
    """Print the requirement exactly as the reading model will be shown it. Free, no model.

    The cheapest check there is on a scan that is about to cost hundreds of dollars: every
    line the model will see comes from the spec, so reading it here is reading the experiment.
    """
    req = _requirement(spec)
    con.print(f"[bold]{req.spec_id}[/]  requirement hash [bold]{req.hash}[/]  "
              f"{len(req.fields)} field(s)")
    typer.echo(req.render())
    dump({"spec_id": req.spec_id, "requirement_hash": req.hash,
          "fields": list(req.field_names), "prompt_version": lab.PROMPT_VERSION,
          "rendered": req.render()}, out)


@label_app.command("scan")
def scan(
    spec: str = _SPEC,
    patients: str = typer.Option("", "--patients",
                                 help="comma list of patient ids; default every patient in "
                                      "--corpus"),
    corpus: str = CORPUS,
    max_usd: float = typer.Option(..., "--max-usd",
                                  help="REQUIRED, no default. The scan stops between batches "
                                       "once the label file's own spend reaches this."),
    max_terms_per_note: int = _MAXTERMS,
    min_term_chars: int = _MINCHARS,
    labels_root: str = _ROOT,
    concurrency: int = typer.Option(8, "--concurrency"),
    max_note_chars: int = typer.Option(24_000, "--max-note-chars",
                                       help="notes longer than this are truncated and the label "
                                            "says so"),
    azure_env: str = typer.Option(lab.AZURE_ENV_PATH, "--azure-env",
                                  help="credentials file, READ as data and never sourced"),
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="plan and price the scan without building a client or "
                                      "calling anything"),
    out: str = typer.Option("", "--out", help="write the scan report JSON here"),
):
    """Read every unlabelled note of these patients once. COSTS MONEY PER NOTE.

    Resume is the file format: a note already in the JSONL is done, error labels included, so
    rerunning after the ceiling trips picks up exactly where it stopped and re-spends nothing.
    """
    terms = lab.TermConfig(max_terms_per_note=max_terms_per_note, min_term_chars=min_term_chars)
    # The model name is part of the store key, so it must be settled BEFORE the store is
    # opened — including in --dry-run, or the dry run would plan against a different file
    # from the one the real run appends to.
    model = f"openai/{lab.DEPLOYMENT}"
    req, store = _store(labels_root, spec, terms, model)
    cfg = lab.ScanConfig(max_usd=max_usd, requirement=req, terms=terms,
                         concurrency=concurrency, max_note_chars=max_note_chars)

    c = Corpus(Path(corpus))
    pids = [p.strip() for p in patients.split(",") if p.strip()] or c.patient_ids()
    # A runner with no client: `scope` and `pending` read the corpus index and the label file
    # and touch nothing else, so the plan is produced by the same code that will do the work
    # rather than by a second estimate free to disagree with it.
    runner = lab.FullScanRunner(corpus=c, store=store, config=cfg, client=None)
    items = runner.scope(pids)
    todo = runner.pending(items)
    spent = store.spend()
    plan = _price(todo, cfg, store)
    plan |= {"spec_id": req.spec_id, "requirement_hash": req.hash, "run_key": store.key,
             "store_dir": str(store.dir), "model": model, "n_patients": len(pids),
             "n_notes_in_scope": len(items), "n_already_labelled": len(items) - len(todo),
             "n_pending": len(todo), "spend_so_far_usd": round(spent, 6), "max_usd": max_usd}
    _print_plan(plan)

    if dry_run:
        con.print("[dim]--dry-run: no client was built and nothing was called[/]")
        dump(plan | {"dry_run": True}, out)
        return
    if spent >= max_usd:
        con.print(f"[red]${spent:.4f} is already on disk for this labelling and the ceiling is "
                  f"${max_usd:.4f}. Raise --max-usd to continue; nothing was called.[/]")
        raise typer.Exit(2)

    client = lab.azure_client(azure_env)
    report = lab.FullScanRunner(corpus=c, store=store, config=cfg, client=client).run(pids)
    con.print(f"[bold]wrote[/] {report.n_written} label(s), {report.n_errors} error(s), "
              f"spend ${report.spend_usd:.4f}")
    if report.aborted:
        con.print(f"[yellow]ABORTED: {report.abort_reason}[/]")
    dump(plan | {"dry_run": False, "report": report.__dict__}, out)
    if report.aborted:
        raise typer.Exit(1)


def _price(todo, cfg: lab.ScanConfig, store: lab.LabelStore) -> dict:
    """A FLOOR on the input side, and the output side named as unpriced.

    `DocMeta.n_chars` is known without opening the note, so the prompt's note payload can be
    bounded before a call is made. What cannot be bounded is the completion, and this module
    will not multiply a guess by a note count and print the product as "the cost" — an
    estimate that reads as the price is how a ceiling gets set below the floor. The empirical
    mean is offered instead whenever this store already holds a priced label, because that
    number was measured on this requirement and this deployment.
    """
    chars = sum(min(m.n_chars, cfg.max_note_chars) for _, m in todo)
    # ~4 characters per token is the standard rough figure and is stated as rough. It bounds
    # the note payload only: the system prompt and the rendered requirement ride on every
    # call too, so this is a floor on a floor.
    floor = lab.cost_usd(chars // 4, 0)
    priced = [ll.cost_usd for ll in store.load().values() if ll.cost_usd > 0]
    mean = round(sum(priced) / len(priced), 6) if priced else None
    return {"n_note_chars_pending": chars,
            "input_cost_floor_usd": round(floor, 6),
            "measured_mean_usd_per_label": mean,
            "projected_usd_at_measured_mean": (round(mean * len(todo), 4) if mean else None),
            "output_cost": "UNPRICED — completion length is not knowable before the call",
            "usd_per_1m": [lab.USD_PER_1M_INPUT, lab.USD_PER_1M_OUTPUT]}


def _print_plan(plan: dict) -> None:
    t = Table("what", "value")
    for k in ("spec_id", "requirement_hash", "run_key", "model", "n_patients",
              "n_notes_in_scope", "n_already_labelled", "n_pending", "spend_so_far_usd",
              "max_usd", "input_cost_floor_usd", "measured_mean_usd_per_label",
              "projected_usd_at_measured_mean"):
        t.add_row(k, str(plan.get(k)))
    con.print(t)
    con.print(f"[dim]{plan['output_cost']}. The input figure is a FLOOR, not a quote; the "
              f"ceiling is what actually stops the run.[/]")
    con.print(f"[dim]labels: {plan['store_dir']}[/]")


@label_app.command("progress")
def progress(
    spec: str = _SPEC,
    max_terms_per_note: int = _MAXTERMS,
    min_term_chars: int = _MINCHARS,
    labels_root: str = _ROOT,
    out: str = typer.Option("", "--out", help="write the summary JSON here"),
):
    """Count what one labelling holds: spend, error rate, standing per field. No PHI leaves.

    Deliberately emits COUNTS and never a row. A label carries a person_id, a note date and a
    verbatim quote of the note, and `tests/test_no_phi_in_tree.py` exists because that
    material got into the tree once already — so this command must stay safe to redirect into
    a file inside the repository.
    """
    terms = lab.TermConfig(max_terms_per_note=max_terms_per_note, min_term_chars=min_term_chars)
    req, store = _store(labels_root, spec, terms, f"openai/{lab.DEPLOYMENT}")
    try:
        labels = list(store.load().values())
    except lab.LabelShapeError as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e

    ok = [ll for ll in labels if ll.ok]
    per_field = {f: dict(Counter(ll.admissibility.verdicts.get(f, "") for ll in ok))
                 for f in req.field_names}
    doc = {"spec_id": req.spec_id, "requirement_hash": req.hash, "run_key": store.key,
           "store_dir": str(store.dir), "n_labels": len(labels), "n_ok": len(ok),
           "n_errors": len(labels) - len(ok), "spend_usd": round(store.spend(), 6),
           # SPLIT, because collapsing them libels a working scan. A label with no quote is a note
           # the model read and correctly found nothing quotable in — on a 321-note scan of three
           # charts, 306 of them. A label whose quote is NOT IN THE TEXT is the model composing, and
           # there was exactly one. Reported as one number, the first run of this printed "305
           # label(s) carry a quote that is not in the note text", which reads as a 95% fabrication
           # rate and would make an operator throw away a scan that worked.
           "n_no_quote": sum(1 for ll in ok if not (ll.admissibility.quote or "").strip()),
           "n_quote_unverified": sum(
               1 for ll in ok
               if (ll.admissibility.quote or "").strip() and not ll.admissibility.quote_verified),
           "n_terms_kept": sum(len(ll.retrieval_terms) for ll in ok),
           "n_terms_proposed": sum(ll.n_terms_proposed for ll in ok),
           "n_terms_hallucinated": sum(ll.n_terms_hallucinated for ll in ok),
           "standing_per_field": per_field,
           "by_doc_type": dict(Counter(ll.doc_type for ll in ok))}
    t = Table("field", *lab.ADMISSIBILITY_VERDICTS)
    for f, counts in per_field.items():
        t.add_row(f, *[str(counts.get(v, 0)) for v in lab.ADMISSIBILITY_VERDICTS])
    con.print(f"[bold]{doc['n_labels']}[/] label(s), {doc['n_errors']} errored, "
              f"${doc['spend_usd']:.4f} spent")
    con.print(t)
    if ok and doc["n_quote_unverified"]:
        # A paraphrased span means the model composed rather than reported, and the verdicts
        # it composed for are what every later number counts.
        con.print(f"[yellow]{doc['n_quote_unverified']} label(s) quote text that is NOT in the "
                  f"note — the model composed rather than reported[/]")
    if ok and doc["n_no_quote"]:
        # Not a warning. Most notes establish nothing, and a model that quotes nothing from them is
        # behaving. It is printed because a silent zero here and a silent 306 look the same.
        con.print(f"[dim]{doc['n_no_quote']} label(s) carry no quote at all, which is what a note "
                  f"that establishes nothing should produce[/]")
    typer.echo(json.dumps({k: doc[k] for k in ("n_labels", "n_ok", "n_errors", "spend_usd")}))
    dump(doc, out)


@label_app.command("export")
def cmd_export(
    spec: str = _SPEC, max_terms_per_note: int = _MAXTERMS, min_term_chars: int = _MINCHARS,
    labels_root: str = _ROOT,
    out: str = typer.Option(..., "--out", help="write the assetdev-shaped labelling here"),
):
    """Re-shape a completed scan into the labelling `acr assets` reads.

    THE TWO ENDS OF THIS PLANE DISAGREED ABOUT A FILE FORMAT, and nothing said so until somebody
    ran the chain end to end. `label scan` writes `labels.jsonl` — one object per line, admissibility
    nested, terms as `{term, reason}` pairs. `improvement/assetdev.Labelling.load` reads a SINGLE
    JSON object with `{model, prompt_hash, spec_hash, notes: [...], indexed_vocabulary}` and notes
    shaped `{patient_id, note_id, doc_type, establishes, mentions, terms}`. Feeding one to the other
    fails on `JSONDecodeError: Extra data: line 2`.

    `assetdev.py:129` says of its own copy of the schema: "Redeclared here rather than imported from
    `acr.improvement.labelling` so the two modules are coupled by a FILE FORMAT, not a class." The
    intent is right — a format is a better seam than a shared class. But a format nobody converts to
    and no test round-trips is two formats, and the comment reads as though the coupling had been
    arranged.

    So this is the conversion, in one place, with `tests/test_labelling.py` round-tripping it. The
    two modules stay coupled by a format; this is the thing that makes the sentence true.
    """
    terms = lab.TermConfig(max_terms_per_note=max_terms_per_note, min_term_chars=min_term_chars)
    req, store = _store(labels_root, spec, terms, f"openai/{lab.DEPLOYMENT}")
    labels = [ll for ll in store.load().values() if ll.ok]
    if not labels:
        con.print("[red]no completed labels in this store — run `acr label scan` first[/]")
        raise typer.Exit(code=2)

    notes = []
    for ll in labels:
        verdicts = ll.admissibility.verdicts or {}
        notes.append({
            "patient_id": ll.patient_id, "note_id": ll.note_id, "doc_type": ll.doc_type,
            "establishes": sorted(f for f, v in verdicts.items() if v == "can_establish"),
            "mentions": sorted(f for f, v in verdicts.items() if v == "merely_mentions"),
            "terms": sorted({t.term for t in ll.retrieval_terms}),
        })
    doc = {
        "schema": "acr.labelling/1",
        "model": labels[0].model,
        "prompt_hash": req.hash,
        "spec_hash": _requirement(spec).spec_hash if hasattr(_requirement(spec), "spec_hash") else req.hash,
        "notes": notes,
        # WHAT THE SCAN ACTUALLY INDEXED, not every term that happens to appear. `assetdev` scores a
        # needle outside this set as an error rather than as a zero, because a term the scan never
        # looked for scoring zero is a lie about the term rather than a fact about it.
        "indexed_vocabulary": sorted({t for n in notes for t in n["terms"]}),
    }
    dump(doc, out)
    con.print(f"{len(notes)} note(s), {len(doc['indexed_vocabulary'])} indexed term(s) -> {out}")
