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
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.table import Table

from ..chartstore.corpus import Corpus
from ..contract.spec import load_spec
from ..core.cli_common import CORPUS, con, dump, read_json
from ..core.local_artifacts import (
    LOCAL_ROOT_ENV,
    RUN_RECORD_GLOB,
    LocalArtifactError,
    LocalArtifactStore,
    require_run_tree,
)
from ..evaluation import evals as E
from ..improvement import refine as R

LOCAL_ROOT = typer.Option(
    None, "--local-root", envvar=LOCAL_ROOT_ENV,
    help="absolute patient-artifact root outside Git; where the case map lives")

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


# --------------------------------------------------------------- the missing `--cases` producer

#: Returned when the key says NOTHING about this run and field. Distinct from `""`, which is the
#: key ASSERTING that abstaining is correct — the distinction `evals.score` calls `n_unkeyed`.
UNKEYED = object()


def _key_value(answer_key: Mapping, patient: str, field: str, spec_id: str):
    """The answer key's value for one field, or `UNKEYED`.

    FOUR FACTS WERE COLLAPSED INTO `""`: a patient the key has no row for, a row that is not a
    mapping, a field absent from the row, and a genuine `None` meaning "abstaining is correct". Only
    the last of those is `""`. The other three made every run on an uncovered patient a failure
    case, printed under "where abstaining was correct and the run answered anyway" — a claim the key
    never made. Measured: a STORE.390 key against five real STORE.400 manifests produced six such
    phantom cases.

    This is the same defect as `n_unkeyed` being pinned at 0 in `evals.score`, reintroduced in a
    producer with no `n_unkeyed` concept — including the `spec_id` half, which `evals._key_row`
    checks and this ignored. `evals._key_row` is the one implementation; this delegates to it.
    """
    row = E._key_row(answer_key, f"{patient}__{spec_id}",
                     _SpecScopedRun(patient_id=patient, spec_id=spec_id))
    if row is None:
        return UNKEYED
    fields = row.get("fields")
    if not isinstance(fields, Mapping) or field not in fields:
        return UNKEYED
    v = fields[field]
    return "" if v is None else str(v)


@dataclass(frozen=True)
class _SpecScopedRun:
    """The two attributes `evals._key_row` reads. Reusing the real lookup means the two scorers
    cannot drift on "does the key speak about this run" — which they already had, in one changeset."""

    patient_id: str
    spec_id: str


def _coded_value(record: "E.RunRecord", field: str) -> str:
    """What this run coded for one field, in the KEY's own convention: `""` means it abstained.

    THE STATUS STRING IS NOT A VALUE. The first version returned `answer["status"]` when no value
    was present, so a run that abstained — `CORPUS_INSUFFICIENT` — compared unequal to the key's
    empty string and a CORRECT abstention was emitted as a failure case. `eval score` called the
    same manifest ABSTAINED_CORRECT: two scorers, one run, opposite verdicts, and this one put a
    wrong count (8, actually 7) into the docs. Whether a run abstained is `RunRecord.abstained`'s
    fact — `status_kind` first, the literal set for older manifests — and this module does not get
    a second opinion.
    """
    if record.abstained:
        return ""
    v = record.value.get(field)
    return "" if v in (None, "") else str(v)


def _notations(value: str) -> list[str]:
    """Every way this corpus might have written the key value.

    MEASURED, not assumed. `chart.search('20230412')` returns zero notes on SYN0001 while
    `chart.search('2023-04-12')` returns two: the corpus matcher is notation-tolerant about
    separators inside a token but does not reformat a date. A producer that searched only the key's
    own notation would find the carrying document in no chart, report `surfaced: False` for every
    date failure, and cut 1 would classify the entire cohort as a retrieval failure — which is the
    exact opposite of what this corpus measures (0 of 11 wrong answers were retrieval failures).
    """
    if not value:
        return []
    if value.isdigit() and len(value) < 8:
        # A SHORT NUMERIC VALUE IS NOT A SEARCHABLE STRING. `STORE.400_522_523`'s `behavior` field
        # has allowable values 0/1/2/3, and on SYN0001 (321 documents) `'0'` matches 321 notes,
        # `'2'` matches 321 and `'3'` matches 172 — every digit appears in a grade, a date, a dose.
        # Falling through to `[value]` made `key_value_in_corpus` a meaningless True, `surfaced`
        # True for any run that opened any note, and cut 1 structurally unable to route a `behavior`
        # failure to RETRIEVAL_FAILURE. The same inversion the 99-date guard prevents, one field
        # over. Retrieval cannot be shown to have missed a bare digit, so nothing is searched for.
        return []
    if len(value) == 8 and value.isdigit():
        y, mo, d = value[:4], value[4:6], value[6:]
        if mo == "99" or d == "99":
            # A constructed partial date is written in no document BY CONSTRUCTION, so there is
            # nothing to search for. NOT `[y]`: a bare year appears in essentially every note of a
            # chart from that year (SYNY02: 27 of them), so returning it flipped
            # `key_value_in_corpus` to True and cut 1 routed the case to §6c retrieval — handing
            # the search team a document that does not exist. `tools/measure_controller_value.py`
            # returns False for the same fact, and the two implementations must agree.
            return []
        return [f"{y}-{mo}-{d}", f"{mo}/{d}/{y}", f"{int(mo)}/{int(d)}/{y}", value]
    return [value]


def _read_note_ids(record: "E.RunRecord") -> set[str]:
    """Which notes were put in front of the agent, from the trace's read calls.

    THE TRACE COMES FROM `RunRecord`, which reads the sibling `.jsonl` beside the manifest — the
    move-safe lookup. The first version re-implemented this from the manifest's RECORDED absolute
    path and hard-refused when it did not resolve; 49 of the 509 manifests in this tree are in
    exactly that state (`tools/archive_runs.sh` moves run directories), every one with its sibling
    present. Two implementations of "where is this run's trace" is how that shipped.

    A MISSING TRACE REFUSES. Returning an empty set would mean "nothing was read", which is the
    value that sends a case to `RETRIEVAL_FAILURE` with no model consulted — so an unreadable trace
    would silently attribute a whole cohort to search. "We do not know" and "it read nothing" are
    different facts and only one of them is a finding.
    """
    # `record.trace == []` is AMBIGUOUS: `RunRecord.from_manifest` returns an empty list both for
    # a missing sibling file and for a trace that recorded zero events. Only the first refuses —
    # the second is a run that genuinely read nothing, and "it read nothing" is a finding. The
    # sibling convention is RunRecord's own (manifest name with `.jsonl`).
    src = Path(record.source)
    sibling = src.with_name(src.name.replace(".manifest.json", ".jsonl"))
    # `sibling != src` is `RunRecord.from_manifest`'s own guard: for a path that does not end in
    # `.manifest.json` the replace is a no-op and the MANIFEST would be treated as its own trace.
    if not record.trace and not (sibling != src and sibling.is_file()):
        raise R.RefineError(
            f"no trace beside {record.source or 'this manifest'}: "
            f"`establishing_evidence_surfaced` is computed from the read calls, and an absent "
            f"trace would report False — which routes the case to a retrieval failure on no "
            f"evidence at all.")
    out: set[str] = set()
    for step in record.tool_calls(E.READ_TOOLS):
        args = step.get("args") or {}
        if args.get("note_id"):
            out.add(str(args["note_id"]))
        for v in (args.get("note_ids") or []):
            out.add(str(v))
    return out


def build_failure_cases(*, manifests, answer_key: Mapping, fields, spec_id: str, corpus,
                        case_map: Mapping[str, str],
                        adjudications: Mapping[str, str]) -> list[dict]:
    """Assemble `refine route`'s `--cases` from run records, an answer key and a case map.

    THE PRODUCER THAT DID NOT SHIP. `--cases` is required, `FailureCase` refuses to hold a real
    person_id, and nothing wrote the file — so `refine` had never routed a real failure. This
    carries FACTS only. `establishing_evidence_surfaced` comes from the trace crossed with the
    corpus; `answer_key_adjudication` comes from a human or is `NOT_ADJUDICATED`. Both are
    deliberately not model judgements: `FailureCase` says letting the reflector decide either
    "would let it route its own gradient", and cut 1 acts on the first before any model is asked.

    THE ABSTENTION CASE HAS A NON-OBVIOUS RIGHT ANSWER. When the key value is empty — abstaining
    was correct and the run answered anyway — no document establishes it, and `False` would send a
    misreading to `RETRIEVAL_FAILURE` where retrieval cannot be the cause. So it is `True`: nothing
    was missing from what the agent saw.
    """
    patient_of_case = dict(case_map)
    case_of_patient = {p: c for c, p in patient_of_case.items()}
    out: list[dict] = []
    n_unkeyed = [0]
    for m in manifests:
        record = E.RunRecord.from_manifest(m)
        patient = record.patient_id
        if record.spec_id != spec_id:
            continue
        case_id = case_of_patient.get(patient)
        if not case_id:
            raise R.RefineError(
                f"{patient!r} is absent from the case map. A failure case may not carry a real "
                f"person_id, and the map is the only thing that pseudonymises one — mint it with "
                f"`acr attribute case-map`.")
        read = None
        for field in fields:
            key_value = _key_value(answer_key, patient, field, spec_id)
            if key_value is UNKEYED:
                # The key says nothing about this run and field, so there is no disagreement to
                # report. Counted, not silently dropped — see `n_unkeyed` in `cmd_cases`.
                n_unkeyed[0] += 1
                continue
            coded = _coded_value(record, field)
            if coded == key_value:
                continue
            if read is None:
                # Only a DISAGREEMENT needs the trace, so a cohort whose agreeing runs lost their
                # traces still builds — the refusal is reserved for the case it would corrupt.
                read = _read_note_ids(record)
            carrying: set[str] = set()
            if key_value:
                chart = corpus.chart(patient)
                for form in _notations(key_value):
                    carrying |= {h.note_id
                                 for h in chart.search(form, False, None, None, None,
                                                       max_hits=100000)}
            # THREE POPULATIONS, and only the middle one is a retrieval failure.
            #   key_value empty          — abstaining was correct; no document establishes it
            #   no document carries it   — the key is CONSTRUCTED (imputed date, inferred across
            #                              notes). `measure_controller_value.py` calls this
            #                              UNSEEDABLE and keeps it out of NEVER_LOOKED for the same
            #                              reason: retrieval cannot surface what does not exist.
            #   carried but never read   — the genuine retrieval failure, and the only one §6c owns
            # Collapsing the first two into `False` hands §6c cases no search could ever fix, and
            # buries the real ones among them. Found on real runs: SYNK01 and SYNK02 are
            # constructed, SYNX02 and SYNX06 are genuine, and the first version reported all four
            # identically.
            in_corpus = bool(carrying)
            surfaced = True if not in_corpus else bool(carrying & read)
            out.append({
                "case_id": case_id,
                "spec_id": spec_id,
                "field": field,
                "coded_value": coded,
                "key_value": key_value,
                "establishing_evidence_surfaced": surfaced,
                "answer_key_adjudication": str(
                    adjudications.get(case_id) or R.NOT_ADJUDICATED),
                # Not a `FailureCase` field, and deliberately reported anyway: it is what separates
                # "the agent saw everything" from "there was nothing to see", which the boolean
                # above cannot express and a reader of the routing will otherwise assume.
                "key_value_in_corpus": in_corpus,
            })
    build_failure_cases.n_unkeyed = n_unkeyed[0]   # read by `cmd_cases` for the printed summary
    return out


@refine_app.command("cases")
def cmd_cases(
    runs: str = typer.Option(..., "--runs", help="run record or directory"),
    answer_key: str = typer.Option(..., "--answer-key"),
    fields: str = typer.Option(..., "--fields", help="comma list, as `eval score --fields`"),
    spec: str = typer.Option(..., "--spec", "-s"),
    corpus: str = CORPUS,
    case_map: str = typer.Option(..., "--case-map",
                                 help="from `acr attribute case-map`; the pseudonymiser"),
    adjudications: str = typer.Option("", "--adjudications",
                                      help="JSON {case_id: ADJUDICATED_KEY_CORRECT|_WRONG}"),
    out: str = typer.Option("", "--out"),
    local_root: str | None = LOCAL_ROOT,
):
    """Assemble the `--cases` file `route` requires. Free: no model, and the chart is only searched.

    WHAT COMES OUT AND WHAT IT COSTS YOU LATER. Every case lands with
    `answer_key_adjudication: NOT_ADJUDICATED` unless `--adjudications` says otherwise, and cut 2
    routes an un-adjudicated case to UNRESOLVED without ever consulting the reflector. That is the
    designed behaviour, not a stub: whether the answer key is right is a human's call, and the
    router refuses to launder it into a spec edit. So a first pass reports mostly UNRESOLVED, and
    the work it names is adjudication — which is also what `meta_evaluate_attributions` has been
    waiting on for thirty cases while two exist.
    """
    store = LocalArtifactStore(local_root) if local_root else None
    try:
        root = require_run_tree(runs, what="runs")
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc
    manifests = [root] if root.is_file() else sorted(root.rglob(RUN_RECORD_GLOB))
    key = read_json(answer_key, "answer key")
    cmap = read_json(case_map, "case map") if not store else json.loads(
        store.require_input(case_map, what="case map").read_text(encoding="utf-8"))
    adj = read_json(adjudications, "adjudications") if adjudications else {}
    spec_id = load_spec(spec).spec_id

    try:
        cases = build_failure_cases(
            manifests=manifests, answer_key=key,
            fields=[f.strip() for f in fields.split(",") if f.strip()],
            spec_id=spec_id, corpus=Corpus(Path(corpus)), case_map=cmap, adjudications=adj)
    except R.RefineError as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e

    # Printed as THREE numbers, because the boolean the router reads collapses two of them and a
    # reader would otherwise take "surfaced" to mean the agent saw the answer.
    n_retrieval = sum(1 for c in cases if not c["establishing_evidence_surfaced"])
    n_unseeded = sum(1 for c in cases
                     if c["key_value"] and not c["key_value_in_corpus"])
    n_abstain = sum(1 for c in cases if not c["key_value"])
    n_unkeyed = getattr(build_failure_cases, "n_unkeyed", 0)
    con.print(f"{len(manifests)} run(s) -> [bold]{len(cases)}[/] disagreement(s) with the key")
    if n_unkeyed:
        # NAMED, not dropped. A run the key says nothing about used to be emitted as a failure with
        # `key_value: ""`, which printed as "abstaining was correct" — a claim the key never made.
        con.print(f"  [yellow]{n_unkeyed} run/field pair(s) the answer key says nothing about, "
                  f"skipped. If that is most of the cohort, the key and the runs are about "
                  f"different contracts.[/]")
    con.print(f"  [bold]{n_retrieval}[/] retrieval failure(s): a document carries the key value "
              f"and the run never opened it")
    con.print(f"  [bold]{n_unseeded}[/] unseedable: the key value appears in NO document, so it is "
              f"constructed and no search could have found it")
    con.print(f"  [bold]{n_abstain}[/] where abstaining was correct and the run answered anyway")
    con.print(f"  [bold]{len(cases) - n_retrieval - n_unseeded - n_abstain}[/] read the "
              f"establishing document and got the reading wrong")
    n_adj = sum(1 for c in cases if c["answer_key_adjudication"] != R.NOT_ADJUDICATED)
    if cases and not n_adj:
        con.print("[yellow]None is adjudicated, so `route` will return UNRESOLVED for every case "
                  "that is not a retrieval failure. That is cut 2 refusing to guess whether the "
                  "answer key is right — supply --adjudications to move past it.[/]")
    if out:
        dump(cases, out)
        con.print(f"→ {out}")
    else:
        con.print_json(data=cases)
