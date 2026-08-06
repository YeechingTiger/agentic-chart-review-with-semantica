"""Invoke the agent-as-a-judge plane with its fence intact: every command that could reach a
model asks `acr.evaluation.evals`' precedence registry first, through `acr.evaluation.judge` itself.

THE FENCE HAS TO SURVIVE THE CLI, AND THE ONLY WAY TO GUARANTEE THAT IS NOT TO RE-IMPLEMENT IT
----------------------------------------------------------------------------------------------
`judge()` refuses a dimension a deterministic evaluator already decides, and there is no flag,
policy value or keyword that turns that off. A CLI that pre-screened the dimension itself —
even correctly, even by reading the same registry — would be a SECOND COPY OF THE JUDGEMENT,
free to drift the first time somebody adds a row. So `panel` never inspects the registry. It
calls `judge()` for real, with a model that raises the moment it is asked anything
(`_PlanOnly`), and lets every refusal in that function fire in its own words. A dry run and a
real run therefore pass through byte-identical precedence, allowlist, packet-type, key-leak
and model-id checks; the only difference is whether a provider is on the other end.

NOTHING RUNS BY DEFAULT AND NOTHING SPENDS BY DEFAULT
------------------------------------------------------
`panel` and `run` each cost money per verdict and neither has a default ceiling: `--max-usd`
is required on both, `--max-calls` and all three cost-class prices are required on `run`
because `JudgeLedger` refuses to be built without them, and `--dry-run` produces the plan and
the price with no client constructed. The budget is checked BEFORE the model is reached — a
limit enforced after the spend is a report.

A judged number is an OPINION. Every artifact written here carries evidence_class=JUDGED and
validation_status=NOT_VALIDATED, and `apply_verdict` is the only door out.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import typer
from rich.table import Table

from ..core import cli_common
from ..core.cli_common import API_BASE, MODEL, con, dump, read_json
from ..evaluation import evals
from ..evaluation import judge as J

judge_app = typer.Typer(add_completion=False, help=(
    "Agent-as-a-judge, fenced in. Judging is refused wherever a deterministic evaluator "
    "exists; `acr eval dimensions` prints the fence. Costs money per verdict."))

_MAXUSD = typer.Option(..., "--max-usd",
                       help="REQUIRED, no default. Judging is refused if the plan would "
                            "exceed it, before any call is made.")


class _PlanOnly:
    """A JudgeModel that satisfies the protocol and refuses to answer.

    It exists so the pre-flight can be `judge()` itself rather than a copy of `judge()`'s
    rules. `model_id` is real because `judge()` requires one and an empty id is its own
    refusal — passing a blank here would make every dry run fail for the wrong reason.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.asked = 0

    def ask(self, prompt: str) -> Mapping[str, Any]:
        self.asked += 1
        raise _Planned(prompt)


class _Planned(Exception):
    """Raised by `_PlanOnly` at the first question: every check upstream of it has passed."""


class JsonJudgeModel:
    """The real seam: an `acr.core.llm` client behind `JudgeModel`.

    The judge prompt asks for JSON and nothing else, and a reply that is not JSON is NOT
    turned into a zero here — `_read_lens` treats a missing score as None, which keeps "the
    judge failed on this case" visible instead of sorting a fabricated zero to the front of a
    human's reading queue.

    Public because `cli_signal` builds the same adapter for --kind judge; a second JSON-mode
    adapter would be a second place for the parsing rules to drift.
    """

    def __init__(self, client, model_id: str):
        self._client, self.model_id = client, model_id
        self.n_calls = 0

    def ask(self, prompt: str) -> Mapping[str, Any]:
        self.n_calls += 1
        resp = self._client.chat([{"role": "user", "content": prompt}])
        try:
            out = json.loads(getattr(resp, "content", "") or "")
        except json.JSONDecodeError:
            return {}
        return out if isinstance(out, Mapping) else {}


def _packet(path: str, keyed: bool):
    """Build the packet from a JSON file, letting `blind_packet`'s key scan do the refusing.

    `AnswerKeyLeak` is caught HERE and not only around the `judge()` call: `blind_packet`
    scans on construction, so a key smuggled in as an artifact refuses one line before the
    fence would have looked at it. Uncaught, that refusal reached the shell as a traceback
    and exit 1 — indistinguishable, to a script, from the command crashing.
    """
    raw = read_json(path, "packet")
    if not isinstance(raw, dict):
        raise typer.BadParameter(f"{path}: expected an object with trace/artifacts/subject_id")
    trace = raw.get("trace") or []
    arts = raw.get("artifacts") or {}
    sid = str(raw.get("subject_id") or "")
    try:
        if keyed:
            return J.keyed_packet(trace, arts, raw.get("answer_key") or {}, sid)
        return J.blind_packet(trace, arts, sid)
    except J.JudgeRefusal as e:
        con.print(f"[red]{type(e).__name__}: {e}[/]")
        raise typer.Exit(2) from e


@judge_app.command("dimensions")
def dimensions(out: str = typer.Option("", "--out", help="write the seam report JSON here")):
    """Print what the judge advertises AND what the registry says about each. Free, no model.

    THE SEAM. Every dimension this module advertised once raised `UnknownDimension` against
    the registry: two halves each correct alone, nothing checking between them. Exits 1 if
    that reopens, so a CI job can hold the seam the same way the test does.
    """
    problems = evals.unknown_dimensions(J.JUDGEABLE_DIMENSIONS)
    # `no_wrap` with a min width on the identifier column, and never rich's default
    # ellipsis. A dimension name is exactly what a reader greps for, and rich silently chops the
    # longest of them on a narrow console — `acr deps` documents the same defect, where
    # a 41-character variable name came out split in half.
    t = Table()
    t.add_column("dimension", min_width=30, no_wrap=True)
    t.add_column("packet")
    t.add_column("registry says")
    for d in J.JUDGEABLE_DIMENSIONS:
        blinded = d in J.KEY_BLINDED_DIMENSIONS
        t.add_row(d, "BLIND" if blinded else "keyed permitted",
                  f"[red]{problems[d][:70]}[/]" if d in problems else "[green]judgeable[/]")
    con.print(t)
    con.print(f"[dim]permitted uses: {', '.join(J.PERMITTED_USES)}; forbidden: "
              f"{', '.join(J.FORBIDDEN_USES)}[/]")
    dump({"advertised": list(J.JUDGEABLE_DIMENSIONS),
          "key_blinded": list(J.KEY_BLINDED_DIMENSIONS),
          "unknown_to_registry": problems,
          "permitted_uses": list(J.PERMITTED_USES)}, out)
    if problems:
        con.print("[red]the judge advertises dimension(s) the precedence registry will not "
                  "accept; every call on them would refuse[/]")
        raise typer.Exit(1)


@judge_app.command("evaluators")
def evaluators(
    directory: str = typer.Option("evaluators", "--dir", help="directory of evaluator YAMLs"),
    out: str = typer.Option("", "--out", help="write the load report JSON here"),
):
    """Load every `assets/evaluators/*.yaml` against the REAL precedence gate. Free, no model.

    `load_evaluators` refuses the whole directory on one bad file rather than skipping it: a
    load that silently drops the evaluator with the failing case and keeps the three that
    pass everything is the exact defect enforcement 4 exists to catch. Exits 2 on refusal.
    """
    try:
        specs = J.load_evaluators(directory, registry=evals.precedence_gate())
    except J.JudgeRefusal as e:
        con.print(f"[red]REFUSED: {e}[/]")
        raise typer.Exit(2) from e
    # `no_wrap` with a min width on the identifier column, and never rich's default
    # ellipsis. A evaluator id is exactly what a reader greps for, and rich silently chops the
    # longest of them on a narrow console — `acr deps` documents the same defect, where
    # a 41-character variable name came out split in half.
    t = Table()
    t.add_column("evaluator_id", min_width=35, no_wrap=True)
    for col in ("dimension", "cost_class", "context", "tools", "must_pass", "must_fail"):
        t.add_column(col)
    rows = []
    for s in specs.values():
        t.add_row(s.evaluator_id, s.dimension, s.cost_class, ",".join(s.context),
                  ",".join(s.tool_names) or "—", str(len(s.must_pass)), str(len(s.must_fail)))
        rows.append({"evaluator_id": s.evaluator_id, "dimension": s.dimension,
                     "cost_class": s.cost_class, "context": list(s.context),
                     "tools": list(s.tool_names), "must_pass": list(s.must_pass),
                     "must_fail": list(s.must_fail), "source": s.source})
    con.print(t)
    con.print(f"[dim]{len(rows)} evaluator(s) loaded; each declares a case it MUST FAIL, "
              f"because an evaluator that scores everything the same looks exactly like a "
              f"clean system. `certify_evaluator` runs them.[/]")
    dump({"directory": directory, "evaluators": rows}, out)


@judge_app.command("panel")
def panel(
    dimension: str = typer.Option(..., "--dimension",
                                  help="one of the judgeable dimensions; anything a "
                                       "deterministic evaluator decides is REFUSED"),
    packet: str = typer.Option(..., "--packet",
                               help="JSON {trace: [...], artifacts: {...}, subject_id: str}"),
    max_usd: float = _MAXUSD,
    usd_per_call: float = typer.Option(..., "--usd-per-call",
                                       help="REQUIRED, no default. One call per lens; an "
                                            "unpriced call reads as free."),
    keyed: bool = typer.Option(False, "--keyed",
                               help="send a KeyedPacket. Refused for every blinded dimension "
                                    "— that refusal is judge()'s, not this command's."),
    model: str = MODEL,
    api_base: str = API_BASE,
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="run every refusal and price the panel without calling"),
    use: str = typer.Option("SCREEN", "--use",
                            help=f"what the verdict is for: {', '.join(J.PERMITTED_USES)}"),
    out: str = typer.Option("", "--out", help="write the verdict JSON here"),
):
    """Ask the built-in lens panel about one case, or refuse. COSTS MONEY PER LENS.

    The refusals you can hit here are `judge()`'s own and are reported verbatim: a dimension
    code already decides, a dimension with no ground truth by construction that is not on the
    allowlist, an answer key reaching a blinded judge, a model that cannot name itself.
    """
    pk = _packet(packet, keyed)
    model_id = model or "planned"
    # THE PRE-FLIGHT IS judge() ITSELF. Everything upstream of the first `model.ask` runs
    # exactly as it will in the real call; `_Planned` is the marker that it all passed.
    probe = _PlanOnly(model_id)
    try:
        J.judge(dimension, pk, registry=evals.precedence_gate(), model=probe)
    except _Planned:
        pass
    except J.JudgeRefusal as e:
        con.print(f"[red]{type(e).__name__}: {e}[/]")
        raise typer.Exit(2) from e
    except (evals.UnknownDimension, ValueError, TypeError) as e:
        con.print(f"[red]{type(e).__name__}: {e}[/]")
        raise typer.Exit(2) from e

    # `J._norm`, not a local `.strip().lower()`: the folding rule is what keeps a padded or
    # miscased name from missing the registry, and a second copy of it here is a second place
    # for it to drift away from the one `judge()` actually applied a moment ago.
    n_lenses = len(J.LENSES[J._norm(dimension)])
    planned = round(n_lenses * usd_per_call, 6)
    plan = {"dimension": J._norm(dimension), "subject_id": pk.subject_id,
            "n_lenses": n_lenses, "usd_per_call": usd_per_call,
            "planned_usd": planned, "max_usd": max_usd,
            "packet": "KEYED" if keyed else "BLIND", "judge_model": model_id}
    con.print(f"[bold]{plan['dimension']}[/] — {n_lenses} lens(es) x ${usd_per_call} = "
              f"${planned} against a ceiling of ${max_usd}")

    if dry_run:
        con.print("[dim]--dry-run: the fence, the allowlist and the key scan all ran; no "
                  "client was built and nothing was called[/]")
        dump(plan | {"dry_run": True}, out)
        return
    if planned > max_usd:
        con.print(f"[red]${planned} of judging would exceed the declared ceiling of "
                  f"${max_usd}. Nothing was called.[/]")
        raise typer.Exit(2)
    if not model:
        raise typer.BadParameter("--model is required for a real run; only --dry-run may go "
                                 "without one")

    real = JsonJudgeModel(cli_common.llm_client(model, api_base), model)
    verdict = J.judge(dimension, pk, registry=evals.precedence_gate(), model=real)
    try:
        applied = J.apply_verdict(verdict, use)
    except J.JudgeCannotGate as e:
        con.print(f"[red]{e}[/]")
        raise typer.Exit(2) from e
    con.print(f"[bold]score[/]: {verdict.score}  incomplete={verdict.incomplete}  "
              f"spent ${round(real.n_calls * usd_per_call, 6)}")
    for r in verdict.lens_readings:
        con.print(f"  {r.lens}: {r.score}  {r.observation[:90]}")
    con.print(f"[yellow]{verdict.notice}[/]")
    dump(plan | {"dry_run": False, "applied": applied}, out)


@judge_app.command("run")
def run_evaluator_cmd(
    evaluator: str = typer.Option(..., "--evaluator", help="evaluator_id, as in assets/evaluators/"),
    directory: str = typer.Option("evaluators", "--dir"),
    context: str = typer.Option(..., "--context",
                                help="JSON of everything the harness can supply; exactly the "
                                     "declared variables are injected and no others"),
    subject_id: str = typer.Option(..., "--subject-id", help="the case being judged"),
    max_usd: float = _MAXUSD,
    max_calls: int = typer.Option(..., "--max-calls",
                                  help="REQUIRED, no default. JudgeLedger refuses to exist "
                                       "without it."),
    price_trace_only: float = typer.Option(..., "--price-trace-only"),
    price_reads_documents: float = typer.Option(..., "--price-reads-documents"),
    price_reruns_searches: float = typer.Option(..., "--price-reruns-searches"),
    model: str = MODEL,
    api_base: str = API_BASE,
    dry_run: bool = typer.Option(False, "--dry-run",
                                 help="load, fence-check, build the context and price it "
                                      "without calling"),
    out: str = typer.Option("", "--out", help="write the verdict and ledger JSON here"),
):
    """Run ONE declared evaluator over one case. COSTS MONEY PER VERDICT.

    Every cost class must be priced, with no default, because an unpriced class reads as free
    and the class that gets forgotten is `reads_documents` — the one whose judges open charts.
    """
    gate = evals.precedence_gate()
    try:
        specs = J.load_evaluators(directory, registry=gate)
    except J.JudgeRefusal as e:
        con.print(f"[red]REFUSED: {e}[/]")
        raise typer.Exit(2) from e
    if evaluator not in specs:
        raise typer.BadParameter(f"no evaluator {evaluator!r} in {directory}; "
                                 f"have {sorted(specs)}")
    spec = specs[evaluator]
    available = read_json(context, "context")
    if not isinstance(available, dict):
        raise typer.BadParameter(f"{context}: expected an object of context variables")

    try:
        ledger = J.JudgeLedger(max_calls=max_calls, max_cost_usd=max_usd,
                               cost_per_call_usd={J.COST_TRACE_ONLY: price_trace_only,
                                                  J.COST_READS_DOCUMENTS: price_reads_documents,
                                                  J.COST_RERUNS_SEARCHES: price_reruns_searches})
        # `build_context` is enforcement 2 and it refuses a declared variable the harness
        # cannot supply, rather than injecting an empty section a judge would answer anyway.
        # Running it in the plan is the point: that refusal must cost nothing to discover.
        injected = J.build_context(spec, available)
    except (J.JudgeRefusal, ValueError, KeyError) as e:
        con.print(f"[red]{type(e).__name__}: {e}[/]")
        raise typer.Exit(2) from e

    price = ledger.price[spec.cost_class]
    plan = {"evaluator_id": spec.evaluator_id, "dimension": spec.dimension,
            "cost_class": spec.cost_class, "subject_id": subject_id,
            "context_injected": sorted(injected), "n_model_calls": 1,
            "planned_usd": price, "max_usd": max_usd, "max_calls": max_calls}
    con.print(f"[bold]{spec.evaluator_id}[/] → {spec.dimension} ({spec.cost_class}) "
              f"1 call x ${price} against ${max_usd} / {max_calls} call(s)")
    con.print(f"[dim]context injected: {', '.join(plan['context_injected']) or 'none'}[/]")

    if dry_run:
        con.print("[dim]--dry-run: loaded, fence-checked and context-closed; no client was "
                  "built and nothing was called[/]")
        dump(plan | {"dry_run": True}, out)
        return
    if not model:
        raise typer.BadParameter("--model is required for a real run; only --dry-run may go "
                                 "without one")

    real = JsonJudgeModel(cli_common.llm_client(model, api_base), model)
    try:
        verdict, rec = J.run_evaluator(spec, available, registry=gate, model=real,
                                       ledger=ledger, subject_id=subject_id)
    except (J.JudgeRefusal, evals.UnknownDimension) as e:
        con.print(f"[red]{type(e).__name__}: {e}[/]")
        raise typer.Exit(2) from e
    con.print(f"[bold]score[/]: {verdict.score}  incomplete={verdict.incomplete}")
    con.print(f"[yellow]{verdict.notice}[/]")
    dump(plan | {"dry_run": False, "verdict": verdict.to_dict(), "run": rec.to_dict(),
                 "ledger": ledger.report()}, out)
