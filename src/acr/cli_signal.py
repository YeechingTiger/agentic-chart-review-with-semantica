"""One place to ask a completed run for signals, whichever way the signal is produced.

THREE WAYS, ONE OUTPUT
----------------------
A signal about a run comes from one of three places and they could not be less alike:

  RULE   deterministic checks over the trace and the manifest. Same input, same output,
         forever. `acr eval` and `acr audit` already do this and no model is reachable from
         either — `tests/test_evals.py::test_no_model_is_reachable_from_this_module` walks the
         import graph of `evals.py` and fails if one appears.
  JUDGE  a model scores the trajectory ITSELF — was the read order sensible, was the effort
         proportionate — on dimensions that have no ground truth by construction. `acr judge`
         already does this behind a fence that refuses every dimension code already decides,
         and blinds the answer key by giving the packet no field to hold one.
  AGENT  a model reads the work log and says why something happened. `acr attribute` already
         does this, under a tool surface that gives it no way to assert a verdict.

All three emit a `SignalEnvelope`, so whatever consumes signals consumes one shape. The two
model-backed kinds are not interchangeable and must not be averaged with the first: JUDGE
carries `evidence_class: JUDGED`, which says the number screens and ranks a human's reading
queue and never gates.

WHY THIS IS A NEW GROUP AND NOT A FLAG ON `acr eval`
----------------------------------------------------
`cli_eval` opens by promising that nothing in it calls a model. Adding `--kind agent` there
would make that sentence false while leaving it on the page. So this group is a thin dispatcher
over three existing surfaces, and the provider-side imports happen inside the branch that needs
them — `tests/test_cli_signal.py::test_module_imports_no_provider_at_module_scope` keeps them
out of module scope.

WHAT THIS MODULE MUST NEVER GROW
--------------------------------
Scoring. If a question can be settled by comparing two values, it belongs in `evals.py` where
it is deterministic and testable. A dispatcher that starts deciding correctness is a second
answer to a question that already has one — and note that the judge kind is not an exception
to this: `judge()` asks the precedence registry FIRST and refuses, and nothing in this file
re-decides that question or offers an argument that would.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from .cli_common import API_BASE, MODEL, con
from .local_artifacts import LOCAL_ROOT_ENV, LocalArtifactError, LocalArtifactStore

#: Progress goes here so that stdout stays exactly one JSON document. `cli_common.con` is the
#: stdout console every other group shares and stays the one that prints the envelope.
_err = Console(stderr=True)

signal_app = typer.Typer(add_completion=False, help=(
    "Ask a completed run for signals. --kind rule runs the deterministic checks and calls no "
    "model; --kind judge runs the fenced trajectory judge; --kind agent runs the diagnostic "
    "agent, whose method comes from eval skills."))

#: The three ways a signal is produced. Not an open set: a fourth would need its own guarantee
#: about what it may and may not decide.
KINDS: tuple[str, ...] = ("rule", "judge", "agent")

#: Which `SignalEnvelope.signal_type` each kind emits. All are already in `kernel.SIGNAL_TYPES`.
#: `judge` shares `EVALUATION_RESULT` with `rule` on purpose — both are measurements OF a run,
#: not explanations of one. What separates them is `deterministic: false` and
#: `evidence_class: JUDGED` on the envelope, which is what `judge.py`'s "never average the two
#: classes" rule looks like once it reaches a consumer reading signals off a queue.
SIGNAL_TYPE_FOR_KIND: dict[str, str] = {
    "rule": "EVALUATION_RESULT",
    "judge": "EVALUATION_RESULT",
    "agent": "ATTRIBUTION_REPORT",
}

#: WHAT THE EVAL AGENT IS ALLOWED TO ASSUME ABOUT THE ANSWER KEY, as three groups of cards.
#: Every one is `slot: eval` and declares what it may judge; see `acr.skills.eval_skills_block`.
#:
#: A failed run has two readings and they license OPPOSITE mistakes. Believe the key and the
#: cause has to be in the run — a term never searched, a type filter that masked the document, a
#: passage read and misjudged. Doubt the key and the question is whether it was ever derivable
#: from THIS chart. Both are real; which one applies is not something the agent can settle from
#: the trace, because the trace looks the same either way.
#:
#: These groups were previously ONE default tuple containing all five, so every diagnosis was
#: told "the key is also a suspect" AND "confirm the value is genuinely documented before you
#: start" in the same system prompt. That is not more method. It hands the agent an exit from
#: every hard failure ("the key may be wrong") and an exit from every unreachable key ("the
#: agent erred"), and its choice between them is recorded nowhere.
KEY_IS_RIGHT_SKILLS: tuple[str, ...] = ("eval-missed-evidence", "eval-overconfidence")

#: Doubt, licensed. A registry value is what a person wrote down: they read outside records this
#: chart does not have, they mistype, and they apply a rule differently than the contract does.
#: README §2.5 already refuses to call such a value gold — this is the card that can act on it.
KEY_IS_SUSPECT_SKILLS: tuple[str, ...] = ("eval-key-challenge",)

#: Neither posture. Both cards defer the correctness question to the deterministic scorer in
#: their own opening lines, so they carry no assumption to conflict with, and both modes get them.
KEY_AGNOSTIC_SKILLS: tuple[str, ...] = ("eval-contrast-traces", "eval-cluster-failures")

#: The posture is an INPUT, chosen per invocation. Two modes and no third: a mode that mixed the
#: groups would be the merged default this replaced, wearing a name.
EVAL_MODES: dict[str, tuple[str, ...]] = {
    "run-fault": KEY_AGNOSTIC_SKILLS + KEY_IS_RIGHT_SKILLS,
    "key-suspect": KEY_AGNOSTIC_SKILLS + KEY_IS_SUSPECT_SKILLS,
}

#: Believing the key is the right default: it is true far more often than not, and the diagnosis
#: it yields is actionable — a fix aimed at the run. Doubt has to be asked for, per run, which is
#: also what makes "we doubted the key here" a legible decision afterwards.
DEFAULT_EVAL_MODE = "run-fault"

#: Turns the attribution agent gets. `acr attribute case` defaults to 12 and this dispatcher
#: copied that, which turned out to be structurally too few HERE: the pipeline has eight stages,
#: and the eval-skills block this seam adds is ~7.5 kB of method the plain command never carries.
#: The first live attribution stopped at 11 of 12 with `cause: UNRESOLVED` and the rationale
#: "model-call limit reached without a gate-valid attribution", having skipped the counterfactual
#: test and the skeptic review — which the report gate then, correctly, refused to call resolved.
#: A default under which the deliverable cannot be produced is not a budget, it is a wall.
DEFAULT_AGENT_MODEL_CALLS = 24
DEFAULT_AGENT_CHART_READS = 12

#: Detector thresholds for the screening pass the agent kind runs before it spends. They are
#: named here rather than defaulted inside `DetectorConfig`, whose docstring refuses defaults on
#: the grounds that "thresholds belong where a reviewer reads them, not buried here where they
#: become folklore" — so this dispatcher declares its own, in the file a reviewer of
#: `acr signal run` opens. They are the widest bands in the README's attribution example: the
#: screen only decides what is worth a diagnosis, never whether an answer was right.
DEFAULT_DETECTOR_ARGS: dict[str, object] = {
    "min_term_chars": 2,
    "max_rejection_repeats": 2,
    "token_band": "0,10000000",
    "turn_band": "0,1000",
}

# The same option as every other command that touches patient-derived artifacts, spelled the
# same way. Both branches end at a `LocalArtifactStore`, which refuses a root inside the
# worktree; without the flag the only way to name one would be the environment, and a
# dispatcher that silently reads a different root than the command it forwards to is a
# boundary bug waiting to be written.
LOCAL_ROOT = typer.Option(
    None, "--local-root", envvar=LOCAL_ROOT_ENV,
    help="absolute patient-artifact root outside Git")


@signal_app.callback()
def signal_main() -> None:
    """Keep `signal` a group, whatever the command count happens to be.

    Typer collapses a single-command app into a bare command, which would make `acr signal run`
    an unexpected-argument error and quietly rename the entry point to `acr signal`. `batch`
    now sits beside `run`, so the collapse cannot fire today — and this stays anyway, because
    otherwise the spelling in every runbook is load-bearing on the group having at least two
    commands, which nobody would think to check before deleting one.
    """


def _store(root: str | None) -> LocalArtifactStore:
    """The same four lines `cli_audit`, `cli_attribute`, `cli_evaluation`, `cli_gold` and
    `cli_repair` each keep locally — a sixth copy rather than a reach into one of theirs,
    because importing `cli_attribute` to borrow it would drag the attribution stack onto the
    rule and judge paths, which this module's docstring promises stays out of them."""
    try:
        return LocalArtifactStore(root)
    except LocalArtifactError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _check_kind(kind: str) -> str:
    """Reject an unknown kind AS THE OPTION IS PARSED, not in the command body.

    Click processes the options that were actually supplied before it notices that a required
    one is missing, so a callback here reports `unknown kind 'vibes'`. The same check in the
    body never runs: `--spec` is missing by then and its `Missing option` wins, which tells an
    operator who mistyped `--kind` about the wrong mistake.
    """
    if kind not in KINDS:
        raise typer.BadParameter(f"unknown kind {kind!r}; expected one of {list(KINDS)}")
    return kind


def _check_mode(mode: str) -> str:
    """Reject an unknown mode AS THE OPTION IS PARSED, for `_check_kind`'s reason.

    The message names the modes, because the whole point of the flag is that a posture is a
    choice somebody makes on purpose — and a refusal that does not say what the choices are
    sends the operator to the source.
    """
    if mode not in EVAL_MODES:
        raise typer.BadParameter(
            f"unknown mode {mode!r}; expected one of {list(EVAL_MODES)}")
    return mode


# Shared verbatim between `run` and `batch`, for the reason the judge options are: two spellings
# of one posture is how a cohort ends up half-diagnosed under each. Declared here rather than
# beside LOCAL_ROOT because the callback has to exist by the time this line is evaluated.
MODE = typer.Option(
    DEFAULT_EVAL_MODE, "--mode", callback=_check_mode,
    help="agent kind only, but CHECKED ON EVERY KIND: which posture toward the answer key. "
         "`run-fault` believes the key and looks for the cause in the run; `key-suspect` asks "
         "whether the key was derivable from this chart at all. Overridden by --eval-skills.")


def _eval_skill_names(raw: str, mode: str = DEFAULT_EVAL_MODE) -> tuple[str, ...]:
    """Which diagnostic skills to offer. Empty string means the mode's set.

    An explicit list still wins: a mode is a named pair of defaults, not a whitelist. Somebody
    diagnosing a one-off has to be able to name three cards without inventing a mode for it.
    """
    if not raw.strip():
        return EVAL_MODES[_check_mode(mode)]
    return tuple(s.strip() for s in raw.split(",") if s.strip())


# The judge kind's own options, shared verbatim between `run` and `batch`. Both prices default
# to None rather than to a number: `acr judge panel` requires `--usd-per-call` and `--max-usd`
# with no default because "an unpriced call reads as free", and reaching the same `judge()`
# through a second front door must not be how somebody acquires a default. Click cannot make an
# option conditionally required, so the refusal lives in `_judge_signal` and names both flags.
DIMENSION = typer.Option(
    "", "--dimension", help="judge kind only: which judged dimension to ask about")
USD_PER_CALL = typer.Option(
    None, "--usd-per-call",
    help="judge kind only. REQUIRED there, no default: one call per lens, and an unpriced "
         "call reads as free.")
MAX_USD = typer.Option(
    None, "--max-usd",
    help="cost ceiling. REQUIRED for --kind judge, where it is checked against the priced "
         "panel before any call; for --kind agent it is the per-run attribution budget "
         "(1.0 when unset). PER RUN in batch, not per cohort.")


@signal_app.command("run")
def signal_run(
    kind: str = typer.Option(..., "--kind", callback=_check_kind,
                             help=f"one of {list(KINDS)}"),
    run: str = typer.Option(..., "--run", help="one *.manifest.json from a completed chart run"),
    spec: str = typer.Option(..., "--spec", "-s", help="the spec that run was made under"),
    gold: str = typer.Option("", "--gold",
                             help="answer key. Enables contrast on the agent kind; IGNORED "
                                  "ENTIRELY on every blinded judged dimension."),
    case_id: str = typer.Option("", "--case-id", help="pseudonymous case id; agent kind only"),
    eval_skills: str = typer.Option(
        "", "--eval-skills",
        help="comma list of eval skills to offer the agent; default is the --mode's set"),
    mode: str = MODE,
    max_model_calls: int = typer.Option(
        DEFAULT_AGENT_MODEL_CALLS, "--max-model-calls", min=1,
        help="agent kind only: turns the attribution pipeline gets. Its eight stages plus the "
             "eval-skills block do not fit in `acr attribute case`'s 12, and a run that stops "
             "short is refused resolution rather than shipped half-diagnosed"),
    max_chart_reads: int = typer.Option(
        DEFAULT_AGENT_CHART_READS, "--max-chart-reads", min=0,
        help="agent kind only: chart reads allowed for discriminating rival causes"),
    dimension: str = DIMENSION,
    usd_per_call: float | None = USD_PER_CALL,
    max_usd: float | None = MAX_USD,
    model: str | None = MODEL,
    api_base: str | None = API_BASE,
    out: str = typer.Option("", "--out", help="write the signal JSON here instead of stdout"),
    local_root: str | None = LOCAL_ROOT,
):
    """Produce signals for ONE completed run.

    The deterministic kind reads the trace and manifest and calls no model. The judge kind
    scores the trajectory itself — an opinion that screens and ranks, never a gate. The agent
    kind reads the same files, plus the answer key when one is supplied, and returns a
    diagnosis — never a verdict, because it has no tool that emits one.
    """
    _check_kind(kind)       # again: the callback covers the CLI, this covers a direct call
    if kind == "rule":
        payload = _rule_signal(run=run, spec=spec, local_root=local_root)
    elif kind == "judge":
        payload = _judged_or_exit(run=run, spec=spec, dimension=dimension, gold=gold,
                                  usd_per_call=usd_per_call, max_usd=max_usd, model=model,
                                  api_base=api_base, local_root=local_root)
    else:
        payload = _agent_signal(run=run, spec=spec, gold=gold, case_id=case_id,
                                eval_skills=_eval_skill_names(eval_skills, mode),
                                max_usd=max_usd, local_root=local_root,
                                max_model_calls=max_model_calls,
                                max_chart_reads=max_chart_reads)
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        con.print(f"→ {out}")
    else:
        con.print_json(text)


def _rule_signal(*, run: str, spec: str, subject_id: str = "",
                 provider_boundary: str = "UNKNOWN", local_root: str | None = None) -> dict:
    """Deterministic checks over one run. No model is imported on this path.

    Delegates to `cli_audit.audit_run_payload`, which is the body of `acr audit run` extracted
    for reuse. Rebuilding the AuditContext here would be a second place where a trajectory is
    assembled from a manifest, and the two would drift.
    """
    from . import evals
    from .cli_audit import audit_run_payload

    report = audit_run_payload(manifest=run, subject_id=subject_id,
                               provider_boundary=provider_boundary, local_root=local_root)
    return {
        "schema": "acr.signal/1",
        "signal_type": SIGNAL_TYPE_FOR_KIND["rule"],
        "kind": "rule",
        "run": run,
        "spec": spec,
        "deterministic": True,
        # `rule_compliance` is deterministic in the registry but can never fire:
        # `answer_checks.ANSWER_CHECK_KINDS` has been empty since 2026-07-30, when all five
        # clinical checks were measured and removed (58 firings destroyed a correct value).
        # Advertising it here would claim a check that no run receives.
        "dimensions": [d.name for d in evals.REGISTRY.values()
                       if d.deterministic and d.name != "rule_compliance"],
        "report": report,
    }


def _agent_signal(*, run: str, spec: str, gold: str, case_id: str,
                  eval_skills: tuple[str, ...], case_map: str = "",
                  max_usd: float | None = None, local_root: str | None = None,
                  max_model_calls: int = DEFAULT_AGENT_MODEL_CALLS,
                  max_chart_reads: int = DEFAULT_AGENT_CHART_READS) -> dict:
    """The diagnostic agent over one run. Provider imports live here, not at module scope."""
    from .skills import SkillError, eval_skills_block

    try:
        # Validates slot and `judges` before spending. A card that is not `slot: eval` is an
        # operator's typo, not a crash: the group's whole claim is that a wrong `--eval-skills`
        # costs nothing, and a traceback is a worse way to say so than a usage error.
        block = eval_skills_block(list(eval_skills))
    except SkillError as exc:
        raise typer.BadParameter(str(exc)) from exc
    # To stderr, not stdout. An agent run takes minutes and an operator deserves to see what
    # method was loaded before it starts, but stdout is the envelope and a progress line in the
    # middle of it is how machine-readable output stops being machine-readable.
    _err.print(f"[dim]{len(eval_skills)} eval skills, {len(block.encode('utf-8'))} bytes[/]")
    from .cli_attribute import attribute_case_payload
    return attribute_case_payload(
        run=run, spec=spec, gold=gold, case_id=case_id, eval_skills_prompt=block,
        signal_type=SIGNAL_TYPE_FOR_KIND["agent"], case_map=case_map, local_root=local_root,
        # `attribute_case_payload`'s own default, restated rather than routed around: the flag
        # is shared with the judge kind, where it is required, so an unset flag here has to mean
        # "the budget `acr attribute case` would have used" and not "no budget".
        max_usd=1.0 if max_usd is None else max_usd,
        max_model_calls=max_model_calls, max_chart_reads=max_chart_reads,
        **DEFAULT_DETECTOR_ARGS)


# ================================================= THE TRAJECTORY JUDGE, THROUGH THE FENCE
#: What of a run manifest a judge is shown. An ALLOWLIST, for the reason `judge.TRACE_KEYS_SHOWN`
#: is one, and for a second reason that is specific to this seam: `judge._render` serialises the
#: artifacts BEFORE the trace and truncates the pair at `PACKET_CHAR_BUDGET`. A manifest handed
#: over whole runs to tens of kilobytes — `develop_plane_candidates` alone can — so it evicts
#: the trajectory, and a trajectory judge shown no trajectory still returns three confident
#: scores.
#: Nothing here is key-bearing; `blind_packet` re-checks that anyway and is the enforcement point.
MANIFEST_KEYS_SHOWN: tuple[str, ...] = (
    "run_id", "patient_id", "spec_id", "model", "answer", "steps", "gate_validated",
    "negative_basis", "plan_revisions", "rejections", "usage", "degradation", "elapsed_s")


def _packet_from_run(*, run: str, gold: str, dimension: str,
                     local_root: str | None = None):
    """Assemble a judge packet from a run's manifest and its sibling trace.

    This is the whole ergonomic point of the kind: `acr judge panel` takes a packet an operator
    hand-builds as JSON, which is a real barrier to ever judging a cohort.

    THE BLIND/KEYED DECISION IS NOT MADE HERE BY POLICY — it falls out of `judge.py`'s own
    constants. For a blinded dimension `gold` is ignored ENTIRELY: no key is read, no keyed
    packet is built and filtered, and the type that comes back has no field one could have
    reached. That is the isolation working as designed rather than being re-promised here.
    """
    from . import judge as J

    store = _store(local_root)
    manifest_path = store.require_input(run, what="manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # The same derivation `cli_audit`, `cli_evaluation`, `evals` and `attribution` all use. A
    # fifth spelling of it would be a fifth thing to fix when run artifacts are renamed.
    trace_path = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".jsonl"))
    trace = ([json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
              if line.strip()] if trace_path.is_file() else [])
    shown = {k: v for k, v in manifest.items() if k in MANIFEST_KEYS_SHOWN}
    subject = str(manifest.get("patient_id") or manifest_path.stem)
    if dimension in J.KEY_PERMITTED_DIMENSIONS and gold:
        key = json.loads(store.require_input(gold, what="answer key").read_text(encoding="utf-8"))
        return J.keyed_packet(trace=trace, artifacts={"manifest": shown},
                              answer_key=key, subject_id=subject)
    return J.blind_packet(trace=trace, artifacts={"manifest": shown}, subject_id=subject)


def _judge_signal(*, run: str, spec: str, dimension: str, gold: str,
                  usd_per_call: float | None, max_usd: float | None,
                  model: str | None, api_base: str | None,
                  local_root: str | None = None) -> dict:
    """One judged verdict as a signal. Every refusal is `judge()`'s own and is raised verbatim.

    This function never asks the precedence registry whether a dimension is judgeable. A CLI
    that pre-screened it — even correctly, even off the same registry — would be a second copy
    of the judgement, free to drift the first time somebody adds a row. `cli_judge` opens with
    that rule; arriving through a different front door does not suspend it.
    """
    from . import cli_common, evals
    from . import judge as J
    from .cli_judge import JsonJudgeModel

    if not dimension:
        raise typer.BadParameter("--kind judge requires --dimension (one of "
                                 f"{list(J.JUDGEABLE_DIMENSIONS)})")
    if usd_per_call is None or max_usd is None:
        raise typer.BadParameter(
            "--kind judge requires --usd-per-call and --max-usd. Neither has a default here "
            "for the reason `acr judge panel` gives for having none: one model call per lens, "
            "and an unpriced call reads as free.")
    dim = J._norm(dimension)
    # Priced from the REAL lens count or not priced at all. The plan said to assume three
    # lenses for an unrecognised dimension, which puts a number on a panel that does not exist;
    # when there is no count, `judge()` below refuses the dimension in its own words and the
    # ceiling never comes up, because nothing was ever going to be called.
    if dim in J.LENSES:
        planned = round(len(J.LENSES[dim]) * usd_per_call, 6)
        if planned > max_usd:
            raise typer.BadParameter(
                f"{len(J.LENSES[dim])} lenses x ${usd_per_call} = ${planned} exceeds --max-usd "
                f"{max_usd}; nothing was called")
    packet = _packet_from_run(run=run, gold=gold, dimension=dim, local_root=local_root)
    if not model:
        raise typer.BadParameter(
            "--model is required: a judged number is conditioned on the model that produced "
            "it, and an unattributable verdict cannot be re-checked when the model changes")
    # `cli_common.llm_client(...)`, through the module and not imported by name — that is what
    # makes one monkeypatch silence the provider for every command group, and the plan's
    # `from .cli_common import llm_client` would have made this group the exception.
    verdict = J.judge(dim, packet, registry=evals.precedence_gate(),
                      model=JsonJudgeModel(cli_common.llm_client(model, api_base), model))
    return {
        "schema": "acr.signal/1",
        "signal_type": SIGNAL_TYPE_FOR_KIND["judge"],
        "kind": "judge",
        "run": run,
        "spec": spec,
        "deterministic": False,
        # `judge.py`'s rule, restated where a consumer of signals reads it: this number screens
        # and ranks. It never gates, and it never averages with a deterministic score.
        "evidence_class": J.EV_JUDGED,
        "dimension": dim,
        "verdict": verdict.to_dict(),
    }


def _judged_or_exit(**kw) -> dict:
    """`_judge_signal`, with the fence's refusals reported as refusals rather than tracebacks.

    Uncaught, a `JudgeRefusal` reaches the shell as a traceback and exit 1 — indistinguishable,
    to a script, from the command crashing. `cli_judge` already learned that. In `batch` the
    same exception is caught one level up and recorded as this run's entry in the array.
    """
    from . import judge as J

    try:
        return _judge_signal(**kw)
    except J.JudgeRefusal as exc:
        _err.print(f"[red]{type(exc).__name__}: {exc}[/]")
        raise typer.Exit(2) from exc


# =============================================================== MANY RUNS, ONE ARRAY
def _manifest_paths(runs: str) -> list[Path]:
    """One manifest, or every manifest below a directory, sorted so two batches line up.

    Sorted, not `rglob` order: two arms of the same cohort are compared by reading their two
    signal arrays side by side, and filesystem order would silently misalign them.

    Plain paths, not `LocalArtifactStore` paths. The store boundary is enforced one layer down,
    inside `audit_run_payload` and `attribute_case_payload`, and re-deciding it here would be a
    second answer to "may this dispatcher read that file" — the mistake `cli_judge` names in its
    own docstring about re-implementing a fence.
    """
    p = Path(runs)
    if p.is_file():
        return [p]
    if not p.is_dir():
        raise typer.BadParameter(f"{runs}: not a file or directory")
    found = sorted(p.rglob("*.manifest.json"))
    if not found:
        raise typer.BadParameter(f"{runs}: no *.manifest.json below it")
    return found


def _patient_to_case(case_map: str, local_root: str | None) -> dict[str, str]:
    """`acr attribute`'s own `{case_id: patient_id}` map, reversed and loaded by its own reader.

    THE SAME FILE AND THE SAME DIRECTION AS `acr attribute --case-map`, deliberately. The plan
    gave this command a private shape — manifest stem to case id — and one flag name meaning two
    different mappings inside one CLI is a trap that costs an operator a whole batch. Reversing
    the existing map also puts the pseudonymisation check (`attribution.safe_case_id`) on this
    path for free, whereas a stem-keyed map would have carried `SYN0001.manifest` into the
    error-case library as a case identifier.
    """
    if not case_map:
        return {}
    from .cli_attribute import _case_map as load_case_map
    return {patient: case
            for case, patient in load_case_map(_store(local_root), case_map).items()}


def _case_id_for(path: Path, patient_to_case: dict[str, str]) -> str:
    """The pseudonymous case id for one manifest, from its patient id.

    With no map the manifest's own `patient_id` is used. That is right for a synthetic corpus
    and refused for a real one: `attribution.safe_case_id` rejects an id that matches the real
    corpus's person shape, so the boundary is checked where it is already implemented rather
    than guessed at here.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    patient = str(raw.get("patient_id") or "")
    return patient_to_case.get(patient) or patient


def _batch_signals(*, kind: str, paths: list[Path], spec: str, gold: str,
                   patient_to_case: dict[str, str], eval_skills: tuple[str, ...],
                   case_map: str = "", dimension: str = "",
                   usd_per_call: float | None = None, max_usd: float | None = None,
                   model: str | None = None, api_base: str | None = None,
                   local_root: str | None = None,
                   # Defaulted to the same constants `run` uses, not to None: the agent branch
                   # below read these two out of thin air, so `--kind agent` raised NameError per
                   # run and the per-run `except` filed it as a bad RUN. A batch of nothing but
                   # that reads as a cohort that failed evaluation, not as a command that has
                   # never once worked.
                   max_model_calls: int = DEFAULT_AGENT_MODEL_CALLS,
                   max_chart_reads: int = DEFAULT_AGENT_CHART_READS) -> list[dict]:
    """Signals for every run, in path order.

    A FAILURE ON ONE RUN IS RECORDED AND THE BATCH CONTINUES. Aborting would discard the signals
    already produced — and on the agent kind, the money already spent producing them. The
    failure lands in the array beside the successes, in the same position the signal would have
    occupied, so the array is still one entry per run and a reader counts both without knowing
    which runs failed in advance.
    """
    out: list[dict] = []
    for path in paths:
        try:
            if kind == "rule":
                out.append(_rule_signal(run=str(path), spec=spec, local_root=local_root))
            elif kind == "judge":
                out.append(_judge_signal(
                    run=str(path), spec=spec, dimension=dimension, gold=gold,
                    usd_per_call=usd_per_call, max_usd=max_usd, model=model,
                    api_base=api_base, local_root=local_root))
            else:
                # Inside the try, not above the loop: a manifest that will not parse is one bad
                # run, and resolving every case id up front would make it the whole batch.
                out.append(_agent_signal(
                    run=str(path), spec=spec, gold=gold,
                    case_id=_case_id_for(path, patient_to_case),
                    eval_skills=eval_skills, case_map=case_map, local_root=local_root,
                    max_model_calls=max_model_calls, max_chart_reads=max_chart_reads))
        except Exception as exc:                # noqa: BLE001 - one bad run is not the batch
            _err.print(f"[red]{path.name}: {type(exc).__name__}: {exc}[/]")
            out.append({"schema": "acr.signal/1", "run": str(path), "kind": kind,
                        "error": f"{type(exc).__name__}: {exc}"})
    return out


@signal_app.command("batch")
def signal_batch(
    kind: str = typer.Option(..., "--kind", callback=_check_kind,
                             help=f"one of {list(KINDS)}"),
    runs: str = typer.Option(..., "--runs",
                             help="a *.manifest.json, or a directory searched recursively"),
    spec: str = typer.Option(..., "--spec", "-s", help="the spec those runs were made under"),
    gold: str = typer.Option("", "--gold", help="answer key; agent kind only"),
    case_map: str = typer.Option(
        "", "--case-map",
        help="JSON {pseudonymous_case_id: patient_id}, the same file `acr attribute` takes; "
             "agent kind only. Without it a run's own patient_id is used as its case id."),
    eval_skills: str = typer.Option("", "--eval-skills",
                                    help="comma list; default is the --mode's set"),
    mode: str = MODE,
    max_model_calls: int = typer.Option(
        DEFAULT_AGENT_MODEL_CALLS, "--max-model-calls", min=1,
        help="agent kind only: turns the attribution pipeline gets. Its eight stages plus the "
             "eval-skills block do not fit in `acr attribute case`'s 12, and a run that stops "
             "short is refused resolution rather than shipped half-diagnosed"),
    max_chart_reads: int = typer.Option(
        DEFAULT_AGENT_CHART_READS, "--max-chart-reads", min=0,
        help="agent kind only: chart reads allowed for discriminating rival causes"),
    dimension: str = DIMENSION,
    usd_per_call: float | None = USD_PER_CALL,
    max_usd: float | None = MAX_USD,
    model: str | None = MODEL,
    api_base: str | None = API_BASE,
    out: str = typer.Option("", "--out", help="write the JSON array here instead of stdout"),
    local_root: str | None = LOCAL_ROOT,
):
    """Produce signals for MANY completed runs. One bad run is recorded, not fatal.

    `--max-usd` is PER RUN on both spending kinds, not per cohort; the line printed to stderr
    before the batch starts multiplies it out so the worst case is on screen before it happens.

    The exit code is not the verdict on any run — read the array. It is 2 only when EVERY run
    failed, because exit 0 over an array of nothing but errors tells a shell script the cohort
    was evaluated when it was not.
    """
    _check_kind(kind)       # again: the callback covers the CLI, this covers a direct call
    paths = _manifest_paths(runs)
    ceiling = ("" if kind == "rule" or max_usd is None
               else f", up to ${round(len(paths) * max_usd, 6)} in total at ${max_usd} each")
    _err.print(f"[dim]{len(paths)} runs, kind={kind}{ceiling}[/]")
    signals = _batch_signals(
        kind=kind, paths=paths, spec=spec, gold=gold,
        patient_to_case=_patient_to_case(case_map, local_root) if kind == "agent" else {},
        eval_skills=_eval_skill_names(eval_skills, mode), case_map=case_map, dimension=dimension,
        usd_per_call=usd_per_call, max_usd=max_usd, model=model, api_base=api_base,
        local_root=local_root,
        max_model_calls=max_model_calls, max_chart_reads=max_chart_reads)
    failed = sum("error" in s for s in signals)
    text = json.dumps(signals, indent=2, ensure_ascii=False, default=str)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        con.print(f"→ {out}  ({failed} of {len(signals)} failed)")
    else:
        con.print_json(text)
        _err.print(f"[dim]{failed} of {len(signals)} failed[/]")
    if failed == len(signals):
        raise typer.Exit(2)
