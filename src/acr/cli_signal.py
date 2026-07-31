"""One place to ask a completed run for signals, whichever way the signal is produced.

TWO WAYS, ONE OUTPUT
--------------------
A signal about a run comes from one of two places and they could not be less alike:

  RULE   deterministic checks over the trace and the manifest. Same input, same output,
         forever. `acr eval` and `acr audit` already do this and no model is reachable from
         either — `tests/test_evals.py::test_no_model_is_reachable_from_this_module` walks the
         import graph of `evals.py` and fails if one appears.
  AGENT  a model reads the work log and says why something happened. `acr attribute` already
         does this, under a tool surface that gives it no way to assert a verdict.

Both emit a `SignalEnvelope`, so whatever consumes signals consumes one shape.

WHY THIS IS A NEW GROUP AND NOT A FLAG ON `acr eval`
----------------------------------------------------
`cli_eval` opens by promising that nothing in it calls a model. Adding `--kind agent` there
would make that sentence false while leaving it on the page. So this group is a thin dispatcher
over the two existing surfaces, and the provider-side imports happen inside the agent branch —
`tests/test_cli_signal.py::test_module_imports_no_provider_at_module_scope` keeps them there.

WHAT THIS MODULE MUST NEVER GROW
--------------------------------
Scoring. If a question can be settled by comparing two values, it belongs in `evals.py` where
it is deterministic and testable. A dispatcher that starts deciding correctness is a second
answer to a question that already has one.
"""
from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from .cli_common import con
from .local_artifacts import LOCAL_ROOT_ENV

#: Progress goes here so that stdout stays exactly one JSON document. `cli_common.con` is the
#: stdout console every other group shares and stays the one that prints the envelope.
_err = Console(stderr=True)

signal_app = typer.Typer(add_completion=False, help=(
    "Ask a completed run for signals. --kind rule runs the deterministic checks and calls no "
    "model; --kind agent runs the diagnostic agent, whose method comes from eval skills."))

#: The two ways a signal is produced. Not an open set: a third would need its own guarantee
#: about what it may and may not decide.
KINDS: tuple[str, ...] = ("rule", "agent")

#: Which `SignalEnvelope.signal_type` each kind emits. Both are already in `kernel.SIGNAL_TYPES`.
SIGNAL_TYPE_FOR_KIND: dict[str, str] = {
    "rule": "EVALUATION_RESULT",
    "agent": "ATTRIBUTION_REPORT",
}

#: The eval skills the diagnostic agent is offered by default. Every one is `slot: eval` and
#: declares what it may judge; see `acr.skills.eval_skills_block`.
DEFAULT_EVAL_SKILLS: tuple[str, ...] = (
    "eval-contrast-traces",
    "eval-cluster-failures",
    "eval-missed-evidence",
    "eval-overconfidence",
)

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
    """Keep `signal` a group while it still has one command.

    Typer collapses a single-command app into a bare command, which would make `acr signal run`
    an unexpected-argument error and quietly rename the entry point to `acr signal`. Task 8
    adds `batch` beside it; the spelling in every runbook must not change when it does.
    """


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


def _eval_skill_names(raw: str) -> tuple[str, ...]:
    """Which diagnostic skills to offer. Empty string means the default set."""
    if not raw.strip():
        return DEFAULT_EVAL_SKILLS
    return tuple(s.strip() for s in raw.split(",") if s.strip())


@signal_app.command("run")
def signal_run(
    kind: str = typer.Option(..., "--kind", callback=_check_kind,
                             help=f"one of {list(KINDS)}"),
    run: str = typer.Option(..., "--run", help="one *.manifest.json from a completed chart run"),
    spec: str = typer.Option(..., "--spec", "-s", help="the spec that run was made under"),
    gold: str = typer.Option("", "--gold", help="answer key; agent kind only, enables contrast"),
    case_id: str = typer.Option("", "--case-id", help="pseudonymous case id; agent kind only"),
    eval_skills: str = typer.Option(
        "", "--eval-skills",
        help="comma list of eval skills to offer the agent; default is all four"),
    out: str = typer.Option("", "--out", help="write the signal JSON here instead of stdout"),
    local_root: str | None = LOCAL_ROOT,
):
    """Produce signals for ONE completed run.

    The deterministic kind reads the trace and manifest and calls no model. The agent kind
    reads the same files, plus the answer key when one is supplied, and returns a diagnosis —
    never a verdict, because it has no tool that emits one.
    """
    _check_kind(kind)       # again: the callback covers the CLI, this covers a direct call
    if kind == "rule":
        payload = _rule_signal(run=run, spec=spec, local_root=local_root)
    else:
        payload = _agent_signal(run=run, spec=spec, gold=gold, case_id=case_id,
                                eval_skills=_eval_skill_names(eval_skills),
                                local_root=local_root)
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
                  eval_skills: tuple[str, ...], local_root: str | None = None) -> dict:
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
        signal_type=SIGNAL_TYPE_FOR_KIND["agent"], local_root=local_root,
        **DEFAULT_DETECTOR_ARGS)
