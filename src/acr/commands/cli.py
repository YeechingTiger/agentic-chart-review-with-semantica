"""Compose the `acr` command line out of one module per command group, and decide nothing.

This file was 1206 lines and reached fifteen modules directly — the widest coupling in the
tree — which meant that adding a command to any group edited the same file as every other
group, and that the top-level entry point had to import `graph`, `intake`, `explain` and the
rest merely to be able to print `--help`. It is now a mounting board: every rule, refusal and
number lives in the group that owns it, and the only thing that can break here is an app that
was not mounted.

THE GROUPS, IN THE ORDER A USER MEETS THEM

  cli_chart      one agent run at a time, against one chart
  cli_pipeline   the cohort-scale L0-L5 artifact chain: extract -> concord -> explain
  cli_plan       what CAN be run, before anything is: `ask` and `deps`, neither reads a chart
  cli_spec       the spec in front of the clinician who owns its decisions
  derive/assets  the retrieval-asset development loop (their sub-apps live in their modules)
  cli_label      THE DEVELOP PLANE: the full scan. Spends money per note; see its docstring
  cli_refine     THE DEVELOP PLANE: §6b's optimizer over text parameters. Spends nothing
  cli_eval       THE EVAL PLANE: precedence registry, detectors, regression harness. Free
  cli_judge      THE EVAL PLANE: agent-as-a-judge, fenced. Spends money per verdict
  cli_evaluation versioned modules and typed local evaluation pipelines
  cli_attribute  tool-using causal attribution over completed run traces

The last four had no entry point at all until now: roughly 4,250 lines of develop-plane and
eval-plane code with tests beside it and no way for anyone to invoke it, which is the same
condition as not existing.

A GROUP MOUNTED WITHOUT A NAME merges its commands into the top level (`acr extract`), and one
mounted WITH a name becomes a sub-command (`acr judge panel`). The first four groups are
nameless because their commands were top-level before this split and moving them would be a
behaviour change wearing a refactor's clothes.
"""
from __future__ import annotations

import importlib

import typer

from ..core import site
from ..core.cli_common import (  # noqa: F401 (re-export)
    CONCORD_SCHEMA,
    EXPLAIN_SCHEMA,
    EXTRACT_SCHEMA,
)
from .cli_chart import chart_app

app = typer.Typer(add_completion=False, help="Agentic EHR chart review.")


@app.callback()
def _gate() -> None:
    """Refuse every command when a real corpus is declared and its identifier shape is not.

    THE GATE HAD NO CALLER. `site.require_person_id_pattern` was written on 2026-08-03 with the
    reasoning that "an inert guard is indistinguishable from a satisfied one", and then wired to
    nothing: a deployment could point `ACR_REAL_CORPUS` at real data with no identifier pattern set,
    and every mask and every person-id refusal in the system would silently do nothing. The gate that
    exists to prevent a silent no-op WAS one.

    A Typer root callback runs before any subcommand, which is the only placement that covers the
    whole surface — putting it in the chart-run path would leave `acr eval`, `acr audit` and
    `acr label` reading the same real corpus ungated.
    """
    site.require_person_id_pattern()


app.add_typer(chart_app)

# The Site Mapping is neither a chart run nor a clinical decision: it is the local-name to
# document-concept table that `doc_type_matches` used to approximate with a substring. Its own
# group because three of its four commands read no chart and spend nothing, and because the
# one that spends runs once per corpus rather than once per run.

# ------------------------------------------------------------------------ development plane
# `assets` does not run the agent and does not read a chart: it develops the retrieval assets a
# run depends on — the keyword lists and the strata — against a complete per-note labelling of
# a small dev set, and refuses to certify one on data the search has seen. Mounted here rather
# than kept as a private entry point because the loop it implements (measure -> propose ->
# evolve -> certify -> adopt) is the only way anything in `assets/specs/` stops being a guess, and a
# development tool nobody can find is a development tool nobody runs.

# `derive` is the FIRST-ORDER member of that same family, and it comes before `assets` in the
# order anyone should use them: count what the labelling already says, price it by grep, cut
# the list, propose the read policy. `assets` hill-climbs, which only refines a list that
# already exists. Both are mounted because the search is still worth running afterwards; the
# derivation is what makes there be something to refine.


# `label` comes before both of them in the order of the loop and did not exist as a command
# until now: `derive` reads a labels.jsonl that only the full scan can produce, so the first
# step of the develop plane was the one step nobody could take.

# ------------------------------------------------------------------------------- eval plane
# `eval` before `judge` deliberately. `eval dimensions` prints the precedence fence, and the
# fence is the thing to read before reaching for a judge: where a deterministic evaluator
# exists a judged opinion is refused, and no flag on `acr judge` will change that.
# `signal` is the one door to a signal about a completed run, whichever way it is produced. It
# is a group of its own rather than `acr eval --kind agent` because `eval` promises it reaches
# no model, and a group cannot keep that promise and also host the diagnostic agent.
# Attribution is model-based and may read the same patient's chart, so it cannot live under
# `eval`, whose import closure and CLI contract guarantee that no model is reachable.



# ----------------------------------------------------------------- optional sibling command groups
#
# ONE `acr` BINARY THAT GROWS WITH WHAT IS INSTALLED. After the 2026-08-03 split the working planes
# ship as separate distributions into the same `acr.*` namespace, and this file is the single entry
# point the owner asked for. It mounts a group when the distribution providing it is importable and
# says nothing when it is not — so `pip install acr-chart-review` gives a CLI that runs charts, and
# adding `acr-eval` beside it grows `acr eval`, `acr judge`, `acr audit` without editing anything.
#
# `ImportError` ONLY, and never a bare `except`. A group whose module is absent is a distribution
# that is not installed, which is expected. A group whose module is present but RAISES on import is
# broken, and swallowing that would hide it behind a subcommand that silently does not exist —
# which is the hardest kind of missing to notice, because `acr --help` simply looks shorter.
#: `(module path relative to `acr`, attribute, mount name or None for top level)`. The path is
#: relative to `acr` and not to `acr.commands`, because two of these Typer apps live inside the plane
#: that owns them (`improvement/derive.py`, `improvement/assetdev.py`) rather than in a `cli_*`
#: module. Assuming `acr.commands` dropped `acr derive` and `acr assets` from the binary silently,
#: which is exactly the failure the ImportError-only rule below exists to prevent — and I caused it
#: by hand in the same edit that wrote that rule.
_OPTIONAL: tuple[tuple[str, str, str | None], ...] = (
    # acr-eval
    ("commands.cli_eval", "eval_app", "eval"),
    ("commands.cli_signal", "signal_app", "signal"),
    ("commands.cli_judge", "judge_app", "judge"),
    ("commands.cli_evaluation", "evaluation_app", "evaluation"),
    ("commands.cli_audit", "audit_app", "audit"),
    ("commands.cli_attribute", "attribute_app", "attribute"),
    # acr-improvement
    ("improvement.derive", "derive_app", "derive"),
    ("improvement.assetdev", "assets_app", "assets"),
    ("commands.cli_label", "label_app", "label"),
    ("commands.cli_refine", "refine_app", "refine"),
    ("commands.cli_repair", "repair_app", "repair"),
    # acr-rules
    ("commands.cli_spec", "spec_app", "spec"),
    ("commands.cli_plan", "plan_app", None),
    ("commands.cli_site_mapping", "site_mapping_app", "site-mapping"),
    ("commands.cli_pipeline", "pipeline_app", None),
    ("commands.cli_gold", "gold_app", "gold"),
)

MOUNTED: list[str] = []
MISSING: list[str] = []

for _mod, _attr, _name in _OPTIONAL:
    try:
        _m = importlib.import_module(f"acr.{_mod}")
    except ImportError:
        MISSING.append(_name or _mod)
        continue
    app.add_typer(getattr(_m, _attr), **({"name": _name} if _name else {}))
    MOUNTED.append(_name or _mod)


if __name__ == "__main__":
    app()
