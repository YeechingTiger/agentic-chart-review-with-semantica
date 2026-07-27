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

The last four had no entry point at all until now: roughly 4,250 lines of develop-plane and
eval-plane code with tests beside it and no way for anyone to invoke it, which is the same
condition as not existing.

A GROUP MOUNTED WITHOUT A NAME merges its commands into the top level (`acr extract`), and one
mounted WITH a name becomes a sub-command (`acr judge panel`). The first four groups are
nameless because their commands were top-level before this split and moving them would be a
behaviour change wearing a refactor's clothes.
"""
from __future__ import annotations

import typer

from .cli_chart import chart_app
from .cli_common import CONCORD_SCHEMA, EXPLAIN_SCHEMA, EXTRACT_SCHEMA  # noqa: F401 (re-export)
from .cli_eval import eval_app
from .cli_judge import judge_app
from .cli_label import label_app
from .cli_pipeline import _variable_records, pipeline_app, read_cohort  # noqa: F401 (re-export)
from .cli_plan import plan_app
from .cli_refine import refine_app
from .cli_spec import spec_app

app = typer.Typer(add_completion=False, help="Agentic EHR chart review.")

app.add_typer(chart_app)
app.add_typer(pipeline_app)
app.add_typer(plan_app)
app.add_typer(spec_app, name="spec")

# ------------------------------------------------------------------------ development plane
# `assets` does not run the agent and does not read a chart: it develops the retrieval assets a
# run depends on — the keyword lists and the strata — against a complete per-note labelling of
# a small dev set, and refuses to certify one on data the search has seen. Mounted here rather
# than kept as a private entry point because the loop it implements (measure -> propose ->
# evolve -> certify -> adopt) is the only way anything in `specs/` stops being a guess, and a
# development tool nobody can find is a development tool nobody runs.
from .assetdev import assets_app  # noqa: E402  (after `app`, so the group can attach)
# `derive` is the FIRST-ORDER member of that same family, and it comes before `assets` in the
# order anyone should use them: count what the labelling already says, price it by grep, cut
# the list, propose the read policy. `assets` hill-climbs, which only refines a list that
# already exists. Both are mounted because the search is still worth running afterwards; the
# derivation is what makes there be something to refine.
from .derive import derive_app  # noqa: E402

app.add_typer(derive_app, name="derive")
app.add_typer(assets_app, name="assets")

# `label` comes before both of them in the order of the loop and did not exist as a command
# until now: `derive` reads a labels.jsonl that only the full scan can produce, so the first
# step of the develop plane was the one step nobody could take.
app.add_typer(label_app, name="label")
app.add_typer(refine_app, name="refine")

# ------------------------------------------------------------------------------- eval plane
# `eval` before `judge` deliberately. `eval dimensions` prints the precedence fence, and the
# fence is the thing to read before reaching for a judge: where a deterministic evaluator
# exists a judged opinion is refused, and no flag on `acr judge` will change that.
app.add_typer(eval_app, name="eval")
app.add_typer(judge_app, name="judge")


if __name__ == "__main__":
    app()
