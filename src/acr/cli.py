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

import typer

from .cli_attribute import attribute_app
from .cli_audit import audit_app
from .cli_chart import chart_app
from .cli_common import CONCORD_SCHEMA, EXPLAIN_SCHEMA, EXTRACT_SCHEMA  # noqa: F401 (re-export)
from .cli_eval import eval_app
from .cli_evaluation import evaluation_app
from .cli_gold import gold_app
from .cli_judge import judge_app
from .cli_label import label_app
from .cli_pipeline import _variable_records, pipeline_app, read_cohort  # noqa: F401 (re-export)
from .cli_plan import plan_app
from .cli_refine import refine_app
from .cli_repair import repair_app
from .cli_signal import signal_app
from .cli_site_mapping import site_mapping_app
from .cli_spec import spec_app

app = typer.Typer(add_completion=False, help="Agentic EHR chart review.")

app.add_typer(chart_app)
app.add_typer(pipeline_app)
app.add_typer(plan_app)
app.add_typer(spec_app, name="spec")

# The Site Mapping is neither a chart run nor a clinical decision: it is the local-name to
# document-concept table that `doc_type_matches` used to approximate with a substring. Its own
# group because three of its four commands read no chart and spend nothing, and because the
# one that spends runs once per corpus rather than once per run.
app.add_typer(site_mapping_app, name="site-mapping")

# ------------------------------------------------------------------------ development plane
# `assets` does not run the agent and does not read a chart: it develops the retrieval assets a
# run depends on — the keyword lists and the strata — against a complete per-note labelling of
# a small dev set, and refuses to certify one on data the search has seen. Mounted here rather
# than kept as a private entry point because the loop it implements (measure -> propose ->
# evolve -> certify -> adopt) is the only way anything in `specs/` stops being a guess, and a
# development tool nobody can find is a development tool nobody runs.
from .assetdev import assets_app

# `derive` is the FIRST-ORDER member of that same family, and it comes before `assets` in the
# order anyone should use them: count what the labelling already says, price it by grep, cut
# the list, propose the read policy. `assets` hill-climbs, which only refines a list that
# already exists. Both are mounted because the search is still worth running afterwards; the
# derivation is what makes there be something to refine.
from .derive import derive_app

app.add_typer(derive_app, name="derive")
app.add_typer(assets_app, name="assets")

# `label` comes before both of them in the order of the loop and did not exist as a command
# until now: `derive` reads a labels.jsonl that only the full scan can produce, so the first
# step of the develop plane was the one step nobody could take.
app.add_typer(label_app, name="label")
app.add_typer(refine_app, name="refine")
app.add_typer(gold_app, name="gold")
app.add_typer(repair_app, name="repair")

# ------------------------------------------------------------------------------- eval plane
# `eval` before `judge` deliberately. `eval dimensions` prints the precedence fence, and the
# fence is the thing to read before reaching for a judge: where a deterministic evaluator
# exists a judged opinion is refused, and no flag on `acr judge` will change that.
app.add_typer(eval_app, name="eval")
# `signal` is the one door to a signal about a completed run, whichever way it is produced. It
# is a group of its own rather than `acr eval --kind agent` because `eval` promises it reaches
# no model, and a group cannot keep that promise and also host the diagnostic agent.
app.add_typer(signal_app, name="signal")
app.add_typer(judge_app, name="judge")
app.add_typer(evaluation_app, name="evaluation")
app.add_typer(audit_app, name="audit")
# Attribution is model-based and may read the same patient's chart, so it cannot live under
# `eval`, whose import closure and CLI contract guarantee that no model is reachable.
app.add_typer(attribute_app, name="attribute")


if __name__ == "__main__":
    app()
