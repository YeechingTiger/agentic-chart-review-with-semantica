"""The retrieval plan told the model three rules, and none of the three exists any more.

`CoveragePlan.render()` reaches the model on **every** model call — `wrap_model_call` rebuilds the
system message with `"PLAN (current):\\n" + plan.render(...)`. It said:

    "RETRIEVAL PLAN — this is the plan, and it governs what you may open."
    "SAMPLED BY THE RUNTIME — you may NOT open these directly."
    "SEARCH TERMS (every one of these must actually be run before a negative is allowed):"

All three enforcements were removed after measurement:

  * `AuditMiddleware._out_of_plan` — the out-of-plan read refusal — removed 2026-07-30. It fired
    138 times, and the bucket it enforced came from a substring matcher that matched
    `Speech-Language-Pathology-Note` and missed `Non-Gyn-Cyto-FNA` (1,285 documents),
    `FN-Aspirate-Report` (881) and `SURG-PATH-RESULT` (231).
  * the required-search refusals left `check_gate`, and `evaluate_gate` runs with `enforce=False`
    with no production call site passing `True` — so an unsearched required term is an advisory
    line in a `verdict: PASS`.

A false rule in the prompt is not untidiness. It is an **uncontrolled intervention**: the model is
being told a constraint that does not exist, on every call, in the arm that is supposed to be the
baseline. Whether telling it changes behaviour is an empirical question nobody has asked; what is
not available is leaving it there and calling the result a base measurement.

## What replaces them

The plan becomes what the rest of this prompt already calls its own guidance — `document_concepts`
says "REFERENCE, NOT INSTRUCTIONS", `experience_block` says "not a rule". The one consequence that
IS enforced stays and is stated as a consequence: the gate refuses an abstention while forced
sampling draws are un-inspected (`answer_gate.py` `_coverage_verdict`).

## And the surface the model can actually reach

`prompt_assets.tool_surface` recorded seven tools. Nine are bound: `revise_plan` is added by
`run_chart_review` and `write_todos` by `TodoListMiddleware`. A manifest that understates the
reachable surface is read by `undeclared-tool-audit` and by anyone asking what this run could do.
"""

from __future__ import annotations

import pytest

from acr.contract.spec import load_spec
from acr.core import site

SPEC = site.specs_root() / "STORE.400_522_523.site_histology_behavior.yaml"


@pytest.fixture(scope="module")
def rendered():
    """A real plan for a real chart, rendered the way `wrap_model_call` renders it."""
    from acr.chartstore.corpus import Corpus
    from acr.review.coverage_planner import plan_from_spec
    chart = Corpus(site.corpus_root()).chart("SYN0001")
    plan = plan_from_spec(load_spec(SPEC), chart)
    assert plan.sample or plan.read_all, "this spec must declare strata for the test to mean anything"
    return plan.render({r["doc_type"]: r["count"] for r in chart.type_summary()})


# ------------------------------------------------------------------ the three false claims

def test_the_plan_does_not_claim_to_govern_what_may_be_opened(rendered):
    """`_out_of_plan` was removed 2026-07-30. Nothing refuses an out-of-plan read."""
    assert "governs what you may open" not in rendered


def test_the_plan_does_not_forbid_opening_a_sampled_type(rendered):
    """The refusal that enforced this was the same removed hook."""
    assert "may NOT open these directly" not in rendered
    assert "you may not open" not in rendered.lower()


def test_the_plan_does_not_claim_every_term_must_be_searched(rendered):
    """`evaluate_gate` is advisory: `enforce=False` and no production caller passes `True`."""
    assert "must actually be run" not in rendered
    assert "before a negative is allowed" not in rendered


# ------------------------------------------------------------------ what it says instead

def test_the_plan_says_it_is_guidance(rendered):
    """The same stance `document_concepts` and `experience_block` already take. A model cannot weigh
    guidance it has been told is a rule."""
    low = rendered.lower()
    assert "guidance" in low or "reference" in low
    assert "not a restriction" in low or "not a rule" in low


def test_the_one_enforced_consequence_is_still_stated(rendered):
    """The gate DOES refuse an abstention while forced-sampling draws are un-inspected. Dropping
    that sentence with the false ones would remove the only true thing the block said about
    enforcement — and the model would meet the refusal with no warning."""
    assert "sampler" in rendered.lower()
    assert "inspect" in rendered.lower() or "hands you the note_ids" in rendered


def test_the_strata_and_the_terms_are_still_there(rendered):
    """The block's actual job. Removing the claims must not remove the content."""
    assert "READ IN FULL" in rendered and "SEARCH" in rendered
    for token in ("Surgical-Pathology-Report",):
        assert token in rendered, f"{token} vanished from the rendered plan"


# ------------------------------------------------------------------ the reachable tool surface

def test_the_manifest_records_every_tool_the_model_can_reach(tmp_path):
    """Nine bound, seven recorded. `revise_plan` comes from `run_chart_review` and `write_todos`
    from `TodoListMiddleware`; both reach the model and neither was in the manifest."""
    pytest.importorskip("deepagents")
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from hooks_harness import ToolScript

    from acr.chartstore.corpus import Corpus
    from acr.review.agent import run_patient

    model = ToolScript(script=[], submit={"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                          "reasoning": "at once"})
    model.seen = []
    m = run_patient(spec=load_spec(SPEC), corpus=Corpus(site.corpus_root()),
                    patient_id="SYN0001", out_dir=tmp_path, model=model,
                    max_model_calls=1, seed=7, run_id="surface")
    ts = m["prompt_assets"]["tool_surface"]
    assert "revise_plan" in ts["names"], "the plan-revision tool is bound and was unrecorded"
    assert "write_todos" in ts["names"], "the library tool is bound and was unrecorded"
    assert ts["n_tools"] == len(ts["names"]) >= 9


def test_the_manifest_says_which_tools_it_could_not_content_hash(tmp_path):
    """AN ADMITTED LIMIT, not a silent one. Only the seven from `build_tool_schemas` have schemas
    here; `revise_plan`'s description is a module constant and `write_todos` belongs to the library,
    whose version is `code_sha`'s business. A reader has to be able to tell which names are hashed
    from which are merely listed — otherwise the hash looks like it covers nine."""
    pytest.importorskip("deepagents")
    from acr.review.run_manifest import prompt_asset_manifest
    from acr.review.tools.toolbox import build_tool_schemas
    spec = load_spec(SPEC)
    schemas = build_tool_schemas(spec)
    ts = prompt_asset_manifest(spec, tool_schemas=schemas,
                               bound_tool_names={"revise_plan", "write_todos",
                                                 *(s["function"]["name"] for s in schemas)})
    assert set(ts["tool_surface"]["not_schema_hashed"]) == {"revise_plan", "write_todos"}
    assert ts["tool_surface"]["n_schema_hashed"] == len(schemas)


def test_an_absent_surface_is_still_an_explicit_absence():
    """`mcp_server` builds its own answer dict; an absent surface and an unhashed one differ."""
    from acr.review.run_manifest import prompt_asset_manifest
    assert prompt_asset_manifest(load_spec(SPEC))["tool_surface"] is None
