"""The retrieval plan and the coverage policy were one flag, so no arm can say which one worked.

`--runtime-profile` moved two independent things at once:

    plan  = plan_from_spec(...)  if starts_with_coverage_assets(profile) else
            plan_from_patient_inventory(...)                     # WHERE THE RUN STARTS LOOKING
    coverage_state["active"] = starts_with_coverage_assets(profile)   # WHETHER COVERAGE IS ENFORCED

`plan_from_spec` is the spec's hand-written strata — document types sorted into read_all / search /
sample, plus the declared keywords. `plan_from_patient_inventory` hands over every type this patient
has and no keywords at all. Those are not two settings of one dial; the first is a supplied prior
about where to look and the second is its absence.

So every arm this repo has ever run varied "better plan" and "stricter policy" TOGETHER.
`tools/run_floor.py`'s own header says "TWO ARMS, one variable: the runtime profile" — and names both
halves in the same breath: the floor arm gets `guideline-only`, which simultaneously drops the strata
plan, drops the five keywords, turns coverage off from call 0 and switches the spec view. A
difference in accuracy between those arms is attributable to none of the four.

`plan_from_spec`'s own docstring calls itself "the arm the develop plane wants to falsify", which
requires holding it while something else moves. Nothing could hold it.

## What this changes, and what it deliberately does not

`--planner` defaults to `profile`, so every recorded run's behaviour is unchanged and no baseline
becomes unreproducible. It selects ONLY which plan the run starts from. Coverage activation, the
positive terms fed to the runtime policy, and the spec view stay on `--runtime-profile`: that is the
separation, and folding any of them in would rebuild the confound under a new name.

The manifest records the resolved planner AND how it was resolved, on the `seed_provenance`
precedent — a reader who cannot tell an explicit choice from an inherited default cannot tell a
reproduced arm from a shopped one.
"""

from __future__ import annotations

import inspect

import pytest

from acr.contract.spec import load_spec
from acr.core import site
from acr.review.agent import ARM_PARAMETERS, PLANNERS, resolve_planner, run_patient

SPEC_390 = site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC_390)


# ------------------------------------------------------------------ the registry

def test_both_planners_are_named_and_reachable():
    """A named registry, not two branches of an `if`. The names are what an arm records."""
    assert set(PLANNERS) == {"spec-strata", "patient-inventory"}


def test_the_default_defers_to_the_profile_and_says_so():
    """Every recorded run took the profile's choice, so the default must reproduce it exactly."""
    for profile, expected in (("current-stratified-coverage", "spec-strata"),
                              ("guideline-only", "patient-inventory")):
        name, how = resolve_planner("", profile)
        assert name == expected, profile
        assert how == "runtime_profile"


def test_an_explicit_planner_wins_and_is_recorded_as_explicit():
    for profile in ("current-stratified-coverage", "guideline-only"):
        assert resolve_planner("patient-inventory", profile) == ("patient-inventory", "explicit")
        assert resolve_planner("spec-strata", profile) == ("spec-strata", "explicit")


def test_an_unknown_planner_refuses_and_names_the_options():
    with pytest.raises(ValueError, match="spec-strata"):
        resolve_planner("whatever-i-typed", "guideline-only")


# ------------------------------------------------------------------ the axis is separable

def test_the_planner_and_the_coverage_policy_can_now_disagree(tmp_path):
    """THE ARM THAT COULD NOT EXIST. `guideline-only` with the spec's strata plan: the prior about
    where to look is supplied, and coverage is still off from the first model call. Without this,
    "the plan helped" and "the gate helped" are one measurement.
    """
    base, arm = _two_runs(tmp_path, runtime_profile="guideline-only",
                          switch={"planner": "spec-strata"})
    # The policy half is untouched by the planner: both runs have coverage inactive because the
    # PROFILE says so. If this ever flips, the confound is back under a different flag.
    assert base["coverage_activation"]["active"] is False
    assert arm["coverage_activation"]["active"] is False
    # And the plan half moved.
    assert base["planner"]["name"] == "patient-inventory"
    assert arm["planner"]["name"] == "spec-strata"
    assert arm["plan"] != base["plan"]


def test_the_strata_planner_supplies_keywords_the_inventory_planner_does_not(tmp_path):
    """What the two planners actually differ by, asserted rather than assumed: the spec's declared
    keywords reach the plan in one and not the other. That is the prior under measurement."""
    base, arm = _two_runs(tmp_path, runtime_profile="guideline-only",
                          switch={"planner": "spec-strata"})
    assert not (base["plan"].get("keywords") or []), "the inventory plan is the absence of a prior"
    assert arm["plan"].get("keywords"), "the strata plan carries the spec's own terms"


def test_the_manifest_says_how_the_planner_was_chosen(tmp_path):
    """`seed_provenance`'s precedent. A reader who cannot tell an explicit choice from an inherited
    default cannot tell a reproduced arm from a shopped one."""
    base, arm = _two_runs(tmp_path, runtime_profile="guideline-only",
                          switch={"planner": "patient-inventory"})
    assert base["planner"] == {"name": "patient-inventory", "provenance": "runtime_profile",
                              "runtime_profile": "guideline-only"}
    # Same plan, different provenance: asking for what you would have got is not a no-op in the
    # record, because it is the difference between a choice and an accident.
    assert arm["planner"]["provenance"] == "explicit"
    assert arm["planner"]["name"] == base["planner"]["name"]


# ------------------------------------------------------------------ it is an arm, so it is hashed

def test_the_planner_is_a_classified_arm_parameter():
    assert ARM_PARAMETERS.get("planner"), (
        "a switch that changes where a run starts looking is part of the arm; unclassified, it "
        "would hash identically to the baseline — the defect `ARM_PARAMETERS` exists to stop")
    assert "planner" in inspect.signature(run_patient).parameters


def test_two_planners_do_not_share_an_arm_hash(tmp_path):
    base, arm = _two_runs(tmp_path, runtime_profile="guideline-only",
                          switch={"planner": "spec-strata"})
    assert base["experiment_config_hash"] != arm["experiment_config_hash"]


def test_asking_for_the_profiles_own_planner_still_changes_the_arm_hash(tmp_path):
    """A DELIBERATE CHOICE, and the honest reading is that it is a different arm.

    The plan is byte-identical, so this could be argued either way. It hashes differently because
    `planner` is recorded with its provenance and the hash covers the manifest's identity: an arm
    where the planner was pinned is reproducible under a profile whose default later changes, and an
    arm that inherited it is not. Two runs that would diverge under a future edit are not one arm.
    """
    base, arm = _two_runs(tmp_path, runtime_profile="guideline-only",
                          switch={"planner": "patient-inventory"})
    assert base["plan"] == arm["plan"]
    assert base["experiment_config_hash"] != arm["experiment_config_hash"]


# ------------------------------------------------------------------ reachable from the commands

def test_every_run_command_exposes_the_flag():
    from acr.commands.cli_chart import batch, consistency, run
    from acr.commands.cli_pipeline import extract
    for fn in (run, batch, consistency, extract):
        assert "planner" in inspect.signature(fn).parameters, fn.__name__


def test_the_flag_refuses_a_bad_value_before_any_model_call():
    import typer

    from acr.commands.cli_chart import _planner
    assert _planner("") == ""
    assert _planner("spec-strata") == "spec-strata"
    with pytest.raises(typer.BadParameter, match="spec-strata"):
        _planner("stratta")


# ------------------------------------------------------------------ helper

def _two_runs(tmp_path, *, runtime_profile: str, switch: dict):
    """The same patient twice through the real runtime, differing by one switch. No model, no cost."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from hooks_harness import run_with_script
    from test_provenance import SHB, _ScriptedLLM

    from acr.chartstore.corpus import Corpus
    corpus, sp = Corpus(site.corpus_root()), load_spec(SHB)
    out = []
    for name, kw in (("base", {}), ("arm", switch)):
        d = tmp_path / name
        d.mkdir()
        llm = _ScriptedLLM({"primary_site": "C341", "histology": "8140", "behavior": "3"})
        m, _ = run_with_script(sp, corpus, "SYN0001", d, llm, run_id=name, max_model_calls=8,
                               runtime_profile=runtime_profile, **kw)
        out.append(m)
    return out


# ------------------------------------------------------------------ and whether it varies at all

def test_the_axis_is_reported_live_or_inert_before_anybody_pays_for_it():
    """MEASURED, AND IT CHANGES WHAT THE REPO BELIEVES ABOUT ITS OWN FLOOR EXPERIMENT.

    `plan_from_spec` degenerates when a contract declares no strata: every type falls to `search`,
    which is exactly what `plan_from_patient_inventory` produces. On `STORE.390` the two plans differ
    only in a `source` label and a per-type `rationale` string — the retrieval surface is identical.

    Every ladder and floor run in this tree used STORE.390. So `tools/run_floor.py`'s "prior versus
    floor" arms, whose whole claim is that the spec's hand-written plan is a supplied prior worth
    falsifying, were comparing two behaviourally identical plans: the entire measured difference came
    from the POLICY half of `--runtime-profile` — coverage activation and the spec view — and none of
    it from the plan.

    Splitting the flag was necessary and is not sufficient. A driver that spends $108 on an axis with
    nothing to vary has bought nothing, so the axis says whether it is live.
    """
    from acr.chartstore.corpus import Corpus
    from acr.review.coverage_planner import planner_axis_is_live

    chart = Corpus(site.corpus_root()).chart("SYN0001")
    inert = planner_axis_is_live(load_spec(SPEC_390), chart)
    assert inert["live"] is False
    assert "no strata" in inert["why"] or "declares no" in inert["why"]

    live = planner_axis_is_live(
        load_spec(site.specs_root() / "STORE.400_522_523.site_histology_behavior.yaml"), chart)
    assert live["live"] is True
    assert live["differs_in"], "a live axis must name what actually moves"


def test_the_label_alone_does_not_count_as_a_live_axis():
    """`source` and `rationale` differ on every spec, including the ones where the surface does not.
    Counting them would report every axis live and the check would be worthless."""
    from acr.chartstore.corpus import Corpus
    from acr.review.coverage_planner import (
        plan_from_patient_inventory,
        plan_from_spec,
        planner_axis_is_live,
    )

    chart = Corpus(site.corpus_root()).chart("SYN0001")
    spec = load_spec(SPEC_390)
    a, b = plan_from_spec(spec, chart).to_dict(), plan_from_patient_inventory(spec, chart).to_dict()
    assert a != b, "the two dicts DO differ — on the label"
    assert a["source"] != b["source"]
    assert planner_axis_is_live(spec, chart)["live"] is False
