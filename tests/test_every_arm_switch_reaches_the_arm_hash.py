"""Two arms that differ must not hash alike, and the next switch must not be able to slip.

`experiment_config_hash` is now the read side's discriminator: since 2026-08-04 `evals.BaselineKey`
carries it and `eval compare` reports it as the part of the arm that moved. That makes every gap in
its INPUTS a gap in the comparison — `key_differences: []` for two runs that genuinely differ, which
is the exact failure the hash was introduced to end, one layer down.

MEASURED, in this tree's own `runs/phaseA`: `Arm0-single-loop` and `Arm1-candidates` carry the same
`experiment_config_hash` (`801fb23df6124fa5`) while `Arm1` declared a candidate and made a candidate-
reasoner call and `Arm0` made none. Two arms, one hash, and a reader comparing them is told they are
the same configuration.

`experiment_config_hash`'s own docstring says there is NO ALLOWLIST inside it and that "the caller
assembles the dict at the point where it knows what varied". The caller then wrote a fixed list of
nine keys, and three switches added to `run_patient` afterwards never joined it:

  * `additional_task_context` — appended verbatim to the system prompt (`agent.py:1688`) and recorded
    in no manifest field at all. Its only caller is `conflict_refinement.py:311`, which injects the
    "OPTIONAL CONFLICT-REFINEMENT BRIEF" — so `--conflict-refine` was an arm whose distinguishing
    prompt content was hashed nowhere.
  * `site_mapping` — reaches `CoverageLedger` and `plan_from_spec`, so it changes the stratification,
    the plan and therefore the gate. `SiteMapping.mapping_hash` has existed all along and reached no
    manifest.
  * `max_usd` — the priced ceiling. A run stopped for spend and a run that finished are not the same
    arm, and `spend.max_usd` was recorded but not hashed.

## The registry is the point, not the three fixes

Three omissions patched by hand would leave the fourth to the next person. `ARM_PARAMETERS` and
`WITHIN_ARM_PARAMETERS` in `agent.py` classify EVERY parameter of `run_patient`, and the test below
fails when a new one is neither. Same shape as `audit_loop.BASIS_REPORTERS` and `skills.SLOTS`: the
decision has to be typed somewhere a reader can see it.

The second list matters as much as the first. `case` and `expansion_budget` must stay OUT: `case`
carries `patient_id` and `latest_document_date`, and `expansion_budget` is priced against the
patient's own plan when the caller supplies none. Hashing either would make `experiment_config_hash`
a per-run id, and a paired comparison would then find that every patient is its own arm.
"""

from __future__ import annotations

import inspect

import pytest

from acr.contract.spec import load_spec
from acr.core import site
from acr.review.agent import ARM_PARAMETERS, WITHIN_ARM_PARAMETERS, run_patient
from acr.review.run_manifest import experiment_config_hash, prompt_asset_manifest

SPEC_390 = site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC_390)


# ------------------------------------------------------------------ the registry covers everything

def test_every_run_patient_parameter_is_classified():
    """The guard that catches the NEXT switch, not the three that are already wrong.

    A parameter in neither list is a switch nobody decided about, and the default outcome of not
    deciding is that two arms hash alike — silently, and only visibly months later when a
    comparison reports `key_differences: []` across a real difference.
    """
    params = set(inspect.signature(run_patient).parameters)
    classified = set(ARM_PARAMETERS) | set(WITHIN_ARM_PARAMETERS)
    assert params - classified == set(), (
        "unclassified `run_patient` parameter(s): add each to ARM_PARAMETERS (with the manifest key "
        "it reaches) or to WITHIN_ARM_PARAMETERS (with the reason it must not be hashed)")
    assert classified - params == set(), "the registry names parameters that no longer exist"


def test_no_parameter_is_in_both_lists():
    assert set(ARM_PARAMETERS) & set(WITHIN_ARM_PARAMETERS) == set()


def test_every_within_arm_exclusion_states_a_reason():
    """An exclusion with an empty reason is an omission wearing a registry entry."""
    for name, why in WITHIN_ARM_PARAMETERS.items():
        assert len(why) > 30, f"{name}: give the reason it must not be hashed"


def test_the_two_parameters_that_would_make_the_hash_a_per_run_id_are_excluded():
    """Named individually because both are tempting and both are wrong.

    `case` carries `patient_id` and `latest_document_date`; `expansion_budget` is priced against the
    patient's own plan when the caller supplies none (`agent.py:1707`). Hashing either turns the arm
    hash into a per-run id, after which a paired comparison finds every patient in its own arm and
    `eval compare` refuses every real pairing as a mixture.
    """
    for name in ("case", "expansion_budget"):
        assert name in WITHIN_ARM_PARAMETERS, name
        assert "per-run" in WITHIN_ARM_PARAMETERS[name] or "patient" in WITHIN_ARM_PARAMETERS[name]


def test_every_arm_parameter_names_the_manifest_key_it_travels_through():
    """A classification with no route is a claim, not a mechanism."""
    for name, key in ARM_PARAMETERS.items():
        assert key, f"{name}: name the manifest key through which it reaches the hash"


# ------------------------------------------------------------------ the three missing switches

def test_the_conflict_refinement_brief_is_hashed(spec):
    """Its only caller injects the brief that DEFINES the refinement arm. Nothing recorded it, so a
    refinement run and a baseline run carried identical prompt-asset manifests."""
    brief = "OPTIONAL CONFLICT-REFINEMENT BRIEF\ncytology vs biopsy"
    with_brief = prompt_asset_manifest(spec, task_context=brief)
    without = prompt_asset_manifest(spec)
    assert without["additional_task_context"] is None, (
        "an absent brief must be an explicit None, not a missing key: a reader cannot tell "
        "'no brief' from 'this manifest predates the field' otherwise")
    entry = with_brief["additional_task_context"]
    assert entry["n_chars"] == len(brief)
    assert entry["content_hash"] and entry["content_hash"] != ""
    assert brief not in str(with_brief), (
        "the brief itself must not be copied into the manifest — it is unbounded operator text "
        "and this file is written beside patient-derived output")


def test_two_different_briefs_do_not_share_a_hash(spec):
    a = prompt_asset_manifest(spec, task_context="brief A")["additional_task_context"]
    b = prompt_asset_manifest(spec, task_context="brief B")["additional_task_context"]
    assert a["content_hash"] != b["content_hash"]


def test_the_site_mapping_reaches_the_prompt_asset_manifest(spec):
    """`mapping_hash` has existed since the mapping did and reached no manifest, so an arm run with
    `--mapping` and one run without recorded the same identity while their plans differed."""
    from acr.contract.site_mapping import (
        Concept,
        SiteMapping,
        TypeAssignment,
        concepts_hash,
    )
    concepts = [Concept("can_establish", "states the diagnosis")]

    def mapping(concept: str) -> SiteMapping:
        return SiteMapping(
            corpus_id="acr_real", concepts=tuple(concepts),
            bound_concepts_hash=concepts_hash(concepts),
            assignments={"Surgical-Pathology-Report":
                         TypeAssignment("Surgical-Pathology-Report", concept, "x", 3)},
            model="test-model", built_at="2026-08-04T00:00:00Z")

    m = mapping("can_establish")
    entry = prompt_asset_manifest(spec, site_mapping=m)["site_mapping"]
    assert entry["mapping_hash"] == m.mapping_hash
    assert entry["n_types"] == 1 and entry["corpus_id"] == "acr_real"
    assert prompt_asset_manifest(spec)["site_mapping"] is None

    # Two mappings that place the SAME type differently are two arms, and the recorded identity has
    # to move with them. A block carrying only `corpus_id` would tie every mapping over one corpus.
    other = prompt_asset_manifest(spec, site_mapping=mapping("UNMAPPED"))["site_mapping"]
    assert other["mapping_hash"] != entry["mapping_hash"]


# ------------------------------------------------------------------ and the hash actually moves

def _cfg(spec, **over):
    base = {"spec_hash": spec.spec_hash, "runtime_profile_hash": "rp1",
            "prompt_assets": {"skills": [], "additional_task_context": None,
                              "site_mapping": None},
            "sample_seed": 1234, "model": "m", "max_usd": 5.0}
    return experiment_config_hash({**base, **over})


def test_the_hash_moves_on_each_of_the_three_switches_that_were_missing(spec):
    base = _cfg(spec)
    assert _cfg(spec, max_usd=1.0) != base, "the priced ceiling"
    assert _cfg(spec, prompt_assets={"skills": [], "site_mapping": None,
                                     "additional_task_context": {"n_chars": 7,
                                                                 "content_hash": "ab"}}) != base, \
        "the conflict-refinement brief"
    assert _cfg(spec, prompt_assets={"skills": [], "additional_task_context": None,
                                     "site_mapping": {"mapping_hash": "cd"}}) != base, \
        "the site mapping"


# ------------------------------------------------------------------ the model identity itself

def test_a_model_with_no_name_still_has_a_stable_identity():
    """FOUND BY THE TEST BELOW, and it was a live per-run id in the field the arm hash leans on.

    `manifest["model"]` was `getattr(model, "model_name", "") or str(model)`, and `str()` on a
    LangChain chat model without `model_name` renders the object repr — memory address included. Two
    runs of one arm therefore recorded different models and hashed differently, so once
    `experiment_config_hash` became the read side's discriminator, `eval compare` would have refused
    every genuine pairing as a mixture.

    No hand-built fixture could catch it: they all type a model name, so the fallback never runs.
    """
    from acr.review.run_manifest import model_identity

    class Nameless:
        pass

    a, b = Nameless(), Nameless()
    assert model_identity(a) == model_identity(b) == "Nameless"
    assert "0x" not in model_identity(a), "an object address is not a model identity"

    class Named:
        model_name = "gpt-5.6-luna"

    assert model_identity(Named()) == "gpt-5.6-luna", "a real provider id still wins"


def test_no_call_site_reconstructs_the_model_identity_by_hand():
    """Three call sites read the same fallback and all three were wrong. One function now, so the
    next reader cannot fix `manifest["model"]` and leave `spend.model` reporting an address."""
    import acr.review.agent as A
    src = inspect.getsource(A)
    assert 'or str(model)' not in src, (
        "a second model-identity expression is a second answer to one question")
    assert src.count("model_identity(model)") >= 3


# ------------------------------------------------------------------ end to end, no model, no cost

def _two_runs(tmp_path, **switch):
    """The same patient twice, differing by ONE switch, through the real runtime.

    Reading the assembly by eye is what let three switches sit outside the hash. This runs it: real
    graph, real middleware, real toolbox, real ledgers, real gate, a scripted provider in place of
    the completions. Both manifests are the ones the runtime wrote.
    """
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from hooks_harness import run_with_script
    from test_provenance import SHB, _ScriptedLLM

    from acr.chartstore.corpus import Corpus
    corpus, sp = Corpus(site.corpus_root()), load_spec(SHB)
    out = []
    for name, kw in (("base", {}), ("arm", switch)):
        d = tmp_path / name
        d.mkdir()
        llm = _ScriptedLLM({"primary_site": "C341", "histology": "8140", "behavior": "3"})
        m, _ = run_with_script(sp, corpus, "SYN0001", d, llm,
                               run_id=name, max_model_calls=8, **kw)
        out.append(m)
    return out


def test_a_task_context_changes_the_arm_hash_of_a_real_run(tmp_path):
    """`--conflict-refine` injects a brief through this parameter and nothing else distinguishes its
    arm. Two real runs, one brief: before this, both manifests carried the same hash."""
    base, arm = _two_runs(tmp_path, additional_task_context="OPTIONAL CONFLICT-REFINEMENT BRIEF")
    assert base["prompt_assets"]["additional_task_context"] is None
    assert arm["prompt_assets"]["additional_task_context"]["n_chars"] == 34
    assert base["experiment_config_hash"] != arm["experiment_config_hash"]


def test_a_ceiling_changes_the_arm_hash_of_a_real_run(tmp_path):
    base, arm = _two_runs(tmp_path, max_usd=0.5)
    assert base["experiment_config_hash"] != arm["experiment_config_hash"]
    assert base["spend"]["max_usd"] != arm["spend"]["max_usd"]


def test_two_runs_of_one_arm_still_share_a_hash(tmp_path):
    """THE PROPERTY THE WHOLE THING RESTS ON, and the one a careless addition breaks. A hash that
    moves between two runs of one arm is a per-run id, after which `eval compare` refuses every real
    pairing as a mixture and the read side is worse off than before it could see the field."""
    a, b = _two_runs(tmp_path)
    assert a["experiment_config_hash"] == b["experiment_config_hash"]


def test_the_runtime_puts_the_ceiling_and_both_assets_into_the_hash_input():
    """Structural, on the one place the input dict is assembled. `run_chart_review` is where the
    manifest is built, so the hash's inputs have to be visible there rather than in whichever CLI
    happened to call it — the reason `code_sha` moved into this function in the first place.
    """
    import acr.review.agent as A
    src = inspect.getsource(A.run_chart_review)
    head = src.split('manifest["experiment_config_hash"]')[1]
    for token in ('"max_usd"', '"prompt_assets"'):
        assert token in head, f"{token} is not an input to experiment_config_hash"
    assert "site_mapping=" in src and "task_context=" in src, (
        "run_chart_review must pass both to prompt_asset_manifest, or the prompt_assets it hashes "
        "cannot contain them")
