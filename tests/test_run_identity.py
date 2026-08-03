"""A manifest read on its own must say what produced it.

WHAT WAS MISSING AND WHY IT MATTERED. `code_sha()` (`core/cli_common.py:49`) has existed since
early on, and it reached exactly two places: the run DIRECTORY NAME, and three pipeline
artifacts. So a manifest moved, copied, or read out of its parent directory carried no code
identity at all — and `agent.py` names the defect in its own comment while not fixing it:
"`292dc90-dirty` is not a reproducible code identity."

Three identities are needed and they answer three different questions:

    code_sha        which code ran. A run under edited code is not the same experiment.
    chart_hash      which DOCUMENTS were read. `tools/generate_corpus.py` is deterministic,
                    so a chart that changes content under a stable patient_id is an edit
                    somebody made — and after 2026-08-03 the held-out adversarial charts are
                    exactly the files where a quiet edit would destroy the result.
    experiment_config_hash
                    which ARM. One value over spec + profile + every prompt asset + the tool
                    surface + seed + model, so two runs that should be comparable can be
                    checked for it rather than assumed to be.

The third exists because the alternative is a reader comparing six fields by eye and getting
it right every time. `tools/analyze_arms.py` already refuses to print across mixed spec
hashes; this is that refusal made general.

THE TOOL SURFACE IS PROMPT CONTENT AND NOTHING HASHED IT. `prompt_asset_manifest` covered the
value domain, the document concepts and the skill cards — not the tool schemas, whose
descriptions are rendered into every model call. `submit_answer`'s description is now built
from the contract's declared outcome space, so it genuinely differs between contracts, and
two runs could differ by a whole tool with no artifact able to tell them apart. That is the
same defect `prompt_assets` was added on 2026-07-30 to fix, one block further along.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from acr.chartstore.corpus import Corpus
from acr.contract.spec import load_spec
from acr.review.run_manifest import chart_hash, experiment_config_hash, prompt_asset_manifest
from acr.review.tools.toolbox import build_tool_schemas

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "patients"
SPEC_390 = ROOT / "assets" / "specs" / "STORE.390.date_of_initial_diagnosis.yaml"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SPEC_390)


# --------------------------------------------------------------- the chart hash

def test_the_chart_hash_covers_content_not_only_names(tmp_path):
    d = tmp_path / "P1"
    d.mkdir()
    (d / "Onc-Med-MD-OP-Progress-Note_2020-01-01.txt").write_text("a", encoding="utf-8")
    before = chart_hash(d)
    (d / "Onc-Med-MD-OP-Progress-Note_2020-01-01.txt").write_text("b", encoding="utf-8")
    assert chart_hash(d) != before, "editing a document must move the hash"


def test_the_chart_hash_covers_names_not_only_content(tmp_path):
    d = tmp_path / "P1"
    d.mkdir()
    (d / "Onc-Med-MD-OP-Progress-Note_2020-01-01.txt").write_text("a", encoding="utf-8")
    before = chart_hash(d)
    (d / "Onc-Med-MD-OP-Progress-Note_2020-01-01.txt").rename(
        d / "Onc-Med-MD-OP-Progress-Note_2021-01-01.txt")
    assert chart_hash(d) != before, "moving a document in time must move the hash"


def test_the_chart_hash_is_stable_across_calls():
    a = chart_hash(CORPUS / "SYN0002")
    assert a == chart_hash(CORPUS / "SYN0002")
    assert a != chart_hash(CORPUS / "SYN0003")


def test_a_missing_chart_hashes_to_nothing_rather_than_raising(tmp_path):
    """A manifest must not fail to be written because a hash could not be taken."""
    assert chart_hash(tmp_path / "nope") == ""


# --------------------------------------------------------------- the tool surface

def test_the_tool_surface_is_hashed(spec):
    block = prompt_asset_manifest(spec, tool_schemas=build_tool_schemas(spec))
    ts = block["tool_surface"]
    assert ts["names"] == [s["function"]["name"] for s in build_tool_schemas(spec)]
    assert len(ts["content_hash"]) == 16


def test_a_contract_with_a_wider_outcome_space_gets_a_different_tool_hash(spec):
    """STORE.390 declares CORPUS_INSUFFICIENT; the default space does not.

    `submit_answer`'s enum AND its description are built from the contract, so the surface
    genuinely differs — and before this hash existed, nothing in either manifest said so.
    """
    with_spec = prompt_asset_manifest(spec, tool_schemas=build_tool_schemas(spec))
    without = prompt_asset_manifest(spec, tool_schemas=build_tool_schemas(None))
    assert (with_spec["tool_surface"]["content_hash"]
            != without["tool_surface"]["content_hash"])


def test_an_absent_tool_surface_is_recorded_as_absent_not_omitted(spec):
    """`mcp_server` builds its own answer dict. An absent key and an unhashed surface are
    different facts and must not share a shape."""
    assert prompt_asset_manifest(spec)["tool_surface"] is None


# --------------------------------------------------------------- the config hash

def _cfg(spec, **over):
    base = {"spec_hash": spec.spec_hash, "runtime_profile_hash": "rp1",
            "prompt_assets": {"skills": []}, "seed": 1234, "model": "m"}
    return experiment_config_hash({**base, **over})


def test_the_config_hash_moves_on_every_axis_an_arm_can_vary(spec):
    base = _cfg(spec)
    assert _cfg(spec, seed=1235) != base, "seed"
    assert _cfg(spec, model="other") != base, "model"
    assert _cfg(spec, runtime_profile_hash="rp2") != base, "runtime profile"
    assert _cfg(spec, prompt_assets={"skills": [{"skill": "x"}]}) != base, "prompt assets"
    assert _cfg(spec, spec_hash="deadbeef") != base, "contract"


def test_the_config_hash_is_order_and_whitespace_stable(spec):
    """Two dicts describing one arm must hash alike, or the guard fires on nothing."""
    a = experiment_config_hash({"a": 1, "b": {"x": [1, 2]}})
    b = experiment_config_hash({"b": {"x": [1, 2]}, "a": 1})
    assert a == b


def test_the_config_hash_ignores_nothing_it_was_given(spec):
    """No allowlist. A field the caller thought worth passing is part of the arm."""
    assert experiment_config_hash({"a": 1}) != experiment_config_hash({"a": 1, "b": 2})


# --------------------------------------------------------------- the manifest carries them

def test_the_runtime_writes_all_three_into_every_manifest():
    """Structural: the keys are assembled where the manifest is, not by a caller.

    A manifest key set by whichever CLI happened to call the runtime is a key that is present
    for `acr batch` and absent for `acr extract`, which is how a cohort ends up unreproducible
    while every individual run looks fine.
    """
    import inspect

    import acr.review.agent as A
    src = inspect.getsource(A.run_chart_review)
    for key in ('"code_sha"', '"chart_hash"', '"experiment_config_hash"'):
        assert key in src, f"{key} is not written by run_chart_review"


def test_a_case_refused_before_it_read_anything_still_says_what_produced_it():
    """The other manifest-writing path.

    A refused case carrying fewer identity keys than a run that ran is a manifest every
    directory-wide reader has to special-case, and the first reader to forget drops it from a
    denominator without saying so.
    """
    import inspect

    import acr.review.agent as A
    src = inspect.getsource(A._case_refused_manifest)
    for key in ('"code_sha"', '"chart_hash"'):
        assert key in src, f"{key} is missing from the refused-case manifest"


def test_a_recorded_manifest_can_be_checked_against_the_tree():
    """The point of the chart hash, stated as the operation it enables."""
    fake = {"patient_id": "SYN0002", "chart_hash": chart_hash(CORPUS / "SYN0002")}
    assert fake["chart_hash"] == chart_hash(CORPUS / fake["patient_id"])
    tampered = json.loads(json.dumps({**fake, "chart_hash": "0" * 16}))
    assert tampered["chart_hash"] != chart_hash(CORPUS / tampered["patient_id"])


def test_every_chart_in_the_corpus_hashes():
    ids = Corpus(CORPUS).patient_ids()
    assert len(ids) >= 21
    assert all(chart_hash(CORPUS / p) for p in ids)
