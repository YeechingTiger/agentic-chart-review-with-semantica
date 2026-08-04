"""Producer→consumer tests whose two halves ship in DIFFERENT distributions.

These belong here and not beside their subject because a producer→consumer test can only run where
both halves exist, and `tools/` ships only in `acr-chart-review`
(`tools/split_repos.py`'s `collect`). Left in `tests/test_incumbent_keywords_match_the_runtime.py`
they routed to `acr-improvement` and failed there with `ModuleNotFoundError: No module named
'build_termcache'` — seven failures across two distributions, in a changeset whose whole subject is
seams between planes.

`(composer)` is what this repository already calls the place for a property that is only true of the
composed tree. `tools/split_repos.py::TESTS_BY_SUBJECT` routes this file there explicitly.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

from acr.contract.spec import load_spec
from acr.contract.strata import spec_declared_keywords
from acr.core import site

ROOT = pathlib.Path(__file__).resolve().parents[1]
SPECS = sorted(site.specs_root().glob("*.yaml"))
TOOLS = ROOT / "tools"


def _norm(terms) -> set[str]:
    return {str(t).strip().lower() for t in terms if str(t).strip()}


@pytest.fixture(autouse=True)
def _require_tools():
    """`tools/` is present in the source tree and in `acr-chart-review`, absent elsewhere. Skipping
    is the honest outcome for a distribution that does not ship the producer — the composed run is
    where this property is actually checked."""
    if not TOOLS.is_dir():
        pytest.skip("this checkout does not ship tools/, so the producer is not here to test")
    sys.path.insert(0, str(TOOLS))


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.name)
def test_the_termcache_needles_cover_the_runtime_list(path):
    """A term the cache does not carry cannot be priced at all — `derive` refuses rather than
    rescanning, which is the right refusal and is why the needle list must be complete.

    `tools/build_termcache.py` is the producer; `acr.improvement.derive` is the consumer. They live
    in different distributions, which is the whole reason this test is in the composer.
    """
    from build_termcache import spec_incumbent
    spec = load_spec(path)
    assert _norm(spec_incumbent(str(path))) == _norm(spec_declared_keywords(spec))


def test_the_answer_key_producer_emits_spec_id():
    """`tools/answer_key_from_corpus.py` writes it; `evals._key_row` reads it. Without this pairing
    the bare-patient-id fallback matched a run of ANY contract and `n_unkeyed` was pinned at 0."""
    out = subprocess.run(
        [sys.executable, str(TOOLS / "answer_key_from_corpus.py"),
         "--spec-key", "STORE.390.date_of_initial_diagnosis",
         "--fields", "date_of_initial_diagnosis", "--out", "/dev/stdout"],
        capture_output=True, text=True, check=False, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    key = json.loads(out.stdout[:out.stdout.rindex("}") + 1])
    row = next(iter(key.values()))
    assert row.get("spec_id") == "STORE.390.date_of_initial_diagnosis"


def test_the_producer_can_declare_the_ids_a_key_covers():
    """An ablation arm is a different `spec_id` over the same variable, with the same correct
    answers. Without `--also-scores` the strict `spec_id` check made all three real UNSTRATIFIED
    manifests unkeyed and the arm's comparison read "nothing changed"."""
    out = subprocess.run(
        [sys.executable, str(TOOLS / "answer_key_from_corpus.py"),
         "--spec-key", "STORE.400_522_523.site_histology_behavior",
         "--fields", "primary_site,histology,behavior",
         "--also-scores", "STORE.400_522_523.site_histology_behavior.UNSTRATIFIED",
         "--out", "/dev/stdout"], capture_output=True, text=True, check=False, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    key = json.loads(out.stdout[:out.stdout.rindex("}") + 1])
    row = next(iter(key.values()))
    assert "STORE.400_522_523.site_histology_behavior.UNSTRATIFIED" in row["spec_ids"]
