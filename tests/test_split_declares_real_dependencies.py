"""A module that ships in the staged repo is not a PyPI dependency.

`scaffold_repos.third_party` resolves every import in a staged distribution and pins it. Its own
docstring says third-party dependencies are COMPUTED, not listed, because "hand-listing them is how a
repo ends up declaring `langchain` because its neighbour needed it". The resolution had one gap: a
module reached through `sys.path` — a sibling under `tools/`, a test importing a fixture from another
test — looks exactly like a PyPI top-level name, so it was reported as an unpinned requirement.

The remedy in place was a hand-list. `NOT_REQUIRED` carried `conftest` and `hooks_harness`, which are
files in `tests/`, alongside `scipy` and `openpyxl`, which are genuinely absent optional packages.
Two different facts in one set: the first pair is "this is not a package at all", the second is "this
package is deliberately not required". A reader cannot tell them apart, and the next same-repo import
gets added to the same list without anybody noticing the category is wrong.

Measured: staging this tree reported `_decision_inputs` (a real file at `tools/_decision_inputs.py`,
imported by all three decision-point scripts through a `sys.path` insert) as an unpinned third-party
requirement. It had been reported on every staging run and read as noise.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import scaffold_repos as S  # noqa: E402


def _stage(tmp_path, **files) -> pathlib.Path:
    d = tmp_path / "acr-chart-review"
    for rel, body in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return d


def test_a_sibling_module_in_the_staged_tree_is_not_a_requirement(tmp_path, capsys):
    """The `tools/_decision_inputs.py` case, which was reported on every staging run."""
    repo = _stage(tmp_path,
                  **{"tools/_decision_inputs.py": "X = 1\n",
                     "tools/analyze_arms.py": "import _decision_inputs\n"})
    out = S.third_party(repo)
    assert not any("_decision_inputs" in r for r in out)
    assert "UNPINNED" not in capsys.readouterr().out


def test_a_test_importing_another_test_module_is_not_a_requirement(tmp_path, capsys):
    """`test_every_arm_switch_reaches_the_arm_hash.py` imports the scripted offline provider from
    `test_provenance`, so both files ship together and neither is a package."""
    repo = _stage(tmp_path,
                  **{"tests/test_provenance.py": "SHB = 1\n",
                     "tests/test_arms.py": "from test_provenance import SHB\n"})
    assert S.third_party(repo) == []
    assert "UNPINNED" not in capsys.readouterr().out


def test_a_genuinely_absent_package_is_still_reported(tmp_path, capsys):
    """The guard must keep its teeth. A name that resolves to no file in the staged tree and to no
    pin is the case it exists for — declaring a dependency the repo cannot install."""
    repo = _stage(tmp_path, **{"src/acr/x.py": "import some_package_nobody_pinned\n"})
    S.third_party(repo)
    err = capsys.readouterr().out
    assert "UNPINNED" in err and "some_package_nobody_pinned" in err


def test_a_real_pin_still_reaches_the_requirements(tmp_path, capsys):
    repo = _stage(tmp_path, **{"src/acr/x.py": "import yaml\n"})
    assert any("yaml" in r.lower() or "pyyaml" in r.lower() for r in S.third_party(repo))


def test_the_deliberate_exclusions_no_longer_hide_same_repo_modules():
    """`NOT_REQUIRED` may only contain names that are genuinely packages this repo does not require.
    `conftest` and `hooks_harness` were files; the resolver now sees them as files, so listing them
    asserts something false about what they are.
    """
    assert "conftest" not in S.NOT_REQUIRED
    assert "hooks_harness" not in S.NOT_REQUIRED
    assert S.NOT_REQUIRED, "the set still holds the genuinely-absent optional packages"
