"""Every plane that reads a completed run must be able to reach `runs/`.

WHAT THIS PINS, and why it is not the same as `test_no_phi_in_tree.py`. Run output lives INSIDE the
worktree, gitignored wholesale, deliberately. `LocalArtifactStore.path` proves the opposite property
— that a path resolves UNDER a local root which is required to be outside the worktree. Any command
that resolves a RUN through the store therefore cannot be pointed at a single run on disk, and the
failure does not look like a bug: the plane simply appears unexercised.

That already happened once. `acr audit run` could not be aimed at any of the 508 manifests in this
tree, `require_run_artifact` was written to fix it, and it was wired into audit and nowhere else —
leaving `attribute`, `eval` and `repair` reading runs through the store. `attribute` had produced
zero proposals since it was written, and this is why.

So the property under test is REACHABILITY, stated once for every plane that has it: a directory of
run records inside the worktree resolves, and a Git-TRACKED run record still refuses. The second half
is what keeps this from being a test that merely deletes a guard.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from acr.core.local_artifacts import LocalArtifactError, require_run_tree


def _manifest(path, patient="SYN0001"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"patient_id": patient, "run_id": "r1"}), encoding="utf-8")
    return path


def test_a_directory_of_runs_inside_a_git_worktree_resolves(tmp_path):
    """The exact shape of `runs/`: a git repo, ignored output, and a manifest below it."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / ".gitignore").write_text("runs/\n", encoding="utf-8")
    _manifest(tmp_path / "runs" / "batch1" / "SYN0001.manifest.json")

    resolved = require_run_tree(tmp_path / "runs", what="runs")

    assert resolved == (tmp_path / "runs").resolve()
    assert sorted(p.name for p in resolved.rglob("*.manifest.json")) == \
        ["SYN0001.manifest.json"]


def test_a_single_manifest_resolves_too(tmp_path):
    m = _manifest(tmp_path / "runs" / "SYN0001.manifest.json")
    assert require_run_tree(m, what="runs") == m.resolve()


def test_a_tracked_run_record_still_refuses(tmp_path):
    """The property that matters is not "outside the worktree" but "Git does not track it"."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    m = _manifest(tmp_path / "runs" / "SYN0001.manifest.json")
    subprocess.run(["git", "add", "-f", str(m)], cwd=tmp_path, check=True)

    with pytest.raises(LocalArtifactError, match="TRACKED"):
        require_run_tree(m, what="runs")


def test_a_directory_holding_a_tracked_run_record_refuses(tmp_path):
    """A batch is scored as a whole; one tracked member is a disclosure in the batch."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _manifest(tmp_path / "runs" / "clean.manifest.json", "SYN0001")
    dirty = _manifest(tmp_path / "runs" / "dirty.manifest.json", "SYN0002")
    subprocess.run(["git", "add", "-f", str(dirty)], cwd=tmp_path, check=True)

    with pytest.raises(LocalArtifactError, match="TRACKED"):
        require_run_tree(tmp_path / "runs", what="runs")


def test_a_missing_path_refuses(tmp_path):
    with pytest.raises(LocalArtifactError, match="not found"):
        require_run_tree(tmp_path / "nope", what="runs")


def test_an_empty_directory_refuses_rather_than_scoring_nothing(tmp_path):
    """Zero runs is not a clean batch; it is a wrong path, and it must not read as success."""
    (tmp_path / "runs").mkdir()
    with pytest.raises(LocalArtifactError, match="no run record"):
        require_run_tree(tmp_path / "runs", what="runs")


@pytest.mark.parametrize("module,line_owner", [
    ("acr.commands.cli_attribute", "attribute"),
    ("acr.commands.cli_evaluation", "eval"),
    ("acr.commands.cli_repair", "repair"),
])
def test_no_plane_resolves_a_run_through_the_local_store(module, line_owner):
    """The regression guard, stated over source text.

    A behavioural test per command would need a manifest each plane will parse; this asks the one
    question that generalises — does any plane still send a RUN path through the store's
    outside-the-worktree proof. It is source-text, and that is a real weakness, but the alternative
    is three fixtures that pin three parsers and would not catch the fourth plane.
    """
    import importlib
    import inspect
    src = inspect.getsource(importlib.import_module(module))
    for i, line in enumerate(src.splitlines(), 1):
        if "store.path(" in line or "require_input(" in line:
            assert not any(w in line for w in ("runs", "manifest", "trace")), (
                f"{module}:{i} resolves a run record through the local store: {line.strip()}\n"
                f"`{line_owner}` cannot then be pointed at runs/, which is inside the worktree "
                f"by design. Use `require_run_tree`.")
