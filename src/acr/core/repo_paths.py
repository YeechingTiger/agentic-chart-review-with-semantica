"""The repo root and the asset directories under it, worked out without counting `__file__` levels.

Why this exists
---------------
Five modules each wrote `Path(__file__).resolve().parents[2]`, and that 2 encoded "I sit one level
below src/acr/". Once the modules moved into plane directories it pointed at `src/`, and the
consequence was not an error: `labelling.py`'s refusal that "labels may not be written inside the
repository" **silently stopped firing**, because the root it compared against had become `src/`.
`test_labels_root_refuses_a_path_inside_the_repository` caught it — DID NOT RAISE.

A level count is a quantity that moves with the directory structure, and "the repo root" is not. So
walk up to the directory that holds `pyproject.toml`, and it stays correct however deep a module is
moved. Raise when there is none rather than falling back to a guess: a path check that guessed its
root wrong is exactly the silent failure above.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

#: The marker file. `pyproject.toml` rather than `.git`: a run out of a built wheel has no `.git`,
#: and at that point these asset directories are not there either, so a caller should get an
#: explicit error instead of a path that exists and is empty.
MARKER = "pyproject.toml"


class RepoRootNotFound(RuntimeError):
    """No directory at or above this file holds the marker file."""


@lru_cache(maxsize=1)
def repo_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if (candidate / MARKER).is_file():
            return candidate
    raise RepoRootNotFound(
        f"no {MARKER} at or above {here}. These asset directories only exist in a source "
        f"checkout; a packaged install has no repo root and callers must not guess one.")


def asset_dir(name: str) -> Path:
    """An asset directory under the repo root, e.g. `skills` / `codes`. Existence is not
    checked — each caller has its own error to raise."""
    return repo_root() / name
