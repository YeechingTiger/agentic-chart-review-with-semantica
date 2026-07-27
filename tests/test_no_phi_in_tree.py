"""No real patient identifier may live in this tree.

The corpus under `/N/project/computable_phenotype/acr_real/` is real PHI. The repo is not a
safe place for any of it, and the leak this guards against already happened: real
`person_id`s were pasted into `skills/` write-ups and into `src/` docstrings as evidence for
design decisions ("patient <id> was coded C349 when 'right upper lobe' was documented").
They were accurate, they were useful, and they were sixteen digits of protected health
information sitting in source control.

The fix was to swap them for the pseudonyms P01..P05 and keep the mapping OUTSIDE the tree,
at `/N/project/computable_phenotype/llm/phi_pseudonym_map.json` (mode 600). The point of a
pseudonym is that the write-up keeps its evidentiary value while the tree stops carrying the
identifier, so nothing was lost by the swap.

WHY THIS IS A TEST AND NOT A REVIEW HABIT
-----------------------------------------
A reviewer notices a sixteen-digit number in a diff exactly once, and then stops noticing.
The observation that motivated the scrub -- that real ids had accumulated across seven
skill documents and three source files without anyone objecting -- is itself the evidence
that human review does not catch this. So it is asserted, on every run.

Two scopes, because the leak and the commit are different events:

  * TRACKED files, via `git ls-files`. This is the one that matters for disclosure: a
    tracked file is a file that gets pushed. It covers content AND path, since
    `runs/azure_real_<person_id>/` carries the identifier in its name and `.gitignore`
    deliberately un-ignores `runs/**/*.manifest.json`.
  * The SOURCE directories in the working tree, tracked or not. `skills/` and half of
    `src/acr/` were untracked when the ids were found in them, so a tracked-only check
    would have reported all clear on the exact files that were leaking.

Failures name the file and line but never echo the identifier -- a test that prints PHI into
a CI log has moved the leak rather than closed it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: The real corpus's person_id shape: a `1168` institutional prefix and twelve more digits.
#: A pattern, deliberately not an example -- writing one real id into the guard that exists
#: to forbid real ids would be its own defect.
PERSON_ID = re.compile(rb"1168[0-9]{12}")

#: Directories that belong to the repo and must be clean whether or not they are committed.
SOURCE_DIRS = ("src", "skills", "specs", "tests", "contracts", "tools", "guidelines")

#: Not scanned. `.venv*` vendors third parties -- google-genai ships a GCP dataset resource
#: id that matches the pattern by coincidence and is not PHI. `corpus/` is the fabricated
#: SYN000x dev corpus. `runs/` is excluded HERE only because it is the experimental record
#: and is not ours to rewrite; the tracked-file scan below still covers every part of it
#: that git would actually publish.
SKIP_PARTS = {".git", ".venv", ".venv-deep", "__pycache__", "node_modules"}

TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".json", ".txt", ".toml", ".cfg", ".ini",
                 ".sh", ".jsonl", ".csv", ".rst", ".html", ".ipynb", ""}


def _redact(match: bytes) -> str:
    """Enough to prove a hit, not enough to be an identifier."""
    s = match.decode("ascii", "replace")
    return f"{s[:4]}{'*' * (len(s) - 4)}"


def _hits(path: Path) -> list[str]:
    try:
        raw = path.read_bytes()
    except (OSError, ValueError):
        return []
    if b"\0" in raw[:8192]:          # binary; nothing to read a person_id out of
        return []
    out = []
    for n, line in enumerate(raw.splitlines(), start=1):
        for m in PERSON_ID.finditer(line):
            out.append(f"{path.relative_to(ROOT)}:{n}: {_redact(m.group(0))}")
    return out


def _tracked() -> list[Path]:
    try:
        raw = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"],
                             capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:  # pragma: no cover
        pytest.skip(f"git unavailable: {e}")
    return [ROOT / p for p in raw.decode().split("\0") if p]


def test_no_real_person_id_in_a_tracked_file():
    """A tracked file is a file that gets pushed. This is the disclosure boundary."""
    found: list[str] = []
    for p in _tracked():
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            found.extend(_hits(p))
    assert not found, (
        f"{len(found)} real person_id(s) in tracked files — use the P01..P05 pseudonyms and "
        "keep the mapping at /N/project/computable_phenotype/llm/phi_pseudonym_map.json:\n"
        + "\n".join(found[:40])
    )


def test_no_real_person_id_in_a_tracked_path():
    """`runs/azure_real_<person_id>/…` carries the id in the directory name, and
    `.gitignore` un-ignores `runs/**/*.manifest.json`, so a run directory for a real patient
    is one `git add -A` away from being published — with the id in the path even if the file
    content is clean."""
    bad = [str(p.relative_to(ROOT)) for p in _tracked() if PERSON_ID.search(bytes(p))]
    assert not bad, (
        f"{len(bad)} tracked path(s) contain a real person_id; rename them to a pseudonym:\n"
        + "\n".join(sorted({b.split('/')[0] + '/…' for b in bad})[:40])
    )


@pytest.mark.parametrize("directory", SOURCE_DIRS)
def test_no_real_person_id_in_the_source_tree(directory: str):
    """Untracked source counts too: `skills/` and much of `src/acr/` were untracked at the
    moment the ids were found sitting in them."""
    base = ROOT / directory
    if not base.is_dir():
        pytest.skip(f"{directory}/ not present")
    found: list[str] = []
    for p in base.rglob("*"):
        if any(part in SKIP_PARTS for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            found.extend(_hits(p))
    assert not found, (
        f"{len(found)} real person_id(s) under {directory}/ — use P01..P05:\n"
        + "\n".join(found[:40])
    )


def test_the_pseudonym_map_is_outside_the_tree_and_not_world_readable():
    """The mapping is the one artefact that must never be in the repo, and the only thing
    that makes the pseudonyms reversible for someone entitled to reverse them."""
    m = Path("/N/project/computable_phenotype/llm/phi_pseudonym_map.json")
    assert ROOT not in m.parents, "the mapping must not live under the repo"
    if not m.exists():
        pytest.skip("mapping not present on this host")
    assert m.stat().st_mode & 0o077 == 0, (
        f"mapping is group/world accessible (mode {m.stat().st_mode & 0o777:o}); chmod 600 it"
    )
