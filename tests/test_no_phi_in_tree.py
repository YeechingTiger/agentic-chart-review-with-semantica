"""No real patient identifier may live in this tree.

The corpus under `$ACR_REAL_CORPUS/` is real PHI. The repo is not a
safe place for any of it, and the leak this guards against already happened: real
`person_id`s were pasted into `assets/skills/` write-ups and into `src/` docstrings as evidence for
design decisions ("patient <id> was coded C349 when 'right upper lobe' was documented").
They were accurate, they were useful, and they were sixteen digits of protected health
information sitting in source control.

The fix was to swap them for the pseudonyms P01..P05 and keep the mapping OUTSIDE the tree,
at `$ACR_PSEUDONYM_MAP` (mode 600). The point of a
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
    `runs/azure_real_<person_id>/` carries the identifier in its name as well as inside
    the manifest.
  * The SOURCE directories in the working tree, tracked or not. `assets/skills/` and half of
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

from acr.core import site

ROOT = Path(__file__).resolve().parents[1]

#: The site's person-id shape, taken from `acr.core.site` so that this scan and the runtime's
#: masking cannot disagree. It used to be a literal here AND in three runtime modules; the four
#: copies were the thing most likely to drift, and the direction they drift in is narrower, which
#: is the direction that fails silently.
#: A pattern, deliberately not an example -- writing one real id into the guard that exists
#: to forbid real ids would be its own defect.
PERSON_ID = (re.compile(site.PHI_SCAN_PATTERN.encode())
             if site.PHI_SCAN_PATTERN else None)

#: Directories that belong to the repo and must be clean whether or not they are committed.
SOURCE_DIRS = ("src", "assets", "tests", "tools")

#: Not scanned. `.venv*` vendors third parties -- google-genai ships a GCP dataset resource
#: id that matches the pattern by coincidence and is not PHI. `corpus/` is the fabricated
#: SYN000x dev corpus. `runs/` is excluded HERE because it is the experimental record and is
#: not ours to rewrite -- and because it is now ignored wholesale by git, so nothing in it is
#: publishable; `test_no_run_output_is_tracked` is what holds that property in place.
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
    if PERSON_ID is None:
        pytest.skip("ACR_PHI_SCAN_PATTERN is not set: no shape to scan for. See acr/core/site.py — three defaults were tried and each was measured wrong.")
    found: list[str] = []
    for p in _tracked():
        if p.is_file() and p.suffix.lower() in TEXT_SUFFIXES:
            found.extend(_hits(p))
    assert not found, (
        f"{len(found)} real person_id(s) in tracked files — use the P01..P05 pseudonyms and "
        "keep the mapping at $ACR_PSEUDONYM_MAP:\n"
        + "\n".join(found[:40])
    )


def test_no_real_person_id_in_a_tracked_path():
    """`runs/azure_real_<person_id>/…` carries the id in the directory name, so a run
    directory for a real patient could be published with the id in the PATH even if the file
    content is clean. `runs/` is ignored now, but `git add -f` overrides an ignore and this
    check does not depend on the ignore being right."""
    if PERSON_ID is None:
        pytest.skip("ACR_PHI_SCAN_PATTERN is not set: no shape to scan for. See acr/core/site.py — three defaults were tried and each was measured wrong.")
    bad = [str(p.relative_to(ROOT)) for p in _tracked() if PERSON_ID.search(bytes(p))]
    assert not bad, (
        f"{len(bad)} tracked path(s) contain a real person_id; rename them to a pseudonym:\n"
        + "\n".join(sorted({b.split('/')[0] + '/…' for b in bad})[:40])
    )


def test_no_run_output_is_tracked():
    """`runs/` NEVER enters git — the ignore is a policy, this is the enforcement.

    The earlier policy was narrower: traces out, `!runs/**/*.manifest.json` in, on the grounds
    that a manifest is 2KB of evidence. It then needed a `runs/*_1168*/` patch to keep real
    patients out, and that patch made the disclosure boundary depend on how a DIRECTORY WAS
    NAMED rather than on what was inside it. A real manifest under a directory that happened
    not to match the glob was still one `git add -A` from being pushed, with the person_id in
    the JSON. `test_no_real_person_id_in_a_tracked_file` would have caught that one — but only
    for the `1168` shape, and only if someone ran the tests before pushing.

    So the rule is now categorical and checkable without knowing anything about ids: no path
    under `runs/` is tracked, whatever it is called and whatever is in it. Run output lives
    outside the repo, under $ACR_LOCAL_ROOT/run/.
    """
    tracked = sorted(str(p.relative_to(ROOT)) for p in _tracked()
                     if p.relative_to(ROOT).parts[:1] == ("runs",))
    assert not tracked, (
        f"{len(tracked)} file(s) under runs/ are tracked. Run output is not a build product "
        "and not evidence git should carry: `git rm -r --cached runs/` (the files stay on "
        "disk). If you need a manifest reviewable, copy the ONE you mean to a path outside "
        "runs/ after checking it for identifiers:\n" + "\n".join(tracked[:40])
    )


@pytest.mark.parametrize("directory", SOURCE_DIRS)
def test_no_real_person_id_in_the_source_tree(directory: str):
    """Untracked source counts too: `assets/skills/` and much of `src/acr/` were untracked at the
    moment the ids were found sitting in them."""
    if PERSON_ID is None:
        pytest.skip("ACR_PHI_SCAN_PATTERN is not set: no shape to scan for. See acr/core/site.py — three defaults were tried and each was measured wrong.")
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
    m = Path("$ACR_PSEUDONYM_MAP")
    assert ROOT not in m.parents, "the mapping must not live under the repo"
    if not m.exists():
        pytest.skip("mapping not present on this host")
    assert m.stat().st_mode & 0o077 == 0, (
        f"mapping is group/world accessible (mode {m.stat().st_mode & 0o777:o}); chmod 600 it"
    )


def test_a_real_corpus_without_the_two_identifier_patterns_is_refused():
    """The link that makes unset patterns safe rather than merely quiet.

    Every person-id refusal, every mask and this file's own byte scan is a no-op without a
    configured shape, and a no-op guard reads exactly like a satisfied one. So a deployment holding
    real data must declare both shapes. This pins the REFUSAL and not a default: three defaults
    were tried and each was measured wrong somewhere nobody had looked — see `acr/core/site.py`,
    and note that the two settings exist because their consumers want OPPOSITE error costs.
    """
    import importlib
    import os as _os

    import acr.core.site as s

    KEYS = ("ACR_REAL_CORPUS", "ACR_PERSON_ID_PATTERN", "ACR_PHI_SCAN_PATTERN")
    saved = {k: _os.environ.get(k) for k in KEYS}

    def reload(**env):
        for k in KEYS:
            v = env.get(k)
            _os.environ.pop(k, None) if v is None else _os.environ.__setitem__(k, v)
        return importlib.reload(s)

    try:
        m = reload(ACR_REAL_CORPUS="/somewhere/real")
        with pytest.raises(RuntimeError, match="ACR_PERSON_ID_PATTERN and ACR_PHI_SCAN_PATTERN"):
            m.require_person_id_pattern()

        # Half-configured is still refused, and the message names the half that is missing.
        m = reload(ACR_REAL_CORPUS="/somewhere/real", ACR_PERSON_ID_PATTERN=r"XX\d{6}")
        with pytest.raises(RuntimeError, match="ACR_PHI_SCAN_PATTERN"):
            m.require_person_id_pattern()

        m = reload(ACR_REAL_CORPUS="/somewhere/real", ACR_PERSON_ID_PATTERN=r"XX\d{6}",
                   ACR_PHI_SCAN_PATTERN=r"XX\d{6}")
        m.require_person_id_pattern()
        assert m.looks_like_a_person_id("XX123456")
        assert not m.looks_like_a_person_id("SYN0001")

        # No real corpus: nothing to protect, and inert guards are the correct state.
        m = reload()
        m.require_person_id_pattern()
        assert not m.looks_like_a_person_id("XX123456")
    finally:
        for k, v in saved.items():
            _os.environ.pop(k, None) if v is None else _os.environ.__setitem__(k, v)
        importlib.reload(s)
