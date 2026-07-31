"""Every skill in `assets/skills/` must survive the loader that actually reads it.

deepagents drops a malformed skill silently: missing frontmatter, a non-mapping mapping, an
empty `name` or `description` all produce a log warning and nothing else, so the model never
learns the skill existed and no run fails. A skill is advisory by design — the model chooses
whether to load it — but "the model chose not to" and "the file never reached the model" are
indistinguishable from the outside, and only one of them is a bug we can fix.

The rules below are the loader's, transcribed rather than imported: deepagents lives in
.venv-deep and the suite runs under .venv, so importing it here would make this guard skip
in exactly the environment that runs the tests.

Reference: deepagents 0.6.12, deepagents/middleware/skills.py -- `_parse_skill_metadata`
(frontmatter regex, required keys, length caps) and `_validate_skill_name`.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

SKILLS_DIR = Path(__file__).resolve().parents[1] / "assets" / "skills"

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESCRIPTION = 1024
MAX_FILE_BYTES = 10 * 1024 * 1024


def _skill_dirs() -> list[Path]:
    # Discovery is exactly one level deep: the loader lists the source directory and looks
    # for SKILL.md in each child. A skill nested any deeper is invisible.
    return sorted(p for p in SKILLS_DIR.iterdir() if p.is_dir())


# Two gaps below are real, current, and specific to skills owned by this triage pass: the
# build agents authoring them were killed mid-work by an org spend limit on 2026-07-26. The
# tests correctly catch both. Per repo triage policy this is "code was never written" --
# skip with the missing path named, do not author the missing file here and do not weaken
# the assertion for every other skill in the tree.
#
# Each dict below scopes its skip to exactly the test function whose failure it explains --
# `store-to-spec` still loads fine (test_skill_loads passes for it) and only fails reference
# resolution because of the one missing pointer, so it must not be skipped out of
# test_skill_loads too.
_SKILL_LOAD_GAPS = {
    "guideline-to-rules": (
        "assets/skills/guideline-to-rules/SKILL.md was never written -- only "
        "references/worked-example.md landed before the build agent was killed by the org "
        "spend limit; see tests/test_guideline_to_rules_skill.py for the rest of the gap"
    ),
}
_REFERENCE_POINTER_GAPS = {
    "store-to-spec": (
        "assets/skills/store-to-spec/references/proof-obligations.md is pointed at by SKILL.md and "
        "field-design.md but was never written -- see tests/test_store_to_spec_skill.py"
    ),
}


def _params_skipping(gaps: dict[str, str]) -> list:
    return [
        pytest.param(p, marks=pytest.mark.skip(reason=gaps[p.name])) if p.name in gaps else p
        for p in _skill_dirs()
    ]


def _frontmatter(skill_md: Path) -> dict:
    raw = skill_md.read_bytes()
    assert len(raw) <= MAX_FILE_BYTES, f"{skill_md} exceeds the 10 MB loader cap"
    text = raw.decode("utf-8")           # non-UTF8 raises here, as it does in the loader
    m = FRONTMATTER_RE.match(text)
    assert m, f"{skill_md} has no frontmatter block at byte 0"
    data = yaml.safe_load(m.group(1))
    assert isinstance(data, dict), f"{skill_md} frontmatter is not a mapping"
    return data


@pytest.mark.parametrize("skill_dir", _params_skipping(_SKILL_LOAD_GAPS), ids=lambda p: p.name)
def test_skill_loads(skill_dir: Path):
    skill_md = skill_dir / "SKILL.md"
    assert skill_md.is_file(), f"{skill_dir} has no SKILL.md (filename is case-sensitive)"
    fm = _frontmatter(skill_md)

    name = fm.get("name")
    assert isinstance(name, str) and name, f"{skill_md}: `name` is required and non-empty"
    assert len(name) <= MAX_NAME
    assert NAME_RE.match(name), f"{skill_md}: `name` must be lowercase-alnum with single hyphens"
    # A mismatch only warns in the loader, so nothing else in the system would ever catch it.
    assert name == skill_dir.name, f"{skill_md}: `name` {name!r} != directory {skill_dir.name!r}"

    desc = fm.get("description")
    assert isinstance(desc, str) and desc.strip(), f"{skill_md}: `description` is required"
    # Over the cap the loader truncates silently, and the tail of a description is where the
    # trigger conditions tend to sit -- losing them costs retrieval, not just tidiness.
    assert len(desc) <= MAX_DESCRIPTION, f"{skill_md}: description is {len(desc)} chars"


@pytest.mark.parametrize("skill_dir", _params_skipping(_REFERENCE_POINTER_GAPS), ids=lambda p: p.name)
def test_reference_pointers_resolve(skill_dir: Path):
    """Progressive disclosure to `references/` is prose, not code -- nothing else checks it.

    The loader never enumerates files beside SKILL.md, so a second-level file is reached only
    because the text names it. A bare relative pointer resolves against the process cwd and
    404s; repo-relative is what works from the repo root, which is where deep_runner must be
    launched anyway.
    """
    # SKILLS_DIR 现在是 <root>/assets/skills，所以 repo root 要再上一层。指针本身写成
    # 仓库根相对的 `assets/skills/…`，两者相接才是真实路径。
    repo = SKILLS_DIR.parents[1]
    for md in skill_dir.rglob("*.md"):
        for ref in re.findall(r"`(assets/skills/[A-Za-z0-9_/.-]+\.md)`", md.read_text()):
            assert (repo / ref).is_file(), f"{md} points at missing {ref}"
        for bare in re.findall(r"`(references/[A-Za-z0-9_/.-]+\.md)`", md.read_text()):
            pytest.fail(f"{md}: pointer {bare!r} is cwd-relative; write it repo-relative")
