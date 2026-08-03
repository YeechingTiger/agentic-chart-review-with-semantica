"""One door to every skill, whatever is behind it.

A skill has always been a directory with a `SKILL.md`: frontmatter plus prose that gets assembled
into a system prompt by slot. This module adds the rest of the doors. A skill may now be:

    kind: prose       the body IS the answer. Rendered into a prompt. What all 24 existing cards
                      are, and the default, so none of them changed.
    kind: script      an executable in the skill's own directory. Run it, return what it printed.
    kind: llm         one model call with the body as the instruction. Return the completion.
    kind: subagent    hand the body to an agent with its own tools and turns. Return its result.

WHAT THIS IS AND IS NOT. It is an ENTRY POINT — a uniform way to find a named unit of work and
invoke it. It is NOT a validation tier, and the distinction is the whole design:

    the door does not check what is behind it; whatever is behind it keeps its own rules.

So `contract.store-390` reached through here still goes through `contract/spec.py`, which still
refuses a spec whose enforced elements carry no provenance record. `tool.chart-documents` still goes
through `assert_tool_surface`, which still refuses a tool the surface did not declare. Those
guarantees live in the modules that own them. Putting them in the loader would mean one place that
has to know every category's rules, which is how a loader becomes a second implementation of
everything it loads.

WHY THIS MATTERS MORE THAN IT LOOKS. On 2026-08-03 this tree had TWENTY independent ways to find a
named, versioned unit of instruction or capability: twelve asset directories each with its own
loader (`specs/`, `skills/`, `module_catalog/`, `codes/`, `contracts/`, `evaluators/`,
`certification_catalog/`, `pipeline_catalog/`, `experience/`, `guidelines/`, `pricing/`, `usecase/`)
and eight registries in code (`TOOL_SCHEMAS`, `builtin_audit_registry`, `evals.REGISTRY`,
`builtin_attribution_registry`, `builtin_runtime_policy_registry`,
`builtin_evaluation_module_registry`, and three separate `from_directory` classmethods). One concept,
implemented twenty times, each with its own discovery, its own manifest and its own hash. A large
part of the answer to "why is this forty thousand lines" is that number.

## The two axes, and why neither collapses into the other

`category` says WHAT a skill is. `kind` says HOW it runs. `slot` says WHERE its prose goes, and is
meaningful only for `kind: prose`.

They are genuinely independent: an `audit` skill can be a script or a subagent; a `policy` skill is
always prose; a `contract` skill is prose plus a payload the runtime reads separately.

And `slot` survives as its own axis rather than folding into `category`, because the distinction it
draws was measured. Before 2026-08-02 eight cards competed for one slot, and what that arrangement
measured was the difference between KINDS of intervention rather than between policies — a tactic
whose precondition did not hold left an arm holding nothing. `tactic`, `policy` and `general` are all
`category: policy`; they are three different assembly positions.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import skills as _skills

#: What a skill IS. Every category can hold any `kind`, which is the point of two axes.
CATEGORIES: tuple[str, ...] = (
    "contract",     # what a task is: the question, the rules, the outcome space, the record shape
    "policy",       # what to do next, or what to do when — always prose
    "tool",         # the capabilities exposed for one shape of data
    "eval",         # how to score a trace, a run, or a final result
    "audit",        # how to inspect a recorded run for PHI, risk, boundary crossings
    "experience",   # a prior somebody MEASURED on a corpus, and which therefore expires
    "improvement",  # how to turn a classified failure into a proposed change to another skill
)

#: How a skill RUNS. `prose` is the default so that every skill written before this module keeps
#: working with no edit — there are 24 of them and none declares a kind.
KINDS: tuple[str, ...] = ("prose", "script", "llm", "subagent")

DEFAULT_KIND = "prose"


class SkillInvocationError(RuntimeError):
    """A skill could not be run. Distinct from `SkillError`, which is about loading one."""


@dataclass(frozen=True)
class SkillResult:
    """What came back, plus enough to tell how it got here.

    `output` is text. Not a parsed object, and not a typed result per category: a skill is a door,
    and a door that reshapes what passes through it is a door with an opinion. A caller that wants
    JSON parses it, and the parse failure is the caller's to report — which is better than this
    module guessing which categories return JSON.
    """

    name: str
    kind: str
    category: str | None
    output: str
    exit_code: int | None = None
    detail: dict[str, Any] | None = None


def skill_kind(name: str, skills_dir: Path | str | None = None) -> str:
    """The declared `kind`, or `prose`.

    UNLIKE `skill_slot`, THIS DEFAULTS. A missing slot is refused because a default would silently
    put two search policies in one slot and the manifest would read like one long policy. A missing
    kind has no such failure: prose is what a skill without an entry point can possibly be, and
    defaulting it is what lets this module arrive without touching 24 files.
    """
    kind = str(_skills._frontmatter(name, skills_dir).get("kind") or DEFAULT_KIND)
    if kind not in KINDS:
        raise _skills.SkillError(
            f"skill {name!r} declares unknown kind {kind!r}; expected one of {list(KINDS)}")
    return kind


def skill_category(name: str, skills_dir: Path | str | None = None) -> str | None:
    """The declared `category`, or `None`.

    Optional on purpose, for now. The 24 existing cards declare a `slot` and no category, and
    inferring one from the slot would write a guess into the manifest as though it were a
    declaration. A skill that wants to be found by category says so.
    """
    fm = _skills._frontmatter(name, skills_dir)
    cat = fm.get("category")
    if cat is None:
        return None
    if cat not in CATEGORIES:
        raise _skills.SkillError(
            f"skill {name!r} declares unknown category {cat!r}; expected one of {list(CATEGORIES)}")
    return str(cat)


def skill_dir(name: str, skills_dir: Path | str | None = None) -> Path:
    """The directory this skill's `SKILL.md` was found in, across every configured root."""
    roots = [Path(skills_dir)] if skills_dir else list(_skills.site.skill_roots())
    for r in roots:
        if (r / name / "SKILL.md").is_file():
            return r / name
    raise _skills.SkillError(
        f"no skill {name!r} under any of {[str(r) for r in roots]}")


def invoke(name: str, *, skills_dir: Path | str | None = None,
           inputs: dict[str, str] | None = None,
           model_call=None, subagent_call=None,
           timeout: int = 600) -> SkillResult:
    """Run one skill and return what it produced.

    `model_call(prompt: str) -> str` and `subagent_call(instruction: str, cwd: Path) -> str` are
    INJECTED rather than imported. Two reasons, and the second is the one that matters: this module
    lives in `contract/`, which may not reach a provider (`tests/test_layering.py` pins that), and a
    test that wants to exercise a `kind: llm` skill has to be able to hand in something that does
    not dial out. `tests/conftest.py` exists because a test once made two real paid calls while
    believing it had muzzled the provider.
    """
    kind = skill_kind(name, skills_dir)
    category = skill_category(name, skills_dir)
    body = _skills.load_skill_body(name, skills_dir)
    here = skill_dir(name, skills_dir)
    env = {**os.environ, **{f"ACR_SKILL_{k.upper()}": v for k, v in (inputs or {}).items()}}

    if kind == "prose":
        # The body IS the answer. Rendering it into a prompt is `skills_block`'s job; this returns
        # it verbatim so a caller that wants the text for anything else gets the same bytes.
        return SkillResult(name, kind, category, body)

    if kind == "script":
        entry = _skills._frontmatter(name, skills_dir).get("entry")
        if not entry:
            raise SkillInvocationError(
                f"skill {name!r} is kind: script and declares no `entry`. Name the executable "
                f"relative to {here}; a script skill without an entry point is prose that will "
                f"fail at the moment somebody trusts it.")
        argv = shlex.split(str(entry))
        target = here / argv[0]
        if not target.is_file():
            raise SkillInvocationError(f"skill {name!r} entry {entry!r} is not a file at {target}")
        # A `.py` entry runs in the CALLER's interpreter, not whatever the shebang finds. The
        # shebang version of this line reached the system Python 3.9 and died on
        # `from datetime import UTC` — a 3.11 name — while the package it was importing was
        # installed in a 3.12 venv two directories away. A skill that runs in a different
        # interpreter than its caller is a skill with a second dependency set nobody declared.
        cmd = ([sys.executable, str(target), *argv[1:]] if target.suffix == ".py"
               else [str(target), *argv[1:]])
        try:
            # The CALLER's working directory, not the skill's. A script run from inside its own
            # directory makes every relative path the caller passed mean something else — the first
            # invocation of this looked up a trace path relative to `assets/skills/<name>/` and
            # reported "no trace at …" for a file that was right there. The skill's own directory is
            # handed over as `ACR_SKILL_DIR` for resources it ships.
            p = subprocess.run(cmd, env={**env, "ACR_SKILL_DIR": str(here)}, timeout=timeout,
                               capture_output=True, text=True, check=False)
        except subprocess.TimeoutExpired as e:
            raise SkillInvocationError(f"skill {name!r} exceeded {timeout}s") from e
        # stderr is CARRIED, not dropped. A script that fails and explains itself on stderr is the
        # normal case, and a door that returns only stdout turns that explanation into silence.
        return SkillResult(name, kind, category, p.stdout, p.returncode,
                           {"stderr": p.stderr[-4000:], "argv": cmd})

    if kind == "llm":
        if model_call is None:
            raise SkillInvocationError(
                f"skill {name!r} is kind: llm and no `model_call` was supplied. It is injected "
                f"rather than imported so that a caller decides whether this run may reach a "
                f"provider — see the note in this module's `invoke`.")
        prompt = body if not inputs else body + "\n\n" + "\n".join(
            f"{k}: {v}" for k, v in inputs.items())
        return SkillResult(name, kind, category, model_call(prompt))

    if kind == "subagent":
        if subagent_call is None:
            raise SkillInvocationError(
                f"skill {name!r} is kind: subagent and no `subagent_call` was supplied.")
        return SkillResult(name, kind, category, subagent_call(body, here))

    raise SkillInvocationError(f"skill {name!r} has kind {kind!r} and no invoker for it")


def by_category(category: str, skills_dir: Path | str | None = None) -> tuple[str, ...]:
    """Every skill declaring this category, across every configured root.

    The lookup the twenty registries were each doing for one kind of thing. Sorted, because a
    caller that iterates has to get the same order twice or a manifest is not comparable.
    """
    if category not in CATEGORIES:
        raise _skills.SkillError(f"unknown category {category!r}; expected one of {list(CATEGORIES)}")
    roots = [Path(skills_dir)] if skills_dir else list(_skills.site.skill_roots())
    found: set[str] = set()
    for r in roots:
        if not r.is_dir():
            continue
        for d in sorted(r.iterdir()):
            if not (d / "SKILL.md").is_file():
                continue
            try:
                if skill_category(d.name, r) == category:
                    found.add(d.name)
            except _skills.SkillError:
                # A malformed skill is not this function's to report. `tests/test_skills_load.py`
                # is what fails over one; a category listing that raises would make every caller
                # of every category depend on every skill being well-formed.
                continue
    return tuple(sorted(found))
