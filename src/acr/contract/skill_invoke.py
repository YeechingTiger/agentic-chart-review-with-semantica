"""What a skill DECLARES about itself, beside the slot `skills.py` reads.

Three readers over one card's `metadata`: `kind` (how it would run), `category` (what it is), and
the directory it was found in. Nothing here runs anything.

## What was here until 2026-08-05, and why it went

This module used to be "one door to every skill" — an `invoke()` that dispatched on `kind` across
four branches:

    prose      the body IS the answer                      24 cards
    script     run an executable in the skill's directory   1 card
    llm        one model call with the body as instruction  0 cards, no invoker ever supplied
    subagent   hand the body to an agent with its own turns 0 cards, no invoker ever supplied

`invoke()` had no callers anywhere in `src/`, `tools/` or `tests/`, and `model_call` /
`subagent_call` — the injected callables the last two branches required — were passed by nobody, so
those two branches could not execute even in principle. 116 of the module's 242 lines were
unreachable.

The `subagent` branch is the one worth naming: it was a hand-rolled version of what
`SubAgentMiddleware` and the `task` tool already provide — isolated context, own tools and turns,
one result returned. Building a second one behind an injection point nothing injected into is the
speculative abstraction `CLAUDE.md` rule 2 forbids, and it sat here for days looking like capability.

`KINDS` now lists only what a card can actually be. Re-add a kind when something can run it.

## The two axes that survive, and why neither collapses into the other

`category` says WHAT a skill is. `kind` says HOW it would run. `slot` says WHERE its prose goes, and
is meaningful only for `kind: prose`.

`slot` survives as its own axis rather than folding into `category`, because the distinction it
draws was measured. Before 2026-08-02 eight cards competed for one slot, and what that arrangement
measured was the difference between KINDS of intervention rather than between policies — a tactic
whose precondition did not hold left an arm holding nothing. `tactic`, `policy` and `general` are
all `category: policy`; they are three different assembly positions.
"""

from __future__ import annotations

from pathlib import Path

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

#: How a skill RUNS, limited to what something in this tree can actually run. `llm` and `subagent`
#: were listed here until 2026-08-05 with zero cards and no invoker; a vocabulary that admits a
#: value nothing can execute turns a typo into a card that loads and then does nothing.
KINDS: tuple[str, ...] = ("prose", "script")

DEFAULT_KIND = "prose"


def skill_kind(name: str, skills_dir: Path | str | None = None) -> str:
    """The declared `kind`, or `prose`.

    UNLIKE `skill_slot`, THIS DEFAULTS. A missing slot is refused because a default would silently
    put two search policies in one slot and the manifest would read like one long policy. A missing
    kind has no such failure: prose is what a skill without an entry point can possibly be, and
    defaulting it is what lets this module arrive without touching 24 files.
    """
    kind = str(_skills.skill_meta(name, "kind", skills_dir) or DEFAULT_KIND)
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
    cat = _skills.skill_meta(name, "category", skills_dir)
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

