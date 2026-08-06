"""Every card must say which slot it is assembled into.

The wrong slot is not a small thing: the card in the `search` slot is the one variable that is
replaced when a controlled comparison is run, and if a `task` card slips in there, the difference
between the two runs no longer comes only from the retrieval strategy — and the conclusion gets
written into the report all the same. So a slot is declared and checked, not guessed from the
directory name.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from acr.contract.skills import (
    SLOTS,
    SkillError,
    SkillStack,
    parse_skill_stack,
    skill_slot,
    skills_manifest,
)
from acr.core import site

SKILLS_DIR = Path(__file__).resolve().parents[1] / "assets" / "skills"
_FM = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# `assets/skills/guideline-to-rules/SKILL.md` was never written — see
# tests/test_skills_load.py:49-55.
_NO_SKILL_MD = {"guideline-to-rules"}


def _skill_names() -> list[str]:
    return sorted(p.name for p in SKILLS_DIR.iterdir()
                  if p.is_dir() and p.name not in _NO_SKILL_MD)


@pytest.mark.parametrize("name", _skill_names())
def test_every_skill_declares_a_known_slot(name: str):
    from acr.contract.skills import skill_meta
    if str(skill_meta(name, "kind") or "prose") != "prose":
        pytest.skip("only prose skills are assembled into a prompt, so only they need a slot")
    slot = skill_slot(name)
    assert slot in SLOTS, f"{name}: slot {slot!r} not one of {SLOTS}"


@pytest.mark.parametrize("name", _skill_names())
def test_declared_slot_matches_the_file(name: str):
    """`skill_slot` reports what DEEPAGENTS parsed out of the card, not a second reading of it.

    This test used to open the file and compare against its own `yaml.safe_load`. That made it the
    tree's second frontmatter parser, and a test that reimplements the thing it checks agrees with
    itself for the wrong reasons. `discover()` is the middleware's own output, so the comparison is
    now between one parse and the code that consumes it.
    """
    from acr.contract.skills import discover, skill_meta
    if str(skill_meta(name, "kind") or "prose") != "prose":
        pytest.skip("a slot is where PROSE goes in a prompt; a script/llm/subagent skill is "
                    "invoked through contract/skill_invoke.py and never rendered")
    parsed = discover()[name]["metadata"]
    assert parsed.get("slot") == skill_slot(name), (
        "and it comes from `metadata`, which is where the Agent Skills spec puts a property the "
        "standard itself does not define")


def test_missing_slot_raises(tmp_path: Path):
    d = tmp_path / "no-slot"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: no-slot\ndescription: x\n---\n\nbody\n", encoding="utf-8")
    with pytest.raises(SkillError, match="declares no `slot`"):
        skill_slot("no-slot", tmp_path)


def test_unknown_slot_raises(tmp_path: Path):
    d = tmp_path / "bad-slot"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: bad-slot\ndescription: x\nmetadata:\n  slot: wherever\n---\n\nbody\n", encoding="utf-8")
    with pytest.raises(SkillError, match="unknown slot 'wherever'"):
        skill_slot("bad-slot", tmp_path)


def test_stack_renders_task_then_policy_then_general():
    stack = SkillStack(task="store-icdo-coding", policy="policy-reactive",
                       general=("chart-triage", "coverage-judgement"))
    assert stack.names() == ("store-icdo-coding", "policy-reactive",
                             "chart-triage", "coverage-judgement")


def test_stack_rejects_a_skill_in_the_wrong_slot():
    """Stuffing a general card into the policy slot — precisely the assembly error that drains a
    controlled comparison of its meaning."""
    with pytest.raises(SkillError, match="chart-triage.*declares slot 'general'.*'policy'"):
        SkillStack(policy="chart-triage").validate()


def test_stack_rejects_an_eval_skill_in_the_chart_agent(tmp_path: Path):
    """A `slot: eval` card belongs to the evaluation agent, so placing it in any slot of a chart
    run is wrong.

    The plan had this test name `eval-contrast-traces` directly, but those cards are not built
    until Task 6, and naming it now would only hit "that card does not exist" and never reach the
    slot check at all. So a throwaway `slot: eval` card is built here, and it tests the same thing.
    """
    d = tmp_path / "eval-contrast-traces"
    d.mkdir()
    (d / "SKILL.md").write_text(
        "---\nname: eval-contrast-traces\ndescription: x\nmetadata:\n  slot: eval\n---\n\nbody\n",
        encoding="utf-8")
    with pytest.raises(SkillError, match="slot 'eval'"):
        SkillStack(general=("eval-contrast-traces",)).validate(tmp_path)


@pytest.mark.parametrize("name", sorted(p.name for p in SKILLS_DIR.iterdir()
                                        if p.name.startswith("eval-")))
@pytest.mark.parametrize("slot", ["task", "policy", "general"])
def test_a_real_eval_card_cannot_be_placed_in_a_chart_slot(name: str, slot: str):
    """The test above builds a card in tmp_path to exercise the mechanism; this one uses the real
    cards, over the path a user would actually type.

    `--skills general=eval-overconfidence` is one slip of the fingers away, and once it is accepted
    the chart-run prompt grows a post-hoc-review passage saying "you are not allowed to score" —
    words spoken to a model that is in the middle of reading a chart, which are neither its task nor
    something any manifest would flag in red. So a real card has to be refused at assembly time.
    """
    placed = SkillStack(**({slot: (name,)} if slot == "general" else {slot: name}))
    with pytest.raises(SkillError, match="slot 'eval'"):
        placed.validate()
    with pytest.raises(SkillError, match="slot 'eval'"):
        parse_skill_stack(f"{slot}={name}", SkillStack())


def test_manifest_carries_the_slot():
    entries = skills_manifest(SkillStack(general=("coverage-judgement",)))
    assert [e["slot"] for e in entries] == ["general"]
    assert entries[0]["skill"] == "coverage-judgement"
    assert entries[0]["content_hash"]


def test_unknown_profile_still_falls_back_to_the_universal_block():
    """The fallback for an unknown profile has to carry the tool contract too — otherwise "every arm
    has it" acquires one exception, and the exception lands on exactly the path nobody ever
    configured on purpose."""
    from acr.review.runtime_profiles import runtime_policy_skills
    assert runtime_policy_skills("not-a-profile").names() == ("tool-contract", "coverage-judgement")


# --- Task 3: `--skills` swaps one card without minting a profile -----------------------------

ROOT = Path(__file__).resolve().parents[1]
SPEC = site.specs_root() / "STORE.400_522_523.site_histology_behavior.yaml"
CORPUS = site.corpus_root()


def test_parse_replaces_the_search_slot():
    base = SkillStack(general=("coverage-judgement",))
    got = parse_skill_stack("policy=policy-reactive", base)
    assert got == SkillStack(policy="policy-reactive", general=("coverage-judgement",))


def test_parse_appends_to_general():
    base = SkillStack(general=("coverage-judgement",))
    got = parse_skill_stack("general=+chart-triage", base)
    assert got.general == ("coverage-judgement", "chart-triage")


def test_parse_replaces_general_without_plus():
    base = SkillStack(general=("coverage-judgement", "chart-triage"))
    assert parse_skill_stack("general=thread-chasing", base).general == ("thread-chasing",)


def test_parse_replaces_general_with_a_pipe_separated_list():
    """`|` is the only way to put more than one card in `general` in a single clause.

    Comma already means "next clause", so a replacement list cannot use it. The separator is
    undiscoverable unless the help text says so, which is why the help text is tested too.
    """
    base = SkillStack(general=("coverage-judgement",))
    got = parse_skill_stack("general=chart-triage|thread-chasing", base)
    assert got.general == ("chart-triage", "thread-chasing")


def test_parse_rejects_an_unknown_slot():
    with pytest.raises(SkillError, match="unknown slot 'polish'"):
        parse_skill_stack("polish=x", SkillStack())


def test_parse_rejects_a_missing_equals():
    with pytest.raises(SkillError, match="expected slot=value"):
        parse_skill_stack("policy-reactive", SkillStack())


def test_parse_validates_placement():
    with pytest.raises(SkillError, match="declares slot 'general'"):
        parse_skill_stack("policy=chart-triage", SkillStack())


def test_parse_empty_value_clears_the_slot():
    base = SkillStack(policy="policy-reactive", general=("coverage-judgement",))
    assert parse_skill_stack("policy=", base).policy is None


def test_parse_empty_string_is_the_base():
    base = SkillStack(general=("coverage-judgement",))
    assert parse_skill_stack("", base) == base


def _scripted_system_prompt(tmp_path: Path, **run_kwargs) -> str:
    """The system message a real run was actually given, with a scripted provider.

    Asserting on the prompt the runtime built, not on a second assembly of it. `skill_stack`
    is read at exactly one place inside `run_patient`, and the only way to see whether that
    place honoured the argument is to read the bytes that reached the model.
    """
    pytest.importorskip("deepagents")
    from hooks_harness import ToolScript

    from acr.chartstore.corpus import Corpus
    from acr.contract.spec import load_spec
    from acr.review.agent import run_patient

    model = ToolScript(script=[], submit={"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                          "reasoning": "the script submits at once"})
    model.seen = []
    run_patient(spec=load_spec(SPEC), corpus=Corpus(CORPUS), patient_id="SYN0001",
                out_dir=tmp_path, model=model, max_model_calls=1, seed=7,
                run_id="skill-stack", **run_kwargs)
    systems = [m.content for turn in model.seen for m in turn
               if getattr(m, "type", None) == "system"]
    assert systems, "the scripted provider never saw a system message"
    return "\n".join(str(c) for c in systems)


def _was_delivered(stack, name: str) -> bool:
    """Whether this card was DELIVERED to the model, which since 2026-08-06 is not the same as
    rendered into the prompt.

    It used to be: `skills_block` concatenated every selected body into the system message, so
    "was the model given this card" was answerable by reading the prompt. Under progressive
    disclosure the body is seeded into the agent's `StateBackend` and `SkillsMiddleware` advertises
    its name and description; the model opens it with `read_file` if it judges the card relevant.
    A card can therefore be correctly delivered and legitimately never appear in the prompt.

    So the question moves to what the run seeded. `skill_files` is the one function that answers it
    and it is the same one `run_chart_review` calls, which keeps the property this file exists for:
    the manifest must not name a card the model was never given.
    """
    from acr.contract.skills import SKILLS_MOUNT, skill_files
    return f"{SKILLS_MOUNT}{name}/SKILL.md" in skill_files(stack)


def test_run_patient_without_a_stack_renders_the_profiles_own(tmp_path: Path):
    """`skill_stack=None` is the path every recorded run has taken; it must not move."""
    from acr.review.runtime_profiles import DEFAULT_RUNTIME_PROFILE, runtime_policy_skills
    stack = runtime_policy_skills(DEFAULT_RUNTIME_PROFILE)
    assert _was_delivered(stack, "coverage-judgement")
    assert not _was_delivered(stack, "tactic-query-formulation")


def test_run_patient_renders_the_stack_it_was_given(tmp_path: Path):
    """An explicit stack replaces the profile's, and it is the override that reaches the model."""
    stack = SkillStack(policy="policy-reactive")
    assert _was_delivered(stack, "policy-reactive")
    assert not _was_delivered(stack, "coverage-judgement"), (
        "the profile's own card was delivered despite being overridden")


def _scripted_manifest_skills(tmp_path: Path, **run_kwargs) -> list[dict]:
    """`prompt_assets.skills` from a real run, as the manifest actually records it."""
    pytest.importorskip("deepagents")
    from hooks_harness import ToolScript

    from acr.chartstore.corpus import Corpus
    from acr.contract.spec import load_spec
    from acr.review.agent import run_patient

    model = ToolScript(script=[], submit={"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                          "reasoning": "the script submits at once"})
    out = run_patient(spec=load_spec(SPEC), corpus=Corpus(CORPUS), patient_id="SYN0001",
                      out_dir=tmp_path, model=model, max_model_calls=1, seed=7,
                      run_id="manifest-stack", **run_kwargs)
    return out["prompt_assets"]["skills"]


def test_the_manifest_records_the_stack_the_model_was_actually_given(tmp_path: Path):
    """The prompt and the manifest must name the SAME cards.

    This is a regression test for a defect found on the first live run against the synthetic
    corpus. `--skills policy=tactic-coverage-pool` was honoured by the prompt builder and
    ignored by `prompt_asset_manifest`, which re-derived the stack from the runtime profile. The
    run therefore produced an artifact asserting it had used the profile's default guidance while
    the model had read a different card — and two arms of a retrieval ablation would have
    compared as identical on exactly the axis they were varying.

    A manifest that names the wrong asset is worse than one that names none: the second is a gap
    a reader can see. Asserting agreement between the two, rather than the value of either, is
    what makes the two halves impossible to drift apart again.
    """
    stack = SkillStack(policy="policy-reactive")
    recorded = _scripted_manifest_skills(tmp_path / "m", skill_stack=stack)

    assert [e["skill"] for e in recorded] == ["policy-reactive"]
    assert [e["slot"] for e in recorded] == ["policy"]
    for entry in recorded:
        assert _was_delivered(stack, entry["skill"]), (
            f"the manifest claims {entry['skill']!r} but the model was never given it")
    assert not _was_delivered(stack, "coverage-judgement"), (
        "the profile's card was delivered despite being overridden")


def test_the_manifest_records_the_profiles_stack_when_nothing_overrode_it(tmp_path: Path):
    """`skill_stack=None` — the path every recorded run took — must not have moved."""
    recorded = _scripted_manifest_skills(tmp_path)
    assert [(e["skill"], e["slot"]) for e in recorded] == [
        ("tool-contract", "general"), ("coverage-judgement", "general")]


def _invoke(monkeypatch, tmp_path: Path, *args):
    """Run a CLI command with the model client wired to explode if it is ever built."""
    from typer.testing import CliRunner

    from acr.commands.cli import app
    from acr.core import cli_common

    def refuse(*a, **kw):
        raise AssertionError("a model client was built before --skills was validated")

    monkeypatch.setattr(cli_common, "chat_model", refuse)
    return CliRunner().invoke(app, [*args, "--spec", str(SPEC), "--corpus", str(CORPUS),
                                    "--out", str(tmp_path)])


def test_a_bad_skills_string_costs_nothing_on_run(monkeypatch, tmp_path: Path):
    """A typo in `--skills` must be refused before the first token is paid for.

    Validation after the client is built is validation that has already spent the run's
    startup, and validation after the first call is a bill for a misspelling.
    """
    r = _invoke(monkeypatch, tmp_path, "run", "SYN0001", "--skills", "polish=x")
    assert r.exit_code != 0
    assert isinstance(r.exception, SkillError), r.output
    assert "unknown slot 'polish'" in str(r.exception)


def test_a_bad_skills_string_costs_nothing_on_batch(monkeypatch, tmp_path: Path):
    """`batch` swallows a per-patient exception, so an unparsed override would be retried.

    The parse therefore has to happen OUTSIDE the loop: inside it, a misspelt slot would be
    printed once per patient as if the chart had failed, and the summary would record ten
    failures whose cause was one typo.
    """
    r = _invoke(monkeypatch, tmp_path, "batch", "--patients", "SYN0001",
                "--skills", "policy=chart-triage")
    assert r.exit_code != 0
    assert isinstance(r.exception, SkillError), r.output
    assert "declares slot 'general'" in str(r.exception)


@pytest.mark.parametrize("command", ["run", "batch"])
def test_the_skills_help_documents_every_form_of_the_syntax(command: str):
    """A syntax nobody can discover is a syntax nobody can use — including `|`.

    `parse_skill_stack` accepts four forms and three of them are invisible from the flag name.
    The pipe is the worst of them: comma is already the clause separator, so the obvious guess
    for "two general cards" silently parses as two clauses instead of one list.
    """
    from typer.testing import CliRunner

    from acr.commands.cli import app

    out = CliRunner().invoke(app, [command, "--help"]).output
    flat = " ".join(out.split())
    assert "--skills" in flat
    for form in ("policy=", "tactics=+", "general=+", "|"):
        assert form in flat, f"{command} --help does not document {form!r}"


# ---------------------------------------------------------------- the experience slot
# 2026-08-02: `experience` has been written into SLOTS all along, and
# `experience-adapter/SKILL.md` has declared `slot: experience` all along, but `SkillStack` had no
# such field and `parse_skill_stack` refused the name outright. Which means the third factor of the
# three-factor experiment — the prior summarised out of the develop set — could not be stacked into
# any run at all. Declaring a slot you cannot fill is not the same as not having the slot: the
# first one looks present in the SLOTS list.

def test_the_experience_slot_can_actually_be_stacked():
    stack = parse_skill_stack("experience=experience-adapter", SkillStack())
    assert stack.experience == ("experience-adapter",)
    assert "experience-adapter" in stack.names()


def test_experience_renders_after_the_tactics_and_before_the_standing_habits():
    """A prior is "something somebody already looked up for you", so it renders after the tactics
    this run chose and before the things every arm has."""
    stack = SkillStack(policy="policy-reactive",
                       tactics=("tactic-coverage-pool",),
                       experience=("experience-adapter",),
                       general=("tool-contract",))
    assert stack.names() == ("policy-reactive", "tactic-coverage-pool",
                             "experience-adapter", "tool-contract")


def test_the_manifest_says_which_card_was_the_experience_one():
    """Otherwise "did this arm have a prior turned on or not" can only be guessed from the card's
    name."""
    rows = skills_manifest(SkillStack(experience=("experience-adapter",)))
    assert [(r["skill"], r["slot"]) for r in rows] == [("experience-adapter", "experience")]


def test_experience_appends_with_plus_like_the_other_list_slots(tmp_path: Path):
    """Two prior cards — a keyword prior plus a document-type prior — is the shape this slot expects.

    The tree holds only one `slot: experience` card today, so the second one is built in tmp_path —
    what is under test is the `+` mechanism, not any particular card.
    """
    for n in ("prior-a", "prior-b"):
        d = tmp_path / n
        d.mkdir()
        (d / "SKILL.md").write_text(
            f"---\nname: {n}\ndescription: x\nmetadata:\n  slot: experience\n---\n\nbody\n",
            encoding="utf-8")
    base = parse_skill_stack("experience=prior-a", SkillStack(), tmp_path)
    assert parse_skill_stack("experience=+prior-b", base, tmp_path).experience == (
        "prior-a", "prior-b")


def test_a_card_placed_in_the_wrong_slot_is_still_refused():
    with pytest.raises(SkillError, match="declares slot"):
        SkillStack(experience=("tool-contract",)).validate()


def test_the_default_profile_delivers_the_universal_card_then_the_standing_habit():
    """WHICH cards the default profile delivers, and in what order.

    This used to pin the rendered BLOCK — every byte of the header, the separators, and both card
    bodies — because `skills_block` concatenated them into the prompt and a reworded header would
    silently move every run onto a different prompt. Progressive disclosure removed the block: the
    bodies are seeded into the agent's backend and the model opens one when it judges the card
    relevant, so there are no assembled bytes left to pin.

    What survives is the property the byte-pinning was protecting — the SELECTION. Two cards, in
    this order, is what every arm is compared against, and `tool-contract` is first because a tool
    fact belongs to every arm rather than to whichever card happened to know it.
    """
    from acr.contract.skills import SKILLS_MOUNT, skill_files
    from acr.review.runtime_profiles import DEFAULT_RUNTIME_PROFILE, runtime_policy_skills
    stack = runtime_policy_skills(DEFAULT_RUNTIME_PROFILE)
    assert stack.names() == ("tool-contract", "coverage-judgement")

    delivered = skill_files(stack)
    assert f"{SKILLS_MOUNT}tool-contract/SKILL.md" in delivered
    assert f"{SKILLS_MOUNT}coverage-judgement/SKILL.md" in delivered
    assert not [k for k in delivered if "tactic-query-formulation" in k], (
        "a card the profile did not select was delivered anyway")
