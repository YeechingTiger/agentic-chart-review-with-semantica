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
import yaml

from acr.contract.skills import (
    SLOTS,
    SkillError,
    SkillStack,
    load_skill_body,
    parse_skill_stack,
    skill_slot,
    skills_block,
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
    import yaml as _y
    _fm = _y.safe_load(_FM.match((SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")).group(1))
    if str(_fm.get("kind") or "prose") != "prose":
        pytest.skip("only prose skills are assembled into a prompt, so only they need a slot")
    slot = skill_slot(name)
    assert slot in SLOTS, f"{name}: slot {slot!r} not one of {SLOTS}"


@pytest.mark.parametrize("name", _skill_names())
def test_declared_slot_matches_the_file(name: str):
    """skill_slot reads exactly that one line in the file, not something inferred elsewhere."""
    text = (SKILLS_DIR / name / "SKILL.md").read_text(encoding="utf-8")
    fm = yaml.safe_load(_FM.match(text).group(1))
    if str(fm.get("kind") or "prose") != "prose":
        pytest.skip("a slot is where PROSE goes in a prompt; a script/llm/subagent skill is "
                    "invoked through contract/skill_invoke.py and never rendered")
    assert fm["slot"] == skill_slot(name)


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
        "---\nname: bad-slot\ndescription: x\nslot: wherever\n---\n\nbody\n", encoding="utf-8")
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
        "---\nname: eval-contrast-traces\ndescription: x\nslot: eval\n---\n\nbody\n",
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


def test_default_profile_renders_the_universal_block_then_the_standing_habits():
    """The bytes the default profile sends the model, nailed down here.

    This test used to be named `..._exactly_what_it_rendered_before`, on these grounds: every run
    in this tree's history ran under the single card `coverage-judgement`, and a refactor that
    casually slipped in two more would make the old and the new runs incomparable. That reason still
    holds, and 2026-08-02 broke it **on purpose**: `tool-contract` was added to every profile.

    Why breaking comparability once was worth it — an outside review pointed out that
    `tactic-query-formulation` knows the tool facts (substring matching, the hit cap, the scan
    order) and the other cards do not, so the differences between the seven arms always had "who
    happened to understand the instrument" mixed into them. A tool fact is not a strategy; it
    belongs to every arm. The break is a one-off, and from here on this test pins the bytes as
    before.
    """
    from acr.review.runtime_profiles import DEFAULT_RUNTIME_PROFILE, runtime_policy_skills
    stack = runtime_policy_skills(DEFAULT_RUNTIME_PROFILE)
    assert stack.names() == ("tool-contract", "coverage-judgement")
    assert skills_block(stack).endswith(load_skill_body("coverage-judgement"))
    # `endswith` only pins the tail. The header and the separators are prompt bytes too, and a
    # reworded header would move every run onto a different prompt while the manifest — which
    # hashes the skill BODY, not the block — went on reporting the same content hash. So the
    # whole block is spelled out here, character for character, as it rendered before slots
    # existed.
    assert skills_block(stack) == "\n".join([
        "METHOD GUIDANCE — JUDGEMENT YOU APPLY, NOT CONDITIONS THE RUNTIME ENFORCES",
        "",
        "Nothing below is checked mechanically. It is how a careful reviewer approaches these "
        "questions, and where it does not fit this chart you should depart from it and say so in "
        "your reasoning. Your departure is recorded, not refused.",
        "",
        "--- skill: tool-contract ---",
        "",
        load_skill_body("tool-contract"),
        "",
        "--- skill: coverage-judgement ---",
        "",
        load_skill_body("coverage-judgement"),
    ])


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


def _was_rendered(prompt: str, name: str) -> bool:
    """Whether this card was assembled into the system prompt the model was given.

    Checked by its `skills_block` separator AND its own opening line, rather than by its whole
    body, because the runtime hands the provider a system message in which part of the text has
    been through `repr` — newlines and apostrophes arrive escaped, so a multi-line body never
    matches verbatim no matter what was rendered. Both probes here are single escape-free lines
    and both are READ FROM THE SKILL, so a reworded card does not need this test edited.
    `skills_block`'s exact bytes are pinned separately by the default-profile test above.
    """
    return (f"--- skill: {name} ---" in prompt
            and load_skill_body(name).splitlines()[0] in prompt)


def test_run_patient_without_a_stack_renders_the_profiles_own(tmp_path: Path):
    """`skill_stack=None` is the path every recorded run has taken; it must not move."""
    prompt = _scripted_system_prompt(tmp_path)
    assert _was_rendered(prompt, "coverage-judgement")
    assert not _was_rendered(prompt, "tactic-query-formulation")


def test_run_patient_renders_the_stack_it_was_given(tmp_path: Path):
    """An explicit stack replaces the profile's, and it is the override that reaches the model."""
    prompt = _scripted_system_prompt(
        tmp_path, skill_stack=SkillStack(policy="policy-reactive"))
    assert _was_rendered(prompt, "policy-reactive")
    assert not _was_rendered(prompt, "coverage-judgement")


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
    prompt = _scripted_system_prompt(tmp_path / "p", skill_stack=stack)

    assert [e["skill"] for e in recorded] == ["policy-reactive"]
    assert [e["slot"] for e in recorded] == ["policy"]
    for entry in recorded:
        assert _was_rendered(prompt, entry["skill"]), (
            f"the manifest claims {entry['skill']!r} but the model was never given it")
    assert not _was_rendered(prompt, "coverage-judgement"), (
        "the profile's card reached the prompt despite being overridden")


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
            f"---\nname: {n}\ndescription: x\nslot: experience\n---\n\nbody\n",
            encoding="utf-8")
    base = parse_skill_stack("experience=prior-a", SkillStack(), tmp_path)
    assert parse_skill_stack("experience=+prior-b", base, tmp_path).experience == (
        "prior-a", "prior-b")


def test_a_card_placed_in_the_wrong_slot_is_still_refused():
    with pytest.raises(SkillError, match="declares slot"):
        SkillStack(experience=("tool-contract",)).validate()
