"""Ten prompt blocks assembled as one `+` chain, so not one of them can be ablated.

`run_patient` builds the system prompt as a single expression
([agent.py](../src/acr/review/agent.py), search `system_prompt=(spec.as_prompt_block`). Every block
in it is a prompt intervention, and none has ever been measured, because measuring one means
editing the runtime. Two of them are large: `skills_block` is 9,117 characters and `baseline_block`
2,902 — together roughly half the static prompt.

Concretely, today you cannot run "baseline minus the tumour anchor" or "baseline minus the
document-concept reference". You can only run "baseline" and "a different baseline after I edited
`agent.py`", which is not an arm, because nothing in the manifest would say what changed.

## What a registry has to get right, in order of how badly it bites

1. **The default selection must be byte-identical to the chain it replaces.** Every recorded run in
   `runs/` was produced by that chain. A refactor that shifts one separator makes every baseline
   unreproducible while every manifest still claims the same `spec_hash`. The test below builds the
   expected string from the block producers directly — a second, independent assembly — because a
   test that called the new assembler to check the new assembler would pass on any bug it contains.

2. **A dropped block removes exactly its own text.** Not a separator too many, not one too few. The
   chain's rule is that a block rendering `""` contributes nothing at all, which reduces the whole
   assembly to `"\\n\\n".join(non-empty)` — and that identity is what makes the refactor safe.

3. **The selection reaches the manifest and the arm hash.** A prompt block dropped and not recorded
   is the `--skills` defect of 2026-07-30 again: the card reached the model and the manifest named
   the profile's default, so two arms of a retrieval ablation looked identical while their prompts
   differed by a whole card.

4. **`spec` and `task` cannot be dropped.** Without the contract there is no question; without the
   standing instruction the model is never told to call a tool. Refusing by name is better than
   discovering it as a run that reads nothing.

## Why not a runtime profile

The five profiles all answer one question — how far must a run search before an absence claim
stands. Which prompt blocks are present is orthogonal to all three of their branch points, and
folding it in would rebuild the confound `--planner` was just split out of.
"""

from __future__ import annotations

import inspect

import pytest

from acr.contract.spec import load_spec
from acr.core import site
from acr.review.prompt_blocks import (
    BLOCKS,
    REQUIRED_BLOCKS,
    PromptBlockError,
    assemble_prompt,
    selected_blocks,
)

SPEC_390 = site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml"
SPEC_400 = site.specs_root() / "STORE.400_522_523.site_histology_behavior.yaml"


@pytest.fixture(scope="module")
def ctx():
    """The context the blocks read, assembled the way `run_patient` assembles it."""
    from acr.contract.skills import SkillStack
    from acr.review.prompt_blocks import PromptContext
    from acr.review.runtime_profiles import resolve_runtime_policy
    asset, _ = resolve_runtime_policy("current-stratified-coverage")
    return PromptContext(spec=load_spec(SPEC_390), patient_id="SYN0001",
                         runtime_profile_asset=asset,
                         skill_stack=SkillStack(general=("tool-contract",
                                                         "coverage-judgement")),
                         retrieval_prior=None, task_context="")


# ------------------------------------------------------------------ (1) byte-identical by default

def _expected(c) -> str:
    """The prompt, rebuilt here from the block producers directly.

    A SECOND, INDEPENDENT ASSEMBLY. Checking the new assembler with the new assembler would pass on
    any bug it contains; this reproduces the arithmetic separately, including the rule that a block
    rendering "" contributes no separator.

    IT NO LONGER REPRODUCES THE `+` CHAIN, and that claim ended deliberately on 2026-08-06. The
    skills block used to render every selected card's BODY; it now renders one sentence saying the
    cards exist and are optional, because `SkillsMiddleware` puts their names and descriptions in
    the prompt and the model opens a body with `read_file` when it judges the card relevant.
    Measured on SYN0001, same spec, same answer (20230412), same gate: 226,367 -> 121,350 tokens,
    $0.0108 -> $0.0077, 10 -> 8 model calls.

    The skills block alone is taken from the registry rather than re-derived, because re-deriving
    one sentence in two places makes a change-detector out of a property test. What stays
    independent here — and is what this oracle is for — is the ORDER and the separator arithmetic.
    """
    from acr.contract.code_tables import code_domain_block
    from acr.contract.trace import rule_citation_block
    from acr.review.agent import TASK
    from acr.review.document_concepts import anchor_block, baseline_block, experience_block
    from acr.review.prompt_blocks import _skills
    from acr.review.runtime_profiles import (
        runtime_policy_instruction,
        uses_clinical_contract_view,
    )
    parts = [
        c.spec.as_prompt_block(
            view=("clinical_contract" if uses_clinical_contract_view(c.runtime_profile_asset.ref)
                  else "full")),
        rule_citation_block(c.spec),
        TASK.format(patient=c.patient_id),
        runtime_policy_instruction(c.runtime_profile_asset.module_id),
        baseline_block(),
        experience_block(None),
        code_domain_block(c.spec),
        anchor_block(),
        _skills(c),
        c.task_context.strip(),
    ]
    return "\n\n".join(p for p in parts if p)


def test_the_default_selection_matches_an_independent_assembly(ctx):
    """THE TEST THAT MATTERS MOST: the registry's order and separator arithmetic, checked by a
    second assembly that does not share its code.

    It was `..._is_byte_identical_to_the_chain_it_replaces`, and every manifest under `runs/`
    before 2026-08-06 was produced by that chain. The prompt changed on purpose when the skills
    block stopped carrying card bodies, so those runs are no longer prompt-comparable with new
    ones — which is what `experiment_config_hash` is for, and it moves."""
    text, _ = assemble_prompt(ctx)
    assert text == _expected(ctx)


def test_it_matches_on_a_contract_with_a_value_domain(ctx):
    """STORE.400 declares `value_domain: icdo3_lung`, so `code_domain_block` renders 5,839 chars
    there and nothing on STORE.390. A conditional block is where an off-by-one separator hides."""
    from dataclasses import replace
    c = replace(ctx, spec=load_spec(SPEC_400))
    text, _ = assemble_prompt(c)
    assert text == _expected(c)
    assert "MORPHOLOGY" in text, "sanity: the value domain really did render for this contract"


def test_it_is_byte_identical_with_a_prior_and_a_task_context(ctx):
    """The two blocks that are empty in every recorded run. If the registry gets their separator
    wrong, no existing baseline would reveal it and the first arm to use them would carry it."""
    from dataclasses import replace

    # BUILT FROM `contract.retrieval_prior` DIRECTLY, not through `improvement.prior.build_prior`.
    # That import routed this file to the composer, so the registry's own specification would not
    # have shipped with `prompt_blocks.py` in `acr-chart-review` — and `acr-chart-review` declares
    # no siblings, so it could not satisfy an `acr.improvement` import anyway. The contract type is
    # CONTRACT_SHARED and travels everywhere.
    from acr.contract.retrieval_prior import (
        FieldPrior,
        Measured,
        RetrievalPrior,
        TermYield,
        prior_digest,
    )
    prior = RetrievalPrior(
        asset_id="T.1", version="1", status="measured",
        measured=Measured(n_patients=2, n_notes=2,
                          patient_digests=(prior_digest("P1"), prior_digest("P2")),
                          spec_id=ctx.spec.spec_id, model="m", prompt_hash="h"),
        fields=(FieldPrior(field_name="date_of_initial_diagnosis", n_answer_bearing=2, n_notes=2,
                           terms=(TermYield(term="carcinoma", n_surfaced_answer_bearing=2,
                                            n_surfaced_other=0, basis="proposed_by_reader"),)),))

    c = replace(ctx, retrieval_prior=prior, task_context="  OPTIONAL BRIEF  ")
    text, _ = assemble_prompt(c)
    assert "RETRIEVAL EXPERIENCE" in text
    assert text.endswith("OPTIONAL BRIEF"), "the trailing block is stripped, as the chain did"
    # AT THESE TWO JOINS, not over the whole prompt. `"\n\n\n" not in text` was asserted here and
    # is false of the chain itself: `spec.as_prompt_block` and `anchor_block` each END with their own
    # newline, so the chain has always emitted "\n\n\n" where those two meet the next block — twice
    # on STORE.390. A registry that removed it would fail the byte-identity test above, and the text
    # of a block is out of this refactor's scope either way. What these two blocks can actually get
    # wrong is their own separator, so that is what is checked.
    assert "\n\nRETRIEVAL EXPERIENCE" in text and "\n\n\nRETRIEVAL EXPERIENCE" not in text
    assert text.endswith("\n\nOPTIONAL BRIEF") and not text.endswith("\n\n\nOPTIONAL BRIEF"), (
        "a stripped block must not leave a doubled separator")


# ------------------------------------------------------------------ (2) the registry itself

def test_the_order_has_the_two_properties_that_matter():
    """WAS A HARDCODED LIST OF THE TEN NAMES, which an adversarial reviewer correctly called a pure
    change-detector: its only failure mode was "you added a block and did not update me", and
    `_expected`'s own `parts` list already pins the whole order with an oracle that can fail for a
    reason. Two facts about the order are load-bearing and neither is the list:

      * `spec` is FIRST. The assembly is `"\n\n".join(non-empty)`, so the opening block is the one
        that never carries a leading separator — moving anything ahead of it changes every prompt.
      * `task_context` is LAST, and it is the only block whose render strips. A stripped block that
        is not last would leave the separator on the wrong side of the gap.
    """
    names = [b.name for b in BLOCKS]
    assert names[0] == "spec" and names[-1] == "task_context"
    assert len(names) == len(set(names)), "a duplicated name would render twice"
    assert all(b.name and b.render for b in BLOCKS), "a block with no producer renders nothing"


def test_dropping_a_block_removes_exactly_its_own_text(ctx):
    """Not a separator too many, not one too few."""
    from acr.review.document_concepts import anchor_block
    full, sizes = assemble_prompt(ctx)
    without, _ = assemble_prompt(ctx, blocks=selected_blocks(
        [b.name for b in BLOCKS if b.name != "anchor"]))
    assert len(full) - len(without) == sizes["anchor"] + 2, (
        "the block plus its one separator, and nothing else")
    assert "WHICH TUMOUR" not in without
    assert "DOCUMENT CONCEPTS" in without, "its neighbours are untouched"
    # THE EXCISION, STATED AS TEXT. `"\n\n\n" not in without` was asserted here and cannot be: the
    # prompt carries "\n\n\n" wherever a block's own text ends with a newline, which `anchor_block`
    # and `spec.as_prompt_block` both do, and the byte-identity test above requires that to survive.
    # This says the same thing the length arithmetic says, without the compensating-error blind spot:
    # what is gone is the block and the ONE separator in front of it, and nothing else moved.
    assert without == full.replace(f"\n\n{anchor_block()}", "", 1)


def test_dropping_the_largest_block_is_the_arm_that_could_not_exist(ctx):
    """Whatever the largest optional block is, an arm must be able to drop it without editing code.

    This asserted `sizes["skills"] > 5_000`, which was true while the skills block carried every
    selected card's body — 9,117 characters, a third of the static prompt. Progressive disclosure
    took it to 438 and the assertion became a fossil of the old composition. The property was never
    about skills being large; it is that the LARGEST optional block is ablatable, since that is the
    arm nobody could run while the prompt was a `+` chain in `agent.py`.
    """
    _, sizes = assemble_prompt(ctx)
    optional = {k: v for k, v in sizes.items() if k not in REQUIRED_BLOCKS and v}
    assert optional, "no optional block rendered; there is no ablation to run"
    largest = max(optional, key=optional.get)
    kept, sizes_kept = assemble_prompt(ctx, blocks=selected_blocks(
        [b.name for b in BLOCKS if b.name != largest]))
    assert sizes_kept.get(largest, 0) == 0, f"{largest!r} still rendered after being dropped"
    full, _ = assemble_prompt(ctx)
    assert len(kept) < len(full), "dropping the largest block did not shorten the prompt"
    # and the named one specifically, since `skills` is the block this registry was built for
    without_skills, _ = assemble_prompt(ctx, blocks=selected_blocks(
        [b.name for b in BLOCKS if b.name != "skills"]))
    assert "METHOD GUIDANCE" not in without_skills


def test_a_block_that_renders_nothing_contributes_no_separator(ctx):
    """The chain's own rule, and the identity the whole refactor rests on: assembly is
    `"\\n\\n".join(non-empty)`. STORE.390 declares no value domain, so that block is empty here."""
    _, sizes = assemble_prompt(ctx)
    assert sizes["value_domain"] == 0
    assert sizes["experience"] == 0 and sizes["task_context"] == 0


# ------------------------------------------------------------------ (4) what cannot be dropped

def test_the_contract_and_the_instruction_cannot_be_dropped():
    assert set(REQUIRED_BLOCKS) == {"spec", "task"}
    for name in REQUIRED_BLOCKS:
        with pytest.raises(PromptBlockError, match=name):
            selected_blocks([b.name for b in BLOCKS if b.name != name])


def test_an_unknown_block_refuses_and_names_the_options():
    with pytest.raises(PromptBlockError, match="anchour"):
        selected_blocks(["spec", "task", "anchour"])


def test_the_empty_selection_means_every_block():
    """`""` on the command line has to reproduce today's behaviour exactly, or every recorded
    baseline becomes a different arm."""
    assert [b.name for b in selected_blocks(None)] == [b.name for b in BLOCKS]
    assert [b.name for b in selected_blocks([])] == [b.name for b in BLOCKS]


# ------------------------------------------------------------------ (3) recorded, and in the hash

def test_the_manifest_records_the_selection_and_not_the_sizes(ctx):
    """The sizes moved to the manifest's own `prompt_block_chars` after adversarial verification
    found that hashing them made the arm hash a per-run id — `_task` embeds the patient id and its
    length varies across the cohort. See `test_two_patients_of_one_arm_share_a_hash`."""
    from acr.review.run_manifest import prompt_asset_manifest
    entry = prompt_asset_manifest(ctx.spec, prompt_blocks=["anchor", "skills"])["blocks"]
    assert entry == {"selected": ["anchor", "skills"]}


def test_an_absent_block_record_is_an_explicit_absence():
    from acr.review.run_manifest import prompt_asset_manifest
    assert prompt_asset_manifest(load_spec(SPEC_390))["blocks"] is None


def test_the_selection_is_a_classified_arm_parameter():
    from acr.review.agent import ARM_PARAMETERS, run_patient
    assert ARM_PARAMETERS.get("prompt_blocks")
    assert "prompt_blocks" in inspect.signature(run_patient).parameters


def test_two_selections_are_two_arms(tmp_path):
    """Through the real runtime, offline. A selection that does not move
    `experiment_config_hash` is a prompt change no comparison can see."""
    base, arm = _two_runs(tmp_path, prompt_blocks="spec,rule_ids,task,runtime_policy,"
                                                  "document_concepts,experience,value_domain,skills")
    assert base["experiment_config_hash"] != arm["experiment_config_hash"]
    assert "anchor" in base["prompt_assets"]["blocks"]["selected"]
    assert "anchor" not in arm["prompt_assets"]["blocks"]["selected"]


def test_the_default_selection_records_every_block(tmp_path):
    base, _ = _two_runs(tmp_path, prompt_blocks="")
    sel = base["prompt_assets"]["blocks"]["selected"]
    assert sel == [b.name for b in BLOCKS]


# ------------------------------------------------------------------ reachable, and unbranched

def test_every_run_command_exposes_the_flag():
    from acr.commands.cli_chart import batch, consistency, run
    from acr.commands.cli_pipeline import extract
    for fn in (run, batch, consistency, extract):
        assert "prompt_blocks" in inspect.signature(fn).parameters, fn.__name__


def test_nothing_outside_the_registry_branches_on_a_block_name():
    """A registry whose consumers still say `if name == "anchor"` is a list, not a mechanism.
    Adding a block must be one entry and nothing else."""
    import acr.review.agent as A
    import acr.review.prompt_blocks as P
    outside = inspect.getsource(A)
    for name in [b.name for b in BLOCKS if b.name not in ("spec", "task")]:
        assert f'== "{name}"' not in outside, f"agent.py branches on the block name {name!r}"
    # In the registry module itself the names appear once each, in their own entry.
    src = inspect.getsource(P)
    for name in [b.name for b in BLOCKS]:
        assert src.count(f'"{name}"') >= 1, name


def test_the_flag_refuses_a_bad_value_before_any_model_call():
    import typer

    from acr.commands.cli_chart import _prompt_blocks
    assert _prompt_blocks("") is None
    assert _prompt_blocks("spec,task") == ["spec", "task"]
    with pytest.raises(typer.BadParameter, match="anchour"):
        _prompt_blocks("spec,task,anchour")


# ------------------------------------------------------------------ helper

def _two_runs(tmp_path, *, prompt_blocks: str):
    """The same patient twice through the real runtime, differing by the block selection."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from hooks_harness import run_with_script
    from test_provenance import SHB, _ScriptedLLM

    from acr.chartstore.corpus import Corpus
    corpus, sp = Corpus(site.corpus_root()), load_spec(SHB)
    out = []
    for name, kw in (("base", {}), ("arm", {"prompt_blocks": prompt_blocks})):
        d = tmp_path / name
        d.mkdir()
        llm = _ScriptedLLM({"primary_site": "C341", "histology": "8140", "behavior": "3"})
        m, _ = run_with_script(sp, corpus, "SYN0001", d, llm, run_id=name, max_model_calls=8, **kw)
        out.append(m)
    return out


# ------------------------------------------------------------------ the size record is not the arm

def test_two_patients_of_one_arm_share_a_hash(tmp_path):
    """FOUND BY ADVERSARIAL VERIFICATION, NOT BY THE 18 TESTS ABOVE.

    `_task` renders `TASK.format(patient=patient_id)`, so the block's character count contains the
    patient id's length. The shipped corpus has both 7- and 6-character ids, so recording `n_chars`
    inside `prompt_assets` — which `experiment_config_hash` hashes wholesale — made the arm hash a
    PER-RUN ID:

        SYN0001   n_chars.task 616   hash c829dca0f7f70804
        SYNK01    n_chars.task 615   hash a32d5c9645b12fd5

    Measured consequence: `derive_baseline_key` over a two-patient cohort returned
    `experiment_config_hash: MIXED` and `eval compare` returned `NOT_COMPARABLE`. `chart batch`
    defaults to the whole corpus, which mixes both id lengths, so this fired on the DEFAULT cohort
    and every batch would have been unscoreable. It is the same defect `model_identity`'s docstring
    records having been caught once already.

    The fix is a distinction the tree already draws elsewhere: the SELECTION defines the arm, the
    SIZES are a per-run observation like `usage`. So they are recorded in different places.
    """
    from acr.evaluation import evals as E
    a = _one_run(tmp_path / "a", "SYN0001")
    b = _one_run(tmp_path / "b", "SYNK01")
    assert len(a["patient_id"]) != len(b["patient_id"]), "the fixture needs two id lengths"
    assert a["experiment_config_hash"] == b["experiment_config_hash"], (
        "two patients of one arm must be one arm")
    recs = [E.RunRecord(a, source="a"), E.RunRecord(b, source="b")]
    assert E.derive_baseline_key(recs).experiment_config_hash != E.MIXED


def test_the_sizes_are_still_recorded_just_not_in_the_arm(tmp_path):
    """Moving them out must not lose them: the per-block character count is how a reader sees that
    `skills` is a third of the prompt, and it is what a cost claim rests on."""
    m = _one_run(tmp_path, "SYN0001")
    assert m["prompt_assets"]["blocks"]["selected"] == [b.name for b in BLOCKS]
    assert "n_chars" not in m["prompt_assets"]["blocks"], "a per-run size cannot ride in the arm"
    sizes = m["prompt_block_chars"]
    # Every rendered block's size is recorded; no threshold, because the point is that the numbers
    # live OUTSIDE the arm hash, not that any one of them is big. `skills` was 9,117 and is now 438.
    assert sizes["n_chars"], "no per-block sizes were recorded at all"
    assert "skills" in sizes["n_chars"]
    assert sizes["total_chars"] == sum(sizes["n_chars"].values())


def test_a_selection_change_still_moves_the_hash(tmp_path):
    """The regression guard on the fix: moving the sizes out must not make the SELECTION invisible,
    which is the whole reason the record exists."""
    base = _one_run(tmp_path / "base", "SYN0001")
    arm = _one_run(tmp_path / "arm", "SYN0001", prompt_blocks="spec,task,skills")
    assert base["experiment_config_hash"] != arm["experiment_config_hash"]


# ------------------------------------------------------------------ every path forwards the flag

def test_the_conflict_refinement_path_forwards_the_arm_switches():
    """A FOURTH CALL SITE, and `--planner` was already being dropped on it before this change.

    `cli_chart.run` builds `runner_kwargs` for `run_conflict_refinement`, which forwards them
    verbatim to the same `run_patient`. A switch accepted on the command line and missing from that
    dict is silently discarded — and the refinement rounds would then all run the profile's default
    while the manifest recorded what the operator asked for on the other three paths.
    """
    import inspect

    from acr.commands import cli_chart
    src = inspect.getsource(cli_chart.run)
    kwargs = src.split("runner_kwargs={")[1].split("})")[0]
    for switch in ('"planner"', '"prompt_blocks"'):
        assert switch in kwargs, f"{switch} is not forwarded to the refinement runner"


def _one_run(out, patient, **kw):
    """One offline scripted run through the real runtime. No model, no cost."""
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from hooks_harness import run_with_script
    from test_provenance import SHB, _ScriptedLLM

    from acr.chartstore.corpus import Corpus
    out.mkdir(parents=True, exist_ok=True)
    llm = _ScriptedLLM({"primary_site": "C341", "histology": "8140", "behavior": "3"})
    m, _ = run_with_script(load_spec(SHB), Corpus(site.corpus_root()), patient, out, llm,
                           run_id=patient, max_model_calls=8, **kw)
    return m
