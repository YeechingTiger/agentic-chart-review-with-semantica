"""The static system prompt as a REGISTRY of named blocks, so one can be dropped and recorded.

`run_patient` assembled this as a single `+` chain of ten expressions. Every one of them is a
prompt intervention with a story behind it -- the tumour anchor came from three runs that answered
about the wrong neoplasm, the document-concept reference replaced a substring list that used to
gate reads -- and not one of them could be removed without editing the runtime. So none has ever
been measured: "baseline" and "baseline after I edited `agent.py`" are not two arms, because
nothing in the manifest would say what changed. Two of the ten are large. On STORE.390 under
`current-stratified-coverage` the whole prompt is 20,531 characters, of which `skills` is 9,117 and
`document_concepts` 2,902 -- more than half between them, and neither has ever been ablated.

THE DEFAULT SELECTION IS BYTE-IDENTICAL TO THE CHAIN IT REPLACES, and that is the one property
here that must not be wrong. Every manifest under `runs/` was produced by that chain; a refactor
shifting one separator makes every recorded baseline unreproducible while every manifest still
claims the same `spec_hash`. The chain's arithmetic reduces to a single identity: a block rendering
`""` contributed nothing at all, not even a separator, and every block after the first was joined
with `"\\n\\n"` -- so the whole assembly is `"\\n\\n".join(p for p in parts if p)`, which is
`assemble_prompt` below. `tests/test_the_prompt_is_a_registry.py` checks it against a SECOND
assembly built from the producers directly, because a test that called this module to check this
module would pass on any bug it contains.

WHAT A BLOCK RENDERS IS NOT THIS MODULE'S BUSINESS. Each `render` calls the producer that already
owned the text, with the arguments the chain gave it, and returns `""` in the same cases. Two
producers end their text with their own newline -- `spec.as_prompt_block` and `anchor_block` -- so
the assembled prompt contains `"\\n\\n\\n"` at those two joins. That is a fact about a block's
text, it was true of every recorded run, and tidying it here would be a prompt change wearing a
refactor's clothes.

WHY THIS IS NOT A RUNTIME PROFILE. The five profiles all answer one question -- how far a run must
search before an absence claim stands -- and which blocks are present is orthogonal to all three of
their branch points. Folding it in would rebuild the confound `--planner` was just split out of:
one flag moving two decisions, and no arm able to attribute a result to either.

WHAT IS DELIBERATELY NOT HERE. `AuditMiddleware.wrap_model_call` rebuilds four more blocks on every
model call -- the PLAN, the coverage asset, the open threads, the pending triggers. They are
rebuilt per call from the live ledgers so that exactly one copy of each exists and it is always
current; that is the fix worth 41% of one measured run's prompt tokens, and it is not a static
selection. No `when` field is declared here in anticipation of them: a field that every entry sets
to the same value and nothing reads is the inert mechanism `tools/verify_structure.py` exists to
name, and the honest time to add it is when the per-call blocks actually move.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from ..contract.code_tables import code_domain_block
from ..contract.retrieval_prior import to_experience_asset
from ..contract.skills import skills_block
from ..contract.trace import rule_citation_block
from .document_concepts import anchor_block, baseline_block, experience_block
from .runtime_profiles import runtime_policy_instruction, uses_clinical_contract_view


@dataclass(frozen=True)
class PromptContext:
    """Everything the blocks read, and nothing else.

    One object rather than ten parameters, for the reason `RunContext` exists: a block that reached
    for something outside this list would be a block whose text depends on state the manifest does
    not record. `skill_stack` is THE STACK THAT WAS RENDERED -- the same object the manifest names,
    not a second derivation from the profile, which is the defect `prompt_asset_manifest`'s own
    docstring records from the first live run against the synthetic corpus.
    """
    spec: Any
    patient_id: str
    runtime_profile_asset: Any
    skill_stack: Any
    retrieval_prior: Any = None
    task_context: str = ""


@dataclass(frozen=True)
class PromptBlock:
    """One named block. `render` returning `""` means the block is ABSENT, separator included.

    `required` is refused by name in `selected_blocks` rather than discovered as a run that read
    nothing, and `why_required` is the sentence a reader gets when it refuses -- an enforcement with
    no stated reason is the kind that gets deleted by whoever it next inconveniences.
    """
    name: str
    render: Callable[[PromptContext], str]
    required: bool = False
    why_required: str = ""


# ================================================================ the blocks, in render order
# Each producer keeps its own text; these functions say WHICH producer, with WHICH arguments, and
# carry the reason the block is in the prompt at all. That reason used to be a comment inside the
# `+` chain, where it could not be read without reading the assembly.

def _spec(c: PromptContext) -> str:
    """THE TASK CONTRACT: the question, the fields, the decision rules and the outcome space.

    Which view is the profile's choice. `clinical_contract` shows the model the clinical question
    without the runtime's own machinery; `full` shows the contract as written.
    """
    return c.spec.as_prompt_block(
        view=("clinical_contract" if uses_clinical_contract_view(c.runtime_profile_asset.ref)
              else "full"))


def _rule_ids(c: PromptContext) -> str:
    """THE RULE IDENTIFIERS, in the prompt rather than only in the finalize question.

    The self-report channel asks at submit time which decision rule was applied, and
    `submit_answer` is reachable from any turn -- an agent asked at the last moment to cite
    identifiers it has never seen invents them, and `rule_attribution.self_reported` then records
    invented ids as if they were a measurement of the agent's reasoning.
    """
    return rule_citation_block(c.spec)


def _task(c: PromptContext) -> str:
    """THE STANDING INSTRUCTION: work by calling tools, and `submit_answer` is gated.

    `TASK` stays in `agent.py`, beside the runtime whose gate contract shapes it, and is imported
    here per call so the two modules do not have to be imported in one order.
    """
    from .agent import TASK
    return TASK.format(patient=c.patient_id)


def _runtime_policy(c: PromptContext) -> str:
    """WHICH SEARCH POLICY IS IN FORCE, said to the model in the profile's own words.

    Without it the arm is invisible from inside the run: five profiles differ in what they will
    accept as an absence claim and the model would have to infer which one it is under.
    """
    return runtime_policy_instruction(c.runtime_profile_asset.module_id)


def _document_concepts(c: PromptContext) -> str:
    """PORTABLE DOCUMENT CONCEPTS, as reference. Standard names and prose saying what each kind of
    document is; no local type string, no ordering, no measurement. It replaces `doc_type_matches`,
    which was the same knowledge written as a substring list that gated reads and fed the coverage
    gate. The model reads this beside `document_type_summary` and decides for itself.
    """
    return baseline_block()


def _experience(c: PromptContext) -> str:
    """THE MEASURED PRIOR, when one was supplied: which document types carried the answer on OTHER
    patients and at what rate, and which terms surfaced an answer-bearing document. `""` when there
    is none, so the baseline arm's prompt is byte-identical to a run predating this channel.
    """
    return experience_block(to_experience_asset(c.retrieval_prior) if c.retrieval_prior else None)


def _value_domain(c: PromptContext) -> str:
    """THE VALUE DOMAIN, when the Task Contract declares one. A run asserted "C341 is right middle
    lobe" and coded C341 over evidence reading "right middle lobe" (C341 is the upper lobe), and
    another coded histology 7205, which is not a morphology. Both are facts about a published
    classification, so the model is shown the table instead of being trusted to recall it -- and
    instead of being refused by a regex afterwards. 5,839 characters on STORE.400, nothing on
    STORE.390, which is why a conditional block is where an off-by-one separator would hide.
    """
    return code_domain_block(c.spec)


def _anchor(c: PromptContext) -> str:
    """WHICH tumour, as opposed to which documents. Three runs answered about the wrong neoplasm and
    the traces could not say why; this asks the model to enumerate its candidates and name the one
    it answered for.
    """
    return anchor_block()


def _skills(c: PromptContext) -> str:
    """METHOD GUIDANCE. Until 2026-07-30 nothing in this tree read a SKILL.md body into a prompt, so
    moving the coverage obligation into `assets/skills/coverage-judgement/` deleted it rather than
    relocating it. The profile chooses which cards load; see `acr.contract.skills`. 9,117 characters
    of the 20,531 in a STORE.390 prompt, and the block whose ablation this registry exists for.
    """
    return skills_block(c.skill_stack)


def _task_context(c: PromptContext) -> str:
    """THE OPERATOR'S BRIEF, appended last. `conflict_refinement` is its only caller and the brief
    is what DEFINES that arm. Stripped here because the chain stripped it: a brief that is all
    whitespace is no brief, and it must not leave a separator behind.
    """
    return c.task_context.strip()


#: The static prompt, in render order. ADDING A BLOCK IS ONE ENTRY -- nothing outside this tuple
#: knows a block's name, which is what makes the selection a mechanism rather than a list.
BLOCKS: tuple[PromptBlock, ...] = (
    PromptBlock("spec", _spec, required=True,
                why_required="without the Task Contract there is no question for the run to "
                             "answer, and a run that answers a question it was not given is worse "
                             "than a refusal at the door"),
    PromptBlock("rule_ids", _rule_ids),
    PromptBlock("task", _task, required=True,
                why_required="the standing instruction is where the model is told to work by "
                             "calling tools and that submit_answer is gated; without it a run "
                             "reads nothing and abstains, which is an expensive way to discover "
                             "a dropped block"),
    PromptBlock("runtime_policy", _runtime_policy),
    PromptBlock("document_concepts", _document_concepts),
    PromptBlock("experience", _experience),
    PromptBlock("value_domain", _value_domain),
    PromptBlock("anchor", _anchor),
    PromptBlock("skills", _skills),
    PromptBlock("task_context", _task_context),
)

#: Derived from the entries, not restated beside them: a second list would be a second place to
#: forget, and the enforcement and its reason belong to the block.
REQUIRED_BLOCKS: tuple[str, ...] = tuple(b.name for b in BLOCKS if b.required)


class PromptBlockError(ValueError):
    """A selection that names something unknown, or drops a block that cannot be dropped."""


def parse_block_names(value: str | Iterable[str] | None) -> list[str] | None:
    """ONE GRAMMAR for the selection, wherever it arrives from. `None` means every block.

    `--prompt-blocks` hands over a comma string, a caller that already validated one hands over the
    parsed list, and both have to mean the same thing -- a flag and a direct call that disagreed
    about `""` would be two behaviours under one recorded name. Empty, blank and `None` all resolve
    to `None`, which is today's behaviour and what every recorded run took.
    """
    if value is None:
        return None
    names = (value.split(",") if isinstance(value, str) else [str(n) for n in value])
    return [n.strip() for n in names if n.strip()] or None


def selected_blocks(names: Iterable[str] | None) -> tuple[PromptBlock, ...]:
    """The blocks a selection asks for, IN REGISTER ORDER. `None` or empty means every block.

    Order comes from `BLOCKS` and never from the caller's list, so `--prompt-blocks skills,spec`
    and `--prompt-blocks spec,skills` are one arm rather than two prompts that hash alike by
    accident. Duplicates collapse for the same reason.

    Refuses rather than silently dropping: a typo'd block name that quietly produced the default
    selection would be an arm the operator believes they ran and nobody can find.
    """
    wanted = list(names or ())
    if not wanted:
        return BLOCKS
    known = {b.name for b in BLOCKS}
    if unknown := [n for n in wanted if n not in known]:
        raise PromptBlockError(
            f"unknown prompt block(s) {', '.join(unknown)}; one of "
            f"{[b.name for b in BLOCKS]}, or leave it empty for every block")
    chosen = set(wanted)
    if missing := [b for b in BLOCKS if b.required and b.name not in chosen]:
        raise PromptBlockError("; ".join(f"{b.name} cannot be dropped: {b.why_required}"
                                         for b in missing))
    return tuple(b for b in BLOCKS if b.name in chosen)


def assemble_prompt(ctx: PromptContext,
                    blocks: Iterable[PromptBlock] | None = None) -> tuple[str, dict[str, int]]:
    """The system prompt, and how many characters each selected block contributed.

    THE JOIN IS THE WHOLE REFACTOR. The chain it replaces opened with the contract, guarded five of
    the remaining nine with `(f"\\n\\n{x}" if x else "")`, and prefixed the other four with a bare
    `"\\n\\n"`. The two forms agree exactly while no unconditionally prefixed block renders empty,
    and none can: the standing instruction, the profile instruction, the concept reference and the
    anchor are non-empty by construction, as is the contract that opens it. So the arithmetic is
    `"\\n\\n".join(non-empty)` and nothing else.

    The sizes are returned rather than recomputed by the caller, because they are what the manifest
    records: a block dropped and not recorded is the `--skills` defect of 2026-07-30 again, where
    the card reached the model and the manifest named the profile's default. A selected block that
    rendered nothing is recorded as `0`, which is a different fact from not being selected at all.
    """
    chosen = BLOCKS if blocks is None else tuple(blocks)
    rendered = [(b.name, b.render(ctx)) for b in chosen]
    return ("\n\n".join(text for _, text in rendered if text),
            {name: len(text) for name, text in rendered})
