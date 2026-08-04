"""The static prompt was the system message AND the opening human turn.

[agent.py](../src/acr/review/agent.py) built one ~25 KB string and used it twice:

    agent = create_agent(model, tools, system_prompt=system_prompt, middleware=middleware)
    ...
    agent.invoke({"messages": [{"role": "user", "content": system_prompt}]}, ...)

LangGraph needs at least one human message to start, so *something* had to go in that turn, and the
whole prompt went in. It then rides in `messages` for the rest of the run while `wrap_model_call`
rebuilds the system message from scratch on every call — so the contract, the rule identifiers, the
document concepts, the tumour anchor and both skill cards are paid for twice per model call, once as
instructions and once as a user asking a question in the voice of a specification.

## Why this is fixed before anything is measured

Cost is a reported outcome of the 100/100 study, and the headline is a ratio: a query-only arm is one
call over a few hundred tokens of search hits, against an agent arm's ~150k. A doubled prompt does
not cancel out of that ratio — it inflates only the denominator. Measuring first and fixing later
would mean rerunning every arm.

It is not only cost. The opening turn is the model's *question*, and a 25 KB question that is a
verbatim copy of its own instructions is a different prompt from a short one. Which of the two
performs better is an empirical matter this repo has no result on; what it cannot do is leave the
duplication in place and call the result a baseline.

## What replaces it

`OPENING_TURN` — one line naming the patient and pointing at the instructions already present. The
patient id is in it because that is the one fact the turn has to carry: the system prompt's TASK
block is formatted with the same id, and a run whose opening turn named a different patient would be
a defect nothing else could see.
"""

from __future__ import annotations

import pytest

from acr.contract.spec import load_spec
from acr.core import site

SPEC = site.specs_root() / "STORE.390.date_of_initial_diagnosis.yaml"
CORPUS = site.corpus_root()


def _turns(tmp_path, patient="SYN0001"):
    """The messages a real scripted run gave the provider. No model, no cost."""
    pytest.importorskip("deepagents")
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from hooks_harness import ToolScript

    from acr.chartstore.corpus import Corpus
    from acr.review.agent import run_patient

    model = ToolScript(script=[], submit={"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                          "reasoning": "the script submits at once"})
    model.seen = []
    m = run_patient(spec=load_spec(SPEC), corpus=Corpus(CORPUS), patient_id=patient,
                    out_dir=tmp_path, model=model, max_model_calls=1, seed=7, run_id="once")
    first = model.seen[0]
    humans = [x for x in first if getattr(x, "type", None) == "human"]
    systems = [x for x in first if getattr(x, "type", None) == "system"]
    return m, humans, systems


def test_the_opening_turn_is_not_a_copy_of_the_instructions(tmp_path):
    """The property, stated as a size relation rather than a magic number.

    A ceiling in bytes would need editing every time a prompt block grows. The relation that has to
    hold is that the opening turn is a small fraction of the instructions — anything else means the
    instructions are being repeated into it.
    """
    _, humans, systems = _turns(tmp_path)
    assert humans, "the graph needs a human turn to start"
    opening = str(humans[0].content)
    instructions = "\n".join(str(s.content) for s in systems)
    assert len(instructions) > 5_000, "sanity: the system prompt really is the large one"
    assert len(opening) < len(instructions) / 10, (
        f"the opening turn is {len(opening)} chars against {len(instructions)} of instructions — "
        f"it is repeating them")


def test_no_prompt_block_is_repeated_into_the_opening_turn(tmp_path):
    """Named blocks, so a partial regression is caught too. Each probe is a heading the block's own
    renderer emits, so a reworded block does not need this test edited."""
    _, humans, _ = _turns(tmp_path)
    opening = str(humans[0].content)
    for heading in ("DOCUMENT CONCEPTS", "WHICH TUMOUR THIS ANSWER IS ABOUT",
                    "METHOD GUIDANCE", "RULE IDENTIFIERS", "RUNTIME SEARCH PROFILE"):
        assert heading not in opening, f"{heading!r} is in the opening turn as well as the system"


def test_the_opening_turn_names_the_patient_it_is_about(tmp_path):
    """The one fact the turn must carry. The TASK block in the instructions is formatted with the
    same id, so an opening turn naming a different patient is a defect nothing else could see."""
    _, humans, _ = _turns(tmp_path, patient="SYN0002")
    assert "SYN0002" in str(humans[0].content)


def test_the_instructions_still_carry_everything(tmp_path):
    """THE OTHER HALF. Removing the duplicate must not remove the content — if a block only ever
    reached the model through the human turn, deleting that turn deletes the block."""
    _, _, systems = _turns(tmp_path)
    instructions = "\n".join(str(s.content) for s in systems)
    for heading in ("DOCUMENT CONCEPTS", "WHICH TUMOUR THIS ANSWER IS ABOUT",
                    "METHOD GUIDANCE", "RULE IDENTIFIERS", "RUNTIME SEARCH PROFILE"):
        assert heading in instructions, f"{heading!r} reached the model through neither channel"


def test_the_run_still_completes_and_terminates_for_a_stated_reason(tmp_path):
    """A prompt change that breaks the loop is not a saving.

    `max_model_calls=1` here, so the call limit binds before the submission clears the gate and the
    status is `NO_ANSWER` — that is the harness, not the prompt. What must hold is that the run
    reaches the manifest, its status is one the contract declares, and the termination reason is one
    of the seven rather than a crash.
    """
    m, _, _ = _turns(tmp_path)
    assert m["termination_reason"] != "RUNTIME_ERROR", m["answer"].get("reasoning")
    assert m["degradation"]["runtime_or_provider_errors"] == 0
    # `NO_ANSWER` is the runtime's own stop status and STORE.390 does not declare it, so
    # `status_kind` is `"undeclared"`. That is pre-existing and it is the honest record — what must
    # hold is that an undeclared status is never read as value-carrying, because `RunRecord.abstained`
    # is `status_kind != "value"` and a wrong answer there scores a failed run as an answer.
    kind = m["answer"]["status_kind"]
    assert kind in ("value", "abstain_evidence", "abstain_spec", "undeclared"), kind
    if kind == "undeclared":
        from acr.contract.answer_contract import status_kind
        assert status_kind(load_spec(SPEC), m["answer"]["status"]) is None, (
            "a status the contract DOES declare must not be recorded as undeclared")
