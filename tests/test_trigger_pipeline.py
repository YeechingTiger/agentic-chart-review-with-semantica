"""THE TRIGGER PIPELINE, END TO END — driven through the real agent loop.

WHY THIS FILE EXISTS
--------------------
Thirty-four trigger tests passed while the centrepiece had never once executed. Every one of
them called `triggers_from_tool_result` / `gate_obligation_triggers` DIRECTLY and asserted on
the returned `Trigger` objects, so nothing ever exercised the line that puts a detected
trigger into the trace:

    self.tracer.emit("trigger", **t.to_dict())      # graph.py:1037, graph.py:1060

`Trigger.to_dict()` carries `kind`, and `Tracer.emit(self, kind, **payload)` takes `kind`
positionally, so the first trigger of any live run raised

    TypeError: Tracer.emit() got multiple values for argument 'kind'

and killed the task. The trace tail named UNLISTED_ANSWER_TERM — the single most valuable
develop-plane signal in the rework — as the killer.

A UNIT TEST OF A DETECTOR IS NOT A TEST OF THE PIPELINE IT FEEDS. So every test below drives
the ACTUAL graph with a scripted model and follows one trigger the whole way:

    tool result -> detection -> the trigger reaches the reflection prompt
                -> the supervisor's typed revision -> the runtime applies it
                -> BEHAVIOUR CHANGES afterwards

Behaviour afterwards is the point. Asserting `plan.keywords == [...]` would pass on a plan
nobody reads, which is the family of bug this whole rework exists to kill.

Two further defects, both of which only show up on the production path and neither of which
a detector unit test can see, are held here too:

  * the up-front planner's own proposed terms were charged against the AGENT's expansion
    budget, so the supervisor was shown `EXPANSION REMAINING: terms -3` at its first
    reflection and its first single-term revision was refused BUDGET_EXHAUSTED;
  * a revision one term over budget was refused WHOLE, taking its thread work down with it,
    so the run ended both budget-exhausted and thread-blocked.

No provider is called anywhere in this file, and no chart text is written into it: every
quote used is looked up from the synthetic corpus at run time.
"""
from __future__ import annotations

import ast
import inspect
import json
import re
from pathlib import Path

import pytest

from acr import run_triggers
from acr.corpus import Corpus
from acr.coverage_planner import (MECHANICALLY_DISCHARGEABLE_MARKERS,
                                  OPEN_REQUEST_ALREADY_OPEN, OPEN_REQUEST_ALREADY_SETTLED,
                                  OPEN_REQUEST_DISCHARGED_ON_READ, OPEN_REQUEST_OPENED,
                                  OPEN_REQUEST_STATUSES, READ_STATE_COMPLETE,
                                  READ_STATE_INCOMPLETE, READ_STATE_LENGTH_UNKNOWN,
                                  READ_STATE_UNREAD, REFUSED_BUDGET,
                                  REFUSED_THREAD_NOOP, SETTLED_BY_READING_TO_THE_END,
                                  TRIGGER_GATE_OBLIGATION_UNREACHABLE,
                                  TRIGGER_UNLISTED_ANSWER_TERM, TRIGGER_UNSETTLED_THREAD,
                                  TRIGGER_ZERO_HIT_SEARCH, ExpansionBudget, OpenRequest,
                                  OpenThreadLedger,
                                  PlanRevision, Trigger, load_marker_catalogue, plan_from_spec,
                                  spec_declared_keywords)
from acr.answer_gate import check_threads
from acr.llm import LLMClient, LLMConfig, LLMResponse
from acr.spec import load_spec
from acr.state import Budget
from acr.trace import Tracer

ROOT = Path(__file__).resolve().parents[1]
SHB = ROOT / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"
CORPUS = ROOT / "corpus" / "patients"

#: A type the scripted planner assigns to `sample`, and which really does hold hits for the
#: term that localises this tumour. Promoting it is the move the fourth trigger forces.
SAMPLED_TYPE = "Chest-CT-W-Contr"
PATHOLOGY_TYPE = "Surgical-Pathology-Document"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SHB)


@pytest.fixture(scope="module")
def chart():
    return Corpus(CORPUS).chart("SYN0001")


# ==========================================================================================
# the scripted model
# ==========================================================================================
class ScriptedLLM(LLMClient):
    """One model, four prompts, no provider.

    Each of `acts` and `reflections` is a queue. An entry may be a literal object or a
    callable taking the prompt it is answering — the callable form is what lets a test say
    "add the term the trigger actually suggested", which is the behaviour under test rather
    than a term the test smuggled in.
    """

    def __init__(self, *, acts=(), reflections=(), assignments=(), planner_keywords=(),
                 finalize=None):
        super().__init__(LLMConfig(model="scripted/none", api_key="none"))
        self.acts = list(acts)
        self.reflections = list(reflections)
        self.assignments = list(assignments)
        self.planner_keywords = list(planner_keywords)
        #: What the FINALIZE prompt is answered with. Defaults to the abstention every test
        #: here wanted until section 12 needed a model that insists on a positive from a run
        #: that has not earned one — which is what SYN0001's finalize actually did.
        self.finalize_reply = finalize
        self.reflect_prompts: list[str] = []
        self.act_prompts: list[str] = []
        self.tool_results: list[tuple[str, dict]] = []
        self.finalized = False
        self._n_tool_seen = 0
        self._n_msgs_seen = 0
        self._n_calls = 0
        #: Every system message the runtime handed this model, in order.
        self.seen_system: list[str] = []

    # -- helpers used by the tests ------------------------------------------------
    def results_for(self, name: str) -> list[dict]:
        return [r for n, r in self.tool_results if n == name]

    def _reply(self, obj, calls=None) -> LLMResponse:
        self.calls += 1
        self.prompt_tokens += 10
        self.completion_tokens += 5
        return LLMResponse(content=json.dumps(obj), tool_calls=calls or [],
                           prompt_tokens=10, completion_tokens=5)

    @staticmethod
    def _resolve(item, prompt):
        return item(prompt) if callable(item) else item

    def chat(self, messages, tools=None):
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")

        if tools is None and "read_all|search|sample" in last:
            return self._reply({"assignments": self.assignments,
                                "keywords": self.planner_keywords})

        if tools is None and "SUFFICIENT|CONTINUE|STUCK" in last:
            self.reflect_prompts.append(last)
            j = (self._resolve(self.reflections.pop(0), last) if self.reflections
                 else {"verdict": "CONTINUE", "reason": "keep working the plan"})
            return self._reply(j)

        if tools is None:
            self.finalized = True
            return self._reply(self.finalize_reply
                               or {"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                   "reasoning": "the scripted run never finalised for real"})

        # act. Collect the tool results the loop has produced since the last act step, so a
        # test can assert on what the runtime actually handed back.
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        for m in tool_msgs[self._n_tool_seen:]:
            try:
                self.tool_results.append((m.get("name", ""), json.loads(m["content"])))
            except json.JSONDecodeError:                      # pragma: no cover - defensive
                self.tool_results.append((m.get("name", ""), {}))
        self._n_tool_seen = len(tool_msgs)
        # EVERYTHING the loop has said to the agent since its last act step, not only the
        # last line: an applied revision is followed by the re-rendered plan, so a supervisor
        # message the model must act on is routinely not the final one.
        self.act_prompts.append("\n".join(
            m["content"] for m in messages[self._n_msgs_seen:] if m.get("role") == "user"
        ) or last)
        self._n_msgs_seen = len(messages)
        self._n_calls += 1
        name, args = (self._resolve(self.acts.pop(0), last) if self.acts
                      else ("document_type_summary", {}))
        return self._reply({}, [{"id": f"c{self._n_calls}", "name": name, "arguments": args}])


def _assignments(chart, sampled=(SAMPLED_TYPE,)):
    """Every type judged, so the plan under test is the scripted one and not a default."""
    out = []
    for t in sorted(chart.doc_types):
        pol = ("read_all" if "Pathology" in t else "sample" if t in sampled else "search")
        out.append({"type": t, "policy": pol, "why": "scripted", "confidence": 0.9})
    return out


def _run(spec, chart, llm, tmp_path, run_id, *, max_steps=8, expansion_budget=None):
    """The live runtime, driven by the same scripts through the adapter.

    Returns (None, manifest, events). The first slot used to be the agent object these tests
    poked at; there is no such object now — the rules live in `AuditMiddleware` and the ledgers
    are reachable from the manifest, which is what a consumer of a finished run actually has.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hooks_harness import run_with_script

    from acr.corpus import Corpus
    corpus = Corpus(ROOT / "corpus" / "patients")
    ctx_out = []
    manifest, events = run_with_script(spec, corpus, chart.patient_id, tmp_path, llm,
                                       run_id=run_id, max_model_calls=max_steps,
                                       ctx_out=ctx_out, expansion_budget=expansion_budget)
    # The live context stands where the agent object used to: same `.plan`, `.threads`,
    # `.coverage`, and it is the one the run actually used.
    return (ctx_out[0] if ctx_out else None), manifest, events


def _triggers(events, kind=None):
    """Trigger events, read out of the trace the way a develop-plane consumer would.

    The trigger's own kind lives under `trigger`, because `kind` is the trace ENVELOPE's
    event type and one key cannot be both.
    """
    rows = [e for e in events if e.get("kind") == "trigger"]
    return [e for e in rows if e.get("trigger") == kind] if kind else rows


def _ledger(spec, chart):
    """A real stratified coverage ledger for this chart. The downgrade rule asks the gate what
    is still missing, so it needs the genuine ledger rather than a stub."""
    from acr.coverage import CoverageLedger, ForcedSampler, strata_from_spec
    docs, _ = chart.list_documents(limit=100_000)
    return CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(7))


def _first_of_type(chart, doc_type):
    docs, _ = chart.list_documents(doc_type_contains=doc_type, limit=5)
    assert docs, f"fixture assumption: SYN0001 has a {doc_type}"
    return docs[0].note_id


# ==========================================================================================
# 0. THE BUG ITSELF — one trigger, one live run, no exception
# ==========================================================================================
def test_a_payload_key_can_no_longer_kill_the_tracer(tmp_path):
    """Instrumentation may lose a field. It may not take the task down with it.

    `kind` is positional-only now, so the collision cannot raise; and a colliding key is
    re-keyed rather than allowed to overwrite the envelope, because a payload that silently
    replaced `kind` would re-file the event as a different type of event and every reader in
    this tree splits on `kind` first.
    """
    tr = Tracer.create(tmp_path, "collide")
    ev = tr.emit("trigger", kind="ZERO_HIT_SEARCH", seq="not-a-sequence", note_id="N1")

    assert ev["kind"] == "trigger" and isinstance(ev["seq"], int)
    assert ev["payload_kind"] == "ZERO_HIT_SEARCH" and ev["payload_seq"] == "not-a-sequence"
    assert ev["reserved_key_collisions"] == ["kind", "seq"] and ev["severity"] == "warning"
    assert ev["note_id"] == "N1", "an uncolliding key must be untouched"


def test_the_trigger_emitter_keeps_the_two_kinds_apart(tmp_path):
    tr = Tracer.create(tmp_path, "shape")
    ev = tr.trigger(**Trigger(kind=TRIGGER_ZERO_HIT_SEARCH, observation="nothing found",
                              step=3).to_dict())

    assert ev["kind"] == "trigger", "the envelope says what KIND of event this is"
    assert ev["trigger"] == TRIGGER_ZERO_HIT_SEARCH, "the payload says which trigger it was"
    assert "reserved_key_collisions" not in ev, (
        "the dedicated emitter must not go through the collision fallback")




def test_a_live_trigger_does_not_kill_the_run(spec, chart, tmp_path):
    """The regression. `emit("trigger", **t.to_dict())` passed `kind` twice and every run
    that detected anything died on the spot — with 34 green trigger tests."""
    llm = ScriptedLLM(acts=[("search_notes", {"query": "pseudomyxoma peritonei"})],
                      assignments=_assignments(chart))
    agent, result, events = _run(spec, chart, llm, tmp_path, "trigger-lives", max_steps=2)

    assert result["answer"], "the run died before it produced an answer"
    (t,) = _triggers(events, TRIGGER_ZERO_HIT_SEARCH)
    assert t["kind"] == "trigger", "the trace envelope must still say what KIND of event it is"
    assert t["trigger"] == TRIGGER_ZERO_HIT_SEARCH
    assert "pseudomyxoma peritonei" in t["observation"]


def test_a_trigger_records_the_step_it_fired_on(spec, chart, tmp_path):
    """`step` must say WHEN, and until 2026-07-28 it said 0 for every trigger ever emitted.

    `_detect` passed the literal `step=0` into `triggers_from_tool_result` and into
    `open_thread`, while every other emitter on the same path passed `ctx.n_model_calls`. On the
    293-document chart that produced six triggers, all six read `step: 0` and the
    `terms_added_at_runtime` entries for the same exchange read `step: 4` — so nothing in the
    manifest could be put in order against anything else, and the four zero-hit searches could
    not be told apart from four simultaneous ones.

    Two misses, so the assertion does not depend on a query that happens to hit: the steps must
    be non-zero AND distinct. A constant returns 0 and 0 and fails both halves.
    """
    llm = ScriptedLLM(acts=[("search_notes", {"query": "pseudomyxoma peritonei"}),
                            ("search_notes", {"query": "chordoma of the clivus"})],
                      assignments=_assignments(chart))
    _, _, events = _run(spec, chart, llm, tmp_path, "trigger-step", max_steps=3)

    steps = [t["step"] for t in _triggers(events, TRIGGER_ZERO_HIT_SEARCH)]
    assert len(steps) == 2, f"both misses should have fired a trigger, got {steps}"
    assert all(s > 0 for s in steps), f"step is the model-call index, never 0: {steps}"
    assert steps[0] != steps[1], (
        f"two triggers on two different model calls must carry different steps, got {steps} — "
        "a field that is always the same number looks like data and is a constant")


def test_every_trigger_field_survives_into_the_trace(spec, chart, tmp_path):
    """The payload is the audit. A trigger whose observation is dropped is a shrug."""
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    llm = ScriptedLLM(acts=[("read_document", {"note_id": nid, "limit": 40})],
                      assignments=_assignments(chart))
    _, _, events = _run(spec, chart, llm, tmp_path, "trigger-payload", max_steps=2)

    (t,) = _triggers(events, TRIGGER_UNSETTLED_THREAD)
    for key in ("observation", "note_id", "doc_type", "marker", "step",
                "terms_proposed", "types_proposed"):
        assert key in t, f"{key!r} was dropped on the way into the trace"
    assert t["note_id"] == nid and t["marker"] == "truncated"



def test_zero_hit_search_reaches_the_agent_and_the_new_term_becomes_an_obligation(spec, chart,
                                                                                 tmp_path):
    """A search that found nothing is an observation the agent must answer, not a shrug.

    It used to be rendered into the next reflect prompt. There is no reflect node; the block
    rides the system message, rebuilt each call and DRAINED so it is said once. What is asserted
    is unchanged: the observation reaches the party that can act on it.
    """
    llm = ScriptedLLM(acts=[("search_notes", {"query": "pseudomyxoma peritonei"}),
                            ("document_type_summary", {}),
                            ("document_type_summary", {})],
                      assignments=_assignments(chart))
    _run(spec, chart, llm, tmp_path, "zero-hit", max_steps=4)

    shown = [m for m in llm.seen_system if TRIGGER_ZERO_HIT_SEARCH in m]
    assert shown, "the zero-hit search never reached the agent"
    assert "OBSERVATIONS THAT REQUIRE A RESPONSE" in shown[0]
    assert "revise_plan" in shown[0], (
        "an observation with no named way to answer it is a shrug with a label")
    # DRAINED: said once, not re-announced on every later turn.
    assert len(shown) == 1, "the observation was repeated instead of drained"





def test_an_unsettled_thread_blocks_submission_and_a_resolution_unblocks_it(spec, chart,
                                                                           tmp_path):
    """The gate refuses while a thread is open, and stops refusing FOR THAT REASON once settled.

    Not "and then accepts": the gate has several obligations in a fixed order — evidence first,
    then the thread — so the script has to record evidence before the thread is what blocks it,
    and a later refusal for coverage is not this test's business. Asserting acceptance would make
    the test pass or fail on an unrelated obligation.

    The settlement arrives through `revise_plan`, which is also the only reason the refusal is
    answerable at all: the gate's `how_to_satisfy` names that call.
    """
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    good = {"status": "FOUND", "value": {"primary_site": "C341", "histology": "8140",
                                         "behavior": "3"},
            "reasoning": "the report is explicit"}

    def cite(prompt):
        hits = llm.results_for("search_notes")
        h = (hits[-1].get("hits") or [{}])[-1] if hits else {}
        return ("record_evidence", {"note_id": h.get("note_id", nid), "start": h.get("start", 0),
                                    "end": h.get("end", 40), "supports": "histology"})

    llm = ScriptedLLM(acts=[
        ("read_document", {"note_id": nid, "limit": 40}),      # opens a `truncated` thread
        ("search_notes", {"query": "carcinoma"}),
        cite,                                                  # so the thread is what blocks
        ("submit_answer", good),
        ("revise_plan", {"resolve_threads": [
            {"thread_id": f"{nid}#{TRUNCATED}", "where_settled": "paged to the end"}]}),
        ("submit_answer", good),
    ], assignments=_assignments(chart))
    ctx, _, events = _run(spec, chart, llm, tmp_path, "thread", max_steps=10)

    def names_the_thread(e):
        return any("unsettled thread" in str(m) for m in (e.get("missing") or []))

    rejects = [e for e in events if e.get("kind") == "answer_rejected"]
    assert rejects, "the submission was never refused"
    blocked = [i for i, e in enumerate(rejects) if names_the_thread(e)]
    assert blocked, f"the thread never blocked a submission; refusals were " \
                    f"{[e.get('why') for e in rejects]}"
    assert check_threads(ctx.threads) == [], "the resolution did not clear the obligation"
    assert not any(names_the_thread(e) for e in rejects[blocked[-1] + 1:]), (
        "the thread was settled and the gate still blocked on it")


# ==========================================================================================
# 3. UNLISTED_ANSWER_TERM — the signal the crash was killing
# ==========================================================================================
def _unlisted_span(chart, keywords, *, doc_type="Onc-Med-MD-OP-Progress-Note"):
    """A real span of a real document that no current search term would have found."""
    docs, _ = chart.list_documents(doc_type_contains=doc_type, limit=10)
    for d in docs:
        text = chart.read(d.note_id, 0, 4000)["text"]
        for start in range(0, max(1, len(text) - 120), 40):
            quote = text[start:start + 90]
            if len(quote.split()) >= 6 and not any(k in quote.lower() for k in keywords):
                return d.note_id, start, start + 90
    raise AssertionError("fixture assumption: SYN0001 has a span no declared term matches")



def test_a_cited_quote_no_term_would_have_found_reaches_the_agent_with_candidates(spec, chart,
                                                                                 tmp_path):
    """Evidence no declared term would have retrieved is evidence about the SPEC's term list.

    The trigger carries candidate terms, and the candidates are the artefact — a trigger with no
    suggestion is a shrug. They reach the agent in the observations block now.
    """
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    llm = ScriptedLLM(acts=[("read_document", {"note_id": nid}),
                            ("document_type_summary", {}),
                            ("document_type_summary", {})],
                      assignments=_assignments(chart))
    _, manifest, events = _run(spec, chart, llm, tmp_path, "unlisted", max_steps=4)

    trig = _triggers(events, TRIGGER_UNLISTED_ANSWER_TERM)
    if not trig:
        pytest.skip("this chart's cited quote is covered by a declared term")
    assert trig[0]["terms_proposed"], (
        "a trigger with no suggestion is a shrug; the candidate terms are the artefact")
    shown = [m for m in llm.seen_system if TRIGGER_UNLISTED_ANSWER_TERM in m]
    assert shown and "candidate terms" in shown[0]



def test_a_gate_obligation_the_plan_forbids_is_reported_as_a_deadlock(spec, chart, tmp_path):
    """The deadlock: the gate says "read these search hits", the plan says "you may not open that
    type", and the old loop spent the rest of its budget in the gap.

    The detector runs every turn now, not only after a refusal — a run whose reads are all
    OUT_OF_PLAN never reaches a submission, which is exactly the case it exists for. Asserted on
    the detector against a plan that genuinely forbids the type holding the hits, because
    manufacturing that state through a scripted run makes the test about the script.
    """
    from acr.run_triggers import detect_gate_obligations

    blocked_type = sorted(_plan_of(spec, chart).sample)[0]
    llm = ScriptedLLM(acts=[("search_notes", {"query": "right upper lobe"}),
                            ("document_type_summary", {}),
                            ("document_type_summary", {})],
                      assignments=_assignments(chart))
    ctx, _, events = _run(spec, chart, llm, tmp_path, "gate-deadlock", max_steps=6)

    assert not ctx.plan.may_open(blocked_type), (
        f"fixture assumption: {blocked_type!r} is sampled away and may not be opened")
    fired = detect_gate_obligations(spec=spec, coverage=ctx.coverage, chart=chart,
                                    plan=ctx.plan, step=1, tracer=ctx.tracer)
    if not fired:
        pytest.skip("this chart's search hits did not land in a forbidden type")
    t = fired[0]
    assert t.kind == TRIGGER_GATE_OBLIGATION_UNREACHABLE
    assert t.types_proposed, (
        "a deadlock report with no candidate type names no way out, and the way out is the "
        "whole reason to report it")


# ==========================================================================================
# 5. THE PLANNER MUST NOT EAT THE AGENT'S BUDGET
# ==========================================================================================
PLANNER_TERMS = ["adenocarcinoma", "lobe", "nodule", "mass", "malignant", "resection",
                 "cytology", "metastasis"]


def _remaining_terms(prompt: str) -> int:
    m = re.search(r"EXPANSION REMAINING: terms (-?\d+)", prompt)
    assert m, "the reflection prompt no longer states the remaining expansion"
    return int(m.group(1))



def test_the_up_front_planners_own_terms_are_not_charged_to_the_agent(spec, chart):
    """Price the cap against the same rows you count it against.

    This used to be asserted through an LLM planner whose keyword proposals became
    `term_provenance` rows: pricing the cap against SPEC-declared terms while counting it
    against EVERY row spent the agent's whole allowance before it reflected once —
    `EXPANSION REMAINING: terms -3`, first revision refused BUDGET_EXHAUSTED, run over at step 1
    of 12. There is no LLM planner now; the plan comes from the spec. The RULE survives and is
    asserted where it lives, in the pricing function's `planner_terms` argument.
    """
    from acr.coverage_planner import documents_by_type, plan_from_spec
    from acr.plan_expansion import price_expansion_budget

    plan = plan_from_spec(spec, chart)
    docs_by_type = documents_by_type(chart)
    priced = price_expansion_budget(plan, docs_by_type, max_revisions=6, supplied=None,
                                    planner_terms=len(plan.keywords))
    assert priced.max_terms_added >= len(plan.keywords), (
        "the cap is counted against EVERY term_provenance row, so a cap priced below the "
        "rows already present is negative allowance before the agent has asked for anything")
    from acr.plan_expansion import headroom
    assert headroom(plan, priced)["terms"] >= 0, "the agent must start with a non-negative allowance"


# ==========================================================================================
# 6. PARTIAL APPLICATION — "you may have 5 of the 6", and thread work is never collateral
# ==========================================================================================
def _revise_tool(spec, chart, *, expansion_budget=None, threads=None):
    """The declared `revise_plan` tool bound to real ledgers. See tests/hooks_harness.py."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hooks_harness import revise_plan_tool
    return revise_plan_tool(spec, chart, expansion_budget=expansion_budget, threads=threads)


def _revise(tool, **kwargs):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hooks_harness import revise
    return revise(tool, **kwargs)


def _plan_of(spec, chart):
    from acr.coverage_planner import plan_from_spec
    return plan_from_spec(spec, chart)


def _tight(**kw):
    base = dict(max_terms_added=2, max_type_promotions=1,
                max_documents_opened_by_promotion=500, max_revisions=6)
    base.update(kw)
    return ExpansionBudget(**base)



def test_one_term_too_many_applies_what_fits_and_reports_back_what_did_not(spec, chart):
    """Two of three land, and the third is NAMED.

    All-or-nothing is right for a monotonicity violation — applying the admissible half of a
    revision that also demoted a type hands back a plan the agent did not propose. A budget
    overrun is a different failure: nothing about the terms is inadmissible, there is simply not
    enough allowance. Asserted on the tool directly; the arithmetic is the subject, and a whole
    run can fail for a dozen unrelated reasons before reaching it.
    """
    tool, ctx = _revise_tool(spec, chart, expansion_budget=_tight())
    before = list(ctx.plan.keywords)
    r = _revise(tool, add_terms=["mucinous", "signet", "cribriform"])

    assert r["applied"] is True
    assert [k for k in ctx.plan.keywords if k not in before] == ["mucinous", "signet"], (
        "all-or-nothing on a one-term overrun: the agent is never told it may have 2 of 3")
    assert "cribriform" not in ctx.plan.keywords
    # NAMED, in the tool's own return. A refusal the model never sees is a refusal it repeats.
    assert r["terms_deferred_for_budget"] == ["cribriform"]
    # A term the run asked for and could not have is evidence about the SPEC's list. Partial
    # application is what keeps it out of `refused_revisions`, so it is harvested by name.
    assert ctx.terms_deferred == ["cribriform"]



def test_a_term_overrun_never_discards_the_thread_work(spec, chart):
    """The thread half of a revision is not collateral damage of a full term list."""
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    threads = OpenThreadLedger()
    threads.open_thread(**_open_kwargs(nid))
    tool, ctx = _revise_tool(spec, chart, expansion_budget=_tight(), threads=threads)

    r = _revise(tool, add_terms=["mucinous", "signet", "cribriform"],
                resolve_threads=[{"thread_id": f"{nid}#{TRUNCATED}",
                                  "where_settled": "paged to the end"}])
    assert r["terms_deferred_for_budget"] == ["cribriform"]
    assert ctx.threads.unresolved() == [], (
        "the settlement was thrown away because the term list overran; the two halves of a "
        "revision are not one transaction")



def test_thread_work_survives_a_retrieval_half_that_is_refused_outright(spec, chart):
    """A hallucinated type refuses the revision WHOLE — correctly, for the retrieval half.

    Thread bookkeeping is not the retrieval half: it cannot violate monotonicity and cannot
    widen what may be opened, so it does not belong to the refusal. A revision that both
    over-reached on types AND settled the thread blocking the answer used to end the run twice
    over — refused, and still thread-blocked, with the resolution nowhere.
    """
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    threads = OpenThreadLedger()
    threads.open_thread(**_open_kwargs(nid))
    tool, ctx = _revise_tool(spec, chart, threads=threads)

    r = _revise(tool, promote_types=[{"type": "Not-A-Real-Type", "to": "read_all"}],
                resolve_threads=[{"thread_id": f"{nid}#{TRUNCATED}",
                                  "where_settled": "read to the end"}])
    assert r["applied"] is False, (
        "the hallucinated type must still be refused; dropping it would leave the agent "
        "believing it had widened a scope it had not")
    assert r["thread_work_salvaged"] is True
    assert check_threads(ctx.threads) == [], "the settlement went down with the refusal"



def test_a_term_overrun_alone_does_not_end_the_run_while_promotions_remain(spec, chart):
    """Terms spent is not expansion spent. A type promotion is still affordable."""
    from acr.plan_expansion import expansion_is_spent

    budget = _tight()
    tool, ctx = _revise_tool(spec, chart, expansion_budget=budget)
    _revise(tool, add_terms=["mucinous", "signet", "cribriform"])
    assert expansion_is_spent(ctx.plan, budget, terms_deferred=list(ctx.terms_deferred)) is False, (
        "the term allowance is gone but the plan can still widen by promoting a type")



def test_the_run_still_ends_honestly_when_expansion_really_is_spent(spec, chart, tmp_path):
    """Partial application must not turn a genuine dead end into a run that quietly keeps going.

    Budget spent with obligations outstanding is EVIDENCE_INSUFFICIENT and it has to be SAID —
    the alternative is looping to the call limit and emitting whatever is in hand, which is a
    silent truncation wearing an answer's clothes.
    """
    llm = ScriptedLLM(acts=[
        ("search_notes", {"query": "pseudomyxoma peritonei"}),
        ("revise_plan", {"add_terms": ["mucinous", "signet", "cribriform"],
                         "promote_types": [{"type": SAMPLED_TYPE, "to": "search"}]}),
        ("document_type_summary", {}),
        ("document_type_summary", {}),
    ], assignments=_assignments(chart))
    ctx, manifest, events = _run(
        spec, chart, llm, tmp_path, "spent", max_steps=8,
        expansion_budget=_tight(max_terms_added=2, max_type_promotions=1))

    assert ctx.outstanding_obligations(), "fixture assumption: the gate is not met"
    assert [e for e in events if e.get("kind") == "expansion_budget_exhausted"], (
        "expansion is over and the obligation is not met; a run that keeps looping to the call "
        "limit and emits whatever is in hand is a silent truncation")
    assert manifest["expansion_stopped"], (
        "the manifest must carry the reason, not just the fact that the run ended")
    assert manifest["answer"]["status"] != "FOUND", (
        "a run that could not finish widening may not report a positive")


# ==========================================================================================
# 12. THE DEADLOCK — a thread that could be opened forever and settled by nothing
#
# WHAT HAPPENED, off the first true end-to-end run (SYN0001, spec
# STORE.400_522_523.site_histology_behavior, run `extract__20260727T200902Z__2d2f55b-dirty`).
# The agent read the pathology report, recorded evidence and submitted FOUND. The gate
# refused it: "a document in this chart deferred its own conclusion and the thread was never
# settled", thread `Surgical-Pathology-Report_2023-04-27#truncated`. It then reflected 18
# times, and from the seventh reflection on every one of them says the same true sentence —
# the report is truncated and its conclusion must be chased before coding.
#
# IT DID CHASE IT. `read_document` at offset 330, `read_document` at offset 480 (which
# returned `truncated: false`), `read_section("")` for the heading list, `read_section("FINAL
# DIAGNOSIS")`, and it recorded the final-diagnosis evidence. Then it spent 434,584 tokens and
# emitted an ungated positive with the thread still open.
#
# Three separate defects, each of which alone would have been survivable:
#
#   1. THE THREAD SHOULD NEVER HAVE OPENED. The whole 521-character report had already been
#      read at step 4 (`offset 0, limit 1200 -> 521 of 521, truncated false`). The thread was
#      opened at step 6 by a 180-character WINDOW read (`offset 330, limit 180`) whose
#      `truncated: true` says nothing about the document and everything about the window.
#   2. AN IDENTICAL THREAD COULD BE REQUESTED FOREVER. Across 14 revisions, 13 admitted, the
#      count requesting `resolve_threads` is ZERO and nine requested a re-open of the very
#      thread that was blocking them. The ledger deduped each one correctly and said nothing,
#      and the loop reported APPLIED nine times. A loop that reports success for doing
#      nothing is how a run spends 400k tokens standing still.
#   3. THERE WAS NO ROUTE FROM "I READ TO THE END OF IT" TO "THE THREAD IS SETTLED", and the
#      route that did exist was invisible where it was needed. `gate_answer` writes a
#      `how_to_satisfy` naming `resolve_threads` — and `graph._n_act` dropped that key on the
#      floor, handing the model `why` and `you_must_still` only. The affordance was in the
#      schema for the whole run and was never reached.
#
# WHAT THESE TESTS HOLD
#   * a document read to its end settles its own `truncated` thread, mechanically, and a
#     window read of an already-complete document opens nothing at all;
#   * `truncated` is the ONLY marker kind that may be discharged that way, and the reason is
#     structural rather than a shortlist somebody maintains;
#   * re-opening an open thread is refused as the no-op it is, in words naming the call that
#     would have helped;
#   * the two settlement routes are visible at the two places the run is blocked — the gate's
#     rejection and the reflection prompt;
#   * and a run that stops with an obligation outstanding abstains instead of shipping a
#     positive with a warning attached.
#
# No provider is called and no chart text is written here.
# ==========================================================================================
TRUNCATED = "truncated"


def _length_of(chart, note_id: str) -> int:
    return chart.read(note_id, 0, 1)["total_chars"]


def _apply(plan, rev, *, threads, step=1):
    return plan.apply_revision(rev, step=step, trigger="test", observation="a test observation",
                               budget=ExpansionBudget(max_terms_added=5, max_type_promotions=5,
                                                      max_documents_opened_by_promotion=5000,
                                                      max_revisions=6),
                               threads=threads, n_docs_by_type={})


def _nothing_outstanding() -> list[str]:
    return []


# ------------------------------------------------------------------ the deterministic route
def test_reading_a_document_to_its_end_settles_its_own_truncated_thread(spec, chart, tmp_path):
    """The route that did not exist. The agent pages to the end and the LEDGER learns it.

    Nothing is asked of any model here: `corpus.PatientChart.read` computes `truncated` from
    `offset + len(chunk) < len(text)`, so the runtime owns both sides of the predicate and can
    compute its complement just as well.
    """
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    n = _length_of(chart, nid)
    llm = ScriptedLLM(
        acts=[("read_document", {"note_id": nid, "offset": 0, "limit": 40}),
              ("read_document", {"note_id": nid, "offset": 40, "limit": n})],
        assignments=_assignments(chart))
    agent, _, events = _run(spec, chart, llm, tmp_path, "settled-by-read", max_steps=3)

    (t,) = _triggers(events, TRIGGER_UNSETTLED_THREAD)
    assert t["marker"] == TRUNCATED, "fixture assumption: the short read opened the thread"
    assert check_threads(agent.threads) == [], (
        "the agent read the document to its end and the ledger never learned — that is the "
        "deadlock: correct work that no mechanism could turn into a settlement")
    (thread,) = agent.threads.threads
    assert thread.state == "resolved"
    assert thread.resolution.startswith(SETTLED_BY_READING_TO_THE_END)
    (settled,) = [e for e in events if e.get("kind") == "threads_settled_by_read"]
    assert settled["thread_ids"] == [thread.thread_id] and settled["note_id"] == nid
    assert agent.threads.to_dict()["n_resolved_mechanically"] == 1, (
        "the manifest must keep 'the runtime saw the end of it' apart from 'the agent said "
        "it chased it'; merged, nobody can tell whether the route is ever reached")


def test_a_window_read_of_an_already_complete_document_opens_nothing(spec, chart, tmp_path):
    """SYN0001's actual sequence, in the order it actually happened.

    Read the whole document, then re-read a 180-character window out of the middle of it. The
    window's `truncated: true` is a fact about the window. Opening a thread on it demands the
    agent chase a conclusion it has already read.
    """
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    n = _length_of(chart, nid)
    assert n > 200, "fixture assumption: the pathology note is long enough to window into"
    llm = ScriptedLLM(
        acts=[("read_document", {"note_id": nid, "offset": 0, "limit": n + 500}),
              ("read_document", {"note_id": nid, "offset": 100, "limit": 60})],
        assignments=_assignments(chart))
    agent, _, events = _run(spec, chart, llm, tmp_path, "window-of-complete", max_steps=3)

    assert not [t for t in _triggers(events, TRIGGER_UNSETTLED_THREAD)
                if t["marker"] == TRUNCATED], (
        "a window read of a document already read in full opened a `truncated` thread — the "
        "obligation is to page to the end, and the end has already been returned")
    assert check_threads(agent.threads) == []
    assert agent.threads.fully_read(nid) is True


def test_a_hole_in_the_middle_is_not_a_read_to_the_end():
    """Head plus tail is not the document, and the deferred conclusion could be in the hole.

    The ledger merges spans and requires contiguity from zero, so this is the one-span test
    and not a `max(end) >= total` test — which head-then-tail would pass.
    """
    L = OpenThreadLedger()
    L.open_thread(note_id="N1", doc_type="Surgical-Pathology-Report", marker=TRUNCATED,
                  obligation="page to the end", excerpt="", step=1)
    assert L.note_read("N1", offset=0, returned_chars=100, total_chars=521, step=2) == []
    assert L.note_read("N1", offset=200, returned_chars=321, total_chars=521, step=3) == []
    assert L.fully_read("N1") is False
    assert [t.thread_id for t in L.unresolved()] == ["N1#truncated"]

    # The hole closed. Now, and only now, the obligation is discharged.
    assert L.note_read("N1", offset=100, returned_chars=100, total_chars=521,
                       step=4) == ["N1#truncated"]
    assert L.fully_read("N1") is True and L.unresolved() == []


def test_truncated_is_the_only_marker_a_machine_may_discharge():
    """The asymmetry is the design and it is checked against the skill's own obligation table.

    `truncated` is a claim about a READ and the runtime holds both sides of it. Every other
    marker names a DOCUMENT — a later report, an addendum, a different class of note, or one
    that is not in this chart at all — and whether the named document was found is a
    judgement no character count settles. So the set may never grow silently.
    """
    assert MECHANICALLY_DISCHARGEABLE_MARKERS == {TRUNCATED}
    catalogue = load_marker_catalogue()
    others = {m.text for m in catalogue.markers} - {TRUNCATED}
    assert others, "fixture assumption: the skill's obligation table parsed"
    assert not (others & MECHANICALLY_DISCHARGEABLE_MARKERS), (
        "a marker naming a document a machine cannot go and find became mechanically "
        "dischargeable; `outside facility` alone would make abstention unreachable")

    # And it holds at the ledger, not merely in the constant.
    L = OpenThreadLedger()
    for marker in ("stains pending", "addendum", "outside facility"):
        L.open_thread(note_id="N1", doc_type="Surgical-Pathology-Report", marker=marker,
                      obligation="chase it", excerpt="", step=1)
    assert L.note_read("N1", offset=0, returned_chars=521, total_chars=521, step=2) == []
    assert len(L.unresolved()) == 3, (
        "reaching the end of the deferring document settled a thread that points at a "
        "different document — the addendum, the later report, the outside institution")


# --------------------------------------------------------------- the no-op, reported as one
def test_re_opening_an_open_thread_is_refused_as_the_no_op_it_is(spec, chart):
    """Identity dedupe was already right. Reporting APPLIED over it was the defect."""
    plan, threads = plan_from_spec(spec, chart), OpenThreadLedger()
    first = _apply(plan, PlanRevision(open_threads=(("N1", "stains pending", "it defers"),)),
                   threads=threads)
    assert first.applied and first.threads_opened == ["N1#stains pending"]

    again = _apply(plan, PlanRevision(open_threads=(("N1", "stains pending", "it defers"),)),
                   threads=threads, step=2)
    assert len(threads.threads) == 1, "the debt was multiplied"
    assert again.threads_opened == []
    assert again.applied is False and again.refusal_class == REFUSED_THREAD_NOOP, (
        "admitted, affordable, monotone — and it moved nothing. Reporting that as APPLIED is "
        "how the same request gets sent nine times")
    assert again.thread_noops == [{"thread_id": "N1#stains pending", "status": "already_open"}]
    (why,) = again.refused
    assert "resolve_threads" in why and '"thread_id": "N1#stains pending"' in why, (
        "the refusal must carry the call that WOULD have helped, with the id filled in; a "
        "refusal an agent cannot act on is one it repeats")
    assert plan.refused_revisions[-1]["refusal_class"] == REFUSED_THREAD_NOOP, (
        "a no-op that leaves no record is one nobody can count across runs")


def test_a_no_op_open_alongside_real_work_does_not_refuse_the_real_work(spec, chart):
    """The refusal is for a revision that was ENTIRELY a no-op. A term that lands still lands,
    and the part that changed nothing is still named."""
    plan, threads = plan_from_spec(spec, chart), OpenThreadLedger()
    _apply(plan, PlanRevision(open_threads=(("N1", "pending", "x"),)), threads=threads)
    out = _apply(plan, PlanRevision(add_terms=("psammoma",),
                                    open_threads=(("N1", "pending", "x"),)),
                 threads=threads, step=3)
    assert out.applied is True and out.terms_added == ["psammoma"]
    assert out.thread_noops and any("resolve_threads" in r for r in out.refused)



def test_the_no_op_refusal_reaches_the_model_in_the_loop_it_understands(spec, chart):
    """The nine identical opens of SYN0001, cut to two, with the answer to them.

    What the model is handed back is the whole point: on the real run it was handed "Your
    revision was APPLIED" and the re-rendered plan, which is indistinguishable from progress.
    The tool's return value is that channel now, so the refusal is in the model's hands by
    construction rather than by a prompt someone remembered to write.
    """
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    tool, ctx = _revise_tool(spec, chart)
    ask = dict(open_threads=[{"note_id": nid, "marker": TRUNCATED,
                              "why": "the read stopped short"}])

    first = _revise(tool, **ask)
    second = _revise(tool, **ask)

    assert len(ctx.threads.threads) == 1, "the second open created a second thread"
    assert second["applied"] is False, "a revision that moved nothing was recorded as one that did"
    told = json.dumps(second)
    assert "REFUSED" in told.upper() and nid in told, (
        "the model was never told its request changed nothing, so it repeats it")


# ------------------------------------------------------ the affordance, where the block is
def test_the_gate_rejection_hands_back_the_way_to_settle_the_thread(spec, chart, tmp_path):
    """`gate_answer` writes `how_to_satisfy`; `_n_act` used to drop it before the model saw it.

    That key is the only place a rejection says the word `resolve_threads`. Telling a run what
    is wrong and not what to do about it is how a loop becomes a deadlock.
    """
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    llm = ScriptedLLM(
        acts=[("read_document", {"note_id": nid, "limit": 40}),
              ("submit_answer", {"status": "EVIDENCE_INSUFFICIENT", "reasoning": "scripted",
                                 "value": {}})],
        assignments=_assignments(chart))
    _run(spec, chart, llm, tmp_path, "gate-says-how", max_steps=4)

    (rejection,) = [r for _, r in llm.tool_results if r.get("accepted") is False]
    assert any("unsettled thread" in m for m in rejection["you_must_still"]), (
        "fixture assumption: the submission was refused for the thread")
    assert "resolve_threads" in rejection.get("how_to_satisfy", ""), (
        "the gate wrote the way out and the boundary threw it away; SYN0001 answered this "
        "rejection with nine more requests to OPEN the thread and none to resolve it")



def test_the_prompt_names_the_settling_call_beside_the_thread_that_blocks(spec, chart, tmp_path):
    """An affordance named a long way from the obstacle it clears does not exist in practice.

    `resolve_threads` sat in a schema at the bottom of the reflect prompt for all eighteen
    reflections of the real run and was never used once; the agent re-opened the same thread
    instead. There is no reflect prompt now — the block rides the system message, rebuilt each
    call — but the rule is the same and so is the test: the way out is printed next to the thing
    it unblocks.
    """
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    llm = ScriptedLLM(acts=[("read_document", {"note_id": nid, "limit": 40}),
                            ("document_type_summary", {}),
                            ("document_type_summary", {})],
                      assignments=_assignments(chart))
    _run(spec, chart, llm, tmp_path, "prompt-names-resolve", max_steps=4)

    shown = [m for call in llm.seen_system for m in [call] if f"{nid}#{TRUNCATED}" in m]
    assert shown, "the agent was never shown the open thread"
    for block in shown:
        threads_block = block.split("UNSETTLED THREADS")[1].split("OBSERVATIONS THAT REQUIRE")[0] \
            if "OBSERVATIONS THAT REQUIRE" in block else block.split("UNSETTLED THREADS")[1]
        assert f'"thread_id": "{nid}#{TRUNCATED}"' in threads_block, (
            "the agent is told the thread blocks it without being told, in the same breath, "
            "the call that settles it")
        assert "dismiss_threads" in threads_block


# ------------------------------------------------------------------------- the termination
def test_a_run_that_stops_owing_an_obligation_cannot_ship_a_value(spec, chart, tmp_path):
    """The end of the SYN0001 run, and the thing it was supposed to be.

    It ended `max_tokens (400000) reached`, went to finalize with the `truncated` thread open,
    and emitted `status FOUND, proof_basis UNGATED, route_to_human true` — a FOUND with a
    warning stapled to it. `concordance.variables_from_answer` promotes each populated field to
    FOUND regardless of the answer's status, so nothing downstream carried the warning and
    C341/8140/3 shipped as established.

    THIS RUNTIME IS IMMUNE BY CONSTRUCTION, which is a stronger claim than the downgrade rule
    and is asserted first: `ctx.answer` is assigned in exactly one place, inside the branch
    where the gate accepted. There is no finalize that can author an answer, so an ungated
    positive cannot come into existence. The downgrade below is the belt to that braces — it
    exists so that a future finalize path cannot reintroduce the defect silently — and it is
    exercised directly rather than through a run that cannot produce its precondition.
    """
    import inspect

    import acr.agent as A

    assigns = re.findall(r"ctx\.answer\s*=\s*(\S+)", inspect.getsource(A))
    assert assigns == ["submitted"], (
        f"expected one assignment, inside the accepted branch; found {assigns}")
    gate = inspect.getsource(A.AuditMiddleware._gate_answer)
    i, j = gate.index('verdict.get("accepted")'), gate.index("ctx.answer = submitted")
    assert i < j, "the answer may only be set after the gate has accepted it"

    # And the guard itself, on the shape the old runtime actually emitted.
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    threads = OpenThreadLedger()
    threads.open_thread(note_id=nid, doc_type=PATHOLOGY_TYPE, marker=TRUNCATED,
                        obligation="page to the end", excerpt="...", step=0)
    ans = {"status": "FOUND",
           "value": {"primary_site": "C341", "histology": "8140", "behavior": "3"}}
    A.downgrade_a_positive_that_owes_something(
        ans, spec=spec, coverage=_ledger(spec, chart), plan=plan_from_spec(spec, chart),
        threads=threads, gate_validated=False, termination="BUDGET_EXHAUSTED")
    assert ans["status"] == "EVIDENCE_INSUFFICIENT"
    assert ans["downgraded_from"] == "FOUND" and ans["outstanding_at_termination"]
    assert ans["withheld_value"] == {"primary_site": "C341", "histology": "8140",
                                     "behavior": "3"}, "nothing is destroyed"
    assert set(ans["value"].values()) == {None}, (
        "the value must not go out intact, or every field is re-promoted to FOUND downstream "
        "and the downgrade is cosmetic exactly where it matters")


def test_a_positive_that_owes_nothing_keeps_its_ungated_label(spec, chart, monkeypatch):
    """The other half, so the downgrade cannot quietly become "no positive ever ships".

    Owing nothing is a run that finished. And a gate-validated FOUND cleared the thread check
    and the decision rules — nothing here has standing to overrule it.
    """
    import acr.agent as A

    # "Owing nothing" is BOTH ledgers clear — the gate's misses and the threads. An empty
    # thread ledger is not enough, because a fresh coverage ledger still owes the gate, and
    # conflating the two is what would make this half of the rule untestable.
    monkeypatch.setattr(A, "outstanding_obligations", lambda *a, **k: [])
    owes_nothing = {"status": "FOUND", "value": {"histology": "8140"}}
    A.downgrade_a_positive_that_owes_something(
        owes_nothing, spec=spec, coverage=_ledger(spec, chart),
        plan=plan_from_spec(spec, chart), threads=OpenThreadLedger(), gate_validated=False,
        termination="BUDGET_EXHAUSTED")
    assert owes_nothing["status"] == "FOUND" and "downgraded_from" not in owes_nothing, (
        "a run that finished keeps its positive; the downgrade must not become "
        "'no positive ever ships'")

    monkeypatch.undo()
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    blocked = OpenThreadLedger()
    blocked.open_thread(note_id=nid, doc_type=PATHOLOGY_TYPE, marker=TRUNCATED,
                        obligation="page to the end", excerpt="...", step=0)
    gated = {"status": "FOUND", "value": {"histology": "8140"}}
    A.downgrade_a_positive_that_owes_something(
        gated, spec=spec, coverage=_ledger(spec, chart), plan=plan_from_spec(spec, chart),
        threads=blocked, gate_validated=True, termination="BUDGET_EXHAUSTED")
    assert gated["status"] == "FOUND" and "downgraded_from" not in gated


SRC = ROOT / "src" / "acr"

#: Every runtime name that hands back an `OpenRequest`. The scan below guards all of them,
#: so a second convenience wrapper cannot reintroduce the sentinel under a new name.
OPEN_REQUEST_RETURNING = ("request_open", "open_thread")


def _open_kwargs(nid="N1", marker=TRUNCATED):
    return dict(note_id=nid, doc_type=PATHOLOGY_TYPE, marker=marker,
                obligation="page to the end", excerpt="the read stopped short", step=1)


def test_three_partial_reads_of_one_document_fire_one_unsettled_thread_trigger(spec, chart,
                                                                                 tmp_path):
    """One document, one obligation. Three short reads are one debt, not three.

    `open_thread` is idempotent per (note, marker) and returns `already_open` the second time;
    the detector branches on that status, so repeated short reads of the same note cannot inflate
    the trigger count. That guard broke once by being tested with `is None`.
    """
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    llm = ScriptedLLM(acts=[
        ("read_document", {"note_id": nid, "offset": 0, "limit": 30}),
        ("read_document", {"note_id": nid, "offset": 0, "limit": 40}),
        ("read_document", {"note_id": nid, "offset": 0, "limit": 50}),
        ("document_type_summary", {}),
    ], assignments=_assignments(chart))
    ctx, _, events = _run(spec, chart, llm, tmp_path, "three-partial-reads", max_steps=6)

    assert len(_triggers(events, TRIGGER_UNSETTLED_THREAD)) == 1, (
        "three short reads of one document are one obligation, not three")
    assert len(ctx.threads.threads) == 1


def test_a_resolved_thread_is_not_announced_again_by_a_later_partial_read(spec, chart, tmp_path):
    """Once settled, a thread must not come back as new debt on the next short read.

    The revision used to arrive through the reflect node's JSON; it arrives through the declared
    `revise_plan` tool now, which is the same operation with an audit trail. What is asserted is
    unchanged: the ledger settles once and stays settled.
    """
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    llm = ScriptedLLM(acts=[
        ("read_document", {"note_id": nid, "offset": 0, "limit": 40}),
        ("revise_plan", {"resolve_threads": [
            {"thread_id": f"{nid}#{TRUNCATED}",
             "where_settled": "paged to the end and read FINAL DIAGNOSIS"}]}),
        ("read_document", {"note_id": nid, "offset": 0, "limit": 60}),
        ("document_type_summary", {}),
    ], assignments=_assignments(chart))
    ctx, manifest, events = _run(spec, chart, llm, tmp_path, "resolved-not-reannounced",
                                 max_steps=6)

    (thread,) = ctx.threads.threads
    assert thread.state == "resolved", "the revision must have settled it"
    assert check_threads(ctx.threads) == [], "a settled thread may not block submission"
    # One trigger for the first short read, and NONE for the one after the resolution.
    assert len(_triggers(events, TRIGGER_UNSETTLED_THREAD)) == 1, (
        "the second short read re-announced a thread that was already settled")


def test_request_open_names_all_four_outcomes_and_open_thread_no_longer_returns_a_sentinel():
    """Four different things can happen; exactly one of them is new debt.

    `already_open`, `already_settled` and `discharged_on_read` are all "nothing was opened",
    and they are three different facts about the run. Collapsing them into a truthy/falsey
    return is what made the difference invisible at the call site.
    """
    L = OpenThreadLedger()
    opened = L.request_open(**_open_kwargs())
    assert opened.status == OPEN_REQUEST_OPENED and opened.opened is True
    assert opened.thread is not None and opened.thread_id == f"N1#{TRUNCATED}"

    again = L.request_open(**_open_kwargs())
    assert again.status == OPEN_REQUEST_ALREADY_OPEN and again.opened is False
    assert again.thread is opened.thread, (
        "the existing thread still travels back — that is what the deadlock fix needs, and "
        "it is exactly why the caller may not read the return as a truth value")

    L.resolve(f"N1#{TRUNCATED}", "read to the end", step=2)
    settled = L.request_open(**_open_kwargs())
    assert settled.status == OPEN_REQUEST_ALREADY_SETTLED and settled.opened is False

    D = OpenThreadLedger()
    D.note_read("N2", offset=0, returned_chars=300, total_chars=300, step=1)
    discharged = D.request_open(**_open_kwargs(nid="N2"))
    assert discharged.status == OPEN_REQUEST_DISCHARGED_ON_READ
    assert discharged.opened is False and discharged.thread is None

    # And the sentinel is gone from the detector's entry point as well: `open_thread` is
    # `request_open` under its old name, not a lossy projection of it.
    assert L.open_thread(**_open_kwargs()).status == OPEN_REQUEST_ALREADY_SETTLED
    assert D.open_thread(**_open_kwargs(nid="N2")).status == OPEN_REQUEST_DISCHARGED_ON_READ
    assert D.open_thread(**_open_kwargs(nid="N3")).status == OPEN_REQUEST_OPENED
    for r in (opened, again, settled, discharged):
        assert r is not None, "`is None` can never again mean 'nothing happened'"


def test_an_open_request_may_not_be_read_as_a_bare_truth_value():
    """The general lesson, enforced by the type rather than recorded in a comment.

    A caller that writes `if threads.open_thread(...)` — or `if r:`, which no static scan can
    see once the value has been bound to a name — gets a TypeError naming the property it
    should have asked for, at the first execution, instead of a silently doubled count.
    """
    L = OpenThreadLedger()
    L.note_read("N2", offset=0, returned_chars=300, total_chars=300, step=1)
    L.request_open(**_open_kwargs())
    every_status = [L.request_open(**_open_kwargs(nid="N9")),        # opened
                    L.request_open(**_open_kwargs()),                # already_open
                    L.request_open(**_open_kwargs(nid="N2"))]        # discharged_on_read
    L.resolve(f"N9#{TRUNCATED}", "settled", step=2)
    every_status.append(L.request_open(**_open_kwargs(nid="N9")))    # already_settled
    assert {r.status for r in every_status} == set(OPEN_REQUEST_STATUSES)

    for r in every_status:
        with pytest.raises(TypeError, match="status"):
            bool(r)
        with pytest.raises(TypeError, match="status"):
            if r:                                    # pragma: no branch - the raise is the test
                pass
        with pytest.raises(TypeError, match="status"):
            _ = not r


def test_every_open_request_status_has_a_declared_meaning_and_only_one_is_new_debt():
    """A status a caller cannot look up is a sentinel with a longer name.

    So the table is the contract: a fifth outcome cannot be returned without being written
    down here, and `opened` is the only one that means an obligation was added.
    """
    assert set(OPEN_REQUEST_STATUSES) == {OPEN_REQUEST_OPENED, OPEN_REQUEST_ALREADY_OPEN,
                                          OPEN_REQUEST_ALREADY_SETTLED,
                                          OPEN_REQUEST_DISCHARGED_ON_READ}
    assert all(len(v) > 20 for v in OPEN_REQUEST_STATUSES.values()), (
        "each status has to say what it MEANS, not restate its own name")
    doc = OpenThreadLedger.request_open.__doc__ or ""
    for s in OPEN_REQUEST_STATUSES:
        assert s in doc, f"{s!r} is returnable and undocumented where it is returned"
        assert OpenRequest(status=s, thread=None, thread_id="t", why="").opened is (
            s == OPEN_REQUEST_OPENED), (
            "a trigger is owed for `opened` and for nothing else")


def test_no_runtime_call_site_reads_an_open_request_as_a_bare_truth_value():
    """The regression, inverted into a scan — this is how the guard died the first time.

    `__bool__` raising catches `if r:` when the line runs; this catches the two forms that
    can sit unexecuted in a rarely-taken branch for a week: the call used directly as a
    condition, and the `is None` test that the old sentinel invited and that now silently
    never fires.
    """
    offenders, seen = [], {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr in OPEN_REQUEST_RETURNING):
                continue
            seen.setdefault(path.name, []).append(node.lineno)
            why = _bare_truth_context(node, parents)
            if why:
                offenders.append(f"{path.name}:{node.lineno}: the result is {why}")

    assert not offenders, (
        "a call site is treating an OpenRequest as a bare truth value:\n  "
        + "\n  ".join(offenders)
        + "\nBranch on `.status` (see OPEN_REQUEST_STATUSES); only `opened` is new debt.")
    assert set(seen) >= {"run_triggers.py", "coverage_planner.py", "agent.py"}, (
        f"fixture assumption: the scan found the runtime's call sites, got {sorted(seen)}")


def _bare_truth_context(node, parents):
    """Name the way this call's result is being used, if that way is a bare truth value."""
    cur, p = node, parents.get(node)
    while isinstance(p, ast.Compare):
        operands = [p.left, *p.comparators]
        if any(isinstance(o, ast.Constant) and o.value is None for o in operands):
            return "compared with None — a sentinel this contract does not have"
        cur, p = p, parents.get(p)
    if isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.Not):
        return "negated with `not`"
    if isinstance(p, ast.BoolOp):
        return "an operand of `and`/`or`"
    if isinstance(p, (ast.If, ast.While, ast.IfExp, ast.Assert)) and p.test is cur:
        return "used directly as a condition"
    if isinstance(p, ast.comprehension) and cur in p.ifs:
        return "used directly as a comprehension filter"
    return ""


def test_the_detector_owes_a_trigger_for_opened_and_for_nothing_else(spec, chart):
    """The guard itself, at the unit the end-to-end tests exercise from above.

    Four calls, four statuses, one trigger. Driven through `detect_from_tool_result` because
    the bug was never in the ledger — the ledger deduped correctly throughout — it was in
    what the caller concluded from the answer.
    """
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    plan, threads = plan_from_spec(spec, chart), OpenThreadLedger()
    markers = load_marker_catalogue()

    def short_read(step, returned=40):
        out = {"note_id": nid, "doc_type": PATHOLOGY_TYPE, "text": "", "truncated": True,
               "offset": 0, "returned_chars": returned, "total_chars": 419}
        return run_triggers.detect_from_tool_result(
            "read_document", {"note_id": nid}, out, step=step, plan=plan, markers=markers,
            threads=threads)

    assert len(short_read(1)) == 1, "the first short read is the one that owes a trigger"
    assert short_read(2) == [], "already_open is a no-op, not new debt"
    threads.resolve(f"{nid}#{TRUNCATED}", "read to the end", step=3)
    assert short_read(4) == [], "already_settled is a paid debt, not new debt"

    fresh = OpenThreadLedger()
    fresh.note_read(nid, offset=0, returned_chars=419, total_chars=419, step=1)
    threads = fresh
    assert short_read(5) == [], "discharged_on_read opened nothing, so nothing is owed"
    assert fresh.threads == [], "fixture assumption: nothing was opened against a whole read"


# ------------------------------------------------- the same shape, hunted down where I own it
def test_a_read_the_ledger_refused_to_record_is_not_reported_as_a_read_that_settled_nothing():
    """`note_read` returns [] for three different things, and one of them is a lost write.

    Same shape as the sentinel: an empty container standing for "nothing to settle" AND for
    "I did not record your read at all". The caller sees the same `[]` either way, so a
    document whose spans were never recorded looks exactly like one that is merely unfinished
    — and `fully_read` would then answer False forever, which is a thread that can never be
    discharged mechanically. The return stays `[]`; what changes is that the refusal is
    written down where the manifest can be read for it.
    """
    L = OpenThreadLedger()
    assert L.note_read("", offset=0, returned_chars=10, total_chars=10, step=1) == []
    assert L.note_read("N1", offset=0, returned_chars=-1, total_chars=10, step=2) == []
    assert L.read_spans == {}, "fixture assumption: neither read was recorded"
    assert [r["step"] for r in L.ignored_reads] == [1, 2]
    assert all(r["why"] for r in L.ignored_reads), "a refusal that says nothing is silence"
    assert L.to_dict()["ignored_reads"] == L.ignored_reads, (
        "a read the runtime dropped is a hole in the coverage the manifest asserts")

    # ...and a read that WAS recorded and settled nothing is not one of them.
    assert L.note_read("N1", offset=0, returned_chars=10, total_chars=419, step=3) == []
    assert len(L.ignored_reads) == 2 and L.read_spans == {"N1": [[0, 10]]}


def test_fully_read_false_does_not_conflate_a_hole_with_a_length_nobody_reported():
    """The other empty-container-as-meaning in the ledger, given names.

    `read_section` contributes true offsets and no document length, so a run that only ever
    read sections knows nothing about whether the document is finished. `fully_read` answers
    False for that, for a hole in the middle and for a document never touched — three states
    that a mechanical discharge decision has to be able to tell apart.
    """
    L = OpenThreadLedger()
    assert L.read_state("N1") == READ_STATE_UNREAD and L.fully_read("N1") is False

    L.note_read("N1", offset=0, returned_chars=100, total_chars=None, step=1)
    assert L.read_state("N1") == READ_STATE_LENGTH_UNKNOWN and L.fully_read("N1") is False

    L.note_read("N1", offset=200, returned_chars=100, total_chars=300, step=2)
    assert L.read_state("N1") == READ_STATE_INCOMPLETE, "characters 100-200 are still a hole"
    assert L.fully_read("N1") is False

    L.note_read("N1", offset=100, returned_chars=100, total_chars=300, step=3)
    assert L.read_state("N1") == READ_STATE_COMPLETE and L.fully_read("N1") is True


def test_a_gate_obligation_detector_that_crashed_is_not_an_empty_list(spec, chart, tmp_path,
                                                                      monkeypatch):
    """Third instance of the shape, in the other function this module owns.

    `detect_gate_obligations` swallows every exception and returns `[]`, which is the same
    value it returns when the plan has no structural deadlock. A trigger family that has
    silently stopped firing and one that has nothing to report are indistinguishable in the
    trace — and the fourth trigger is the one that breaks deadlocks, so the failure mode is a
    run that burns its budget with nothing recorded about why nobody stopped it.
    """
    llm = ScriptedLLM(acts=[("document_type_summary", {})], assignments=_assignments(chart))
    agent, _, _ = _run(spec, chart, llm, tmp_path, "detector-probe", max_steps=2)

    assert "tracer" in inspect.signature(run_triggers.detect_gate_obligations).parameters, (
        "the detector has nowhere to report its own failure to")
    assert (inspect.signature(run_triggers.detect_gate_obligations)
            .parameters["tracer"].default is inspect.Parameter.empty), (
        "an optional reporting channel is one a new call site forgets, which puts the "
        "silence straight back")

    def boom(*a, **k):
        raise RuntimeError("the gate blew up")

    monkeypatch.setattr(run_triggers, "check_gate", boom)
    tr = Tracer.create(tmp_path, "detector-blew-up")
    out = run_triggers.detect_gate_obligations(spec=spec, coverage=agent.coverage, chart=chart,
                                               plan=agent.plan, step=3, tracer=tr)

    assert out == [], "trigger detection may never take a run down with it"
    (ev,) = [e for e in tr.events if e["kind"] == "gate_obligation_detection_failed"]
    assert ev["severity"] == "error" and ev["step"] == 3
    assert "RuntimeError" in ev["error"] and "the gate blew up" in ev["error"]


