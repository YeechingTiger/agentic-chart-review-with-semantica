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

import inspect
import json
import re
from pathlib import Path

import pytest

from acr import deep_runner, graph
from acr.corpus import Corpus
from acr.coverage_planner import (REFUSED_BUDGET, TRIGGER_GATE_OBLIGATION_UNREACHABLE,
                                  TRIGGER_UNLISTED_ANSWER_TERM, TRIGGER_UNSETTLED_THREAD,
                                  TRIGGER_ZERO_HIT_SEARCH, ExpansionBudget, Trigger,
                                  spec_declared_keywords)
from acr.graph import ChartReviewAgent, check_threads
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

    def __init__(self, *, acts=(), reflections=(), assignments=(), planner_keywords=()):
        super().__init__(LLMConfig(model="scripted/none", api_key="none"))
        self.acts = list(acts)
        self.reflections = list(reflections)
        self.assignments = list(assignments)
        self.planner_keywords = list(planner_keywords)
        self.reflect_prompts: list[str] = []
        self.act_prompts: list[str] = []
        self.tool_results: list[tuple[str, dict]] = []
        self.finalized = False
        self._n_tool_seen = 0
        self._n_msgs_seen = 0
        self._n_calls = 0

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
            return self._reply({"status": "EVIDENCE_INSUFFICIENT", "value": {},
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
    agent = ChartReviewAgent(spec, llm, budget=Budget(max_steps=max_steps), out_dir=tmp_path,
                             sample_seed=7, expansion_budget=expansion_budget)
    result = agent.run(chart, run_id=run_id)
    events = [json.loads(l) for l in (tmp_path / f"{run_id}.jsonl").read_text(
        encoding="utf-8").splitlines() if l.strip()]
    return agent, result, events


def _triggers(events, kind=None):
    """Trigger events, read out of the trace the way a develop-plane consumer would.

    The trigger's own kind lives under `trigger`, because `kind` is the trace ENVELOPE's
    event type and one key cannot be both.
    """
    rows = [e for e in events if e.get("kind") == "trigger"]
    return [e for e in rows if e.get("trigger") == kind] if kind else rows


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


def test_no_runtime_pipes_a_trigger_dict_straight_into_emit():
    """The typo, inverted into an assertion. It cost every live run its first trigger while
    34 detector unit tests stayed green, and it is two characters from coming back."""
    call = re.compile(r'^\s*[\w.]+\.emit\("trigger"', re.M)      # statements, not prose
    for mod in (graph, deep_runner):
        assert not call.search(inspect.getsource(mod)), (
            f"{mod.__name__} is passing a Trigger dict into emit() again; `kind` is in it")


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


# ==========================================================================================
# 1. ZERO_HIT_SEARCH — a search that found nothing widens the term list
# ==========================================================================================
def test_zero_hit_search_reaches_the_supervisor_and_the_new_term_becomes_an_obligation(
        spec, chart, tmp_path):
    llm = ScriptedLLM(
        acts=[("search_notes", {"query": "pseudomyxoma peritonei"})],
        reflections=[{"verdict": "CONTINUE", "reason": "the term list is wrong for this chart",
                      "revision": {"add_terms": ["mucinous"]}}],
        assignments=_assignments(chart))
    agent, _, events = _run(spec, chart, llm, tmp_path, "zero-hit", max_steps=4)

    (hit,) = llm.results_for("search_notes")
    assert hit["n_hits"] == 0, "fixture assumption: this query finds nothing in SYN0001"

    # ...it reached the supervisor as an observation requiring a response...
    assert f"[{TRIGGER_ZERO_HIT_SEARCH}]" in llm.reflect_prompts[0]
    # ...the runtime applied the answer...
    applied = [e for e in events if e.get("kind") == "plan_revision" and e["applied"]]
    assert applied and applied[0]["outcome"]["terms_added"] == ["mucinous"]
    # ...and the plan the agent now works is the widened one.
    assert "mucinous" in agent.plan.keywords
    assert any("mucinous" in p for p in llm.act_prompts[1:]), (
        "the term was added to a plan the model was never shown again")
    # THE REAL SUBSEQUENT BEHAVIOUR: you added the term, you must now run the search.
    assert any("mucinous" in m for m in agent._outstanding_obligations())


# ==========================================================================================
# 2. UNSETTLED_THREAD — a truncated read blocks the answer until it is settled
# ==========================================================================================
def test_an_unsettled_thread_blocks_submission_and_a_resolution_unblocks_it(spec, chart,
                                                                            tmp_path):
    """The 8046 error, wired end to end: the addendum 353 characters past where the read
    stopped is exactly this trigger."""
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    # EVIDENCE_INSUFFICIENT and not FOUND: an open thread bears on a negative exactly as it
    # does on a positive (the addendum you never opened is the document that would have
    # changed it), and this way the gate is not answering "no evidence recorded" instead.
    submit = ("submit_answer", {"status": "EVIDENCE_INSUFFICIENT", "reasoning": "scripted",
                                "value": {}})
    llm = ScriptedLLM(
        acts=[("read_document", {"note_id": nid, "limit": 40}), submit, submit],
        reflections=[
            {"verdict": "CONTINUE", "reason": "read the rest before answering"},
            {"verdict": "CONTINUE", "reason": "the addendum settled the stain",
             "revision": {"resolve_threads": [{"thread_id": f"{nid}#truncated",
                                               "how": "read to the end; the addendum is final"}]}},
        ],
        assignments=_assignments(chart))
    agent, _, events = _run(spec, chart, llm, tmp_path, "thread", max_steps=6)

    (t,) = _triggers(events, TRIGGER_UNSETTLED_THREAD)
    assert t["marker"] == "truncated"
    rejects = [e for e in events if e.get("kind") == "answer_rejected"]
    assert len(rejects) >= 2, "fixture assumption: the scripted agent submitted twice"
    assert any("unsettled thread" in m for m in rejects[0]["missing"]), (
        "an open thread must REFUSE the answer; advice a model may decline is not a control")
    assert not any("unsettled thread" in m for m in rejects[1]["missing"]), (
        "the thread was resolved at reflection and the gate still blocked on it")
    assert check_threads(agent.threads) == []
    assert agent.threads.threads[0].state == "resolved"


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


def test_a_cited_quote_no_term_would_have_found_adds_the_term_it_suggested(spec, chart,
                                                                           tmp_path):
    kws = spec_declared_keywords(spec)
    nid, start, end = _unlisted_span(chart, kws)

    def add_the_suggested_term(prompt: str):
        """Answer with what the trigger actually proposed — not with a term the test knew."""
        m = re.search(r"candidate terms: \[([^\]]*)\]", prompt)
        assert m, "the trigger reached the supervisor without its candidate terms"
        first = m.group(1).split(",")[0].strip().strip("'\"")
        return {"verdict": "CONTINUE", "reason": "the retrieval plan did not lead here",
                "revision": {"add_terms": [first]}}

    llm = ScriptedLLM(
        acts=[("record_evidence", {"note_id": nid, "start": start, "end": end,
                                   "supports": "site"})],
        reflections=[add_the_suggested_term],
        assignments=_assignments(chart))
    agent, result, events = _run(spec, chart, llm, tmp_path, "unlisted", max_steps=4)

    (t,) = _triggers(events, TRIGGER_UNLISTED_ANSWER_TERM)
    assert t["note_id"] == nid and t["terms_proposed"], (
        "a trigger with no suggestion is a shrug; the candidate terms are the artefact")
    assert f"[{TRIGGER_UNLISTED_ANSWER_TERM}]" in llm.reflect_prompts[0]

    added = [r for r in agent.plan.term_provenance
             if r["trigger"] == TRIGGER_UNLISTED_ANSWER_TERM]
    assert len(added) == 1, "the most valuable develop-plane signal never reached the plan"
    assert added[0]["term"] in t["terms_proposed"]
    assert added[0]["observation"], "a term with no observation is not a candidate spec edit"
    # And it lands in the manifest as a candidate spec edit, which is the whole point of it.
    cand = result["develop_plane_candidates"]["terms_added_at_runtime"]
    assert any(c["term"] == added[0]["term"] for c in cand)


# ==========================================================================================
# 4. GATE_OBLIGATION_UNREACHABLE — the deadlock, broken by a promotion
# ==========================================================================================
def test_a_gate_obligation_the_plan_forbids_is_broken_by_the_promotion_it_forces(spec, chart,
                                                                                 tmp_path):
    """The deadlock: the gate says "read these search hits", the plan says "you may not open
    that type", and the old loop spent the rest of its budget in the gap."""
    blocked_type = "Endoscopy"          # a `may_mention` stratum type the planner sampled away
    ct = _first_of_type(chart, blocked_type)

    def promote_what_it_named(prompt: str):
        m = re.search(r"candidate types: \[([^\]]*)\]", prompt)
        assert m, "the gate trigger reached the supervisor without its candidate types"
        first = m.group(1).split(",")[0].strip().strip("'\"")
        return {"verdict": "CONTINUE", "reason": "the hits are in a type I may not open",
                "revision": {"promote_types": [{"type": first, "to": "search"}]}}

    llm = ScriptedLLM(
        acts=[("read_document", {"note_id": ct}),                 # refused: OUT_OF_PLAN
              ("search_notes", {"query": "right upper lobe"}),    # hits land in that type
              ("read_document", {"note_id": ct})],                # allowed after promotion
        reflections=[{"verdict": "CONTINUE", "reason": "look for the site first"},
                     promote_what_it_named],
        assignments=_assignments(chart, sampled=(blocked_type,)))
    agent, _, events = _run(spec, chart, llm, tmp_path, "gate-deadlock", max_steps=6)

    t = _triggers(events, TRIGGER_GATE_OBLIGATION_UNREACHABLE)[0]
    assert blocked_type in t["types_proposed"]

    reads = llm.results_for("read_document")
    assert reads[0].get("error") == "OUT_OF_PLAN", "fixture assumption: the plan governed"
    assert reads[-1].get("note_id") == ct, (
        "the promotion was applied to the plan object but the dispatch guard still refused")
    assert agent.plan.policy_for(blocked_type) == "search"


# ==========================================================================================
# 5. THE PLANNER MUST NOT EAT THE AGENT'S BUDGET
# ==========================================================================================
PLANNER_TERMS = ["adenocarcinoma", "lobe", "nodule", "mass", "malignant", "resection",
                 "cytology", "metastasis"]


def _remaining_terms(prompt: str) -> int:
    m = re.search(r"EXPANSION REMAINING: terms (-?\d+)", prompt)
    assert m, "the reflection prompt no longer states the remaining expansion"
    return int(m.group(1))


def test_the_up_front_planners_own_terms_are_not_charged_to_the_agent(spec, chart, tmp_path):
    """PLAN_PROMPT asks the planner for a keyword list, and the planner's proposals are
    `term_provenance` rows like any other. Pricing the cap against SPEC-declared terms while
    counting it against EVERY row spent the agent's whole allowance before it reflected once:
    `EXPANSION REMAINING: terms -3`, first revision refused BUDGET_EXHAUSTED, run over at
    step 1 of 12."""
    assert len(PLANNER_TERMS) > len(spec_declared_keywords(spec)), (
        "fixture assumption: the planner proposes more terms than the spec declares, which "
        "is what drove the remaining count negative")
    llm = ScriptedLLM(
        acts=[("search_notes", {"query": "pseudomyxoma peritonei"})],
        reflections=[{"verdict": "CONTINUE", "reason": "one term should fix this",
                      "revision": {"add_terms": ["mucinous"]}}],
        assignments=_assignments(chart), planner_keywords=PLANNER_TERMS)
    agent, result, events = _run(spec, chart, llm, tmp_path, "budget-split", max_steps=6)

    assert _remaining_terms(llm.reflect_prompts[0]) == len(spec_declared_keywords(spec)), (
        "the supervisor was shown an allowance the planner had already spent")
    rev = [e for e in events if e.get("kind") == "plan_revision"]
    assert rev and rev[0]["applied"], (
        f"the first single-term revision of the run was refused: {rev and rev[0]['outcome']}")
    assert "mucinous" in agent.plan.keywords
    assert not [e for e in events if e.get("kind") == "expansion_budget_exhausted"], (
        "the run ended on an expansion budget the agent had never spent")

    # The manifest already separated the two counts; the BUDGET now makes the same split.
    b = result["expansion_budget"]
    assert b["planner_proposed_terms"] == len(PLANNER_TERMS)
    assert b["max_terms_added_by_reflection"] == len(spec_declared_keywords(spec))
    assert result["replan"]["terms_added"] == len(PLANNER_TERMS) + 1
    assert result["replan"]["terms_added_by_reflection"] == 1


# ==========================================================================================
# 6. PARTIAL APPLICATION — "you may have 5 of the 6", and thread work is never collateral
# ==========================================================================================
def _tight(**kw):
    base = dict(max_terms_added=2, max_type_promotions=1,
                max_documents_opened_by_promotion=500, max_revisions=6)
    base.update(kw)
    return ExpansionBudget(**base)


def test_one_term_too_many_applies_what_fits_and_reports_back_what_did_not(spec, chart,
                                                                          tmp_path):
    llm = ScriptedLLM(
        acts=[("search_notes", {"query": "pseudomyxoma peritonei"})],
        reflections=[{"verdict": "CONTINUE", "reason": "three terms would cover the field",
                      "revision": {"add_terms": ["mucinous", "signet", "cribriform"]}}],
        assignments=_assignments(chart))
    agent, result, events = _run(spec, chart, llm, tmp_path, "partial", max_steps=6,
                                 expansion_budget=_tight())

    assert agent.plan.keywords[-2:] == ["mucinous", "signet"], (
        "all-or-nothing on a one-term overrun: the agent is never told it may have 2 of 3")
    assert "cribriform" not in agent.plan.keywords
    (part,) = [e for e in events if e.get("kind") == "revision_partially_applied"]
    assert part["deferred_terms"] == ["cribriform"]
    # REPORTED BACK, in the loop the model already understands. A refusal it never sees is a
    # refusal it repeats.
    told = [m for m in llm.act_prompts if "cribriform" in m]
    assert told, "what was refused never reached the model"
    assert any(REFUSED_BUDGET in m for m in told)
    # A term the run asked for and could not have is evidence about the spec's list. Partial
    # application is what keeps it out of `refused_revisions`, so it is harvested by name.
    d = result["develop_plane_candidates"]
    assert d["terms_deferred_for_budget"] == ["cribriform"]
    assert result["expansion_budget"]["exhausted"] is False, (
        "terms are spent but a type promotion is still affordable; the plan can still widen")


def test_a_term_overrun_never_discards_the_thread_work(spec, chart, tmp_path):
    """The compound failure: an over-budget revision also carried the resolution of the very
    thread that was blocking the answer, so the run ended budget-exhausted AND thread-blocked."""
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    llm = ScriptedLLM(
        acts=[("read_document", {"note_id": nid, "limit": 40}),
              ("search_notes", {"query": "pseudomyxoma peritonei"})],
        reflections=[
            {"verdict": "CONTINUE", "reason": "settle the thread and widen the terms",
             "revision": {"add_terms": ["mucinous", "signet", "cribriform"],
                          "resolve_threads": [{"thread_id": f"{nid}#truncated",
                                               "how": "read to the end"}]}}],
        assignments=_assignments(chart))
    agent, _, events = _run(spec, chart, llm, tmp_path, "partial-thread", max_steps=6,
                            expansion_budget=_tight())

    assert check_threads(agent.threads) == [], (
        "the thread resolution was thrown away because one term did not fit the budget")
    assert agent.threads.threads[0].state == "resolved"
    assert "cribriform" not in agent.plan.keywords


def test_thread_work_survives_a_retrieval_half_that_is_refused_outright(spec, chart, tmp_path):
    """A hallucinated type refuses the revision WHOLE — correctly, for the retrieval half.
    Thread bookkeeping is not the retrieval half and must not go down with it."""
    nid = _first_of_type(chart, PATHOLOGY_TYPE)
    llm = ScriptedLLM(
        acts=[("read_document", {"note_id": nid, "limit": 40}),
              ("document_type_summary", {})],
        reflections=[
            {"verdict": "CONTINUE", "reason": "settle the thread, and open the imaging",
             "revision": {"promote_types": [{"type": "Not-A-Real-Type", "to": "read_all"}],
                          "resolve_threads": [{"thread_id": f"{nid}#truncated",
                                               "how": "read to the end"}]}}],
        assignments=_assignments(chart))
    agent, _, events = _run(spec, chart, llm, tmp_path, "salvage", max_steps=6)

    assert [e for e in events if e.get("kind") == "plan_revision" and not e["applied"]], (
        "the hallucinated type must still be refused; dropping it would leave the agent "
        "believing it had widened a scope it had not")
    (salv,) = [e for e in events if e.get("kind") == "thread_work_salvaged"]
    assert salv["threads_resolved"] == [f"{nid}#truncated"]
    assert check_threads(agent.threads) == []


def test_a_term_overrun_alone_does_not_end_the_run_while_promotions_remain(spec, chart,
                                                                           tmp_path):
    """STICKY EXHAUSTION. One refused term used to mark the plan spent forever, ending the
    run even though it could still promote a type — the widening move that was actually
    needed."""
    llm = ScriptedLLM(
        acts=[("search_notes", {"query": "pseudomyxoma peritonei"})],
        reflections=[{"verdict": "CONTINUE", "reason": "widen the terms",
                      "revision": {"add_terms": ["mucinous", "signet", "cribriform"]}},
                     {"verdict": "CONTINUE", "reason": "now open the imaging",
                      "revision": {"promote_types": [{"type": SAMPLED_TYPE, "to": "search"}]}}],
        assignments=_assignments(chart))
    agent, _, events = _run(spec, chart, llm, tmp_path, "sticky", max_steps=6,
                            expansion_budget=_tight())

    assert agent.plan.policy_for(SAMPLED_TYPE) == "search", (
        "the promotion the run still had budget for was refused as budget-exhausted")
    promoted = [e for e in events if e.get("kind") == "plan_revision"
                and e["outcome"]["types_promoted"]]
    assert promoted, "the promotion never happened; the term overrun ended the run first"
    spent = [e for e in events if e.get("kind") == "expansion_budget_exhausted"]
    assert all(e["seq"] > promoted[0]["seq"] for e in spent), (
        "the run was declared spent before it had spent the promotion allowance it still had")


def test_the_run_still_ends_honestly_when_expansion_really_is_spent(spec, chart, tmp_path):
    """The other half. Partial application must not turn a genuine dead end into a run that
    quietly keeps going: budget spent with obligations outstanding is EVIDENCE_INSUFFICIENT
    and it has to be SAID."""
    llm = ScriptedLLM(
        acts=[("search_notes", {"query": "pseudomyxoma peritonei"})],
        reflections=[{"verdict": "CONTINUE", "reason": "widen everything",
                      "revision": {"add_terms": ["mucinous", "signet", "cribriform"],
                                   "promote_types": [{"type": SAMPLED_TYPE, "to": "search"}]}}],
        assignments=_assignments(chart))
    agent, result, events = _run(
        spec, chart, llm, tmp_path, "spent", max_steps=8,
        expansion_budget=_tight(max_terms_added=2, max_type_promotions=1))

    assert agent._outstanding_obligations(), "fixture assumption: the gate is not met"
    assert [e for e in events if e.get("kind") == "expansion_budget_exhausted"], (
        "expansion is over and the obligation is not met; a run that keeps looping to "
        "max_steps and emits whatever is in hand is a silent truncation")
    assert result["answer"]["status"] == "EVIDENCE_INSUFFICIENT"
