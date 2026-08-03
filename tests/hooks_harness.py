"""One hooks-driven harness, so tests of LIVE rules do not die with the runtime that is gone.

WHY THIS EXISTS. Deleting `graph.py` orphaned 56 tests across six files. Roughly half of them
test mechanics that went with it — the reflect node's verdict vocabulary, the replan rate in the
manifest, the wording of the monotonicity note — and those are gone for good. The other half
test rules that are STILL IN THE PRODUCT and merely borrowed the old loop as a harness:

    reading a document to its end settles its own `truncated` thread
    a window read of an already-complete document opens nothing
    an unsettled thread blocks submission and a resolution unblocks it
    a sampled type may not be opened, and the refusal names the way out
    SPEC_INSUFFICIENT is reached through the gate, not around it

Two of those pin fixes made the same day the runtime was replaced. Deleting them would remove
the only coverage of live audit rules and leave a green suite that had stopped asking.

WHAT IT CANNOT DO. The old doubles were litellm-shaped and some scripted the reflect node
through `json_chat`. The hooks runtime has no reflect node and never calls `json_chat`, so a
script written for it has nothing to answer — those tests are testing a mechanism, not a rule,
and they are the ones that go.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("langchain_core.language_models.chat_models")

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from acr.review.agent import run_patient


class ToolScript(BaseChatModel):
    """Replays a fixed list of tool calls, then submits. One entry per model turn.

    A list rather than a transcript-reader because these tests want to drive a SPECIFIC
    sequence — read this document at this offset, then that section — which is exactly what the
    rules under test key on.
    """

    script: list = []
    submit: dict = {}
    turn: int = 0
    seen: list = []

    @property
    def _llm_type(self) -> str:
        return "tool-script"

    def bind_tools(self, tools, **kw):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kw) -> ChatResult:
        self.seen.append(list(messages))
        i, self.turn = self.turn, self.turn + 1
        if i < len(self.script):
            name, args = self.script[i]
            call = {"name": name, "args": dict(args), "id": f"t{i}"}
        else:
            call = {"name": "submit_answer", "args": dict(self.submit or {
                "status": "EVIDENCE_INSUFFICIENT", "value": {},
                "reasoning": "the script ran out"}), "id": f"t{i}"}
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content="", tool_calls=[call]))])


def run_scripted(spec, corpus, patient_id, tmp_path, *, script, submit=None, max_model_calls=12,
                 run_id="scripted", seed=7):
    """Drive the real runtime with a scripted provider. Returns (manifest, trace events).

    Everything but the completions is real: the graph, the middleware, the toolbox, the
    coverage ledger, the thread ledger and the gate.
    """
    model = ToolScript(script=list(script), submit=dict(submit or {}))
    model.seen = []
    m = run_patient(spec=spec, corpus=corpus, patient_id=patient_id, out_dir=Path(tmp_path),
                    model=model, max_model_calls=max_model_calls, seed=seed, run_id=run_id)
    trace = Path(tmp_path) / f"{run_id}.jsonl"
    events = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if l.strip()]
    return m, events


def triggers(events, kind=None):
    """Trigger events, read the way a develop-plane consumer would.

    The trigger's own kind lives under `trigger`, because `kind` is the trace ENVELOPE's event
    type and one key cannot be both.
    """
    rows = [e for e in events if e.get("kind") == "trigger"]
    return [e for e in rows if e.get("trigger") == kind] if kind else rows


class LitellmScriptAdapter(BaseChatModel):
    """Wrap a litellm-shaped scripted client as a `BaseChatModel`.

    The surviving tests script the provider through `chat(messages, tools) -> LLMResponse`, and
    NONE of them call `json_chat` — the ones that did were scripting the reflect node, which is
    a mechanism rather than a rule, and they went with it. So the translation is mechanical:
    convert LangChain messages to the dict shape the script reads, hand back its tool calls as
    an `AIMessage`.

    This is what let ~25 tests of LIVE audit rules survive the runtime change instead of being
    deleted with it. Two of them pin fixes made the same day, and one of them caught a real
    regression on the first run: the port had dropped `_record_reads`, so a document read to its
    end no longer settled its own `truncated` thread and the deadlock came back.
    """

    inner: object = None

    @property
    def _llm_type(self) -> str:
        return "litellm-script-adapter"

    def bind_tools(self, tools, **kw):
        return self

    @staticmethod
    def _as_dicts(messages: list[BaseMessage]) -> list[dict]:
        out = []
        for m in messages:
            t = getattr(m, "type", None)
            role = {"human": "user", "ai": "assistant", "system": "system",
                    "tool": "tool"}.get(t, "user")
            d = {"role": role, "content": m.content if isinstance(m.content, str) else str(m.content)}
            if role == "tool":
                d["name"] = getattr(m, "name", "") or ""
                d["tool_call_id"] = getattr(m, "tool_call_id", "") or ""
            out.append(d)
        return out

    def _generate(self, messages: list[BaseMessage], stop=None,
                  run_manager: CallbackManagerForLLMRun | None = None, **kw) -> ChatResult:
        # The SYSTEM message is where the plan, the open threads and the mechanical
        # observations ride. Recorded on the inner script so a test can assert on what the
        # agent was actually shown rather than on what a prompt template claims it shows.
        for m in messages:
            if getattr(m, "type", None) == "system":
                txt = m.content if isinstance(m.content, str) else str(m.content)
                getattr(self.inner, "seen_system", []).append(txt)
        r = self.inner.chat(self._as_dicts(messages), tools=[{"x": 1}])
        calls = [{"name": c["name"], "args": c.get("arguments") or {},
                  "id": c.get("id") or f"a{i}"}
                 for i, c in enumerate(getattr(r, "tool_calls", None) or [])]
        return ChatResult(generations=[ChatGeneration(
            message=AIMessage(content=getattr(r, "content", "") or "", tool_calls=calls))])


def run_with_script(spec, corpus, patient_id, tmp_path, llm, *, run_id="scripted",
                    max_model_calls=12, seed=7, ctx_out=None, expansion_budget=None):
    """Drive the real runtime with a litellm-shaped scripted client. (manifest, events).

    `ctx_out` is a list the runtime appends its live `RunContext` to, so a test can assert on
    the plan and the ledgers the run actually used rather than on a second assembly of them.
    """
    m = run_patient(spec=spec, corpus=corpus, patient_id=patient_id, out_dir=Path(tmp_path),
                    model=LitellmScriptAdapter(inner=llm), max_model_calls=max_model_calls,
                    seed=seed, run_id=run_id, ctx_out=ctx_out,
                    expansion_budget=expansion_budget)
    trace = Path(tmp_path) / f"{run_id}.jsonl"
    events = [json.loads(l) for l in trace.read_text(encoding="utf-8").splitlines() if l.strip()]
    return m, events


def revise_plan_tool(spec, chart, *, expansion_budget=None, threads=None):
    """The declared `revise_plan` tool, bound to real ledgers. (tool, ctx).

    Several tests here are about the ARITHMETIC of a revision — what fits in the budget, what is
    refused as redundant, whether the thread half survives a refused retrieval half — and they
    used to reach it by scripting a reflect node and running the whole loop. The operation is a
    tool call now, so it can be invoked directly. That is shorter and it is also a better test:
    a full run can fail for a dozen unrelated reasons before it reaches the arithmetic.
    """
    from acr.contract.trace import Tracer
    from acr.core.state import EvidenceLedger
    from acr.review.agent import RunContext, make_revise_plan_tool
    from acr.review.coverage import CoverageLedger, ForcedSampler, strata_from_spec
    from acr.review.coverage_planner import (
        ExpansionBudget,
        OpenThreadLedger,
        load_marker_catalogue,
        plan_from_spec,
    )
    from acr.review.tools import Toolbox

    docs, _ = chart.list_documents(limit=100_000)
    evidence = EvidenceLedger()
    coverage = CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(7))
    plan = plan_from_spec(spec, chart)
    ctx = RunContext(spec=spec, chart=chart, plan=plan, coverage=coverage,
                     threads=threads or OpenThreadLedger(),
                     catalogue=load_marker_catalogue(),
                     tracer=Tracer.create(Path("/tmp") / "acr-revise-tests"),
                     gate=lambda submitted: {"accepted": False},
                     toolbox=Toolbox(chart, evidence, coverage))
    budget = expansion_budget or ExpansionBudget(max_terms_added=40, max_type_promotions=8,
                                                 max_documents_opened_by_promotion=40,
                                                 max_revisions=6)
    return make_revise_plan_tool(ctx, budget), ctx


def revise(tool, **kwargs) -> dict:
    """Call the tool the way the model does and parse what it hands back."""
    return json.loads(tool.func(**kwargs))
