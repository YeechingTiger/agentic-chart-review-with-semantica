"""The chart-review agent as a LangChain/deepagents graph: our rules live in HOOKS.

WHY THIS REPLACES `graph.py`
`graph.py` is 1,197 lines of hand-written plan/act/reflect/finalize nodes plus the edges
between them. Every one of those nodes has a hook with the same shape, and the hook is
better in each case because the framework owns when it fires:

    graph.py node                     hook                     what the hook gives us
    ----------------------------------------------------------------------------------
    _n_plan, first entry              before_agent             runs once, before any call
    the PLAN block in the prompt      wrap_model_call          per-call and IMMUTABLE
    _n_act dispatch + refusals        wrap_tool_call           sees EVERY tool call
    _n_reflect + the REPLAN edge      after_model              fires after every decision
    _n_finalize                       after_agent              runs once, at the end
    Budget(max_steps=...)             ModelCallLimitMiddleware library

THE PLAN BELONGS IN `wrap_model_call`, AND THAT IS THE WHOLE 41%
`graph.py` appended `plan.render()` to the message list — once per plan-node entry and again
in every reflect message that announced an applied revision. Measured on a real 293-document
chart: 6,310 characters, ELEVEN copies, each re-sent on all forty-nine later calls. ~425,000
of that run's 1,030,179 prompt tokens — 41% — spent re-reading ten stale copies of a plan
whose current version sat at the bottom of the same prompt.

`ModelRequest.override(system_message=...)` returns a NEW request and never touches
`messages`. The plan is rebuilt from the live `CoveragePlan` on every call, so there is
exactly one copy and it is always current. Not because a dedupe pass removed the others —
because there is nowhere for them to accumulate. Deduping was the patch; this is the fix.

THE TYPED CHANNEL IS A TOOL, NOT A NODE
`graph.py`'s reflect node carried one typed channel doing two jobs: widen the plan, and settle
open threads. Replacing reflect with hooks dropped both, and the thread half is the one that
bites — the gate refuses while a thread is open and tells the agent to `resolve_threads` or
`dismiss_threads`, tools that then did not exist. SYN0002 resubmitted until
GraphRecursionError at 72. `revise_plan` restores it as a declared tool, so the proposal is
typed, `wrap_tool_call` audits it like any other call, and `CoveragePlan.apply_revision`
enforces monotonicity by refusing demotions. Revisions are proposed by the MODEL — the
trigger detector only renders candidates into a prompt (`graph.py:601-604`) — which is why
this is a tool and not something `after_model` could derive.

WHY `create_agent` AND NOT `create_deep_agent`
`create_deep_agent` injects nine tools nobody asked for: ls, glob, grep, read_file,
write_file, edit_file, execute, task, write_todos. Four are read paths, and a read that does
not go through `Toolbox.dispatch` is invisible to the `CoverageLedger` — the gate would still
stamp `gate_validated: true` over a chart the ledger never saw read. Worse, under the
`FilesystemBackend(root_dir=".")` the skills path uses, `read` and `grep` reach ABSOLUTE
paths outside root_dir:

    read("/N/project/computable_phenotype/acr_real/ground_truth.csv")
        -> ReadResult(error=None, content="person_id,...,gt_primary_site,gt_histology,...")

That is the answer key, reachable from the RUN plane. No recorded run exercised it — all
seven had `skills_enabled=0` and called no built-in tool — but nobody having walked through
an open door is not a boundary. `create_agent` takes exactly the tools it is given.
"""
from __future__ import annotations

import json
import collections
from dataclasses import dataclass, field
from typing import Any, Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (ModelCallLimitMiddleware, TodoListMiddleware,
                                         hook_config)
from langchain.agents.middleware.types import (AgentMiddleware, ModelRequest, ToolCallRequest)
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool

from .coverage_planner import OPEN_REQUEST_OPENED, triggers_from_tool_result
from .tool_surface import LIBRARY_TOOLS, ToolSurfaceError, assert_tool_surface  # noqa: F401

#: Tools whose results the coverage ledger must see. Named here rather than inferred from the
#: toolbox, because this list answers "what counts as having looked" — a claim about the audit
#: rule, not about which functions happen to exist.
READ_TOOLS = ("read_document", "read_section", "read_documents_batch")

#: The standing instruction. Beside the runtime it drives, now that the runner it used to live
#: in is gone. The gate's contract — a rejection is the instruction for what to do next — is
#: the whole reason this prompt is shaped the way it is.
TASK = """Determine the answer for patient {patient} using ONLY this chart.

Work by calling tools: list the documents, search them, read what matters, and record every
claim with record_evidence and a verbatim quote. When you are ready call submit_answer.

submit_answer is GATED. If the proof obligation is not met it will be rejected with the
reason, and you must act on that reason and submit again. A rejection is not a failure; it
is the instruction for what to do next.

If a rule refuses every value the record can support, that is a finding about the
SPECIFICATION: submit SPEC_INSUFFICIENT and name the check at fault."""


@dataclass
class RunContext:
    """Everything the audit hooks read or advance. Plain data; the graph is the library's."""
    spec: Any
    chart: Any
    plan: Any
    coverage: Any
    threads: Any
    catalogue: Any
    tracer: Any
    gate: Callable[[dict], dict]
    toolbox: Any = None
    submitted: dict = field(default_factory=dict)
    accepted: bool = False
    answer: dict = field(default_factory=dict)
    rejections: list = field(default_factory=list)
    declared: set[str] = field(default_factory=set)
    revisions: list = field(default_factory=list)
    rejection_fingerprints: collections.Counter = field(
        default_factory=collections.Counter)
    n_model_calls: int = 0
    no_tool_call: int = 0
    #: Floor of 2 is the library's own: one rejection is not a loop.
    max_rejection_repeats: int = 3
    stalled: dict | None = None


class AuditMiddleware(AgentMiddleware):
    """The audit plane, spread across the four hooks that structurally suit each rule.

    One class rather than four, because these rules share one `RunContext` and splitting them
    would put the ledger behind an interface that three of them mutate. What is deliberate is
    WHICH HOOK each rule sits in — that is the design, not the packaging.
    """

    def __init__(self, ctx: RunContext):
        super().__init__()
        self.ctx = ctx

    # ------------------------------------------------------------------- before_agent
    def before_agent(self, state, runtime) -> dict | None:
        """Once, before the first call: say what the plan was BEFORE the model touched it.

        Emitted here rather than at construction so the trace's first plan event is inside
        the run it describes. A plan recorded outside the run cannot be replayed against it.
        """
        self.ctx.tracer.emit("retrieval_plan", runtime="deepagents",
                             source=self.ctx.plan.source, plan=self.ctx.plan.to_dict(),
                             marker_catalogue=self.ctx.catalogue.source, revisable=True,
                             revisable_why=("the agent revises through the declared "
                                            "`revise_plan` tool; apply_revision refuses "
                                            "demotions, so the plan can only widen"))
        return None

    # ---------------------------------------------------------------- wrap_model_call
    def wrap_model_call(self, request: ModelRequest, handler):
        """Put the CURRENT plan in the system message. It cannot accumulate — see module doc.

        Appended to the system message rather than pushed as a user turn, for the same reason
        `TodoListMiddleware` does it: the system message is replaced wholesale on every call,
        so yesterday's plan is not still in the thread arguing with today's.
        """
        self.ctx.n_model_calls += 1
        block = "PLAN (current):\n" + self.ctx.plan.render(self._docs_by_type())
        sm = request.system_message
        content = ((sm.content if isinstance(sm.content, str) else str(sm.content)) + "\n\n"
                   if sm is not None else "") + block
        return handler(request.override(system_message=SystemMessage(content=content)))

    def _docs_by_type(self) -> dict[str, int]:
        return {r["doc_type"]: r["count"] for r in self.ctx.chart.type_summary()}

    # -------------------------------------------------------------------- after_model
    @hook_config(can_jump_to=["model"])
    def after_model(self, state, runtime) -> dict | None:
        """A model turn that calls no tool ends `create_agent`. Do not let it end THIS run.

        WHAT THIS COSTS WHEN IT IS MISSING. On a 293-document real chart the run stopped at
        13 of 50 allowed calls with `NO_ANSWER`, no error and no budget exhausted — the gate
        had rejected once, a revision came back partly refused, three new threads opened, and
        the model replied with prose instead of a tool call. `create_agent` reads "no tool
        calls" as "done" and goes to END. So the run did not fail; it wandered off, and the
        manifest recorded a silence that looks exactly like a chart with nothing in it.

        `graph.py` had this and called it `act_no_tool_call`: append "Continue by calling a
        tool", loop. The library equivalent is a jump, and it is BETTER than the old counter
        because the message says what is still outstanding instead of nagging generically.

        Not a nag loop: `ModelCallLimitMiddleware` still owns termination, so an agent that
        will not act runs out of calls and stops with a budget reason, which is a fact a
        reader can act on. Silence is not.
        """
        msgs = state.get("messages") or []
        last = msgs[-1] if msgs else None
        if last is None or getattr(last, "type", None) != "ai":
            return None
        if getattr(last, "tool_calls", None):
            return None
        if self.ctx.accepted:
            # It said its piece after a gate-validated answer. That is the run finishing.
            return None
        self.ctx.no_tool_call += 1
        open_ids = [t.thread_id for t in self.ctx.threads.unresolved()]
        nudge = ("You ended a turn without calling a tool, and no answer has passed the gate "
                 "yet, so nothing has been recorded. ")
        if open_ids:
            nudge += (f"{len(open_ids)} thread(s) still block submit_answer: {open_ids[:5]}. "
                      "Settle each with revise_plan — resolve_threads if you found where the "
                      "deferred text was settled, dismiss_threads with a reason if it cannot "
                      "be settled from this chart. ")
        nudge += "Call a tool now, or call submit_answer."
        self.ctx.tracer.emit("act_no_tool_call", severity="warning",
                             n=self.ctx.no_tool_call, unresolved=open_ids,
                             message="model turn produced no tool call; jumping back to model")
        return {"messages": [{"role": "user", "content": nudge}], "jump_to": "model"}

    # ---------------------------------------------------------------- wrap_tool_call

    # ---------------------------------------------------------------- wrap_tool_call
    def _refuse(self, request: ToolCallRequest, payload: dict) -> ToolMessage:
        return ToolMessage(content=json.dumps(payload, default=str)[:8000],
                           tool_call_id=request.tool_call["id"],
                           name=request.tool_call["name"], status="error")

    def _undeclared(self, name: str) -> dict | None:
        if name in self.ctx.declared or name in LIBRARY_TOOLS:
            return None
        # Reached only if something bound a tool after `assert_tool_surface` ran. Refused
        # rather than logged: an undeclared tool is by definition one whose effect on the
        # coverage ledger nobody has reasoned about.
        self.ctx.tracer.emit("undeclared_tool_refused", severity="error", tool=name)
        return {"error": "UNDECLARED_TOOL", "tool": name,
                "message": ("This tool is not part of the declared surface for this task and "
                            "its result cannot be admitted as evidence.")}

    def _out_of_plan(self, name: str, args: dict) -> dict | None:
        """The retrieval plan, enforced at dispatch rather than suggested in a prompt."""
        if name not in READ_TOOLS:
            return None
        ids = ([args.get("note_id")] if name != "read_documents_batch"
               else list(args.get("note_ids") or []))
        drawn = {n for v in self.ctx.coverage.drawn.values() for n in v}
        blocked = []
        for nid in ids:
            meta = self.ctx.chart._docs.get(str(nid))
            if meta is None or str(nid) in drawn:
                continue
            if not self.ctx.plan.may_open(meta.doc_type):
                blocked.append({"note_id": str(nid), "doc_type": meta.doc_type})
        if not blocked:
            return None
        self.ctx.tracer.emit("plan_refused_open", severity="warning", tool=name, blocked=blocked)
        return {"error": "OUT_OF_PLAN", "blocked": blocked,
                "types": sorted({b["doc_type"] for b in blocked}),
                "message": ("The retrieval plan assigns these types to `sample`: the runtime's "
                            "sampler draws from them and you may not open them directly. This "
                            "is NOT evidence that they hold nothing.")}

    def wrap_tool_call(self, request: ToolCallRequest, handler):
        """Every tool call, including one a library adds tomorrow.

        In `graph.py` these rules lived inside the act node's dispatch loop and in
        `deep_runner` inside each wrapped tool function. Both are per-tool, so both are silent
        about a tool they do not wrap — which is how nine injected tools would have sailed
        past. This hook is the last line, not the first.
        """
        name = request.tool_call["name"]
        args = request.tool_call.get("args") or {}
        for check in (self._undeclared(name), self._out_of_plan(name, args)):
            if check is not None:
                return self._refuse(request, check)
        result = handler(request)
        # BEFORE detection, always. A read that completes a document must settle its thread
        # before the same result is scanned for markers, or a window read of an
        # already-complete document can re-open what it just closed.
        self._record_reads(name, self._payload(result))
        if name == "submit_answer":
            # THE GATE, and it must be here rather than inside the tool. `_t_submit_answer`
            # records the submission and returns `{"received": true, "note": "pending
            # validation"}` — a receipt, not a verdict. If that receipt reached the model
            # unchanged the run would read it as acceptance and stop, which is precisely how
            # ungated FOUND answers got stamped `gate_validated: true` in the old runtime.
            # The receipt is replaced by the verdict, so the only thing the model can see is
            # the judgement.
            return self._gate_answer(request)
        self._detect(name, args, result)
        return result

    def _gate_answer(self, request: ToolCallRequest) -> ToolMessage:
        """One gate, shared with `graph.py`. A rejection carries the way out, not just a no."""
        submitted = dict(self.ctx.toolbox.submitted or {})
        verdict = self.ctx.gate(submitted)
        if not verdict.get("accepted"):
            stall = self._stalled(submitted, verdict)
            if stall is not None:
                return ToolMessage(content=json.dumps(stall, default=str)[:4000],
                                   tool_call_id=request.tool_call["id"], name="submit_answer",
                                   status="error")
        self.ctx.tracer.tool("submit_answer", submitted, verdict,
                             ok=bool(verdict.get("accepted")), ms=0.0)
        if verdict.get("accepted"):
            self.ctx.accepted = True
            self.ctx.answer = submitted
        else:
            self.ctx.rejections.append(verdict)
            self._detect_deadlock()
        return ToolMessage(content=json.dumps(verdict, default=str)[:8000],
                           tool_call_id=request.tool_call["id"], name="submit_answer",
                           status="success" if verdict.get("accepted") else "error")

    def _detect_deadlock(self) -> None:
        """An obligation the CURRENT plan structurally cannot discharge is a deadlock.

        THE FOURTH TRIGGER, and it was not ported. A gate that says "read these search hits"
        while the plan says "you may not open that type" is not a rejection the agent can
        satisfy — it is a loop, and the old runtime spent the rest of its budget inside one.
        The `revise_plan` tool is the way out, so the detector fires here, right after the
        refusal, where the agent is about to decide what to do next.

        `tracer` is required by the detector and passed for the reason its own docstring
        gives: it swallows its own exceptions, so without a channel to say so a detector that
        has stopped working is indistinguishable from a run with no deadlock to report.
        """
        from .run_triggers import detect_gate_obligations
        for t in detect_gate_obligations(spec=self.ctx.spec, coverage=self.ctx.coverage,
                                         chart=self.ctx.chart, plan=self.ctx.plan,
                                         step=self.ctx.n_model_calls,
                                         tracer=self.ctx.tracer):
            self.ctx.tracer.trigger(runtime="deepagents-hooks", **t.to_dict())

    def _stalled(self, submitted: dict, verdict: dict) -> dict | None:
        """Stop a run that is being refused the same way for the same answer, over and over.

        MEASURED, NOT HYPOTHETICAL. Two of ten real charts spent their entire 50-call budget
        on 26 and 28 rejections, of which 24 and 22 were the identical
        `not_less_specific`/`conflict_requires_nos` pair on a byte-identical value. That
        particular contradiction is fixed, but the SHAPE is not specific to it: any refusal the
        agent cannot satisfy produces it, and the run's only signal was a spent budget.

        `evals.py` already detects this — `--max-rejection-repeats`, whose help says "the
        library floor is 2: one rejection is not a loop". That detector reads FINISHED runs, so
        it can only tell you afterwards what you paid for. The same rule at runtime turns the
        spend into a labelled stop.

        The fingerprint is (rejection reason + coded value). Reason alone would stop an agent
        making real progress against a recurring obligation; value alone would stop one being
        refused for two different reasons. Only the pair repeating means nothing is moving.
        """
        fp = (str(verdict.get("why") or ""), json.dumps(submitted.get("value") or {}, sort_keys=True))
        self.ctx.rejection_fingerprints[fp] += 1
        n = self.ctx.rejection_fingerprints[fp]
        if n < self.ctx.max_rejection_repeats:
            return None
        self.ctx.stalled = {"fingerprint_repeats": n, "why": fp[0],
                            "value": submitted.get("value") or {}}
        self.ctx.tracer.emit("rejection_loop", severity="error", repeats=n, why=fp[0],
                             message=("the same rejection fired on the same value this many "
                                      "times; the run is stopped rather than allowed to spend "
                                      "the rest of its budget"))
        return {"accepted": False, "stop": True,
                "why": (f"REJECTION_LOOP: this exact answer has been refused {n} times for the "
                        f"same reason ({fp[0]}). Resubmitting it again cannot succeed."),
                "what_to_do": ("Change the VALUE, or submit EVIDENCE_INSUFFICIENT with the "
                               "reason you cannot satisfy this rule. If the rule itself cannot "
                               "be satisfied by any value the record supports, submit "
                               "SPEC_INSUFFICIENT and name the answer_check at fault — that is "
                               "a finding about the specification and it is wanted.")}

    def _record_reads(self, name: str, out: dict) -> None:
        """Hand every read's extent to the thread ledger, and trace what it settled.

        THE ROUTE FROM "I READ TO THE END OF IT" TO "THE THREAD IS SETTLED". It was ported from
        the runtime this one replaced, and it had to be: dropping it re-created the deadlock it
        was written to fix. On a scripted probe the partial read opened a `truncated` thread,
        the following full read settled nothing, and the run ended with the thread outstanding —
        the same shape as the run that paged to the end of a report thirteen times because
        nothing connected "I read it" to "the thread is discharged".

        Deliberately mechanical: no model is asked whether the document is finished, because the
        runtime computed `truncated` from the character counts and can compute the complement
        just as well. `truncated` is the only marker this may ever settle — see
        `MECHANICALLY_DISCHARGEABLE_MARKERS`.
        """
        threads = self.ctx.threads
        if threads is None or not isinstance(out, dict) or out.get("error"):
            return
        reads: list[tuple[str, int, int, int | None]] = []
        if name == "read_document" and out.get("note_id") and "returned_chars" in out:
            reads.append((str(out["note_id"]), int(out.get("offset") or 0),
                          int(out.get("returned_chars") or 0), out.get("total_chars")))
        elif name == "read_documents_batch":
            for d in (out.get("documents") or []):
                reads.append((str(d.get("note_id", "")), 0, len(str(d.get("text", ""))),
                              d.get("total_chars")))
        elif name == "read_section" and out.get("note_id") and "start" in out and "end" in out:
            # A named section carries TRUE offsets and no document length, so it contributes
            # coverage and can never on its own prove the document is complete. Reading FINAL
            # DIAGNOSIS tells you nothing about what sits after it.
            reads.append((str(out["note_id"]), int(out.get("start") or 0),
                          max(0, int(out.get("end") or 0) - int(out.get("start") or 0)), None))
        for note_id, offset, returned, total in reads:
            settled = threads.note_read(note_id, offset=offset, returned_chars=returned,
                                        total_chars=total, step=self.ctx.n_model_calls)
            if settled:
                self.ctx.tracer.emit(
                    "threads_settled_by_read", thread_ids=settled, note_id=note_id,
                    total_chars=threads.doc_length.get(note_id),
                    message=("the document has now been returned in full by reads in this run, "
                             "so its `truncated` thread is discharged deterministically — the "
                             "runtime owns both sides of that predicate and does not need to "
                             "ask"))

    @staticmethod
    def _payload(result: Any) -> dict:
        """The tool result as a dict. One parser, so read-recording and marker detection can
        never disagree about what the tool actually returned."""
        try:
            out = json.loads(getattr(result, "content", "") or "{}")
        except (json.JSONDecodeError, TypeError):
            return {}
        return out if isinstance(out, dict) else {}

    def _detect(self, name: str, args: dict, result: Any) -> None:
        """Triggers and threads, from the result the model just received.

        After the call and outside the refusal path on purpose: a refused call produced no
        evidence, and opening a thread against text nobody read puts debt on the run for a
        document it never saw.
        """
        payload = self._payload(result)
        if not payload:
            return
        quote = str(payload.get("quote", "")) if name == "record_evidence" else ""
        for t in triggers_from_tool_result(name, args, payload, plan=self.ctx.plan,
                                           catalogue=self.ctx.catalogue, step=0, quote=quote):
            if t.kind == "UNSETTLED_THREAD":
                m = self.ctx.catalogue.by_text().get(t.marker)
                req = self.ctx.threads.open_thread(
                    note_id=t.note_id, doc_type=t.doc_type, marker=t.marker,
                    obligation=(m.obligation if m else "unsettled"),
                    excerpt=t.observation, step=0)
                # Branch on the typed status. `is None` was the old sentinel test and it
                # counted every short read as a new thread once `open_thread` began handing
                # back the existing one.
                if req.status != OPEN_REQUEST_OPENED:
                    continue
            self.ctx.tracer.trigger(runtime="deepagents", **t.to_dict())


def build_agent(*, model, tools: list[StructuredTool], system_prompt: str, ctx: RunContext,
                backend, max_model_calls: int, summarization_model=None,
                keep_messages: int = 20):
    """The graph. Every node comes from the library; every rule comes from a hook.

    Middleware order is composition order. `AuditMiddleware` is last so its `wrap_tool_call`
    sits closest to the tool and sees the call after any earlier middleware has rewritten it,
    and so its `wrap_model_call` appends the plan after `TodoListMiddleware` has added its own
    system-prompt block.
    """
    from deepagents.middleware.summarization import SummarizationMiddleware

    ctx.declared = {t.name for t in tools}
    middleware = [
        # Planning. Todos live in STATE and `write_todos` REPLACES the list, so a revised plan
        # leaves no stale copy in the transcript.
        TodoListMiddleware(),
        # Context. Compaction plus offload of oversized tool results to the backend.
        SummarizationMiddleware(model=summarization_model or model, backend=backend,
                                keep=("messages", keep_messages)),
        # The budget the CLI can finally reach, as a library concern rather than a dataclass
        # every construction site forgot to pass.
        ModelCallLimitMiddleware(thread_limit=max_model_calls, exit_behavior="end"),
        AuditMiddleware(ctx),
    ]
    agent = create_agent(model, tools, system_prompt=system_prompt, middleware=middleware)
    assert_tool_surface(agent, ctx.declared)
    return agent


def recursion_limit_for(agent, max_model_calls: int, *, slack: int = 8) -> int:
    """Derive LangGraph's step limit from the graph, because guessing it costs a whole run.

    LangGraph counts SUPER-STEPS, not model calls: one turn is the model node, the tools node
    and every per-turn middleware node. This graph has six such nodes, so a 50-call budget
    needs ~300 steps and the `max_model_calls * 3` I first wrote allowed 150 — which is
    exactly where a real 293-document run died, at 25 model calls of the 50 it was granted,
    with `GraphRecursionError` and no answer. The budget the operator set was never reachable.

    Derived rather than tuned. A middleware added later changes the node count, and a constant
    would go quietly wrong again; this reads the graph that was actually built. `slack` covers
    the one-time before_agent/after_agent nodes and the final model turn.

    The point is WHICH limit stops a run. `ModelCallLimitMiddleware` stops with a reason a
    reader can act on; the recursion limit stops with a stack trace. The first must always
    bind, so this is set above it.
    """
    per_turn = sum(1 for n in agent.nodes
                   if n != "__start__" and not n.endswith((".before_agent", ".after_agent")))
    return max_model_calls * max(per_turn, 2) + slack


# ===================================================================== the run
# `after_agent` is where finalize belongs, but the manifest also needs the elapsed time, the
# usage the callbacks collected and the run directory — none of which are agent state. So the
# hook advances `ctx`, and this function, which owns those, assembles the record. The split is
# the same one `graph.py` had between `_n_finalize` and `run`.

def run_chart_review(*, spec, chart, toolbox, coverage, evidence, plan, threads, catalogue,
                     tracer, gate, model, tools, system_prompt, backend, max_model_calls,
                     out_dir, elapsed_fn, expansion_budget) -> dict:
    """One patient, one spec, through the library's graph. Returns the manifest."""
    from .answer_contract import NO_COVERAGE_CLAIM, attach_coverage_claim
    from .plan_expansion import budget_report, expansion_is_spent, headroom
    from .answer_contract import (assert_answer_is_reportable, build_spec_gap,
                                  strip_value_from_spec_insufficient)

    ctx = RunContext(spec=spec, chart=chart, plan=plan, coverage=coverage, threads=threads,
                     catalogue=catalogue, tracer=tracer, gate=gate, toolbox=toolbox)
    # The typed channel goes in with the chart tools, so it is declared, audited by
    # `wrap_tool_call`, and counted in the tool surface like everything else.
    tools = list(tools) + [make_revise_plan_tool(ctx, expansion_budget)]
    agent = build_agent(model=model, tools=tools, system_prompt=system_prompt, ctx=ctx,
                        backend=backend, max_model_calls=max_model_calls)

    crashed = False
    try:
        agent.invoke({"messages": [{"role": "user", "content": system_prompt}]},
                     config={"recursion_limit": recursion_limit_for(agent, max_model_calls)})
    except Exception as e:  # noqa: BLE001 -- a crashed run must still leave its trace
        crashed = True
        tracer.emit("runtime_error", severity="error", error=f"{type(e).__name__}: {e}")
    # A crash and a spent budget end the loop the same way and mean different things to the
    # person reading the manifest, so they are not folded into one word.
    termination = "RUNTIME_ERROR" if crashed else "BUDGET_EXHAUSTED"

    answer = dict(ctx.answer or {"status": "NO_ANSWER"})
    # THE THREE FIELDS `_n_finalize` SETS AND THIS FUNCTION DROPPED. Measured on ten real
    # charts: every run came out with `proof_basis: None` and `answer.evidence: []` while the
    # ledger held 3-10 items. Nothing raised, because the ledger copy lives at the manifest's
    # top level and `evals.py` reads that first. But `explain.py:358` selects on
    # `status == "FOUND" and e.evidence and e.proof_basis == "WITNESS"`, so every positive
    # this runtime produced would have been silently dropped from L5 — a whole arm's results
    # missing from the explanation layer with no error anywhere.
    if answer.get("status") == "FOUND":
        # Witness proof: one qualifying document settles it, which is what the FOUND branch of
        # the gate checks. It never claims the universe was searched, so no coverage ledger is
        # attached here.
        answer["proof_basis"] = "WITNESS"
        answer["witness_count"] = len(evidence.items)
        if not ctx.accepted:
            answer["proof_basis"] = "UNGATED"
            answer["route_to_human"] = True
            tracer.emit("ungated_positive", severity="warning", termination=termination)
    answer["evidence"] = evidence.to_list()
    spec_gap = None
    if answer.get("status") == "SPEC_INSUFFICIENT":
        spec_gap, remedy = build_spec_gap(spec, answer, reported_by="agent",
                                          gate_validated=ctx.accepted)
        answer.update({"spec_gap": spec_gap, "remedy_class": remedy,
                       "proof_basis": "NOT_APPLICABLE",
                       "coverage_note": ("no coverage claim is made — SPEC_INSUFFICIENT is a "
                                         "statement about the specification, not this chart")})
        strip_value_from_spec_insufficient(answer, tracer)
    for k in ("spec_section", "spec_quote", "uncovered_fields"):
        answer.pop(k, None)
    if answer.get("status") == "EVIDENCE_INSUFFICIENT":
        attach_coverage_claim(answer, gate_validated=ctx.accepted,
                              ledger=coverage.to_dict(), ungated_basis=termination)
    claim = ({"coverage_attested": answer["coverage_attested"]} if "coverage_attested" in answer
             else {"coverage_note": answer.get("coverage_note") or NO_COVERAGE_CLAIM})
    # Refuses an unearned ledger AND a gate-validated negative that arrives without one.
    assert_answer_is_reportable(answer)

    manifest = {
        "runtime": "deepagents-hooks", "patient_id": chart.patient_id,
        "spec_id": spec.spec_id, "spec_hash": spec.spec_hash,
        "answer": answer, "spec_gap": spec_gap, "gate_validated": ctx.accepted,
        "rejections": ctx.rejections, "rule_attribution": tracer.rule_attribution(),
        "plan": plan.to_dict(),
        "open_threads": {**threads.to_dict(), "marker_catalogue": catalogue.source},
        # UNDEFINED, NOT ZERO. There is no `revise_plan` tool yet, so no revision can be
        # proposed; reporting 0.0 would claim the agent had nothing to add on an axis that was
        # never measured. See the `after_model` comment for what building it requires.
        # Now measurable: `revise_plan` is a declared tool, so a request is an event and a
        # refusal is an event. Counted from what the tool recorded, not from an intention.
        "replan": {"n_requests": len(ctx.revisions),
                   "n_applied": sum(1 for r in ctx.revisions if r["applied"]),
                   "n_refused": sum(1 for r in ctx.revisions if not r["applied"]),
                   "revisions": ctx.revisions},
        # The budget the plan was priced for, in the manifest. "EXPANSION_BUDGET_EXHAUSTED"
        # without the numbers cannot be told from a chart that needed no widening, and the old
        # runtime recorded them.
        "expansion_budget": {
            **budget_report(plan, expansion_budget, source="priced_against_plan",
                            planner_terms=len(plan.keywords)),
            "exhausted": expansion_is_spent(plan, expansion_budget, terms_deferred=[]),
            "headroom": headroom(plan, expansion_budget)},
        "n_model_calls": ctx.n_model_calls, "max_model_calls": max_model_calls,
        "recursion_limit": recursion_limit_for(agent, max_model_calls),
        # Non-zero means the model tried to stop without answering and was sent back.
        "no_tool_call_recoveries": ctx.no_tool_call,
        # Non-null means the run was stopped for looping, not for lack of budget.
        "rejection_loop": ctx.stalled,
        "elapsed_s": elapsed_fn(), **claim, "evidence": evidence.to_list(),
    }
    (out_dir / f"{tracer.run_id}.manifest.json").write_text(json.dumps(manifest, indent=2))
    tracer.emit("run_end", accepted=ctx.accepted, rejections=len(ctx.rejections),
                n_model_calls=ctx.n_model_calls)
    return manifest


# ============================================================== the revision tool
# THE PORT LOST THIS AND SYN0002 DEADLOCKED ON IT.
#
# In `graph.py` the reflect node carried ONE typed channel that did two jobs: widen the
# retrieval plan, and settle open threads. Replacing reflect with hooks dropped both. The
# plan half I noticed. The thread half I did not, and it is the half that bites first: the
# gate refuses an answer while a thread is open and its `how_to_satisfy` reads "resolve_threads
# in your next reflection, or dismiss_threads with a reason" — instructions for tools that did
# not exist. On SYN0002 the marker was `outside facility`, which the catalogue documents as
# "the one marker you cannot resolve" because it names a document that is not in the record.
# The only legal move is to dismiss it. With no way to dismiss, the agent resubmitted until
# GraphRecursionError at 72. Telling a run what is wrong and not what to do about it is how a
# loop becomes a deadlock — the same sentence already in graph.py, re-earned.
#
# ONE tool, not two, because `CoveragePlan.apply_revision` already applies both halves under
# one budget and one monotonicity rule. Two tools would be two callers of that rule.

REVISE_PLAN_DESCRIPTION = """Widen the retrieval plan and settle open threads.

The plan may only GROW: add search terms, promote a document type toward more reading. A
request to remove a term or demote a type is refused — scope that can shrink is not a scope.

Threads: resolve_threads when you found where the deferred text was settled (say where);
dismiss_threads when it cannot be settled from this chart at all (say why). A thread naming an
outside facility or an outside institution CANNOT be resolved by reading, because the document
is not in this record — dismiss it with that reason. An open thread blocks submit_answer."""


def make_revise_plan_tool(ctx: RunContext, budget) -> StructuredTool:
    """The typed channel, as a declared tool so `wrap_tool_call` audits it like any other."""
    from .coverage_planner import PlanRevision

    def _revise(add_terms: list | None = None, promote_types: list | None = None,
                resolve_threads: list | None = None, dismiss_threads: list | None = None) -> str:
        def pairs(rows, second):
            out = []
            for r in rows or []:
                if isinstance(r, dict):
                    out.append((str(r.get("thread_id", "")), str(r.get(second, "")) or "unstated"))
                elif isinstance(r, (list, tuple)) and len(r) == 2:
                    out.append((str(r[0]), str(r[1])))
            return tuple(out)

        rev = PlanRevision(
            add_terms=tuple(str(t) for t in (add_terms or [])),
            promote_types=tuple((str(t), "read_all") for t in (promote_types or [])),
            resolve_threads=pairs(resolve_threads, "where_settled"),
            dismiss_threads=pairs(dismiss_threads, "reason"))
        outcome = ctx.plan.apply_revision(
            rev, step=ctx.n_model_calls, trigger="agent_request",
            observation="requested by the agent through revise_plan", budget=budget,
            threads=ctx.threads,
            n_docs_by_type={r["doc_type"]: r["count"] for r in ctx.chart.type_summary()},
            known_types=[r["doc_type"] for r in ctx.chart.type_summary()])
        ctx.revisions.append({"requested": rev.__dict__ if hasattr(rev, "__dict__")
                              else str(rev), "applied": bool(outcome.applied),
                              "refused": list(outcome.refused)})
        ctx.tracer.emit("plan_revision", runtime="deepagents-hooks",
                        applied=bool(outcome.applied), refused=list(outcome.refused),
                        refusal_class=getattr(outcome, "refusal_class", None))
        return json.dumps({
            "applied": bool(outcome.applied), "refused": list(outcome.refused),
            "unresolved_threads": [t.thread_id for t in ctx.threads.unresolved()],
            # The refusals are returned verbatim rather than summarised: an agent told only
            # "partly applied" re-sends the part that already landed, which is the loop this
            # channel exists to end.
            "note": ("the plan is re-rendered for you on the next turn; do not re-send what "
                     "was applied")}, default=str)[:6000]

    return StructuredTool.from_function(
        func=_revise, name="revise_plan", description=REVISE_PLAN_DESCRIPTION,
        args_schema={"type": "object", "properties": {
            "add_terms": {"type": "array", "items": {"type": "string"}},
            "promote_types": {"type": "array", "items": {"type": "string"}},
            "resolve_threads": {"type": "array", "items": {"type": "object", "properties": {
                "thread_id": {"type": "string"}, "where_settled": {"type": "string"}}}},
            "dismiss_threads": {"type": "array", "items": {"type": "object", "properties": {
                "thread_id": {"type": "string"}, "reason": {"type": "string"}}}}}})


def run_patient(*, spec, corpus, patient_id: str, out_dir, model, max_model_calls: int,
                seed: int = 1234, expansion_budget=None, run_id: str | None = None) -> dict:
    """Assemble the ledgers, tools and gate for one patient and run it.

    The assembly lived in a scratch harness while this runtime was being proven. It belongs
    here: a CLI that builds ledgers itself is a second place where the coverage ledger and the
    plan can be wired to different objects, and that asymmetry is what `assert_answer_is_
    reportable` exists to refuse.
    """
    import time

    from .coverage import CoverageLedger, ForcedSampler, strata_from_spec
    from .coverage_planner import OpenThreadLedger, load_marker_catalogue, plan_from_spec
    from .plan_expansion import price_expansion_budget
    from .answer_gate import gate_answer
    from .audit import _callbacks
    from .state import Budget, EvidenceLedger
    from .tools.toolbox import Toolbox
    from .trace import Tracer
    from deepagents.backends import StateBackend

    chart = corpus.chart(patient_id)
    docs, _ = chart.list_documents(limit=100_000)
    tracer = Tracer.create(out_dir, run_id)
    tracer.emit("run_start", patient_id=patient_id, runtime="deepagents-hooks",
                spec_id=spec.spec_id, spec_hash=spec.spec_hash, n_documents=len(docs))
    tracer.bind_spec(spec)

    evidence = EvidenceLedger()
    coverage = CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(seed))
    toolbox = Toolbox(chart, evidence, coverage,
                      known_doc_types=corpus.doc_type_vocabulary())
    plan = plan_from_spec(spec, chart)
    threads = OpenThreadLedger()
    markers = load_marker_catalogue()
    if markers.degraded:
        tracer.emit("marker_catalogue_degraded", severity="error", detail=markers.degraded)
    tools = chart_tools(toolbox, tracer)

    # THE GATE, CALLED DIRECTLY. `deep_runner` reaches it by constructing a whole
    # `ChartReviewAgent` and calling its `_gate`, which is three lines forwarding to
    # `gate_answer` — so a 1,197-line runtime was instantiated per patient to borrow one
    # function, and every front end that wanted the audit rule inherited a dependency on the
    # loop it was trying not to use. `gate_answer` takes exactly the objects assembled above.
    # One implementation of the rule, and now no holder around it.
    def gate(submitted: dict) -> dict:
        return gate_answer(spec, submitted, evidence=evidence, coverage=coverage, chart=chart,
                           tracer=tracer, threads=threads, plan=plan)

    t0 = time.time()
    return run_chart_review(
        spec=spec, chart=chart, toolbox=toolbox, coverage=coverage, evidence=evidence,
        plan=plan, threads=threads, catalogue=markers, tracer=tracer, gate=gate,
        model=model, tools=tools,
        system_prompt=spec.as_prompt_block() + "\n\n" + TASK.format(patient=patient_id),
        backend=StateBackend(), max_model_calls=max_model_calls, out_dir=out_dir,
        elapsed_fn=lambda: round(time.time() - t0, 1),
        # PRICED AGAINST THE PLAN, not a constant. The old runtime computed this from the
        # plan's own size — how many types it may promote, how many documents that opens —
        # and this one shipped `ExpansionBudget(40, 8, 40, 6)`, four numbers that fit no
        # particular chart. A 34-document chart and a 293-document chart were given the same
        # room to widen, which makes "the expansion budget was exhausted" a fact about the
        # constant rather than about the chart.
        expansion_budget=expansion_budget or price_expansion_budget(
            plan, {r["doc_type"]: r["count"] for r in chart.type_summary()},
            max_revisions=6, supplied=None, planner_terms=len(plan.keywords)))


# ===================================================== the chart tools, as LangChain tools
# MOVED OFF `deep_runner`, and stripped of two things it was doing that the middleware
# already does. `deep_runner._make_tools` wrapped every tool with its own copy of the plan
# refusal AND its own call to `triggers_from_tool_result`, so on this runtime both ran twice:
#
#   * The refusal was dead code — `wrap_tool_call` fires before the tool function, so the
#     middleware always refused first. Dead, but not harmless: its message read "Under this
#     runtime the plan cannot be widened — there is no typed reflection channel here", which
#     stopped being true the moment `revise_plan` existed. An agent told it cannot ask is an
#     agent that will not ask.
#   * The trigger detection genuinely ran twice. It did not show up as double-counted
#     triggers only because `OpenThreadLedger.open_thread` returns `already_open` the second
#     time and the loop `continue`s before emitting — every thread trigger was deduplicated
#     BY ACCIDENT, and a kind without a ledger behind it (ZERO_HIT_SEARCH) had no such luck.
#
# One owner each: the plan refusal and the trigger detector live in `AuditMiddleware`, and
# this function only adapts the toolbox's schemas.

def chart_tools(toolbox, tracer) -> list[StructuredTool]:
    """The toolbox's OpenAI-style schemas as LangChain tools. Dispatch and trace, nothing else."""
    tools = []
    for s in toolbox.schemas():
        fn_spec = s["function"]
        name = fn_spec["name"]

        def _call(_name=name, **kwargs):
            out, ms = toolbox.dispatch(_name, kwargs)
            tracer.tool(_name, kwargs, out, ok="error" not in (out or {}), ms=ms)
            return json.dumps(out, default=str)[:20000]

        tools.append(StructuredTool.from_function(
            func=_call, name=name, description=fn_spec.get("description", ""),
            args_schema=fn_spec.get("parameters") or {"type": "object", "properties": {}}))
    return tools
