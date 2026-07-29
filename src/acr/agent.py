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
    #: Set when the gate STOPPED ASKING because nothing further could satisfy it — an exclusion
    #: sample that turned up a hit, or an elusion bound frozen over its cap. The run ends and the
    #: abstention stands, but `gate_validated` must stay FALSE: the answer earned no coverage
    #: claim, it merely ran out of things that could earn one. Kept apart from `accepted` because
    #: `accepted` means "the gate said yes" everywhere else in this file, and conflating the two
    #: is how an unearned ledger gets stamped `GATE_VALIDATED`.
    coverage_unreachable: list = field(default_factory=list)
    declared: set[str] = field(default_factory=set)

    @property
    def gate_validated(self) -> bool:
        """DID THE GATE SAY YES, as opposed to having stopped asking.

        Five call sites needed this and each passed `ctx.accepted`, which was the same thing
        until `COVERAGE_UNREACHABLE` existed. Patching them one at a time is how the first
        attempt shipped with the manifest's own `gate_validated: true` sitting next to
        `negative_basis: COVERAGE_UNREACHABLE` -- `attach_coverage_claim` and
        `provenance_for_run` had been corrected and the manifest field had not. One property,
        one name that says which question it answers, and no site left to forget.
        """
        return bool(self.accepted and not self.coverage_unreachable)
    revisions: list = field(default_factory=list)
    rejection_fingerprints: collections.Counter = field(
        default_factory=collections.Counter)
    n_model_calls: int = 0
    no_tool_call: int = 0
    undeclared_tools: int = 0
    #: Repeats WITH THE LEDGERS FROZEN. Not a repeat count — see `_stalled` for why the
    #: earlier version of that cost a correct answer on a real chart.
    max_frozen_repeats: int = 3
    rejection_progress: dict = field(default_factory=dict)
    #: Terms a revision asked for that the budget could not pay for. Harvested, not
    #: discarded: the ask is evidence about the spec's declared list.
    terms_deferred: list = field(default_factory=list)
    #: Detected mechanically, not yet shown to the model. Drained by
    #: `wrap_model_call`; a trigger nobody is told about is a trigger that did not
    #: happen as far as the run is concerned.
    pending_triggers: list = field(default_factory=list)
    spend: Any = None
    spend_stopped: str | None = None
    expansion_stopped: str | None = None
    stalled: dict | None = None


    def outstanding_obligations(self) -> list[str]:
        """What this run still owes, asked of the context rather than of a runtime object.

        A method because it is a real question about a run — "is this finished?" — and every
        caller must get the same answer from the same two ledgers. When the old runtime had two
        ways to compute it they disagreed about whether a run had finished.
        """
        return outstanding_obligations(self.spec, self.coverage, self.plan, self.threads)


class AuditMiddleware(AgentMiddleware):
    """The audit plane, spread across the four hooks that structurally suit each rule.

    One class rather than four, because these rules share one `RunContext` and splitting them
    would put the ledger behind an interface that three of them mutate. What is deliberate is
    WHICH HOOK each rule sits in — that is the design, not the packaging.
    """

    def __init__(self, ctx: RunContext, budget=None):
        super().__init__()
        self.ctx = ctx
        #: The expansion budget, so the dead-end test can be asked once per turn.
        self._budget = budget

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
        parts = ["PLAN (current):\n" + self.ctx.plan.render(self._docs_by_type())]
        if (obligations := self._threads_block()):
            parts.append(obligations)
        if (observed := self._triggers_block()):
            parts.append(observed)
        sm = request.system_message
        content = ((sm.content if isinstance(sm.content, str) else str(sm.content)) + "\n\n"
                   if sm is not None else "") + "\n\n".join(parts)
        return handler(request.override(system_message=SystemMessage(content=content)))

    def _expansion_spent_with_obligations(self) -> str | None:
        """True only for the CONJUNCTION. Budget spent with everything discharged is a run that
        finished; obligations outstanding with budget left is a run that should keep going. Only
        both at once is a dead end, and a dead end has to be SAID."""
        from .plan_expansion import expansion_is_spent
        if self._budget is None:
            return None
        if not expansion_is_spent(self.ctx.plan, self._budget,
                                  terms_deferred=list(self.ctx.terms_deferred)):
            return None
        outstanding = self.ctx.outstanding_obligations()
        if not outstanding:
            return None
        return f"EXPANSION_BUDGET_EXHAUSTED with {len(outstanding)} obligation(s) outstanding"

    def _threads_block(self) -> str:
        """The open threads, each beside the call that settles it.

        AN AFFORDANCE NAMED A LONG WAY FROM THE OBSTACLE IT CLEARS DOES NOT EXIST IN PRACTICE.
        On the run this was written for, `resolve_threads` was in a schema at the bottom of the
        prompt for all eighteen reflections and the agent never once used it; it re-opened the
        same thread instead. So the way out is printed next to the thing it unblocks.
        """
        open_ = self.ctx.threads.unresolved() if self.ctx.threads else []
        if not open_:
            return ""
        rows = []
        for t in open_[:10]:
            rows.append(
                f"  - {t.thread_id}: {getattr(t, 'obligation', '') or 'unsettled'}\n"
                f'    settle it with revise_plan(resolve_threads=[{{"thread_id": '
                f'"{t.thread_id}", "where_settled": "..."}}])\n'
                f'    or, if this chart cannot settle it, revise_plan(dismiss_threads=[...]) '
                f"with a reason")
        return ("UNSETTLED THREADS — each of these blocks submit_answer until it is settled:\n"
                + "\n".join(rows))

    def _triggers_block(self) -> str:
        """What was detected mechanically since the last call. DRAINED, so it is said once.

        The detector ran and told only the trace. The agent — the only party that can act on an
        observation — was never shown it, so a mechanism that fires, records, and changes
        nothing looked from the outside exactly like a mechanism that works.
        """
        pending, self.ctx.pending_triggers = list(self.ctx.pending_triggers), []
        if not pending:
            return ""
        rows = []
        for t in pending[:10]:
            line = f"  - {t.kind}: {t.observation}"
            if getattr(t, "terms_proposed", None):
                line += f"\n    candidate terms: {list(t.terms_proposed)}"
            if getattr(t, "types_proposed", None):
                line += f"\n    candidate types: {list(t.types_proposed)}"
            rows.append(line)
        return ("OBSERVATIONS THAT REQUIRE A RESPONSE — detected mechanically since your last "
                "turn. You are not being asked whether anything happened; these happened. For "
                "each one either widen the plan with revise_plan or proceed knowing it stands:\n"
                + "\n".join(rows))

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
        # PRICE THE RUN AS IT GOES. This is the limit that is meant to bind — the absurd case,
        # tens of dollars on one patient — and it is the only one that should. Step caps and
        # loop brakes were cutting working runs short while the budget went untouched.
        self.ctx.spend.add(getattr(last, "usage_metadata", None))
        # EVERY TURN, not only after a refusal. A gate obligation the CURRENT plan structurally
        # cannot discharge is a deadlock whether or not the agent has tried to submit yet — and
        # on a run whose reads are refused OUT_OF_PLAN it never gets as far as submitting, which
        # is exactly the case the detector exists for.
        self._detect_deadlock()
        if (spent := self._expansion_spent_with_obligations()) and not self.ctx.accepted:
            # EXPANSION HAS A BUDGET AND RUNNING OUT OF IT IS A RESULT. The alternative — keep
            # looping to the call limit and emit whatever is in hand — is a silent truncation
            # dressed as an answer. This exits labelled, so the manifest carries a reason.
            self.ctx.expansion_stopped = spent
            self.ctx.tracer.emit("expansion_budget_exhausted", severity="warning",
                                 outstanding=self.ctx.outstanding_obligations()[:20],
                                 terms_deferred=list(self.ctx.terms_deferred),
                                 message=("the plan can no longer widen and the proof obligation "
                                          "is still not met. This is EVIDENCE_INSUFFICIENT and "
                                          "it is honest; it is not a pass and not a truncation"))
            return {"jump_to": "end"}
        if (why := self.ctx.spend.exceeded()) and not self.ctx.accepted:
            self.ctx.spend_stopped = why
            self.ctx.tracer.emit("cost_ceiling_reached", severity="error", why=why,
                                 spend=self.ctx.spend.report(),
                                 message=("stopped on cost, not on steps: whatever this run was "
                                          "doing, it was not worth this much"))
            return {"jump_to": "end"}
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
        self.ctx.undeclared_tools += 1
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
        types = sorted({b["doc_type"] for b in blocked})
        return {
            "error": "OUT_OF_PLAN", "blocked": blocked, "types": types,
            "message": ("The retrieval plan assigns these document types to `sample`: the "
                        "runtime's sampler draws from them and you may not open them directly. "
                        "This is NOT evidence that they hold nothing."),
            # THE WAY OUT, in the same message as the refusal. Dropping this key is how a
            # refusal becomes a deadlock: the agent is told it may not open the type and not
            # told that it can ask for the type to be promoted. The old runtime carried it and
            # this one had lost it — the same defect as the gate rejection that withheld
            # `how_to_satisfy`, which cost a run nine repeats of the wrong request.
            "how_to_proceed": (
                "if you have a reason to think this type bears on the answer, promote it with "
                f"revise_plan(promote_types=[{types[0]!r}]). The plan may only ever widen, so "
                "the promotion is recorded and permanent."),
        }

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
            out = {"accepted": True}
            if verdict.get("coverage_unreachable"):
                # Accepted so the run ENDS, not because the coverage obligation was met. Told to
                # the agent too: it submitted an answer it had been refused four times and needs
                # to know it stood for a different reason than the fifth attempt succeeding.
                self.ctx.coverage_unreachable = list(verdict["coverage_unreachable"])
                out["coverage_unreachable"] = self.ctx.coverage_unreachable
                out["note"] = ("accepted, and routed to a human. The proof obligation was NOT "
                               "met and no coverage claim is attached: the listed failures "
                               "cannot be discharged in this run, so asking you again would "
                               "have been asking for something that does not exist.")
        else:
            self.ctx.rejections.append(verdict)
            # A REJECTION IS AN EVENT, not a line in a tool log. `tracer.rejected` is what the
            # eval plane's `rejection_loop` detector and `rule_attribution` read; emitting only
            # `tracer.tool` left both blind, so a run that argued with the gate twenty times
            # looked identical to one that submitted once.
            self.ctx.tracer.rejected(verdict.get("why", ""), verdict.get("missing") or [],
                                     submitted)
            self._detect_deadlock()
            # SHAPED FOR THE AGENT, not the raw verdict. `missing` is the gate's word for its own
            # bookkeeping; `you_must_still` is an instruction. And `how_to_satisfy` is the only
            # place a thread rejection says `resolve_threads` — dropping it at this boundary is
            # what turned one run into nine repeats of the wrong request.
            out = {"accepted": False, "why": verdict.get("why"),
                   "you_must_still": verdict.get("missing") or []}
            if verdict.get("how_to_satisfy"):
                out["how_to_satisfy"] = verdict["how_to_satisfy"]
        return ToolMessage(content=json.dumps(out, default=str)[:8000],
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
        from .run_triggers import detect_gate_obligations  # noqa: F401
        for t in detect_gate_obligations(spec=self.ctx.spec, coverage=self.ctx.coverage,
                                         chart=self.ctx.chart, plan=self.ctx.plan,
                                         step=self.ctx.n_model_calls,
                                         tracer=self.ctx.tracer):
            self.ctx.pending_triggers.append(t)
            self.ctx.tracer.trigger(runtime="deepagents-hooks", **t.to_dict())

    def _stalled(self, submitted: dict, verdict: dict) -> dict | None:
        """Stop a run only when NOTHING IS MOVING — not when a refusal has repeated.

        THE FIRST VERSION OF THIS COST A CORRECT ANSWER. It counted repeats of
        (reason, value) and stopped at three. On ten real charts it fired on two, and one of
        them — a 293-document chart — was stopped into a WRONG histology it would otherwise
        have had more turns to correct. Measured against the same ten runs: nobody came near
        the 50-call ceiling (the busiest used 30), and a call costs $0.008, so a full 50-call
        run is about $0.40. The brake was the binding constraint on quality and the budget was
        not binding at all. That is backwards: a limit exists to stop the absurd case — tens of
        dollars, millions of tokens on one patient — not to cut a working run short.

        So the test is PROGRESS, not repetition. A refusal that repeats while the agent is
        still recording evidence or still successfully widening the plan is a run doing its
        job under a hard rule. Only a refusal that repeats with the ledgers frozen means the
        loop cannot advance, and that is cheap to detect exactly because the ledgers are ours.
        """
        fp = (str(verdict.get("why") or ""),
              json.dumps(submitted.get("value") or {}, sort_keys=True))
        # What "moving" means, measured off the ledgers rather than inferred from the text.
        progress = (len(self.ctx.toolbox.evidence.items) if self.ctx.toolbox else 0,
                    sum(1 for r in self.ctx.revisions if r["applied"]),
                    len(self.ctx.threads.threads) - len(self.ctx.threads.unresolved())
                    if self.ctx.threads else 0)
        prev = self.ctx.rejection_progress.get(fp)
        self.ctx.rejection_progress[fp] = progress
        if prev is None or progress != prev:
            # Either the first time this refusal has been seen, or something advanced since
            # the last one. Not a stall.
            self.ctx.rejection_fingerprints[fp] = 0
            return None
        self.ctx.rejection_fingerprints[fp] += 1
        n = self.ctx.rejection_fingerprints[fp]
        if n < self.ctx.max_frozen_repeats:
            return None
        self.ctx.stalled = {"frozen_repeats": n, "why": fp[0],
                            "value": submitted.get("value") or {}, "progress": list(progress)}
        self.ctx.tracer.emit("rejection_loop", severity="error", repeats=n, why=fp[0],
                             progress=list(progress),
                             message=("the same refusal fired on the same value with no new "
                                      "evidence, no applied revision and no settled thread in "
                                      "between; the loop cannot advance"))
        return {"accepted": False, "stop": True,
                "why": (f"REJECTION_LOOP: this answer has been refused {n} times for the same "
                        f"reason with nothing recorded in between ({fp[0]}). Resubmitting it "
                        f"cannot succeed."),
                "what_to_do": ("Change the VALUE, or record evidence that answers the rule, or "
                               "widen the plan with revise_plan. If no value the record "
                               "supports can satisfy this rule, submit SPEC_INSUFFICIENT and "
                               "name the answer_check at fault — that is a finding about the "
                               "specification and it is wanted.")}

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
        # `step=self.ctx.n_model_calls`, not the literal 0 this passed until 2026-07-28. Every
        # other emitter on this path already uses the counter (`_gate_answer`, `_record_reads`,
        # `revise_plan`), so a run's trigger records read `step: 0` while its
        # `terms_added_at_runtime` records read `step: 4` for the same exchange. A field that is
        # always the same number looks like data and is a constant: it made the six triggers on
        # SYN0002 unorderable against anything else in the manifest.
        for t in triggers_from_tool_result(name, args, payload, plan=self.ctx.plan,
                                           catalogue=self.ctx.catalogue,
                                           step=self.ctx.n_model_calls, quote=quote):
            if t.kind == "UNSETTLED_THREAD":
                m = self.ctx.catalogue.by_text().get(t.marker)
                req = self.ctx.threads.open_thread(
                    note_id=t.note_id, doc_type=t.doc_type, marker=t.marker,
                    obligation=(m.obligation if m else "unsettled"),
                    excerpt=t.observation, step=self.ctx.n_model_calls)
                # Branch on the typed status. `is None` was the old sentinel test and it
                # counted every short read as a new thread once `open_thread` began handing
                # back the existing one.
                if req.status != OPEN_REQUEST_OPENED:
                    continue
            self.ctx.pending_triggers.append(t)
            self.ctx.tracer.trigger(runtime="deepagents", **t.to_dict())


def build_agent(*, model, tools: list[StructuredTool], system_prompt: str, ctx: RunContext,
                backend, max_model_calls: int, summarization_model=None,
                keep_messages: int = 20, max_usd: float = 5.0, expansion_budget=None):
    """The graph. Every node comes from the library; every rule comes from a hook.

    Middleware order is composition order. `AuditMiddleware` is last so its `wrap_tool_call`
    sits closest to the tool and sees the call after any earlier middleware has rewritten it,
    and so its `wrap_model_call` appends the plan after `TodoListMiddleware` has added its own
    system-prompt block.
    """
    from deepagents.middleware.summarization import SummarizationMiddleware

    ctx.declared = {t.name for t in tools}
    if ctx.spend is None:
        from .spend import Spend
        ctx.spend = Spend(max_usd=max_usd, model=getattr(model, "model_name", "") or str(model))
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
        AuditMiddleware(ctx, budget=expansion_budget),
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

def outstanding_obligations(spec, coverage, plan, threads) -> list[str]:
    """Everything this run still owes: the gate's misses plus the unsettled threads.

    A free function so the finalize path and the deadlock detector ask the same question. When
    the old runtime had two ways to compute it they disagreed about whether a run had finished.
    """
    from .answer_gate import check_gate, check_threads
    try:
        missing = list(check_gate(spec, coverage, plan).missing)
    except Exception:      # noqa: BLE001 - a broken gate must not decide the answer's status
        missing = []
    return missing + check_threads(threads)


def downgrade_a_positive_that_owes_something(ans: dict, *, spec, coverage, plan, threads,
                                             gate_validated: bool, termination: str,
                                             tracer=None) -> None:
    """A run that stopped owing an obligation may not walk out with a positive.

    NOT PORTED WITH THE RUNTIME, and that is worse than it sounds. The hooks runtime was setting
    `proof_basis: UNGATED` and `route_to_human: True` on such an answer and shipping the VALUE
    intact — and `concordance.variables_from_answer` promotes every populated field to FOUND
    regardless of the answer's status. So the warning was carried by nothing downstream reads,
    which is precisely the defect this rule exists to prevent: a real run once ended
    `max_tokens (400000) reached` with a `truncated` thread open and emitted
    C341/8140/3 as established.

    THE THREE CONDITIONS, all required:
      * the answer is FOUND;
      * it never passed the gate — a gate-validated FOUND cleared the thread check and the
        decision rules, and nothing here has standing to second-guess it;
      * an obligation is genuinely outstanding. Owing nothing is a run that finished, and it
        keeps its UNGATED FOUND.

    THE VALUE GOES WITH THE STATUS. Left in place it is re-promoted field by field and the
    downgrade is cosmetic exactly where it matters. It is preserved verbatim under
    `withheld_value` so nothing is destroyed and a reviewer can see what the model wanted to
    say; it simply is not asserted by a run that did not finish.
    """
    if ans.get("status") != "FOUND" or gate_validated:
        return
    obligations = outstanding_obligations(spec, coverage, plan, threads)
    if not obligations:
        return
    ans["status"] = "EVIDENCE_INSUFFICIENT"
    ans["downgraded_from"] = "FOUND"
    ans["downgraded_because"] = (
        f"the run stopped ({termination}) with {len(obligations)} obligation(s) still "
        f"outstanding and never passed the answer gate; a positive asserted from that position "
        f"is a guess with a warning attached, and the honest status is an abstention")
    ans["outstanding_at_termination"] = obligations[:20]
    if ans.get("value"):
        ans["withheld_value"] = ans["value"]
        # Explicit nulls, not a dropped key: `variables_from_answer` reads a missing field as a
        # silence and an explicit null as the answer's own status, and the second is what this is.
        ans["value"] = {f.name: None for f in spec.fields}
    if tracer is not None:
        tracer.emit("positive_downgraded_at_termination", severity="warning",
                    termination=termination, outstanding=obligations[:20],
                    withheld_value=ans.get("withheld_value"),
                    message=("FOUND was emitted by a run that stopped with an obligation "
                             "outstanding and no gate pass; recorded as EVIDENCE_INSUFFICIENT "
                             "with the proposed value withheld"))


def run_chart_review(*, spec, chart, toolbox, coverage, evidence, plan, threads, catalogue,
                     tracer, gate, model, tools, system_prompt, backend, max_model_calls,
                     out_dir, elapsed_fn, expansion_budget, ctx_out=None,
                     max_usd: float = 5.0, seed_record: dict | None = None) -> dict:
    """One patient, one spec, through the library's graph. Returns the manifest."""
    from .answer_contract import NO_COVERAGE_CLAIM, attach_coverage_claim
    from .coverage_planner import MONOTONICITY_VS_LEDGER
    from .plan_expansion import budget_report, expansion_is_spent, headroom
    from .answer_contract import (assert_answer_is_reportable, build_spec_gap,
                                  strip_value_from_spec_insufficient)

    seed_record = seed_record or {"effective": None, "provenance": "unrecorded",
                                  "caller_supplied": None}
    ctx = RunContext(spec=spec, chart=chart, plan=plan, coverage=coverage, threads=threads,
                     catalogue=catalogue, tracer=tracer, gate=gate, toolbox=toolbox)
    # The typed channel goes in with the chart tools, so it is declared, audited by
    # `wrap_tool_call`, and counted in the tool surface like everything else.
    if ctx_out is not None:
        # A caller that wants the live ledgers gets THIS context, not a second assembly of its
        # own. Two places wiring the plan and the coverage ledger is the asymmetry the whole
        # audit layer exists to refuse, and a test harness is not exempt from it.
        ctx_out.append(ctx)
    tools = list(tools) + [make_revise_plan_tool(ctx, expansion_budget)]
    agent = build_agent(model=model, tools=tools, system_prompt=system_prompt, ctx=ctx,
                        backend=backend, max_model_calls=max_model_calls, max_usd=max_usd,
                        expansion_budget=expansion_budget)

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

    # `value` present even on NO_ANSWER: `variables_from_answer` reads a MISSING field as a
    # silence and an explicit null as the answer's own statement, and a run that produced nothing
    # is making a statement.
    answer = dict(ctx.answer or {"status": "NO_ANSWER",
                                 "value": {f.name: None for f in spec.fields}})
    # A SPEC THIS CHART CANNOT ANSWER AT ALL. `data_source: outside_notes` means the variable is
    # not derivable from the notes, so whatever the run produced is rewritten — and `cli_pipeline`
    # already tells the operator this happens ("every run will return SPEC_INSUFFICIENT /
    # WRONG_DATA_SOURCE by design"), which the runtime had stopped doing. The rewrite lands AFTER
    # the gate, so there is no rejection available and the value has to be taken here or it
    # smuggles itself out under a status that disclaims it.
    # UNCONDITIONAL, including over NO_ANSWER. `cli_pipeline` tells the operator "EVERY run will
    # return SPEC_INSUFFICIENT / WRONG_DATA_SOURCE by design", and it is right: the variable is
    # not derivable from notes, so what this particular run managed to do does not change the
    # answer. Gating the rewrite on "the run produced something" would let a run that produced
    # nothing report NO_ANSWER — a statement about this chart — for a spec whose problem is that
    # it is not about charts.
    forced_from = None
    if spec.data_source == "outside_notes":
        forced_from = answer.get("status")
        answer["status"] = "SPEC_INSUFFICIENT"
    # THE THREE FIELDS `_n_finalize` SETS AND THIS FUNCTION DROPPED. Measured on ten real
    # charts: every run came out with `proof_basis: None` and `answer.evidence: []` while the
    # ledger held 3-10 items. Nothing raised, because the ledger copy lives at the manifest's
    # top level and `evals.py` reads that first. But `explain.py:358` selects on
    # `status == "FOUND" and e.evidence and e.proof_basis == "WITNESS"`, so every positive
    # this runtime produced would have been silently dropped from L5 — a whole arm's results
    # missing from the explanation layer with no error anywhere.
    # BEFORE the FOUND labelling: a downgraded answer must not also be given WITNESS.
    downgrade_a_positive_that_owes_something(
        answer, spec=spec, coverage=coverage, plan=plan, threads=threads,
        gate_validated=ctx.gate_validated, termination=termination, tracer=tracer)
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
        spec_gap, remedy = build_spec_gap(
            spec, answer, reported_by=("runtime" if forced_from is not None else "agent"),
            gate_validated=ctx.gate_validated)
        if forced_from is not None:
            spec_gap["forced_over_status"] = forced_from
        answer.update({"spec_gap": spec_gap, "remedy_class": remedy,
                       "proof_basis": "NOT_APPLICABLE",
                       "coverage_note": ("no coverage claim is made — SPEC_INSUFFICIENT is a "
                                         "statement about the specification, not this chart")})
        strip_value_from_spec_insufficient(answer, tracer)
    for k in ("spec_section", "spec_quote", "uncovered_fields"):
        answer.pop(k, None)
    if answer.get("status") == "EVIDENCE_INSUFFICIENT":
        # `ctx.accepted and not ctx.coverage_unreachable`: the gate ends the run in both cases and
        # only one of them earned a coverage claim. Passing `ctx.accepted` alone would attach the
        # ledger and stamp GATE_VALIDATED over a run whose exclusion sampling was invalidated,
        # which is the precise thing `assert_answer_is_reportable` exists to refuse.
        attach_coverage_claim(
            answer, gate_validated=ctx.gate_validated,
            ledger=coverage.to_dict(),
            ungated_basis=("COVERAGE_UNREACHABLE" if ctx.coverage_unreachable else termination))
        if ctx.coverage_unreachable:
            answer["coverage_unreachable"] = list(ctx.coverage_unreachable)
            answer["coverage_note"] = (
                "no coverage claim is made — the proof obligation cannot be met on this chart: "
                + "; ".join(ctx.coverage_unreachable))
    claim = ({"coverage_attested": answer["coverage_attested"]} if "coverage_attested" in answer
             else {"coverage_note": answer.get("coverage_note") or NO_COVERAGE_CLAIM})
    # Refuses an unearned ledger AND a gate-validated negative that arrives without one.
    assert_answer_is_reportable(answer)

    manifest = {
        # IDENTITY. A manifest a reader cannot tie back to a patient, a spec version, a model
        # and a trace is a number with no provenance, and this runtime was emitting fifteen
        # fewer keys than the one it replaced — including the one below that decides whether a
        # gate pass may be reported as a validated answer at all.
        "run_id": tracer.run_id,
        "model": getattr(model, "model_name", "") or str(model),
        "spec_version": spec.spec_version,
        "trace": str(tracer.path),
        "runtime": "deepagents-hooks", "patient_id": chart.patient_id,
        "spec_id": spec.spec_id, "spec_hash": spec.spec_hash,
        "answer": answer, "spec_gap": spec_gap, "gate_validated": ctx.gate_validated,
        "rejections": ctx.rejections, "rule_attribution": tracer.rule_attribution(),
        "plan": plan.to_dict(),
        # THE DEVELOP-PLANE HARVEST. Lost in the port because this function assembles its own
        # manifest and never calls `run_manifest.build_manifest`, where the block lived. It is
        # the channel the improvement loop reads: what the spec DECLARED versus what a real
        # chart forced the run to add. Without it every run still rescued itself at runtime and
        # nothing recorded that the spec's list had been insufficient.
        "develop_plane_candidates": {
            "spec_declared_terms": list(plan.initial_keywords),
            "terms_added_at_runtime": list(plan.term_provenance),
            "types_promoted_at_runtime": list(plan.promotion_log),
            "refused_revisions": list(plan.refused_revisions),
            # A term the run ASKED FOR and the budget could not pay for is evidence about the
            # spec's list too, and partial application is exactly what stops it from landing in
            # `refused_revisions` — the revision was applied, only the tail of its term list was
            # not. It would otherwise disappear from the harvest.
            "terms_deferred_for_budget": list(ctx.terms_deferred),
            "what_this_is": ("candidate spec edits observed on a real chart. Score the spec's "
                             "list against spec_declared_terms, NEVER against the expanded "
                             "list — a runtime rescue folded back into the baseline erases the "
                             "evidence that the baseline was wrong"),
            "trace": str(tracer.path),
        },
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
        # DEGRADATION, with this runtime's OWN counters. The old loop counted plan/reflect
        # node fallbacks; those nodes are the library's now, so `deep_runner` reported
        # `degradation: None` and this runtime reported nothing at all — which reads as "clean"
        # to every consumer that checks it. These are the four ways THIS runtime can quietly do
        # less than it claims. Read them before any other number: a non-zero entry means the
        # behaviour a result is being cited for may not have been exercised.
        "degradation": {
            "no_tool_call_recoveries": ctx.no_tool_call,
            "undeclared_tool_calls": ctx.undeclared_tools,
            "rejection_loop_stopped": 1 if ctx.stalled else 0,
            "marker_catalogue_incomplete": 1 if catalogue.degraded else 0,
            "coverage_unreachable": 1 if ctx.coverage_unreachable else 0,
        },
        # WHAT IT COST AND WHAT IT WAS ALLOWED TO COST, in one place. `spend_stopped` is
        # non-null only when the ceiling is what ended the run.
        "spend": ctx.spend.report() if ctx.spend else None,
        "spend_stopped": ctx.spend_stopped,
        "expansion_stopped": ctx.expansion_stopped,
        # WHAT THE SPEC'S PROVENANCE PERMITS THIS RUN TO CLAIM, which is a separate question
        # from whether the gate passed. The gate proves the search was done; it cannot prove the
        # search terms were right. `reportable_as_validated` inside this block is the field a
        # downstream filter must read, never `gate_validated` alone — and this runtime was not
        # emitting the block at all, so every consumer had only the stronger-looking flag.
        # `and not ctx.coverage_unreachable` for the same reason as `attach_coverage_claim`: a
        # run the gate stopped asking is not a run the gate validated, and this block decides
        # what the answer may be REPORTED as.
        "provenance": spec.provenance_for_run(
            answer.get("value") or {}, str(answer.get("status") or ""),
            gate_validated=ctx.gate_validated),
        "negative_basis": answer.get("negative_basis"),
        #: Non-empty when the coverage bar could not be met on this chart at all. This is a
        #: finding about the STRATIFICATION, not about the patient, and it is what the develop
        #: plane should act on: the document type that produced the exclusion hit belongs in
        #: `may_mention` in the SPEC, not rescued per-run.
        "coverage_unreachable": list(ctx.coverage_unreachable),
        "steps": ctx.n_model_calls,
        "plan_revisions": len(ctx.revisions),
        "suspected_recognition_failures": len(getattr(coverage, "suspected_recognition_failures",
                                                      []) or []),
        "monotonicity_vs_ledger": MONOTONICITY_VS_LEDGER,
        # The seed and where it came from, always. A run whose seed was caller-supplied was
        # sampled with a number the operator chose, and a reader who cannot see that cannot tell
        # a reproduced draw from a shopped one.
        "sample_seed": seed_record["effective"],
        "seed_provenance": seed_record["provenance"],
        "seed_is_caller_supplied": seed_record["caller_supplied"],
        # The shape `evals.py` reads. `spend` prices the run; this is the token accounting.
        "usage": {"llm_calls": ctx.n_model_calls,
                  "prompt_tokens": ctx.spend.prompt if ctx.spend else 0,
                  "cached_tokens": ctx.spend.cached if ctx.spend else 0,
                  "completion_tokens": ctx.spend.completion if ctx.spend else 0,
                  "total_tokens": ((ctx.spend.prompt + ctx.spend.completion)
                                   if ctx.spend else 0)},
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

Promote to `search` to have a type's documents searched, or `read_all` to have every document
of it read. `search` is the smaller step; ask for it unless you need the whole type.

Threads: resolve_threads when you found where the deferred text was settled (say where);
dismiss_threads when it cannot be settled from this chart at all (say why). A thread naming an
outside facility or an outside institution CANNOT be resolved by reading, because the document
is not in this record — dismiss it with that reason. An open thread blocks submit_answer."""


def make_revise_plan_tool(ctx: RunContext, budget) -> StructuredTool:
    """The typed channel, as a declared tool so `wrap_tool_call` audits it like any other."""
    from .coverage_planner import REFUSED_THREAD_NOOP, PlanRevision

    def _revise(add_terms: list | None = None, promote_types: list | None = None,
                open_threads: list | None = None, resolve_threads: list | None = None,
                dismiss_threads: list | None = None) -> str:
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
            # THE TARGET IS THE AGENT'S TO NAME. Forcing `read_all` made every promotion the
            # largest possible one: a type the plan had put in `sample` jumped straight to
            # reading every document of it, when `search` is the smaller step and usually the
            # right one. Monotonicity does not require the biggest move, only that no move
            # shrinks — `apply_revision` still refuses a demotion.
            promote_types=tuple(
                ((str(t.get("type", "")), str(t.get("to", "search"))) if isinstance(t, dict)
                 else (str(t), "search"))
                for t in (promote_types or [])),
            # OPENING is part of the channel too. The runtime opens threads from markers on
            # its own, but an agent that notices a deferral the detector's catalogue does not
            # list has no other way to record it — and the THREAD_NOOP refusal class exists
            # precisely to answer a request to open one that is already open.
            open_threads=tuple(
                (str(t.get("note_id", "")), str(t.get("marker", "")), str(t.get("why", "")))
                if isinstance(t, dict) else tuple(t) for t in (open_threads or [])),
            resolve_threads=pairs(resolve_threads, "where_settled"),
            dismiss_threads=pairs(dismiss_threads, "reason"))
        # PARTIAL ON A BUDGET OVERRUN, all-or-nothing on monotonicity. `fit_terms_to_budget`
        # owns that distinction and this tool was skipping it, so a request for three terms with
        # room for two was refused whole — the agent never learned it could have had two, and
        # re-sent all three. Its docstring is the argument: nothing about the requested terms is
        # inadmissible, there is simply not enough allowance, and that is a different failure
        # from a revision that also tried to demote a type.
        from .plan_expansion import fit_terms_to_budget
        rev, deferred = fit_terms_to_budget(rev, ctx.plan, budget)
        ctx.terms_deferred.extend(t for t in deferred if t not in ctx.terms_deferred)
        outcome = ctx.plan.apply_revision(
            rev, step=ctx.n_model_calls, trigger="agent_request",
            observation="requested by the agent through revise_plan", budget=budget,
            threads=ctx.threads,
            n_docs_by_type={r["doc_type"]: r["count"] for r in ctx.chart.type_summary()},
            known_types=[r["doc_type"] for r in ctx.chart.type_summary()])
        ctx.revisions.append({"requested": rev.__dict__ if hasattr(rev, "__dict__")
                              else str(rev), "applied": bool(outcome.applied),
                              "refused": list(outcome.refused),
                              "terms_deferred": deferred})
        # THE THREAD HALF IS NOT COLLATERAL DAMAGE OF THE RETRIEVAL HALF. A revision that both
        # over-reached on types AND resolved the thread blocking the answer used to end the run
        # twice over — refused, and still thread-blocked, with the resolution nowhere. Thread
        # bookkeeping cannot violate monotonicity or widen what may be opened, so it does not
        # belong to the refusal. Re-sent through `apply_revision` rather than applied against
        # the ledger here, so a resolution's semantics stay defined in one place.
        salvaged = None
        if not outcome.applied and getattr(outcome, "refusal_class", None) != REFUSED_THREAD_NOOP:
            threads_only = PlanRevision(open_threads=rev.open_threads,
                                        resolve_threads=rev.resolve_threads,
                                        dismiss_threads=rev.dismiss_threads)
            if not threads_only.is_empty():
                salvaged = ctx.plan.apply_revision(
                    threads_only, step=ctx.n_model_calls, trigger="thread_work_salvage",
                    observation="the retrieval half of this revision was refused", budget=budget,
                    threads=ctx.threads,
                    n_docs_by_type={r["doc_type"]: r["count"] for r in ctx.chart.type_summary()},
                    known_types=[r["doc_type"] for r in ctx.chart.type_summary()])
                ctx.tracer.emit("thread_work_salvaged", severity="warning",
                                applied=salvaged.applied,
                                threads_opened=salvaged.threads_opened,
                                threads_resolved=salvaged.threads_resolved,
                                threads_dismissed=salvaged.threads_dismissed,
                                refused=list(salvaged.refused),
                                message=("the retrieval half was refused; its thread operations "
                                         "were re-applied on their own rather than discarded "
                                         "with it"))
        if deferred:
            ctx.tracer.emit("revision_partially_applied", severity="warning",
                            deferred_terms=list(deferred), applied=bool(outcome.applied),
                            message=("the term list did not fit the remaining expansion budget; "
                                     "what fitted was applied and the rest is named so the "
                                     "agent does not re-send it"))
        ctx.tracer.emit("plan_revision", runtime="deepagents-hooks",
                        applied=bool(outcome.applied), refused=list(outcome.refused),
                        refusal_class=getattr(outcome, "refusal_class", None))
        return json.dumps({
            "applied": bool(outcome.applied), "refused": list(outcome.refused),
            "thread_work_salvaged": bool(salvaged and salvaged.applied),
            "unresolved_threads": [t.thread_id for t in ctx.threads.unresolved()],
            # NAMED, not counted. An agent told only "partly applied" re-sends the part that
            # already landed, which is the loop this channel exists to end.
            "terms_deferred_for_budget": list(deferred),
            # The refusals are returned verbatim rather than summarised: an agent told only
            # "partly applied" re-sends the part that already landed, which is the loop this
            # channel exists to end.
            "note": ("the plan is re-rendered for you on the next turn; do not re-send what "
                     "was applied")}, default=str)[:6000]

    return StructuredTool.from_function(
        func=_revise, name="revise_plan", description=REVISE_PLAN_DESCRIPTION,
        args_schema={"type": "object", "properties": {
            "add_terms": {"type": "array", "items": {"type": "string"}},
            "promote_types": {"type": "array", "items": {"type": "object", "properties": {
                "type": {"type": "string"},
                "to": {"type": "string", "enum": ["search", "read_all"],
                       "description": "search is the smaller step; ask for read_all only when "
                                      "every document of the type must be read"}}}},
            "resolve_threads": {"type": "array", "items": {"type": "object", "properties": {
                "thread_id": {"type": "string"}, "where_settled": {"type": "string"}}}},
            "open_threads": {"type": "array", "items": {"type": "object", "properties": {
                "note_id": {"type": "string"}, "marker": {"type": "string"},
                "why": {"type": "string"}}}},
            "dismiss_threads": {"type": "array", "items": {"type": "object", "properties": {
                "thread_id": {"type": "string"}, "reason": {"type": "string"}}}}}})


def run_patient(*, spec, corpus, patient_id: str, out_dir, model, max_model_calls: int,
                seed: int = 1234, expansion_budget=None, run_id: str | None = None,
                ctx_out: list | None = None, max_usd: float = 5.0) -> dict:
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
    from .trace import Tracer, rule_citation_block
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
        # THE RULE IDENTIFIERS GO IN THE PROMPT, not only into the finalize question. The
        # self-report channel asks at submit time which decision rule was applied, and
        # `submit_answer` is reachable from any turn — an agent asked at the last moment to cite
        # identifiers it has never seen invents them, and `rule_attribution.self_reported` then
        # records invented ids as if they were a measurement of the agent's reasoning.
        system_prompt=(spec.as_prompt_block()
                       + (f"\n\n{cite}" if (cite := rule_citation_block(spec)) else "")
                       + "\n\n" + TASK.format(patient=patient_id)),
        backend=StateBackend(), max_model_calls=max_model_calls, out_dir=out_dir,
        elapsed_fn=lambda: round(time.time() - t0, 1), ctx_out=ctx_out,
        max_usd=max_usd,
        seed_record={"effective": seed, "provenance": "caller_supplied",
                     "caller_supplied": True},
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
