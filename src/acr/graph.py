"""The run plane's loop: plan -> act -> reflect -> (act | replan | finalize), and the single
route by which a run produces an answer.

Why a graph and not a plain ReAct loop
--------------------------------------
A plain loop decides "what next" inside the same generation that just read a document,
which makes the stopping decision an afterthought. Here `reflect` is a separate node with
one job: look at what has actually been gathered and rule CONTINUE / REPLAN / SUFFICIENT /
STUCK. Replanning is therefore a first-class, traceable event rather than a drift in the
model's internal monologue.

THE PLAN IS THE COVERAGE PLAN, AND THERE IS ONLY ONE
----------------------------------------------------
This module used to carry a second plan: a prose list of {id, goal, rationale} produced by
a PLAN_PROMPT, rendered into the message list, and read by no code anywhere. Two greps
settled it —

    grep 's["plan"]'                 src/acr/graph.py   -> nothing
    grep 'CoveragePlan|policy_for'   src/acr/graph.py   -> nothing

— so the plan the agent could revise governed nothing, and the plan that governed retrieval
was never consulted by the loop. That is why REPLAN fired 0 times in 291 actions across 37
runs. REPLAN and CONTINUE were mechanically identical: both appended text. A model asked
"does something learned change what should be done next?" about a goal like "find the
pathology report" correctly answers no — the GOAL never changes. What changes is the
RETRIEVAL SCOPE, and the retrieval scope was not in the plan.

So: the prose plan is deleted, `coverage_planner.CoveragePlan` is built once up front, it is
rendered into the agent's messages, it is ENFORCED in `_n_act` (a `sample` type may not be
opened at all unless the runtime's sampler drew it), and it is what reflection revises —
monotonically, in a typed object the runtime applies. REPLAN is no longer a verdict the
model may pick; it is recorded by the runtime when, and only when, a revision actually
changed what the agent may open or must search.

WHAT THIS MODULE DOES NOT DECIDE
--------------------------------
The loop drives four collaborators and rules on none of their questions, because each of
them has to give the same answer to the other two front ends (`mcp_server`, `deep_runner`)
as it gives here:

  * `answer_gate`      whether a submitted answer may stand. THERE IS EXACTLY ONE GATE.
                       `_gate` and `_check_gate` below are thin delegates and nothing in
                       this file recomputes a verdict — a second copy of the gate judgement
                       grew here once before and had to be removed.
  * `answer_contract`  what an answer owes at emission, asserted in `_n_finalize`.
  * `run_triggers`     which mechanical observations oblige the next reflection.
  * `plan_expansion`   what a monotone widening costs and when widening is over.
  * `run_manifest`     the record the finished run leaves behind.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from .answer_contract import (SPEC_SECTIONS, assert_answer_is_reportable, build_spec_gap,
                              strip_value_from_spec_insufficient)
from .answer_gate import check_gate, check_threads, gate_answer, keyword_hits_among_drawn
from .corpus import PatientChart
from .llm import LLMClient, extract_json
from .spec import ExtractionSpec
from .coverage import (SEED_CALLER, SEED_DERIVED, CoverageLedger, ForcedSampler, GateResult,
                       derive_sample_seed, strata_from_spec)
from .coverage_planner import (REFUSED_BUDGET, REFUSED_THREAD_NOOP, TRIGGERS, CoveragePlan,
                               ExpansionBudget, MONOTONICITY_VS_LEDGER, OpenThreadLedger,
                               PlanRevision, RevisionOutcome, Trigger, documents_by_type,
                               load_marker_catalogue, plan_coverage, plan_from_spec)
from .plan_expansion import (budget_report, expansion_is_spent, fit_terms_to_budget,
                             headroom, install_plan_block, price_expansion_budget)
from .run_manifest import ExpansionRecord, RunCounters, SeedRecord, build_manifest
from .run_triggers import detect_from_tool_result, detect_gate_obligations
from .state import Budget, EvidenceLedger, RunState
from .tools import Toolbox
from .trace import Tracer, rule_citation_block

#: The routing vocabulary of `_after_reflect`. REPLAN is still in it and still routes to the
#: plan node, but it is now DERIVED BY THE RUNTIME from an applied revision rather than
#: chosen by the model — see `_n_reflect`. That is the whole fix: when REPLAN was something a
#: supervisor could assert, asserting it changed nothing, so it was never asserted.
VERDICTS = {"CONTINUE", "REPLAN", "SUFFICIENT", "STUCK"}

#: What the model may return. REPLAN is absent on purpose. A model that returns it anyway is
#: recorded and read as CONTINUE: the revision object, not the word, is what replans.
MODEL_VERDICTS = {"CONTINUE", "SUFFICIENT", "STUCK"}

#: Tool calls the retrieval plan is allowed to refuse. Reading only. Searching a `sample`
#: type stays legal — the plan says what must be READ, and a cheap metadata-level search that
#: turns up a hit is exactly the observation that should trigger a promotion.
PLAN_GOVERNED_TOOLS = ("read_document", "read_section", "read_documents_batch")


SYSTEM = """You are a cancer-registry abstractor reviewing one patient's chart.

You work exactly the way a careful human abstractor does: first see what documents exist \
and when, then narrow by document type and date, then search, then read only what matters, \
and quote what you find.

Rules that are not negotiable:
- Ground every assertion in a recorded quote. Use record_evidence before you answer.
- Never infer a finding from a document type that cannot establish it. Imaging does not \
establish histology. A restatement in a consult note is weaker than the primary report it \
claims to summarise; when they disagree, cite both and prefer the primary.
- Absence of a mention is not evidence of absence until you have actually looked. The \
specification tells you what "looked" means for this variable.
- If the specification does not cover the case, say SPEC_INSUFFICIENT. If the specification \
is clear but the chart lacks the evidence, say EVIDENCE_INSUFFICIENT. These are different \
answers and choosing the wrong one is an error.
- SPEC_INSUFFICIENT is a report about the SPECIFICATION, so it must name what is wrong with \
it: set spec_section to the part at fault, say in your own words what it fails to cover, \
and quote the sentence you mean if one exists. It must NOT carry a value — if you can code \
a field, answer FOUND for it and prove it.
- The RETRIEVAL PLAN below is binding. Document types it assigns to `sample` are drawn by \
the runtime's sampler and you may not open them yourself; a read of one is refused. That is \
not evidence the type holds nothing. If you find a reason to think it bears on the answer, \
widen the plan at the next supervisor step — the plan may only ever widen, never narrow.
- A document that DEFERS ITS OWN CONCLUSION — "pending", "see addendum", "correlate \
clinically", an outside facility, or a read that came back truncated — has opened a thread, \
and an open thread blocks your answer. Chase it: page to the end of the same document, then \
its section list, then later documents. A deferred conclusion is not evidence for the \
conclusion; it is an instruction about where to look next.

Call one tool at a time and read its result before deciding the next move."""

# NOTE: there is no PLAN_PROMPT any more. The prose plan of {id, goal, rationale} that used
# to live here was read by no code — it was rendered into the message list and that was all.
# The plan is now `coverage_planner.CoveragePlan`, built once up front by
# `coverage_planner.plan_coverage` (or, when no model is available or the planner degrades,
# derived from the spec's own strata by `plan_from_spec`) and rendered by `plan.render()`.
# Keeping both would have left the bug exactly where it was: two plans, one of which matters.

REFLECT_PROMPT = """You are supervising a chart review in progress. Judge only what has \
actually been gathered, and revise the retrieval plan if the observations below say it was \
wrong.

SPECIFICATION QUESTION: {question}

PROOF OBLIGATION FOR A NEGATIVE ANSWER:
{obligation}

THE PLAN (there is only one, and it governs what may be opened):
{plan}

UNSETTLED THREADS — an OPEN one blocks your answer, and the only two ways past it are \
`resolve_threads` (say where it was settled) and `dismiss_threads` (say why it does not bear \
on the answer). Both are recorded. Opening a thread ADDS an obligation; re-opening one that \
is already open discharges nothing and is refused as a no-op. If you have already chased a \
thread — paged to the end, read the section list, looked at later documents — then RESOLVE \
it in this revision, because the work does not count until the ledger is told:
{threads}

OBSERVATIONS THAT REQUIRE A RESPONSE — these were detected mechanically since the last \
reflection. You are not being asked whether anything happened; these happened. For each one \
either widen the plan or say in `reason` why widening is not warranted:
{triggers}

EVIDENCE RECORDED SO FAR:
{evidence}

COVERAGE SO FAR:
{coverage}

STEPS USED: {step}/{max_steps}
EXPANSION REMAINING: {budget}

HOW THE PLAN MAY BE REVISED — this is enforced in code, not requested:
- ADD a search term. Anything you add becomes a search the gate then REQUIRES to have run.
- PROMOTE a document type toward more reading: sample -> search -> read_all.
- OPEN a thread, or RESOLVE one (say where it was settled) or DISMISS one (say why it does \
not bear on the answer). An unsettled thread blocks submission, and only the last two settle \
one — a second OPEN of a thread already open is a no-op and is refused as one.
- You may NEVER remove a term, demote a type, or drop a type from the plan. A revision that \
is not a superset of the current plan is REFUSED WHOLE and recorded as refused.

Rule one verdict:
- SUFFICIENT  the recorded evidence already answers the question, or the proof obligation \
for a negative has been met and nothing was found.
- CONTINUE    keep going.
- STUCK       further search is futile; the honest answer is an abstention. Choose this when \
the expansion budget is spent and obligations are still outstanding.
There is no REPLAN verdict. The runtime records a replan when — and only when — your \
revision actually changes what may be opened or what must be searched.

Reply with JSON only:
{{"verdict":"SUFFICIENT|CONTINUE|STUCK","reason":"one or two sentences",
  "revision":{{"add_terms":["..."],
              "promote_types":[{{"type":"<exact type string>","to":"search|read_all"}}],
              "open_threads":[{{"note_id":"...","marker":"...","why":"..."}}],
              "resolve_threads":[{{"thread_id":"...","how":"..."}}],
              "dismiss_threads":[{{"thread_id":"...","reason":"..."}}]}}}}
Send an empty revision object when nothing needs to widen."""

FINALIZE_PROMPT = """{spec_block}

{rule_citations}

You are writing the final answer for patient {patient_id}.

You may use ONLY the evidence below. Anything not in this ledger does not exist for the \
purposes of this answer.

EVIDENCE LEDGER:
{evidence}

COVERAGE ACHIEVED:
{coverage}

{gate_note}

Apply the decision rules to the evidence and produce the answer.

Reply with JSON only:
{{"status":"FOUND|EVIDENCE_INSUFFICIENT|SPEC_INSUFFICIENT",
  "value":{{{value_keys}}},
  "reasoning":"which rules you applied to which evidence",
  "rules_applied":["the identifiers above for the rules you actually used"],
  "evidence_ids":["E1","E2"]}}

If and only if the status is SPEC_INSUFFICIENT, add these and leave every value null — the \
specification cannot both fail to cover the case and decide it:
  "spec_section": one of {spec_sections},
  "spec_quote": the sentence you mean, verbatim from the specification above, or omit it if \
no such sentence exists,
  "uncovered_fields": the output fields it does not cover, or omit for the whole answer"""


class ChartReviewAgent:
    def __init__(
        self,
        spec: ExtractionSpec,
        llm: LLMClient,
        *,
        budget: Budget | None = None,
        reflect_every: int = 3,
        out_dir: str | Path = "runs",
        sample_seed: int | None = None,
        expansion_budget: ExpansionBudget | None = None,
    ):
        self.spec = spec
        self.llm = llm
        self.budget = budget or Budget()
        self.reflect_every = reflect_every
        self.out_dir = Path(out_dir)
        # None does NOT mean "unbounded" and does not mean "some default". It means "price it
        # against the plan", which `plan_expansion.price_expansion_budget` does with no
        # literal in sight: each cap is "no more than the commitment the plan was already
        # priced at". A caller may override, and the manifest records which of the two
        # happened.
        self.expansion_budget = expansion_budget
        self.expansion_budget_source = ("caller_supplied" if expansion_budget is not None
                                        else "priced_against_plan")
        # Recorded in the trace so an audit can confirm which documents were drawn, and so
        # a run replays deterministically. Two ablation arms must share it to be comparable.
        #
        # None does NOT mean "draw one at random" any more. It means "derive it from
        # (patient, spec_id)", the way `mcp_server` always has — `run()` fills it in once the
        # patient is known. A caller-supplied seed is honoured, but the manifest says so:
        # a seed the caller chose is a seed the caller could have chosen again.
        self.sample_seed = sample_seed
        self.seed_provenance = SEED_CALLER if sample_seed is not None else SEED_DERIVED
        # How many `term_provenance` rows existed before the agent reflected once — i.e. the
        # up-front planner's own proposals. The expansion budget is priced in REFLECTION
        # terms and counted in ALL rows, so the two are reconciled here and nowhere else;
        # see `plan_expansion.price_expansion_budget`. Set in `run()`, defined here because
        # callers that borrow the object without running it (deep_runner, tests) read the
        # budget helpers.
        self._planner_terms = 0
        #: Terms a revision asked for and the budget could not pay for. Kept because partial
        #: application means a term overrun no longer records a refusal, and "the agent hit
        #: the cap" must remain observable — see `plan_expansion.expansion_is_spent`.
        self._terms_deferred: list[str] = []
        # Defined before `run()` for the same reason: a borrowed agent must be able to report
        # zero rather than raise AttributeError at the one moment a reader wants the number.
        self._counters = RunCounters()
        self._graph = self._build()

    # ------------------------------------------------------------------ graph
    def _build(self):
        g = StateGraph(RunState)
        g.add_node("plan", self._n_plan)
        g.add_node("act", self._n_act)
        g.add_node("reflect", self._n_reflect)
        g.add_node("finalize", self._n_finalize)
        g.add_edge(START, "plan")
        g.add_edge("plan", "act")
        g.add_conditional_edges("act", self._after_act, {"reflect": "reflect", "finalize": "finalize"})
        g.add_conditional_edges("reflect", self._after_reflect,
                                {"act": "act", "plan": "plan", "finalize": "finalize"})
        g.add_edge("finalize", END)
        return g.compile()

    # ------------------------------------------------------------------ the one plan
    def _build_plan(self) -> CoveragePlan:
        """Build the retrieval plan ONCE, before the loop, and record where it came from.

        Up front and frozen-at-birth is the property that makes it auditable: the reviewing
        agent consumes a plan it did not author, and may only widen it. Two sources, and the
        difference is recorded rather than smoothed over — a planner guess must never be
        readable as a curated site binding, which is what `CoveragePlan.source` is for.
        """
        if self.llm is None:
            return plan_from_spec(self.spec, self.chart)
        try:
            p = plan_coverage(self.spec, self.chart, self.llm)
        except Exception as e:      # noqa: BLE001 - a broken planner must not lose the run
            self._counters.plan_fallbacks += 1
            self.tracer.emit("plan_fallback_used", severity="error",
                             error=f"{type(e).__name__}: {e}",
                             message=("the coverage planner raised; falling back to the "
                                      "spec's own strata. The run is NOT exercising the "
                                      "planner and no conclusion about it may be drawn"))
            return plan_from_spec(self.spec, self.chart)
        if not (p.read_all or p.search or p.sample):
            # The planner returned nothing assignable — the reasoning-channel trap that used
            # to be laundered into a one-line prose goal. Loud, countable, and it falls back
            # to a real plan (the spec's strata) rather than to a generic sentence.
            self._counters.plan_fallbacks += 1
            self.tracer.emit("plan_fallback_used", severity="error",
                             message=("the coverage planner produced no usable assignment; "
                                      "falling back to the spec's declared strata"))
            return plan_from_spec(self.spec, self.chart)
        return p

    # ------------------------------------------------------------------ nodes
    def _n_plan(self, s: RunState) -> dict:
        """Render the plan into the working messages. Entered once at START, and again after
        every APPLIED revision — which is what makes a widened scope actually reach the model
        rather than sitting in a Python object nobody shows it."""
        rev = s.get("plan_revisions", 0)
        self.tracer.plan(self.plan.to_dict(), rev)
        # The rule identifiers go in the working prompt, not only in the finalize prompt,
        # because the self-report is collected at submit_answer — and `submit_answer` is
        # reachable from any act step. An agent asked at the last moment to cite identifiers
        # it has never seen will invent them, and we would be measuring our own prompt.
        cite = rule_citation_block(self.spec)
        msgs = s.get("messages") or [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self.spec.as_prompt_block()
             + (f"\n\n{cite}" if cite else "")
             + f"\n\nPATIENT: {s['patient_id']}\nBegin. Work the plan; call one tool at a time."},
        ]
        header = ("PLAN (revision %d — the scope was widened; the additions are marked):\n"
                  % rev if rev else "PLAN:\n")
        msgs = install_plan_block(msgs, header + self.plan.render(self._docs_by_type))
        return {"plan": self.plan.to_dict(), "plan_revisions": rev + 1, "messages": msgs}

    def _n_act(self, s: RunState) -> dict:
        msgs = list(s["messages"])
        r = self.llm.chat(msgs, tools=self.toolbox.schemas())
        self.tracer.llm("act", r.content, [c["name"] for c in r.tool_calls], {"total": r.total_tokens})

        if not r.tool_calls:
            msgs.append({"role": "assistant", "content": r.content or ""})
            self._counters.act_no_tool_call += 1
            self.tracer.emit("act_no_tool_call", severity="warning",
                             content_chars=len(r.content or ""),
                             used_reasoning_channel=r.used_reasoning_channel,
                             message="act step produced neither a tool call nor usable text")
            msgs.append({"role": "user", "content":
                         "Continue by calling a tool. If you are ready, call submit_answer."})
            return {"messages": msgs, "step": s.get("step", 0) + 1}

        msgs.append({"role": "assistant", "content": r.content or "",
                     "tool_calls": [{"id": c["id"], "type": "function",
                                     "function": {"name": c["name"],
                                                  "arguments": json.dumps(c["arguments"])}}
                                    for c in r.tool_calls]})
        rejections = list(s.get("rejections", []))
        done = False
        gate_validated = bool(s.get("gate_validated"))
        answer = s.get("answer") or {}

        for c in r.tool_calls:
            refusal = self._plan_refusal(c["name"], c["arguments"] or {})
            if refusal is not None:
                # THE PLAN GOVERNS WHAT MAY BE OPENED. Not as advice in a prompt — as a
                # refusal at dispatch. A plan the agent can ignore is the prose plan again.
                out, ms = refusal, 0.0
                self._counters.plan_refused_opens += 1
                self.tracer.emit("plan_refused_open", severity="warning", tool=c["name"],
                                 blocked=refusal["blocked"])
            else:
                out, ms = self.toolbox.dispatch(c["name"], c["arguments"])
            self.tracer.tool(c["name"], c["arguments"], out, ok="error" not in out, ms=ms)
            # BEFORE detection, always. `_record_reads` is what makes the deterministic
            # settlement route reachable, and it has to run first for both of its halves: a
            # read that completes a document must settle the thread before the same result is
            # scanned, and a window read of an already-complete document must not be able to
            # open one at all.
            self._record_reads(c["name"], out, s.get("step", 0))
            self._detect_triggers(c["name"], c["arguments"] or {}, out, s.get("step", 0))

            if c["name"] == "submit_answer":
                verdict = self._gate(self.toolbox.submitted or {})
                if verdict["accepted"]:
                    answer = self.toolbox.submitted or {}
                    done = True
                    gate_validated = True          # the ONLY place this becomes true
                    if self._counters.steps_to_gate_pass is None:
                        # Answers existence and cost in one run: whether the obligation is
                        # reachable at all, and what it costs per patient per variable. If
                        # this lands at 23, a 20-step budget was short by 3 and no second
                        # run is needed to learn that.
                        self._counters.steps_to_gate_pass = s.get("step", 0) + 1
                        self.tracer.emit("gate_passed",
                                         step=self._counters.steps_to_gate_pass,
                                         rejections_before=len(rejections))
                    out = {"accepted": True}
                else:
                    rejections.append(verdict)
                    self.tracer.rejected(verdict["why"], verdict["missing"], self.toolbox.submitted)
                    out = {"accepted": False, "why": verdict["why"],
                           "you_must_still": verdict["missing"]}
                    if verdict.get("how_to_satisfy"):
                        # THE GATE WROTE THE WAY OUT AND THIS LINE USED TO THROW IT AWAY. The
                        # thread rejection's `how_to_satisfy` is the only place the word
                        # `resolve_threads` appears in a rejection, and it was dropped at the
                        # boundary: the agent was told "this thread blocks you" without being
                        # told in the same breath how to settle it. On SYN0001 that rejection
                        # was answered with nine more requests to OPEN the same thread and
                        # zero requests to resolve it. Telling a run what is wrong and not
                        # what to do about it is how a loop becomes a deadlock.
                        out["how_to_satisfy"] = verdict["how_to_satisfy"]

            msgs.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                         "content": json.dumps(out, ensure_ascii=False, default=str)[:6000]})

        return {"messages": msgs, "step": s.get("step", 0) + 1, "done": done,
                "gate_validated": gate_validated,
                "answer": answer, "rejections": rejections,
                "evidence": self.evidence.to_list(), "coverage": self.coverage.to_dict()}

    # -------------------------------------------------------- plan enforcement + triggers
    def _plan_refusal(self, name: str, args: dict) -> dict | None:
        """Refuse a read of a `sample` type the runtime did not draw. Returns None to allow.

        The escape hatch is deliberate and is the whole design: the refusal names the type
        and tells the agent to promote it in the next reflection. An agent that has found a
        reason to open a sampled type gets to open it — by widening the plan on the record,
        which is monotone and auditable, rather than by quietly wandering out of scope.
        """
        if name not in PLAN_GOVERNED_TOOLS:
            return None
        ids = ([args.get("note_id")] if name != "read_documents_batch"
               else list(args.get("note_ids") or []))
        drawn = {n for v in self.coverage.drawn.values() for n in v}
        blocked: list[dict] = []
        for nid in ids:
            meta = self.chart._docs.get(str(nid))
            if meta is None or str(nid) in drawn:
                # Unknown ids fall through to the toolbox, which distinguishes a fabricated
                # note_id from absence — a distinction this guard must not blur. A drawn
                # document is the RUNTIME's choice, never the agent's, so it is always open.
                continue
            if not self.plan.may_open(meta.doc_type):
                blocked.append({"note_id": str(nid), "doc_type": meta.doc_type})
        if not blocked:
            return None
        types = sorted({b["doc_type"] for b in blocked})
        return {
            "error": "OUT_OF_PLAN",
            "blocked": blocked,
            "message": ("The retrieval plan assigns these document types to `sample`: the "
                        "runtime's sampler draws from them and you may not open them "
                        "directly. This is NOT evidence that they hold nothing."),
            "types": types,
            "how_to_proceed": ("if you have a reason to think this type bears on the answer, "
                               "promote it at the next reflection with "
                               f"promote_types=[{{'type': {types[0]!r}, 'to': 'search'}}]. "
                               "The plan may only ever widen, so the promotion is recorded "
                               "and permanent."),
        }

    def _record_reads(self, name: str, out: dict, step: int) -> None:
        """Hand every read's extent to the thread ledger, and trace what it settled.

        THE ROUTE FROM "I READ TO THE END OF IT" TO "THE THREAD IS SETTLED". It did not exist:
        on SYN0001 the agent paged to the end of the truncated report, listed its sections and
        read FINAL DIAGNOSIS, and the ledger learned nothing from any of it because nothing
        was telling the ledger what had been read. This is that wire, and it is deliberately
        mechanical — no model is asked whether the document is finished, because the runtime
        computed `truncated` from the character counts in the first place and can compute the
        complement just as well. See `OpenThreadLedger.note_read`, and
        `MECHANICALLY_DISCHARGEABLE_MARKERS` for why `truncated` is the only marker this may
        ever settle.
        """
        threads = getattr(self, "threads", None)
        if threads is None or not isinstance(out, dict) or out.get("error"):
            return
        reads: list[tuple[str, int, int, int | None]] = []
        if name == "read_document" and out.get("note_id") and "returned_chars" in out:
            reads.append((str(out["note_id"]), int(out.get("offset") or 0),
                          int(out.get("returned_chars") or 0), out.get("total_chars")))
        elif name == "read_documents_batch":
            for d in (out.get("documents") or []):
                # The batch reader always starts at zero and returns `text`; it reports
                # `total_chars` so a short excerpt of a long note stays visibly short.
                reads.append((str(d.get("note_id", "")), 0, len(str(d.get("text", ""))),
                              d.get("total_chars")))
        elif name == "read_section" and out.get("note_id") and "start" in out and "end" in out:
            # A named section carries TRUE offsets and no document length, so it contributes
            # coverage and can never on its own prove the document is complete. That is the
            # honest accounting: reading FINAL DIAGNOSIS tells you nothing about what sits
            # after it.
            reads.append((str(out["note_id"]), int(out.get("start") or 0),
                          max(0, int(out.get("end") or 0) - int(out.get("start") or 0)), None))
        for note_id, offset, returned, total in reads:
            settled = threads.note_read(note_id, offset=offset, returned_chars=returned,
                                        total_chars=total, step=step)
            if settled:
                self.tracer.emit(
                    "threads_settled_by_read", thread_ids=settled, note_id=note_id,
                    total_chars=threads.doc_length.get(note_id),
                    message=("the document has now been returned in full by reads in this "
                             "run, so its `truncated` thread is discharged deterministically "
                             "— the runtime owns both sides of that predicate and does not "
                             "need to ask"))

    def _detect_triggers(self, name: str, args: dict, out: dict, step: int) -> None:
        """Queue whatever `run_triggers` read off this tool result. No model is asked here."""
        for t in detect_from_tool_result(name, args, out, step=step, plan=self.plan,
                                         markers=self.markers, threads=self.threads):
            self._record_trigger(t)

    def _record_trigger(self, t: Trigger) -> None:
        """Count one detected trigger, queue it for the next reflection, and trace it.

        `tracer.trigger` and not `tracer.emit("trigger", **t.to_dict())`: the trigger's own
        `kind` collided with the trace envelope's `kind` and every run that detected anything
        died on the spot with a TypeError. One emitter, used by both detection paths, so the
        two cannot drift into two event shapes.
        """
        self._pending_triggers.append(t)
        self._trigger_counts[t.kind] = self._trigger_counts.get(t.kind, 0) + 1
        self.tracer.trigger(**t.to_dict())

    def _gate_triggers(self, step: int) -> None:
        """Queue the obligations the CURRENT plan structurally cannot discharge; see
        `run_triggers.detect_gate_obligations` for why a deadlock is not a rejection."""
        # The tracer is not optional there: the detector swallows its own exceptions, so
        # without a channel to say so a fourth trigger that has stopped working is
        # indistinguishable from a run with no deadlock to report.
        for t in detect_gate_obligations(spec=self.spec, coverage=self.coverage,
                                         chart=self.chart, plan=self.plan, step=step,
                                         tracer=self.tracer):
            self._record_trigger(t)

    # ------------------------------------------------------------- the expansion budget
    # Thin delegates to `plan_expansion`, which owns the arithmetic. They stay as methods
    # because the reflect node and both budget edges read them, and a run that priced its
    # allowance two different ways is the defect the split exists to prevent.
    def _price_expansion_budget(self) -> ExpansionBudget:
        return price_expansion_budget(self.plan, self._docs_by_type,
                                      max_revisions=self.budget.max_plan_revisions,
                                      supplied=self.expansion_budget,
                                      planner_terms=self._planner_terms)

    def _expansion_headroom(self) -> dict[str, int]:
        return headroom(self.plan, self._expansion_budget)

    def _expansion_budget_report(self) -> dict:
        return budget_report(self.plan, self._expansion_budget,
                             source=self.expansion_budget_source,
                             planner_terms=self._planner_terms)

    def _fit_terms_to_budget(self, rev: PlanRevision) -> tuple[PlanRevision, list[str]]:
        return fit_terms_to_budget(rev, self.plan, self._expansion_budget)

    def _expansion_is_spent(self) -> bool:
        return expansion_is_spent(self.plan, self._expansion_budget,
                                  terms_deferred=self._terms_deferred)

    def _n_reflect(self, s: RunState) -> dict:
        step = s.get("step", 0)
        self._gate_triggers(step)
        triggers = list(self._pending_triggers)
        self._pending_triggers = []
        self._counters.reflections += 1
        h = self._expansion_headroom()
        if any(v < 0 for v in h.values()):
            # Unreachable by construction — `apply_revision` refuses anything that would
            # overspend — so it is an error and not a clamp. A negative allowance shown to
            # the supervisor is the defect this budget split exists to remove; hiding one
            # behind max(0, ...) without saying so would put it back invisibly.
            self.tracer.emit("expansion_headroom_negative", severity="error", headroom=h,
                             budget=self._expansion_budget_report(),
                             message=("more expansion has been spent than the budget allows; "
                                      "the remaining allowance shown to the supervisor is "
                                      "clamped at zero and is NOT the true count"))
        remaining = (f"terms {max(h['terms'], 0)}, "
                     f"type promotions {max(h['type_promotions'], 0)}, "
                     f"revisions {max(h['revisions'], 0)}")
        prompt = REFLECT_PROMPT.format(
            question=self.spec.question,
            obligation="\n".join(f"  - {x}" for x in self.spec.proof_obligation.required_coverage) or "  (none)",
            plan=self.plan.render(self._docs_by_type),
            threads=self.threads.render(),
            triggers=("\n".join(f"  - [{t.kind}] {t.observation}"
                                + (f"  candidate terms: {list(t.terms_proposed)}"
                                   if t.terms_proposed else "")
                                + (f"  candidate types: {list(t.types_proposed)}"
                                   if t.types_proposed else "")
                                for t in triggers)
                      or "  (none detected — no observation obliges a revision)"),
            evidence=self.evidence.render(),
            coverage=self.coverage.render(),
            budget=remaining,
            step=step, max_steps=self.budget.max_steps,
        )
        msgs_ref = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        r = self.llm.chat(msgs_ref)
        # require="verdict": the reply may carry more than one JSON object (gpt-5.6-luna
        # leaks a tool-channel preamble object first), and the first one is not the answer.
        j = extract_json(r.content, require="verdict")
        raw_verdict = j.get("verdict")

        # Same trap as the planner, same model, same completion budget — and this prompt is
        # LONGER because it carries the evidence ledger. An unparsed reply silently becoming
        # CONTINUE is indistinguishable from a supervisor that read the evidence and judged
        # "keep going", which makes every CONTINUE in the trace uninterpretable. Retry once,
        # then record the degradation loudly instead of laundering it into a verdict.
        if raw_verdict is None or str(raw_verdict).upper() not in VERDICTS:
            self.tracer.emit("reflect_empty_retry", first_attempt_chars=len(r.content),
                             used_reasoning_channel=r.used_reasoning_channel,
                             completion_tokens=r.completion_tokens, raw_verdict=raw_verdict)
            old = self.llm.cfg.max_tokens
            self.llm.cfg.max_tokens = max(old * 2, 4096)
            try:
                r = self.llm.chat(msgs_ref)
            finally:
                self.llm.cfg.max_tokens = old
            j = extract_json(r.content, require="verdict")
            raw_verdict = j.get("verdict")

        degraded = raw_verdict is None or str(raw_verdict).upper() not in VERDICTS
        if degraded:
            verdict = "CONTINUE"
            self._counters.reflect_fallbacks += 1
            self.tracer.emit("reflect_fallback_used", severity="error", raw_verdict=raw_verdict,
                             message=("the supervisor returned nothing usable after a retry; "
                                      "defaulting to CONTINUE. This verdict carries NO "
                                      "information and no conclusion about reflection "
                                      "behaviour may be drawn from this step"))
        else:
            verdict = str(raw_verdict).upper()
        if verdict == "REPLAN":
            # No longer a verdict the model may assert. Counted, because a model still
            # reaching for the word is a fact about the prompt, and read as CONTINUE: the
            # REVISION replans, not the label. When the label was the mechanism, asserting
            # it changed nothing and so it was never asserted.
            self._counters.model_asserted_replan += 1
            self.tracer.emit("model_asserted_replan", severity="warning",
                             message=("the supervisor returned the REPLAN verdict, which no "
                                      "longer exists; the revision object is what replans"))
            verdict = "CONTINUE"

        # ------------------------------------------------------- apply the typed revision
        # IF THE RUNTIME DOES NOT APPLY IT, IT DID NOT HAPPEN. Everything below mutates the
        # object that actually governs retrieval, or refuses to and records the refusal.
        rev = PlanRevision.from_json(j.get("revision"))
        outcome = None
        deferred: list[str] = []
        salvage = None
        if not rev.is_empty():
            # WHAT OBSERVATION CAUSED IT. The detected triggers when there were any; the
            # supervisor's own stated reason when there were none. An empty observation would
            # make an unprompted widening indistinguishable from a forced one downstream, and
            # the develop plane needs to know which of the two it is holding.
            why = "; ".join(t.observation for t in triggers) or (
                f"unprompted by any detected trigger; supervisor's reason: "
                f"{str(j.get('reason', '')).strip() or '(none given)'}")
            trigger_label = ",".join(sorted({t.kind for t in triggers})) or "unprompted"
            # YOU MAY HAVE 5 OF THE 6. Trim the term list to what the budget can pay for
            # BEFORE it is priced, so one term too many no longer refuses the whole revision.
            to_apply, deferred = self._fit_terms_to_budget(rev)
            outcome = self.plan.apply_revision(
                to_apply, step=step, trigger=trigger_label,
                observation=why[:800],
                budget=self._expansion_budget, threads=self.threads,
                n_docs_by_type=self._docs_by_type,
                known_types=self.toolbox.known_doc_types)
            if deferred:
                self._terms_deferred.extend(deferred)
                outcome.refused.append(
                    f"{REFUSED_BUDGET}: {len(deferred)} of the {len(rev.add_terms)} terms you "
                    f"asked for did not fit the remaining expansion budget and were NOT "
                    f"added: {deferred}")
                self.tracer.emit("revision_partially_applied", severity="warning",
                                 requested_terms=list(rev.add_terms),
                                 applied_terms=list(outcome.terms_added),
                                 deferred_terms=list(deferred),
                                 headroom=self._expansion_headroom(),
                                 message=("the term list was trimmed to the remaining budget "
                                          "rather than the revision being refused whole; the "
                                          "deferred terms are reported back to the agent"))
            self.tracer.emit("plan_revision", applied=outcome.applied,
                             severity=("info" if outcome.applied and not deferred
                                       else "warning"),
                             requested=rev.to_dict(), applied_subset=to_apply.to_dict(),
                             deferred_terms=list(deferred),
                             outcome=outcome.to_dict(),
                             triggers=[t.kind for t in triggers])
            if not outcome.applied:
                self._counters.revisions_refused += 1
                # THE THREAD WORK IS NOT COLLATERAL. Retried on its own below; see
                # `_salvage_thread_work`. Except when the refusal IS the thread work: a
                # revision refused THREAD_NOOP has already had every thread operation in it
                # processed, and re-sending them would produce the same no-op a second time
                # and record it twice.
                if outcome.refusal_class != REFUSED_THREAD_NOOP:
                    salvage = self._salvage_thread_work(to_apply, step=step,
                                                        trigger=trigger_label, observation=why)
                # Back to the agent, in the loop it already understands. A refusal the model
                # never sees is a refusal it repeats.
                triggers = triggers + [Trigger(
                    kind="REVISION_REFUSED", step=step,
                    observation=f"{outcome.refusal_class}: {'; '.join(outcome.refused)[:300]}")]

        replanned = bool(outcome and outcome.applied and outcome.changed_retrieval())
        if replanned:
            self._counters.revisions_applied += 1
            # THE runtime-derived REPLAN. It is true exactly when the retrieval scope moved.
            verdict = "REPLAN"

        self.tracer.reflect(verdict, j.get("reason", ""), len(self.evidence.items))
        upd: dict[str, Any] = {"reflection": {"verdict": verdict, "reason": j.get("reason", ""),
                                              "degraded": degraded,
                                              "revision": (outcome.to_dict() if outcome else None),
                                              "triggers": [t.kind for t in triggers]}}
        if replanned:
            upd["plan"] = self.plan.to_dict()
        msgs = list(s["messages"])
        tail = ""
        refresh_plan = False
        if outcome is not None and outcome.applied and deferred:
            # PARTIAL APPLICATION IS ONLY HONEST IF IT IS SAID. An agent that asked for six
            # terms, got five and was told nothing would believe it had all six and would
            # never re-ask for the sixth.
            # "The plan now reads: <full listing>" — five times in one run, on top of the six
            # the plan node wrote. What the agent needs here is that the revision landed and
            # what did not fit; the plan itself is re-installed once, at the end of the
            # transcript, by `install_plan_block`. See its docstring for the measurement.
            tail = (f"Your revision was APPLIED IN PART ({REFUSED_BUDGET} on the term list "
                    f"only). These terms did NOT fit the remaining expansion budget and were "
                    f"not added: {deferred}. Re-ask for "
                    f"the one that matters most if budget frees up. The current plan is at "
                    f"the end of this thread.")
            refresh_plan = True
        elif outcome is not None and outcome.applied:
            tail = ("Your revision was APPLIED. The current plan is at the end of this thread.")
            refresh_plan = True
            if outcome.refused:
                # APPLIED IS NOT THE SAME AS "ALL OF IT LANDED". A revision that promoted a
                # type already at read_all and re-opened a thread already open used to come
                # back as an unqualified APPLIED, and the parts that changed nothing were
                # invisible — so they were sent again, and again. Whatever moved nothing is
                # named here, in the same message that says what did.
                tail += ("\nPART OF IT CHANGED NOTHING and was not counted as progress:\n  - "
                         + "\n  - ".join(outcome.refused))
        elif outcome is not None:
            tail = (f"Your revision was REFUSED ({outcome.refusal_class}) and the refusal is "
                    f"recorded:\n  - " + "\n  - ".join(outcome.refused))
            if salvage is not None and salvage.applied:
                tail += ("\nThe THREAD work in it was kept and applied — resolved: "
                         f"{salvage.threads_resolved}, dismissed: {salvage.threads_dismissed}, "
                         f"opened: {salvage.threads_opened}. Only the retrieval half was "
                         "refused; do not re-send the thread operations.")
        elif verdict == "SUFFICIENT":
            tail = "If you have what you need, call submit_answer now."
        msgs.append({"role": "user", "content":
                     f"[supervisor] verdict={verdict}. {j.get('reason','')}\n" + tail})
        if refresh_plan:
            # The revision moved retrieval, so the resident plan block is stale. Replace it —
            # the message above says WHAT changed, this puts the current plan back at the end
            # where the next act step reads it.
            msgs = install_plan_block(msgs, "PLAN (current):\n"
                                      + self.plan.render(self._docs_by_type))
        upd["messages"] = msgs
        return upd

    def _salvage_thread_work(self, rev: PlanRevision, *, step: int, trigger: str,
                             observation: str) -> RevisionOutcome | None:
        """Re-apply the thread half of a revision whose retrieval half was refused.

        A refused revision used to discard its thread operations too, so a revision that both
        over-reached on terms and RESOLVED THE THREAD BLOCKING THE ANSWER ended the run twice
        over: budget-exhausted and thread-blocked, with the resolution nowhere. Thread
        bookkeeping is not the retrieval half — `changed_retrieval()` says so itself, and no
        thread operation can violate monotonicity or widen what may be opened — so it does
        not belong to the refusal.

        Re-sent through `apply_revision` rather than applied against the ledger here, so the
        semantics of a resolution (and of an unreasoned dismissal, which is refused) stay
        defined in exactly one place.
        """
        threads_only = PlanRevision(open_threads=rev.open_threads,
                                    resolve_threads=rev.resolve_threads,
                                    dismiss_threads=rev.dismiss_threads)
        if threads_only.is_empty():
            return None
        out = self.plan.apply_revision(
            threads_only, step=step, trigger=trigger, observation=observation[:800],
            budget=self._expansion_budget, threads=self.threads,
            n_docs_by_type=self._docs_by_type, known_types=self.toolbox.known_doc_types)
        self.tracer.emit("thread_work_salvaged", severity="warning", applied=out.applied,
                         threads_opened=out.threads_opened,
                         threads_resolved=out.threads_resolved,
                         threads_dismissed=out.threads_dismissed,
                         refused=out.refused,
                         message=("the retrieval half of this revision was refused; its "
                                  "thread operations were re-applied on their own rather "
                                  "than discarded with it"))
        return out

    def _downgrade_a_positive_that_owes_something(self, ans: dict, s: RunState) -> None:
        """A run that stopped owing an obligation may not walk out with a positive.

        WHAT SYN0001 DID. It exhausted `max_tokens (400000)`, went to finalize with the
        `truncated` thread still open, and emitted:

            status FOUND, proof_basis UNGATED, route_to_human true,
            unsettled_threads ['Surgical-Pathology-Report_2023-04-27#truncated']

        That is a FOUND with a warning stapled to it. The manifest says so, `explain` says so,
        and `concordance.variables_from_answer` promotes each populated field to FOUND
        regardless of the answer's status — so the "warning" is carried by nothing that
        downstream reads. The honest label for "the budget ran out and the chart still has a
        question outstanding" is EVIDENCE_INSUFFICIENT, which is precisely what
        `ExpansionBudget`'s own docstring already promised and what `_after_reflect` already
        does for the expansion budget. The token budget got the same treatment here.

        THE THREE CONDITIONS, all required:
          * the answer is FOUND;
          * it never passed `gate_answer` — a gate-validated FOUND cleared the thread check
            and the decision rules, and nothing here has standing to second-guess it;
          * an obligation is genuinely outstanding, by the same `_outstanding_obligations`
            the dead-end edge uses. No obligation outstanding is a run that finished, and it
            keeps its UNGATED FOUND.

        THE VALUE GOES WITH THE STATUS. Left in place it would be re-promoted to FOUND field
        by field, and the downgrade would be cosmetic exactly where it matters most — the
        SYN0001 answer would still ship C341/8140/3 as established. It is preserved verbatim
        under `withheld_value` so nothing is destroyed and a reviewer can see what the model
        wanted to say; it simply is not asserted by a run that did not finish.
        """
        if ans.get("status") != "FOUND" or s.get("gate_validated"):
            return
        outstanding = self._outstanding_obligations()
        if not outstanding:
            return
        termination = getattr(self, "_termination", None) or "BUDGET_EXHAUSTED"
        ans["status"] = "EVIDENCE_INSUFFICIENT"
        ans["downgraded_from"] = "FOUND"
        ans["downgraded_because"] = (
            f"the run stopped ({termination}) with {len(outstanding)} obligation(s) still "
            f"outstanding and never passed the answer gate; a positive asserted from that "
            f"position is a guess with a warning attached, and the honest status is an "
            f"abstention")
        ans["outstanding_at_termination"] = outstanding[:20]
        if ans.get("value"):
            ans["withheld_value"] = ans["value"]
            # Explicit nulls, not a dropped key: `concordance.variables_from_answer` reads a
            # missing field as a silence and an explicit null as the answer's own status, and
            # the second is what this is. The run has a candidate and cannot vouch for it.
            ans["value"] = {f.name: None for f in self.spec.fields}
        self.tracer.emit("positive_downgraded_at_termination", severity="warning",
                         termination=termination, outstanding=outstanding[:20],
                         withheld_value=ans.get("withheld_value"),
                         message=("FOUND was emitted by a run that stopped with an obligation "
                                  "outstanding and no gate pass; recorded as "
                                  "EVIDENCE_INSUFFICIENT with the proposed value withheld"))

    def _n_finalize(self, s: RunState) -> dict:
        if s.get("answer") and s.get("done"):
            ans = s["answer"]
        else:
            gate = self._check_gate()
            note = ("The proof obligation for a negative answer is SATISFIED."
                    if gate.verdict == "PASS" else
                    "The proof obligation for a negative answer is NOT satisfied; outstanding: "
                    + "; ".join(gate.missing)
                    + ". You may not assert a confident negative — prefer EVIDENCE_INSUFFICIENT.")
            keys = ", ".join(f'"{f.name}": null' for f in self.spec.fields) or '"value": null'
            prompt = FINALIZE_PROMPT.format(
                spec_block=self.spec.as_prompt_block(), patient_id=s["patient_id"],
                rule_citations=rule_citation_block(self.spec),
                evidence=self.evidence.render(), coverage=self.coverage.render(),
                gate_note=note, value_keys=keys,
                spec_sections=", ".join(SPEC_SECTIONS))
            r = self.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            self.tracer.llm("finalize", r.content, usage={"total": r.total_tokens})
            ans = extract_json(r.content, require="status")
            # This answer never passes through `gate_answer` — finalize authors it when the
            # agent never called submit_answer — so its self-report has to be collected here
            # or the one path that produces an UNGATED answer is also the one path with no
            # attribution at all, which is precisely backwards.
            self.tracer.self_reported_rules([ans.get("rules_applied"), ans.get("reasoning")],
                                            where="finalize")

        forced_from = None
        if self.spec.data_source == "outside_notes":
            # Recorded before the overwrite: the loop needs to know this SPEC_INSUFFICIENT is
            # a constant the runtime imposes on every chart, not an agent's judgement, and
            # after the assignment that distinction is gone.
            forced_from = ans.get("status")
            ans["status"] = "SPEC_INSUFFICIENT"
        if not ans.get("status"):
            self._counters.finalize_defaults += 1
            self.tracer.emit("finalize_status_defaulted", severity="error",
                             message=("the model produced no status; defaulting to "
                                      "EVIDENCE_INSUFFICIENT. This label is NOT evidence "
                                      "that the agent reached that conclusion"))
            ans["status"] = "EVIDENCE_INSUFFICIENT"

        self._downgrade_a_positive_that_owes_something(ans, s)

        # Positive and negative findings are proved differently, so they are labelled
        # differently. A negative additionally has three possible bases demanding three
        # different downstream owners: emitting the same EVIDENCE_INSUFFICIENT for all of
        # them is how "the agent gave up" gets filed as "the chart really has nothing".
        if ans.get("status") == "FOUND":
            # Witness proof: one qualifying document settles it, and the gate for FOUND
            # checks exactly that. It never claims the universe was searched. Attaching
            # coverage_attested here would advertise a stronger claim than anything that was
            # verified — the same error as a check that records but cannot refuse, committed
            # on the way out instead of on the way in.
            ans["proof_basis"] = "WITNESS"
            ans["witness_count"] = len(self.evidence.items)
            if not s.get("gate_validated"):
                # Left without going through submit_answer, so not even the witness standard
                # was applied. Different reason from a coverage failure, same consequence.
                ans["proof_basis"] = "UNGATED"
                ans["route_to_human"] = True
                self.tracer.emit("ungated_positive", severity="warning",
                                 termination=getattr(self, "_termination", None))
        elif ans.get("status") == "SPEC_INSUFFICIENT":
            # A statement about the SPECIFICATION, so it gets no negative_basis and no
            # coverage ledger — those describe how well this chart was searched, and the
            # claim is not about this chart. Attaching one is the category error that
            # crashed every run reaching this status. What it does get is the report the
            # §6b optimizer routes on; see build_spec_gap.
            gap, remedy = build_spec_gap(
                self.spec, ans, reported_by=("runtime" if forced_from is not None else "agent"),
                gate_validated=bool(s.get("gate_validated")))
            if forced_from is not None:
                gap["forced_over_status"] = forced_from
            ans["spec_gap"] = gap
            ans["remedy_class"] = remedy
            ans["proof_basis"] = "NOT_APPLICABLE"
            ans["coverage_note"] = ("no coverage claim is made — SPEC_INSUFFICIENT is a "
                                    "statement about the specification, not about this chart")
            # The gate refuses a submitted value, but two paths arrive here without passing
            # it: the outside_notes overwrite above, and a finalize-authored answer written
            # when the agent never called submit_answer.
            strip_value_from_spec_insufficient(ans, self.tracer)
            self.tracer.emit("spec_insufficient_reported",
                             reported_by=gap["reported_by"], remedy_class=remedy,
                             spec_section=gap["spec_section"], routable=gap["routable"],
                             uncovered_fields=gap["uncovered_fields"],
                             has_spec_quote=bool(gap["spec_quote"]))
            if not gap["routable"]:
                # Reachable only when finalize authored the answer itself, so there is no
                # loop left to reject into. Loud and counted: the manifest still lands, the
                # status is still true, and the one thing that makes it USEFUL is missing.
                self._counters.spec_gaps_unroutable += 1
                self.tracer.emit("spec_gap_unroutable", severity="error",
                                 spec_section=gap["spec_section"],
                                 agent_words_supplied=gap["agent_words_supplied"],
                                 message=("SPEC_INSUFFICIENT was reported without naming the "
                                          "part of the specification at fault. The status is "
                                          "recorded; it cannot be routed to any text."))
        elif s.get("gate_validated"):
            ans["negative_basis"] = "GATE_VALIDATED"
            ans["proof_obligation"] = self._check_gate().to_dict()
            ans["coverage_attested"] = self.coverage.to_dict()
        else:
            ans["negative_basis"] = getattr(self, "_termination", None) or "BUDGET_EXHAUSTED"
            ans["route_to_human"] = True
            # Deliberately withholding the ledger: this answer never passed the gate, and
            # attaching it would let the answer read as though it had.
            ans["coverage_note"] = ("no coverage claim is made — this answer did not pass the "
                                    "proof obligation")
            self.tracer.emit("unvalidated_negative", severity="warning",
                             negative_basis=ans["negative_basis"])

        for k in ("spec_section", "spec_quote", "uncovered_fields"):
            # Folded into spec_gap already. Leaving the raw inputs beside the assembled block
            # invites a reader to trust the copy that the gate never validated — and on the
            # finalize-authored path it never did.
            ans.pop(k, None)
        ans["evidence"] = self.evidence.to_list()
        # Travels with the answer, not only in the manifest. An unresolved thread on a
        # finalize-authored answer (the one path with no gate to refuse it) is the exact
        # shape of the 8046 error, and the answer has to carry the fact that it exists.
        if self.threads.threads:
            ans["threads"] = self.threads.to_dict()
            if self.threads.unresolved():
                ans["route_to_human"] = True
                ans["unsettled_threads"] = [t.thread_id for t in self.threads.unresolved()]
                self.tracer.emit("answer_carries_unsettled_threads", severity="warning",
                                 thread_ids=ans["unsettled_threads"], status=ans.get("status"))
        assert_answer_is_reportable(ans)   # enforced at emission, not merely intended
        return {"answer": ans, "done": True}

    # ------------------------------------------------------------------ edges
    def _after_act(self, s: RunState) -> str:
        if s.get("done"):
            return "finalize"
        if self._over_budget(s):
            return "finalize"
        return "reflect" if s.get("step", 0) % self.reflect_every == 0 else "reflect"

    def _after_reflect(self, s: RunState) -> str:
        refl = s.get("reflection") or {}
        v = refl.get("verdict")
        if v is None:
            # Reaching the edge with no verdict at all means the node did not run. Treating
            # that as CONTINUE would hide a broken graph behind normal-looking behaviour.
            self._counters.reflect_fallbacks += 1
            self.tracer.emit("reflect_missing_at_edge", severity="error",
                             message="no verdict present when routing; defaulting to CONTINUE")
            v = "CONTINUE"
        if self._over_budget(s):
            self._termination = "BUDGET_EXHAUSTED"
            return "finalize"
        if v == "SUFFICIENT":
            # NOT to finalize. There must be exactly one route by which an answer is
            # produced, and it runs through submit_answer, because that is where the gate
            # lives. Routing SUFFICIENT to finalize gave the graph a second inbound edge to
            # the answer and the proof obligation was simply skipped — the run still printed
            # a proof_obligation field, computed but unable to refuse, which is a comment
            # wearing the costume of a check. reflect keeps its judgement; it expresses it as
            # "go submit", not as "we are done".
            self.tracer.emit("reflect_sufficient_routed_to_submit",
                             message="supervisor judged the evidence sufficient; routing to "
                                     "submit_answer so the proof obligation still applies")
            return "act"
        if v == "STUCK":
            # A give-up is not an answer and must NOT be asked to prove coverage: it asserts
            # no coverage. It goes straight out, labelled as unvalidated.
            self._termination = "AGENT_GAVE_UP"
            return "finalize"
        if self._expansion_exhausted_with_obligations():
            # EXPANSION HAS A BUDGET, and running out of it is a result, not a nuisance.
            # The alternative — keep looping until max_steps and emit whatever is in hand — is
            # a silent truncation dressed as an answer. This exits labelled, so the manifest
            # carries EXPANSION_BUDGET_EXHAUSTED rather than a shrug, and `finalize` gives it
            # no coverage claim because it never passed the gate.
            self._termination = "EXPANSION_BUDGET_EXHAUSTED"
            self.tracer.emit("expansion_budget_exhausted", severity="warning",
                             budget=self._expansion_budget_report(),
                             terms_added=len(self.plan.terms_added()),
                             terms_deferred=list(self._terms_deferred),
                             promotions=len(self.plan.promotion_log),
                             outstanding=self._outstanding_obligations(),
                             message=("the plan can no longer widen and the proof obligation "
                                      "is still not met. This is EVIDENCE_INSUFFICIENT and it "
                                      "is honest; it is not a pass and it is not a truncation"))
            return "finalize"
        if v == "REPLAN" and s.get("plan_revisions", 0) < self.budget.max_plan_revisions:
            return "plan"
        return "act"

    def _outstanding_obligations(self) -> list[str]:
        try:
            missing = list(check_gate(self.spec, self.coverage, self.plan).missing)
        except Exception:      # noqa: BLE001
            missing = []
        return missing + check_threads(self.threads)

    def _expansion_exhausted_with_obligations(self) -> bool:
        """True when widening is over and the obligations are not discharged.

        Both halves matter. Budget spent with everything discharged is a run that finished;
        obligations outstanding with budget left is a run that should keep going. Only the
        conjunction is a dead end, and a dead end has to be SAID.
        """
        return self._expansion_is_spent() and bool(self._outstanding_obligations())

    def _over_budget(self, s: RunState) -> bool:
        why = self.budget.exceeded(step=s.get("step", 0),
                                   tokens=self.llm.prompt_tokens + self.llm.completion_tokens,
                                   elapsed=time.time() - self._t0)
        if why:
            self.tracer.emit("budget_exceeded", reason=why)
            # LABELLED HERE AND NOT AT THE EDGE. Both `_after_act` and `_after_reflect` route
            # to finalize on exhaustion, and only the reflect edge used to set `_termination`
            # — so a run that ran out of tokens between two act steps arrived at finalize with
            # no termination reason at all. That is the SYN0001 run exactly: its
            # `ungated_positive` event carries `termination: null`, and the answer it labelled
            # was FOUND. One owner, so the two edges cannot disagree about why a run stopped.
            if getattr(self, "_termination", None) is None:
                self._termination = "BUDGET_EXHAUSTED"
        return bool(why)

    # ------------------------------------------------------------------ gate
    # Delegates, deliberately one line each. THERE IS EXACTLY ONE GATE and it lives in
    # `answer_gate`; a method here that recomputed any part of a verdict would be the second
    # copy that had to be removed once already.
    def _check_gate(self) -> GateResult:
        return check_gate(self.spec, self.coverage, getattr(self, "plan", None))

    def _keyword_hits_among_drawn(self) -> set[str]:
        return keyword_hits_among_drawn(self.spec, self.coverage, self.chart)

    def _gate(self, submitted: dict) -> dict:
        return gate_answer(self.spec, submitted, evidence=self.evidence,
                           coverage=self.coverage, chart=self.chart, tracer=self.tracer,
                           threads=getattr(self, "threads", None),
                           plan=getattr(self, "plan", None))

    # ------------------------------------------------------------------ run
    def run(self, chart: PatientChart, run_id: str | None = None,
            known_doc_types: list[str] | None = None) -> dict:
        self.chart = chart
        self.evidence = EvidenceLedger()
        docs, _ = chart.list_documents(limit=100_000)
        strata = strata_from_spec(self.spec)
        # Derived, not drawn. `ForcedSampler(None)` used to reach for `random.randrange`,
        # which made every unseeded run a fresh roll of the validation draw: rerun until the
        # sample is kind. `--seed` still wins, and `seed_provenance` in the manifest below is
        # how a reader tells the two apart.
        self.effective_seed = (self.sample_seed if self.sample_seed is not None
                               else derive_sample_seed(chart.patient_id, self.spec.spec_id))
        self.coverage = CoverageLedger(docs, strata, ForcedSampler(self.effective_seed))
        # Corpus-wide type vocabulary keeps "this patient has none" (a finding) separable
        # from "no such type" (a typo). Without it the toolbox says so in its own error.
        self.toolbox = Toolbox(chart, self.evidence, self.coverage,
                               known_doc_types=known_doc_types)
        self.tracer = Tracer.create(self.out_dir, run_id)
        self._t0 = time.time()
        self._counters = RunCounters()
        self._termination = None
        self._planner_terms = 0
        self._terms_deferred = []
        self._pending_triggers: list[Trigger] = []
        self._trigger_counts: dict[str, int] = {k: 0 for k in TRIGGERS}

        self.tracer.run_start(patient_id=chart.patient_id, model=self.llm.cfg.model,
                              **self.spec.identity(), n_documents=len(chart),
                              sample_seed=self.effective_seed,
                              seed_provenance=self.seed_provenance)
        # Before any rule can be cited, write down what the rules ARE. An id in a trace whose
        # spec has since been edited is unreadable without the fingerprint that travelled
        # with the run, and "which rule was in play" is the question this whole block exists
        # to answer six months later.
        self.tracer.bind_spec(self.spec)

        # THE ONE PLAN. Built once, before the loop, and never rebuilt: a plan re-derived
        # mid-run would be a fresh model guess with no monotonicity relation to the one the
        # agent has been working against, which is a narrowing wearing a widening's clothes.
        self._docs_by_type = documents_by_type(chart)
        self.threads = OpenThreadLedger()
        self.markers = load_marker_catalogue()
        if self.markers.degraded:
            self.tracer.emit("marker_catalogue_degraded", severity="error",
                             detail=self.markers.degraded,
                             message=("thread detection is running on an incomplete marker "
                                      "set; an unsettled thread may pass unnoticed"))
        self.plan = self._build_plan()
        # Whatever the planner proposed is already in `term_provenance` and is NOT the
        # agent's expansion allowance. Counted before the budget is priced against it.
        self._planner_terms = len(self.plan.term_provenance)
        self._expansion_budget = self._price_expansion_budget()
        self.tracer.emit("retrieval_plan", source=self.plan.source,
                         plan=self.plan.to_dict(),
                         expansion_budget=self._expansion_budget_report(),
                         expansion_budget_source=self.expansion_budget_source,
                         marker_catalogue=self.markers.source,
                         monotonicity_vs_ledger=MONOTONICITY_VS_LEDGER)

        final = self._graph.invoke(
            {"patient_id": chart.patient_id, "spec_id": self.spec.spec_id, "step": 0,
             "max_steps": self.budget.max_steps, "plan_revisions": 0, "rejections": []},
            {"recursion_limit": self.budget.max_steps * 4 + 20},
        )

        result = build_manifest(
            spec=self.spec, patient_id=chart.patient_id, model=self.llm.cfg.model,
            plan=self.plan, coverage=self.coverage, threads=self.threads,
            markers=self.markers, tracer=self.tracer, final=final, counters=self._counters,
            expansion=ExpansionRecord(
                report=self._expansion_budget_report(),
                exhausted=self._expansion_is_spent(),
                refused_at_least_once=self.plan.budget_exhausted(self._expansion_budget),
                terms_deferred=list(self._terms_deferred)),
            seed=SeedRecord(effective=self.effective_seed, provenance=self.seed_provenance,
                            caller_supplied=self.sample_seed is not None),
            triggers_fired=self._trigger_counts, usage=self.llm.usage(),
            run_budget=self.budget.report,
            elapsed_s=round(time.time() - self._t0, 2))
        self.tracer.run_end(**{k: v for k, v in result.items() if k != "answer"},
                            status=result.get("answer", {}).get("status"))
        self.tracer.write_manifest(result)
        return result
