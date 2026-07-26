"""The agent: plan -> act -> reflect -> (act | replan | finalize).

Why a graph and not a plain ReAct loop
--------------------------------------
A plain loop decides "what next" inside the same generation that just read a document,
which makes the stopping decision an afterthought. Here `reflect` is a separate node with
one job: look at what has actually been gathered and rule CONTINUE / REPLAN / SUFFICIENT /
STUCK. Replanning is therefore a first-class, traceable event rather than a drift in the
model's internal monologue.

Why the answer is gated in code
-------------------------------
`submit_answer` does not end the run. It is validated: a negative or absent answer is
rejected unless the spec's proof obligation is satisfied by the *computed* coverage ledger.
The rejection, with its reasons, is fed back to the model as an observation. Prompting a
model to "be sure you looked everywhere" is a wish; checking the ledger is a control.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from langgraph.graph import END, START, StateGraph

from .corpus import PatientChart
from .llm import LLMClient, extract_json
from .spec import ExtractionSpec
from .coverage import (CoverageLedger, ForcedSampler, GateResult, evaluate_gate,
                       strata_from_spec)
from .state import Budget, EvidenceLedger, RunState
from .tools import Toolbox
from .trace import Tracer

VERDICTS = {"CONTINUE", "REPLAN", "SUFFICIENT", "STUCK"}


class CoverageClaimError(AssertionError):
    """An answer advertised a coverage claim it did not earn."""


def assert_coverage_claim_is_earned(ans: dict) -> None:
    """`coverage_attested` may appear on exactly one kind of answer.

    A coverage ledger asserts "I searched the defined universe". Only a negative that passed
    the proof obligation has established that. A witness-proved positive never claimed it; a
    give-up and a budget exhaustion never earned it. Attaching the ledger anywhere else makes
    the answer advertise a stronger claim than was verified, and — because it looks exactly
    like a verified one downstream — nothing would catch it.

    Checked at the point of emission rather than left as an intention, since the whole family
    of bugs this guards against consists of intentions that the code did not keep.
    """
    has_ledger = "coverage_attested" in ans
    earned = (ans.get("status") == "EVIDENCE_INSUFFICIENT"
              and ans.get("negative_basis") == "GATE_VALIDATED")
    if has_ledger and not earned:
        raise CoverageClaimError(
            f"coverage_attested attached to status={ans.get('status')!r} "
            f"negative_basis={ans.get('negative_basis')!r} proof_basis={ans.get('proof_basis')!r}; "
            "only a gate-validated EVIDENCE_INSUFFICIENT may carry a coverage claim"
        )
    if earned and not has_ledger:
        raise CoverageClaimError(
            "a gate-validated negative must carry its coverage_attested ledger — "
            "the claim is only auditable if the evidence for it travels with it"
        )

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

Call one tool at a time and read its result before deciding the next move."""

PLAN_PROMPT = """{spec_block}

PATIENT: {patient_id}
CHART SHAPE: {n_docs} documents, {n_types} document types, {date_lo} .. {date_hi}
DOCUMENT TYPES PRESENT: {types}

Write a short investigation plan: 3-6 steps, each a concrete goal with a one-line rationale.
Plan around the evidence you need and the proof obligation you must satisfy, not around \
reading everything.

Reply with JSON only:
{{"plan":[{{"id":"1","goal":"...","rationale":"..."}}]}}"""

REFLECT_PROMPT = """You are supervising a chart review in progress. Judge only what has \
actually been gathered.

SPECIFICATION QUESTION: {question}

PROOF OBLIGATION FOR A NEGATIVE ANSWER:
{obligation}

CURRENT PLAN:
{plan}

EVIDENCE RECORDED SO FAR:
{evidence}

COVERAGE SO FAR:
{coverage}

STEPS USED: {step}/{max_steps}

Rule one verdict:
- SUFFICIENT  the recorded evidence already answers the question, or the proof obligation \
for a negative has been met and nothing was found.
- CONTINUE    the current plan is still the right one; keep going.
- REPLAN      something learned changes what should be done next; supply a revised plan.
- STUCK       further search is futile; the honest answer is an abstention.

Reply with JSON only:
{{"verdict":"CONTINUE|REPLAN|SUFFICIENT|STUCK","reason":"one or two sentences",
 "revised_plan":[{{"id":"1","goal":"...","rationale":"..."}}]}}
(revised_plan only when verdict is REPLAN)"""

FINALIZE_PROMPT = """{spec_block}

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
  "evidence_ids":["E1","E2"]}}"""


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
    ):
        self.spec = spec
        self.llm = llm
        self.budget = budget or Budget()
        self.reflect_every = reflect_every
        self.out_dir = Path(out_dir)
        # Recorded in the trace so an audit can confirm which documents were drawn, and so
        # a run replays deterministically. Two ablation arms must share it to be comparable.
        self.sample_seed = sample_seed
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

    # ------------------------------------------------------------------ nodes
    def _n_plan(self, s: RunState) -> dict:
        summary = self.chart.type_summary()
        docs, total = self.chart.list_documents(limit=1)
        alldocs, _ = self.chart.list_documents(limit=10_000)
        lo = alldocs[0].date.isoformat() if alldocs else "-"
        hi = alldocs[-1].date.isoformat() if alldocs else "-"
        prompt = PLAN_PROMPT.format(
            spec_block=self.spec.as_prompt_block(),
            patient_id=s["patient_id"], n_docs=total, n_types=len(summary),
            date_lo=lo, date_hi=hi,
            types=", ".join(r["doc_type"] for r in summary[:24]),
        )
        msgs_plan = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        r = self.llm.chat(msgs_plan)
        self.tracer.llm("plan", r.content, usage={"total": r.total_tokens,
                                                  "reasoning_channel": r.used_reasoning_channel})
        plan = extract_json(r.content).get("plan", [])

        # A hybrid reasoning model can burn the whole completion budget on thinking and
        # return empty text. Silently substituting a one-line plan makes that look like a
        # working planner: every run in this repo did exactly that, for every replan, without
        # anything going red. Retry once with room to answer, then fail LOUDLY.
        if not isinstance(plan, list) or not plan:
            self.tracer.emit("plan_empty_retry", first_attempt_chars=len(r.content),
                             used_reasoning_channel=r.used_reasoning_channel,
                             completion_tokens=r.completion_tokens)
            old = self.llm.cfg.max_tokens
            self.llm.cfg.max_tokens = max(old * 3, 4096)
            try:
                r2 = self.llm.chat(msgs_plan)
            finally:
                self.llm.cfg.max_tokens = old
            self.tracer.llm("plan_retry", r2.content, usage={"total": r2.total_tokens})
            plan = extract_json(r2.content).get("plan", [])

        planning_failed = not isinstance(plan, list) or not plan
        if planning_failed:
            plan = [{"id": "1",
                     "goal": "Identify and read the documents that could establish the answer.",
                     "rationale": "FALLBACK — the planner returned nothing usable"}]
            # Loud, countable, and visible in the manifest. A degraded planner must never
            # again be indistinguishable from a working one.
            self.tracer.emit("plan_fallback_used", severity="error",
                             message=("planner produced no usable plan after a retry; the run "
                                      "is proceeding on a generic one-step goal and its "
                                      "replanning behaviour is NOT being exercised"))
            self._plan_fallbacks = getattr(self, "_plan_fallbacks", 0) + 1
        for st in plan:
            st.setdefault("status", "pending")
        rev = s.get("plan_revisions", 0)
        self.tracer.plan(plan, rev)
        msgs = s.get("messages") or [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": self.spec.as_prompt_block()
             + f"\n\nPATIENT: {s['patient_id']}\nBegin. Work the plan; call one tool at a time."},
        ]
        msgs = msgs + [{"role": "user", "content": "PLAN:\n" + json.dumps(plan, ensure_ascii=False, indent=2)}]
        return {"plan": plan, "plan_revisions": rev + 1, "messages": msgs}

    def _n_act(self, s: RunState) -> dict:
        msgs = list(s["messages"])
        r = self.llm.chat(msgs, tools=self.toolbox.schemas())
        self.tracer.llm("act", r.content, [c["name"] for c in r.tool_calls], {"total": r.total_tokens})

        if not r.tool_calls:
            msgs.append({"role": "assistant", "content": r.content or ""})
            self._act_no_tool_call += 1
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
            out, ms = self.toolbox.dispatch(c["name"], c["arguments"])
            self.tracer.tool(c["name"], c["arguments"], out, ok="error" not in out, ms=ms)

            if c["name"] == "submit_answer":
                verdict = self._gate(self.toolbox.submitted or {})
                if verdict["accepted"]:
                    answer = self.toolbox.submitted or {}
                    done = True
                    gate_validated = True          # the ONLY place this becomes true
                    out = {"accepted": True}
                else:
                    rejections.append(verdict)
                    self.tracer.rejected(verdict["why"], verdict["missing"], self.toolbox.submitted)
                    out = {"accepted": False, "why": verdict["why"], "you_must_still": verdict["missing"]}

            msgs.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                         "content": json.dumps(out, ensure_ascii=False, default=str)[:6000]})

        return {"messages": msgs, "step": s.get("step", 0) + 1, "done": done,
                "gate_validated": gate_validated,
                "answer": answer, "rejections": rejections,
                "evidence": self.evidence.to_list(), "coverage": self.coverage.to_dict()}

    def _n_reflect(self, s: RunState) -> dict:
        prompt = REFLECT_PROMPT.format(
            question=self.spec.question,
            obligation="\n".join(f"  - {x}" for x in self.spec.proof_obligation.required_coverage) or "  (none)",
            plan=json.dumps(s.get("plan", []), ensure_ascii=False),
            evidence=self.evidence.render(),
            coverage=self.coverage.render(),
            step=s.get("step", 0), max_steps=self.budget.max_steps,
        )
        msgs_ref = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        r = self.llm.chat(msgs_ref)
        j = extract_json(r.content)
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
            j = extract_json(r.content)
            raw_verdict = j.get("verdict")

        degraded = raw_verdict is None or str(raw_verdict).upper() not in VERDICTS
        if degraded:
            verdict = "CONTINUE"
            self._reflect_fallbacks += 1
            self.tracer.emit("reflect_fallback_used", severity="error", raw_verdict=raw_verdict,
                             message=("the supervisor returned nothing usable after a retry; "
                                      "defaulting to CONTINUE. This verdict carries NO "
                                      "information and no conclusion about reflection "
                                      "behaviour may be drawn from this step"))
        else:
            verdict = str(raw_verdict).upper()

        self.tracer.reflect(verdict, j.get("reason", ""), len(self.evidence.items))
        upd: dict[str, Any] = {"reflection": {"verdict": verdict, "reason": j.get("reason", ""),
                                              "degraded": degraded}}
        if verdict == "REPLAN" and isinstance(j.get("revised_plan"), list) and j["revised_plan"]:
            plan = j["revised_plan"]
            for st in plan:
                st.setdefault("status", "pending")
            upd["plan"] = plan
        msgs = list(s["messages"])
        msgs.append({"role": "user", "content":
                     f"[supervisor] verdict={verdict}. {j.get('reason','')}\n"
                     + ("Revised plan:\n" + json.dumps(upd.get('plan'), ensure_ascii=False, indent=2)
                        if verdict == "REPLAN" else
                        "If you have what you need, call submit_answer now." if verdict == "SUFFICIENT" else "")})
        upd["messages"] = msgs
        return upd

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
                evidence=self.evidence.render(), coverage=self.coverage.render(),
                gate_note=note, value_keys=keys)
            r = self.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            self.tracer.llm("finalize", r.content, usage={"total": r.total_tokens})
            ans = extract_json(r.content)

        if self.spec.data_source == "outside_notes":
            ans["status"] = "SPEC_INSUFFICIENT"
            ans["remedy_class"] = "WRONG_DATA_SOURCE"
        if not ans.get("status"):
            self._finalize_defaults += 1
            self.tracer.emit("finalize_status_defaulted", severity="error",
                             message=("the model produced no status; defaulting to "
                                      "EVIDENCE_INSUFFICIENT. This label is NOT evidence "
                                      "that the agent reached that conclusion"))
            ans["status"] = "EVIDENCE_INSUFFICIENT"

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

        ans["evidence"] = self.evidence.to_list()
        assert_coverage_claim_is_earned(ans)   # enforced at emission, not merely intended
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
            self._reflect_fallbacks += 1
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
        if v == "REPLAN" and s.get("plan_revisions", 0) < self.budget.max_plan_revisions:
            return "plan"
        return "act"

    def _over_budget(self, s: RunState) -> bool:
        why = self.budget.exceeded(step=s.get("step", 0),
                                   tokens=self.llm.prompt_tokens + self.llm.completion_tokens,
                                   elapsed=time.time() - self._t0)
        if why:
            self.tracer.emit("budget_exceeded", reason=why)
        return bool(why)

    # ------------------------------------------------------------------ gate
    def _check_gate(self) -> GateResult:
        """Evaluate the spec's proof obligation against the stratified ledger."""
        fn = getattr(self.spec.proof_obligation, "for_negative", {}) or {}
        gate_spec = dict(fn.get("gate") or {})
        if not gate_spec and fn.get("required_coverage"):
            # A spec written before stratification: fall back to the keyword checks it does
            # declare, rather than silently passing everything.
            gate_spec = {"required_keywords_all_searched": True}
        strata = self.coverage.stratum_results()
        g = evaluate_gate(gate_spec, strata)
        for kw in getattr(self.spec.proof_obligation, "required_keywords", []) or []:
            if not any(kw.lower() in t or t in kw.lower() for t in self.coverage.searched_terms):
                g.missing.append(f"required search not performed: {kw!r}")
        if not self.coverage.listed_documents:
            g.missing.append("must list the patient's documents before asserting absence")
        g.verdict = "PASS" if not g.missing else "FAIL"
        return g

    def _gate(self, submitted: dict) -> dict:
        status = submitted.get("status", "")
        if not self.evidence.items and status == "FOUND":
            return {"accepted": False, "why": "no evidence recorded",
                    "missing": ["record at least one verbatim quote with record_evidence before answering FOUND"]}
        if status == "EVIDENCE_INSUFFICIENT":
            # Runtime-forced validation sampling. Drawn by the sampler, never by the agent:
            # a model choosing which unread documents to check is validating its own
            # judgement with its own judgement.
            # Credit whatever the agent has already read against the outstanding draw before
            # deciding it still owes anything.
            self.coverage.resolve_sample_verdicts(self.evidence.cited_notes())
            pending = self.coverage.pending_samples()
            if pending:
                lines = []
                for stratum, docs in pending.items():
                    for d in docs:
                        lines.append(f"  {stratum}: {d.note_id} ({d.doc_type}, {d.date})")
                self.tracer.emit("forced_sampling", seed=self.coverage.sampler.seed,
                                 counts={k: len(v) for k, v in pending.items()})
                ids = [d.note_id for docs in pending.values() for d in docs]
                return {"accepted": False,
                        "why": "validation sampling not yet done — the runtime has drawn these",
                        "how_to_satisfy": ("call read_documents_batch with note_ids set to the "
                                           "list below, in one step; then record_evidence for "
                                           "any that turn out to be relevant, then resubmit"),
                        "note_ids": ids,
                        "missing": ["these were drawn by the runtime, not chosen by you:"] + lines}
            gate = self._check_gate()
            if gate.verdict != "PASS":
                return {"accepted": False,
                        "why": "the proof obligation for asserting absence is not yet met",
                        "missing": gate.missing}
        return {"accepted": True, "why": "", "missing": []}

    # ------------------------------------------------------------------ run
    def run(self, chart: PatientChart, run_id: str | None = None,
            known_doc_types: list[str] | None = None) -> dict:
        self.chart = chart
        self.evidence = EvidenceLedger()
        docs, _ = chart.list_documents(limit=100_000)
        strata = strata_from_spec(self.spec)
        self.coverage = CoverageLedger(docs, strata, ForcedSampler(self.sample_seed))
        # Corpus-wide type vocabulary keeps "this patient has none" (a finding) separable
        # from "no such type" (a typo). Without it the toolbox says so in its own error.
        self.toolbox = Toolbox(chart, self.evidence, self.coverage,
                               known_doc_types=known_doc_types)
        self.tracer = Tracer.create(self.out_dir, run_id)
        self._t0 = time.time()
        self._plan_fallbacks = 0
        self._reflect_fallbacks = 0
        self._finalize_defaults = 0
        self._act_no_tool_call = 0
        self._termination = None

        self.tracer.run_start(patient_id=chart.patient_id, model=self.llm.cfg.model,
                              **self.spec.identity(), n_documents=len(chart))

        final = self._graph.invoke(
            {"patient_id": chart.patient_id, "spec_id": self.spec.spec_id, "step": 0,
             "max_steps": self.budget.max_steps, "plan_revisions": 0, "rejections": []},
            {"recursion_limit": self.budget.max_steps * 4 + 20},
        )

        answer = final.get("answer", {})
        result = {
            "run_id": self.tracer.run_id,
            "patient_id": chart.patient_id,
            **self.spec.identity(),
            "model": self.llm.cfg.model,
            "answer": answer,
            "plan": final.get("plan", []),
            "plan_revisions": final.get("plan_revisions", 0),
            "steps": final.get("step", 0),
            "negative_basis": (final.get("answer") or {}).get("negative_basis"),
            "gate_validated": bool(final.get("gate_validated")),
            "rejections": final.get("rejections", []),
            "usage": self.llm.usage(),
            # If this is non-zero the planner degraded and the run's replanning behaviour
            # was never actually exercised — read any conclusion about planning accordingly.
            # Any non-zero entry here means a node degraded silently and the corresponding
            # behaviour was NOT exercised. Read every conclusion against this block first.
            "degradation": {
                "plan_fallbacks": getattr(self, "_plan_fallbacks", 0),
                "reflect_fallbacks": getattr(self, "_reflect_fallbacks", 0),
                "finalize_defaults": getattr(self, "_finalize_defaults", 0),
                "act_no_tool_call": getattr(self, "_act_no_tool_call", 0),
            },
            "elapsed_s": round(time.time() - self._t0, 2),
            "trace": str(self.tracer.path),
        }
        self.tracer.run_end(**{k: v for k, v in result.items() if k != "answer"},
                            status=answer.get("status"))
        self.tracer.write_manifest(result)
        return result
