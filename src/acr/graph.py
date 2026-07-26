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
from .state import (Budget, CoverageLedger, EvidenceLedger, RunState, check_proof_obligation)
from .tools import Toolbox
from .trace import Tracer

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
    ):
        self.spec = spec
        self.llm = llm
        self.budget = budget or Budget()
        self.reflect_every = reflect_every
        self.out_dir = Path(out_dir)
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
        r = self.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        self.tracer.llm("plan", r.content, usage={"total": r.total_tokens})
        plan = extract_json(r.content).get("plan", [])
        if not isinstance(plan, list) or not plan:
            plan = [{"id": "1", "goal": "Identify and read the documents that could establish the answer.",
                     "rationale": "fallback plan"}]
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
        answer = s.get("answer") or {}

        for c in r.tool_calls:
            out, ms = self.toolbox.dispatch(c["name"], c["arguments"])
            self.tracer.tool(c["name"], c["arguments"], out, ok="error" not in out, ms=ms)

            if c["name"] == "submit_answer":
                verdict = self._gate(self.toolbox.submitted or {})
                if verdict["accepted"]:
                    answer = self.toolbox.submitted or {}
                    done = True
                    out = {"accepted": True}
                else:
                    rejections.append(verdict)
                    self.tracer.rejected(verdict["why"], verdict["missing"], self.toolbox.submitted)
                    out = {"accepted": False, "why": verdict["why"], "you_must_still": verdict["missing"]}

            msgs.append({"role": "tool", "tool_call_id": c["id"], "name": c["name"],
                         "content": json.dumps(out, ensure_ascii=False, default=str)[:6000]})

        return {"messages": msgs, "step": s.get("step", 0) + 1, "done": done,
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
        r = self.llm.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
        j = extract_json(r.content)
        verdict = str(j.get("verdict", "CONTINUE")).upper()
        if verdict not in {"CONTINUE", "REPLAN", "SUFFICIENT", "STUCK"}:
            verdict = "CONTINUE"
        self.tracer.reflect(verdict, j.get("reason", ""), len(self.evidence.items))
        upd: dict[str, Any] = {"reflection": {"verdict": verdict, "reason": j.get("reason", "")}}
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
            gate = check_proof_obligation(self.spec, self.coverage)
            note = ("The proof obligation for a negative answer is SATISFIED."
                    if gate.satisfied else
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

        gate = check_proof_obligation(self.spec, self.coverage)
        if self.spec.data_source == "outside_notes":
            ans["status"] = "SPEC_INSUFFICIENT"
        ans.setdefault("status", "EVIDENCE_INSUFFICIENT")
        ans["proof_obligation"] = gate.to_dict()
        ans["evidence"] = self.evidence.to_list()
        ans["coverage_attested"] = self.coverage.to_dict()
        return {"answer": ans, "done": True}

    # ------------------------------------------------------------------ edges
    def _after_act(self, s: RunState) -> str:
        if s.get("done"):
            return "finalize"
        if self._over_budget(s):
            return "finalize"
        return "reflect" if s.get("step", 0) % self.reflect_every == 0 else "reflect"

    def _after_reflect(self, s: RunState) -> str:
        v = (s.get("reflection") or {}).get("verdict", "CONTINUE")
        if self._over_budget(s):
            return "finalize"
        if v in ("SUFFICIENT", "STUCK"):
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
    def _gate(self, submitted: dict) -> dict:
        status = submitted.get("status", "")
        if not self.evidence.items and status == "FOUND":
            return {"accepted": False, "why": "no evidence recorded",
                    "missing": ["record at least one verbatim quote with record_evidence before answering FOUND"]}
        if status == "EVIDENCE_INSUFFICIENT":
            gate = check_proof_obligation(self.spec, self.coverage)
            if not gate.satisfied:
                return {"accepted": False,
                        "why": "the proof obligation for asserting absence is not yet met",
                        "missing": gate.missing}
        return {"accepted": True, "why": "", "missing": []}

    # ------------------------------------------------------------------ run
    def run(self, chart: PatientChart, run_id: str | None = None,
            known_doc_types: list[str] | None = None) -> dict:
        self.chart = chart
        self.evidence = EvidenceLedger()
        self.coverage = CoverageLedger()
        # Corpus-wide type vocabulary keeps "this patient has none" (a finding) separable
        # from "no such type" (a typo). Without it the toolbox says so in its own error.
        self.toolbox = Toolbox(chart, self.evidence, self.coverage,
                               known_doc_types=known_doc_types)
        self.tracer = Tracer.create(self.out_dir, run_id)
        self._t0 = time.time()

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
            "rejections": final.get("rejections", []),
            "usage": self.llm.usage(),
            "elapsed_s": round(time.time() - self._t0, 2),
            "trace": str(self.tracer.path),
        }
        self.tracer.run_end(**{k: v for k, v in result.items() if k != "answer"},
                            status=answer.get("status"))
        self.tracer.write_manifest(result)
        return result
