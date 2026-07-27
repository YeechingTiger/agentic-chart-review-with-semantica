"""A deepagents runtime for the same chart-review task, alongside the LangGraph one.

Run with:
    .venv-deep/bin/python -m acr.deep_runner SYN0002 --out runs/deep_SYN0002

WHY THIS SITS BESIDE graph.py RATHER THAN REPLACING IT
The LangGraph path produced today's numbers. Deleting it would leave the deepagents results
with no baseline, and the comparison is the only way to tell an agent-architecture effect
from ordinary temperature-1 variance. Both paths therefore share the same spec, corpus,
ledgers and — critically — the same gate.

WHAT IS REUSED UNCHANGED
    Corpus / PatientChart      corpus.py
    EvidenceLedger             state.py
    CoverageLedger, strata     coverage.py
    answer_checks              answer_checks.py
    THE GATE                   graph.ChartReviewAgent._gate
    WHAT MAY CLAIM COVERAGE    graph.assert_coverage_claim_is_earned, via
                               assert_answer_is_reportable

The gate is reused by *holding* a ChartReviewAgent and calling its `_gate`, never running
its graph. That looks indirect, and it is deliberate: the gate is the audit rule, and a
second copy of an audit rule is a liability — the two drift, and then a run's validation
means whichever copy happened to execute. One implementation, two front ends.

WHAT DEEPAGENTS SUPPLIES
Its own planning (todo) and sub-agent middleware replace the hand-written plan/reflect
nodes. That is the substantive difference to measure. Note the trade: `plan_fallbacks` and
`reflect_fallbacks` are counters over OUR nodes, so they do not exist here. A deepagents run
cannot report the degradation block, which means it cannot be compared on that axis at all —
state that wherever these runs are cited.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .answer_checks import check_answer  # noqa: F401  (used via the shared gate)
from .corpus import Corpus
from .coverage import CoverageLedger, ForcedSampler, strata_from_spec
from .coverage_planner import (MONOTONICITY_VS_LEDGER, OPEN_REQUEST_OPENED, OpenThreadLedger,
                               documents_by_type, load_marker_catalogue, plan_from_spec,
                               triggers_from_tool_result)
from .graph import (SPEC_SECTIONS, ChartReviewAgent, assert_answer_is_reportable,
                    build_spec_gap, strip_value_from_spec_insufficient)
from .spec import load_spec
from .state import Budget, EvidenceLedger
from .tools.toolbox import Toolbox
from .trace import Tracer

TASK = """Determine the answer for patient {patient} using ONLY this chart.

Work by calling tools: list the documents, search them, read what matters, and record every
claim with record_evidence and a verbatim quote. When you are ready call submit_answer.

submit_answer is GATED. If the proof obligation is not met it will be rejected with the
reason, and you must act on that reason and submit again. A rejection is not a failure; it
is the instruction for what to do next."""

#: What a manifest says when it makes no coverage claim at all. A key, not an omission:
#: "this run claimed nothing" and "this manifest predates the field" must stay distinguishable
#: to a reader filtering a directory of them.
NO_COVERAGE_CLAIM = "no coverage claim is made — see answer.status"


def attach_coverage_claim(answer: dict, *, gate_validated: bool, ledger: dict,
                          ungated_basis: str) -> None:
    """Everything this runtime says about coverage, derived from the gate and nothing else.

    A coverage ledger asserts "I searched the universe this spec defines". The only thing
    that establishes that is the proof obligation, and the only thing that evaluates the
    proof obligation is `ChartReviewAgent._gate`. So the ledger is attached on the branch the
    gate accepted, and on the other branch the answer says in words that it makes no claim —
    because downstream an unearned ledger is indistinguishable from an earned one, which is
    the entire failure mode.

    Written ONTO THE ANSWER, exactly where `graph._n_finalize` writes it. The manifest used
    to carry a top-level `coverage_attested` and the answer to carry nothing, which put the
    claim outside the reach of `assert_answer_is_reportable` — so the one rule that says who
    may claim coverage was never asked. It is asked now, and it refuses in both directions:
    an unearned ledger raises, and a gate-validated negative WITHOUT its ledger raises too.

    Only EVIDENCE_INSUFFICIENT belongs here. FOUND is proved by witness and never claimed the
    universe was searched; SPEC_INSUFFICIENT is not a claim about this chart at all.
    """
    if gate_validated:
        answer["negative_basis"] = "GATE_VALIDATED"
        answer["coverage_attested"] = ledger
        return
    # `ungated_basis` and not a literal: a negative that never passed the gate still owes the
    # reader WHY it ended, and every value but GATE_VALIDATED routes to a human.
    answer["negative_basis"] = ungated_basis
    answer["route_to_human"] = True
    answer["coverage_note"] = ("no coverage claim is made — this answer did not pass the "
                               "proof obligation")


def _make_tools(toolbox: Toolbox, tracer: Tracer, *, plan, catalogue, threads, chart):
    """Wrap the existing OpenAI-style tool schemas as LangChain StructuredTools.

    THE PLAN AND THE THREAD LEDGER ARE ENFORCED HERE TOO. A retrieval plan that binds one
    runtime and not the other makes the scope of a run silently conditional on which binary
    the operator happened to launch — the same asymmetry `assert_answer_is_reportable`
    exists to refuse. deepagents supplies its own planning middleware, so it gets no typed
    reflection channel and cannot widen the plan; what it does get is the same floor.
    """
    from langchain_core.tools import StructuredTool

    tools = []
    for s in toolbox.schemas():
        fn_spec = s["function"]
        name = fn_spec["name"]

        def _call(_name=name, **kwargs):
            refusal = _plan_refusal(_name, kwargs, plan=plan, chart=chart, coverage=toolbox.coverage)
            if refusal is not None:
                tracer.emit("plan_refused_open", severity="warning", runtime="deepagents",
                            tool=_name, blocked=refusal["blocked"])
                return json.dumps(refusal)[:20000]
            out, ms = toolbox.dispatch(_name, kwargs)
            tracer.tool(_name, kwargs, out, ok="error" not in out, ms=ms)
            for t in triggers_from_tool_result(
                    _name, kwargs, out if isinstance(out, dict) else {}, plan=plan,
                    catalogue=catalogue, step=0,
                    quote=str((out or {}).get("quote", "")) if _name == "record_evidence" else ""):
                if t.kind == "UNSETTLED_THREAD":
                    m = catalogue.by_text().get(t.marker)
                    # THE SAME GUARD AS `run_triggers.detect_from_tool_result`, and it broke
                    # the same way: `is None` was the sentinel test, `open_thread` began
                    # handing back the existing thread, and this runtime started counting
                    # short reads as threads too. Branch on the status; only `opened` is new
                    # debt. Both front ends must agree here — a trigger count that depends on
                    # which binary the operator launched is not a measurement.
                    request = threads.open_thread(
                        note_id=t.note_id, doc_type=t.doc_type, marker=t.marker,
                        obligation=(m.obligation if m else "unsettled"),
                        excerpt=t.observation, step=0)
                    if request.status != OPEN_REQUEST_OPENED:
                        continue
                # `tracer.trigger`, not `emit("trigger", **t.to_dict())`: the trigger's own
                # `kind` collides with the trace envelope's, which used to raise a TypeError
                # and kill the run at its first trigger. One emitter, so this runtime's
                # trigger events keep the same shape as `graph`'s.
                tracer.trigger(runtime="deepagents", **t.to_dict())
            return json.dumps(out)[:20000]

        tools.append(StructuredTool.from_function(
            func=_call, name=name, description=fn_spec.get("description", ""),
            args_schema=fn_spec.get("parameters") or {"type": "object", "properties": {}},
        ))
    return tools


def _plan_refusal(name: str, args: dict, *, plan, chart, coverage) -> dict | None:
    """Same refusal as `graph.ChartReviewAgent._plan_refusal`, and it has to be the same.

    Kept as a free function so the two front ends share the RULE even though they cannot
    share the dispatch loop. A `sample` type is the runtime sampler's to draw; a document the
    sampler drew is always open.
    """
    if name not in ("read_document", "read_section", "read_documents_batch"):
        return None
    ids = ([args.get("note_id")] if name != "read_documents_batch"
           else list(args.get("note_ids") or []))
    drawn = {n for v in coverage.drawn.values() for n in v}
    blocked = []
    for nid in ids:
        meta = chart._docs.get(str(nid))
        if meta is None or str(nid) in drawn:
            continue
        if not plan.may_open(meta.doc_type):
            blocked.append({"note_id": str(nid), "doc_type": meta.doc_type})
    if not blocked:
        return None
    return {"error": "OUT_OF_PLAN", "blocked": blocked,
            "types": sorted({b["doc_type"] for b in blocked}),
            "message": ("The retrieval plan assigns these types to `sample`: the runtime's "
                        "sampler draws from them and you may not open them directly. This is "
                        "NOT evidence that they hold nothing. Under this runtime the plan "
                        "cannot be widened — there is no typed reflection channel here — so "
                        "work the types the plan does allow.")}


def _callbacks():
    """Audit + optional langfuse. deepagents bypasses LiteLLM, so sitecustomize's hook
    never fires here and a run's cost would otherwise be invisible -- as it was on the
    first deepagents run."""
    cbs = []
    try:
        from lc_callback import make_handler, langfuse_handler
        for h in (make_handler(), langfuse_handler()):
            if h is not None:
                cbs.append(h)
    except Exception:
        pass
    return cbs


def _model(model_name: str, api_base: str | None, api_key: str | None, temperature: float):
    from langchain_openai import ChatOpenAI
    # gpt-5.6-luna rejects any temperature but 1 (see llm/.azure_env); passing 0 here 400s
    # on the first call exactly as it did through litellm.
    return ChatOpenAI(model=model_name, base_url=api_base, api_key=api_key,
                      temperature=temperature, timeout=600, max_retries=3,
                      callbacks=_callbacks())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("patient")
    ap.add_argument("--spec", default="specs/STORE.400_522_523.site_histology_behavior.yaml")
    ap.add_argument("--corpus", default="corpus/patients")
    ap.add_argument("--out", default="runs/deep")
    ap.add_argument("--max-steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--temperature", type=float,
                    default=float(os.getenv("ACR_TEMPERATURE", "1")))
    ap.add_argument("--skills", default=None,
                    help="directory of Agent Skills to expose (progressive disclosure). "
                         "Arm B of the skill validation; omit for arm A.")
    a = ap.parse_args(argv)

    spec = load_spec(Path(a.spec))
    corpus = Corpus(Path(a.corpus))
    chart = corpus.chart(a.patient)
    docs, _ = chart.list_documents(limit=100_000)

    out_dir = Path(f"{a.out}__{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    tracer = Tracer.create(out_dir)
    tracer.emit("run_start", patient_id=a.patient, model=os.getenv("ACR_MODEL", "?"),
                runtime="deepagents", spec_id=spec.spec_id, spec_hash=spec.spec_hash,
                n_documents=len(docs))
    # Same rule catalog as the LangGraph path. Attribution recorded under one runtime and not
    # the other makes "which rule was in play" silently conditional on which binary the
    # operator ran, which is the asymmetry `assert_answer_is_reportable` exists to refuse.
    tracer.bind_spec(spec)

    evidence = EvidenceLedger()
    coverage = CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(a.seed))
    toolbox = Toolbox(chart, evidence, coverage,
                      known_doc_types=corpus.doc_type_vocabulary())

    # THE ONE PLAN, on this front end too. Derived from the spec's own strata rather than
    # from a model: deepagents owns the planning middleware here, and inserting a second
    # LLM planner would make the arms differ in two things at once and the comparison would
    # measure nothing.
    plan = plan_from_spec(spec, chart)
    threads = OpenThreadLedger()
    markers = load_marker_catalogue()
    if markers.degraded:
        tracer.emit("marker_catalogue_degraded", severity="error", detail=markers.degraded)
    tracer.emit("retrieval_plan", runtime="deepagents", source=plan.source,
                plan=plan.to_dict(), marker_catalogue=markers.source,
                monotonicity_vs_ledger=MONOTONICITY_VS_LEDGER,
                revisable=False,
                revisable_why=("deepagents supplies its own planning middleware and there is "
                               "no typed reflection channel to apply a revision through. The "
                               "plan is a floor here, not a revisable object — say so "
                               "wherever these runs are compared against the LangGraph arm"))

    # Gate holder: shares our ledgers, never runs its own graph. See module docstring.
    gatekeeper = ChartReviewAgent(spec, llm=None, budget=Budget(max_steps=a.max_steps),
                                  out_dir=out_dir, sample_seed=a.seed)
    gatekeeper.chart = chart
    gatekeeper.evidence = evidence
    gatekeeper.coverage = coverage
    gatekeeper.toolbox = toolbox
    gatekeeper.tracer = tracer
    gatekeeper.plan = plan
    gatekeeper.threads = threads

    tools = _make_tools(toolbox, tracer, plan=plan, catalogue=markers, threads=threads,
                        chart=chart)

    # submit_answer must go through the gate, or `gate_validated` means nothing — which is
    # exactly the defect found in the LangGraph path, where FOUND answers were accepted
    # unchecked and still stamped True.
    from langchain_core.tools import StructuredTool

    state = {"accepted": False, "rejections": [], "answer": {}}

    def _submit(status: str, reasoning: str = "", value: dict | None = None,
                spec_section: str = "", spec_quote: str = "",
                uncovered_fields: list | None = None):
        toolbox.submitted = {"status": status, "value": value or {}, "reasoning": reasoning,
                             "spec_section": spec_section, "spec_quote": spec_quote,
                             "uncovered_fields": list(uncovered_fields or [])}
        verdict = gatekeeper._gate(toolbox.submitted)
        tracer.tool("submit_answer", toolbox.submitted, verdict,
                    ok=bool(verdict.get("accepted")), ms=0.0)
        if verdict.get("accepted"):
            state["accepted"] = True
            state["answer"] = dict(toolbox.submitted)
        else:
            state["rejections"].append(verdict)
        return json.dumps(verdict)[:8000]

    tools = [t for t in tools if t.name != "submit_answer"] + [
        StructuredTool.from_function(
            func=_submit, name="submit_answer",
            description=("Submit the final answer. GATED: may be rejected with a reason you "
                         "must act on. status is FOUND | EVIDENCE_INSUFFICIENT | "
                         "SPEC_INSUFFICIENT. SPEC_INSUFFICIENT is a report about the "
                         "SPECIFICATION: name the part at fault in spec_section (one of "
                         + ", ".join(SPEC_SECTIONS) + "), say in your own words what it "
                         "fails to cover, and send no value."),
            args_schema={"type": "object",
                         "properties": {"status": {"type": "string"},
                                        "reasoning": {"type": "string"},
                                        "value": {"type": "object"},
                                        # Present here for the same reason as on the other
                                        # two surfaces: an abstention channel that works on
                                        # one runtime and not another is a finding nobody
                                        # can interpret afterwards.
                                        "spec_section": {"type": "string"},
                                        "spec_quote": {"type": "string"},
                                        "uncovered_fields": {"type": "array",
                                                             "items": {"type": "string"}}},
                         "required": ["status"]},
        )]

    from deepagents import create_deep_agent
    # Skills are ADVISORY by construction -- progressive disclosure means the model decides
    # whether to load them. That is why the gate and the forced sampler are code and not
    # skills: an audit rule the model may decline to read is not an audit rule. Skills carry
    # the coding JUDGEMENT, where the failures were; the deterministic checks stay as backstop.
    kw = {}
    if a.skills:
        from deepagents.backends import FilesystemBackend
        kw["skills"] = [a.skills]
        kw["backend"] = FilesystemBackend(root_dir=".", virtual_mode=False)
        tracer.emit("skills_enabled", source=a.skills)
    agent = create_deep_agent(
        _model(os.getenv("ACR_MODEL_NAME", "gpt-5.6-luna"),
               os.getenv("ACR_API_BASE"), os.getenv("ACR_API_KEY"), a.temperature),
        tools,
        system_prompt=spec.as_prompt_block() + "\n\n" + TASK.format(patient=a.patient),
        **kw,
    )

    t0 = time.time()
    crashed = False
    try:
        # recursion_limit bounds tool-call rounds the way max_steps bounds our own loop.
        agent.invoke({"messages": [{"role": "user",
                                    "content": TASK.format(patient=a.patient)}]},
                     config={"recursion_limit": a.max_steps * 3})
    except Exception as e:  # noqa: BLE001 -- a crashed run must still leave its trace
        crashed = True
        tracer.emit("runtime_error", severity="error", error=f"{type(e).__name__}: {e}")
        print(f"!! deepagents raised: {type(e).__name__}: {e}", file=sys.stderr)
    elapsed = round(time.time() - t0, 1)
    # How the loop ended, for an answer that never passed the gate. Not a constant: filing a
    # crash as BUDGET_EXHAUSTED would tell the reader the agent ran out of room to work when
    # in fact the runtime fell over, and those have different owners.
    termination = "RUNTIME_ERROR" if crashed else "BUDGET_EXHAUSTED"

    answer = dict(state["answer"] or {"status": "NO_ANSWER"})
    forced_from = None
    if spec.data_source == "outside_notes" and answer.get("status") != "NO_ANSWER":
        forced_from = answer.get("status")
        answer["status"] = "SPEC_INSUFFICIENT"
    spec_gap = None
    if answer.get("status") == "SPEC_INSUFFICIENT":
        # This runtime never crashed on SPEC_INSUFFICIENT — it wrote a manifest carrying a
        # bare status code and, below, a coverage ledger attached to it regardless. Both are
        # the same category error the LangGraph path raised on: coverage describes how well
        # this chart was searched, and the claim is about the specification.
        spec_gap, remedy = build_spec_gap(
            spec, answer, reported_by=("runtime" if forced_from is not None else "agent"),
            gate_validated=state["accepted"])
        if forced_from is not None:
            spec_gap["forced_over_status"] = forced_from
        answer["spec_gap"] = spec_gap
        answer["remedy_class"] = remedy
        answer["proof_basis"] = "NOT_APPLICABLE"
        # Said, not left to be inferred from an absent key, and said in the same words
        # `graph._n_finalize` uses: searching every document in the record cannot make a
        # silent spec speak, so coverage of this chart is beside the point here.
        answer["coverage_note"] = ("no coverage claim is made — SPEC_INSUFFICIENT is a "
                                   "statement about the specification, not about this chart")
        strip_value_from_spec_insufficient(answer, tracer)
        tracer.emit("spec_insufficient_reported", runtime="deepagents",
                    reported_by=spec_gap["reported_by"], remedy_class=remedy,
                    spec_section=spec_gap["spec_section"], routable=spec_gap["routable"])
    for k in ("spec_section", "spec_quote", "uncovered_fields"):
        answer.pop(k, None)
    # THE COVERAGE CLAIM IS THE GATE'S TO MAKE, on this front end as on the other.
    if answer.get("status") == "EVIDENCE_INSUFFICIENT":
        attach_coverage_claim(answer, gate_validated=state["accepted"],
                              ledger=coverage.to_dict(), ungated_basis=termination)
    # Mirrored out of the answer, never recomputed. See the manifest key below.
    coverage_claim = ({"coverage_attested": answer["coverage_attested"]}
                      if "coverage_attested" in answer
                      else {"coverage_note": answer.get("coverage_note") or NO_COVERAGE_CLAIM})
    # Now enforceable rather than intended: this refuses an unearned ledger AND a
    # gate-validated negative that arrives without one.
    assert_answer_is_reportable(answer)

    manifest = {
        "runtime": "deepagents",
        "patient_id": a.patient,
        "spec_id": spec.spec_id, "spec_hash": spec.spec_hash,
        "answer": answer,
        # Lifted to the top level so a directory of manifests can be filtered for spec
        # gaps without parsing every answer. Null, not omitted: "this run reported none" and
        # "this manifest predates the field" have to stay distinguishable.
        "spec_gap": spec_gap,
        "gate_validated": state["accepted"],
        "rejections": state["rejections"],
        # WHICH SPEC RULE WAS IN PLAY, in the manifest so a finished run can be attributed
        # without replaying its trace. Same block, same two channels, as `graph.run`.
        "rule_attribution": tracer.rule_attribution(),
        # Both term lists, on both front ends. `initial_keywords` and `keywords` are equal
        # here because nothing can widen the plan under this runtime — which is a finding
        # about the runtime, and it is stated rather than left to be inferred from equality.
        "plan": plan.to_dict(),
        "open_threads": {**threads.to_dict(), "marker_catalogue": markers.source},
        # DELIBERATELY NULL, not zero, exactly as `degradation` is. The replan rate counts
        # applied revisions per reflection, and this runtime has neither. Reporting 0.0 would
        # claim "the agent had nothing to add" on an axis that was never measured — and that
        # is the reading that would send the whole criterion to a workflow.
        "replan": None,
        "replan_note": ("not measurable under deepagents: there is no typed reflection "
                        "channel, so the plan cannot be revised and the rate is undefined, "
                        "not zero"),
        "monotonicity_vs_ledger": MONOTONICITY_VS_LEDGER,
        "elapsed_s": elapsed,
        # Deliberately absent, not zero: plan_fallbacks / reflect_fallbacks count OUR nodes,
        # and deepagents supplies its own. Reporting zeros here would claim a clean run on an
        # axis that was never measured.
        "degradation": None,
        "degradation_note": ("not measurable under deepagents: the counters instrument the "
                             "hand-written plan/reflect nodes, which this runtime replaces"),
        # A COPY OF THE ANSWER'S CLAIM, and only ever a copy. `evals.py` reads coverage off
        # the manifest's top level while `explain.py` reads it off the answer, so a top-level
        # key computed on its own terms is two readers that can disagree about whether a run
        # attested coverage — this defect one level up. It is assembled above, next to the
        # answer, immediately before the check that has to be able to see it.
        **coverage_claim,
        "evidence": evidence.to_list(),
    }
    mp = out_dir / f"{tracer.run_id}.manifest.json"
    mp.write_text(json.dumps(manifest, indent=2))
    tracer.emit("run_end", accepted=state["accepted"], rejections=len(state["rejections"]),
                elapsed_s=elapsed)

    print(f"runtime          deepagents")
    print(f"manifest         {mp}")
    print(f"status           {manifest['answer'].get('status')}")
    print(f"value            {json.dumps(manifest['answer'].get('value'))}")
    if spec_gap:
        print(f"spec gap         {manifest['answer'].get('remedy_class')} in "
              f"{spec_gap['spec_section']} (by {spec_gap['reported_by']}, "
              f"{'routable' if spec_gap['routable'] else 'NOT routable'})")
        print(f"  agent words    {spec_gap['agent_words'][:160]}")
    print(f"gate_validated   {state['accepted']}")
    print(f"rejections       {len(state['rejections'])}")
    print(f"evidence items   {len(evidence.items)}")
    print(f"plan             read_all={len(plan.read_all)} search={len(plan.search)} "
          f"sample={len(plan.sample)} terms={len(plan.keywords)} (not revisable here)")
    print(f"open threads     {len(threads.unresolved())} unresolved of {len(threads.threads)}")
    print(f"elapsed          {elapsed}s")
    for r in state["rejections"][:3]:
        print("  rej:", json.dumps(r.get('missing') or r.get('why'))[:200])
    return 0


if __name__ == "__main__":
    sys.exit(main())
