"""SPEC_INSUFFICIENT: the abstention channel that could not be used at all.

An agent has three ways to answer. FOUND and EVIDENCE_INSUFFICIENT are claims about THIS
CHART. SPEC_INSUFFICIENT is not — it says "your SPECIFICATION does not cover this case", and
it is the highest-precision input the spec-improvement loop can ever receive, because it is
the agent naming the defect itself rather than a reviewer inferring one from a wrong answer.

It crashed every run that produced it. `graph._n_finalize` attached a coverage ledger to any
non-FOUND status, `assert_coverage_claim_is_earned` correctly refused it, the exception
escaped, and `cli.extract` caught it and wrote nothing — so the run left a truncated trace
and no manifest. On disk, a run that crashed and a run that never happened were the same
thing.

Then: across 38 real runs SPEC_INSUFFICIENT was reported ZERO times, and that was written up
as evidence the model under-uses its abstention channel. The conclusion was wrong. A broken
reporting channel whose output reads as a clean result is the worst shape a defect can take,
and this repo has produced that shape four times.

So the tests here are not "it no longer raises". They assert the four things that make the
channel worth having:

  1. it COMPLETES and leaves a MANIFEST — crash and never-ran must differ on disk;
  2. the manifest says WHY in a form the §6b optimizer can route on — which field, which part
     of the spec, the agent's own words, which rule;
  3. all THREE front ends do it — graph, deepagents and MCP — because a signal conditional on
     which runtime was used is a signal nobody can interpret;
  4. it CANNOT be used to carry a value past the gate. Handed an unproved exit, an agent will
     take it, and this repo has already shipped one.

Everything runs offline against the synthetic corpus with a scripted client. No provider is
called, no real chart is read.
"""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

import acr.answer_contract as AC
from acr import answer_gate as G
from acr.concordance import variables_from_answer
from acr.corpus import Corpus
from acr.coverage import CoverageLedger, ForcedSampler, strata_from_spec
from acr.answer_contract import (SPEC_SECTIONS, CoverageClaimError, SpecGapError,
                                 assert_answer_is_reportable, assert_spec_gap_is_reported,
                                 build_spec_gap)
from acr.answer_gate import gate_answer
from acr.llm import LLMClient, LLMConfig, LLMResponse
from acr.spec import load_spec
from acr.state import Budget, EvidenceLedger

ROOT = Path(__file__).resolve().parents[1]
SHB = ROOT / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"
#: `data_source: outside_notes`. Every run of it is FORCED to SPEC_INSUFFICIENT at finalize,
#: which means every run of it hit the crash — a spec that is broken on 100% of charts by
#: construction, shipped, with a test suite that never ran it end to end.
OUTSIDE = ROOT / "specs" / "STORE.610.class_of_case.yaml"
CORPUS = ROOT / "corpus" / "patients"

#: A quote lifted verbatim from the shipped spec, so the gate's citation check has something
#: true to accept. Not a patient quote — spec text only; nothing in this file touches a chart.
REAL_SPEC_QUOTE = "the site of ORIGIN, not the site of a biopsy"


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def shb():
    return load_spec(SHB)


@pytest.fixture(scope="module")
def outside():
    return load_spec(OUTSIDE)


@pytest.fixture(scope="module")
def chart():
    return Corpus(CORPUS).chart("SYN0002")


def _ledgers(spec, chart):
    docs, _ = chart.list_documents(limit=100_000)
    return EvidenceLedger(), CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(7))


def _gate(spec, chart, submitted: dict) -> dict:
    ev, cov = _ledgers(spec, chart)
    return gate_answer(spec, submitted, evidence=ev, coverage=cov, chart=chart)


GOOD_SUBMIT = {
    "status": "SPEC_INSUFFICIENT",
    "value": {},
    "reasoning": ("The specimen is a neoadjuvant resection and rule decision_rule.1 tells me "
                  "to code the site of origin, but nothing here says which of two synchronous "
                  "primaries is being reported."),
    "spec_section": "decision_rule",
    "spec_quote": REAL_SPEC_QUOTE,
    "uncovered_fields": ["primary_site"],
}


class ScriptedLLM(LLMClient):
    """Drives the real graph with fixed completions. Only the model boundary is stubbed —
    the toolbox, both ledgers, the gate and finalize all genuinely run."""

    def __init__(self, submit_args: dict | None, finalize: dict):
        super().__init__(LLMConfig(model="scripted/none", api_key="none"))
        self.submit_args = submit_args
        self.finalize = finalize
        self.rejections_seen: list[dict] = []

    def _reply(self, obj, calls=None):
        self.calls += 1
        self.prompt_tokens += 10
        self.completion_tokens += 5
        return LLMResponse(content=json.dumps(obj), tool_calls=calls or [],
                           prompt_tokens=10, completion_tokens=5)

    def chat(self, messages, tools=None):
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if tools is None:
            if "SUFFICIENT|CONTINUE|STUCK" in last:      # reflect: REPLAN is no longer a verdict a model may pick
                return self._reply({"verdict": "CONTINUE", "reason": "still gathering"})
            if "FOUND|EVIDENCE_INSUFFICIENT|SPEC_INSUFFICIENT" in last:
                return self._reply(self.finalize)
            return self._reply({"plan": [{"id": "1", "goal": "read the pathology",
                                          "rationale": "it would establish the answer"}]})
        for m in reversed(messages):
            if m.get("role") == "tool" and m.get("name") == "submit_answer":
                body = json.loads(m["content"])
                if body.get("accepted") is False:
                    self.rejections_seen.append(body)
                break
        if self.submit_args is None:
            return self._reply({}, [{"id": "c0", "name": "list_documents", "arguments": {}}])
        return self._reply({}, [{"id": "c1", "name": "submit_answer",
                                 "arguments": dict(self.submit_args)}])


def _run(spec, chart, tmp_path, submit_args, finalize=None, max_steps=4):
    """Drive the live runtime. ONE submission channel, so `finalize` folds into it.

    The old loop had two ways to produce an answer — the agent's `submit_answer` and a finalize
    prompt that could author one itself — and these tests used the second to exercise paths the
    first cannot reach (a runtime-forced rewrite, for instance). There is one channel now: an
    answer exists only if it went through the gate. So a script that only set `finalize` is
    submitting that, which is also the honest translation: whatever the run means to say, it says
    through `submit_answer`.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hooks_harness import run_with_script

    from acr.corpus import Corpus
    llm = ScriptedLLM(submit_args or finalize or {"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                                 "reasoning": "nothing established"},
                      finalize or {"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                   "reasoning": "nothing established"})
    manifest, _ = run_with_script(spec, Corpus(ROOT / "corpus" / "patients"), chart.patient_id,
                                  tmp_path, llm, run_id="spec-insufficient",
                                  max_model_calls=max_steps)
    return manifest, llm


def test_spec_insufficient_completes_and_writes_a_manifest(shb, chart, tmp_path):
    """The regression, end to end. Before the fix this raised CoverageClaimError out of
    `agent.run`, leaving a half-written trace and no manifest at all."""
    res, _ = _run(shb, chart, tmp_path, GOOD_SUBMIT)

    assert res["answer"]["status"] == "SPEC_INSUFFICIENT"
    manifests = list(Path(tmp_path).glob("*.manifest.json"))
    assert len(manifests) == 1, (
        "a SPEC_INSUFFICIENT run must leave a manifest. Without one, a run that crashed and a "
        "run that was never launched are the same thing in the run directory."
    )
    doc = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert doc["answer"]["status"] == "SPEC_INSUFFICIENT"
    assert doc["answer"]["spec_gap"], "the manifest must carry the report, not just the code"


def test_the_status_is_reached_through_the_gate_not_around_it(shb, chart, tmp_path):
    """A well-formed report is ACCEPTED by submit_answer — the channel is usable, not merely
    survivable. If this ever falls back to the finalize-authored path it still 'passes'
    superficially, so assert on the gate."""
    res, _ = _run(shb, chart, tmp_path, GOOD_SUBMIT)
    assert res["gate_validated"] is True
    assert res["answer"]["spec_gap"]["reported_by"] == "agent"
    assert res["answer"]["spec_gap"]["routable"] is True


# ======================================================== 2. the manifest says WHY
def test_the_report_names_the_field_the_section_the_words_and_the_rules(shb, chart, tmp_path):
    """A bare status code is useless to the loop that will read it. Each assertion below is
    one of the four things §6b's router needs before it can decide which text to change."""
    res, _ = _run(shb, chart, tmp_path, GOOD_SUBMIT)
    gap = res["answer"]["spec_gap"]

    # WHICH FIELD
    assert gap["uncovered_fields"] == ["primary_site"]
    assert gap["fields_scope"] == "named_fields"
    # WHICH PART OF THE SPEC, from a closed vocabulary — "the bit about staging" is not a
    # destination an optimizer can route to.
    assert gap["spec_section"] == "decision_rule"
    assert gap["spec_section"] in SPEC_SECTIONS
    # ITS OWN WORDS, verbatim and unsummarised.
    assert gap["agent_words"] == GOOD_SUBMIT["reasoning"]
    assert gap["agent_words_supplied"] is True
    # WHICH RULE. `invoked_rules` is what the agent cited; `section_rule_ids` is what existed
    # to cite. Keeping them apart is what distinguishes "no rule applied" from "no rule
    # exists", which are different findings about the spec.
    assert "decision_rule.1" in gap["invoked_rules"]
    assert gap["misattributed_rule_count"] == 0
    assert "decision_rule.2" in gap["section_rule_ids"]
    assert gap["section_rule_ids_available"] is True
    # The identity of the text being complained about, at the version complained about.
    assert gap["spec_id"] == shb.spec_id and gap["spec_hash"] == shb.spec_hash


def test_a_hallucinated_rule_id_is_never_promoted_to_a_citation(shb):
    """A gradient routed at a rule that does not exist would find the spec 'silent' about it
    and propose adding a rule that is already there."""
    gap, _ = build_spec_gap(
        shb, {"reasoning": "rules_applied: decision_rule.1, decision_rule.99",
              "spec_section": "decision_rule"},
        reported_by="agent", gate_validated=True)
    assert gap["invoked_rules"] == ["decision_rule.1"]
    assert gap["misattributed_rule_count"] == 1


def test_the_block_holds_a_count_not_a_list_of_invented_ids(shb):
    """`Tracer.rule_attribution` is the authority on misattributions and caps its list so a
    model emitting a thousand invented ids cannot grow a manifest without bound. A second
    unbounded copy here would defeat that cap and give two numbers free to disagree."""
    flood = " ".join(f"decision_rule.{i}" for i in range(100, 400))
    gap, _ = build_spec_gap(shb, {"reasoning": flood, "spec_section": "decision_rule"},
                            reported_by="agent", gate_validated=True)
    assert gap["misattributed_rule_count"] == 300
    assert not any(isinstance(v, list) and len(v) > 50 for v in gap.values())


def test_a_report_that_cannot_be_routed_says_so_and_is_counted(shb, chart, tmp_path):
    """`finalize` authors the answer when the agent never submitted, and there is no loop
    left to reject into. It must not crash, must not invent a section, and must not look
    like a clean report — the last of those is how this defect returns."""
    res, _ = _run(shb, chart, tmp_path, submit_args=None,
                  finalize={"status": "SPEC_INSUFFICIENT", "value": {},
                            "reasoning": "the spec does not address this"})
    gap = res["answer"]["spec_gap"]
    assert gap["spec_section"] == AC.SPEC_SECTION_UNATTRIBUTED
    assert gap["routable"] is False
    assert res["degradation"]["spec_gaps_unroutable"] == 1, (
        "an unroutable report is the channel half-working, and half-working has historically "
        "been read as working. It belongs in the degradation block."
    )


def test_a_runtime_forced_abstention_is_not_filed_as_an_agent_signal(outside, chart, tmp_path):
    """STORE.610 returns SPEC_INSUFFICIENT for every chart by design. Pooling those with real
    agent reports would bury 38 runs' worth of signal under a constant — and this spec was
    ALSO the one that crashed on 100% of its runs."""
    res, _ = _run(outside, chart, tmp_path, submit_args=None,
                  finalize={"status": "FOUND", "value": {"class_of_case": "10"},
                            "reasoning": "coded from the face sheet"})
    ans = res["answer"]
    assert ans["status"] == "SPEC_INSUFFICIENT"
    assert ans["remedy_class"] == AC.REMEDY_WRONG_DATA_SOURCE
    assert ans["spec_gap"]["reported_by"] == "runtime"
    assert ans["spec_gap"]["spec_section"] == "data_source"
    assert ans["spec_gap"]["forced_over_status"] == "FOUND"
    assert list(Path(tmp_path).glob("*.manifest.json")), "and it still writes its manifest"


def test_an_excluded_case_is_not_filed_as_a_gap(shb):
    """`when_not_to_use` firing is the specification WORKING. Filing it as a gap would have
    the optimizer 'fix' deliberate exclusions."""
    _, remedy = build_spec_gap(shb, {"reasoning": "explicitly excluded",
                                     "spec_section": "when_not_to_use"},
                               reported_by="agent", gate_validated=True)
    assert remedy == AC.REMEDY_CASE_EXCLUDED


# ======================================================== 3. the category error, both ways
def test_a_coverage_claim_on_spec_insufficient_is_still_refused():
    """The fix is the CATEGORY, not the crash. Coverage describes how well this chart was
    searched; SPEC_INSUFFICIENT is not about this chart. The claim must still be impossible."""
    with pytest.raises(CoverageClaimError, match="only a gate-validated"):
        assert_answer_is_reportable({"status": "SPEC_INSUFFICIENT",
                                     "coverage_attested": {"mode": "stratified_exclusion"}})


def test_finalize_does_not_route_spec_insufficient_into_the_coverage_branch():
    """Structural, because the behavioural test above only covers the paths someone thought
    of. The crash was one `elif` ordering: SPEC_INSUFFICIENT fell into the gate-validated
    negative branch and was handed a ledger."""
    fin = inspect.getsource(G.ChartReviewAgent._n_finalize)
    i_spec = fin.index('ans.get("status") == "SPEC_INSUFFICIENT"')
    i_cov = fin.index('elif s.get("gate_validated"):')
    assert i_spec < i_cov, (
        "SPEC_INSUFFICIENT must be handled before the gate-validated negative branch, or it "
        "is handed a coverage ledger it may not carry and the run dies at emission"
    )
    # Comments stripped first: the branch explains in prose that it withholds these, and a
    # naive substring check would trip over its own documentation.
    spec_branch = "\n".join(ln for ln in fin[i_spec:i_cov].splitlines()
                            if not ln.strip().startswith("#"))
    assert "coverage_attested" not in spec_branch
    assert "negative_basis" not in spec_branch, (
        "SPEC_INSUFFICIENT is not a negative about the chart and must not borrow its vocabulary"
    )


def test_a_bare_status_code_is_refused_at_emission():
    """Exempting SPEC_INSUFFICIENT from the coverage gate without imposing the spec_gap
    obligation would turn the crash into a shrug."""
    with pytest.raises(SpecGapError, match="spec_gap block"):
        assert_spec_gap_is_reported({"status": "SPEC_INSUFFICIENT"})


@pytest.mark.parametrize("mutate, match", [
    (lambda g, a: g.pop("reported_by"), "reported_by"),
    (lambda g, a: g.update(spec_section="the staging bit"), "spec_section"),
    (lambda g, a: g.pop("routable"), "routable"),
    (lambda g, a: g.pop("agent_words"), "agent_words"),
    (lambda g, a: a.update(remedy_class="LOOKS_FINE"), "remedy_class"),
])
def test_each_routing_field_is_individually_required(shb, mutate, match):
    gap, remedy = build_spec_gap(shb, GOOD_SUBMIT, reported_by="agent", gate_validated=True)
    ans = {"status": "SPEC_INSUFFICIENT", "spec_gap": gap, "remedy_class": remedy}
    assert_spec_gap_is_reported(ans)          # intact, it passes
    mutate(gap, ans)
    with pytest.raises(SpecGapError, match=match):
        assert_spec_gap_is_reported(ans)


def test_the_other_two_statuses_are_untouched_by_the_spec_gap_rule():
    """A guard that fires on the wrong status is a new bug, not a fix."""
    assert_answer_is_reportable({"status": "FOUND", "proof_basis": "WITNESS",
                                 "witness_count": 2, "value": {"histology": "8140"}})
    assert_answer_is_reportable({"status": "EVIDENCE_INSUFFICIENT",
                                 "negative_basis": "GATE_VALIDATED",
                                 "coverage_attested": {"mode": "stratified_exclusion"}})


# ======================================================== 4. REFUSE THE ABUSE
def test_spec_insufficient_may_not_carry_a_value_past_the_gate(shb, chart):
    """The exit an agent will take if you leave it open: it cannot meet the FOUND standard,
    so it declares the specification inadequate and ships the code anyway."""
    v = _gate(shb, chart, dict(GOOD_SUBMIT, value={"primary_site": "C349", "histology": "8046"}))
    assert v["accepted"] is False
    assert "cannot carry a coded value" in v["why"]
    blob = " ".join(v["missing"])
    assert "primary_site" in blob and "histology" in blob
    assert "FOUND" in blob, (
        "the rejection must say what to do instead. A rejection the agent cannot act on is "
        "how one real run burned a 400k-token budget without revising."
    )


def test_the_smuggled_value_would_have_been_read_as_an_established_fact(shb, chart):
    """Why the refusal above matters, demonstrated rather than asserted.

    `concordance.variables_from_answer` promotes ANY populated field to FOUND regardless of
    the answer's status — deliberately, because a spec may license reporting a field while
    abstaining overall. So a value riding on SPEC_INSUFFICIENT does not merely survive: it
    enters L4 as an established value, having met no proof standard at all.
    """
    smuggled = {"status": "SPEC_INSUFFICIENT", "value": {"primary_site": "C349"}}
    flat = variables_from_answer(smuggled, ["primary_site"], source="t")
    assert flat["primary_site"].status == "FOUND", (
        "this is the downstream behaviour the gate must protect, not a bug in the flattener"
    )

    v = _gate(shb, chart, dict(GOOD_SUBMIT, value={"primary_site": "C349"}))
    assert v["accepted"] is False


def test_the_forced_path_strips_a_value_it_cannot_reject(outside, chart, tmp_path):
    """`data_source: outside_notes` rewrites a FOUND into SPEC_INSUFFICIENT after the gate has
    already run, so there is no rejection available. It kept the value — the same smuggling
    route, opened by the runtime itself rather than by the agent."""
    res, _ = _run(outside, chart, tmp_path, submit_args=None,
                  finalize={"status": "FOUND", "value": {"class_of_case": "10"},
                            "reasoning": "coded from the face sheet"})
    ans = res["answer"]
    assert ans["value"] == {}
    assert ans["value_withheld"] == ["class_of_case"], (
        "dropped, and SAID to be dropped — a silent strip is its own small lie"
    )
    assert "class_of_case" in ans["value_withheld_why"] or ans["value_withheld_why"]
    assert variables_from_answer(ans, ["class_of_case"], source="t")[
        "class_of_case"].status == "SPEC_INSUFFICIENT"


def test_an_emitted_answer_can_never_hold_both_the_status_and_a_value(shb):
    """Belt and braces at the emission point, since two paths reach it without the gate."""
    gap, remedy = build_spec_gap(shb, GOOD_SUBMIT, reported_by="agent", gate_validated=True)
    with pytest.raises(SpecGapError, match="populated value"):
        assert_spec_gap_is_reported({"status": "SPEC_INSUFFICIENT", "spec_gap": gap,
                                     "remedy_class": remedy,
                                     "value": {"primary_site": "C349"}})


# ======================================================== the gate's other refusals
def test_an_unnamed_section_is_rejected_recoverably(shb, chart):
    """'The spec is unclear' with no destination is the path of least resistance §6b predicts,
    and it cannot be routed anywhere."""
    v = _gate(shb, chart, {"status": "SPEC_INSUFFICIENT", "reasoning": "it does not say"})
    assert v["accepted"] is False
    blob = " ".join(v["missing"])
    assert "spec_section" in blob and "decision_rule" in blob, (
        "the rejection has to list the legal sections, or the agent cannot comply"
    )


def test_an_invented_section_is_rejected(shb, chart):
    v = _gate(shb, chart, dict(GOOD_SUBMIT, spec_section="vibes"))
    assert v["accepted"] is False and "vibes" in " ".join(v["missing"])


def test_silence_is_rejected(shb, chart):
    v = _gate(shb, chart, dict(GOOD_SUBMIT, reasoning=""))
    assert v["accepted"] is False
    assert "own words" in " ".join(v["missing"])


def test_a_fabricated_spec_quote_is_rejected(shb, chart):
    """refine.py's citation mask is the one thing standing between a plausible rewrite and an
    unjustified spec edit. A quote that is not in the spec would sail straight through it."""
    v = _gate(shb, chart, dict(GOOD_SUBMIT,
                               spec_quote="Always code the biopsy site when in doubt."))
    assert v["accepted"] is False
    assert "does not appear in the specification" in " ".join(v["missing"])


def test_omitting_the_quote_is_legitimate(shb, chart):
    """A GAP is precisely the case where no sentence exists to quote. Requiring one would make
    the most valuable report impossible to file."""
    sub = dict(GOOD_SUBMIT)
    sub.pop("spec_quote")
    assert _gate(shb, chart, sub)["accepted"] is True
    gap, _ = build_spec_gap(shb, sub, reported_by="agent", gate_validated=True)
    assert gap["spec_quote"] is None


def test_a_field_the_spec_does_not_declare_is_rejected(shb, chart):
    v = _gate(shb, chart, dict(GOOD_SUBMIT, uncovered_fields=["laterality"]))
    assert v["accepted"] is False and "laterality" in " ".join(v["missing"])


def test_the_report_is_not_asked_to_prove_coverage(shb, chart):
    """The whole point. An untouched chart fails the coverage gate outright — and must not
    block a statement that is not about the chart."""
    ev, cov = _ledgers(shb, chart)
    assert G.check_gate(shb, cov).verdict == "FAIL", "precondition: coverage is not satisfied"
    v = gate_answer(shb, GOOD_SUBMIT, evidence=ev, coverage=cov, chart=chart)
    assert v["accepted"] is True, (
        "no amount of reading a chart can make a silent specification speak; demanding a "
        "coverage proof here is the category error that crashed the run"
    )


def test_evidence_insufficient_still_needs_the_gate(shb, chart):
    """The exemption must not leak to the status that IS a claim about coverage."""
    v = _gate(shb, chart, {"status": "EVIDENCE_INSUFFICIENT", "value": {},
                           "reasoning": "nothing found"})
    assert v["accepted"] is False


# ======================================================== 3. every front end
def test_the_mcp_surface_reports_it_too(shb, chart, tmp_path):
    """The MCP path never crashed — it signed a bare status code, and on the forced path it
    signed the caller's value with it. Different symptom, same missing channel."""
    from acr.mcp_server import ChartReviewService

    svc = ChartReviewService(str(CORPUS), str(ROOT / "specs"))
    plan = svc.call("coverage.plan", {"patient": "SYN0002", "spec_id": shb.spec_id})
    rid = plan["run_id"]

    bad = svc.call("gate.check", {"run_id": rid,
                                  "answer": dict(GOOD_SUBMIT, value={"primary_site": "C349"})})
    assert bad["verdict"] == "FAIL" and bad["answer"] is None

    ok = svc.call("gate.check", {"run_id": rid, "answer": dict(GOOD_SUBMIT)})
    assert ok["verdict"] == "PASS", ok
    gap = ok["answer"]["spec_gap"]
    assert gap["spec_section"] == "decision_rule"
    assert gap["uncovered_fields"] == ["primary_site"]
    assert gap["agent_words"] == GOOD_SUBMIT["reasoning"]
    assert "decision_rule.1" in gap["invoked_rules"]
    assert "coverage_attested" not in ok["answer"]
    assert ok["answer"]["remedy_class"] == AC.REMEDY_SPEC_DOES_NOT_COVER


def test_the_mcp_surface_will_not_sign_a_caller_supplied_gap_block(shb):
    """The block is ASSEMBLED by the server from inputs it validated. A caller-supplied one
    would be a report whose quote nobody checked, which is precisely what the citation mask
    downstream assumes has happened."""
    from acr.mcp_server import ChartReviewService

    svc = ChartReviewService(str(CORPUS), str(ROOT / "specs"))
    rid = svc.call("coverage.plan", {"patient": "SYN0002",
                                     "spec_id": shb.spec_id})["run_id"]
    out = svc.call("gate.check", {"run_id": rid, "answer": dict(
        GOOD_SUBMIT, spec_gap={"spec_section": "decision_rule", "routable": True,
                               "agent_words": "trust me"})})
    assert "spec_gap" in out["ignored_client_claims"]
    assert out["answer"]["spec_gap"]["agent_words"] == GOOD_SUBMIT["reasoning"]


def test_the_mcp_forced_path_also_strips_the_value(outside):
    """The forced rewrite runs after the gate on this surface too, so the gate's refusal is
    unavailable and the value has to be taken away at emission."""
    from acr.mcp_server import ChartReviewService

    svc = ChartReviewService(str(CORPUS), str(ROOT / "specs"))
    rid = svc.call("coverage.plan", {"patient": "SYN0002",
                                     "spec_id": outside.spec_id})["run_id"]
    hits = svc.call("chart.search", {"patient": "SYN0002", "query": "carcinoma",
                                     "max_hits": 1})["hits"]
    out = svc.call("gate.check", {"run_id": rid, "answer": {
        "status": "FOUND", "value": {"class_of_case": "10"},
        "reasoning": "coded it anyway",
        "evidence": [{"note_id": hits[0]["note_id"], "start": hits[0]["start"],
                      "end": hits[0]["end"], "supports": "class_of_case"}]}})
    assert out["verdict"] == "PASS", out
    ans = out["answer"]
    assert ans["status"] == "SPEC_INSUFFICIENT"
    assert ans["value"] == {} and ans["value_withheld"] == ["class_of_case"]
    assert ans["spec_gap"]["reported_by"] == "runtime"


def test_both_front_ends_use_the_one_builder_and_the_one_assertion():
    """Structural, and the reason is in the bug: the same status meant three different things
    on three runtimes — a crash on the hand-written loop, a bare code on MCP, a bare code plus
    an unearned coverage ledger on deepagents. Nobody noticed, because each surface looked fine
    alone.

    TWO front ends now, not three: the hand-written loop is gone and `agent.py` is the runtime.
    The count is not the point — the point is that every surface which can emit an answer goes
    through the same three calls. Assert on the shared call, not on the output shape: correct
    copies today are copies free to drift tomorrow, and the drift is invisible from inside any
    one of them.
    """
    import acr.agent as A
    import acr.mcp_server as M

    for mod in (A, M):
        src = inspect.getsource(mod)
        assert "build_spec_gap" in src, f"{mod.__name__} cannot assemble a spec gap"
        assert "strip_value_from_spec_insufficient" in src, (
            f"{mod.__name__} does not refuse the value-smuggling exit")
        assert "assert_answer_is_reportable" in src, (
            f"{mod.__name__} emits answers without the emission-time checks")


def test_the_manifest_no_longer_attests_coverage_unconditionally():
    """A manifest that carried a coverage ledger for every status claimed the universe had been
    searched on runs that never searched it — including SPEC_INSUFFICIENT, which is not a
    statement about this chart at all. Same category error as the crash, silent because nothing
    checked it. Asserted against `agent.run_chart_review`, which is where the branch now lives.
    """
    import acr.agent as A

    src = inspect.getsource(A.run_chart_review)
    i = src.index("attach_coverage_claim")
    window = src[max(0, i - 400):i + 200]
    assert 'answer.get("status") == "EVIDENCE_INSUFFICIENT"' in window, (
        "the ledger must be conditional on the one status that earns it")


def test_every_front_end_offers_the_reporting_fields_to_the_model():
    """A gate that demands `spec_section` from a model that was never given the parameter is an
    unwinnable loop: it would be rejected forever for omitting something it cannot send."""
    from acr.tools.toolbox import TOOL_SCHEMAS

    submit = next(s for s in TOOL_SCHEMAS if s["function"]["name"] == "submit_answer")
    props = submit["function"]["parameters"]["properties"]
    for k in ("spec_section", "spec_quote", "uncovered_fields"):
        assert k in props, f"the toolbox cannot send {k}"

    from acr.mcp_server import MCP_TOOLS
    desc = next(t for t in MCP_TOOLS if t["name"] == "gate.check")["inputSchema"]
    assert "spec_section" in json.dumps(desc), "the MCP caller is never told to send it"


def test_the_toolbox_passes_the_fields_through_rather_than_dropping_them():
    """Dropping them silently would make the gate's rejection unanswerable."""
    from acr.corpus import Corpus as C
    from acr.tools import Toolbox

    ch = C(CORPUS).chart("SYN0002")
    spec = load_spec(SHB)
    ev, cov = _ledgers(spec, ch)
    tb = Toolbox(ch, ev, cov)
    out, _ = tb.dispatch("submit_answer", dict(GOOD_SUBMIT))
    assert out["received"] is True
    assert tb.submitted["spec_section"] == "decision_rule"
    assert tb.submitted["spec_quote"] == REAL_SPEC_QUOTE
    assert tb.submitted["uncovered_fields"] == ["primary_site"]


# ======================================================== the CLI leg
def test_extract_writes_the_gap_into_the_artifact_and_exits_clean(monkeypatch, tmp_path):
    """`cli.extract` caught the crash and wrote the run off as an error, so the cohort artifact
    the §6b loop reads never contained a single spec gap."""
    from typer.testing import CliRunner

    from acr.cli import app

    llm = ScriptedLLM(GOOD_SUBMIT, {"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                    "reasoning": "n/a"})
    monkeypatch.setattr("acr.cli_common.llm_client", lambda *a, **k: llm)
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\n", encoding="utf-8")

    r = CliRunner().invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                                 "--variables", "primary_site",
                                 "--max-steps", "4", "--seed", "7",
                                 "--out", str(tmp_path / "runs")])
    assert r.exit_code == 0, r.output
    (path,) = list((tmp_path / "runs").glob("extract__*/extract.json"))
    doc = json.loads(path.read_text(encoding="utf-8"))

    assert doc["n_failed_runs"] == 0
    row = doc["patients"][0]["runs"][0]
    assert row["status"] == "SPEC_INSUFFICIENT"
    assert row["spec_gap"]["spec_section"] == "decision_rule"
    assert row["remedy_class"] == AC.REMEDY_SPEC_DOES_NOT_COVER
    # Countable at the top level, so "zero spec gaps" is a number someone looked at rather
    # than an absence nobody noticed for 38 runs.
    assert doc["spec_gaps"]["total"] == 1
    assert doc["spec_gaps"]["agent_reported"] == 1
    assert doc["spec_gaps"]["unroutable"] == 0
    assert doc["spec_gaps"]["by_section"] == {"decision_rule": 1}


def test_a_crashed_run_is_distinguishable_from_one_that_never_happened(monkeypatch, tmp_path):
    """The other half of the original defect. Whatever raises next, the run directory must
    record that something was attempted and died."""
    from typer.testing import CliRunner

    from acr.cli import app

    class Boom(ScriptedLLM):
        def chat(self, messages, tools=None):
            raise RuntimeError("simulated provider failure")

    monkeypatch.setattr("acr.cli_common.llm_client", lambda *a, **k: Boom(None, {}))
    (tmp_path / "c.csv").write_text("patient_id\nSYN0001\n", encoding="utf-8")

    r = CliRunner().invoke(app, ["extract", "--cohort", str(tmp_path / "c.csv"),
                                 "--variables", "primary_site", "--max-steps", "2",
                                 "--out", str(tmp_path / "runs")])
    assert r.exit_code == 1
    stubs = list((tmp_path / "runs").glob("extract__*/*.failed.json"))
    assert len(stubs) == 1, "a crashed run must leave a mark on disk"
    stub = json.loads(stubs[0].read_text(encoding="utf-8"))
    assert stub["outcome"] == "RUN_RAISED" and stub["answer"] is None
    assert not list((tmp_path / "runs").glob("extract__*/*.manifest.json")), (
        "and the mark must NOT be a manifest — it is not an answer"
    )
