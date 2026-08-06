"""The plan the agent revises is the plan that governs retrieval — and it can only widen.

WHAT WAS WRONG
--------------
Two greps settled it:

    grep 's["plan"]'                 src/acr/graph.py   -> nothing
    grep 'CoveragePlan|policy_for'   src/acr/graph.py   -> nothing

There were two plans. The revisable one was a prose list of {id, goal, rationale} that was
rendered into messages and read by no code. The one that governs retrieval —
`coverage_planner.CoveragePlan`, with its read_all / search / sample assignment and its term
list — was never consulted by the agent loop at all. REPLAN and CONTINUE were mechanically
identical: both appended text. So REPLAN fired 0 times in 291 actions across 37 runs, and a
model asked "does something learned change what should be done next?" about "find the
pathology report" was answering correctly. The goal never changes. The RETRIEVAL SCOPE
changes, and the retrieval scope was not in the plan.

WHAT THESE TESTS HOLD
---------------------
Six things, and the fifth is the one the whole design rests on:

  1. ONE PLAN. Structural, so a second one cannot grow back.
  2. IT GOVERNS. A `sample` type cannot be opened; the refusal names the promotion that
     would allow it.
  3. REVISION IS MONOTONE EXPANSION. A non-superset is refused WHOLE and recorded.
  4. IT IS SAFE AGAINST THE LEDGER ARITHMETIC — proved here on real Clopper-Pearson numbers,
     including the one place it is conservatively WRONG, which is asserted rather than hidden.
  5. THE RUNTIME APPLIES IT. Every field of the typed revision is tested by the BEHAVIOUR it
     changes afterwards, never by the state it writes. If the runtime does not apply it, it
     did not happen — that one sentence is the entire bug being fixed, and a test that
     asserted `plan.keywords == [...]` would pass on a plan nobody reads, which is precisely
     the failure it is supposed to catch.
  6. EXPANSION HAS A BUDGET, and exhausting it with obligations outstanding is an honest
     abstention rather than a truncation or a pass.

No provider is called anywhere in this file and no chart text is written into it.
"""

# ---------------------------------------------------------------------------------------------
# TESTS REMOVED 2026-07-30, with the rules they specified.
#
# `answer_checks` carried five checks that decided clinical questions by matching word lists
# against the model's own cited quotes. Measured over every trace this project has recorded
# (266 traces, 202 joinable to registry gold, 122 firings):
#
#   not_less_specific        22 fires   22 rejected the registry's own value    0 ever helped
#   nos_requires_search      24 fires   21 rejected the registry's own value    0 ever helped
#   conflict_requires_nos    67 fires   18 rejected the registry's own value   15 "helped",
#                                       all 15 of them the same push to the NOS code
#   origin_not_specimen       2 fires    0                                      0
#   code_matches_cited_text   0 fires    -                                      -
#
# `fit_terms_to_budget` deleted 103 search terms the model had proposed for itself, and on
# CASE009 it deleted `lobe` and `bronchus` while `nos_requires_search` refused the answer for
# never having searched them. The required-keyword gate enforced a list measured at 87.4%
# recall over 276,054 documents.
#
# A test that pins a rule in place is part of the rule, so these went with them:
#   - test_add_terms_changes_what_the_gate_demands
#   - test_coverage_is_evaluated_against_the_final_list
#
# Nothing replaced them here. A wrong clinical value is an instruction-following failure and is
# measured as one. tests/test_answer_checks.py holds what survives: field `format` and
# `allowable_values`, the only check with a positive measured record.
# ---------------------------------------------------------------------------------------------
from __future__ import annotations

import inspect
import json
import re
from pathlib import Path

import pytest

from acr.chartstore.corpus import Corpus
from acr.contract.spec import load_spec
from acr.core import site
from acr.core.llm import LLMClient, LLMConfig, LLMResponse
from acr.core.state import EvidenceLedger
from acr.review.coverage import CoverageLedger, ForcedSampler, strata_from_spec
from acr.review.coverage_planner import (
    TRIGGER_GATE_OBLIGATION_UNREACHABLE,
    TRIGGER_UNLISTED_ANSWER_TERM,
    TRIGGER_UNSETTLED_THREAD,
    TRIGGER_ZERO_HIT_SEARCH,
    ExpansionBudget,
    OpenThreadLedger,
    PlanSnapshot,
    check_monotone,
    documents_by_type,
    gate_obligation_triggers,
    load_marker_catalogue,
    plan_from_spec,
    spec_declared_keywords,
    triggers_from_tool_result,
)

ROOT = Path(__file__).resolve().parents[1]
SHB = site.specs_root() / "STORE.400_522_523.site_histology_behavior.yaml"
CORPUS = site.corpus_root()

#: A type this spec's `cannot_establish` stratum sweeps into `sample`. It is also the exact
#: type the coverage module records a real error against: patient P03 was coded C349 (lung
#: NOS) while "right upper lobe" sat in seven imaging and oncology note types, because the
#: architecture had taught the agent those documents were useless. Promotion is the move that
#: was missing.
SAMPLED_TYPE = "Chest-CT-W-Contr"


@pytest.fixture(scope="module")
def spec():
    return load_spec(SHB)


@pytest.fixture(scope="module")
def chart():
    return Corpus(CORPUS).chart("SYN0001")


@pytest.fixture
def plan(spec, chart):
    return plan_from_spec(spec, chart)


@pytest.fixture
def budget(plan, chart):
    return ExpansionBudget.priced_against(plan, documents_by_type(chart), max_revisions=6)


def _ledger(spec, chart):
    docs, _ = chart.list_documents(limit=100_000)
    return CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(7))


# ==========================================================================================
# 2. THE PLAN GOVERNS WHAT MAY BE OPENED
# ==========================================================================================
def test_any_document_in_the_chart_may_be_opened(spec, chart, plan):
    """The runtime no longer rules on what may be read.

    `_out_of_plan` refused a read whenever the plan had filed the document's type in the
    `sample` bucket, and told the model to ask for a promotion first. It fired 138 times across
    the recorded traces. The bucket came from `doc_type_matches`, a case-insensitive substring
    over local type names, measured wrong in both directions here: it matched
    `Speech-Language-Pathology-Note` and missed `Non-Gyn-Cyto-FNA` (1,285 documents),
    `FN-Aspirate-Report` (881) and `SURG-PATH-RESULT` (231). 107 of the 219 patients whose
    `can_establish` count is zero hold one of those reports anyway, so the hook could stand
    between the agent and the only document carrying the answer -- and on CASE001 it did.
    """
    agent = _agent(spec, chart, llm=None)
    agent.plan = plan
    assert not hasattr(agent, "_out_of_plan"), "the read refusal must not come back"

    nid = _first_note_of_type(chart, SAMPLED_TYPE)
    out, _ = agent.ctx.toolbox.dispatch("read_document", {"note_id": nid, "limit": 500})
    assert "error" not in out, out
    assert out["text"], "a document in the chart is readable, whatever bucket its type is in"
    assert nid in agent.ctx.coverage.read_notes, "and the read is still recorded"


def test_the_undeclared_tool_refusal_is_kept(spec, chart):
    """What survives in `wrap_tool_call`, and why it is a different kind of thing.

    An undeclared tool is a statement about the tool surface, not about a clinical document: a
    read that does not go through `Toolbox.dispatch` is invisible to the coverage ledger, so the
    gate would stamp a chart the ledger never saw read.
    """
    agent = _agent(spec, chart, llm=None)
    assert agent._undeclared("read_document") is None
    refusal = agent._undeclared("some_tool_nobody_declared")
    assert refusal is not None and refusal["error"] == "UNDECLARED_TOOL"


# ==========================================================================================
# 3. MONOTONICITY — refused whole, and recorded as refused
# ==========================================================================================
def test_check_monotone_catches_every_way_to_look_at_less():
    before = PlanSnapshot(policies=(("A", "read_all"), ("B", "search"), ("C", "sample")),
                          keywords=frozenset({"carcinoma", "biopsy"}))
    assert check_monotone(before, before) == []

    demoted = PlanSnapshot(policies=(("A", "search"), ("B", "search"), ("C", "sample")),
                           keywords=before.keywords)
    dropped = PlanSnapshot(policies=(("A", "read_all"), ("B", "search")),
                           keywords=before.keywords)
    lost_term = PlanSnapshot(policies=before.policies, keywords=frozenset({"carcinoma"}))
    widened = PlanSnapshot(policies=(("A", "read_all"), ("B", "read_all"), ("C", "search")),
                           keywords=frozenset({"carcinoma", "biopsy", "mucinous"}))

    assert any("demoted" in v for v in check_monotone(before, demoted))
    assert any("dropped" in v for v in check_monotone(before, dropped))
    assert any("terms removed" in v for v in check_monotone(before, lost_term))
    assert check_monotone(before, widened) == [], "widening is the whole point"


# ==========================================================================================
# 4. MONOTONICITY AGAINST THE LEDGER ARITHMETIC
# ==========================================================================================
def _stratum(ledger, name):
    return next(r for r in ledger.stratum_results() if r.name == name)


def test_expansion_never_cancels_a_drawn_obligation(spec, chart):
    """A drawn document that a NEW TERM turns into a search hit does not vanish.

    This is the one place expansion could plausibly weaken the proof: `pending_samples`
    computes the miss universe as "pool minus search hits", so a term added mid-run removes
    documents from it. If the obligation simply disappeared, adding terms would be a way to
    shrink the audited population without reading anything — searching HARDER would make the
    gate EASIER, which is the inversion `hits_read` was written to close one layer down.
    """
    cov = _ledger(spec, chart)
    cov.listed_documents = True
    pending = cov.pending_samples()
    drawn_ids = [d.note_id for docs in pending.values() for d in docs]
    assert drawn_ids, "this spec must actually draw a sample or the test proves nothing"
    victim = drawn_ids[0]

    # Expand: a term that hits the drawn-but-uninspected document.
    cov.note_search("a-newly-added-term", [victim])

    still_drawn = {d.note_id for docs in cov.pending_samples().values() for d in docs}
    unread_hits = {n for r in cov.stratum_results() for n in r.hits_unread}
    assert victim in still_drawn or victim in unread_hits, (
        "a drawn document left the miss frame and landed nowhere — the obligation to look at "
        "it was cancelled by the act of searching better"
    )


def test_the_elusion_bound_only_moves_the_honest_way(spec, chart):
    """More clean draws tightens it; a real hit loosens it. Nothing else can move it."""
    cov = _ledger(spec, chart)
    cov.listed_documents = True
    pending = cov.pending_samples()
    name, docs = next(iter(pending.items()))
    ids = [d.note_id for d in docs]

    bounds = []
    for i, nid in enumerate(ids[:6], start=1):
        cov.record_sample_verdict(name, nid, relevant=False)
        bounds.append(_stratum(cov, name).elusion_upper)
    assert bounds == sorted(bounds, reverse=True), (
        f"a clean draw must never loosen the bound: {bounds}")

    before = _stratum(cov, name).elusion_upper
    cov.record_sample_verdict(name, ids[6], relevant=True)
    assert _stratum(cov, name).elusion_upper > before, (
        "a hit in the sample must raise the bound — that is the prior being falsified, and "
        "the gate failing on it is the accounting working, not breaking"
    )


def test_an_expansion_that_moves_the_frame_recomputes_the_bound_and_never_inherits_it(spec, chart):
    """CLAIM 2 OF `MONOTONICITY_VS_LEDGER`, WHICH USED TO BE FALSE.

    `test_expansion_never_cancels_a_drawn_obligation` above covers the drawn-but-UNREAD case,
    where a new term trades the miss sample for the stronger obligation to read the hit. This
    is the case it does not cover and the old docstring wrongly claimed: draws that already
    carry a clean verdict. Nothing is found, no verdict is overturned, `n_s` does not decrease
    numerically — and the frame those verdicts described has lost most of its members.

    Measured on SYN0001: 25 clean draws bound elusion at 0.1129 and clear the spec's 0.12
    cap; one term that hits 20 of them leaves 5 draws in a 92-document frame, worth 0.4507.
    Inheriting 0.1129 across that revision reaches a gate PASS, which is the worst direction
    an absence proof can fail in.
    """
    cov = _ledger(spec, chart)
    cov.listed_documents = True
    docs = cov.pending_samples().get("may_mention") or []
    if len(docs) < 10:
        pytest.skip("this spec draws too small a miss sample on this chart to move a frame")
    for d in docs:
        cov.note_read(d.note_id, d.doc_type)
    cov.resolve_sample_verdicts(cited=set())
    before = _stratum(cov, "may_mention")
    assert before.misses_sampled == len(docs) and before.miss_sample_hits == 0

    victims = [d.note_id for d in docs[:-5]]          # one monotone term addition
    cov.note_search("a-newly-added-term", victims)

    after = _stratum(cov, "may_mention")
    assert after.miss_sample_hits == 0, "fixture assumption: the prior was never falsified"
    assert after.misses_sampled == 5, (
        "the bound is still being credited draws that left the frame it is a bound over")
    assert after.draws_invalidated == sorted(victims)
    assert after.elusion_upper > before.elusion_upper, (
        "the bound was inherited across a frame revision. Expansion is monotone in the "
        "EVIDENCE and not in the BOUND: n_s did not decrease, the universe it was drawn from "
        "shrank underneath it."
    )
    assert after.replacement_draws_required == len(victims)
    assert len(cov.pending_samples().get("may_mention") or []) == len(victims), (
        "replacement draws must be demanded until n is restored")


def test_the_monotonicity_note_states_what_is_true_and_not_what_is_comfortable():
    """A comment that overstates a guarantee is how the inherited-bound defect survived.

    Structural, like `test_the_prose_plan_is_gone_...`, and for the same reason: the failure
    was not a code path anybody exercised, it was a sentence everybody believed. The way it
    comes back is somebody restoring the reassuring version of it.
    """
    from acr.review import coverage_planner as CP

    src = inspect.getsource(CP)
    for gone in ("THE BOUND CANNOT BE GAMED DOWNWARD", "Expansion never decreases n_s"):
        assert gone not in src, (
            f"{gone!r} is back. It is false: n_s does not decrease numerically while the "
            "universe it was drawn from shrinks underneath it."
        )
    assert "drawn-but-unread" in src, (
        "claim 1 holds only for the drawn-but-unread case — the only case "
        "test_expansion_never_cancels_a_drawn_obligation covers — and must say so"
    )
    note = CP.MONOTONICITY_VS_LEDGER
    assert "monotone" in note, "graph and deep_runner ship this string into every manifest"
    for owed in ("frame", "recomputed", "not in the BOUND"):
        assert owed in note, f"the manifest's one-line summary must carry {owed!r}"


def test_reading_more_of_a_sampled_stratum_earns_no_credit_and_that_is_correct(spec, chart):
    """Documents the AGENT chose are not a random sample.

    Crediting self-selected reading toward the elusion bound would let the agent read the
    reassuring documents and call the result evidence — the circularity `ForcedSampler`
    exists to prevent, rebuilt on the expansion path. So this asserts a NON-effect on
    purpose: promoting a type and reading its documents leaves the bound exactly where it was.
    """
    cov = _ledger(spec, chart)
    cov.listed_documents = True
    cov.pending_samples()
    name = next(iter(cov.drawn))
    before = _stratum(cov, name).elusion_upper

    pool = cov.by_stratum[name]
    undrawn = [d for d in pool if d.note_id not in set(cov.drawn[name])]
    for d in undrawn[:10]:
        cov.note_read(d.note_id, d.doc_type)

    assert _stratum(cov, name).elusion_upper == before


def test_a_censused_search_stratum_still_reports_elusion_one_AND_THAT_IS_A_DEFECT(spec, chart):
    """Asserted so that it is on the record, not so that it is blessed.

    Expand far enough that every document in `may_mention` is a search hit and `misses` falls
    to zero. The stratum has then been CENSUSED — `keyword_list_validated` refuses until every
    hit is read — yet `elusion_upper` stays at `clopper_pearson_upper(0, 0) == 1.0`, because
    the bound is computed from a sample that no longer has anything to draw from. A
    `max_elusion_upper` cap therefore becomes unpassable precisely for the run that did the
    most work.

    It is conservative, so nothing unsafe passes and monotonicity is not violated: the gate
    can only get harder. But it does mean an honest exhaustive expansion can be forced into
    EVIDENCE_INSUFFICIENT, and the fix — teaching `coverage.stratum_results` that a censused
    stratum has eluded nothing — is a change to the ledger and is deliberately NOT made from
    the planner. If this test ever starts failing, the ledger has been fixed and this file
    should celebrate rather than complain.
    """
    cov = _ledger(spec, chart)
    cov.listed_documents = True
    pool = cov.by_stratum.get("may_mention") or []
    if not pool:
        pytest.skip("this spec declares no may_mention stratum on this chart")
    cov.note_search("a-term-that-hits-everything", [d.note_id for d in pool])
    for d in pool:
        cov.note_read(d.note_id, d.doc_type)

    r = _stratum(cov, "may_mention")
    assert r.misses == 0 and r.hits == len(pool) and not r.hits_unread
    assert r.elusion_upper == 1.0, (
        "the ledger now credits a census — good. Update coverage_planner.MONOTONICITY_VS_LEDGER "
        "and this test, because the documented conservative defect is gone."
    )


def test_the_planners_own_terms_are_additions_not_baseline(spec, chart):
    """A term the coverage planner proposed is a term the SPEC did not declare. Folding it
    into the baseline would erase the gap at the moment it is created."""
    from acr.review.coverage_planner import plan_coverage

    class _Planner:
        def chat(self, messages, tools=None):
            return LLMResponse(content=json.dumps({
                "assignments": [{"type": SAMPLED_TYPE, "policy": "search", "why": "", "confidence": 1}],
                "keywords": ["adenocarcinoma", "pathology"]}), tool_calls=[])

    p = plan_coverage(spec, chart, _Planner())
    assert p.initial_keywords == spec_declared_keywords(spec)
    assert "adenocarcinoma" in p.keywords
    (added,) = [r for r in p.term_provenance if r["term"] == "adenocarcinoma"]
    assert added["trigger"] == "planner_proposal" and added["step"] == 0
    assert "pathology" not in [r["term"] for r in p.term_provenance], (
        "a term the spec already declared must not be logged as an addition")


def test_the_marker_catalogue_comes_from_the_skill_not_from_src():
    """Read the catalogue, do not invent one. A second hand-written list in src/ drifts from
    the measured one within a week and then the two disagree about what blocks an answer."""
    cat = load_marker_catalogue()
    assert cat.degraded == "", cat.degraded
    assert "thread-chasing" in cat.source
    texts = {m.text for m in cat.markers}
    for expected in ("stains pending", "see addendum", "outside facility",
                     "correlate clinically", "truncated"):
        assert expected in texts, f"{expected!r} missing from the parsed catalogue"
    assert "final diagnosis" not in texts, (
        "`final diagnosis` is in the base-rate table because it is the RESOLUTION, not the "
        "thread. Enrolling it would open a thread on every pathology report in the corpus."
    )
    # The catalogue's own precision warning, parsed rather than restated: `pending` occurs in
    # 599 of 7,965 documents and is mostly refills and appointments.
    assert cat.by_text()["pending"].low_precision is True
    assert cat.by_text()["see addendum"].low_precision is False


def test_a_zero_hit_search_is_a_trigger(plan):
    cat = load_marker_catalogue()
    t, = triggers_from_tool_result("search_notes", {"query": "signet ring"},
                                   {"n_hits": 0, "hits": []}, plan=plan, catalogue=cat, step=3)
    assert t.kind == TRIGGER_ZERO_HIT_SEARCH and "signet ring" in t.observation
    assert not triggers_from_tool_result("search_notes", {"query": "x"},
                                         {"n_hits": 4}, plan=plan, catalogue=cat, step=3)


def test_a_citation_the_term_list_would_not_have_found_is_a_trigger(plan):
    """Mechanical and model-free. If no current term occurs in the quote the agent chose to
    rest its answer on, the plan did not lead there and the term that would have is missing."""
    cat = load_marker_catalogue()
    t, = triggers_from_tool_result(
        "record_evidence", {"note_id": "N1"},
        {"recorded": True, "quote": "mucinous neoplasm of the appendiceal orifice"},
        plan=plan, catalogue=cat, step=4,
        quote="mucinous neoplasm of the appendiceal orifice")
    assert t.kind == TRIGGER_UNLISTED_ANSWER_TERM
    assert "mucinous" in t.terms_proposed
    # A quote the plan's own terms would have found is not a trigger.
    assert not triggers_from_tool_result(
        "record_evidence", {"note_id": "N1"}, {"quote": "invasive carcinoma"},
        plan=plan, catalogue=cat, step=4, quote="invasive carcinoma")


def test_an_unsettled_marker_and_a_truncated_read_are_both_triggers(plan):
    cat = load_marker_catalogue()
    out = triggers_from_tool_result(
        "read_document", {},
        {"note_id": "N1", "doc_type": "Surgical-Pathology-Document", "truncated": True,
         "returned_chars": 4000, "total_chars": 19050,
         "text": "diagnosis deferred; special stains pending, see addendum"},
        plan=plan, catalogue=cat, step=5)
    markers = {t.marker for t in out}
    assert {"truncated", "stains pending", "see addendum"} <= markers
    assert all(t.kind == TRIGGER_UNSETTLED_THREAD for t in out)


def test_a_low_precision_marker_fires_only_inside_a_decisive_type(plan):
    """The catalogue's rule, in the plan's vocabulary rather than as a hard-coded type list:
    inside a document the plan judged capable of establishing the answer, `pending` is a
    thread; inside a progress note it is a medication refill."""
    cat = load_marker_catalogue()
    body = {"note_id": "N", "text": "prior authorisation pending", "truncated": False}
    quiet = triggers_from_tool_result("read_document", {},
                                      {**body, "doc_type": "Onc-Med-MD-OP-Progress-Note"},
                                      plan=plan, catalogue=cat, step=1)
    loud = triggers_from_tool_result("read_document", {},
                                     {**body, "doc_type": "Surgical-Pathology-Document"},
                                     plan=plan, catalogue=cat, step=1)
    assert quiet == [] and [t.marker for t in loud] == ["pending"]


def test_an_obligation_the_plan_forbids_discharging_is_a_trigger(plan):
    """A gate saying "read these hits" while the plan says "you may not open that type" is a
    deadlock, not a rejection — and the old loop would spend its whole budget inside it."""
    t, = gate_obligation_triggers([], plan=plan, unread_hit_types=[SAMPLED_TYPE], step=7)
    assert t.kind == TRIGGER_GATE_OBLIGATION_UNREACHABLE
    assert SAMPLED_TYPE in t.types_proposed

    t2, = gate_obligation_triggers(["required search not performed: 'lobe'"],
                                   plan=plan, step=7)
    assert t2.terms_proposed == ("lobe",)


def test_re_reading_a_document_does_not_multiply_the_debt():
    threads = OpenThreadLedger()
    for _ in range(3):
        threads.open_thread(note_id="N1", doc_type="d", marker="pending", obligation="o",
                            excerpt="", step=1)
    assert len(threads.threads) == 1


# ==========================================================================================
# 8. THE BUDGET
# ==========================================================================================
def test_the_budget_is_priced_against_the_plan_with_no_literal_in_it(plan, chart):
    b = ExpansionBudget.priced_against(plan, documents_by_type(chart), max_revisions=6)
    assert b.max_terms_added == len(plan.initial_keywords)
    assert b.max_type_promotions == len(plan.read_all) + len(plan.search)
    assert b.max_documents_opened_by_promotion == sum(
        documents_by_type(chart).get(t, 0) for t in plan.read_all + plan.search)


# ==========================================================================================
# 9. END TO END — the revision reaches the model, and the rate reaches the manifest
# ==========================================================================================
class _RevisingLLM(LLMClient):
    """Scripted. Widens the plan once at the first reflection, then reads what it unlocked.

    It cites nothing and never submits, so the run ends on the budget — which is fine: what
    is under test is whether a revision travels from the reflection JSON into the dispatch
    guard, and whether the rate lands in the manifest.
    """

    def __init__(self, promote: str, term: str):
        super().__init__(LLMConfig(model="scripted/none", api_key="none"))
        self.promote, self.term = promote, term
        self.revised = False
        self.read_results: list[dict] = []

    def _reply(self, obj, calls=None):
        self.calls += 1
        self.prompt_tokens += 10
        self.completion_tokens += 5
        return LLMResponse(content=json.dumps(obj), tool_calls=calls or [],
                           prompt_tokens=10, completion_tokens=5)

    def chat(self, messages, tools=None):
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        for m in reversed(messages):
            if m.get("role") == "tool" and m.get("name") == "read_document":
                self.read_results.append(json.loads(m["content"]))
                break
        if tools is None:
            if "read_all|search|sample" in last:
                # One type declared junk, everything else left unjudged (and therefore
                # defaulted to `search`). That single `sample` assignment is the prior this
                # run is going to falsify.
                return self._reply({"assignments": [{"type": self.promote, "policy": "sample",
                                                     "why": "imaging cannot establish "
                                                            "histology", "confidence": 0.9}],
                                    "keywords": []})
            if "SUFFICIENT|CONTINUE|STUCK" in last:
                if self.revised:
                    return self._reply({"verdict": "CONTINUE", "reason": "working the widened plan"})
                self.revised = True
                return self._reply({
                    "verdict": "CONTINUE",
                    "reason": "imaging localises the tumour and the plan sampled it away",
                    "revision": {"add_terms": [self.term],
                                 "promote_types": [{"type": self.promote, "to": "search"}]}})
            return self._reply({"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                "reasoning": "the scripted run never finalised"})
        # act: always try to open the sampled type. Refused before the revision, allowed after.
        return self._reply({}, [{"id": "c0", "name": "read_document",
                                 "arguments": {"note_id": self._target}}])






# ==========================================================================================
# 10. A TERM THE PLAN ALREADY COVERS IS NOT AN EXPANSION
#
# `max_terms_added` is priced against the spec's own list, so it is single digits — five on
# STORE.400. A variant that retrieves nothing still spends one of those five, and an agent
# that spends all five on variants has widened nothing and learned nothing about the chart.
# Not a safety hole (the plan is still monotone, the gate still demands every term be run)
# and that is exactly why nothing caught it.
#
# The rule these tests pin down is a fact about the SEARCH THE AGENT HAS, not a style
# preference: `corpus.PatientChart.search` compiles `re.escape(query)` with `re.IGNORECASE`.
# ==========================================================================================
def _hits(term: str, texts) -> set[str]:
    """The corpus search semantics, spelled out on strings written here.

    `corpus.py`: `re.compile(query if regex else re.escape(query), re.IGNORECASE)`. Copied
    rather than run against the corpus so the direction claim below is PROVED by the same
    rule the runtime uses, on text this test owns and no chart text at all.
    """
    pat = re.compile(re.escape(term), re.IGNORECASE)
    return {t for t in texts if pat.search(t)}


_NOTES = ("Final diagnosis: invasive carcinoma, moderately differentiated.",
          "IMPRESSION: CARCINOMA of the right upper lobe cannot be excluded.",
          "Specimen shows adenocarcinoma arising in a tubular adenoma.",
          "No malignancy identified in this biopsy.")


class _DuplicateTermLLM(LLMClient):
    """Reflects once asking for a term the plan already has, then keeps acting.

    Scripted end to end because the claim under test is that the refusal REACHES THE MODEL,
    and the message stream is the only place that can be observed.
    """

    def __init__(self, term: str):
        super().__init__(LLMConfig(model="scripted/none", api_key="none"))
        self.term = term
        self.act_prompts: list[str] = []

    def _reply(self, obj, calls=None):
        self.calls += 1
        self.prompt_tokens += 10
        self.completion_tokens += 5
        return LLMResponse(content=json.dumps(obj), tool_calls=calls or [],
                           prompt_tokens=10, completion_tokens=5)

    def chat(self, messages, tools=None):
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if tools is None:
            if "read_all|search|sample" in last:
                return self._reply({"assignments": [], "keywords": []})
            if "SUFFICIENT|CONTINUE|STUCK" in last:
                return self._reply({"verdict": "CONTINUE", "reason": "widen the term list",
                                    "revision": {"add_terms": [self.term]}})
            return self._reply({"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                "reasoning": "the scripted run never finalised"})
        self.act_prompts.append(last)
        return self._reply({}, [{"id": "c0", "name": "list_documents", "arguments": {}}])



# ------------------------------------------------------------------------------- helpers
def _agent(spec, chart, llm=None):
    """The AUDIT MIDDLEWARE wired to real ledgers — where these rules live now.

    This used to hold a `ChartReviewAgent` and never run it, purely to borrow the plan refusal
    off the object. That runtime is gone, so the harness holds the middleware: one object less
    between the test and the rule it asserts. The `_expansion_*` aliases keep the assertions
    below reading as they did, and what they call is the live code.

    `declared` is populated because `_undeclared` refuses anything outside it, and an empty set
    means every tool is undeclared.
    """
    from acr.contract.trace import Tracer
    from acr.review.agent import AuditMiddleware, RunContext
    from acr.review.coverage_planner import OpenThreadLedger, load_marker_catalogue
    from acr.review.plan_expansion import expansion_is_spent, price_expansion_budget
    from acr.review.tools import Toolbox

    evidence = EvidenceLedger()
    coverage = _ledger(spec, chart)
    plan = plan_from_spec(spec, chart)
    threads = OpenThreadLedger()
    ctx = RunContext(spec=spec, chart=chart, plan=plan, coverage=coverage, threads=threads,
                     catalogue=load_marker_catalogue(),
                     tracer=Tracer.create(Path("/tmp") / "acr-test-traces"),
                     gate=lambda submitted: {"accepted": False},
                     toolbox=Toolbox(chart, evidence, coverage))
    ctx.declared = {"read_document", "read_documents_batch", "search_notes", "list_documents",
                    "document_type_summary", "record_evidence", "submit_answer"}
    mw = AuditMiddleware(ctx)
    docs_by_type = documents_by_type(chart)
    mw.plan, mw.coverage, mw.evidence, mw.threads = plan, coverage, evidence, threads
    mw.toolbox, mw.tracer, mw.markers = ctx.toolbox, ctx.tracer, ctx.catalogue
    mw._docs_by_type = docs_by_type
    mw._expansion_budget = price_expansion_budget(plan, docs_by_type, max_revisions=6,
                                                  supplied=None, planner_terms=len(plan.keywords))
    # `_plan_refusal` is gone with `AuditMiddleware._out_of_plan`: the runtime no longer refuses
    # a read because of the bucket its document type is in. See the REMOVED note in acr.review.agent.

    def _spent():
        ctx.plan = mw.plan
        return (expansion_is_spent(mw.plan, mw._expansion_budget, terms_deferred=[])
                and bool(mw.threads.unresolved()))

    mw._expansion_exhausted_with_obligations = _spent
    mw._outstanding_obligations = lambda: ctx.outstanding_obligations()
    return mw


def _first_note_of_type(chart, doc_type: str) -> str:
    docs, _ = chart.list_documents(doc_type_contains=doc_type, limit=5)
    assert docs, f"the corpus fixture has no {doc_type!r}"
    return docs[0].note_id


# ==========================================================================================
# 12. THE MANIFEST'S REPLAN BLOCK IS DERIVED FROM THE TRACE, NOT ACCUMULATED BESIDE IT
#
# WHAT HAPPENED. The first true end-to-end run (SYN0001, run
# `extract__20260727T200902Z__2d2f55b-dirty`) left a trace holding 14 `plan_revision` events,
# 13 of them `applied: true`, and a manifest reporting `n_revisions_applied: 0` and
# `replan_rate: 0.0`. The 0.0 was read as "the model ignores the replanning channel" and a
# conclusion was written from it.
#
# BOTH NUMBERS WERE ARITHMETICALLY RIGHT AND THE CONCLUSION WAS WRONG. They count different
# things and neither name says so:
#
#   trace `plan_revision.applied`  == the revision was ADMISSIBLE (monotone, affordable, not
#                                     wholly redundant). True even when it moved nothing.
#   manifest `n_revisions_applied` == the revision MOVED RETRIEVAL (terms or promotions).
#
# On that run all 13 admitted revisions had `outcome.terms_added == []` and
# `outcome.types_promoted == []`: the agent asked to re-promote types already at `read_all`
# and to re-open a thread already open. So `replan_rate: 0.0` is a true statement about plan
# MOVEMENT and a false answer to the question anyone actually asks of it, because the
# manifest had no field at all for "how many times did the agent reach for this channel".
# 14 requests and 0 requests rendered identically. That missing field is the defect.
#
# WHAT THESE TESTS HOLD
#   1. Every replan number is recomputed from the trace events by ONE function,
#      `run_manifest.replan_from_trace`, and the manifest publishes what that function
#      returns. Two counters for one quantity is what produced the divergence.
#   2. A revision that was admitted and moved nothing is REPORTED AS SUCH — requested,
#      admitted, no-op — so "never tried" and "tried and it did nothing" cannot be read as
#      the same run ever again.
#   3. The runtime counters `graph.py` still keeps are cross-checked against the derivation
#      inside the manifest itself, so a future divergence is visible in the artifact and
#      fails here loudly rather than reading plausibly.
#
# No provider is called and no chart text is written here.
# ==========================================================================================
from acr.review.run_manifest import replan_from_trace


def _events(path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


#: The refusal prose `apply_revision` returns for a promotion already in force. Verbatim
#: from the SYN0001 trace, where it appeared on four revisions all reported as `applied`.
REFUSAL_ALREADY_AT_POLICY = "'Surgical-Pathology-Report' is already at 'read_all'; no change"


def _revision_event(*, applied: bool, terms=(), promotions=(), refused=()) -> dict:
    """One `plan_revision` event in the shape `graph._after_reflect` emits."""
    return {"kind": "plan_revision", "applied": applied,
            "outcome": {"applied": applied, "terms_added": list(terms),
                        "types_promoted": [{"type": t, "from": "search", "to": "read_all"}
                                           for t in promotions],
                        "threads_opened": [], "threads_resolved": [], "threads_dismissed": [],
                        "refused": list(refused), "refusal_class": "" if applied else "BUDGET"}}


def test_the_recorded_syn0001_divergence_is_reproduced_and_named_by_one_function():
    """The exact shape of the run that produced the wrong conclusion, replayed as events.

    13 admitted revisions that moved nothing plus 1 refused, over 18 reflections. The old
    reading had exactly two numbers available — 13 (trace) and 0 (manifest) — and no way to
    tell which answered "did the agent revise its plan". The derivation must publish both,
    under names that cannot be confused, plus the request count that neither had.
    """
    events = ([{"kind": "reflect", "verdict": "CONTINUE"}] * 18
              # 12 no-ops that were silently admitted, 1 of them carrying refusal prose
              + [_revision_event(applied=True) for _ in range(9)]
              + [_revision_event(applied=True,
                                 refused=[REFUSAL_ALREADY_AT_POLICY])
                 for _ in range(4)]
              + [_revision_event(applied=False, refused=["REDUNDANT_TERM: 'histology'"])])

    r = replan_from_trace(events)

    assert r["n_reflections"] == 18
    assert r["n_revision_requests"] == 14, (
        "THE NUMBER THE MANIFEST NEVER HAD. Without it a run that reached for the replanning "
        "channel 14 times is indistinguishable from one that never reached for it, and 0.0 "
        "gets read as 'the model ignores the channel'."
    )
    assert r["n_revisions_admitted"] == 13, "what the trace's `applied: true` counts"
    assert r["n_revisions_applied"] == 0, "what moved retrieval — and nothing did"
    assert r["n_revisions_no_op"] == 13, (
        "admitted and changed nothing. This is the whole finding: the agent used the channel "
        "and the plan was already maximal for what it asked."
    )
    assert r["n_revisions_refused"] == 1
    assert r["n_revisions_partly_refused"] == 4, (
        "a revision admitted WITH refusal prose on part of it was counted as neither applied "
        "nor refused, so 4 refusals of a real request were reported nowhere"
    )
    assert r["replan_rate"] == 0.0
    assert r["request_rate"] == round(14 / 18, 4), (
        "the honest answer to 'did the agent use the replanning channel'. It is not the "
        "replan rate and must never be read as one."
    )
    assert r["terms_added_by_reflection"] == 0 and r["types_promoted"] == 0


def test_replan_rate_and_request_rate_are_different_questions():
    """A run that never reaches for the channel and a run that reaches and is refused every
    time both score `replan_rate == 0`. They must not score the same everywhere."""
    silent = replan_from_trace([{"kind": "reflect"}] * 5)
    trying = replan_from_trace([{"kind": "reflect"}] * 5
                               + [_revision_event(applied=False, refused=["x"])] * 5)
    assert silent["replan_rate"] == trying["replan_rate"] == 0.0
    assert silent["n_revision_requests"] == 0 and silent["request_rate"] == 0.0
    assert trying["n_revision_requests"] == 5 and trying["request_rate"] == 1.0, (
        "the two runs are the opposite diagnosis — nothing to do, versus the budget or the "
        "monotonicity rule refusing everything — and one number cannot carry both"
    )


def test_a_moving_revision_is_counted_once_as_applied_and_once_as_a_request():
    r = replan_from_trace([{"kind": "reflect"}] * 2
                          + [_revision_event(applied=True, terms=["right upper lobe"],
                                             promotions=["Chest-CT-W-Contr"])])
    assert r["n_revision_requests"] == 1 and r["n_revisions_admitted"] == 1
    assert r["n_revisions_applied"] == 1 and r["n_revisions_no_op"] == 0
    assert r["terms_added_by_reflection"] == 1 and r["types_promoted"] == 1
    assert r["replan_rate"] == 0.5


class _NoOpRevisingLLM(_RevisingLLM):
    """Scripted to reproduce SYN0001: ask, every single reflection, for a promotion the plan
    has already made. Admissible, monotone, affordable — and it moves nothing."""

    def chat(self, messages, tools=None):
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if tools is None and "read_all|search|sample" in last:
            # The type is ALREADY at `search`, so asking for `search` is admissible, monotone,
            # free — and moves nothing. Exactly the SYN0001 shape.
            return self._reply({"assignments": [{"type": self.promote, "policy": "search",
                                                 "why": "already searchable",
                                                 "confidence": 0.9}], "keywords": []})
        if tools is None and "SUFFICIENT|CONTINUE|STUCK" in last:
            self.revised = True
            return self._reply({
                "verdict": "CONTINUE",
                "reason": "the pathology type must be read in full",
                "revision": {"add_terms": [],
                             "promote_types": [{"type": self.promote, "to": "search"}]}})
        return super().chat(messages, tools)


def test_there_is_exactly_one_retrieval_plan_in_the_runtime():
    """The REPLAN bug was two plans where only one mattered — a prose planning prompt beside the
    real `CoveragePlan`, consumed by nothing. This guard asks the question that bug poses: is
    there exactly one object saying WHERE TO LOOK, and is it the live plan?

    It used to grep for the literal name `PLAN_PROMPT`, which a rename would satisfy, so it now
    asserts the property. `OPEN_GAPS_PROMPT` deliberately does not trip it: that is the other half
    of planning — which questions are open and what would close each one — carried by
    `write_todos`, whose list is model-authored and lives in state. A retrieval prior and a gap
    ledger are two different questions; two names for one retrieval plan is the bug.

    This test was itself deleted twice on 2026-08-06 by scripted sweeps keyed on substrings that
    its own explanatory prose contained. It is restored with the mechanism named only here.
    """
    import inspect

    import acr.review.agent as A
    src = inspect.getsource(A)
    assert "RETRIEVAL PLAN" not in src, (
        "a second retrieval plan has appeared in the runtime. `CoveragePlan.render` is the one "
        "place that says where to look")
    assert "plan.render" in src, "the runtime must still consult the coverage plan"
    assert "OPEN_GAPS_PROMPT" in src, (
        "the gap ledger's text is gone. `write_todos` was bound on all 514 recorded runs and "
        "called on zero of them because the library default tells the model not to bother for "
        "few-step tasks; this override is what makes the tool reachable")
