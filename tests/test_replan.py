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
from acr.review.answer_gate import check_threads, gate_answer
from acr.review.coverage import CoverageLedger, ForcedSampler, strata_from_spec
from acr.review.coverage_planner import (
    POLICY_RANK,
    REFUSED_BUDGET,
    REFUSED_NOT_MONOTONE,
    REFUSED_REDUNDANT_TERM,
    REFUSED_UNKNOWN_TYPE,
    TRIGGER_GATE_OBLIGATION_UNREACHABLE,
    TRIGGER_UNLISTED_ANSWER_TERM,
    TRIGGER_UNSETTLED_THREAD,
    TRIGGER_ZERO_HIT_SEARCH,
    ExpansionBudget,
    OpenThreadLedger,
    PlanRevision,
    PlanSnapshot,
    check_monotone,
    documents_by_type,
    gate_obligation_triggers,
    load_marker_catalogue,
    normalise_term,
    plan_from_spec,
    redundant_against,
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


def _apply(plan, rev, *, budget, step=1, threads=None, chart=None, trigger="test"):
    return plan.apply_revision(
        rev, step=step, trigger=trigger, observation="a test observation", budget=budget,
        threads=threads,
        n_docs_by_type=documents_by_type(chart) if chart is not None else {})



def test_the_prose_plan_is_gone_and_the_coverage_plan_is_wired_in():
    """The two greps from the finding, inverted into an assertion.

    Structural on purpose. A behavioural test only covers the paths someone thought of; the
    failure was an entire object that nothing referenced, and the way that returns is somebody
    reintroducing a "planning prompt" beside the real plan. Asserted on the runtime that holds
    the plan now.
    """
    import acr.review.agent as A

    src = inspect.getsource(A)
    assert not re.search(r"^\w*PLAN_PROMPT\s*=", src, re.MULTILINE), (
        "a second planning prompt has appeared. The REPLAN bug existed because there were two "
        "plans and only one mattered; a second one is the bug returning")
    for token in ("plan.render", "apply_revision"):
        assert token in src, f"the runtime must consult the coverage plan; {token!r} is absent"
    # `may_open` is deliberately NOT in this list any more. The runtime used to consult it to
    # REFUSE A READ, and that hook was removed on 2026-07-30: see `_out_of_plan: REMOVED` in
    # acr.review.agent. Which documents to open is the model's decision.
    assert "_out_of_plan" not in src or "REMOVED" in src, (
        "the read refusal is back. It fired 138 times over the recorded traces, and the bucket "
        "it enforced came from a substring over local type names that missed the FNA and "
        "surgical-pathology reports carrying the answer for 107 patients")




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


def test_a_demotion_is_refused_whole_and_recorded(plan, budget, chart):
    """All-or-nothing. Applying the admissible half of a mixed revision would hand back a
    plan the agent did not propose and cannot see, and the next revision would be computed
    against it."""
    before = plan.snapshot()
    out = _apply(plan, PlanRevision(add_terms=("mucinous",),
                                    promote_types=(("Surgical-Pathology-Document", "search"),)),
                 budget=budget, chart=chart)

    assert out.applied is False
    assert out.refusal_class == REFUSED_NOT_MONOTONE
    assert any("demoted" in r for r in out.refused)
    assert plan.snapshot() == before, "a refused revision must change nothing at all"
    assert "mucinous" not in plan.keywords, (
        "the admissible term rode in on a refused revision — partial application is how a "
        "plan the agent never proposed becomes the baseline for the next one"
    )
    # RECORDED AS REFUSED. A refusal nobody can count is a refusal that will be repeated.
    (row,) = plan.refused_revisions
    assert row["refusal_class"] == REFUSED_NOT_MONOTONE and row["step"] == 1
    assert row["requested"]["add_terms"] == ["mucinous"]


def test_a_hallucinated_document_type_is_refused_not_dropped(plan, budget, chart):
    """Dropping it would leave the agent believing it had widened a scope it had not."""
    out = _apply(plan, PlanRevision(promote_types=(("Not-A-Real-Type", "read_all"),)),
                 budget=budget, chart=chart)
    assert out.applied is False and out.refusal_class == REFUSED_UNKNOWN_TYPE


def test_a_promotion_is_applied_and_carries_its_provenance(plan, budget, chart):
    assert plan.policy_for(SAMPLED_TYPE) == "sample"
    out = plan.apply_revision(
        PlanRevision(promote_types=((SAMPLED_TYPE, "search"),)),
        step=9, trigger=TRIGGER_UNLISTED_ANSWER_TERM,
        observation="a read surfaced a lobe name no term would have found",
        budget=budget, n_docs_by_type=documents_by_type(chart))

    assert out.applied and out.changed_retrieval()
    assert plan.policy_for(SAMPLED_TYPE) == "search"
    assert POLICY_RANK[plan.policy_for(SAMPLED_TYPE)] > POLICY_RANK["sample"]
    (row,) = plan.promotion_log
    # WHICH type, WHEN, and WHAT OBSERVATION CAUSED IT. Without the last of the three this is
    # a list of words rather than a develop-plane candidate.
    assert row["from"] == "sample" and row["to"] == "search" and row["step"] == 9
    assert row["trigger"] == TRIGGER_UNLISTED_ANSWER_TERM and row["observation"]


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


# ==========================================================================================
# 5. TWO TERM LISTS, FOR TWO PURPOSES
# ==========================================================================================
def test_the_initial_list_is_the_specs_and_survives_expansion(spec, chart, plan, budget):
    declared = spec_declared_keywords(spec)
    assert plan.initial_keywords == declared and declared, "fixture assumption"

    _apply(plan, PlanRevision(add_terms=("mucinous", "right upper lobe")),
           budget=budget, chart=chart)

    assert plan.initial_keywords == declared, (
        "the runtime rescue overwrote the baseline. That erases the evidence that the spec's "
        "list was wrong, and that evidence is the whole input to §6c — on this corpus "
        "STORE.400's five terms miss the diagnosis for 31.7% of patients, and folding the "
        "rescue back in reads as 0%."
    )
    assert set(declared) < set(plan.keywords)
    assert plan.terms_added() == ["mucinous", "right upper lobe"]


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


# ==========================================================================================
# 7. THE TYPED REVISION IS APPLIED — tested by the behaviour each field changes
# ==========================================================================================
def test_promote_types_still_widens_the_plan_but_no_longer_gates_a_read(spec, chart, plan,
                                                                          budget):
    """Promotion still records that the agent widened its own plan -- monotonically, refusing a
    demotion -- which is a useful audit fact. What it no longer does is unlock a read, because
    reads are not locked."""
    nid = _first_note_of_type(chart, SAMPLED_TYPE)
    assert plan.policy_for(SAMPLED_TYPE) == "sample"

    agent = _agent(spec, chart, llm=None)
    out, _ = agent.ctx.toolbox.dispatch("read_document", {"note_id": nid, "limit": 300})
    assert "error" not in out, "readable before any promotion"

    outcome = _apply(plan, PlanRevision(promote_types=((SAMPLED_TYPE, "read_all"),)),
                     budget=budget, chart=chart)
    assert outcome.applied
    assert plan.policy_for(SAMPLED_TYPE) == "read_all", "the widening is still recorded"

def test_a_text_matched_marker_opens_a_thread_and_advises_without_blocking(spec, chart, plan,
                                                                           budget):
    """`stains pending` is a SUBSTRING SCAN, and it no longer refuses an answer.

    It used to. Measured over every recorded trace on 2026-07-30: 39 thread refusals, 11 of them
    (28%) rejecting a tuple that was exactly the registry's. `addendum` refused 40 times while
    `read_section("ADDENDUM")` could address that heading in 0 of the 2,401 documents containing
    the word.

    What the thread still does is everything except refuse: it is opened, it is recorded, and
    `render()` puts it in the prompt with the settling call and the thread_id filled in. Whether
    a `pending` matters to this question is a clinical judgement, and the model is the thing
    being asked to make it.
    """
    threads = OpenThreadLedger()
    ev, cov = EvidenceLedger(), _ledger(spec, chart)
    nid = _first_note_of_type(chart, "Surgical-Pathology-Document")
    from acr.core.state import Evidence
    ev.add(Evidence(nid, "Surgical-Pathology-Document", "2019-01-01", 0, 5, "xxxxx",
                    "histology"))
    submitted = {"status": "FOUND", "value": {"histology": "8046"}, "reasoning": "coded"}

    out = _apply(plan, PlanRevision(open_threads=((nid, "stains pending", "the report defers"),)),
                 budget=budget, chart=chart, threads=threads)
    assert out.threads_opened == [f"{nid}#stains pending"], "still detected and opened"
    assert threads.unresolved(), "still on the ledger"
    assert "stains pending" in threads.render(), "still reaches the model as advice"

    assert check_threads(threads) == [], "a text-matched marker must not refuse the answer"
    verdict = gate_answer(spec, submitted, evidence=ev, coverage=cov, chart=chart,
                          threads=threads, plan=plan)
    assert not any("unsettled thread" in m for m in verdict["missing"])

    out = _apply(plan, PlanRevision(resolve_threads=((f"{nid}#stains pending",
                                                      "the addendum is later in the same file"),)),
                 budget=budget, chart=chart, threads=threads, step=4)
    assert out.threads_resolved == [f"{nid}#stains pending"], "resolution still recorded"


def test_a_computed_truncated_marker_still_blocks_the_answer(spec, chart, plan, budget):
    """The one that survived, and the reason it is not the same kind of thing.

    `truncated` is computed from the character counts of the run's own read against the length
    that read reported. It is a fact about what this run did, not a guess about what a word
    means; it cannot be wrong about the corpus; and the agent discharges it by reading to the
    end.
    """
    from acr.review.coverage_planner import MARKER_TRUNCATED, marker_blocks_answer
    assert marker_blocks_answer(MARKER_TRUNCATED)
    assert not marker_blocks_answer("stains pending")
    assert not marker_blocks_answer("addendum")

    threads = OpenThreadLedger()
    ev, cov = EvidenceLedger(), _ledger(spec, chart)
    nid = _first_note_of_type(chart, "Surgical-Pathology-Document")
    from acr.core.state import Evidence
    ev.add(Evidence(nid, "Surgical-Pathology-Document", "2019-01-01", 0, 5, "xxxxx",
                    "histology"))
    submitted = {"status": "FOUND", "value": {"histology": "8046"}, "reasoning": "coded"}

    threads.open_thread(note_id=nid, doc_type="Surgical-Pathology-Document",
                        marker=MARKER_TRUNCATED, excerpt="read stopped 353 characters short",
                        obligation="page to the end before reasoning about it", step=1)
    assert check_threads(threads), "a computed marker still refuses"
    verdict = gate_answer(spec, submitted, evidence=ev, coverage=cov, chart=chart,
                          threads=threads, plan=plan)
    assert verdict["accepted"] is False
    assert any("unsettled thread" in m for m in verdict["missing"])


def test_dismiss_threads_requires_a_reason_and_records_it(plan, budget, chart):
    threads = OpenThreadLedger()
    threads.open_thread(note_id="N1", doc_type="Surgical-Pathology-Document",
                        marker="stains pending", obligation="the lab had not finished",
                        excerpt="", step=1)
    out = _apply(plan, PlanRevision(dismiss_threads=(("N1#stains pending", ""),)),
                 budget=budget, chart=chart, threads=threads)
    assert out.threads_dismissed == [] and threads.unresolved(), (
        "an unreasoned dismissal reads exactly like an unnoticed thread")
    assert threads.refused_dismissals

    out = _apply(plan, PlanRevision(dismiss_threads=(("N1#stains pending",
                                                      "the stain bears on histology and this "
                                                      "answer codes only the site"),)),
                 budget=budget, chart=chart, threads=threads, step=6)
    assert out.threads_dismissed == ["N1#stains pending"] and not threads.unresolved()
    (t,) = threads.to_dict()["threads"]
    assert t["state"] == "dismissed" and "codes only the site" in t["resolution"]


def test_a_thread_is_not_counted_as_a_replan(plan, budget, chart):
    """Resolving a thread is bookkeeping. Counting it would reinflate the replan rate with
    exactly the sort of no-op the old REPLAN verdict was."""
    threads = OpenThreadLedger()
    threads.open_thread(note_id="N1", doc_type="d", marker="pending", obligation="o",
                        excerpt="", step=1)
    out = _apply(plan, PlanRevision(resolve_threads=(("N1#pending", "found it"),)),
                 budget=budget, chart=chart, threads=threads)
    assert out.applied and not out.changed_retrieval()
    assert plan.revisions_applied == 0


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


def test_expansion_beyond_the_budget_is_refused_and_recorded(plan, chart):
    tight = ExpansionBudget(max_terms_added=1, max_type_promotions=1,
                            max_documents_opened_by_promotion=10_000, max_revisions=6)
    assert _apply(plan, PlanRevision(add_terms=("one",)), budget=tight, chart=chart).applied
    out = _apply(plan, PlanRevision(add_terms=("two",)), budget=tight, chart=chart)

    assert out.applied is False and out.refusal_class == REFUSED_BUDGET
    assert "two" not in plan.keywords
    assert plan.budget_exhausted(tight) is True
    # NEVER a silent truncation. The refusal is in the plan, and `_after_reflect` reads
    # `budget_exhausted` to route the run to an honest abstention.
    assert plan.refused_revisions[-1]["refusal_class"] == REFUSED_BUDGET




def test_budget_exhausted_with_obligations_outstanding_is_an_honest_dead_end(spec, chart, plan):
    """Both halves matter, and so does the third: the plan must have BUMPED into the cap.

    `expansion_is_spent` deliberately answers False for a run that never tried to widen, however
    small its budget — otherwise a tight budget would end a run that had asked for nothing. So a
    dead end is: it asked, it was refused, and an obligation is still outstanding.
    """
    from acr.review.plan_expansion import expansion_is_spent

    zero = ExpansionBudget(max_terms_added=0, max_type_promotions=0,
                           max_documents_opened_by_promotion=0, max_revisions=6)
    agent = _agent(spec, chart, llm=None)
    agent.plan = plan
    agent.threads = OpenThreadLedger()
    agent._expansion_budget = zero

    assert expansion_is_spent(plan, zero, terms_deferred=[]) is False, (
        "budget untouched is not exhaustion; a run that never tried to widen must keep going")

    _apply(plan, PlanRevision(add_terms=("mucinous",)), budget=zero, chart=chart)
    assert expansion_is_spent(plan, zero, terms_deferred=[]) is True, (
        "the plan asked, was refused, and has no room left")
    assert agent._outstanding_obligations(), "fixture assumption: the gate is not yet met"

    # And the runtime says so rather than merely stopping at the call limit.
    import acr.review.agent as A
    src = inspect.getsource(A.AuditMiddleware._expansion_spent_with_obligations)
    assert "expansion_is_spent" in src and "outstanding" in src


def test_a_spent_budget_with_nothing_outstanding_is_not_a_dead_end(spec, chart, plan,
                                                                   monkeypatch):
    agent = _agent(spec, chart, llm=None)
    agent.plan = plan
    agent.threads = OpenThreadLedger()
    agent._expansion_budget = ExpansionBudget(0, 0, 0, max_revisions=6)
    _apply(plan, PlanRevision(add_terms=("x",)), budget=agent._expansion_budget, chart=chart)
    monkeypatch.setattr(agent, "_outstanding_obligations", list)
    assert agent._expansion_exhausted_with_obligations() is False


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


def test_a_case_variant_is_the_same_search_and_must_not_cost_a_term_slot(plan, chart):
    """The verified defect: "CARCINOMA" beside "carcinoma" was accepted as a distinct term.

    Priced first, because that is the damage — the plan gains nothing and the allowance is
    one smaller. With a cap of one, the variant used to consume the only slot the run had.
    """
    assert "carcinoma" in plan.keywords, "fixture assumption"
    assert _hits("CARCINOMA", _NOTES) == _hits("carcinoma", _NOTES), (
        "fixture assumption: the runtime's search is case-insensitive, so the variant "
        "retrieves exactly nothing new")
    one = ExpansionBudget(max_terms_added=1, max_type_promotions=1,
                          max_documents_opened_by_promotion=10_000, max_revisions=6)

    out = _apply(plan, PlanRevision(add_terms=("CARCINOMA",)), budget=one, chart=chart)

    assert out.terms_added == [] and plan.terms_added() == []
    assert plan.keywords.count("carcinoma") == 1 and "CARCINOMA" not in plan.keywords
    # THE SLOT SURVIVED. This is the whole point: the next revision, the one that carries a
    # real term, must still be affordable.
    real = _apply(plan, PlanRevision(add_terms=("mucinous",)), budget=one, chart=chart, step=2)
    assert real.applied and real.terms_added == ["mucinous"]


@pytest.mark.parametrize("variant", ["  carcinoma  ", "\tcarcinoma\n", "final   diagnosis",
                                     "Final Diagnosis"])
def test_whitespace_and_case_variants_are_the_same_term(plan, chart, budget, variant):
    """Surrounding whitespace, internal runs, and case. A padded term is not a new search —
    and a double-spaced one is a strictly NARROWER search, since `re.escape` keeps both
    spaces and the note has one."""
    before = list(plan.keywords)
    out = _apply(plan, PlanRevision(add_terms=(variant,)), budget=budget, chart=chart)
    assert plan.keywords == before and out.terms_added == []


def test_a_narrower_term_is_redundant_and_a_broader_one_is_the_whole_point(plan, chart, budget):
    """BOTH WAYS ROUND, because inverting this rule would refuse the only additions worth
    paying for.

    Substring search means hits(narrow) is contained in hits(broad). So adding
    "adenocarcinoma" when the plan already runs "carcinoma" buys nothing, while adding
    "carcin" when the plan runs "carcinoma" is a genuine widening — it reaches documents the
    plan cannot currently see.
    """
    assert _hits("adenocarcinoma", _NOTES) < _hits("carcinoma", _NOTES), (
        "fixture assumption: the longer term returns a strict subset")
    assert _hits("carcin", _NOTES) >= _hits("carcinoma", _NOTES), (
        "fixture assumption: the shorter term returns a superset")

    narrower = _apply(plan, PlanRevision(add_terms=("adenocarcinoma",)),
                      budget=budget, chart=chart)
    assert "adenocarcinoma" not in plan.keywords
    assert narrower.terms_added == [], (
        "every document 'adenocarcinoma' could return is already returned by 'carcinoma'")
    assert any("carcinoma" in why for why in narrower.refused), (
        "the refusal must name the term that already covers it, or the agent cannot tell "
        "whether to try a different word or a shorter one")

    broader = _apply(plan, PlanRevision(add_terms=("carcin",)), budget=budget, chart=chart,
                     step=2)
    assert broader.applied and broader.terms_added == ["carcin"], (
        "a SHORTER term reaches documents the plan cannot currently see; refusing it would "
        "block the only kind of expansion that is worth a budget slot")
    assert broader.changed_retrieval() is True


@pytest.mark.parametrize("order", [("carcin", "adenocarcinoma"), ("adenocarcinoma", "carcin")])
def test_one_revision_pays_once_for_a_term_and_its_narrower_variant(plan, chart, order):
    """Order-independent: the price of a revision must not depend on the order the model
    happened to list its terms in, because nothing downstream can reproduce that."""
    one = ExpansionBudget(max_terms_added=1, max_type_promotions=1,
                          max_documents_opened_by_promotion=10_000, max_revisions=6)
    out = _apply(plan, PlanRevision(add_terms=order), budget=one, chart=chart)

    assert out.applied and out.terms_added == ["carcin"], (
        "two terms, one search: 'adenocarcinoma' is inside 'carcin' and cannot add a "
        "document to it")
    assert len(plan.terms_added()) == 1


def test_a_revision_that_was_all_variants_is_applied_and_the_duplication_is_reported(plan,
                                                                                     chart,
                                                                                     budget):
    """The redundancy is still told to the agent; it no longer rejects the revision.

    The old behaviour refused the whole revision, on the reasoning that reporting "APPLIED" over
    a plan that did not move would let the agent believe it had widened its search. That reasoning
    depended on the keyword list being a CONTRACT -- the gate discharged against it, so a term
    that was not really added would be demanded later. Both halves are gone: `check_gate` no
    longer refuses over an unsearched term and `fit_terms_to_budget` no longer trims, so the list
    is a record of what the model said it would search. A refusal over a record costs a round trip
    and buys nothing.
    """
    out = _apply(plan, PlanRevision(add_terms=("CARCINOMA", "  biopsy")),
                 budget=budget, chart=chart)

    assert out.applied is True, "a revision of duplicates is not a failure"
    assert out.refusal_class == "", "nothing was refused, so nothing is classified as refused"
    assert out.changed_retrieval() is False, "nothing new was added, and that is still true"
    # Still reported, through the channel that already carried the already-at-that-policy
    # promote no-ops on the applied path: a request that vanished without a word is one nobody
    # can audit.
    assert len(out.refused) == 2, "the agent is still told why"
    for term, covered_by in (("carcinoma", "carcinoma"), ("biopsy", "biopsy")):
        assert any(repr(term) in why and repr(covered_by) in why for why in out.refused)
    assert plan.budget_exhausted(budget) is False


def test_a_variant_alongside_a_real_term_still_lands_and_is_still_reported(plan, chart, budget):
    """Partial, in the same shape as the budget overrun: what is admissible is applied, and
    what was dropped travels back rather than vanishing."""
    out = _apply(plan, PlanRevision(add_terms=("mucinous", "SPECIMEN")),
                 budget=budget, chart=chart)

    assert out.applied and out.terms_added == ["mucinous"]
    assert plan.terms_added() == ["mucinous"], "the variant must not appear in the harvest"
    assert any(REFUSED_REDUNDANT_TERM in why and "'specimen'" in why for why in out.refused)


def test_the_normalisation_is_case_and_whitespace_only(plan, chart, budget):
    """Not stemming, not punctuation folding. A different WORD is a different search, and
    quietly folding one onto another would drop a term the agent is still charged for."""
    assert normalise_term("  Final   Diagnosis ") == "final diagnosis"
    assert redundant_against("carcinoid", ["carcinoma"]) is None
    assert redundant_against("carcinomas", ["carcinoma"]) == "carcinoma"

    out = _apply(plan, PlanRevision(add_terms=("carcinoid",)), budget=budget, chart=chart)
    assert out.applied and out.terms_added == ["carcinoid"], (
        "'carcinoid' shares a prefix with 'carcinoma' and is a different word; folding them "
        "would lose a search the agent asked for and was charged for")


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



def test_the_duplicate_refusal_reaches_the_model_in_the_loop_it_understands(spec, chart):
    """A duplicate it is not told about is one it will send again, one slot at a time.

    The refusal used to be rendered into the next reflect prompt. It is the `revise_plan` tool's
    RETURN VALUE now, which is stronger: the model has it in hand by construction rather than by
    a prompt someone remembered to write.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hooks_harness import revise, revise_plan_tool

    tool, ctx = revise_plan_tool(spec, chart)
    assert "carcinoma" in ctx.plan.keywords, "fixture assumption: the spec already covers it"
    before = list(ctx.plan.keywords)

    r = revise(tool, add_terms=["CARCINOMA"])
    told = " ".join(r["refused"])
    assert REFUSED_REDUNDANT_TERM in told, "the refusal never reached the model"
    assert "carcinoma" in told.lower(), "the refusal must name the covering term"
    assert ctx.plan.keywords == before, "a redundant term must cost no budget and change nothing"


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



def test_a_revision_that_moved_nothing_is_reported_as_a_request_not_as_silence(spec, chart):
    """THE SYN0001 FIX. Every request asks for a promotion already in force.

    The retrieval scope never moves — correctly — but the record must say, in the same block,
    that the channel WAS used and that every use was a no-op. A reader who concludes "the model
    ignores the replanning channel" has to be contradicted by the record.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hooks_harness import revise, revise_plan_tool

    tool, ctx = revise_plan_tool(spec, chart)
    already = sorted(ctx.plan.read_all)[0]
    for _ in range(3):
        revise(tool, promote_types=[{"type": already, "to": "search"}])

    assert len(ctx.revisions) == 3, "three asks must be three recorded asks"
    assert all(not r["applied"] for r in ctx.revisions), (
        "a promotion already in force moves nothing")
    assert any(r["refused"] for r in ctx.revisions), (
        "silence and a refusal are different facts; the record must carry the second")




