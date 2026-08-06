"""A trace has to say WHICH SPEC RULE was in play, and how much that claim is worth.

Before this, a trace held tool calls, arguments and messages and nothing that linked any of
it to the specification. So when an answer was wrong, nothing could say which rule was
applied, misapplied or missing — and §6b's optimizer, handed a failure and no attribution,
reverse-engineers a story from the outcome. That is the mechanism by which a loop starts
confidently rewriting text that was never at fault.

Three sources feed the attribution and they are NOT equally trustworthy:

    (a) which answer_check rejected a submission, with the value and the quote that fired it,
        and how many times in a row              — DETERMINISTIC
    (b) which evidence rule admitted or refused each cited document at gate time
                                                 — DETERMINISTIC
    (c) which decision_rule / conflict_rule the agent SAYS it applied
                                                 — SELF_REPORTED

The assertions below are mostly of the "this must be refused / this must stay separate"
shape, because the failure this work exists to prevent is not an absent record. It is a
record that reads like a measurement and is not one: a hallucinated rule citation stored
beside a computed rejection, indistinguishable one function call downstream.

No chart text and no real identifier appears anywhere here. The quotes are written for these
tests; the only real documents are the synthetic SYN000x charts already in the tree.
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
#   - test_a_check_that_fires_on_an_absence_records_no_quote
#   - test_a_content_identity_is_preferred_to_a_position_where_the_spec_has_one
#   - test_a_rejection_names_the_rule_the_value_and_the_quote_that_fired_it
#   - test_a_specific_code_is_still_refused_when_the_record_conflicts
#   - test_inserting_a_rule_above_an_answer_check_does_not_re_point_its_id
#   - test_one_documented_lobe_still_forbids_the_nos_code
#   - test_the_exemption_is_driven_by_the_spec_and_not_by_code
#   - test_the_shipped_spec_carries_the_check_and_refuses_the_real_answer
#   - test_two_lobes_in_one_answer_cannot_be_coded_as_one_of_them
#
# Nothing replaced them here. A wrong clinical value is an instruction-following failure and is
# measured as one. tests/test_answer_checks.py holds what survives: field `format` and
# `allowable_values`, the only check with a positive measured record.
# ---------------------------------------------------------------------------------------------
from __future__ import annotations

import json
from pathlib import Path

import pytest

from acr.chartstore.corpus import Corpus
from acr.contract.answer_checks import (
    Violation,
    check_answer,
    check_answer_detail,
    check_field_formats,
    check_field_formats_detail,
)
from acr.contract.spec import load_spec
from acr.contract.trace import (
    MAX_KEPT_UNRECOGNISED,
    PROV_DETERMINISTIC,
    PROV_SELF_REPORTED,
    Tracer,
    parse_rule_citations,
    rule_catalog,
    rule_catalog_hash,
    rule_citation_block,
    rule_index,
)
from acr.core import site
from acr.core.state import Evidence, EvidenceLedger
from acr.review.answer_gate import gate_answer
from acr.review.coverage import (
    ADMITTED,
    REFUSED,
    UNDECLARED,
    CoverageLedger,
    ForcedSampler,
    admissibility_for_citations,
    strata_from_spec,
)

ROOT = Path(__file__).resolve().parents[1]
SHB = site.specs_root() / "STORE.400_522_523.site_histology_behavior.yaml"
ALL_SPECS = sorted((site.specs_root()).rglob("*.yaml"))


def test_tracer_serializes_concurrent_event_sequence_and_file_order(tmp_path):
    """Parallel tool completions still produce one analysis-ready application log."""
    from concurrent.futures import ThreadPoolExecutor

    tracer = Tracer.create(tmp_path, "parallel")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda value: tracer.emit("tool", value=value), range(80)))

    rows = [
        json.loads(line)
        for line in tracer.path.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["seq"] for row in rows] == list(range(80))
    assert [row["seq"] for row in tracer.events] == list(range(80))


def _spec():
    return load_spec(SHB)


def _loadable_specs():
    out = []
    for p in ALL_SPECS:
        try:
            out.append((p, load_spec(p)))
        except Exception:      # noqa: BLE001 - another team owns assets/specs/; a broken one is
            continue           # their finding, not a reason this module cannot be tested
    return out


# ===================================================================== 1. the id scheme
def test_every_rule_of_every_spec_gets_an_id():
    """A rule with no id cannot be attributed, and an un-attributable rule is a rule the
    optimizer will never be told about."""
    for path, spec in _loadable_specs():
        cat = rule_catalog(spec)
        assert cat, f"{path.name}: no rules catalogued at all"
        assert len(cat) == len({r.rule_id for r in cat}), f"{path.name}: duplicate rule ids"
        assert len(cat) >= len(spec.decision_rule) + len(spec.conflict_rules)


def test_the_same_rule_gets_the_same_id_across_runs_of_the_same_spec_version():
    """The whole scheme rests on this. An id that moves between two loads of one file
    attributes today's failure to yesterday's sentence."""
    for path, spec in _loadable_specs():
        first = [(r.rule_id, r.text_sha) for r in rule_catalog(spec)]
        second = [(r.rule_id, r.text_sha) for r in rule_catalog(load_spec(path))]
        assert first == second, path.name


def test_a_positional_id_carries_a_fingerprint_so_a_moved_position_is_detectable():
    """Positional ids are honest only if a reader can tell when the slot changed meaning.
    Without the fingerprint an old manifest silently re-points at a different sentence."""
    spec = _spec()
    before = rule_index(spec)["decision_rule.1"].text_sha
    spec.decision_rule = ["something else entirely"] + list(spec.decision_rule[1:])
    after = rule_index(spec)["decision_rule.1"]
    assert after.rule_id == "decision_rule.1"
    assert after.text_sha != before, "a changed sentence must change the fingerprint"


def test_rule_ids_are_the_numbers_the_agent_is_actually_shown():
    """1-based, matching `as_prompt_block`'s "DECISION RULES: 1." and specview's `rule.1`.
    Asking a model to cite `decision_rule.0` for the rule printed as 1 measures our indexing
    convention, not its reasoning."""
    spec = _spec()
    block = rule_citation_block(spec)
    assert "decision_rule.1" in block and "decision_rule.0" not in block
    idx = rule_index(spec)
    assert idx["decision_rule.1"].text.startswith(spec.decision_rule[0][:20])
    assert idx["decision_rule.1"].view_id == "rule.1"


def test_the_catalog_hash_moves_when_a_rule_moves():
    """An attribution names ids; the hash is what says those ids meant the same thing."""
    spec = _spec()
    h = rule_catalog_hash(rule_catalog(spec))
    spec.decision_rule = list(spec.decision_rule) + ["a new rule"]
    assert rule_catalog_hash(rule_catalog(spec)) != h


def test_the_catalog_never_raises_on_a_malformed_spec():
    """A trace that can abort a run is worse than an imperfect identifier."""
    class _Junk:
        spec_id = "junk"
        decision_rule = ["one"]
        conflict_rules = [{"if": "a"}]
        evidence_rules = {"counts_as_evidence": "not even a list"}
        answer_checks = ["a bare string"]
        fields: list = []
        abstention: dict = {}
        proof_obligation = None
    cat = rule_catalog(_Junk())
    assert [r.rule_id for r in cat][:3] == ["decision_rule.1", "conflict_rule.1",
                                            "evidence_rule.counts_as_evidence.1"]
    assert any(r.rule_id.startswith("answer_check.unparsed") for r in cat)


def test_two_rules_can_never_share_one_id():
    """One id speaking for two rules credits one of them with the other's failures — the
    exact mis-attribution the ids exist to prevent."""
    class _Dupes:
        spec_id = "dupes"
        decision_rule: list = []
        conflict_rules: list = []
        evidence_rules: dict = {}
        # Two checks with the same field/kind/first-nos-value: one content key, two rules.
        answer_checks = [{"field": "f", "kind": "not_less_specific", "nos_values": ["X"]},
                         {"field": "f", "kind": "not_less_specific", "nos_values": ["X"]}]
        fields: list = []
        abstention: dict = {}
        proof_obligation = None
    cat = rule_catalog(_Dupes())
    assert len({r.rule_id for r in cat}) == len(cat)
    assert any(r.ambiguous_id for r in cat), "a disambiguated id must say that it was one"


# ============================================== 2. (a) DETERMINISTIC: which check rejected
def _ev(quote: str, note_id: str = "N1", supports: str = "primary_site") -> list[dict]:
    return [{"note_id": note_id, "doc_type": "Pathology", "date": "2020-01-01",
             "start": 0, "end": len(quote), "quote": quote, "supports": supports,
             "stance": "supports"}]


def test_the_message_only_form_is_a_projection_of_the_attributed_one():
    """Two implementations of one check drift; then a rejection and its attribution disagree
    and nothing raises."""
    spec = _spec()
    ev = _ev("mass in the right upper lobe")
    value = {"primary_site": "C349"}
    assert (check_answer(spec.answer_checks, value, ev, ["lobe", "bronchus"])
            == [v.message for v in check_answer_detail(spec.answer_checks, value, ev,
                                                       ["lobe", "bronchus"])])
    bad = {"primary_site": "C3412"}
    assert (check_field_formats(spec.fields, bad)
            == [v.message for v in check_field_formats_detail(spec.fields, bad)])


def test_a_format_rejection_is_attributed_to_the_field_rule_not_to_a_decision_rule():
    """"C3412" is a contract violation, not a clinical judgement. Routing it at the prose
    would have a clinician re-reading a rule that had nothing to do with it."""
    v = check_field_formats_detail(_spec().fields, {"primary_site": "C3412"})
    assert [x.rule_id for x in v] == ["field_format.primary_site"]
    assert v[0].rule_kind == "field_format"


def test_the_repeat_count_separates_a_loop_from_three_unrelated_rejections(tmp_path):
    """A run was rejected twice for coding 8046 over "favor squamous" and then burned a 400k
    token budget without revising. That is a signal about the rejection MESSAGE — an
    engineer's parameter — and a cumulative total cannot express it."""
    t = Tracer.create(tmp_path, "streaks")
    t.bind_spec(_spec())
    rid = "answer_check.histology.not_less_specific.8000"
    v = Violation(rule_id=rid, rule_kind="not_less_specific", field="histology",
                  coded_value="8000", message="m", trigger="squamous")
    t.answer_check_outcome([v])
    t.answer_check_outcome([v])
    assert t.rule_rejection_max_streak[rid] == 2
    assert t.rule_attribution()["deterministic"]["rejection_loops"] == [rid]


def test_a_clean_evaluation_between_two_rejections_breaks_the_streak(tmp_path):
    """Otherwise "rejected, fixed, later rejected again for the same rule" reads as a loop,
    and the message gets blamed for a run that in fact responded to it."""
    t = Tracer.create(tmp_path, "gap")
    t.bind_spec(_spec())
    rid = "answer_check.histology.not_less_specific.8000"
    v = Violation(rule_id=rid, rule_kind="not_less_specific", field="histology",
                  coded_value="8000", message="m")
    t.answer_check_outcome([v])
    t.answer_check_outcome([])            # the agent revised; the check passed
    t.answer_check_outcome([v])
    att = t.rule_attribution()["deterministic"]
    assert att["rejections_by_rule"][rid] == 3 - 1
    assert att["max_consecutive_by_rule"][rid] == 1
    assert att["rejection_loops"] == [], "two non-consecutive rejections are not a loop"


def test_truncating_the_rejection_rows_never_truncates_the_loop_signal(tmp_path):
    """A run that resubmitted the same refused answer forty times is forty copies of one
    message. The rows are capped; the counts that make the loop visible are not."""
    from acr.contract.trace import MAX_KEPT_REJECTION_ROWS
    t = Tracer.create(tmp_path, "cap")
    t.bind_spec(_spec())
    rid = "answer_check.primary_site.not_less_specific.C349"
    v = Violation(rule_id=rid, rule_kind="not_less_specific", field="primary_site",
                  coded_value="C349", message="m")
    for _ in range(MAX_KEPT_REJECTION_ROWS + 10):
        t.answer_check_outcome([v])
    att = t.rule_attribution()["deterministic"]
    assert len(att["answer_check_rejections"]) == MAX_KEPT_REJECTION_ROWS
    assert att["rejections_truncated"] is True
    assert att["n_rejection_rows"] == MAX_KEPT_REJECTION_ROWS + 10
    assert att["max_consecutive_by_rule"][rid] == MAX_KEPT_REJECTION_ROWS + 10
    assert att["rejection_loops"] == [rid]


def test_the_manifest_rejection_row_points_at_the_quote_instead_of_copying_it(tmp_path):
    """The evidence ledger is already in the manifest. A second copy of a chart quote is a
    second thing to redact and a second thing to go stale."""
    t = Tracer.create(tmp_path, "quotes")
    t.bind_spec(_spec())
    t.answer_check_outcome([Violation(rule_id="answer_check.primary_site.origin_not_specimen",
                                      rule_kind="origin_not_specimen", field="primary_site",
                                      coded_value="C340", message="m",
                                      trigger="biopsy", quote="Bronchus, biopsy",
                                      evidence_index=0)])
    row = t.rule_attribution()["deterministic"]["answer_check_rejections"][0]
    assert "quote" not in row and row["evidence_index"] == 0
    ev = [e for e in t.events if e["kind"] == "rule_rejection"][0]
    assert ev["violations"][0]["quote"] == "Bronchus, biopsy", "the trace keeps the text"


# ========================================= 3. (b) DETERMINISTIC: admissibility at gate time
def _ledger(spec, pid="SYN0001"):
    docs, _ = Corpus(site.corpus_root()).chart(pid).list_documents(limit=100_000)
    return CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(7))


def test_the_gate_writes_down_which_evidence_rule_admitted_each_citation():
    """The stratification already computes it for every document; it was recorded nowhere,
    so "was the agent allowed to use that document for that field" had to be re-derived
    after the fact and hoped to match."""
    spec = _spec()
    led = _ledger(spec)
    note = led.by_stratum["can_establish"][0]
    rows = admissibility_for_citations(
        spec, led, [{"note_id": note.note_id, "doc_type": note.doc_type}], ["histology"])
    assert len(rows) == 1
    assert rows[0]["stratum"] == "can_establish"
    assert rows[0]["rule_id"] == "evidence_rule.stratum.can_establish.establishes"
    assert rows[0]["by_field"]["histology"]["verdict"] == ADMITTED


def test_a_refusal_is_recorded_as_a_refusal_and_names_the_rule_that_refused():
    """"Radiology can localise a mass; it cannot establish histology" is the spec sentence;
    `establishes` is that sentence where code can read it. A refusal nobody wrote down is a
    rule the optimizer cannot be shown to have been applied."""
    spec = _spec()
    led = _ledger(spec)
    pool = led.by_stratum["cannot_establish"]
    if not pool:
        pytest.skip("this synthetic chart has no document in the cannot_establish stratum")
    rows = admissibility_for_citations(
        spec, led, [{"note_id": pool[0].note_id, "doc_type": pool[0].doc_type}],
        ["histology", "primary_site"])
    assert rows[0]["by_field"]["histology"]["verdict"] == REFUSED
    assert rows[0]["by_field"]["primary_site"]["verdict"] == ADMITTED, \
        "the same stratum admits primary_site — imaging localises, it just cannot type"
    assert rows[0]["rule_id"] == "evidence_rule.stratum.cannot_establish.establishes"


def test_an_empty_establishes_list_is_undeclared_and_not_silently_resolved():
    """`establishes: []` means "every field" to the runtime (derive.py) and "no field" to a
    stratum its author named `cannot_establish` and filled with EKGs. Picking a side here
    would bury an authoring fault that is exactly the divergent-readings evidence §6b admits
    as a FORM gradient."""
    spec = _spec()
    led = _ledger(spec)
    for s in led.specs:
        s.establishes = []
    note = led.by_stratum["can_establish"][0]
    rows = admissibility_for_citations(
        spec, led, [{"note_id": note.note_id, "doc_type": note.doc_type}], ["histology"])
    row = rows[0]["by_field"]["histology"]
    assert row["verdict"] == UNDECLARED
    assert "convention" in row["why"] and "ambiguous" in row["why"].lower()


def test_admissibility_is_scoped_by_the_coded_fields_and_not_by_the_supports_label():
    """Scoping on a model-authored `supports` string is the bug that let a run code 8046 over
    a report saying "favor squamous": three quotes, three labelling styles, two silently
    dropped, and the check then validated nothing."""
    spec = _spec()
    led = _ledger(spec)
    note = led.by_stratum["can_establish"][0]
    ev = [{"note_id": note.note_id, "doc_type": note.doc_type,
           "supports": "Final pathologic diagnosis establishes morphology and site"}]
    rows = admissibility_for_citations(spec, led, ev, ["histology", "behavior"])
    assert set(rows[0]["by_field"]) == {"histology", "behavior"}


def test_an_admissibility_record_says_the_gate_did_not_enforce_it():
    """Nothing in `gate_answer` refuses a citation on these grounds today. A reader who
    assumed otherwise would conclude the ledger had already been filtered and stop looking
    for the failure that is still in it."""
    spec = _spec()
    led = _ledger(spec)
    note = led.by_stratum["can_establish"][0]
    rows = admissibility_for_citations(
        spec, led, [{"note_id": note.note_id, "doc_type": note.doc_type}], ["histology"])
    assert rows[0]["enforced_by_gate"] is False


# ================================================== 4. (c) SELF_REPORTED: what the agent says
def test_a_declared_rule_the_agent_names_is_recognised():
    known = [r.rule_id for r in rule_catalog(_spec())]
    good, bad = parse_rule_citations(
        "Coded from the final pathology.\nrules_applied: decision_rule.2, conflict_rule.1",
        known)
    assert good == ["decision_rule.2", "conflict_rule.1"] and bad == []


def test_a_rule_that_does_not_exist_is_discarded_and_counted_never_stored_as_a_citation():
    """Storing a hallucinated citation is worse than storing nothing: the optimizer routes a
    gradient at a rule that is not there, finds the spec silent about it, and files a
    SPEC_GAP against a sentence nobody ever wrote."""
    known = [r.rule_id for r in rule_catalog(_spec())]
    good, bad = parse_rule_citations("rules_applied: decision_rule.2, decision_rule.99", known)
    assert good == ["decision_rule.2"]
    assert bad == ["decision_rule.99"]


def test_the_parser_never_repairs_a_near_miss():
    """A parser that guesses what the model meant manufactures the citation the citation
    requirement exists to demand, and the guess is indistinguishable from a real one."""
    known = [r.rule_id for r in rule_catalog(_spec())]
    good, bad = parse_rule_citations("I applied decision rule 2 and the second conflict rule",
                                     known)
    assert good == [] and bad == []


def test_the_self_report_is_marked_as_a_self_report_everywhere_it_appears(tmp_path):
    """A self-report and a deterministic fact must never be indistinguishable downstream —
    not in the event, not in the manifest, not after somebody flattens the manifest."""
    t = Tracer.create(tmp_path, "marking")
    t.bind_spec(_spec())
    t.self_reported_rules("rules_applied: decision_rule.1", where="submit_answer")
    t.answer_check_outcome([Violation(rule_id="answer_check.primary_site.origin_not_specimen",
                                      rule_kind="origin_not_specimen", field="primary_site",
                                      coded_value="C340", message="m")])
    kinds = {e["kind"]: e for e in t.events}
    assert kinds["rules_self_reported"]["provenance"] == PROV_SELF_REPORTED
    assert kinds["rule_rejection"]["provenance"] == PROV_DETERMINISTIC
    att = t.rule_attribution()
    assert att["self_reported"]["provenance"] == PROV_SELF_REPORTED
    assert att["deterministic"]["provenance"] == PROV_DETERMINISTIC
    # ...and on the individual row, because sub-objects get flattened downstream and a row
    # that loses its parent key must still say what it is.
    assert att["deterministic"]["answer_check_rejections"][0]["provenance"] == PROV_DETERMINISTIC


def test_the_two_channels_are_never_merged_into_one_list_of_rules(tmp_path):
    """The single most damaging shape this could take. One "rules involved" list and the
    difference between a check that provably fired and a rule the model thought about is
    gone."""
    t = Tracer.create(tmp_path, "unmerged")
    t.bind_spec(_spec())
    rid = "answer_check.primary_site.not_less_specific.C349"
    t.self_reported_rules("rules_applied: decision_rule.1", where="submit_answer")
    t.answer_check_outcome([Violation(rule_id=rid, rule_kind="not_less_specific",
                                      field="primary_site", coded_value="C349", message="m")])
    att = t.rule_attribution()
    assert set(att["deterministic"]["rejected_by"]) == {rid}
    assert att["self_reported"]["rules_claimed"] == ["decision_rule.1"]
    assert rid not in att["self_reported"]["rules_claimed"]
    assert "decision_rule.1" not in att["deterministic"]["rejected_by"]


def test_a_flood_of_invented_ids_cannot_grow_the_manifest_without_bound(tmp_path):
    t = Tracer.create(tmp_path, "flood")
    t.bind_spec(_spec())
    t.self_reported_rules(" ".join(f"decision_rule.{i}" for i in range(100, 400)),
                          where="submit_answer")
    att = t.rule_attribution()["self_reported"]
    assert len(att["unrecognised_rule_ids"]) == MAX_KEPT_UNRECOGNISED
    assert att["unrecognised_ids_truncated"] is True
    assert att["n_unrecognised"] >= 300, "the count is kept even when the strings are not"


def test_an_unbound_catalog_produces_no_hallucination_rate_at_all(tmp_path):
    """A front end that forgets `bind_spec` must not manufacture a 100% misattribution rate.
    With nothing to check against, "recognised" and "hallucinated" are both unavailable, and
    reporting either would be a measurement of our wiring dressed as a fact about the model.
    """
    t = Tracer.create(tmp_path, "unbound")
    t.self_reported_rules("rules_applied: decision_rule.2", where="submit_answer")
    ev = [e for e in t.events if e["kind"] == "rules_self_reported"][0]
    assert ev["catalog_bound"] is False
    assert ev["unclassified_rule_ids"] == ["decision_rule.2"]
    att = t.rule_attribution()["self_reported"]
    assert att["n_unrecognised"] == 0 and att["rules_claimed"] == []


def test_the_prompt_asks_only_for_identifiers_the_spec_declares():
    """Ask for a free-text rule name and we guarantee a hallucination rate we then measure."""
    spec = _spec()
    block = rule_citation_block(spec)
    known = {r.rule_id for r in rule_catalog(spec)}
    good, bad = parse_rule_citations(block, known)
    assert good and not bad, f"the prompt itself names undeclared rules: {bad}"
    assert "answer_check." not in block, \
        "answer_checks are the code's channel; an agent naming one tells us nothing new"


# ============================================================ 5. end to end, in the manifest
class _ScriptedLLM:
    """A fixed tool script. No provider, no network, no cost.

    It cites a span out of a real search result because a made-up note_id is refused by the
    toolbox and the run would never reach the gate — and the gate is where attribution is
    written.
    """

    def __init__(self, value: dict, reasoning: str):
        from acr.core.llm import LLMClient, LLMConfig
        self._c = LLMClient(LLMConfig(model="scripted/none", api_key="none"))
        self.cfg = self._c.cfg
        self.value = value
        self.reasoning = reasoning
        self.prompt_tokens = self.completion_tokens = 0
        self.submits = 0
        self.prompts: list[str] = []

    def usage(self) -> dict:
        return {"prompt": 0, "completion": 0}

    def _reply(self, obj, calls=None):
        from acr.core.llm import LLMResponse
        return LLMResponse(content=json.dumps(obj), tool_calls=calls or [],
                           prompt_tokens=10, completion_tokens=5)

    def chat(self, messages, tools=None):
        last = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        # Every user turn, not just the last: the rule catalogue is planted in the opening
        # message and the check below is that the agent HAD it when it was asked to cite.
        self.prompts.extend(str(m.get("content") or "") for m in messages)
        if tools is None:
            if "SUFFICIENT|CONTINUE|STUCK" in last:      # reflect: REPLAN is no longer a verdict a model may pick
                return self._reply({"verdict": "CONTINUE", "reason": "still gathering"})
            if "FOUND|EVIDENCE_INSUFFICIENT|SPEC_INSUFFICIENT" in last:
                return self._reply({"status": "EVIDENCE_INSUFFICIENT", "value": {},
                                    "reasoning": "never finalised"})
            return self._reply({"plan": [{"id": "1", "goal": "find the pathology report",
                                          "rationale": "it establishes the histology"}]})
        hits = next((json.loads(m["content"]).get("hits") or []
                     for m in reversed(messages)
                     if m.get("role") == "tool" and m.get("name") == "search_notes"), [])
        if not hits:
            return self._reply({}, [{"id": "c0", "name": "list_documents", "arguments": {}},
                                    {"id": "c1", "name": "search_notes",
                                     "arguments": {"query": "carcinoma"}}])
        h = hits[0]
        self.submits += 1
        return self._reply({}, [
            {"id": f"e{self.submits}", "name": "record_evidence",
             "arguments": {"note_id": h["note_id"], "start": h["start"], "end": h["end"],
                           "supports": "histology"}},
            {"id": f"s{self.submits}", "name": "submit_answer",
             "arguments": {"status": "FOUND", "value": self.value,
                           "reasoning": self.reasoning}}])


def _run(tmp_path, value, reasoning, run_id):
    llm = _ScriptedLLM(value, reasoning)
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from hooks_harness import run_with_script

    from acr.chartstore.corpus import Corpus
    manifest, trace = run_with_script(_spec(), Corpus(site.corpus_root()), "SYN0001",
                                      tmp_path, llm, run_id=run_id, max_model_calls=6)
    return llm, manifest, manifest, trace


def test_a_completed_run_can_be_attributed_from_its_manifest_alone(tmp_path):
    """The §6b loop reads a directory of finished runs. Attribution recoverable only by
    replaying a JSONL file is attribution nobody computes."""
    _, result, manifest, _ = _run(
        tmp_path, {"primary_site": "C341", "histology": "8140", "behavior": "3"},
        "Coded from the final pathology.\nrules_applied: decision_rule.2, conflict_rule.1",
        "attr-clean")
    att = manifest["rule_attribution"]
    assert att is result["rule_attribution"] or att == result["rule_attribution"]
    assert att["spec_hash"] == manifest["spec_hash"]
    assert att["rule_catalog_hash"] and att["rule_catalog"]
    assert att["self_reported"]["rules_claimed"] == ["conflict_rule.1", "decision_rule.2"]
    assert att["deterministic"]["n_gate_evaluations"] >= 1
    assert att["deterministic"]["rejected_by"] == [], "this answer is clean"
    # every id in the block resolves against the catalog shipped with the same manifest
    catalog_ids = {r["rule_id"] for r in att["rule_catalog"]}
    assert set(att["self_reported"]["rules_claimed"]) <= catalog_ids


def test_the_trace_carries_the_catalog_so_an_id_is_readable_after_the_spec_moves_on(tmp_path):
    _, _, _, trace = _run(tmp_path, {"primary_site": "C341", "histology": "8140",
                                     "behavior": "3"},
                          "rules_applied: decision_rule.2", "attr-catalog")
    cat = [e for e in trace if e["kind"] == "rule_catalog"]
    assert len(cat) == 1, "bound exactly once, before anything can cite it"
    assert cat[0]["provenance"] == PROV_DETERMINISTIC
    entry = next(r for r in cat[0]["rules"] if r["rule_id"] == "decision_rule.2")
    assert entry["text"] and entry["text_sha"]


def test_a_shape_miss_is_recorded_against_its_rule_without_refusing_the_answer(tmp_path):
    """RECORDED, NOT REFUSED, as of 2026-07-30.

    `field_format` was the last check in `gate_answer` that judged the CONTENT of an answer, and
    it is gone: the constraint is already in the prompt (`as_prompt_block` renders every field's
    `format` and `allowable_values`, and STORE.400's own description says "no decimal point"), so
    a model that writes `C3412` against it has failed to follow an instruction rather than been
    under-informed. Four of that check's six useful firings rejected `C34.9`/`C34.11`/`C34.2` --
    the punctuated form ICD-O-3 itself writes -- so it was largely creating the round trips it
    then resolved.

    What must survive is the MEASUREMENT: the trace still records which declared shape the answer
    missed, so the eval plane can count instruction-following failures instead of a gate silently
    absorbing them.
    """
    _, result, manifest, trace = _run(
        tmp_path, {"primary_site": "C3412", "histology": "8140", "behavior": "3"},
        "rules_applied: decision_rule.1", "attr-shape-recorded")

    misses = [e for e in trace if e["kind"] == "answer_shape_miss"]
    assert misses, "the shape miss must still reach the trace"
    assert "field_format.primary_site" in misses[0]["rules"]
    assert misses[0]["refused"] is False

    # Its OWN event kind, deliberately. `rule_rejection` and the consecutive-rejection streak
    # counter are accounting about refusals; recording an accepted answer there would make the
    # manifest report a rejection that never happened.
    assert not [e for e in trace if e["kind"] == "rule_rejection"], (
        "nothing refuses over a field shape any more")
    assert result["gate_validated"] is True, "the answer stands"


def test_the_agent_is_shown_the_identifiers_before_it_is_asked_to_cite_them(tmp_path):
    """Asked at the last moment to cite ids it has never seen, a model invents them, and the
    self-report channel then measures our prompt rather than its reasoning."""
    llm, _, _, _ = _run(tmp_path, {"primary_site": "C341", "histology": "8140",
                                   "behavior": "3"},
                        "rules_applied: decision_rule.2", "attr-prompt")
    assert any("RULE IDENTIFIERS FOR THIS SPECIFICATION" in p for p in llm.prompts)


def test_attribution_never_changes_a_gate_verdict(tmp_path):
    """Recording must be inert. A caller may call `gate_answer` with no tracer at all, and
    the two front ends must not be able to reach different verdicts."""
    spec = _spec()
    led = _ledger(spec)
    ledger = EvidenceLedger()
    note = led.by_stratum["can_establish"][0]
    ledger.add(Evidence(note_id=note.note_id, doc_type=note.doc_type, date="2020-01-01",
                        start=0, end=10, quote="adenocarcinoma", supports="histology"))
    chart = Corpus(site.corpus_root()).chart("SYN0001")
    submitted = {"status": "FOUND", "value": {"primary_site": "C349", "histology": "8140"},
                 "reasoning": "rules_applied: decision_rule.1"}
    untraced = gate_answer(spec, submitted, evidence=ledger, coverage=led, chart=chart)
    traced = gate_answer(spec, submitted, evidence=ledger, coverage=led, chart=chart,
                         tracer=Tracer.create(tmp_path, "inert"))
    assert untraced == traced


# ---------------------------------------------------- conflict_requires_nos, from a real run
# `not_less_specific` guards one direction only: coding NOS when the record is specific.
# Nothing guarded the other. On the 10-patient real batch of 2026-07-27, one answer quoted an
# operative note ("coming and arising from the right middle lobe") AND a pathology header
# ("Lung, right lower lobe") in the SAME evidence list, coded C342, and passed the gate. The
# registry coded C349. The quotes below are that answer's, trimmed.

CONFLICT_CHECK = [{
    "field": "primary_site", "kind": "conflict_requires_nos", "nos_value": "C349",
    "mutually_exclusive": [["upper lobe", "RUL", "LUL"], ["middle lobe", "RML"],
                           ["lower lobe", "RLL", "LLL"], ["main bronchus"]],
    "message": "say which statement describes the origin, or code C349."}]


def _two_lobes():
    return [{"field": "primary_site", "quote": "coming and arising from the right middle lobe",
             "supports": "primary_site", "stance": "supports"},
            {"field": "primary_site", "quote": 'B. Lung, right lower lobe "excision"',
             "supports": "primary_site", "stance": "supports"}]


def test_conceding_the_conflict_is_the_remedy_and_is_not_refused():
    """C349 IS the answer this check asks for; firing on it would reject the fix."""
    assert check_answer_detail(CONFLICT_CHECK, {"primary_site": "C349"}, _two_lobes()) == []


def test_one_lobe_named_twice_is_not_a_conflict():
    ev = [{"field": "primary_site", "quote": "right middle lobe mass", "supports": "primary_site"},
          {"field": "primary_site", "quote": "RML lesion, 2.1 cm", "supports": "primary_site"}]
    assert check_answer_detail(CONFLICT_CHECK, {"primary_site": "C342"}, ev) == []


def test_the_check_reads_only_what_the_answer_cited():
    """Reading the chart here would ask the agent to defend text it never saw."""
    ev = [{"field": "primary_site", "quote": "right middle lobe mass", "supports": "primary_site"}]
    assert check_answer_detail(CONFLICT_CHECK, {"primary_site": "C342"}, ev) == []


def _site_checks():
    from acr.contract.spec import load_spec
    return load_spec("assets/specs/STORE.400_522_523.site_histology_behavior.yaml").answer_checks


def _kinds(ev, code):
    v = check_answer_detail(_site_checks(),
                            {"primary_site": code, "histology": "8070", "behavior": "3"},
                            ev, searched=["lobe", "bronchus"])
    return sorted({x.rule_kind for x in v})


# REMOVED 2026-07-30 with the checks it was about: `test_self_contradicting_evidence_leaves_nos
# _reachable`. It asserted that at least one primary_site code remained admissible when the
# cited evidence named three mutually exclusive lobes -- the deadlock `not_less_specific` and
# `conflict_requires_nos` could put a run into, where NOS was refused for being too vague and
# every specific code was refused for conflicting with the record. One real run resubmitted 24
# times into that trap and burned its whole call budget, and the value it was refused first was
# the registry's answer.
#
# With both checks gone the property is vacuous: nothing refuses any code, so every candidate is
# admissible and the assertion cannot fail for any reason connected to what it was testing.


