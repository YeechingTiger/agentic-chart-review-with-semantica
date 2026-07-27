"""The three invariants that make the MCP surface auditable rather than decorative.

Exposing the toolbox over MCP moves the trust boundary. Inside `ChartReviewAgent` the model
was a component we constructed; across MCP the caller is an arbitrary client that can send
any JSON it likes, in any order, as many times as it likes. Every guarantee that used to
hold because no code path existed to break it now has to hold because a check refuses.

The three that carry the weight, from the design doc:

  1. the sampling draw is the server's, not the caller's;
  2. `gate.check` is the only thing that can mark an answer validated;
  3. registry ground truth never shares a session with extraction.

Each is tested by what it must REFUSE. A permission check that has stopped checking looks
exactly like a clean pass from the outside, which is why `test_gate_must_reject.py` exists
one layer down and why every assertion here is of the same shape.

Almost everything below drives `ChartReviewService.call` in-process. That is the single entry
point the MCP adapter shims over, so it is the same code path the wire takes — a suite that
reached the handlers by some other route would be proving something about a path no client
can take. One test at the end runs a real client over the protocol, because the shim itself
is the one place an in-process guarantee could still be lost.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from acr.mcp_server import (
    MCP_TOOLS,
    STEERING_ARGS,
    ChartReviewService,
    RunSession,
    build_mcp_server,
)

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "corpus" / "patients"
SPECS = ROOT / "specs"
SITE = "STORE.400_522_523.site_histology_behavior"

# SYN0001 is the FOUND case (tissue diagnosis present); SYN0002 is the evidence-gap case
# whose biopsy was done at an outside hospital. Both are needed: a negative-only suite
# cannot tell a working gate from one that refuses everything.
FOUND_PATIENT, GAP_PATIENT = "SYN0001", "SYN0002"
# A real span in SYN0001, from a search hit: the word "adenocarcinoma" in the microscopic
# description. Deliberately NOT the SPECIMEN line — `origin_not_specimen` fires when every
# cited quote looks like a specimen label, and that is a check we want live, not dodged.
FOUND_EVIDENCE = [{"note_id": "Surgical-Pathology-Report_2023-04-27",
                   "start": 467, "end": 481, "supports": "histology"}]
FOUND_VALUE = {"primary_site": "C341", "histology": "8140", "behavior": "3"}


def service(**kw) -> ChartReviewService:
    kw.setdefault("seed_secret", b"unit-test-secret")
    kw.setdefault("truth_token", "eval-token")
    return ChartReviewService(CORPUS, SPECS, **kw)


def plan(svc: ChartReviewService, patient: str = GAP_PATIENT) -> str:
    return svc.call("coverage.plan", {"patient": patient, "spec_id": SITE})["run_id"]


def required_searches(svc: ChartReviewService, run_id: str) -> list[str]:
    """Every keyword the run's strata oblige the caller to search for."""
    specs = svc._runs_by_id[run_id].coverage.specs
    return list(dict.fromkeys(k for s in specs for k in s.required_keywords))


def discharge_the_sample(svc: ChartReviewService, run_id: str, patient: str) -> None:
    """Do the work the gate asks for: list the chart, run the searches the spec requires,
    then read every document the server drew.

    The searches are not decoration. `may_mention` is
    `search_then_read_hits_and_sample_misses`, so its miss sample only measures anything
    relative to a search that actually ran — skip them and every document is trivially a
    "miss". See test_a_run_with_zero_searches_must_not_reach_keyword_list_validated, which
    pins the version of this helper that did no searching at all.
    """
    svc.call("chart.list_documents", {"patient": patient, "limit": 5})
    for kw in required_searches(svc, run_id):
        found = svc.call("chart.search", {"patient": patient, "query": kw})
        # Read the hits. That clause sits in the middle of the policy name and used to be
        # enforced by nothing: a hit is excluded from the miss-sampling frame, so an unread
        # hit is audited by neither route.
        for h in found.get("hits", []):
            svc.call("chart.read",
                     {"note_id": h["note_id"], "patient": patient, "limit": 1200})
    drawn = svc.call("coverage.pending_samples", {"run_id": run_id})
    for docs in drawn["strata"].values():
        for d in docs:
            svc.call("chart.read", {"note_id": d["note_id"], "patient": patient, "limit": 1200})


# ------------------------------------------------------------------ invariant 1: the seed
def test_a_caller_supplied_seed_is_refused_not_ignored():
    """Silently ignoring it is worse than refusing it.

    A caller that believes it pinned the seed will publish a "reproducible" sample that
    nobody can reproduce, and the mismatch surfaces only when an auditor tries.
    """
    svc = service()
    run_id = plan(svc)
    honest = svc.call("coverage.pending_samples", {"run_id": run_id})

    steered = svc.call("coverage.pending_samples", {"run_id": run_id, "seed": 1})
    assert steered["error"] == "SAMPLE_IS_SERVER_DRAWN"
    assert "seed" in steered["rejected_arguments"]
    assert svc.call("coverage.pending_samples", {"run_id": run_id}) == honest, (
        "a rejected steering attempt must not perturb the outstanding draw either"
    )


@pytest.mark.parametrize("arg", sorted(STEERING_ARGS))
def test_every_way_of_steering_the_draw_is_refused(arg):
    """`seed` is the obvious one. `n`, `note_ids` and `stratum` reach the same end by
    choosing the sample directly instead of choosing the randomness that picks it."""
    svc = service()
    out = svc.call("coverage.pending_samples", {"run_id": plan(svc), arg: 1})
    assert out["error"] == "SAMPLE_IS_SERVER_DRAWN", f"{arg} steered the draw"


def test_the_declared_schema_admits_nothing_but_run_id():
    """Defence in depth: the SDK validates against inputSchema before the handler runs, so
    the schema must not quietly re-open what the handler closes."""
    tool = next(t for t in MCP_TOOLS if t["name"] == "coverage.pending_samples")
    assert set(tool["inputSchema"]["properties"]) == {"run_id"}
    assert tool["inputSchema"]["additionalProperties"] is False


def test_the_seed_is_reproducible_from_the_server_secret():
    """The draw has to be replayable by an auditor months later, or "the seed is recorded"
    buys nothing."""
    a, b = service(), service()
    ra, rb = plan(a), plan(b)
    da = a.call("coverage.pending_samples", {"run_id": ra})
    db = b.call("coverage.pending_samples", {"run_id": rb})
    assert da["sample_seed"] == db["sample_seed"]
    assert da["strata"] == db["strata"] and da["n_outstanding"] > 0

    other = service(seed_secret=b"a-different-secret")
    do = other.call("coverage.pending_samples", {"run_id": plan(other)})
    assert do["sample_seed"] != da["sample_seed"], (
        "the seed must depend on the server secret, or it is not server-held at all"
    )


def test_an_ephemeral_secret_says_so_instead_of_implying_reproducibility(monkeypatch):
    """A generated secret dies with the process. Claiming a replayable draw anyway is the
    failure `note_type_source: NOT_WIRED` exists to prevent one layer down."""
    monkeypatch.delenv("ACR_SAMPLE_SEED_SECRET", raising=False)
    svc = ChartReviewService(CORPUS, SPECS, truth_token="t")
    assert "NOT_REPRODUCIBLE" in svc.seed_provenance
    assert "NOT_REPRODUCIBLE" in svc.call(
        "coverage.pending_samples", {"run_id": plan(svc)})["seed_provenance"]


def test_replanning_cannot_reroll_the_seed():
    """The one hole a run_id-derived seed would leave open: mint runs until the draw looks
    convenient. Freezing the plan per (patient, spec) closes it — same question, same seed."""
    svc = service()
    first = svc.call("coverage.plan", {"patient": GAP_PATIENT, "spec_id": SITE})
    second = svc.call("coverage.plan", {"patient": GAP_PATIENT, "spec_id": SITE})
    assert first["minted"] is True and second["minted"] is False
    assert first["run_id"] == second["run_id"]
    assert first["sample_seed"] == second["sample_seed"]
    assert first["plan"] == second["plan"] and second["frozen"] is True


def test_the_sample_is_large_enough_to_be_a_sample():
    """A draw of one satisfies the letter of forced sampling and establishes nothing; the
    spec's min_sample is what makes the elusion bound mean something."""
    svc = service()
    drawn = svc.call("coverage.pending_samples", {"run_id": plan(svc)})
    assert drawn["n_outstanding"] >= 25 and drawn["drawn_by"] == "server"


# --------------------------------------------------------- invariant 2: only gate.check
def test_no_other_tool_can_mark_an_answer_validated():
    """Every tool on the surface, called for real. Only one of them may flip the bit."""
    svc = service()
    run_id = plan(svc, FOUND_PATIENT)
    run = svc._runs_by_id[run_id]
    args = {
        "chart.type_summary": {"patient": FOUND_PATIENT},
        "chart.list_documents": {"patient": FOUND_PATIENT},
        "chart.search": {"patient": FOUND_PATIENT, "query": "carcinoma"},
        "chart.read": {"note_id": FOUND_EVIDENCE[0]["note_id"], "patient": FOUND_PATIENT},
        "chart.timeline": {"patient": FOUND_PATIENT},
        "coverage.plan": {"patient": FOUND_PATIENT, "spec_id": SITE},
        "coverage.pending_samples": {"run_id": run_id},
        "registry.truth": {"patient": FOUND_PATIENT, "variable": SITE, "token": "eval-token"},
    }
    for name, a in args.items():
        svc.call(name, a)
        assert run.validated is False, f"{name} validated an answer"


def test_a_self_attested_answer_is_stripped_and_still_refused():
    """The whole point of moving the gate server-side. A client that sends
    `gate_validated: true` must be told its claim was discarded, not have it honoured and
    not have it dropped in silence."""
    svc = service()
    run_id = plan(svc)
    out = svc.call("gate.check", {"run_id": run_id, "answer": {
        "status": "EVIDENCE_INSUFFICIENT", "reasoning": "trust me",
        "gate_validated": True, "coverage_attested": {"strata": []},
        "proof_basis": "WITNESS", "negative_basis": "GATE_VALIDATED"}})

    assert out["verdict"] == "FAIL"
    assert out["answer"] is None
    assert svc._runs_by_id[run_id].validated is False
    assert set(out["ignored_client_claims"]) == {
        "gate_validated", "coverage_attested", "proof_basis", "negative_basis"}


def test_a_passing_gate_is_what_sets_validated():
    """The gate has to be passable by doing the work, or it is broken rather than strict."""
    svc = service()
    run_id = plan(svc)
    run = svc._runs_by_id[run_id]
    discharge_the_sample(svc, run_id, GAP_PATIENT)

    out = svc.call("gate.check", {"run_id": run_id, "answer": {
        "status": "EVIDENCE_INSUFFICIENT", "value": {},
        "reasoning": "the confirming pathology was done at an outside hospital"}})

    assert out["verdict"] == "PASS", out["missing"]
    assert run.validated is True
    assert out["answer"]["negative_basis"] == "GATE_VALIDATED"
    assert out["answer"]["coverage_attested"]["sample_seed"] == run.sample_seed


def test_a_run_with_zero_searches_must_not_reach_keyword_list_validated():
    """THE GATE INVERSION: doing LESS work must not make the gate EASIER to pass.

    `may_mention` carries policy `search_then_read_hits_and_sample_misses`. Run no searches
    at all and `search_hit_notes` stays empty, so every one of the stratum's 105 documents is
    trivially a "miss". The sampler dutifully draws its 25 from the whole stratum, the agent
    reads them, cites none of them — and the ledger announced `keyword_list_validated: True`.

    So the agent that searched properly and turned up one relevant document in its miss
    sample FAILED the gate, while the agent that never searched at all PASSED it. The check
    that exists to price the search obligation was refunding it.

    Asserts the stratum verdict and not merely the gate verdict, so this keeps testing the
    inversion even if some unrelated check fails the gate for its own reasons.
    """
    svc = service()
    run_id = plan(svc)
    run = svc._runs_by_id[run_id]

    # Everything the discharged run does EXCEPT searching.
    svc.call("chart.list_documents", {"patient": GAP_PATIENT, "limit": 5})
    drawn = svc.call("coverage.pending_samples", {"run_id": run_id})
    for docs in drawn["strata"].values():
        for d in docs:
            svc.call("chart.read",
                     {"note_id": d["note_id"], "patient": GAP_PATIENT, "limit": 1200})

    out = svc.call("gate.check", {"run_id": run_id, "answer": {
        "status": "EVIDENCE_INSUFFICIENT", "value": {},
        "reasoning": "nothing found — but nothing was searched for either"}})

    assert run.coverage.searched_terms == [], "precondition: this run searched for nothing"
    may = next(r for r in run.coverage.stratum_results() if r.name == "may_mention")
    assert may.misses_sampled >= 25, "precondition: the miss sample was drawn and inspected"
    assert may.miss_sample_hits == 0, "precondition: the clean sample that used to buy a pass"
    assert may.keywords_unsearched == may.required_keywords != []

    assert may.keyword_list_validated is False, (
        "a keyword list nobody searched cannot have been validated by sampling its misses")
    assert out["verdict"] == "FAIL"
    assert run.validated is False
    assert any("never ran" in m for m in out["missing"]), out["missing"]


def test_an_untouched_chart_cannot_pass_the_gate():
    """Guards the regression `test_gate_must_reject` guards downstream: a checker that has
    stopped checking is indistinguishable from a clean pass."""
    svc = service()
    out = svc.call("gate.check", {"run_id": plan(svc), "answer": {
        "status": "EVIDENCE_INSUFFICIENT", "reasoning": "nothing found"}})
    assert out["verdict"] == "FAIL" and out["missing"]


def test_a_witness_proved_positive_carries_no_coverage_claim():
    """A FOUND never claimed the universe was searched. Attaching the ledger would advertise
    a stronger claim than anything that was verified."""
    svc = service()
    run_id = plan(svc, FOUND_PATIENT)
    out = svc.call("gate.check", {"run_id": run_id, "answer": {
        "status": "FOUND", "value": FOUND_VALUE, "reasoning": "tissue diagnosis",
        "evidence": FOUND_EVIDENCE}})

    assert out["verdict"] == "PASS", out["missing"]
    assert out["answer"]["proof_basis"] == "WITNESS"
    assert "coverage_attested" not in out["answer"]


def test_a_malformed_code_is_refused_before_any_clinical_judgement():
    """`C3412` reached a manifest on 2026-07-26 because the declared format was rendered into
    the prompt and never enforced. The MCP front end must not reopen that."""
    svc = service()
    run_id = plan(svc, FOUND_PATIENT)
    out = svc.call("gate.check", {"run_id": run_id, "answer": {
        "status": "FOUND", "value": {**FOUND_VALUE, "primary_site": "C3412"},
        "reasoning": "x", "evidence": FOUND_EVIDENCE}})

    assert out["verdict"] == "FAIL"
    assert any("C3412" in m for m in out["missing"])
    assert svc._runs_by_id[run_id].validated is False


def test_a_fabricated_quote_cannot_enter_the_ledger():
    """Evidence arrives with the answer, so the server re-reads every span against the file.
    An out-of-range span clips to empty and is rejected rather than stored."""
    svc = service()
    run_id = plan(svc, FOUND_PATIENT)
    out = svc.call("gate.check", {"run_id": run_id, "answer": {
        "status": "FOUND", "value": FOUND_VALUE, "reasoning": "x",
        "evidence": [dict(FOUND_EVIDENCE[0], start=999_000, end=999_100)]}})

    assert out["verdict"] == "FAIL"
    assert out["evidence_recorded"][0]["accepted"] is False
    assert svc._runs_by_id[run_id].evidence.items == []


def test_validated_has_exactly_one_origin_in_the_source():
    """If this count ever exceeds one, the gate has grown a second door.

    Mirrors `test_one_route_to_an_answer.test_gate_validated_has_exactly_one_origin`: the
    behavioural tests above can only probe the tools that exist today, so the source-level
    count is what catches a new writer added tomorrow.
    """
    import re

    body = (ROOT / "src" / "acr" / "mcp_server.py").read_text(encoding="utf-8")
    code = "\n".join(ln for ln in body.splitlines() if not ln.strip().startswith("#"))
    assignments = re.findall(r"\.validated\s*=\s*True", code)
    assert len(assignments) == 1, (
        f"validated is set True in {len(assignments)} places; only gate.check may set it"
    )
    assert "validated" in RunSession.__dataclass_fields__


# ------------------------------------------------- invariant 3: truth never meets extraction
def test_registry_truth_is_unavailable_without_the_credential(monkeypatch):
    """The correct state for a server doing extraction. Unset env means unavailable, not
    open — an absent credential must never degrade to an absent check."""
    monkeypatch.delenv("ACR_REGISTRY_TRUTH_TOKEN", raising=False)
    svc = ChartReviewService(CORPUS, SPECS, seed_secret=b"s")
    for token in ("", "anything", "None"):
        out = svc.call("registry.truth",
                       {"patient": GAP_PATIENT, "variable": SITE, "token": token})
        assert out["error"] == "REGISTRY_TRUTH_NOT_CONFIGURED"


def test_the_credential_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("ACR_REGISTRY_TRUTH_TOKEN", "from-the-env")
    svc = ChartReviewService(CORPUS, SPECS, seed_secret=b"s")
    assert svc.call("registry.truth", {"patient": GAP_PATIENT, "variable": SITE,
                                       "token": "wrong"})["error"] == "REGISTRY_TRUTH_FORBIDDEN"
    ok = svc.call("registry.truth", {"patient": GAP_PATIENT, "variable": SITE,
                                     "token": "from-the-env"})
    assert ok["truth"]["status"] == "EVIDENCE_INSUFFICIENT"


def test_truth_is_refused_for_a_patient_this_session_reviewed():
    """The stated rule: ground truth leaking into an extraction run voids every number
    downstream, so the refusal has to happen before the read, not be audited after it."""
    svc = service()
    svc.call("chart.search", {"patient": GAP_PATIENT, "query": "carcinoma"})
    out = svc.call("registry.truth",
                   {"patient": GAP_PATIENT, "variable": SITE, "token": "eval-token"})

    assert out["error"] == "REGISTRY_TRUTH_WOULD_LEAK"
    assert "chart.search" in out["served"]


def test_a_coverage_call_alone_is_enough_to_quarantine_the_patient():
    """`coverage.plan` reads no document text, but it commits the session to reviewing this
    patient. Truth after it is still truth during extraction."""
    svc = service()
    plan(svc)
    assert svc.call("registry.truth", {"patient": GAP_PATIENT, "variable": SITE,
                                       "token": "eval-token"})["error"] == "REGISTRY_TRUTH_WOULD_LEAK"


def test_extraction_is_refused_after_truth_was_served():
    """The reordering attack, and the reason the check runs in both directions.

    A one-directional rule — "no truth after extraction" — is defeated by asking for the
    answer first and then opening the chart. The harm is identical either way: the ground
    truth is in the caller's context while the chart is being read.
    """
    svc = service()
    got = svc.call("registry.truth",
                   {"patient": GAP_PATIENT, "variable": SITE, "token": "eval-token"})
    assert got["truth"]["status"] == "EVIDENCE_INSUFFICIENT"

    for name, args in (("chart.type_summary", {"patient": GAP_PATIENT}),
                       ("chart.search", {"patient": GAP_PATIENT, "query": "carcinoma"}),
                       ("coverage.plan", {"patient": GAP_PATIENT, "spec_id": SITE})):
        out = svc.call(name, args)
        assert out["error"] == "EXTRACTION_AFTER_TRUTH_REFUSED", f"{name} ran after truth"


def test_the_quarantine_follows_the_run_not_just_the_patient_argument():
    """`coverage.pending_samples` and `gate.check` name a run_id, not a patient. Resolving
    the patient through the run is what stops the quarantine being sidestepped by argument
    shape alone."""
    svc = service()
    run_id = plan(svc)
    svc.truth_served.add(GAP_PATIENT)     # as if truth had been served in this session
    for name, args in (("coverage.pending_samples", {"run_id": run_id}),
                       ("gate.check", {"run_id": run_id, "answer": {"status": "FOUND"}})):
        assert svc.call(name, args)["error"] == "EXTRACTION_AFTER_TRUTH_REFUSED"


def test_a_refused_truth_call_does_not_quarantine_the_patient():
    """A wrong token discloses nothing, so it must not lock the patient out of extraction —
    otherwise any caller can deny review of any patient by guessing at the credential."""
    svc = service()
    assert svc.call("registry.truth", {"patient": GAP_PATIENT, "variable": SITE,
                                       "token": "guess"})["error"] == "REGISTRY_TRUTH_FORBIDDEN"
    assert GAP_PATIENT not in svc.truth_served
    assert svc.call("chart.type_summary", {"patient": GAP_PATIENT})["n_documents"] > 0


def test_the_quarantine_is_per_patient_not_per_session():
    """Blanket-locking the session would make a mixed eval run impossible and would push
    users to disable the check entirely."""
    svc = service()
    svc.call("chart.type_summary", {"patient": GAP_PATIENT})
    out = svc.call("registry.truth",
                   {"patient": FOUND_PATIENT, "variable": SITE, "token": "eval-token"})
    assert out["truth"]["status"] == "FOUND"


def test_ground_truth_is_not_reachable_through_the_chart_tools():
    """`_ground_truth.json` sits inside the patient directory. If the chart layer ever starts
    serving non-.txt files, invariant 3 is bypassed without any tool being called."""
    svc = service()
    listing = svc.call("chart.list_documents", {"patient": GAP_PATIENT, "limit": 10_000})
    assert not any("ground_truth" in d["note_id"] for d in listing["documents"])
    hits = svc.call("chart.search", {"patient": GAP_PATIENT, "query": "designer_notes"})
    assert hits["n_hits"] == 0


# ------------------------------- invariant 3, part two: the quarantine key is not a string
#
# The audit finding these pin: the quarantine was keyed on WHAT THE CALLER TYPED, while the
# filesystem was reached by joining that same string onto the corpus root. Two different
# resolvers for one question. `./SYN0002` and `SYN0002` name one directory to `pathlib` and
# two patients to `set.add`, so truth was served and the chart stayed open. Every test below
# is one member of that class, and the class — not the three spellings in the report — is
# what has to be closed.

# Spellings that all reach ONE patient directory on disk, or try to reach outside the corpus
# entirely. None may both serve truth and leave that patient extractable.
DODGY_SPELLINGS = [
    "./SYN0002",              # the audit's first spelling
    "SYN0002/",               # the audit's second
    "../patients/SYN0002",    # the audit's third
    ".//SYN0002",             # doubled separator
    "SYN0002/.",              # trailing dot component
    "SYN0002//",              # trailing doubled separator
    "SYN0003/../SYN0002",     # `..` at depth, arriving back at the target
    "SYN0002/../SYN0002",     # `..` through the target itself
    "..",                     # the corpus root as a patient
    "SYN0002\\",              # windows separator, meaningless on posix but not a patient id
    "SYN%30002",              # percent literal
    "SYN0002%2F",             # url-encoded separator, undecoded
    "%2e%2fSYN0002",          # url-encoded `./`
    "SYN0002 ",               # trailing space
    " SYN0002",               # leading space
    "SYN0002.",               # trailing dot
    "syn0002",                # case difference: the same directory on a case-insensitive fs
    "SYN0002​",          # zero-width space
    "ＳＹＮ０００２",   # fullwidth lookalikes, NFKC -> SYN0002
    "SYN0002\x00",            # null byte truncation
]


def _canonical_ids(svc: ChartReviewService) -> set[str]:
    return set(svc.corpus.patient_ids())


@pytest.mark.parametrize("spelling", DODGY_SPELLINGS)
def test_no_spelling_both_serves_truth_and_leaves_the_chart_open(spelling):
    """The whole defect in one assertion, stated so it cannot be satisfied by patching
    spellings one at a time.

    A call is allowed to be refused, and it is allowed to serve truth — what it may never do
    is serve truth under a name the quarantine does not recognise. Before the fix,
    `registry.truth('./SYN0002')` returned SYN0002's registry answer and then
    `chart.search(patient='SYN0002')` ran normally, because `truth_served` held the literal
    string `'./SYN0002'` while `corpus.root / './SYN0002'` had opened SYN0002's directory.
    """
    svc = service()
    got = svc.call("registry.truth",
                   {"patient": spelling, "variable": SITE, "token": "eval-token"})
    if "truth" not in got:
        return  # refused: nothing leaked, nothing to quarantine

    for name, args in (("chart.type_summary", {"patient": GAP_PATIENT}),
                       ("chart.search", {"patient": GAP_PATIENT, "query": "carcinoma"}),
                       ("coverage.plan", {"patient": GAP_PATIENT, "spec_id": SITE})):
        out = svc.call(name, args)
        assert out.get("error") == "EXTRACTION_AFTER_TRUTH_REFUSED", (
            f"{spelling!r} served truth for SYN0002 and {name} still ran; "
            f"ledger={sorted(svc.truth_served)}")


@pytest.mark.parametrize("spelling", DODGY_SPELLINGS)
def test_no_spelling_reopens_a_chart_the_session_already_quarantined(spelling):
    """The same hole in the other direction, which is the one that actually gets used.

    Reviewing the chart first and then asking for the answer is the ordering a careless
    evaluation script falls into. The forward check refused `SYN0002` and every variant
    walked straight past it, so the reordering defence of
    `test_extraction_is_refused_after_truth_was_served` bought nothing.
    """
    svc = service()
    svc.call("chart.search", {"patient": GAP_PATIENT, "query": "carcinoma"})
    assert svc.call("registry.truth", {"patient": GAP_PATIENT, "variable": SITE,
                                       "token": "eval-token"})["error"] == \
        "REGISTRY_TRUTH_WOULD_LEAK"

    got = svc.call("registry.truth",
                   {"patient": spelling, "variable": SITE, "token": "eval-token"})
    assert "truth" not in got, (
        f"{spelling!r} served ground truth for a chart this session had already read: {got}")


def test_the_patient_argument_can_never_leave_the_corpus_root():
    """A patient id that is itself a path. `self.corpus.root / patient` discards the root
    outright when `patient` is absolute — `Path('/a/b') / '/etc'` is `Path('/etc')` — so the
    argument was a read primitive pointed anywhere the server process could reach, including
    a real-corpus directory mounted beside this one.
    """
    svc = service()
    outside = [
        str(CORPUS / FOUND_PATIENT),          # absolute, inside — still not an id
        "/etc",                               # absolute, outside
        "../../corpus/patients/SYN0001",      # relative escape and re-entry
        "../../..",                           # climb out of the corpus
        "SYN0002/../SYN0001",                 # ask as one patient, land on another
    ]
    for spelling in outside:
        got = svc.call("registry.truth",
                       {"patient": spelling, "variable": SITE, "token": "eval-token"})
        assert "truth" not in got, f"{spelling!r} read truth as a path: {got}"
        assert got["error"] in ("MALFORMED_PATIENT_ID", "UNKNOWN_PATIENT"), got
        chart = svc.call("chart.type_summary", {"patient": spelling})
        assert chart.get("error") in ("MALFORMED_PATIENT_ID", "UNKNOWN_PATIENT"), chart


def test_a_spelling_that_is_not_a_known_patient_is_refused_before_the_filesystem():
    """Refusing at the index, not at `open()`, is what makes the rule stable: a spelling that
    happens to hit a directory is not thereby a patient, and one that misses must not leak
    whether the directory exists."""
    svc = service()
    for spelling in ("SYN9999", "", "   ", ".", "..", "_ground_truth.json",
                     ".doc_type_vocab.json"):
        got = svc.call("chart.type_summary", {"patient": spelling})
        assert got.get("error") in ("MALFORMED_PATIENT_ID", "UNKNOWN_PATIENT",
                                    "PATIENT_REQUIRED"), (spelling, got)


def test_a_non_string_patient_argument_is_refused_not_stringified():
    """`str(pid)` turned every JSON type into a plausible-looking id: `['SYN0002']` became
    `"['SYN0002']"`, a distinct quarantine key that no longer matched anything."""
    svc = service()
    for junk in (None, 0, 12, ["SYN0002"], {"patient": "SYN0002"}, True):
        got = svc.call("chart.type_summary", {"patient": junk})
        assert "error" in got and "n_documents" not in got, (junk, got)


def test_the_quarantine_ledgers_hold_canonical_ids_and_nothing_else():
    """The ledger is the boundary's memory. If a caller can write its own spelling into it,
    the boundary has no memory at all — it has the caller's."""
    svc = service()
    known = _canonical_ids(svc)
    for spelling in DODGY_SPELLINGS + [GAP_PATIENT, FOUND_PATIENT]:
        svc.call("chart.type_summary", {"patient": spelling})
        svc.call("registry.truth",
                 {"patient": spelling, "variable": SITE, "token": "eval-token"})

    assert svc.extraction_touched <= known, sorted(svc.extraction_touched - known)
    assert svc.truth_served <= known, sorted(svc.truth_served - known)
    assert {a["patient"] for a in svc.access_log} <= known
    # And the two planes still cannot overlap on any one patient, whatever was typed.
    assert not (svc.extraction_touched & svc.truth_served)


def test_case_and_unicode_lookalikes_do_not_mint_a_second_identity():
    """On a case-insensitive filesystem `syn0002` and `SYN0002` open one directory, so keying
    on the raw string gives one chart two quarantine entries. NFKC folding matters for the
    same reason: fullwidth digits normalise to ASCII on the way to `open()`."""
    svc = service()
    served = svc.call("registry.truth",
                      {"patient": GAP_PATIENT, "variable": SITE, "token": "eval-token"})
    assert "truth" in served
    for spelling in ("syn0002", "SYN0002", "Syn0002",
                     "ＳＹＮ０００２"):
        out = svc.call("chart.type_summary", {"patient": spelling})
        assert out.get("error") in ("EXTRACTION_AFTER_TRUTH_REFUSED", "MALFORMED_PATIENT_ID",
                                    "UNKNOWN_PATIENT"), (spelling, out)


def test_the_audit_log_keeps_the_spelling_even_though_the_ledger_does_not():
    """Canonicalising must not erase the fact that a caller reached this chart under another
    name. The ledger has to key on identity or the quarantine fails; the log has to keep the
    spelling or a caller probing for a bypass looks exactly like ordinary traffic."""
    svc = service()
    svc.call("chart.type_summary", {"patient": "syn0002"})
    (entry,) = svc.access_log
    assert entry["patient"] == GAP_PATIENT
    assert entry["requested_as"] == "syn0002"

    svc.call("chart.type_summary", {"patient": GAP_PATIENT})
    assert "requested_as" not in svc.access_log[1]


SYN0002_NOTE = "Endo-Diab-MD-OP-Progress-Note_2013-07-02"


@pytest.mark.parametrize("spelling", DODGY_SPELLINGS)
def test_chart_read_canonicalises_too_despite_having_its_own_patient_branch(spelling):
    """`chart.read` is the ONLY tool with a bespoke branch in `_patient_for`, because its
    patient may be implied by the note_id instead of named. A bespoke branch is where a
    boundary drifts: it is the one place a future edit can stop calling `_canonical_patient`
    without any other tool noticing, and a read charged to the spelling instead of the
    identity is the whole defect back again one tool at a time.

    Asserted in the direction that actually gets used — read first, then ask for the answer.
    """
    svc = service()
    svc.call("chart.read", {"note_id": SYN0002_NOTE, "patient": spelling, "limit": 200})
    got = svc.call("registry.truth",
                   {"patient": GAP_PATIENT, "variable": SITE, "token": "eval-token"})
    if svc.extraction_touched:
        assert svc.extraction_touched == {GAP_PATIENT}, sorted(svc.extraction_touched)
        assert got.get("error") == "REGISTRY_TRUTH_WOULD_LEAK", (
            f"chart.read({spelling!r}) opened SYN0002's chart under a name the quarantine did "
            f"not recognise: {got}")


@pytest.mark.parametrize("spelling", DODGY_SPELLINGS)
def test_no_spelling_mints_a_second_run_over_one_chart(spelling):
    """A second run for one (chart, spec) is a second coverage ledger AND a second server
    draw, which is invariant 1 defeated through the patient field rather than through `seed`.

    `test_replanning_cannot_reroll_the_seed` freezes the run on `(patient, spec_id)`, so it
    only holds while one chart has one patient key. Every alias that mints its own run hands
    the caller a fresh sample to choose between — and a fresh, empty proof obligation to pass
    the gate against.
    """
    svc = service()
    first = svc.call("coverage.plan", {"patient": GAP_PATIENT, "spec_id": SITE})
    again = svc.call("coverage.plan", {"patient": spelling, "spec_id": SITE})
    if "run_id" not in again:
        return  # refused: no second ledger exists to choose between
    assert again["run_id"] == first["run_id"], (
        f"{spelling!r} minted a second run over SYN0002: {again['run_id']} vs "
        f"{first['run_id']}; runs={sorted(svc._runs_by_id)}")
    assert again["minted"] is False and again["sample_seed"] == first["sample_seed"]
    assert len(svc._runs) == 1


@pytest.mark.parametrize("spelling", DODGY_SPELLINGS)
def test_the_truth_handler_refuses_a_spelling_when_reached_directly(spelling):
    """`_h_registry_truth` re-canonicalises rather than trusting what it was handed, and its
    comment says so. Pinned, because a comment is not a check: this method builds the only
    path in the module that is still assembled by joining a patient onto a directory, and the
    argument reaching it canonical is a property of `call`, not of the method.
    """
    svc = service()
    out = svc._h_registry_truth(patient=spelling, variable=SITE, token="eval-token")
    if "truth" in out:
        # An alias of a real patient (case, unicode form) is allowed to resolve and answer.
        # What it may not do is answer UNDER THE SPELLING: the reported patient is the key the
        # ledger would be charged, so a spelling echoed back here is a ledger entry nobody can
        # match against a later chart call.
        assert out["patient"] == GAP_PATIENT, (
            f"the handler answered for {spelling!r} without resolving it: {out['patient']!r}")
    else:
        assert out["error"] in ("MALFORMED_PATIENT_ID", "UNKNOWN_PATIENT", "PATIENT_REQUIRED",
                                "AMBIGUOUS_PATIENT_ID"), out


def test_the_call_that_quarantines_is_the_call_that_discloses():
    """A finding, not a feature, and the sharpest form of it.

    The spelling fix makes the ledger key trustworthy. It does not make the quarantine
    preventive: the truth VALUE is in the payload of the very call that writes the ledger
    entry. There is no ordering in which the entry is written first and the value withheld —
    `registry.truth` has no dry-run, no "would this be allowed" form, and no way to learn that
    a patient is already extraction-touched without either being told (refused) or being told
    the answer. So for the first call on any patient, quarantine is strictly retrospective.

    If a preventive mode is ever added, this test should start failing and be replaced.
    """
    svc = service()
    out = svc.call("registry.truth",
                   {"patient": GAP_PATIENT, "variable": SITE, "token": "eval-token"})
    assert "truth" in out and out["truth"], "the payload the ledger entry is written beside"
    assert GAP_PATIENT in svc.truth_served
    assert not any(k in out for k in ("withheld", "dry_run", "value_withheld")), (
        "if registry.truth grew a way to quarantine WITHOUT disclosing, this finding is stale")


def test_the_quarantine_ledger_is_known_to_this_module_and_no_other():
    """The reason the boundary does not survive a different front end, stated as a fact about
    the source rather than as a claim in a docstring.

    `truth_served` is process memory inside one class. No other module reads it, nothing
    writes it to disk, and nothing under `runs/` records it — so the CLI and LangGraph paths
    review a chart with no knowledge that ground truth was served for it over MCP, and a
    reconnect to a fresh server starts clean. A grep is the right shape of test here: the
    behavioural version can only probe the front ends that exist today.
    """
    src = ROOT / "src" / "acr"
    holders = sorted(p.name for p in src.rglob("*.py")
                     if "truth_served" in p.read_text(encoding="utf-8"))
    assert holders == ["mcp_server.py"], (
        f"the quarantine ledger is now consulted by {holders}; if a second front end really "
        "does check it, this finding is narrower than it was and the docstring must say so")

    # And nothing in this module carries the ledger off the heap. Checked line by line rather
    # than by grepping the file for write verbs, which matches `json.dumps` in the MCP shim
    # and would pass or fail for reasons that have nothing to do with the ledger.
    body = (src / "mcp_server.py").read_text(encoding="utf-8").splitlines()
    writes = [ln.strip() for ln in body
              if ("truth_served" in ln or "extraction_touched" in ln)
              and any(w in ln for w in ("write", "dump", "json.", "Path(", "sqlite", "redis"))]
    assert writes == [], (
        f"the ledger is being persisted ({writes}); if a reconnect no longer starts clean, "
        "that finding needs re-testing rather than deleting")
    assert isinstance(service().truth_served, set), "process memory, and nothing more"


def test_a_note_id_cannot_be_a_path_either():
    """`chart.read` takes a note_id, and a note_id is looked up in an index rather than
    joined onto a directory. Pinned because the same mistake in the same file is what this
    section exists for: an id that is secretly a path charges one patient's read to another.
    """
    svc = service()
    svc.call("chart.type_summary", {"patient": GAP_PATIENT})
    for note_id in ("../SYN0001/Surgical-Pathology-Report_2023-04-27",
                    "../_ground_truth", "/etc/passwd", "_ground_truth"):
        got = svc.call("chart.read", {"note_id": note_id, "patient": GAP_PATIENT})
        assert "text" not in got, (note_id, got)
        assert "error" in got, (note_id, got)


def _tiny_corpus(root: Path, pid: str) -> None:
    """A one-patient synthetic corpus. Text is invented here; nothing is copied from a chart."""
    d = root / pid
    d.mkdir(parents=True)
    (d / "Surgical-Pathology-Report_2020-01-02.txt").write_text(
        "FINAL DIAGNOSIS:\nsynthetic fixture text, no patient data\n", encoding="utf-8")
    (d / "_ground_truth.json").write_text(
        json.dumps({"patient_id": pid, "ground_truth": {SITE: {"status": "FOUND"}}}),
        encoding="utf-8")


def test_a_symlinked_patient_directory_is_one_identity_not_two(tmp_path):
    """Two names, one directory — the same defeat as `./SYN0002` with the filesystem doing
    the aliasing instead of the string. Identity has to be resolved against what the name
    OPENS, not against the name, or a corpus with a sharded/relinked layout reopens the hole
    that the string checks just closed.
    """
    root = tmp_path / "patients"
    _tiny_corpus(root, "SYN0002")
    (root / "ALIAS0002").symlink_to(root / "SYN0002", target_is_directory=True)

    svc = ChartReviewService(root, SPECS, seed_secret=b"s", truth_token="eval-token")
    got = svc.call("registry.truth",
                   {"patient": "SYN0002", "variable": SITE, "token": "eval-token"})
    assert "truth" in got, got
    out = svc.call("chart.type_summary", {"patient": "ALIAS0002"})
    assert out.get("error") == "EXTRACTION_AFTER_TRUTH_REFUSED", out
    # The real directory names the identity, not the link, so the ledger reads the same
    # whichever way the corpus was laid out on this machine.
    assert svc.truth_served == {"SYN0002"}


def test_two_charts_that_differ_only_by_case_are_refused_not_merged(tmp_path):
    """Case folding is right for one chart under two names and catastrophic for two charts
    under one folded name — it would charge SYN0002's ground truth to syn0002's ledger entry
    and leave the other chart open. Refuse the pair; do not pick.

    The FIRST lookup is the one under test. An ambiguity discovered while building the index
    but consulted before the build would be invisible to exactly the call that opens a chart.
    """
    root = tmp_path / "patients"
    _tiny_corpus(root, "SYN0002")
    _tiny_corpus(root, "syn0002")   # a distinct directory, distinct inode

    # A chart call on a cold service, because `registry.truth` re-canonicalises inside the
    # handler and would mask the ordering: by the second resolution the index has been built
    # and the ambiguity is on record. The chart path has no second check, so it is the one
    # that shows whether the FIRST resolution was safe.
    cold = ChartReviewService(root, SPECS, seed_secret=b"s", truth_token="eval-token")
    assert cold.call("chart.type_summary",
                     {"patient": "SYN0002"}).get("error") == "AMBIGUOUS_PATIENT_ID"
    assert cold.extraction_touched == set()

    svc = ChartReviewService(root, SPECS, seed_secret=b"s", truth_token="eval-token")
    first = svc.call("registry.truth",
                     {"patient": "syn0002", "variable": SITE, "token": "eval-token"})
    assert first.get("error") == "AMBIGUOUS_PATIENT_ID", first
    assert svc.truth_served == set()


# ------------------------------- invariant 3, part three: the quarantine price is the
# CREDENTIAL CHECK, not the payload
#
# Canonicalising the patient closed the spelling class. It left a second way to ask the
# registry a question for free: ask about a variable that is not in the answer key. The token
# is verified, the patient is resolved, the ground-truth file is opened and parsed — and then
# the handler returns UNKNOWN_VARIABLE, which `call` treats as "nothing was served" and
# declines to quarantine. The caller learns which variables this patient is scored on, and
# whether this patient is in the answer key at all, and the chart stays open.
#
# The rule these pin: once the eval credential has been verified against a resolved patient,
# the session has consulted the registry about that patient. What came back does not matter.

def test_an_authenticated_probe_for_an_unknown_variable_still_quarantines():
    """The variable does not have to exist for the question to have been asked.

    `registry.truth(SYN0002, '__probe__', <valid token>)` reads SYN0002's answer key off disk
    and reports its key set in `available`. That is registry knowledge about this patient
    obtained with the eval credential, so it costs the same as a value. Before this test the
    ledger stayed empty and all 293 documents remained readable.
    """
    svc = service()
    out = svc.call("registry.truth", {"patient": GAP_PATIENT, "variable": "__probe__",
                                      "token": "eval-token"})
    assert out["error"] == "UNKNOWN_VARIABLE"
    assert out["available"], "the probe disclosed the answer key's variable set"
    assert GAP_PATIENT in svc.truth_served, (
        "an authenticated registry question was answered for this patient and charged to "
        f"nobody; ledger={sorted(svc.truth_served)}")
    assert svc.call("chart.type_summary",
                    {"patient": GAP_PATIENT})["error"] == "EXTRACTION_AFTER_TRUTH_REFUSED"


def test_the_answer_key_cohort_cannot_be_enumerated_for_free(tmp_path):
    """NO_GROUND_TRUTH and UNKNOWN_VARIABLE are different answers to "is this patient scored?"

    Distinguishing them tells an extraction session which patients are in the eval set — the
    shape of the benchmark it is being measured on — and, unquarantined, it can be asked for
    every patient in the corpus before a single chart is opened.
    """
    root = tmp_path / "patients"
    _tiny_corpus(root, "SYN0002")            # has an answer key
    unscored = root / "SYN0009"              # has none
    unscored.mkdir(parents=True)
    (unscored / "Progress-Note_2020-03-04.txt").write_text(
        "synthetic fixture text, no patient data\n", encoding="utf-8")

    for pid in ("SYN0002", "SYN0009"):
        svc = ChartReviewService(root, SPECS, seed_secret=b"s", truth_token="eval-token")
        out = svc.call("registry.truth",
                       {"patient": pid, "variable": "__probe__", "token": "eval-token"})
        assert out["error"] in {"UNKNOWN_VARIABLE", "NO_GROUND_TRUTH"}
        assert pid in svc.truth_served, f"{pid}: probed the answer key and paid nothing"
        assert svc.call("chart.type_summary",
                        {"patient": pid})["error"] == "EXTRACTION_AFTER_TRUTH_REFUSED"


def test_an_unauthenticated_probe_still_does_not_quarantine():
    """The other half, or the fix above becomes a denial-of-service primitive.

    Charging the quarantine to a caller who never proved it holds the credential would let
    any client lock any patient out of review by guessing. The credential check is the line:
    below it nothing is disclosed and nothing is charged; above it everything is charged.
    A holder of the token could already quarantine a patient by asking a real question, so
    this adds no capability it did not have.
    """
    svc = service()
    for variable in (SITE, "__probe__"):
        out = svc.call("registry.truth",
                       {"patient": GAP_PATIENT, "variable": variable, "token": "guess"})
        assert out["error"] == "REGISTRY_TRUTH_FORBIDDEN"
        assert "available" not in out, "a forbidden call disclosed the answer key's shape"
    for spelling in ("../patients/SYN0002", "NOSUCHPATIENT"):
        svc.call("registry.truth",
                 {"patient": spelling, "variable": SITE, "token": "eval-token"})
    assert svc.truth_served == set()
    assert svc.call("chart.type_summary", {"patient": GAP_PATIENT})["n_documents"] > 0


# ---------------------------------------- what the quarantine does NOT do, pinned on purpose
def test_the_quarantine_is_damage_limitation_and_the_test_suite_says_so():
    """Read this one as a finding, not a feature.

    `registry.truth` RETURNS the answer and quarantines afterwards. By the time the ledger is
    written the truth is already in the caller's context; every later refusal only stops the
    same session reading more chart. The ledger is process-memory, so a second service over
    the same corpus — a reconnect, a second client, a restarted server — starts clean and
    will happily extract the patient whose answer the caller is still holding.

    Fixing the spelling class does not change this. It is asserted here so that nobody reads
    a green invariant-3 section as "ground truth cannot reach an extraction run".
    """
    first = service()
    got = first.call("registry.truth",
                     {"patient": GAP_PATIENT, "variable": SITE, "token": "eval-token"})
    assert "truth" in got
    assert first.call("chart.type_summary",
                      {"patient": GAP_PATIENT})["error"] == "EXTRACTION_AFTER_TRUTH_REFUSED"

    second = service()
    assert second.truth_served == set(), "the ledger is not persisted anywhere"
    assert second.call("chart.type_summary", {"patient": GAP_PATIENT})["n_documents"] > 0


def test_truth_for_one_patient_rides_into_every_other_chart_in_the_session():
    """A finding, not a feature — and the one the per-patient rule cannot answer.

    `test_the_quarantine_is_per_patient_not_per_session` defends per-patient scoping on the
    grounds that a blanket lock would make mixed eval runs impossible. That is true, and it
    is only sound if a truth payload says nothing that generalises. This one does: the answer
    key carries a free-text `why` — here "No pathology in the record; histology must NOT be
    inferred from imaging" — which is corpus-design guidance, not a label. Once it is in the
    context, the same session extracts the remaining eleven charts with the fixture author's
    decision rule in front of it, and every refusal fires for SYN0002 alone.

    Asserted so the green invariant-3 section cannot be read as "ground truth does not reach
    an extraction run". It reaches ten of them; it just does not reach the one it names.
    """
    svc = service()
    truth = svc.call("registry.truth",
                     {"patient": GAP_PATIENT, "variable": SITE, "token": "eval-token"})["truth"]
    assert isinstance(truth.get("why"), str) and truth["why"], (
        "the payload is a label only; if that is now true, this finding can be closed")

    others = [p for p in svc.corpus.patient_ids() if p != GAP_PATIENT]
    still_open = [p for p in others
                  if not svc.call("chart.type_summary", {"patient": p}).get("error")]
    assert still_open == others, (
        "per-patient scoping is what this pins; if the quarantine went session-wide, "
        "delete this test and the finding with it")


def test_one_process_holds_both_the_extraction_surface_and_the_truth_credential():
    """The credential separation is a token, not a boundary.

    `ACR_REGISTRY_TRUTH_TOKEN` is read by the same object that serves `chart.read`, from the
    same environment, in the same process. A caller holding the token can review a chart and
    read the registry over one connection; the only thing between them is this class's own
    bookkeeping. A real separation would be a second server, under a second identity, with no
    corpus text on its surface at all.
    """
    svc = service()
    assert svc._truth_token, "the extraction surface is holding the eval credential"
    assert "registry.truth" in ChartReviewService.HANDLERS
    assert any(n.startswith("chart.") for n in ChartReviewService.HANDLERS)


# ------------------------------------------------------------------------ the surface
def test_every_declared_tool_is_dispatchable_and_vice_versa():
    """A tool advertised but not wired answers every call with UNKNOWN_TOOL; a tool wired but
    not advertised is unreachable from a client. Both are silent."""
    assert {t["name"] for t in MCP_TOOLS} == set(ChartReviewService.HANDLERS)
    svc = service()
    for name, handler in ChartReviewService.HANDLERS.items():
        assert hasattr(svc, handler), name


def test_the_declared_surface_is_the_one_in_the_design_doc():
    assert {t["name"] for t in MCP_TOOLS} == {
        "chart.type_summary", "chart.list_documents", "chart.search", "chart.read",
        "chart.timeline", "coverage.plan", "coverage.pending_samples", "gate.check",
        "registry.truth"}


def test_list_documents_returns_metadata_and_no_document_text():
    """The cheap orienting call must stay cheap, and must not become a way to bulk-export a
    chart under a name that says it will not."""
    svc = service()
    out = svc.call("chart.list_documents", {"patient": GAP_PATIENT, "limit": 5})
    assert out["documents"] and all(
        set(d) == {"note_id", "doc_type", "date", "seq", "n_chars"} for d in out["documents"])
    assert "text" not in json.dumps(out)


def test_chart_calls_before_a_plan_say_they_count_towards_nothing():
    """Otherwise a caller reads a hundred documents, then cannot understand why the gate
    reports that nothing was reviewed."""
    svc = service()
    early = svc.call("chart.type_summary", {"patient": GAP_PATIENT})
    assert early["coverage_recorded_into"] == [] and "coverage_note" in early

    run_id = plan(svc)
    later = svc.call("chart.type_summary", {"patient": GAP_PATIENT})
    assert later["coverage_recorded_into"] == [run_id]
    assert svc._runs_by_id[run_id].coverage.type_summary_seen is True


def test_an_ambiguous_note_id_is_refused_rather_than_guessed():
    """A note_id is a filename stem, unique only within a patient: 259 of 3,447 stems in this
    corpus occur under more than one. Guessing an owner would charge one patient's read to
    another in the quarantine ledger."""
    svc = service()
    shared = "Prescriptions-Filled-RxHub_2020-05-24"
    for pid in ("SYN0001", "SYN0006"):
        svc.call("chart.list_documents", {"patient": pid, "limit": 1})

    out = svc.call("chart.read", {"note_id": shared})
    assert out["error"] == "AMBIGUOUS_NOTE_ID"
    assert set(out["candidates"]) >= {"SYN0001", "SYN0006"}
    assert svc.call("chart.read", {"note_id": shared, "patient": "SYN0001"})["text"]


def test_an_unopened_note_id_is_refused_rather_than_searched_for():
    svc = service()
    out = svc.call("chart.read", {"note_id": "Surgical-Pathology-Report_2023-04-27"})
    assert out["error"] == "NOTE_OWNER_UNKNOWN"


def test_unknown_run_ids_and_tools_are_named_not_swallowed():
    svc = service()
    assert svc.call("coverage.pending_samples", {"run_id": "nope"})["error"] == "UNKNOWN_RUN_ID"
    assert svc.call("chart.teleport", {"patient": GAP_PATIENT})["error"] == "UNKNOWN_TOOL"
    assert svc.call("coverage.plan", {"patient": GAP_PATIENT,
                                      "spec_id": "STORE.999"})["error"] == "UNKNOWN_SPEC"


def test_the_service_never_raises_out_of_call():
    """A tool error is an observation the caller can act on; an exception kills the session
    and takes every frozen plan and drawn sample with it."""
    svc = service()
    for args in ({}, {"patient": 42}, {"patient": GAP_PATIENT, "limit": "lots"}):
        assert isinstance(svc.call("chart.list_documents", args), dict)


def test_the_gate_is_the_agent_s_gate_and_not_a_second_copy():
    """Two gate implementations that can disagree is the two-ledger failure one layer up."""
    import inspect

    from acr import graph, mcp_server
    assert mcp_server.gate_answer is graph.gate_answer
    assert "gate_answer(" in inspect.getsource(mcp_server.ChartReviewService._h_gate_check)
    assert "gate_answer(" in inspect.getsource(graph.ChartReviewAgent._gate)


def test_the_mcp_adapter_is_a_shim_over_the_same_entry_point():
    """If the wire path diverged from `call`, every test above would be exercising a route no
    client can take."""
    import inspect

    assert "service.call(" in inspect.getsource(build_mcp_server)
    assert build_mcp_server(service()) is not None


def test_the_service_works_without_the_mcp_sdk_installed():
    """The SDK is an optional extra, so the import has to stay inside `build_mcp_server`.

    Hoisting it to module scope would make `acr.mcp_server` unimportable wherever the extra
    is not installed, and the failure would land on `acr.graph`'s importers rather than on
    whoever wanted a server. Checked by actually blocking the import, not by reading the
    source: a transitive import through any other acr module would be just as fatal.
    """
    import subprocess
    import sys

    probe = (
        "import sys\n"
        "for m in list(sys.modules):\n"
        "    if m == 'mcp' or m.startswith('mcp.'): del sys.modules[m]\n"
        "sys.meta_path.insert(0, type('Block', (), {'find_module': None, 'find_spec':\n"
        "    staticmethod(lambda name, path=None, target=None: (_ for _ in ()).throw(\n"
        "        ImportError('mcp blocked')) if name.split('.')[0] == 'mcp' else None)})())\n"
        "from acr.mcp_server import ChartReviewService\n"
        f"svc = ChartReviewService({str(CORPUS)!r}, {str(SPECS)!r}, seed_secret=b's')\n"
        "out = svc.call('coverage.pending_samples', {'run_id': 'x', 'seed': 1})\n"
        "assert out['error'] == 'UNKNOWN_RUN_ID', out\n"
        "print('OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                       cwd=ROOT, check=False,
                       env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0 and "OK" in r.stdout, r.stderr[-800:]


def test_the_invariants_survive_a_real_client_session():
    """Everything above drives `call` in-process. This drives a genuine MCP client over the
    protocol, because the shim is where an in-process guarantee could still be lost — a
    handler refusal is worth nothing if the transport never reaches it, or reaches it with
    arguments the SDK reshaped on the way.

    It also pins a second line of defence: `additionalProperties: false` makes the SDK reject
    a stray `seed` during input validation, before the handler is entered at all.
    """
    anyio = pytest.importorskip("anyio")
    pytest.importorskip("mcp")
    from mcp.shared.memory import create_connected_server_and_client_session as connected

    svc = service()

    async def exercise() -> dict:
        async with connected(build_mcp_server(svc)) as client:
            await client.initialize()
            tools = [t.name for t in (await client.list_tools()).tools]
            planned = json.loads((await client.call_tool(
                "coverage.plan", {"patient": GAP_PATIENT, "spec_id": SITE}
            )).content[0].text)
            drawn = json.loads((await client.call_tool(
                "coverage.pending_samples", {"run_id": planned["run_id"]}
            )).content[0].text)
            steered = await client.call_tool(
                "coverage.pending_samples", {"run_id": planned["run_id"], "seed": 7})
            leaked = json.loads((await client.call_tool(
                "registry.truth", {"patient": GAP_PATIENT, "variable": SITE,
                                   "token": "eval-token"})).content[0].text)
            return {"tools": tools, "drawn": drawn, "leaked": leaked,
                    "steered_is_error": steered.isError}

    got = anyio.run(exercise)
    assert set(got["tools"]) == set(ChartReviewService.HANDLERS)
    assert got["drawn"]["drawn_by"] == "server" and got["drawn"]["n_outstanding"] >= 25
    assert got["steered_is_error"] is True, "the transport must refuse a seed as well"
    assert got["leaked"]["error"] == "REGISTRY_TRUTH_WOULD_LEAK"
