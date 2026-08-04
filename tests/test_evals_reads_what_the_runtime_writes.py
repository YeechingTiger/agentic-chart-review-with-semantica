"""The evaluation plane against the shapes the RUNTIME actually writes, not the ones it expects.

Every test here was written from a defect the 2026-08-03 pipeline audit found by reading real
manifests instead of fixtures. The common cause: `tests/test_evals.py::manifest` hand-authors
`coverage_attested` and a scalar `query`, and the runtime writes `coverage_state` and a LIST. Each
half passed its own tests forever.

Three properties, all measured over this tree's own 509 manifests:

1. `search_notes` is BATCHED — `toolbox.py` accepts a list of terms and records one coverage entry
   per term. 953 of 3,988 recorded search events carry a list-valued `query`. `searched_terms`
   stringified the whole list, so `str(["bx", "adenocarcinoma"])` was one opaque "term" and
   `detect_degenerate_search` could not see `bx` inside it. It reported 1 finding over the tree
   while `coverage_state.searched_terms` shows 5 degenerate terms.

2. The manifest fallback read `coverage_attested`, a key the runtime has never written — 13 of 509
   manifests carry it, none recent; 502 carry `coverage_state.searched_terms`.

3. `answer_key_from_corpus.py` writes BARE PATIENT IDS and no `spec_id`, so `score`'s composite
   lookup `f"{patient}__{spec}"` never hit, the bare fallback always did, `run.spec_id` was compared
   to nothing, and `n_unkeyed` was pinned at 0 whether coverage was complete or not. Runs of a
   DIFFERENT SPEC scored against the key silently: on `runs/`, 8 cross-spec runs dragged a published
   74.6% down from the spec's actual 75.8%.
"""

from __future__ import annotations

import json

from acr.evaluation import evals as E

SPEC = "SYN.400.site_histology"
OTHER_SPEC = "SYN.390.date_of_diagnosis"
CFG = E.DetectorConfig(min_term_chars=4, max_rejection_repeats=2,
                       token_band=(1_000, 200_000), turn_band=(3, 60))


def _runtime_manifest(tmp_path, patient="SYN0001", *, spec=SPEC, searched=("adenocarcinoma",),
                      queries=(), value=None, status="FOUND", name=None):
    """A manifest and trace in the shape `run_patient` writes: `coverage_state`, path-valued trace.

    `queries` are recorded as `search_notes` trace events, each exactly as the toolbox records
    them — which may be a LIST when the model batched its terms.
    """
    d = tmp_path / "runs"
    d.mkdir(exist_ok=True)
    stem = name or patient
    m = d / f"{stem}.manifest.json"
    tp = d / f"{stem}.jsonl"
    tp.write_text("".join(
        json.dumps({"seq": i, "kind": "tool", "tool": "search_notes", "args": {"query": q}}) + "\n"
        for i, q in enumerate(queries)), encoding="utf-8")
    m.write_text(json.dumps({
        "patient_id": patient, "spec_id": spec, "spec_hash": "abc123",
        "answer": {"status": status, "status_kind": "value", "value": value or {}},
        "gate_validated": True,
        "usage": {"total_tokens": 50_000, "llm_calls": 12},
        "spend": {"usd": 0.42, "priced": True},
        "coverage_state": {"n_read": 7, "searched_terms": list(searched)},
        "trace": str(tp),
    }), encoding="utf-8")
    return m


# ---------------------------------------------------------------- 1. batched search terms

def test_a_batched_search_contributes_each_term_separately(tmp_path):
    r = E.RunRecord.from_manifest(
        _runtime_manifest(tmp_path, searched=(), queries=[["bx", "adenocarcinoma"]]))
    assert sorted(r.searched_terms) == ["adenocarcinoma", "bx"]


def test_a_degenerate_term_inside_a_batch_is_detected(tmp_path):
    """The measured consequence: `bx` appears 3x in this tree and was invisible."""
    r = E.RunRecord.from_manifest(
        _runtime_manifest(tmp_path, searched=(), queries=[["bx", "adenocarcinoma"]]))
    found = E.detect_degenerate_search(r, min_term_chars=4)
    assert [f.detector for f in found] == ["degenerate_search"]
    assert found[0].evidence["term"] == "bx"


def test_a_scalar_query_still_works(tmp_path):
    """The unbatched form must not regress — most recorded searches are scalars."""
    r = E.RunRecord.from_manifest(
        _runtime_manifest(tmp_path, searched=(), queries=["adenocarcinoma"]))
    assert r.searched_terms == ["adenocarcinoma"]


def test_a_stringified_list_never_reaches_the_detector(tmp_path):
    """The bug's signature: a term that is `"['bx', 'adeno']"` is not a term anyone searched."""
    r = E.RunRecord.from_manifest(
        _runtime_manifest(tmp_path, searched=(), queries=[["bx", "adeno"]]))
    assert not any(t.startswith("[") for t in r.searched_terms)


# ---------------------------------------------------------------- 2. the manifest fallback

def test_the_trace_and_the_manifest_are_not_added_together(tmp_path):
    """Two sources for one fact, summed. The manifest half read `coverage_attested` — a key the
    runtime never writes — so only the trace contributed and the concatenation looked harmless.
    Repointing it at `coverage_state` made both non-empty and every term counted twice: a real run
    yielded 12 terms where 6 were searched, and `detect_degenerate_search` reported 10 findings over
    this tree where 5 occurrences exist."""
    r = E.RunRecord.from_manifest(_runtime_manifest(
        tmp_path, searched=("carcinoma", "bx"), queries=["carcinoma", "bx"]))
    assert sorted(r.searched_terms) == ["bx", "carcinoma"]
    assert len(E.detect_degenerate_search(r, min_term_chars=4)) == 1


def test_a_repeated_search_is_counted_once_per_call(tmp_path):
    """Trace-first is per-CALL on purpose: searching the same term three times is three events, and
    a rejection-loop reading needs that. The manifest summary would flatten it to one."""
    r = E.RunRecord.from_manifest(_runtime_manifest(
        tmp_path, searched=("bx",), queries=["bx", "bx", "bx"]))
    assert r.searched_terms == ["bx", "bx", "bx"]


def test_searched_terms_reads_coverage_state_when_there_is_no_trace(tmp_path):
    """502 of 509 manifests carry `coverage_state.searched_terms`; the fallback read a key the
    runtime never writes, so a trace-less manifest reported zero terms searched."""
    m = _runtime_manifest(tmp_path, searched=("carcinoma", "bx"), queries=())
    (tmp_path / "runs" / "SYN0001.jsonl").unlink()
    r = E.RunRecord.from_manifest(m)
    assert not r.trace
    assert sorted(r.searched_terms) == ["bx", "carcinoma"]


def test_the_legacy_coverage_attested_key_is_still_read(tmp_path):
    """13 manifests on disk carry it. Dropping support would silently zero their terms."""
    m = tmp_path / "legacy.manifest.json"
    m.write_text(json.dumps({
        "patient_id": "P1", "spec_id": SPEC, "answer": {"status": "FOUND"},
        "coverage_attested": {"n_read": 3, "searched_terms": ["carcinoma"]}}), encoding="utf-8")
    assert E.RunRecord.from_manifest(m).searched_terms == ["carcinoma"]


def test_n_documents_read_also_falls_back_to_coverage_state(tmp_path):
    m = _runtime_manifest(tmp_path, queries=())
    (tmp_path / "runs" / "SYN0001.jsonl").unlink()
    assert E.RunRecord.from_manifest(m).n_documents_read == 7


# ---------------------------------------------------------------- 3. cross-spec scoring

KEY = {"SYN0001": {"fields": {"primary_site": "C34.9"}, "spec_id": SPEC},
       "SYN0002": {"fields": {"primary_site": "C50.9"}, "spec_id": SPEC}}
BKEY = E.BaselineKey(commit="c", spec_hash="h", model="m", date="2026-08-04")


def test_a_run_of_a_different_spec_is_unkeyed_not_scored(tmp_path):
    """`n_unkeyed` was pinned at 0: the composite lookup never hit because the key has bare ids,
    and the bare fallback matched regardless of spec. A run of another contract scored as a wrong
    answer against a key that says nothing about it."""
    right = E.RunRecord.from_manifest(_runtime_manifest(
        tmp_path, "SYN0001", value={"primary_site": "C34.9"}, name="right"))
    wrong_spec = E.RunRecord.from_manifest(_runtime_manifest(
        tmp_path, "SYN0002", spec=OTHER_SPEC, value={"primary_site": "C50.9"}, name="other"))

    rep = E.score([right, wrong_spec], KEY, fields=["primary_site"], key=BKEY)

    assert rep.totals["n_unkeyed"] == 1
    scored = [i for i in rep.per_instance if i.spec_id == OTHER_SPEC]
    assert all(o.outcome == E.NO_KEY for i in scored for o in i.outcomes), \
        "a cross-spec run must not receive a correctness verdict"


def test_a_key_without_spec_id_still_matches_on_the_bare_id(tmp_path):
    """Backwards compatibility: keys written before the producer emitted `spec_id` must still
    score, or every recorded baseline becomes unreadable."""
    legacy = {"SYN0001": {"fields": {"primary_site": "C34.9"}}}
    r = E.RunRecord.from_manifest(_runtime_manifest(
        tmp_path, "SYN0001", value={"primary_site": "C34.9"}))
    rep = E.score([r], legacy, fields=["primary_site"], key=BKEY)
    assert rep.totals["n_unkeyed"] == 0
    assert rep.per_instance[0].outcomes[0].outcome == E.EXACT


def test_the_composite_instance_id_form_matches(tmp_path):
    """The form `score` prefers, and which nothing produced. It must work when someone writes it."""
    composite = {f"SYN0001__{SPEC}": {"fields": {"primary_site": "C34.9"}}}
    r = E.RunRecord.from_manifest(_runtime_manifest(
        tmp_path, "SYN0001", value={"primary_site": "C34.9"}))
    rep = E.score([r], composite, fields=["primary_site"], key=BKEY)
    assert rep.totals["n_unkeyed"] == 0


ABLATION = f"{SPEC}.UNSTRATIFIED"


def test_an_ablation_arm_of_the_same_variable_still_scores(tmp_path):
    """The regression the `spec_id` check introduced, measured on files already on disk.

    `assets/specs/ablation/STORE.400_522_523.unstratified.yaml` loads with
    `spec_id == "STORE.400_522_523.site_histology_behavior.UNSTRATIFIED"`, and the corpus has no
    separate ground-truth key for it — the whole point of an ablation is that the CORRECT ANSWERS
    are identical and only the retrieval policy differs. The strict `spec_id` comparison made all 3
    real UNSTRATIFIED manifests unkeyed: `exact_match_den` over the STORE.400 runs fell 3 -> 2 and
    every rate for the arm alone became `None`, which `compare()` reads as "nothing changed" rather
    than refusing.
    """
    key = {"SYN0001": {"fields": {"primary_site": "C34.9"},
                       "spec_id": SPEC, "spec_ids": [SPEC, ABLATION]}}
    r = E.RunRecord.from_manifest(_runtime_manifest(
        tmp_path, "SYN0001", spec=ABLATION, value={"primary_site": "C34.9"}))
    rep = E.score([r], key, fields=["primary_site"], key=BKEY)
    assert rep.totals["n_unkeyed"] == 0
    assert rep.per_instance[0].outcomes[0].outcome == E.EXACT


def test_a_genuinely_different_spec_is_still_unkeyed(tmp_path):
    """The check must keep doing its job: declaring extra ids must not reopen the hole."""
    key = {"SYN0001": {"fields": {"primary_site": "C34.9"},
                       "spec_id": SPEC, "spec_ids": [SPEC, ABLATION]}}
    r = E.RunRecord.from_manifest(_runtime_manifest(
        tmp_path, "SYN0001", spec="STORE.390.date_of_initial_diagnosis",
        value={"primary_site": "C34.9"}))
    rep = E.score([r], key, fields=["primary_site"], key=BKEY)
    assert rep.totals["n_unkeyed"] == 1


def test_a_cohort_that_is_entirely_unkeyed_says_so_loudly(tmp_path):
    """`None` rates read as "no change" to a reader and to `compare`. A key that covers none of the
    runs is a wrong pairing, not a clean result."""
    key = {"SYN9999": {"fields": {"primary_site": "C34.9"}, "spec_id": SPEC}}
    r = E.RunRecord.from_manifest(_runtime_manifest(
        tmp_path, "SYN0001", value={"primary_site": "C34.9"}))
    rep = E.score([r], key, fields=["primary_site"], key=BKEY)
    assert rep.totals["n_unkeyed"] == 1
    assert rep.totals.get("all_instances_unkeyed") is True
