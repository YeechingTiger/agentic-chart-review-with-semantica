"""The retrieval-asset development layer, tested offline against labellings we author here.

EVERY FIXTURE IS SYNTHETIC AND NO TEST TOUCHES THE NETWORK OR A CHART. That is not a
convenience: a suite that needs the real corpus runs on one host, costs money, and stops being
run — and this is the module whose whole job is to stop people guessing. The cohort below is
thirty lines of Python and carries the cases that matter:

  * a patient whose histology exists only in a progress note, in wording the shipped keyword
    list does not contain
  * a patient with ZERO establishing notes, who must never count as having lost an answer
  * "patient", a term with perfect recall and terrible precision, which every recall-only
    measurement in this project so far would have adopted
  * a document type in `cannot_establish` that demonstrably states the field — where the data
    has something to say and still may not say it
  * a dev/test overlap, which must raise
  * a semantic asset, which must refuse to auto-adopt
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import MISSING, fields, replace
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from acr import assetdev as A
from acr.spec import load_spec

runner = CliRunner()
FIELD = "histology"
PATH, CYTO, PROG, IMG, ADD = ("Surgical-Pathology-Report", "Cytology-Report", "Progress-Note",
                              "Head-CT-WWO-Contr", "Onc-Addendum")
#: Everything the imaginary scan indexed. Passed explicitly so an unindexed needle raises.
VOCAB = frozenset({"carcinoma", "adenocarcinoma", "small cell", "sclc", "specimen", "biopsy",
                   "contrast", "cough", "patient"})


def cohort(n: int = 60) -> list[A.NoteLabel]:
    """Deterministic in `n` and nothing else, so a failure here is a failure and not a seed."""
    out: list[A.NoteLabel] = []

    def add(pid, i, doc_type, terms, establishes=(), mentions=()):
        out.append(A.NoteLabel(pid, f"{doc_type}_2021-01-{(i % 28) + 1:02d}", doc_type,
                               frozenset(establishes), frozenset(mentions), frozenset(terms)))

    for i in range(n):
        pid, blank, path_absent = f"SYN{i:04d}", i % 13 == 6, i % 4 == 0 and i % 13 != 6
        if not blank and not path_absent:
            add(pid, i, PATH, {"carcinoma", "specimen", "biopsy", "patient"}, [FIELD])
        if path_absent:  # the histology exists only here, in wording nobody searched for
            add(pid, i, PROG, {"small cell", "sclc", "patient"}, [FIELD])
        for j in (1, 2, 3):  # read exhaustively by the shipped spec, establishes nothing, ever
            add(pid, i + j, CYTO, {"specimen", "patient"})
        if not blank:
            add(pid, i + 4, PROG, {"carcinoma", "patient"}, mentions=[FIELD])
        for j in range(5, 11):  # the pile "patient" would drag in, and it says nothing
            add(pid, i + j, PROG, {"cough", "patient"})
        add(pid, i + 11, IMG, {"contrast", "patient"})
        if i % 5 == 0 and not blank:  # swept into `cannot_establish`, and states the histology
            add(pid, i + 12, ADD, {"carcinoma", "specimen", "patient"}, [FIELD])
    return out


def labelling(n: int = 60) -> A.Labelling:
    return A.Labelling("haiku-cheap-scan", "p" * 8, "s" * 8, tuple(cohort(n)), VOCAB)


def spec_doc(keywords=("carcinoma",)) -> dict:
    def prov(el):
        return {"element": el, "origin": "model_authored", "status": "draft",
                "basis": "no external source; authored in a test fixture"}

    strata = [
        {"name": "can_establish", "policy": "exhaustive", "establishes": [FIELD],
         "match": {"doc_type_matches": ["Pathology", "Cytology"]}},
        {"name": "may_mention", "policy": "search_then_read_hits_and_sample_misses",
         "match": {"doc_type_matches": ["Progress-Note", "Consult"]},
         "required_keywords": list(keywords), "min_sample_of_misses": 25},
        {"name": "cannot_establish", "policy": "validate_by_sampling",
         "match": {"rest": True}, "min_sample": 25},
    ]
    return {
        "spec_id": "SYNTH.999.histology_for_tests", "question": "What is the histology?",
        "fields": [{"name": FIELD, "type": "string"}],
        "proof_obligation": {"for_positive": "One pathology report.",
                             "for_negative": {"mode": "stratified_exclusion", "strata": strata}},
        "provenance": [prov(f"proof_obligation.for_negative.strata[{s['name']}].{leaf}")
                       for s in strata for leaf in
                       ({"can_establish": ("match", "establishes"),
                         "may_mention": ("match", "required_keywords", "min_sample_of_misses"),
                         "cannot_establish": ("match", "min_sample")}[s["name"]])],
    }


@pytest.fixture
def spec_path(tmp_path) -> Path:
    p = tmp_path / "SYNTH.999.histology.yaml"
    p.write_text(yaml.safe_dump(spec_doc(), sort_keys=False), encoding="utf-8")
    load_spec(p)  # the fixture is only useful if it is a spec
    return p


@pytest.fixture
def plan(spec_path) -> A.RetrievalPlan:
    return A.RetrievalPlan.from_spec(load_spec(spec_path), FIELD)


@pytest.fixture
def lab() -> A.Labelling:
    return labelling()


@pytest.fixture
def split(tmp_path, lab) -> A.Split:
    return A.make_split(lab.patient_ids(), seed=0, path=tmp_path / "split.json",
                        today="2026-07-27")


def digest_tree(root: Path) -> dict:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# ------------------------------------------------------------------------------------ plan
def test_the_plan_read_off_the_spec_is_the_one_the_runtime_would_run(spec_path, plan):
    """A development plane that models retrieval differently from `coverage.assign_strata`
    measures a system nobody is going to run."""
    from datetime import date

    from acr.corpus import DocMeta
    from acr.coverage import assign_strata, strata_from_spec

    docs = [DocMeta(t, t, date(2021, 1, 1), 1, 10) for t in (PATH, CYTO, PROG, IMG, ADD)]
    runtime = {d.note_id: name for name, group
               in assign_strata(docs, strata_from_spec(load_spec(spec_path))).items()
               for d in group}
    assert {d.doc_type: plan.stratum_of(d.doc_type) for d in docs} == runtime
    assert (plan.reads_for("can_establish"), plan.reads_for("may_mention"),
            plan.reads_for("cannot_establish")) == (A.READ_ALL, A.READ_HITS, A.READ_NONE)


# ----------------------------------------------------------------------------------- split
def test_a_split_is_one_stored_file_carrying_its_own_seed_and_no_patient_ids(tmp_path, lab):
    s = A.make_split(lab.patient_ids(), seed=3, path=tmp_path / "s.json", today="2026-07-27")
    body = (tmp_path / "s.json").read_text()
    assert json.loads(body)["seed"] == 3 and json.loads(body)["created_on"] == "2026-07-27"
    assert not any(pid in body for pid in lab.patient_ids()), "the split leaked identifiers"
    assert A.Split.load(tmp_path / "s.json").split_hash == s.split_hash
    assert not s.overlap and len(s.dev) + len(s.test) == len(lab.patient_ids())


def test_rewriting_a_split_path_with_a_different_seed_is_refused(tmp_path, lab):
    """Seed shopping: whichever of the two you quote, the reader cannot see the other existed."""
    A.make_split(lab.patient_ids(), seed=1, path=tmp_path / "s.json")
    with pytest.raises(A.AssetDevelopmentError, match="already holds split"):
        A.make_split(lab.patient_ids(), seed=2, path=tmp_path / "s.json")


def test_the_seed_is_not_a_parameter_of_any_measurement():
    """It is accepted by make_split and by nothing else downstream."""
    import inspect

    for fn in (A.measure, A.propose, A.evolve, A.certify, A.adopt):
        assert "seed" not in inspect.signature(fn).parameters, fn.__name__


# --------------------------------------------------------------------------------- metrics
def test_all_four_numbers_are_reported_and_they_do_not_agree(plan, lab, split):
    m = A.measure(plan, lab, split, A.DEV)
    assert 0.0 < m.recall < 1.0 and 0.0 < m.precision < 1.0
    assert m.notes_read_per_patient > 0 and m.n_read < m.n_notes
    # the failure that is not money: patients whose only establishing note went unread
    assert 0 < m.patients_losing_the_answer < m.patients_with_an_answer


def test_a_patient_whose_chart_never_had_the_answer_cannot_lose_it(plan, tmp_path):
    """You cannot lose what was never there, so a blank chart is in no denominator."""
    notes = (A.NoteLabel("A", "n1", PROG, frozenset({FIELD}), terms=frozenset({"sclc"})),
             A.NoteLabel("B", "n1", PROG, terms=frozenset({"cough"})))
    lab = A.Labelling("m", "p", "s", notes, frozenset({"sclc", "cough", "carcinoma"}))
    sp = A.make_split(["A", "B"], seed=0, test_frac=0.5, path=tmp_path / "s.json")
    both = A.measure(plan, lab, sp, A.DEV, notes=list(notes))
    assert both.n_patients == 2 and both.patients_with_an_answer == 1
    assert both.patients_losing_the_answer == 1  # A only, never B


def test_an_unindexed_needle_raises_rather_than_scoring_a_silent_zero(plan, lab, split):
    with pytest.raises(A.UnindexedTermError, match="prefixes nothing"):
        A.measure(plan.with_keywords("may_mention", ["myxopapillary"]), lab, split, A.DEV)


# ---------------------------------------------------------------------- propose and evolve
def test_propose_offers_add_drop_and_stem_and_never_looks_at_test(plan, lab, split):
    cands = A.propose(plan, lab, split, kind=A.KEYWORDS)
    assert {"add", "drop", "stem"} <= {c.rationale.split()[0] for c in cands}
    assert all(c.element.endswith("strata[may_mention].required_keywords") for c in cands)
    # a word that exists only in the held-out half must never reach a candidate
    marked = tuple(n if A._digest(n.patient_id) in split.members(A.DEV)
                   else replace(n, terms=n.terms | {"zzzheldout"}) for n in lab.notes)
    held = A.Labelling(lab.model, lab.prompt_hash, lab.spec_hash, marked, VOCAB | {"zzzheldout"})
    assert not any("zzzheldout" in c.value
                   for c in A.propose(plan, held, split, kind=A.KEYWORDS))


def test_a_term_with_perfect_recall_and_terrible_precision_is_not_adopted(plan, lab, split):
    """"patient" is in every note ever written. Recall alone would take it; the bill says no."""
    greedy = A.measure(plan.with_keywords("may_mention", ["patient"]), lab, split, A.DEV)
    ev = A.evolve(plan, lab, split, kind=A.KEYWORDS)
    won = A.measure(ev.plan, lab, split, A.DEV)
    assert greedy.recall >= won.recall, "the greedy term really does have the better recall"
    assert greedy.precision < won.precision and greedy.n_read > won.n_read
    assert "patient" not in ev.candidate.value
    assert {"sclc", "small cell"} & set(ev.candidate.value), ev.candidate.value


def test_evolve_climbs_on_dev_only_and_writes_nothing(tmp_path, spec_path, plan, lab, split):
    before = digest_tree(tmp_path)
    ev = A.evolve(plan, lab, split, kind=A.KEYWORDS)
    assert digest_tree(tmp_path) == before, "evolve() wrote something"
    assert ev.baseline.patients_losing_the_answer >= ev.final.patients_losing_the_answer
    assert any(line.startswith("ACCEPT") for line in ev.log)
    assert ev.split_hash == split.split_hash


# --------------------------------------------------------------------------------- certify
def test_certify_refuses_a_dev_test_overlap(tmp_path, plan, lab, split):
    ev = A.evolve(plan, lab, split, kind=A.KEYWORDS)
    leaky = replace(split, dev=split.dev + (split.test[0],)).save(tmp_path / "leaky.json")
    with pytest.raises(A.SplitLeakError, match="also"):
        A.certify(ev, lab, leaky, model=lab.model)


def test_certify_refuses_to_score_on_the_half_the_search_optimised(plan, lab, split):
    ev = A.evolve(plan, lab, split, kind=A.KEYWORDS)
    with pytest.raises(A.ScoredOnDevError, match="training score"):
        A.certify(ev, lab, split, model=lab.model, on=A.DEV)


def test_certify_refuses_a_split_that_was_never_stored(plan, lab, split):
    ev = A.evolve(plan, lab, split, kind=A.KEYWORDS)
    with pytest.raises(A.AssetDevelopmentError, match="written to disk"):
        A.certify(ev, lab, replace(split, path=None), model=lab.model)


def test_the_certified_number_comes_from_the_test_half(plan, lab, split):
    ev = A.evolve(plan, lab, split, kind=A.KEYWORDS)
    cert = A.certify(ev, lab, split, model=lab.model, today="2026-07-27")
    assert cert.test.n_patients == len(split.test) and cert.dev.n_patients == len(split.dev)
    assert cert.verdict == "supports" and cert.certified_on == "2026-07-27"
    assert A.Certification.from_dict(json.loads(json.dumps(cert.to_dict()))) == cert


# ------------------------------------------------------------------------ negative control
def no_signal_labelling(n: int = 30) -> A.Labelling:
    """A corpus that holds an answer and says NOTHING about who has it.

    Every progress note reads the same, and exactly one of each patient's four establishes the
    histology. A hill-climb still improves its dev objective here — by adding the one term there
    is and reading the whole pile — and that improvement survives on the held-out half, so no
    amount of test-set recall exposes it. It is not retrieval; it is reading everything. Permute
    which patient owns the answer and the identical search gains the identical amount, which is
    the only measurement that can tell the two apart.
    """
    notes = []
    for i in range(n):
        pid = f"NUL{i:04d}"
        for j in range(4):
            notes.append(A.NoteLabel(pid, f"{PROG}_{j}", PROG,
                                     frozenset({FIELD}) if j == i % 4 else frozenset(),
                                     frozenset(), frozenset({"cough"})))
        notes.append(A.NoteLabel(pid, f"{IMG}_0", IMG, terms=frozenset({"contrast"})))
    return A.Labelling("haiku-cheap-scan", "p" * 8, "s" * 8, tuple(notes), VOCAB)


def test_a_gain_that_survives_shuffling_the_labels_is_refused_loudly(tmp_path, plan):
    """The whole point. The search wins on dev, the win holds up on the held-out half, and it is
    worth nothing: permuted labels buy the same win. Certification must fail, with both numbers
    in the message, rather than report a verdict a reader would believe."""
    lab = no_signal_labelling()
    sp = A.make_split(lab.patient_ids(), seed=0, path=tmp_path / "null.json")
    ev = A.evolve(plan, lab, sp, kind=A.KEYWORDS)
    assert ev.candidate is not None, "the fixture must give the search something to find"
    held_out = A.measure(ev.plan, lab, sp, A.TEST)
    assert held_out.answer_coverage > A.measure(plan, lab, sp, A.TEST).answer_coverage, \
        "and it must hold up on the test half, or the control is not the thing catching it"

    with pytest.raises(A.NegativeControlFailed) as e:
        A.certify(ev, lab, sp, model=lab.model)
    msg = str(e.value)
    assert "NEGATIVE CONTROL FAILED" in msg
    for number in (A._negative_control(ev, lab, sp, A.TEST, A.scope(lab, sp, A.TEST)).real_gain,
                   A._negative_control(ev, lab, sp, A.TEST, A.scope(lab, sp, A.TEST)).shuffled_max):
        assert f"{number:+.4f}" in msg, (number, msg)


def test_the_control_cannot_be_skipped_by_a_flag_a_default_or_an_absent_argument(plan, lab, split):
    """A control a caller can decline is a report section. `certify` takes no argument that turns
    it off, the record has no default that stands in for one, and the CLI exposes no switch."""
    import inspect

    assert not [p for p in inspect.signature(A.certify).parameters
                if any(w in p for w in ("control", "shuffl", "skip", "force", "unsafe", "seed"))]
    held = next(f for f in fields(A.Certification) if f.name == "control")
    assert held.default is MISSING and held.default_factory is MISSING, \
        "a defaulted control is one an absent argument can skip"
    out = runner.invoke(A.assets_app, ["certify", "--help"]).output
    assert not [w for w in ("control", "shuffle", "skip", "force", "unsafe", "seed")
                if f"--{w}" in out or f"--no-{w}" in out]
    # and the object a passing run produces is the claim itself, not a note about one
    cert = A.certify(A.evolve(plan, lab, split, kind=A.KEYWORDS), lab, split, model=lab.model)
    assert cert.control.passed


def test_the_shuffle_moves_the_answers_and_disturbs_nothing_else(lab):
    """Per-patient note counts and the multiset of answers survive; who owns which answer does
    not. A shuffle that also tore up the note structure would leave every plan bad on the
    permuted corpus, every candidate would look like a discovery, and the control would pass
    whatever it was given."""
    shuffled = A._shuffle(lab, seed=7)
    where = [(n.patient_id, n.note_id, n.doc_type, n.terms) for n in lab.notes]

    assert [(n.patient_id, n.note_id, n.doc_type, n.terms) for n in shuffled.notes] == where
    assert (Counter(n.patient_id for n in shuffled.notes)
            == Counter(n.patient_id for n in lab.notes)), "per-patient note counts moved"
    assert (Counter((tuple(sorted(n.establishes)), tuple(sorted(n.mentions))) for n in shuffled.notes)
            == Counter((tuple(sorted(n.establishes)), tuple(sorted(n.mentions))) for n in lab.notes)), \
        "the marginal distribution of answers moved"
    assert (Counter(n.doc_type for n in shuffled.notes if FIELD in n.establishes)
            == Counter(n.doc_type for n in lab.notes if FIELD in n.establishes)), \
        "answers left their document type, which is more than the patient link"
    assert shuffled.hash == lab.hash and shuffled.vocabulary == lab.vocabulary

    owners = ({n.patient_id for n in x.notes if FIELD in n.establishes} for x in (lab, shuffled))
    assert next(owners) != next(owners), "the patient-answer link survived the shuffle"
    assert A._shuffle(lab, seed=7) == shuffled and A._shuffle(lab, seed=8) != shuffled


def test_the_rule_the_margin_and_the_seeds_are_written_into_the_certification(tmp_path, plan, lab,
                                                                              split):
    """A threshold nobody can find is a threshold nobody can argue with, so the rule that was
    applied, the margin, every seed and every shuffled gain go into the record."""
    cert = A.certify(A.evolve(plan, lab, split, kind=A.KEYWORDS), lab, split, model=lab.model)
    c = cert.control
    assert len(c.seeds) == len(c.shuffled_gains) == A.CONTROL_SHUFFLES == 19
    assert len(set(c.seeds)) == 19 and c.margin == A.CONTROL_MARGIN == 0.02
    assert c.rule == A.CONTROL_RULE and str(A.CONTROL_MARGIN) in c.rule
    assert c.passed and c.real_gain > c.shuffled_max + c.margin and c.real_gain > 0

    body = cert.to_dict()["negative_control"]
    assert body["passed"] and body["rule"] == A.CONTROL_RULE and body["seeds"] == list(c.seeds)
    assert A.Certification.from_dict(json.loads(json.dumps(cert.to_dict()))) == cert
    # derived from what is being certified — not passed in, and not the same for another split
    other = A.make_split(lab.patient_ids(), seed=5, path=tmp_path / "other.json")
    assert c.seeds == A._control_seeds(lab, split) != A._control_seeds(lab, other)


def test_the_control_is_scored_through_the_guards_on_the_same_held_out_half(monkeypatch, plan,
                                                                            lab, split):
    """Not a second, unguarded path into the test data: every shuffled rerun is scored through
    the same `_guards`, on the same split object, for the same half."""
    seen, real_guards = [], A._guards

    def spy(evolution, labelling, sp, on):
        seen.append((sp, on, labelling.hash, len(labelling.notes)))
        return real_guards(evolution, labelling, sp, on)

    monkeypatch.setattr(A, "_guards", spy)
    A.certify(A.evolve(plan, lab, split, kind=A.KEYWORDS), lab, split, model=lab.model)
    assert len(seen) == 1 + A.CONTROL_SHUFFLES
    assert all(sp is split and on == A.TEST for sp, on, _, _ in seen)
    assert {h for _, _, h, _ in seen} == {lab.hash}
    assert {n for *_, n in seen} == {len(lab.notes)}, "a rerun scored a different set of notes"

    # and the guards BITE on that path: a shuffled labelling with nothing left in the held-out
    # half refuses, where an unguarded control would quietly score whatever remained.
    monkeypatch.setattr(A, "_shuffle", lambda labelling, seed: replace(
        labelling, notes=tuple(n for n in labelling.notes
                               if A._digest(n.patient_id) in split.members(A.DEV))))
    with pytest.raises(A.AssetDevelopmentError, match="no notes for any of the"):
        A.certify(A.evolve(plan, lab, split, kind=A.KEYWORDS), lab, split, model=lab.model)


# ----------------------------------------------------------------------------------- adopt
def test_a_certified_keyword_list_auto_adopts_with_its_provenance(spec_path, plan, lab, split):
    """Keywords are retrieval-only: they change what text arrives, never what an answer means."""
    cert = A.certify(A.evolve(plan, lab, split, kind=A.KEYWORDS), lab, split, model=lab.model,
                     spec_id="SYNTH.999.histology_for_tests")
    out = A.adopt(cert, spec_path, run="unit-test")
    assert out.outcome == "adopted" and out.rule.startswith("RETRIEVAL_ONLY")

    reloaded = load_spec(spec_path)
    element = "proof_obligation.for_negative.strata[may_mention].required_keywords"
    rec = reloaded.provenance_index[element]
    written = A.RetrievalPlan.from_spec(reloaded, FIELD).keywords_for("may_mention")
    assert list(written) == [str(v) for v in cert.candidate.value]
    assert rec.origin == "corpus_derived" and rec.status == "measured"
    assert rec.measured["verdict"] == "supports" and rec.measured["run"] == "unit-test"
    for k in ("recall", "precision", "notes_read_per_patient", "patients_losing_the_answer"):
        assert k in rec.measured, k
    assert len([r for r in reloaded.provenance if r.element == element]) == 1


def test_an_adoption_that_cannot_land_leaves_the_spec_byte_identical(spec_path, plan, lab, split):
    cert = A.certify(A.evolve(plan, lab, split, kind=A.KEYWORDS), lab, split, model=lab.model)
    nowhere = replace(cert.candidate,
                      element="proof_obligation.for_negative.strata[invented].required_keywords")
    before = spec_path.read_bytes()
    with pytest.raises(A.AdoptionAborted, match="no stratum"):
        A.adopt(replace(cert, candidate=nowhere), spec_path)
    assert spec_path.read_bytes() == before


def test_a_stratum_change_only_ever_becomes_a_proposal(tmp_path, spec_path, plan, lab, split):
    """A stratum encodes admissibility. The labels can say where the information IS; they
    cannot say what may ESTABLISH it, so there is no path from a measurement to the spec."""
    ev = A.evolve(plan, lab, split, kind=A.STRATA)
    assert ev.candidate is not None and ev.candidate.is_semantic
    cert = A.certify(ev, lab, split, model=lab.model, spec_id="SYNTH.999.histology_for_tests")
    before = spec_path.read_bytes()
    out = A.adopt(cert, spec_path, proposals_dir=tmp_path / "proposals", today="2026-07-27")

    assert out.outcome == "proposal_emitted" and out.rule.startswith("SEMANTIC")
    assert spec_path.read_bytes() == before, "a measurement rewrote an evidence rule"
    body = yaml.safe_load(Path(out.proposal_path).read_text())
    assert body["STATUS"].startswith("PROPOSED")
    assert body["signature"] == {"reviewed_by": None, "reviewed_on": None, "decision": None,
                                 "note": "accept | reject | accept_with_changes"}
    assert "MAY ESTABLISH" in body["the_question_for_the_clinician"]


def test_the_provenance_record_carries_the_control_the_number_survived(plan, lab, split):
    """The record a reader sees has to say what the number was tested against, in numbers: a
    certification that does not quote its control is indistinguishable from one that had none."""
    cert = A.certify(A.evolve(plan, lab, split, kind=A.KEYWORDS), lab, split, model=lab.model)
    rec = cert.provenance_record()
    assert rec.measured["negative_control"] == "passed"
    assert rec.measured["control_real_gain"] > rec.measured["control_shuffled_max"]
    assert rec.measured["control_rule"] == A.CONTROL_RULE
    assert len(rec.measured["control_seeds"]) == A.CONTROL_SHUFFLES
    assert "shuffled labels" in rec.basis and "permutation test" in rec.basis


# ------------------------------------------------------------------------------------- CLI
def test_cli_split_measure_evolve_certify_adopt(tmp_path, spec_path, lab):
    lab_path = tmp_path / "labels.json"
    lab_path.write_text(json.dumps({
        "model": lab.model, "prompt_hash": lab.prompt_hash, "spec_hash": lab.spec_hash,
        "indexed_vocabulary": sorted(VOCAB),
        "notes": [{"patient_id": n.patient_id, "note_id": n.note_id, "doc_type": n.doc_type,
                   "establishes": sorted(n.establishes), "mentions": sorted(n.mentions),
                   "terms": sorted(n.terms)} for n in lab.notes]}))
    sp, cert = tmp_path / "split.json", tmp_path / "cert.json"
    base = ["--spec", str(spec_path), "--field", FIELD, "--labelling", str(lab_path),
            "--split", str(sp)]

    def run(*args):
        r = runner.invoke(A.assets_app, list(args))
        assert r.exit_code == 0, r.output + str(r.exception)
        return r.output

    run("split", "--labelling", str(lab_path), "--out", str(sp), "--seed", "0")
    assert "recall" in run("measure", *base)
    evolved = run("evolve", *base)
    assert "ACCEPT" in evolved and "wrote nothing" in evolved
    assert "verdict: supports" in run("certify", *base, "--out", str(cert))
    assert "adopted" in run("adopt", "--spec", str(spec_path), "--cert", str(cert))
    load_spec(spec_path)
