"""The `guideline-to-rules` skill must stay true to the engine it teaches.

A skill is prose, and prose about executable rules rots silently. The reader of this one has
a guideline PDF open and is about to write YAML that `acr.concordance` will run against real
patients; every claim the skill makes is a claim about `validate_guideline` and `assess_one`
that can be checked here rather than discovered when a rule loads wrong.

Three kinds of drift are checked, and each is a mistake this repo has already made once in
another form:

  vocabulary  The skill names outcomes, statuses and ops. `registry_catalog` had to learn
              that a name which nearly matches is worse than one that does not match at all
              (`Speech-Language-Pathology-Note` swept into a pathology stratum by a
              substring). An op named in the skill that `_OPS` does not have sends the
              analyst to write a guideline that will not load, and the error will arrive
              hours later pointing at their YAML rather than at our documentation.

  the worked example is the shipped rule  `references/worked-example.md` ends on the real
              NSCLC-ADJ-SYSTEMIC-II-IIIA block. If it ends on a *stale copy* of it, the
              analyst learns a shape nobody runs. So the block is compared to
              `guidelines/nccn_nsclc_subset.yaml` structurally, and the five-patient cohort
              beside it is actually scored.

  the failure catalogue is checked against the validator  Each example in
              `references/failure-catalogue.md` declares `# expect: REJECTED` or
              `# expect: LOADS_CLEAN` and is run through `validate_guideline`. The
              LOADS_CLEAN ones carry the weight: they are the failures the engine *cannot*
              see, which is the reason the catalogue exists at all. If validation is ever
              strengthened to catch one, this test goes red and the catalogue gets rewritten
              instead of quietly lying about where the last line of defence is.

The cohort in the worked example is invented — P01..P05, invented dates, no corpus was read
to write it.
"""
from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any, get_args

import pytest
import yaml

from acr import concordance as C
from acr.spec import Status

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "skills" / "guideline-to-rules"
SKILL = SKILL_DIR / "SKILL.md"
WORKED = SKILL_DIR / "references" / "worked-example.md"
CATALOGUE = SKILL_DIR / "references" / "failure-catalogue.md"

GUIDELINE_PATH = ROOT / "guidelines" / "nccn_nsclc_subset.yaml"
ADJ = "NSCLC-ADJ-SYSTEMIC-II-IIIA"

# --------------------------------------------------------------------- known, current gaps
#
# The build agent authoring guideline-to-rules was killed mid-work by the org spend limit on
# 2026-07-26. Only references/worked-example.md landed; SKILL.md and
# references/failure-catalogue.md were never written (see tests/test_skills_load.py's
# per-skill loader check for SKILL.md's absence specifically). The four tests that exercise
# WORKED against the real shipped guideline still pass and are left alone -- only the tests
# that need SKILL.md and/or CATALOGUE are skipped, by name, below.
_SKILL_MISSING = not SKILL.is_file()
_CATALOGUE_MISSING = not CATALOGUE.is_file()

skip_no_skill = pytest.mark.skipif(
    _SKILL_MISSING,
    reason=(
        "skills/guideline-to-rules/SKILL.md was never written (build agent killed by org "
        "spend limit; only references/worked-example.md landed)"
    ),
)
skip_no_catalogue = pytest.mark.skipif(
    _CATALOGUE_MISSING,
    reason="skills/guideline-to-rules/references/failure-catalogue.md was never written",
)

#: A fenced yaml block that opts in to being checked. Blocks without the marker are
#: illustrative fragments and are left alone.
TAGGED_BLOCK = re.compile(r"```yaml\n(#\s*expect:\s*(\w+)\n(?:#[^\n]*\n)*)(.*?)```", re.DOTALL)
BECAUSE = re.compile(r"#\s*because:\s*(.+)")

#: Vocabulary the skill is allowed to use in backticked SHOUTING_CASE. Everything that can
#: be derived from the engine is derived, so a rename over in `concordance.py` surfaces here
#: as a failing doc rather than as an analyst reading a word the code no longer uses.
_ENGINE_WORDS = set(get_args(C.Outcome)) | set(get_args(C.Resolution)) \
    | set(get_args(C.Ternary)) | set(get_args(Status)) | {"SCORABLE"}
_RUNTIME_WORDS = {
    "NOT_EXTRACTED",        # concordance.VariableValue default; no Literal to read it from
    "WRONG_DATA_SOURCE",    # STORE.610's answer when asked of the notes
    "GATE_VALIDATED", "AGENT_GAVE_UP", "BUDGET_EXHAUSTED",   # negative_basis values
    "PLACEHOLDER_REQUIRES_CLINICAL_INPUT",                   # the recurrence spec's refusal
    "NOT_BOUND",            # source_authority.version_binding in the shipped guideline
    "CANNOT_DISTINGUISH",   # explain.py verdict, referenced when handing off to L5
}
_DOC_MARKERS = {"REJECTED", "LOADS_CLEAN", "SHIPPED", "SCENARIOS"}
_METHOD_WORDS = {"ELIGIBILITY", "ACTION", "TIMING", "EXCEPTION"}   # the four predicate kinds
ALLOWED_WORDS = _ENGINE_WORDS | _RUNTIME_WORDS | _DOC_MARKERS | _METHOD_WORDS

SHOUTED = re.compile(r"`([A-Z][A-Z_]{3,})`")


def _docs() -> list[Path]:
    return [SKILL, WORKED, CATALOGUE]


def _blocks(path: Path) -> list[tuple[str, str, Any]]:
    """(tag, because, parsed) for every tagged yaml block in a doc."""
    out = []
    for header, tag, body in TAGGED_BLOCK.findall(path.read_text(encoding="utf-8")):
        m = BECAUSE.search(header)
        out.append((tag, (m.group(1).strip() if m else ""), yaml.safe_load(body)))
    return out


def _wrap(recs: Any) -> C.Guideline:
    """One or more example recommendations, wrapped in just enough guideline to validate.

    The real `value_sets` are lent to the example so that `in_set` resolves; the real
    `unknown_value_codes` are NOT, because they are checked against what the whole file
    reads and a one-recommendation excerpt would fail that for reasons the example is not
    about.
    """
    real = yaml.safe_load(GUIDELINE_PATH.read_text(encoding="utf-8"))
    return C.parse_guideline({
        "guideline_id": "EXAMPLE-FROM-THE-FAILURE-CATALOGUE",
        "guideline_version": "0.0.0",
        "value_sets": real["value_sets"],
        "recommendations": recs if isinstance(recs, list) else [recs],
    })


def _flatten(x: Any) -> Any:
    """Structure-equal, whitespace-insensitive. Reflowing a rationale is not a drift."""
    if isinstance(x, dict):
        return {k: _flatten(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_flatten(v) for v in x]
    if isinstance(x, str):
        return re.sub(r"\s+", " ", x).strip()
    return x


@pytest.fixture(scope="module")
def guideline() -> C.Guideline:
    return C.load_guideline(GUIDELINE_PATH)


# ------------------------------------------------------------------------ the files exist
@skip_no_skill
@skip_no_catalogue
def test_the_skill_ships_with_both_references():
    assert SKILL.is_file(), f"{SKILL} is missing"
    for ref in (WORKED, CATALOGUE):
        assert ref.is_file(), f"{ref} is missing"
        # test_skills_load.py checks that pointers resolve; this checks they are made at all,
        # because a reference nothing points at is never read by a model.
        assert f"skills/guideline-to-rules/references/{ref.name}" in SKILL.read_text(), \
            f"SKILL.md never points at {ref.name}, so progressive disclosure dead-ends"


# ------------------------------------------------------------------------ the vocabulary
@skip_no_skill
def test_every_op_named_in_the_skill_exists_and_every_op_is_named():
    """Both directions. An op the engine lacks costs the analyst a failed load; an op the
    skill omits costs them a workaround for something already built."""
    named = set(re.findall(r"`(all_of|any_of|not|equals|not_equals|in_set|not_in_set|matches|"
                           r"is_true|is_false|is_present|is_absent|at_least|at_most|"
                           r"days_between|on_or_before)`", SKILL.read_text(encoding="utf-8")))
    engine = set(C._OPS)
    assert named - engine == set(), f"skill names ops the engine does not have: {named - engine}"
    assert engine - named == set(), f"engine ops the skill never mentions: {engine - named}"


@pytest.mark.parametrize(
    "doc",
    [pytest.param(SKILL, marks=skip_no_skill), WORKED, pytest.param(CATALOGUE, marks=skip_no_catalogue)],
    ids=lambda p: p.name,
)
def test_no_invented_status_or_outcome_names(doc: Path):
    used = set(SHOUTED.findall(doc.read_text(encoding="utf-8")))
    unknown = used - ALLOWED_WORDS
    assert unknown == set(), (
        f"{doc.name} uses vocabulary the system does not define: {sorted(unknown)}. "
        f"An analyst cannot act on a status that does not exist.")


# ------------------------------------------------------------- the worked example is real
def test_the_worked_example_ends_on_the_recommendation_that_actually_ships(guideline):
    shipped = [r for r in guideline.raw["recommendations"] if r["id"] == ADJ]
    assert len(shipped) == 1
    blocks = [b for tag, _, b in _blocks(WORKED) if tag == "SHIPPED"]
    assert len(blocks) == 1, "worked-example.md must carry exactly one `# expect: SHIPPED` block"
    assert _flatten(blocks[0]) == _flatten(shipped[0]), (
        "the worked example has drifted from guidelines/nccn_nsclc_subset.yaml; the analyst "
        "would be copying a shape nobody runs")


def test_the_worked_example_cohort_scores_the_way_the_document_says(guideline):
    """The five invented patients in the worked example, actually put through the engine."""
    scen = [b for tag, _, b in _blocks(WORKED) if tag == "SCENARIOS"]
    assert len(scen) == 1
    base, cases = scen[0]["base"], scen[0]["cases"]
    assert len(cases) >= 5, "the cohort must cover all five outcomes plus the window's floor"
    rec = guideline.recommendation(ADJ)

    seen = set()
    for case in cases:
        variables = dict(base, **(case.get("overrides") or {}))
        r = C.assess_one(rec, variables, guideline)
        assert r.outcome == case["outcome"], (
            f"{case['id']}: document says {case['outcome']}, engine says {r.outcome} "
            f"({r.reason})")
        if case.get("exception_id"):
            assert r.exception_id == case["exception_id"], f"{case['id']}: wrong exception"
        for name in case.get("blocking_inputs") or []:
            assert name in r.blocking_inputs, (
                f"{case['id']}: {name!r} is not in blocking_inputs {r.blocking_inputs}")
        seen.add(r.outcome)
    assert seen == set(get_args(C.Outcome)), f"cohort never reaches {set(get_args(C.Outcome)) - seen}"


def test_removing_the_lower_bound_turns_neoadjuvant_therapy_concordant(guideline):
    """Section 4 of the skill, proved rather than asserted.

    The same patient — systemic therapy dated before the operation — is NON_CONCORDANT under
    the shipped rule and CONCORDANT under one whose window has only a ceiling. Nothing in
    `validate_guideline` objects to the second rule; `days_between` needs only one bound.
    """
    scen = [b for tag, _, b in _blocks(WORKED) if tag == "SCENARIOS"][0]
    neo = next(c for c in scen["cases"] if c.get("id") == "P01b")
    variables = dict(scen["base"], **(neo.get("overrides") or {}))

    assert C.assess_one(guideline.recommendation(ADJ), variables, guideline
                        ).outcome == "NON_CONCORDANT"

    raw = copy.deepcopy(guideline.raw)
    rec = next(r for r in raw["recommendations"] if r["id"] == ADJ)
    for cond in rec["satisfied_when"]:
        cond.pop("min_days", None)
    unbounded = C.parse_guideline(raw)
    assert C.validate_guideline(unbounded) == [], \
        "the point of this test is that the unbounded rule loads clean"
    assert C.assess_one(unbounded.recommendation(ADJ), variables, unbounded
                        ).outcome == "CONCORDANT"


def test_editing_the_operationalisation_number_moves_the_hash_but_not_the_score(guideline):
    """Both halves of what the skill claims about `operationalisation:`.

    Changing a signed-off number changes `guideline_hash`, so labels made before and after
    are not comparable — that is the intended behaviour. But the block and the condition are
    two separate literals, so editing only the block moves the hash and changes nothing that
    runs. The catalogue documents that as the drift it is.
    """
    scen = [b for tag, _, b in _blocks(WORKED) if tag == "SCENARIOS"][0]
    base = scen["base"]

    raw = copy.deepcopy(guideline.raw)
    rec = next(r for r in raw["recommendations"] if r["id"] == ADJ)
    rec["source"]["operationalisation"]["max_days_surgery_to_adjuvant"] = 90
    edited = C.parse_guideline(raw)

    assert edited.guideline_hash != guideline.guideline_hash
    before = C.assess_one(guideline.recommendation(ADJ), base, guideline)
    after = C.assess_one(edited.recommendation(ADJ), base, edited)
    assert (before.outcome, after.outcome) == ("CONCORDANT", "CONCORDANT")
    assert before.guideline_hash != after.guideline_hash


# ------------------------------------------------------------------ the failure catalogue
@skip_no_catalogue
def test_the_catalogue_has_examples_of_both_kinds():
    tags = [t for t, _, _ in _blocks(CATALOGUE)]
    assert tags.count("REJECTED") >= 4, "too few examples the validator does catch"
    assert tags.count("LOADS_CLEAN") >= 4, "too few examples the validator cannot catch"


@skip_no_catalogue
def test_every_rejected_example_is_actually_rejected():
    for tag, because, rec in _blocks(CATALOGUE):
        if tag != "REJECTED":
            continue
        name = rec["id"] if isinstance(rec, dict) else "<list>"
        errors = C.validate_guideline(_wrap(rec))
        assert errors, f"{name}: catalogue says REJECTED, validate_guideline accepted it"
        assert because, f"{name}: a REJECTED example must quote the error with `# because:`"
        joined = "; ".join(errors)
        assert because in joined, (
            f"{name}: catalogue quotes {because!r}, validator said {joined!r}")


@skip_no_catalogue
def test_every_loads_clean_example_really_does_load_clean():
    """These are the ones that matter. Each is a rule that is wrong about patients and right
    about syntax, so the only thing standing between it and a published rate is the analyst.
    """
    n = 0
    for tag, _, rec in _blocks(CATALOGUE):
        if tag != "LOADS_CLEAN":
            continue
        name = rec["id"] if isinstance(rec, dict) else "<list>"
        errors = C.validate_guideline(_wrap(rec))
        assert errors == [], (
            f"{name}: the catalogue presents this as a defect the validator CANNOT see, but "
            f"it now reports {errors}. Move it to REJECTED and rewrite the section.")
        n += 1
    assert n >= 4


@skip_no_catalogue
def test_bundling_two_recommendations_changes_the_reported_rate():
    """Section 1, in numbers. One patient, one set of facts, two ways of writing the rule."""
    by_id = {}
    for _, _, rec in _blocks(CATALOGUE):
        for r in (rec if isinstance(rec, list) else [rec]):
            if isinstance(r, dict) and r.get("id"):
                by_id[r["id"]] = r
    bundled = by_id["EXAMPLE-BUNDLED"]
    split = [by_id["EXAMPLE-SPLIT-ADJUVANT"], by_id["EXAMPLE-SPLIT-SURVEILLANCE"]]

    scen = [b for tag, _, b in _blocks(WORKED) if tag == "SCENARIOS"][0]
    patient = dict(scen["base"])
    # Adjuvant therapy delivered on time; surveillance imaging proved absent by the gate.
    patient["surveillance_imaging_within_12_months"] = {
        "status": "FOUND", "value": None, "negative_basis": "GATE_VALIDATED"}

    g1, g2 = _wrap(bundled), _wrap(split)
    one = C.summarise(C.assess(patient, g1))
    two = C.summarise(C.assess(patient, g2))

    assert [r.outcome for r in C.assess(patient, g1)] == ["NON_CONCORDANT"]
    assert sorted(r.outcome for r in C.assess(patient, g2)) == ["CONCORDANT", "NON_CONCORDANT"]
    assert (one["denominator"], one["concordance_rate"]) == (1, 0.0)
    assert (two["denominator"], two["concordance_rate"]) == (2, 0.5)
