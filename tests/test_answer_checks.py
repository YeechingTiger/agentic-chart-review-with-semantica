"""The answer checks, tested directly — which until 2026-07-28 nothing did.

`answer_checks.py` decides whether a submitted answer is accepted. It had no test file of its
own; it was exercised sideways by `test_rule_attribution` (which cares about rule IDS, not
verdicts), `test_stage_spec`, `test_refine` and `test_speclint`. So the module that can turn a
right answer into a wrong one had its verdicts asserted nowhere, and that is exactly what
happened on the ten-patient real batch of 2026-07-28.

THE MEASURED FAILURE, wired here as a control. A run coded histology 8046 — the registry's
answer — citing "poorly differentiated non-small cell carcinoma". The check's
`contradicted_by` list contains "small cell". `_norm` does not strip hyphens, so
`"small cell" in "non-small cell carcinoma"` is True: the check read a negation as its
opposite, refused the correct answer, and its message then told the agent that
'"favor squamous cell carcinoma" supports 8070 over 8046' — a worked example hardcoded from a
different chart. The agent submitted 8070 and that was accepted.

Both halves are asserted below: the negation must not count, and the message must not name a
code.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from acr.answer_checks import _norm, _occurs_unnegated, check_answer_detail
from acr.spec import load_spec

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "specs" / "STORE.400_522_523.site_histology_behavior.yaml"

HISTOLOGY_NOS = {"field": "histology", "kind": "not_less_specific",
                 "nos_values": ["8000", "8010", "8046"],
                 "contradicted_by": ["squamous", "adenocarcinoma", "small cell", "large cell"]}


def ev(quote: str, field: str = "histology") -> list[dict]:
    return [{"field": field, "note_id": "Path-Report_2020-01-01", "quote": quote}]


# ------------------------------------------------------------------ the negation, in isolation
@pytest.mark.parametrize("phrase,text,unnegated", [
    # The measured case. "non-" makes the phrase the tail of its own negation.
    ("small cell", "poorly differentiated non-small cell carcinoma", False),
    ("small cell", "non small cell carcinoma of the lung", False),
    # ...and the same words, actually asserted.
    ("small cell", "small cell carcinoma of the lung", True),
    ("small cell", "negative for small cell carcinoma", False),
    ("adenocarcinoma", "no evidence of adenocarcinoma", False),
    # Bare "no" is NOT a negator, on purpose: this sentence asserts adenocarcinoma.
    ("adenocarcinoma", "there is no doubt this is adenocarcinoma", True),
    # Per occurrence, not per blob: the second mention is not negated by the first.
    ("adenocarcinoma", "no evidence of adenocarcinoma left; adenocarcinoma present right", True),
    # A sentence boundary ends a negation's reach.
    ("adenocarcinoma", "negative for small cell. adenocarcinoma is present", True),
    # "non-" applies only immediately before the phrase, not to the whole sentence.
    ("lower lobe", "non small cell carcinoma of the lower lobe", True),
])
def test_negation_is_read_per_occurrence(phrase, text, unnegated):
    assert _occurs_unnegated(phrase, _norm(text)) is unnegated


# ---------------------------------------------------- the failure that reached a real answer
def test_a_negated_mention_does_not_refuse_the_nos_code():
    """8046 IS the answer for "non-small cell carcinoma". The check must not refuse it.

    This is the regression. Before the fix this returned one violation whose message named
    8070, and the run that received it complied.
    """
    quote = ("Final Diagnosis: Positive for malignant cells. Poorly differentiated "
             "non-small cell carcinoma. See note.")
    assert check_answer_detail([HISTOLOGY_NOS], {"histology": "8046"}, ev(quote)) == []


def test_an_unnegated_mention_still_refuses_the_nos_code():
    """The fix must not disarm the check. Same code, same list, a real specific mention."""
    v = check_answer_detail([HISTOLOGY_NOS], {"histology": "8046"},
                            ev("Final Diagnosis: small cell carcinoma, extensive stage."))
    assert len(v) == 1 and v[0].trigger == "small cell", (
        "a genuine more-specific mention must still be caught")


def test_the_live_spec_refuses_no_answer_on_the_chart_that_was_misjudged():
    """Through the spec as shipped, not a hand-built check — the shipped list is the risk."""
    spec = load_spec(str(SPEC))
    checks = [c if isinstance(c, dict) else c.model_dump() for c in spec.answer_checks]
    quote = "Poorly differentiated non-small cell carcinoma. See note."
    hits = [v for v in check_answer_detail(checks, {"histology": "8046"}, ev(quote))
            if v.field == "histology"]
    assert hits == [], f"the shipped spec still refuses 8046: {[h.message for h in hits]}"


# -------------------------------------------------------------- the message must not answer
# ------------------------------------------- the code and the words must be the same claim
LOBE_EVIDENCE = [
    {"field": "primary_site", "note_id": "Operative-Note-Report_2022-08-25",
     "quote": "PREOPERATIVE DIAGNOSIS: 1. Left lower lobe lung mass 2. COPD"},
    {"field": "primary_site", "note_id": "Rad-Onc-Consult_2022-09-19",
     "quote": "Stage IIIB NSCLC of Left Lower Lobe Lung (LLL)"},
]


def _site_violations(coded: str, evidence=None):
    spec = load_spec(str(SPEC))
    checks = [c if isinstance(c, dict) else c.model_dump() for c in spec.answer_checks]
    return [v for v in check_answer_detail(checks, {"primary_site": coded},
                                           evidence if evidence is not None else LOBE_EVIDENCE)
            if v.field == "primary_site"]


def test_a_site_code_that_names_a_different_lobe_than_the_evidence_is_refused():
    """The measured miss. Nine "left lower lobe", one "LLL", coded C342 — middle lobe.

    `answer_check_rejections` was EMPTY on the real run: `not_less_specific` only asks whether
    the code is too vague, `conflict_requires_nos` only fires when two groups appear, and
    `origin_not_specimen` only reads document headers. Nothing compared the code to the words.
    """
    v = _site_violations("C342")
    kinds = [x.rule_kind for x in v]
    assert "code_matches_cited_text" in kinds, f"C342 must be refused; fired: {kinds}"
    msg = next(x.message for x in v if x.rule_kind == "code_matches_cited_text")
    assert "C343" in msg, "the refusal must say which code the evidence actually names"


def test_the_code_the_evidence_names_passes():
    """C343 is the registry's answer here. A check that refuses it too is not a check."""
    assert [x.rule_kind for x in _site_violations("C343")] == []


def test_the_nos_value_is_not_refused_twice_for_two_different_reasons():
    """C349 is this check's `nos_value`, so it is out of scope here.

    C349 IS refused on this evidence — by `not_less_specific` and `nos_requires_search`, which
    is right, since the record names one lobe unambiguously. But `code_matches_cited_text` must
    not pile on: an answer refused twice with two different remedies is the C349/C348 trap this
    file's `conflict_requires_nos` comment describes, where no value satisfies everything.
    """
    kinds = [x.rule_kind for x in _site_violations("C349")]
    assert "code_matches_cited_text" not in kinds, f"the NOS value is exempt; fired: {kinds}"
    assert kinds, "C349 should still be refused, by the vagueness checks"


def test_the_check_stays_silent_when_the_evidence_names_no_lobe_at_all():
    """That is `not_less_specific`'s question. Firing here would state the wrong reason."""
    ev_no_lobe = [{"field": "primary_site", "note_id": "Path_2020-01-01",
                   "quote": "Specimen: Lung, right. Invasive adenocarcinoma."}]
    kinds = [x.rule_kind for x in _site_violations("C342", ev_no_lobe)]
    assert "code_matches_cited_text" not in kinds, (
        f"no enumerated lobe is named, so this check has nothing to say; fired: {kinds}")


def test_a_negated_lobe_does_not_satisfy_the_code():
    """Same negation rule as the histology check: "no lower lobe involvement" is not evidence
    of a lower-lobe primary, and must not license C343."""
    ev_neg = [{"field": "primary_site", "note_id": "Rad_2020-01-01",
               "quote": "Right upper lobe mass. No evidence of lower lobe involvement."}]
    kinds = [x.rule_kind for x in _site_violations("C343", ev_neg)]
    assert "code_matches_cited_text" in kinds, (
        f"a negated lobe mention must not support the code; fired: {kinds}")


def test_the_two_lobe_vocabularies_have_not_drifted_apart():
    """`conflict_requires_nos.mutually_exclusive` and `code_matches_cited_text.code_wordings`
    are the same lobe vocabulary written twice — once to detect a conflict between groups, once
    to check the code against the group. Two copies agree until they do not, nothing raises, and
    one check starts recognising a wording the other cannot see."""
    spec = load_spec(str(SPEC))
    checks = [c if isinstance(c, dict) else c.model_dump() for c in spec.answer_checks]
    groups = next(c["mutually_exclusive"] for c in checks
                  if c.get("kind") == "conflict_requires_nos" and c.get("field") == "primary_site")
    wordings = next(c["code_wordings"] for c in checks
                    if c.get("kind") == "code_matches_cited_text"
                    and c.get("field") == "primary_site")
    assert {frozenset(g) for g in groups} == {frozenset(v) for v in wordings.values()}, (
        "the two declarations no longer describe the same lobes:\n"
        f"  conflict_requires_nos: {[sorted(g) for g in groups]}\n"
        f"  code_wordings        : { {k: sorted(v) for k, v in wordings.items()} }")


# -------------------------------------------------------- a declared check must be a real one
def test_a_spec_declaring_an_unimplemented_check_kind_is_refused():
    """`check_answer_detail` is an if/elif chain with no final else, so a misspelled kind matched
    nothing and raised nothing. The rule showed up in the YAML and in the manifest's
    `rule_catalog`, and produced zero rejections forever — which reads exactly like a check that
    looked and found nothing."""
    from acr.spec import ProvenanceError, bind_provenance

    spec = load_spec(str(SPEC))
    spec.answer_checks = list(spec.answer_checks) + [
        {"field": "histology", "kind": "code_matches_cited_txt", "nos_values": ["8046"]}]
    with pytest.raises(ProvenanceError, match="nothing implements"):
        bind_provenance(spec)


def test_a_check_message_names_no_code_it_does_not_itself_declare():
    """A check that supplies the answer is not checking it.

    The histology message ended '"favor squamous cell carcinoma" supports 8070 over 8046' — a
    worked example from one chart, delivered as instruction on every firing. It fired on a chart
    whose own searches for "squamous" and "squamous cell carcinoma" both returned zero hits, and
    the agent wrote the code the message named.

    THE LINE IS NOT "no codes in messages". The three `primary_site` messages name C349, and
    should: C349 is that check's own `nos_values`/`nos_value`, and saying what the code the agent
    already chose ASSERTS ("C349 asserts the subsite is unknown") is the rule speaking about
    itself. 8070 was declared nowhere in the histology check — it came from a different chart.
    So the invariant is: a message may name the codes the check declares, and no others.
    """
    import re
    spec = load_spec(str(SPEC))
    checks = [c if isinstance(c, dict) else c.model_dump() for c in spec.answer_checks]
    code = re.compile(r"\b(?:C\d{3}|\d{4})\b")
    offenders = []
    for c in checks:
        declared = {str(v) for v in (c.get("nos_values") or [])}
        if c.get("nos_value"):
            declared.add(str(c["nos_value"]))
        foreign = sorted(set(code.findall(str(c.get("message") or ""))) - declared)
        if foreign:
            offenders.append((c.get("field"), c.get("kind"), foreign))
    assert not offenders, (
        "an answer_check message names a code the check does not declare, so it is dictating "
        f"an answer rather than describing the obligation: {offenders}")
