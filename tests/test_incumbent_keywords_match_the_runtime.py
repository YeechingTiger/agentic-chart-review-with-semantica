"""The develop plane must price candidates against the list a RUN actually searches.

THE SEAM. `review/coverage_planner.spec_declared_keywords` unions three places a spec can declare
terms — `proof_obligation.required_keywords`, `for_negative.strata[].required_keywords`, and
`for_negative.claims[].strata[].required_keywords` — and `_blank_plan` seeds both
`initial_keywords` and `keywords` from it, so THAT is what a run searches.

The develop plane went through `contract.strata.strata_from_spec`, which reads only the two
stratum locations and never `proof_obligation.required_keywords`. Three readers did it:
`assetdev.RetrievalPlan.from_spec`, `derive._stage123`, and `tools/build_termcache.py::needles_from`.
The termcache half lives in `tests/test_producers_across_distributions.py`, because `tools/` ships in
only one distribution and a test that spans two can only run in the composed tree.

WHY IT IS NOT COSMETIC. `derive.price_terms` scores every candidate's MARGINAL recall over the
incumbent list, so a term the runtime already searches gets credited with the answers it rescues;
`assetdev.measure` scores the incumbent by replaying that plan and `certify` certifies improvement
over it. Then `adopt` writes into `strata[]`, which the runtime unions with the terms the develop
plane could not see — so **the deployed configuration is not the one that was certified.** Measured:
`STORE.700_880.stage.yaml` declares 6 terms at `proof_obligation.required_keywords`, one of which
(`tnm`) appears nowhere in its strata — runtime 11 terms, develop 10.

This file is a producer→consumer test: it asks the two halves the same question about every shipped
contract and requires the same answer.
"""

from __future__ import annotations

import pytest

from acr.contract.spec import load_spec
from acr.core import site
from acr.improvement.assetdev import RetrievalPlan
from acr.review.coverage_planner import spec_declared_keywords

SPECS = sorted(p for p in site.specs_root().glob("*.yaml"))


def _norm(terms) -> set[str]:
    return {str(t).strip().lower() for t in terms if str(t).strip()}


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.name)
def test_the_shipped_specs_are_a_real_population(path):
    assert path.is_file()


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.name)
def test_derive_prices_against_the_runtime_list(path):
    from acr.improvement.derive import incumbent_keywords
    spec = load_spec(path)
    assert _norm(incumbent_keywords(spec)) == _norm(spec_declared_keywords(spec))


@pytest.mark.parametrize("path", SPECS, ids=lambda p: p.name)
def test_the_retrieval_plan_sees_every_declared_term(path):
    """`RetrievalPlan` keeps keywords PER STRATUM, and `proof_obligation.required_keywords` is
    scoped to no stratum — the runtime unions them all into one search list, so they belong on the
    stratum whose list is the one being developed. What must hold is the union."""
    spec = load_spec(path)
    fields = list(getattr(spec, "fields", []) or [])
    if not fields:
        pytest.skip("no fields declared")
    name = str(getattr(fields[0], "name", fields[0]))
    try:
        plan = RetrievalPlan.from_spec(spec, name)
    except Exception as e:                      # noqa: BLE001 — a spec with no strata is not this test's subject
        pytest.skip(f"no developable plan: {type(e).__name__}: {e}")
    seen = {t for _, terms in plan.keywords for t in terms}
    missing = _norm(spec_declared_keywords(spec)) - _norm(seen)
    assert not missing, (
        f"the develop plane cannot see {sorted(missing)}, which a run searches. Every candidate "
        f"would be priced against a shorter list than the one deployed.")


def test_at_least_one_spec_actually_exercises_the_defect():
    """An INERT parametrised test passes because it had nothing to examine. This asserts the
    population contains the case: some shipped spec declares a term ONLY at the
    `proof_obligation.required_keywords` level, so the tests above are not vacuous."""
    from acr.contract.strata import strata_from_spec
    exercised = []
    for path in SPECS:
        spec = load_spec(path)
        strata_only = _norm(k for st in strata_from_spec(spec) for k in st.required_keywords)
        if _norm(spec_declared_keywords(spec)) - strata_only:
            exercised.append(path.name)
    assert exercised, ("no shipped spec declares a term outside its strata, so every test in this "
                       "file is inert. Keep one that does, or delete this file.")
