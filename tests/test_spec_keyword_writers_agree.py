"""Two implementations of "write a keyword list into a spec", and they disagreed.

`derive._KW_ELEMENT_RE` made the claims segment NON-CAPTURING and then descended unconditionally
into `for_negative["strata"]`. `assetdev._ELEMENT_RE` captures `claim` and descends into `claims[]`
first. So for a contract whose keywords live under a CLAIM — `STORE.1860_1880.first_recurrence.yaml`
has `strata: []` at the top level and its lists under `claims[disease_free_interval_existed]` —
the regex matched, the write went looking in the wrong place, and `derive` aborted with

    no stratum 'can_establish'; adopting into one that does not exist creates a list nothing reads

which is FALSE: the stratum exists, one level down. One of five shipped contract shapes could not be
adopted through the derive plane, and the refusal named the wrong cause. Worse for a spec carrying
the same stratum name in both places: the write would target the wrong element.

Nothing was corrupted — the provenance `element_hash` check aborts before persisting — but a stage
that cannot run on a fifth of the contracts is a stage nobody can rely on.
"""

from __future__ import annotations

import pytest
import yaml

from acr.core import site
from acr.improvement import assetdev as A

CLAIM_SCOPED = site.specs_root() / "STORE.1860_1880.first_recurrence.yaml"


def _claim_element(doc: dict) -> str | None:
    """The `required_keywords` element path of the first claim-scoped stratum in this spec."""
    fn = (doc.get("proof_obligation") or {}).get("for_negative") or {}
    for claim in (fn.get("claims") or []):
        for st in (claim.get("strata") or []):
            if "required_keywords" in st:
                return (f"proof_obligation.for_negative.claims[{claim['id']}]"
                        f".strata[{st['name']}].required_keywords")
    return None


@pytest.fixture
def claim_spec():
    if not CLAIM_SCOPED.is_file():
        pytest.skip(f"{CLAIM_SCOPED.name} is not in this checkout")
    doc = yaml.safe_load(CLAIM_SCOPED.read_text(encoding="utf-8"))
    element = _claim_element(doc)
    if not element:
        pytest.skip("this spec no longer declares keywords under a claim")
    return doc, element


def test_the_population_is_real(claim_spec):
    """Guards the rest: if no shipped spec is claim-scoped, every test below is inert."""
    doc, element = claim_spec
    assert "claims[" in element
    assert not ((doc.get("proof_obligation") or {}).get("for_negative") or {}).get("strata"), \
        "this spec was expected to declare its strata only under a claim"


def test_assetdev_writes_a_claim_scoped_element(claim_spec):
    doc, element = claim_spec
    A._set_keywords(doc, element, ["recurrence", "relapse"], CLAIM_SCOPED)
    fn = doc["proof_obligation"]["for_negative"]
    written = [st["required_keywords"]
               for c in fn["claims"] for st in (c.get("strata") or [])
               if "required_keywords" in st]
    assert ["recurrence", "relapse"] in written


def test_derive_writes_the_same_element_the_same_way(claim_spec):
    """The regression: `derive` must reach the identical element, not abort on a false cause."""
    doc, element = claim_spec
    from acr.improvement.derive import _set_keywords as derive_set
    derive_set(doc, element, ["recurrence", "relapse"], CLAIM_SCOPED)
    fn = doc["proof_obligation"]["for_negative"]
    written = [st["required_keywords"]
               for c in fn["claims"] for st in (c.get("strata") or [])
               if "required_keywords" in st]
    assert ["recurrence", "relapse"] in written


def test_both_writers_produce_byte_identical_documents(claim_spec):
    """The property that keeps them from drifting again: same input, same output document."""
    import copy

    from acr.improvement.derive import _set_keywords as derive_set
    doc, element = claim_spec
    a, b = copy.deepcopy(doc), copy.deepcopy(doc)
    A._set_keywords(a, element, ["recurrence"], CLAIM_SCOPED)
    derive_set(b, element, ["recurrence"], CLAIM_SCOPED)
    assert yaml.safe_dump(a, sort_keys=False) == yaml.safe_dump(b, sort_keys=False)


def test_a_top_level_stratum_still_writes(claim_spec):
    """The unscoped form must not regress — four of five shipped specs use it."""
    from acr.improvement.derive import _set_keywords as derive_set
    doc = {"proof_obligation": {"for_negative": {"strata": [
        {"name": "can_establish", "required_keywords": ["old"]}]}}}
    derive_set(doc, "proof_obligation.for_negative.strata[can_establish].required_keywords",
               ["new"], CLAIM_SCOPED)
    assert doc["proof_obligation"]["for_negative"]["strata"][0]["required_keywords"] == ["new"]


def test_a_missing_stratum_still_refuses(claim_spec):
    """The refusal is right; only its trigger was wrong.

    `derive.AdoptionAborted`, not assetdev's: `derive`'s callers catch a `DerivationError`, so the
    delegate's exception is re-raised at the boundary rather than allowed to escape as a traceback.
    """
    from acr.improvement.derive import AdoptionAborted as DeriveAborted
    from acr.improvement.derive import _set_keywords as derive_set
    doc = {"proof_obligation": {"for_negative": {"strata": [{"name": "other"}]}}}
    with pytest.raises(DeriveAborted, match="no stratum"):
        derive_set(doc, "proof_obligation.for_negative.strata[can_establish].required_keywords",
                   ["new"], CLAIM_SCOPED)
