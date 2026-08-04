"""The Site Mapping had no consumer, so the migration `spec lint` mandates could not be executed.

THE TRAP, exactly as an author walking into it meets it:

  1. Author a contract with `match.doc_type_matches` — raw local note-type substrings.
  2. `acr spec lint` fires F10 at TIER 1: *"Raw local note types do not belong in a Task Contract …
     Replace it with `means:` prose and a Site Mapping."*
  3. Do that. Pay for `acr site-mapping build` over the corpus (one model call per distinct type).
  4. Every `acr run` now dies before the first model call:
     `SiteMappingError: stratum 'pathology' selects documents through a Site Mapping … Refusing
     rather than stratifying`.

Because `run_patient` built `CoverageLedger(docs, strata_from_spec(spec), ForcedSampler(seed))` and
never passed `mapping=`, had no parameter for one, and no run command had a `--mapping` flag. The
only readers of `SiteMapping.from_dict` were `site-mapping review` and `site-mapping diff` — inside
the group that PRODUCES the file. The refusal's own remedy ("pass it to `assign_strata`") was
unfollowable from any CLI.

No shipped spec uses `means:` — `grep -rn "means:" assets/specs/` finds nothing — which is why no
test and no run ever constructed a mapped stratum against the runtime. `strata.py` and `coverage.py`
are unit-tested with `mapping=` passed explicitly, exercising the plumbing the production call site
did not use.
"""

from __future__ import annotations

import json

import pytest

from acr.chartstore.corpus import DocMeta
from acr.contract.site_mapping import SiteMapping, SiteMappingError
from acr.contract.strata import StratumSpec
from acr.review.coverage import CoverageLedger

CONCEPT = "pathology"


def _docs():
    from datetime import date
    return [DocMeta("n1", "SURG-PATH-RESULT", date(2021, 1, 2), 1, 100),
            DocMeta("n2", "Chest-CT-W-Contr", date(2021, 1, 3), 1, 100)]


def _strata():
    """One mapped stratum plus a rest stratum — the shape `spec lint` F10 tells you to write.

    `means:` is TOP-LEVEL and `rest:` lives under `match:` — `StratumSpec.from_dict` reads them
    from different places, and getting either wrong makes `is_mapped` False and this whole file
    inert. `test_the_fixture_is_actually_mapped` is the guard.
    """
    return [StratumSpec.from_dict({"name": CONCEPT, "policy": "search_hits",
                                   "means": "a pathology report"}),
            StratumSpec.from_dict({"name": "rest", "policy": "validate_by_sampling",
                                   "match": {"rest": True}})]


def test_the_fixture_is_actually_mapped():
    """Without this the refusal tests pass by never constructing a mapped stratum at all."""
    assert _strata()[0].is_mapped
    assert _strata()[1].rest


@pytest.fixture
def mapping_file(tmp_path):
    """A mapping JSON of the shape `acr site-mapping build --out` writes."""
    from acr.contract.site_mapping import Concept, TypeAssignment, concepts_hash
    concepts = (Concept(name=CONCEPT, means="a pathology report"),)
    m = SiteMapping(
        corpus_id="test-corpus", concepts=concepts,
        bound_concepts_hash=concepts_hash(concepts),
        assignments={"SURG-PATH-RESULT": TypeAssignment(
            doc_type="SURG-PATH-RESULT", concept=CONCEPT, why="it is one", n_documents=1)},
        model="test", built_at="2026-08-04", provenance="model_assigned")
    p = tmp_path / "mapping.json"
    p.write_text(json.dumps(m.to_dict()), encoding="utf-8")
    return p


def test_a_mapped_stratum_without_a_mapping_refuses(mapping_file):
    """The refusal itself is right and stays — this pins that it is the thing being worked around."""
    with pytest.raises(SiteMappingError, match="Site Mapping"):
        CoverageLedger(_docs(), _strata())


def test_a_mapped_stratum_with_a_mapping_stratifies(mapping_file):
    mapping = SiteMapping.from_dict(json.loads(mapping_file.read_text(encoding="utf-8")))
    ledger = CoverageLedger(_docs(), _strata(), mapping=mapping)
    assert ledger is not None


def test_run_patient_accepts_a_mapping(mapping_file):
    """The parameter that did not exist. Signature-level, so it costs no model call."""
    import inspect

    from acr.review.agent import run_patient
    assert "site_mapping" in inspect.signature(run_patient).parameters


def test_the_run_commands_expose_the_flag():
    """`run`, `batch` and `extract` all take a spec, so all three can meet a mapped stratum."""
    import inspect

    from acr.commands.cli_chart import batch, run
    from acr.commands.cli_pipeline import extract
    for fn in (run, batch, extract):
        assert "site_mapping" in inspect.signature(fn).parameters, fn.__name__


def test_a_mapped_spec_with_no_flag_names_the_flag(tmp_path, monkeypatch):
    """A refusal from inside the ledger tells an author about `assign_strata`, which is not
    something they can pass. The door must name `--mapping`."""
    from acr.commands import cli_chart
    msg = cli_chart._require_mapping_for(_strata(), "")
    assert msg and "--mapping" in msg and "site-mapping build" in msg


def test_an_unmapped_spec_needs_no_flag():
    """Four of five shipped specs declare `doc_type_matches` and must keep running with no flag."""
    from acr.commands import cli_chart
    plain = [StratumSpec.from_dict({"name": "path", "policy": "search_hits",
                                   "match": {"doc_type_matches": ["PATH"]}})]
    assert cli_chart._require_mapping_for(plain, "") is None
