"""There must be exactly one coverage ledger.

Two independent accounts of "how much was covered" can disagree, nothing raises when they
do, and you are left with two numbers and no way to choose which to believe. An earlier
revision of this repo had precisely that — a flat ledger in `state.py` and a stratified one
in `coverage.py` — and worse, the stratified one's docstring already claimed to have
replaced the flat one. The claim was true of the intention and false of the code, and
nothing failed.

The value of these tests is not that they pass today. It is that the next person who leaves
"the old class, just for compatibility" behind gets a red line instead of a silent second
source of truth.
"""
from __future__ import annotations

import inspect

import acr.core.state
import acr.review.coverage
import acr.review.tools.toolbox


def test_flat_ledger_is_gone_not_merely_unused():
    assert not hasattr(acr.core.state, "CoverageLedger"), (
        "the flat CoverageLedger must be deleted from state.py, not left importable — "
        "anything still importable will eventually be imported"
    )
    assert not hasattr(acr.core.state, "check_proof_obligation"), (
        "the flat gate went with it; the stratified ledger is what the gate consumes now"
    )


def test_coverage_is_the_only_definition():
    assert hasattr(acr.review.coverage, "CoverageLedger")
    src = inspect.getsource(acr.core.state)
    assert "class CoverageLedger" not in src


def test_everything_imports_the_stratified_ledger():
    assert acr.review.tools.toolbox.CoverageLedger is acr.review.coverage.CoverageLedger


def test_the_ledger_reports_which_mode_it_is_in():
    """An unstratified run is a legitimate ablation arm, but it must say so in the trace,
    otherwise the two arms are indistinguishable after the fact."""
    from pathlib import Path

    from acr.chartstore.corpus import Corpus

    corpus = Path(__file__).resolve().parents[1] / "corpus" / "patients"
    docs, _ = Corpus(corpus).chart("SYN0002").list_documents(limit=10_000)

    bare = acr.review.coverage.CoverageLedger(docs, [])
    assert bare.to_dict()["mode"] == "unstratified"

    strata = acr.review.coverage.strata_from_spec(
        __import__("acr.contract.spec", fromlist=["load_spec"]).load_spec(
            Path(__file__).resolve().parents[1] / "assets" / "specs"
            / "STORE.400_522_523.site_histology_behavior.yaml"
        )
    )
    assert strata, "this spec declares strata; strata_from_spec must find them"
    full = acr.review.coverage.CoverageLedger(docs, strata)
    assert full.to_dict()["mode"] == "stratified_exclusion"
    assert full.to_dict()["sample_seed"] is not None, "the seed has to reach the trace"


def test_the_runtime_imports_the_stratified_ledger_too():
    """`acr.graph` used to be checked here. It is gone; the runtime that replaced it is not
    exempt from the one-ledger rule, and the import is lazy so it is asserted at the call."""
    import inspect

    import pytest
    pytest.importorskip("langchain.agents")
    import acr.review.agent
    src = inspect.getsource(acr.review.agent.run_patient)
    assert "from .coverage import CoverageLedger" in src, (
        "the runtime must take the ledger from acr.review.coverage and not define or import another")
