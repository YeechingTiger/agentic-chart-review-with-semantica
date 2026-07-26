"""There is exactly one route by which an answer becomes gate-validated.

An invariant hung on one edge has to account for every edge into the node it protects. The
gate lived on `submit_answer`; `finalize` had a second inbound edge, from reflect's
SUFFICIENT verdict; and so a run could reach an answer having skipped the proof obligation
entirely. It still printed a `proof_obligation` field — computed, but with no power to
refuse — which is a comment dressed as a check, and it is what made the bypass look
inspected.

Behavioural tests only cover the paths someone thought of. These are structural: they assert
the shape of the graph and the single origin of `gate_validated`, so a future third edge to
`finalize` fails here rather than in production.
"""
from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

import acr.graph as G

SRC = inspect.getsource(G)


def test_sufficient_routes_to_submission_not_to_the_exit():
    """reflect keeps its judgement, but expresses it as 'go submit', not 'we are done'."""
    body = inspect.getsource(G.ChartReviewAgent._after_reflect)
    m = re.search(r'if v == "SUFFICIENT":(.*?)if v == "STUCK":', body, re.S)
    assert m, "SUFFICIENT and STUCK must be handled separately — they are different signals"
    assert '"act"' in m.group(1), "SUFFICIENT must route to act (and thus to submit_answer)"
    assert '"finalize"' not in m.group(1), (
        "routing SUFFICIENT straight to finalize reopens the bypass around the gate"
    )


def test_stuck_is_not_asked_to_prove_coverage():
    """A give-up asserts no coverage, so requiring it to prove coverage is incoherent."""
    body = inspect.getsource(G.ChartReviewAgent._after_reflect)
    m = re.search(r'if v == "STUCK":(.*?)if v == "REPLAN"', body, re.S)
    assert m and '"finalize"' in m.group(1)
    assert "AGENT_GAVE_UP" in m.group(1), "a give-up has to be labelled as one"


def test_gate_validated_has_exactly_one_origin():
    """If this count ever exceeds one, the gate has grown a second door."""
    assignments = re.findall(r"gate_validated\s*=\s*True", SRC)
    assert len(assignments) == 1, (
        f"gate_validated is set True in {len(assignments)} places; it must be set only where "
        "submit_answer is accepted"
    )
    acted = inspect.getsource(G.ChartReviewAgent._n_act)
    assert "gate_validated = True" in acted


def test_an_unvalidated_answer_carries_no_coverage_claim():
    """Attaching the ledger to an answer that never passed the gate lets it read as if it had."""
    fin = inspect.getsource(G.ChartReviewAgent._n_finalize)
    # Anchor on the gate_validated branch; the function has other else-clauses. Strip
    # comments first — the ungated branch explains in prose that it withholds
    # coverage_attested, and a naive substring check would trip over its own documentation.
    def code_only(block: str) -> str:
        return "\n".join(ln for ln in block.splitlines() if not ln.strip().startswith("#"))

    _, tail = fin.split('if s.get("gate_validated"):', 1)
    validated, unvalidated = (code_only(x) for x in tail.split("        else:", 1))

    assert "coverage_attested" in validated, "a gated answer should carry its ledger"
    assert "coverage_attested" not in unvalidated, (
        "the ungated branch must not attach coverage_attested — that is what lets an "
        "unvalidated negative read as a verified one"
    )
    assert "route_to_human" in unvalidated


def test_every_negative_declares_its_basis():
    """EVIDENCE_INSUFFICIENT means three different things with three different remedies:
    a verified negative can be filed, a give-up and a budget exhaustion must reach a human."""
    fin = inspect.getsource(G.ChartReviewAgent._n_finalize)
    assert "negative_basis" in fin
    for basis in ("GATE_VALIDATED", "AGENT_GAVE_UP", "BUDGET_EXHAUSTED"):
        assert basis in SRC, f"{basis} must be a reachable basis"


def test_finalize_is_the_only_node_wired_to_end():
    """A second exit would be a second way to produce an answer."""
    build = inspect.getsource(G.ChartReviewAgent._build)
    ends = re.findall(r"add_edge\(\s*([A-Za-z_\"']+)\s*,\s*END\s*\)", build)
    assert ends == ['"finalize"'], f"exactly one node may reach END, found {ends}"


def test_finalize_inbound_edges_are_enumerated():
    """Both routes into finalize must be accounted for, and neither may claim validation
    unless it actually passed the gate."""
    build = inspect.getsource(G.ChartReviewAgent._build)
    assert build.count('"finalize"') >= 2, "finalize has more than one inbound edge by design"
    fin = inspect.getsource(G.ChartReviewAgent._n_finalize)
    assert 's.get("gate_validated")' in fin, (
        "finalize must branch on whether the answer was actually gated, not assume it"
    )
