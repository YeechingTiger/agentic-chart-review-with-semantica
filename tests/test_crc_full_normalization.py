"""Regression guards for the checked-in CRC full-universe authoring bundle."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "assets" / "usecase" / "crc" / "core_v1"
NORMALIZATION = BUNDLE / "normalization"


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(value, dict)
    return value


def _walk_ast(node: dict):
    yield node
    for operand in node.get("operands") or []:
        yield from _walk_ast(operand)
    for key in ("operand", "condition", "then", "else"):
        if isinstance(node.get(key), dict):
            yield from _walk_ast(node[key])


def test_full_normalization_validator_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(NORMALIZATION / "validate_full_normalization.py"),
            str(BUNDLE),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "280 rules, 576 variables/contracts, 39 categories, 8 group plans" in result.stdout


def test_no_ambiguous_empty_boolean_ast_is_published() -> None:
    rules = _load(NORMALIZATION / "normalized_rules.yaml")["rules"]
    nodes = [
        node
        for rule in rules
        for block in rule["requirements"].values()
        for node in _walk_ast(block["evidence_logic"])
    ]
    assert not [
        node
        for node in nodes
        if node.get("op") in {"all_of", "any_of"} and not node.get("operands")
    ]
    assert sum(node.get("op") == "unresolved" for node in nodes) == 82
    assert sum(node.get("op") == "constant" for node in nodes) == 52


def test_coverage_layers_remain_distinct() -> None:
    coverage = _load(BUNDLE / "guideline_universe_evidence_coverage.yaml")
    claims = coverage["coverage_claims"]
    assert claims["reportable_full_rule_evaluability"] == {
        "status": "CANDIDATE_STRUCTURALLY_ASSESSED",
        "full": 0,
        "partial": 0,
        "none": 280,
        "not_assessed": 0,
        "denominator": 280,
        "reason": (
            "All candidates have evidence ASTs and canonical variable bindings; "
            "coverage is a conservative candidate structural assessment with "
            "registrar review still pending."
        ),
    }
    component = claims["component_evidence_reach"]
    assert component["partial"] == 97
    assert "no substring" in component["method"]
    assert claims["observed_crc_coverage"]["status"] == "NOT_ASSESSED"


def test_every_coarsened_use_has_a_versioned_projection_contract() -> None:
    projections = _load(NORMALIZATION / "registry_projections.yaml")["projections"]
    assert len(projections) == 9
    by_variable = {row["canonical_variable_id"]: row for row in projections}
    for row in projections:
        assert row["transformation_id"].endswith(".v1")
        assert row["input_domain"]
        assert row["output_domain"]
        assert row["effective_years"]
        assert row["explicit_loss_list"]

    coverage = _load(NORMALIZATION / "evidence_coverage.yaml")
    coarsened_uses = [
        row
        for row in coverage["variable_uses"]
        if row["canonical_to_registry_projection"] == "coarsened"
    ]
    assert len(coarsened_uses) == 333
    for row in coarsened_uses:
        assert row["transformation_id"] == by_variable[row["variable_id"]]["transformation_id"]
