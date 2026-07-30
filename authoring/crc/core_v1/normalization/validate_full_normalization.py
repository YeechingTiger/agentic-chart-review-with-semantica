#!/usr/bin/env python3
"""Validate checked-in full CRC normalization artifacts without rebuilding them."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


BLOCKS = ("eligibility", "action", "timing", "exceptions")
OPS = {"all_of", "any_of", "not", "conditional", "constant", "unresolved"}


def load(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected mapping")
    return value


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def digest(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def leaves(node: Any, where: str) -> set[str]:
    if not isinstance(node, dict):
        raise ValueError(f"{where}: AST node must be a mapping")
    if node.get("variable"):
        return {str(node["variable"])}
    op = str(node.get("op") or "")
    if op not in OPS:
        raise ValueError(f"{where}: invalid op {op!r}")
    if op == "constant":
        if not isinstance(node.get("value"), bool):
            raise ValueError(f"{where}: constant requires a Boolean value")
        return set()
    if op == "unresolved":
        if not str(node.get("reason_code") or "").strip():
            raise ValueError(f"{where}: unresolved requires reason_code")
        return set()
    if op == "not":
        return leaves(node.get("operand"), f"{where}.operand")
    if op == "conditional":
        if node.get("condition") is None or node.get("then") is None:
            raise ValueError(f"{where}: conditional requires condition and then")
        out: set[str] = set()
        for key in ("condition", "then", "else"):
            if node.get(key) is not None:
                out.update(leaves(node[key], f"{where}.{key}"))
        return out
    operands = as_list(node.get("operands"))
    if not operands:
        raise ValueError(
            f"{where}: {op} cannot be empty; use constant or unresolved explicitly"
        )
    out: set[str] = set()
    for index, operand in enumerate(operands):
        out.update(leaves(operand, f"{where}.operands[{index}]"))
    return out


def validate(bundle: Path) -> str:
    root = bundle / "normalization"
    universe = load(bundle / "intake" / "rule_universe.yaml")
    source_denominator = load(bundle / "intake" / "source_denominator.yaml")
    source_register = load(bundle / "source_register.yaml")
    normalized = load(root / "normalized_rules.yaml")
    variables_doc = load(root / "canonical_variables.yaml")
    bindings_doc = load(root / "concept_bindings.yaml")
    projections_doc = load(root / "registry_projections.yaml")
    groups_doc = load(root / "execution_group_plan.yaml")
    coverage = load(root / "evidence_coverage.yaml")
    gaps = load(root / "gap_assessment.yaml")
    manifest = load(root / "manifest.yaml")

    source_rows = as_list(universe.get("candidates"))
    source_index = {str(row.get("candidate_id")): row for row in source_rows}
    rules = as_list(normalized.get("rules"))
    rule_index = {str(row.get("candidate_id")): row for row in rules}
    if len(source_index) != len(source_rows):
        raise ValueError("rule_universe has duplicate candidate IDs")
    if len(rule_index) != len(rules):
        raise ValueError("normalized_rules has duplicate candidate IDs")
    if set(rule_index) != set(source_index):
        raise ValueError("normalized rule IDs do not exactly match the intake universe")
    if normalized.get("ast_normalized_count") != len(rules):
        raise ValueError("ast_normalized_count is stale")
    register_ids = {
        str(row.get("source_id")) for row in as_list(source_register.get("sources"))
    }
    source_slices = {
        str(row.get("source_id")): row
        for row in as_list(source_denominator.get("sources"))
    }
    for candidate_id, source in source_index.items():
        source_id = str(source.get("source_id") or "")
        if source_id not in source_slices:
            raise ValueError(
                f"{candidate_id}: source_id {source_id!r} absent from source_denominator"
            )
        register_source_id = str(
            source_slices[source_id].get("register_source_id") or ""
        )
        if register_source_id not in register_ids:
            raise ValueError(
                f"{candidate_id}: source slice {source_id!r} does not resolve to source_register"
            )

    variables = as_list(variables_doc.get("variables"))
    variable_index = {str(row.get("variable_id")): row for row in variables}
    if len(variable_index) != len(variables):
        raise ValueError("canonical_variables has duplicate IDs")
    used: set[str] = set()
    for candidate_id, rule in rule_index.items():
        declared = set(as_list(rule.get("required_variables")))
        union: set[str] = set()
        for block_name in BLOCKS:
            block = (rule.get("requirements") or {}).get(block_name) or {}
            block_vars = {str(value) for value in as_list(block.get("variables"))}
            ast_vars = leaves(block.get("evidence_logic"), f"{candidate_id}.{block_name}")
            if block_vars != ast_vars:
                raise ValueError(f"{candidate_id}.{block_name}: variables differ from AST leaves")
            union.update(block_vars)
        if declared != union:
            raise ValueError(f"{candidate_id}: required_variables differs from block union")
        used.update(union)
    if used != set(variable_index):
        raise ValueError(
            f"canonical variable set mismatch; unused={sorted(set(variable_index)-used)}, "
            f"undefined={sorted(used-set(variable_index))}"
        )

    bindings = as_list(bindings_doc.get("bindings"))
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for row in bindings:
        by_candidate.setdefault(str(row.get("candidate_id")), []).append(row)
    for candidate_id, source in source_index.items():
        expected = {str(value) for value in as_list(source.get("critical_variable_concepts"))}
        actual = {
            str(row.get("source_concept"))
            for row in by_candidate.get(candidate_id, [])
        }
        if actual != expected:
            raise ValueError(f"{candidate_id}: source concept binding set mismatch")
        for row in by_candidate.get(candidate_id, []):
            targets = {str(value) for value in as_list(row.get("canonical_variable_ids"))}
            if not targets or not targets <= set(rule_index[candidate_id]["required_variables"]):
                raise ValueError(f"{candidate_id}/{row.get('source_concept')}: invalid targets")

    contract_dir = root / "variable_contracts"
    contract_paths = sorted(contract_dir.glob("*.yaml"))
    contract_vars: set[str] = set()
    contract_hash_rows: list[tuple[str, str]] = []
    for path in contract_paths:
        contract = load(path)
        variable_id = str(contract.get("canonical_variable_id") or "")
        if variable_id in contract_vars:
            raise ValueError(f"duplicate contract for {variable_id}")
        contract_vars.add(variable_id)
        value = dict(contract)
        recorded = str(value.pop("contract_hash", ""))
        if recorded != digest(value):
            raise ValueError(f"{path}: stale contract_hash")
        if not as_list(contract.get("boundary_cases")):
            raise ValueError(f"{path}: boundary_cases must be non-empty")
        if (contract.get("conformance") or {}).get("verdict") != "needs_revision":
            raise ValueError(
                f"{path}: candidate full-universe contract must remain needs_revision"
            )
        if (
            (contract.get("execution_binding") or {}).get("runtime_status")
            != "planned_not_materialized"
        ):
            raise ValueError(
                f"{path}: runtime status must remain planned_not_materialized"
            )
        provenance = as_list(contract.get("provenance"))
        if not provenance:
            raise ValueError(f"{path}: per-enforced-element provenance is missing")
        provenance_elements = {
            str(row.get("element") or "")
            for row in provenance
            if isinstance(row, dict)
        }
        required_provenance = {
            "roles",
            "datatype",
            "value_domain",
            "temporal_meaning",
            "mapping_level",
            "registry_mapping",
            "question",
            "applicability_guard",
            "field_contract.name",
            "field_contract.type",
            "field_contract.nullable",
            "field_contract.value_domain",
            "evidence_contract.positive_witness",
            "evidence_contract.negative_or_unknown_proof",
            "proof_obligation",
            "abstention",
            "boundary_cases",
        }
        if not required_provenance <= provenance_elements:
            raise ValueError(
                f"{path}: provenance missing {sorted(required_provenance-provenance_elements)}"
            )
        contract_hash_rows.append((str(contract.get("contract_id")), recorded))
    if contract_vars != set(variable_index):
        raise ValueError("contract set does not equal canonical variable set")

    projections = as_list(projections_doc.get("projections"))
    if projections_doc.get("projection_contract_count") != len(projections):
        raise ValueError("registry_projections.projection_contract_count is stale")
    projection_by_variable: dict[str, dict[str, Any]] = {}
    transformation_ids: set[str] = set()
    for row in projections:
        variable_id = str(row.get("canonical_variable_id") or "")
        transformation_id = str(row.get("transformation_id") or "")
        if variable_id in projection_by_variable or transformation_id in transformation_ids:
            raise ValueError("duplicate registry projection variable or transformation_id")
        value = dict(row)
        recorded = str(value.pop("transformation_hash", ""))
        if recorded != digest(value):
            raise ValueError(f"{transformation_id}: stale transformation_hash")
        for key in (
            "input_domain",
            "output_domain",
            "effective_years",
            "explicit_loss_list",
        ):
            if not row.get(key):
                raise ValueError(f"{transformation_id}: {key} is required")
        if (
            (row.get("transformation") or {}).get("implementation_status")
            != "planned_not_materialized"
        ):
            raise ValueError(
                f"{transformation_id}: implementation must remain planned_not_materialized"
            )
        projection_by_variable[variable_id] = row
        transformation_ids.add(transformation_id)
    expected_projection_variables = {
        variable_id
        for variable_id, variable in variable_index.items()
        if variable.get("mapping_level") == "registry_coarsened"
    }
    if set(projection_by_variable) != expected_projection_variables:
        raise ValueError(
            "registry projection contracts do not equal registry_coarsened variables"
        )

    group_member_rows = [
        str(variable_id)
        for group in as_list(groups_doc.get("groups"))
        for variable_id in as_list(group.get("canonical_variable_ids"))
    ]
    group_members = set(group_member_rows)
    if len(group_member_rows) != len(group_members):
        raise ValueError("execution group plan assigns at least one variable more than once")
    if group_members != set(variable_index):
        raise ValueError("execution group plan does not cover every canonical variable")

    category_files = sorted((root / "rules_by_category").glob("*.yaml"))
    category_rule_ids: list[str] = []
    for path in category_files:
        category_rule_ids.extend(
            str(row.get("candidate_id")) for row in as_list(load(path).get("rules"))
        )
    if Counter(category_rule_ids) != Counter(rule_index.keys()):
        raise ValueError("rules_by_category does not partition normalized rules exactly once")

    if len(as_list(coverage.get("rules"))) != len(rules):
        raise ValueError("evidence coverage rule rows are stale")
    variable_uses = as_list(coverage.get("variable_uses"))
    expected_variable_uses = Counter(
        (
            str(rule["candidate_id"]),
            f"requirements.{block}.evidence_logic",
            str(variable_id),
        )
        for rule in rules
        for block in BLOCKS
        for variable_id in as_list(rule["requirements"][block].get("variables"))
    )
    actual_variable_uses = Counter(
        (
            str(row.get("candidate_id")),
            str(row.get("predicate_path")),
            str(row.get("variable_id")),
        )
        for row in variable_uses
    )
    if actual_variable_uses != expected_variable_uses:
        raise ValueError("evidence coverage variable-use rows do not match rule AST uses")
    for row in variable_uses:
        if row.get("semantic_support") not in {
            "exact", "coarsened", "absent", "not_applicable", "unknown"
        }:
            raise ValueError("invalid variable-use semantic_support")
        if row.get("temporal_alignment") not in {
            "exact", "conditional", "misaligned", "unknown"
        }:
            raise ValueError("invalid variable-use temporal_alignment")
        if row.get("provenance_suitability") not in {
            "agreement_target", "weak_label", "same_source_dependent", "not_truth", "unknown"
        }:
            raise ValueError("invalid variable-use provenance_suitability")
        projection = row.get("canonical_to_registry_projection")
        if projection not in {"direct", "coarsened", "none", "unknown"}:
            raise ValueError("invalid canonical_to_registry_projection")
        if projection == "coarsened":
            variable_id = str(row.get("variable_id") or "")
            expected_id = str(
                projection_by_variable.get(variable_id, {}).get(
                    "transformation_id"
                )
                or ""
            )
            if not expected_id or row.get("transformation_id") != expected_id:
                raise ValueError(
                    f"{variable_id}: coarsened use lacks its versioned transformation_id"
                )
        elif row.get("transformation_id"):
            raise ValueError(
                "only coarsened variable uses may name a transformation_id"
            )
    if len(as_list(gaps.get("variable_gaps"))) != len(variables):
        raise ValueError("gap assessment variable rows are stale")
    expected_manifest = {
        "source_candidate_count": len(source_rows),
        "normalized_rule_count": len(rules),
        "canonical_variable_count": len(variables),
        "variable_contract_count": len(contract_paths),
        "registry_projection_contract_count": len(projections),
        "execution_group_plan_count": len(as_list(groups_doc.get("groups"))),
        "category_file_count": len(category_files),
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(f"manifest.{key}: expected {expected}, got {manifest.get(key)!r}")
    if manifest.get("source_slice_crosswalk_status") != "complete":
        raise ValueError("manifest.source_slice_crosswalk_status must be complete")
    if manifest.get("contract_set_hash") != digest(contract_hash_rows):
        raise ValueError("manifest.contract_set_hash is stale")

    return (
        f"valid: {len(rules)} rules, {len(variables)} variables/contracts, "
        f"{len(category_files)} categories, {len(as_list(groups_doc.get('groups')))} group plans"
    )


def main() -> int:
    bundle = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    try:
        print(validate(bundle))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERROR {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
