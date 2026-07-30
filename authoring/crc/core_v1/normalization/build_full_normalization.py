#!/usr/bin/env python3
"""Merge module authoring into a complete CRC rule/variable normalization bundle.

The builder is intentionally strict about coverage and traceability:

* every intake candidate must occur in exactly one module;
* every source concept use must have an explicit candidate-scoped binding;
* every rule block must have an evidence AST whose leaves equal its variable list;
* every canonical variable must have exactly one authoring extraction contract.

The resulting contracts are authoring units. Execution groups are plans until a reviewed
native multi-field ExtractionSpec is materialized; no clinical or production approval is
implied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


BLOCKS = ("eligibility", "action", "timing", "exceptions")
AST_OPS = {"all_of", "any_of", "not", "conditional", "constant", "unresolved"}
MAPPING_LEVELS = {
    "registry_direct",
    "registry_coarsened",
    "chart_extension",
    "derived",
    "outside_current_sources",
}
COARSENED_PROJECTION_LOSSES: dict[str, list[str]] = {
    "crc.ici_contraindication": [
        "ICI-specific reason is collapsed into a broad biological-response-modifier code.",
        "Onset, severity, duration, resolution, and decision-time applicability are lost.",
        "Patient preference and contraindication may not remain distinguishable.",
    ],
    "crc.metastatic_status": [
        "Decision-time disease status is collapsed into diagnosis/recurrence-oriented registry items.",
        "Initial metastatic disease and later recurrence may not remain distinguishable.",
        "Evidence source, lesion context, resectability, and decision date are lost.",
    ],
    "crc.mmr_ihc_integrated_status": [
        "Individual MLH1, PMS2, MSH2, and MSH6 results are lost.",
        "IHC method, controls, specimen, result date, subclonal loss, and discordance are lost.",
        "The registry item combines MMR and MSI summary semantics.",
    ],
    "crc.msi_status": [
        "Raw MSI assay status is not distinguishable from an MMR-IHC-derived summary.",
        "Assay method, loci/panel, QC, specimen, result date, and discordance are lost.",
        "Indeterminate and not-assessed states may collapse into standard summary codes.",
    ],
    "crc.patient_preference": [
        "Preference among clinically acceptable options is reduced to coarse refusal/nonreceipt coding.",
        "Discussion content, alternatives, decision date, and change over time are lost.",
        "Access barriers and contraindications may not remain distinguishable from preference.",
    ],
    "crc.systemic_regimen": [
        "Exact agents and named regimen are reduced to a broad chemotherapy summary class.",
        "Line, intent, dose, schedule, cycles, completion, and progression context are lost.",
        "Later-line and outside-facility treatment are not represented as complete longitudinal history.",
    ],
    "crc.systemic_therapy_start_date": [
        "A generic indexed systemic-therapy start is reduced to a first-course registry date.",
        "Line, regimen, intent, ordering versus administration, and later starts are lost.",
        "Outside-facility and recurrent-disease treatment timing may be absent.",
    ],
    "crc.treatment_contraindication": [
        "Modality-specific clinical contraindications are reduced to coarse non-treatment reasons.",
        "Severity, temporality, reversibility, and the indexed treatment option are lost.",
        "Preference, access, comorbidity, and true contraindication may not remain distinguishable.",
    ],
    "crc.treatment_sequence": [
        "Exact event order and dates are reduced to broad modality-sequence categories.",
        "Regimen, line, intent, overlap, interruption, and completion are lost.",
        "Later-line and outside-facility sequences are not represented completely.",
    ],
}


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected mapping")
    return value


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def canonical_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "_", value.casefold()).strip("_")


def safe_variable_suffix(variable_id: str) -> str:
    """Preserve namespace boundaries so dots cannot collide with underscores."""
    parts = variable_id.removeprefix("crc.").split(".")
    return "__".join(safe_name(part) for part in parts)


def resolve_alias(variable_id: str, aliases: dict[str, str]) -> str:
    seen: set[str] = set()
    current = variable_id
    while current in aliases:
        if current in seen:
            raise ValueError(f"canonical alias cycle contains {current!r}")
        seen.add(current)
        current = aliases[current]
    return current


def rewrite_ast_aliases(node: Any, aliases: dict[str, str]) -> None:
    if not isinstance(node, dict):
        return
    if node.get("variable"):
        node["variable"] = resolve_alias(str(node["variable"]), aliases)
        return
    for operand in as_list(node.get("operands")):
        rewrite_ast_aliases(operand, aliases)
    for key in ("operand", "condition", "then", "else"):
        if node.get(key) is not None:
            rewrite_ast_aliases(node[key], aliases)


def ast_leaves(node: Any, where: str) -> set[str]:
    if not isinstance(node, dict):
        raise ValueError(f"{where}: AST node must be a mapping")
    if node.get("variable"):
        variable = str(node["variable"])
        if not variable.startswith("crc."):
            raise ValueError(f"{where}: leaf {variable!r} is not a crc.* canonical variable")
        return {variable}
    op = str(node.get("op") or "")
    if op not in AST_OPS:
        raise ValueError(f"{where}: unsupported AST op {op!r}")
    if op == "constant":
        if not isinstance(node.get("value"), bool):
            raise ValueError(f"{where}: constant requires a Boolean value")
        return set()
    if op == "unresolved":
        if not str(node.get("reason_code") or "").strip():
            raise ValueError(f"{where}: unresolved requires reason_code")
        return set()
    if op == "not":
        if "operand" not in node:
            raise ValueError(f"{where}: not requires operand")
        return ast_leaves(node["operand"], f"{where}.operand")
    if op == "conditional":
        leaves: set[str] = set()
        for key in ("condition", "then", "else"):
            if key in node and node[key] is not None:
                leaves.update(ast_leaves(node[key], f"{where}.{key}"))
        if "condition" not in node or "then" not in node:
            raise ValueError(f"{where}: conditional requires condition and then")
        return leaves
    operands = as_list(node.get("operands"))
    leaves: set[str] = set()
    for index, operand in enumerate(operands):
        leaves.update(ast_leaves(operand, f"{where}.operands[{index}]"))
    return leaves


def normalize_variable(raw: dict[str, Any], existing: dict[str, Any] | None) -> dict[str, Any]:
    variable = {key: value for key, value in raw.items() if value is not None}
    if existing:
        # The reviewed seed inventory remains authoritative for an already-canonical ID.
        # Module authors may add definition/evidence notes, but cannot silently downgrade a
        # verified STORE mapping or change its domain/timepoint.
        for key in (
            "variable_id",
            "label",
            "datatype",
            "value_domain",
            "temporal_meaning",
            "mapping_level",
            "registry_mapping",
            "source_documents",
            "excluded_sources",
            "missingness_semantics",
            "conformance",
        ):
            if existing.get(key) is not None:
                variable[key] = existing[key]
        variable["seed_inventory_authority"] = True
    variable_id = str(variable.get("variable_id") or "")
    if not variable_id.startswith("crc."):
        raise ValueError(f"invalid variable_id {variable_id!r}")
    mapping_level = str(variable.get("mapping_level") or "chart_extension")
    if mapping_level.endswith("_candidate"):
        mapping_level = mapping_level.removesuffix("_candidate")
    registry_mapping = variable.get("registry_mapping") or {}
    if (
        mapping_level in {"registry_direct", "registry_coarsened"}
        and str(registry_mapping.get("standard") or "none") == "none"
        and str(registry_mapping.get("status") or "").startswith(
            ("no_verified", "not_verified")
        )
    ):
        mapping_level = "chart_extension"
        conformance = variable.setdefault("conformance", {})
        issues = conformance.setdefault("issues", [])
        issue = (
            "Registry-like label has no verified STORE/NAACCR mapping; treated as "
            "chart_extension until registrar review."
        )
        if issue not in issues:
            issues.append(issue)
        conformance["verdict"] = "needs_revision"
    if mapping_level not in MAPPING_LEVELS:
        raise ValueError(f"{variable_id}: invalid mapping_level {mapping_level!r}")
    variable.setdefault("label", variable_id.removeprefix("crc.").replace("_", " "))
    variable.setdefault("datatype", "string")
    variable.setdefault("value_domain", {"kind": "open_coded_text", "status": "candidate"})
    variable.setdefault("temporal_meaning", "indexed rule decision time; source-specific anchor retained")
    variable["mapping_level"] = mapping_level
    variable.setdefault("registry_mapping", {"standard": "none", "status": "no_verified_mapping"})
    variable.setdefault("source_documents", ["source-specific establishing document class pending review"])
    variable.setdefault("excluded_sources", ["problem-list carry-forward without indexed-event support"])
    variable.setdefault(
        "missingness_semantics",
        "Null means the reviewed evidence does not establish a value; it is not a negative clinical finding.",
    )
    variable.setdefault("suggested_execution_group", "CRC.FULL.chart_misc")
    variable.setdefault(
        "conformance",
        {
            "verdict": "needs_revision",
            "issues": ["full-universe generated contract requires clinical and evidence-owner review"],
        },
    )
    return variable


def state_for_variable(variable: dict[str, Any]) -> str:
    mapping_level = str(variable["mapping_level"])
    if mapping_level == "registry_direct":
        return (
            "full"
            if str(variable["variable_id"]) in {"crc.primary_site", "crc.histology"}
            else "partial"
        )
    return {
        "registry_coarsened": "partial",
        "derived": "none",
        "chart_extension": "none",
        "outside_current_sources": "none",
    }[mapping_level]


def combine(op: str, values: list[str]) -> str:
    if not values:
        return "full"
    if op == "any_of":
        if "full" in values:
            return "full"
        if "partial" in values:
            return "partial"
        return "none"
    if "none" in values:
        return "none"
    if "partial" in values:
        return "partial"
    return "full"


def evaluate_ast(node: dict[str, Any], variables: dict[str, dict[str, Any]]) -> str:
    if node.get("variable"):
        return state_for_variable(variables[str(node["variable"])])
    op = str(node["op"])
    if op == "constant":
        return "full" if node["value"] else "none"
    if op == "unresolved":
        return "none"
    if op == "not":
        return evaluate_ast(node["operand"], variables)
    if op == "conditional":
        return combine(
            "all_of",
            [
                evaluate_ast(node[key], variables)
                for key in ("condition", "then", "else")
                if node.get(key) is not None
            ],
        )
    return combine(op, [evaluate_ast(value, variables) for value in as_list(node.get("operands"))])


BLOCK_GAP_CODES = {
    "eligibility": "DENOMINATOR_VARIABLES_INCOMPLETE",
    "timing": "TEMPORAL_ANCHOR_MISSING",
    "exceptions": "EXCEPTION_MODEL_MISSING",
}


def make_empty_block_explicit(
    block: dict[str, Any],
    block_name: str,
    computability: dict[str, Any],
) -> None:
    """Replace ambiguous empty conjunctions with explicit authoring semantics."""
    node = block.get("evidence_logic")
    if (
        not isinstance(node, dict)
        or str(node.get("op") or "") not in {"all_of", "any_of"}
        or as_list(node.get("operands"))
        or as_list(block.get("variables"))
    ):
        return
    blocker = BLOCK_GAP_CODES.get(block_name)
    blockers = {str(value) for value in as_list(computability.get("blockers"))}
    if blocker and blocker in blockers:
        block["evidence_requirement_status"] = "unresolved"
        block["evidence_logic"] = {
            "op": "unresolved",
            "reason_code": blocker,
            "expression": block.get("expression"),
        }
        return
    block["evidence_requirement_status"] = "no_source_stated_evidence_requirement"
    block["evidence_logic"] = {
        "op": "constant",
        "value": True,
        "meaning": "no_source_stated_evidence_requirement",
    }


def replace_nested_empty_operators(node: Any) -> None:
    """Make nested no-requirement branches explicit after root-gap handling."""
    if not isinstance(node, dict) or node.get("variable"):
        return
    op = str(node.get("op") or "")
    if op in {"all_of", "any_of"}:
        operands = as_list(node.get("operands"))
        if not operands:
            node.clear()
            node.update(
                {
                    "op": "constant",
                    "value": True,
                    "meaning": "no_evidence_requirement_on_this_branch",
                }
            )
            return
        for operand in operands:
            replace_nested_empty_operators(operand)
    for key in ("operand", "condition", "then", "else"):
        if node.get(key) is not None:
            replace_nested_empty_operators(node[key])


def coverage_dimensions(variable: dict[str, Any]) -> dict[str, str]:
    """Conservative candidate mapping for one rule-variable use."""
    mapping_level = str(variable["mapping_level"])
    variable_id = str(variable["variable_id"])
    if mapping_level == "registry_direct":
        temporal = (
            "exact"
            if variable_id in {"crc.primary_site", "crc.histology"}
            else "conditional"
        )
        return {
            "semantic_support": "exact",
            "temporal_alignment": temporal,
            "provenance_suitability": "agreement_target",
            "canonical_to_registry_projection": "direct",
        }
    if mapping_level == "registry_coarsened":
        return {
            "semantic_support": "coarsened",
            "temporal_alignment": "conditional",
            "provenance_suitability": "weak_label",
            "canonical_to_registry_projection": "coarsened",
        }
    return {
        "semantic_support": "absent",
        "temporal_alignment": "unknown",
        "provenance_suitability": "not_truth",
        "canonical_to_registry_projection": "none",
    }


def build_contract(variable: dict[str, Any]) -> dict[str, Any]:
    variable_id = str(variable["variable_id"])
    field_name = safe_variable_suffix(variable_id)
    contract = {
        "contract_schema_version": "2.0",
        "contract_id": f"CRC.FULL.VAR.{field_name}",
        "contract_version": "0.1.0",
        "authoring_status": "candidate",
        "clinical_use": "NOT_FOR_CLINICAL_USE",
        "canonical_variable_id": variable_id,
        "label": variable["label"],
        "roles": variable["roles"],
        "datatype": variable["datatype"],
        "value_domain": variable["value_domain"],
        "temporal_meaning": variable["temporal_meaning"],
        "mapping_level": variable["mapping_level"],
        "registry_mapping": variable["registry_mapping"],
        "question": (
            f"What is the {variable['label']} for the indexed CRC tumor, test, treatment, "
            f"or decision at {variable['temporal_meaning']}?"
        ),
        "applicability_guard": {
            "disease": "colorectal_cancer",
            "rule_uses": sorted(
                {row["candidate_id"] for row in variable["rule_uses"]}
            ),
        },
        "field_contract": {
            "name": field_name,
            "type": variable["datatype"],
            "nullable": True,
            "value_domain": variable["value_domain"],
            "description": variable["label"],
        },
        "evidence_contract": {
            "establishing_source_documents": variable["source_documents"],
            "excluded_sources": variable["excluded_sources"],
            "positive_witness": (
                "A tumor- and time-linked source owned by the relevant clinical discipline "
                f"explicitly establishes {variable_id}."
            ),
            "negative_or_unknown_proof": {
                "mode": "candidate_review_required",
                "required_coverage": [
                    "Review every available establishing document class in the indexed rule window."
                ],
                "statement": variable["missingness_semantics"],
            },
            "conflict_policy": (
                "Retain conflicting citations; prefer the indexed primary source and timepoint. "
                "Abstain when source precedence does not resolve the conflict."
            ),
        },
        "proof_obligation": {
            "for_positive": (
                f"At least one tumor- and time-linked establishing source explicitly supports "
                f"{variable_id}; copied or unanchored mentions are insufficient."
            ),
            "for_negative_or_unknown": {
                "mode": "candidate_review_required",
                "required_document_classes": variable["source_documents"],
                "excluded_document_classes": variable["excluded_sources"],
                "runtime_enforcement": "not_materialized",
            },
        },
        "missingness_semantics": variable["missingness_semantics"],
        "abstention": {
            "EVIDENCE_INSUFFICIENT": "The available indexed evidence does not establish the variable.",
            "SPEC_INSUFFICIENT": "The candidate contract does not cover this evidence pattern or timepoint.",
        },
        "boundary_cases": [
            {
                "case_id": "wrong_tumor_or_timepoint",
                "input": (
                    f"{variable['label']} is documented only for another tumor, specimen, "
                    "line, or later decision."
                ),
                "expected": "Do not back-project; return EVIDENCE_INSUFFICIENT for the indexed use.",
            },
            {
                "case_id": "conflicting_establishing_sources",
                "input": "Two admissible indexed sources disagree and precedence does not resolve them.",
                "expected": "Retain both citations and abstain EVIDENCE_INSUFFICIENT.",
            },
            {
                "case_id": "mention_without_establishing_source",
                "input": "Only a copied problem-list or unanchored summary mention is available.",
                "expected": "Do not emit a positive value.",
            },
            {
                "case_id": "registry_projection_is_coarser",
                "input": "A registry value exists but omits a distinction required by this variable.",
                "expected": (
                    "Keep the registry value as a validation projection only; do not infer "
                    "the detailed canonical value."
                ),
            },
        ],
        "execution_binding": {
            "execution_group_id": variable["suggested_execution_group"],
            "output_field": field_name,
            "runtime_status": "planned_not_materialized",
        },
        "source_concepts": variable["source_concepts"],
        "conformance": variable["conformance"],
        "review_requirements": ["clinical_domain_reviewer", "registrar", "engineer"],
        "provenance": [
            {
                "element": element,
                "origin": "model_authored",
                "basis": (
                    "No external source directly specifies this executable element; generated "
                    "from the candidate rule/variable normalization and requires named review."
                ),
                "status": "review_pending",
            }
            for element in (
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
                "evidence_contract.establishing_source_documents",
                "evidence_contract.excluded_sources",
                "evidence_contract.positive_witness",
                "evidence_contract.negative_or_unknown_proof",
                "evidence_contract.conflict_policy",
                "proof_obligation",
                "missingness_semantics",
                "abstention",
                "boundary_cases",
                "execution_binding",
            )
        ],
    }
    contract["contract_hash"] = canonical_hash(contract)
    return contract


def build_projection_contract(variable: dict[str, Any]) -> dict[str, Any]:
    variable_id = str(variable["variable_id"])
    if variable_id not in COARSENED_PROJECTION_LOSSES:
        raise ValueError(f"{variable_id}: no explicit coarsened projection loss contract")
    registry_mapping = variable["registry_mapping"]
    projection = {
        "projection_schema_version": "1.0",
        "transformation_id": f"CRC.PROJECTION.{safe_variable_suffix(variable_id)}.v1",
        "transformation_version": "0.1.0",
        "clinical_use": "NOT_FOR_CLINICAL_USE",
        "canonical_variable_id": variable_id,
        "direction": "canonical_to_registry",
        "projection_kind": "coarsened",
        "input_domain": variable["value_domain"],
        "output_domain": {
            "standard": registry_mapping.get("standard"),
            "item_name": registry_mapping.get("item_name"),
            "item_number": registry_mapping.get("item_number"),
            "xml_id": registry_mapping.get("xml_id"),
            "code_domain_status": "registrar_crosswalk_review_pending",
        },
        "effective_years": registry_mapping.get("effective_years"),
        "transformation": {
            "operation": "versioned_value_crosswalk",
            "implementation_status": "planned_not_materialized",
            "unknown_handling": (
                "Preserve documented unknown/not-applicable distinctions where the target "
                "standard supports them; otherwise abstain from agreement scoring."
            ),
        },
        "explicit_loss_list": COARSENED_PROJECTION_LOSSES[variable_id],
        "review_status": {
            "clinical": "pending",
            "registrar": "pending",
            "runtime": "pending",
        },
        "provenance": {
            "origin": "model_authored",
            "basis": (
                "No external source directly specifies this canonical-to-registry crosswalk; "
                "the target item is registered, but value-level transformation requires "
                "registrar review."
            ),
            "status": "review_pending",
        },
    }
    projection["transformation_hash"] = canonical_hash(projection)
    return projection


def build(bundle: Path, *, replace: bool) -> dict[str, Any]:
    universe = load_yaml(bundle / "intake" / "rule_universe.yaml")
    source_denominator = load_yaml(bundle / "intake" / "source_denominator.yaml")
    source_register = load_yaml(bundle / "source_register.yaml")
    seed_inventory = load_yaml(bundle / "variable_inventory.yaml")
    seed_variables = {
        str(row["variable_id"]): row for row in as_list(seed_inventory.get("variables"))
    }
    universe_rows = as_list(universe.get("candidates"))
    universe_index = {str(row["candidate_id"]): row for row in universe_rows}
    registered_source_ids = {
        str(row.get("source_id"))
        for row in as_list(source_register.get("sources"))
    }
    denominator_sources = {
        str(row.get("source_id")): row
        for row in as_list(source_denominator.get("sources"))
    }
    unresolved_source_slices = sorted(
        {
            str(row.get("source_id"))
            for row in universe_rows
            if str(row.get("source_id")) not in denominator_sources
            or str(
                denominator_sources.get(str(row.get("source_id")), {}).get(
                    "register_source_id"
                )
            )
            not in registered_source_ids
        }
    )
    if unresolved_source_slices:
        raise ValueError(
            "rule-universe source slices do not resolve through source_denominator to "
            f"source_register: {unresolved_source_slices}"
        )
    module_dir = bundle / "normalization" / "modules"
    module_paths = sorted(module_dir.glob("*.yaml"))
    if not module_paths:
        raise ValueError(f"{module_dir}: no module YAML files")
    alias_path = bundle / "normalization" / "canonical_aliases.yaml"
    alias_doc = load_yaml(alias_path) if alias_path.is_file() else {}
    aliases = {
        str(row.get("alias")): str(row.get("canonical"))
        for row in as_list(alias_doc.get("aliases"))
    }
    for alias, canonical in aliases.items():
        if not alias.startswith("crc.") or not canonical.startswith("crc.") or alias == canonical:
            raise ValueError(f"{alias_path}: invalid alias {alias!r} -> {canonical!r}")
        resolve_alias(alias, aliases)

    normalized_by_id: dict[str, dict[str, Any]] = {}
    definitions: dict[str, dict[str, Any]] = {}
    definition_sources: dict[str, list[str]] = defaultdict(list)
    bindings_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    module_counts: dict[str, int] = {}

    for path in module_paths:
        module = load_yaml(path)
        module_id = str(module.get("module_id") or path.stem)
        module_counts[module_id] = len(as_list(module.get("candidates")))
        module_defs: dict[str, dict[str, Any]] = {}
        for row in as_list(module.get("canonical_variables")):
            raw_id = str(row.get("variable_id") or "")
            variable_id = resolve_alias(raw_id, aliases)
            normalized_row = dict(row)
            normalized_row["variable_id"] = variable_id
            module_defs[variable_id] = normalized_row
        for variable_id, raw in module_defs.items():
            candidate_definition = normalize_variable(raw, seed_variables.get(variable_id))
            if variable_id in definitions:
                left = {
                    key: definitions[variable_id].get(key)
                    for key in ("datatype", "value_domain", "temporal_meaning", "mapping_level")
                }
                right = {
                    key: candidate_definition.get(key)
                    for key in ("datatype", "value_domain", "temporal_meaning", "mapping_level")
                }
                if left != right:
                    conflict = {
                        "module_id": module_id,
                        "existing": left,
                        "incoming": right,
                    }
                    definitions[variable_id].setdefault(
                        "normalization_conflicts", []
                    ).append(conflict)
                    issues = (
                        (definitions[variable_id].setdefault("conformance", {}))
                        .setdefault("issues", [])
                    )
                    issue = (
                        "Cross-module variable-definition conflict requires clinical "
                        f"canonicalization review: {module_id}"
                    )
                    if issue not in issues:
                        issues.append(issue)
                    definitions[variable_id]["conformance"]["verdict"] = "needs_revision"
            else:
                definitions[variable_id] = candidate_definition
            definition_sources[variable_id].append(module_id)

        for candidate in as_list(module.get("candidates")):
            candidate_id = str(candidate.get("candidate_id") or "")
            if candidate_id not in universe_index:
                raise ValueError(f"{path}: unknown candidate_id {candidate_id!r}")
            if candidate_id in normalized_by_id:
                raise ValueError(f"{candidate_id}: assigned to multiple normalization modules")
            source = universe_index[candidate_id]
            requirements = candidate.get("requirements") or {}
            computability = candidate.get(
                "computability",
                {"status": "partially_specified", "blockers": source.get("blockers", [])},
            )
            required_variables: set[str] = set()
            for block_name in BLOCKS:
                block = requirements.get(block_name)
                if not isinstance(block, dict):
                    raise ValueError(f"{candidate_id}.{block_name}: missing requirement block")
                block["variables"] = sorted(
                    {
                        resolve_alias(str(value), aliases)
                        for value in as_list(block.get("variables"))
                    }
                )
                make_empty_block_explicit(block, block_name, computability)
                replace_nested_empty_operators(block.get("evidence_logic"))
                rewrite_ast_aliases(block.get("evidence_logic"), aliases)
                variables = {str(value) for value in as_list(block.get("variables"))}
                leaves = ast_leaves(
                    block.get("evidence_logic"),
                    f"{candidate_id}.{block_name}.evidence_logic",
                )
                if variables != leaves:
                    raise ValueError(
                        f"{candidate_id}.{block_name}: variables != AST leaves; "
                        f"missing={sorted(variables-leaves)}, extra={sorted(leaves-variables)}"
                    )
                required_variables.update(variables)
            normalized_by_id[candidate_id] = {
                "candidate_id": candidate_id,
                "parent_recommendation_id": source.get("parent_recommendation_id"),
                "related_parent_recommendation_ids": source.get(
                    "related_parent_recommendation_ids", []
                ),
                "candidate_kind": source.get("candidate_kind"),
                "source_id": source.get("source_id"),
                "source_anchor": source.get("source_anchor"),
                "category": source.get("category"),
                "title": source.get("title"),
                "core_or_supplemental": source.get("core_or_supplemental"),
                "source_context": {
                    key: source.get(key)
                    for key in ("eligibility", "action", "timing", "exceptions", "predicate_logic")
                },
                "requirements": requirements,
                "required_variables": sorted(required_variables),
                "computability": computability,
                "review_status": "pending",
                "clinical_use": "NOT_FOR_CLINICAL_USE",
            }

        for binding in as_list(module.get("concept_bindings")):
            candidate_id = str(binding.get("candidate_id") or "")
            if candidate_id not in universe_index:
                raise ValueError(f"{path}: binding names unknown candidate {candidate_id!r}")
            source_concept = str(
                binding.get("source_concept")
                or binding.get("critical_variable_concept")
                or ""
            )
            if not source_concept:
                raise ValueError(f"{path}: empty source concept binding")
            variable_ids = [
                resolve_alias(str(value), aliases)
                for value in as_list(
                    binding.get("canonical_variable_ids") or binding.get("variable_ids")
                )
            ]
            if not variable_ids:
                raise ValueError(f"{candidate_id}/{source_concept}: binding has no variables")
            bindings_by_candidate[candidate_id].append(
                {
                    "candidate_id": candidate_id,
                    "source_concept": source_concept,
                    "canonical_variable_ids": variable_ids,
                    "relation": binding.get("relation") or "candidate",
                    "rationale": binding.get("rationale")
                    or "Module-authored clinical-semantic binding; human review pending.",
                }
            )

    expected_ids = set(universe_index)
    actual_ids = set(normalized_by_id)
    if actual_ids != expected_ids:
        raise ValueError(
            f"module candidate coverage mismatch; missing={sorted(expected_ids-actual_ids)}, "
            f"extra={sorted(actual_ids-expected_ids)}"
        )

    all_variable_uses: dict[str, list[dict[str, str]]] = defaultdict(list)
    source_concepts_by_variable: dict[str, set[str]] = defaultdict(set)
    for candidate_id, source in universe_index.items():
        expected_concepts = {str(value) for value in as_list(source.get("critical_variable_concepts"))}
        bindings = bindings_by_candidate.get(candidate_id, [])
        bound_concepts = {row["source_concept"] for row in bindings}
        if bound_concepts != expected_concepts:
            raise ValueError(
                f"{candidate_id}: concept bindings mismatch; "
                f"missing={sorted(expected_concepts-bound_concepts)}, "
                f"extra={sorted(bound_concepts-expected_concepts)}"
            )
        rule = normalized_by_id[candidate_id]
        rule_variables = set(rule["required_variables"])
        bound_variables = {
            variable_id
            for binding in bindings
            for variable_id in binding["canonical_variable_ids"]
        }
        if not bound_variables <= rule_variables:
            raise ValueError(
                f"{candidate_id}: concept bindings reference variables absent from rule AST: "
                f"{sorted(bound_variables-rule_variables)}"
            )
        for variable_id in rule_variables:
            if variable_id not in definitions and variable_id not in seed_variables:
                raise ValueError(f"{candidate_id}: no definition for {variable_id}")
            for block_name in BLOCKS:
                if variable_id in as_list(rule["requirements"][block_name].get("variables")):
                    all_variable_uses[variable_id].append(
                        {"candidate_id": candidate_id, "requirement_role": block_name}
                    )
        for binding in bindings:
            for variable_id in binding["canonical_variable_ids"]:
                source_concepts_by_variable[variable_id].add(binding["source_concept"])

    used_variables = set(all_variable_uses)
    canonical_variables: dict[str, dict[str, Any]] = {}
    for variable_id in sorted(used_variables):
        variable = normalize_variable(
            definitions.get(variable_id, {}),
            seed_variables.get(variable_id),
        )
        variable["roles"] = sorted(
            {row["requirement_role"] for row in all_variable_uses[variable_id]}
        )
        variable["rule_uses"] = sorted(
            all_variable_uses[variable_id],
            key=lambda row: (row["candidate_id"], row["requirement_role"]),
        )
        variable["source_concepts"] = sorted(source_concepts_by_variable[variable_id])
        variable["definition_sources"] = sorted(definition_sources.get(variable_id, []))
        canonical_variables[variable_id] = variable

    out_root = bundle / "normalization"
    contract_dir = out_root / "variable_contracts"
    if contract_dir.exists():
        if not replace:
            raise ValueError(f"{contract_dir} exists; pass --replace")
        shutil.rmtree(contract_dir)
    contract_dir.mkdir(parents=True, exist_ok=True)
    contracts: list[dict[str, Any]] = []
    for variable in canonical_variables.values():
        contract = build_contract(variable)
        contracts.append(contract)
        path = contract_dir / f"{contract['contract_id']}.yaml"
        path.write_text(
            yaml.safe_dump(contract, sort_keys=False, allow_unicode=True, width=120),
            encoding="utf-8",
        )

    rules = [normalized_by_id[str(row["candidate_id"])] for row in universe_rows]
    variable_rows = list(canonical_variables.values())
    coarsened_variable_ids = {
        str(variable["variable_id"])
        for variable in variable_rows
        if variable["mapping_level"] == "registry_coarsened"
    }
    if coarsened_variable_ids != set(COARSENED_PROJECTION_LOSSES):
        raise ValueError(
            "coarsened projection contracts differ from canonical inventory: "
            f"missing={sorted(coarsened_variable_ids-set(COARSENED_PROJECTION_LOSSES))}, "
            f"extra={sorted(set(COARSENED_PROJECTION_LOSSES)-coarsened_variable_ids)}"
        )
    projection_contracts = [
        build_projection_contract(canonical_variables[variable_id])
        for variable_id in sorted(coarsened_variable_ids)
    ]
    projection_by_variable = {
        str(row["canonical_variable_id"]): row
        for row in projection_contracts
    }
    category_dir = out_root / "rules_by_category"
    if category_dir.exists():
        if not replace:
            raise ValueError(f"{category_dir} exists; pass --replace")
        shutil.rmtree(category_dir)
    category_dir.mkdir(parents=True, exist_ok=True)
    rules_by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rule in rules:
        rules_by_category[str(rule["category"])].append(rule)
    for category, category_rules in sorted(rules_by_category.items()):
        (category_dir / f"{safe_name(category)}.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": "2.0",
                    "category": category,
                    "clinical_use": "NOT_FOR_CLINICAL_USE",
                    "candidate_count": len(category_rules),
                    "rules": category_rules,
                },
                sort_keys=False,
                allow_unicode=True,
                width=120,
            ),
            encoding="utf-8",
        )
    groups: dict[str, list[str]] = defaultdict(list)
    for variable in variable_rows:
        groups[str(variable["suggested_execution_group"])].append(str(variable["variable_id"]))

    coverage_rows: list[dict[str, Any]] = []
    for rule in rules:
        block_states = {
            block: evaluate_ast(rule["requirements"][block]["evidence_logic"], canonical_variables)
            for block in BLOCKS
        }
        coverage_rows.append(
            {
                "candidate_id": rule["candidate_id"],
                "category": rule["category"],
                "blocks": block_states,
                "denominator_evaluability": block_states["eligibility"],
                "concordance_evaluability": combine(
                    "all_of",
                    [block_states[name] for name in ("eligibility", "action", "timing")],
                ),
                "nonconcordance_defensibility": combine(
                    "all_of", [block_states[name] for name in BLOCKS]
                ),
                "observed_crc_availability": "NOT_ASSESSED",
            }
        )
    coverage_counts = Counter(
        row["nonconcordance_defensibility"] for row in coverage_rows
    )
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in coverage_rows:
        category_counts[str(row["category"])][str(row["nonconcordance_defensibility"])] += 1

    variable_use_rows: list[dict[str, Any]] = []
    category_use_counts: dict[str, dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    for rule in rules:
        category = str(rule["category"])
        for block_name in BLOCKS:
            for variable_id in as_list(
                rule["requirements"][block_name].get("variables")
            ):
                variable = canonical_variables[str(variable_id)]
                dimensions = coverage_dimensions(variable)
                row = {
                    "candidate_id": rule["candidate_id"],
                    "category": category,
                    "predicate_path": f"requirements.{block_name}.evidence_logic",
                    "requirement_role": block_name,
                    "variable_id": variable_id,
                    **dimensions,
                    "mapping_review_status": "candidate_registrar_review_pending",
                    "registry_mapping": variable["registry_mapping"],
                }
                if variable["mapping_level"] == "registry_coarsened":
                    row["transformation_id"] = projection_by_variable[
                        str(variable_id)
                    ]["transformation_id"]
                variable_use_rows.append(row)
                for dimension, value in dimensions.items():
                    category_use_counts[category][dimension][value] += 1

    gap_rows: list[dict[str, Any]] = []
    for variable in variable_rows:
        mapping_level = str(variable["mapping_level"])
        registry_coverage = {
            "registry_direct": "direct",
            "registry_coarsened": "coarsened",
            "derived": "derived",
            "chart_extension": "none",
            "outside_current_sources": "none",
        }[mapping_level]
        gap_classes = ["RUNTIME_MATERIALIZATION_PENDING", "DATA_NOT_PROFILED", "CLINICAL_REVIEW_PENDING"]
        if mapping_level == "registry_coarsened":
            gap_classes.append("REGISTRY_TOO_COARSE")
        elif mapping_level in {"chart_extension", "outside_current_sources"}:
            gap_classes.append("NO_REGISTRY_FIELD")
        if mapping_level == "outside_current_sources":
            gap_classes.append("SOURCE_OR_EVIDENCE_OWNER_UNRESOLVED")
        if registry_coverage in {"direct", "coarsened"}:
            gap_classes.append("REGISTRAR_REVIEW_PENDING")
        gap_rows.append(
            {
                "variable_id": variable["variable_id"],
                "rule_use_count": len(variable["rule_uses"]),
                "rule_uses": variable["rule_uses"],
                "mapping_level": mapping_level,
                "registry_coverage": registry_coverage,
                "authoring_contract": "present_candidate",
                "runtime_materialization": "planned_not_materialized",
                "data_coverage": "NOT_ASSESSED",
                "gap_classes": gap_classes,
                "severity": (
                    "blocking"
                    if mapping_level in {"chart_extension", "outside_current_sources"}
                    else "major"
                ),
            }
        )
    gap_class_counts = Counter(
        gap_class for row in gap_rows for gap_class in row["gap_classes"]
    )
    top_unlocks = sorted(
        (
            {
                "variable_id": row["variable_id"],
                "rule_use_count": row["rule_use_count"],
                "mapping_level": row["mapping_level"],
            }
            for row in gap_rows
        ),
        key=lambda row: (-int(row["rule_use_count"]), str(row["variable_id"])),
    )[:50]

    documents = {
        out_root / "normalized_rules.yaml": {
            "schema_version": "2.0",
            "bundle_id": "CRC.CORE.FULL_NORMALIZATION.v1",
            "clinical_use": "NOT_FOR_CLINICAL_USE",
            "candidate_count": len(rules),
            "ast_normalized_count": len(rules),
            "rules": rules,
        },
        out_root / "canonical_variables.yaml": {
            "schema_version": "2.0",
            "bundle_id": "CRC.CORE.FULL_NORMALIZATION.v1",
            "clinical_use": "NOT_FOR_CLINICAL_USE",
            "canonical_variable_count": len(variable_rows),
            "variables": variable_rows,
        },
        out_root / "concept_bindings.yaml": {
            "schema_version": "2.0",
            "binding_scope": "candidate_scoped",
            "bindings": [
                row
                for candidate in universe_rows
                for row in bindings_by_candidate[str(candidate["candidate_id"])]
            ],
        },
        out_root / "registry_projections.yaml": {
            "projection_schema_version": "1.0",
            "clinical_use": "NOT_FOR_CLINICAL_USE",
            "projection_contract_count": len(projection_contracts),
            "projections": projection_contracts,
        },
        out_root / "execution_group_plan.yaml": {
            "schema_version": "1.0",
            "status": "planned_not_materialized",
            "group_count": len(groups),
            "groups": [
                {
                    "execution_group_id": group_id,
                    "canonical_variable_ids": sorted(variable_ids),
                    "review_status": "compatibility_review_pending",
                }
                for group_id, variable_ids in sorted(groups.items())
            ],
        },
        out_root / "evidence_coverage.yaml": {
            "coverage_schema_version": "2.0",
            "clinical_use": "NOT_FOR_CLINICAL_USE",
            "logic_status": "normalized",
            "assessment_status": "CANDIDATE_STRUCTURAL_MAPPING_REVIEW_PENDING",
            "summary": {
                "executable_candidates": len(rules),
                "canonical_variables": len(variable_rows),
                "variable_contracts": len(contracts),
                "coarsened_projection_contracts": len(projection_contracts),
                "variable_use_rows": len(variable_use_rows),
                "mapping_level_counts": dict(
                    sorted(Counter(str(row["mapping_level"]) for row in variable_rows).items())
                ),
                "registry_only_nonconcordance_defensibility": {
                    state: coverage_counts[state] for state in ("full", "partial", "none")
                },
                "observed_crc_availability": "NOT_ASSESSED",
            },
            "by_category": {
                category: {
                    **{
                        state: counts[state]
                        for state in ("full", "partial", "none")
                    },
                    "variable_use_dimensions": {
                        dimension: dict(sorted(values.items()))
                        for dimension, values in sorted(
                            category_use_counts[category].items()
                        )
                    },
                }
                for category, counts in sorted(category_counts.items())
            },
            "variable_uses": variable_use_rows,
            "rules": coverage_rows,
        },
        out_root / "gap_assessment.yaml": {
            "schema_version": "2.0",
            "clinical_use": "NOT_FOR_CLINICAL_USE",
            "guideline_denominator": {
                "executable_candidates": len(rules),
                "ast_normalized": len(rules),
            },
            "summary": {
                "canonical_variables": len(variable_rows),
                "variable_contracts": len(contracts),
                "runtime_materialized_variables": 0,
                "data_profile_status": "NOT_ASSESSED",
                "mapping_level_counts": dict(
                    sorted(Counter(str(row["mapping_level"]) for row in variable_rows).items())
                ),
                "gap_class_counts": dict(sorted(gap_class_counts.items())),
            },
            "top_variable_unlocks": top_unlocks,
            "variable_gaps": gap_rows,
        },
        out_root / "manifest.yaml": {
            "schema_version": "1.0",
            "bundle_id": "CRC.CORE.FULL_NORMALIZATION.v1",
            "clinical_use": "NOT_FOR_CLINICAL_USE",
            "source_candidate_count": len(universe_rows),
            "normalized_rule_count": len(rules),
            "distinct_source_concept_count": len(
                {
                    str(concept)
                    for candidate in universe_rows
                    for concept in as_list(candidate.get("critical_variable_concepts"))
                }
            ),
            "candidate_scoped_binding_status": "complete",
            "source_slice_crosswalk_status": "complete",
            "source_document_hash_status": "pending",
            "source_register_documents_without_hash": sum(
                str(row.get("document_hash") or "") == "NOT_CAPTURED"
                for row in as_list(source_register.get("sources"))
            ),
            "evidence_block_root_counts": dict(
                sorted(
                    Counter(
                        (
                            "variable"
                            if rule["requirements"][block].get("evidence_logic", {}).get(
                                "variable"
                            )
                            else str(
                                rule["requirements"][block]
                                .get("evidence_logic", {})
                                .get("op")
                            )
                        )
                        for rule in rules
                        for block in BLOCKS
                    ).items()
                )
            ),
            "computability_status_counts": dict(
                sorted(
                    Counter(
                        str((rule.get("computability") or {}).get("status"))
                        for rule in rules
                    ).items()
                )
            ),
            "canonical_variable_count": len(variable_rows),
            "variable_contract_count": len(contracts),
            "registry_projection_contract_count": len(projection_contracts),
            "execution_group_plan_count": len(groups),
            "category_file_count": len(rules_by_category),
            "module_counts": module_counts,
            "canonical_alias_count": len(aliases),
            "contract_set_hash": canonical_hash(
                [(row["contract_id"], row["contract_hash"]) for row in contracts]
            ),
            "review_status": {
                "clinical": "pending",
                "registrar": "pending",
                "runtime_materialization": "pending",
            },
        },
    }
    for path, document in documents.items():
        path.write_text(
            yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=120),
            encoding="utf-8",
        )

    mapping_counts = Counter(str(row["mapping_level"]) for row in variable_rows)
    gap_lines = [
        "# CRC full-normalization gap assessment",
        "",
        "> Candidate authoring output — **NOT FOR CLINICAL USE**.",
        "",
        "## Blocking result",
        "",
        f"- Evidence-AST normalized rules: **{len(rules)}/{len(universe_rows)}**.",
        f"- Explicit unresolved requirement blocks: **"
        f"{sum(rule['requirements'][block]['evidence_logic'].get('op') == 'unresolved' for rule in rules for block in BLOCKS)}**; "
        "no empty conjunction is published.",
        f"- Canonical variables and one-variable contracts: **{len(variable_rows)}/{len(contracts)}**.",
        f"- Versioned coarsened registry projection contracts: **{len(projection_contracts)}/{len(coarsened_variable_ids)}**.",
        "- Every variable contract has boundary cases and per-enforced-element candidate provenance.",
        f"- Source slice crosswalk: **{len(rules)}/{len(rules)} resolved**; "
        f"**{sum(str(row.get('document_hash') or '') == 'NOT_CAPTURED' for row in as_list(source_register.get('sources')))}** "
        "registered source documents still lack an immutable hash.",
        "- Runtime-materialized full-universe variables: **0**; execution groups remain plans pending compatibility review.",
        "- Observed CRC data coverage: **NOT_ASSESSED**.",
        "",
        "## STORE-centered mapping",
        "",
        "| mapping level | variables |",
        "|---|---:|",
    ]
    for level in sorted(MAPPING_LEVELS):
        gap_lines.append(f"| {level} | {mapping_counts[level]} |")
    gap_lines += [
        "",
        "## Highest-use variables to unlock",
        "",
        "| variable | rule-block uses | mapping |",
        "|---|---:|---|",
    ]
    for row in top_unlocks[:30]:
        gap_lines.append(
            f"| {row['variable_id']} | {row['rule_use_count']} | {row['mapping_level']} |"
        )
    gap_lines += [
        "",
        "Every row has an authoring contract. That does not imply a reviewed native runtime "
        "spec, a validated evidence proof, or observed availability in CRC linked data.",
        "",
    ]
    (out_root / "gap_assessment.md").write_text("\n".join(gap_lines), encoding="utf-8")

    rule_category_by_id = {
        str(rule["candidate_id"]): str(rule["category"]) for rule in rules
    }

    def append_variable_table(
        lines: list[str],
        *,
        title: str,
        rows: list[dict[str, Any]],
        explanation: str,
    ) -> None:
        lines += [
            f"## {title}",
            "",
            explanation,
            "",
            "| variable | rule-block uses | roles | guideline categories | example rules |",
            "|---|---:|---|---|---|",
        ]
        for row in sorted(
            rows,
            key=lambda item: (-int(item["rule_use_count"]), str(item["variable_id"])),
        ):
            rule_uses = as_list(row.get("rule_uses"))
            roles = ", ".join(
                sorted({str(use["requirement_role"]) for use in rule_uses})
            )
            categories = ", ".join(
                sorted(
                    {
                        rule_category_by_id[str(use["candidate_id"])]
                        for use in rule_uses
                    }
                )
            )
            candidate_ids = sorted(
                {str(use["candidate_id"]) for use in rule_uses}
            )
            examples = ", ".join(candidate_ids[:6])
            if len(candidate_ids) > 6:
                examples += f", +{len(candidate_ids) - 6} more"
            lines.append(
                f"| `{row['variable_id']}` | {row['rule_use_count']} | "
                f"{roles} | {categories} | {examples} |"
            )
        lines.append("")

    registry_gap_lines = [
        "# CRC guideline variables versus cancer registry",
        "",
        "> Candidate authoring inventory — **NOT FOR CLINICAL USE**.",
        "",
        "## Summary",
        "",
        "| registry relationship | variables | interpretation |",
        "|---|---:|---|",
        f"| no registry field: linked-chart extension | {mapping_counts['chart_extension']} | "
        "The guideline fact must be extracted from linked clinical documents or events. |",
        f"| no registry field: outside current sources | {mapping_counts['outside_current_sources']} | "
        "The fact needs a new external source, policy, label, calendar, or evidence owner. |",
        f"| derived, not stored directly | {mapping_counts['derived']} | "
        "The fact must be computed from reviewed inputs. |",
        f"| registry field exists but is coarsened | {mapping_counts['registry_coarsened']} | "
        "The registry can support a lossy validation projection, not the canonical rule fact. |",
        f"| registry-direct candidate mapping | {mapping_counts['registry_direct']} | "
        "The canonical fact has a candidate direct NAACCR mapping, still pending review. |",
        "",
        f"Therefore **{mapping_counts['chart_extension'] + mapping_counts['outside_current_sources']}** "
        "canonical variables have `NO_REGISTRY_FIELD`, and "
        f"**{mapping_counts['chart_extension'] + mapping_counts['outside_current_sources'] + mapping_counts['derived']}** "
        "are not stored directly in the registry.",
        "",
    ]
    append_variable_table(
        registry_gap_lines,
        title=f"Missing registry fields — linked-chart extensions ({mapping_counts['chart_extension']})",
        rows=[
            row for row in gap_rows if row["mapping_level"] == "chart_extension"
        ],
        explanation=(
            "These are required guideline facts with no accepted registry field. "
            "They should be added to the STORE-centered linked-chart feature layer, "
            "not silently inferred from registry summaries."
        ),
    )
    append_variable_table(
        registry_gap_lines,
        title=(
            "Missing registry fields — outside current sources "
            f"({mapping_counts['outside_current_sources']})"
        ),
        rows=[
            row
            for row in gap_rows
            if row["mapping_level"] == "outside_current_sources"
        ],
        explanation=(
            "These variables cannot currently be established from the registered "
            "registry/chart source set and need an explicit new source or owner."
        ),
    )
    append_variable_table(
        registry_gap_lines,
        title=f"Derived variables not stored directly ({mapping_counts['derived']})",
        rows=[row for row in gap_rows if row["mapping_level"] == "derived"],
        explanation=(
            "These variables need a reviewed derivation and input provenance. "
            "They do not necessarily require a new raw data field."
        ),
    )

    registry_gap_lines += [
        f"## Registry fields with scope mismatch ({mapping_counts['registry_coarsened']})",
        "",
        "These variables have a candidate registry projection, but information is lost. "
        "They cannot be treated as exact rule evidence.",
        "",
        "| variable | rule-block uses | principal loss |",
        "|---|---:|---|",
    ]
    for row in sorted(
        (
            row
            for row in gap_rows
            if row["mapping_level"] == "registry_coarsened"
        ),
        key=lambda item: (-int(item["rule_use_count"]), str(item["variable_id"])),
    ):
        losses = COARSENED_PROJECTION_LOSSES[str(row["variable_id"])]
        registry_gap_lines.append(
            f"| `{row['variable_id']}` | {row['rule_use_count']} | {losses[0]} |"
        )
    registry_gap_lines += [
        "",
    ]
    append_variable_table(
        registry_gap_lines,
        title=f"Registry-direct candidate mappings ({mapping_counts['registry_direct']})",
        rows=[
            row for row in gap_rows if row["mapping_level"] == "registry_direct"
        ],
        explanation=(
            "These are the only candidate exact canonical-to-registry mappings. "
            "They remain subject to registrar, clinical, temporal-scope, and data-profile review."
        ),
    )
    (out_root / "registry_variable_gap_inventory.md").write_text(
        "\n".join(registry_gap_lines),
        encoding="utf-8",
    )

    summary = {
        "rules": len(rules),
        "variables": len(variable_rows),
        "contracts": len(contracts),
        "groups": len(groups),
        "coverage": {state: coverage_counts[state] for state in ("full", "partial", "none")},
    }
    (out_root / "SUMMARY.md").write_text(
        "\n".join(
            [
                "# CRC full normalization",
                "",
                "> Candidate authoring output — **NOT FOR CLINICAL USE**.",
                "",
                f"- Normalized rules with evidence AST: **{summary['rules']}**.",
                f"- Computability disposition: **"
                f"{sum((rule.get('computability') or {}).get('status') == 'partially_specified' for rule in rules)} "
                f"partially specified / "
                f"{sum((rule.get('computability') or {}).get('status') == 'not_computable' for rule in rules)} "
                f"not computable**.",
                f"- Empty requirement semantics: **"
                f"{sum(rule['requirements'][block]['evidence_logic'].get('op') == 'unresolved' for rule in rules for block in BLOCKS)} "
                f"explicit unresolved / "
                f"{sum(rule['requirements'][block]['evidence_logic'].get('op') == 'constant' for rule in rules for block in BLOCKS)} "
                f"explicit no-source-requirement constants**; no empty conjunction is published.",
                f"- Rule categories: **{len(rules_by_category)}**.",
                f"- Distinct source concepts: **{len({str(concept) for candidate in universe_rows for concept in as_list(candidate.get('critical_variable_concepts'))})}**; every candidate-scoped use has an explicit binding.",
                f"- Canonical variables: **{summary['variables']}**.",
                f"- One-variable contracts: **{summary['contracts']}**.",
                f"- Versioned coarsened registry projection contracts: **{len(projection_contracts)}**.",
                f"- Planned grouped execution passes: **{summary['groups']}**.",
                f"- STORE-only structural rule coverage/nonconcordance defensibility: "
                f"**{summary['coverage']['full']} full / {summary['coverage']['partial']} partial / "
                f"{summary['coverage']['none']} none**.",
                "- Observed CRC availability: **NOT_ASSESSED**.",
                "- Clinical, registrar, and runtime-materialization review remain pending.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    summary = build(args.bundle.resolve(), replace=args.replace)
    print(
        f"normalized {summary['rules']} rules, {summary['variables']} variables, "
        f"{summary['contracts']} contracts, {summary['groups']} group plans; "
        f"coverage={summary['coverage']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
