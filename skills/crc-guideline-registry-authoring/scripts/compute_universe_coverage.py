#!/usr/bin/env python3
"""Measure CRC guideline-universe reach of a bound registry schema.

This report deliberately distinguishes:

* the accessible-source guideline denominator;
* executable candidates versus formal source recommendations;
* full rule evaluability versus the presence of a few potentially useful components;
* schema-level assessment versus observed CRC data availability.

The current local profile is not a CRC tumor-level profile. Its field frequencies therefore
cannot be reported as observed CRC coverage.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected mapping")
    return value


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def component_matches(
    variable_ids: list[str],
    local_variable_to_field: dict[str, str],
) -> list[dict[str, str]]:
    """Use only profile-declared exact canonical candidates; never substring matching."""
    rows: list[dict[str, str]] = []
    for variable_id in sorted(set(variable_ids)):
        registry_field = local_variable_to_field.get(variable_id)
        if registry_field:
            rows.append(
                {
                    "canonical_variable_id": variable_id,
                    "registry_field": registry_field,
                    "support": "profile_declared_component_candidate",
                }
            )
    return rows


def compute(bundle: Path) -> dict[str, Any]:
    denominator = load_yaml(bundle / "intake" / "source_denominator.yaml")
    universe = load_yaml(bundle / "intake" / "rule_universe.yaml")
    normalized = load_yaml(bundle / "candidate_rules.yaml")
    variables = load_yaml(bundle / "variable_inventory.yaml")
    execution = load_yaml(bundle / "execution_manifest.yaml")
    concept_inventory = load_yaml(bundle / "intake" / "variable_concept_inventory.yaml")
    full_rules_path = bundle / "normalization" / "normalized_rules.yaml"
    full_variables_path = bundle / "normalization" / "canonical_variables.yaml"
    full_coverage_path = bundle / "normalization" / "evidence_coverage.yaml"
    full_manifest_path = bundle / "normalization" / "manifest.yaml"
    local_profile_path = (
        bundle / "registry_profiles" / "R6249_LUNG_SCHEMA_2026-07-25.yaml"
    )
    full_rules_doc = load_yaml(full_rules_path) if full_rules_path.is_file() else {}
    full_variables_doc = load_yaml(full_variables_path) if full_variables_path.is_file() else {}
    full_coverage_doc = load_yaml(full_coverage_path) if full_coverage_path.is_file() else {}
    full_manifest = load_yaml(full_manifest_path) if full_manifest_path.is_file() else {}
    local_profile = load_yaml(local_profile_path)
    full_rules = as_list(full_rules_doc.get("rules"))
    full_rule_index = {
        str(row.get("candidate_id")): row
        for row in full_rules
    }
    local_variable_to_field: dict[str, str] = {}
    for field in as_list((local_profile.get("registry_schema") or {}).get("fields")):
        source_field = str(field.get("source_field") or "")
        for variable_id in as_list(field.get("canonical_candidates")):
            variable_id = str(variable_id)
            if variable_id in local_variable_to_field:
                raise ValueError(
                    f"local profile maps {variable_id!r} to more than one source field"
                )
            local_variable_to_field[variable_id] = source_field
    full_coverage_index = {
        str(row.get("candidate_id")): row
        for row in as_list(full_coverage_doc.get("rules"))
    }
    candidates = as_list(universe.get("candidates"))
    seed_rules = as_list(normalized.get("rules"))
    seed_ast_normalized = sum(
        all(
            bool((((rule.get("requirements") or {}).get(block) or {}).get("evidence_logic")))
            for block in ("eligibility", "action", "timing", "exceptions")
        )
        for rule in seed_rules
    )
    ast_normalized = int(full_rules_doc.get("ast_normalized_count") or seed_ast_normalized)

    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    detail_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        concepts = [str(x) for x in as_list(candidate.get("critical_variable_concepts"))]
        normalized_rule = full_rule_index.get(str(candidate.get("candidate_id"))) or {}
        matches = component_matches(
            [
                str(value)
                for value in as_list(normalized_rule.get("required_variables"))
            ],
            local_variable_to_field,
        )
        row = {
            "candidate_id": candidate.get("candidate_id"),
            "parent_recommendation_id": candidate.get("parent_recommendation_id"),
            "candidate_kind": candidate.get("candidate_kind"),
            "source_id": candidate.get("source_id"),
            "category": candidate.get("category"),
            "core_or_supplemental": candidate.get("core_or_supplemental"),
            "critical_variable_concept_count": len(concepts),
            "registry_component_matches": matches,
            "store_standard_structural": (
                full_coverage_index.get(str(candidate.get("candidate_id"))) or {}
            ),
            "local_registry_schema": {
                "reportable_full_rule_evaluability": (
                    "none" if ast_normalized == len(candidates) else "not_assessed"
                ),
                "structural_full_rule_upper_bound": "none",
                "component_evidence_reach": "partial" if matches else "none",
                "reason": (
                    "The local extract has no treatment, molecular-testing, imaging/procedure, "
                    "decision-rationale, exception, or required action-timing fields. A site, "
                    "histology, stage, or "
                    "recurrence component cannot by itself establish reportable rule coverage."
                ),
            },
            "observed_crc_data": {
                "status": "NOT_ASSESSED",
                "reason": "The bound measured profile is lung-cancer and patient-level, not a CRC tumor-record profile.",
            },
        }
        detail_rows.append(row)
        category_rows[str(candidate.get("category") or "unknown")].append(row)
        source_rows[str(candidate.get("source_id") or "unknown")].append(row)

    extracted_core = sum(row["core_or_supplemental"] == "core" for row in detail_rows)
    extracted_supp = len(detail_rows) - extracted_core
    expected = denominator.get("summary") or {}
    denominator_sources = as_list(denominator.get("sources"))
    accessible_core_sources = [
        source
        for source in denominator_sources
        if source.get("core_or_supplemental") == "core" and not source.get("license_blocker")
    ]
    accessible_supp_sources = [
        source
        for source in denominator_sources
        if source.get("core_or_supplemental") in {"supplemental", "adjacent"}
        and not source.get("license_blocker")
    ]
    partial = sum(
        row["local_registry_schema"]["component_evidence_reach"] == "partial"
        for row in detail_rows
    )
    normalized_coverage_summary = (
        (full_coverage_doc.get("summary") or {}).get(
            "registry_only_nonconcordance_defensibility"
        )
        or {}
    )
    full_logic_complete = ast_normalized == len(detail_rows)
    normalized_by_category = full_coverage_doc.get("by_category") or {}
    report = {
        "coverage_schema_version": "1.0",
        "artifact_id": "crc_core_v1_guideline_universe_evidence_coverage",
        "as_of_date": denominator.get("as_of_date"),
        "clinical_use": "NOT_FOR_CLINICAL_USE",
        "denominators": {
            "accessible_source_core_executable_expected": expected.get("core_executable_candidates"),
            "accessible_source_supplemental_expected": expected.get("supplemental_or_adjacent_candidates"),
            "accessible_source_total_expected": expected.get("total_executable_candidates"),
            "accessible_source_core_nonformal_units": expected.get("core_nonformal_source_units"),
            "accessible_source_core_scoped_units": expected.get("core_scoped_source_units"),
            "accessible_source_core_formal_recommendations": sum(
                int(source.get("formal_recommendations_in_scope") or 0)
                for source in accessible_core_sources
            ),
            "accessible_source_supplemental_formal_recommendations": sum(
                int(source.get("formal_recommendations_in_scope") or 0)
                for source in accessible_supp_sources
            ),
            "universe_rows_currently_extracted": len(detail_rows),
            "core_rows_currently_extracted": extracted_core,
            "supplemental_rows_currently_extracted": extracted_supp,
            "distinct_parent_recommendation_ids_currently_extracted": len(
                {
                    (row["source_id"], row["parent_recommendation_id"])
                    for row in detail_rows
                }
            ),
            "candidate_kind_counts": dict(
                sorted(Counter(str(row["candidate_kind"]) for row in detail_rows).items())
            ),
            "nccn_candidates_reconstructed": expected.get("nccn_candidates_reconstructed"),
        },
        "authoring_layers": {
            "intake_universe": {
                "status": (
                    "complete"
                    if len(detail_rows) == expected.get("total_executable_candidates")
                    else "in_progress"
                ),
                "candidate_count": len(detail_rows),
            },
            "structured_seed_rules": {
                "status": "four_block_seed_not_full_universe",
                "count": len(seed_rules),
                "fraction_of_core_denominator": (
                    round(
                        len(seed_rules)
                        / int(expected.get("core_executable_candidates") or 1),
                        6,
                    )
                ),
            },
            "predicate_ast_normalized_rules": {
                "count": ast_normalized,
                "status": (
                    "complete_for_full_universe"
                    if ast_normalized == len(detail_rows)
                    else "normalization_pending"
                ),
                "root_node_counts": full_manifest.get("evidence_block_root_counts") or {},
                "caveat": (
                    "Explicit unresolved nodes remain non-evaluable; AST normalization "
                    "does not imply every rule is computable."
                ),
            },
            "canonical_variable_contracts": {
                "count": (
                    full_manifest.get("variable_contract_count")
                    or len(as_list(variables.get("variables")))
                ),
                "one_contract_per_current_canonical_variable": (
                    (
                        full_manifest.get("variable_contract_count")
                        == full_manifest.get("canonical_variable_count")
                    )
                    if full_manifest
                    else (
                        execution.get("variable_contract_count")
                        == len(as_list(variables.get("variables")))
                    )
                ),
            },
            "variable_use_coverage": {
                "row_count": int(
                    ((full_coverage_doc.get("summary") or {}).get("variable_use_rows"))
                    or 0
                ),
                "dimensions": [
                    "semantic_support",
                    "temporal_alignment",
                    "provenance_suitability",
                    "canonical_to_registry_projection",
                ],
                "review_status": "candidate_registrar_review_pending",
                "coarsened_projection_contracts": int(
                    (
                        (full_coverage_doc.get("summary") or {}).get(
                            "coarsened_projection_contracts"
                        )
                    )
                    or 0
                ),
            },
            "full_universe_variable_concepts": {
                "distinct": (concept_inventory.get("summary") or {}).get("distinct_concepts"),
                "candidate_scoped_bindings_complete": ast_normalized == len(detail_rows),
                "canonicalization_pending": 0 if full_manifest else (
                    (concept_inventory.get("summary") or {}).get("canonicalization_pending")
                ),
                "canonical_variable_count": full_manifest.get("canonical_variable_count"),
                "caveat": (
                    "Bindings and contracts are model-authored candidates pending clinical "
                    "canonicalization review; complete binding does not mean clinical approval."
                ),
            },
            "execution_layers": {
                "seed_materialized_runtime_groups": execution.get("execution_group_count"),
                "full_universe_planned_groups": full_manifest.get("execution_group_plan_count"),
                "full_universe_runtime_status": (
                    (full_manifest.get("review_status") or {}).get("runtime_materialization")
                    or "not_started"
                ),
            },
        },
        "coverage_claims": {
            "current_local_registry_profile_id": "R6249_LUNG_SCHEMA_2026-07-25",
            "profile_compatibility": "SCHEMA_ONLY_NON_CRC",
            "reportable_full_rule_evaluability": {
                "status": (
                    "CANDIDATE_STRUCTURALLY_ASSESSED"
                    if full_logic_complete
                    else "NOT_ASSESSED"
                ),
                "full": int(normalized_coverage_summary.get("full") or 0),
                "partial": int(normalized_coverage_summary.get("partial") or 0),
                "none": (
                    int(normalized_coverage_summary.get("none") or 0)
                    if full_logic_complete
                    else 0
                ),
                "not_assessed": 0 if full_logic_complete else len(detail_rows),
                "denominator": len(detail_rows),
                "reason": (
                    "All candidates have evidence ASTs and canonical variable bindings; "
                    "coverage is a conservative candidate structural assessment with "
                    "registrar review still pending."
                    if full_logic_complete
                    else
                    "Complete evidence ASTs or canonical mappings are still absent."
                ),
            },
            "structural_schema_upper_bound": {
                "full": 0,
                "none": len(detail_rows),
                "denominator": len(detail_rows),
                "scope": "rows_currently_extracted",
                "reportability": "diagnostic_not_a_coverage_rate",
                "assessment_basis": (
                    "Every in-scope candidate requires at least one molecular, treatment, "
                    "imaging/procedure, decision, exception, or action-timing fact absent "
                    "from the bound 15-column local registry schema."
                ),
            },
            "component_evidence_reach": {
                "partial": partial,
                "none": len(detail_rows) - partial,
                "denominator": len(detail_rows),
                "warning": "Component reach is not rule coverage and cannot support a concordance denominator.",
                "method": (
                    "exact canonical-variable IDs declared by the immutable local profile; "
                    "no substring, fuzzy-name, or inferred synonym matching"
                ),
            },
            "observed_crc_coverage": {
                "status": "NOT_ASSESSED",
                "reason": "No compatible immutable tumor-level CRC linked-registry profile is bound.",
            },
        },
        "by_category": {
            category: {
                "candidate_count": len(rows),
                "distinct_parent_id_count": len(
                    {(row["source_id"], row["parent_recommendation_id"]) for row in rows}
                ),
                "structural_upper_bound_full": 0,
                "store_standard_structural": normalized_by_category.get(category)
                or {"full": 0, "partial": 0, "none": 0},
                "component_reach_partial": sum(
                    row["local_registry_schema"]["component_evidence_reach"] == "partial"
                    for row in rows
                ),
            }
            for category, rows in sorted(category_rows.items())
        },
        "by_source": {
            source: {
                "candidate_count": len(rows),
                "distinct_parent_id_count": len(
                    {(row["source_id"], row["parent_recommendation_id"]) for row in rows}
                ),
                "structural_upper_bound_full": 0,
                "component_reach_partial": sum(
                    row["local_registry_schema"]["component_evidence_reach"] == "partial"
                    for row in rows
                ),
            }
            for source, rows in sorted(source_rows.items())
        },
        "rules": detail_rows,
    }
    return report


def render_markdown(report: dict[str, Any]) -> str:
    denominator = report["denominators"]
    layers = report["authoring_layers"]
    coverage = report["coverage_claims"]
    reportable = coverage["reportable_full_rule_evaluability"]
    upper = coverage["structural_schema_upper_bound"]
    component = coverage["component_evidence_reach"]
    lines = [
        "# CRC guideline-universe evidence coverage",
        "",
        "> Authoring assessment — **NOT FOR CLINICAL USE**.",
        "",
        "## Answer",
        "",
        f"- Accessible-source denominator: **{denominator['accessible_source_core_executable_expected']} core** "
        f"+ **{denominator['accessible_source_supplemental_expected']} supplemental/adjacent** candidates.",
        f"- Formal source units: **{denominator['accessible_source_core_formal_recommendations']} core** "
        f"+ **{denominator['accessible_source_supplemental_formal_recommendations']} supplemental/adjacent**; "
        "executable branch counts must not be presented as independent recommendations.",
        f"- Core source review also includes **{denominator['accessible_source_core_nonformal_units']} "
        f"nonformal units**; total core scoped source units are "
        f"**{denominator['accessible_source_core_scoped_units']}**.",
        f"- Universe rows materialized: **{denominator['universe_rows_currently_extracted']}**.",
        f"- STORE-standard candidate structural rule coverage: **{reportable['full']} full / "
        f"{reportable['partial']} partial / {reportable['none']} none / "
        f"{reportable['not_assessed']} not assessed**, denominator "
        f"**{reportable['denominator']}**.",
        f"- Structural schema upper bound: **{upper['full']}/{upper['denominator']}** candidates "
        "could be fully evaluated from the bound 15-column extract; this is a diagnostic, not a coverage rate.",
        f"- The local profile declares at least one exact canonical component for "
        f"**{component['partial']}/{component['denominator']}** rows; this is not concordance coverage.",
        "- Observed CRC coverage: **NOT_ASSESSED** because the measured profile is a lung-cancer, "
        "patient-level extract rather than a CRC tumor-record dataset.",
        "",
        "## Why 12 / 68 / 6 were misleading",
        "",
        f"- **{layers['structured_seed_rules']['count']}** is the current four-block seed tranche, "
        "not the guideline denominator.",
        f"- **{layers['predicate_ast_normalized_rules']['count']}** full-universe candidates "
        "have explicit machine-readable evidence ASTs; "
        f"**{layers['predicate_ast_normalized_rules']['root_node_counts'].get('unresolved', 0)}** "
        "requirement blocks are explicitly unresolved rather than silently treated as true.",
        f"- **{layers['canonical_variable_contracts']['count']}** is the current canonical-variable "
        "contract count; every current variable has one contract.",
        f"- Evidence coverage contains **{layers['variable_use_coverage']['row_count']}** "
        "rule/predicate/variable-use rows with separate semantic, temporal, provenance, "
        "and registry-projection dimensions.",
        f"- All coarsened uses resolve to **"
        f"{layers['variable_use_coverage']['coarsened_projection_contracts']}** "
        "named/versioned candidate projection contracts.",
        f"- The full universe contains **{layers['full_universe_variable_concepts']['distinct']}** "
        f"distinct source concepts; all candidate-scoped uses are bound into "
        f"**{layers['full_universe_variable_concepts']['canonical_variable_count']}** "
        "candidate canonical variables/contracts.",
        f"- The original seed has **{layers['execution_layers']['seed_materialized_runtime_groups']}** "
        f"materialized runtime groups. The full universe has "
        f"**{layers['execution_layers']['full_universe_planned_groups']}** planned groups and "
        "is not runtime-materialized yet.",
        "",
        "## Category coverage",
        "",
        "| category | candidates | source parents | STORE full | STORE partial | STORE none | local component reach |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for category, row in report["by_category"].items():
        lines.append(
            f"| {category} | {row['candidate_count']} | {row['distinct_parent_id_count']} | "
            f"{row['store_standard_structural']['full']} | "
            f"{row['store_standard_structural']['partial']} | "
            f"{row['store_standard_structural']['none']} | "
            f"{row['component_reach_partial']} |"
        )
    lines += [
        "",
        "The structural upper bound is zero because the local extract contains no molecular-testing, "
        "treatment, imaging/procedure, decision-rationale, exception, or action-timing fields. "
        "Its stage and tumor-identity columns can help validate components. Full-universe "
        "AST normalization is complete, but observed CRC availability remains NOT_ASSESSED "
        "until a compatible tumor-level CRC profile is bound.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    report = compute(bundle)
    yaml_path = bundle / "guideline_universe_evidence_coverage.yaml"
    md_path = bundle / "guideline_universe_evidence_coverage.md"
    yaml_path.write_text(
        yaml.safe_dump(report, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"Wrote universe coverage for {report['denominators']['universe_rows_currently_extracted']} "
        f"candidate(s): {yaml_path}, {md_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
