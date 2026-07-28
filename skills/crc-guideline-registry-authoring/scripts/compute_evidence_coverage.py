#!/usr/bin/env python3
"""Compute structural CRC registry evidence coverage without inventing data availability.

The report separates:
  * registry -> guideline-predicate semantic support;
  * canonical agent output -> registry validation projection;
  * standards-level structural coverage from observed compatible-data coverage.

Rules without an evidence AST receive a conservative flat-list calculation, explicitly
labelled LOGIC_NOT_NORMALIZED. That result is not promoted to measured rule coverage.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


BLOCKS = ("eligibility", "action", "timing", "exceptions")
STATES = ("full", "partial", "none", "not_assessed")
SEMANTIC_LEVELS = ("exact", "coarsened", "absent", "not_applicable", "unknown")
PROJECTION_LEVELS = ("direct", "coarsened", "none", "unknown")


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return value


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def registry_is_named(variable: dict[str, Any]) -> bool:
    mapping = variable.get("registry_mapping") or {}
    return str(mapping.get("standard") or "").lower() not in {"", "none", "null"}


def variable_axes(variable: dict[str, Any]) -> dict[str, str]:
    """Return independent registry-to-rule and agent-to-registry axes."""
    explicit = variable.get("evidence_coverage") or {}
    r2r = explicit.get("registry_to_rule") or {}
    projection = explicit.get("canonical_to_registry") or {}
    level = variable.get("mapping_level")

    fallback_semantic = {
        "registry_direct": "exact",
        "registry_coarsened": "coarsened",
        "chart_extension": "absent",
        "derived": "coarsened" if registry_is_named(variable) else "absent",
        "outside_current_sources": "absent",
    }.get(level, "unknown")
    fallback_temporal = {
        "registry_direct": "exact",
        "registry_coarsened": "conditional",
        "chart_extension": "misaligned",
        "derived": "conditional",
        "outside_current_sources": "unknown",
    }.get(level, "unknown")
    fallback_projection = {
        "registry_direct": "direct",
        "registry_coarsened": "coarsened",
        "chart_extension": "coarsened" if registry_is_named(variable) else "none",
        "derived": "coarsened" if registry_is_named(variable) else "none",
        "outside_current_sources": "none",
    }.get(level, "unknown")

    semantic = str(r2r.get("semantic") or fallback_semantic)
    temporal = str(r2r.get("temporal") or fallback_temporal)
    projected = str(projection.get("level") or fallback_projection)
    if semantic not in SEMANTIC_LEVELS:
        semantic = "unknown"
    if projected not in PROJECTION_LEVELS:
        projected = "unknown"
    structural = semantic_temporal_state(semantic, temporal)
    return {
        "semantic": semantic,
        "temporal": temporal,
        "projection": projected,
        "structural": structural,
        "axis_status": "explicit" if explicit else "inferred_from_mapping_level",
    }


def semantic_temporal_state(semantic: str, temporal: str) -> str:
    if semantic in {"absent", "not_applicable"} or temporal == "misaligned":
        return "none"
    if semantic == "exact" and temporal == "exact":
        return "full"
    if semantic in {"exact", "coarsened"} and temporal in {"exact", "conditional"}:
        return "partial"
    return "not_assessed"


def combine(op: str, states: list[str]) -> str:
    if not states:
        return "full"
    if op == "any_of":
        if "full" in states:
            return "full"
        if "partial" in states:
            return "partial"
        if all(state == "none" for state in states):
            return "none"
        return "not_assessed"
    # all_of, conditional and unknown operators use the conservative conjunction.
    if "none" in states:
        return "none"
    if "not_assessed" in states:
        return "not_assessed"
    if "partial" in states:
        return "partial"
    return "full"


def leaf_ids(node: Any) -> set[str]:
    if isinstance(node, str):
        return {node}
    if not isinstance(node, dict):
        return set()
    if node.get("variable"):
        return {str(node["variable"])}
    values: set[str] = set()
    for operand in as_list(node.get("operands")):
        values.update(leaf_ids(operand))
    for key in ("condition", "then", "else", "operand"):
        if node.get(key) is not None:
            values.update(leaf_ids(node[key]))
    return values


def evaluate_ast(node: Any, variable_index: dict[str, dict[str, Any]]) -> str:
    if isinstance(node, str):
        return variable_axes(variable_index.get(node) or {})["structural"]
    if not isinstance(node, dict):
        return "not_assessed"
    if node.get("variable"):
        return variable_axes(variable_index.get(str(node["variable"])) or {})["structural"]
    op = str(node.get("op") or "all_of")
    if op == "not":
        return evaluate_ast(node.get("operand"), variable_index)
    if op == "conditional":
        parts = [
            evaluate_ast(node.get(key), variable_index)
            for key in ("condition", "then", "else")
            if node.get(key) is not None
        ]
        return combine("all_of", parts)
    return combine(op, [evaluate_ast(x, variable_index) for x in as_list(node.get("operands"))])


def block_coverage(
    block: dict[str, Any], variable_index: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    variables = [str(x) for x in as_list(block.get("variables"))]
    ast = block.get("evidence_logic")
    if ast:
        mentioned = leaf_ids(ast)
        missing = sorted(set(variables) - mentioned)
        extra = sorted(mentioned - set(variables))
        return {
            "status": evaluate_ast(ast, variable_index),
            "logic_status": "normalized",
            "variables": variables,
            "ast_missing_declared_variables": missing,
            "ast_undeclared_variables": extra,
        }
    return {
        "status": combine(
            "all_of",
            [variable_axes(variable_index.get(variable_id) or {})["structural"] for variable_id in variables],
        ),
        "logic_status": "LOGIC_NOT_NORMALIZED",
        "variables": variables,
        "ast_missing_declared_variables": variables,
        "ast_undeclared_variables": [],
    }


def profile_summary(bundle: Path, variable_ids: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for path in sorted((bundle / "registry_profiles").glob("*.yaml")):
        profile = load_yaml(path)
        dataset = profile.get("dataset") or {}
        fields = as_list((profile.get("registry_schema") or {}).get("fields"))
        mapped = {
            str(variable_id)
            for field in fields
            for variable_id in as_list((field or {}).get("canonical_candidates"))
        }
        out.append(
            {
                "profile_id": profile.get("profile_id"),
                "profile_status": profile.get("profile_status"),
                "disease": dataset.get("disease"),
                "case_unit": dataset.get("case_unit"),
                "compatibility_with_crc": dataset.get("compatibility_with_crc"),
                "canonical_variables_present_in_schema": len(mapped & variable_ids),
                "canonical_variables_present": sorted(mapped & variable_ids),
                "crc_observed_availability": (
                    (profile.get("crc_observed_availability") or {}).get("status")
                    or "NOT_ASSESSED"
                ),
            }
        )
    return out


def count_states(values: list[str]) -> dict[str, int]:
    counter = Counter(values)
    return {state: counter[state] for state in STATES}


def compute(bundle: Path) -> dict[str, Any]:
    rule_doc = load_yaml(bundle / "candidate_rules.yaml")
    variable_doc = load_yaml(bundle / "variable_inventory.yaml")
    rules = as_list(rule_doc.get("rules"))
    variables = as_list(variable_doc.get("variables"))
    variable_index = {str(v.get("variable_id")): v for v in variables}
    variable_ids = set(variable_index)

    rows: list[dict[str, Any]] = []
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    use_semantic: Counter[str] = Counter()
    use_temporal: Counter[str] = Counter()
    use_projection: Counter[str] = Counter()

    for rule in rules:
        blocks: dict[str, dict[str, Any]] = {}
        normalized = True
        missing_cut: set[str] = set()
        uses_seen: set[tuple[str, str]] = set()
        for block_name in BLOCKS:
            block = (rule.get("requirements") or {}).get(block_name) or {}
            result = block_coverage(block, variable_index)
            blocks[block_name] = result
            normalized = normalized and result["logic_status"] == "normalized"
            for variable_id in result["variables"]:
                axes = variable_axes(variable_index.get(variable_id) or {})
                if (block_name, variable_id) not in uses_seen:
                    use_semantic[axes["semantic"]] += 1
                    use_temporal[axes["temporal"]] += 1
                    use_projection[axes["projection"]] += 1
                    uses_seen.add((block_name, variable_id))
                if axes["structural"] == "none":
                    missing_cut.add(variable_id)

        denominator = blocks["eligibility"]["status"]
        concordance = combine(
            "all_of",
            [blocks[name]["status"] for name in ("eligibility", "action", "timing")],
        )
        nonconcordance = combine("all_of", [blocks[name]["status"] for name in BLOCKS])
        if not normalized:
            reportable = {
                "denominator_evaluability": "not_assessed",
                "concordance_evaluability": "not_assessed",
                "nonconcordance_defensibility": "not_assessed",
            }
        else:
            reportable = {
                "denominator_evaluability": denominator,
                "concordance_evaluability": concordance,
                "nonconcordance_defensibility": nonconcordance,
            }
        row = {
            "rule_id": rule.get("rule_id"),
            "parent_recommendation_id": rule.get("parent_recommendation_id") or rule.get("rule_id"),
            "candidate_kind": rule.get("candidate_kind") or "unspecified",
            "category": rule.get("category"),
            "logic_status": "normalized" if normalized else "LOGIC_NOT_NORMALIZED",
            "blocks": blocks,
            "reportable_registry_only": reportable,
            "conservative_flat_registry_support": {
                "denominator_evaluability": denominator,
                "concordance_evaluability": concordance,
                "nonconcordance_defensibility": nonconcordance,
            },
            "registry_validation": {
                "direct_variable_uses": sum(
                    1
                    for block in blocks.values()
                    for variable_id in block["variables"]
                    if variable_axes(variable_index.get(variable_id) or {})["projection"] == "direct"
                ),
                "coarsened_variable_uses": sum(
                    1
                    for block in blocks.values()
                    for variable_id in block["variables"]
                    if variable_axes(variable_index.get(variable_id) or {})["projection"] == "coarsened"
                ),
                "unvalidated_variable_uses": sum(
                    1
                    for block in blocks.values()
                    for variable_id in block["variables"]
                    if variable_axes(variable_index.get(variable_id) or {})["projection"] == "none"
                ),
            },
            "structural_missing_variables": sorted(missing_cut),
            "observed_crc_availability": "NOT_ASSESSED",
        }
        rows.append(row)
        category_rows[str(rule.get("category"))].append(row)

    categories: dict[str, Any] = {}
    for category, category_rules in sorted(category_rows.items()):
        categories[category] = {
            "executable_candidates": len(category_rules),
            "formal_recommendations": len(
                {str(row["parent_recommendation_id"]) for row in category_rules}
            ),
            "logic_normalized": sum(row["logic_status"] == "normalized" for row in category_rules),
            "reportable_nonconcordance_defensibility": count_states(
                [
                    row["reportable_registry_only"]["nonconcordance_defensibility"]
                    for row in category_rules
                ]
            ),
            "conservative_flat_nonconcordance_support": count_states(
                [
                    row["conservative_flat_registry_support"]["nonconcordance_defensibility"]
                    for row in category_rules
                ]
            ),
            "top_structural_bottlenecks": Counter(
                variable_id
                for row in category_rules
                for variable_id in row["structural_missing_variables"]
            ).most_common(15),
        }

    profiles = profile_summary(bundle, variable_ids)
    return {
        "coverage_schema_version": "1.0",
        "bundle_id": rule_doc.get("bundle_id"),
        "clinical_use": "NOT_FOR_CLINICAL_USE",
        "coverage_claim": {
            "scope": "structured_seed_rules_only_not_guideline_denominator",
            "standards_level": "structural_upper_bound",
            "observed_crc": "NOT_ASSESSED",
            "reason": "No compatible tumor-level linked CRC registry profile is bound.",
        },
        "summary": {
            "executable_candidates": len(rows),
            "formal_recommendations": len(
                {str(row["parent_recommendation_id"]) for row in rows}
            ),
            "canonical_variables": len(variable_index),
            "logic_normalized_rules": sum(row["logic_status"] == "normalized" for row in rows),
            "reportable_registry_only_nonconcordance": count_states(
                [
                    row["reportable_registry_only"]["nonconcordance_defensibility"]
                    for row in rows
                ]
            ),
            "conservative_flat_registry_support_nonconcordance": count_states(
                [
                    row["conservative_flat_registry_support"]["nonconcordance_defensibility"]
                    for row in rows
                ]
            ),
            "variable_use_semantic_support": dict(sorted(use_semantic.items())),
            "variable_use_temporal_alignment": dict(sorted(use_temporal.items())),
            "agent_validation_projection": dict(sorted(use_projection.items())),
        },
        "profiles": profiles,
        "categories": categories,
        "rules": rows,
    }


def render_markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# CRC registry evidence coverage",
        "",
        "> Authoring assessment — **NOT FOR CLINICAL USE**.",
        "",
        "> **SEED LAYER ONLY.** This file covers the 12 structured seed rules, not the "
        "280-candidate guideline universe. Use `guideline_universe_evidence_coverage.md` "
        "for denominator-level reporting.",
        "",
        "## Blocking interpretation",
        "",
        f"- Executable candidates: **{summary['executable_candidates']}**.",
        f"- Canonical variables: **{summary['canonical_variables']}**.",
        f"- Rules with normalized evidence AST: **{summary['logic_normalized_rules']}**.",
        "- Observed CRC availability: **NOT_ASSESSED**; no compatible tumor-level CRC profile is bound.",
        "",
        "A rule can be useful for registry validation while remaining unevaluable from registry "
        "data alone. Coarsened projection coverage is not guideline-rule coverage.",
        "",
        "## Category coverage",
        "",
        "| category | candidates | formal parents | AST normalized | full | partial | none | not assessed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, value in report["categories"].items():
        states = value["reportable_nonconcordance_defensibility"]
        lines.append(
            f"| {category} | {value['executable_candidates']} | "
            f"{value['formal_recommendations']} | {value['logic_normalized']} | "
            f"{states['full']} | {states['partial']} | {states['none']} | "
            f"{states['not_assessed']} |"
        )
    lines += [
        "",
        "## Conservative flat-list support",
        "",
        "This is an upper-bound diagnostic for rules whose AST is not normalized; it is not a "
        "reportable coverage rate.",
        "",
        "| category | full | partial | none | not assessed |",
        "|---|---:|---:|---:|---:|",
    ]
    for category, value in report["categories"].items():
        states = value["conservative_flat_nonconcordance_support"]
        lines.append(
            f"| {category} | {states['full']} | {states['partial']} | "
            f"{states['none']} | {states['not_assessed']} |"
        )
    lines += [
        "",
        "## Registry profiles",
        "",
        "| profile | disease | unit | CRC compatibility | mapped canonical variables | CRC observed |",
        "|---|---|---|---|---:|---|",
    ]
    for profile in report["profiles"]:
        lines.append(
            f"| {profile['profile_id']} | {profile['disease']} | {profile['case_unit']} | "
            f"{profile['compatibility_with_crc']} | "
            f"{profile['canonical_variables_present_in_schema']} | "
            f"{profile['crc_observed_availability']} |"
        )
    lines += [
        "",
        "## Variable-use axes",
        "",
        f"- Registry → rule semantic support: `{summary['variable_use_semantic_support']}`.",
        f"- Temporal alignment: `{summary['variable_use_temporal_alignment']}`.",
        f"- Agent → registry validation projection: `{summary['agent_validation_projection']}`.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--output-yaml", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    report = compute(bundle)
    yaml_path = args.output_yaml or bundle / "evidence_coverage.yaml"
    md_path = args.output_md or bundle / "evidence_coverage.md"
    yaml_path.write_text(
        yaml.safe_dump(report, sort_keys=False, allow_unicode=True, width=120),
        encoding="utf-8",
    )
    md_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        f"Wrote evidence coverage for {report['summary']['executable_candidates']} rule(s), "
        f"{report['summary']['canonical_variables']} variable(s): {yaml_path}, {md_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
