#!/usr/bin/env python3
"""Validate a CRC guideline-to-registry authoring bundle without patient data.

This is an authoring-integrity validator. It deliberately does not certify clinical
correctness, registry completeness, or performance on linked data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


SOURCE_STATUSES = {
    "version_bound",
    "source_pending",
    "superseded_or_update_pending",
    "context_only",
}
RULE_CATEGORIES = {
    "molecular_testing",
    "molecular_testing_operations",
    "localized_colon_treatment",
    "locally_advanced_rectal_treatment",
    "metastatic_systemic_treatment",
    "local_metastatic_treatment",
    "nonoperative_management",
    "surveillance",
    "hereditary_risk_workup",
}
COMPUTABILITY = {"fully_specified", "partially_specified", "not_computable"}
MAPPING_LEVELS = {
    "registry_direct",
    "registry_coarsened",
    "chart_extension",
    "derived",
    "outside_current_sources",
}
SPEC_VERDICTS = {
    "conformant_candidate",
    "needs_revision",
    "blocked_source",
    "blocked_runtime",
    "not_assessed",
}
DATA_COVERAGE = {
    "NOT_ASSESSED",
    "FIELD_PRESENT_PROFILED",
    "FIELD_ABSENT_PROFILED",
    "SOURCE_PRESENT_PROFILED",
    "SOURCE_ABSENT_PROFILED",
}
SEVERITIES = {"none", "minor", "major", "blocking"}
GAP_CLASSES = {
    "NO_REGISTRY_FIELD",
    "REGISTRY_TOO_COARSE",
    "NO_EXTRACTION_SPEC",
    "SPEC_NONCONFORMANT",
    "SOURCE_VERSION_UNBOUND",
    "TEMPORAL_ANCHOR_MISSING",
    "EXCEPTION_MODEL_MISSING",
    "RUNTIME_OPERATOR_MISSING",
    "DATA_NOT_PROFILED",
    "DATA_FIELD_ABSENT",
    "DATA_HIGH_MISSINGNESS",
    "DATA_TIME_COVERAGE_MISMATCH",
    "CLINICAL_REVIEW_PENDING",
    "REGISTRAR_REVIEW_PENDING",
}
REQUIREMENT_ROLES = {"eligibility", "action", "timing", "exceptions"}
ORIGINS = {
    "store_manual",
    "ajcc_manual",
    "corpus_derived",
    "model_authored",
    "clinician",
}
FACT_STATUSES = {
    "explicit_guideline_fact",
    "inferred_from_guideline",
    "local_operationalization",
    "source_not_extracted",
}
UNIVERSE_CANDIDATE_KINDS = {
    "formal_recommendation",
    "executable_branch",
    "operational_qualifier",
    "good_practice",
    "adjacent",
}


class Report:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, where: str, message: str) -> None:
        self.errors.append(f"{where}: {message}")

    def warn(self, where: str, message: str) -> None:
        self.warnings.append(f"{where}: {message}")

    def require(self, condition: Any, where: str, message: str) -> None:
        if not condition:
            self.error(where, message)


def load_yaml(path: Path, report: Report) -> Any:
    if not path.is_file():
        report.error(str(path), "required file is missing")
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:  # pragma: no cover - defensive CLI boundary
        report.error(str(path), f"cannot parse YAML: {exc}")
        return {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def nonempty(value: Any) -> bool:
    return bool(str(value or "").strip())


def require_keys(report: Report, where: str, obj: Any, keys: list[str]) -> None:
    if not isinstance(obj, dict):
        report.error(where, "must be a mapping")
        return
    for key in keys:
        report.require(key in obj and obj[key] not in (None, "", []), where, f"missing {key!r}")


def duplicates(values: list[str]) -> list[str]:
    return sorted(key for key, count in Counter(values).items() if count > 1)


def validate_manifest(root: Path, report: Report) -> dict[str, Any]:
    manifest = load_yaml(root / "manifest.yaml", report)
    require_keys(
        report,
        "manifest",
        manifest,
        [
            "schema_version",
            "bundle_id",
            "disease",
            "focus",
            "authoring_status",
            "clinical_use",
            "generated_on",
            "care_settings",
            "data_profile",
        ],
    )
    report.require(
        manifest.get("clinical_use") == "NOT_FOR_CLINICAL_USE",
        "manifest",
        "clinical_use must be NOT_FOR_CLINICAL_USE for an authoring bundle",
    )
    profile = manifest.get("data_profile") or {}
    require_keys(report, "manifest.data_profile", profile, ["status"])
    if profile.get("status") == "NOT_ASSESSED":
        report.require(
            not profile.get("profile_id"),
            "manifest.data_profile",
            "profile_id must be empty when data were not assessed",
        )
    else:
        report.require(
            nonempty(profile.get("profile_id")),
            "manifest.data_profile",
            "a measured data claim requires profile_id",
        )
    return manifest


def validate_sources(root: Path, report: Report) -> tuple[dict[str, dict], dict[str, Any]]:
    doc = load_yaml(root / "source_register.yaml", report)
    sources = as_list(doc.get("sources"))
    report.require(bool(sources), "source_register", "sources must be non-empty")
    index: dict[str, dict] = {}
    for i, source in enumerate(sources):
        where = f"source_register.sources[{i}]"
        require_keys(
            report,
            where,
            source,
            [
                "source_id",
                "authority",
                "title",
                "source_type",
                "version",
                "publication_or_update_date",
                "status",
                "url",
                "accessed_on",
                "scope",
                "use",
                "limitations",
                "license",
                "document_hash",
                "review_status",
            ],
        )
        source_id = str(source.get("source_id") or "")
        if source_id in index:
            report.error(where, f"duplicate source_id {source_id!r}")
        index[source_id] = source
        report.require(
            source.get("status") in SOURCE_STATUSES,
            where,
            f"status must be one of {sorted(SOURCE_STATUSES)}",
        )
        if source.get("status") == "version_bound":
            version = str(source.get("version") or "").strip().lower()
            report.require(
                version not in {"current", "latest", "unknown", "pending"},
                where,
                "version_bound source needs an exact version/date/DOI, not a moving label",
            )
    return index, doc


def validate_intake(root: Path, report: Report) -> dict[str, Any]:
    denominator = load_yaml(root / "intake" / "source_denominator.yaml", report)
    universe = load_yaml(root / "intake" / "rule_universe.yaml", report)
    summary = denominator.get("summary") or {}
    require_keys(
        report,
        "intake.source_denominator.summary",
        summary,
        [
            "core_executable_candidates",
            "core_formal_recommendations",
            "core_nonformal_source_units",
            "core_scoped_source_units",
            "supplemental_or_adjacent_candidates",
            "total_executable_candidates",
            "nccn_candidates_reconstructed",
        ],
    )
    sources = {
        str(source.get("source_id") or ""): source
        for source in as_list(denominator.get("sources"))
    }
    report.require(bool(sources), "intake.source_denominator", "sources must be non-empty")
    candidates = as_list(universe.get("candidates"))
    ids: list[str] = []
    for i, candidate in enumerate(candidates):
        where = f"intake.rule_universe.candidates[{i}]"
        require_keys(
            report,
            where,
            candidate,
            [
                "candidate_id",
                "parent_recommendation_id",
                "candidate_kind",
                "source_id",
                "source_anchor",
                "category",
                "title",
                "core_or_supplemental",
                "eligibility",
                "action",
                "timing",
                "exceptions",
                "logic_operator",
                "predicate_logic",
                "critical_variable_concepts",
                "activation_status",
            ],
        )
        report.require(
            "blockers" in candidate and isinstance(candidate.get("blockers"), list),
            where,
            "blockers must be present as a list; an empty list is valid",
        )
        candidate_id = str(candidate.get("candidate_id") or "")
        ids.append(candidate_id)
        report.require(
            candidate.get("candidate_kind") in UNIVERSE_CANDIDATE_KINDS,
            where,
            f"candidate_kind must be one of {sorted(UNIVERSE_CANDIDATE_KINDS)}",
        )
        report.require(
            candidate.get("core_or_supplemental") in {"core", "supplemental", "adjacent"},
            where,
            "core_or_supplemental must be core, supplemental, or adjacent",
        )
        report.require(
            candidate.get("source_id") in sources,
            where,
            f"source_id {candidate.get('source_id')!r} is not in source_denominator",
        )
    for duplicate in duplicates(ids):
        report.error("intake.rule_universe", f"duplicate candidate_id {duplicate!r}")
    core = sum(candidate.get("core_or_supplemental") == "core" for candidate in candidates)
    supplemental = len(candidates) - core
    core_formal_parent_projection = {
        str(candidate.get("parent_recommendation_id"))
        for candidate in candidates
        if candidate.get("core_or_supplemental") == "core"
        and candidate.get("candidate_kind") not in {"good_practice", "adjacent"}
    }
    accessible_core_sources = [
        source
        for source in sources.values()
        if source.get("core_or_supplemental") == "core" and not source.get("license_blocker")
    ]
    source_formal = sum(
        int(source.get("formal_recommendations_in_scope") or 0)
        for source in accessible_core_sources
    )
    source_nonformal = sum(
        int(source.get("nonformal_source_units_in_scope") or 0)
        for source in accessible_core_sources
    )
    report.require(
        summary.get("core_formal_recommendations")
        == source_formal
        == len(core_formal_parent_projection),
        "intake.source_denominator",
        "core formal denominator must agree across source rows, summary, and projected "
        f"distinct parent IDs; summary={summary.get('core_formal_recommendations')}, "
        f"sources={source_formal}, parents={len(core_formal_parent_projection)}",
    )
    report.require(
        summary.get("core_nonformal_source_units") == source_nonformal
        and summary.get("core_scoped_source_units") == source_formal + source_nonformal,
        "intake.source_denominator",
        "core nonformal/scoped source-unit counts do not reconcile",
    )
    report.require(
        core == summary.get("core_executable_candidates"),
        "intake.rule_universe",
        f"core candidate count must be {summary.get('core_executable_candidates')}, got {core}",
    )
    report.require(
        supplemental == summary.get("supplemental_or_adjacent_candidates"),
        "intake.rule_universe",
        f"supplemental/adjacent count must be {summary.get('supplemental_or_adjacent_candidates')}, got {supplemental}",
    )
    report.require(
        len(candidates) == summary.get("total_executable_candidates"),
        "intake.rule_universe",
        f"total candidate count must be {summary.get('total_executable_candidates')}, got {len(candidates)}",
    )
    coverage = load_yaml(root / "guideline_universe_evidence_coverage.yaml", report)
    coverage_denominator = coverage.get("denominators") or {}
    report.require(
        coverage_denominator.get("universe_rows_currently_extracted") == len(candidates),
        "guideline_universe_evidence_coverage",
        "coverage report is stale relative to rule_universe",
    )
    report.require(
        ((coverage.get("coverage_claims") or {}).get("observed_crc_coverage") or {}).get("status")
        == "NOT_ASSESSED",
        "guideline_universe_evidence_coverage",
        "observed CRC coverage must remain NOT_ASSESSED without a compatible CRC profile",
    )
    report.require(
        (root / "guideline_universe_evidence_coverage.md").is_file(),
        "guideline_universe_evidence_coverage.md",
        "reviewer view is missing",
    )
    concepts_doc = load_yaml(root / "intake" / "variable_concept_inventory.yaml", report)
    generated = concepts_doc.get("generated_from") or {}
    concepts = as_list(concepts_doc.get("concepts"))
    concept_summary = concepts_doc.get("summary") or {}
    concept_ids = [str(row.get("critical_variable_concept") or "") for row in concepts]
    expected_concepts = {
        str(concept)
        for candidate in candidates
        for concept in as_list(candidate.get("critical_variable_concepts"))
    }
    report.require(
        generated.get("candidate_count") == len(candidates),
        "intake.variable_concept_inventory",
        "candidate_count is stale relative to rule_universe",
    )
    report.require(
        not duplicates(concept_ids) and all(concept_ids),
        "intake.variable_concept_inventory",
        "critical_variable_concept values must be non-empty and unique",
    )
    report.require(
        set(concept_ids) == expected_concepts,
        "intake.variable_concept_inventory",
        f"concept set must exactly equal rule_universe; "
        f"missing={sorted(expected_concepts - set(concept_ids))}, "
        f"extra={sorted(set(concept_ids) - expected_concepts)}",
    )
    mapped = sum(row.get("canonicalization_status") == "mapped_existing" for row in concepts)
    pending = sum(
        row.get("canonicalization_status") == "canonicalization_pending" for row in concepts
    )
    report.require(
        mapped + pending == len(concepts),
        "intake.variable_concept_inventory",
        "canonicalization_status must be mapped_existing or canonicalization_pending",
    )
    report.require(
        concept_summary.get("distinct_concepts") == len(concepts)
        and concept_summary.get("mapped_existing") == mapped
        and concept_summary.get("canonicalization_pending") == pending,
        "intake.variable_concept_inventory",
        "summary counts are stale",
    )
    return universe


def requirement_variables(rule: dict[str, Any]) -> tuple[set[str], dict[str, set[str]]]:
    by_role: dict[str, set[str]] = {}
    requirements = rule.get("requirements") or {}
    for role in REQUIREMENT_ROLES:
        block = requirements.get(role) or {}
        by_role[role] = {str(v) for v in as_list(block.get("variables")) if nonempty(v)}
    return set().union(*by_role.values()), by_role


def validate_rules(
    root: Path, source_index: dict[str, dict], report: Report
) -> tuple[dict[str, dict], dict[str, Any]]:
    doc = load_yaml(root / "candidate_rules.yaml", report)
    rules = as_list(doc.get("rules"))
    report.require(bool(rules), "candidate_rules", "rules must be non-empty")
    index: dict[str, dict] = {}
    for i, rule in enumerate(rules):
        where = f"candidate_rules.rules[{i}]"
        require_keys(
            report,
            where,
            rule,
            [
                "rule_id",
                "category",
                "title",
                "authoring_status",
                "clinical_use",
                "fact_status",
                "review_status",
                "source_refs",
                "context",
                "requirements",
                "required_variables",
                "computability",
            ],
        )
        rule_id = str(rule.get("rule_id") or "")
        if rule_id in index:
            report.error(where, f"duplicate rule_id {rule_id!r}")
        index[rule_id] = rule
        report.require(
            rule.get("clinical_use") == "NOT_FOR_CLINICAL_USE",
            where,
            "clinical_use must be NOT_FOR_CLINICAL_USE",
        )
        report.require(
            rule.get("category") in RULE_CATEGORIES,
            where,
            f"category must be one of {sorted(RULE_CATEGORIES)}",
        )
        fact_status = rule.get("fact_status") or {}
        require_keys(
            report,
            f"{where}.fact_status",
            fact_status,
            ["source_context", "rule_projection"],
        )
        report.require(
            fact_status.get("source_context") in FACT_STATUSES,
            f"{where}.fact_status",
            f"source_context must be one of {sorted(FACT_STATUSES)}",
        )
        report.require(
            fact_status.get("rule_projection") == "local_operationalization",
            f"{where}.fact_status",
            "rule_projection must disclose local_operationalization",
        )
        context = rule.get("context") or {}
        require_keys(
            report,
            f"{where}.context",
            context,
            [
                "site",
                "histology",
                "disease_setting",
                "stage",
                "line_of_therapy",
                "molecular_state",
                "exclusions",
            ],
        )
        requirements = rule.get("requirements") or {}
        for role in REQUIREMENT_ROLES:
            block = requirements.get(role)
            require_keys(report, f"{where}.requirements.{role}", block, ["expression", "variables"])
        union, _ = requirement_variables(rule)
        declared = {str(v) for v in as_list(rule.get("required_variables"))}
        if union != declared:
            report.error(
                where,
                "required_variables must exactly equal the union of the four requirement "
                f"blocks; missing={sorted(union - declared)}, extra={sorted(declared - union)}",
            )
        comp = rule.get("computability") or {}
        require_keys(report, f"{where}.computability", comp, ["status", "blockers"])
        report.require(
            comp.get("status") in COMPUTABILITY,
            f"{where}.computability",
            f"status must be one of {sorted(COMPUTABILITY)}",
        )
        if comp.get("status") == "fully_specified":
            report.require(
                not as_list(comp.get("blockers")),
                f"{where}.computability",
                "fully_specified rule cannot carry blockers",
            )
        for j, source_ref in enumerate(as_list(rule.get("source_refs"))):
            ref_where = f"{where}.source_refs[{j}]"
            require_keys(
                report,
                ref_where,
                source_ref,
                ["source_id", "anchor", "fact_status", "paraphrase"],
            )
            report.require(
                source_ref.get("fact_status") in FACT_STATUSES,
                ref_where,
                f"fact_status must be one of {sorted(FACT_STATUSES)}",
            )
            source_id = source_ref.get("source_id")
            report.require(
                source_id in source_index,
                ref_where,
                f"unknown source_id {source_id!r}",
            )
            source = source_index.get(source_id) or {}
            if source.get("status") != "version_bound":
                report.require(
                    comp.get("status") != "fully_specified",
                    ref_where,
                    "a rule backed by an unbound/context source cannot be fully_specified",
                )
    return index, doc


def validate_variables(
    root: Path, report: Report
) -> tuple[dict[str, dict], dict[str, Any]]:
    doc = load_yaml(root / "variable_inventory.yaml", report)
    variables = as_list(doc.get("variables"))
    report.require(bool(variables), "variable_inventory", "variables must be non-empty")
    index: dict[str, dict] = {}
    for i, variable in enumerate(variables):
        where = f"variable_inventory.variables[{i}]"
        require_keys(
            report,
            where,
            variable,
            [
                "variable_id",
                "label",
                "roles",
                "datatype",
                "value_domain",
                "temporal_meaning",
                "mapping_level",
                "registry_mapping",
                "extraction",
                "source_documents",
                "excluded_sources",
                "missingness_semantics",
                "conformance",
            ],
        )
        variable_id = str(variable.get("variable_id") or "")
        if variable_id in index:
            report.error(where, f"duplicate variable_id {variable_id!r}")
        index[variable_id] = variable
        report.require(
            variable.get("mapping_level") in MAPPING_LEVELS,
            where,
            f"mapping_level must be one of {sorted(MAPPING_LEVELS)}",
        )
        conformance = variable.get("conformance") or {}
        require_keys(report, f"{where}.conformance", conformance, ["verdict", "issues"])
        report.require(
            conformance.get("verdict") in SPEC_VERDICTS,
            f"{where}.conformance",
            f"verdict must be one of {sorted(SPEC_VERDICTS)}",
        )
        registry = variable.get("registry_mapping") or {}
        if variable.get("mapping_level") in {"registry_direct", "registry_coarsened"}:
            require_keys(
                report,
                f"{where}.registry_mapping",
                registry,
                ["standard", "item_name", "item_number", "xml_id", "effective_years"],
            )
        extraction = variable.get("extraction") or {}
        require_keys(report, f"{where}.extraction", extraction, ["spec_id", "field", "status"])
    return index, doc


def enforced_spec_paths(spec: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for field in as_list(spec.get("fields")):
        name = field.get("name", "?")
        if field.get("format"):
            paths.append(f"fields[{name}].format")
        if field.get("allowable_values"):
            paths.append(f"fields[{name}].allowable_values")
    for check in as_list(spec.get("answer_checks")):
        nos_values = as_list(check.get("nos_values"))
        suffix = f".{nos_values[0]}" if nos_values else ""
        paths.append(
            f"answer_checks[{check.get('field', '?')}.{check.get('kind', 'not_less_specific')}{suffix}]"
        )
    negative = ((spec.get("proof_obligation") or {}).get("for_negative") or {})
    for key in ("required_coverage", "required_keywords"):
        if negative.get(key):
            paths.append(f"proof_obligation.for_negative.{key}")
    for key in (negative.get("gate") or {}):
        paths.append(f"proof_obligation.for_negative.gate.{key}")
    holders: list[tuple[str, dict]] = [("proof_obligation.for_negative", negative)]
    for claim in as_list(negative.get("claims")):
        holders.append((f"proof_obligation.for_negative.claims[{claim.get('id')}]", claim))
    for prefix, holder in holders:
        for stratum in as_list(holder.get("strata")):
            base = f"{prefix}.strata[{stratum.get('name')}]"
            for key in ("match", "partition_by", "establishes", "required_keywords", "min_sample", "min_sample_of_misses"):
                if stratum.get(key) not in (None, [], {}):
                    paths.append(f"{base}.{key}")
    return paths


def validate_specs(
    root: Path, source_index: dict[str, dict], report: Report
) -> tuple[dict[str, dict], dict[tuple[str, str], dict]]:
    spec_dir = root / "candidate_specs"
    if not spec_dir.is_dir():
        report.error(str(spec_dir), "required directory is missing")
        return {}, {}
    specs: dict[str, dict] = {}
    fields: dict[tuple[str, str], dict] = {}
    global_fields: dict[str, str] = {}
    for path in sorted(spec_dir.glob("*.yaml")):
        spec = load_yaml(path, report)
        where = f"candidate_specs/{path.name}"
        require_keys(
            report,
            where,
            spec,
            [
                "spec_id",
                "spec_version",
                "source_authority",
                "provenance",
                "question",
                "applicability_guard",
                "fields",
                "decision_rule",
                "evidence_rules",
                "conflict_rules",
                "proof_obligation",
                "abstention",
                "boundary_cases",
                "search_hints",
            ],
        )
        spec_id = str(spec.get("spec_id") or "")
        if spec_id in specs:
            report.error(where, f"duplicate spec_id {spec_id!r}")
        specs[spec_id] = spec
        authority = spec.get("source_authority") or {}
        require_keys(report, f"{where}.source_authority", authority, ["document", "items", "source_refs"])
        for source_id in as_list(authority.get("source_refs")):
            report.require(
                source_id in source_index,
                f"{where}.source_authority",
                f"unknown source_id {source_id!r}",
            )
        field_names: list[str] = []
        for i, field in enumerate(as_list(spec.get("fields"))):
            field_where = f"{where}.fields[{i}]"
            require_keys(report, field_where, field, ["name", "type", "nullable", "description"])
            name = str(field.get("name") or "")
            field_names.append(name)
            fields[(spec_id, name)] = field
            if name in global_fields:
                report.error(
                    field_where,
                    f"field {name!r} is also produced by {global_fields[name]}; one canonical field must have one owner",
                )
            global_fields[name] = spec_id
            if field.get("format"):
                try:
                    re.compile(str(field["format"]))
                except re.error as exc:
                    report.error(field_where, f"format is not a valid Python regex: {exc}")
            report.require(
                not (field.get("format") and field.get("allowable_values")),
                field_where,
                "format and allowable_values must not both be set",
            )
            report.require(
                bool(field.get("format") or field.get("allowable_values") or field.get("type") in {"date", "boolean", "integer", "number"}),
                field_where,
                "string field has no enforceable value domain",
            )
        for name in duplicates(field_names):
            report.error(where, f"duplicate field name {name!r}")
        evidence = spec.get("evidence_rules") or {}
        require_keys(report, f"{where}.evidence_rules", evidence, ["counts_as_evidence", "does_not_count"])
        proof = spec.get("proof_obligation") or {}
        require_keys(report, f"{where}.proof_obligation", proof, ["for_positive", "for_negative"])
        negative = proof.get("for_negative") or {}
        require_keys(report, f"{where}.proof_obligation.for_negative", negative, ["mode", "statement"])
        abstention = spec.get("abstention") or {}
        require_keys(
            report,
            f"{where}.abstention",
            abstention,
            ["EVIDENCE_INSUFFICIENT", "SPEC_INSUFFICIENT"],
        )
        search_hints = {str(x).casefold() for x in as_list(spec.get("search_hints"))}
        required_keywords = {str(x).casefold() for x in as_list(negative.get("required_keywords"))}
        missing_hints = required_keywords - search_hints
        if missing_hints:
            report.error(
                f"{where}.search_hints",
                f"required keywords are not reachable as exact search hints: {sorted(missing_hints)}",
            )
        for stratum in as_list(negative.get("strata")):
            unknown = set(as_list(stratum.get("establishes"))) - set(field_names)
            if unknown:
                report.error(
                    f"{where}.proof_obligation",
                    f"stratum {stratum.get('name')!r} establishes unknown fields {sorted(unknown)}",
                )
        enforced = enforced_spec_paths(spec)
        if duplicates(enforced):
            report.error(where, f"enforced paths are not uniquely addressable: {duplicates(enforced)}")
        provenance = as_list(spec.get("provenance"))
        records = {str(rec.get("element")): rec for rec in provenance}
        if len(records) != len(provenance):
            report.error(where, "duplicate provenance element")
        missing = sorted(set(enforced) - set(records))
        stale = sorted(set(records) - set(enforced))
        if missing:
            report.error(where, f"enforced elements missing provenance: {missing}")
        if stale:
            report.error(where, f"provenance records name non-enforced elements: {stale}")
        for element, record in records.items():
            rec_where = f"{where}.provenance[{element}]"
            require_keys(report, rec_where, record, ["element", "origin", "basis", "status"])
            report.require(
                record.get("origin") in ORIGINS,
                rec_where,
                f"origin must be one of {sorted(ORIGINS)}",
            )
            if record.get("origin") == "model_authored":
                report.require(
                    "no external source" in str(record.get("basis") or "").casefold(),
                    rec_where,
                    "model_authored basis must contain 'no external source'",
                )
    report.require(bool(specs), "candidate_specs", "at least one candidate spec is required")
    return specs, fields


def validate_variable_links(
    variable_index: dict[str, dict],
    rule_index: dict[str, dict],
    specs: dict[str, dict],
    spec_fields: dict[tuple[str, str], dict],
    report: Report,
) -> None:
    required = set()
    for rule in rule_index.values():
        required.update(str(v) for v in as_list(rule.get("required_variables")))
    missing = sorted(required - set(variable_index))
    if missing:
        report.error("variable_inventory", f"rule-required variables are undeclared: {missing}")
    for variable_id, variable in variable_index.items():
        extraction = variable.get("extraction") or {}
        spec_id = str(extraction.get("spec_id") or "")
        field = str(extraction.get("field") or "")
        status = extraction.get("status")
        where = f"variable_inventory[{variable_id}].extraction"
        if status == "specified":
            report.require(spec_id in specs, where, f"unknown candidate spec {spec_id!r}")
            report.require(
                (spec_id, field) in spec_fields,
                where,
                f"field {field!r} is not produced by {spec_id!r}",
            )
        elif status == "existing_project_spec":
            report.warn(where, "existing project spec was not loaded by the portable validator")
        elif status in {"planned", "not_specifiable"}:
            pass
        else:
            report.error(where, "status must be specified, existing_project_spec, planned, or not_specifiable")


def canonical_hash(value: Any) -> str:
    blob = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def validate_variable_contracts(
    root: Path,
    variable_index: dict[str, dict],
    specs: dict[str, dict],
    spec_fields: dict[tuple[str, str], dict],
    report: Report,
) -> None:
    contract_dir = root / "variable_contracts"
    report.require(contract_dir.is_dir(), str(contract_dir), "required directory is missing")
    contracts: dict[str, dict[str, Any]] = {}
    bindings: dict[tuple[str, str], str] = {}
    for path in sorted(contract_dir.glob("*.yaml")) if contract_dir.is_dir() else []:
        contract = load_yaml(path, report)
        where = f"variable_contracts/{path.name}"
        require_keys(
            report,
            where,
            contract,
            [
                "contract_schema_version",
                "contract_id",
                "contract_version",
                "authoring_status",
                "clinical_use",
                "canonical_variable_id",
                "field_contract",
                "registry_mapping",
                "evidence_coverage",
                "question",
                "applicability_guard",
                "evidence_contract",
                "missingness_semantics",
                "abstention",
                "execution_binding",
                "contract_hash",
            ],
        )
        variable_id = str(contract.get("canonical_variable_id") or "")
        if variable_id in contracts:
            report.error(where, f"duplicate contract for canonical variable {variable_id!r}")
        contracts[variable_id] = contract
        report.require(variable_id in variable_index, where, f"unknown canonical variable {variable_id!r}")
        report.require(
            path.name == f"{contract.get('contract_id')}.yaml",
            where,
            "filename must equal <contract_id>.yaml",
        )
        report.require(
            contract.get("clinical_use") == "NOT_FOR_CLINICAL_USE",
            where,
            "clinical_use must be NOT_FOR_CLINICAL_USE",
        )
        expected_hash_input = dict(contract)
        recorded_hash = expected_hash_input.pop("contract_hash", None)
        report.require(
            recorded_hash == canonical_hash(expected_hash_input),
            where,
            "contract_hash does not match canonical contract content",
        )
        binding = contract.get("execution_binding") or {}
        require_keys(report, f"{where}.execution_binding", binding, ["execution_group_id", "output_field"])
        key = (
            str(binding.get("execution_group_id") or ""),
            str(binding.get("output_field") or ""),
        )
        if key in bindings:
            report.error(where, f"execution binding {key!r} is already owned by {bindings[key]!r}")
        bindings[key] = variable_id
        report.require(key[0] in specs, where, f"unknown execution group {key[0]!r}")
        report.require(key in spec_fields, where, f"unknown materialized field {key!r}")
        inventory = variable_index.get(variable_id) or {}
        extraction = inventory.get("extraction") or {}
        report.require(
            key == (str(extraction.get("spec_id") or ""), str(extraction.get("field") or "")),
            where,
            "contract binding disagrees with variable_inventory.extraction",
        )
        report.require(
            contract.get("field_contract") == spec_fields.get(key),
            where,
            "field_contract differs from the grouped runtime field",
        )
        inventory_contract = inventory.get("variable_contract") or {}
        report.require(
            inventory_contract.get("contract_id") == contract.get("contract_id")
            and inventory_contract.get("contract_hash") == recorded_hash,
            where,
            "variable_inventory.variable_contract lineage does not match",
        )

    expected = set(variable_index)
    report.require(
        set(contracts) == expected,
        "variable_contracts",
        f"must contain exactly one contract per canonical variable; "
        f"missing={sorted(expected - set(contracts))}, extra={sorted(set(contracts) - expected)}",
    )

    groups = load_yaml(root / "execution_groups.yaml", report)
    group_rows = {
        str(row.get("execution_group_id") or ""): row
        for row in as_list(groups.get("groups"))
    }
    report.require(set(group_rows) == set(specs), "execution_groups", "must enumerate every grouped runtime spec")
    for spec_id, spec in specs.items():
        declared = group_rows.get(spec_id) or {}
        members = {
            (spec_id, str(row.get("output_field") or ""))
            for row in as_list(declared.get("member_contracts"))
        }
        actual = {key for key in bindings if key[0] == spec_id}
        report.require(
            members == actual,
            f"execution_groups[{spec_id}]",
            f"member bindings mismatch; expected={sorted(actual)}, got={sorted(members)}",
        )
        unbound = {
            str(row.get("field") or "")
            for row in as_list(declared.get("unbound_runtime_fields"))
            if nonempty((row or {}).get("classification")) and nonempty((row or {}).get("rationale"))
        }
        all_fields = {str(field.get("name") or "") for field in as_list(spec.get("fields"))}
        report.require(
            {field for group, field in actual if group == spec_id} | unbound == all_fields,
            f"execution_groups[{spec_id}]",
            "every runtime field must be contract-bound or explicitly classified as supporting/projection-only",
        )

    manifest = load_yaml(root / "execution_manifest.yaml", report)
    report.require(
        manifest.get("variable_contract_count") == len(variable_index),
        "execution_manifest",
        f"variable_contract_count must be {len(variable_index)}",
    )
    report.require(
        manifest.get("execution_group_count") == len(specs),
        "execution_manifest",
        f"execution_group_count must be {len(specs)}",
    )
    materialized_dir = root / "materialized_specs"
    report.require(materialized_dir.is_dir(), str(materialized_dir), "required directory is missing")
    for spec_id, source_spec in specs.items():
        materialized = load_yaml(materialized_dir / f"{spec_id}.yaml", report)
        report.require(
            materialized == source_spec,
            f"materialized_specs/{spec_id}.yaml",
            "materialized grouped spec is stale relative to candidate source template",
        )


def validate_gaps(
    root: Path,
    manifest: dict[str, Any],
    rule_index: dict[str, dict],
    variable_index: dict[str, dict],
    report: Report,
) -> dict[str, Any]:
    doc = load_yaml(root / "gap_assessment.yaml", report)
    reviewer_view = root / "gap_assessment.md"
    report.require(
        reviewer_view.is_file(),
        "gap_assessment.md",
        "required reviewer view is missing",
    )
    if reviewer_view.is_file():
        reviewer_text = reviewer_view.read_text(encoding="utf-8")
        report.require(
            "NOT FOR CLINICAL USE" in reviewer_text,
            "gap_assessment.md",
            "reviewer view must disclose NOT FOR CLINICAL USE",
        )
    require_keys(
        report,
        "gap_assessment",
        doc,
        ["bundle_id", "data_profile", "summary", "variable_coverage"],
    )
    report.require(
        doc.get("bundle_id") == manifest.get("bundle_id"),
        "gap_assessment",
        "bundle_id does not match manifest",
    )
    rows = as_list(doc.get("variable_coverage"))
    actual_variables: set[str] = set()
    gap_class_counts: Counter[str] = Counter()
    for i, row in enumerate(rows):
        where = f"gap_assessment.variable_coverage[{i}]"
        require_keys(
            report,
            where,
            row,
            [
                "variable_id",
                "rule_uses",
                "registry_coverage",
                "spec_coverage",
                "data_coverage",
                "gap_classes",
                "severity",
                "remediation",
            ],
        )
        variable_id = str(row.get("variable_id") or "")
        if variable_id in actual_variables:
            report.error(where, f"duplicate variable coverage row {variable_id!r}")
        actual_variables.add(variable_id)
        report.require(variable_id in variable_index, where, f"unknown variable_id {variable_id!r}")
        actual_uses: dict[str, set[str]] = {}
        for j, use in enumerate(as_list(row.get("rule_uses"))):
            use_where = f"{where}.rule_uses[{j}]"
            require_keys(report, use_where, use, ["rule_id", "requirement_roles"])
            rule_id = str(use.get("rule_id") or "")
            roles = set(as_list(use.get("requirement_roles")))
            report.require(
                rule_id in rule_index,
                use_where,
                f"unknown rule_id {rule_id!r}",
            )
            report.require(
                roles <= REQUIREMENT_ROLES,
                use_where,
                f"unknown requirement roles {sorted(roles - REQUIREMENT_ROLES)}",
            )
            if rule_id in actual_uses:
                report.error(use_where, f"duplicate rule use {rule_id!r}")
            actual_uses[rule_id] = roles
        expected_uses: dict[str, set[str]] = {}
        for rule_id, rule in rule_index.items():
            _, by_role = requirement_variables(rule)
            roles = {role for role, values in by_role.items() if variable_id in values}
            if roles:
                expected_uses[rule_id] = roles
        if actual_uses != expected_uses:
            report.error(
                where,
                "rule_uses mismatch: expected "
                f"{ {key: sorted(value) for key, value in expected_uses.items()} }",
            )
        report.require(
            row.get("data_coverage") in DATA_COVERAGE,
            where,
            f"data_coverage must be one of {sorted(DATA_COVERAGE)}",
        )
        gaps = set(as_list(row.get("gap_classes")))
        gap_class_counts.update(gaps)
        report.require(
            gaps <= GAP_CLASSES,
            where,
            f"unknown gap classes {sorted(gaps - GAP_CLASSES)}",
        )
        report.require(
            row.get("severity") in SEVERITIES,
            where,
            f"severity must be one of {sorted(SEVERITIES)}",
        )
        mapping_level = (variable_index.get(variable_id) or {}).get("mapping_level")
        expected_registry_coverage = {
            "registry_direct": "direct",
            "registry_coarsened": "coarsened",
            "chart_extension": "none",
            "derived": "deterministic_or_clinical_derivation",
            "outside_current_sources": "none",
        }.get(mapping_level)
        report.require(
            row.get("registry_coverage") == expected_registry_coverage,
            where,
            "registry_coverage disagrees with variable_inventory mapping_level: "
            f"expected {expected_registry_coverage!r}",
        )
        if row.get("data_coverage") == "NOT_ASSESSED":
            report.require(
                "DATA_NOT_PROFILED" in gaps,
                where,
                "NOT_ASSESSED data coverage requires DATA_NOT_PROFILED gap",
            )
        verdict = ((variable_index.get(variable_id) or {}).get("conformance") or {}).get("verdict")
        expected_spec_coverage = (
            "candidate_conformant"
            if verdict == "conformant_candidate"
            else "candidate_needs_revision"
        )
        report.require(
            row.get("spec_coverage") == expected_spec_coverage,
            where,
            "spec_coverage disagrees with variable_inventory conformance: "
            f"expected {expected_spec_coverage!r}",
        )
    expected_variables = {
        str(variable_id)
        for rule in rule_index.values()
        for variable_id in as_list(rule.get("required_variables"))
    }
    missing = sorted(expected_variables - actual_variables)
    extra = sorted(actual_variables - expected_variables)
    if missing:
        report.error("gap_assessment", f"missing variable coverage rows: {missing}")
    if extra:
        report.error("gap_assessment", f"rows exist for unrequired variables: {extra}")
    summary = doc.get("summary") or {}
    require_keys(
        report,
        "gap_assessment.summary",
        summary,
        [
            "candidate_rules",
            "required_variables",
            "registry_direct",
            "registry_coarsened",
            "derived",
            "chart_extension",
            "conformant_candidates",
            "needs_revision",
            "rule_category_counts",
            "gap_class_counts",
        ],
    )
    category_counts = Counter(str(rule.get("category")) for rule in rule_index.values())
    mapping_counts = Counter(
        str(variable_index[variable_id].get("mapping_level"))
        for variable_id in expected_variables
        if variable_id in variable_index
    )
    verdict_counts = Counter(
        str(((variable_index[variable_id].get("conformance") or {}).get("verdict")))
        for variable_id in expected_variables
        if variable_id in variable_index
    )
    expected_summary_scalars = {
        "candidate_rules": len(rule_index),
        "required_variables": len(expected_variables),
        "registry_direct": mapping_counts["registry_direct"],
        "registry_coarsened": mapping_counts["registry_coarsened"],
        "derived": mapping_counts["derived"],
        "chart_extension": mapping_counts["chart_extension"],
        "conformant_candidates": verdict_counts["conformant_candidate"],
        "needs_revision": verdict_counts["needs_revision"],
    }
    for key, expected in expected_summary_scalars.items():
        report.require(
            summary.get(key) == expected,
            "gap_assessment.summary",
            f"{key} must be {expected}, got {summary.get(key)!r}",
        )
    report.require(
        summary.get("rule_category_counts") == dict(sorted(category_counts.items())),
        "gap_assessment.summary.rule_category_counts",
        f"must equal {dict(sorted(category_counts.items()))}",
    )
    report.require(
        summary.get("gap_class_counts") == dict(sorted(gap_class_counts.items())),
        "gap_assessment.summary.gap_class_counts",
        f"must equal {dict(sorted(gap_class_counts.items()))}",
    )
    profile = doc.get("data_profile") or {}
    if profile.get("status") != "NOT_ASSESSED":
        require_keys(report, "gap_assessment.data_profile", profile, ["profile_id", "dataset_snapshot"])
    return doc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Path to the authoring bundle")
    parser.add_argument(
        "--warnings-as-errors",
        action="store_true",
        help="Return non-zero when warnings are present",
    )
    args = parser.parse_args()
    root = args.bundle.resolve()
    report = Report()
    report.require(root.is_dir(), str(root), "bundle directory does not exist")
    if report.errors:
        print("\n".join(f"ERROR {x}" for x in report.errors))
        return 1

    manifest = validate_manifest(root, report)
    source_index, _ = validate_sources(root, report)
    intake = validate_intake(root, report)
    rule_index, _ = validate_rules(root, source_index, report)
    variable_index, _ = validate_variables(root, report)
    specs, spec_fields = validate_specs(root, source_index, report)
    validate_variable_links(variable_index, rule_index, specs, spec_fields, report)
    validate_variable_contracts(root, variable_index, specs, spec_fields, report)
    validate_gaps(root, manifest, rule_index, variable_index, report)

    for warning in report.warnings:
        print(f"WARNING {warning}")
    for error in report.errors:
        print(f"ERROR {error}")
    print(
        f"Validated {len(as_list(intake.get('candidates')))} universe candidate(s), "
        f"{len(rule_index)} structured seed rule(s), {len(variable_index)} variable contract(s), "
        f"{len(specs)} grouped candidate spec(s): "
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)."
    )
    return 1 if report.errors or (args.warnings_as_errors and report.warnings) else 0


if __name__ == "__main__":
    sys.exit(main())
