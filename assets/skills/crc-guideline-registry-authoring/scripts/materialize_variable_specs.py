#!/usr/bin/env python3
"""Materialize one authoring contract per CRC variable and grouped runtime specs.

Variable contracts own identity, field semantics, registry mapping, evidence, and
missingness. Execution groups own one-pass behavior and compile to the repository's existing
multi-field ExtractionSpec shape. This preserves one-variable review/versioning without
turning 68 variables into 68 chart passes.

Generated contracts and materialized specs remain candidates. Compilation does not create
clinical, registrar, corpus, or runtime approval.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any

import yaml


STOPWORDS = {
    "crc",
    "status",
    "indexed",
    "registry",
    "summary",
    "clinical",
    "treatment",
    "documented",
    "decision",
    "result",
    "date",
    "whether",
}
FORMAT_CANDIDATES = [
    "C180",
    "8140",
    "2026",
    "202601",
    "20260101",
    "cT3",
    "cN1",
    "1.0",
    "10",
    "SP1",
    "FOLFOX",
    "sample",
]


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected mapping")
    return value


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def safe_spec_suffix(variable_id: str) -> str:
    suffix = variable_id.removeprefix("crc.")
    return re.sub(r"[^a-z0-9_]+", "_", suffix.lower()).strip("_")


def keywords(variable: dict[str, Any], field_name: str, parent: dict[str, Any]) -> list[str]:
    text = " ".join(
        [
            str(variable.get("label") or ""),
            field_name.replace("_", " "),
            str((variable.get("registry_mapping") or {}).get("item_name") or ""),
        ]
    )
    words = [
        match.group(0)
        for match in re.finditer(r"[A-Za-z][A-Za-z0-9+-]*", text)
        if match.group(0).casefold() not in STOPWORDS and len(match.group(0)) >= 3
    ]
    preferred: list[str] = []
    for term in as_list(parent.get("search_hints")):
        term_s = str(term)
        if any(word.casefold() in term_s.casefold() or term_s.casefold() in word.casefold() for word in words):
            preferred.append(term_s)
    for word in words:
        if word.casefold() not in {value.casefold() for value in preferred}:
            preferred.append(word)
    if not preferred:
        preferred = [field_name.replace("_", " ")]

    # Runtime matches search terms bidirectionally. Keep a minimal antichain so one term
    # cannot silently discharge another.
    antichain: list[str] = []
    for term in preferred:
        folded = term.casefold()
        if any(folded in old.casefold() or old.casefold() in folded for old in antichain):
            continue
        antichain.append(term)
        if len(antichain) == 3:
            break
    return antichain or [preferred[0]]


def field_provenance(parent: dict[str, Any], field_name: str) -> list[dict[str, Any]]:
    prefix = f"fields[{field_name}]."
    return [
        copy.deepcopy(record)
        for record in as_list(parent.get("provenance"))
        if str((record or {}).get("element") or "").startswith(prefix)
    ]


def filtered_checks(parent: dict[str, Any], field_name: str) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(check)
        for check in as_list(parent.get("answer_checks"))
        if str((check or {}).get("field") or "") == field_name
    ]


def check_provenance(parent: dict[str, Any], checks: list[dict[str, Any]], field_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in as_list(parent.get("provenance")):
        element = str((record or {}).get("element") or "")
        if not element.startswith(f"answer_checks[{field_name}."):
            continue
        records.append(copy.deepcopy(record))
    if checks and not records:
        raise ValueError(f"{parent.get('spec_id')}#{field_name}: answer checks lack provenance")
    return records


def sample_for_field(field: dict[str, Any]) -> tuple[str, str] | None:
    allowable = as_list(field.get("allowable_values"))
    if allowable:
        accepted = str(allowable[0])
        rejected = "__INVALID__"
        if rejected in {str(value) for value in allowable}:
            rejected = "__OUT_OF_DOMAIN__"
        return accepted, rejected
    pattern = field.get("format")
    if not pattern:
        return None
    compiled = re.compile(str(pattern))
    accepted = next((candidate for candidate in FORMAT_CANDIDATES if compiled.fullmatch(candidate)), None)
    if accepted is None:
        raise ValueError(f"{field.get('name')}: no generated example satisfies {pattern!r}")
    rejected = next(
        candidate
        for candidate in ("", "__INVALID__", " ", "C18", "20261340")
        if not compiled.fullmatch(candidate)
    )
    return accepted, rejected


def canonical_hash(value: Any) -> str:
    blob = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def inferred_coverage(variable: dict[str, Any]) -> dict[str, Any]:
    mapping_level = str(variable.get("mapping_level") or "")
    named_registry = str((variable.get("registry_mapping") or {}).get("standard") or "").lower() not in {
        "",
        "none",
        "null",
    }
    semantic = {
        "registry_direct": "exact",
        "registry_coarsened": "coarsened",
        "chart_extension": "absent",
        "derived": "coarsened" if named_registry else "absent",
        "outside_current_sources": "absent",
    }.get(mapping_level, "unknown")
    temporal = {
        "registry_direct": "exact",
        "registry_coarsened": "conditional",
        "chart_extension": "misaligned",
        "derived": "conditional",
        "outside_current_sources": "unknown",
    }.get(mapping_level, "unknown")
    projection = {
        "registry_direct": "direct",
        "registry_coarsened": "coarsened",
        "chart_extension": "coarsened" if named_registry else "none",
        "derived": "coarsened" if named_registry else "none",
        "outside_current_sources": "none",
    }.get(mapping_level, "unknown")
    return {
        "registry_to_rule": {
            "semantic": semantic,
            "temporal": temporal,
            "decision_boundary_preserved": semantic == "exact" and temporal == "exact",
            "status": "inferred_from_mapping_level_requires_use_specific_review",
        },
        "canonical_to_registry": {
            "level": projection,
            "transformation_id": None if projection != "coarsened" else "PENDING_VERSIONED_PROJECTION",
            "status": "candidate",
        },
        "observed_availability": {
            "status": "NOT_ASSESSED",
            "profile_id": None,
        },
    }


def build_contract(variable: dict[str, Any], parent: dict[str, Any], field: dict[str, Any]) -> dict[str, Any]:
    variable_id = str(variable["variable_id"])
    field_name = str(field["name"])
    terms = keywords(variable, field_name, parent)
    mapping_level = str(variable.get("mapping_level") or "")
    contract_kind = {
        "registry_direct": "registry_projection_and_chart_extraction",
        "registry_coarsened": "canonical_extraction_with_lossy_registry_projection",
        "chart_extension": "chart_extraction",
        "derived": "derivation",
        "outside_current_sources": "blocked_source_contract",
    }.get(mapping_level, "candidate_extraction")
    contract = {
        "contract_schema_version": "1.0",
        "contract_id": f"CRC.VAR.{safe_spec_suffix(variable_id)}",
        "contract_version": "0.1.0",
        "authoring_status": "candidate",
        "clinical_use": "NOT_FOR_CLINICAL_USE",
        "canonical_variable_id": variable_id,
        "contract_kind": contract_kind,
        "label": variable.get("label"),
        "roles": copy.deepcopy(variable.get("roles") or []),
        "temporal_meaning": variable.get("temporal_meaning"),
        "field_contract": copy.deepcopy(field),
        "registry_mapping": copy.deepcopy(variable.get("registry_mapping") or {}),
        "evidence_coverage": copy.deepcopy(variable.get("evidence_coverage") or inferred_coverage(variable)),
        "source_authority": copy.deepcopy(parent.get("source_authority") or {}),
        "question": (
            f"What is the {variable.get('label')} for the indexed CRC tumor or decision "
            f"at {variable.get('temporal_meaning')}?"
        ),
        "applicability_guard": copy.deepcopy(parent.get("applicability_guard") or {}),
        "evidence_contract": {
            "establishing_source_documents": copy.deepcopy(variable.get("source_documents") or []),
            "excluded_sources": copy.deepcopy(variable.get("excluded_sources") or []),
            "positive_witness": (
                f"A tumor-linked source appropriate to {variable.get('temporal_meaning')} "
                f"establishes {field_name}."
            ),
            "negative_or_unknown_proof": {
                "required_coverage": [
                    "Review every available establishing source class for this variable in the indexed time window."
                ],
                "required_keywords": terms,
                "statement": str(variable.get("missingness_semantics") or ""),
            },
            "conflict_policy": (
                "Retain conflicting citations, apply the reviewed temporal/source precedence, "
                "and abstain when the conflict remains unresolved."
            ),
        },
        "missingness_semantics": variable.get("missingness_semantics"),
        "abstention": {
            "EVIDENCE_INSUFFICIENT": f"The available indexed evidence does not establish {field_name}.",
            "SPEC_INSUFFICIENT": f"The requested meaning of {field_name} is outside this candidate contract.",
        },
        "execution_binding": {
            "execution_group_id": parent.get("spec_id"),
            "output_field": field_name,
            "runtime_shape": "multi_field_ExtractionSpec",
            "runtime_status": "candidate_materialized_group",
        },
        "field_provenance": field_provenance(parent, field_name),
        "conformance": copy.deepcopy(variable.get("conformance") or {}),
        "review_requirements": [clinical_owner(mapping_level), "registrar", "engineer"],
    }
    contract["contract_hash"] = canonical_hash(contract)
    return contract


def clinical_owner(mapping_level: str) -> str:
    if mapping_level in {"registry_direct", "registry_coarsened"}:
        return "clinical_domain_reviewer"
    return "clinical_domain_reviewer"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--replace", action="store_true", help="replace existing generated contract/materialized directories")
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    inventory = load_yaml(bundle / "variable_inventory.yaml")
    parents = {
        str(spec.get("spec_id")): spec
        for path in sorted((bundle / "candidate_specs").glob("*.yaml"))
        for spec in [load_yaml(path)]
    }
    parent_fields = {
        (spec_id, str(field.get("name"))): field
        for spec_id, spec in parents.items()
        for field in as_list(spec.get("fields"))
    }
    variables = as_list(inventory.get("variables"))
    out_dir = bundle / "variable_contracts"
    materialized_dir = bundle / "materialized_specs"
    if out_dir.exists():
        if not args.replace:
            raise SystemExit(f"{out_dir} already exists; pass --replace to rebuild generated candidates")
        shutil.rmtree(out_dir)
    if materialized_dir.exists():
        if not args.replace:
            raise SystemExit(f"{materialized_dir} already exists; pass --replace to rebuild")
        shutil.rmtree(materialized_dir)
    out_dir.mkdir(parents=True)
    materialized_dir.mkdir(parents=True)

    groups: dict[str, dict[str, Any]] = {}
    index_rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    seen_fields: set[str] = set()
    for variable in variables:
        variable_id = str(variable.get("variable_id") or "")
        extraction = variable.get("extraction") or {}
        parent_id = str(extraction.get("spec_id") or "")
        field_name = str(extraction.get("field") or "")
        parent = parents.get(parent_id)
        field = parent_fields.get((parent_id, field_name))
        if not parent or not field:
            raise ValueError(f"{variable_id}: cannot resolve {parent_id}#{field_name}")
        if field_name in seen_fields:
            raise ValueError(f"{variable_id}: field {field_name!r} already has a variable owner")
        seen_fields.add(field_name)
        contract = build_contract(variable, parent, field)
        contract_path = out_dir / f"{contract['contract_id']}.yaml"
        contract_path.write_text(
            yaml.safe_dump(contract, sort_keys=False, allow_unicode=True, width=120),
            encoding="utf-8",
        )
        variable["variable_contract"] = {
            "contract_id": contract["contract_id"],
            "contract_version": contract["contract_version"],
            "contract_hash": contract["contract_hash"],
            "path": f"variable_contracts/{contract['contract_id']}.yaml",
        }
        variable["evidence_coverage"] = contract["evidence_coverage"]
        index_rows.append(
            {
                "variable_id": variable_id,
                "contract_id": contract["contract_id"],
                "contract_hash": contract["contract_hash"],
                "execution_group_id": parent_id,
                "output_field": field_name,
            }
        )

        group = groups.setdefault(
            parent_id,
            {
                "execution_group_id": parent_id,
                "source_template": f"candidate_specs/{parent_id}.yaml",
                "materialized_spec": f"materialized_specs/{parent_id}.yaml",
                "runtime_status": "candidate_loadable",
                "rationale": "Members share the parent template's evidence universe, timing family, or owner; variable contracts remain independently hashed and reviewed.",
                "member_contracts": [],
            },
        )
        group["member_contracts"].append(
            {"contract_id": contract["contract_id"], "output_field": field_name}
        )
        sample = sample_for_field(field)
        if sample:
            examples.append(
                {
                    "contract_id": contract["contract_id"],
                    "execution_group_id": parent_id,
                    "field": field_name,
                    "accept": sample[0],
                    "reject": sample[1],
                }
            )

    (bundle / "variable_inventory.yaml").write_text(
        yaml.safe_dump(inventory, sort_keys=False, allow_unicode=True, width=160),
        encoding="utf-8",
    )
    # Materialized runtime specs retain the existing one-pass shape. Their content is copied
    # exactly; the manifest carries variable-contract lineage until a compiler embeds hashes
    # into per-field extension metadata.
    for parent_id, parent in parents.items():
        (materialized_dir / f"{parent_id}.yaml").write_text(
            yaml.safe_dump(parent, sort_keys=False, allow_unicode=True, width=120),
            encoding="utf-8",
        )
        bound_fields = {
            member["output_field"] for member in groups.get(parent_id, {}).get("member_contracts", [])
        }
        all_fields = {str(field.get("name")) for field in as_list(parent.get("fields"))}
        groups[parent_id]["unbound_runtime_fields"] = [
            {
                "field": field_name,
                "classification": "supporting_or_projection_only_review_required",
                "rationale": "The grouped runtime template produces this field, but no canonical variable contract currently owns it.",
            }
            for field_name in sorted(all_fields - bound_fields)
        ]
        groups[parent_id]["materialized_spec_content_hash"] = canonical_hash(parent)

    (bundle / "execution_groups.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "bundle_id": inventory.get("bundle_id"),
                "clinical_use": "NOT_FOR_CLINICAL_USE",
                "groups": list(groups.values()),
            },
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
        encoding="utf-8",
    )
    (bundle / "variable_index.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "bundle_id": inventory.get("bundle_id"),
                "variables": index_rows,
            },
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
        encoding="utf-8",
    )
    (bundle / "execution_manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "bundle_id": inventory.get("bundle_id"),
                "materialization_status": "candidate",
                "variable_contract_count": len(index_rows),
                "execution_group_count": len(groups),
                "groups": [
                    {
                        "execution_group_id": group["execution_group_id"],
                        "materialized_spec": group["materialized_spec"],
                        "materialized_spec_content_hash": group["materialized_spec_content_hash"],
                        "member_contract_hashes": [
                            {
                                "contract_id": row["contract_id"],
                                "contract_hash": next(
                                    item["contract_hash"]
                                    for item in index_rows
                                    if item["contract_id"] == row["contract_id"]
                                ),
                            }
                            for row in group["member_contracts"]
                        ],
                    }
                    for group in groups.values()
                ],
            },
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
        encoding="utf-8",
    )
    (bundle / "variable_spec_examples.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "bundle_id": inventory.get("bundle_id"),
                "examples": examples,
            },
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
        encoding="utf-8",
    )
    print(
        f"Materialized {len(variables)} one-variable contract(s), "
        f"{len(groups)} grouped runtime spec(s), and {len(examples)} domain example(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
