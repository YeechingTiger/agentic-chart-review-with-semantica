#!/usr/bin/env python3
"""Build the full-universe CRC critical-variable concept inventory.

The rule-universe vocabulary is source-authored.  Similar-looking tokens are not
automatically synonyms: ``braf_status`` is not silently collapsed into
``crc.braf_v600e_status``, and a generic treatment date is not silently collapsed
into the CoC first-course date. Mappings therefore come from the explicit
candidate-scoped normalization artifact when it is supplied. The curated table
below is retained only as the pre-normalization fallback.

By default the command refuses to generate an artifact until the declared rule
universe is complete.  ``--allow-incomplete`` exists only for development and
must not be used to publish the checked-in inventory.
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml


HERE = Path(__file__).resolve().parent
DEFAULT_RULE_UNIVERSE = HERE / "rule_universe.yaml"
DEFAULT_VARIABLE_INVENTORY = HERE.parent / "variable_inventory.yaml"
DEFAULT_OUTPUT = HERE / "variable_concept_inventory.yaml"
DEFAULT_FULL_BINDINGS = HERE.parent / "normalization" / "concept_bindings.yaml"
DEFAULT_FULL_VARIABLE_INVENTORY = HERE.parent / "normalization" / "canonical_variables.yaml"
DEFAULT_FULL_CONTRACT_DIR = HERE.parent / "normalization" / "variable_contracts"


# Every mapping is an affirmative clinical-semantic assertion.  Do not replace
# this table with name normalization, edit distance, substring matching, or a
# generated alias list.
CURATED_EXISTING_MAPPINGS: dict[str, dict[str, Any]] = {
    "assay_method": {
        "canonical_variable_ids": ["crc.molecular_assay_method"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the method used for the indexed molecular assay.",
    },
    "braf_v600e": {
        "canonical_variable_ids": ["crc.braf_v600e_status"],
        "mapping_relation": "exact",
        "rationale": "The source concept explicitly names the BRAF V600E result, not generic BRAF status.",
    },
    "clinical_T": {
        "canonical_variable_ids": ["crc.clinical_t"],
        "mapping_relation": "exact",
        "rationale": "The source use is pretreatment clinical T, despite its source-specific capitalization.",
    },
    "concordance_status": {
        "canonical_variable_ids": ["crc.mmr_msi_discordance_status"],
        "mapping_relation": "exact",
        "rationale": "The use concerns the relationship between the indexed MMR-IHC and MSI methods.",
    },
    "discordance_flag": {
        "canonical_variable_ids": ["crc.mmr_msi_discordance_status"],
        "mapping_relation": "exact",
        "rationale": "The use explicitly identifies discordance between MMR-IHC and MSI results.",
    },
    "cN": {
        "canonical_variable_ids": ["crc.clinical_n"],
        "mapping_relation": "exact",
        "rationale": "The source concept is pretreatment clinical N for rectal-cancer risk and treatment selection.",
    },
    "cT": {
        "canonical_variable_ids": ["crc.clinical_t"],
        "mapping_relation": "exact",
        "rationale": "The source concept is pretreatment clinical T for rectal-cancer risk and treatment selection.",
    },
    "ecog": {
        "canonical_variable_ids": ["crc.ecog_performance_status"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the decision-time ECOG performance status.",
    },
    "emvi": {
        "canonical_variable_ids": ["crc.emvi_status"],
        "mapping_relation": "exact",
        "rationale": "The source uses are pretreatment MRI extramural venous invasion status.",
    },
    "extramural_depth": {
        "canonical_variable_ids": ["crc.extramural_depth_mm"],
        "mapping_relation": "exact",
        "rationale": "The source concept is pretreatment MRI extramural depth; the canonical variable fixes millimetres.",
    },
    "four_mmr_proteins": {
        "canonical_variable_ids": [
            "crc.mmr_mlh1",
            "crc.mmr_pms2",
            "crc.mmr_msh2",
            "crc.mmr_msh6",
        ],
        "mapping_relation": "composite",
        "rationale": "The source concept explicitly requires the four individual MMR-IHC protein results.",
    },
    "germline_result": {
        "canonical_variable_ids": ["crc.germline_result"],
        "mapping_relation": "exact",
        "rationale": "The source and canonical concepts both represent the high-level germline test result.",
    },
    "histology": {
        "canonical_variable_ids": ["crc.histology"],
        "mapping_relation": "exact",
        "rationale": "The source uses require colorectal histologic identity or confirmation.",
    },
    "ici_contraindication": {
        "canonical_variable_ids": ["crc.ici_contraindication"],
        "mapping_relation": "exact",
        "rationale": "The source concept is an ICI-specific treatment contraindication.",
    },
    "ici_agent": {
        "canonical_variable_ids": ["crc.ici_agent"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the indexed immune-checkpoint agent or combination.",
    },
    "intersphincteric_plane": {
        "canonical_variable_ids": ["crc.intersphincteric_plane_status"],
        "mapping_relation": "exact",
        "rationale": "The source concept is pretreatment intersphincteric-plane involvement status.",
    },
    "mlh1": {
        "canonical_variable_ids": ["crc.mmr_mlh1"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the indexed MLH1 IHC protein result.",
    },
    "mlh1_methylation": {
        "canonical_variable_ids": ["crc.mlh1_promoter_methylation_status"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the MLH1 promoter methylation result used in Lynch triage.",
    },
    "molecular_report_date": {
        "canonical_variable_ids": ["crc.molecular_result_date"],
        "mapping_relation": "exact",
        "rationale": "The source clock ends at final molecular-report availability.",
    },
    "mdt": {
        "canonical_variable_ids": ["crc.multidisciplinary_review"],
        "mapping_relation": "exact",
        "rationale": "The source use requires documented multidisciplinary review before the indexed decision.",
    },
    "metastatic_status": {
        "canonical_variable_ids": ["crc.metastatic_status"],
        "mapping_relation": "exact",
        "rationale": "The source concept is whether metastatic disease is established at the indexed assessment.",
    },
    "mrf": {
        "canonical_variable_ids": ["crc.mrf_status"],
        "mapping_relation": "exact",
        "rationale": "The source concept is pretreatment mesorectal-fascia threatened/involved status.",
    },
    "mrf_status": {
        "canonical_variable_ids": ["crc.mrf_status"],
        "mapping_relation": "exact",
        "rationale": "The source and canonical concepts are pretreatment mesorectal-fascia status.",
    },
    "mri_date": {
        "canonical_variable_ids": ["crc.pelvic_mri_date"],
        "mapping_relation": "exact",
        "rationale": "The source uses are the pretreatment pelvic MRI exam date.",
    },
    "mri_protocol": {
        "canonical_variable_ids": ["crc.rectal_mri_protocol"],
        "mapping_relation": "exact",
        "rationale": "The source concept is use of a dedicated high-resolution rectal MRI protocol.",
    },
    "mri_quality": {
        "canonical_variable_ids": ["crc.rectal_mri_quality"],
        "mapping_relation": "exact",
        "rationale": "The source concept is adequacy of the pretreatment rectal MRI for decision making.",
    },
    "msh2": {
        "canonical_variable_ids": ["crc.mmr_msh2"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the indexed MSH2 IHC protein result.",
    },
    "msh6": {
        "canonical_variable_ids": ["crc.mmr_msh6"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the indexed MSH6 IHC protein result.",
    },
    "msi_result": {
        "canonical_variable_ids": ["crc.msi_status"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the raw indexed MSI assay result.",
    },
    "msi_status": {
        "canonical_variable_ids": ["crc.msi_status"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the raw indexed MSI result used by the rule.",
    },
    "patient_preference": {
        "canonical_variable_ids": ["crc.patient_preference"],
        "mapping_relation": "exact",
        "rationale": "The source concept explicitly represents the patient's preference for the indexed option.",
    },
    "pms2": {
        "canonical_variable_ids": ["crc.mmr_pms2"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the indexed PMS2 IHC protein result.",
    },
    "preference": {
        "canonical_variable_ids": ["crc.patient_preference"],
        "mapping_relation": "exact",
        "rationale": "Across its rule uses, preference is the patient's treatment preference in shared decision making.",
    },
    "primary_sidedness": {
        "canonical_variable_ids": ["crc.primary_tumor_sidedness"],
        "mapping_relation": "exact",
        "rationale": "The source concept is primary-tumor left/right sidedness for treatment selection.",
    },
    "progression_date": {
        "canonical_variable_ids": ["crc.progression_date"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the qualifying progression date before the indexed later-line decision.",
    },
    "regimen": {
        "canonical_variable_ids": ["crc.systemic_regimen"],
        "mapping_relation": "exact",
        "rationale": "The source uses identify the indexed systemic regimen, including adjuvant regimens.",
    },
    "resectability": {
        "canonical_variable_ids": ["crc.resectability_status"],
        "mapping_relation": "exact",
        "rationale": "The source concept is resectability immediately relevant to the indexed treatment decision.",
    },
    "specimen_date": {
        "canonical_variable_ids": ["crc.molecular_specimen_date"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the date of the indexed molecular specimen.",
    },
    "sphincter_relation": {
        "canonical_variable_ids": ["crc.sphincter_relation"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the pretreatment tumor relationship to the sphincter complex.",
    },
    "sequence": {
        "canonical_variable_ids": ["crc.treatment_sequence"],
        "mapping_relation": "exact",
        "rationale": "The source uses distinguish induction, consolidation, and radiation-first multimodality sequences.",
    },
    "treatment_line": {
        "canonical_variable_ids": ["crc.line_of_therapy"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the indexed systemic line of therapy.",
    },
    "treatment_sequence": {
        "canonical_variable_ids": ["crc.treatment_sequence"],
        "mapping_relation": "exact",
        "rationale": "The source concept is the indexed multimodality treatment sequence.",
    },
    "tumor_deposits": {
        "canonical_variable_ids": ["crc.mri_tumor_deposits"],
        "mapping_relation": "exact",
        "rationale": "The source uses are pretreatment MRI tumor-deposit findings in rectal cancer.",
    },
    "tumor_height": {
        "canonical_variable_ids": ["crc.rectal_tumor_height"],
        "mapping_relation": "exact",
        "rationale": "The source concept is pretreatment rectal tumor height for risk and treatment selection.",
    },
}


def load_mapping(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a YAML mapping")
    return data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonempty_string(value: Any, where: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{where}: expected a non-empty string")
    return text


def build(
    rule_universe_path: Path,
    variable_inventory_path: Path,
    *,
    allow_incomplete: bool,
    full_bindings_path: Path | None = None,
    full_variable_inventory_path: Path | None = None,
    full_contract_dir: Path | None = None,
) -> dict[str, Any]:
    universe = load_mapping(rule_universe_path)
    inventory = load_mapping(variable_inventory_path)
    candidates = universe.get("candidates")
    variables = inventory.get("variables")
    if not isinstance(candidates, list):
        raise ValueError(f"{rule_universe_path}: candidates must be a list")
    if not isinstance(variables, list):
        raise ValueError(f"{variable_inventory_path}: variables must be a list")

    expected = universe.get("candidate_count_expected")
    if not isinstance(expected, int) or expected <= 0:
        raise ValueError(f"{rule_universe_path}: candidate_count_expected must be a positive integer")
    if len(candidates) != expected and not allow_incomplete:
        raise ValueError(
            f"rule universe is incomplete: found {len(candidates)} candidates, expected {expected}; "
            "refusing to generate the publishable concept inventory"
        )

    canonical: dict[str, dict[str, Any]] = {}
    for index, variable in enumerate(variables):
        if not isinstance(variable, dict):
            raise ValueError(f"{variable_inventory_path}: variables[{index}] must be a mapping")
        variable_id = nonempty_string(
            variable.get("variable_id"), f"{variable_inventory_path}: variables[{index}].variable_id"
        )
        if variable_id in canonical:
            raise ValueError(f"{variable_inventory_path}: duplicate variable_id {variable_id!r}")
        canonical[variable_id] = variable

    unknown_targets = sorted(
        target
        for mapping in CURATED_EXISTING_MAPPINGS.values()
        for target in mapping["canonical_variable_ids"]
        if target not in canonical
    )
    if unknown_targets:
        raise ValueError(f"curated mappings name unknown canonical variables: {unknown_targets}")

    uses: dict[str, list[dict[str, str]]] = defaultdict(list)
    candidate_ids: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ValueError(f"{rule_universe_path}: candidates[{index}] must be a mapping")
        candidate_id = nonempty_string(
            candidate.get("candidate_id"),
            f"{rule_universe_path}: candidates[{index}].candidate_id",
        )
        if candidate_id in candidate_ids:
            raise ValueError(f"{rule_universe_path}: duplicate candidate_id {candidate_id!r}")
        candidate_ids.add(candidate_id)
        source_id = nonempty_string(
            candidate.get("source_id"), f"{candidate_id}.source_id"
        )
        category = nonempty_string(
            candidate.get("category"), f"{candidate_id}.category"
        )
        parent_id = nonempty_string(
            candidate.get("parent_recommendation_id"),
            f"{candidate_id}.parent_recommendation_id",
        )
        concepts = candidate.get("critical_variable_concepts")
        if not isinstance(concepts, list) or not concepts:
            raise ValueError(f"{candidate_id}.critical_variable_concepts must be a non-empty list")
        normalized = [
            nonempty_string(value, f"{candidate_id}.critical_variable_concepts")
            for value in concepts
        ]
        if len(normalized) != len(set(normalized)):
            raise ValueError(f"{candidate_id}: duplicate critical_variable_concepts")
        for concept in normalized:
            uses[concept].append(
                {
                    "candidate_id": candidate_id,
                    "parent_recommendation_id": parent_id,
                    "source_id": source_id,
                    "category": category,
                }
            )

    stale_curated = sorted(set(CURATED_EXISTING_MAPPINGS) - set(uses))
    if stale_curated:
        raise ValueError(
            "curated mappings are stale or not represented in this universe: "
            + ", ".join(stale_curated)
        )

    full_binding_index: dict[tuple[str, str], dict[str, Any]] = {}
    full_canonical: dict[str, dict[str, Any]] = {}
    full_contract_variables: set[str] = set()
    full_inputs = (full_bindings_path, full_variable_inventory_path, full_contract_dir)
    if any(value is not None for value in full_inputs):
        if not all(value is not None for value in full_inputs):
            raise ValueError(
                "full-universe canonicalization requires bindings, canonical variables, "
                "and the variable-contract directory together"
            )
        assert full_bindings_path is not None
        assert full_variable_inventory_path is not None
        assert full_contract_dir is not None
        bindings_doc = load_mapping(full_bindings_path)
        full_variables_doc = load_mapping(full_variable_inventory_path)
        bindings = bindings_doc.get("bindings")
        full_variables = full_variables_doc.get("variables")
        if not isinstance(bindings, list):
            raise ValueError(f"{full_bindings_path}: bindings must be a list")
        if not isinstance(full_variables, list):
            raise ValueError(f"{full_variable_inventory_path}: variables must be a list")
        for index, variable in enumerate(full_variables):
            if not isinstance(variable, dict):
                raise ValueError(
                    f"{full_variable_inventory_path}: variables[{index}] must be a mapping"
                )
            variable_id = nonempty_string(
                variable.get("variable_id"),
                f"{full_variable_inventory_path}: variables[{index}].variable_id",
            )
            if variable_id in full_canonical:
                raise ValueError(
                    f"{full_variable_inventory_path}: duplicate variable_id {variable_id!r}"
                )
            full_canonical[variable_id] = variable
        for index, binding in enumerate(bindings):
            if not isinstance(binding, dict):
                raise ValueError(f"{full_bindings_path}: bindings[{index}] must be a mapping")
            candidate_id = nonempty_string(
                binding.get("candidate_id"), f"{full_bindings_path}: bindings[{index}].candidate_id"
            )
            source_concept = nonempty_string(
                binding.get("source_concept"),
                f"{full_bindings_path}: bindings[{index}].source_concept",
            )
            key = (candidate_id, source_concept)
            if key in full_binding_index:
                raise ValueError(f"{full_bindings_path}: duplicate candidate-scoped binding {key}")
            targets = binding.get("canonical_variable_ids")
            if not isinstance(targets, list) or not targets:
                raise ValueError(f"{full_bindings_path}: {key} has no canonical_variable_ids")
            normalized_targets = [
                nonempty_string(value, f"{full_bindings_path}: {key}.canonical_variable_ids")
                for value in targets
            ]
            unknown = sorted(set(normalized_targets) - set(full_canonical))
            if unknown:
                raise ValueError(f"{full_bindings_path}: {key} names unknown variables {unknown}")
            full_binding_index[key] = {
                **binding,
                "canonical_variable_ids": normalized_targets,
            }
        expected_bindings = {
            (use["candidate_id"], concept)
            for concept, concept_uses in uses.items()
            for use in concept_uses
        }
        if set(full_binding_index) != expected_bindings:
            raise ValueError(
                "candidate-scoped binding set differs from rule_universe: "
                f"missing={sorted(expected_bindings - set(full_binding_index))}, "
                f"extra={sorted(set(full_binding_index) - expected_bindings)}"
            )
        if not full_contract_dir.is_dir():
            raise ValueError(f"{full_contract_dir}: variable-contract directory is missing")
        for contract_path in sorted(full_contract_dir.glob("*.yaml")):
            contract = load_mapping(contract_path)
            variable_id = nonempty_string(
                contract.get("canonical_variable_id"),
                f"{contract_path}: canonical_variable_id",
            )
            if variable_id in full_contract_variables:
                raise ValueError(
                    f"{full_contract_dir}: more than one contract for {variable_id!r}"
                )
            full_contract_variables.add(variable_id)
        if full_contract_variables != set(full_canonical):
            raise ValueError(
                "full variable-contract set differs from canonical inventory: "
                f"missing={sorted(set(full_canonical) - full_contract_variables)}, "
                f"extra={sorted(full_contract_variables - set(full_canonical))}"
            )

    concepts: list[dict[str, Any]] = []
    for concept in sorted(uses):
        concept_uses = sorted(uses[concept], key=lambda item: item["candidate_id"])
        if full_binding_index:
            scoped_bindings = [
                full_binding_index[(use["candidate_id"], concept)]
                for use in concept_uses
            ]
            targets = sorted(
                {
                    target
                    for binding in scoped_bindings
                    for target in binding["canonical_variable_ids"]
                }
            )
            missing_contract_targets = sorted(set(targets) - full_contract_variables)
            item = {
                "critical_variable_concept": concept,
                "canonicalization_status": "mapped_existing",
                "existing_canonical_variable_ids": targets,
                "mapping_relation": "candidate_scoped_explicit",
                "mapping_rationale": (
                    "Every use is explicitly bound at candidate scope; target differences "
                    "are retained rather than collapsed by spelling or substring."
                ),
                "missing_variable_contract": bool(missing_contract_targets),
                "candidate_scoped_bindings": [
                    {
                        "candidate_id": use["candidate_id"],
                        "canonical_variable_ids": binding["canonical_variable_ids"],
                        "relation": binding.get("relation"),
                    }
                    for use, binding in zip(concept_uses, scoped_bindings, strict=True)
                ],
            }
            if missing_contract_targets:
                item["canonical_variables_missing_contract"] = missing_contract_targets
        else:
            mapping = CURATED_EXISTING_MAPPINGS.get(concept)
            if mapping:
                targets = list(mapping["canonical_variable_ids"])
                missing_contract_targets = [
                    target for target in targets
                    if not isinstance(canonical[target].get("variable_contract"), dict)
                ]
                item = {
                    "critical_variable_concept": concept,
                    "canonicalization_status": "mapped_existing",
                    "existing_canonical_variable_ids": targets,
                    "mapping_relation": mapping["mapping_relation"],
                    "mapping_rationale": mapping["rationale"],
                    "missing_variable_contract": bool(missing_contract_targets),
                }
                if missing_contract_targets:
                    item["canonical_variables_missing_contract"] = missing_contract_targets
            else:
                item = {
                    "critical_variable_concept": concept,
                    "canonicalization_status": "canonicalization_pending",
                    "existing_canonical_variable_ids": [],
                    "mapping_relation": None,
                    "mapping_rationale": (
                        "No clinically unambiguous mapping to the existing canonical inventory "
                        "has been established; no name-based merge was attempted."
                    ),
                    "missing_variable_contract": True,
                }
        item.update(
            {
                "use_count": len(concept_uses),
                "rule_uses": [
                    {
                        "candidate_id": use["candidate_id"],
                        "parent_recommendation_id": use["parent_recommendation_id"],
                    }
                    for use in concept_uses
                ],
                "source_uses": sorted({use["source_id"] for use in concept_uses}),
                "category_uses": sorted({use["category"] for use in concept_uses}),
            }
        )
        concepts.append(item)

    mapped = sum(c["canonicalization_status"] == "mapped_existing" for c in concepts)
    pending = len(concepts) - mapped
    missing_contract = sum(bool(c["missing_variable_contract"]) for c in concepts)
    return {
        "schema_version": "1.0",
        "artifact_id": "crc_core_v1_variable_concept_inventory",
        "clinical_use": "NOT_FOR_CLINICAL_USE",
        "generated_from": {
            "rule_universe": str(rule_universe_path.relative_to(HERE.parent.parent.parent.parent)),
            "rule_universe_sha256": sha256(rule_universe_path),
            "candidate_count": len(candidates),
            "candidate_count_expected": expected,
            "variable_inventory": str(
                (full_variable_inventory_path or variable_inventory_path).relative_to(
                    HERE.parent.parent.parent.parent
                )
            ),
            "variable_inventory_sha256": sha256(
                full_variable_inventory_path or variable_inventory_path
            ),
            "existing_canonical_variable_count": len(full_canonical or canonical),
            "candidate_scoped_bindings": (
                str(full_bindings_path.relative_to(HERE.parent.parent.parent.parent))
                if full_bindings_path is not None
                else None
            ),
            "candidate_scoped_bindings_sha256": (
                sha256(full_bindings_path) if full_bindings_path is not None else None
            ),
            "variable_contract_count": (
                len(full_contract_variables) if full_binding_index else len(canonical)
            ),
        },
        "canonicalization_policy": {
            "automatic_name_matching": False,
            "mapped_existing_definition": (
                "An explicit curated assertion at global or candidate scope that the source "
                "concept has the same clinical semantics as the named canonical variable or "
                "variables for that use."
            ),
            "canonicalization_pending_definition": (
                "No safe existing-canonical mapping has been established. Similar spelling "
                "does not establish equivalence."
            ),
            "missing_contract_count_caveat": (
                "This is a count of source concepts not covered by an existing variable "
                "contract, not a forecast of distinct new contracts: clinical canonicalization "
                "may later split or merge pending concepts."
            ),
        },
        "summary": {
            "distinct_concepts": len(concepts),
            "mapped_existing": mapped,
            "canonicalization_pending": pending,
            "concepts_without_variable_contract": missing_contract,
        },
        "concepts": concepts,
    }


def dump_yaml(document: dict[str, Any]) -> str:
    return yaml.safe_dump(
        document,
        sort_keys=False,
        allow_unicode=True,
        width=120,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rule-universe", type=Path, default=DEFAULT_RULE_UNIVERSE)
    parser.add_argument("--variable-inventory", type=Path, default=DEFAULT_VARIABLE_INVENTORY)
    parser.add_argument("--full-bindings", type=Path, default=DEFAULT_FULL_BINDINGS)
    parser.add_argument(
        "--full-variable-inventory",
        type=Path,
        default=DEFAULT_FULL_VARIABLE_INVENTORY,
    )
    parser.add_argument("--full-contract-dir", type=Path, default=DEFAULT_FULL_CONTRACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="development only: allow candidate_count != candidate_count_expected",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify that --output is identical to regenerated content",
    )
    args = parser.parse_args()

    try:
        document = build(
            args.rule_universe.resolve(),
            args.variable_inventory.resolve(),
            allow_incomplete=args.allow_incomplete,
            full_bindings_path=args.full_bindings.resolve(),
            full_variable_inventory_path=args.full_variable_inventory.resolve(),
            full_contract_dir=args.full_contract_dir.resolve(),
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(str(exc)) from exc
    rendered = dump_yaml(document)
    if args.check:
        if not args.output.is_file():
            raise SystemExit(f"{args.output}: generated artifact is missing")
        if args.output.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"{args.output}: stale; regenerate with this script")
        print(
            f"valid: {document['summary']['distinct_concepts']} concepts, "
            f"{document['summary']['mapped_existing']} mapped, "
            f"{document['summary']['canonicalization_pending']} pending"
        )
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        f"wrote {args.output}: {document['summary']['distinct_concepts']} concepts, "
        f"{document['summary']['mapped_existing']} mapped, "
        f"{document['summary']['canonicalization_pending']} pending, "
        f"{document['summary']['concepts_without_variable_contract']} without a variable contract"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
