# CRC registry evidence coverage

> Authoring assessment — **NOT FOR CLINICAL USE**.

> **SEED LAYER ONLY.** This file covers the 12 structured seed rules, not the 280-candidate guideline universe. Use `guideline_universe_evidence_coverage.md` for denominator-level reporting.

## Blocking interpretation

- Executable candidates: **12**.
- Canonical variables: **68**.
- Rules with normalized evidence AST: **0**.
- Observed CRC availability: **NOT_ASSESSED**; no compatible tumor-level CRC profile is bound.

A rule can be useful for registry validation while remaining unevaluable from registry data alone. Coarsened projection coverage is not guideline-rule coverage.

## Category coverage

| category | candidates | formal parents | AST normalized | full | partial | none | not assessed |
|---|---:|---:|---:|---:|---:|---:|---:|
| hereditary_risk_workup | 1 | 1 | 0 | 0 | 0 | 0 | 1 |
| locally_advanced_rectal_treatment | 3 | 3 | 0 | 0 | 0 | 0 | 3 |
| metastatic_systemic_treatment | 4 | 4 | 0 | 0 | 0 | 0 | 4 |
| molecular_testing | 4 | 4 | 0 | 0 | 0 | 0 | 4 |

## Conservative flat-list support

This is an upper-bound diagnostic for rules whose AST is not normalized; it is not a reportable coverage rate.

| category | full | partial | none | not assessed |
|---|---:|---:|---:|---:|
| hereditary_risk_workup | 0 | 0 | 1 | 0 |
| locally_advanced_rectal_treatment | 0 | 0 | 3 | 0 |
| metastatic_systemic_treatment | 0 | 0 | 4 | 0 |
| molecular_testing | 0 | 0 | 4 | 0 |

## Registry profiles

| profile | disease | unit | CRC compatibility | mapped canonical variables | CRC observed |
|---|---|---|---|---:|---|
| CRC_LINKED_PROFILE_PENDING | colorectal_carcinoma | tumor_record | None | 0 | NOT_ASSESSED |
| R6249_LUNG_SCHEMA_2026-07-25 | lung_cancer | patient_after_last_row_deduplication | SCHEMA_ONLY_NON_CRC | 4 | NOT_ASSESSED |

## Variable-use axes

- Registry → rule semantic support: `{'absent': 83, 'coarsened': 49, 'exact': 21}`.
- Temporal alignment: `{'conditional': 53, 'exact': 21, 'misaligned': 79}`.
- Agent → registry validation projection: `{'coarsened': 52, 'direct': 21, 'none': 80}`.
