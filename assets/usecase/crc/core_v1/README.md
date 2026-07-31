# CRC guideline-to-registry authoring bundle

> Candidate authoring assets — **NOT FOR CLINICAL USE**.

## Current truth

| layer | count | meaning |
|---|---:|---|
| core formal recommendations | 218 | formal clinical-guideline denominator |
| core nonformal source units | 37 | good-practice statements, qualifiers, and explicit source branches |
| core scoped source units | 255 | 218 formal plus 37 nonformal units reviewed |
| core executable candidates | 275 | formal units split into executable branches/qualifiers where needed |
| supplemental/adjacent candidates | 5 | excluded from the core rate |
| full normalized candidate rules | 280 | every candidate has four explicit machine-readable blocks; 82 blocks remain unresolved |
| computability disposition | 253 partial; 27 not computable | explicit blockers remain; AST presence is not clinical executability |
| rule categories | 39 | category views are generated from the same 280-rule source of truth |
| full-universe source concepts | 568 | vocabulary aggregated from all 280 candidates |
| canonical variables | 576 | candidate/timepoint-specific distinctions are retained where concepts cannot safely merge |
| variable extraction contracts | 576 | exactly one independently hashed authoring contract per canonical variable |
| coarsened registry projection contracts | 9 | named/versioned loss contracts for all coarsened variables |
| rule-variable coverage uses | 2,406 | semantic, temporal, provenance, and registry-projection dimensions are separate |
| planned full-universe execution groups | 8 | execution plan only; full-universe runtime specs are not materialized |
| materialized seed runtime specs | 6 | legacy 12-rule seed execution layer, not the full-universe spec count |

The bound local registry profile is a 15-column lung-cancer, patient-level extract with no
treatment or molecular fields. Because the 280-rule universe now has explicit evidence ASTs
and candidate variable bindings, candidate STORE-standard structural rule coverage is
reportable as **0 full / 0 partial / 280 none**. The local extract has at least one exact
profile-declared canonical component for **97/280**, but component reach is not rule
coverage and cannot support a concordance denominator. Observed CRC coverage separately
remains `NOT_ASSESSED` until a compatible immutable tumor-level CRC profile is bound.

## Start here

- Source denominator: `intake/source_denominator.yaml`
- Full candidate universe: `intake/rule_universe.yaml`
- Normalized rules with evidence AST: `normalization/normalized_rules.yaml`
- Rules grouped into 39 category views: `normalization/rules_by_category/`
- Full canonical-variable inventory: `normalization/canonical_variables.yaml`
- One-variable extraction contracts: `normalization/variable_contracts/`
- Coarsened canonical-to-registry projections: `normalization/registry_projections.yaml`
- Full normalization coverage and gaps: `normalization/evidence_coverage.yaml`,
  `normalization/gap_assessment.md`
- Coverage answer: `guideline_universe_evidence_coverage.md`
- Full-universe execution plan: `normalization/execution_group_plan.yaml`
- Seed runtime grouping and lineage: `execution_groups.yaml`, `execution_manifest.yaml`
- Consolidated development gaps: `gap_assessment.md`

Rebuild/check with the scripts in
`skills/crc-guideline-registry-authoring/scripts/` and
`normalization/validate_full_normalization.py`.
