---
name: crc-guideline-registry-authoring
description: Compile colorectal cancer treatment and molecular-testing guidance into a version-bound source denominator, context-preserving candidate rules, one contract per canonical variable, STORE/NAACCR mappings, and registry-only versus agent-validation evidence coverage. Use when deciding which CRC guidelines belong in scope, extracting recommendations and executable branches, translating them into predicate ASTs, measuring how much of each rule the cancer registry can evaluate, mapping detailed agent outputs back to registry validation targets, checking extraction specs against this repository's conventions, or preparing a bundle for clinical/registrar review. Do not use candidates as clinical decision support until source binding and clinical sign-off are recorded.
slot: task
---

# CRC guideline and registry authoring

## Purpose

Build a reviewable authoring bundle that keeps seven claims separate:

1. what an identified source actually says;
2. how the source context was normalized into a candidate rule;
3. which variables the rule needs;
4. whether STORE/NAACCR or a chart extraction spec can supply each variable;
5. whether the available data has actually been profiled.
6. whether registry evidence can decide the guideline predicate;
7. whether an agent output can be validated directly or through a lossy registry projection.

Never promote a candidate because it parses. Parsing proves structure, not clinical validity.

## Load the right references

- Read [source-policy.md](references/source-policy.md) before selecting or quoting guidance.
- Read [rule-contract.md](references/rule-contract.md) before creating or changing rules.
- Read [spec-contract.md](references/spec-contract.md) before mapping variables or writing specs.
- Read [gap-contract.md](references/gap-contract.md) before claiming coverage or producing a gap assessment.
- Read [evidence-coverage-contract.md](references/evidence-coverage-contract.md) before
  calculating registry coverage or agent-validation coverage.

When editing an existing repository spec, also load that repository's own spec-authoring skill. In this project that is `skills/store-to-spec/SKILL.md`; its runtime-specific rules take precedence over the portable baseline here.

## Required bundle

Create or update this shape outside the production `specs/` and `guidelines/` directories:

```text
<bundle>/
  manifest.yaml
  source_register.yaml
  intake/
    source_denominator.yaml
    rule_universe.yaml
    variable_concept_inventory.yaml
  normalization/
    normalized_rules.yaml
    canonical_variables.yaml
    concept_bindings.yaml
    registry_projections.yaml
    variable_contracts/
      *.yaml
    rules_by_category/
      *.yaml
    execution_group_plan.yaml
    evidence_coverage.yaml
    gap_assessment.yaml
    gap_assessment.md
    manifest.yaml
  candidate_rules.yaml
  variable_inventory.yaml
  variable_contracts/
    *.yaml
  variable_index.yaml
  execution_groups.yaml
  materialized_specs/
    *.yaml
  execution_manifest.yaml
  registry_profiles/
    *.yaml
  guideline_universe_evidence_coverage.yaml
  guideline_universe_evidence_coverage.md
  evidence_coverage.yaml
  evidence_coverage.md
  gap_assessment.yaml
  gap_assessment.md
```

Use the production directories only after explicit clinical/registrar approval.

## Workflow

### 1. Set the intake boundary

Record disease, geography, diagnosis years, care settings, and focused clinical questions. For the initial CRC tranche, prefer:

- treatment-relevant molecular testing;
- molecularly selected therapy in metastatic disease;
- treatment selection for locally advanced rectal cancer;
- treatment/process timing only when the source states an operational threshold.

Record adjacent topics such as hereditary-risk workup or surveillance as dependencies; do not silently expand scope.

### 2. Build the source register

Search official publishers and standards bodies. Record the exact version/date/status and a stable URL. Classify each source as:

- `version_bound`: exact content was inspected and may support a candidate;
- `source_pending`: authority belongs in scope but the exact licensed/current version was not inspected;
- `superseded_or_update_pending`: usable only with an explicit limitation;
- `context_only`: evidence background, not a recommendation source.

NCCN content is commonly licensed. Do not reconstruct it from memory or secondary summaries. Put it in the register as `source_pending` until the authorized version is available.

### 3. Extract recommendations with context

For every recommendation and executable branch capture:

- source anchor and recommendation type/strength;
- disease site and histology;
- stage, resectability, treatment setting, and line;
- molecular state and laterality when relevant;
- action, comparator, sequence/timing, and exceptions;
- ambiguity or dependence on preference, multidisciplinary review, or contraindication.
- `parent_recommendation_id` and `candidate_kind`, so branches do not inflate the formal
  recommendation denominator.

Paraphrase by default. Keep any short quotation within applicable copyright limits. If context is missing, block the rule instead of filling it from clinical memory.

### 4. Normalize the rule layer

Represent each candidate with four explicit requirement blocks:

- `eligibility`: who enters the denominator;
- `action`: what should or should not happen;
- `timing`: when or in what order;
- `exceptions`: when non-performance may be concordant or unevaluable.

Give every rule one primary category and a stable ID. A rule may be `fully_specified`, `partially_specified`, or `not_computable`. “No data yet” is not the same as “the rule is underspecified.”

Give every block a machine-readable `evidence_logic` AST. A flat variable list is a
traceability index, not executable coverage logic. Use `all_of`, `any_of`, `not`, and
`conditional`; do not treat “IHC and/or MSI” as a requirement for every listed assay.

For a full-universe normalization, require exactly one normalized rule per intake candidate,
an exact candidate-scoped binding for every `critical_variable_concept`, and a generated
category view derived from the same normalized source of truth. Do not use fuzzy or
substring matching to satisfy concept bindings.

Never leave `all_of` or `any_of` empty. Encode a genuinely absent source requirement as an
explicit Boolean `constant`; encode a denominator, timing, or exception gap as `unresolved`
with a reason code. Coverage must treat `unresolved` as non-evaluable.

Every candidate `source_id` must resolve through the source-denominator slice to a registered
source. A crosswalk is not a source snapshot: missing document hashes remain promotion
blockers.

### 5. Derive variables from the rule

Collect every variable named by the four blocks. Give it one canonical ID and classify it:

- `registry_direct`: STORE/NAACCR semantics are sufficient;
- `registry_coarsened`: a registry field exists but collapses a distinction the rule needs;
- `chart_extension`: evidence exists in notes/reports but no adequate registry field exists;
- `derived`: computed only from separately specified inputs;
- `outside_current_sources`: neither registry nor current chart sources establish it.

Verify item numbers, XML IDs, effective years, and code meanings against the current standard. A similar label is not a verified mapping.

Before claiming full-universe variable coverage, aggregate every source-level
`critical_variable_concept` and explicitly map it to the canonical inventory. Never use
substring, fuzzy-name, or spelling normalization as clinical equivalence. Report unmatched
concepts as `canonicalization_pending`; their count is a gap count, not necessarily the
eventual number of new variables because review may split or merge concepts.

### 6. Author one variable contract per canonical variable

Give every canonical variable one independently versioned and hashed extraction, projection,
or derivation contract. Write:

- question and applicability guard;
- fields and enforceable domains;
- positive and negative evidence;
- conflict resolution;
- proof obligation and two abstentions;
- boundary cases and missingness semantics;
- per-enforced-element provenance.

Separately group compatible variable contracts into `execution_groups.yaml` when one agent
pass should answer them together. The execution group is an optimization and consistency
boundary; it never replaces variable-level ownership, versioning, coverage, or scoring.
Compile those groups into `materialized_specs/`, which use the repository's native
multi-field `ExtractionSpec` shape. Only materialized grouped specs are runtime inputs:
do not run one chart pass per authoring variable contract.

Unknown registry codes are positive missingness claims. Preserve `test ordered/result unavailable`, `not applicable`, and `not documented/not assessed` when the standard distinguishes them.

### 7. Compute evidence coverage and gaps

Compute separately:

- `guideline_universe_evidence_coverage`: the full accessible-source denominator, the
  normalized-rule fraction, and full-rule versus component-only reach of the bound registry
  schema;
- `registry_to_rule`: exact/coarsened/absent semantic support for each variable use;
- `canonical_to_registry`: direct/coarsened/none validation projection;
- `registry_only_rule_evaluability`: denominator, concordance, and defensible
  non-concordance coverage from the predicate AST;
- `observed_availability`: only from a compatible immutable data profile.

- `spec_gap`: rule-required variables absent or semantically too coarse in the registry/spec layer;
- `data_gap`: variables whose linked-data field presence, completeness, temporal validity, or source-note coverage was actually measured.

The structural coverage artifact must include one row per
`(rule_id, predicate_path, variable_id)` use with semantic support, temporal alignment,
provenance suitability, and canonical-to-registry projection. Global variable labels alone
do not establish rule-use coverage. Local schema component reach must use an explicit
canonical-ID mapping from the bound profile; never infer it with substring matching.

If no dataset profile was run, set data coverage to `NOT_ASSESSED`. Never infer it from the manual's required-status column.

### 8. Validate, review, and promote

Run:

```bash
.venv/bin/python <bundle>/intake/build_variable_concept_inventory.py --check
.venv/bin/python <bundle>/normalization/validate_full_normalization.py <bundle>
.venv/bin/python skills/crc-guideline-registry-authoring/scripts/validate_bundle.py <bundle>
```

For an authorized authoring rebuild, additionally run the write-producing commands below.
For a read-only assessment, do not run them against the checked-in bundle; use a temporary
copy if deterministic regeneration must be tested.

```bash
.venv/bin/python skills/crc-guideline-registry-authoring/scripts/materialize_variable_specs.py <bundle> --replace
.venv/bin/python skills/crc-guideline-registry-authoring/scripts/compute_universe_coverage.py <bundle>
.venv/bin/python skills/crc-guideline-registry-authoring/scripts/compute_evidence_coverage.py <bundle>
```

This repository requires Python 3.11+; use its virtual-environment interpreter rather than the system Python. Also run the host repository's native spec loader/tests when available. The portable validator checks authoring integrity; it does not replace clinical review or runtime tests.

Before promotion require:

- exact source-version binding;
- no unresolved rule variables;
- native spec conformance;
- clinician review of rule meaning;
- registrar review of STORE/NAACCR mappings;
- an explicit data-profile run ID for any data-coverage claim.

## Output discipline

Lead with the gap table, then the rules and specs that produced it. Label all clinical candidates `NOT FOR CLINICAL USE`. Keep source facts distinct from model-authored normalization. Record blockers rather than guessing, especially for contraindication, preference-sensitive care, performance status, non-concordance reasons, and rapidly changing drug recommendations.
