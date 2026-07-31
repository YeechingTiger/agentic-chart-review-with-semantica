# Evidence-coverage contract

## Purpose

Answer two different questions without conflating them:

1. **Registry-only rule evaluability:** can the registry evidence decide a guideline
   predicate or an entire rule?
2. **Agent validation coverage:** can a detailed agent output be compared directly, or after
   a lossy projection, with a registry field?

A registry item can be useful for validation while being insufficient for guideline
evaluability. For example, RX Summ--Chemo can validate a projected single/multi-agent class
but cannot reconstruct a regimen, treatment line, intent, dose, cycle, or progression.

## Denominators

Report both:

- `formal_recommendations`: source-numbered recommendations or good-practice statements;
- `executable_candidates`: source-specific branches and operational qualifiers produced
  from those recommendations.

Every executable candidate must name `parent_recommendation_id` and `candidate_kind`.
Never report branch coverage as if each branch were a separate formal recommendation.

## Variable-use coverage

Coverage belongs to a `(rule_id, predicate_path, variable_id)` use, not only to a variable.
The same registry item can preserve one predicate boundary and collapse another.

Record four independent dimensions:

### 1. Semantic support: registry to canonical fact

- `exact`: the item preserves the decision boundary used by this predicate;
- `coarsened`: a related item exists but collapses a required distinction;
- `absent`: no registry item establishes the canonical fact;
- `not_applicable`: the item does not apply for this site, histology, or diagnosis year;
- `unknown`: mapping has not been reviewed.

### 2. Temporal alignment

- `exact`: registry and rule use the same tumor/event/timepoint;
- `conditional`: aligned only when stated conditions hold, such as first-course treatment;
- `misaligned`: diagnosis-time or first-course data cannot answer a later decision;
- `unknown`.

### 3. Provenance suitability

- `agreement_target`: sufficiently independent and precise for direct comparison;
- `weak_label`: abstracted/coarsened evidence useful only with caveats;
- `same_source_dependent`: agent and registry may derive from the same chart source;
- `not_truth`: do not score the agent against it as truth;
- `unknown`.

### 4. Observed availability

An observed claim requires a named immutable profile and reports:

- candidate and eligible tumor counts;
- field presence;
- informative and non-informative counts;
- unknown/not-applicable code distribution;
- diagnosis-year and facility coverage.

Use `NOT_ASSESSED` when the profile is absent or belongs to a non-compatible disease,
schema, time period, or tumor unit.

## Projection coverage: canonical fact to registry

Record this separately from semantic support:

- `direct`: canonical output and registry item are directly comparable;
- `coarsened`: apply a named, versioned projection before comparison;
- `none`: registry cannot validate this output;
- `unknown`.

Every coarsened projection needs `transformation_id`, input domain, output domain, effective
years, and an explicit loss list.

## Predicate AST

Natural-language expressions and flat variable lists are insufficient. Each requirement
block must carry a machine-readable evidence AST:

```yaml
evidence_logic:
  op: any_of
  operands:
    - variable: crc.msi_status
    - op: all_of
      operands:
        - variable: crc.mmr_mlh1
        - variable: crc.mmr_pms2
        - variable: crc.mmr_msh2
        - variable: crc.mmr_msh6
```

Allowed authoring operators are:

- `all_of`
- `any_of`
- `not`
- `conditional`
- `variable`
- `constant` — only for an explicitly inapplicable/no-source-requirement block, with a
  Boolean `value` and a recorded meaning;
- `unresolved` — an explicit non-evaluable dependency with a `reason_code`, used when the
  source or local operationalization does not yet define the required predicate.

The AST describes evidence dependencies, not the clinical value comparison. Clinical
operators and value sets remain in the executable guideline rule.

Never publish an empty `all_of` or `any_of`. Although an empty conjunction can be assigned a
mathematical truth value, it cannot distinguish “the source states no requirement” from “the
author failed to model the denominator, timing anchor, or exception.” Use `constant` or
`unresolved` so coverage cannot silently treat a missing predicate as fully supported.

If no AST exists, report only a conservative flat-list upper bound and label it
`LOGIC_NOT_NORMALIZED`; do not call it measured rule coverage.

## Rule-level coverage

Report three outcomes:

1. `denominator_evaluability`: eligibility can be decided.
2. `concordance_evaluability`: eligibility, action, and timing can be decided.
3. `nonconcordance_defensibility`: eligibility, action, timing, and relevant exceptions can
   be decided.

Evaluate the AST with three-valued logic. A known false branch may settle `all_of`, and a
known true branch may settle `any_of`; unknown inputs do not always make the whole block
unknown.

Structural registry-only status:

- `full`: at least one complete AST path has exact semantic and temporal support;
- `partial`: registry supplies related evidence but no complete exact path;
- `none`: no decision path can be completed;
- `not_assessed`: rule logic or mappings are not normalized.

Do not use a mean variable percentage as the primary rule metric. One missing critical
predicate can make an otherwise well-populated rule unevaluable.

## Category summaries

For every category report:

- formal-recommendation count;
- executable-candidate count;
- full/partial/none/not-assessed rule counts;
- exact/coarsened/absent variable-use counts;
- temporally exact/conditional/misaligned counts;
- direct/coarsened/unvalidated agent-validation counts;
- top minimum missing cut sets and variable unlock scores.

Always display the count of fully registry-only evaluable rules. Do not let a high
variable-use percentage hide a zero-rule result.

## Profile compatibility

A profile must state disease, site/schema, tumor versus patient unit, diagnosis years, and
snapshot. A lung registry profile may test generic infrastructure or expose a local schema,
but it cannot establish observed CRC evidence coverage. Report it as
`SCHEMA_ONLY_NON_CRC`, never as a CRC availability profile.
