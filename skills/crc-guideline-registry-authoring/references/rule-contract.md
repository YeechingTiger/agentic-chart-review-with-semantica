# Candidate rule contract

## Category taxonomy

Choose one primary category:

- `molecular_testing`
- `molecular_testing_operations`
- `localized_colon_treatment`
- `locally_advanced_rectal_treatment`
- `metastatic_systemic_treatment`
- `local_metastatic_treatment`
- `nonoperative_management`
- `surveillance`
- `hereditary_risk_workup`

Use tags for secondary relationships.

## Minimum rule shape

```yaml
- rule_id: CRC.<CATEGORY>.<STABLE_NAME>.v1
  category: molecular_testing
  title: Short reviewer-facing title
  authoring_status: candidate
  clinical_use: NOT_FOR_CLINICAL_USE
  fact_status:
    source_context: explicit_guideline_fact
    rule_projection: local_operationalization
  review_status: pending
  source_refs:
    - source_id: SOURCE.ID
      anchor: "Recommendation 1"
      fact_status: explicit_guideline_fact
      recommendation_type: recommendation
      evidence_quality: source_reported_value_or_not_reported
      strength: source_reported_value_or_not_reported
      paraphrase: Concise context-preserving paraphrase.
  context:
    site: [colon, rectum]
    histology: [adenocarcinoma]
    disease_setting: metastatic
    stage: not_applicable
    line_of_therapy: before_anti_egfr
    molecular_state: []
    exclusions: []
  requirements:
    eligibility:
      expression: Human-readable denominator.
      variables: [crc.disease_setting, crc.anti_egfr_plan]
    action:
      expression: Human-readable expected action.
      variables: [crc.kras_status, crc.nras_status]
    timing:
      expression: Before treatment starts.
      variables: [crc.molecular_result_date, crc.anti_egfr_start_date]
    exceptions:
      expression: Explicit exceptions or none stated by source.
      variables: []
  required_variables: [...]
  computability:
    status: partially_specified
    blockers: [...]
```

## Four-block discipline

`required_variables` must equal the union of variables in the four requirement blocks. A variable used only in prose is invisible to gap analysis and therefore invalid.

Eligibility and action must never share an ambiguous temporal state. For example, “metastatic” needs a time anchor relative to therapy if recurrence can occur after diagnosis. “RAS wild type” needs a result available before anti-EGFR initiation, not a result obtained years later.

## Computability states

- `fully_specified`: each predicate has an operational expression and every input has a conformant spec.
- `partially_specified`: clinical meaning is clear, but one or more inputs, exceptions, or temporal anchors are not yet operational.
- `not_computable`: source context is incomplete, recommendation is preference-only without a reviewable decision record, or required evidence is outside current sources.

Most new rules should begin as `partially_specified`.

## Recommendation versus quality measure

A recommendation does not automatically define a binary quality measure. Before treating it as concordance:

- identify all source-stated options;
- retain conditional/weak strength;
- model patient preference and multidisciplinary judgment when material;
- distinguish “recommended,” “may be offered,” and “should not be routinely offered”;
- capture contraindications and documented reasons for deviation.

If those conditions are not extractable, report `UNEVALUABLE` or a specific gap. Do not label the case non-concordant.

## Rule provenance

Source-derived content:

- population/context stated by the source;
- recommendation action;
- source-stated timing, strength, and exceptions.

Model-authored normalization:

- canonical variable IDs;
- Boolean/logical projection;
- mapping of prose to event dates;
- additional exception categories;
- thresholds not explicit in the source.

Keep both visible. A model-authored exception must not silently broaden the recommendation.

Allowed `fact_status` values are `explicit_guideline_fact`,
`inferred_from_guideline`, `local_operationalization`, and—only for a registered but
uninspected dependency—`source_not_extracted`.
