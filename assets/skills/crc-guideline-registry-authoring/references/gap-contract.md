# Gap-assessment contract

## Keep three layers separate

1. `guideline_denominator`: the complete declared accessible-source universe, with formal
   recommendations distinguished from executable branches.
2. `rule_need`: a variable is required by an eligibility/action/timing/exception predicate.
3. `spec_coverage`: a registry mapping or extraction spec represents the needed semantics.
4. `data_coverage`: a named linked-data profile demonstrated field/source availability and completeness.

A variable can pass layer 2 and fail layer 3. It can also be present in data while failing layer 2 because the field is too coarse.

## Gap classes

- `NO_REGISTRY_FIELD`
- `REGISTRY_TOO_COARSE`
- `NO_EXTRACTION_SPEC`
- `SPEC_NONCONFORMANT`
- `SOURCE_VERSION_UNBOUND`
- `TEMPORAL_ANCHOR_MISSING`
- `EXCEPTION_MODEL_MISSING`
- `RUNTIME_OPERATOR_MISSING`
- `DATA_NOT_PROFILED`
- `DATA_FIELD_ABSENT`
- `DATA_HIGH_MISSINGNESS`
- `DATA_TIME_COVERAGE_MISMATCH`
- `CLINICAL_REVIEW_PENDING`
- `REGISTRAR_REVIEW_PENDING`

## Severity

- `blocking`: rule cannot produce a defensible concordance state.
- `major`: rule can run only in a restricted subset or with material semantic loss.
- `minor`: improves traceability or detail but does not change the decision boundary.

## Required outputs

`gap_assessment.yaml` is machine-readable and includes:

- bundle and profile IDs;
- counts by rule category and gap class;
- one row per canonical variable, with complete `rule_uses` entries naming every rule and requirement role;
- registry coverage;
- spec coverage and conformance;
- data coverage with measurement provenance;
- gap class, severity, and remediation.

`gap_assessment.md` is the reviewer view:

1. source denominator and normalization coverage;
2. blocking gaps first;
3. rule-by-rule coverage matrix;
4. STORE-direct variables;
5. chart extensions;
6. unprofiled data claims;
7. promotion checklist.

## Data profile contract

A data-coverage statement needs:

- immutable run/profile ID;
- dataset snapshot or extract date;
- diagnosis-year range and CRC case definition;
- row and patient/tumor counts;
- field presence and non-null rates;
- value-frequency table including unknown codes;
- temporal coverage;
- chart document-type counts when a chart extension is claimed.

Without this, write `NOT_ASSESSED`; do not write “likely available.”
