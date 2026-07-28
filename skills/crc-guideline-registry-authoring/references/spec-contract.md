# STORE-centered extraction-spec contract

## Mapping levels

- `registry_direct`: registry semantics answer the rule variable without losing a decision boundary.
- `registry_coarsened`: registry field exists, but it merges values/timing needed by the rule.
- `chart_extension`: no adequate registry field; a report/note can establish the variable.
- `derived`: deterministic transform over other specified variables.
- `outside_current_sources`: current inputs cannot establish it.

Required collection does not mean complete in the linked dataset. It also does not mean the item is clinically sufficient.

## Variable record and contract

Each canonical variable needs:

- stable `variable_id` and label;
- roles among `eligibility`, `action`, `timing`, `exception`, `outcome`;
- datatype and normalized value domain;
- temporal meaning;
- mapping level;
- verified standard name, item number, XML ID, effective years, and code crosswalk where applicable;
- exactly one independently versioned and hashed variable contract;
- one execution-group ID, output field, and materialization status;
- establishing source documents and excluded sources;
- missingness semantics;
- conformance assessment.

The authoring contract is the unit of semantic ownership and review. It is not necessarily
the unit of chart execution.

## Split and merge

Compile compatible contracts into a grouped runtime `ExtractionSpec` when one pass over the
same evidence, with the same timing and owner, should answer them together. Split when:

- evidence owners differ;
- clinical and pathologic timepoints differ;
- copying one field into another is a major feared error;
- an exception/non-concordance reason requires a different search universe.

Molecular summary SSDIs may be one pass. Regimen detail and a registry chemotherapy summary are not necessarily one field: the latter often collapses specific drugs, lines, sequence, and intent.

Required artifacts:

- `variable_contracts/*.yaml`: exactly one per canonical variable;
- `variable_index.yaml`: variable-to-contract/group/field lookup;
- `execution_groups.yaml`: one-pass grouping and intentional supporting fields;
- `materialized_specs/*.yaml`: generated native multi-field runtime specs;
- `execution_manifest.yaml`: deterministic hashes and lineage.

Only materialized grouped specs go to the runtime. Running every variable contract directly
would turn 68 variables into 68 passes and would defeat the repository's existing
variable-to-group resolver.

## Minimum extraction spec

A conformant spec includes:

- `spec_id`, semantic version, source authority, and provenance;
- one registrar-readable `question`;
- `applicability_guard`;
- fields with `allowable_values` or a real runtime regex where possible;
- evidence that counts and evidence that does not count;
- conflict resolution and source precedence;
- positive proof witness;
- negative/unknown proof obligation;
- distinct `EVIDENCE_INSUFFICIENT` and `SPEC_INSUFFICIENT`;
- boundary cases;
- search hints that can discharge any required search terms;
- answer checks for unknown/NOS codes when supported by the runtime.

## Missingness

Preserve standard distinctions:

- test performed and positive/negative;
- test ordered but result absent;
- test not applicable;
- not documented or unknown whether assessed;
- indeterminate/equivocal when the standard combines it with unknown.

If the registry collapses clinically distinct states, mark `registry_coarsened` and create a chart extension rather than pretending the SSDI is lossless.

## Provenance and review

Every runtime-enforced element must identify its origin and verification status. A manual citation needs an item/section/page locator. Model-authored logic must explicitly state that it has no external source. Clinical and registrar sign-off are separate:

- clinician: rule meaning and clinical exceptions;
- registrar: STORE/NAACCR item mapping and coding;
- engineer: runtime shape and executable checks.

No single sign-off substitutes for the other two.

## Conformance verdicts

- `conformant_candidate`: structurally complete, version-bound, but not yet clinically approved.
- `needs_revision`: correctable structural or semantic defects.
- `blocked_source`: exact standard/guideline content unavailable.
- `blocked_runtime`: host runtime cannot enforce the required distinction.
- `not_assessed`: not yet reviewed.
