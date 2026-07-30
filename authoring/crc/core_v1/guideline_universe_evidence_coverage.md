# CRC guideline-universe evidence coverage

> Authoring assessment — **NOT FOR CLINICAL USE**.

## Answer

- Accessible-source denominator: **275 core** + **5 supplemental/adjacent** candidates.
- Formal source units: **218 core** + **5 supplemental/adjacent**; executable branch counts must not be presented as independent recommendations.
- Core source review also includes **37 nonformal units**; total core scoped source units are **255**.
- Universe rows materialized: **280**.
- STORE-standard candidate structural rule coverage: **0 full / 0 partial / 280 none / 0 not assessed**, denominator **280**.
- Structural schema upper bound: **0/280** candidates could be fully evaluated from the bound 15-column extract; this is a diagnostic, not a coverage rate.
- The local profile declares at least one exact canonical component for **97/280** rows; this is not concordance coverage.
- Observed CRC coverage: **NOT_ASSESSED** because the measured profile is a lung-cancer, patient-level extract rather than a CRC tumor-record dataset.

## Why 12 / 68 / 6 were misleading

- **12** is the current four-block seed tranche, not the guideline denominator.
- **280** full-universe candidates have explicit machine-readable evidence ASTs; **82** requirement blocks are explicitly unresolved rather than silently treated as true.
- **576** is the current canonical-variable contract count; every current variable has one contract.
- Evidence coverage contains **2406** rule/predicate/variable-use rows with separate semantic, temporal, provenance, and registry-projection dimensions.
- All coarsened uses resolve to **9** named/versioned candidate projection contracts.
- The full universe contains **568** distinct source concepts; all candidate-scoped uses are bound into **576** candidate canonical variables/contracts.
- The original seed has **6** materialized runtime groups. The full universe has **8** planned groups and is not runtime-materialized yet.

## Category coverage

| category | candidates | source parents | STORE full | STORE partial | STORE none | local component reach |
|---|---:|---:|---:|---:|---:|---:|
| clinical_trial | 1 | 1 | 0 | 0 | 1 | 0 |
| colon_adjuvant | 24 | 22 | 0 | 0 | 24 | 0 |
| genomic_testing | 9 | 9 | 0 | 0 | 9 | 0 |
| genomic_treatment | 1 | 1 | 0 | 0 | 1 | 0 |
| germline_testing | 1 | 1 | 0 | 0 | 1 | 0 |
| historical_evidence | 3 | 3 | 0 | 0 | 3 | 0 |
| liver_metastasis_local | 8 | 7 | 0 | 0 | 8 | 8 |
| liver_transplant | 1 | 1 | 0 | 0 | 1 | 1 |
| lung_metastasis_local | 1 | 1 | 0 | 0 | 1 | 1 |
| lynch_reflex | 6 | 4 | 0 | 0 | 6 | 0 |
| metastatic_mdt | 4 | 4 | 0 | 0 | 4 | 4 |
| metastatic_primary | 1 | 1 | 0 | 0 | 1 | 1 |
| metastatic_resectability | 1 | 1 | 0 | 0 | 1 | 1 |
| metastatic_staging | 5 | 5 | 0 | 0 | 5 | 5 |
| metastatic_systemic | 50 | 25 | 0 | 0 | 50 | 50 |
| molecular_interpretation | 5 | 5 | 0 | 0 | 5 | 0 |
| molecular_operations | 23 | 23 | 0 | 0 | 23 | 1 |
| molecular_reporting | 2 | 2 | 0 | 0 | 2 | 0 |
| molecular_testing | 21 | 17 | 0 | 0 | 21 | 4 |
| nodal_metastasis_local | 2 | 2 | 0 | 0 | 2 | 2 |
| oligometastatic_local | 6 | 6 | 0 | 0 | 6 | 6 |
| oligometastatic_systemic | 4 | 4 | 0 | 0 | 4 | 4 |
| ovarian_metastasis_local | 1 | 1 | 0 | 0 | 1 | 1 |
| peritoneal_metastasis_local | 5 | 4 | 0 | 0 | 5 | 5 |
| pharmacogenomics | 5 | 1 | 0 | 0 | 5 | 0 |
| rectal_adjuvant | 1 | 1 | 0 | 0 | 1 | 0 |
| rectal_assessment | 15 | 15 | 0 | 0 | 15 | 0 |
| rectal_immunotherapy | 3 | 3 | 0 | 0 | 3 | 0 |
| rectal_neoadjuvant | 11 | 11 | 0 | 0 | 11 | 0 |
| rectal_organ_preservation | 10 | 10 | 0 | 0 | 10 | 0 |
| rectal_radiation | 8 | 8 | 0 | 0 | 8 | 0 |
| rectal_restaging | 4 | 4 | 0 | 0 | 4 | 0 |
| rectal_selective_radiation | 1 | 1 | 0 | 0 | 1 | 0 |
| rectal_surgery | 3 | 3 | 0 | 0 | 3 | 0 |
| rectal_surveillance | 2 | 2 | 0 | 0 | 2 | 0 |
| rectal_timing | 9 | 9 | 0 | 0 | 9 | 0 |
| rectal_tnt | 18 | 17 | 0 | 0 | 18 | 0 |
| shared_decision_making | 4 | 4 | 0 | 0 | 4 | 3 |
| survivorship_adjacent | 1 | 1 | 0 | 0 | 1 | 0 |

The structural upper bound is zero because the local extract contains no molecular-testing, treatment, imaging/procedure, decision-rationale, exception, or action-timing fields. Its stage and tumor-identity columns can help validate components. Full-universe AST normalization is complete, but observed CRC availability remains NOT_ASSESSED until a compatible tumor-level CRC profile is bound.
