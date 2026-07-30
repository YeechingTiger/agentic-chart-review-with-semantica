# CRC core v1: guideline denominator, evidence coverage, and gap assessment

> Authoring candidate — **NOT FOR CLINICAL USE**. Assessed 2026-07-27.

The accessible-source intake denominator is 218 core formal recommendations plus 37
nonformal source units, materialized as 275 executable core candidates, plus 5
supplemental/adjacent candidates.
NCCN remains licensed-pending and zero NCCN rules were reconstructed. The original 12 rules
are a runtime-materialized seed tranche, not the guideline denominator. The full 280-rule
universe now has evidence ASTs, candidate-scoped canonical bindings, and one authoring
contract for each of 576 canonical variables.

## Full-universe evidence-coverage answer

| layer | result | interpretation |
|---|---:|---|
| normalized candidate rules | 280/280 | all four blocks have explicit ASTs; 82 blocks carry an unresolved node |
| computability disposition | 253 partial / 27 not computable | source, temporal, denominator, exception, or local-operationalization blockers remain |
| canonical variables with contracts | 576/576 | each has boundary cases and per-enforced-element candidate provenance |
| coarsened registry projections | 9/9 | each has a transformation ID, domains, effective years, and explicit loss list |
| rule-variable coverage uses | 2,406 | each records semantic, temporal, provenance, and projection status |
| source-slice resolution | 280/280 | rule source slices resolve through the denominator to the source register |
| captured source document hashes | 0/19 | every registered source remains promotion-blocked on immutable snapshot/hash |
| candidate STORE structural rule coverage | 0 full / 0 partial / 280 none | no complete rule is defensibly evaluable from current candidate STORE mappings alone |
| local 15-column component reach | 97/280 | exact canonical IDs declared by the profile can validate components only |
| full-universe runtime materialization | 0 variables | eight execution groups are planned but not compiled/reviewed as native specs |
| observed linked CRC availability | `NOT_ASSESSED` | no compatible immutable tumor-level CRC profile is bound |

The 568 source concepts do not map one-to-one to variable IDs. The normalized inventory has
576 variables because clinically unsafe merges were avoided and candidate-specific
timepoints, indexed decisions, and operational evidence distinctions were retained. All
bindings and contracts are authoring candidates pending clinical, registrar, and runtime
review.

## Blocking gaps in development order

1. **Bind current sources and adjudicate conflicts.** Obtain authorized NCCN Colon, Rectal, and Hereditary snapshots with version/effective date/hash. Review ESMO 2026 versus ASCO 2022 metastatic branches. Do not activate one source family by silently overwriting another.
2. **Repair production tumor identity and stage specs.** Resolve provenance, date formats, item/edition binding, and candidate/production ownership before any CRC run.
3. **Review the 576 candidate contracts.** Prioritize the 4 verified registry-direct and 9 registry-coarsened variables, then the highest-use chart variables. Resolve the recorded cross-module definition conflicts before runtime compilation.
4. **Profile the linked registry extract.** Measure field presence, diagnosis-year availability, non-null rate, code distribution, and facility/state variation using a compatible tumor-level CRC snapshot. Until this is done, every observed-data claim stays `NOT_ASSESSED`.
5. **Validate molecular chart extensions.** Start with individual MMR proteins, assay/method/QC/specimen/date, discordance, expanded-RAS coverage, and MLH1 methylation. These unlock the most rules and have direct registry comparison targets.
6. **Validate rectal MRI extraction.** Establish a radiologist-reviewed terminology and threshold contract, then stratify source coverage by dedicated MRI versus other imaging.
7. **Build longitudinal treatment representation.** Add regimen terminology, exact agents, line, intent, dates, sequence, outside exposure, response/progression, and current label/source adapters.
8. **Build non-concordance reasons only after the above.** Contraindication, refusal/preference, comorbidity/fitness, missing outside care, guideline conflict, and source/spec insufficiency must remain distinct. Nonreceipt alone is never a reason.

## Outcome

The authoring layers are:

| layer | count | current result |
|---|---:|---|
| accessible-source executable candidates | 280 | 275 core plus 5 supplemental/adjacent; NCCN excluded pending an authorized snapshot |
| full normalized rules | 280 | evidence AST and exact candidate-scoped variable binding exist for every requirement block |
| full canonical variables/contracts | 576 | 576/576 authoring contracts; clinical and runtime review pending |
| registry-direct variables | 4 | candidate agreement targets with verified STORE/NAACCR mappings |
| registry-coarsened variables | 9 | registry supplies evidence but loses distinctions required by at least one rule |
| derived variables | 7 | each derivation still requires reviewed inputs and logic |
| chart extensions | 533 | includes 13 registry-like labels downgraded because no mapping was verified |
| outside-current-source variables | 23 | require a new evidence source or explicit unevaluable state |
| planned full-universe groups | 8 | not yet runtime materialized |
| materialized seed runtime specs | 6 | execution batches for the original 12-rule tranche only |
| compatible CRC linked-data profiles | 0 | observed CRC coverage is `NOT_ASSESSED`, not “missing” |

The full rule-to-variable traceability graph is now complete at the authoring layer. This is
not a claim that all variables are clinically approved, natively executable, or observed in
the linked data.

## Rules by category

| category | candidate | rule-layer meaning | status |
|---|---|---|---|
| molecular testing | `CRC.MOL.MMR_ICI_METHOD.v1` | eligibility → valid method/QC/result → result before ICI → discordance exception | partially specified |
| molecular testing | `CRC.MOL.MMR_MSI_DISCORDANCE.v1` | discordant/indeterminate/subclonal result → review/repeat/final interpretation | partially specified |
| molecular testing | `CRC.MOL.EXTENDED_RAS_BEFORE_ANTIEGFR.v1` | anti-EGFR considered → complete KRAS/NRAS coverage/result before start | partially specified; CAP update check required |
| hereditary workup | `CRC.MOL.MLH1_LOSS_LYNCH_REFLEX.v1` | MLH1-loss pattern → BRAF/methylation/communication/referral | partially blocked by unbound NCCN hereditary source |
| rectal assessment | `CRC.RECT.MMR_BEFORE_TREATMENT.v1` | LARC → MMR/MSI result before treatment | partially specified |
| rectal assessment | `CRC.RECT.MRI_BEFORE_TREATMENT.v1` | LARC → dedicated MRI and complete risk report before treatment | partially specified |
| rectal treatment | `CRC.RECT.TNT_HIGH_RISK_PMMR.v1` | pMMR/MSS low/high-risk LARC → initial TNT, with explicit exceptions | partially specified |
| rectal treatment | `CRC.RECT.DMMR_ICI_INITIAL.v1` | MSI-H/dMMR LARC → initial ICI strategy; preserve contraindication | agent/duration realization pending current-source review |
| metastatic treatment | `CRC.MCRC.PMMR_FIRST_LINE_BACKBONE.v1` | untreated unresectable pMMR/MSS → doublet for most; selected triplet | source-specific, current reconciliation pending |
| metastatic treatment | `CRC.MCRC.MSIH_FIRST_LINE_ASCO2022.v1` | ASCO 2022 first-line pembrolizumab branch | blocked as a universal current rule |
| metastatic treatment | `CRC.MCRC.RAS_SIDEDNESS_ASCO2022.v1` | ASCO 2022 RAS/sidedness anti-EGFR branch | blocked as a universal current rule |
| metastatic treatment | `CRC.MCRC.BRAF_POST_LINE_ASCO2022.v1` | ASCO 2022 post-line BRAF combination branch | blocked as a universal current rule |

The three blocked metastatic rules are still useful as source-specific audit candidates. They must not be merged into one “current standard”: ESMO 2026 changes first-line MSI-H/dMMR, RAS/BRAF/location selection, and BRAF treatment placement. Licensed NCCN and a jurisdiction/date-specific regulatory adapter remain required for US certification.

## STORE/NAACCR coverage

### Good registry agreement targets

- Primary site `#400`, histology `#522`, and clinical TNM items can anchor the tumor and staging context after production-spec repair.
- KRAS `#3866`, BRAF `#3940`, and NRAS `#3941` have direct summary mappings.
- MSI/MMR `#3890` is a strong summary-level agreement target.
- First-course treatment dates and summaries can validate whether broad modalities were recorded.

### Registry semantic loss that changes a rule

- `#3890` merges MSI and MMR and loses method, four-protein pattern, discordance, QC, specimen, and result date.
- KRAS/NRAS SSDIs do not prove complete expanded-RAS exon/codon coverage or exact variants.
- Chemotherapy `#1390` and immunotherapy `#1410` do not contain regimen, drug, line, intent, dose, cycles, completion, or progression.
- First-course dates do not represent the complete longitudinal treatment history.
- STORE has no adequate resectability, multidisciplinary conclusion, ECOG, organ constraints, detailed contraindication, or preference field.
- STORE has no adequate pretreatment rectal MRI fields for tumor height, MRF, EMVI, extramural depth, sphincter/intersphincteric relation, or MRI tumor deposits.

Therefore the registry remains the validation spine, but not the complete clinical truth layer.

## Spec-conformance assessment

The six new candidate specs pass the native loader and have per-enforced-element provenance:

- `CRC.STORE.molecular_summary`
- `CRC.CHART.molecular_detail`
- `CRC.CHART.rectal_mri_risk`
- `CRC.STORE_CHART.systemic_treatment`
- `CRC.CHART.decision_exceptions`
- `CRC.CANDIDATE.disease_context`

No variable is currently rated `conformant_candidate`. The KRAS, NRAS, and BRAF summary contracts load natively, but their proof obligation requires review of molecular reports/pathology addenda while the runtime currently proves only keyword searches. Until a document-category coverage operator is implemented and tested, codes 8/9 can satisfy the runtime without satisfying the declared evidence semantics.

There is also a repository-wide positive-proof gap: `for_positive.witness` and stratum
`establishes` are parsed and exposed for review, but the current answer gate does not verify
that a `FOUND` citation came from an allowed witness stratum for the answered field. The
completed `store-to-spec` proof-obligation reference records this limitation; full-universe
contracts must retain `needs_revision` until the runtime check and boundary tests exist.

All 68 variables in the runtime-materialized seed tranche are therefore rated
`needs_revision`, mainly because:

- value domains, source precedence, search terms, and negative proof have not been measured on CRC charts;
- the runtime lacks a document-category coverage operator for the STORE molecular-summary absence proof;
- the CAP 2017 expanded-RAS rule is under update, so KRAS/NRAS/expanded-RAS rows retain a current-source blocker;
- disease setting, LARC, sidedness, line, resectability, MRI thresholds, regimen terminology, and exception categories contain local operationalization;
- the production owner/collision for site, histology, and stage is unresolved;
- specialist review is pending.

The existing five files in top-level `specs/` were checked independently. All five fail the current native provenance gate:

| existing spec | native result | principal implication |
|---|---|---|
| `STORE.390.date_of_initial_diagnosis` | fail: 13 unprovenanced enforced elements | cannot be a production dependency |
| `STORE.400_522_523.site_histology_behavior` | fail: 22 unprovenanced enforced elements | CRC case routing/identity owner must be repaired |
| `STORE.610.class_of_case` | fail: 2 unprovenanced enforced elements | registry-only context is not currently loadable |
| `STORE.700_880.stage` | fail: 36 unprovenanced enforced elements | clinical/pathologic stage owner must be repaired |
| `STORE.1860_1880.first_recurrence` | fail: 12 unprovenanced enforced elements | recurrence cannot yet support later-line logic |

This assessment ignores patient data, as requested. It is a contract/runtime consistency finding.

## Full-universe development gap

The full normalized universe now covers, rather than merely defers:

- stage II/III colon adjuvant treatment;
- selective CRT, TNT sequence, and cCR/nonoperative management;
- HER2, NTRK/RET, POLE/POLD1/TMB, KRAS G12C, ctDNA, and modern BRAF branches;
- local treatment of resectable/oligometastatic disease.

The remaining work is clinical/registrar adjudication of the generated ASTs, mappings, value
domains, conflicts, and negative-proof requirements; compiling the eight planned execution
groups into native runtime specs; testing boundary behavior; and measuring a compatible CRC
linked-data snapshot. Authoring completeness is not runtime or clinical completeness.
