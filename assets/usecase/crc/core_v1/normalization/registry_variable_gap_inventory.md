# CRC guideline variables versus cancer registry

> Candidate authoring inventory — **NOT FOR CLINICAL USE**.

## Summary

| registry relationship | variables | interpretation |
|---|---:|---|
| no registry field: linked-chart extension | 533 | The guideline fact must be extracted from linked clinical documents or events. |
| no registry field: outside current sources | 23 | The fact needs a new external source, policy, label, calendar, or evidence owner. |
| derived, not stored directly | 7 | The fact must be computed from reviewed inputs. |
| registry field exists but is coarsened | 9 | The registry can support a lossy validation projection, not the canonical rule fact. |
| registry-direct candidate mapping | 4 | The canonical fact has a candidate direct NAACCR mapping, still pending review. |

Therefore **556** canonical variables have `NO_REGISTRY_FIELD`, and **563** are not stored directly in the registry.

## Missing registry fields — linked-chart extensions (533)

These are required guideline facts with no accepted registry field. They should be added to the STORE-centered linked-chart feature layer, not silently inferred from registry summaries.

| variable | rule-block uses | roles | guideline categories | example rules |
|---|---:|---|---|---|
| `crc.decision_date` | 92 | timing | liver_metastasis_local, liver_transplant, lung_metastasis_local, metastatic_mdt, metastatic_primary, metastatic_resectability, metastatic_staging, metastatic_systemic, nodal_metastasis_local, oligometastatic_local, oligometastatic_systemic, ovarian_metastasis_local, peritoneal_metastasis_local, shared_decision_making | ASCO22-LOCAL-01, ASCO22-LOCAL-02, ASCO22-LOCAL-03, ASCO22-LOCAL-04, ASCO22-LOCAL-05, ASCO22-LOCAL-06, +86 more |
| `crc.localized_rectal.disease_status_at_decision` | 86 | eligibility | rectal_adjuvant, rectal_assessment, rectal_immunotherapy, rectal_neoadjuvant, rectal_organ_preservation, rectal_radiation, rectal_restaging, rectal_selective_radiation, rectal_surgery, rectal_surveillance, rectal_timing, rectal_tnt, shared_decision_making | ASCO24-LARC-1.1, ASCO24-LARC-1.2, ASCO24-LARC-1.3, ASCO24-LARC-2.1, ASCO24-LARC-2.2, ASCO24-LARC-3.1, +80 more |
| `crc.line_of_therapy` | 58 | eligibility, timing | liver_metastasis_local, metastatic_resectability, metastatic_systemic, molecular_testing, oligometastatic_systemic | ASCO22-LOCAL-07, ASCO22-LOCAL-12, ASCO22-MCRC-01, ASCO22-MCRC-02, ASCO22-MCRC-03, ASCO22-MCRC-04, +51 more |
| `crc.rectal_treatment.start_date` | 58 | action, timing | rectal_assessment, rectal_immunotherapy, rectal_neoadjuvant, rectal_radiation, rectal_selective_radiation, rectal_timing, rectal_tnt | ASCO24-LARC-1.1, ASCO24-LARC-1.2, ASCO24-LARC-1.3, ASCO24-LARC-2.1, ASCO24-LARC-2.2, ASCO24-LARC-3.1, +51 more |
| `crc.rectal_treatment.decision_date` | 56 | timing | rectal_assessment, rectal_immunotherapy, rectal_neoadjuvant, rectal_radiation, rectal_selective_radiation, rectal_tnt | ASCO24-LARC-1.1, ASCO24-LARC-1.2, ASCO24-LARC-1.3, ASCO24-LARC-2.1, ASCO24-LARC-2.2, ASCO24-LARC-3.1, +50 more |
| `crc.documented_reason_for_deviation_at_decision` | 41 | exceptions | colon_adjuvant, rectal_adjuvant, rectal_assessment, rectal_immunotherapy, rectal_neoadjuvant, rectal_radiation, rectal_restaging, rectal_selective_radiation, rectal_surveillance, rectal_timing, rectal_tnt | ASCO19-S3-DUR-01, ASCO22-S2-1.1, ASCO22-S2-2.1a, ASCO22-S2-2.1b, ASCO22-S2-3.1b, ASCO24-LARC-2.1, +35 more |
| `crc.rectal_treatment.planned_pathway_at_decision` | 41 | action | rectal_immunotherapy, rectal_neoadjuvant, rectal_radiation, rectal_selective_radiation, rectal_tnt | ASCO24-LARC-2.1, ASCO24-LARC-2.2, ASCO24-LARC-3.1, ASCO24-LARC-4.1, ASCO24-LARC-6.1, ASTRO25-KQ1-03, +35 more |
| `crc.local_treatment_at_decision` | 26 | action | liver_metastasis_local, liver_transplant, lung_metastasis_local, metastatic_primary, metastatic_staging, nodal_metastasis_local, oligometastatic_local, ovarian_metastasis_local, peritoneal_metastasis_local | ASCO22-LOCAL-01, ASCO22-LOCAL-02, ASCO22-LOCAL-05, ASCO22-LOCAL-06, ASCO22-LOCAL-07, ASCO22-LOCAL-09, +20 more |
| `crc.local_treatment_date` | 25 | timing | liver_metastasis_local, liver_transplant, lung_metastasis_local, metastatic_primary, nodal_metastasis_local, oligometastatic_local, ovarian_metastasis_local, peritoneal_metastasis_local | ASCO22-LOCAL-01, ASCO22-LOCAL-02, ASCO22-LOCAL-05, ASCO22-LOCAL-06, ASCO22-LOCAL-07, ASCO22-LOCAL-09, +19 more |
| `crc.colon_adjuvant.decision_date` | 24 | timing | colon_adjuvant | ASCO19-S3-DUR-01, ASCO19-S3-DUR-02, ASCO22-S2-1.1, ASCO22-S2-1.2, ASCO22-S2-1.3, ASCO22-S2-1.4, +18 more |
| `crc.localized_colon.resected_status_at_adjuvant_decision` | 24 | eligibility | colon_adjuvant | ASCO19-S3-DUR-01, ASCO19-S3-DUR-02, ASCO22-S2-1.1, ASCO22-S2-1.2, ASCO22-S2-1.3, ASCO22-S2-1.4, +18 more |
| `crc.rectal_stage.clinical_t.pretreatment` | 24 | action, eligibility | rectal_assessment, rectal_neoadjuvant, rectal_organ_preservation, rectal_selective_radiation, rectal_surgery, rectal_tnt | ASCO24-LARC-2.1, ASTRO25-KQ1-01, ASTRO25-KQ1-04, ASTRO25-KQ1-05, ASTRO25-KQ1-07, ASTRO25-KQ2-04, +18 more |
| `crc.expanded_ras_result` | 21 | eligibility | metastatic_systemic | ASCO22-MCRC-07, ASCO22-MCRC-08, ASCO22-MCRC-09, ASCO22-MCRC-11, E26-SYS-1L-03, E26-SYS-1L-04, +15 more |
| `crc.colon_adjuvant.start_date` | 20 | timing | colon_adjuvant | ASCO19-S3-DUR-01, ASCO19-S3-DUR-02, ASCO22-S2-1.1, ASCO22-S2-1.2, ASCO22-S2-1.3, ASCO22-S2-1.4, +14 more |
| `crc.colon_adjuvant.therapy_plan_at_decision` | 20 | action | colon_adjuvant | ASCO19-S3-DUR-01, ASCO19-S3-DUR-02, ASCO22-S2-1.1, ASCO22-S2-1.2, ASCO22-S2-1.3, ASCO22-S2-1.4, +14 more |
| `crc.ecog_performance_status` | 18 | action, eligibility, exceptions | genomic_testing, liver_transplant, metastatic_systemic, rectal_assessment, rectal_tnt, survivorship_adjacent | ASCO-SGT-01, ASCO-SGT-12, ASCO22-MCRC-03, ASCO24-LARC-3.1, ASTRO25-KQ2-08b, E26-LOCAL-14, +8 more |
| `crc.rectal_stage.clinical_n.pretreatment` | 18 | action, eligibility | rectal_assessment, rectal_neoadjuvant, rectal_organ_preservation, rectal_surgery, rectal_tnt | ASCO24-LARC-1.3, ASTRO25-KQ1-01, ASTRO25-KQ1-05, ASTRO25-KQ1-07, ASTRO25-KQ2-04, ASTRO25-KQ3-01, +12 more |
| `crc.age_at_decision` | 16 | eligibility, exceptions | colon_adjuvant, liver_transplant, metastatic_systemic, rectal_tnt | ASCO22-MCRC-03, ASCO22-S2-1.2, ASCO24-LARC-3.1, ASTRO25-KQ2-08b, E26-LOCAL-14, E26-SYS-1L-01, +7 more |
| `crc.rectal_assessment.completed_components.pretreatment` | 15 | action | rectal_assessment | ASCO24-LARC-1.1, ASCO24-LARC-1.2, ASCO24-LARC-1.3, ASTRO25-KQ1-01, ASTRO25-KQ1-02, ER25-ASSESS-01, +9 more |
| `crc.rectal_anatomy.third.pretreatment` | 14 | action, eligibility | rectal_assessment, rectal_immunotherapy, rectal_neoadjuvant, rectal_organ_preservation, rectal_tnt | ASTRO25-KQ3-02, ER25-ASSESS-03, ER25-NOM-01, ER25-TX-01, ER25-TX-02, ER25-TX-03, +8 more |
| `crc.rectal_response.overall_category.post_neoadjuvant` | 14 | eligibility | rectal_immunotherapy, rectal_neoadjuvant, rectal_organ_preservation, rectal_restaging, rectal_surgery, rectal_timing | ASCO24-LARC-5.1, ASTRO25-KQ1-09, ASTRO25-KQ3-01, ASTRO25-KQ3-02, ER25-NOM-01, ER25-NOM-02, +8 more |
| `crc.therapy_regulatory_availability_at_decision` | 14 | eligibility, exceptions | metastatic_systemic | E26-SYS-1L-01, E26-SYS-1L-05, E26-SYS-1L-06, E26-SYS-2L-24, E26-SYS-3L-29, E26-SYS-3L-30, +3 more |
| `crc.colon_stage.pathologic_t.resection` | 12 | action, eligibility | colon_adjuvant | ASCO19-S3-DUR-01, ASCO19-S3-DUR-02, ASCO22-S2-1.1, ASCO22-S2-1.2, ASCO22-S2-1.3, ASCO22-S2-2.1b, +6 more |
| `crc.colon_adjuvant.regimen` | 11 | action | colon_adjuvant | ASCO19-S3-DUR-01, ASCO19-S3-DUR-02, ASCO22-S2-2.1a, ASCO22-S2-2.1b, ASCO22-S2-3.1b, ASCO22-S2-4.1, +5 more |
| `crc.liver_metastasis_resectability_at_decision` | 11 | eligibility, exceptions | liver_metastasis_local, liver_transplant, metastatic_mdt | ASCO22-LOCAL-06, ASCO22-LOCAL-09, ASCO22-LOCAL-11, E26-LOCAL-11, E26-LOCAL-12, E26-LOCAL-13, +1 more |
| `crc.rectal_nom.decision_date` | 11 | timing | rectal_organ_preservation, shared_decision_making | ASCO24-LARC-5.1, ASTRO25-KQ1-07, ASTRO25-KQ3-01, ASTRO25-KQ3-02, ER25-NOM-01, ER25-NOM-02, +5 more |
| `crc.rectal_surgery.definitive_surgery_date` | 11 | action, timing | rectal_radiation, rectal_timing | ASTRO25-KQ1-08, ER25-RESTAGE-04, ER25-RESTAGE-05, ER25-RESTAGE-06, ER25-RESTAGE-07, ER25-RESTAGE-08 |
| `crc.rectal_treatment.sequence` | 11 | action | rectal_organ_preservation, rectal_timing, rectal_tnt | ASCO24-LARC-3.1, ASTRO25-KQ2-05, ASTRO25-KQ2-06, ASTRO25-KQ2-07, ASTRO25-KQ2-08a, ASTRO25-KQ3-05, +5 more |
| `crc.molecular_assay_method` | 10 | action, eligibility | genomic_testing, metastatic_systemic, molecular_testing, rectal_assessment | ASCO-SGT-09, ASCO24-LARC-1.1, ASTRO25-KQ1-02, CAP17-BIO-02, E26-MOL-009, E26-MOL-016, +4 more |
| `crc.molecular_variant_pathogenicity` | 10 | action, eligibility, exceptions | germline_testing, molecular_interpretation, molecular_testing, pharmacogenomics | ASCO-SGT-05, ASCO-SGT-06, E26-MOL-017, EC20-DPD-HET-01, EC20-DPD-HOM-01, SITC23-CRC-03 |
| `crc.rectal_nom.management_plan_at_decision` | 10 | action | rectal_organ_preservation | ASCO24-LARC-5.1, ASTRO25-KQ1-07, ASTRO25-KQ3-01, ASTRO25-KQ3-02, ER25-NOM-01, ER25-NOM-02, +4 more |
| `crc.rectal_response.assessment_date.post_neoadjuvant` | 10 | action, timing | rectal_restaging, rectal_timing | ASTRO25-KQ3-06, ER25-RESTAGE-01, ER25-RESTAGE-02, ER25-RESTAGE-03, ER25-RESTAGE-09, ER25-RESTAGE-10, +1 more |
| `crc.treatment_selection_rationale` | 10 | eligibility, exceptions | liver_metastasis_local, metastatic_systemic | ASCO22-LOCAL-07, ASCO22-MCRC-03, ASCO22-MCRC-10, E26-SYS-1L-08, E26-SYS-1L-12, E26-SYS-1L-16, +1 more |
| `crc.comorbidity_constraints` | 9 | action, exceptions | colon_adjuvant, lung_metastasis_local, metastatic_systemic, rectal_tnt, survivorship_adjacent | ASCO22-MCRC-03, ASCO22-S2-1.3, ASCO24-LARC-3.1, ASTRO25-KQ2-08b, E26-LOCAL-16, EC20-ADJ-III-02, +3 more |
| `crc.molecular_variant_gene` | 9 | action, eligibility | genomic_testing, molecular_interpretation, molecular_reporting, molecular_testing | ASCO-SGT-09, CAP17-BIO-01, CAP17-BIO-21, E26-MOL-009, E26-MOL-010, E26-MOL-017, +2 more |
| `crc.rectal_mri.mrf_distance_mm.pretreatment` | 9 | action, eligibility | rectal_assessment, rectal_neoadjuvant, rectal_selective_radiation, rectal_tnt | ASCO24-LARC-1.3, ASTRO25-KQ1-04, ASTRO25-KQ2-06, ER25-ASSESS-06, ER25-TX-09, ER25-TX-14, +3 more |
| `crc.rectal_mri.emvi_status.pretreatment` | 8 | action, eligibility | rectal_assessment, rectal_selective_radiation, rectal_surgery, rectal_tnt | ASCO24-LARC-1.3, ASCO24-LARC-2.1, ASTRO25-KQ1-04, ASTRO25-KQ1-05, ASTRO25-KQ2-06, ER25-ASSESS-06, +2 more |
| `crc.regulatory_jurisdiction` | 8 | eligibility, exceptions | genomic_testing, genomic_treatment, molecular_testing, rectal_immunotherapy | ASCO-SGT-01, ASCO-SGT-09, ASCO-SGT-14, ASCO24-LARC-6.1, E26-MOL-012, E26-MOL-018, +1 more |
| `crc.resectability_status` | 8 | eligibility, exceptions | metastatic_primary, metastatic_resectability, metastatic_systemic, oligometastatic_local, ovarian_metastasis_local | ASCO22-MCRC-01, E26-LOCAL-02, E26-LOCAL-03, E26-LOCAL-06, E26-LOCAL-19, E26-STAGE-07, +1 more |
| `crc.braf_variant` | 7 | eligibility, exceptions | metastatic_systemic | ASCO22-MCRC-13, E26-SYS-1L-05, E26-SYS-1L-06, E26-SYS-1L-07, E26-SYS-1L-08, E26-SYS-2L-23 |
| `crc.colon_adjuvant.planned_or_delivered_duration_months` | 7 | timing | colon_adjuvant | ASCO19-S3-DUR-01, ASCO19-S3-DUR-02, ASCO22-S2-4.1, EC20-ADJ-II-HIGH-DUR-01, EC20-ADJ-III-02, EC20-ADJ-III-03, +1 more |
| `crc.dpd_status.before_fluoropyrimidine` | 7 | exceptions | colon_adjuvant, rectal_neoadjuvant, rectal_radiation, rectal_tnt | ASTRO25-KQ2-03, ASTRO25-KQ2-08a, EC20-ADJ-III-01, EC20-ADJ-III-04, ER25-TX-06, ER25-TX-07, +1 more |
| `crc.metastasis_ablatability_at_decision` | 7 | eligibility, exceptions | liver_metastasis_local, oligometastatic_local | E26-LOCAL-03, E26-LOCAL-11, E26-LOCAL-12, E26-LOCAL-13 |
| `crc.molecular_variant_hgvs` | 7 | action, eligibility | clinical_trial, genomic_treatment, molecular_interpretation, molecular_testing | ASCO-SGT-05, ASCO-SGT-13, ASCO-SGT-14, CAP17-BIO-01, E26-MOL-009, E26-MOL-017, +1 more |
| `crc.multidisciplinary_review` | 7 | action, eligibility, exceptions | liver_metastasis_local, metastatic_mdt, metastatic_resectability, oligometastatic_local | ASCO22-LOCAL-03, ASCO22-LOCAL-08, ASCO22-LOCAL-09, ASCO22-LOCAL-11, E26-LOCAL-06, E26-STAGE-06, +1 more |
| `crc.tmb_value` | 7 | action, eligibility, exceptions | genomic_testing, molecular_interpretation, molecular_testing | ASCO-SGT-08, CAP22-MMR-002, CAP22-MMR-003, E26-MOL-017, SITC23-CRC-03 |
| `crc.genetics_referral_status` | 6 | action, exceptions | lynch_reflex, molecular_testing | CAP17-BIO-03, CAP17-BIO-04, CAP22-MMR-004, E26-MOL-015, EC20-LYNCH-MSH-01, EC20-LYNCH-SPORADIC-01 |
| `crc.metastatic_lesion_count_at_decision` | 6 | eligibility | liver_metastasis_local, lung_metastasis_local, oligometastatic_local | ASCO22-LOCAL-10, E26-LOCAL-01, E26-LOCAL-03, E26-LOCAL-05, E26-LOCAL-13, E26-LOCAL-16 |
| `crc.molecular_specimen_id` | 6 | action, eligibility | genomic_testing, molecular_operations, molecular_testing | CAP17-BIO-08, E26-MOL-004, E26-MOL-009, E26-MOL-011, E26-MOL-016, SITC23-CRC-01 |
| `crc.molecular_variant_allele_fraction` | 6 | action | molecular_operations, molecular_reporting, molecular_testing | CAP17-BIO-01, CAP17-BIO-02, CAP17-BIO-19, E26-MOL-009, E26-MOL-010, E26-MOL-011 |
| `crc.pathology.regional_nodes_examined_count.resection` | 6 | action, eligibility | colon_adjuvant | ASCO22-S2-1.2, ASCO22-S2-1.4, EC20-ADJ-II-HIGH-OX-01, EC20-ADJ-II-INT-01, EC20-ADJ-II-LOW-01, EC20-RISK-02 |
| `crc.planned_drug` | 6 | eligibility | molecular_testing, pharmacogenomics | E26-MOL-019, EC20-DPD-01, EC20-DPD-HET-01, EC20-DPD-HOM-01, EC20-DPD-U150-01, EC20-DPD-U16-01 |
| `crc.prior_systemic_therapy` | 6 | eligibility, exceptions | metastatic_systemic | E26-SYS-2L-24, E26-SYS-3L-25, E26-SYS-3L-30, E26-SYS-3L-31, E26-SYS-3L-38 |
| `crc.rectal_mri.mrf_status.pretreatment` | 6 | action, eligibility | rectal_assessment, rectal_neoadjuvant, rectal_surgery, rectal_tnt | ASCO24-LARC-2.1, ASTRO25-KQ1-05, ER25-ASSESS-07, ER25-TX-09, ER25-TX-10, ER25-TX-11 |
| `crc.rectal_neoadjuvant.systemic_regimen` | 6 | action | rectal_neoadjuvant, rectal_tnt | ASTRO25-KQ2-08a, ASTRO25-KQ2-08b, ER25-TX-07, ER25-TX-08, ER25-TX-15, ER25-TX-20 |
| `crc.shared_decision_discussion` | 6 | action, eligibility, exceptions | liver_metastasis_local, shared_decision_making | ASCO22-LOCAL-04, ASCO22-LOCAL-09, ASCO22-LOCAL-10, ASCO22-MCRC-05, ASCO22-MCRC-12 |
| `crc.treatment_response_before_local_decision` | 6 | eligibility, exceptions | liver_metastasis_local, oligometastatic_local, oligometastatic_systemic | ASCO22-LOCAL-06, E26-LOCAL-04, E26-LOCAL-07, E26-LOCAL-09 |
| `crc.colon_stage.pathologic_n.resection` | 5 | action, eligibility | colon_adjuvant | ASCO19-S3-DUR-01, ASCO19-S3-DUR-02, EC20-ADJ-III-02, EC20-ADJ-III-03, EC20-RISK-02 |
| `crc.colon_stage_ii.high_risk_factors` | 5 | eligibility | colon_adjuvant | ASCO22-S2-1.1, ASCO22-S2-2.1b, ASCO22-S2-3.1a, ASCO22-S2-3.1b, EC20-ADJ-II-LOW-01 |
| `crc.fusion_in_frame_status` | 5 | action, eligibility | genomic_testing, metastatic_systemic, molecular_testing | ASCO-SGT-09, ASCO-SGT-10, E26-MOL-018, E26-SYS-3L-33, E26-SYS-3L-35 |
| `crc.ici_plan` | 5 | eligibility | molecular_interpretation, molecular_testing | ASCO-SGT-07, CAP22-MMR-001, CAP22-MMR-002, SITC23-CRC-02, SITC23-CRC-03 |
| `crc.mdt_review_date` | 5 | timing | metastatic_mdt, oligometastatic_local | ASCO22-LOCAL-03, ASCO22-LOCAL-08, ASCO22-LOCAL-11, E26-LOCAL-01, E26-STAGE-06 |
| `crc.mlh1_promoter_methylation_status` | 5 | action, eligibility | lynch_reflex | CAP17-BIO-03, CAP22-MMR-004, E26-MOL-015, EC20-LYNCH-MLH1-01, EC20-LYNCH-SPORADIC-01 |
| `crc.mmr_mlh1` | 5 | action, eligibility | lynch_reflex, molecular_testing | CAP17-BIO-03, CAP17-BIO-04, CAP22-MMR-001, E26-MOL-014, EC20-LYNCH-MLH1-01 |
| `crc.mmr_msh2` | 5 | action, eligibility | lynch_reflex, molecular_testing | CAP17-BIO-03, CAP17-BIO-04, CAP22-MMR-001, E26-MOL-014, EC20-LYNCH-MSH-01 |
| `crc.mmr_msh6` | 5 | action, eligibility | lynch_reflex, molecular_testing | CAP17-BIO-03, CAP17-BIO-04, CAP22-MMR-001, E26-MOL-014, EC20-LYNCH-MSH-01 |
| `crc.mmr_pms2` | 5 | action, eligibility | lynch_reflex, molecular_testing | CAP17-BIO-03, CAP17-BIO-04, CAP22-MMR-001, E26-MOL-014, EC20-LYNCH-MLH1-01 |
| `crc.molecular_test_order_date` | 5 | eligibility, timing | molecular_operations, molecular_testing | CAP17-BIO-14, CAP17-BIO-16, E26-MOL-006, E26-MOL-007 |
| `crc.molecular_testing_policy_version` | 5 | eligibility, exceptions, timing | molecular_operations | CAP17-BIO-14, CAP17-BIO-15, CAP17-BIO-16 |
| `crc.msi_pcr_result` | 5 | action, eligibility | molecular_interpretation, molecular_operations, molecular_testing | ASCO-SGT-07, CAP22-MMR-001, CAP22-MMR-003, CAP22-MMR-005, CAP22-MMR-007 |
| `crc.oxaliplatin_fitness_at_decision` | 5 | eligibility, exceptions | colon_adjuvant, metastatic_systemic, oligometastatic_systemic | E26-LOCAL-08, E26-SYS-1L-05, E26-SYS-1L-07, EC20-ADJ-III-01 |
| `crc.rectal_neoadjuvant.completion_date` | 5 | action, timing | rectal_restaging, rectal_timing | ASTRO25-KQ3-06, ER25-RESTAGE-04, ER25-RESTAGE-09 |
| `crc.rectal_neoadjuvant.systemic_duration_months` | 5 | timing | rectal_immunotherapy, rectal_neoadjuvant, rectal_tnt | ASTRO25-KQ2-08a, ASTRO25-KQ2-08b, ER25-TX-07, ER25-TX-15, ER25-TX-22 |
| `crc.rectal_radiation.fraction_count` | 5 | action | rectal_radiation, rectal_tnt | ASTRO25-KQ2-01, ASTRO25-KQ2-02, ASTRO25-KQ3-03, ASTRO25-KQ3-04, ER25-TX-06 |
| `crc.rectal_radiation.radiosensitizer` | 5 | action | rectal_radiation, rectal_tnt | ASTRO25-KQ2-01, ASTRO25-KQ2-03, ASTRO25-KQ3-03, ASTRO25-KQ3-04, ER25-TX-06 |
| `crc.rectal_radiation.total_dose_cgy` | 5 | action | rectal_radiation, rectal_tnt | ASTRO25-KQ2-01, ASTRO25-KQ2-02, ASTRO25-KQ3-03, ASTRO25-KQ3-04, ER25-TX-06 |
| `crc.rectal_risk.high_risk_features.pretreatment` | 5 | eligibility | rectal_neoadjuvant, rectal_radiation, rectal_tnt | ASCO24-LARC-2.2, ASTRO25-KQ1-03, ER25-TX-02, ER25-TX-05, ER25-TX-12 |
| `crc.rectal_treatment.goal` | 5 | exceptions | rectal_organ_preservation, rectal_radiation, rectal_surgery, rectal_tnt | ASCO24-LARC-4.1, ASTRO25-KQ1-05, ASTRO25-KQ3-05, ER25-NOM-01, ER25-TX-13 |
| `crc.staging_assessment_date` | 5 | timing | metastatic_staging | E26-STAGE-01, E26-STAGE-02, E26-STAGE-03, E26-STAGE-04, E26-STAGE-05 |
| `crc.trial_availability` | 5 | action, eligibility, exceptions | clinical_trial, genomic_treatment, metastatic_systemic | ASCO-SGT-13, ASCO-SGT-14, E26-SYS-3L-30 |
| `crc.biopsy_date` | 4 | action, eligibility, timing | molecular_operations | E26-MOL-002, E26-MOL-003 |
| `crc.braf_variant_hgvs` | 4 | action, eligibility | lynch_reflex, molecular_testing | CAP17-BIO-02, E26-MOL-011, E26-MOL-012, EC20-LYNCH-SPORADIC-01 |
| `crc.colon_adjuvant.risk_assessment_documented` | 4 | action | colon_adjuvant | EC20-RISK-01, EC20-RISK-02, EC20-RISK-03, EC20-RISK-04 |
| `crc.colon_stage.pathologic_stage_group.resection` | 4 | action, eligibility | colon_adjuvant | ASCO22-S2-1.1, ASCO22-S2-1.3, EC20-ADJ-III-01, EC20-RISK-02 |
| `crc.complete_macroscopic_resection_feasibility_at_decision` | 4 | eligibility, exceptions | peritoneal_metastasis_local | ASCO22-LOCAL-01, E26-LOCAL-17 |
| `crc.genomic_test_status` | 4 | action | genomic_testing | ASCO-SGT-01, ASCO-SGT-03, ASCO-SGT-12, SITC23-CRC-01 |
| `crc.her2_amplification_result` | 4 | eligibility, exceptions | metastatic_systemic | E26-SYS-3L-29, E26-SYS-3L-30, E26-SYS-3L-31 |
| `crc.her2_protein_expression_result` | 4 | eligibility, exceptions | metastatic_systemic | E26-SYS-3L-29, E26-SYS-3L-30, E26-SYS-3L-31 |
| `crc.mdt_participating_specialties` | 4 | eligibility | metastatic_mdt | ASCO22-LOCAL-03, ASCO22-LOCAL-08, ASCO22-LOCAL-11, E26-STAGE-06 |
| `crc.metastatic_lesion_location_at_decision` | 4 | eligibility, exceptions | liver_metastasis_local, lung_metastasis_local | E26-LOCAL-11, E26-LOCAL-12, E26-LOCAL-16 |
| `crc.metastatic_lesion_size_at_decision` | 4 | eligibility | liver_metastasis_local, lung_metastasis_local | ASCO22-LOCAL-10, E26-LOCAL-11, E26-LOCAL-12, E26-LOCAL-16 |
| `crc.metastatic_sites_at_decision` | 4 | eligibility | metastatic_mdt, metastatic_staging, oligometastatic_local | E26-LOCAL-01, E26-LOCAL-05, E26-STAGE-01, E26-STAGE-06 |
| `crc.mmr_ihc_controls_status` | 4 | action, eligibility | lynch_reflex, molecular_operations, molecular_testing | CAP17-BIO-12, CAP22-MMR-001, E26-MOL-014, EC20-LYNCH-MSH-01 |
| `crc.molecular_specimen_date` | 4 | eligibility, timing | molecular_operations | CAP17-BIO-08, E26-MOL-004 |
| `crc.molecular_specimen_site` | 4 | action, eligibility | molecular_operations, molecular_testing | CAP17-BIO-08, E26-MOL-001, E26-MOL-004, E26-MOL-006 |
| `crc.multidisciplinary_team.decision` | 4 | action | rectal_adjuvant, rectal_assessment, rectal_neoadjuvant, rectal_surgery | ER25-ASSESS-01, ER25-TX-20, ER25-TX-24, ER25-TX-25 |
| `crc.pathology.grade.resection` | 4 | action, eligibility | colon_adjuvant | ASCO22-S2-1.2, ASCO22-S2-1.4, ASCO22-S2-2.1b, EC20-RISK-03 |
| `crc.pretreatment_uracil_unit` | 4 | eligibility, exceptions | pharmacogenomics | EC20-DPD-U150-01, EC20-DPD-U16-01 |
| `crc.prior_anti_egfr_before_decision` | 4 | eligibility, exceptions | metastatic_systemic | E26-SYS-2L-24, E26-SYS-3L-26, E26-SYS-3L-28 |
| `crc.prior_ici_before_decision` | 4 | eligibility | metastatic_systemic | E26-SYS-1L-01, E26-SYS-2L-21, E26-SYS-2L-22, E26-SYS-2L-23 |
| `crc.prior_irinotecan_before_decision` | 4 | eligibility | metastatic_systemic | E26-SYS-3L-27, E26-SYS-3L-29, E26-SYS-3L-32, E26-SYS-3L-37 |
| `crc.prior_oxaliplatin_before_decision` | 4 | eligibility | metastatic_systemic | E26-SYS-1L-06, E26-SYS-3L-29, E26-SYS-3L-32, E26-SYS-3L-37 |
| `crc.prior_systemic_line_count_at_decision` | 4 | eligibility, exceptions | liver_metastasis_local, liver_transplant, metastatic_systemic | ASCO22-MCRC-13, E26-LOCAL-14, E26-LOCAL-15 |
| `crc.progression_date` | 4 | eligibility, timing | metastatic_systemic, molecular_testing | ASCO22-MCRC-13, E26-MOL-013, E26-SYS-3L-34 |
| `crc.rectal_nom.surveillance_capacity` | 4 | exceptions | rectal_organ_preservation | ASCO24-LARC-5.1, ASTRO25-KQ3-01, ER25-NOM-01, ER25-NOM-02 |
| `crc.rectal_radiation.chemoradiation_completion_date` | 4 | action, timing | rectal_timing | ER25-RESTAGE-05, ER25-RESTAGE-07 |
| `crc.rectal_radiation.start_date` | 4 | action, timing | rectal_timing | ER25-RESTAGE-10, ER25-RESTAGE-11 |
| `crc.rectal_response.endoscopy_category.post_neoadjuvant` | 4 | action, eligibility | rectal_organ_preservation, rectal_restaging, rectal_selective_radiation | ASCO24-LARC-5.1, ASTRO25-KQ1-04, ER25-RESTAGE-01, ER25-TX-21 |
| `crc.rectal_response.repeat_assessment_date.after_near_ccr` | 4 | action, timing | rectal_timing | ER25-RESTAGE-10, ER25-RESTAGE-11 |
| `crc.rectal_restaging.completed_modalities.post_neoadjuvant` | 4 | action | rectal_restaging | ASTRO25-KQ3-06, ER25-RESTAGE-01, ER25-RESTAGE-02, ER25-RESTAGE-03 |
| `crc.rectal_surgery.intent_at_neoadjuvant_decision` | 4 | eligibility | rectal_neoadjuvant, rectal_tnt | ER25-TX-01, ER25-TX-02, ER25-TX-11, ER25-TX-18 |
| `crc.rectal_surgery.planned_procedure` | 4 | action | rectal_organ_preservation, rectal_surgery | ASTRO25-KQ1-05, ASTRO25-KQ1-07, ER25-NOM-05, ER25-TX-24 |
| `crc.rectal_tnt.systemic_component` | 4 | action | rectal_tnt | ASTRO25-KQ1-06, ASTRO25-KQ2-05, ASTRO25-KQ3-05, ER25-TX-13 |
| `crc.tmb_unit` | 4 | action, eligibility, exceptions | genomic_testing, molecular_interpretation, molecular_testing | ASCO-SGT-08, CAP22-MMR-002, E26-MOL-017 |
| `crc.treatment_center_expertise` | 4 | eligibility, exceptions | liver_metastasis_local, peritoneal_metastasis_local | ASCO22-LOCAL-01, ASCO22-LOCAL-02, E26-LOCAL-15, E26-LOCAL-17 |
| `crc.treatment_toxicity_risk_at_decision` | 4 | action, exceptions | colon_adjuvant, metastatic_systemic, shared_decision_making | ASCO22-MCRC-05, E26-SYS-1L-12, E26-SYS-1L-17, EC20-RISK-04 |
| `crc.trial_context` | 4 | exceptions | historical_evidence, metastatic_systemic, peritoneal_metastasis_local | CAP17-BIO-06, CAP17-BIO-07, E26-LOCAL-18, SITC23-CRC-04 |
| `crc.anti_egfr_washout_interval` | 3 | eligibility, timing | metastatic_systemic, molecular_testing | E26-MOL-013, E26-SYS-3L-28 |
| `crc.bevacizumab_contraindication_at_decision` | 3 | exceptions | metastatic_systemic | E26-SYS-1L-18, E26-SYS-3L-37, E26-SYS-3L-38 |
| `crc.cea_at_decision` | 3 | eligibility, exceptions | liver_transplant, oligometastatic_local | E26-LOCAL-05, E26-LOCAL-14 |
| `crc.colon_presentation.obstruction` | 3 | eligibility | colon_adjuvant, rectal_assessment | ASCO22-S2-1.2, ASCO22-S2-1.4, ER25-ASSESS-03 |
| `crc.estimated_life_expectancy` | 3 | action, eligibility, exceptions | colon_adjuvant, genomic_testing | ASCO-SGT-01, EC20-RISK-04 |
| `crc.fusion_partner` | 3 | action | genomic_testing, molecular_testing | ASCO-SGT-09, ASCO-SGT-10, E26-MOL-018 |
| `crc.fusion_test_status` | 3 | action | genomic_testing | ASCO-SGT-09, ASCO-SGT-10, ASCO-SGT-11 |
| `crc.genomic_panel_content` | 3 | action, eligibility | genomic_testing | ASCO-SGT-02, ASCO-SGT-11, SITC23-CRC-01 |
| `crc.germline_result` | 3 | action, exceptions | germline_testing, lynch_reflex | ASCO-SGT-06, E26-MOL-015, EC20-LYNCH-SPORADIC-01 |
| `crc.mmr_loss_pattern` | 3 | action, eligibility | lynch_reflex, molecular_testing | CAP17-BIO-04, CAP22-MMR-004, E26-MOL-015 |
| `crc.molecular_assay_lod` | 3 | action, eligibility | molecular_operations, molecular_testing | CAP17-BIO-10, CAP17-BIO-19, E26-MOL-013 |
| `crc.molecular_assay_validation_status` | 3 | action, exceptions | molecular_operations, molecular_testing | ASCO-SGT-07, CAP17-BIO-09 |
| `crc.molecular_assay_version` | 3 | action, eligibility | molecular_operations | ASCO-SGT-04, CAP17-BIO-10, CAP17-BIO-11 |
| `crc.molecular_clinical_evidence` | 3 | action, exceptions | genomic_treatment, molecular_interpretation | ASCO-SGT-05, ASCO-SGT-14 |
| `crc.molecular_final_interpretation` | 3 | action, exceptions | molecular_interpretation, molecular_reporting | CAP17-BIO-21, CAP22-MMR-005 |
| `crc.molecular_specimen_adequacy` | 3 | eligibility, exceptions | molecular_operations | CAP17-BIO-08, E26-MOL-004 |
| `crc.molecular_variant_codon` | 3 | action | molecular_reporting, molecular_testing | CAP17-BIO-01, E26-MOL-009, E26-MOL-010 |
| `crc.molecular_variant_exon` | 3 | action | molecular_reporting, molecular_testing | CAP17-BIO-01, E26-MOL-009, E26-MOL-010 |
| `crc.molecular_variant_hgvs_c` | 3 | action, eligibility | molecular_reporting | CAP17-BIO-21, E26-MOL-010 |
| `crc.molecular_variant_hgvs_p` | 3 | action, eligibility | molecular_reporting | CAP17-BIO-21, E26-MOL-010 |
| `crc.ntrk_fusion_result` | 3 | eligibility, exceptions | metastatic_systemic | E26-SYS-3L-33, E26-SYS-3L-34 |
| `crc.oxaliplatin_contraindication_at_decision` | 3 | exceptions | metastatic_systemic | E26-SYS-1L-06, E26-SYS-1L-07, E26-SYS-1L-15 |
| `crc.pathology.lymphovascular_invasion.resection` | 3 | action, eligibility | colon_adjuvant | ASCO22-S2-1.2, ASCO22-S2-1.4, EC20-RISK-03 |
| `crc.pathology.perineural_invasion.resection` | 3 | action, eligibility | colon_adjuvant | ASCO22-S2-1.2, ASCO22-S2-1.4, EC20-RISK-03 |
| `crc.patient.oxaliplatin_neuropathy_risk_at_decision` | 3 | exceptions | colon_adjuvant | ASCO19-S3-DUR-02, ASCO22-S2-3.1a, ASCO22-S2-4.1 |
| `crc.peritoneal_metastatic_disease_at_decision` | 3 | eligibility | metastatic_staging, peritoneal_metastasis_local | ASCO22-LOCAL-01, E26-LOCAL-18, E26-STAGE-04 |
| `crc.pole_pold1_proofreading_domain_status` | 3 | action, exceptions | molecular_interpretation, molecular_testing | E26-MOL-017, SITC23-CRC-03 |
| `crc.pole_pold1_variant` | 3 | eligibility, exceptions | metastatic_systemic | E26-SYS-3L-36, SITC23-CRC-04 |
| `crc.prior_fluoropyrimidine_before_decision` | 3 | eligibility | metastatic_systemic | E26-SYS-3L-29, E26-SYS-3L-32, E26-SYS-3L-37 |
| `crc.progression_before_decision` | 3 | eligibility, exceptions | metastatic_systemic | E26-SYS-2L-23, E26-SYS-3L-25 |
| `crc.rectal_anatomy.anal_verge_distance_cm.pretreatment` | 3 | action, eligibility | rectal_assessment, rectal_selective_radiation | ASCO24-LARC-1.3, ASTRO25-KQ1-04, ER25-ASSESS-03 |
| `crc.rectal_anatomy.height_category.pretreatment` | 3 | eligibility | rectal_surgery, rectal_tnt | ASCO24-LARC-2.1, ASTRO25-KQ1-05, ASTRO25-KQ2-06 |
| `crc.rectal_immunotherapy.agent` | 3 | action | rectal_immunotherapy | ASCO24-LARC-6.1, ASTRO25-KQ1-09, ER25-TX-22 |
| `crc.rectal_mri.image_quality.pretreatment` | 3 | action, eligibility | rectal_assessment, rectal_tnt | ASCO24-LARC-1.2, ER25-ASSESS-05, ER25-TX-05 |
| `crc.rectal_mri.lateral_node_status.pretreatment` | 3 | action, eligibility | rectal_assessment, rectal_tnt | ASTRO25-KQ2-06, ER25-ASSESS-06, ER25-TX-11 |
| `crc.rectal_neoadjuvant.protocol_name` | 3 | action | rectal_assessment, rectal_tnt | ASCO24-LARC-1.2, ER25-ASSESS-05, ER25-TX-08 |
| `crc.rectal_radiation.course_type` | 3 | action | rectal_radiation, rectal_tnt | ASCO24-LARC-4.1, ASTRO25-KQ2-05, ASTRO25-KQ2-07 |
| `crc.rectal_response.dre_category.post_neoadjuvant` | 3 | action | rectal_organ_preservation, rectal_restaging | ASCO24-LARC-5.1, ER25-RESTAGE-01, ER25-TX-21 |
| `crc.rectal_response.mri_category.post_neoadjuvant` | 3 | action | rectal_organ_preservation, rectal_restaging | ASCO24-LARC-5.1, ER25-RESTAGE-01, ER25-TX-21 |
| `crc.rectal_stage.clinical_stage_group.pretreatment` | 3 | eligibility | rectal_radiation, rectal_tnt, survivorship_adjacent | ASTRO25-KQ1-03, ASTRO25-KQ1-06, EC26-EXERCISE-01 |
| `crc.rectal_surgery.decision_date` | 3 | timing | rectal_surgery | ASTRO25-KQ1-05, ER25-NOM-05, ER25-TX-24 |
| `crc.rectal_tnt.radiation_component` | 3 | action | rectal_tnt | ASTRO25-KQ1-06, ASTRO25-KQ3-05, ER25-TX-13 |
| `crc.rectal_treatment.organ_preservation_goal` | 3 | exceptions | rectal_neoadjuvant, rectal_organ_preservation, rectal_tnt | ASCO24-LARC-2.2, ER25-TX-03, ER25-TX-12 |
| `crc.rectal_tumor.maximum_diameter_cm.pretreatment` | 3 | eligibility | rectal_neoadjuvant, rectal_organ_preservation | ASTRO25-KQ3-02, ER25-TX-01, ER25-TX-04 |
| `crc.recurrence_risk_group` | 3 | eligibility | molecular_testing, pharmacogenomics | EC20-DPD-ALT-01, EC20-GEX-01, EC20-IMMUNOSCORE-01 |
| `crc.source_effective_date` | 3 | eligibility, exceptions, timing | historical_evidence | CAP17-BIO-05 |
| `crc.stage_group` | 3 | eligibility | genomic_testing, molecular_testing | ASCO-SGT-01, EC20-IMMUNOSCORE-01, EC20-MMR-01 |
| `crc.total_perioperative_systemic_therapy_duration` | 3 | exceptions, timing | oligometastatic_systemic | ASCO22-LOCAL-12, E26-LOCAL-09, E26-LOCAL-10 |
| `crc.treatment_toxicity_at_decision` | 3 | exceptions | oligometastatic_systemic | ASCO22-LOCAL-12, E26-LOCAL-09, E26-LOCAL-10 |
| `crc.tumor_mutational_burden_result` | 3 | eligibility, exceptions | metastatic_systemic | E26-SYS-3L-36, SITC23-CRC-04 |
| `crc.alternative_adjuvant_regimen` | 2 | action, exceptions | pharmacogenomics | EC20-DPD-ALT-01 |
| `crc.anti_egfr_plan` | 2 | eligibility | molecular_testing | CAP17-BIO-01, E26-MOL-009 |
| `crc.biomarker_predictive_claim` | 2 | exceptions | molecular_testing | EC20-GEX-01, EC20-IMMUNOSCORE-01 |
| `crc.broad_tumor_ngs_available` | 2 | eligibility | molecular_testing | E26-MOL-017, E26-MOL-018 |
| `crc.cea.value_at_adjuvant_decision` | 2 | action | colon_adjuvant, rectal_assessment | EC20-RISK-03, ER25-ASSESS-02 |
| `crc.colon_presentation.perforation` | 2 | eligibility | colon_adjuvant | ASCO22-S2-1.2, ASCO22-S2-1.4 |
| `crc.ctdna_resistance_alterations` | 2 | eligibility, exceptions | metastatic_systemic | E26-SYS-3L-28 |
| `crc.cytoreductive_surgery_event` | 2 | action, eligibility | peritoneal_metastasis_local | ASCO22-LOCAL-05 |
| `crc.diagnosis_date` | 2 | eligibility, timing | molecular_testing | E26-MOL-006 |
| `crc.disease_control_duration_before_decision` | 2 | exceptions, timing | liver_transplant | E26-LOCAL-14 |
| `crc.dpd_phenotype` | 2 | action | molecular_testing | E26-MOL-019, EC20-DPD-01 |
| `crc.dpyd_function_status_at_decision` | 2 | eligibility | metastatic_systemic | E26-SYS-1L-18, E26-SYS-1L-19 |
| `crc.dpyd_genotype` | 2 | action | molecular_testing | E26-MOL-019, EC20-DPD-01 |
| `crc.dpyd_variant` | 2 | eligibility | pharmacogenomics | EC20-DPD-HET-01, EC20-DPD-HOM-01 |
| `crc.dpyd_zygosity` | 2 | eligibility | pharmacogenomics | EC20-DPD-HET-01, EC20-DPD-HOM-01 |
| `crc.extra_peritoneal_disease_at_decision` | 2 | eligibility, exceptions | peritoneal_metastasis_local | ASCO22-LOCAL-01, E26-LOCAL-17 |
| `crc.extra_target_organ_metastatic_disease_at_decision` | 2 | eligibility | nodal_metastasis_local | E26-LOCAL-20, E26-LOCAL-21 |
| `crc.extrahepatic_disease_ever_before_decision` | 2 | eligibility, exceptions | liver_transplant | E26-LOCAL-14 |
| `crc.first_oncology_consult_date` | 2 | eligibility, timing | molecular_operations | E26-MOL-002 |
| `crc.fluoropyrimidine_initial_dose` | 2 | action | pharmacogenomics | EC20-DPD-HET-01, EC20-DPD-U16-01 |
| `crc.fluoropyrimidine_treatment_decision` | 2 | action | pharmacogenomics | EC20-DPD-HOM-01, EC20-DPD-U150-01 |
| `crc.fusion_assay_nucleic_acid` | 2 | action | genomic_testing, molecular_testing | ASCO-SGT-10, E26-MOL-018 |
| `crc.genomic_variant_classes` | 2 | action | genomic_testing | ASCO-SGT-03, SITC23-CRC-01 |
| `crc.hipec_agent` | 2 | action | peritoneal_metastasis_local | ASCO22-LOCAL-05, E26-LOCAL-18 |
| `crc.hipec_event` | 2 | action | peritoneal_metastasis_local | ASCO22-LOCAL-05, E26-LOCAL-18 |
| `crc.ici_biomarker_basis` | 2 | action | molecular_interpretation | CAP22-MMR-002, SITC23-CRC-02 |
| `crc.kras_variant` | 2 | eligibility, exceptions | metastatic_systemic | E26-SYS-3L-32 |
| `crc.liver_function_at_decision` | 2 | eligibility, exceptions | liver_metastasis_local | E26-LOCAL-15 |
| `crc.liver_metastasis_burden_at_decision` | 2 | eligibility | liver_metastasis_local, metastatic_staging | ASCO22-LOCAL-06, E26-STAGE-03 |
| `crc.liver_mri_result` | 2 | action, eligibility | metastatic_staging | E26-STAGE-03 |
| `crc.liver_only_metastatic_disease_at_decision` | 2 | eligibility, exceptions | liver_transplant | E26-LOCAL-14 |
| `crc.local_treatment_event` | 2 | eligibility | oligometastatic_systemic | E26-LOCAL-08, E26-LOCAL-09 |
| `crc.local_treatment_feasibility_at_decision` | 2 | eligibility, exceptions | oligometastatic_local | E26-LOCAL-04 |
| `crc.lynch_risk_communication` | 2 | action | lynch_reflex | CAP22-MMR-004, E26-MOL-015 |
| `crc.mcrc_suspicion_date` | 2 | eligibility, timing | molecular_operations | E26-MOL-001 |
| `crc.mdt_recommendation` | 2 | action | metastatic_mdt, oligometastatic_local | E26-LOCAL-01, E26-STAGE-06 |
| `crc.metastasis_timing` | 2 | eligibility | metastatic_primary, oligometastatic_local | E26-LOCAL-02, E26-LOCAL-05 |
| `crc.mmr_msi_discordance_status` | 2 | eligibility, exceptions | molecular_interpretation, molecular_testing | CAP22-MMR-005, E26-MOL-014 |
| `crc.mmr_msi_interpretive_rereview_status` | 2 | action, exceptions | molecular_interpretation | CAP22-MMR-005 |
| `crc.molecular_final_status` | 2 | action, exceptions | molecular_operations | CAP22-MMR-006 |
| `crc.molecular_order_authority` | 2 | eligibility, exceptions | molecular_operations | CAP17-BIO-16 |
| `crc.molecular_panel_version` | 2 | action | molecular_testing | CAP22-MMR-001, E26-MOL-006 |
| `crc.molecular_pathologist_reviewer` | 2 | action | molecular_operations | CAP17-BIO-18, E26-MOL-004 |
| `crc.molecular_result_date` | 2 | timing | molecular_operations | CAP17-BIO-20, E26-MOL-007 |
| `crc.molecular_specimen_inventory` | 2 | eligibility | molecular_operations | CAP17-BIO-15, E26-MOL-004 |
| `crc.molecular_testing_failure_reason` | 2 | action, exceptions | molecular_operations | CAP17-BIO-13, CAP17-BIO-15 |
| `crc.molecular_tumor_fraction` | 2 | action, eligibility | molecular_operations | CAP17-BIO-18, CAP17-BIO-19 |
| `crc.molecular_variant_transcript` | 2 | action | molecular_reporting | CAP17-BIO-21, E26-MOL-010 |
| `crc.msi_ngs_result` | 2 | action, eligibility | molecular_interpretation, molecular_testing | ASCO-SGT-07, CAP22-MMR-005 |
| `crc.ntrk_resistance_mutation` | 2 | eligibility, exceptions | metastatic_systemic | E26-SYS-3L-34 |
| `crc.off_label_treatment_rationale` | 2 | action, exceptions | genomic_treatment | ASCO-SGT-14 |
| `crc.organ_function_constraints` | 2 | exceptions | metastatic_systemic | ASCO22-MCRC-03, E26-SYS-1L-03 |
| `crc.pathology.resection_margin_status` | 2 | action, eligibility | colon_adjuvant, rectal_adjuvant | EC20-RISK-03, ER25-TX-25 |
| `crc.pathology.tumor_budding_grade.resection` | 2 | eligibility | colon_adjuvant | ASCO22-S2-1.2, ASCO22-S2-1.4 |
| `crc.pet_staging_result` | 2 | action, eligibility | metastatic_staging | E26-STAGE-05 |
| `crc.plasma_collection_date` | 2 | timing | molecular_testing | E26-MOL-008, E26-MOL-013 |
| `crc.postoperative_systemic_therapy_duration` | 2 | timing | oligometastatic_systemic | ASCO22-LOCAL-12, E26-LOCAL-10 |
| `crc.preoperative_systemic_therapy_duration` | 2 | timing | oligometastatic_systemic | ASCO22-LOCAL-12, E26-LOCAL-10 |
| `crc.preoperative_systemic_therapy_event` | 2 | eligibility, exceptions | oligometastatic_systemic | E26-LOCAL-08 |
| `crc.pretreatment_uracil_level` | 2 | action | molecular_testing | E26-MOL-019, EC20-DPD-01 |
| `crc.pretreatment_uracil_value` | 2 | eligibility | pharmacogenomics | EC20-DPD-U150-01, EC20-DPD-U16-01 |
| `crc.primary_tumor_resection_before_decision` | 2 | eligibility, exceptions | liver_transplant | E26-LOCAL-14 |
| `crc.primary_tumor_symptom_status_at_decision` | 2 | eligibility, exceptions | metastatic_primary | E26-LOCAL-02 |
| `crc.prior_anti_egfr_response` | 2 | eligibility | metastatic_systemic, molecular_testing | E26-MOL-013, E26-SYS-3L-28 |
| `crc.prior_braf_targeted_therapy_before_decision` | 2 | eligibility, exceptions | metastatic_systemic | ASCO22-MCRC-13 |
| `crc.prior_encorafenib_cetuximab_before_decision` | 2 | eligibility, exceptions | metastatic_systemic | E26-SYS-2L-23 |
| `crc.prior_trk_inhibitor_before_decision` | 2 | eligibility | metastatic_systemic | E26-SYS-3L-33, E26-SYS-3L-34 |
| `crc.radiation_feasibility_at_decision` | 2 | exceptions | liver_metastasis_local, nodal_metastasis_local | ASCO22-LOCAL-06, E26-LOCAL-21 |
| `crc.rectal_mri.date.pretreatment` | 2 | timing | rectal_assessment | ASCO24-LARC-1.2, ER25-ASSESS-05 |
| `crc.rectal_mri.tumor_deposit_status.pretreatment` | 2 | action, eligibility | rectal_assessment, rectal_tnt | ASCO24-LARC-1.3, ASCO24-LARC-2.1 |
| `crc.rectal_neoadjuvant.chemotherapy_completion_date` | 2 | action, timing | rectal_timing | ER25-RESTAGE-08 |
| `crc.rectal_nom.start_date` | 2 | timing | rectal_surveillance | ASTRO25-KQ3-07, ER25-NOM-04 |
| `crc.rectal_nom.surveillance_adherence_capacity` | 2 | exceptions | rectal_organ_preservation, shared_decision_making | ER25-NOM-03, ER25-TX-23 |
| `crc.rectal_radiation.short_course_start_date` | 2 | action, timing | rectal_timing | ER25-RESTAGE-06 |
| `crc.rectal_response.dre_performed.post_neoadjuvant` | 2 | action | rectal_assessment, rectal_restaging | ASTRO25-KQ3-06, ER25-ASSESS-02 |
| `crc.rectal_risk.local_recurrence_features.pretreatment` | 2 | eligibility | rectal_tnt | ASTRO25-KQ2-05, ASTRO25-KQ2-07 |
| `crc.rectal_surgery.resectability_at_salvage_decision` | 2 | exceptions | rectal_surgery | ER25-NOM-05, ER25-TX-24 |
| `crc.rectal_surveillance.completed_events` | 2 | action | rectal_surveillance | ASTRO25-KQ3-07, ER25-NOM-04 |
| `crc.rectal_surveillance.ct_event_dates` | 2 | timing | rectal_surveillance | ASTRO25-KQ3-07, ER25-NOM-04 |
| `crc.rectal_surveillance.dre_event_dates` | 2 | eligibility | rectal_surveillance | ASTRO25-KQ3-07, ER25-NOM-04 |
| `crc.rectal_surveillance.endoscopy_event_dates` | 2 | timing | rectal_surveillance | ASTRO25-KQ3-07, ER25-NOM-04 |
| `crc.rectal_surveillance.mri_event_dates` | 2 | timing | rectal_surveillance | ASTRO25-KQ3-07, ER25-NOM-04 |
| `crc.rectal_surveillance.year_from_nom_start` | 2 | timing | rectal_surveillance | ASTRO25-KQ3-07, ER25-NOM-04 |
| `crc.rectal_treatment.nonoperative_management_goal` | 2 | eligibility | rectal_radiation, rectal_tnt | ASTRO25-KQ1-06, ASTRO25-KQ2-01 |
| `crc.resectability_reassessment_event` | 2 | action, eligibility | metastatic_resectability, oligometastatic_local | E26-LOCAL-04, E26-STAGE-07 |
| `crc.ret_fusion_result` | 2 | eligibility, exceptions | metastatic_systemic | E26-SYS-3L-35 |
| `crc.retroperitoneal_nodal_metastasis_count_at_decision` | 2 | eligibility | nodal_metastasis_local | E26-LOCAL-20, E26-LOCAL-21 |
| `crc.retroperitoneal_nodal_metastasis_sites_at_decision` | 2 | eligibility | nodal_metastasis_local | E26-LOCAL-20, E26-LOCAL-21 |
| `crc.retroperitoneal_nodal_resectability_at_decision` | 2 | eligibility, exceptions | nodal_metastasis_local | E26-LOCAL-20 |
| `crc.staging_ct_protocol` | 2 | action, eligibility | metastatic_staging | E26-STAGE-02 |
| `crc.staging_laparoscopy_result` | 2 | action, eligibility | metastatic_staging | E26-STAGE-04 |
| `crc.targeted_combination_availability_at_decision` | 2 | eligibility | metastatic_systemic | E26-SYS-1L-07, E26-SYS-1L-08 |
| `crc.tmb_assay_panel` | 2 | eligibility | molecular_interpretation, molecular_testing | CAP22-MMR-002, CAP22-MMR-003 |
| `crc.tmb_high_cutoff` | 2 | action, eligibility | genomic_testing, molecular_testing | ASCO-SGT-08, CAP22-MMR-003 |
| `crc.transplant_organ_policy_eligibility_at_decision` | 2 | eligibility, exceptions | liver_transplant | E26-LOCAL-14 |
| `crc.treatment_response_goal_at_decision` | 2 | eligibility, exceptions | metastatic_systemic | E26-SYS-1L-16 |
| `crc.trial_identifier` | 2 | eligibility, exceptions | molecular_operations | CAP22-MMR-007 |
| `crc.tumor_type` | 2 | eligibility | genomic_treatment, molecular_interpretation | ASCO-SGT-14, SITC23-CRC-02 |
| `crc.actionable_molecular_subtype_at_decision` | 1 | eligibility | metastatic_systemic | E26-SYS-3L-25 |
| `crc.adverse_prognostic_factor_count_at_decision` | 1 | eligibility | oligometastatic_local | E26-LOCAL-07 |
| `crc.anti_egfr_contraindication_at_decision` | 1 | exceptions | metastatic_systemic | E26-SYS-1L-19 |
| `crc.anti_egfr_rash_risk_at_decision` | 1 | exceptions | shared_decision_making | ASCO22-MCRC-12 |
| `crc.approved_biomarker_set` | 1 | eligibility | genomic_testing | ASCO-SGT-03 |
| `crc.aspirin_treatment_context` | 1 | exceptions | historical_evidence | CAP17-BIO-06 |
| `crc.baseline_laboratory.cbc_completed` | 1 | action | rectal_assessment | ER25-ASSESS-02 |
| `crc.baseline_laboratory.liver_tests_completed` | 1 | action | rectal_assessment | ER25-ASSESS-02 |
| `crc.baseline_laboratory.renal_tests_completed` | 1 | action | rectal_assessment | ER25-ASSESS-02 |
| `crc.biomarker_linked_therapy_eligibility` | 1 | eligibility | genomic_testing | ASCO-SGT-02 |
| `crc.biomarker_prognostic_use` | 1 | action | molecular_testing | EC20-GEX-01 |
| `crc.biopsy_nonperformance_reason` | 1 | exceptions | molecular_operations | E26-MOL-002 |
| `crc.braf_summary_status` | 1 | eligibility | historical_evidence | CAP17-BIO-05 |
| `crc.braf_variant_class` | 1 | action | molecular_testing | E26-MOL-011 |
| `crc.candidate_biomarker_linked_therapy` | 1 | eligibility | genomic_testing | ASCO-SGT-01 |
| `crc.chemotherapy_contraindication_at_decision` | 1 | exceptions | metastatic_systemic | E26-SYS-1L-20 |
| `crc.clinician_treatment_choice_at_decision` | 1 | action | metastatic_systemic | ASCO22-MCRC-02 |
| `crc.colon_adjuvant.end_date` | 1 | timing | survivorship_adjacent | EC26-EXERCISE-01 |
| `crc.colon_adjuvant.risk_group` | 1 | eligibility | colon_adjuvant | EC20-ADJ-II-HIGH-DUR-01 |
| `crc.colon_adjuvant.start_delay_reason` | 1 | exceptions | colon_adjuvant | EC20-ADJ-TIME-01 |
| `crc.colon_recurrence_risk.at_adjuvant_decision` | 1 | eligibility | colon_adjuvant | EC20-RISK-01 |
| `crc.colon_stage_ii.high_risk_factor_count` | 1 | eligibility | colon_adjuvant | EC20-ADJ-II-HIGH-OX-01 |
| `crc.colon_stage_ii.minor_risk_factors` | 1 | eligibility | colon_adjuvant | EC20-ADJ-II-INT-01 |
| `crc.colon_surgery.primary_resection_date` | 1 | timing | colon_adjuvant | EC20-ADJ-TIME-01 |
| `crc.colonoscopy.extent_at_initial_workup` | 1 | action | rectal_assessment | ER25-ASSESS-03 |
| `crc.conventional_imaging_result_at_decision` | 1 | eligibility | metastatic_staging | E26-STAGE-05 |
| `crc.ctdna.status_at_adjuvant_decision` | 1 | eligibility | colon_adjuvant | ASCO22-S2-1.4 |
| `crc.ctdna_assay_result` | 1 | action | molecular_testing | E26-MOL-008 |
| `crc.ctdna_braf_status` | 1 | action | molecular_testing | E26-MOL-013 |
| `crc.ctdna_egfr_ectodomain_status` | 1 | action | molecular_testing | E26-MOL-013 |
| `crc.ctdna_kras_status` | 1 | action | molecular_testing | E26-MOL-013 |
| `crc.ctdna_nras_status` | 1 | action | molecular_testing | E26-MOL-013 |
| `crc.ctdna_specimen_date` | 1 | timing | metastatic_systemic | E26-SYS-3L-28 |
| `crc.ctdna_tumor_fraction` | 1 | action | molecular_testing | E26-MOL-008 |
| `crc.cytoreductive_surgery_candidacy_at_decision` | 1 | eligibility | metastatic_mdt | ASCO22-LOCAL-03 |
| `crc.decision.adjuvant_benefit_discussed` | 1 | action | colon_adjuvant | EC20-RISK-01 |
| `crc.decision.adjuvant_toxicity_discussed` | 1 | exceptions | colon_adjuvant | EC20-RISK-01 |
| `crc.decision.benefit_harm_discussion_documented` | 1 | action | colon_adjuvant | ASCO22-S2-1.3 |
| `crc.decision.counseling_documented` | 1 | action | shared_decision_making | ER25-NOM-03 |
| `crc.decision.oxaliplatin_benefit_discussed` | 1 | exceptions | colon_adjuvant | ASCO22-S2-3.1a |
| `crc.direct_mmr_msi_assay_status` | 1 | action | molecular_interpretation | CAP22-MMR-002 |
| `crc.direct_mmr_msi_test_status` | 1 | action | molecular_testing | CAP22-MMR-003 |
| `crc.disease_extent.at_localized_treatment_decision` | 1 | eligibility | rectal_assessment | ER25-MOL-02 |
| `crc.expanded_ras_coverage` | 1 | action | molecular_testing | CAP17-BIO-01 |
| `crc.extrahepatic_disease_at_decision` | 1 | eligibility | liver_metastasis_local | E26-LOCAL-15 |
| `crc.family_history_status` | 1 | exceptions | germline_testing | ASCO-SGT-06 |
| `crc.fluoropyrimidine_contraindication` | 1 | eligibility | pharmacogenomics | EC20-DPD-ALT-01 |
| `crc.fluoropyrimidine_dose_action` | 1 | exceptions | molecular_testing | E26-MOL-019 |
| `crc.fusion_assay` | 1 | action | genomic_testing | ASCO-SGT-11 |
| `crc.fusion_confirmation_status` | 1 | action | genomic_testing | ASCO-SGT-10 |
| `crc.gene_expression_signature_result` | 1 | action | molecular_testing | EC20-GEX-01 |
| `crc.genetic_counseling_status` | 1 | action | germline_testing | ASCO-SGT-06 |
| `crc.genetics_referral_date` | 1 | timing | lynch_reflex | E26-MOL-015 |
| `crc.genomic_driver_result` | 1 | eligibility | genomic_testing | ASCO-SGT-11 |
| `crc.genomic_laboratory` | 1 | action | molecular_operations | ASCO-SGT-04 |
| `crc.genomic_laboratory_certification_id` | 1 | action | molecular_operations | ASCO-SGT-04 |
| `crc.genomic_panel_coverage` | 1 | action | genomic_testing | ASCO-SGT-03 |
| `crc.germline_test_status` | 1 | action | germline_testing | ASCO-SGT-06 |
| `crc.germline_testing_indication` | 1 | eligibility | germline_testing | ASCO-SGT-06 |
| `crc.her2_copy_number` | 1 | action | molecular_testing | E26-MOL-016 |
| `crc.her2_ihc_score` | 1 | action | molecular_testing | E26-MOL-016 |
| `crc.her2_ish_ratio` | 1 | action | molecular_testing | E26-MOL-016 |
| `crc.her2_ngs_copy_number` | 1 | action | molecular_testing | E26-MOL-016 |
| `crc.histologic_confirmation_date` | 1 | timing | molecular_operations | E26-MOL-001 |
| `crc.ici_doublet_infeasibility_reason_at_decision` | 1 | eligibility | metastatic_systemic | E26-SYS-1L-02 |
| `crc.ild_pneumonitis_risk_at_decision` | 1 | exceptions | metastatic_systemic | E26-SYS-3L-31 |
| `crc.immunoscore` | 1 | action | molecular_testing | EC20-IMMUNOSCORE-01 |
| `crc.indexed_therapy` | 1 | action | historical_evidence | CAP17-BIO-05 |
| `crc.intensive_therapy_infeasibility_reason_at_decision` | 1 | eligibility | metastatic_systemic | E26-SYS-1L-18 |
| `crc.irinotecan_contraindication_at_decision` | 1 | exceptions | metastatic_systemic | E26-SYS-1L-14 |
| `crc.irinotecan_toxicity_at_decision` | 1 | exceptions | metastatic_systemic | E26-SYS-3L-27 |
| `crc.irinotecan_toxicity_risk_at_decision` | 1 | exceptions | metastatic_systemic | E26-SYS-1L-04 |
| `crc.iv_contrast_contraindication_at_decision` | 1 | exceptions | metastatic_staging | E26-STAGE-01 |
| `crc.liver_directed_radiation_candidacy_at_decision` | 1 | eligibility | metastatic_mdt | ASCO22-LOCAL-08 |
| `crc.liver_directed_therapy_at_decision` | 1 | action | liver_metastasis_local | E26-LOCAL-15 |
| `crc.liver_dominant_disease_at_decision` | 1 | eligibility | liver_metastasis_local | E26-LOCAL-15 |
| `crc.liver_metastasis_distribution_at_decision` | 1 | eligibility | liver_metastasis_local | ASCO22-LOCAL-07 |
| `crc.liver_mri_contrast_status` | 1 | eligibility | metastatic_staging | E26-STAGE-03 |
| `crc.local_therapy_organ_constraints_at_decision` | 1 | exceptions | liver_metastasis_local | E26-LOCAL-12 |
| `crc.local_treatment_candidacy_at_decision` | 1 | eligibility | metastatic_staging | E26-STAGE-03 |
| `crc.local_treatment_intent_at_decision` | 1 | eligibility | liver_metastasis_local | ASCO22-LOCAL-09 |
| `crc.localized_colon.resected_status_at_survivorship_decision` | 1 | eligibility | survivorship_adjacent | EC26-EXERCISE-01 |
| `crc.localized_rectal.metastatic_marker_orders` | 1 | action | rectal_assessment | ER25-MOL-02 |
| `crc.lung_metastasis_resectability_at_decision` | 1 | eligibility | lung_metastasis_local | E26-LOCAL-16 |
| `crc.lynch_risk_communication_date` | 1 | timing | lynch_reflex | CAP22-MMR-004 |
| `crc.lynch_risk_communication_recipient` | 1 | action | lynch_reflex | CAP22-MMR-004 |
| `crc.metastasis_surgery_event` | 1 | action | ovarian_metastasis_local | E26-LOCAL-19 |
| `crc.metastatic_recurrence_date` | 1 | timing | liver_metastasis_local | E26-LOCAL-13 |
| `crc.metastatic_status.at_initial_staging` | 1 | action | rectal_assessment | ER25-ASSESS-08 |
| `crc.mmr_ihc_antibody_clone` | 1 | eligibility | molecular_operations | CAP17-BIO-12 |
| `crc.mmr_ihc_platform` | 1 | eligibility | molecular_operations | CAP17-BIO-12 |
| `crc.mmr_ihc_validation_version` | 1 | action | molecular_operations | CAP17-BIO-12 |
| `crc.mmr_loss_area` | 1 | eligibility | molecular_operations | CAP22-MMR-007 |
| `crc.mmr_loss_extent` | 1 | eligibility | molecular_operations | CAP22-MMR-007 |
| `crc.mmr_lost_protein` | 1 | eligibility | molecular_operations | CAP22-MMR-007 |
| `crc.mmr_msi_ngs_equivalence_status` | 1 | exceptions | molecular_testing | CAP22-MMR-001 |
| `crc.mmr_msi_ngs_validation_comparator` | 1 | exceptions | molecular_testing | CAP22-MMR-001 |
| `crc.molecular.test_indication` | 1 | exceptions | rectal_assessment | ER25-MOL-02 |
| `crc.molecular_accession_date` | 1 | timing | molecular_operations | CAP17-BIO-17 |
| `crc.molecular_adequacy_report_field` | 1 | action | molecular_operations | CAP17-BIO-18 |
| `crc.molecular_alternate_method` | 1 | action | molecular_operations | CAP22-MMR-006 |
| `crc.molecular_assay_accuracy` | 1 | action | molecular_operations | CAP17-BIO-10 |
| `crc.molecular_assay_intended_use` | 1 | eligibility | molecular_operations | CAP17-BIO-10 |
| `crc.molecular_assay_precision` | 1 | action | molecular_operations | CAP17-BIO-10 |
| `crc.molecular_decision_date` | 1 | timing | molecular_testing | E26-MOL-012 |
| `crc.molecular_disease_context` | 1 | eligibility | molecular_interpretation | ASCO-SGT-05 |
| `crc.molecular_evidence_level` | 1 | eligibility | clinical_trial | ASCO-SGT-13 |
| `crc.molecular_extraction_date` | 1 | timing | molecular_operations | E26-MOL-005 |
| `crc.molecular_functional_evidence` | 1 | action | molecular_interpretation | ASCO-SGT-05 |
| `crc.molecular_indeterminate_reason` | 1 | eligibility | molecular_operations | CAP22-MMR-006 |
| `crc.molecular_local_tat_target` | 1 | action | molecular_operations | CAP17-BIO-13 |
| `crc.molecular_ordering_actor` | 1 | action | molecular_operations | CAP17-BIO-14 |
| `crc.molecular_ordering_role` | 1 | eligibility | molecular_operations | CAP17-BIO-16 |
| `crc.molecular_panel_design` | 1 | action | molecular_operations | CAP17-BIO-13 |
| `crc.molecular_peer_reviewers` | 1 | action | molecular_operations | CAP22-MMR-006 |
| `crc.molecular_procedure_date` | 1 | timing | molecular_operations | CAP17-BIO-20 |
| `crc.molecular_repeat_block_id` | 1 | action | molecular_operations | CAP22-MMR-006 |
| `crc.molecular_report_legacy_designation` | 1 | action | molecular_reporting | CAP17-BIO-21 |
| `crc.molecular_result_released_status` | 1 | action | molecular_operations | E26-MOL-007 |
| `crc.molecular_sectioning_plan` | 1 | action | molecular_operations | CAP17-BIO-15 |
| `crc.molecular_selected_specimen_id` | 1 | action | molecular_operations | E26-MOL-004 |
| `crc.molecular_sendout_required` | 1 | eligibility | molecular_operations | CAP17-BIO-17 |
| `crc.molecular_shipment_date` | 1 | timing | molecular_operations | CAP17-BIO-17 |
| `crc.molecular_specimen_fixation` | 1 | eligibility | molecular_operations | CAP17-BIO-09 |
| `crc.molecular_specimen_preference_decision` | 1 | action | molecular_operations | CAP17-BIO-08 |
| `crc.molecular_specimen_processing` | 1 | eligibility | molecular_operations | CAP17-BIO-09 |
| `crc.molecular_specimen_quality` | 1 | action | molecular_operations | CAP17-BIO-18 |
| `crc.molecular_specimen_quantity` | 1 | action | molecular_operations | CAP17-BIO-18 |
| `crc.molecular_specimen_receipt_date` | 1 | timing | molecular_operations | E26-MOL-007 |
| `crc.molecular_specimen_type` | 1 | eligibility | molecular_operations | CAP17-BIO-09 |
| `crc.molecular_test_order_status` | 1 | action | molecular_operations | CAP17-BIO-16 |
| `crc.molecular_testing_status` | 1 | action | molecular_testing | E26-MOL-006 |
| `crc.molecular_testing_trigger` | 1 | eligibility | molecular_operations | CAP17-BIO-14 |
| `crc.molecular_tissue_exhaustion_status` | 1 | action | molecular_operations | CAP17-BIO-15 |
| `crc.molecular_tissue_use_plan` | 1 | action | molecular_operations | CAP17-BIO-13 |
| `crc.molecular_validation_plan` | 1 | action | molecular_operations | CAP17-BIO-11 |
| `crc.molecular_validation_results` | 1 | action | molecular_operations | CAP17-BIO-11 |
| `crc.mri_contraindication_at_decision` | 1 | exceptions | metastatic_staging | E26-STAGE-03 |
| `crc.msi_assay_method` | 1 | action | molecular_testing | E26-MOL-014 |
| `crc.multidisciplinary_team.review_date` | 1 | timing | rectal_assessment | ER25-ASSESS-01 |
| `crc.multidisciplinary_team.review_documented` | 1 | action | rectal_tnt | ER25-TX-05 |
| `crc.multidisciplinary_team.specialties_present` | 1 | action | rectal_assessment | ER25-ASSESS-01 |
| `crc.neuropathy_at_decision` | 1 | exceptions | metastatic_systemic | E26-SYS-1L-04 |
| `crc.ntrk_gene` | 1 | eligibility | genomic_testing | ASCO-SGT-10 |
| `crc.off_label_treatment_decision` | 1 | action | genomic_treatment | ASCO-SGT-14 |
| `crc.oral_chemotherapy_feasibility_at_decision` | 1 | eligibility | metastatic_systemic | ASCO22-MCRC-02 |
| `crc.ovarian_metastasis_laterality` | 1 | eligibility | ovarian_metastasis_local | E26-LOCAL-19 |
| `crc.ovarian_metastasis_status_at_decision` | 1 | eligibility | ovarian_metastasis_local | E26-LOCAL-19 |
| `crc.pathology.histology.resection` | 1 | action | colon_adjuvant | EC20-RISK-03 |
| `crc.pathology.lymphoid_response.resection` | 1 | action | colon_adjuvant | EC20-RISK-03 |
| `crc.pathology_report_date` | 1 | timing | molecular_operations | E26-MOL-003 |
| `crc.patient.financial_toxicity_at_decision` | 1 | exceptions | survivorship_adjacent | EC26-EXERCISE-01 |
| `crc.patient.frailty_at_decision` | 1 | exceptions | rectal_neoadjuvant | ER25-TX-01 |
| `crc.patient.oxaliplatin_intolerance` | 1 | exceptions | colon_adjuvant | EC20-ADJ-III-04 |
| `crc.patient.treatment_fitness_at_decision` | 1 | exceptions | rectal_tnt | ER25-TX-08 |
| `crc.pdl1_assay` | 1 | action | molecular_interpretation | SITC23-CRC-02 |
| `crc.pdl1_cps` | 1 | action | molecular_interpretation | SITC23-CRC-02 |
| `crc.pdl1_tps` | 1 | action | molecular_interpretation | SITC23-CRC-02 |
| `crc.peritoneal_cancer_index_at_decision` | 1 | eligibility | peritoneal_metastasis_local | E26-LOCAL-17 |
| `crc.peritoneal_only_metastatic_disease_at_decision` | 1 | eligibility | peritoneal_metastasis_local | E26-LOCAL-17 |
| `crc.pik3ca_variant` | 1 | eligibility | historical_evidence | CAP17-BIO-06 |
| `crc.plasma_ctdna_assay` | 1 | action | molecular_testing | E26-MOL-008 |
| `crc.post_enrichment_tumor_percent` | 1 | action | molecular_operations | E26-MOL-005 |
| `crc.postoperative_systemic_regimen` | 1 | action | oligometastatic_systemic | E26-LOCAL-09 |
| `crc.pre_enrichment_tumor_percent` | 1 | eligibility | molecular_operations | E26-MOL-005 |
| `crc.preoperative_systemic_regimen` | 1 | eligibility | oligometastatic_systemic | E26-LOCAL-09 |
| `crc.primary_tumor_nodal_status` | 1 | eligibility | oligometastatic_local | E26-LOCAL-05 |
| `crc.prior_anti_egfr_agent` | 1 | eligibility | molecular_testing | E26-MOL-013 |
| `crc.prior_anti_her2_before_decision` | 1 | eligibility | metastatic_systemic | E26-SYS-3L-31 |
| `crc.prior_biologic_therapy_before_decision` | 1 | eligibility | metastatic_systemic | E26-SYS-3L-37 |
| `crc.prior_braf_inhibitor_exposure` | 1 | eligibility | molecular_testing | E26-MOL-012 |
| `crc.prior_oxaliplatin_date` | 1 | timing | metastatic_systemic | E26-SYS-1L-06 |
| `crc.prior_systemic_therapy_before_local_decision` | 1 | eligibility | liver_metastasis_local | ASCO22-LOCAL-06 |
| `crc.progression_on_irinotecan_before_decision` | 1 | eligibility | metastatic_systemic | E26-SYS-3L-27 |
| `crc.pten_assay_method` | 1 | eligibility | historical_evidence | CAP17-BIO-07 |
| `crc.pten_result` | 1 | eligibility | historical_evidence | CAP17-BIO-07 |
| `crc.pulmonary_reserve_at_decision` | 1 | eligibility | lung_metastasis_local | E26-LOCAL-16 |
| `crc.rapid_result_need_reason` | 1 | eligibility | molecular_testing | E26-MOL-008 |
| `crc.rectal_adjuvant.decision_date` | 1 | timing | rectal_adjuvant | ER25-TX-25 |
| `crc.rectal_adjuvant.start_date` | 1 | timing | rectal_adjuvant | ER25-TX-25 |
| `crc.rectal_adjuvant.therapy_plan_at_decision` | 1 | action | rectal_adjuvant | ER25-TX-25 |
| `crc.rectal_anatomy.rt_relevant_features.pretreatment` | 1 | exceptions | rectal_radiation | ASCO24-LARC-4.1 |
| `crc.rectal_brachytherapy.feasibility` | 1 | exceptions | rectal_organ_preservation | ER25-TX-04 |
| `crc.rectal_cancer.diagnosis_date` | 1 | timing | rectal_assessment | ER25-ASSESS-01 |
| `crc.rectal_diagnostic_biopsy.date` | 1 | timing | rectal_assessment | ER25-MOL-01 |
| `crc.rectal_erus.date.pretreatment` | 1 | timing | rectal_assessment | ER25-ASSESS-04 |
| `crc.rectal_erus.t_category_result.pretreatment` | 1 | action | rectal_assessment | ER25-ASSESS-04 |
| `crc.rectal_mri.enlarged_regional_node_count.pretreatment` | 1 | eligibility | rectal_selective_radiation | ASTRO25-KQ1-04 |
| `crc.rectal_mri.extramural_depth_mm.pretreatment` | 1 | action | rectal_assessment | ER25-ASSESS-06 |
| `crc.rectal_mri.intersphincteric_plane_status.pretreatment` | 1 | eligibility | rectal_tnt | ASCO24-LARC-2.1 |
| `crc.rectal_mri.largest_lateral_node_short_axis_mm.pretreatment` | 1 | action | rectal_assessment | ER25-ASSESS-07 |
| `crc.rectal_mri.protocol.pretreatment` | 1 | action | rectal_assessment | ASTRO25-KQ1-01 |
| `crc.rectal_mri.report_available.pretreatment` | 1 | action | rectal_assessment | ER25-ASSESS-05 |
| `crc.rectal_mri.sphincter_relation.pretreatment` | 1 | action | rectal_assessment | ASCO24-LARC-1.3 |
| `crc.rectal_mri.suspicious_regional_node_count.pretreatment` | 1 | action | rectal_assessment | ER25-ASSESS-07 |
| `crc.rectal_neoadjuvant.biologic_agent` | 1 | action | rectal_neoadjuvant | ER25-TX-15 |
| `crc.rectal_neoadjuvant.intolerance` | 1 | exceptions | rectal_neoadjuvant | ER25-TX-20 |
| `crc.rectal_neoadjuvant.irinotecan_included` | 1 | action | rectal_tnt | ER25-TX-08 |
| `crc.rectal_neoadjuvant.progression_date` | 1 | timing | rectal_neoadjuvant | ER25-TX-20 |
| `crc.rectal_nom.local_regrowth_date` | 1 | timing | rectal_surgery | ER25-NOM-05 |
| `crc.rectal_nom.local_regrowth_extent` | 1 | eligibility | rectal_surgery | ER25-NOM-05 |
| `crc.rectal_nom.shared_decision_making_documented` | 1 | action | shared_decision_making | ER25-NOM-03 |
| `crc.rectal_nom.surveillance_access` | 1 | exceptions | shared_decision_making | ER25-NOM-03 |
| `crc.rectal_nom.surveillance_plan` | 1 | action | rectal_organ_preservation | ER25-TX-23 |
| `crc.rectal_post_immunotherapy.recurrence_date` | 1 | timing | rectal_surgery | ER25-TX-24 |
| `crc.rectal_postoperative.clinical_risk_assessment` | 1 | eligibility | rectal_adjuvant | ER25-TX-25 |
| `crc.rectal_radiation.concurrent_systemic_therapy` | 1 | action | rectal_radiation | ASTRO25-KQ2-02 |
| `crc.rectal_radiation.indication` | 1 | eligibility | rectal_radiation | ASTRO25-KQ1-08 |
| `crc.rectal_radiation.omission_eligibility` | 1 | eligibility | rectal_radiation | ASTRO25-KQ1-03 |
| `crc.rectal_radiation.timing_relative_to_surgery` | 1 | timing | rectal_radiation | ASTRO25-KQ1-08 |
| `crc.rectal_response.assessment_completed.week12_dostarlimab` | 1 | action | rectal_timing | ER25-RESTAGE-12 |
| `crc.rectal_response.assessment_completed.week24_dostarlimab` | 1 | action | rectal_timing | ER25-RESTAGE-12 |
| `crc.rectal_response.biopsy_performed.post_neoadjuvant` | 1 | action | rectal_restaging | ER25-RESTAGE-03 |
| `crc.rectal_response.ccr_confirmation_date` | 1 | timing | rectal_organ_preservation | ER25-TX-23 |
| `crc.rectal_response.ct_performed.post_neoadjuvant` | 1 | action | rectal_restaging | ASTRO25-KQ3-06 |
| `crc.rectal_response.endoscopy_performed.post_neoadjuvant` | 1 | action | rectal_restaging | ASTRO25-KQ3-06 |
| `crc.rectal_response.imaging_category.post_neoadjuvant` | 1 | eligibility | rectal_selective_radiation | ASTRO25-KQ1-04 |
| `crc.rectal_response.mri_performed.post_neoadjuvant` | 1 | action | rectal_restaging | ASTRO25-KQ3-06 |
| `crc.rectal_response.suspicious_finding.post_neoadjuvant` | 1 | eligibility | rectal_restaging | ER25-RESTAGE-03 |
| `crc.rectal_restaging.modality_used` | 1 | action | rectal_restaging | ER25-RESTAGE-02 |
| `crc.rectal_risk.distant_recurrence_risk.pretreatment` | 1 | eligibility | rectal_tnt | ASCO24-LARC-3.1 |
| `crc.rectal_risk.high_risk_status.pretreatment` | 1 | action | rectal_assessment | ER25-ASSESS-07 |
| `crc.rectal_risk.lower_risk_status.pretreatment` | 1 | eligibility | rectal_tnt | ASTRO25-KQ2-04 |
| `crc.rectal_stage.pathologic_stage_group.resection` | 1 | eligibility | rectal_adjuvant | ER25-TX-25 |
| `crc.rectal_surgery.local_excision_plan` | 1 | action | rectal_radiation | ASTRO25-KQ3-04 |
| `crc.rectal_surgery.sphincter_preserving_feasibility` | 1 | exceptions | rectal_neoadjuvant | ER25-TX-14 |
| `crc.rectal_tnt.feasibility` | 1 | exceptions | rectal_neoadjuvant | ER25-TX-10 |
| `crc.rectal_tnt.infeasibility_reason` | 1 | exceptions | rectal_neoadjuvant | ER25-TX-10 |
| `crc.rectal_treatment.planned_neoadjuvant_pathway` | 1 | action | rectal_neoadjuvant | ER25-TX-19 |
| `crc.rectal_treatment.prior_neoadjuvant_therapy` | 1 | eligibility | rectal_adjuvant | ER25-TX-25 |
| `crc.rectal_treatment.response_goal` | 1 | exceptions | rectal_neoadjuvant | ASCO24-LARC-2.2 |
| `crc.rectal_treatment.sphincter_preservation_goal` | 1 | exceptions | rectal_organ_preservation | ASTRO25-KQ1-07 |
| `crc.regimen_contraindication_at_decision` | 1 | exceptions | metastatic_systemic | ASCO22-MCRC-01 |
| `crc.resectability_assessment_date` | 1 | timing | metastatic_resectability | E26-STAGE-07 |
| `crc.resectability_imaging_date` | 1 | timing | metastatic_resectability | E26-STAGE-07 |
| `crc.rna_quality` | 1 | action | genomic_testing | ASCO-SGT-11 |
| `crc.sirt_event` | 1 | action | liver_metastasis_local | ASCO22-LOCAL-07 |
| `crc.somatic_variant_result` | 1 | eligibility | metastatic_systemic | ASCO22-MCRC-09 |
| `crc.specialty_center_referral_event` | 1 | action | peritoneal_metastasis_local | ASCO22-LOCAL-02 |
| `crc.staging.ct_abdomen_findings.pretreatment` | 1 | action | rectal_assessment | ER25-ASSESS-08 |
| `crc.staging.ct_chest_abdomen_date.pretreatment` | 1 | timing | rectal_assessment | ER25-ASSESS-08 |
| `crc.staging.ct_chest_findings.pretreatment` | 1 | action | rectal_assessment | ER25-ASSESS-08 |
| `crc.staging_ct_contrast_status` | 1 | eligibility | metastatic_staging | E26-STAGE-01 |
| `crc.staging_ct_date` | 1 | timing | metastatic_staging | E26-STAGE-01 |
| `crc.staging_ct_quality` | 1 | eligibility | metastatic_staging | E26-STAGE-02 |
| `crc.staging_ct_result` | 1 | action | metastatic_staging | E26-STAGE-01 |
| `crc.staging_laparoscopy_feasibility_at_decision` | 1 | exceptions | metastatic_staging | E26-STAGE-04 |
| `crc.surgical_candidacy_at_decision` | 1 | eligibility | metastatic_staging | E26-STAGE-04 |
| `crc.survivorship.exercise_counseling_date` | 1 | timing | survivorship_adjacent | EC26-EXERCISE-01 |
| `crc.survivorship.exercise_program_access` | 1 | exceptions | survivorship_adjacent | EC26-EXERCISE-01 |
| `crc.survivorship.physical_ability_for_exercise` | 1 | exceptions | survivorship_adjacent | EC26-EXERCISE-01 |
| `crc.survivorship.structured_exercise_counseling` | 1 | action | survivorship_adjacent | EC26-EXERCISE-01 |
| `crc.survivorship.structured_exercise_plan` | 1 | action | survivorship_adjacent | EC26-EXERCISE-01 |
| `crc.systemic_therapy_event` | 1 | action | peritoneal_metastasis_local | ASCO22-LOCAL-01 |
| `crc.targeted_dissection_status` | 1 | action | molecular_operations | CAP22-MMR-007 |
| `crc.tissue_available_for_testing` | 1 | exceptions | genomic_testing | ASCO-SGT-02 |
| `crc.tissue_biopsy_feasibility` | 1 | eligibility | molecular_testing | E26-MOL-008 |
| `crc.tmb_germline_filtering_method` | 1 | action | genomic_testing | ASCO-SGT-08 |
| `crc.tmb_panel_territory` | 1 | action | genomic_testing | ASCO-SGT-08 |
| `crc.tmb_variants_counted` | 1 | action | genomic_testing | ASCO-SGT-08 |
| `crc.treatment_response_at_resectability_assessment` | 1 | eligibility | metastatic_resectability | E26-STAGE-07 |
| `crc.trial_referral_status` | 1 | action | clinical_trial | ASCO-SGT-13 |
| `crc.tumor_biology.at_adjuvant_decision` | 1 | action | colon_adjuvant | EC20-RISK-04 |
| `crc.tumor_enrichment_method` | 1 | action | molecular_operations | E26-MOL-005 |
| `crc.tumor_enrichment_status` | 1 | action | molecular_operations | CAP17-BIO-19 |
| `crc.tumor_marker_status_at_decision` | 1 | eligibility | metastatic_staging | E26-STAGE-05 |
| `crc.tumor_ngs_date` | 1 | timing | genomic_testing | SITC23-CRC-01 |
| `crc.unused_eligible_subtype_therapy_at_decision` | 1 | eligibility | metastatic_systemic | E26-SYS-3L-25 |

## Missing registry fields — outside current sources (23)

These variables cannot currently be established from the registered registry/chart source set and need an explicit new source or owner.

| variable | rule-block uses | roles | guideline categories | example rules |
|---|---:|---|---|---|
| `crc.regulatory_approval_date` | 10 | eligibility, exceptions, timing | genomic_testing, molecular_operations, molecular_testing | ASCO-SGT-01, ASCO-SGT-03, ASCO-SGT-12, CAP17-BIO-11, E26-MOL-018 |
| `crc.unresolved.metastatic_biology` | 4 | eligibility, exceptions | liver_metastasis_local, nodal_metastasis_local, oligometastatic_local | E26-LOCAL-03, E26-LOCAL-13, E26-LOCAL-20 |
| `crc.genomic_testing_access_constraint` | 2 | exceptions | clinical_trial, genomic_testing | ASCO-SGT-02, ASCO-SGT-13 |
| `crc.laboratory_proficiency_program` | 2 | action, exceptions | molecular_operations | CAP17-BIO-22 |
| `crc.molecular_turnaround_clock_definition` | 2 | exceptions, timing | molecular_operations | E26-MOL-007 |
| `crc.service_cohort_metric` | 2 | action | molecular_operations | CAP17-BIO-17, CAP17-BIO-20 |
| `crc.unresolved.adverse_biomarker_set` | 2 | eligibility, exceptions | oligometastatic_local | E26-LOCAL-07 |
| `crc.working_day_calendar_version` | 2 | timing | molecular_operations | CAP17-BIO-17, CAP17-BIO-20 |
| `crc.disease_specific_genomic_options` | 1 | eligibility | genomic_testing | ASCO-SGT-12 |
| `crc.genomic_accreditation_jurisdiction` | 1 | action | molecular_operations | ASCO-SGT-04 |
| `crc.genomic_testing_cost_constraint` | 1 | exceptions | genomic_testing | ASCO-SGT-02 |
| `crc.laboratory_alternative_quality_assurance` | 1 | action | molecular_operations | CAP17-BIO-22 |
| `crc.laboratory_corrective_action` | 1 | action | molecular_operations | CAP17-BIO-22 |
| `crc.laboratory_proficiency_cycle_results` | 1 | action | molecular_operations | CAP17-BIO-22 |
| `crc.laboratory_qi_metrics` | 1 | action | molecular_operations | CAP17-BIO-22 |
| `crc.molecular_knowledgebase_version` | 1 | timing | molecular_interpretation | ASCO-SGT-05 |
| `crc.service_cohort_denominator` | 1 | action | molecular_operations | E26-MOL-003 |
| `crc.service_cohort_numerator` | 1 | action | molecular_operations | E26-MOL-003 |
| `crc.standard_treatment_options` | 1 | action | clinical_trial | ASCO-SGT-13 |
| `crc.tumor_agnostic_biomarker_set` | 1 | action | genomic_testing | ASCO-SGT-12 |
| `crc.unresolved.guideline_evidence_context` | 1 | eligibility | metastatic_systemic | ASCO22-MCRC-04 |
| `crc.unresolved.oligometastatic_clinical_score` | 1 | eligibility | oligometastatic_local | E26-LOCAL-05 |
| `crc.unresolved.oligometastatic_prognostic_factor_set` | 1 | eligibility | oligometastatic_local | E26-LOCAL-04 |

## Derived variables not stored directly (7)

These variables need a reviewed derivation and input provenance. They do not necessarily require a new raw data field.

| variable | rule-block uses | roles | guideline categories | example rules |
|---|---:|---|---|---|
| `crc.primary_tumor_sidedness` | 17 | eligibility | metastatic_systemic | ASCO22-MCRC-07, ASCO22-MCRC-08, ASCO22-MCRC-11, E26-SYS-1L-08, E26-SYS-1L-09, E26-SYS-1L-10, +11 more |
| `crc.historical_rule_activation_status` | 3 | action | historical_evidence | CAP17-BIO-05, CAP17-BIO-06, CAP17-BIO-07 |
| `crc.ici_biomarker_eligibility` | 2 | action | molecular_interpretation, molecular_testing | E26-MOL-014, SITC23-CRC-03 |
| `crc.braf_targeted_therapy_eligibility` | 1 | action | molecular_testing | E26-MOL-012 |
| `crc.her2_targeted_therapy_eligibility` | 1 | action | molecular_testing | E26-MOL-016 |
| `crc.lynch_suspicion_status` | 1 | action | lynch_reflex | EC20-LYNCH-MSH-01 |
| `crc.somatic_mlh1_pathway_support` | 1 | action | lynch_reflex | EC20-LYNCH-SPORADIC-01 |

## Registry fields with scope mismatch (9)

These variables have a candidate registry projection, but information is lost. They cannot be treated as exact rule evidence.

| variable | rule-block uses | principal loss |
|---|---:|---|
| `crc.metastatic_status` | 97 | Decision-time disease status is collapsed into diagnosis/recurrence-oriented registry items. |
| `crc.systemic_regimen` | 59 | Exact agents and named regimen are reduced to a broad chemotherapy summary class. |
| `crc.systemic_therapy_start_date` | 54 | A generic indexed systemic-therapy start is reduced to a first-course registry date. |
| `crc.mmr_ihc_integrated_status` | 43 | Individual MLH1, PMS2, MSH2, and MSH6 results are lost. |
| `crc.msi_status` | 39 | Raw MSI assay status is not distinguishable from an MMR-IHC-derived summary. |
| `crc.patient_preference` | 27 | Preference among clinically acceptable options is reduced to coarse refusal/nonreceipt coding. |
| `crc.treatment_contraindication` | 6 | Modality-specific clinical contraindications are reduced to coarse non-treatment reasons. |
| `crc.ici_contraindication` | 5 | ICI-specific reason is collapsed into a broad biological-response-modifier code. |
| `crc.treatment_sequence` | 3 | Exact event order and dates are reduced to broad modality-sequence categories. |

## Registry-direct candidate mappings (4)

These are the only candidate exact canonical-to-registry mappings. They remain subject to registrar, clinical, temporal-scope, and data-profile review.

| variable | rule-block uses | roles | guideline categories | example rules |
|---|---:|---|---|---|
| `crc.primary_site` | 95 | eligibility | liver_metastasis_local, liver_transplant, lung_metastasis_local, metastatic_mdt, metastatic_primary, metastatic_resectability, metastatic_staging, metastatic_systemic, molecular_testing, nodal_metastasis_local, oligometastatic_local, oligometastatic_systemic, ovarian_metastasis_local, peritoneal_metastasis_local, shared_decision_making | ASCO22-LOCAL-01, ASCO22-LOCAL-02, ASCO22-LOCAL-03, ASCO22-LOCAL-04, ASCO22-LOCAL-05, ASCO22-LOCAL-06, +89 more |
| `crc.braf_v600e_status` | 23 | action, eligibility, exceptions | liver_transplant, lynch_reflex, metastatic_systemic, oligometastatic_local | ASCO22-MCRC-07, ASCO22-MCRC-11, CAP17-BIO-03, CAP22-MMR-004, E26-LOCAL-05, E26-LOCAL-06, +16 more |
| `crc.treatment_start_date` | 2 | timing | molecular_operations, molecular_testing | E26-MOL-001, E26-MOL-019 |
| `crc.histology` | 1 | action | molecular_operations | E26-MOL-001 |
