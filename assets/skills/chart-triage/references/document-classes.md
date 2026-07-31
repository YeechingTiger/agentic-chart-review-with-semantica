# Document classes: what the type names in this corpus actually are

Third-level detail: load when a type name is unfamiliar, when a filter is behaving oddly, or
when you are about to conclude a class of evidence is absent. Each entry is the
classification plus the reason, because the reason is what transfers to the next site.

All counts measured 2026-07-26 against the corpus type vocabulary, **1,516 distinct types**.

## The shape of the vocabulary

| family | types | note |
|---|---|---|
| imaging studies (CT, MRI, PET, US, XR, fluoro, mammo, echo, angio) | **674 (44%)** | the largest family by far, and mostly noise for most questions |
| pathology / cytology **reports**, by name | 28 | the decisive class for tissue facts, and 2% of the vocabulary |
| needle / biopsy / aspirate names | 49 | **split** between reports and image-guided procedure notes — see below |
| progress and clinic notes | ~126 | restatement class: mentions results, rarely establishes them |
| consults and H&Ps | ~115 | good for history and outside references, weak as primary evidence |

Two consequences. First, a random or breadth-first read order spends its budget on imaging.
Second, the class you usually need is 28 names out of 1,516 and none of them is discoverable
by guessing.

## Pathology and cytology reports, by name

```
Anatomic-Pathology-Report            Pathology-Report
Cytology-Bronchial-Brushings         Pathology-Report-Miscellaneous
Cytology-Bronchoalveolar-Lavage      Surgical-Pathology-Document
Cytology-Misc                        Surgical-Pathology-Report
Cytology-Other                       SURG-PATH-RESULT
Cytology-Report                      SURGICAL-PATH
Dermatopathology-Report              Tissue-Pathology-Report
Dermpath-Spec-A-Report               Medical-Pathology-Report
Gyn-Cytology-Report                  Non-Gynecologic-Cytology-Study
Hematopathology-Report               Microscopic-Observation-ID-Cyto-Stain
IMMUNOHISTOLOGY-RPT                  Pathologist-Review-Smear-Bld
Immunoperoxidase                     PAP-Smear-Interp
MOLECULAR-PATHOLOGY                  Pap-Smear-Liq-Based-Prep-w-HPV
Path-Rpt-Addendum                    Pap-Smear-Liquid-Based-Prep
```

Only **11** of these 28 contain the substring `patholog`, and a twelfth type that does —
`Speech-Language-Pathology-Note` — is not one of them. That single ratio is why a substring
filter cannot stand in for the class.

## Tissue acquisition names: report or procedure?

The `Needle` / `Bx` / `Aspirate` family (49 types) contains both, and the distinction is the
suffix, not the organ:

| pattern | what it is | examples |
|---|---|---|
| `-Report`, `-Rpt`, bare organ + `Biopsy` | the **pathologist's** findings | `Fine-Needle-Report`, `FN-Aspirate-Report`, `Core-Needle-Biopsy`, `Ventricular-Biopsy`, `Bone-Marrow-Aspirate-Bx-Clot` |
| `-Guide`, `-Guid`, `-Placement`, `-Localiz`, `-Wire`, `-Perc` | the **radiologist's** account of obtaining the sample | `Lung-Bx-W-CT-Guid`, `Liver-Bx-W-US-Guide`, `Needle-Placement-CT-Guide`, `Breast-Needle-Wire-Placement-US-Guide`, `Spine-Needle-Cath-Localiz-Fluoro` |

**Thirty-two of the 49** carry a guidance or technique suffix — that is, most of the family
is radiology, not pathology. They are not worthless: they establish that a sample was taken,
from where, and on what date, which settles temporal and specimen-site questions even though
they carry no diagnosis. Do not cite them for the diagnosis, and do not cite the site they
name as the site of origin.

Direct evidence that this matters: on the five real charts reviewed on 2026-07-26, the only
tissue diagnosis for patient `P01` sat in a single `Fine-Needle-Report`, and
for patient `P04` in two `FN-Aspirate-Report`s. Neither chart contains any type
whose name mentions pathology or cytology.

## Look-alikes to exclude

| type | what it actually is |
|---|---|
| `Speech-Language-Pathology-Note`, `Aud-Speech-Path-Initial-Eval`, `Aud-Speech-Path-Procedure` | speech and language therapy |
| `PAP-Previous-Biopsy` | a screening-history data field |
| `Histo-capsulatum-Ag-Qn-EIA` | a histoplasma antigen assay |
| `Blood-Pathogens-NAA-+-non-probe-Panel` | microbiology |
| `Breast-Specimen-Mammo` | a radiograph of the excised specimen, taken in theatre |
| `Pelvis-Surgical-Lat-Hip-XR` | an intra-operative film |

Every one of these matches at least one substring a reasonable person would use for
pathology.

## Where the resolution documents live

`Addendum`, `Path-Rpt-Addendum` and `Discharge-Summary-Staff-Addendum` are their own types.
`ED-Corrections-Report-Document` is the amendment channel for ED notes. When a report defers
its own conclusion, these are the types to list — see `assets/skills/thread-chasing/SKILL.md`.

`Tumor-Board-Recommendation-Note` is a single type and is the closest thing in the corpus to
a synthesised, multidisciplinary statement of the case. It is a restatement and cannot be
primary evidence for a tissue fact, but it is an excellent index into which documents matter
and what the treating team believed.

## A note on absence

A class being absent from a chart is a real and reportable finding, and it is not
distributed at random: charts missing a pathology class are disproportionately transfers,
outside work-ups and declined biopsies. Report **which class** is absent rather than that
the answer was not found — the two support completely different downstream conclusions.
