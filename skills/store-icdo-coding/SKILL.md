---
name: store-icdo-coding
description: How to assign ICD-O-3 primary site, histology and behaviour from a chart under CoC STORE rules. Use when coding a solid tumour's topography or morphology from pathology, cytology, operative or imaging reports - especially when reports disagree, a diagnosis is hedged ("favor X"), the specimen came from a biopsy or metastasis, or no tissue diagnosis exists.
slot: task
license: MIT
---

# Coding ICD-O-3 site, histology and behaviour

Applies to solid tumours under CoC STORE 2025. Not for lymphoma, leukaemia or other
haematopoietic neoplasms — those follow separate rules; answer SPEC_INSUFFICIENT.

## The three mistakes that actually happen

These are not hypothetical. Each was made on a real chart in this project on 2026-07-26,
by a competent model that had the rules in its prompt. Knowing the rule is not the same as
applying it under pressure, so check yourself against these three before submitting.

### 1. Coding the specimen site instead of the site of origin

A pathology report names the site the **tissue came from**. That is frequently not where
the tumour **arose**.

> Report read: `"Bronchus, distal right main mass, biopsy"` → coded `C340` (main bronchus).
> The same document said `"5 x 6 cm right hilar mass … obstructing RUL"`.
> Correct answer was `C341` (upper lobe).

Before coding a site, ask: *is this quote telling me where the tumour started, or where the
needle went?* A specimen header (`Specimen Received:`, `…, biopsy:`, `FNA`) answers the
second question only. Find a statement of origin — operative findings, imaging describing
the dominant mass, or the clinical impression — and cite that too.

If the specimen is from a **metastatic** site, code the histology of the metastasis and
behaviour 3, but the topography is still the site of **origin**.

### 2. Taking the less specific reading when a more specific one is present

> Report read: `"Poorly differentiated non-small cell carcinoma, favor squamous cell
> carcinoma, special stains pending"`, and elsewhere `"best classified as a non-small cell
> carcinoma, lung primary, NOS"`. Coded `8046` (NSCLC, NOS). Correct was `8070` (squamous).

The STORE conflict rule: where the record is more specific than the summary line, **take the
more specific reading**. The canonical example is microscopic "adenocarcinoma" over a final
line of "carcinoma NOS" → 8140, not 8010.

Two riders:
- `"special stains pending"` is an unfinished thread. Look for the addendum or a later
  report before settling for the interim NOS line. If it genuinely is not in the chart, say
  so in your reasoning.
- `8000` (cancer, NOS) and `8010` (carcinoma, NOS) are **not** interchangeable. If the
  physician wrote carcinoma, code 8010.

### 3. Coding an NOS subsite without looking for the specific one

> Pathology said only `"Right lung"` → coded `C349` (lung, NOS). But `"right upper lobe"`
> appeared across seven other document types. Correct was `C341`.

**A NOS code is a positive claim that the specific value is not documented.** It carries the
same burden as any other claim of absence. Before coding `C349`, search for `lobe`,
`bronchus`, and laterality; before any `…9` subsite, search for the subsites.

Radiology is admissible for this. The rule "imaging cannot establish histology or
behaviour" is about **morphology**, not topography — radiology localises a mass perfectly
well, and the spec says so explicitly.

## Order of work

1. Read every pathology and cytology report in full. Judge by what a document **is**, not
   what its name contains: `Fine-Needle-Report`, `Core-Needle-Biopsy`, `FN-Aspirate-Report`
   and `IMMUNOHISTOLOGY-RPT` are pathology reports even though none of those names contains
   the word "pathology".
2. Take histology and behaviour **only** from a tissue or cytology diagnosis.
3. Take the site of origin from wherever it is documented — pathology, operative note, or
   imaging.
4. Before submitting, run the three checks above against what you actually cited.

## Behaviour

- Any malignant invasion, however focal → `3`.
- `2` (in situ) requires the process be confined to the epithelium with **no** stromal
  invasion identified.
- Metastatic specimen → `3`.
- Do not upgrade behaviour on anatomical spread alone. An atypical meningioma invading skull
  bone is `1`; meninges can invade bone without being malignant.

## When the chart cannot support an answer

If there is no pathology in the record, do **not** infer histology from imaging or from a
clinical assertion, and do not treat "biopsy was done at an outside facility" as evidence of
what that biopsy showed. Answer `EVIDENCE_INSUFFICIENT` for histology and behaviour, state
plainly that the diagnosis is radiographic or clinical without tissue confirmation — and
still report `primary_site` if the site of origin is documented.

Further worked cases: `skills/store-icdo-coding/references/boundary-cases.md`. The bare
relative form resolves against the process cwd and 404s; this path is the one that opens.
