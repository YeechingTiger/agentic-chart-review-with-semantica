---
name: store-staging
description: How to record AJCC 8th edition clinical and pathologic stage and SEER Summary Stage for a lung primary from a chart. Use when assigning cT/cN/cM, pT/pN/pM, a stage group or a summary stage - especially when the only stage statement sits in a resection synoptic report, when the record hedges with "at least T2a" or "pending", when imaging and pathology disagree, when neoadjuvant therapy came first, or when no stage is documented and 99, TX or NX look like the easy answer.
slot: task
license: MIT
---

# Recording AJCC stage and Summary Stage for a lung primary

Applies to lung primaries diagnosed 2018-01-01 onward, staged under AJCC 8th edition. Not
for post-neoadjuvant (yc/yp) staging, small cell recorded only as limited/extensive,
haematopoietic neoplasms, or cases staged under AJCC 7th — for those, answer
SPEC_INSUFFICIENT.

## The four mistakes this criterion invites

Unlike `store-icdo-coding`, **none of these has been observed on a chart in this project** —
this criterion has not been run yet. Each is the structural twin of a failure that *was*
measured on 2026-07-26 for site and histology, by a competent model that had the rules in
its prompt. The twinning is the claim being made here. When a run produces a real error,
replace the constructed example below with the real quote and name the chart.

| measured, site/histology | its twin here |
|---|---|
| coded the biopsy site `C340` as the site of origin | reads the clinical stage off the resection |
| coded `8046` over "favor squamous" | codes `cT2` over a documented `cT2a` |
| coded `C349` having run zero searches | codes `99` / `cTX` / `cNX` having run none |
| *(no twin — specific to staging)* | treats "at least cT2a" as a determination |

### 1. Reading the clinical stage off the resection

> Constructed, not observed: a synoptic report reads `"AJCC pathologic stage: pT2a pN1
> cM0 — Stage IIB"` and the run fills `clinical_stage_group: IIB`. A specimen cannot
> report what was known before treatment began.

Before filling any clinical field, ask: *is this quote telling me what was known BEFORE the
first treatment, or what the specimen showed after it?*

- Clinical stage is everything acquired pre-treatment — exam, imaging, endoscopy, biopsy,
  surgical staging. It is fixed at that point and the resection does not revise it.
- Pathologic stage needs the primary resected, and `pN` needs the nodes actually removed. A
  needle or endoscopic biopsy never produces a pathologic stage group.
- The `cM0` inside a pathologic synoptic is the clinical M borrowed to complete the p-group.
  **There is no `pM0`**, and the spec's format check refuses it — that rejection is telling
  you a value was copied across the c/p boundary.
- Neoadjuvant therapy makes both sets `yc`/`yp`, a third pair of items. `SPEC_INSUFFICIENT`.

### 2. Taking the undivided category when the subcategory is documented

> Constructed: `"3.2 cm mass with visceral pleural invasion"` recorded as `cT2`. AJCC 8th
> has no undivided T2 to assign; size and pleural invasion each give `cT2a`.

The same rule that makes "favor squamous" into `8070`: where the record is more specific
than the line you were about to copy, take the more specific reading. Staging subdivides
further than histology does, so there is more to lose.

- `T1mi`/`T1a`/`T1b`/`T1c` split at 1, 2 and 3 cm; `T2a`/`T2b` at 4 cm; `T3` at 5, `T4` at
  7 cm.
- Those digits are not cosmetic. At N0 M0 they are stages `IA1`, `IA2`, `IA3`, `IB`, `IIA`.
- A stage group must be consistent with its own T, N and M. Where a stated group disagrees
  with the stated categories, prefer the categories and record the contradiction with
  `stance=contradicts` — an unrecorded contradiction looks identical to an unnoticed one.

### 3. Coding 99, TX or NX without having looked

> Constructed: `clinical_stage_group: 99` with zero searches run — the exact shape of the
> `C349` failure, where "right upper lobe" was sitting in seven other note types.

`99`, `cTX` and `cNX` are **positive claims**. `cNX` asserts the regional nodes could not be
assessed; it is not the code for "I did not read the staging CT". Before any of them, run
the six searches the runtime gates on: `stage`, `tnm`, `tumor size`, `pleural`, `lymph
node`, `metasta`. A rejection here means you have not looked, not that you are wrong.

Where each field lives is not where you would guess. The clinical stage group is usually in
an oncology or tumour-board assessment rather than in pathology; size and pleural invasion
are in the radiology report; `M1` is in imaging.

### 4. Treating a bound as a determination

> Constructed: `"clinically at least T2a, N status pending mediastinoscopy"` recorded as
> `cT2a cN0` → stage `IB`.

A lower bound constrains the answer; it does not determine it. Neither does "pending".
Record what *is* determined, abstain on the rest, and do not round. This is the one point
where this skill and `store-icdo-coding` pull in opposite directions, so keep the three
apart:

| phrasing | what it is | what to do |
|---|---|---|
| "favor squamous", "consistent with" | committing with reservation | code the specific value |
| "at least T2a", "T2 or greater" | refusing an upper bound | do not assign the group |
| "pending", "see addendum" | an unfinished thread | chase it before settling |

## Order of work

1. Fix which primary you are staging. A second cancer carries its own stage and the record
   rarely labels whose is whose.
2. Fix the date of first treatment. That one date decides which evidence is clinical.
3. Read every pathology report and every oncology, thoracic-surgery and tumour-board
   assessment in full — the runtime reads that stratum exhaustively, so budget for it.
4. Run the six gated searches, then read the hits in imaging and progress notes.
5. Fill the clinical fields and the pathologic fields separately. Leave a field null rather
   than borrow one across the boundary.
6. Before submitting, run the four checks above against what you actually cited.

## Summary Stage is not a translation of the AJCC group

Summary Stage 2018 is assembled from documented extent of disease across the whole record:
`0` in situ, `1` localised, `2` regional by direct extension, `3` regional nodes only, `4`
both, `7` distant, `9` unknown. It survives a missing AJCC group — a PET showing an adrenal
deposit gives `7` even when no clinician ever wrote a stage.

## When the chart cannot support an answer

Answer `EVIDENCE_INSUFFICIENT`, and say which categories are determinable and which are not.
Two abstentions here are findings rather than gaps, and should be stated that way: a primary
that was never resected **has** no pathologic stage, and a bounded statement leaves a real
lower bound on the record. Unknown stage is not missing at random — it clusters on patients
who never reached an oncology consultation and on outside-facility workups.

Further worked cases: `skills/store-staging/references/staging-boundary-cases.md`.
