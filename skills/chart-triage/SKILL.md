---
name: chart-triage
description: Use at the start of any chart review and whenever you must decide which documents to open - orienting in a chart of hundreds of documents, choosing what to read first, deciding whether a document type is the kind that can answer your question, a type-name filter returning nothing or returning obviously wrong documents, or judging whether a chart genuinely lacks a class of evidence. Covers read order (type before date, date before content), judging a document by what it is rather than by what its name contains, and which document classes are decisive for which kind of question.
slot: general
license: MIT
---

# Triaging a chart before reading it

Read order: **type before date, date before content.** One `document_type_summary` call
gives you the whole shape of the record — every type, its count, its date span — for the
cost of one step. Deciding what to read from the type inventory is cheap; discovering it by
reading is not. Applies to any variable.

## Judge a document by what it IS, not by what its name contains

This corpus has **1,516 distinct document types**. Type names are site-local strings, not a
vocabulary, and a substring filter over them is a lexical accident.

Measured on 2026-07-26. The substring pair `["Pathology", "Cytology"]` matches **18** of the
1,516 types. One of those 18 is `Speech-Language-Pathology-Note`, which is not a pathology
report. A further **32** types report on tissue or cells and contain neither word — among
them `Core-Needle-Biopsy`, `Fine-Needle-Report`, `FN-Aspirate-Report`, `FNA-Report`,
`IMMUNOHISTOLOGY-RPT`, `SURG-PATH-RESULT`, `SURGICAL-PATH`, `Immunoperoxidase`,
`Specimen-Site`, `Path-Rpt-Addendum`, `Bronchial-Wash-Cyto`,
`Microscopic-Observation-ID-Cyto-Stain`.

What that cost, on the five real charts reviewed that day: **two of the five had zero
documents matching `Pathology|Cytology`, and both had a tissue diagnosis** — patient
`P01` in a single `Fine-Needle-Report`, patient `P04` in two
`FN-Aspirate-Report`s. A filter-shaped review of those charts would have abstained on a
question the chart answers.

Ask four questions of a type name, in this order:

1. **Who wrote it?** A pathologist, a radiologist, a surgeon, a nurse, a billing system.
2. **What did they look at?** Tissue or cells under a microscope; an image; the patient; a
   prior document.
3. **Is it a report of a finding, or a record of an act?** `Lung-Bx-W-CT-Guid` and
   `Needle-Placement-US-Guide` are the radiologist describing how the sample was obtained.
   The diagnosis is in a different document.
4. **Is it primary or a restatement?** A progress note repeating a result is not the result.

## Name traps, both directions

- `-Guide`, `-Guid`, `-Placement`, `-Localiz`, `-Wire`, `-Stereotactic` — 32 of the 49
  biopsy-named types carry one, and every one is a procedure or imaging note rather than a
  diagnostic report. They tell you a sample was taken and when, which is genuinely useful
  for temporal and specimen-site questions.
- `Speech-Language-Pathology-Note`, `Aud-Speech-Path-Initial-Eval` — a different profession.
- `PAP-Previous-Biopsy` — a screening-history field, not a report.
- `Histo-capsulatum-Ag-Qn-EIA`, `Blood-Pathogens-NAA-+-non-probe-Panel` — microbiology.
- Conversely, ALL-CAPS and abbreviated names (`SURG-PATH-RESULT`, `IMMUNOHISTOLOGY-RPT`)
  are usually feeds from an older system and are frequently the decisive documents.

Fuller listing: `skills/chart-triage/references/document-classes.md`.

## Which classes are decisive for which KIND of question

The variable changes; these five kinds do not.

| kind of fact | decisive classes | not admissible / not sufficient |
|---|---|---|
| about tissue or cells (histology, grade, margins, receptors, molecular) | pathology, cytology, needle/aspirate reports, immunostain reports and their addenda | imaging, problem lists, a progress note repeating a result |
| about anatomic location or extent (site, laterality, size, spread) | pathology and operative notes **and** cross-sectional imaging — imaging localises even where it cannot diagnose | a specimen header, which names where the needle went, not where the tumour arose |
| about when an event happened | the document generated **by** the event — operative note, report collection date, administration record | a later note recalling the date, unless nothing else exists |
| about what was given or done | orders, administration and infusion records, operative and procedure notes | a plan. A documented intention is not an event |
| about status or summary judgements (stage assigned, class of case, vital status) | `Discharge-Summary`, `Tumor-Board-Recommendation-Note`, registry-facing summaries | inference from the clinical course; some of these are genuinely outside the notes |

When a spec's field is a location and its search hints are all diagnostic terms, the second
row is the one being skipped — see `skills/keyword-strategy/SKILL.md`.

## Absence is a finding, but only after you enumerate

`list_documents(doc_type_contains=...)` returns `type_exists_but_empty: true` when your
substring names a real type somewhere in the corpus and *this* patient has none of it. That
distinguishes a typo from a real absence, which is worth a great deal — but it says nothing
about whether the patient has the *class* you meant, because your substring was never the
class. On the synthetic chart `SYN0002`, a filter of `Path` returned
`type_exists_but_empty: true`, and that chart's 293 documents really do span 17 types none
of which is pathology. The filter was right by luck; the reasoning that produced it was not.

Before concluding a class is absent, enumerate the class from the type inventory by asking
the four questions above of every type name in it — not by trying more substrings.

## Order of work

1. `document_type_summary` — one call, the whole shape, before anything else.
2. Partition the type list into: can answer this question, might mention it, cannot bear on
   it. Do it by what each type is, not by what it is called.
3. Within the deciding types, order by date around the index event, not from either end of
   the record. The decisive document usually sits near the event; the oldest documents are
   usually unrelated history.
4. Read the deciding types in full before searching for anything. `list_documents` returns
   60 per page by default and documents run past 20,000 characters — check `more` and
   `truncated` rather than assuming you have seen it all.
5. Only then search, to find mentions in the types you classified as "might mention".
