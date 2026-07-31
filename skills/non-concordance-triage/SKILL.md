---
name: non-concordance-triage
description: How to explain why a patient was NON_CONCORDANT with a guideline recommendation, by sorting the case into one of four causes with four different owners - care gap (clinician), documentation gap (health information management), extraction error (us), justified exception (nobody, this is correct care). Use when triaging concordance failures, when deciding whether recommended care is missing from the chart or missing from the world, when a registry value disagrees with an extracted one, or when a documented refusal, poor performance status, comorbidity or trial enrolment might mean the recommendation never applied.
slot: general
license: MIT
---

# Triaging a non-concordant case into its actual cause

Applies to one patient against one recommendation already scored `NON_CONCORDANT` by the
rule engine. `CONCORDANT` and `NOT_ASSESSABLE` cases are out of scope — `NOT_ASSESSABLE`
especially, because a recommendation whose inputs were `EVIDENCE_INSUFFICIENT` was never
scored at all, and explaining it folds it into a rate it does not belong in.

You are handed a case packet from `scaffold_explanation`. Read `causes` first: standings of
`SUPPORTED`, `OPEN` and `ELIMINATED` were computed from the coverage ledger before you were
called. **An `ELIMINATED` standing is not yours to overturn** — it was decided from counters
you cannot address and did not produce, and `assert_cause_is_earned` will reject a final
answer that names one. Your work is the `OPEN` causes and the `questions` list.

## The four mistakes that actually happen

L5 has not yet been run against a real cohort here, so unlike `store-icdo-coding` these are
not all replays of this project's own traces. Mistakes 1 and 4 are what the design predicts
and what the concordance literature does routinely; mistakes 2 and 3 arrive from L2 failures
this project **did** measure on 2026-07-26. Treat the list as a checklist, not as folklore.

### 1. Guessing between a care gap and a documentation gap

> Chart holds no record of adjuvant chemotherapy. Run ended `BUDGET_EXHAUSTED`, no coverage
> ledger. Tempting reading: "the patient did not get chemotherapy" → **A, care gap**.
> Correct answer: `CANNOT_DISTINGUISH`.

"Not documented" and "not done" produce the identical observation, and nothing you can read
in the chart separates them — the separating information is not in the chart. Only a
stratified coverage proof does it, and only partly: it establishes that the absence is real
rather than a retrieval failure. When `verdict` is `CANNOT_DISTINGUISH`, say so, and name
what would settle it.

The mirror error is over-reading a proof you *do* have. `B: SUPPORTED` means we proved the
care is not documented **in this chart**. Care delivered at another facility looks exactly
like that. So a proven B does not promote to A, and A stays `OPEN` — write the documentation
finding, and name A as unresolved rather than dismissing it.

### 2. Filing a justified exception as a care gap

> A patient who declined chemotherapy, counted as a quality failure. This is the single most
> common way published concordance rates are wrong.

Before writing A, work the exception catalogue explicitly: performance status, comorbidity
or contraindication, patient refusal, clinical trial enrolment, hospice or goals-of-care
change, death before the recommended interval elapsed. Each verdict needs its own cited
evidence, **to the same standard as the primary variable** — a witness-proved finding with a
quote, not a passing mention in a social-work note.

Two riders, and they pull in opposite directions:
- A refusal documented for a *different* decision does not license this one. "Declined
  surgery" is not "declined chemotherapy". Check that the exception's date and subject match
  the recommendation's window and modality; that scope judgement is why the code leaves A
  open when D is supported instead of eliminating it for you.
- Conversely, do not demand the guideline's own exact wording. ECOG 3 recorded in an
  oncology note is a performance-status exception whether or not anyone wrote "not a
  candidate".

### 3. Ruling out extraction error because there is no registry row

> Registry coverage is 1,788 of 8,894 patients. On the other 80% there is no truth value,
> and "no disagreement found" is then a statement about the registry, not about our
> extraction.

C stays live wherever truth is absent, and the packet names exactly which variables lack it.
Do the thing the registry cannot: re-read the cited spans in `variables[].evidence` and ask
whether the coded value actually follows from them.

When truth *does* exist and disagrees, C is `SUPPORTED` — but disagreement is not proof we
are wrong. Measured here on 2026-07-26: the pathologist wrote "best classified as non-small
cell carcinoma, NOS" after IHC and the registrar coded squamous from a hedged "favor". The
chart can be right. Route the case to us and adjudicate the chart, not the registry.

### 4. Reporting one number

A single non-concordance rate fuses four findings with four owners. B is filed against
records management, C against this codebase, D is not a defect at all — and a fused rate
tells a clinician to change behaviour that may already have been correct. Report the causes
separately, including the `CANNOT_DISTINGUISH` bucket, and never let that bucket be folded
into A "for readability".

## Order of work

1. Read `causes` and `forbidden`. Note what is already `ELIMINATED` and stop reasoning about
   it. Note whether `verdict` is `CANNOT_DISTINGUISH`; if it is, A and B are closed to you.
2. Work D before A. An unexamined exception catalogue is the most likely single error, and a
   supported D changes who owns the case entirely.
3. Work C on the evidence, not the registry: does each coded value follow from its quotes?
4. Write the answer as a cause plus its owner plus the evidence, and list every cause left
   open. An open cause you did not mention reads downstream as one you ruled out.

## What would settle A versus B, when neither is available

- external administration or claims data for the recommended treatment
- a health-information-exchange feed or an outside-records request for the gap period
- the interior-gap window analysis, where the recommendation has a follow-up interval:
  records either side and nothing between is an evidence gap, never an absence of events
- direct contact with the patient or the treating practice

Naming which of these is needed is a useful output. "Unknown" without it is not.

## When the case cannot support a single cause

Answer `CANNOT_DISTINGUISH` and still report everything else you established: which causes
are eliminated, which remain open, what evidence you cited, and what specific artefact — a
rerun with a stratified spec, a registry row, an outside-records request — would resolve it.
A named open question is a finding. A confident wrong owner is a wasted chart review.

The deterministic rules, and the exact reason behind every standing you were handed, are in
`src/acr/explain.py`. Nothing in this layer has ground truth in this project; do not
describe any output of it as validated.
