---
name: coverage-judgement
description: Use when your answer is about to claim that something is NOT in the chart - the case answer is EVIDENCE_INSUFFICIENT, a field is being left empty, or your reasoning is about to say a value is not documented. Tells you which questions to settle before claiming absence, why a not-otherwise-specified code is a positive claim rather than an absence, how to judge which local document types could establish the answer at all, and what to write down about the looking. Not needed for a positive answer resting on a witness you have read.
slot: general
license: MIT
---

# Deciding whether you have looked at enough of the chart

This is your judgement to make. Nothing in the runtime will refuse your answer for looking too
little, and nothing will accept it because you looked a lot. What the runtime does is count what
you did and write it down beside your answer, so that a reviewer can later tell the difference
between *you decided the chart was adequately searched* and *nobody looked*.

## When this question even arises

Only when your answer makes a claim of **absence**. Three shapes count:

- the case answer is `EVIDENCE_INSUFFICIENT`;
- a field is left empty;
- you are about to say, in reasoning, that something is not documented.

A positive answer resting on a witness you have read does not need this. If a pathology report
states the diagnosis and you have read it, you are done — go and answer. Reading further to feel
thorough costs money and, measured on this project's own runs, changes the answer for the worse
more often than for the better.

A not-otherwise-specified code is **not** an absence claim. `8046` (non-small cell carcinoma,
NOS), `8010`, `8000` and `C349` are real registry answers — together about 10% and 9.6% of this
corpus respectively — and a conflict-resolution rule legitimately lands on them. If you coded NOS
because that is what the rules give you, you have made a positive claim and you owe a witness for
it, not a search proof.

## What to ask yourself before claiming absence

1. **Which kinds of document could settle this at all?** For histology and behaviour, that is a
   report where a pathologist or cytopathologist states a diagnosis from tissue or cells. Note
   that this corpus names those documents many ways — `Surgical-Pathology-Document`,
   `Cytology-Report`, but also `Non-Gyn-Cyto-FNA`, `FN-Aspirate-Report`, `SURG-PATH-RESULT`,
   `Microscopic-Observation-ID-Cyto-Stain`. Read the type list you were given and judge each name
   on what the document *is*, not on whether it contains the word "pathology". A name containing
   "pathology" may be a speech-language therapy note.
2. **Have you actually looked in them?** Not "did a search return nothing" — did you open the
   documents that could carry the answer.
3. **Did you finish the ones you opened?** A read that stopped short is the one thing the runtime
   still refuses on, because it can compute it: the diagnosis is often in the last paragraph, and
   an addendum that changes it is often after that. Page to the end.
4. **Did you try more than one wording?** No single term covers this corpus. Measured over its
   12,190 diagnosis-bearing documents: `carcinoma` appears in 57.5%, `cancer` in 32.7%,
   `malignan` in 50.1%, `tumor` in 36.8% — and 23.9% contain none of those. 677 documents have
   `cancer` and not `carcinoma`. If one search came back thin, the term is the first suspect.
   Search is cheap; issue several.
5. **What would change your mind?** If you can name a document that would overturn your answer,
   go and look for it. If you cannot, say so.

## What the runtime will tell you

You may see advisory lines about strata, samples and residual bounds — how many documents of a
kind exist, how many you reviewed, what chance of a missed document your sampling leaves. Those
are computed facts and they are worth reading. They are not conditions. If an advisory says you
reviewed 4 of 12 documents that could establish the answer, decide whether the other 8 matter
here and act on that decision.

## Say what you decided

When you claim absence, state in your reasoning:

- which kinds of document you judged capable of settling it, and why the rest could not;
- what you searched and what you read;
- what remains unexamined, and why you judged it not worth examining.

That paragraph is the deliverable. An absence claim with no account of the looking is the thing
this project has been trying to prevent, and it turns out a refusal was never the way to get it —
five previous attempts at enforcing coverage in code are recorded in
`docs/COVERAGE_THREE_ARM_PILOT.md`, and all of them cost more correct answers than they saved.

## Why this is a skill and not a gate

Measured on ten real charts with registry ground truth:

| arm | exact match | what coverage cost |
|---|---|---|
| `guideline-only` (no coverage) | 3/10, 10/10 gate-valid | — |
| `conditional-negative-coverage` | 3/10, 5/10 gate-valid | activated on 7 cases, completed 1; recovered 2 field values, lost 13 |
| `always-coverage` | 2/10, 5/10 gate-valid | recovered 0, lost 15 |

Across every recorded trace, coverage obligations produced about 150 answer rejections, 27 of
which refused a tuple that was exactly the registry's answer. One run submitted the correct answer
ten times and was rejected into a call-limit failure.

The counting was never the problem. The refusing was.
