# What the deterministic rules cost, and what replaced them

## Summary

Every deterministic rule that judged the *content* of a chart-review answer was measured against
every trace this project has recorded — 266 traces, 219 manifests, 202 traces joinable to
registry gold — and then removed. What is left in the gate judges *provenance* only.

The headline number: **60 of 254 recorded answer rejections (24%) refused a tuple that was
exactly the registry's answer.** Twelve runs held the exact registry answer in hand and shipped
something else; eight of those twelve shipped nothing at all.

This document is the measurement record. The two pilots it supersedes in their forward-looking
sections are [SEARCH_PLANNING_PILOT.md](SEARCH_PLANNING_PILOT.md) and
[COVERAGE_THREE_ARM_PILOT.md](COVERAGE_THREE_ARM_PILOT.md); their measured results stand
unchanged and are not restated here.

## Method

Deterministic, no model calls, no cost. Every `rule_rejection` and `answer_rejected` event in
every recorded trace carries the rule id, the rule kind, the field and the value that was
refused. `ground_truth.csv` covers all 1,788 patients, so every firing joins to a gold tuple on
`patient_id` without an adjudication step.

Two denominators are used and they are not interchangeable:

- **field-level** — did this rule refuse the registry's value *for the field it judges*. The
  right denominator for judging a rule, since a rule only ever looks at one field.
- **submission-level** — was the whole three-field tuple that got refused exactly the registry's.
  The right denominator for the cost of the gate as a whole.

## The answer checks

Field-level, 122 recorded firings:

| rule | fires | refused the registry's own value | ever helped |
|---|---:|---:|---:|
| `not_less_specific` | 22 | **22 (100%)** | **0** |
| `nos_requires_search` | 24 | **21 (88%)** | **0** |
| `conflict_requires_nos` | 67 | 18 (27%) | 15 |
| `field_format` | 7 | 0 | 6 |
| `origin_not_specimen` | 2 | 0 | 0 |
| `code_matches_cited_text` | **0** | — | — |

"Ever helped" means the firing refused a wrong value and the run then reached the registry's
answer. Net across all five clinical checks: **58 firings destroyed a correct value, 21 preceded
a correct one, 39 did nothing.**

Three of them never helped once and need no further argument. The other three are worth stating
precisely, because two of them looked defensible right up to the measurement.

### `conflict_requires_nos` was a biased coin, not a check

15 helps against 18 harms reads like breaking even. It is not: **all 15 helps were the same
event, a push to `C349`**, because "or code `C349`" is the only remedy its message offered.

```
rejected C341   -> final C349  (gold C349)      rejected C343   -> final C349  (gold C349)
rejected C348   -> final C349  (gold C349)      rejected C34.2  -> final C349  (gold C349)
   ... 15 of these, and not one help that produced a specific subsite
```

`C349` is the registry's answer for 9.6% of this corpus; `C341` alone is 52.7%. A rule whose only
move pays off at the base rate of the value it moves toward is a coin, and 53% of its firings
(34 of 67) refused a wrong value and ended on another wrong value — pure friction.

### `origin_not_specimen` was a round trip

Fired twice. Both times the model resubmitted the identical value.

### `code_matches_cited_text` never fired

Zero firings in 266 traces, while sitting on the reject path with a lobe→code table. Untested
code guarding production.

### `field_format` had the only positive record, and was still removed

7 firings, 0 refusals of a registry value, 6 that preceded the registry answer. It caught `C3412`
and `C3432` — codes that are not codes.

It was removed anyway, and the reason is in what it caught: **4 of those 6 useful firings
rejected `C34.9`, `C34.11` and `C34.2` — the punctuated form ICD-O-3 itself writes.** It was
largely creating the round trips it then resolved. And the constraint is already in the prompt:
`as_prompt_block` renders every field's `format` and `allowable_values`, and STORE.400's own field
description reads "no decimal point". A model that writes `C34.9` against that has failed to
follow an instruction. That is a fact to measure, not a hole to plug with a regex.

The measurement survives as an `answer_shape_miss` trace event carrying `refused: False`, so the
eval plane counts instruction-following failures instead of a gate absorbing them silently.

Two known gaps are pinned as tests rather than left in a docstring:
`tests/test_answer_checks.py::test_the_punctuated_icdo3_form_is_still_refused_and_that_is_a_known_defect`
(the fix is deterministic notation normalisation, `C34.1` → `C341`) and
`…::test_a_well_formed_but_invented_morphology_still_passes_and_that_is_the_open_gap`
(`\d{4}` passes `9999`; closing it needs a real ICD-O-3 code table, and a shape regex is not one).

### One contradiction no single rule could show

On CASE009 of the planning ablation the runtime deleted `lobe` and `bronchus` from the retrieval
plan for budget (`fit_terms_to_budget`), and `nos_requires_search` then refused the answer because
the run "never searched for `['lobe', 'bronchus']`". **One rule punished the model for not running
a term another rule had taken away.** That run submitted the registry-correct `C341` five times,
was refused five times, and shipped `C349`.

## The thread markers

39 thread refusals; **11 (28%) refused a tuple that was exactly the registry's.** By marker:

```
truncated 111   addendum 40   in consultation 10   additional sections 9   pending 8
outside facility 5   clinical correlation 4   outside hospital 3   synoptic report 3
see synoptic 3   amended 1                                            (197 detections)
```

All but `truncated` are substrings scanned across document text, parsed out of a Markdown table in
`skills/thread-chasing/SKILL.md`. `addendum` refused 40 times while `read_section("ADDENDUM")`
could address that heading in **0 of the 2,401 documents containing the word** — an obligation
whose tool could never reach its target.

`truncated` still blocks, and the difference is categorical rather than one of degree: it is
*computed* from the character counts of the run's own read against the length that read reported.
It is a fact about what the run did, it cannot be wrong about the corpus, and it discharges
automatically once the document is fully read. The other twenty are now advisory — still detected,
still opened as threads, still rendered into the prompt with the settling call and thread id
filled in, still in the manifest. What went away is the refusal.

## The document stratifier

`can_establish` selected its documents with `doc_type_matches: ["Pathology", "Cytology"]`,
case-insensitively, as a substring. Over the corpus's 1,516 distinct type names and 276,054
documents it **misses**:

| type name | documents |
|---|---:|
| `Non-Gyn-Cyto-FNA` | 1,285 |
| `FN-Aspirate-Report` | 881 |
| `SURG-PATH-RESULT` | 231 |
| `Microscopic-Observation-ID-Cyto-Stain` | 31 |
| `Fine-Needle-Aspiration` | 5 |

and **matches** `Speech-Language-Pathology-Note`.

**107 of the 219 patients whose `can_establish` count is zero in fact hold one of the missed
reports** — 6.0% of the cohort, told by the architecture that no document in the chart could
establish histology while an FNA diagnosis sat in it. Only 112 (6.3%) genuinely hold none.

Note that `ground_truth.csv`'s own `can_establish` column was computed by the same matcher, so the
spec's claim that the stratifier "was validated against ground_truth.csv at 1787 of 1787 exact
matches" is circular: it validated the matcher against its own output.

The planning ablation contains both halves of the proof. **CASE001** — no type name matched, so
the cytology FNA in the chart was filed under a stratum named `cannot_establish`, and the run
returned `EVIDENCE_INSUFFICIENT` for histology and behaviour; the registry coded 8070. **CASE006**
— the same chart state, but the run happened to open its FN-Aspirate-Report anyway, cited it as a
witness, and returned the registry answer exactly.

The stratum's `establishes` list also decided, per field, whether a cited document was *admissible*
(`admissibility_for_citations`), so the substring error propagated into the evidence record.

## The read permission

`AuditMiddleware._out_of_plan` **refused a read**. If the plan had filed a document's type in the
`sample` bucket the agent could not open it: the call returned `error: OUT_OF_PLAN` and the model
was told to request a promotion first. It fired **138 times**, and the bucket it enforced came from
the substring matcher above. On CASE001 it stood between the agent and the only document in the
chart that carried the answer.

`plan_refused_open` recorded all 138 with an empty `refusal_class`, so none of them is attributable.

## Section addressing

`read_section` addressed headings with `^[A-Z][A-Z0-9 /&'\-]{2,60}:\s*$` — ALL-CAPS, nothing after
the colon. Over the 12,221 diagnosis-bearing documents:

| heading | documents containing | addressable | reachable |
|---|---:|---:|---:|
| `final diagnosis` | 7,390 | 170 | **2.3%** |
| `final pathologic diagnosis` | 6,262 | 332 | 5.3% |
| `microscopic description` | 5,914 | 479 | 8.1% |
| `diagnosis` | 12,082 | 42 | 0.3% |
| `addendum` | 2,401 | **0** | **0.0%** |

The reason is in the real spellings — Title Case dominates and the regex admits none of it:

```
no    3899  'Pre-Operative Diagnosis:'        YES   308  'FINAL PATHOLOGIC DIAGNOSIS:'
no    2807  'Final Diagnosis:'                YES   104  'FINAL DIAGNOSIS:'
no    2154  'Final Cytologic Diagnosis:'      no     28  '***DIAGNOSIS***'
no    1489  'Final Pathologic Diagnosis:'
```

Worse than the low recall: called with no section argument the tool returned
`available_sections`, so in 97.7% of documents the model asked what was in the file and got back
a list with the final diagnosis missing from it. That is a tool answering wrongly, not failing to
help.

Removing it also eliminated a state. `read_section` reported true offsets and no document length,
and that was the only source of `READ_STATE_LENGTH_UNKNOWN` — a run that had only read sections
could not tell it had left a hole. Every read now reports `total_chars`, so `truncated` is always
computable. The state is retained as the truthful answer for a malformed read result.

The synthetic dev corpus is part of why this survived so long: it writes ALL-CAPS headings
(`CLINICAL HISTORY:`, `FINDINGS:`) that the regex matched perfectly.

## What replaced retrieval

### Search is notation-tolerant, never synonym-aware

A run of whitespace or hyphen in the query now matches any run of whitespace or hyphen in the
text. Documents found, over 12,190 diagnosis-bearing documents:

| phrase | literal | folded | gained |
|---|---:|---:|---:|
| `non-small cell` | 2,251 | 2,476 | **+225 (+10.0%)** |
| `small cell carcinoma` | 1,537 | 1,552 | +15 |
| `right upper lobe` | 1,270 | 1,277 | +7 |
| `right lower lobe` | 876 | 883 | +7 |
| `squamous cell carcinoma` | 2,615 | 2,623 | +8 |
| ten phrases, total | 16,797 | 17,064 | **+267 (+1.6%)** |

The hyphen case is real; the line-wrap case is small — 1.6% overall, against a prior expectation
that hard wraps were a major loss. Both are free and neither needs a word list.

Synonyms stay out, and the measurement is why. Share of diagnosis-bearing documents each single
term appears in:

```
carcinoma 57.5%   malignan 50.1%   tumor 36.8%   cancer 32.7%
adenocarcinoma 27.6%   neoplasm 4.3%   tumour 0.0%
```

677 documents (5.6%) contain `cancer` and not `carcinoma`, and **23.9% contain none of those
seven**. No fixed list is close to complete, so a matcher that expanded a term into a synonym set
would be guessing for the model and hiding its own miss rate inside a hit. Recall comes from the
model issuing several searches and reading what comes back — which is what deleting the term
budget made possible.

The snippet window widened from ±160 to ±250 characters, so a hit list can be decided from
without opening the documents.

### Document concepts are reference

`src/acr/document_concepts.py` holds seven portable concepts — `definitive_pathology`,
`specimen_acquisition`, `operative_localization`, `cross_sectional_imaging`,
`specialist_assessment`, `molecular_or_ancillary`, `administrative` — each with prose saying what
the document is and which fields it can establish on its own. No local type string appears in it,
no ordering, no measurement, no keyword list. Its prompt header says so.

Priority is deliberately absent because it depends on the field: for histology an imaging report
is inert, and for primary site the same report may be the only thing that names the lobe. This
project got that wrong in the other direction once, coding lung-NOS while `right upper lobe` sat
in seven imaging and oncology note types.

A measured prior is a separate tier. `experience_block()` returns nothing until a certified asset
is passed in, and there is none: the SEARCH_PLANNING_PILOT arm injected the five spec-derived
keywords `pathology, biopsy, final diagnosis, specimen, carcinoma`, which had already been
measured at 87.4% recall over 276,054 documents — missing an answer-bearing document for 31.7% of
patients because the list has `carcinoma` and not `cancer`. That arm scored 3/10 against native
planning's 4/10. **The negative result says nothing about priors in general; it says an
uncertified list does not help.**

## Coverage

`evaluate_gate` is advisory by default. The arithmetic is unchanged — strata counts, reviewed
counts, Clopper-Pearson residual bounds — and routes to `advisories` instead of `missing`, so it
reaches the model as information about its own run. `gate_answer` and `gate.check` both forward it,
because asking the model to judge coverage without showing it what the runtime counted is not a
design.

`terminal` stays empty in advisory mode: "asking again will not change this" is a statement about
a refusal, and it also keeps `terminal ⊆ missing` true in both modes.

Coverage obligations produced roughly 150 answer rejections across the recorded traces, **27 of
which refused a tuple that was exactly the registry's**. The obligation now lives in
`skills/coverage-judgement/SKILL.md`. `enforce=True` is retained for the diagnostic arm, which
COVERAGE_THREE_ARM_PILOT.md already said was not a candidate default.

## What the gate still enforces

Provenance, not content:

- at least one recorded quote before a `FOUND`;
- every cited quote re-read from disk at its `(note_id, start, end)` — the model's own quote text
  is discarded, so a fabricated quote cannot enter the ledger;
- `truncated`, computed from character counts;
- patient scope, spend limit, model-call limit;
- undeclared tools — a statement about the tool surface, since a read that bypasses
  `Toolbox.dispatch` is invisible to the ledger.

None of these is a clinical judgement.

## Baselines the accuracy numbers should be read against

From `ground_truth.csv`, 1,788 patients:

| field | majority value | share |
|---|---|---:|
| `behavior` | `3` | **99.7%** |
| `primary_site` | `C341` | 52.7% |
| `histology` | `8140` | 37.7% |
| three-field exact, constant `C341/8140/3` | | **19.9%** |

So the planning ablation's 4/10 exact match sits against a ~2/10 constant-predictor baseline, and
`behavior` carries almost no information while still being a multiplier on exact match. In that
ablation `behavior` never failed independently: it was empty exactly when `histology` was empty, in
all three abstaining cases, so reporting it as a separate field metric counts one failure mode
three times.

`8000`/`8010`/`8046` together are the registry's answer for 10.8% of the corpus and `C349` for
9.6% — which is why treating a NOS-shaped value as a claim of missing evidence
(`negative_claim_reasons`) was wrong, and why a rule whose remedy was "or code the NOS value" was
a coin.

## Where the losses actually were

Per-case attribution of the planning ablation's native arm, which is the cohort the removed rules
were tuned on. Eleven field misses, in three buckets — **and not one of them was a document the
search failed to find**:

| bucket | field misses | cause |
|---|---:|---|
| abstention | 6 | 3 cases × (histology + behavior); 2 of the 3 caused by the stratifier |
| histology subtype precision | 2 | `8140` for gold `8230`; `8973` for gold `8972` |
| primary-site NOS calibration | 1 | coded `C341`, gold `C349` |

The remaining loss in the preplanned arm was CASE009: gold `C341`, refused five times, shipped
`C349`.

Search planning was ablated against a cohort containing **zero retrieval failures**, which is why
nine of ten output tuples were identical. The null result was a property of the design, not a
finding about priors.

## Reproducing this

Every number above is deterministic string matching against `ground_truth.csv` and the recorded
traces. No model was in the loop, so unlike an evolved artifact none of it expires when the model
changes. The scripts were scratch; the joins are `patient_id` against `ground_truth.csv` and
`rule_id` / `rule_kind` / `coded_value` off the `rule_rejection` and `answer_rejected` events.
