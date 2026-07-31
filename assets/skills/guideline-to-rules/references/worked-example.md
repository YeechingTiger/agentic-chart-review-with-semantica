# Worked example: one paragraph of guideline prose to one executable rule

The rule built here is `NSCLC-ADJ-SYSTEMIC-II-IIIA`, which ships in
`assets/guidelines/nccn_nsclc_subset.yaml`. Every choice below is shown with the person who owns
it. The final block is compared, character-structure for character-structure, against the
shipped file by `tests/test_guideline_to_rules_skill.py`, so if this page and the file
disagree the suite goes red rather than teaching you a shape nobody runs.

The five patients at the end are invented. No chart was read to write this page.

---

## 0. The prose you start with

A paraphrase of the guideline position, in the shape guideline prose actually arrives in:

> Patients with completely resected stage II or IIIA disease should be offered adjuvant
> systemic therapy after recovery from surgery; regimen selection is informed by molecular
> results, and patients should be followed with surveillance imaging thereafter.

That is **three recommendations in one sentence**: give adjuvant therapy, choose it using
molecular results, then image on a schedule. It is also two eligibility populations, because
"completely resected" is not the same population as "stage II or IIIA".

## 1. Isolate one — and write down what you discarded

Kept: *give adjuvant systemic therapy to completely resected stage II–IIIA disease.*

Discarded, each to be written as its own rule or not at all:

| discarded clause | why it is a separate rule |
|---|---|
| "regimen selection is informed by molecular results" | different action, different timing anchor, different exception set. It became `NSCLC-BIOMARKER-BEFORE-FIRST-LINE`, with its own eligibility (stage IV, non-squamous) that barely overlaps this one. |
| "followed with surveillance imaging thereafter" | the interval is not stated anywhere in the prose. The recurrence spec refuses to invent one and writes `PLACEHOLDER_REQUIRES_CLINICAL_INPUT` instead; a rule cannot be written from a placeholder. |

Writing the discards down is the deliverable, not bookkeeping. An unwritten discard comes
back as pressure to bolt a fourth condition onto `satisfied_when` a month later, and
`tests/test_guideline_to_rules_skill.py::test_bundling_two_recommendations_changes_the_reported_rate`
measures what that costs: the same patient, same facts, scores 0/1 bundled and 1/2 split.

## 2. Decompose into the four predicate kinds

Each phrase of the kept sentence becomes a predicate of exactly one kind, and each predicate
names a variable that must be declared in `required_inputs`. The middle column is the one
that catches sloppiness: if you cannot say which kind a phrase is, it is usually two phrases.

| phrase | kind | variable | source |
|---|---|---|---|
| lung primary | `ELIGIBILITY` | `primary_site` | extraction spec, item 400 |
| non-small cell | `ELIGIBILITY` | `histology` | extraction spec, item 522 |
| malignant | `ELIGIBILITY` | `behavior` | extraction spec, item 523 |
| treated at this facility | `ELIGIBILITY` | `class_of_case` | registry limited dataset, item 610 |
| stage II or IIIA | `ELIGIBILITY` | `pathologic_stage_group` | extraction spec |
| "completely resected" (a) anatomic operation | `ELIGIBILITY` | `surgical_resection_extent` | not yet extractable |
| "completely resected" (b) negative margins | `ELIGIBILITY` | `surgical_margins` | not yet extractable |
| "adjuvant systemic therapy" | `ACTION` | `adjuvant_systemic_therapy_class` | not yet extractable |
| "after recovery from surgery" | `TIMING` | two dates, below | not yet extractable |
| performance status, comorbidity, refusal, trial, death | `EXCEPTION` | five variables | mixed |

Two things to notice.

**"Completely resected" split into two variables.** One phrase, two facts, recorded in two
different places in a chart — the operative note says lobectomy, the pathology synoptic says
margins. Left as one variable it would have needed a reader to decide what "complete" meant,
which is the disqualifier in section 7 of the skill.

**`class_of_case` is declared `registry_limited_dataset`, not `extraction_spec`.** The spec
for item 610 answers `WRONG_DATA_SOURCE` when asked of the notes, because class of case is a
fact about the reporting facility's relationship to the patient and is not written anywhere
in a chart. Declaring it as an extraction target would have produced a variable that is
`SPEC_INSUFFICIENT` on every patient and a rule that is `NOT_ASSESSABLE` on every patient.
Own the honest source, even when it is one you do not have yet.

## 3. The one clinical decision on this page: what "after recovery from surgery" means

The guideline says "after recovery from surgery". `days_between` needs an integer.

Somebody has to pick the integer, and the whole point of the `operationalisation:` block is
that the somebody is named and is a clinician:

```yaml
operationalisation:
  max_days_surgery_to_adjuvant: 120
  requires_clinical_signoff: true
  rationale: >
    NCCN states that adjuvant therapy follows recovery from surgery; it does not
    publish a day count. 120 days is an operationalisation chosen so that a normal
    post-operative recovery is not scored as a gap. ...
```

What follows from that number is not cosmetic:

- A patient who started therapy on day 121 is `NON_CONCORDANT`. Nothing clinical happened at
  midnight on day 120; the label did.
- `guideline_hash` is a hash of the whole file, so editing 120 to 90 changes it. Labels
  carrying different hashes are not comparable and must not be pooled. That is the intended
  behaviour, not a nuisance — and the test asserts the hash moves.
- The same test asserts the other half, which is the trap: the block says 120 and the
  condition in `satisfied_when` also says 120, and **they are two separate literals that
  nothing checks against each other**. Editing only the block moves the hash and changes no
  patient's outcome. Edit both, in one commit, or the number that was signed off is not the
  number that runs.

There is a second operationalised number on this page and it is worth seeing that it is
easy to miss: the exception fires at `ecog_performance_status_after_surgery` ≥ 2. Two rather
than three is a clinical cutpoint of exactly the same character as the 120, and the shipped
`operationalisation:` block does not mention it. `validate_guideline` checks only that a
block exists, not that it accounts for every number in the rule. Account for every number.

## 4. Bound both sides of the window

The timing predicate is written with a floor as well as a ceiling:

```yaml
- {op: days_between, from: date_of_definitive_surgery,
   to: date_of_first_adjuvant_systemic_therapy, min_days: 0, max_days: 120}
```

`min_days: 0` is not defensive tidiness. Take patient P01 below — concordant, therapy at day
56 — and move that one date to 41 days *before* the operation, which is P01b. That is
neoadjuvant therapy, and it does not satisfy an adjuvant recommendation. Measured, in
`test_removing_the_lower_bound_turns_neoadjuvant_therapy_concordant`:

| rule | P01b scores |
|---|---|
| as shipped, `min_days: 0` | `NON_CONCORDANT` |
| identical, with `min_days` deleted | `CONCORDANT` |

The deleted-floor version passes `validate_guideline` with zero complaints; `days_between`
requires only one of the two bounds. The engine will never tell you.

## 5. The exception catalogue

Five exceptions, each a declared variable, each with the evidence that discharges it:

| exception | evidence standard | why it is here |
|---|---|---|
| performance status ≥ 2 | a dated assessment after the resection, before the window closed | an undated performance status from before the operation is not evidence about fitness after it |
| patient refused | an explicit statement of refusal | the absence of a chemotherapy note is not a refusal, and treating it as one is the failure this whole catalogue exists for |
| contraindicating comorbidity | a *named* contraindication, not "frailty" | "frailty" is a judgement, and judgements do not belong in a rule |
| therapeutic trial enrolment | a named protocol with an enrolment date | protocol therapy is correct care that does not look like the recommended class |
| died before the window closed | date of death inside the same 0–120 day window | note it reuses the operationalised number; changing 120 changes who counts as having had the chance |

Held to the same standard as the primary variable, in both directions. `FOUND` with `false`
means the coverage proof looked and ruled it out, and the case can be a care gap.
`EVIDENCE_INSUFFICIENT` means nobody looked, and the case is `NOT_ASSESSABLE` — patient P04
below. The engine implements this ordering: exceptions are evaluated before an unknown
action is reported, because a documented contraindication settles the case whether or not
the action can be established.

## 6. Which unknowns make it unscorable

Walk every declared input and answer the question for each. For this rule the answer is the
same fifteen times: unknown makes it unscorable. Two are worth stating explicitly because
they look like they might have defaults and do not.

- `pathologic_stage_group` = `99`. A registry sentinel: present, well-formed, and meaning
  "nobody established this". It is declared in `unknown_value_codes` so it resolves to
  `UNKNOWN`. Without that declaration it would fail set membership like any other
  non-member and the patient would be reported `NOT_APPLICABLE` — a positive claim that they
  are outside the population, made from a value that asserts nothing.
- `patient_refused_adjuvant_systemic_therapy` missing. The tempting default is `false`
  ("nobody wrote that they refused, so they didn't"). That default is how a patient who
  declined treatment becomes a care gap. There is no safe default here. P04 is this case.

The closest thing to a safe default in the whole rule is `date_of_death`, and only because
the registry limited dataset supplies vital status for everyone in it — it is not a default,
it is a source.

## 7. What was kept out

- "Completely resected" → two objective variables, not a reader's assessment of completeness.
- "Adjuvant systemic therapy" → a membership test against a declared class list. Note what
  the list leaves out: single-agent chemotherapy is deliberately absent, so a patient who got
  it is `NON_CONCORDANT` rather than quietly counted as treated. That is a clinical decision
  visible in a value set an oncologist can review, not a judgement made at scoring time.
- "Should be offered" → scored as *received*. Offers are not reliably documented; the rule
  says what it measures and the exception catalogue carries the offers that were declined.

---

## The finished rule

```yaml
# expect: SHIPPED
id: NSCLC-ADJ-SYSTEMIC-II-IIIA
title: "Adjuvant systemic therapy after complete resection of stage II-IIIA NSCLC"
statement: >
  A patient with completely resected (anatomic resection, negative margins) pathologic
  stage II or IIIA non-small cell lung cancer should receive adjuvant systemic therapy
  of a guideline-concordant class, started within the post-operative window.
source:
  basis: "NCCN NSCLC — adjuvant treatment after surgical resection"
  paraphrase: true
  operationalisation:
    max_days_surgery_to_adjuvant: 120
    requires_clinical_signoff: true
    rationale: >
      NCCN states that adjuvant therapy follows recovery from surgery; it does not
      publish a day count. 120 days is an operationalisation chosen so that a normal
      post-operative recovery is not scored as a gap. It is a threshold a clinician must
      set, not an engineer — it is declared here rather than buried in code so that
      changing it changes the guideline hash and therefore breaks label comparability,
      which is the intended behaviour.
    min_days_surgery_to_adjuvant: 0
    min_days_rationale: >
      Systemic therapy dated before the operation is neoadjuvant, and neoadjuvant therapy
      does not satisfy an adjuvant recommendation. Without a lower bound the date
      arithmetic would accept it.

required_inputs:
  - {name: primary_site,          source: extraction_spec, spec_id: STORE.400_522_523.site_histology_behavior, item: "STORE [400]"}
  - {name: histology,             source: extraction_spec, spec_id: STORE.400_522_523.site_histology_behavior, item: "STORE [522]"}
  - {name: behavior,              source: extraction_spec, spec_id: STORE.400_522_523.site_histology_behavior, item: "STORE [523]"}
  - {name: class_of_case,         source: registry_limited_dataset, item: "STORE [610]",
     why: "not derivable from notes — assets/specs/STORE.610 answers SPEC_INSUFFICIENT / WRONG_DATA_SOURCE by design"}
  - {name: pathologic_stage_group, source: extraction_spec, spec_id: STORE.700_880.stage, item: "AJCC 8th pathologic stage group"}
  - {name: surgical_resection_extent,   source: not_yet_extractable, item: "NAACCR RX Summ--Surg Prim Site [1290]"}
  - {name: surgical_margins,            source: not_yet_extractable, item: "NAACCR RX Summ--Surgical Margins [1320]"}
  - {name: adjuvant_systemic_therapy_class,        source: not_yet_extractable, item: "first-course systemic therapy class"}
  - {name: date_of_definitive_surgery,             source: not_yet_extractable, item: "NAACCR Date of Most Definitive Surg [3170]"}
  - {name: date_of_first_adjuvant_systemic_therapy, source: not_yet_extractable, item: "NAACCR Date Systemic Therapy Started [3230]"}
  - {name: date_of_death,                          source: registry_limited_dataset, item: "date of death"}
  - {name: ecog_performance_status_after_surgery,  source: not_yet_extractable, item: "documented post-operative ECOG PS"}
  - {name: patient_refused_adjuvant_systemic_therapy, source: not_yet_extractable, item: "NAACCR Reason for No Systemic Therapy [1380] = refused"}
  - {name: contraindication_to_systemic_therapy,   source: not_yet_extractable, item: "documented contraindicating comorbidity"}
  - {name: clinical_trial_enrollment,              source: not_yet_extractable, item: "documented therapeutic trial enrolment"}

applies_when:
  - {op: matches, var: primary_site, pattern: "C34\\d"}
  - {op: in_set,  var: histology, set: nsclc_histology}
  - {op: equals,  var: behavior, value: "3"}
  - {op: in_set,  var: class_of_case, set: analytic_treated_at_reporting_facility}
  - {op: in_set,  var: pathologic_stage_group, set: stage_II_IIIA}
  - {op: in_set,  var: surgical_resection_extent, set: anatomic_resection}
  - {op: equals,  var: surgical_margins, value: "negative"}

satisfied_when:
  - {op: in_set, var: adjuvant_systemic_therapy_class, set: guideline_concordant_adjuvant_systemic}
  - {op: days_between, from: date_of_definitive_surgery,
     to: date_of_first_adjuvant_systemic_therapy, min_days: 0, max_days: 120}

# Each exception is itself an extracted variable and is held to the same evidentiary
# standard as the primary ones: FOUND=false means the coverage proof ruled it out,
# EVIDENCE_INSUFFICIENT means nobody looked and the case is NOT_ASSESSABLE.
exceptions:
  - id: performance_status_precludes_systemic_therapy
    label: "post-operative ECOG performance status 2 or worse"
    evidence_standard: "a dated PS assessment after the resection and before the adjuvant window closed"
    when: [{op: at_least, var: ecog_performance_status_after_surgery, value: 2}]
  - id: patient_refused
    label: "patient declined adjuvant systemic therapy"
    evidence_standard: "an explicit statement of refusal, not an absence of a chemotherapy note"
    when: [{op: is_true, var: patient_refused_adjuvant_systemic_therapy}]
  - id: contraindicating_comorbidity
    label: "documented contraindication to systemic therapy"
    evidence_standard: "a named contraindication, e.g. dialysis-dependent renal failure, not 'frailty'"
    when: [{op: is_true, var: contraindication_to_systemic_therapy}]
  - id: therapeutic_clinical_trial
    label: "enrolled on a therapeutic clinical trial"
    evidence_standard: "a named protocol with a documented enrolment date"
    when: [{op: is_true, var: clinical_trial_enrollment}]
  - id: died_before_window_closed
    label: "died before adjuvant therapy could be delivered"
    evidence_standard: "date of death within the post-operative window"
    when: [{op: days_between, from: date_of_definitive_surgery, to: date_of_death,
            min_days: 0, max_days: 120}]
```

## The rule under five invented patients

Every row is executed against the shipped rule by
`test_the_worked_example_cohort_scores_the_way_the_document_says`. Two of the six are in a
concordance denominator; four are not, and each of the four is excluded for a different
reason.

```yaml
# expect: SCENARIOS
base:
  primary_site:                            {status: FOUND, value: "C341"}
  histology:                               {status: FOUND, value: "8140"}
  behavior:                                {status: FOUND, value: "3"}
  class_of_case:                           {status: FOUND, value: "10"}
  pathologic_stage_group:                  {status: FOUND, value: "IIB"}
  surgical_resection_extent:               {status: FOUND, value: "lobectomy"}
  surgical_margins:                        {status: FOUND, value: "negative"}
  adjuvant_systemic_therapy_class:         {status: FOUND, value: "platinum_doublet_chemotherapy"}
  date_of_definitive_surgery:              {status: FOUND, value: "20190304"}
  date_of_first_adjuvant_systemic_therapy: {status: FOUND, value: "20190429"}
  date_of_death:                           {status: FOUND, value: null, negative_basis: GATE_VALIDATED}
  ecog_performance_status_after_surgery:   {status: FOUND, value: "1"}
  patient_refused_adjuvant_systemic_therapy: {status: FOUND, value: false}
  contraindication_to_systemic_therapy:    {status: FOUND, value: false}
  clinical_trial_enrollment:               {status: FOUND, value: false}

cases:
  # Platinum doublet on day 56. In the denominator, in the numerator.
  - id: P01
    outcome: CONCORDANT
    overrides: {}

  # The same record with one date moved: therapy 41 days BEFORE the operation. This is
  # neoadjuvant therapy and the floor is what refuses it.
  - id: P01b
    outcome: NON_CONCORDANT
    overrides:
      date_of_first_adjuvant_systemic_therapy: {status: FOUND, value: "20190122"}

  # No adjuvant therapy, and the coverage gate closed on that absence. This is the care gap,
  # and it is the only kind of row that belongs in a non-concordance count.
  - id: P02
    outcome: NON_CONCORDANT
    overrides:
      adjuvant_systemic_therapy_class:         {status: FOUND, value: null, negative_basis: GATE_VALIDATED}
      date_of_first_adjuvant_systemic_therapy: {status: FOUND, value: null, negative_basis: GATE_VALIDATED}

  # Identical to P02 except that the refusal is documented. Correct care; not a gap.
  - id: P03
    outcome: EXCEPTION_DOCUMENTED
    exception_id: patient_refused
    overrides:
      adjuvant_systemic_therapy_class:         {status: FOUND, value: null, negative_basis: GATE_VALIDATED}
      date_of_first_adjuvant_systemic_therapy: {status: FOUND, value: null, negative_basis: GATE_VALIDATED}
      patient_refused_adjuvant_systemic_therapy: {status: FOUND, value: true}

  # Identical to P02 except that nobody established whether a refusal was documented. P03
  # and P04 are indistinguishable from the chart alone; defaulting P04 to false makes it P02.
  - id: P04
    outcome: NOT_ASSESSABLE
    blocking_inputs: [patient_refused_adjuvant_systemic_therapy]
    overrides:
      adjuvant_systemic_therapy_class:         {status: FOUND, value: null, negative_basis: GATE_VALIDATED}
      date_of_first_adjuvant_systemic_therapy: {status: FOUND, value: null, negative_basis: GATE_VALIDATED}
      patient_refused_adjuvant_systemic_therapy: {status: EVIDENCE_INSUFFICIENT, value: null}

  # Stage IA. Determinately outside the population — not an unknown, and it must not dilute
  # the unknowns either.
  - id: P05
    outcome: NOT_APPLICABLE
    overrides:
      pathologic_stage_group: {status: FOUND, value: "IA2"}
```

## Who owned each decision

| decision | owner |
|---|---|
| which recommendation to isolate, and what the discards are | analyst |
| which phrase is `ELIGIBILITY` versus `ACTION` versus `TIMING` | analyst |
| 120 days for "recovery from surgery" | thoracic oncologist, signed off |
| performance status cutpoint of 2 rather than 3 | thoracic oncologist, **not yet signed off** — see section 3 |
| which drug classes count as guideline-concordant | thoracic oncologist |
| which histology codes are non-small cell, and that 8000/8010 are unknowns rather than exclusions | oncologist with the registrar |
| that class of case comes from the registry and not from notes | registrar |
| that `days_between` treats an imprecise date as an interval | engineer |
| binding the paraphrase to a specific guideline version | thoracic oncologist; still `NOT_BOUND` in the shipped file |

The last row is the honest state of this file: it is an engineering subset for validating
the rule engine, and no number computed from it may be reported as a concordance rate until
the binding and the sign-offs exist.
