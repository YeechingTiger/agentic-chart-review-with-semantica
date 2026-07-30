# Coverage Policy Three-Arm Pilot

## Purpose

This pilot isolates three different claims that had previously been mixed together:

1. a model can review a chart using only the clinical Task Contract;
2. a coverage proof may help when an answer makes a negative-shaped claim;
3. universal coverage may prevent premature closure.

The arms use the same ten local patient charts, task, model, seed, per-run budget, answer
checks, evidence gate, patient boundary, and open-thread control. Patient artifacts and the
unresolved registry reference remain outside the Git worktree. The repository contains only
aggregate results.

For this cohort the registry values are operator-confirmed benchmark ground truth. Exact
match is therefore reported as task accuracy. A separate chart-derivability adjudication is
still useful for causal attribution: it distinguishes an agent miss from a source-corpus
evidence gap, but it does not change the benchmark label.

## Frozen arms

| Arm | Initial retrieval knowledge | When coverage is required |
|---|---|---|
| `guideline-only` | Clinical contract and patient document inventory | Never; targeted field-level abstention is allowed |
| `conditional-negative-coverage` | Same as guideline-only | On case abstention, missing populated fields, or configured NOS/unknown-shaped values |
| `always-coverage` | Coverage terms and document strata from the first turn | Before either positive or negative output |

All three retain deterministic field formats, answer checks, admissible positive evidence,
closure of discovered conflicts/threads, patient scope, and cost limits. Coverage is the
experimental variable; those shared controls are not.

## Results

| Arm | Final statuses | Gate-valid cases | Coverage active / passed | Model calls | Rejections | Documents read | Elapsed minutes | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `guideline-only` | 7 FOUND, 3 field-level EVIDENCE_INSUFFICIENT | 10/10 | 0/0 | 115 | 6 | 25 | 8.32 | $1.682 |
| `conditional-negative-coverage` | 4 FOUND, 1 EVIDENCE_INSUFFICIENT, 5 NO_ANSWER | 5/10 | 7/1 | 244 | 57 | 383 | 36.31 | $4.993 |
| `always-coverage` | 3 FOUND, 2 SPEC_INSUFFICIENT, 5 NO_ANSWER | 5/10 | 10/3 | 274 | 71 | 609 | 56.77 | $5.276 |

Registry-ground-truth exact match was 3/10, 3/10, and 2/10 respectively. The two earlier
profiles both achieved 5/10:

| Earlier profile | Exact match | Field matches: site / histology / behavior |
|---|---:|---:|
| `current-stratified-coverage` | 5/10 | 7/10 / 5/10 / 8/10 |
| `witness-first-baseline` | 5/10 | 9/10 / 6/10 / 9/10 |

The earlier witness-first profile was not a guideline-only baseline. It exposed the
spec-derived keyword and document-type plan from the first model turn, then stopped on a
witness. The new guideline-only arm deliberately removed those retrieval-experience assets.
It lost two previously exact cases: one became over-specific at the primary-site field and
one abstained on histology and behavior.

Conditional coverage activated in seven cases. Only one activated case completed its
coverage proof. Five activated cases ended at a model-call limit or rejection loop. Compared
with guideline-only, conditional coverage recovered two previously missing field values in
one case, but lost thirteen populated field values across five cases.

Always coverage recovered no field value that guideline-only lacked and lost fifteen
populated field values across six cases. Its failures included rejection loops, model-call
limits, and one graph recursion limit. Starting coverage earlier therefore did not repair
the proof-execution problem.

The CODE evaluation pipeline produced 60 results for 30 distinct trajectories. There were
14 quality failures, concentrated in the two coverage arms. Re-running the evaluation
produced no additional events, confirming stable trajectory identity and ledger
idempotency. Application audit correlated PHI locations under the approved provider
boundary into zero boundary incidents.

## Interpretation

This pilot shows that the new arms are less accurate than the two earlier runs on the
registry-ground-truth benchmark. It also shows that the current conditional and always-on
implementations are not viable online policies: their proof-execution cost and failure rate
overwhelm any observed benefit.

The main failure is architectural rather than a missing prompt sentence:

- Coverage definition and coverage execution are coupled.
- A language model performs mechanical search, sampling, and ledger bookkeeping.
- Case-level coverage lets one difficult field block already established fields.
- NOS value shape is treated as a proxy for missing evidence, even when NOS may be the
  correct result of a conflict rule.
- Coverage, answer checks, and open-thread controls can produce long interacting rejection
  loops.
- Full tool results accumulate in the model context, increasing token and wall-clock cost.
- The conditional and always-on gates sometimes destroyed an already correct candidate.
  Conditional runs submitted the exact gold answer before final failure in three cases;
  always-on runs did so in two cases.
- A configured NOS value was treated as a generic absence claim. In one case the
  conflict-resolving gold answer was submitted exactly ten times and rejected into a
  model-call-limit failure.

Universal “read everything” is therefore not a default runtime requirement. It remains an
offline/high-risk verification experiment until it demonstrates critical-miss reduction on
adjudicated held-out gold and passes cost and process certification.

## Corrected experimental design

The three-arm comparison confounded two independent interventions:

1. whether retrieval experience is exposed before search;
2. whether a coverage proof can block the answer.

The next experiment must use a factorial design:

| Retrieval experience | Coverage gate | Purpose |
|---|---|---|
| off | off | Native guideline-only baseline |
| on | off | Measure keyword/note-prior acceleration and accuracy |
| off | field-conditional | Measure coverage without experience |
| on | field-conditional | Test whether experience makes verification reachable |

`always-coverage` remains a diagnostic fifth arm, not a candidate default.

Each patient needs multiple independent trajectories or a deterministic model configuration.
The current `--seed` controls the runtime sampler, not model generation, and all real runs
used temperature 1.0. One trajectory per patient therefore cannot separate policy effect
from model variance. The code bundle must also be content-addressed; `292dc90-dirty` is not a
reproducible code identity.

The two coverage-off cells were subsequently rerun with a frozen source bundle and
temperature zero. Native planning achieved 4/10 exact matches and up-front retrieval
experience achieved 3/10; nine final field tuples were identical. The preplanned arm used
fewer whole-document reads and slightly less priced model cost, but more searches, model
calls, and elapsed time. The only primary output regression exposed an entity-insensitive
site-conflict check rather than a simple witness-retrieval miss. See
[SEARCH_PLANNING_PILOT.md](SEARCH_PLANNING_PILOT.md) for the paired design, stability runs,
and resulting SearchPlan contract.

## SUPERSEDED FROM HERE (2026-07-30)

The results and the interpretation above stand. The design below does not: it proposes a
*better-executed* coverage gate — a deterministic coverage executor, three named strategies, a
progress gate — and coverage stopped being a gate at all. See
[DETERMINISTIC_RULES_REMOVED.md](DETERMINISTIC_RULES_REMOVED.md).

The measurement that settled it, over every recorded trace rather than these ten cases: coverage
obligations produced roughly **150 answer rejections, 27 of which refused a tuple that was exactly
the registry's answer**. Across all rejection reasons, 60 of 254 (24%) refused the exact registry
tuple, and twelve runs held it and shipped something else — eight of them shipping nothing at all.

This document already contained the reason, in its own interpretation section: *"the conditional and
always-on gates sometimes destroyed an already correct candidate"*, and *"a configured NOS value was
treated as a generic absence claim"* — one run submitting the gold answer ten times before a
call-limit failure. What is added now is that the pattern is not a fixable execution defect. "Have
I looked at enough of this chart to say something is absent?" is a clinical judgement about one
patient and one question, and every version of it written in code has cost more correct answers
than it saved.

What was built instead:

- **`evaluate_gate` is advisory by default.** The arithmetic is untouched — strata counts,
  reviewed counts, Clopper-Pearson bounds — and routes to `advisories` instead of `missing`.
  `gate_answer` and `gate.check` forward it, because asking the model to judge coverage without
  showing it what the runtime counted is not a design. `enforce=True` remains for the diagnostic
  arm this document already declined to make a default.
- **The obligation is prose.** `skills/coverage-judgement/SKILL.md`: when the question arises at
  all (only on an absence claim), why a NOS code is a positive claim rather than a confession,
  how to judge which local type names can establish a field, and what to write down about the
  looking. Declining is recorded, not refused — which was always the real requirement, and it
  never needed a refusal.
- **The `nos_or_unknown` activation branch is gone.** It inferred an absence claim from the
  *shape* of a value. `8000`/`8010`/`8046` are the registry's own answer for 10.8% of the corpus
  and `C349` for 9.6%, and `conflict_requires_nos` used to *order* the agent toward one of them —
  so the same value could be demanded by one rule and treated as unproven absence by another. A
  missing field is still a negative claim; that is a fact about the submission, not an inference
  from a code table.
- **The stratifier that fed the gate is gone too**, and it was wrong in a way none of these arms
  could see: `["Pathology", "Cytology"]` as a substring matched `Speech-Language-Pathology-Note`
  and missed `Non-Gyn-Cyto-FNA` (1,285 documents), `FN-Aspirate-Report` (881) and
  `SURG-PATH-RESULT` (231). **107 of the 219 patients with a zero `can_establish` count hold one
  of those reports anyway.** `guideline-only`'s three field-level abstentions were partly this,
  not the arm.
- **The read permission is gone.** `_out_of_plan` refused a read whenever the plan had filed the
  document's type in the `sample` bucket — 138 times, over buckets from that same substring.

The field-level design below is still the better *shape* for the question, and the per-field
record it proposes (`witness_found`, `negative_claim`, `negative_reason`, …) is worth keeping as
something the model reports and the runtime records. What must not come back is the last line of
it: a progress gate that refuses.

---

## Next design (NOT BUILT — see above)

The next coverage version should be a field-level verification pass:

```text
initial extraction
→ field hypotheses and missingness provenance
→ select only unresolved/high-risk fields
→ deterministic coverage executor
→ send only evidence/conflict deltas to the model
→ progress gate
→ preserve completed fields; route unresolved fields
```

Each field should record:

```text
witness_found
negative_claim
negative_reason
coverage_strategy
coverage_required
coverage_passed
unresolved_conflicts
```

At least three coverage strategies should remain distinct:

- `source-exhaustion`: enumerate and finish sources capable of establishing the field;
- `targeted-falsification`: search specifically for evidence that would overturn the
  current hypothesis;
- `residual-sampling`: estimate risk in the remaining low-prior document universe.

The policy selects among them based on field semantics and risk. It must not automatically
stack all three.

Progress-based stopping is required. A verification round must add admissible evidence,
close a conflict/thread, or reduce outstanding field obligations. Otherwise it stops and
routes the unresolved field. Rejection efficiency and incremental evidence yield become
first-class evaluation metrics.

## Defects found and repaired after the frozen run

The arms were not modified during the comparison. Two shared framework defects discovered
by the pilot were repaired afterward:

- A populated field inside `EVIDENCE_INSUFFICIENT` bypassed field-format checks. The gate now
  validates every populated partial field.
- Canonical `Trajectory.created_at` used ingestion time, so repeated evaluation changed its
  content hash and defeated append-only idempotency. It now uses the recorded run timestamp;
  coverage state and spend are also preserved by the adapter.

These repairs require a new runtime version before another clinical comparison. The frozen
results above remain the measurement of the original three-arm implementation.
