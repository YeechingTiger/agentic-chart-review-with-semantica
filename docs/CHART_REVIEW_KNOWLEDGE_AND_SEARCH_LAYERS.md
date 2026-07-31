# Chart Review Knowledge Placement and Search Escalation

## 1. Knowledge assets are not one specification

ACR separates the source of truth from the means used to find and enforce it:

| Asset | Owns | Must not own |
|---|---|---|
| Source Guideline | Versioned manual, guideline, codebook, and locatable source text | Local retrieval behavior |
| Task Contract | Fields, value meaning, entity/time scope, evidence standing, precedence, witness, missingness, conflicts, and human-review boundary | Keywords, raw local note types, sampling thresholds, tool steps, budgets |
| Retrieval Experience | Measured field terms, document-concept priors, successful rescue paths, and yield statistics | Clinical truth |
| Site Mapping | Local note-type name to portable document concept | Source precedence or coding semantics |
| Skill | Reusable method such as entity anchoring, timeline reconstruction, or thread chasing | Hard permission or answer authority |
| Runtime Policy | Search order, escalation, stopping, sampling design, and budget allocation | The meaning of the answer |
| Runtime Control | Deterministic conditions on an answer's **provenance**: evidence exists, every quote re-reads at its offsets, the patient is in scope, the budget is unspent, no read is unfinished | Any judgement about an answer's **content** — which code is right, whether the record is specific enough, whether the chart was searched enough |
| Evaluator | Post-run quality, causal attribution, and repair routing | Direct production mutation |

The placement test is: **does this rule define the correct answer, or does it change the
probability/cost of finding it?** The former belongs in the Task Contract; the latter belongs
in Experience, Mapping, Skill, or Policy.

The model receives a compiled clinical-contract view. Retrieval assets are injected by the
selected runtime profile, so a retrieval experiment never silently changes clinical
semantics.

### A second placement test, added 2026-07-30: enforced or offered?

Placement says *which asset owns a rule*. It does not say whether the runtime may refuse an answer
over it, and this tree got that wrong five times in the same direction. Every deterministic rule
that judged an answer's **content** has now been measured and removed —
[DETERMINISTIC_RULES_REMOVED.md](DETERMINISTIC_RULES_REMOVED.md) has the numbers, and the headline
is that **60 of 254 recorded answer rejections (24%) refused a tuple that was exactly the
registry's**.

So the second test is: **is this a fact about what the run did, or a judgement about what the
answer means?**

| | enforced (may refuse) | offered (reference) |
|---|---|---|
| what it is | a fact about the run, decidable without clinical knowledge | a clinical judgement |
| examples | evidence exists; a quote re-reads at its offsets; a read stopped short; patient scope; spend; undeclared tool | which code the record supports; whether NOS is right; whether the chart was searched enough; which local type can settle a field |
| how it fails | it cannot be wrong about the corpus — it describes the run | it is a word list written from one chart and applied to every chart |
| a model that disagrees | is refused, and told exactly what would satisfy it | departs from it and says so in its reasoning; the departure is recorded |

A clinical rule in the Task Contract still reaches the model — as instruction, rendered into the
prompt. A wrong value against a stated instruction is an **instruction-following failure**, measured
as one (`answer_shape_miss`, and the evaluators) rather than patched with a regex.

## 2. Three-arm search experiment

> **The shared controls listed here are historical.** As written, the arms retained "field
> formats, answer checks, source precedence, discovered-conflict/thread closure" as enforced
> conditions. None of those is enforced now — see the second placement test above and
> [DETERMINISTIC_RULES_REMOVED.md](DETERMINISTIC_RULES_REMOVED.md). What every arm retains today
> is provenance only: evidence exists, quotes re-read at their offsets, `truncated` reads are
> unfinished, patient scope, spend and call limits, declared tool surface. The arm definitions
> below are otherwise accurate and are what the two pilots ran.
>
> `guideline-only` is now closest to the intended baseline, and even it was not clean: its three
> field-level abstentions were caused in part by the document stratifier telling runs their charts
> held no pathology while an FNA diagnosis sat in them.

All three arms retain patient scope, read-only tools, positive evidence, field formats,
answer checks, source precedence, discovered-conflict/thread closure, budgets, and audit.
`SPEC_INSUFFICIENT` remains a statement about the contract and is exempt from chart coverage.

### `guideline-only`

- Model sees the clinical Task Contract and the current patient's document inventory.
- No task keywords, note-type prior, coverage strata, or retrieval hints are supplied.
- It chooses terms, note types, reading order, and expansions itself.
- A targeted `EVIDENCE_INSUFFICIENT` may be accepted after the patient inventory is listed,
  but it carries `negative_basis=GUIDELINE_ONLY_TARGETED` and no coverage attestation.

This arm measures native model-plus-guideline capability.

### `conditional-negative-coverage`

- Starts identically to `guideline-only`; the proof asset is hidden.
- Coverage activates when the proposed answer is `EVIDENCE_INSUFFICIENT`, contains a missing
  field, or uses a configured NOS/unknown-shaped value.
- The runtime then reveals the coverage plan, forces its own samples, and rejects the answer
  until obligations pass or become unreachable.
- A positive answer with no negative-shaped field stops on its ordinary witness and conflict
  controls.

This arm measures the incremental safety and cost of coverage where a claim of absence is
actually being made.

### `always-coverage`

- Coverage terms and document strata are active from the first model call.
- Positive and negative answers both require the configured coverage proof.
- A positive answer that passes is recorded as `WITNESS_PLUS_COVERAGE`; it still does not put
  `coverage_attested` inside the answer because coverage is not its semantic proof.
- Coverage that cannot be earned routes to review instead of manufacturing consensus.

This is an experimental upper-bound arm, not a recommended default.

## 3. “Read everything” is field-specific — and it is the model's call

Coverage never means blindly reading every note. For the field making a negative-shaped claim, the
questions to settle are:

1. which kinds of document could establish this field at all;
2. whether they were actually opened — not whether a search came back empty;
3. whether the ones opened were finished, because the diagnosis is often in the last paragraph and
   an addendum that changes it is often after that;
4. whether more than one wording was tried, since no single term covers this corpus: `carcinoma`
   appears in 57.5% of diagnosis-bearing documents, `cancer` in 32.7%, and 23.9% contain neither
   nor five other candidates;
5. what remains unexamined, and why it was judged not worth examining;
6. what would change the answer, and whether it was looked for.

**These are asked of the model, not enforced against it.** The list above is
`assets/skills/coverage-judgement/SKILL.md`; the runtime counts what happened —
strata, reviewed counts, Clopper-Pearson residual bounds — and hands the counts back as
`advisories`. Five earlier versions of this section were enforced instead, and all five cost more
correct answers than they saved: coverage obligations produced roughly 150 answer rejections across
the recorded traces, 27 of which refused a tuple that was exactly the registry's.

The answer owes an account of the looking, and that account is the deliverable. An absence claim
with no account was the thing enforcement existed to prevent, and it turns out a refusal was never
what produced one.

Entity and temporal anchoring precede all three arms. Retrieval success attached to the wrong
tumor, specimen, episode, or time point is not a correct chart review.

## 4. Error-to-asset routing

| Observed causal fact | Repair target |
|---|---|
| Correct witness exists but never surfaced | Retrieval Experience, Site Mapping, search tool, or Policy |
| Witness surfaced but was not read completely | Skill or Policy |
| Witness was read but ignored/misinterpreted | Skill/model; Task Contract only if ambiguous |
| Evidence attached to wrong entity/time | Task Contract form if undefined; otherwise Skill/check |
| Wrong source won despite explicit precedence | Skill, or the Task Contract's precedence prose — **not** a deterministic check |
| Precedence or value meaning is missing/ambiguous | `SPEC_FORM` proposal |
| Current semantic rule is clinically wrong | `SPEC_CONTENT` question and human adjudication |
| Correct answer rejected | Delete the rule that rejected it, or demote it to advisory. It is not a tuning problem — see below |
| Invalid answer accepted | Evaluator and Skill. A value that violates stated instruction is an instruction-following failure and is counted, not gated |
| Value is well-formed but not a real code | Value domain — a **code table**, which is a fact, not a word list |
| Repeated path is faster but does not define truth | Candidate Retrieval Experience |
| Registry disagrees without chart-observable adjudication | Human queue only |

Two rows above changed on 2026-07-30 and the reason is worth stating, because the old routing is
what produced the loop. *"Correct answer rejected or invalid answer accepted → Runtime
Control/check implementation"* routed both directions of failure at the same target, so every
observed miss became a reason to write another check, and each check was written from the one chart
that motivated it. Measured over every recorded trace, the five clinical checks that accumulated
that way destroyed a correct value 58 times and preceded a correct one 21 times. A correct answer
rejected now routes at *deleting the rule*, and the two directions route separately.

Experience never becomes guideline merely because it repeats. Promotion into a Task Contract
requires a normative source or signed human semantic decision and independent validation. And a
Task Contract rule reaches the model as **instruction**: promotion does not make it enforceable.

## 5. Evaluation contract

Production may run sequentially, but causal experiments use independent paired arms with the
same patients, model, seed, budget, Task Contract, and tool surface.

- `guideline-only` vs `conditional-negative-coverage`: false abstention, critical miss,
  review routing, and incremental cost on negative-shaped cases.
- `guideline-only` vs `always-coverage`: total cost and regressions caused by treating
  coverage as a universal requirement.
- Positive cases: time to first valid witness, evidence validity, conflicts closed, documents
  read, calls, tokens, and cost.
- Negative/partial cases: chart-observable abstention accuracy, coverage validity, missed
  witness, and `COVERAGE_UNREACHABLE`.

Registry authority is declared per EvaluationTask. An unresolved registry reference is only
a disagreement signal; an operator-confirmed registry benchmark is ground truth for task
accuracy. Chart derivability remains a separate attribution variable in either case: it
distinguishes model/spec/retrieval error from a corpus evidence gap and determines whether a
semantic repair is justified.

The first ten-case real-chart coverage pilot and the resulting field-level verification
design are recorded in
[COVERAGE_THREE_ARM_PILOT.md](COVERAGE_THREE_ARM_PILOT.md). The controlled comparison of
native model planning against an up-front spec-derived retrieval plan is recorded in
[SEARCH_PLANNING_PILOT.md](SEARCH_PLANNING_PILOT.md).
