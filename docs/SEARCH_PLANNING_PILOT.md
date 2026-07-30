# Search Planning Pilot

## Question

Does giving the chart-review agent a search plan before it starts produce a more accurate
answer than letting the model plan from the clinical task contract and the patient's document
inventory?

The answer from this pilot is **not yet**. A precomputed plan changed search behavior and
sometimes reduced work, but it did not improve accuracy on the ten-case registry-ground-truth
benchmark. Search planning should therefore remain a versioned RuntimePolicy experiment, not
a new clinical rule or a mandatory checklist.

## What was isolated

The clinical Task Contract, evidence gate, answer checks, patient scope, model, temperature,
sampling seed, model-call limit, cost limit, tool surface, and ten local patient charts were
held fixed. No patient artifact was written to the repository.

| Arm | Model's own planning | Up-front retrieval experience | Coverage gate |
|---|---|---|---|
| `native-planning` | Yes | None | Off |
| `experience-preplanned` | Yes | Spec-derived keywords and document-type strata | Off |

The preplanned arm did not force completion of the strata. It still stopped on a grounded
positive witness. This matters: the experiment tests retrieval priors, not the previously
observed rejection behavior of an always-on coverage proof.

The primary analysis used one trajectory per case and arm. Only the one case whose primary
outputs disagreed received two additional trajectories per arm. These stability runs are
reported separately and do not inflate the full-cohort denominator.

## Primary results

| Metric | Native planning | Experience-preplanned |
|---|---:|---:|
| Three-field exact match | **4/10** | 3/10 |
| Primary-site match | **9/10** | 8/10 |
| Histology match | 5/10 | 5/10 |
| Behavior match | 7/10 | 7/10 |
| Gate-valid final output | 10/10 | 10/10 |
| Model calls | **139** | 143 |
| Search tool calls | **64** | 97 |
| Read-document tool calls | 72 | **40** |
| Unique documents read | 38 | **34** |
| All tool calls | 246 | 246 |
| Priced model cost | $2.450 | **$2.245** |
| Sum of per-run elapsed time | **13.55 min** | 16.41 min |

Nine of ten final field tuples were identical. The preplanned arm shifted work from reading
whole documents to issuing more targeted searches, but this did not produce a cohort-level
accuracy gain. Its total priced cost was about 8% lower, while model calls and elapsed time
were higher.

The only primary disagreement was CASE009. Across the primary run plus two stability runs:

| Arm | Exact trajectories |
|---|---:|
| Native planning | 3/3 |
| Experience-preplanned | 2/3 |

This is a useful warning signal, not a stable treatment-effect estimate. The cohort is small,
and temperature zero does not guarantee identical remote-model executions.

## Mechanism findings

The aggregate accuracy result hides three different effects.

### A plan can accelerate a hard successful case

In CASE002 both arms returned the exact registry answer. The preplanned arm reduced:

- model calls from 27 to 15;
- unique documents read from 12 to 4;
- priced cost from $0.558 to $0.277;
- elapsed time from 273 to 120 seconds.

This is the desired use of retrieval experience: reach the same decisive witness with less
unfocused reading.

### A plan can add work without changing the answer

In CASE004 both arms ended with the same non-exact partial result. The preplanned arm
increased model calls from 12 to 17, unique documents from 4 to 5, cost from $0.264 to
$0.517, and elapsed time from 83 to 219 seconds. A broad plan is therefore not intrinsically
an efficiency improvement.

### Better retrieval can expose a downstream entity/gate defect

In the primary CASE009 preplanned run, the agent found the definitive upper-lobe tumor
witness, but it also surfaced a separate lower-lobe finding. The current deterministic
answer check collapsed lobe mentions across cited evidence without a sufficiently strong
tumor-entity association. It repeatedly rejected the registry-correct `C341` candidate as a
same-tumor site conflict. The agent eventually yielded to the gate and returned `C349`.

The trace supports this causal chain:

```text
broader targeted retrieval
→ evidence from a second lesion enters the ledger
→ entity-insensitive site-conflict check fires
→ correct specific-site candidates are rejected
→ final answer retreats to NOS
```

The preplanned arm used 11 searches, 10 evidence records, 16 answer submissions, and 7
recorded rejections in this run, versus 5 searches, 3 evidence records, 4 submissions, and
1 rejection in the native run. In the two additional preplanned trajectories this failure
did not recur, which explains the 2/3 stability result.

This is not evidence that less retrieval is generally safer. It is evidence that higher
recall requires entity-aware evidence and conflict checks.

The truth-blind CODE evaluation pipeline marked both arms structurally valid. That is
expected: evidence validity and internal gate consistency cannot determine whether the gate
rejected a gold answer. A separate GOLD outcome evaluator and causal attribution are needed
for that question.

## SUPERSEDED FROM HERE (2026-07-30)

Everything above is the measurement and it stands. Everything below was the design proposed on
the strength of it, and it was not built — the `SearchPlan` contract, the three runtime phases and
the retrieval-experience asset schema all assume a runtime that enforces a plan. That runtime has
been dismantled. See [DETERMINISTIC_RULES_REMOVED.md](DETERMINISTIC_RULES_REMOVED.md).

What changed the conclusion is not a new experiment but a fuller reading of this one. Per-case
attribution of the native arm's eleven field misses puts **none of them on a document the search
failed to find**: six are abstentions (two of the three caused by the document stratifier, which
told the run its chart held no pathology while an FNA diagnosis sat in it), two are histology
subtype precision, one is primary-site NOS calibration. This pilot ablated search planning against
a cohort containing **zero retrieval failures**, which is why nine of ten output tuples were
identical. The null result is a property of the design, not a finding about priors.

The other correction: the arm's "prior" was the five spec-derived keywords
`pathology, biopsy, final diagnosis, specimen, carcinoma`, which the spec's own provenance had
already measured at **87.4% recall over 276,054 documents**, missing an answer-bearing document for
31.7% of patients because the list has `carcinoma` and not `cancer`. So the treatment was a
known-falsified asset, and the result says an uncertified list does not help rather than that
priors do not.

What was built instead:

- **Document concepts are reference, not a plan.** `src/acr/document_concepts.py` gives the model
  seven portable concept descriptions and the patient's own type list; it decides what to open.
  No ordering, no keyword list, no measurement claimed. A certified prior would render under its
  own heading with its measurement attached and remain declinable; there is none yet.
- **The stopping rule is the model's.** `skills/coverage-judgement/SKILL.md` carries the
  questions to settle before claiming absence. `evaluate_gate` still counts and now advises.
- **Search got notation tolerance instead of a plan.** Folding whitespace and hyphens found
  +225 documents on `non-small cell` (+10.0%) and +1.6% over ten phrases. Synonyms stay out: the
  best single term covers 57.5% of diagnosis-bearing documents and seven together miss 23.9%, so
  recall comes from the model searching several times — which deleting the term budget made
  possible.

The two-by-two retrieval ablation below is still the right experiment, with two changes: the
keyword asset has to be certified before it is an arm at all, and the cohort has to be **sampled
to contain retrieval-hard cases** or it will return another null. Registry exact match should be
reported against the constant-predictor baseline of 19.9%, and `behavior` — 99.7% a single value,
and never failing independently of `histology` in this cohort — should not be a third multiplier
on the primary endpoint.

---

## Design decision (NOT BUILT — see above)

### The plan is a typed runtime artifact, not a Skill

The reusable method for constructing or revising a plan can be a Skill. The actual plan
produced for a task and patient is a typed `SearchPlan` owned by a RuntimePolicy:

```text
SearchPlan
  target fields and entity/time anchors
  candidate queries
    query text
    field and evidence role
    provenance: model | validated experience | conflict-generated
    expected information gain
  document-concept priorities
    definitive / corroborating / localization / low-prior
  positive-witness stopping rule
  escalation triggers
  falsification queries
  per-phase budget
```

A local Site Mapping translates portable document concepts such as `definitive pathology`
or `operative localization` into corpus-specific note-type names. Raw local note names do
not belong in the clinical Task Contract.

### Three runtime phases

The preferred runtime is not “plan everything, then read everything”:

```text
Phase 1: hypothesis-directed search
  model forms an initial plan from the Task Contract
  validated retrieval experience is offered as a prior, not a checklist

Phase 2: witness and interpretation
  read the smallest definitive source set
  attach evidence to field + tumor/entity + time + evidence role
  stop a positive field when its witness is valid and discovered conflicts are closed

Phase 3: conditional verification
  activate only for a negative/partial claim, unresolved conflict, entity ambiguity,
  known high-risk field, or failed proof obligation
  search specifically for evidence that would falsify the current hypothesis
  stop when a round adds no evidence, closes no conflict, and reduces no obligation
```

Global “look at everything” remains a diagnostic arm. It is not the third phase.

## What a retrieval-experience asset should contain

The current broad terms (`pathology`, `biopsy`, `final diagnosis`, `specimen`,
`carcinoma`) are candidate assets, not validated knowledge. A production asset should be
field- and role-specific:

```yaml
asset_id: lung-site-histology-retrieval
version: 0.1.0
task_scope: site_histology_behavior

queries:
  - id: definitive-histology
    field: histology
    evidence_role: definitive_diagnosis
    terms: [final diagnosis, addendum, amended, pathology consult]

  - id: site-origin
    field: primary_site
    evidence_role: origin_not_container
    terms: [tumor site, arises from, origin, lobe, bronchus]

  - id: invasive-behavior
    field: behavior
    evidence_role: invasion
    terms: [invasive, microinvasion, in situ]

document_concepts:
  - concept: definitive_pathology
    priority: 1
    action: read_final_and_addenda
  - concept: operative_localization
    priority: 2
    action: read_if_origin_unsettled
  - concept: specialist_summary
    priority: 3
    action: search
  - concept: imaging_localization
    priority: 4
    action: search_if_pathology_lacks_origin

safeguards:
  - conflict_requires_same_entity_and_time_scope
  - specimen_container_does_not_define_origin
  - separate_lesion_does_not_create_same_tumor_conflict
  - later_addendum_supersedes_preliminary_diagnosis
```

Candidates may be generated from the spec and successful traces, but they become
`VALIDATED_EXPERIENCE` only after held-out testing. Useful asset metrics are:

- decisive-witness recall and time to first witness;
- marginal yield of each query and document concept;
- full-document reads avoided;
- false-conflict and wrong-entity rates;
- exact accuracy, critical misses, abstention, cost, and subgroup regressions.

Terms that merely appear often are not automatically useful. A term can have high hit rate
and negative information gain.

## Next experiment

Before another planning comparison, repair or isolate the entity-insensitive site-conflict
check exposed by CASE009. Otherwise an improved retriever is unfairly penalized by a
downstream control defect.

Then use a two-by-two retrieval ablation with coverage still off:

| Arm | Candidate keywords | Note-concept priority |
|---|---|---|
| A | Off | Off |
| B | On | Off |
| C | Off | On |
| D | On | On |

Each case should receive multiple independently registered trajectories. Report case-level
paired accuracy and efficiency, not a pooled count that treats repeated trajectories as new
patients. Only after a retrieval prior is certified should field-conditional verification
be crossed on/off in a second experiment.

Adoption requires:

- no critical or subgroup accuracy regression;
- no increase in wrong-entity or false-conflict errors;
- a predeclared practical gain in witness recall, accuracy, or cost;
- reproducibility on a sealed patient cohort;
- versioned lineage for the spec, plan asset, model, tools, and checks.

The conclusion of this pilot is therefore narrower than “planning does not work”:
**unvalidated up-front planning is not more accurate by default; validated, field-specific
retrieval priors may improve search efficiency, but only when downstream evidence and gate
logic remain correct under the additional evidence they surface.**
