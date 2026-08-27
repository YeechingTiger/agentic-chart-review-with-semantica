# The core story

## The problem

A wrong chart-review answer rarely says where the work first went wrong. The agent may have missed
a source family, searched too narrowly, skipped a candidate note, misjudged a phrase, applied the
wrong conflict rule, or stopped too early. A raw model/tool trace contains the evidence needed to
investigate, but it is too detailed and unstable to be the primary unit of human review or
cross-run analysis.

The product goal is therefore not “summarize the trace.” It is:

> Let a human follow the consequential choices in order, judge each choice, identify the earliest
> questionable step, and route the remedy to retrieval, guideline, instrumentation, or model
> capability—while retaining exact drill-down to what actually happened.

## One run, three representations

```text
observable execution       semantic audit units          reusable intelligence
Langtrace + receipts  →    Decision Episodes       →     Semantica ContextGraph
```

These representations have different jobs.

### 1. Langtrace is the execution evidence

The Codex harness emits the model and tool spans of the actual run. Deterministic replay converts
the canonical export into immutable ReAct cycles with state before, action, observation, and state
after. Reconstruction may interpret those cycles; it may not add, delete, reorder, or move them.

The agent also records concise Runtime Decision Testimony for material choices: the question it
faced, chosen option, alternatives, rationale, claimed basis, cited refs, checked discriminating
facts, and uncertainty. The chart server seals each testimony with resolved refs, current state,
source event, schema, and hash. This is an auditable commitment, not private chain-of-thought and
not proof that the commitment was correct.

### 2. Decision Episodes are the audit units

An Atomic Decision is one material commitment among meaningful alternatives that one reviewer can
judge with one verdict. Choosing keywords, choosing notes to open, judging one note's standing,
resolving a conflict, and deciding to stop are separate decisions. Mechanical pagination and
transport are not decisions.

A Decision Episode contains one decision-bearing cycle and any immediately supporting execution
cycles. Luna assigns the evolving semantic function and subject after the run. Deterministic code
checks boundaries, refs, execution facts, and causal support. This split preserves both needs:

- runtime testimony says what the acting agent claimed at that moment;
- post-run reconstruction can adopt a better category or taxonomy without rerunning the review.

The selected analysis is append-only. A human chain never silently mixes episodes from different
reconstruction passes.

### 3. Semantica is the reusable decision layer

Verified episodes become native `ContextGraph` decisions. Their structured scenario describes the
pre-decision question in readable, de-identified language; outcome records the choice. A separate
metadata fingerprint preserves stable structural indexing without replacing the human scenario.
Explicit dependencies use Semantica's
`CAUSED`, `INFLUENCED`, or `PRECEDENT_FOR` relationships. Task clauses become independently
versioned native Policies, and `APPLIED_POLICY` is created only when runtime evidence actually
cited or checked the clause.

ACR does not implement a competing analytics graph. It supplies chart-review projection, identity
guards, provenance checks, and the human-readable view. Semantica supplies persistence, native
similar-decision retrieval, causal and impact analysis, policy versioning, and affected-decision
queries.

## What a human can do

### Audit one run

Read the decisions chronologically:

1. Was the note inventory sufficient?
2. Were search terms and source families reasonable?
3. Were all material candidates opened?
4. Was each note's standing judged correctly?
5. Was conflict resolved under the right evidence and rule?
6. Was stopping justified, and did the submitted answer follow?

Each step shows question, choice, rationale, policy/ref status, remaining model judgment, review
attention, state change, and raw Langtrace links. The abstraction reduces the default reading load;
it does not delete the source evidence.

### Compare repeated runs

Semantica searches native, human-readable decision scenarios within a category. ACR then limits
candidates to the intended cohort and different runs. A true same-point comparison additionally
requires the same case evidence and atomic question; a coarse structural match alone is never
called an exact disagreement.

If both choices relied on `own_knowledge` and neither applied a Policy, the system routes the pair
as `UNGROUNDED_OUTCOME_DIVERGENCE`. That tells a guideline author where ungoverned judgment was
inconsistent. It does not decide which outcome is clinically correct.

### Review a policy change

A Policy revision is append-only and linked to the prior version. Semantica returns decisions with
historical direct bindings to the changed version. That result is a precise re-audit queue: these
decisions used the clause. It is not a counterfactual claim that their final answers would change;
that requires a paired rerun with the revised Task Presentation.

## Fidelity contract

The projection is useful only while its provenance remains visible:

| Claim | Authority |
|---|---|
| tool execution, ref resolution, sealed receipt | server fact |
| what the agent said it relied on | self-reported testimony |
| cycle boundary, executed retrieval outcome, ref validation | deterministic derivation |
| decision function, question, semantic grouping | model reconstruction |
| selected analysis or step disposition | human adjudication |

No layer may upgrade ref resolution into semantic entailment, temporal adjacency into causation,
similarity into precedent, or policy impact into an answer-change prediction. Missing testimony is
shown as missing rather than filled with a plausible story.

## Checked evidence

The postdoc walkthrough is executable and checked against persisted real OpenRouter Luna/Terra
runs:

1. [Experiment notebook](../notebooks/01_run_chart_review_experiments.executed.ipynb) explains the
   chart-only agent and shows the answers from six available historical runs. The set is not
   presented as a balanced accuracy experiment.
2. [Audit notebook](../notebooks/02_trace_to_decision_chain.executed.ipynb) reduces one real run
   from 158 harness records / 22 canonical events / 21 cycles to eight chronological Decision
   Episodes. It identifies retrieval coverage as the main review question while preserving exact
   cycle accounting and raw drill-down on all eight.
3. [Semantica notebook](../notebooks/03_semantica_decision_intelligence.executed.ipynb) shows one
   readable native Decision, retrieves similar decisions, finds a true same-note task-only
   disagreement, and constructs a direct Policy-binding re-audit queue.

These are system-mechanism checks. A clinical expert is still needed to adjudicate correctness, a
paired rerun is needed to test a policy revision counterfactually, and a human study is still
needed to measure reviewer accuracy or time.
