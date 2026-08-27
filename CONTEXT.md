# Domain context

## Goal

Help a human audit a chart-review agent, locate the decision that caused a bad answer or weak
coverage, and decide whether the remedy belongs in the guideline, retrieval behavior,
instrumentation, or model capability.

The system does not attempt to expose private chain-of-thought. It records concise, reviewable
commitments and preserves the observable trace that supports or contradicts them.

## Core language

**Task Contract**
The chart-review requirement: target Field, acceptable evidence, exclusions, conflict rules,
proof obligations, output shape, and discriminating facts. It is the normative domain artifact.

**Task Presentation**
The exact material offered to one run. `task_only` contains the task/output contract;
`policy_bundle` also contains versioned guideline clauses. Its content hash is part of run identity.

**Semantica Policy**
A native, versioned Semantica rule object. A Task Contract clause may be projected as a Policy, but
the concepts are not synonyms: the Contract is the authored domain document; Policy is the graph
representation used for binding, versioning, and impact queries.

**Layer-1 event**
One append-only chart-review event emitted by the tool server: model/run metadata, tool call and
result, gate result, or accepted answer. Langtrace is the canonical remote store.

**ReAct Cycle**
A deterministic replay unit containing state before, action, observation, and state after. Cycles
are immutable during reconstruction: they cannot be added, removed, reordered, duplicated, or
moved.

**Runtime Decision Testimony**
The agent's concise contemporaneous account of a material choice: facing question, selected
option, alternatives, rationale, claimed basis, cited refs, checked discriminating facts, and
uncertainty. It is `SELF_REPORTED`, not automatically correct.

**Runtime Decision Receipt**
The server-sealed envelope around testimony plus server facts: resolved refs, state at recording,
source event, schema, and content hash. It proves what was said and what refs resolved; it does not
prove that the rule semantically entailed the choice.

**Atomic Decision**
One material commitment among meaningful alternatives that one reviewer can judge with one
verdict. If the first half can be right while the second is wrong, they are two Atomic Decisions.
Choosing a keyword batch, selecting a note set to open, judging one note's standing, resolving a
conflict, and deciding to stop are different decisions.

**Decision Episode**
The audit envelope for exactly one Atomic Decision: one decision-bearing cycle followed by zero or
more contiguous support cycles that execute or observe that commitment. Mechanical pagination,
transport, and termination remain non-decisions.

**Decision function**
The post-run semantic kind (`where_to_look`, `standing`, `which_wins`, `enough`, and the other
controlled types in `decision_types.py`). It is deliberately assigned after the run so the
taxonomy can evolve without rerunning the agent.

**Decision subject**
What the choice acted on: inventory, source family, query batch, document set, evidence item,
evidence relationship, sufficiency, or answer. Function and subject together provide a stable
similarity key without collapsing distinct retrieval choices.

**Decision grounding**
Three separate audit questions:

1. Did every claimed reference resolve against material actually offered or observed?
2. Does the referenced clause semantically entail the choice?
3. What judgment remained for the model, including operational discretion or outside knowledge?

Only the first is mechanically established by reference resolution. The human view must never
present it as proof of the second.

**Causal assertion**
An evidenced `CAUSED`, `INFLUENCED`, or `PRECEDENT_FOR` relationship. Temporal adjacency alone is
not causation. Explicit runtime refs such as `decision:N`, `finding:N`, `search:q`, and `note:id`
can deterministically establish influence.

**Analysis selection**
An append-only choice of which reconstructed analysis is authoritative for one run. A chain query
must not silently mix episodes from multiple reconstruction passes or verifier versions.

## Truth and provenance

Every important field keeps one of these meanings:

- `SERVER_FACT`: tool execution, reference resolution, sealed receipt identity.
- `SELF_REPORTED`: what the acting agent explicitly testified at runtime.
- `DETERMINISTIC_DERIVED`: fixed replay, reference checks, cycle boundaries, execution outcomes.
- `MODEL_RECONSTRUCTED`: Luna's post-run interpretation of the fixed trace.
- `HUMAN_ADJUDICATED`: a reviewer's explicit disposition or selection.

These sources may be shown together but must not be collapsed into fictional chain-of-thought.

## Semantica projection

Only verified Decision Episodes become Semantica Decision nodes. Cycles, testimony, state,
findings, policies, gates, and answers remain ordinary graph nodes linked to them. ACR supplies the
domain projection and human-readable view; it does not replace ContextGraph analytics with a
parallel graph implementation.

Semantica similarity returns candidates for human comparison, not proof that two situations are
equivalent. Policy impact returns historically bound decisions to re-audit, not proof that their
answers would change under a new policy version.

## Human audit order

A useful review reads chronologically:

1. Was the note inventory complete enough?
2. Were the search terms and source families reasonable?
3. Did the agent open all material candidates?
4. Was each note's standing judged correctly?
5. Were conflicts and dates resolved under the right clauses?
6. Was stopping justified, and does the submitted answer follow?

Each step must show the question, choice, reason, policy/reference status, remaining model
judgment, review attention, and a drill-down link to raw Langtrace.

## Scope boundary

The maintained code is limited to chart corpus/spec contracts, Codex chart-review execution,
enhanced Langtrace capture/replay, Decision Episode reconstruction and verification, Semantica
projection/queries, and the human review UI/notebook. Legacy labelling, separate audit-agent,
evaluation, improvement, authoring, and alternate runtime stacks are intentionally out of scope.
