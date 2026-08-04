# From AgentLoop to ACR: running, evaluating and continuously improving a medical chart-review agent

> Status: methodology baseline  
> Document date: 2026-07-29  
> Applies to: `agentic-chart-review`  
> Source scope: full coverage of AgentLoop's product capabilities and user guides; the
> Alibaba-Cloud-only onboarding steps, OpenAPI operations, RAM fields and billing details are
> indexed only and are not ACR design requirements.

The stable contracts at the implementation level, the Module/Skill/Capability/Stage boundaries,
the independent Audit plane and the compatibility migration are in
[`ACR_MODULE_ARCHITECTURE_V2.md`](ACR_MODULE_ARCHITECTURE_V2.md).

## 0. Conclusion first

ACR must not call everything that "checks the agent" an eval. A complete system contains at
least the following seven layers:

| Layer | When it runs | Core question | Can it affect the current answer |
|---|---|---|---|
| Task contract | frozen before the run | what is the task asking, and what evidence and answers count | Yes, it defines the answer boundary |
| In-request controls | inside the current request | is this answer allowed to be submitted | Yes, it can reject, abstain or route to a human |
| Runtime policies | inside the current request | how the agent searches, expands and allocates compute | Yes, but they are policies awaiting validation, not automatically correct |
| Observability | recorded during the current request | what actually happened | No, it only records facts |
| Audit | preflight during the run, or correlated after it | did a safety, privacy or authority violation occur | Preflight can block; a post-hoc incident does not rewrite an answer that already exists |
| Evaluation & attribution | after the trace completes | how good are the result and the process, and why did it fail | No, it only scores, attributes and routes |
| Experiment & optimization | development / before release | does a given change genuinely improve quality | It only decides whether the next version ships |

The distinctions that matter most:

1. **A requirement is not an evaluator.** Patient isolation, evidence admissibility, field format
   and the proof obligation on a negative answer are contracts the system executes.
2. **A policy is not a requirement.** Which terms to search, how many documents to read, whether
   to run multiple agents, whether to "see everything" — these are policies with a cost, and they
   must earn the right to keep existing through experiment.
3. **Evaluator type is not evaluator category.** `CODE / LLM / AGENT / HUMAN` describe "what
   executes the evaluation"; safety, correctness, evidence, process, efficiency and attribution
   describe "what is being evaluated".
4. **A gate pass is not a correct answer.** A gate can only prove that the obligations the system
   defined were executed. It cannot prove that the search terms, the evidence universe and the
   clinical rules are themselves right.
5. **An eval signal does not modify the agent directly.** It first becomes an evidenced failure
   event and a repair obligation, then modifies one asset that has an owner, and then
   passes paired validation.

---

## 1. What AgentLoop actually is

AgentLoop's product through-line is not Agent-as-a-Judge on its own. It is:

```text
Observability / Trace
        ↓
Audit + Evaluation
        ↓
Dataset / Annotation / Bad Cases
        ↓
Experiment
        ↓
Prompt / Skill / Context assets
        ↓
Production traces
        ↺
```

The official documentation treats the complete trajectory as the base data unit: the same
trajectory serves observability, audit, evaluation, experiment, data processing and experience
mining at once. The hands-on order in the official QuickStart is create an AgentSpace, connect
observability, build an evaluator, create an evaluation task, import and annotate a dataset,
manage Prompt/Skill, use Memory, and finally close the data flywheel.

What is useful to ACR is this **closed-loop division of labour**, not the specific implementation
in the Alibaba Cloud console, SLS or eBPF.

### 1.1 Capability map and what ACR takes

| AgentLoop capability | What it does officially | ACR decision |
|---|---|---|
| AgentSpace | resource, permission and data boundary for a team or business domain | Adopt with changes: local workspace + task bundle + local artifact root |
| Observability | trace, metrics and topology for Agent/Tool/Model/Retriever/Memory | Adopt the idea directly: one unified trace ontology |
| Audit | raw facts → finding → incident → investigation | Adopt the two-level model directly; the patient boundary is additionally pulled forward into a runtime control |
| Evaluation | Evaluator + Evaluation Task + Analysis | Adopt the three-part structure directly |
| Agent-as-a-Judge | the evaluator carries its own Prompt, Skill and tools and reads the trajectory | Adopt with changes: same patient, read-only, capability broker, may not confirm clinical truth |
| Experiment | repeatable, comparable tests over models, Prompts, Agents and tool configuration | Adopt directly; add chart-observable gold and paired patient comparison |
| Dataset | carries traces, annotations, gold, bad cases and experiment cases | Adopt with changes: patient data is referenced locally outside the repo only, never copied into a shared library |
| Pipeline | filter, dedupe, sample, cluster, AI processing, write into a dataset | Adopt the idea directly: a local append-only event pipeline |
| Annotation | templated human annotation | Adopt with changes: role-based adjudication by registrar/clinician/engineer |
| Prompt/Skill assets | version, diff, label, staged rollout and runtime loading | Adopt directly; extend to specs, tools, runtime policies and evaluators |
| Experience Library | distil reusable "ways of doing things" from successful and failed trajectories | Adopt later: only PHI-free, validated method assets are allowed |
| Memory | persist long-term context about a user or an environment | Not used for patient charts by default; cross-patient memory is forbidden |
| OpenAPI/RAM/Billing | cloud product integration, permissions and cost | Indexed only, no Alibaba Cloud integration |

### 1.2 A few concepts in AgentLoop that are easy to confuse

#### Trace and Trajectory

- A trace is the call-chain fact of one request: model, tools, retrieval, inputs and outputs,
  latency and tokens.
- A trajectory is the complete representation of an agent execution, in a form suited to later
  evaluation and experience mining.
- ACR does not need to store unverifiable long-form CoT. It needs to store structured events,
  model/tool I/O, the evidence ledger, plan revisions, gate decisions, termination and hash
  lineage.

#### Audit and Evaluation

- AgentLoop Audit first provides neutral Audit Facts: Session, Token, Tool and behavioural facts.
  Only Risk Audit correlates a safety or boundary signal from a low-fidelity `Finding` into a
  high-fidelity `Incident`.
- ACR v1 adopts application-level Risk Audit and puts the full Audit Facts platform, eBPF and
  entity investigation out of scope. Audit remains an independent governance plane, not another
  name for a CODE evaluator.
- Evaluation is about quality: correctness, evidence, task completion, tool selection, execution
  efficiency and so on.
- "PHI appeared in a request to an external provider" is an incident; "the answer cited the wrong
  pathology report" is a quality error. Both can happen in the same trajectory, but they cannot be
  expressed as the same kind of conclusion.

#### The three components of Evaluation

1. **Evaluator**: one versioned evaluation standard.
2. **Evaluation Task**: binds the data source, sampling, variable mapping, evaluator and run
   policy.
3. **Analysis**: aggregates trends, filters bad cases, drills down into a single trajectory.

An evaluator cannot simultaneously double as task orchestration, budget, selector and
certification. ACR v2 uses `ModuleAsset` to define an evaluator, `PipelineProfile` to define
conditions and dependencies, `EvaluationTask` to bind data and permissions, and
`CertificationSuite` to hold certification standards separately. The old `EvaluatorSpec` survives
only as a compatibility adapter.

#### Experience and Memory

- Experience answers "how is this kind of task usually done better" — for example, look at the
  definitive resection first, then handle biopsy/resection precedence.
- Memory answers "what has been true of this user or this environment in the past".
- A patient chart is not Memory. ACR must not write patient content into memory that is
  recallable across runs or across patients.
- ACR Experience may store only a general method, its applicability conditions, the source
  trajectory hashes, the validation metrics and a version. It may not store patient answers,
  source text or unvalidated "experience".

#### AgentLoop's "online" does not mean inside the same request

- "Continuous/online evaluation" means evaluation is triggered automatically once a new trace
  appears, usually nearline post-run.
- "Online experiment" in the Experiment documentation means the AgentLoop server side calls the
  target endpoint; "offline" means the run is started from the user's own machine. It is not a
  synonym for a runtime gate.
- ACR consistently uses `IN_REQUEST`, `POST_RUN_CONTINUOUS` and `OFFLINE_DEVELOPMENT`, and avoids
  writing only real-time/online.

---

## 2. ACR's standard layering

### 2.1 Task contract

A task contract is domain semantics. It is not navigation instructions for the agent, and it is
not an evaluator.

Every task package should freeze:

- the fields, their value domain, statuses and the output schema;
- patient/entity/time scope;
- evidence eligibility and source precedence;
- what a positive witness must be;
- the observable scope and the proof obligation for a negative/absence claim;
- the abstention boundary;
- conflict rules and the boundaries where a human must decide;
- the source, version, owner and sign-off for every semantic element.

What gets reused across domains is not one giant spec, it is the same contract schema. A cancer
registry, medication use, recurrence and a cardiovascular phenotype may each have a different
spec, but they share one execution and evaluation framework.

### 2.2 In-request controls: the hard controls inside the current request

A runtime control must satisfy at least one of these conditions:

- it is a non-negotiable safety/authority boundary;
- an output that does not satisfy it is logically invalid;
- it is a cheap, deterministic, replayable check;
- there is a defined reject, abstain or human-review behaviour on failure.

The controls ACR should keep:

| Control | Why it belongs in the runtime | Behaviour on failure |
|---|---|---|
| Current patient scope | cross-patient evidence makes the answer invalid | refuse the tool call immediately |
| Provider/PHI boundary | protection is only effective before the data is sent | preflight block |
| Tool allowlist/read-only | an unrecorded read corrupts the evidence ledger | refuse the call |
| Spec, policy and tool version freeze | an answer that cannot be replayed cannot be compared | refuse to start, or mark as unvalidated |
| Evidence admissibility | FOUND must have an eligible witness | refuse the submission |
| Field format/value checks | a structurally invalid answer must not leave the system | refuse the submission |
| Negative proof obligation | "absent / not enough evidence" is a global claim | abstain/review while it is not closed |
| Discovered conflict/open thread | a contradiction the chart itself exposed cannot be ignored | keep investigating, or route to human review |
| Hard budget | prevents unbounded loops and unbounded spend | `BUDGET_EXHAUSTED`/review |

A runtime control is not responsible for judging that the whole clinical answer is correct, and it
should not contain an experimentally unvalidated "best search path".

### 2.3 Runtime policies: run-time performance policies

A policy decides how the agent fulfils the contract:

- initial search terms and document-type routing;
- the order documents are read in;
- sampling frame and sample size;
- plan expansion;
- conflict refinement;
- the trigger conditions for multiple trajectories;
- model, temperature, context management;
- experience retrieval;
- how cost is divided between searching, reading and confirming.

A policy may affect the current run, but it must not be promoted to a requirement just because it
"looks thorough". Every policy must:

1. have a stable `policy_id/version/hash`;
2. record the failure mode it is expected to improve;
3. be replaceable by a feature flag or a profile;
4. be run in a paired experiment under the same task contract;
5. be rollback-able on failure, without modifying task semantics.

### 2.4 Observability: the neutral fact layer

Observability answers only "what happened":

- run/session/trace/span IDs;
- model/tool/retrieval/gate/output events;
- input and output hashes, versions, latency, tokens, cost;
- documents listed/searched/read;
- the evidence and coverage ledger;
- plan revisions, rejections, retries, termination;
- provider boundary and patient scope.

A trace field must not be deleted because one evaluator temporarily does not need it; the same
trace still has to serve audit, evaluation, experiment and attribution later.

### 2.5 Audit: the safety evidence chain

ACR adopts AgentLoop's two-level model:

```text
Raw facts
  → Finding (high recall, single-point signal)
  → correlation
  → Incident (evidence chain closed, actionable)
  → investigation / disposition
```

Examples:

- text that looks like a name is detected in a log: Finding;
- that name enters an external provider request along with chart content: Incident;
- another patient ID appears in a tool argument: Incident;
- a local path outside the repo contains patient content but has not crossed a declared boundary:
  needs investigation, and does not automatically amount to a leak.

Two lines of defence should coexist:

- `IN_REQUEST`: preflight on patient scope, provider trust boundary, tool allowlist and so on;
- `POST_RUN_CONTINUOUS`: correlate, from the complete trace, the PHI/data-flow incidents that were
  missed.

The current implementation uses application events only; it does not claim to detect a
process/file/network side effect that the application did not report. `RuntimeEvidenceRef` is a
boundary for a future adapter, not an eBPF capability that already exists.

### 2.6 Evaluation: independent quality evaluation

#### Two orthogonal classification axes

"How to evaluate" and "what to evaluate" must be kept apart.

**Execution mode:**

| Type | Capability | Authority |
|---|---|---|
| CODE | deterministic, replayable, cheap | may create an incident or block a version from shipping |
| LLM | single turn, no tools, explicit rubric | screening and ranking only |
| AGENT | reads the trajectory, calls declared tools, minimal supplementary reads | investigation, attribution and routing only |
| HUMAN | clinical/registry/spec semantic adjudication | confirms gold, approves a semantic change |

The native types in the AgentLoop API are `CODE / LLM / AGENT`; `HUMAN` is an authority plane ACR
adds for medical accountability and adjudication.

**Evaluation domain:**

| Category | What it evaluates | Typical implementation |
|---|---|---|
| Safety & boundary | PHI, cross-patient, egress, authority | CODE audit + human investigation |
| Outcome & abstention | is the value/status correct, is it over-asserted | CODE against gold; HUMAN adjudicates |
| Evidence & grounding | quote, document standing, source precedence, unread contradictions | CODE + AGENT |
| Process & control integrity | are gate/ledger/termination consistent, are tool/plan behaviours anomalous | CODE |
| Reliability & efficiency | provider failure, retry, cost, latency, documents read | CODE |
| Causal attribution | which defect explains the target error, which class of asset should change | AGENT + HUMAN |

`Agent-as-a-Judge` is an execution mode, not a seventh evaluation domain.

#### CODE first

When a deterministic answer to the same sub-question exists, LLM/AGENT must not re-judge it:

- whether a quote offset reproduces: CODE;
- field exact match: CODE;
- whether a read crossed patients: CODE;
- whether a passage of clinical text genuinely supports a given code: may need AGENT/HUMAN;
- whether the registry value is the truth of the current chart: HUMAN.

#### Eval must not affect an answer that is already finished

A post-run evaluator may:

- flag the run;
- create an incident;
- queue the case for human review;
- block a candidate bundle from shipping;
- create a bad-case/repair obligation.

It may not quietly rewrite the original run's answer, or write its own conjecture down as
confirmed truth.

### 2.7 Causal attribution: from finding a defect to explaining the target error

Finding a real defect does not mean it explains the target error. Attribution must be bound to an
explicit target event:

```text
Target event
  → reconstruct retrieval → evidence → interpretation → coding → gate → output
  → enumerate rival causes
  → smallest discriminating probe
  → safe counterfactual replay
  → skeptic review
  → attribution gate
```

Every cause should be labelled:

- `relation_to_target = EXPLAINS | CONTRIBUTES | UNRELATED_DEFECT | UNKNOWN`
- `causal_strength = OBSERVED | PLAUSIBLE | COUNTERFACTUAL_SUPPORTED | HUMAN_CONFIRMED`

Only `EXPLAINS + COUNTERFACTUAL_SUPPORTED` can become a strong model-side attribution; clinical
content still needs human confirmation.

### 2.8 Experiment and release

Evaluation describes current performance; Experiment answers "did a given change cause an
improvement".

A valid experiment must hold fixed:

- the patient set and the chart snapshot;
- the task contract/spec hash;
- the model/provider and the seed set;
- the budget;
- the evaluator bundle;
- the truth/adjudication version.

and must change only the assets declared in advance:

- the spec;
- the retrieval/runtime policy;
- the Prompt/Skill;
- a tool/check;
- the model;
- the experience bundle.

Results must be compared case by case, reporting accuracy, abstention, overclaim, evidence
validity, subgroup, cost, latency, calls and documents read together. A rise in the mean must not
cover up a critical regression.

---

## 3. Coverage: it has to be split into four things

"Require the agent to see everything" mixes four different concepts together.

### 3.1 Claim obligation: what proof the answer owes

This is the task contract.

- For a positive `FOUND`: one eligible witness is usually enough.
- For a negative, an absence or `EVIDENCE_INSUFFICIENT`: the observed scope must be declared, and
  it must be shown that no eligible witness exists within that scope, or which evidence gaps
  prevent a conclusion must be named.
- For source precedence, multiple tumors/entities or multiple time points: even a positive answer
  may have to prove that no higher-precedence source exists, or that the conflict is closed.

### 3.2 Acquisition policy: how the proof is obtained

This is a replaceable runtime policy:

- exhaustive read;
- keyword search;
- document-type routing;
- sample misses;
- time-window coverage;
- conflict-triggered expansion.

They may raise recall, and they may also raise cost, create a rejection loop, or cause a wrong
abstention.

### 3.3 Enforcement: when an answer is refused

This is a runtime control:

- the positive witness is not eligible: refuse `FOUND`;
- the negative proof is not closed: no strong negative conclusion may be emitted;
- a discovered conflict/open thread is not closed: keep investigating or route to a human;
- the search policy has no viable expansion left: return an explicit gap rather than manufacturing
  a consensus.

### 3.4 Coverage evaluation: does this approach work

This is post-run evaluation/experiment:

- is the gate consistent with the ledger;
- did it miss a witness that was found later;
- did it lower critical overclaim;
- did it raise chart-observable exact match;
- did it raise abstention, loops, cost and subgroup regression.

### 3.5 ACR's default coverage position

1. "Read every document" is not adopted as a general requirement across tasks.
2. A proof obligation is kept only for negative/absence claims and for tasks that genuinely need
   global precedence.
3. A positive answer is allowed to stop once the witness is eligible and discovered conflicts are
   closed.
4. The agent may decide its own navigation path; the runtime may sample misses independently, so
   that the agent cannot prove its own search was sufficient using documents it selected itself.
5. The current stratified coverage, keywords, sample size and thresholds are all engineering
   assumptions; they cannot be claimed to improve performance until they are validated on real
   chart-observable gold.

### 3.6 Coverage ablation

Four arms are proposed:

| Arm | Behaviour |
|---|---|
| A. Witness-only | stop after a positive witness; a negative states insufficiency directly, with no broad closure |
| B. Negative-proof | closure only starts when a negative conclusion is about to be submitted |
| C. Current stratified | the current stratified search + forced sampling |
| D. Adaptive | widen the search only on conflict, entity/time ambiguity or gate risk |

Until chart-observable adjudication of the first round of 10 real cases is complete, only the
following can be compared:

- calls, cost, documents read, latency;
- rejection loops, provider/runtime degradation;
- evidence admissibility, gate consistency;
- the change in registry disagreement.

Only fields a human has confirmed are derivable from the current chart may enter conclusions about
accuracy, overclaim and causal improvement. Ten cases are still an exploratory pilot and cannot
approve a global policy on their own.

---

## 4. How an eval signal improves the system

Eval output must be routed to a named asset owner, not attributed uniformly to "the model is not
good enough".

| Observed signal | The first distinction to make | Allowed repair target |
|---|---|---|
| Gold witness was never surfaced | wrong search terms, or a class missing from the evidence universe | retrieval asset/runtime policy |
| Witness was read but not used | source standing, semantic understanding, or entity/time | Skill/Prompt; the spec form if necessary |
| The correct answer was rejected by the gate | the task obligation is wrong, or the control implementation is wrong | proof contract or deterministic control |
| A wrong answer passed the gate | the gate missed it, or the task semantics are wrong | answer check/control or spec content |
| Registry disagreement | is the registry chart-derivable | human adjudication; no automatic patch |
| Runs disagree with each other | value, evidence, entity, time, or provider | targeted attribution/experiment |
| `SPEC_INSUFFICIENT` loop | a real spec gap, or a coverage dead-end that was misrouted | gate/termination routing |
| PHI/cross-patient incident | is a preflight missing | runtime safety control |
| Retries, timeouts, cost spikes | provider, tool, policy, or prompt | runtime/tool/model/policy |
| One successful path is repeatedly effective | does it hold across cases, does it leak PHI | certified experience asset |

### 4.1 Truth mode caps what can be concluded

| Mode | Signals available | Conclusions allowed |
|---|---|---|
| BLIND | trace, spec, detector, repeated runs, chart probe | anomaly, hypothesis, test obligation |
| REGISTRY_REFERENCE | unadjudicated registry values | disagreement, adjudication obligation |
| GOLD | chart-observable gold and witness | confirmed mismatch, contrastive failure packet |
| HUMAN | role-based signed adjudication | confirmed gold, semantic approval, disposition |

Without gold you can still improve safety, runtime reliability, gate consistency, evidence
admissibility and cost; you cannot treat a clinical correctness hypothesis as truth.

### 4.2 The loop for one repair

```text
Trace
  → deterministic screening
  → selected LLM/AGENT evaluation
  → human adjudication when required
  → deterministic bad-case cluster
  → one target event + one repair obligation
  → change one versioned asset
  → paired validation
  → sealed certification
  → canary/shadow
  → production monitoring
```

Forbidden:

- an evaluator modifying a production spec directly;
- a majority vote standing in for the evidence gate;
- model confidence standing in for human confirmation;
- teaching the agent to guess outside-chart information in order to match the registry;
- tuning endlessly on the same batch of diagnosis cases and then calling it a sealed test.

---

## 5. The framework development phase and the task usage phase

### 5.1 What the framework development phase should deliver

These capabilities are shared by every chart-review task:

- one unified trace ontology and an immutable run manifest;
- the patient/provider/tool capability broker;
- independent hashes for the task contract, the runtime controls and the runtime policy;
- the `CODE / LLM / AGENT / HUMAN` runners and their authority boundaries;
- audit `Finding → Incident` correlation;
- the evaluator registry, the task runner, the analysis/event store;
- truth-mode isolation;
- the attribution target/counterfactual/skeptic gate;
- experiment, paired comparison, release gate;
- asset lineage: spec, Prompt, Skill, Tool, model, policy, evaluator, experience;
- local-only PHI storage and an append-only bad-case library;
- certification and drift monitoring of the evaluators themselves.

The framework must not hard-code lung cancer, STORE, or the clinical knowledge of one department.

### 5.2 Done when each new task is onboarded

A task package contains at least:

- the extraction spec / field contract;
- the evidence source policy and precedence;
- the entity/time model;
- the positive/negative proof obligation;
- retrieval assets;
- task-specific skills;
- task-specific deterministic checks;
- the evaluator profile;
- the gold/registry/blind data policy;
- subgroup and critical-error definitions;
- the runtime policy experiment;
- a human owner and sign-off.

### 5.3 Used when a task runs in production

Each case runs, by default, only:

- the frozen task contract;
- the cheapest safe runtime policy;
- the in-request controls;
- a complete trace;
- a cheap CODE audit.

Only anomalous cases trigger:

- conflict refinement;
- the unread-evidence evaluator;
- causal attribution;
- human review.

The production RUN plane carries no gold and no registry reference.

### 5.4 Used when a task is continuously improved

- Continuous post-run CODE evaluation: every trace, or a high proportion of them.
- LLM/AGENT evaluation: anomaly selection and cost-controlled sampling.
- Human adjudication: registry disagreement, clinical semantics and semantic patches.
- Periodic experiment: new model/spec/policy/skill/tool/evaluator/experience versions.
- Bad-case cluster: clustered by target/cause/parameter, not by free-text summary.

---

## 6. The boundary of what ACR adopts from AgentLoop

### 6.1 Adopted directly

- trajectory-first observability;
- the Evaluator / Evaluation Task / Analysis division of labour;
- CODE/LLM/AGENT evaluators;
- the two-level audit finding/incident model;
- the trace → dataset → annotation → experiment → asset data flywheel;
- version, diff, label and staged rollout for Prompt/Skill;
- experience is a method, not an answer.

### 6.2 Adopted with changes

- `HUMAN` as an independent authority plane;
- patient-scoped chart tools and a PHI-local artifact boundary;
- chart-observable gold, rather than following the registry blindly;
- a deterministic check takes precedence on the same sub-question;
- an agent evaluator may only screen and route;
- a semantic patch requires human sign-off;
- a dataset stores only local references and hashes, never a copy of the real chart;
- Memory is disabled by default; Experience must be PHI-free, validated and revocable.

### 6.3 Not adopted

- patient data is not sent to Alibaba Cloud;
- nothing depends on SLS, ARMS, MSE or eBPF in order to run;
- large volumes of free-text CoT are not stored;
- an evaluator is not allowed to push its own changes live;
- cross-patient memory is not allowed;
- online scoring is not treated as a clinical gate for the current request;
- reading the whole chart is not forced because the process "looks complete".

---

## 7. What this means concretely for the current repository

This section used to list, module by module, what each flat module should be called (`agent.py`:
runtime orchestration, `answer_gate.py`: in-request controls, and so on), and to point out that
README's "audit layer — this is the product" conflated three things. That list **has been
replaced by the directory structure itself**: `src/acr/` is now split into ten packages by plane,
the directory a module sits in is which layer it belongs to, and "who it may depend on" is
asserted by `tests/test_layering.py`.

To read this section now, look at the table in README §2.0: plane -> `src/acr/<dir>` ->
`assets/<dir>` -> the question it answers. A naming list that a person has to diff against the
code by hand is exactly the kind of thing it was written to correct.

## 8. Coverage matrix for the official documentation

Status legend:

- **Deep read**: already used in this methodology.
- **Indexed**: the capability boundary is confirmed, but the cloud-platform operational detail is
  not unpacked.
- **Adopted**: ACR adopts the core idea directly.
- **Adapted**: the idea is adopted, with medical, PHI or local boundaries added.
- **Not integrated**: its existence is recorded, but ACR does not connect to that cloud service.

### 8.1 Product, concepts and quick start

| Document | Status | Use in ACR |
|---|---|---|
| [What is AgentLoop](https://help.aliyun.com/en/document_detail/3033860.html) | Deep read / adapted | the overall closed loop and trajectory-first |
| [Core concepts](https://help.aliyun.com/zh/document_detail/3042001.html) | Deep read / adopted | AgentSpace, Trajectory, Dataset, Pipeline, Evaluation, Experience, Memory |
| [QuickStart end-to-end](https://help.aliyun.com/en/document_detail/3033823.html) | Deep read / adapted | the order of the closed loop and the relationships between assets |
| [Billing](https://help.aliyun.com/zh/document_detail/3044490.html) | Indexed / not integrated | evidence that evaluation, experiment and data each carry their own cost |
| [RAM permission reference](https://help.aliyun.com/en/document_detail/3033852.html) | Indexed / adapted | the least-privilege idea |
| [OpenAPI operations](https://help.aliyun.com/en/document_detail/3041792.html) | Indexed / not integrated | the API capability boundary |

### 8.2 Observability

| Document | Status | Use in ACR |
|---|---|---|
| [AI agent observability](https://help.aliyun.com/en/document_detail/3042586.html) | Deep read / adopted | capability map |
| [Trace](https://help.aliyun.com/en/document_detail/3042591.html) | Deep read / adopted | request/span structure, replay |
| [Session analysis](https://help.aliyun.com/en/cms/cloudmonitor-2-0/conversational-analysis-of-ai-agent) | Deep read / adapted | run/session aggregation, cost and error drill-down |
| [Scenario-based analysis](https://help.aliyun.com/zh/document_detail/3042597.html) | Indexed / adapted | multi-dimensional trend analysis |
| [Connecting an AI Agent to application monitoring](https://help.aliyun.com/zh/document_detail/3046111.html) | Indexed / not integrated | the probe and OpenTelemetry onboarding idea |

The per-framework and per-product onboarding pages (LangChain, Dify, OpenAI, AgentScope, Hermes,
Coding Agents and so on) are all filed under the Access Center index; they change how data is
collected, not the ACR methodology.

### 8.3 Audit

| Document | Status | Use in ACR |
|---|---|---|
| [AI Agent audit](https://help.aliyun.com/zh/document_detail/3045691.html) | Deep read / adopted | the overall audit layering |
| [Audit onboarding guide](https://help.aliyun.com/zh/document_detail/3045692.html) | Deep read / adapted | application facts + runtime facts |
| [Audit management](https://help.aliyun.com/zh/document_detail/3045693.html) | Deep read / adapted | independent switches, continuous tasks |
| [Audit rule reference](https://help.aliyun.com/zh/document_detail/3045694.html) | Deep read / adopted | Finding → Incident |
| [Risk event field reference](https://help.aliyun.com/zh/document_detail/3045695.html) | Indexed / adapted | incident schema |
| [Risk audit](https://help.aliyun.com/zh/document_detail/3045696.html) | Deep read / adapted | risk ranking and the entry point for investigation |
| [Entity investigation](https://help.aliyun.com/zh/document_detail/3045697.html) | Deep read / adapted | patient/provider/tool/entity correlation |
| [Audit facts](https://help.aliyun.com/zh/document_detail/3045698.html) | Deep read / adopted | a neutral fact is not a risk |

### 8.4 Evaluation

| Document | Status | Use in ACR |
|---|---|---|
| [Evaluation overview](https://help.aliyun.com/zh/document_detail/3042179.html) | Deep read / adopted | Evaluator/Task/Analysis |
| [Evaluators](https://help.aliyun.com/zh/document_detail/3042180.html) | Deep read / adapted | built-in dimensions, Prompt, Skill, MCP |
| [Evaluation tasks](https://help.aliyun.com/zh/document_detail/3042181.html) | Deep read / adopted | trace/log/dataset, sampling, continuous/historical |
| [Evaluator API schema](https://help.aliyun.com/zh/document_detail/3045378.html) | Deep read / adapted | CODE/LLM/AGENT, variable mapping, versions |

### 8.5 Experiment

| Document | Status | Use in ACR |
|---|---|---|
| [Experiment operations guide](https://help.aliyun.com/zh/document_detail/3046601.html) | Deep read / adapted | LLM/Agent experiments, the online/offline definitions, trajectory variables |

Offline result reporting, service registration and the console operation pages are filed under the
index; ACR uses local paired experiments and does not report patient experiment results to
AgentLoop.

### 8.6 Data Center, Pipeline and Annotation

| Document | Status | Use in ACR |
|---|---|---|
| [Dataset overview](https://help.aliyun.com/zh/document_detail/3042278.html) | Deep read / adapted | structured data assets |
| [Dataset console onboarding](https://help.aliyun.com/zh/document_detail/3042280.html) | Deep read / adapted | trace/CSV/manual sources |
| [Data annotation](https://help.aliyun.com/zh/document_detail/3041820.html) | Deep read / adapted | the human adjudication UI/schema idea |
| [Pipeline user guide](https://help.aliyun.com/zh/cms/cloudmonitor-2-0/user-guide-for-agentloop-pipeline) | Deep read / adopted | cleaning, three-level dedupe, cluster sampling, AI processing |
| [Pipeline reference](https://help.aliyun.com/zh/cms/cloudmonitor-2-0/product-feature-documentation) | Indexed / not integrated | operators/API/limits |

### 8.7 Agent Assets

| Document | Status | Use in ACR |
|---|---|---|
| [Agent asset overview](https://help.aliyun.com/zh/document_detail/3041729.html) | Deep read / adopted | independent versions, status, labels |
| [Prompt management](https://help.aliyun.com/zh/document_detail/3041730.html) | Deep read / adapted | draft/publish/diff/label/dynamic loading |
| [Skill management](https://help.aliyun.com/zh/document_detail/3041731.html) | Deep read / adopted | SKILL.md, import, version, labels |

### 8.8 Experience and Memory

| Document | Status | Use in ACR |
|---|---|---|
| [Experience Library product introduction](https://help.aliyun.com/zh/document_detail/3047255.html) | Deep read / adapted | a method asset, not an answer |
| [Experience Library user guide](https://help.aliyun.com/zh/document_detail/3047254.html) | Deep read / adapted | trajectory mining, continuous mining, runtime Skill recall |
| [Memory module console onboarding](https://help.aliyun.com/zh/cms/cloudmonitor-2-0/memory-module-console-operating-guidelines) | Deep read / not adopted by default | Facts/Episodic/Summary, retrieval and lifecycle |
| [Memory integration in QuickStart](https://help.aliyun.com/en/document_detail/3033823.html) | Deep read / not adopted by default | the Mem0 interface and long-term/short-term memory |

---

## 9. Methodology acceptance questions

Before any new ACR mechanism is added, answer these first:

1. Does it belong to task contract, control, policy, observability, audit, evaluation, experiment
   or experience?
2. Does it run at `IN_REQUEST`, `POST_RUN_CONTINUOUS` or `OFFLINE_DEVELOPMENT`?
3. On what authority may it block the current answer?
4. What dimension does it evaluate, and what executes the evaluation?
5. Does it depend on gold; without gold, what is the cap on its conclusions?
6. Which asset does it modify once it finds a signal, and who owns that asset?
7. How does it prove its benefit through a paired experiment?
8. Could it copy PHI, cross patients, leak the answer key, or accumulate unvalidated "experience"?
9. What is its stopping condition?
10. If it is removed, which proven metric degrades?

If there is no evidence for question 10, the mechanism should be labelled `EXPERIMENTAL_POLICY`;
it cannot become a permanent hard requirement just because it "looks thoroughly thought through".
