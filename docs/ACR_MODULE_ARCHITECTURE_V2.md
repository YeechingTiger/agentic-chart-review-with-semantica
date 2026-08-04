# ACR module architecture v2

> Status: implemented compatible-evolution baseline  
> Date: 2026-07-29

## 1. Design principles

Only a unit that can run on its own and that has typed input/output, authority, a stopping
condition and a certification lifecycle is a module.

| Name | Meaning | Top-level module? |
|---|---|---|
| Module | A capability that can be run and certified independently | yes |
| Skill | Method guidance internal to an agent evaluator | no |
| Capability | A tool power bounded by scope and budget | no |
| Stage | An execution phase inside one module | no |
| Pipeline | The conditions, dependencies and data bindings over modules | not a module |
| Task | The data, truth, budget and authority bindings of one run | not a module |

The system uses four planes instead of putting every check into Eval:

```text
Execution
  → canonical Trajectory
  → Audit              behavioural facts and governance boundaries
  → Evaluation         task quality and causal attribution
  → Improvement        repair, paired validation, adoption
```

The existing deepagents extraction stays where it is. The canonical trajectory is built directly
from the current trace/manifest; the old EvalLoop, the duplicate evaluator catalog and the
compatibility CLI have been removed.

## 2. Stable Kernel

`acr.kernel` defines the shared objects that stay stable across tasks:

- `AssetRef`: the content-addressed reference for every spec, policy, prompt, skill, tool,
  evaluator, audit rule and repair strategy.
- `Trajectory`: the immutable, analysis-ready record of one complete agent execution.
- `TargetRef`: the explicit target a signal is about — run, field, evidence, tool, gate.
- `SignalEnvelope`: the thin envelope shared by Audit, Evaluation, Attribution and Repair output.

`TrajectoryAdapter` produces the canonical trajectory from the extraction manifest/trace. Chart
text stays in local artifacts outside the repo; free text in a trajectory event keeps only a hash
and a length, and analysis modules read the original local reference under an explicit grant.

## 3. Module Families

`acr.modules` defines only five kinds of module protocol:

1. `RuntimePolicy`: search, coverage, expansion and stopping policy.
2. `AuditRule`: truth-blind behavioural boundary checks.
3. `Evaluator`: CODE/LLM/AGENT/HUMAN quality evaluation.

> 2026-08-03: this document used to list five kinds. The other two were `RuntimeControl`
> (in-request allow/deny/require) and `RepairStrategy` (signal → repair obligation routing). The
> single implementation of each had zero references in production code, so both were deleted along
> with their protocols. The `acr.repair_loop` guarantees described in section 8 therefore never
> took effect on any live path.

YAML may refer only to an explicitly registered `implementation_id`. The system does not allow
dynamic import of arbitrary Python code from YAML.

## 4. Asset, Pipeline, Task and Certification

The old `EvaluatorSpec` held the evaluator, the selector, dependencies, the budget and synthetic
fixtures all at once. v2 splits it apart:

- `ModuleAsset`: module identity, I/O, runner, truth modes, capability requests and maximum
  authority.
- `PipelineProfile`: nodes, conditions, dependencies, input bindings, the capability allowlist, the
  budget ceiling and the authority ceiling.
- `EvaluationTask`: trajectory cohort, truth mode, model, seed, the actual budget and the grants.
- `CertificationSuite`: must-pass/must-fail fixtures, the calibration cohort and thresholds.

A task's effective capability is:

```text
module requested
∩ pipeline allowed
∩ task granted
∩ current patient scope
```

A task can only narrow capability, budget and authority, never widen them.

## 5. Audit and Evaluation

### Audit

AgentLoop's Audit has two parts:

- Audit Facts: session, turn, tool, token and behavioural facts.
- Risk Audit: Finding → correlation → Incident → investigation.

ACR v1 implements only the application-level Risk Audit:

- patient boundary
- PHI/provider boundary
- undeclared tool
- local artifact boundary
- trajectory integrity
- hard runtime-control conformance

Audit does not receive a `TruthContext`, does not judge clinical correctness, and does not produce
semantic spec repair. Audit output is stored separately in:

```text
<local-root>/audit/findings.jsonl
<local-root>/audit/incidents.jsonl
```

eBPF, process/file/network collection, entity graphs, real-time alerting and SIEM are out of scope
for v1.

### Evaluation

The `EvaluationContext` of `acr.evaluation_pipeline` keeps ordinary typed channels separate from
`TruthContext`. A BLIND channel recursively rejects gold, answer keys and registry references.

`EvaluationResult` contains no `AuditFinding` or `AuditIncident`. A security signal may enter
attribution as a reference, but it cannot be re-adjudicated by an evaluator.

Built-in v2 CODE evaluators:

- `evidence-validity`
- `gate-effectiveness`

`causal-attribution@2.0.0` is registered as an AGENT `ModuleAsset`; the tool loop itself is still
executed by `acr attribute`. Its targeted probe covers the unread-evidence contradiction check, so
a duplicate standalone contradiction runtime is no longer maintained.

## 6. The internal modules of Attribution

What used to be `attribution_modules` in fact denotes stages inside one evaluator, not eight
independent evaluators. The formal names are `AttributionStage`, `AttributionStageProfile` and
`AttributionStageRegistry`.

```text
target framing
→ trace reconstruction
→ cause hypothesis
→ targeted probe
→ counterfactual replay
→ skeptic review
→ attribution gate
```

The whole sequence still produces exactly one `AttributionReport`.

## 7. Coverage

Coverage is split into:

- `CoveragePolicy`: how to search, which belongs to RuntimePolicy.
- `CoverageState`: what was actually read, which belongs to Trajectory.
- coverage effectiveness: whether the policy works, which belongs to Evaluation/Experiment.

`acr.runtime_profiles` provides two comparable baselines:

- `witness-first-baseline`
- `current-stratified-coverage`

They are already wired into `run_patient` and the `--runtime-profile` option of
`acr run|batch|consistency` and `acr extract`. The default is still `current-stratified-coverage`.
Every manifest and every `run_start` event stores the profile ref, its version and its content
hash.

`witness-first-baseline` still enforces patient scope, tool authority, field format, the answer
check, positive evidence and the open-thread control; it requires listing the patient's documents
first, but it runs neither forced sampling of negatives nor the stratified exclusion gate. An
`EVIDENCE_INSUFFICIENT` it accepts records `negative_basis=WITNESS_FIRST_BASELINE` explicitly,
carries no `coverage_attested`, and is not reported as a coverage-validated answer.

Any "you must read everything" performance claim has to be compared under the same patient, model,
seed and budget on accuracy, critical miss, overclaim, abstention, evidence validity, documents
read and cost.

## 8. Improvement

`acr.repair_loop` guarantees:

- An Audit Incident routes only to security/control repair.
- An Evaluation signal routes, after attribution, to the spec/retrieval/skill/gate/runtime owner.
- `REGISTRY_REFERENCE` can only produce an adjudication/clinician question.
- Semantic repair must have both GOLD and human adjudication.
- Every proposal still needs paired validation, and must not be accepted when there is a critical,
  per-case or subgroup regression.

## 9. Catalog and CLI

Separate catalogs:

```text
assets/module_catalog/
assets/pipeline_catalog/
assets/certification_catalog/
```

CLI:

- `acr audit rules|run|summarize|incidents`
- `acr evaluation modules|validate|run|batch|summarize|compare`
- the original `acr eval` stays pure CODE
- `acr attribute` is the only model-using attribution entry point

## 10. Explicitly not doing

- eBPF and full-stack host/process/network collection
- dynamic Python plugin import
- cloud console, dashboard, RBAC, SIEM
- a general-purpose data lake or vector dataset platform
- automatic Memory/Experience injection
- online automatic modification or publication of a semantic spec
- LLM-driven security blocking
