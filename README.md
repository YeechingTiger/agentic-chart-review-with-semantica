# agentic-chart-review

An agent that reads a patient chart the way a cancer registrar does, under a **frozen,
machine-checkable specification**, and cannot report an answer it has not proved it earned.

This file is the handover document. It describes what the pipeline is, what every component
takes and returns, how to run each one, and — at the end, in detail — **what is not done**.

The design question is not "can a model read a note". It is: *what has to be in the input, and
what has to be enforced in code, for two independent executors — a human abstractor and an LLM
agent — to reach the same label for the same reason?*

---

## 0. The one idea

A wrong answer and an unproved answer are different failures, and only the second can be caught
mechanically. So the system is built so that **every claim carries the evidence for itself**,
and a claim that cannot show its evidence is refused rather than shipped with a warning
attached.

Four consequences run through everything:

1. **The gate.** `submit_answer` is the only way an answer comes into existence, and it goes
   through `answer_gate.gate_answer`. A rejection is not a failure; it is the instruction for
   what to do next, and it always names the call that satisfies it.
2. **Enforced / advisory / declarative.** Rules that must hold are **code** (`answer_checks`,
   `coverage`). Rules that are judgement are **skills** the model may decline to read
   (`skills/`). Everything domain-specific is **YAML** a registrar can review (`specs/`).
   Clinical knowledge never lives in Python.
3. **Two planes.** DEVELOP may see the answer key; RUN may not. The only channel between them
   is a provenance record plus a human signature.
4. **A green gate is not a validated answer.** `provenance.reportable_as_validated` is the field
   a downstream filter must read — never `gate_validated` alone. The gate proves the search was
   done. It cannot prove the search terms were right.

**The spec states the decision boundary, not the navigation path.** How the agent finds evidence
is its own business; what counts as evidence, and what must be true before it may assert a
negative, are not. Every spec is content-hashed, and two labels are comparable only under the
same `spec_hash`.

**Two ways to not know, and they mean different things.** `EVIDENCE_INSUFFICIENT` is a claim
about the chart. `SPEC_INSUFFICIENT` is a claim about the specification — the agent's channel
for saying the rules do not cover this case. Pooling them destroys the only signal that tells
you which one to fix.

---

## 1. The pipeline

```
                    specs/*.yaml            corpus/patients/<id>/*.txt
                    (the question)          (the chart)
                          │                        │
   L0  acr ask ───────────┤                        │      route a question to a spec,
       acr extract        │                        │      or to an explicit gap list
                          ▼                        ▼
   L1-L3  ┌──────────────────────────────────────────────────┐
          │  acr extract   ← THE AGENT                        │
          │  LangChain/deepagents graph + our rules in hooks  │
          │  every read through the toolbox, every answer     │
          │  through the gate                                 │
          └──────────────────────────────────────────────────┘
                          │
                          ▼  extract.json          ← acr.extract/1
                             per patient x spec: value, status, evidence,
                             coverage ledger, provenance, spend, trace path
                          │
   L4     acr concord ────┤  + guidelines/*.yaml   a rule engine, NO model
                          ▼  concord.json          ← acr.concord/1
                             per recommendation: CONCORDANT / NON_CONCORDANT /
                             INDETERMINATE, and which variable decided it
                          │
   L5     acr explain ────┤                        a rule layer, NO model
                          ▼  explain.json          ← acr.explain/1
                             for each selected case, the causes the ledger
                             ELIMINATES — never a cause it invents
```

Everything after L3 is deterministic and replays from a file. `acr concord` and `acr explain`
call no model and read no chart, which is why a number they produce can be re-derived six weeks
later without paying for the agent again.

---

## 2. Components

Line counts are the current tree. "No model" means the module *cannot* reach a provider —
`tests/test_evals.py` walks the import closure and fails if one appears.

### 2.1 The runtime — 2,107 lines

| module | what it is |
|---|---|
| `agent.py` | **The agent.** `create_agent` from LangChain plus deepagents middleware; our rules live in hooks. Replaced a 1,197-line hand-written ReAct loop. |
| `tool_surface.py` | The whitelist. Refuses an agent carrying a tool nobody declared. |
| `spend.py` | The cost ceiling, priced from `audit/prices.json`. |
| `audit.py` | Token/cost accounting, wired to LiteLLM or LangChain callbacks. |
| `corpus.py` | `PatientChart` — the only thing that touches note files. |
| `state.py` | `EvidenceLedger`, `Budget`. |
| `llm.py` | LiteLLM client, used by the develop-plane commands. |

**Where each rule lives, and why that hook:**

| concern | hook | why that one |
|---|---|---|
| record the plan before the first call | `before_agent` | runs once, inside the run it describes |
| the CURRENT plan, open threads, mechanical observations | `wrap_model_call` | `ModelRequest.override` never touches `messages`, so none of it can accumulate |
| tool allowlist, plan refusal, **the gate**, read recording, trigger detection | `wrap_tool_call` | sees every tool call, including one a library adds tomorrow |
| cost ceiling, expansion dead end, deadlock detector, no-tool-call recovery | `after_model` | fires after every model decision |
| call budget | `ModelCallLimitMiddleware` | the library's |
| context compaction | `SummarizationMiddleware` | the library's |
| the todo list | `TodoListMiddleware` | todos live in STATE, not in messages |

`revise_plan` is a **declared tool**, not a hook: the plan may only widen, and a proposal the
model makes must be auditable like any other tool call.

**Why `create_agent` and not `create_deep_agent`:** the latter injects nine tools nobody asked
for — `ls glob grep read_file write_file edit_file execute task write_todos`. Four are read
paths, and a read that does not go through `Toolbox.dispatch` is invisible to the
`CoverageLedger`, so the gate would still stamp `gate_validated: true` over a chart the ledger
never saw read. Under `FilesystemBackend(root_dir=".")` its `read` and `grep` reach absolute
paths outside root_dir — including `ground_truth.csv`. No recorded run exercised that; an
unexercised open door is not a boundary. `tool_surface.assert_tool_surface` is the boundary, and
it is a **whitelist** — a blacklist of today's nine would pass the tenth.

### 2.2 The audit layer — 6,508 lines. **This is the product.**

| module | what it decides |
|---|---|
| `answer_gate.py` | `gate_answer` — the single decision on whether an answer may be emitted. Every front end calls THIS. |
| `answer_checks.py` | Deterministic checks on a submitted value (`not_less_specific`, `conflict_requires_nos`, `origin_not_specimen`, …). Declared per spec; no clinical knowledge in the file. |
| `answer_contract.py` | What an answer owes at emission: `assert_answer_is_reportable`, `build_spec_gap`, `attach_coverage_claim`, `strip_value_from_spec_insufficient`. |
| `coverage.py` | The stratified coverage ledger, the forced sampler, the elusion bound (Clopper–Pearson). |
| `coverage_planner.py` | The ONE retrieval plan, monotone expansion, the thread ledger, the marker catalogue, trigger detection. |
| `plan_expansion.py` | The arithmetic of the expansion budget: pricing, partial application, when widening is over. |
| `run_triggers.py` | `detect_gate_obligations` — an obligation the current plan structurally cannot discharge. |
| `spec.py` / `speclint.py` | Load, freeze-hash and lint a spec; bind provenance and refuse an unmarked enforced element. |
| `trace.py` / `run_manifest.py` | The trace and the run record. |

**Coverage is stratified, not a count.** Each spec assigns document types to
`can_establish` (search exhaustively) / `may_mention` (search, then sample the misses) /
`cannot_establish` (validate the exclusion by sampling). `max_tolerated_hits: 0` — one hit in a
`cannot_establish` stratum overturns the declaration. The elusion bound is Clopper–Pearson, and
it is tied to **the frame it was drawn from**: adding a search term shrinks the miss frame, so
draws taken before the term was added no longer bound the population that remains. Monotone
expansion is monotone in the *evidence*, not in the *bound*.

### 2.3 Downstream, no model — 3,983 lines

`concordance.py` (L4 rule engine) · `explain.py` (L5 cause elimination) · `deps.py` (what a
recommendation reads, and what a spec edit invalidates) · `registry_catalog.py` (variable → spec
resolution; a variable belongs to exactly one spec, and ambiguity is an error rather than a
merge) · `intake.py` (question → spec routing).

### 2.4 The clinician's view — 1,648 lines

`specview/` renders a spec as prose a registrar can review and sign, with every element's
provenance and measurement beside it.

### 2.5 The develop plane — 3,608 lines. **Never run on data.**

`labelling.py` (read every note of a dev set once) · `derive.py` (labels → keywords + read
policy) · `assetdev.py` (evolve / certify / adopt retrieval assets). Consumes the answer key;
must never run in the same process as a RUN-plane job.

### 2.6 The eval plane — 3,066 lines

`evals.py` (precedence registry, abnormal-behaviour detectors, regression harness) ·
`judge.py` (agent-as-a-judge, fenced) · `refine.py` (route a classified failure at the text
parameter that caused it).

**The fence:** where a deterministic check exists, a model judge is *forbidden*, not
discouraged. `correctness` is `==`. A task-completion judge is refused outright because it
**launders abstention** — it scores a correct `EVIDENCE_INSUFFICIENT` as a failure, and
optimising against that teaches the agent to guess on exactly the subpopulation where the stakes
are highest. The fence is **per sub-question**, not per dimension.

### 2.7 Other front end — 941 lines

`mcp_server.py` exposes the chart tools and the gate over MCP. It shares `gate_answer`
(pinned by `tests/test_mcp_server.py`) but **not** the answer contract — see §5.2.

---

## 3. How to run each component

One environment. `langchain`, `deepagents`, `langfuse` and `pytest` are all in `.venv`.

```bash
cd /N/project/computable_phenotype/xh_project/agentic-chart-review
set -a && . /N/project/computable_phenotype/llm/.azure_env && set +a
export ACR_AUDIT_LOG=/N/project/computable_phenotype/llm/run/<name>/audit.jsonl
```

`.azure_env` exports `ACR_API_BASE`, `ACR_API_KEY`, `ACR_MODEL`. **It is chmod 600 and outside
the git tree. Keep it that way.**

### Look before you spend

```bash
.venv/bin/acr patients --corpus corpus/patients      # who is in the corpus
.venv/bin/acr chart SYN0001                          # one chart's doc-type summary
.venv/bin/acr specs                                  # specs and their freeze hashes
.venv/bin/acr extract --cohort c.csv --variables histology --dry-run
```

`--dry-run` resolves the variables, prices the work, and calls no model.

### L0–L3 · the agent

```bash
.venv/bin/acr extract \
  --cohort cohort.txt \                 # csv/tsv/txt/json of patient ids
  --variables primary_site,histology,behavior \
  --corpus /N/project/computable_phenotype/acr_real/patients \
  --max-steps 50 --temperature 1 --seed 1234 \
  --out runs/
```

- **in:** a cohort file, a variable list, `specs/`, a corpus
- **out:** `runs/extract__<utc>__<sha>/extract.json`, plus one `.jsonl` trace and one
  `.manifest.json` per (patient × spec)
- The unit of work is the **spec**, not the variable. `--variables primary_site,histology` is
  ONE pass because that spec answers both; asking per variable pays for the chart twice and can
  return two different sites for one patient.
- `--max-steps` is **model calls**, not plan/act/reflect cycles — there is no reflect node. The
  limit meant to bind is the cost ceiling in `spend.py` ($5/patient default).
- `--temperature` defaults to **1.0**: `gpt-5.6-luna` rejects any other value and 400s on the
  first call.

Single chart, for debugging: `acr chart run SYN0001 --spec specs/….yaml`.
Same chart N times, to measure self-consistency (which is *stability, not validity*):
`acr chart consistency SYN0001 --spec … --n 3`.

### L4 · concordance — no model

```bash
.venv/bin/acr concord --guideline guidelines/….yaml -i runs/…/extract.json -o concord.json
```

### L5 · explanation — no model

```bash
.venv/bin/acr explain -i concord.json -o explain.json
```

### The spec, in front of a clinician

```bash
.venv/bin/acr spec review   --spec specs/….yaml     # prose + provenance + measurements
.venv/bin/acr spec signoff  --spec specs/….yaml     # records reviewer, date, element hash
```

### The eval plane — no model

```bash
.venv/bin/acr eval dimensions            # THE FENCE: what a judge may and may not decide
.venv/bin/acr eval score   --runs runs/…/ --answer-key key.json \
                           --fields primary_site,histology,behavior \
                           --commit $(git rev-parse --short HEAD) --spec-hash … \
                           --model … --date … --baseline baseline.json
.venv/bin/acr eval detect  --runs runs/…/ --min-term-chars 3 --max-rejection-repeats 2 \
                           --token-band 20000,1500000 --turn-band 3,60
.venv/bin/acr eval compare --before a.json --after b.json     # exits 1 on REGRESSION
```

**`eval compare` needs `ACR_PSEUDONYM_KEY` set** on real data. Without it every real person_id
masks to one constant token, the per-instance index collapses, and `compare` returns
`NOT_COMPARABLE` (exit 2) rather than a wrong `0 regressions`. It reported exactly that once,
over a comparison that actually held 6 improvements and 3 regressions.

`eval detect` has **no default thresholds** on purpose: a detector that ran with numbers nobody
chose reports "nothing fired" indistinguishably from "nothing looked".

### The develop plane — spends money, consumes the answer key

```bash
.venv/bin/acr label scan   --spec … --dev-set … --max-usd 40    # refuses to start without --max-usd
.venv/bin/acr derive terms --labelling … --min-yield … --min-precision …
.venv/bin/acr assets evolve   …        # writes nothing
.venv/bin/acr assets adopt    …        # the only writer
```

Every `derive` threshold is a **required option**. There are no defaults, because a threshold
nobody chose is a finding nobody owns.

---

## 4. Reading a manifest

The fields that decide whether a number may be used:

| field | read it as |
|---|---|
| `answer.status` | `FOUND` / `EVIDENCE_INSUFFICIENT` / `SPEC_INSUFFICIENT` / `NO_ANSWER` |
| `gate_validated` | the proof obligation was met. **Not** "the answer is right." |
| `provenance.reportable_as_validated` | **the field a filter must read.** False while any element used is `draft`. |
| `provenance.weakest_status` / `counts_by_origin` | which elements dragged it down |
| `answer.proof_basis` | `WITNESS` (one qualifying document) / `UNGATED` / `NOT_APPLICABLE` |
| `answer.downgraded_from` + `withheld_value` | the run stopped owing an obligation; the value is preserved but **not asserted** |
| `degradation` | any non-zero entry means a node did less than it claims. **Read this first.** |
| `spend` / `spend_stopped` | what it cost, what it was allowed to cost, whether cost ended it |
| `expansion_stopped` | the plan could no longer widen and the obligation stood |
| `rejection_loop` | stopped because the ledgers froze, not because the budget ran out |
| `develop_plane_candidates` | candidate spec edits observed on a real chart. Score the spec against `spec_declared_terms`, **never** against the expanded list — a runtime rescue folded back into the baseline erases the evidence that the baseline was wrong. |

---

## 5. What is left

Ordered by what blocks trust.

### 5.1 The current runtime has never touched a real chart — **do this first**

Every real-patient number in this repo's history (5/10 exact, 10/10 gate-validated, $1.54 for
ten patients) was produced at commit `fbc7d82`, which **predates all sixteen audit fixes**.
Those manifests have 20 keys and no `provenance` block at all.

The sixteen fixes and the reframed limits are verified **by tests only**. Re-run the ten-patient
cohort on current code before quoting any figure:

```bash
B=/N/project/computable_phenotype/llm/run/hooks_current && mkdir -p $B
cp /N/project/computable_phenotype/llm/run/real_ten/{cohort.txt,answer_key.json} $B/
export ACR_AUDIT_LOG=$B/audit.jsonl
.venv/bin/acr extract --cohort $B/cohort.txt --variables primary_site,histology,behavior \
  --corpus /N/project/computable_phenotype/acr_real/patients \
  --max-steps 50 --temperature 1 --out $B/runs
```

Watch **the 293-document and 34-document patients** specifically: both were cut short by the old
rejection brake, one of them into a wrong histology. The brake now tests whether the ledgers
moved rather than whether a refusal repeated, so those two are the cases that should change.

**Do not run two jobs against the deployment at once.** Doing so cost three of ten patients to
HTTP 429, and the dead runs looked like agent behaviour until their traces were read.

### 5.2 `mcp_server.py` is the next `graph.py`

It shares the gate and **none** of the answer contract:

```
provenance_for_run       0 references
downgrade_a_positive     0
attach_coverage_claim    0
develop_plane            0
rule_citation_block      0
```

The sixteen gaps fixed in `agent.py` were "a port dropped them". These are "they were never
there". Same class of defect, unaddressed. `tests/test_spec_insufficient.py` already asserts
that *both* front ends go through the shared builders — extend that assertion to the rest.

### 5.3 Needs a human, not code

- **The 101-document patient** scores 0/3 and passes the gate. Its reasoning quotes the spec's
  own `not_less_specific` message verbatim ("favor squamous supports 8070 over 8046"); the
  registry coded 8046. **The spec and the registry disagree, and a registrar has to rule.** That
  rule's provenance is `model_authored / draft`.
- **`cached_input_per_1m = $0.10`** in `audit/prices.json` was stated, not measured against an
  invoice. Every cost conclusion's magnitude depends on it — at full price the same tokens cost
  4.3× more.
- **All 90 provenance records are `model_authored`, all `draft`, zero `store_manual`.** No
  registrar has read a line of any spec. This is why `reportable_as_validated` is false
  everywhere, and it is the correct answer until someone signs.

### 5.4 Numbers with no evidence behind them

- `max_frozen_repeats = 3` in `agent.py`. The *judgement* is now right (stop when the ledgers
  freeze, not when a refusal repeats) but the threshold is a guess. It needs arms at 2, 3 and 5
  on one cohort.
- `spend.py`'s `max_usd = 5.0` is ~33× a typical run ($0.15). Deliberately generous so it never
  ends a working run — and therefore never yet fired, so it is untested in anger.

### 5.5 Never run on data

The develop plane (`labelling` / `derive` / `assetdev`, 3,608 lines) has a CLI and tests and
**has never performed a single real scan**. Until it does, every keyword list in `specs/` is
model-authored and unmeasured — which is exactly what `develop_plane_candidates` exists to fix.

### 5.6 Architectural, unsolved

**Quarantine is not a boundary.** One process holds both the RUN-plane and DEVELOP-plane
credentials, the answer key ships in the payload of the very call that quarantines, and the
ledger is process memory no other front end reads. The spelling class is closed and pinned by
62 tests; the boundary is not.

**`langfuse` is wired and off.** `lc_callback.langfuse_handler()` attaches when `LANGFUSE_HOST`
is set. It needs a self-hosted instance inside the approved boundary — six services, and this
cluster has no container runtime on the login node. Note that `mask` does **not** cover
LangChain callback spans; only `mask_otel_spans` does. Getting that wrong sends chart text to
`cloud.langfuse.com`.

### 5.7 Branch state

Four commits on `deepagents-only`, not merged:

```
0a23097  The migration is finished: 74 failures to zero, and graph.py is gone
a56a61e  Sixteen audit rules the port had dropped, found by refusing to delete the tests
1e54ad8  Restore _record_reads, which my own edit deleted, and pin it so it cannot go a third time
e7e61bb  Port the three things the hooks runtime was missing, before deleting the one that had them
```

`1410 passed, 12 skipped, 0 failed`.

---

## 6. Rules for whoever takes this over

**Non-negotiable, from IRB and from this tree's history:**

1. Never write a real person_id (`1168` + 12 digits) or note text into any file under the repo.
   `tests/test_no_phi_in_tree.py` enforces it — and it has caught me.
2. The only endpoint approved for PHI is the Azure deployment in `.azure_env`. That approval does
   **not** generalise. It is why Langfuse Cloud and AgentLoop are unusable here.
3. `/N/project/computable_phenotype/acr_real/` is the real corpus. `corpus/patients/` is
   synthetic and PHI-free — develop against it.
4. Never `rm -rf` under `runs/`; use `tools/archive_runs.sh`.
5. **Another team works in this tree.** Do not touch `authoring/`,
   `skills/crc-guideline-registry-authoring/`, `tests/test_crc_*.py`. **Do not run
   `git stash -u`** — it stashes their untracked work. I did, and it looked like I had broken
   four of their tests.
6. Write run outputs outside the repo (`/N/project/computable_phenotype/llm/run/`). `/N/slate/`
   is near quota.

**Three lessons that cost real time:**

**A check that cannot fail is worse than no check.** Found seven times here: a coverage gate that
never read its own required-keywords field; a keyword matcher that `t` satisfied; a tool-surface
test skipped in one venv and uninvokable in the other, so it ran nowhere; `_callbacks()`
swallowing `ModuleNotFoundError` into an empty list so every run recorded `usage: null`. When you
add a guard, **mutate the code and watch the test go red** before you believe it.

**Do not edit by range slice.** `s[s.index('def X'):s.index('def Y')]` swallows everything
between the boundaries. It silently deleted `_record_reads` (twice — once in the port, once by my
own hand), then `SRC`, `OPEN_REQUEST_RETURNING` and `_open_kwargs`. Each time the *call site*
survived, so the failure looked like a broken runtime rather than a missing rule. Use an
AST-exact single-node replacement.

**And the one that matters most.** The instruction that produced most of this work was "delete the
tests that only existed for the old runtime". Thirty-six of them turned out to be the only
coverage of rules still in the product, and migrating them instead of deleting them surfaced
**sixteen audit rules the runtime had silently stopped enforcing** — including two fixed hours
earlier the same day. Every one would have become "how the new architecture behaves", under a
green suite. When a test looks like scaffolding, check what it is holding up.
