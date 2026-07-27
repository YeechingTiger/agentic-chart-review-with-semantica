# agentic-chart-review

An EHR chart-review agent that works the way a human abstractor does — see what documents
exist and when, narrow by type and date, search, read only what matters, quote what it
finds — under a **frozen, machine-checkable extraction specification**.

The design question this repo answers is not "can a model read a note". It is: *what has to
be in the input, and what has to be enforced in code, for two independent executors — a
human abstractor and an LLM agent — to reach the same label for the same reason?*

---

## Four ideas that shape the whole thing

**1. The specification is the contract, and it is frozen and hashed.**
A spec states the decision boundary and the evidentiary rules but deliberately *not* the
navigation path. How the agent finds the evidence is its own business; what counts as
evidence, and what must be true before it may assert a negative, are not. Every spec is
content-hashed, and a label is only comparable to another label produced under the same
`spec_hash`.

**2. Proof obligation is a first-class field, and it is checked in code.**
Most extraction failures are not misreadings — they are confident negatives asserted after
an incomplete search. Each spec declares what "you looked" *means* for that variable. The
agent's `submit_answer` is **rejected** if the computed coverage ledger does not satisfy
that obligation. Coverage is derived from real tool calls, never self-reported. Prompting a
model to "be sure you looked everywhere" is a wish; checking the ledger is a control.

**3. There are two ways to not know, and they mean different things.**

| status | meaning |
|---|---|
| `FOUND` | the answer is established by recorded evidence |
| `EVIDENCE_INSUFFICIENT` | the spec is clear; **the chart** lacks the evidence |
| `SPEC_INSUFFICIENT` | **the spec** does not cover this case, or the variable is not derivable from notes at all |

Collapsing these destroys the signal that tells you whether to fix your specification or go
get more data.

**4. Sort every piece by whether it must be *enforced*, merely *known*, or *declared*.**

| | Enforced | Advisory | Declarative |
|---|---|---|---|
| what | the plan, forced sampling, the gate, format checks, the quarantine | how an abstractor works; how to code a variable | what a variable means; what counts as evidence |
| where | **code** | **skills** (`skills/*/SKILL.md`) | **spec** (`specs/*.yaml`) |
| may the model decline it? | **no** | yes — progressive disclosure means it *chooses* to load | n/a |
| improved by | adversarial attack + tests | evolution against a held-out reward | corpus measurement + clinician review |
| signed off by | engineer | engineer | **clinician** |

Making a judgement rule *enforced* turns a wrong answer into no answer. Making an audit rule
*advisory* is what "let the agent decide what to read" means — the thing under audit choosing
its own scope.

---

## Two planes, and a gate between them

```
╔═ DEVELOP PLANE ════════════════════════════════════════════════════╗
║  Consumes an answer key. Learns keyword lists, document-type        ║
║  policy, spec wording. Held-out splits. Kill criteria.              ║
╚═════════════════════════╤══════════════════════════════════════════╝
                          │  the ONLY channel: a provenance record
                          │  plus a human signature. Not a code path.
╔═════════════════════════▼══════════════════════════════════════════╗
║  RUN PLANE — never sees the answer key.                             ║
║  Produces answers, evidence, and proofs about what was not found.   ║
╚═════════════════════════════════════════════════════════════════════╝
```

If an answer key reaches the run plane, every number downstream is void and nothing in the
output says so. That is why `registry.truth` sits behind a separate credential and why one
call to it quarantines the run.

---

## The modules

46 modules in five groups. Each names its single responsibility in its own docstring; the
table is that sentence, shortened.

### Run plane — L0 to L5

| module | does |
|---|---|
| `intake` | **L0.** An arbitrary question becomes a routing decision, or an explicit gap list |
| `registry_catalog` | **L0b.** A user's variable names to the specs that produce them. Exact match; ambiguity is an error |
| `coverage_planner` | **L1.** *The* plan: what may be opened, what is searched with which terms, what is only sampled. Also the open-thread ledger |
| `graph` | **L2.** The loop: plan → act → reflect → (act \| replan \| finalize), and the single route to an answer |
| `run_triggers` | Detect, mechanically and without asking a model, the observations a reflection must answer for |
| `plan_expansion` | The arithmetic of the expansion budget: what a monotone widening costs, and when widening is over |
| `answer_gate` | **L3.** The single decision on whether a submitted answer may stand — returned as a recoverable rejection, not raised |
| `answer_contract` | What an answer owes at emission, asserted by every runtime rather than intended by each |
| `answer_checks` | Deterministic checks on a submitted answer, from the spec's own `answer_checks` |
| `coverage` | What has to be true before a negative answer is allowed. Strata, forced sampling, the elusion bound |
| `concordance` | **L4.** Guideline concordance, decided by rule and never by a model |
| `explain` | **L5.** Why a case is non-concordant. Four causes, and they must not become one number |
| `deps` | **L4.5.** Which variables a rule reads, in both directions, and what a spec edit invalidates |
| `state` · `trace` · `run_manifest` | The plan and the ledgers; the trace and its rule attribution; the record a finished run leaves |
| `corpus` · `llm` · `tools/toolbox` | Chart access, provider access, the tool schemas |

### Spec layer

| module | does |
|---|---|
| `spec` | Load, validate, freeze. Provenance is enforced: an unprovenanced enforced element does not load |
| `speclint` | Spec completeness in four tiers — formal, against the corpus, against an answer key, and irreducibly human |
| `specview/` | Seven modules that turn a spec into the document a clinician reads and signs: `statements`, `decisions`, `prose`, `basis`, `measurements`, `render`, `signoff` |

### Develop plane

| module | does |
|---|---|
| `labelling` | **The full scan.** One cheap reading of every note in a development set against one requirement. Per field: does this note bear on the question, and which terms would find it |
| `derive` | **First-order derivation.** Read the labelling, price the words by grep, write the plan. The model says what *indicates* the answer; grep says what it *costs* to search for |
| `assetdev` | The second-order version: hill-climb a candidate, certify on a held-out split |
| `refine` | One reflective optimizer over **every** text parameter the agent reads — spec, skills, prompts, rejection messages — with a different update policy per parameter |

### Eval plane

| module | does |
|---|---|
| `evals` | The precedence registry (what may be judged versus what is already decided exactly), the runtime abnormality detectors, and the regression harness over manifests |
| `judge` | Agent-as-a-judge, fenced by that registry. A judged number is an **opinion** and never a gate. Evaluators are `evaluators/*.yaml`, not code |

### Front ends

`cli` composes ten command groups and decides nothing: `cli_chart`, `cli_pipeline`,
`cli_plan`, `cli_spec`, `cli_label`, `cli_eval`, `cli_judge`, `cli_refine`, `cli_common`.
`mcp_server` exposes the same capability as one MCP tool surface. `deep_runner` is a second
runtime over the same gate.

---

## How they connect

```
  a question or a cohort
        │
   intake ─── registry_catalog ──► spec (frozen, hashed, provenanced)
        │                            │
   coverage_planner ◄────────────────┘   THE plan: read_all / search / sample
        │                                revision is MONOTONE EXPANSION only
   ┌────▼───────────────────────────────────────────────────┐
   │ graph      plan → act ⇄ reflect → finalize             │
   │            run_triggers detect · plan_expansion prices  │
   └────┬───────────────────────────────────────────────────┘
        │ submit_answer
   answer_gate ── coverage (strata · forced sampling · elusion bound)
        │         answer_checks · answer_contract
        ├─► rejected, with a reason the agent can act on
        └─► accepted ──► run_manifest + trace
                              │
                        concordance (L4, no model) ──► explain (L5, four causes)
```

Two rules make the loop auditable rather than decorative. **Forced sampling draws from a
server-held seed** — if the agent could choose its own validation sample the circularity is
back. **The gate is the only thing that may mark an answer validated**, and there is exactly
one of it; `test_gate_validated_has_exactly_one_origin` exists because a second copy grew
once and had to be removed.

The develop plane runs offline over the same corpus and writes back **only** through a
provenance record. Keyword lists may be adopted automatically once certified, because they
change which text reaches the agent and never what an answer means. Document-type policy,
evidence rules and thresholds are **semantic** — they change what a correct answer *is* — so
the plane may only propose them, and a clinician signs.

---

## Install

Server / GPU / vLLM: see **[DEPLOY.md](DEPLOY.md)**.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
./.venv/bin/python tools/generate_corpus.py --out corpus/patients   # synthetic, PHI-free
```

**Always call the venv's interpreter directly.** The system Python is 3.9 and cannot import
`acr`; a bare `python3.12` has no pytest. Both failures look like a broken checkout:

```bash
./.venv/bin/python -m pytest      # correct
python3 -m pytest                 # ModuleNotFoundError: acr
```

## Use

```bash
acr patients                        # what's in the corpus
acr chart SYN0001                   # document-type summary — what the agent sees first
acr specs                           # available specs with freeze hashes
acr spec lint specs/…yaml           # four tiers, and what no linter can check
acr spec review specs/…yaml         # the document a clinician reads and signs

acr ask "does this patient have resected stage II NSCLC"    # route, do not run
acr extract --cohort c.txt --variables primary_site,histology,behavior
acr concord --guideline guidelines/nccn_nsclc_subset.yaml --input extract.json
acr explain --input concord.json
acr deps --guideline guidelines/…yaml       # what each rule reads, and the gaps

acr eval dimensions                 # THE FENCE: what may be judged, and what is already exact
acr eval detect --runs runs/…       # the abnormality detectors over recorded runs
acr judge evaluators                # load evaluators/*.yaml against the real fence
acr refine parameters               # the text parameters, and who may update each
```

Nothing spends by default. Every command that can call a model requires an explicit cost
ceiling with no default, and `--dry-run` plans and prices without calling.

---

## The run plane in detail

`reflect` is a **separate node** with one job: look at what has been gathered and rule
`CONTINUE` / `REPLAN` / `SUFFICIENT` / `STUCK`. In a plain ReAct loop the stopping decision
is an afterthought made inside the same generation that just read a document.

Revision is restricted to **monotone expansion** — add a term, promote a document type
toward more reading, open a thread; never remove, demote or shrink. That restriction is what
lets the audited party adjust its own scope at all: expansion can only make the evidence
stronger.

**But monotone in the evidence is not monotone in the bound.** Adding a term moves documents
out of the *miss* frame the sample was drawn from. Measured on a real ledger: one added term
left 5 of 25 draws inside the frame, and the ledger went on reporting the 25-draw bound of
0.1129 against a 0.12 cap where the surviving draws earned 0.4507. A sample is now tied to
the frame it was drawn from — the bound is recomputed over surviving draws, replacements are
forced until *n* is restored, and a run that cannot restore *n* is refused.

`finalize` sees **only the evidence ledger**, never the scratchpad, so the model cannot
remember an uncited detail into the final answer.

### Tools

| tool | purpose |
|---|---|
| `list_documents` | metadata only — type, date, size. Never returns text |
| `document_type_summary` | counts and date span per type; cheap orientation in a large chart |
| `search_notes` | search, returns note_id + character offsets |
| `read_document` | paginated read with stable, citable offsets |
| `read_section` | jump to `IMPRESSION`, `FINAL DIAGNOSIS`, … |
| `timeline` | chronological view — needed to establish intervals |
| `record_evidence` | pin a verbatim span; the quote is sliced out of the document by offset, so it cannot be model-authored |
| `submit_answer` | validated, and rejected if the proof obligation is unmet |

---

## Corpus format

One flat directory per patient; the filename carries the metadata:

```
corpus/patients/SYN0001/Surgical-Pathology-Report_2023-04-27.txt
```

`<DocType>_<YYYY-MM-DD>[__<n>].txt`. `src/acr/corpus.py` is the only module that knows this;
point it at a different backend and nothing else changes.

The bundled corpus is **synthetic and PHI-free**, and each patient is built around a
deliberate evidence pattern so the specs are actually exercised:

| patient | pattern | what it tests |
|---|---|---|
| SYN0001 | cytology precedes pathology | the `[390]` date boundary rule |
| SYN0002 | biopsy done at an outside hospital | must answer `EVIDENCE_INSUFFICIENT`, not infer histology |
| SYN0003 | metastatic at presentation | recurrence `70`, not `00` |
| SYN0004 | recurrence after a disease-free interval | requires establishing **both** halves |
| SYN0005 | carcinoma in situ | behaviour `2`, not a reflexive `3` |
| SYN0006 | patient declined biopsy | second evidence-gap mechanism |
| SYN0007 | intraductal carcinoma with focal invasion | behaviour `3` despite in-situ wording |
| SYN0008 | consult contradicts the imaging impression | prefer the primary report, cite both |

Ground truth lives in `_ground_truth.json` next to the notes, reachable only through the
quarantined credential.

---

## Run output is data, not build product

A trace that documented a bug in code that has since been fixed **cannot be regenerated** —
the path that produced it no longer exists. One batch was already lost this way; see
`runs/_archive/NOTES.md`.

- Directories are `runs/<label>__<UTC>__<code-sha>/`, created with `exist_ok=False`.
- Manifests from the **synthetic** corpus are committed (~2KB, evidence). Traces are not.
- **Real-patient run output never enters git**, manifest or not: a real run writes the
  patient id into the directory name and into the manifest.
- To clear runs use `tools/archive_runs.sh`. **Never `rm -rf runs/…`.**

Without the trace you cannot distinguish a correct answer from a lucky one, which is the
whole reason this is built the way it is.

---

## On self-consistency

`acr consistency` measures **stability, not correctness.** A model can settle on one wrong
reading of an ambiguous specification and repeat it perfectly. High self-consistency with
low agreement against an adjudicated reference is the signature of a shared misreading, and
it is a finding about the *specification*.

Replicate runs of one model are **not** independent raters: their errors are correlated by
construction. Do not pool them as if they were separate abstractors.

---

## Status

v0.2. 1,374 tests pass. Six specs, three guideline recommendations, eight skills, three
evaluators.

**Real:** the spec layer with enforced provenance (90 records — 90 `model_authored`, 90
`draft`, which is the finding and not an omission); the run plane end to end on the synthetic
corpus; the rule engine; the four-cause scaffold; the linter, which finds four tier-1
failures in the shipped histology spec.

**Built but never run against data:** the develop plane and the eval plane. Libraries with
tests.

**Known broken:**

- The quarantine is **not a boundary**. The spelling class is closed and pinned by 62 tests,
  but one process holds both credentials and the answer key ships in the payload of the very
  call that quarantines. Damage limitation, not separation.
- The first true end-to-end run **deadlocked**. The agent correctly identified a truncated
  pathology report as its blocker, said so in twelve consecutive reflections, read to the end
  of the document — and then re-opened the same thread thirteen times, because nothing
  connects *"I read it"* to *"the thread is settled"*. It exhausted a 400k-token budget.
- `replan_rate` in the manifest read `0.0` where the trace held thirteen applied revisions.
  A conclusion was drawn from that reading before the instrument was checked.

**Not validated against real charts. No clinical use.**

## Licence

MIT.
