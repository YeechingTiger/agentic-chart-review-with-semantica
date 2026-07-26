# agentic-chart-review

An EHR chart-review agent that works the way a human abstractor does — see what documents
exist and when, narrow by type and date, search, read only what matters, quote what it
finds — under a **frozen, machine-checkable extraction specification**.

The design question this repo answers is not "can a model read a note". It is: *what has to
be in the input, and what has to be enforced in code, for two independent executors — a
human abstractor and an LLM agent — to reach the same label for the same reason?*

---

## Three ideas that shape the whole thing

**1. The specification is the contract, and it is frozen and hashed.**
A spec states the decision boundary and the evidentiary rules but deliberately *not* the
navigation path. How the agent finds the evidence is its own business; what counts as
evidence, and what must be true before it may assert a negative, are not. Every spec is
content-hashed, and a label is only comparable to another label produced under the same
`spec_hash`.

**2. Proof obligation is a first-class field, and it is checked in code.**
Most extraction failures are not misreadings — they are confident negatives asserted after
an incomplete search. Each spec declares what "you looked" *means* for that variable:
which searches must have run, which document types must have been reviewed. The agent's
`submit_answer` is **rejected** by the graph if the computed coverage ledger does not
satisfy that obligation. Coverage is derived from real tool calls, never self-reported.
Prompting a model to "be sure you looked everywhere" is a wish; checking the ledger is a
control.

**3. There are two ways to not know, and they mean different things.**

| status | meaning |
|---|---|
| `FOUND` | the answer is established by recorded evidence |
| `EVIDENCE_INSUFFICIENT` | the spec is clear; **the chart** lacks the evidence |
| `SPEC_INSUFFICIENT` | **the spec** does not cover this case, or the variable is not derivable from notes at all |

Collapsing these two into one "unknown" destroys the signal that tells you whether to fix
your specification or go get more data.

---

## Install

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"

# generate the synthetic, PHI-free dev corpus (8 patients, ~166 documents, 2010-2025)
python tools/generate_corpus.py --out corpus/patients
```

Any LiteLLM-supported provider works. Local by default:

```bash
ollama serve &
ollama pull qwen3.6:35b
```

## Use

```bash
acr patients                                    # what's in the corpus
acr chart SYN0001                               # document-type summary — what the agent sees first
acr specs                                       # available specs with freeze hashes

acr run SYN0001 --spec specs/STORE.400_522_523.site_histology_behavior.yaml
acr batch --spec specs/STORE.1860_1880.first_recurrence.yaml
acr trace runs/<run-id>.jsonl                   # replay what the agent actually did

# stability, which is NOT accuracy — see below
acr consistency SYN0002 --spec specs/... --n 5 --temperature 0.7
```

Switch models without touching code:

```bash
acr run SYN0001 --spec specs/… -m ollama_chat/qwen3.6:35b
acr run SYN0001 --spec specs/… -m hosted_vllm/Qwen/Qwen3.6-35B --api-base http://localhost:8000/v1
acr run SYN0001 --spec specs/… -m openai/gpt-4.1
acr run SYN0001 --spec specs/… -m anthropic/claude-sonnet-4-5
```

or via env: `ACR_MODEL`, `ACR_API_BASE`, `ACR_API_KEY`, `ACR_TEMPERATURE`.

---

## The agent

```
START → plan → act ⇄ reflect → finalize → END
                 ↑        │
                 └── replan
```

`reflect` is a **separate node** with one job: look at what has actually been gathered and
rule `CONTINUE` / `REPLAN` / `SUFFICIENT` / `STUCK`. In a plain ReAct loop the stopping
decision is an afterthought made inside the same generation that just read a document; here
replanning is a first-class, traceable event. That is what produces the behaviour you want
— the agent notices something, revises what it intends to do next, and stops when it has
enough rather than when it runs out of turns.

`finalize` sees **only the evidence ledger**, never the scratchpad, so the model cannot
"remember" an uncited detail into the final answer.

### Tools

The moves a human abstractor makes:

| tool | purpose |
|---|---|
| `list_documents` | metadata only — type, date, size. Never returns text. |
| `document_type_summary` | counts and date span per type; cheap orientation in a large chart |
| `search_notes` | case-insensitive / regex search, returns note_id + character offsets |
| `read_document` | paginated read with stable, citable offsets |
| `read_section` | jump to `IMPRESSION`, `FINAL DIAGNOSIS`, `ASSESSMENT AND PLAN`, … |
| `timeline` | chronological view — needed to establish intervals |
| `record_evidence` | pin a verbatim span; required before answering |
| `submit_answer` | validated, and rejected if the proof obligation is unmet |

---

## Corpus format

One flat directory per patient; the filename carries the metadata:

```
corpus/patients/SYN0001/
    Chest-CT-W-Contr_2023-04-06.txt
    Surgical-Pathology-Document_2023-04-12.txt
    Onc-Med-MD-OP-Progress-Note_2023-04-12.txt
    Surgical-Pathology-Report_2023-04-27.txt
    EKG_2019-03-24.txt
    Prescriptions-Filled-RxHub_2015-04-13.txt
    ...
```

`<DocType>_<YYYY-MM-DD>[__<n>].txt`, where `__<n>` disambiguates same-type-same-date
documents. `src/acr/corpus.py` is the only module that knows this; point it at a different
backend and nothing else changes.

The bundled corpus is **synthetic and PHI-free**, generated deterministically, and each
patient is built around a deliberate evidence pattern so the specs are actually exercised:

| patient | pattern | what it tests |
|---|---|---|
| SYN0001 | cytology precedes pathology | the `[390]` date boundary rule |
| SYN0002 | biopsy done at an outside hospital | must answer `EVIDENCE_INSUFFICIENT`, not infer histology |
| SYN0003 | metastatic at presentation | recurrence `70` (never disease-free), not `00` |
| SYN0004 | recurrence after a disease-free interval | requires establishing **both** halves |
| SYN0005 | carcinoma in situ | behaviour `2`, not a reflexive `3` |
| SYN0006 | patient declined biopsy | second evidence-gap mechanism |
| SYN0007 | intraductal carcinoma with focal invasion | behaviour `3` despite in-situ wording |
| SYN0008 | consult contradicts the imaging impression | prefer the primary report, cite both |

Ground truth for each lives in `_ground_truth.json` next to the notes.

---

## Traces

Every run appends JSONL: plan revisions, every tool call with full input and output, every
reflection verdict, every rejected answer, and the final coverage attestation. A run also
writes a `.manifest.json`. `Tracer.to_capg()` reshapes a trace into the observation-tree
form a CAPG-style provenance-graph adapter consumes.

Without the trace you cannot distinguish a correct answer from a lucky one — which is the
whole reason this is built the way it is.

---

## On self-consistency

`acr consistency` runs the same spec N times and reports agreement. Read the output
carefully: **it measures stability, not correctness.** A model can settle on one wrong
reading of an ambiguous specification and repeat it perfectly. High self-consistency with
low agreement against an adjudicated reference is the signature of a shared misreading, and
it is a finding about the *specification*, not about the model.

For the same reason, replicate runs of one model are **not** independent raters: their
errors are correlated by construction. Do not pool them as if they were separate abstractors.

---

## Status

v0.1 — the spec format, the corpus layer, the tools, the graph, and the enforcement gate are
implemented and runnable end to end against a local model. Specs currently cover four
variables (`[390]`, `[400]+[522]+[523]`, `[1860]+[1880]`, and `[610]` as a not-derivable-from-notes
example). Not validated against real charts; no clinical use.

## Licence

MIT.
