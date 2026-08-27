# Agentic Chart Review with Semantica

This repository helps a human audit a chart-review agent at the level where the work can actually
be improved: which evidence it looked for, which notes it opened, how it judged each finding, how
it resolved conflicts, and why it stopped with its final answer.

The central design is:

```text
Codex chart-review agent
        ↓
local Langtrace execution record
        ↓
deterministic ReAct cycles
        ↓
Luna Decision Episode reconstruction + verifier
        ↓
Semantica ContextGraph
        ↓
human audit, cross-run comparison, and policy-impact review
```

Raw trace remains the execution evidence, but it is too detailed and unstable to be the primary
human review surface. A Decision Episode is one consequential choice that one reviewer can judge
with one verdict. Verified episodes become native Semantica decisions, with direct links back to
the cycles, runtime testimony, evidence, policy clauses, and provenance that produced them.

## What each layer is for

| Layer | Authority | Main use |
|---|---|---|
| Codex + chart tools | Actual agent/tool execution | Run the chart review inside a chart-only tool boundary |
| Langtrace | Canonical observable run record | Replay what happened; preserve model/tool spans and server facts |
| ReAct cycles | Deterministic state/action/observation units | Fix the execution skeleton before semantic interpretation |
| Decision Episodes | Post-run audit units | Let a human follow and judge consequential choices in order |
| Semantica ContextGraph | Reusable decision layer | Similar cases, divergence, causal traversal, policy binding, impact queues |
| Semantica Provenance | Lineage and integrity | Distinguish server fact, self-report, derivation, reconstruction, and human review |
| Semantica Explorer | Interactive delivery shell | Review the Decision Chain and save durable human adjudications |

ACR does not expose private chain-of-thought. At runtime the agent records concise Decision
Testimony: the question, choice, alternatives, rationale, claimed basis, cited refs, checked facts,
and uncertainty. The chart server seals that testimony into a receipt. Post-run Luna can assign a
new semantic taxonomy without rerunning the agent, while the original testimony remains visible.

## Install

Requirements:

- Python 3.12 and [`uv`](https://docs.astral.sh/uv/)
- Node/npm for the Codex CLI and optional Semantica Explorer build
- Docker Compose for local Langtrace

Clone the project, then install the lean graph/notebook profile:

```bash
git clone --branch main --single-branch https://github.com/YeechingTiger/agentic-chart-review-with-semantica.git
cd agentic-chart-review-with-semantica

uv venv --python 3.12
uv pip install -e ".[ledger,ledger-ui,dev,notebook]"
uv pip install --no-deps semantica==0.6.6

npm install -g @openai/codex@latest
codex --version
```

Semantica is installed with `--no-deps` intentionally. Its full package declares large embedding,
NLP, vision, and analytics dependency families that this graph-only profile disables. The six
runtime dependencies ACR needs are supplied by the project and base dependencies. Semantica is
pinned because the current adapter depends on its persistence and decision-index behavior.

The notebook and CLI work with released Semantica. The human review UI additionally needs ACR's
Semantica Explorer Decisions extension:

```bash
tools/install_semantica_decision_review.sh
```

That script checks out the pinned upstream Semantica commit, applies the repository patch, builds
Explorer, and installs it into the active environment.

## Start local Langtrace

Langtrace is the reconstruction source, not an optional logging sidecar. The official self-hosted
Docker Compose setup is:

```bash
git clone https://github.com/Scale3-Labs/langtrace.git
cd langtrace
docker compose up -d
```

Open `http://localhost:3000`, create a project, and generate its API key. The official setup guide
is [Langtrace Docker Compose Setup](https://docs.langtrace.ai/hosting/hosting_options/docker-compose).
If your Compose port mapping uses another port, put that port in `LANGTRACE_API_HOST`.

Back in this repository, copy the environment template and fill in the values:

```bash
cp .env.example .env
set -a
source .env
set +a
```

Keys must stay in the environment. Do not put them in a notebook cell, CLI argument, committed
artifact, or screenshot.

## Start with the three notebooks

```bash
.venv/bin/jupyter lab notebooks/
```

Run them in order:

1. [01 — Run chart-review agent (checked output)](notebooks/01_run_chart_review_experiments.executed.ipynb)
   explains the ReAct-style chart agent and its seven tools, runs or reuses the small cohort, and
   shows only the returned answer beside synthetic gold. [Runnable source](notebooks/01_run_chart_review_experiments.ipynb).
2. [02 — Trace to Decision Chain (checked output)](notebooks/02_trace_to_decision_chain.executed.ipynb)
   treats the raw trace as a long receipt, then walks one real case through eight human-auditable
   questions. It shows why search coverage—not the final date comparison—is the main review point.
   [Runnable source](notebooks/02_trace_to_decision_chain.ipynb).
3. [03 — Semantica Decision Intelligence (checked output)](notebooks/03_semantica_decision_intelligence.executed.ipynb)
   treats ContextGraph as an indexed decision-card box. It shows one readable card, similar-card
   retrieval, a true same-evidence disagreement, and the re-audit queue for a changed Policy.
   [Runnable source](notebooks/03_semantica_decision_intelligence.ipynb).

Each source notebook is accompanied by an `.executed.ipynb` produced from persisted real
OpenRouter Luna/Terra runs. The executed companion is a compact reading copy: it keeps the saved
outputs and source-cell hashes but omits implementation text; use the source notebook to inspect or
rerun code. Run data itself stays under `runs/` and never enters Git.

Notebook 1 defaults to `reuse` when the local checked cohort exists. To generate a new cohort:

```bash
export ACR_TUTORIAL_MODE=live
# 2 paired cases × 2 instruction sets × Luna
```

Two Luna reconstruction passes must agree on episode alignment before the notebook selects an
analysis. Luna reconstruction uses the LiteLLM/OpenRouter model id
`openrouter/openai/gpt-5.6-luna`; the Codex harness requests `openai/gpt-5.6-luna` or Terra through
the OpenRouter Responses-compatible endpoint.

## Essential CLI

The notebooks are the teaching path; these commands are the operational equivalents:

```bash
# Run one real policy-guided review
acr run assets/specs/STORE.390.date_of_initial_diagnosis.yaml \
  corpus/patients/SYNX03 --out runs/mvp --task-arm policy_bundle

# Read canonical Langtrace and reconstruct two append-only analyses
acr trace runs/mvp/<RUN_DIR>
acr reconstruct runs/mvp/<RUN_DIR> --ledger runs/mvp/ledger.json --passes 2

# Select exactly one analysis before human review
acr select-analysis <RUN_ID> <ANALYSIS_ID> --ledger runs/mvp/ledger.json \
  --selected-by reviewer --reason "two passes aligned"
acr chain <RUN_ID> --run-dir runs/mvp/<RUN_DIR> --ledger runs/mvp/ledger.json

# Semantica-native decision intelligence
acr similar <EPISODE_ID> --ledger runs/mvp/ledger.json
acr causal-trace <EPISODE_ID> --ledger runs/mvp/ledger.json
acr impact <EPISODE_ID> --ledger runs/mvp/ledger.json
acr policy-impact <POLICY_ID> --from-version <OLD> --to-version <NEW> \
  --ledger runs/mvp/ledger.json

# Human audit UI
acr review-ui <RUN_ID> --run-dir runs/mvp/<RUN_DIR> \
  --ledger runs/mvp/ledger.json --port 8877
```

`task_only` is useful for locating inconsistent ungoverned judgment. `policy_bundle` is useful for
testing whether decisions cite offered clauses and for retrieving the exact historical decisions
bound to a changed Policy version. Similarity proposes comparison candidates; it does not create
clinical precedent. Policy impact returns a re-audit queue; it does not predict that an answer
would change.

## Verify

```bash
.venv/bin/python -m pytest tests/test_mvp_*.py -q
.venv/bin/ruff check src tests tools

# Rebuild and execute the maintained walkthroughs against the local sealed cohort
.venv/bin/python tools/build_postdoc_notebooks.py
for nb in 01_run_chart_review_experiments 02_trace_to_decision_chain 03_semantica_decision_intelligence; do
  .venv/bin/jupyter nbconvert --to notebook --execute "notebooks/${nb}.ipynb" \
    --output "${nb}.executed.ipynb" --output-dir notebooks \
    --ExecutePreprocessor.timeout=900
done
.venv/bin/python tools/prepare_executed_notebooks.py
```

The ordinary test suite uses fixed traces and stub extractors so it can test verifier behavior
under reconstruction mistakes. Paid OpenRouter coverage is opt-in. The executed notebooks cover
the real-provider boundary and retain the interpretation limits beside every result.

For the conceptual model, read [The core story](docs/CORE_STORY.md). For detailed operations and
finding routes, read [Runbook](docs/RUNBOOK.md). Controlled terms live in [CONTEXT.md](CONTEXT.md).
