# Runbook

## 1. Install

```bash
uv venv --python 3.12
uv pip install -e ".[ledger,ledger-ui,dev,notebook]"
uv pip install --no-deps semantica==0.6.6
npm install -g @openai/codex@latest
codex --version
```

Semantica is pinned because the graph projection depends on its current persistence and decision
registry behavior. The lean profile disables embeddings and automatic NLP extraction. To use the
human review UI, install the Semantica Explorer Decisions extension as well:

```bash
tools/install_semantica_decision_review.sh
```

## 2. Start local Langtrace

Use Langtrace's official Docker Compose deployment:

```bash
git clone https://github.com/Scale3-Labs/langtrace.git
cd langtrace
docker compose up -d
```

Open `http://localhost:3000`, create a project, and generate its API key. See the official
[Docker Compose guide](https://docs.langtrace.ai/hosting/hosting_options/docker-compose). Deleting
the Compose volumes deletes the trace store; do not run `docker compose down -v` against evidence
you still need.

## 3. Configure providers

```bash
cp .env.example .env
# Fill values, then:
set -a
source .env
set +a
```

Required for a real run:

```text
ACR_API_BASE / ACR_API_KEY / ACR_MODEL
OPENROUTER_API_KEY
LANGTRACE_API_HOST / LANGTRACE_PROJECT_ID / LANGTRACE_API_KEY
```

`SEMANTICA_API_KEY` is additionally required for the local review UI. Keys belong only in the
environment. Never place them in a notebook cell, CLI argument captured in history, run artifact,
or Git.

## 4. Run the three-notebook walkthrough

Start Jupyter from the repository root:

```bash
.venv/bin/jupyter lab notebooks/
```

Run in order:

1. [01_run_chart_review_experiments.ipynb](../notebooks/01_run_chart_review_experiments.ipynb)
   creates or reuses the multi-run cohort. Existing real-provider evidence is reused by default.
   Set `ACR_TUTORIAL_MODE=live` to make new paid calls. `pilot` runs two cases × two arms × Luna;
   `full` additionally uses Terra as an acting review model.
2. [02_trace_to_decision_chain.ipynb](../notebooks/02_trace_to_decision_chain.ipynb) reads one
   explicitly selected analysis from protocol records through Langtrace, ReAct cycles, and the
   human Decision Chain. Override its input with `ACR_AUDIT_LEDGER` and `ACR_AUDIT_RUN_ID`.
3. [03_semantica_decision_intelligence.ipynb](../notebooks/03_semantica_decision_intelligence.ipynb)
   performs cross-run Semantica queries. Override its graph with `ACR_INTELLIGENCE_LEDGER`.

The `.executed.ipynb` companions are checked walkthroughs from persisted Luna/Terra runs. They make
no claim that the convenience cohort is a powered evaluation.

## 5. Run the closed loop from the CLI

```bash
acr run assets/specs/STORE.390.date_of_initial_diagnosis.yaml \
  corpus/patients/SYNX03 --out runs/mvp --task-arm policy_bundle

acr trace runs/mvp/<RUN_DIR>
acr reconstruct runs/mvp/<RUN_DIR> --ledger runs/mvp/ledger.json --passes 2
acr select-analysis <RUN_ID> <ANALYSIS_ID> --ledger runs/mvp/ledger.json \
  --selected-by reviewer --reason "two passes aligned"
acr chain <RUN_ID> --run-dir runs/mvp/<RUN_DIR> --ledger runs/mvp/ledger.json
```

The closed loop must pass these boundaries:

1. Codex completes a real chart review through the chart-only tool boundary.
2. Langtrace reads back the same trace and reports a complete export.
3. Deterministic replay builds a fixed, exhaustive ReAct skeleton.
4. Two Luna reconstructions agree on episode alignment.
5. One analysis is selected explicitly and projected to Semantica.
6. The human view has a conclusion, raw drill-down, and evidenced causal history where asserted.

## 6. Use decision-intelligence queries

```bash
acr insights <RUN_ID> --ledger runs/mvp/ledger.json
acr similar <EPISODE_ID> --ledger runs/mvp/ledger.json
acr causal-trace <EPISODE_ID> --ledger runs/mvp/ledger.json
acr impact <EPISODE_ID> --ledger runs/mvp/ledger.json
acr policy-impact <POLICY_ID> --from-version <OLD> --to-version <NEW> \
  --ledger runs/mvp/ledger.json
```

Authority limits:

- similarity returns comparison candidates, not clinical precedent;
- causal trace contains only explicit ACR assertions, not temporal adjacency;
- impact returns candidates, not counterfactual causation;
- policy impact returns historically direct-bound decisions to re-audit, not predicted answer
  changes or automatic non-compliance.

## 7. Open the human view

```bash
acr review-ui <RUN_ID> --run-dir /absolute/path/to/<RUN_ID> \
  --ledger /absolute/path/to/ledger.json --port 8877
```

Review in order. For a questionable step, inspect the runtime receipt and source refs, then the
cited clause and semantic entailment, then the raw Langtrace drill-down. A green ref-resolution
status proves that the reference existed; it does not prove that the rule entailed the choice.

## 8. Route a finding

| Observation | First route |
|---|---|
| inventory, query, or note-set choice was weak | retrieval behavior |
| similar scenarios repeatedly diverge without policy grounding | guideline question |
| receipt/testimony is absent, compound, or cross-episode | instrumentation |
| a cited clause resolves but does not entail the choice | rule application / guideline wording |
| correct rules and evidence still yield a bad judgment | model capability |
| changed Policy has historical bindings | targeted re-audit, then paired rerun if needed |

## 9. Verify the maintained surface

```bash
.venv/bin/python -m pytest tests/test_mvp_*.py -q
.venv/bin/ruff check src tests tools
.venv/bin/python tools/build_postdoc_notebooks.py

for notebook in notebooks/0*.ipynb; do
  .venv/bin/python -m json.tool "$notebook" >/dev/null
done

acr --help
```

The paid live test is opt-in. Ordinary tests use fixed traces and stub extractors to verify
behavior under reconstruction mistakes; the executed notebooks cover the real-provider boundary.
