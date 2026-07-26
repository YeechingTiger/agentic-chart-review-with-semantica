# Running this on a server

Everything needed is in the repo: code, specs, and the synthetic corpus (12 patients, 3,736
documents, PHI-free, committed deliberately so a fresh clone reproduces the experiments).
The only external dependency is a model endpoint.

The bottleneck on the laptop was inference, not the agent. Measured on qwen3.6:35b under
Ollama: **3.5s per act step at 3.5k tokens rising to 45s at 10k**, so a single 30-step run
took 15–25 minutes and the four-arm ablation ran sequentially for over an hour. On a GPU
server with vLLM the same work should drop by roughly an order of magnitude, and the arms can
run concurrently instead of one at a time.

---

## 1. Get the code

The repo is **private**. Pick whichever auth the server already has.

```bash
# GitHub CLI (easiest if gh is installed and logged in)
gh repo clone YeechingTiger/agentic-chart-review
cd agentic-chart-review

# or SSH, if the server's key is on the GitHub account
git clone git@github.com:YeechingTiger/agentic-chart-review.git

# or HTTPS with a fine-grained PAT (Contents: read)
git clone https://<TOKEN>@github.com/YeechingTiger/agentic-chart-review.git
```

## 2. Install

Python **3.12** or newer. `uv` is the fastest route; plain venv works too.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # if uv is absent
uv venv --python 3.12
uv pip install -e ".[dev]"

./.venv/bin/python -m pytest -q                     # expect: 53 passed
```

**Always call `./.venv/bin/python` directly.** A bare `python3` is often 3.9 and cannot
import `acr`; a bare `python3.12` usually has no pytest. Both failures look like a broken
checkout.

The corpus is already in the tree. To regenerate it (deterministic, byte-identical):

```bash
./.venv/bin/python tools/generate_corpus.py --out corpus/patients
```

## 3. Serve the model

### vLLM — recommended on a GPU box

**Size it from the hardware, don't copy the flags below.** Check `nvidia-smi` first and pick:

| Total VRAM | What fits | `--tensor-parallel-size` |
|---|---|---|
| ≥ 80 GB on one card | bf16 (~70 GB weights + KV cache) | 1 |
| 2 × 40–48 GB | bf16 split across both | 2 |
| 4 × 24 GB | bf16 split across four | 4 |
| 1 × 40–48 GB | AWQ or GPTQ 4-bit (~20 GB) | 1 |
| < 24 GB | use Ollama, or a hosted API | — |

`--tensor-parallel-size` must divide the number of visible GPUs. Leave headroom for KV
cache: at `--max-model-len 32768` budget several GB per concurrent request, and lower
`--gpu-memory-utilization` (default 0.9) if the server shares the card with anything else.

```bash
uv pip install vllm
vllm serve Qwen/Qwen3.6-35B \
  --port 8000 \
  --tensor-parallel-size 2 \
  --max-model-len 32768 \
  --enable-auto-tool-choice --tool-call-parser hermes
```

`--enable-auto-tool-choice` is **required** — the agent is built on tool calling and will do
nothing useful without it.

```bash
export ACR_MODEL="hosted_vllm/Qwen/Qwen3.6-35B"
export ACR_API_BASE="http://localhost:8000/v1"
```

### Ollama — simpler, slower

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull qwen3.6:35b
export ACR_MODEL="ollama_chat/qwen3.6:35b"
```

### A hosted API instead

LiteLLM means no code changes, only the model string.

```bash
export ACR_MODEL="openai/gpt-4.1"        ; export ACR_API_KEY=sk-...
export ACR_MODEL="anthropic/claude-sonnet-4-5" ; export ACR_API_KEY=sk-ant-...
```

## 4. Check it works before committing to a long run

```bash
./.venv/bin/acr patients                 # 12 patients listed
./.venv/bin/acr specs                    # 4 specs with freeze hashes
./.venv/bin/acr chart SYN0002            # document-type summary

# one run, verbose — should finish in a couple of minutes on a GPU box
./.venv/bin/acr run SYN0002 \
  --spec specs/STORE.400_522_523.site_histology_behavior.yaml \
  --max-steps 30 --reflect-every 3 --seed 1234 --out runs/smoke
```

Read `degradation` in the manifest first. Any non-zero entry means a node silently fell back
and the corresponding behaviour was not exercised — conclusions drawn from that run about
planning, reflection or finalisation do not hold.

```bash
./.venv/bin/python -c "
import json,glob; d=json.load(open(glob.glob('runs/smoke__*/*.manifest.json')[0]))
print('status              ', d['answer'].get('status'))
print('basis               ', d['answer'].get('proof_basis') or d['answer'].get('negative_basis'))
print('gate_validated      ', d['gate_validated'])
print('steps_to_gate_pass  ', d['steps_to_gate_pass'])
print('degradation         ', d['degradation'])
print('tokens / seconds    ', d['usage']['total_tokens'], d['elapsed_s'])"
```

## 5. The ablation

Four runs: two specs (stratified vs not) × two patients (SYN0002 whose correct answer is
EVIDENCE_INSUFFICIENT, SYN0001 whose correct answer is FOUND). Same seed throughout, or the
arms are not comparable.

```bash
S=specs/STORE.400_522_523.site_histology_behavior.yaml
U=specs/ablation/STORE.400_522_523.unstratified.yaml
for pt in SYN0002 SYN0001; do
  ./.venv/bin/acr run $pt --spec $U --max-steps 30 --reflect-every 3 --seed 1234 --out runs/aprime_$pt
  ./.venv/bin/acr run $pt --spec $S --max-steps 30 --reflect-every 3 --seed 1234 --out runs/b_$pt
done
```

With enough GPU memory these can run concurrently — append `&` and `wait`. Keep an eye on
whether vLLM is batching them, since contention will distort the timing comparison even
though it leaves the labels alone.

The single thing to look for: **does any run report `gate_validated: true`?** No run ever has.
Every answer so far came out of budget exhaustion or, before it was fixed, a bypass — labels
happened to be right while the system had reached none of them. One `GATE_VALIDATED` would be
the first end-to-end auditable determination this system has produced.

## 6. Things that will bite

**Run directories are `runs/<label>__<UTC>__<code-sha>/` and are never reused.** A
configuration under different code is a different experiment. Never `rm -rf runs/…` — use
`tools/archive_runs.sh`, which moves rather than deletes. One batch was already lost this
way and two of its traces can never be regenerated because the code path that produced them
was removed; see `runs/_archive/NOTES.md`.

**Tool calling must be on.** Without `--enable-auto-tool-choice` on vLLM the agent produces
text and calls nothing, which looks like a hung run rather than a misconfiguration.

**Reasoning models need output headroom.** `max_tokens` defaults to 4096 because qwen3.6
spent 1394 completion tokens of thinking to emit a 48-character answer; at 1536 the thinking
consumed the budget, the text channel came back empty, and the planner silently ran on a
one-line stub for every run before it was caught. Do not trim this on the theory that a short
answer needs a small budget.

**Context length.** `num_ctx` / `--max-model-len` of 32768 is the tested setting. Act-step
prompts reached ~10k tokens on a 293-document chart; a larger corpus will push higher.

**Real data never enters this repo.** The committed corpus is fabricated. Point `--corpus` at
a path outside the tree for anything real; `.gitignore` already excludes `corpus_real/` and
`data_real/`.

## 7. Pointing at a different corpus

`src/acr/corpus.py` is the only module that knows the on-disk layout. It expects one flat
directory per patient with filenames `<DocType>_<YYYY-MM-DD>[__<n>].txt`. A different backend
means replacing that one module; nothing else changes.

```bash
./.venv/bin/acr run <PATIENT> --spec <SPEC> --corpus /secure/path/patients
```

---

## Appendix — instruction block for an agent doing the setup

Paste this to a coding agent on the target machine. It is self-contained: it says what to
detect, what to decide, and what to report back.

```text
Deploy and run the agentic chart review project in this repo.

SETUP
1. Read DEPLOY.md and RESULTS.md first. RESULTS.md says what has and has not been
   established; do not re-derive it.
2. Install: uv venv --python 3.12 && uv pip install -e ".[dev]"
   Verify with ./.venv/bin/python -m pytest -q — expect 53 passed.
   Always invoke ./.venv/bin/python directly; a bare python3 is usually 3.9 and cannot
   import acr, which looks like a broken checkout.
3. The corpus is committed (12 synthetic patients, 3736 documents). Do not generate data.

MODEL — detect the hardware and choose the configuration yourself
4. Run nvidia-smi. Using the sizing table in DEPLOY.md, decide between bf16 and a 4-bit
   quantisation, and set --tensor-parallel-size to divide the visible GPU count. Serve
   Qwen/Qwen3.6-35B with vLLM. --enable-auto-tool-choice is mandatory: without it the agent
   emits text, calls no tools, and the run looks hung rather than misconfigured.
   Fall back to Ollama if there is no usable GPU.
5. Export ACR_MODEL and ACR_API_BASE to match whatever you started, then smoke-test with a
   single ./.venv/bin/acr chart SYN0002 before running anything long.

FIRST TASK — this is the point of the exercise
6. Run exactly this one and stop:
     ./.venv/bin/acr run SYN0002 \
       --spec specs/STORE.400_522_523.site_histology_behavior.yaml \
       --max-steps 30 --reflect-every 3 --seed 1234 --out runs/b_SYN0002

   It exercises the stratified proof obligation with forced validation sampling — the only
   core mechanism never yet shown to be satisfiable. The unstratified obligation already
   passes in 18 steps (see RESULTS.md); this one is unknown.

7. Report: status and basis, gate_validated, steps_to_gate_pass, rejections, the degradation
   block, suspected_recognition_failures, tokens and elapsed, and the per-stratum figures
   from coverage_attested (how many drawn, how many read, any sample hits).

   Read degradation FIRST. Any non-zero entry means a node silently fell back and that run's
   conclusions about planning, reflection or finalisation do not hold — say so and stop
   rather than continuing.

THEN, only if the first run is clean
8. The four-arm ablation: two specs (specs/STORE.400_522_523.site_histology_behavior.yaml
   and specs/ablation/STORE.400_522_523.unstratified.yaml) crossed with two patients
   (SYN0002, correct answer EVIDENCE_INSUFFICIENT; SYN0001, correct answer FOUND). Same
   --seed 1234 throughout or the arms are not comparable. Concurrency is fine if VRAM allows,
   but note in the report that it happened, since contention distorts timing even though it
   leaves labels alone.

TWO RULES THAT ARE NOT NEGOTIABLE
- Never rm -rf anything under runs/. Use tools/archive_runs.sh, which moves rather than
  deletes. A batch was already lost that way and two of its traces cannot be regenerated,
  because the code path that produced them no longer exists.
- Do not lower max_tokens from 4096. qwen3.6 spends ~1394 completion tokens thinking to emit
  a 48-character answer; at a smaller budget the text channel returns empty and the planner
  silently degrades to a one-line stub for every run.
```
