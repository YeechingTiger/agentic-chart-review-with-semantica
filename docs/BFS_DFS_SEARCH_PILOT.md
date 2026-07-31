# Breadth / Depth / Combined Search Pilot

## Question

Does the SHAPE of the chart traversal change accuracy — sweeping wide, chasing leads, or
both — holding the clinical contract, the model, the seed and everything else fixed?

`SEARCH_PLANNING_PILOT.md` answered a different question: where the plan COMES FROM (the
model's own vs a dev-set-derived prior). It found the prior did not improve accuracy. Traversal
shape was never varied, and it is the axis DeepEvidence reports an ablation on — that neither
breadth nor depth alone was sufficient and only the combination was best. Whether that holds
for a chart, which is one patient's record rather than a federation of knowledge bases, is
an open question and this pilot is how it gets answered here.

## Arms

Four, differing in exactly one skill. The arm is IDENTIFIED by the skill it installs, so that
no name in this document has to be turned into another name to be run:

| Arm | `--skills` |
|---|---|
| `search-native` (control) | `search=search-native` |
| `search-breadth-first` | `search=search-breadth-first` |
| `search-depth-first` | `search=search-depth-first` |
| `search-breadth-then-depth` (combined) | `search=search-breadth-then-depth` |

Everything else is held: same spec, same corpus, same model, same `--seed 1234`, same
`--runtime-profile`, same cost and call ceilings, same `task` and `general` skill slots.

## Which spec, and why not the obvious one

**`assets/specs/STORE.390.date_of_initial_diagnosis.yaml`**, on all twelve synthetic patients.

The first attempt used `STORE.400_522_523.site_histology_behavior`, and it had to be abandoned
five charts in. That spec declares `value_domain: icdo3_lung` — eight lung topography codes and
fifty-five lung morphologies. **Two of the twelve synthetic patients have a lung primary**, and
one of those two has a key status of `EVIDENCE_INSUFFICIENT`. So an exact match was structurally
possible on exactly ONE chart.

The other ten returned `SPEC_INSUFFICIENT`, and they were right to. The agent found the answer
every time — its own words on three of them were "pancreatic head adenocarcinoma", "left breast
upper-outer-quadrant infiltrating ductal carcinoma", "urinary bladder urothelial carcinoma in
situ", all three matching the key — and then reported that the contract could not express it.
That is the channel working: `SPEC_INSUFFICIENT` is a claim about the specification, not about
the chart.

**A spec gap is not a retrieval failure, so those charts cannot discriminate between traversal
arms.** No amount of better searching fixes a value domain. Had the pilot run to completion it
would have produced a table in which all four arms scored identically badly, and the honest
reading of that table — "traversal shape does not matter" — would have been an artifact of
asking the question on charts where it could not be answered.

`STORE.390` has a key for all twelve, all `FOUND`, and no value domain at all: the answer is a
date. It is also a genuine retrieval problem of the kind the arms differ on — the rule is *the
FIRST date of diagnosis, whether clinically or histologically established*, so a run has to find
the earliest establishing document rather than the most definitive one, and `SYN0001`'s designer
note says it was written to exercise exactly the cytology-precedes-pathology boundary.

The general lesson, which cost about $0.30 to learn: **check that the corpus can express the
spec's value domain before spending on arms.** One command would have said so —

```bash
PYTHONPATH=src .venv/bin/python -c "
import json, glob
for p in sorted(glob.glob('corpus/patients/SYN*/_ground_truth.json')):
    g = json.load(open(p))
    k = g['ground_truth']['STORE.400_522_523.site_histology_behavior']
    print(g['patient_id'], k.get('primary_site'), k.get('status'))
"
```

## Protocol

The loop iterates the SKILL NAMES, not arm nicknames, and derives the output directory from
the skill rather than the other way round. An earlier draft ran over short arm names and built
`search=search-$arm`; that only ever emits a real skill while every nickname happens to equal
its card name minus the prefix, and the arm called "combined" in the table above does not.
A wrong run directory is a tidiness problem; a wrong skill name is a refused run at best and a
silently different experiment at worst.

The list is spelled out in each loop rather than held in a variable: `for skill in $ARMS` word-
splits under bash and does NOT under zsh, which is the shell on the machines this gets pasted
into, and there it would hand `acr` one arm whose name is all four names.

Output goes under `runs/`, which `.gitignore` covers. An earlier attempt wrote to `runs-pilot/`,
which it does not cover — that breaks the tree's own rule that run output never enters git, and
it also stamps every run id `-dirty`, so the provenance of the whole pilot records a modified
tree that was only modified by the pilot's own output.

```bash
for skill in search-native search-breadth-first search-depth-first search-breadth-then-depth; do
  acr batch --spec assets/specs/STORE.390.date_of_initial_diagnosis.yaml \
            --skills "search=$skill" --seed 1234 --max-usd 1.0 \
            --out "runs/pilot/${skill#search-}"
done

for skill in search-native search-breadth-first search-depth-first search-breadth-then-depth; do
  acr signal batch --kind rule --runs "runs/pilot/${skill#search-}"* \
                   --spec assets/specs/STORE.390.date_of_initial_diagnosis.yaml \
                   --out "signals/pilot-${skill#search-}.json"
done
```

Note the `*` in the second loop: `--out runs/pilot/native` lands in
`runs/pilot/native__<timestamp>__<sha>`, because `cli_common.unique_run_dir` stamps the name so
two runs of one command cannot overwrite each other. A scorer that globs the literal arm name
finds nothing and reports it as an arm that produced no output.

### Where output may live, which is not the same answer for every command

`acr batch` and `acr eval score` are happy with `runs/` inside the worktree. `acr signal
--kind judge` and `--kind agent` are NOT: they go through `LocalArtifactStore`, which refuses a
root that is relative *and* refuses one that resolves inside the Git worktree —

```
Invalid value: local artifact root resolves inside the Git worktree: …/runs/pilot/native__…
```

That is the "patient-derived artifacts never enter the repo" rule enforced in code rather than
left to `.gitignore`, and it is stricter than ignoring the directory. On a server the root is
the external run tree (`/N/project/computable_phenotype/llm/run/`, per README §3). Locally, the
manifest and its `.jsonl` have to be copied somewhere outside the checkout before either
model-calling kind will look at them. Worth knowing before you schedule a cohort and discover it
at the scoring step.

Verify the four skill names resolve before spending anything. `--skills` is parsed and
validated before the first model call, so a typo costs nothing at run time — but it costs a
scheduling round trip, and this check is free:

```bash
PYTHONPATH=src .venv/bin/python -c "
from acr.skills import skill_slot
for s in 'search-native search-breadth-first search-depth-first search-breadth-then-depth'.split():
    assert skill_slot(s) == 'search', s
print('four arms resolve into the search slot')
"
```

The second loop needs the `acr signal` group, which lands with the batch signal entry point;
until then, score the four run directories with whatever deterministic scorer is current. No
judged opinion enters the table either way.

## What is measured

Primary, from the deterministic scorer only — a judged opinion decides nothing here:

| Metric | Why it is in the table |
|---|---|
| exact date match | the headline, and the only one that is the task |
| wrong date | answered, and answered incorrectly — the failure a better traversal should remove |
| abstained where the key has a date | the OTHER failure, and not the same one: it costs an answer without asserting a wrong one |
| correct abstentions retained | the failure mode this repo guards hardest: an arm that trades a correct `EVIDENCE_INSUFFICIENT` for a guess has made things worse at any accuracy |
| `SPEC_INSUFFICIENT` count | a spec gap is not a retrieval failure. If this is non-zero the arm is being scored on charts where traversal cannot matter — see the spec section above |
| documents read per patient | the bill, and breadth's expected cost |
| read calls | which traversal actually ran, independent of what the skill said |
| model calls | the other bill |
| caused-read fraction (`evals.detect_uncaused_reads`) | how much of each arm is causally legible afterwards |

## Powering, stated before the run

Twelve patients. One chart moves every rate by more than eight points, so this pilot can detect
a large effect and nothing else. **A difference of one or two cases is NOT a result and must be
reported as underpowered** — `MIN_PATIENTS_FOR_SUPPORT` is 20 in `assetdev.py` for this reason.
What twelve cases CAN do is expose a traversal that is grossly worse, and rule an arm out.

Stated here, before any number exists, because the temptation to read a two-case gap as a
finding arrives only after seeing one.

## Results

Run 2026-07-31, twelve synthetic patients per arm, `--seed 1234`, `gpt-5.6-luna` via
OpenRouter, `STORE.390` spec hash unchanged across arms, `task` and `general` skill slots
identical across arms. Accuracy from `acr eval score`; operational columns read from the
manifests and traces; cost priced post hoc from recorded `usage` against `assets/pricing/prices.json`
(the runs predate that table, so their own `spend.usd` is null).

| | native | breadth-first | depth-first | breadth-then-depth |
|---|---:|---:|---:|---:|
| exact match | 11/12 (91.7%) | **12/12 (100%)** | 10/12 (83.3%) | **12/12 (100%)** |
| abstained where the key has a date | 1 | 0 | 1 | 0 |
| gate-valid | 12/12 | 12/12 | 12/12 | 12/12 |
| `SPEC_INSUFFICIENT` | 0 | 0 | 0 | 0 |
| searches / patient | 10.5 | **31.8** | **8.6** | 22.4 |
| documents opened / patient | **22.4** | 8.6 | 9.2 | **7.8** |
| read calls / patient | 4.2 | 2.7 | 3.2 | 3.3 |
| tokens (mean) | 258k | 223k | **195k** | 257k |
| USD / patient | 0.0148 | 0.0122 | **0.0106** | 0.0139 |
| caused-read fraction | 1.00 | 1.00 | 1.00 | 1.00 |
| `evidence_span_overlap` findings | 1 | 6 | 1 | 4 |

Total spend across all 48 runs: $0.62.

### What this pilot DID establish

**The traversal skills change behaviour, sharply and in the direction each card describes.**
Breadth-first issues 3.7× the searches of depth-first (31.8 vs 8.6) and opens a third of the
documents native does (8.6 vs 22.4). Depth-first is the cheapest arm on every operational axis.
Those gaps are large, monotone across the arms that share a mechanism, and far outside anything
twelve charts could produce by chance. Before this run, `_PROFILE_SKILLS` was an empty dict and
no recorded experiment had ever varied the guidance a run received — so the first thing worth
knowing is that the slot mechanism does what it claims.

**A behavioural signature falls out of it.** `evidence_span_overlap` fires on 6 of 12
breadth-first runs against 1 of 12 for native and depth-first. That is what sweeping wide looks
like in the ledger: one passage surfaced by several search terms and recorded more than once.
The arms differ in a way an audit can see without being told which arm it is reading.

### What this pilot did NOT establish, and cannot

**Which traversal is more accurate.** The whole spread is two cases — 10, 11, 12, 12 out of 12.
The powering section above committed, before any number existed, to reporting a one- or two-case
difference as underpowered, and that is what this is. `eval compare` calls depth-first a
REGRESSION against native, and its per-subgroup output is worth reading for what it is:

```
REGRESSED SYN0004  EXACT -> MISMATCH
REGRESSED SYN0005  EXACT -> ABSTAINED_MISSED
SUBGROUP pattern:documented recurrence after a disease-free interval  1.0 -> 0.0  (n=1)
SUBGROUP pattern:in situ disease — behavior 2                         1.0 -> 0.0  (n=1)
```

Two subgroups of ONE going from 1.0 to 0.0 is the same two cases counted a second way, not
independent corroboration. The tool is right to flag it — a subgroup collapse is exactly what
an aggregate hides, and it must surface — but "n=1" is doing all the work in those rows.

**Whether the combination beats either alone.** `breadth-then-depth` matched `breadth-first` at
12/12 while spending 15% more per patient. On twelve charts with one case of headroom that is
consistent with the DeepEvidence ablation, with the opposite, and with neither.

### The ceiling problem, and what to run next

Native already scores 91.7%. **There is one case of headroom in the entire cohort**, so this
design can detect an arm that is grossly worse and essentially nothing else — which is what it
did. Two ways forward, and the first is cheaper:

1. **A task with headroom.** `STORE.1860_1880.first_recurrence` has a FOUND key for 7 of 12 and
   is a harder retrieval problem (a recurrence has to be distinguished from the initial disease
   across a timeline). `STORE.400_522_523` is unusable here for the value-domain reason above.
2. **More charts.** `MIN_PATIENTS_FOR_SUPPORT` is 20 in `assetdev.py`, and at ~$0.013 per
   patient-arm, 40 patients × 4 arms is about $2. The corpus generator is `tools/generate_corpus.py`.

Until one of those runs, the honest summary is: **the mechanism works and is measurable; the
accuracy question is open.**
