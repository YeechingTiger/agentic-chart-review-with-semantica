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

**`specs/STORE.390.date_of_initial_diagnosis.yaml`**, on all twelve synthetic patients.

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
  acr batch --spec specs/STORE.390.date_of_initial_diagnosis.yaml \
            --skills "search=$skill" --seed 1234 --max-usd 1.0 \
            --out "runs/pilot/${skill#search-}"
done

for skill in search-native search-breadth-first search-depth-first search-breadth-then-depth; do
  acr signal batch --kind rule --runs "runs/pilot/${skill#search-}"* \
                   --spec specs/STORE.390.date_of_initial_diagnosis.yaml \
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

Not yet complete. The first attempt was abandoned for the spec reason above; the `STORE.390`
run is in progress.
