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

```bash
for skill in search-native search-breadth-first search-depth-first search-breadth-then-depth; do
  acr batch --spec specs/STORE.400_522_523.site_histology_behavior.yaml \
            --skills "search=$skill" --seed 1234 --out "runs/bfs-${skill#search-}"
done

for skill in search-native search-breadth-first search-depth-first search-breadth-then-depth; do
  acr signal batch --kind rule --runs "runs/bfs-${skill#search-}" \
                   --spec specs/STORE.400_522_523.site_histology_behavior.yaml \
                   --out "signals/bfs-${skill#search-}.json"
done
```

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
| three-field exact match | the headline, and the only one that is the task |
| per-field match (site / histology / behavior) | a traversal can help one field and hurt another |
| correct abstentions retained | the failure mode this repo guards hardest: an arm that trades a correct EVIDENCE_INSUFFICIENT for a guess has made things worse at any accuracy |
| documents read per patient | the bill, and breadth's expected cost |
| search calls / read calls | which traversal actually ran, independent of what the skill said |
| open threads left unresolved | depth's expected advantage, measured directly |
| priced model cost | the other bill |
| caused-read fraction (`evals.detect_uncaused_reads`) | how much of each arm is causally legible afterwards |

## Powering, stated before the run

Ten patients. One chart moves every rate by ten points, so this pilot can detect a large
effect and nothing else. A difference of one or two cases is NOT a result and must be reported
as underpowered — `MIN_PATIENTS_FOR_SUPPORT` is 20 in `assetdev.py` for this reason. What ten
cases CAN do is expose a traversal that is grossly worse, and rule an arm out.

## Results

Not yet run.
