# The candidate ledger: two attempts, both falsified, and the numbers

Removed 2026-08-03. The behaviour it existed to produce is now asked for in prose —
`assets/skills/policy-hypothesis-set/SKILL.md` — and enforced nowhere.

This file exists so nobody rebuilds it without knowing what has already been measured. Both
approaches are falsified on their own terms; a third would need a different premise, not a better
implementation of one of these.

## What the thing was for

Every contract in this tree arbitrates between readings. STORE.390 has four `conflict_rules` and
every one of them decides between two dates that could each be the answer. Nothing in the runtime
represented "a reading I have not yet ruled out", so those rules could only be followed in prose,
and when they were not followed nothing could say which reading was dropped or why.

## Attempt 1 — a skill card

`tactic-counterevidence` already says the target thing in plain words: *name the most plausible
alternative value explicitly, then search for what would establish THAT*. Run paired against
twelve charts, same seed, same profile, card on versus off:

| | off | on |
|---|---|---|
| distinct values submitted per run | 1.00 | 1.00 |
| runs that ever submitted two | 0/12 | 0/12 |
| reasoning mentions an alternative | 4/12 | **1/12** |
| words of reasoning | 99 | 111 |
| accuracy, informed charts | 5/6 | 3/6 |

Asking for it made the model write more about the answer it already had, and mention alternatives
LESS. n is small; the direction is not the point — the point is that nothing moved on the
mechanism the card names.

## Attempt 2 — a typed ledger and an independent reasoner call

A `Candidate` dataclass, a `CandidateLedger`, an answerability axis kept apart from the values,
explicit conflict sets, structured discriminators, five invariants, and a separate structured
model call whose only output was candidate updates and which had no chart tools, no gate and no
way to submit.

**It worked.** Frozen evaluation, 14 charts x 3 repeats, protocol locked before the first call:

| | clear | competing |
|---|---|---|
| gold-answer survival | 80% | 83% |
| rejection precision | 90% | 100% |
| rejections citing a real contract rule | 41% | 51% |
| answerability correct | 80% | 88% |
| candidate precision | 63% | 82% |
| **false competition** | **40%** | — |
| conflict set formed | — | 38% |

And when it said a question was ALREADY_RESOLVED it was telling the truth: 13 resolved-claims, 13
correct selections, zero cases of declaring a question settled and then picking wrong.

**What it never did** was hold two competing VALUES. Across 13 candidates in Phase A, not one pair
was value-against-value; every second candidate was an abstention.

## Attempt 2b — a mechanical seeder, to give the reasoner a set to compare

Extract every type-compatible value from the recorded evidence and seed them all; let the reasoner
prune with recorded reasons. Deliberately over-inclusive, on the argument that a recorded
rejection beats a value nobody listed.

Deterministic: 42/42 replays identical, 42/42 matching what each run recorded. Gold recall 100%
clear, 82% competing. `DOCUMENT_DATE` had a marginal recall of 19 — gold candidates no other
source found.

**And it was wrong in three ways.**

1. **It only functioned on dates.** Of five contracts, two declare a field the extractor
   recognised and in both cases that field is a date. The other three no-opped SILENTLY and wrote
   `n_declared: 0`, which in a manifest is indistinguishable from "the model found nothing".
2. **Over-inclusion cost answers.** All three SYN0002 runs rejected the gold answer as "the
   progress note's own service date" and answered CORPUS_INSUFFICIENT. Seeding every document
   date had taught the reasoner that document dates are noise; on that chart the document date IS
   the answer.
3. **The principle was an artifact of one value-space shape.** A date's value space is unbounded,
   so its candidates must be found in the record. A categorical target with
   `value_domain: icdo3_lung` has a declared table — seeding everything there means seeding
   several hundred codes, and the work is subset SELECTION, not extraction. A stage field's
   `format: cT(X|0|is|1mi|...)` is itself the value domain. Enumerate-then-prune was never
   general.

## The finding that outlived all of it

Every wrong answer in the valid batch, attributed to whether more searching would have helped.
The decisive document is grounded in an outcome — what runs of the same chart that got the answer
RIGHT chose to cite.

```
11 wrong answers in 42 runs
more searching WOULD have helped   0 / 11
more searching would NOT have      7 / 11   (4 more reachable by neither)

CITED_BUT_MISJUDGED  4   off by 2-6 days, having cited the right document
GOLD_REJECTED        3   SYN0002, all three runs
UNSEEDABLE           4   constructed values, written in no document
NEVER_LOOKED         0
READ_NOT_CITED       0
```

**Not one failure was a retrieval failure.** In every failing run the agent opened the document
that carries the answer. What it got wrong was reading it. So a Strategic Controller — which
decides whether to keep searching — has no job on this variable, and the architecture ordering
that puts one next is answering a question this variable does not ask.

## Two defects it found in code that stayed

- `20150099` shipped with `gate_validated: True`. It violates the contract's declared format —
  month `00` — and was recorded as `answer_shape_miss` with `refused: False`. The format check was
  left advisory because it had once destroyed values that were RIGHT in another notation
  (`C34.9` versus `C341`). `20150099` is right in no notation. The advisory/refusing line is drawn
  in the wrong place.
- A `TypeError` in the reasoner killed 35 of 42 runs in the first frozen evaluation, and the trace
  recorded only `str(e)`. `runtime_error` now carries the traceback.

## What was kept

`Evidence.evidence_id`; the corpus's `gold_candidates` / `gold_rejections` / `candidate_stratum`
annotations on 14 charts, which are hand-authored knowledge about which readings are defensible
and are the input any future approach needs; and `tools/measure_agency.py` and
`tools/measure_policy_value.py`, which read manifests and never depended on the ledger.
