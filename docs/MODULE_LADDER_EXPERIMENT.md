# The module ladder: what each arm adds, what it can prove, and what blocks it today

The ladder is B0 through B7, each arm adding one thing. This document completes it with the
parts a ladder specification usually leaves out and that this repository has already paid to
learn — powering stated in advance, which comparisons are primary, what a null result would
mean, and which arms are *structurally incapable* of moving the metric they would be scored on.

Nothing here is a result. It is what has to be true before a result means anything.

## The ladder

| arm | adds | plane | can it change the answer? |
|---|---|---|---|
| B0 | nothing — empty search slot, minimal target definition, base tools | review | — (the baseline) |
| B1 | simple eval: evidence support, search coverage, temporal consistency, registry/spec discrepancy | evaluation | **no** |
| B2 | experience: note-type and keyword library | review | yes |
| B3 | search policy (one card in the search slot) | review | yes |
| B4 | experience + policy | review | yes |
| B5 | deterministic audit | audit | **no** |
| B6 | advanced eval | evaluation | **no** |
| B7 | eval-driven follow-up — the eval's findings trigger an action | review←evaluation | yes |

### The flaw in the ladder as written, and the fix

**B1, B5 and B6 cannot move accuracy, and scoring them on it guarantees a null.** An evaluator
that only reports does not change the answer — by design; `EVALUATION_RESULT` may mark, queue,
block a bundle or open a repair obligation, and may never rewrite a completed run. Only B7 closes
that loop.

So those three arms are measured on a different quantity: **issue discovery**, scored against
adjudicated issues. That means B1/B5/B6 are blocked on the adjudication set existing, and
reporting them on accuracy would produce three confident zeros that say nothing about the
modules.

State it this way instead:

- **Accuracy ladder** — B0, B2, B3, B4, B7. Primary metric: exact match on the answerable
  adjudicated subset.
- **Discovery ladder** — B1, B5, B6. Primary metric: true findings per run, and the
  known/novel/false-alarm split below. No accuracy claim.

## The five numbers, per arm, against B0

Reporting only "how many did it fix" hides the cost of an aggressive module.

| number | definition |
|---|---|
| error recovery rate | of B0's wrong answers, how many this arm gets right |
| **harm rate** | **of B0's correct answers, how many this arm gets wrong** |
| net improvement | recovery − harm |
| abstention improvement | where the corpus cannot answer, is this arm more willing to say so |
| extra cost | tool calls and tokens spent per error corrected |

`eval compare` already emits per-case REGRESSED/IMPROVED, which is the raw material for the first
three; `abstention_correctness` is the fourth; `usage` and `spend` are the fifth. What is missing
is assembling them into these five rows.

### A sixth number, new since the plan was written

**Compliance.** `evidence_chain` reports `grounding_ratio` and `max_depth` per run. For any arm
whose card asks for something observable, this separates two conclusions that used to be
indistinguishable:

> the policy does not help  vs  **the policy was never executed**

Measured, not hypothetical: the first version of the pointer instruction produced 0 of 18 links
across a whole run while the card asked for it plainly. Without a compliance column that arm
would have been written up as "no effect".

## Two subsets, different questions

**Answerable adjudicated subset** — evidence sufficient and a human decided. Scores the accuracy
ladder: exact match, evidence correctness, recovery/harm, cost.

**Uncertain / unanswerable subset** — the human could not decide either. Scores: correct
abstention, spec-ambiguity detection, absence of falsely confident answers, correct escalation.

A case that neither the reviewer nor the chart can settle is not a failure to be scored; folding
it into a single accuracy denominator destroys the only signal that says which one to fix.

## Stratify by corpus answerability

High / Moderate / Low, per target, from the availability profile. Each stratum reports the full
set above.

The expected shape is not uniform and the plan should say so before the run: in high-answerability
cases a retrieval policy can move accuracy; in low-answerability cases **the gain that matters is
fewer confidently wrong answers, not more right ones**. An arm that improves the second and not
the first has still improved.

## POWERING, STATED BEFORE ANY NUMBER EXISTS

The synthetic cohort is 21 patients. One chart moves every rate by roughly five points.
`MIN_PATIENTS_FOR_SUPPORT` is 20 in `assetdev.py`.

**A difference of one or two cases is NOT a result and must be reported as underpowered.** This
is written here, before the run, because the temptation to read a two-case gap as a finding
arrives only after seeing one — the same commitment `BFS_DFS_SEARCH_PILOT.md` made and then had
to honour when its spread came out 10/11/12/12.

Eight arms is 28 pairwise comparisons. **Declare the primary comparisons in advance**: every arm
against B0, paired per case. Everything else is exploratory and says so in the table.

## Pre-flight checks that are not optional

Each one exists because skipping it has already cost something.

1. **The corpus must be able to express the spec's value domain.** The first BFS/DFS attempt was
   abandoned five charts in: `STORE.400_522_523` declares `value_domain: icdo3_lung`, only two of
   twelve patients have a lung primary, and ten runs correctly returned `SPEC_INSUFFICIENT`. All
   arms would have scored identically badly and "traversal shape does not matter" would have been
   an artifact. **A spec gap is not a retrieval failure.** One query answers it before any spend.

2. **`SPEC_INSUFFICIENT` count must be zero or explained.** Non-zero means the arm is being scored
   on charts where the module cannot matter.

3. **The provider must be muzzled outside deliberate runs.** `tests/conftest.py` does this after a
   test was found making real paid calls while believing it had simulated a crash.

4. **Derived assets must not leak the answer.** Before B2 can be trusted, a keyword derived on the
   development set must be checked against the gold value and its notation variants — a term that
   IS the answer scores perfectly on dev and is worthless on test. The permutation control in
   `assetdev.certify` catches this statistically; the direct check does not exist yet.

## Anti-circularity

A case used to discover a problem may not be the main evidence that the fix works.

- **Discovery / Calibration** — see everything, design modules here. Usable afterwards for
  debugging and worked examples, never for the headline number.
- **Development / Regression** — tune, ablate, check for regressions.
- **Frozen Test** — run only after spec, experience, keywords, policies, audit rules and eval
  skills are all frozen. Findings here are recorded; they may not modify the system and then
  re-report the same test.

Every case carries `informed_module_design: true|false`. True means development only.

If N is too small for a clean split, use K-fold cross-fitting, with the rule that **a case's own
data may never have contributed to the experience, rules or skills active when it is scored.**

### Two generalisations, not one

1. **Error-pattern generalisation** — in *new* cases of the same shape, is the error rarer? Not
   "was the original case fixed".
2. **Novel-issue detection** — does the eval find corner cases nobody wrote down? Classify every
   test finding as **known issue type / novel issue type / false alarm**. This is what separates
   an open-ended auditor from a checklist.

### Verified acceptance, borrowed from SkillBoost (arXiv 2607.26643)

A card revision is accepted only if it stays within a regression bound — not merely if it shows
net gain. Its structured-exploitation half also argues for **section-level identity in the cards**,
so a regression can be attributed to the paragraph that caused it rather than to the whole card.
Neither is implemented.

## Recording a failure so it becomes a mechanism, not a patch

```
Observed failure / Evidence / Likely cause /
Proposed module / Expected mechanism / Cases used to derive it
```

The last line is what makes the anti-circularity rule enforceable.

## What blocks each arm today

| arm | state | blocked on |
|---|---|---|
| B0 | never recorded as a baseline, though every historical run used this configuration | nothing — **run it** |
| B1 | five eval cards and `--truth-mode` exist; no four-check simplified version | adjudicated issues to score against |
| B2 | no experience library at all | answer-leak filter, then a calibration pass |
| B3 | eight cards, four measured | a task with headroom; the SYNX charts supply it |
| B4 | — | B2 |
| B5 | six audit rules, coverage three-arm pilot has results | adjudicated issues |
| B6 / B7 | — | everything above |

And two things block the *design* rather than any single arm:

- **Corpus answerability profiling does not exist.** Without it the Low stratum cannot be formed,
  and "the agent found nothing" cannot be told from "this corpus could never have answered". On
  synthetic charts this is invisible because every chart is answerable by construction — which is
  exactly why the BFS/DFS pilot could skip the whole question.
- **No adjudication workflow.** Without it there is no answerable-adjudicated subset, so the
  accuracy ladder has no denominator on real data and the discovery ladder has nothing to score
  against.

## The order that follows from this

1. **B0 baseline, recorded.** Every arm compares to a floor that has never been measured.
2. **Answer-leak filter.** Cheap, deterministic, and B2 is untrustworthy without it.
3. **Corpus answerability profiling.** The one gap that makes every other number ambiguous.
4. **Evidence-pack builder** — turn the `REGISTRY_REFERENCE` attribution into a two-sided packet
   for a human. This is what makes adjudication affordable, and adjudication gates B1/B5/B6.
5. Then the ladder, on data where its numbers mean something.


---

# E1 + E2 results, 2026-08-01

Seven arms x eighteen charts (twelve base + six SYNX adversarial), `STORE.390`, seed 1234,
`gpt-5.6-luna`, 126 runs, about 70 minutes, roughly $1.6. B0 is the empty search slot.

| arm | exact | trap sprung | abstained | recovery | **harm** | net | grounding | max depth | calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0-base** | 15/18 | 1 | 2 | — | — | — | 0.49 | 4 | 10.1 |
| native | 15/18 | 2 | 1 | 1 | 1 | 0 | 0.54 | 4 | 11.5 |
| breadth-first | 15/18 | 1 | 2 | 0 | 0 | 0 | **0.74** | 4 | 11.3 |
| depth-first | 14/18 | 2 | 1 | 0 | 1 | −1 | 0.53 | 3 | 9.9 |
| breadth-then-depth | 15/18 | 2 | 1 | 1 | 1 | 0 | **0.70** | 4 | 11.3 |
| information-gain | **16/18** | 1 | 1 | 1 | 0 | **+1** | 0.55 | 3 | 10.4 |
| latest-first | 14/18 | 1 | 3 | 0 | 1 | −1 | 0.54 | **5** | 11.3 |

B0's three failures: SYN0002, SYNX02, SYNX06.

## What this establishes

**Nothing about accuracy, and that is the finding.** The whole spread is 14 to 16 out of 18. The
powering section above committed, before any number existed, to reporting a one- or two-case
difference as underpowered, and that is exactly what this is. `information-gain` at +1 net is one
chart.

**But the baseline itself is the result.** B0 — no search card at all — scores 15/18, matching or
beating five of the six cards. Every card measured before today was compared against a floor
nobody had measured. It turns out the floor is high, and on this task **no card has yet bought a
point**.

**The task is saturated.** Fifteen of eighteen correct leaves three cases of headroom, of which
one is an abstention and two are traps. That is the same ceiling problem `BFS_DFS_SEARCH_PILOT`
diagnosed on twelve charts, and adding the six adversarial charts did not fix it: the traps
sprang once or twice per arm, not systematically. **The SYNX charts are not hard enough for the
current agent**, so they do not supply the discrimination they were built for.

**Harm is real and would have been invisible.** Four arms have harm ≥ 1 — they turned a case B0
got right into one it got wrong — and three of those have net ≤ 0. Reported as accuracy alone,
`native` and `breadth-then-depth` look identical to B0 at 15/18; the paired view shows each
recovered one case and broke a different one. A ladder that reports only the headline would have
called them "no change".

## The one clean signal: grounding is driven by SHAPE, not by instruction

`breadth-first` 0.74 and `breadth-then-depth` 0.70 against 0.49–0.55 for everything else — and
**neither of those two cards was ever given the pointer instruction.** It went to `depth-first`
and `information-gain`, which came back at 0.53 and 0.55.

So the causal story is not the one the instruction predicted. A wide sweep produces steps that
each hang naturally off the inventory or a prior search; a lead-chase produces jumps whose origin
is harder to name, and telling the model to name it did not close the gap. This contradicts the
prediction made when the pointer landed — that `depth-first` would go deepest — and it is
falsifiable, which is more than the accuracy column offers.

`latest-first` reaches depth 5, the deepest of any arm, on a grounding ratio of only 0.54: it
builds one long chain rather than many short ones. That is exactly the navigation shape its card
describes, and it is visible in the trace rather than asserted.

## What to do next, from these numbers

1. **A harder task, not more charts.** STORE.390 is saturated at B0. `STORE.1860_1880.first_recurrence`
   has a FOUND key for 7 of 12 and requires separating a recurrence from the initial disease
   across a timeline. Adding charts to a saturated task adds cost, not power.
2. **Corpus answerability profiling.** Of the three residual failures, how many are "the corpus
   could never have answered"? Without the stratum that question cannot be asked, and it is the
   difference between "the policy does not help" and "this case has no answer to find".
3. **Do not add a ninth card.** Eight cards, seven measured today, none beating an empty slot.
