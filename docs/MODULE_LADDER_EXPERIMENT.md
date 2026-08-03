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

**Implemented 2026-08-03, and this paragraph was a wish for a week before it was.** The flag was
named here and existed in no code, so every number this document reports was computed over
charts that were designed by watching runs fail — SYNX01-06 — while the cards being scored were
written from those same failures. `_ground_truth.json` now carries `informed_module_design` and
`designed_from` for all 27 charts; `tools/analyze_arms.py` counts the two populations apart and
refuses to present a headline number over an all-informed batch; `tools/answer_key_from_corpus.py`
emits `held_out:` as a subgroup so `eval compare` fails an arm that gains only where its cards
came from. Six held-out charts exist (SYNY01-06), each derived from a contract clause no other
chart exercises and from no run result.

**Every ladder number recorded above this line is on the informed population.** They are kept
because they are what happened; they are not evidence about a card, and re-running the ladder
against SYNY is the first thing that would produce any.

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

---

# E3 — Is the cost of a card in its search rule or in its stopping rule?

**Pre-registered before the run. 2026-08-02.**

## Where this question came from

E2 left "the cards lose to an empty slot" unexplained. The first explanation offered — that the
cards fail to invent retrieval terms — was **measured and refused**: counting every search term
against the spec's five required ones, every card invents terms, and most invent MORE than B0.

| arm | 搜索/人 | 自创词/人 | 占比 |
|---|---|---|---|
| B0-base | 7.4 | 2.4 | 32% |
| native | 9.7 | 4.7 | 48% |
| depth-first | 8.2 | 3.4 | 41% |
| information-gain | 9.0 | 4.1 | 45% |

(`breadth-first`'s 18% is not comparable — a pipe-joined multi-term query counts as one term
under the new tool surface. Measurement artefact of mine, not a property of the arm.)

So the loss is not in term generation. On SYNX05 specifically, `depth-first` issued four
searches, all required terms, zero invented — on a chart where it invents freely elsewhere. It
stopped. And its card told it to: *"Follow one thread until it resolves. A resolved thread is
worth more than three half-followed ones."* The 2019-02-15 pathology report resolves the thread
and is the declared `naive_answer` bait.

The GOLD-mode diagnosis agrees without being told this: *"the run chose the earliest qualifying
date **among the later documents it saw**"* — the failure is in what it stopped before reading.

**Hypothesis: a policy is as much a stopping rule as a search rule, and the stopping rule is
where these cards pay.** An empty slot has no satisfaction criterion, so nothing can satisfy it.

## Manipulation

One variable: the section that states when the traversal is complete, deleted. Nothing else.
The cut was made by script so the removed bytes could be printed and checked, and the diffs are
pure deletions (plus the frontmatter no longer promising the removed content).

| card | removed | bytes |
|---|---|---|
| `search-depth-first` | "How far, and when to stop following" | 318 |
| `search-breadth-first` | "When the sweep is done" | 373 |
| `search-information-gain` | "When the highest-gain action is to stop" | 445 |

`depth-first`'s cut initially also took the anti-circling rule ("never revisit a document you
have already read"), which is a loop guard and not a completion criterion. It was put back under
its own heading. Had it stayed out, the arm would have varied in two things at once.

**Six arms, not three.** `depth-first` and `information-gain` were measured on the OLD single-keyword
tool surface; comparing them to a variant on today's surface would confound the manipulation with
the tool change. Both members of each pair are re-run today.

## Predicted direction — and the arm that is the control

The three criteria are not of one kind, and that is what makes this testable rather than a
guess that shorter cards do better:

- **`depth-first`, `information-gain` — criterion satisfiable EARLY** ("the thread resolved",
  "no action would change the answer"). Removing it predicts **more documents read per patient**.
- **`breadth-first` — criterion is DEMANDING** ("every type searched or explicitly excluded").
  Removing it predicts **fewer searches**, and if anything fewer reads. **Opposite direction.**

So if all three nostop variants improve together, the effect is *less text*, not *less stopping*,
and the hypothesis is refuted even by a favourable-looking accuracy table.

## Endpoint, and what this run cannot decide

**Primary: documents read per patient, and whether the establishing document was read at all.**
Not accuracy. 18 binary outcomes cannot resolve the 1–2 chart differences at issue — the same
pre-committed clause that made E2's 15→16 uninterpretable applies here, and stating the endpoint
afterwards is how that clause gets evaded. Accuracy is reported as description.

**Falsifier:** if reads/patient does not rise for `depth-first` and `information-gain`, the
early-stop explanation is wrong and SYNX05 needs another cause.

---

# E4 — The floor under E1, E2 and E3

**Pre-registered before the run. 2026-08-02.** E3 is suspended until this returns.

## What was wrong with every experiment above

`B0-base` is described throughout this document as "the empty search slot" and treated as the
unassisted floor. It is not empty. Every arm — B0 included — receives a retrieval plan before
its first search, built from the spec's hand-written strata:

```
B0-base on SYNX05   source=spec_strata
  read_all = [Onc-Med-MD-OP-Progress-Note, Surgical-Pathology-Report]
  search   = [... Endo-Diab-MD-OP-Progress-Note ...]
  initial_keywords = [diagnosis, diagnosed, carcinoma, biopsy, malignancy]
```

`read_all` is the two documents carrying SYNX05's bait answer 20190215. The type carrying the
real one is demoted to `search`. Neither `pancreatic` nor `cancer` — the two terms that
distinguish the runs that answer this chart correctly — is in the list.

There is a second layer in the prompt itself: `as_prompt_block(view="full")` prints
`required searches:`, `document types that must be reviewed:` and `SEARCH HINTS`.

So E1/E2/E3 measured **card on top of a prior** against **prior alone**. "No card beats every
card" is true and says less than it appeared to: the prior was never on the other side of a
comparison. It is precisely the kind of asset `assetdev.certify` exists to certify — a claim
about where to look — and being hand-written YAML it bypasses the permutation control and the
answer-leak filter both. No development set was ever involved.

`coverage_planner.plan_from_spec` states this in its own docstring: *"That is the arm the
develop plane wants to falsify."* It was designed to be the thing under test. It became the
ground under every test instead.

This is the same error as E1's, one level down. E1 found that every card had been compared
against an unmeasured floor. That floor was standing on an unmeasured prior.

## The ablation already existed

`guideline-only` — `search_terms=()`, `required_strata=()`, and `uses_clinical_contract_view`
true so the prompt hides retrieval detail. Described in the registry as *"Clinical task contract
only: no task keywords, note-type priors, or negative coverage proof."* Built, shipped, never
run against the stratified profile on STORE.390.

| arm | profile | plan source |
|---|---|---|
| `prior` | `current-stratified-coverage` | `spec_strata` |
| `floor` | `guideline-only` | `patient_inventory_only` |

Both with `--skills search=`. No card in either. One variable.

## Prediction

If the prior earns its place, `floor` reads more, searches more and answers worse. If it is a
liability, `floor` matches or beats it on the six SYNX charts — the only ones with a
wrong-but-reachable answer to be steered into.

A one-patient probe on SYNX05 returned the gold 20181107 from four self-chosen searches
(`cancer, carcinoma, malignan, diagnos` — note the stems, which no spec term supplies). It
discriminates nothing: B0 with the prior also answers that chart correctly. It is recorded as
the reason this run exists, not as evidence for its outcome.

## What E3 cost, and the guard that came out of it

The E3 run was killed at arm 2. Arm 1 had run against the spec as it stood; the `establishes`
fix was committed while it was in flight, so arms 2–6 would have run against a different spec
— and the change was on the measured axis, since it decides whether a progress-note citation
is admissible at all.

Nothing in the tree would have caught it. `analyze_arms.py` now collects `spec_hash` per arm and
refuses to print a comparison across more than one. Verified against the contaminated data
rather than a fixture:

```
拒绝比较：这些臂跑在 2 个不同的 spec 版本上
  depth-first-stop    ['1b8834600b7b']
  depth-first-nostop  ['3e3aa2ca6ea6']
```

## Where the prior should come from instead

The experience library (B2): learned on a development set, certified by the permutation control
and screened by the answer-leak filter, versioned as an asset. That is what B2 is for. A
retrieval prior hand-written into the contract and shipped to every run is the thing it exists
to replace — and E4 measures what that hand-written one is currently worth.

## Amendment, made after launch and before any result was read

`guideline-only` moves two things, not one. Beside dropping the retrieval prior it drops the
coverage proof:

| arm | `search_terms` | `required_strata` | negative proof |
|---|---|---|---|
| `prior` STRATIFIED_COVERAGE | the five | the three | always required |
| `floor` GUIDELINE_ONLY | none | none | none |

So a `prior`−`floor` difference confounds "no keyword prior" with "weaker gate". It bites on
SYNX06, whose declared `naive_answer` is EVIDENCE_INSUFFICIENT — a weaker gate makes abstention
cheaper, and abstention is that chart's trap.

The arm that separates them also already exists and is also unused: `conditional-negative-coverage`
— `search_terms=()` but `required_strata` retained, coverage activated only when the answer makes
a negative-shaped claim. `prior`−`conditional` isolates the keyword and type prior;
`conditional`−`floor` isolates the gate.

It is being run as a third arm after the first two. Recorded here as an amendment rather than
folded into the prediction above, because the prediction was written without it and a
pre-registration edited to match what you later thought of is not one.

## E4 results

Three arms, eighteen charts each, one spec hash (`3e3aa2ca6ea6`), no card in any of them.

| arm | 读/人 | 词/人 | 自创词/人 | 全部 | SYNX | 中饵 | 读到关键 |
|---|---|---|---|---|---|---|---|
| `conditional` | 5.2 | 16.1 | 13.2 | **17/18** | 5/6 | 1/6 | 5/6 |
| `floor` | 5.8 | 22.1 | 18.3 | 16/18 | 5/6 | 1/6 | 5/6 |
| `prior` | 12.8 | 14.8 | 4.8 | 15/18 | 3/6 | 2/6 | 2/6 |

**The prior arm read twice as much and found less.** 12.8 documents per patient against 5.8,
and on the six adversarial charts it reached the establishing document twice where the other
two reached it five times.

### The confound is resolved

`conditional` was added to separate "no retrieval prior" from "weaker gate". Its retrieval plan
turns out to be identical to `floor`'s — `patient_inventory_only`, `read_all=[]`,
`initial_keywords=[]` — because `required_strata` feeds the proof obligation, not the plan. So
`conditional`−`floor` is the gate alone, and on the adversarial charts it is nothing: 5/6 and
5/6, 1/6 sprung and 1/6, 5.2 reads and 5.8.

The gate is not what makes the floor beat the prior. `prior`−`floor` is the retrieval prior.

### Why, mechanically — this part is not statistical

On SYNX05 the `can_establish` stratum contains 23 documents: 22 `Onc-Med-MD-OP-Progress-Note`
spanning 2009 to 2020, and one pathology report. Its policy is `exhaustive`.

`prior` read 23 documents. **The same 23.** Not a near match — the identical set, all of it,
and nothing else. `floor` read four, one of which was
`Endo-Diab-MD-OP-Progress-Note_2018-11-07`, the document carrying the answer.

The shape is structural, not a property of this chart:

| chart | documents | mandated by `exhaustive` | of those, routine oncology follow-up |
|---|---|---|---|
| SYNX01 | 303 | 26 | 25 |
| SYNX02 | 330 | 27 | 26 |
| SYNX03 | 312 | 24 | 22 |
| SYNX04 | 299 | 18 | 16 |
| SYNX05 | 287 | 23 | 22 |
| SYNX06 | 317 | 21 | 21 |

85–100% of what the stratum compels a run to read is a decade of routine clinic notes. **The
cost does not depend on the stratum picking the right types.** Pick them right and the run
still spends its budget reading twenty of them before it may look anywhere else.

### What the gate does buy

`20999999` — year 2099, month 99, day 99 — was produced by `floor` on SYN0002 and by `prior` on
SYNX06, and never by `conditional`. It is format-valid: `(19|20)\d{2}(0[1-9]|1[0-2]|99)(...|99)`
accepts it. The run wanted to say the year was not establishable and the value space has no way
to say it — `decision_rule[5]` orders the year approximated, and the only imputation flag is
`month_day_imputed`. Nothing records that a year was estimated, so an approximated date and a
witnessed one are indistinguishable downstream.

`conditional` avoids it because a negative-shaped claim activates the coverage proof, and
discharging it finds the date. `prior` has that proof always on and still failed — it had spent
its reads on the stratum first.

**So the two halves want opposite settings**: drop the retrieval prior, keep the negative-claim
proof. That is `conditional`, and it is the best arm on every column.

### What this does not show

- **SYNX02 is failed by all three.** Its requirement is "sweeping a type that states no
  diagnosis", and nothing here does that. It is the one chart where a breadth policy might earn
  its keep, and it remains open.
- **The six SYNX charts are built to punish priors.** On the twelve ordinary charts `prior` is
  12/12 and `floor` is 11/12. A prior costs nothing where it is right. What the real mix of
  right-and-wrong looks like on a clinical corpus, this cannot say.
- **`读到关键` is a proxy** — "read a document stamped with the gold diagnosis date". On SYNX01
  `prior` answered correctly while scoring `没读到`, because that chart resolves through a
  retrospective remark that states the date without the imaging being opened. The proxy
  overcounts misses; the SYNX accuracy column is the hard one.
- **Accuracy is underpowered**, as pre-committed. 17/16/15 over 18 charts decides nothing. The
  read counts and the 23-for-23 mechanism are what carry this.

### Consequence for the ladder above

E1, E2 and E3 compared cards against `B0-base`, which ran this prior. Their finding — "no card
beats every card" — stands as measured but describes a narrower thing than it appeared to: no
card improves on the prior, while the prior itself costs 2 of 6 adversarial charts and half the
read budget. The search-card question should be re-asked against `conditional`, not `B0-base`.
