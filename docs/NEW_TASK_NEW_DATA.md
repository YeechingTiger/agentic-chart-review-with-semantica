# A new task on new data, in order

> Every command and script named below was verified to exist on 2026-08-03, and every phase was run
> against the synthetic corpus that day. Where a stage refused, the refusal is quoted verbatim.

Written 2026-08-03 after running the whole chain end to end for the first time. Every ordering
constraint here is one the tooling enforces or one that cost something to discover; where a stage
refuses, the refusal is quoted, because a refusal you have read is a refusal you will not fight.

The shape of the thing: **a contract says what the answer must be, a corpus holds the documents, an
agent runs one against the other, and everything else measures whether any of it earned its place.**
Seven phases. Two of them are decision points where the honest outcome may be *stop*.

---

## Phase 0 — the data, before anything

**1. Point at the documents.** `export ACR_CORPUS=/path/to/patients`, one directory per subject.

**2. Filenames are load-bearing, so check them before spending anything.**

```bash
acr check-corpus --strict   # exits 1 on an unreadable filename OR a patient with no documents
```

`corpus.FILENAME_RE` is `<Doc-Type>_<YYYY-MM-DD>[__<n>].txt` — the type and the date come from the
NAME, and every date filter and every type sweep works off them. A file whose stem the loader cannot
parse is skipped, correctly, because guessing a date is worse than missing one — but it was skipped
**silently**, which is the expensive kind: the document is invisible to every run, the run still
answers, still passes its gate, and still reports coverage over the documents it did see.
`PatientChart.unreadable_filenames` is what the command above reads.
`tests/test_adversarial_corpus.py::test_no_unparseable_document_is_sitting_in_the_corpus` exists
because that is not hypothetical. On the synthetic corpus this finds nothing (27 patients, 8,126
documents, 0 unreadable); on a real export it is the first command to run, and `--strict` is what
belongs in a pipeline, because a corpus that half-loaded is not a corpus.

**3. If the data is real, say so, and the system will refuse until you finish.**

```bash
export ACR_REAL_CORPUS=/path/to/real
export ACR_PERSON_ID_PATTERN='<the identifier shape at your site>'
export ACR_PHI_SCAN_PATTERN='<narrower: what may never be committed>'
```

Two patterns, not one, because the consumers want opposite error costs — the runtime's mask must not
false-negative, the pre-commit scan must not false-positive. `site.require_person_id_pattern()` raises
when `ACR_REAL_CORPUS` is set and either is missing:

> Every person-id refusal, every mask and the pre-commit scan are inert without them, and an inert
> guard is indistinguishable from a satisfied one.

There is no default. Three were tried and each was measured wrong somewhere nobody had looked; see
`src/acr/core/site.py`.

**4. Look before you spend.** `acr patients`, then `acr chart <subject>` — the document-type summary
is literally what the agent sees first, and reading it tells you whether the question is answerable
from this corpus at all.

---

## Phase 1 — the contract

**5. Ask whether one already exists.** `acr ask "<your question>"` routes to a contract that can
answer it, or reports an explicit gap. Never a guess. Zero model calls, no chart read.

**6. Author one if not.** A contract is a YAML file stating a question, NUMBERED decision rules, the
conflict rules that order them when two apply, the evidence rules that say what counts as support,
and the closed set of outcomes a run may conclude. `assets/skills/store-to-spec/SKILL.md` is the card
that teaches this; read `references/field-design.md` beside it.

Two things that are not optional:

- **Every enforced element needs a provenance record**, or `load_spec` raises
  `UnprovenancedElementError`. The default provenance is `model_authored`, deliberately, and not the
  manual named in the header — because naming a standards body at the top of a file is not evidence
  that the sentence three hundred lines down came out of it.
- **The outcome space is part of the contract.** A value, an abstention because the evidence is
  insufficient, an abstention because the contract does not cover the case, and a failure are FOUR
  different things. Collapsing them loses the only signal that separates "the record is silent" from
  "we never asked the right question".

> **If the lint tells you to migrate to `means:` + a Site Mapping, you now can.** Until 2026-08-04
> `acr site-mapping build` produced a file no run command could consume: `run_patient` never passed
> `mapping=` and had no parameter for one, so every mapped contract died in `StratumSpec.matches`
> before the first model call. `--mapping` now exists on `run`, `batch` and `extract`, `ACR_SITE_MAPPING`
> covers `acr-mcp`, and the door refuses by naming the flag. Both call sites are wired — the ledger
> AND `plan_from_spec`, because fixing only the first left the run dying one line later.

**7. Lint it.** `acr spec lint <spec>` — eleven checks in four tiers that cost four different things
to run and mean four different things when they pass. Tier 1 must be clean. **Do not read a clean
lint as coverage**: five of the eleven produce zero findings over all six shipped contracts. They are
regression guards in a passing state.

**8. Put it in front of whoever owns the decisions.**

```bash
acr spec review <spec> --out review.md      # a document a domain expert reads in ten minutes
acr spec signoff --spec <spec> --element <id> --reviewer <name>
```

The sign-off carries the element's CONTENT HASH, so the next render reports the approval as withdrawn
the moment the wording changes. Assent to a sentence is not assent to whatever that sentence is
edited into.

---

## Phase 2 — a floor, and the first place to stop

**9. One chart, once.** `acr run <subject> --spec <spec> --runtime-profile guideline-only --out runs/first`
— about twenty seconds. Read the trace: `acr trace <the .jsonl>`.

**10. Build an answer key.** Synthetic corpus: `tools/answer_key_from_corpus.py --spec-key <id>
--fields a,b,c --out key.json`. Real corpus: `acr gold` stages registry values as LOCAL UNRESOLVED
references, never as truth.

**11. The floor.** `acr batch` over the cohort with NO method cards, then:

```bash
acr eval score --runs <dir> --answer-key key.json --fields a,b,c --baseline floor.json
```

**The baseline key is no longer typed.** `--commit`, `--spec-hash`, `--model` and `--date` were
required options reconciled against nothing, and this document instructed you to fill them in. Every
manifest records `code_sha`, `spec_hash`, `model` and `experiment_config_hash`, so those are read;
the date comes from the run id, or from the batch directory for the runs whose `run_id` is a patient
id. The four flags remain as optional ASSERTIONS — pass one and a run that recorded something else
stops the command by name. A part no manifest recorded contradicts nothing, so older runs still
score.

Two things this changes for you at step 22. `eval score` prints `MIXED` for any identity field that
varies across the runs you pointed it at, and `eval compare` then REFUSES that baseline as an
endpoint: a mixed baseline averages more than one arm, and a delta against it prices the difference
between the arms and the difference between the mixtures as one number. Score one arm directory at a
time. And two arms that differ only in their PROMPT — a skill card, a prior, a mapping, a
conflict-refinement brief — are now distinguishable, because `experiment_config_hash` reaches the key.
Before 2026-08-04 they compared as the same configuration and `key_differences` was empty.

**12. DECISION POINT — is an agent the right tool here?**

```bash
python tools/measure_agency.py <runs> --answer-key key.json   # --spec inferred from the runs
```

All three decision points (steps 12, 13, 22) were hardcoded to `STORE.390` and read gold from
`corpus/index.json`, a file only `tools/generate_corpus.py` writes — so on any corpus but the
shipped synthetic one they died with `FileNotFoundError`. They take `--answer-key` now: the same file
`acr eval score` takes.

**On a real corpus you must still author that file.** `tools/answer_key_from_corpus.py` reads
`_ground_truth.json`, which exists only in the synthetic corpus, and **`acr gold` is not its
real-data equivalent** — it stages registry values as LOCAL UNRESOLVED references and audits
derivability; it emits nothing in `eval score`'s key format. That producer does not exist yet, and
saying otherwise here was the same class of error as the rest of this document's corrections. `analyze_arms.py` additionally
REFUSES to print its held-out column when the corpus carries no design metadata, because not knowing
which charts are contaminated is not the same as none being contaminated.

`measure_agency.py` takes the contract's OWN vocabulary, searches the corpus with
it, and asks whether the document that carried the answer is among the hits.

> If reachability is high and decisive evidence arrives on contract terms across most variables, then
> for those variables this system is an expensive way to run a keyword search, and the honest
> recommendation is a query.

Run this before investing in anything below it.

---

## Phase 3 — the second place to stop

**13. Where are the errors, actually?**
`python tools/measure_controller_value.py <runs>` attributes every wrong answer:
`NEVER_LOOKED` / `READ_NOT_CITED` / `CITED_BUT_MISJUDGED` / `GOLD_REJECTED` / `UNSEEDABLE`.

On `STORE.390`, of eleven wrong answers: **`NEVER_LOOKED 0`, `READ_NOT_CITED 0`.** Not one failure was
a retrieval failure — the agent opened the document carrying the answer every single time and got the
reading wrong.

**If your failures look like that, Phase 4 buys nothing.** Better search cannot fix a misreading, and
the work belongs in the contract's decision rules or in the reading policy. Skip to Phase 5.

---

## Phase 4 — experience: only if retrieval IS the problem

This is the expensive phase and the one with the most ordering constraints.

**14. The full scan.** Read every note of a development set once, cheaply, against ONE requirement.

```bash
acr label scan --spec <spec> --patients A,B,C \
  --max-usd 4 --max-terms-per-note 8 --min-term-chars 4 --concurrency 8
```

`--dry-run` first: it prices the input as a FLOOR and says so — completion length is not knowable
before the call, and `--max-usd` is what actually stops the run. Three subjects of ~300 notes each
cost about $2.50 and take a few minutes.

**15. Read the labelling before using it.** `acr label progress --spec <spec> --max-terms-per-note 8
--min-term-chars 4`. Two numbers matter and they are printed apart on purpose:

- *"N label(s) carry no quote at all"* — expected, and not a warning. Most notes establish nothing.
- *"N label(s) quote text that is NOT in the note"* — the model composing. On the first real scan this
  was **zero**, while an earlier version of this report conflated the two and claimed 305.
- The hallucinated-term rate: 61 of 439 proposed terms (14%) on that scan.

**16. Bridge the format.** `acr label export --spec <spec> … --out labelling.json`. The scan writes
JSONL; the certification half reads a single JSON object. They are coupled by a file format
deliberately, and the conversion is this command.

**17. Build the bitmaps.** `python tools/build_termcache.py --labels <labels.jsonl> --spec <spec>
--fields a,b,c`. One bit per (document, candidate term), plus an oracle bit per document from the
labelling. Uses the CORPUS'S own matcher — a price computed against a different matcher is a price for
a search nobody performs.

**18. Price the candidates.** `acr derive terms …`. It may correctly report **nothing to add**: on the
scan above, 50 candidates and ZERO in the cut, because the contract's five incumbent keywords already
surfaced all 27 answer-bearing documents. That is a result, not a failure.

**19. Certify on held-out data.**

```bash
acr assets split   --labelling labelling.json --out split.json
acr assets measure --spec <spec> --field <f> --labelling labelling.json --split split.json
acr assets evolve  …          # hill-climb on DEV ONLY
acr assets certify …          # control against shuffled labels, score on TEST
acr assets adopt   --spec <spec> --cert cert.json
```

**Two refusals you will meet, and both are correct:**

- `cannot split 1 patient(s) into two non-empty halves` — label at least two subjects.
- `'<keyword>' prefixes nothing in this labelling's N-term vocabulary, so the labels cannot say what
  it would surface` — **the ordering constraint nothing documents**: you cannot certify a contract's
  EXISTING keyword list against a labelling that never indexed those keywords. Scoring them zero
  would be a lie about the terms rather than a fact about them. Either seed the scan with the
  incumbent list, or drop it from what you are measuring.

---

## Phase 5 — did the intervention earn its place

**20. Freeze the protocol BEFORE the first model call.** `docs/POLICY_LADDER_PROTOCOL.md` is the
worked example: what runs, how many seeds, how it will be read, and the predictions **registered in
advance** — including one registered against the thing being tested. A prediction written afterwards
is a description.

**21. Run arms that differ in exactly one thing.** `python tools/run_ladder.py --group policy
--seed 1234` (then 2345, 3456). `batch` runs each chart once at temperature 1.0, so **a repeat is a
separate seed**: one seed cannot separate a card's effect from run-to-run variance, and an effect
whose ranking flips between seeds is noise wearing a result's clothes.

**22. Read it out.** `python tools/analyze_arms.py <experiment>`. It REFUSES rather than footnotes:
a comparison across mixed contract hashes stops; a chart that informed a method's design is never
folded into a headline number. Held-out charts are counted apart and the held-out denominator is
printed small on purpose.

**23. Assert the mechanisms are not inert.** `python tools/verify_mechanisms.py` and
`python tools/verify_structure.py`.

---

## Phase 6 — the record, and what it carries

**24. Detectors.** `acr eval detect --runs <dir> --min-term-chars 4 --max-rejection-repeats 3
--token-band lo,hi --turn-band lo,hi`. **Declare the thresholds.** With none declared, `findings` is
empty "because nothing looked, not because nothing fired" — and the report says so.

**25. Audit the trace you have already written to disk.**

```bash
acr audit run --manifest <m> --subject-id <s> --local-root "$ACR_LOCAL_ARTIFACT_ROOT"
python -c "…" # or the script skill: audit-phi-in-trace
```

This is the only thing in the system that answers *does a record we already wrote carry an
identifier, and where* — no object graph can, because the leak is in content, not structure. Location
decides severity: a hit in a MODEL'S OUTPUT or an ARTIFACT PATH left the process; a hit in a document
body a read tool returned is the corpus being the corpus.

Over twenty runs of the synthetic corpus this reported 4 IRB findings: an MRN-shaped string in the
SUBMITTED ANSWER's evidence quote, because the agent had quoted a document HEADER
(`MRN: … / Patient: … / DOB: …`) as its evidence. Synthetic there. On a real chart that is a real
record number, a real name and a real date of birth, in the artifact a human pastes into a report.

**26. Attribute the wrong answers.**

> **The "there are only 2 records" explanation was wrong.** `meta_evaluate_attributions` reads the
> human root cause from `row["primary_cause"]`; `AdjudicationEvent.to_dict()` — the only producer —
> emitted `decision`, validated against `LIFECYCLE`, and `LIFECYCLE ∩ CAUSES` is empty. So the
> calibration returned zero pairs for ANY number of adjudications and reported a case shortage.
> Fixed 2026-08-04: `acr attribute adjudicate --primary-cause <one of CAUSES>`, and the report now
> names the format problem when rows exist and none carries a cause. First real run:
> `n_adjudicated_pairs: 2, macro_f1: 0.333`.

```bash
acr attribute case-map --runs <dir> --out case-map.json      # mint the pseudonyms FIRST
acr attribute batch --runs <dir> --spec <spec> --case-map case-map.json \
  --min-term-chars 4 --max-rejection-repeats 3 --token-band 2000,60000 --turn-band 3,40 \
  --max-usd 1.20 --max-model-calls 24
```

`--case-map` is required by three commands and **nothing produced one until 2026-08-03**, which is
why `attribute` had never emitted a proposal. The map is `{case_id: patient_id}`, minted as an HMAC
under a 256-bit key kept in the local root — salted rather than a plain digest, because an unsalted
hash of a medical record number is reversible by enumeration. Re-running is additive: reminting a
case id would split that patient's history into two cases.

**Give it enough CALLS, not just enough dollars.** The first real batch selected 1 of 27 runs, spent
$0.009 of a $0.60 ceiling, and returned UNRESOLVED — *"model-call limit reached without a gate-valid
attribution (7/8)"*. The protocol requires a final targeted confirmation round, and the call budget
is what binds. At 24 calls the same case resolved: `EVIDENCE_GAP`, `POSSIBLE`, 15 calls, 3 chart
reads, `repair_route: HUMAN_REVIEW_TEST_OBLIGATION`.

Treat the cause as a hypothesis: this plane's own accuracy has never been measured
(`meta_evaluate_attributions` wants 30 adjudicated cases; there are 2 records). And note that
`semantic_patch_allowed` is `False` in BLIND mode by design — a patch needs gold AND adjudication.

---

## Phase 7 — change something, deliberately

**27. Route failures at the text.**

```bash
acr refine cases --runs <dir> --answer-key key.json --fields <f> --spec <spec> \
  --case-map case-map.json --out cases.json          # free; the other missing producer
acr refine route --cases cases.json --verdicts verdicts.json --spec-text <id>=<spec>
```

`refine route` proposes an edit to the specific text parameter that could have caused a classified
failure — a prompt, a card, a contract clause. It never applies one: paired validation is a separate
decision, a semantic change needs gold AND human adjudication, and a `REGISTRY_REFERENCE` truth mode
can only produce a question for an expert.

**`refine cases` separates three populations that the router's own boolean cannot.** Cut 1 sends
`establishing_evidence_surfaced: false` to §6c retrieval without consulting any model, so what goes
into that field decides the whole diagnosis:

| population | what it means | is it retrieval? |
|---|---|---|
| a document carries the key value, the run never opened it | genuine retrieval failure | **yes** |
| the key value appears in NO document | the key is CONSTRUCTED — imputed, or inferred across notes | no; no search could find it |
| the key says abstaining was correct | no document establishes an abstention | no |

On 27 runs of STORE.390 that is **7 disagreements: 2 genuine retrieval, 3 unseedable, 2
read-and-misjudged.** `tools/measure_controller_value.py`, an independent implementation, reports the
same 7.

**An earlier version of this document said 8, and that was wrong** — three ways, all fixed
2026-08-04. `_coded_value` returned the run's STATUS STRING when no value was present, so a run that
correctly abstained on a chart where the key says abstention is correct compared `"CORPUS_INSUFFICIENT"`
against `""` and was emitted as a failure; `eval score` called the same manifest `ABSTAINED_CORRECT`.
`_notations` returned `[year]` for a `99`-partial date, and a bare year is in essentially every note
of a chart from that year (SYNY02: 27 of them), so a CONSTRUCTED date read as present and cut 1
routed it to §6c retrieval — the exact inversion this section warns about. And the trace was read
from the manifest's recorded absolute path rather than the sibling `.jsonl`, so a moved run tree
refused with a false statement (49 of 509 manifests here are in that state; `tools/archive_runs.sh`
is what moves them). All three were duplicate implementations of facts `RunRecord` already owned.

**Then it stops, and the stop is the point.** Every case lands `NOT_ADJUDICATED`, and cut 2 routes
an un-adjudicated case to UNRESOLVED without asking the reflection model anything. Whether the
answer key is right is a human's call, and the router refuses to launder it into a spec edit. So a
first pass names its own next task: adjudication — the same thing
`meta_evaluate_attributions` has been waiting on for thirty cases while two exist.

**28. Then go back to Phase 5**, because an intervention nobody measured is a change and not an
improvement.

---

## The discipline that makes any of it mean anything

**A chart used to design a method cannot score it.** Every chart carries
`informed_module_design: true|false` and `designed_from`, and `analyze_arms.py` refuses to fold an
informed chart into a headline. The six SYNY charts exist because the other twenty-one had already
been used to build the methods being measured.

**A run's identity is stamped, not remembered.** Every manifest carries the contract's hash, the
corpus's content hash, the code SHA and the resolved arm. `code_sha()` marks a dirty tree, which is
why you do not commit while a batch is in flight: it splits one experiment across two code
identities.

**Run output is never committed.** `runs/` is ignored wholesale. A tracked run record is a disclosure
that has already happened.
