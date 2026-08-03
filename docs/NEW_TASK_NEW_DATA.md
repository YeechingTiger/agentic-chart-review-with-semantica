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

**2. Filenames are load-bearing.** `corpus.FILENAME_RE` is
`<Doc-Type>_<YYYY-MM-DD>[__<n>].txt` — the type and the date come from the NAME, and every date
filter and every type sweep works off them. A file whose stem the loader cannot parse is skipped
**silently**: it is invisible to every run, the run still answers, still passes its gate, and still
reports coverage over the documents it did see.
`tests/test_adversarial_corpus.py::test_no_unparseable_document_is_sitting_in_the_corpus` exists
because that is not hypothetical.

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
acr eval score --runs <dir> --answer-key key.json --fields a,b,c \
  --commit <sha> --spec-hash <hash> --model <m> --date <d>
```

**12. DECISION POINT — is an agent the right tool here?**
`python tools/measure_agency.py <runs>` takes the contract's OWN vocabulary, searches the corpus with
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
acr audit run --manifest <m> --subject-id <s> --local-root "$ACR_LOCAL_ROOT"
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

**26. Attribute the wrong answers.** `acr attribute` — a model-based cause for each, from an agent
that has never been shown the key. Treat its output as a hypothesis: this plane's own accuracy has
never been measured (`meta_evaluate_attributions` wants 30 adjudicated cases; there are 2 records).

---

## Phase 7 — change something, deliberately

**27. Route failures at the text.** `acr refine` proposes an edit to the specific text parameter that
could have caused a classified failure — a prompt, a card, a contract clause. It never applies one:
paired validation is a separate decision, a semantic change needs gold AND human adjudication, and a
`REGISTRY_REFERENCE` truth mode can only produce a question for an expert.

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
