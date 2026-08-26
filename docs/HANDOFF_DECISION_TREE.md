# Handoff: the decision-tree module

One chart-review run, read back as the decisions it was made of, in semantica.
Everything below is verified against this repo at the commit that added it.

---

## 1. What was decided, and why it matters before you touch anything

The question was whether the agent should narrate its decisions **in our taxonomy at
runtime** (A), or run unconstrained and be **reconstructed afterwards** (B).

**It is B — but not "runtime collects nothing".** The seam is drawn at exactly one place:

| | collected at runtime | applied afterwards |
|---|---|---|
| `used` — which documents/searches a decision rested on | ✅ | ✗ |
| `grounding` — contract / card / chart / precedent / own_knowledge | ✅ | ✗ |
| `decision_type` — which of 13 kinds this judgment was | ✗ | ✅ |
| how the run splits into big and small points | ✗ | ✅ |

**Why the taxonomy is post-hoc.** It is the one part of this system that has never met real
data. Asking a model to sort a judgment into an unsettled thirteen-way vocabulary at the moment
it acts forces everything that fits nothing into `other` — or worse, into a type that merely
looks close — and the trace keeps only the forced choice. Applied afterwards, changing the
taxonomy costs one re-extraction instead of re-running every review ever done.

**Why `used` and `grounding` are not.** Both are facts about the model's state at the moment it
decided, and a later reader cannot recover either. It sees which documents were opened, so it
would mark every citation verified; it sees a conclusion and a contract that would support it,
so it would write `grounding: contract`. The gap between what a run *consulted* and what it
*looked at*, and the gap between the contract and the model's own clinical knowledge, are the
two findings this instrument exists to produce. Both are optional fields — a required field
gets filled with filler, and filler in the one channel nobody can check is worse than a gap
anybody can see.

The split is pinned by a test, not by discipline:
`tests/test_mvp_toolserver.py::test_the_runtime_does_not_know_the_taxonomy` asserts that
`toolserver.py` never imports `decision_types`.

---

## 2. The four modules

```
warrants.py        RUNTIME vocabulary: GROUNDING_KINDS, INPUT_KINDS, and RunFacts —
                   ONE implementation of "does this citation stand", built either from
                   the live ToolState or replayed from a finished trace
decision_types.py  the TAXONOMY: 13 types in 3 groups, big/small levels. Read only by
                   the reconstructor. Never by the runtime
toolserver.py      the tool surface + Layer-1 trace + the gate
reconstruct.py     trace -> decision tree -> semantica
ledger.py          semantica as the account book (no parallel store, see §6)
```

---

## 3. Run it

```bash
uv venv && uv pip install -e .            # plus the semantica pin, see pyproject
export $(grep -v '^#' .env | xargs)       # ACR_MODEL / ACR_API_BASE / ACR_API_KEY

# 1. one review, with the ledger recording live
python -m acr.mvp.cli run assets/specs/STORE.390.date_of_initial_diagnosis.yaml \
       corpus/patients/SYN0001 --out runs/mvp --ledger runs/mvp/ledger.json

# 2. read the run top to bottom
python -m acr.mvp.cli trace runs/mvp/<RUN_DIR>

# 3. read it back as a typed decision tree  <-- the new verb, needs an LLM
python -m acr.mvp.cli reconstruct runs/mvp/<RUN_DIR> \
       --ledger runs/mvp/ledger.json \
       --model openrouter/anthropic/claude-sonnet-4.5 \
       --api-key-env OPENROUTER_API_KEY \
       --passes 2

# 4. the audit chain, and what has settled or diverged across runs
python -m acr.mvp.cli chain SYN0001 --ledger runs/mvp/ledger.json
python -m acr.mvp.cli precipitate --ledger runs/mvp/ledger.json --level big
```

`--model` is a **LiteLLM** id, so any provider LiteLLM speaks works
(`openrouter/...`, `anthropic/...`, `openai/...`); `--base-url` overrides the endpoint.
The reconstructor goes through `semantica.llms.LiteLLM`, so there is no HTTP client of
ours anywhere in this path.

**Note the two different providers.** The *agent* runs on codex against `ACR_API_BASE`
(a Responses-API endpoint). The *reconstructor* runs on LiteLLM against `--api-key-env`.
They are deliberately separable — reading a run back with a different model than performed it
is the normal case, not an edge case.

---

## 4. Reading the output

```
3 big point(s), 3 small
provenance: 3 self-reported, 3 reconstructed
1 quote(s) did not hold up against the record and were dropped to reconstructed
own_knowledge: 2, of which 2 the run actually said so — only those are questions for a domain expert
FALSE WARRANT at seq 5: decided "date the case by the earlier pathology"
  citing note:Onc-Med-MD-OP-Progress-Note_2023-04-12 — never read or surfaced in this run
```

**The quote is the load-bearing part of the whole design.** Asking a model how confident it is
produces a number that means nothing. The extractor is instead asked *which line it read each
judgment off*, verbatim — a claim that either matches the sheet or does not. So provenance is
computed, never self-assessed:

| | meaning |
|---|---|
| `deterministic` | the server recorded it: the tool called, the gate's verdict, the span |
| `self_reported` | the model said it during the run, and the quote proves where |
| `reconstructed` | this reader inferred it; nobody said it |

Quotes are checked against the **self-reported lines only**. Quoting a server fact back at us
proves the server observed something, never that the model thought it.

This is why `own_knowledge` is reported twice over. `own_knowledge` that is **self-reported**
means a run told you the contract ran out there — that is a question for a domain expert.
The same label **reconstructed** means only that the reader thought so. A report that mixed
them would spend expert attention on noise.

**`--passes N`** extracts the same unchanged run N times and reports the drift, storing only
the first. If two readings of one run disagree about how many decisions it held, the taxonomy
is not sharp enough to compare across runs yet, and nothing downstream fixes that. **Look at
this number before you look at anything else.**

**The audit chain carries both.** `step` rows are the model's own words, unclassified, as it
spoke them; `big:` rows are this reader's analysis. Testimony and analysis, side by side and
labelled — not duplication.

---

## 5. What the verifier assumes

That the extractor is wrong, and reports how:

- a span anchored to no real seq is **dropped with its children** and counted
- a citation to a document the run never opened is **marked false**, not believed
- a big-point type arriving in the small list **keeps its name and moves level** — that is a
  segmentation error, not a naming one, and discarding either half loses information
- an unknown type becomes `other` and **keeps what was claimed**
- seqs no point covers are listed as **stretches this reading does not explain**

---

## 6. Two things about semantica you will otherwise rediscover the hard way

**`save_to_file` writes nodes, edges and links — and nothing else.** The decision registry
(`_decisions` and the category/entity indexes over it) is in-memory, so after a reload
`find_precedents_by_scenario` and `get_decision_insights` answer as though nothing was ever
recorded. `SemanticaLedger._rehydrate` replays the persisted decision nodes and their
`involves` edges back into that registry at construction. **We keep no parallel store** — the
old `.index.json` sidecar is gone; the case is an entity, so the audit anchor is a graph query.

**Its default similarity threshold is unreachable here.** It scores a precedent
`0.7 * content + 0.3 * structural`, and structural needs `advanced_analytics`, which stays off
for the PHI posture. So the score caps at 0.7 while the default threshold is 0.5, and two
decisions facing an *identical* situation measure **0.467** on content (its similarity mixes
`reasoning` and the entity list into the text) → **0.327** combined → rejected. `precedents()`
therefore states its floor on *content* similarity and converts. Measured on real ledgers:
unrelated ≈ 0.07, identical ≈ 0.47, so the 0.3 default separates them with room on both sides.

`precipitate` deliberately does **not** use semantica's search: it clusters on `scenario`
alone, so two runs that faced the same situation and reasoned differently land in the same
group instead of being split by the very field whose disagreement is the finding.

---

## 7. Where to point this next

The taxonomy is a hypothesis. This module exists to falsify it. In rough order:

1. **Run `--passes 2` on ten real runs.** Drift in `n_big` is the first honest signal about
   whether the vocabulary is sharp enough. Everything else waits on this.
2. **Count what lands in `other`.** A big `other` bucket names the types the list is missing.
   Read what is in it before adding anything — the granularity rule is that two types split
   only when their divergences go to different people and change different things.
3. **Count `own_knowledge`, self-reported only.** Each one is a place the contract did not
   reach, and the point of the whole exercise is that these become questions for a human, not
   rules we invent.
4. **Then** the situation slug (`{level}:{type}:{slug}`), which is where comparability
   actually lives — but built from the real distribution, not guessed a second time.

Deliberately not built yet, all waiting on that data: L1–L4 levels, PointDefinitionRegistry,
counterfactual impact replay, the three-way triage, and `PolicyEngine` for the rule lifecycle.

---

## 8. Known limits

- **The reconstructor has never run against a real LLM.** Every test stubs the extractor, on
  purpose — a test that called a model would measure the model, and the cases under test are
  what happens when the extractor is *wrong*, which cannot be provoked reliably from a live
  one. The first real call is yours; expect to tune `build_prompt`.
- `rule:` references are recorded as claimed, never checked. The contract's clause ids are
  enumerable, so this is doable — it is left until a real run cites a clause that does not
  exist, so the check is built against a real failure rather than an imagined one.
- Layer 2 (the harness event stream) is archived and read into the sheet, but is never
  load-bearing: delete it and the trace still reads, minus the self-reported thought lines.
