# The policy ladder — protocol, frozen 2026-08-03 before the first model call

The core architecture, run end to end: one task contract, one agent loop with seven tools, one
policy card drawn per run, and a record nothing downstream may rewrite. This document is the
protocol. It is written before the first run so that the predictions below cannot be adjusted to
whatever comes back.

## What is being run

| | |
|---|---|
| contract | `assets/specs/STORE.390.date_of_initial_diagnosis.yaml` |
| runtime profile | `guideline-only` |
| charts | all 27 with a `STORE.390` ground truth — 21 informed, 6 held-out (SYNY01–06) |
| arms | `B0-base`, `pol-reactive`, `pol-infogain`, `pol-hypothesis-set` |
| repeats | 3, as three seeds: 1234 / 2345 / 3456, each in its own output root |
| total | 4 × 27 × 3 = **324 runs** |

Every arm is `tool-contract, coverage-judgement` plus at most one policy card, so the only thing
that changes between arms is the policy. `B0-base` has no policy card at all.

## Why three seeds and not one

`batch` runs each chart once and the model runs at temperature 1.0. A single run per (arm, chart)
cell cannot separate a card's effect from run-to-run variance, and reporting one anyway is the
mistake this tree has already made. Three seeds, three independent readings of the same ladder,
each analysed on its own. **An effect whose ranking flips between seeds is noise wearing a
result's clothes.** They are not to be averaged into a single number without saying so.

## How it will be read

`tools/analyze_arms.py` refuses to fold an informed chart into a headline number, so the informed
21 and the held-out 6 are reported apart. The held-out denominator is six. Printing it small is
the point: a number over six charts that were not used to build the thing being measured is worth
more than a number over twenty-one that were.

Failures are recorded, not fixed. A crash is a data point about the code, and patching mid-batch
splits one experiment across two code identities — `code_sha()` carries a `-dirty` flag for
exactly this reason. **No commits while the batch is in flight.**

## Predictions, registered now

1. **`pol-reactive` and `pol-infogain` will not improve accuracy.** Both are retrieval-and-stopping
   interventions. The attribution of eleven wrong answers in the last valid batch was
   `NEVER_LOOKED 0`, `READ_NOT_CITED 0` — not one failure was a retrieval failure. If either arm
   moves accuracy, my model of where the errors live is wrong and that is the finding.

2. **`pol-hypothesis-set` is the only arm with a mechanism.** It intervenes on judgement, not
   retrieval: enumerate the ways the contract says an answer can be established, then close each
   one. It is also the prose replacement for a deleted enforcement layer, so this arm answers a
   second question — whether the difference between the two falsified attempts was the framing or
   the whole idea.

3. **The sharpest cell is SYNY02 and SYNY04.** Their gold values `20159999` and `20191099` are
   partial dates that appear as no string in any document — they are constructed by a decision
   rule, not found. `pol-hypothesis-set`'s central claim is that "a rule may produce a candidate
   that appears nowhere in the text". If the card works anywhere, it works here. If it does not
   move these two, the card is decoration.

4. **Registered against myself:** a prose card asking for counterevidence has already been measured
   in this tree, and it moved alternative mentions 4/12 → **1/12** — the wrong direction, while
   spending 11% more words. So the honest prior on card 3 is *no effect or a negative one*. I am
   not expecting this to work.

## What would falsify what

- All four arms equal on held-out ⇒ the policy slot does not matter for this variable, and the
  three cards are cost without effect.
- `pol-hypothesis-set` improves the informed 21 but not the held-out 6 ⇒ the card is scoring on
  its own development set. It was written from the same failures those charts were designed from.
- Any arm below `B0-base` ⇒ a card that makes the model worse, which is a result and belongs in
  the record next to the two falsified attempts in `docs/CANDIDATE_LEDGER_REMOVED.md`.

---

## The 2026-08-03 run was ABANDONED. Do not read the manifests in `runs/ladder-policy-s1234/` as this experiment.

Started 11:08, stopped by the owner about 90 minutes in. What is on disk:

| arm | charts completed |
|---|---|
| `B0-base` | 27 of 27 |
| `pol-reactive` | 27 of 27 |
| `pol-infogain` | partial |
| `pol-hypothesis-set` | none |

Seeds 2345 and 3456 never started. 75 manifests exist against a protocol that calls for 324.

**Why this note is here rather than nowhere.** Two complete arms invite exactly one comparison —
`B0-base` against `pol-reactive`, 27 charts each — and that comparison is not the experiment. It is
one seed, so it cannot separate a card's effect from run-to-run variance at temperature 1.0, which
is the whole reason the protocol specifies three. Reporting it would be the mistake this document
was written to prevent, made against this document's own data.

The four registered predictions stand unmeasured. The partial data is kept because deleting it
would leave nothing to explain the gap in `runs/`, and because `B0-base` at `ded2fc8` over all 27
charts is a usable FLOOR for the current contract — the previous floor was stale, the spec hash
having moved twice. Use it as a floor. Do not use it as an arm comparison.
