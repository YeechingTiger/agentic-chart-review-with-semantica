# Archive notes

Run output is an experimental record. Deleting it destroys data that code changes can make
permanently unreproducible. Use `tools/archive_runs.sh` — never `rm -rf runs/...`.

---

## 2026-07-25 — data lost to `rm -rf`

Four ablation runs (`aprime_SYN0002`, `b_SYN0002`, `aprime_SYN0001`, `b_SYN0001`, all at
code `f8a5c25^`) were deleted before the corrected batch was launched. `runs/` was in
`.gitignore` and had never been committed, so nothing was recoverable from git. The new batch
then **reused the same directory names**, so anyone opening `runs/aprime_SYN0002` afterwards
would see post-fix results with no indication that a same-named, different-conclusion run had
ever existed.

### What was lost, and why two items cannot be regenerated

**`b_SYN0002` — the only primary record of the gate bypass.** Its reflect node ruled
SUFFICIENT at 312s on the grounds *"Coverage confirms zero documents capable of establishing
histology"*, read directly off the stratified ledger's `can_establish N=0`, and the graph
routed it straight to `finalize`, skipping `submit_answer` and therefore the entire proof
obligation. That trace is the sole primary evidence for two separate findings:

1. the bypass existed, and
2. **stratification gave the agent a correct signal it acted on** — the one positive result
   about stratification obtained so far.

It cannot be re-run. Commit `306e88a` removed the SUFFICIENT → finalize edge, so that
configuration now routes through `submit_answer`. The path no longer exists in the code.

**`aprime_SYN0002` — the only record of the gate refusing an answer before the fix.**
13 steps, `submit_answer` called twice, one rejection, then a revised answer. Also
unreproducible: the surrounding code has changed in ways that alter the path.

### What survives

- `2026-07-25_pre-single-route-fix/b_SYN0002.rendered-path.txt` — a **rendered summary**
  salvaged from a session transcript, not the raw trace. Reflect reasons truncated to ~90
  characters, tool arguments and results elided, plan text cut at 72 characters, no per-call
  token counts. **Secondary evidence.**
- Summary numbers quoted in the session transcript:

  | arm/patient | status | steps | submit | rejected | termination | tokens | time |
  |---|---|---|---|---|---|---|---|
  | aprime_SYN0002 | EVIDENCE_INSUFFICIENT | 13 | 2 | 1 | submit accepted | 170,155 | 817s |
  | b_SYN0002 | EVIDENCE_INSUFFICIENT | 7 | 0 | 0 | reflect SUFFICIENT → finalize (bypass) | 59,325 | 341s |
  | aprime_SYN0001 | FOUND | 9 | 1 | 0 | submit accepted | 116,092 | 526s |
  | b_SYN0001 | FOUND | 9 | 0 | 0 | reflect SUFFICIENT → finalize (bypass) | 93,918 | 686s |

  Also recorded: `plan_fallbacks = 0` for all four (the planner was genuinely working); the
  `degradation` block was **absent**, because those runs predate the counters — reflect,
  finalize and act degradation went unmeasured in that batch.

### Constraint on citing this batch

If a paper cites "before the fix, the stratified arm never submitted", it must state that
the primary traces were destroyed, that only a rendered summary survives for one of the four
runs, and that the observation cannot be reproduced because the code path was removed. If
that provenance cannot be stated, the observation cannot be used.

### What changed as a result

- Run directories are now `runs/<label>__<UTC>__<code-sha>/`, created with `exist_ok=False`.
  A configuration run under different code is a different experiment, so the sha is part of
  the identity and names are never reused.
- `manifest.json` files are tracked in git (~2KB each). Traces stay out; manifests are
  evidence.
- `tools/archive_runs.sh` moves directories into `runs/_archive/<timestamp>/`.
- README states that run output is an experimental record.
