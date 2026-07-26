# Results

## 2026-07-26 — first end-to-end auditable determination

`runs/aprime_SYN0002__20260726T035724Z__183f3f3/` — committed, traces and manifest.

```
status                EVIDENCE_INSUFFICIENT
negative_basis        GATE_VALIDATED          ← first time
gate_validated        True
steps_to_gate_pass    18
rejections            1
degradation           {plan: 0, reflect: 0, finalize: 0, act: 0}
suspected_recognition_failures  0
tokens / elapsed      313,400 / 1218s
```

Patient SYN0002, whose biopsy was performed at an outside hospital: the correct answer is
that the chart cannot establish histology. The agent reached that answer *and* proved it had
looked, which is the thing this system exists to do and had never once done.

Three details make it worth more than a pass:

- **`rejections: 1`** — the gate refused an answer and the agent then satisfied it. Not a
  rubber stamp.
- **`degradation` all zero** — no node silently fell back, so this run's behaviour is
  interpretable. First time that can be said of any run here.
- **`steps_to_gate_pass: 18`** — the operational number. It is what a patient-variable costs.

### What this does and does not establish

**Established:** the `enumerated` proof obligation is satisfiable end to end, in 18 steps,
with clean instrumentation.

**Not established:** the **stratified** obligation — the one carrying forced validation
sampling, 25 documents from `cannot_establish` and 25 from `may_mention`'s search misses — has
still never been satisfied. That is the actual design contribution of this project and its
feasibility remains unknown. `b_SYN0002` is the run that would answer it.

**Not tested:** SYN0001, the FOUND case, was not run in this batch.

### A judgement this overturns

`steps_to_gate_pass = 18 < 20`. Earlier 20-step runs did not fail on budget; they failed on
three bugs — the sampling deadlock, `gate_validated` never declared as a state channel, and
one-document-per-read. With those fixed, 20 steps would have sufficed.

So moving to a GPU is about **speed, not feasibility**. The laptop can complete the work; it
takes 20 minutes a run and forces the ablation arms to run one after another.

---

## Superseded batches

See `runs/_archive/NOTES.md`. The 20-step batch is kept as the worked example of a label that
means nothing: 4/4 correct, 0/4 validated, every answer produced by budget exhaustion. Without
`proof_basis` / `negative_basis` it would have been written up as a perfect score.
