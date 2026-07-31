---
name: eval-contrast-traces
description: Use when two runs of the same spec reached different answers, or one run matched the answer key and another did not, and you need to locate where their paths diverged. Tells you how to align two traces step by step, which differences are causal and which are noise, and how to state a divergence point as a claim someone could check. Does not settle which answer was right - ask the deterministic scorer for that.
slot: eval
judges: [search_behaviour, divergence_point, plan_adherence]
license: MIT
---

# Putting two traces side by side

You have two work logs for the same question. One of them may have matched the answer key —
**ask the scorer, do not infer it from how confident the reasoning sounds.**

## Align before you compare

Traces are not comparable turn by turn: one run may spend three calls where the other spends
one. Align on EVENTS, not on turn numbers:

1. first search issued
2. first admissible witness read
3. first widening after an empty result
4. the read the final answer cites
5. submission

A run that never reached one of these has a divergence point at that event, and that is
usually the whole finding.

## Which differences matter

Most differences between two traces are noise: different phrasing, different order among
equally-productive searches, one extra confirmatory read. A difference is worth reporting when
it changes what text the model ever saw. Three that do:

- **A term one run tried and the other did not**, where the term hit.
- **A document one run opened and the other did not**, where the document carries the field.
- **A stopping decision** taken at different evidence states.

A difference in wording of reasoning, with identical reads, is not a divergence — it is the
same run described twice.

## Stating the finding

Name the earliest event where the paths differ, quote both sides from the trace, and say what
the later run would have had to do differently to arrive where the other one did. If you
cannot point at a step, you have found a correlation, and say that instead — a divergence you
cannot locate is a real observation and a false explanation.
