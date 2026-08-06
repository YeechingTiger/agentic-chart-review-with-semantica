---
name: eval-overconfidence
description: Use when a run submitted a definite answer that did not match the answer key, and you need to find what it treated as sufficient. Tells you how to read the evidence a run actually rested on, which evidence patterns precede confident errors, and how to separate a reasoning failure from a contract gap. Does not determine that the answer was wrong - ask the deterministic scorer.
license: MIT
metadata:
  slot: eval
  judges: evidence_sufficiency_reasoning witness_standing contract_gap
---

# A confident answer that did not hold

**The scorer tells you it did not match.** Your question is narrower and more useful: what did
the run treat as enough?

## Read what it rested on, not what it said

Go to the cited evidence first, before the reasoning. The reasoning is a story told after the
reads; the citation is what the answer is actually made of. Four patterns recur:

- **A single witness of the wrong standing.** The cited document mentions the value but is not
  a type the contract lets establish it. The run treated a mention as an establishment.
- **An interim line.** The citation is from a preliminary or pending report whose final version
  says something else, and the run never chased the thread.
- **An inference across two documents.** Neither cited document states the value; the answer is
  a join the run performed. Sometimes correct, never admissible on its own.
- **The right document, the wrong span.** The value cited is real text from a real report about
  a different specimen, date, or entity.

## Reasoning failure or contract gap

These need different owners, and confusing them wastes a fix.

- If the contract clearly covers this case and the run misapplied it: instruction-following
  failure. Say which sentence of the contract the run departed from.
- If the contract does not settle the case — two documents of equal standing disagree and no
  precedence rule applies — then this is a **contract gap** and the correct behaviour was
  `SPEC_INSUFFICIENT`. Report it as a gap and quote the ambiguity. A run punished for a gap it
  correctly walked into learns to guess instead.

Distinguishing these two is the main value of this skill. When you cannot, say which evidence
would settle it.
