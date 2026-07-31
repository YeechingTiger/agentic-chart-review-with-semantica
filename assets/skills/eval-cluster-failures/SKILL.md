---
name: eval-cluster-failures
description: Use when several runs across a cohort came out wrong and you must tell whether they share one cause or have several. Tells you which trace features to cluster on, why clustering on the wrong answer's value misleads, how many cases a cluster needs before it is worth reporting, and how to name a cluster so that a fix can be aimed at it. Does not decide which runs were wrong - ask the deterministic scorer for the list.
slot: eval
judges: [failure_grouping, shared_mechanism, cluster_support]
license: MIT
---

# Telling one problem from six

Six wrong answers can be one bug or six. The difference decides whether anything is worth
fixing, and it is not visible in the answers themselves.

**Get the list of wrong runs from the scorer.** Which ones missed is not yours to decide.

## Cluster on behaviour, not on the answer

The tempting axis is the value that came out wrong — all the runs that said `C349`. That axis
groups by SYMPTOM, and a symptom shared by two different mechanisms produces a cluster nobody
can fix. Cluster on what the run DID:

- **the last search before submission** — what it was looking for when it gave up
- **whether an admissible witness was ever read** — never-read and read-but-misjudged are
  different failures with different owners
- **where the answer's citation came from** — which document type, which section
- **the shape of the stop** — budget exhausted, no more ideas, or an affirmative decision

## Support

A cluster of one is an anecdote. Say the size every time you name a cluster, and when a
cluster has one or two members say explicitly that it is under-supported rather than reporting
it beside a cluster of nine as though they were comparable. The cohort here is ten cases;
almost everything will be under-supported, and saying so is the finding.

## Naming

Name a cluster by its mechanism, in a sentence that says what would have to change:
"the witness was in an imaging report and imaging was never searched" is a name a fix can be
aimed at. "Primary site errors" is a bucket, not a cause.
