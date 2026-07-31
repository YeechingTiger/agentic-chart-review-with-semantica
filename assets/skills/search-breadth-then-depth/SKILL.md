---
name: search-breadth-then-depth
description: Use as the default traversal when neither coverage nor lead-chasing alone fits - sweep the document inventory to build a candidate pool, then chase the leads the sweep raised. Tells you the handover point between the two modes, which leads earn a chase and which do not, when to return to the sweep, and how to record which mode produced the decisive read. Combines the breadth-first and depth-first methods rather than choosing between them.
slot: search
license: MIT
---

# Sweep, then chase what the sweep raised

Two traversals, in one order, with a stated handover. Each covers the other's blind spot: a
sweep records deferred conclusions and walks past them; a chase reads a narrow slice and
cannot support a claim of absence.

## Phase 1 — sweep

Inventory the chart by document type. One search per candidate type, same terms, type filter
varied. Record the pool — which documents matched, which type, which date — BEFORE reading
closely. The pool is the denominator any later absence claim rests on.

## The handover

Leave the sweep when it has done what only it can do: every type searched or explicitly
excluded. At that moment you hold two things — a pool of candidates, and a list of the
questions the sweep RAISED without answering. The second list is the input to phase 2.

Do not hand over early because you found a hit. A hit that the contract lets establish the
field outright can end the whole run, but it does not license skipping the rest of the sweep
if your answer will also assert that nothing else was documented.

## Phase 2 — chase

Rank the raised questions by whether the deferral sits in a document type that COULD establish
the field, and chase them in that order. Follow one thread until it resolves or until the next
hop would be a guess. Do not start a second thread while a higher-ranked one is unresolved.

## Returning to the sweep

Go back when a chase turns up a document type absent from your inventory — an outside report,
a scanned addendum filed under an unexpected type. That is new territory, and the sweep is the
tool for territory. Sweep only the new type, then resume the chase where you left it.

## Recording which mode found it

For each read, say whether it came from the sweep or from a chase, and for a chase, which
thread. The whole reason for running two traversals is to learn which one earns its cost, and
a run that cannot say which mode produced the decisive read cannot contribute to that.
