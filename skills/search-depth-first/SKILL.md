---
name: search-depth-first
description: Use when a document points somewhere else and you want to follow the lead to where the question was settled, rather than sampling more of the chart. Tells you which pointers are worth following and in what order, how far to follow one before abandoning it, how to avoid walking in a circle, and what following a lead cannot tell you about the rest of the chart. Pairs with but does not replace a broad sweep.
slot: search
license: MIT
---

# Following one lead to where it ends

A chart is written forward in time by people who did not yet know the answer. Any document
may defer its own conclusion, and the resolving text is usually reachable from the deferral.
This traversal chases that, one thread at a time, to its end.

## Which pointers to follow, in order

1. **Deferral in a document that COULD establish the field.** A pathology report saying
   stains are pending is the highest-value pointer in a chart: the thing it defers is exactly
   the thing you need, and the resolution is usually in the same file or its addendum.
2. **An explicit cross-reference.** "See addendum", "per outside records", "correlate with
   the 3/14 biopsy" — a named destination. Follow the name.
3. **A hedge that a later document would have resolved.** "Favor squamous" invites a later
   definite statement. Search forward in time from that date.

## How far, and when to stop following

Follow one thread until it resolves, or until the next hop would be a guess rather than a
named destination. A resolved thread is worth more than three half-followed ones: the value
is in reaching the settled statement, and a chain abandoned in the middle has cost the reads
and produced nothing citable.

Never revisit a document you have already read in full during this chase — that is the circle
this traversal is prone to. Keep the thread's hops in mind and, if a hop returns you to a
document already read, the thread is exhausted, not continuing.

## What following a lead cannot do

It tells you nothing about the documents no lead pointed at. A chase that ends in a confident,
well-cited answer has still read a narrow slice of the chart, and if your answer claims that
something is ABSENT, the slice is not the basis for that claim — the rest of the chart is,
and you have not looked at it.
