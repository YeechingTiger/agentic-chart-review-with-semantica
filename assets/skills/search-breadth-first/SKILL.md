---
name: search-breadth-first
description: Use when deciding how to traverse a chart and you want coverage before depth - building the full candidate pool of documents that could carry the field before reading any of them closely. Tells you how to sweep the document inventory by type, what to record about each candidate, when a sweep is complete, and what a sweep cannot tell you. Pairs with but does not replace chasing an individual lead.
slot: search
license: MIT
---

# Sweeping wide before reading deep

Build the pool first. One `document_type_summary` call gives you every type, its count and
its date span; a type-filtered search over each candidate type gives you the documents that
could carry the field. Only then start reading.

## The sweep

1. **Inventory by type.** Which types exist in THIS chart, and how many of each. Types the
   contract can never accept still get listed — you need them to say what you excluded.
2. **One search per candidate type.** Same terms, type filter varied. This is what makes the
   result a comparison rather than a walk: a term that hits in pathology and misses in
   imaging tells you something about the term; a term tried only in pathology does not.
3. **Record the pool before reading.** Which documents matched, which type, which date. The
   pool is your denominator, and once you begin reading you will stop being able to say what
   you started with.

## When the sweep is done

A sweep is complete when every type in the inventory has been searched or explicitly
excluded, with a reason for each exclusion. It is not complete because you found a hit —
a hit ends the sweep only if the contract lets that document type establish the field
outright, and even then the rest of the pool is what a later absence claim rests on.

## What a sweep cannot do

It cannot follow a lead. A pathology report saying "stains pending", a note saying "see
outside records", an addendum referenced but not returned — a sweep records these as pool
members and walks on. Each of them is a question the sweep has now RAISED and not answered.

So a sweep alone under-reads exactly the documents that were about to become decisive. When
your pool contains a deferred conclusion, the sweep has done its job and the next move is not
another sweep.
