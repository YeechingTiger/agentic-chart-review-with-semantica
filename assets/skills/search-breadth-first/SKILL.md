---
name: search-breadth-first
description: Use when deciding how to traverse a chart and you want coverage before depth - building the full candidate pool of documents that could carry the field before reading any of them closely. Tells you how to sweep the document inventory by type, what to record about each candidate, when a sweep is complete, and what a sweep cannot tell you. Pairs with but does not replace chasing an individual lead.
slot: search
license: MIT
---

# Sweeping wide before reading deep

Build the pool before you commit to a reading order — but not before you read at all. One
`document_type_summary` call gives you every type, its count and its date span; searches over
the candidate types give you the documents that could carry the field.

Open something early. Reading is how you learn what this record calls things, which wording
discriminates here, and whether you already hold the answer — and a sweep conducted before any
of that is a sweep conducted in the wrong vocabulary. The pool is a DENOMINATOR you assemble,
not a gate you must finish passing through.

## The sweep

1. **Inventory by type.** Which types exist in THIS chart, and how many of each. Types the
   contract can never accept still get listed — you need them to say what you excluded.
2. **Widen along a NEW AXIS each time, never by repeating a term under a filter.**
   `search_notes` already covers the whole record, so re-running a term under a type filter
   returns a subset of what you already hold. It reads as more coverage and is less. The axes
   that add something are a DIFFERENT TERM, a DIFFERENT DATE WINDOW, or a class of document you
   have not looked at at all.
3. **Search locates; reading answers.** A hit is a pointer, not a finding. Sweeping is the part
   of this method that can consume a whole budget while opening nothing, and a wide pass that
   read less than a narrow one was wider by every count except the one that decides.
4. **Write the pool down as you go.** Which documents matched, which class, which date. The
   pool is what any later statement about coverage rests on, and once you are deep in reading
   you will no longer be able to reconstruct what you started with.

## When the sweep is done

Not "every type has been searched". A record can hold dozens or hundreds of kinds of
document, most of which cannot bear on this question, and walking all of them rewards formal
coverage over the coverage that decides anything.

A sweep is complete when every SOURCE CLASS that the current claim's proof obligation requires
has been searched or excluded with a reason. What the claim is determines the list: a positive
answer needs the classes that could establish it and the classes that could contradict it; an
absence claim needs every class that could have carried the thing said to be missing.

And note what the pool is not. It holds what your terms matched. A relevant document that none
of your terms reached is not in it, so the pool is a denominator for what you searched, never
for what the record contains.

## What a sweep cannot do

It cannot follow a lead. A document that defers its own conclusion to a result not yet
available, one that points at a source held elsewhere, an amendment referenced but not
returned — a sweep records these as pool members and walks on. Each of them is a question the sweep has now RAISED and not answered.

So a sweep alone under-reads exactly the documents that were about to become decisive. When
your pool contains a deferred conclusion, the sweep has done its job and the next move is not
another sweep.
