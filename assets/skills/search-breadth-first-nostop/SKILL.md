---
name: search-breadth-first-nostop
description: NO-STOP VARIANT — the section stating when the traversal is complete has been removed; identical to `search-breadth-first` otherwise. Use when deciding how to traverse a chart and you want coverage before depth - building the full candidate pool of documents that could carry the field before reading any of them closely. Tells you how to sweep the document inventory by type, what to record about each candidate, and what a sweep cannot tell you. Pairs with but does not replace chasing an individual lead.
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
2. **Widen along a NEW AXIS each time, never by repeating a term under a filter.**
   `search_notes` already searches the whole chart, so running the same term again with a
   document-type filter returns a subset of what you already have. Measured: this card's first
   version said "same terms, type filter varied" and produced 506 searches over eighteen charts,
   460 of them type-filtered, `biopsy` alone eighty-six times — for the same accuracy as issuing
   no searches beyond the obvious ones. The axes that actually add something are a DIFFERENT
   TERM, a DIFFERENT DATE WINDOW, or a type you have not looked at at all.
3. **Search locates; reading answers.** A hit is a pointer, not a finding. The same first
   version spent its budget sweeping and opened FEWER documents per patient than a run with no
   card at all (2.6 against 3.3) — wider by every count except the one that decides the answer.
4. **Record the pool before reading.** Which documents matched, which type, which date. The
   pool is your denominator, and once you begin reading you will stop being able to say what
   you started with.

## What a sweep cannot do

It cannot follow a lead. A pathology report saying "stains pending", a note saying "see
outside records", an addendum referenced but not returned — a sweep records these as pool
members and walks on. Each of them is a question the sweep has now RAISED and not answered.

So a sweep alone under-reads exactly the documents that were about to become decisive. When
your pool contains a deferred conclusion, the sweep has done its job and the next move is not
another sweep.
