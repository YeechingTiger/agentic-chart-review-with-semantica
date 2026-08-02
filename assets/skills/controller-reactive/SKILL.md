---
name: controller-reactive
description: Use when deciding how to search a chart and when to stop searching, and no keyword list or note-type prior has been supplied. Tells you how to choose terms from the task contract and the patient's own document inventory, how to widen after an empty result, and how to judge that further searching would not change your answer. Not a rule about how much searching is enough - that judgement stays yours and is recorded, not enforced.
slot: controller
license: MIT
---

# Choosing your own search, and deciding when it is done

Nobody has handed you a term list. That is the point of this arm: the task contract says what
the answer must mean, the document inventory says what this chart contains, and the searching
in between is yours.

## Where terms come from

Read the contract's field definitions first, then `document_type_summary`. Terms come from
three places, in this order of reliability:

1. **The contract's own words** — the names of the values it asks you to distinguish. These
   are the words a report writer would have used for the same concept.
2. **The record's own vocabulary** — the words this record uses for itself. Take them from
   text you have READ. Document type names are a weaker source than they look: a type name can
   be administrative and never appear in any document's body, so treat it as a signal about
   WHICH SOURCES to open, not as a term to search.
3. **Synonyms you supply** — the least reliable, because you are guessing at a local
   convention you have not seen. Try these after the first two return nothing.

## Widening after an empty result

An empty search is information: either the term is wrong or the concept is absent. You cannot
tell which from the miss alone, so widen along ONE axis at a time and note which:

- **Shorter stem** — `amend` before `amendment`; abbreviations and truncations are how dictated
  and hand-entered text actually reads.
- **A different word for the same thing** — the concept's other name, not a related concept.
- **A different document type** — same term, somewhere else in the chart.

Widening along two axes at once means a hit tells you nothing about which change earned it.

## Deciding you are done

You are done when you can say what a further search would have to find in order to change your
answer, AND you have examined the sources you can currently identify as able to carry it, AND
you have written down which possible sources you did not cover — because you could not name
them, because the record does not contain them, or because you could not reach them.

That last clause is the part that keeps the test honest. Without it, "I looked where it would
be" only means you looked where you thought to look, and a confident wrong guess about where
things live ends the run early and leaves no trace that it did.

Two things are NOT reasons to stop: running out of ideas for terms, and having read a lot of
documents. Neither is a statement about the chart.

If your answer claims something is absent, the coverage-judgement skill applies on top of this
one — an absence claim owes more than a positive one.
