---
name: search-native
description: Use when deciding how to search a chart and when to stop searching, and no keyword list or note-type prior has been supplied. Tells you how to choose terms from the task contract and the patient's own document inventory, how to widen after an empty result, and how to judge that further searching would not change your answer. Not a rule about how much searching is enough - that judgement stays yours and is recorded, not enforced.
slot: search
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
2. **The chart's vocabulary** — a term that appears in this patient's document type names
   costs nothing to try and is written in the local dialect.
3. **Synonyms you supply** — the least reliable, because you are guessing at a local
   convention you have not seen. Try these after the first two return nothing.

## Widening after an empty result

An empty search is information: either the term is wrong or the concept is absent. You cannot
tell which from the miss alone, so widen along ONE axis at a time and note which:

- **Shorter stem** — `adenocarc` before `adenocarcinoma`; abbreviations and truncations are how
  dictation actually reads.
- **A different word for the same thing** — the concept's other name, not a related concept.
- **A different document type** — same term, somewhere else in the chart.

Widening along two axes at once means a hit tells you nothing about which change earned it.

## Deciding you are done

You are done when you can say what a further search would have to find in order to change your
answer, and you have looked where that thing would be. That sentence is the test. If you cannot
write it, you are not done; if you can write it and you have looked there, more searching is
spending without a hypothesis.

Two things are NOT reasons to stop: running out of ideas for terms, and having read a lot of
documents. Neither is a statement about the chart.

If your answer claims something is absent, the coverage-judgement skill applies on top of this
one — an absence claim owes more than a positive one.
