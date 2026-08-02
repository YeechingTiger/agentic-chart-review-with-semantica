---
name: tool-contract
description: Use on every run without exception - what the seven tools actually do, as opposed to what their names suggest. Covers how the search matches, what the per-term hit cap hides and in which direction, what a snippet is and is not, what a citation requires, and how one step is recorded as following from another. Facts about the instruments, not advice about how to use them.
slot: general
license: MIT
---

# What the tools actually do

Every run gets this, whatever else it gets. A search policy that happened to know how the
search matches, competing against one that did not, would be measuring knowledge of the
instrument rather than the merit of the policy.

Nothing here tells you what to look for. These are properties of the instruments.

## `search_notes` matches substrings, not words and not meanings

Case-insensitive substring. `carc` finds `carcinoma`; `heart` finds `heartburn`. There is no
word boundary, no stemming engine, no ranking, and **no synonym folding** — two words for one
thing are two searches, and the tool will never tell you that you missed the other one.

It IS tolerant of one thing: a run of whitespace or hyphens in your query matches any run of
whitespace or hyphens in the text, so a phrase written with a hyphen, without one, or broken
across a line wrap is found by any of the three spellings.

**A zero-hit result is a fact about the string you typed.** It is not a fact about the
document set, and it is not a fact about the world. The concept may be present under another
name, in another notation, or in a document this corpus does not contain.

## One call takes many terms, and the hits stay attributed

Pass a list. Each term is searched separately and results come back under `by_term`, so which
term surfaced which document is preserved. Passing five terms in one call is not the same as
merging five searches: the attribution is what lets a later reader see why you opened
something.

## The cap is per term, and it hides the LATER documents

`max_hits` defaults to 25 and applies to each term independently. Documents are scanned in
date order, oldest first, and scanning stops at the cap — so a truncated result is the
**earliest** matches and silently omits everything after them.

`truncated: true` on a term means you are holding a slice, not a total. Reasoning over it as
though it were the whole is the error the flag exists to prevent. Narrow with a type filter or
a date window, or raise `max_hits`.

## A snippet is context, not evidence

A hit carries roughly 250 characters around the match. That is enough to decide whether the
document is worth opening and not enough to know what it says: the sentence may be negated,
attributed to someone else, describing a plan rather than a finding, or belonging to a
different subject than the one you are asking about.

`record_evidence` takes character offsets into a document, and you obtain offsets by reading
the document. **A search that leads to no read has located something and established nothing.**

## What each tool returns

| tool | returns | does not return |
|---|---|---|
| `list_documents` | type, date, size, id — filterable, pageable | any text |
| `document_type_summary` | every type in this record, count, date span | any text |
| `search_notes` | hits with offsets and a snippet, per term | the document |
| `read_document` | text from an offset, up to a limit | more than the limit |
| `read_documents_batch` | a fixed number of characters from each of several | the rest of them |
| `record_evidence` | an entry in the ledger | any judgement about it |
| `submit_answer` | the end of the run | a second chance |

There are seven. There is no semantic search, no section addressing, no timeline, no
similarity, and no tool that will tell you whether you have looked enough.

## Recording what followed from what

Every call takes `because` — why you are making it — and `after_event`, an integer naming the
EARLIER step that caused this one. Every tool result carries its own `step` number; copy that
number. The step you name must already have happened.

Prose is for whoever reads the run. `after_event` is for checking it: without it, a chain of
reasoning and a sequence of unrelated actions look identical in the record. Leaving it out is
not refused, but that step then belongs to no chain.
